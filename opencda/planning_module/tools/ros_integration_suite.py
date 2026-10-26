#!/usr/bin/env python3
"""Run and evaluate repeatable CP-X ROS closed-loop experiments.

The suite deliberately leaves ROS process management outside this script.  A
single long-running Planner/Bridge/MTR stack is part of the experiment: cases
are executed sequentially so map/session leakage is observable instead of
being hidden by restarting ROS between scenarios.

Examples
--------
List the experiment matrix::

    python opencda/planning_module/tools/ros_integration_suite.py list

Run the A-F matrix three times, keeping ROS alive::

    python opencda/planning_module/tools/ros_integration_suite.py run \
      --suite cp_af --trials 3

Analyze the most recent live outputs without running CARLA::

    python opencda/planning_module/tools/ros_integration_suite.py analyze \
      --suite cp_af --live
"""

from __future__ import print_function

import argparse
import datetime
import json
import shutil
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_ROOT = REPO_ROOT / "artifacts" / "debug"
DEFAULT_ARCHIVE_ROOT = REPO_ROOT / "artifacts" / "ros_integration_runs"


CASES = OrderedDict(
    [
        ("smoke_right_turn", dict(scenario="cpx_single_right_lane_turn", dirs=["right_turn_highspeed"], completion=True)),
        ("smoke_left_turn", dict(scenario="cpx_single_left_lane_turn_isolated", dirs=["single_left_lane_turn_isolated"], completion=True)),
        ("smoke_lane_change", dict(scenario="cpx_lane_change_speed_12", dirs=["cpx_lane_change_speed_12"], completion=True)),
        ("A_on", dict(scenario="cpx_cp_red_light_violator", dirs=["cp_red_light_on"], completion=True)),
        ("A_off", dict(scenario="cpx_cp_red_light_violator_off", dirs=["cp_red_light_off"], completion=True)),
        ("B_on", dict(scenario="cpx_cp_bend_oncoming", dirs=["cp_bend_oncoming_on"], completion=True)),
        # The OFF arm is an effect-size baseline.  Its unsafe/non-completing
        # result is reported, never described as a safety pass.
        ("B_off", dict(scenario="cpx_cp_bend_oncoming_off", dirs=["cp_bend_oncoming_off"], completion=False, baseline_unsafe=True)),
        ("C_on", dict(scenario="cpx_cp_vru_crossing", dirs=["cp_vru_crossing_on"], completion=True)),
        ("C_off", dict(scenario="cpx_cp_vru_crossing_off", dirs=["cp_vru_crossing_off"], completion=True)),
        ("D_on", dict(scenario="cpx_cp_roadway_object", dirs=["cp_roadway_object_on"], completion=True)),
        ("D_off", dict(scenario="cpx_cp_roadway_object_off", dirs=["cp_roadway_object_off"], completion=False)),
        ("E_on", dict(scenario="cpx_two_cav_merge_conflict", dirs=["two_cav_merge_cav1", "two_cav_merge_cav2"], completion=True, require_conflict=True)),
        ("E_off", dict(scenario="cpx_two_cav_merge_conflict_baseline", dirs=["two_cav_merge_off_cav1", "two_cav_merge_off_cav2"], completion=True, forbid_conflict=True)),
        ("F_on", dict(scenario="cpx_cp_lane_closure", dirs=["cp_lane_closure"], completion=True, required_lane=630149)),
        ("F_off", dict(scenario="cpx_cp_lane_closure_off", dirs=["cp_lane_closure_off"], completion=False, forbidden_lane=630149)),
        ("mtr_cut_in", dict(scenario="cpx_two_vehicle_cut_in_mtr_six_modes", dirs=["cut_in_mtr_six_modes"], completion=True, modes=6)),
        ("mtr_lead_braking", dict(
            scenario="cpx_two_cav_lead_braking_mtr_multimodal",
            dirs=["lead_braking_mtr_ego", "lead_braking_mtr_lead"],
            completion=True,
            modes=6,
            forbid_lane_change=True,
        )),
        ("mtr_merge_2cav", dict(scenario="cpx_two_cav_merge_conflict_mtr_multimodal", dirs=["two_cav_merge_mtr_cav1", "two_cav_merge_mtr_cav2"], completion=True, modes=6)),
        ("mtr_merge_3cav", dict(scenario="cpx_three_cav_merge_mtr_multimodal", dirs=["three_cav_merge_mtr_cav1", "three_cav_merge_mtr_cav2", "three_cav_merge_mtr_cav3"], completion=True, modes=6, require_lateral_tracker=True)),
        ("mtr_merge_4cav", dict(scenario="cpx_four_cav_merge_mtr_multimodal", dirs=["four_cav_merge_mtr_cav1", "four_cav_merge_mtr_cav2", "four_cav_merge_mtr_cav3", "four_cav_merge_mtr_cav4"], completion=True, modes=6, require_lateral_tracker=True)),
    ]
)


SUITES = OrderedDict(
    [
        ("smoke", ["smoke_right_turn", "smoke_left_turn", "smoke_lane_change"]),
        ("cp_af", ["A_on", "A_off", "B_on", "B_off", "C_on", "C_off", "D_on", "D_off", "E_on", "E_off", "F_on", "F_off"]),
        ("prediction", ["mtr_cut_in", "mtr_lead_braking", "mtr_merge_2cav"]),
        ("scale", ["mtr_merge_2cav", "mtr_merge_3cav", "mtr_merge_4cav"]),
        # Ordering is intentional: it exercises map switching and verifies
        # that an ON run cannot contaminate the immediately following OFF run.
        ("lifecycle", ["A_on", "F_on", "F_off", "C_on", "A_off"]),
        ("full", list(CASES)),
    ]
)


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values):
    return sum(values) / float(len(values)) if values else None


def _first_cycle(rows, predicate):
    for row in rows:
        if predicate(row):
            value = row.get("cycle_time_s")
            return float(value) if value is not None else None
    return None


def _is_complete(status):
    if status.get("termination_reason") in {
        "route_destination_stopped",
        "destination_reached",
    }:
        return True
    states = list(status.get("cav_states", []) or [])
    subject = int(status.get("subject_cav_index", 0) or 0)
    return bool(states and subject < len(states) and states[subject].get("agent_finished"))


def _subject_cav_state(status):
    """Return the per-CAV status represented by one archived CAV log."""
    states = list(status.get("cav_states", []) or [])
    subject = int(status.get("subject_cav_index", 0) or 0)
    if 0 <= subject < len(states):
        return dict(states[subject] or {})
    return {}


def _analyze_log(directory):
    status_path = directory / "run_status.json"
    log_path = directory / "ros_planner_control_output.jsonl"
    missing = [str(path.name) for path in (status_path, log_path) if not path.is_file()]
    if missing:
        return {"directory": str(directory), "failures": ["missing:" + name for name in missing]}
    status = _read_json(status_path)
    subject_state = _subject_cav_state(status)
    rows = _read_jsonl(log_path)
    if not rows:
        return {"directory": str(directory), "failures": ["empty_ros_control_log"]}

    states = [str(row.get("ros_control_state", "")) for row in rows]
    non_ros_frames = sum(str(row.get("source", "")) != "ros" for row in rows)
    # VehicleManager records the executable fallback state as
    # ``stale_control``.  Older archived logs used ``stale``.  Accept both so
    # the regression report reflects the emergency-brake ticks that actually
    # occurred instead of silently reporting 0.00%.
    stale_frames = sum(
        state in {"stale", "stale_control"} for state in states
    )
    active_frames = sum(state == "active" for state in states)
    reused_active_frames = sum(
        state == "active" and bool(row.get("control_reused", False))
        for state, row in zip(states, rows)
    )
    lateral_tracker_frames = sum(
        state == "active"
        and bool(row.get("lateral_execution_tracker_active", False))
        for state, row in zip(states, rows)
    )
    session_ids = sorted(set(str(row.get("session_id", "")) for row in rows if row.get("session_id")))
    latencies = [float(row["input_to_output_time_ms"]) for row in rows if row.get("input_to_output_time_ms") is not None]
    planner_times = [float(row["planning_cycle_time_ms"]) for row in rows if row.get("planning_cycle_time_ms") is not None]
    target_lanes = sorted(set(int(row["route_physical_target_lane_id"]) for row in rows if isinstance(row.get("route_physical_target_lane_id"), (int, float))))
    lane_change_frames = sum(
        str(row.get("behavior_decision", "")).strip().lower()
        in {"lane_change_left", "lane_change_right"}
        for row in rows
    )
    conflicts = sum(bool(str(row.get("cav_conflict_summary", ""))) for row in rows)
    revisions = [int(row.get("prediction_ros_revision", -1) or -1) for row in rows]
    attached_modes = max(int(row.get("prediction_bridge_attached_mode_count", 0) or 0) for row in rows)
    attached_ages = [
        float(row["prediction_ros_frame_age_s"])
        for row in rows
        if int(row.get("prediction_bridge_attached_mode_count", 0) or 0) > 0
        and row.get("prediction_ros_frame_age_s") is not None
    ]
    target_speeds = [
        float(row["target_speed_mps"])
        for row in rows
        if row.get("target_speed_mps") is not None
    ]
    cruise_target = max(target_speeds) if target_speeds else None
    failures = []
    if status.get("exception"):
        failures.append("scenario_exception")
    if active_frames == 0:
        failures.append("no_active_ros_control")
    if non_ros_frames:
        failures.append("non_ros_control_output")
    if len(session_ids) != 1:
        failures.append("expected_one_session_per_cav_log")
    if stale_frames / float(len(rows)) > 0.02:
        failures.append("stale_control_rate_above_2_percent")
    return {
        "directory": str(directory),
        "failures": failures,
        "termination_reason": str(status.get("termination_reason", "")),
        "completed": _is_complete(status),
        "ticks": int(status.get("completed_ticks", 0) or 0),
        "collision_count": int(status.get("collision_count", 0) or 0),
        "road_boundary_breach_count": int(status.get("road_boundary_breach_count", 0) or 0),
        "remaining_distance_m": float(
            subject_state.get(
                "distance_to_destination_m",
                status.get("distance_to_destination_m", 0.0),
            )
            or 0.0
        ),
        "rows": len(rows),
        "active_frames": active_frames,
        "reused_active_frames": reused_active_frames,
        "lateral_tracker_frames": lateral_tracker_frames,
        "lateral_tracker_rate": (
            lateral_tracker_frames / float(reused_active_frames)
            if reused_active_frames
            else 0.0
        ),
        "non_ros_frames": non_ros_frames,
        "stale_frames": stale_frames,
        "stale_rate": stale_frames / float(len(rows)),
        "session_ids": session_ids,
        "mean_input_to_output_ms": _mean(latencies),
        "max_input_to_output_ms": max(latencies) if latencies else None,
        "mean_planning_cycle_ms": _mean(planner_times),
        "max_attached_modes": attached_modes,
        "max_attached_prediction_age_s": max(attached_ages) if attached_ages else None,
        "prediction_revision_span": max(revisions) - min(revisions) if revisions else 0,
        "conflict_frames": conflicts,
        "target_lanes": target_lanes,
        "lane_change_frames": lane_change_frames,
        "first_corridor_cycle_s": _first_cycle(
            rows, lambda row: int(row.get("cav_total_qp_row_count", 0) or 0) > 0
        ),
        "first_relevant_obstacle_cycle_s": _first_cycle(
            rows,
            lambda row: int(row.get("mpc_obstacle_relevant_count", 0) or 0) > 0,
        ),
        "first_speed_reduction_cycle_s": _first_cycle(
            rows,
            lambda row: cruise_target is not None
            and row.get("target_speed_mps") is not None
            and float(row["target_speed_mps"]) < cruise_target - 0.5,
        ),
    }


def analyze_case(case_name, contract, case_root):
    logs = [_analyze_log(case_root / name) for name in contract["dirs"]]
    failures = []
    for log in logs:
        failures.extend("%s:%s" % (Path(log["directory"]).name, item) for item in log.get("failures", []))
    complete_values = [log.get("completed") for log in logs if "completed" in log]
    expected_completion = bool(contract.get("completion", True))
    if complete_values and any(bool(value) != expected_completion for value in complete_values):
        failures.append("completion_outcome_mismatch")
    collision_count = max([int(log.get("collision_count", 0)) for log in logs] or [0])
    safety_pass = collision_count == 0 and all(int(log.get("road_boundary_breach_count", 0)) == 0 for log in logs)
    if not safety_pass and not contract.get("baseline_unsafe"):
        failures.append("safety_violation")
    all_sessions = [session_id for log in logs for session_id in log.get("session_ids", [])]
    if len(logs) > 1 and len(set(all_sessions)) != len(logs):
        failures.append("multi_cav_session_not_isolated")
    required_modes = int(contract.get("modes", 0) or 0)
    if required_modes and not any(int(log.get("max_attached_modes", 0)) >= required_modes for log in logs):
        failures.append("six_mode_prediction_not_attached")
    if required_modes and not any(int(log.get("prediction_revision_span", 0)) > 0 for log in logs):
        failures.append("prediction_revision_did_not_advance")
    if required_modes and any(
        log.get("max_attached_prediction_age_s") is not None
        and float(log["max_attached_prediction_age_s"]) > 0.500001
        for log in logs
    ):
        failures.append("attached_prediction_older_than_0_5s")
    if contract.get("require_lateral_tracker") and any(
        int(log.get("lateral_tracker_frames", 0)) == 0 for log in logs
    ):
        failures.append("lateral_execution_tracker_not_observed")
    conflict_frames = sum(int(log.get("conflict_frames", 0)) for log in logs)
    if contract.get("require_conflict") and conflict_frames == 0:
        failures.append("cooperative_conflict_path_not_exercised")
    if contract.get("forbid_conflict") and conflict_frames != 0:
        failures.append("conflict_leaked_into_baseline")
    lanes = sorted(set(lane for log in logs for lane in log.get("target_lanes", [])))
    if contract.get("required_lane") is not None and int(contract["required_lane"]) not in lanes:
        failures.append("required_target_lane_missing")
    if contract.get("forbidden_lane") is not None and int(contract["forbidden_lane"]) in lanes:
        failures.append("forbidden_target_lane_present")
    lane_change_frames = sum(int(log.get("lane_change_frames", 0)) for log in logs)
    if contract.get("forbid_lane_change") and lane_change_frames:
        failures.append("unexpected_lane_change_decision")
    return {
        "case": case_name,
        "scenario": contract["scenario"],
        "contract_pass": not failures,
        "safety_pass": safety_pass,
        "baseline_unsafe_by_design": bool(contract.get("baseline_unsafe", False)),
        "failures": failures,
        "logs": logs,
    }


def _selected_cases(args):
    names = list(SUITES[args.suite])
    if getattr(args, "cases", None):
        requested = [item.strip() for item in args.cases.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(CASES))
        if unknown:
            raise SystemExit("Unknown case(s): %s" % ", ".join(unknown))
        names = requested
    return names


def _port_open(port):
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError:
        return False
    finally:
        if sock is not None:
            sock.close()


def preflight():
    checks = OrderedDict([("CARLA:2000", 2000), ("ROS localization:5051", 5051), ("ROS perception:5052", 5052), ("ROS traffic light:5053", 5053), ("ROS V2X:5054", 5054), ("ROS cooperative:5055", 5055)])
    results = OrderedDict((name, _port_open(port)) for name, port in checks.items())
    for name, ready in results.items():
        print("%-24s %s" % (name, "READY" if ready else "MISSING"))
    return all(results.values())


def _archive_case(case_name, contract, destination, started_ns):
    destination.mkdir(parents=True, exist_ok=True)
    for name in contract["dirs"]:
        source = LIVE_ROOT / name
        status = source / "run_status.json"
        if not status.is_file() or status.stat().st_mtime_ns < started_ns:
            raise RuntimeError("%s did not produce a fresh run_status.json" % case_name)
        shutil.copytree(str(source), str(destination / name))


def command_run(args):
    if not preflight():
        raise SystemExit("Preflight failed; start CARLA and the ROS integration stack first.")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = Path(args.output_root).expanduser().resolve() / timestamp
    selected = _selected_cases(args)
    metadata = {"suite": args.suite, "cases": selected, "trials": int(args.trials), "carla_version": args.carla_version, "started_at": timestamp}
    archive_root.mkdir(parents=True, exist_ok=False)
    (archive_root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for trial in range(1, int(args.trials) + 1):
        for case_name in selected:
            contract = CASES[case_name]
            command = [args.python, str(REPO_ROOT / "opencda.py"), "-t", contract["scenario"], "-v", args.carla_version]
            print("\n[trial %d] %s: %s" % (trial, case_name, " ".join(command)), flush=True)
            started_ns = time.time_ns() if hasattr(time, "time_ns") else int(time.time() * 1e9)
            log_dir = archive_root / ("trial_%02d" % trial) / case_name
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "console.log").open("w", encoding="utf-8") as output:
                process = subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")
                    output.write(line)
                return_code = process.wait()
            if return_code:
                if not args.continue_on_failure:
                    raise SystemExit("%s exited with status %d" % (case_name, return_code))
                continue
            _archive_case(case_name, contract, log_dir, started_ns)
    print("\nArchived experiment suite: %s" % archive_root)
    return command_analyze(args, root=archive_root)


def _write_reports(report, output_base):
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# ROS Integration Experiment Summary",
        "",
        "| Trial | Case | Contract | Safety | Completion | Collisions | Stale | Tracker | Mean I/O | Max I/O | Modes | Sessions | Failures |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for trial in report["trials"]:
        for result in trial["results"]:
            logs = result["logs"]
            completion = "/".join(
                "yes" if log.get("completed") else "no"
                for log in logs
                if "completed" in log
            ) or "missing"
            collisions = max([int(log.get("collision_count", 0)) for log in logs] or [0])
            stale_rate = max([float(log.get("stale_rate", 0.0)) for log in logs] or [0.0])
            tracker_rate = min(
                [float(log.get("lateral_tracker_rate", 0.0)) for log in logs]
                or [0.0]
            )
            mean_latency = max(
                [float(log["mean_input_to_output_ms"]) for log in logs if log.get("mean_input_to_output_ms") is not None]
                or [0.0]
            )
            max_latency = max(
                [float(log["max_input_to_output_ms"]) for log in logs if log.get("max_input_to_output_ms") is not None]
                or [0.0]
            )
            modes = max([int(log.get("max_attached_modes", 0)) for log in logs] or [0])
            sessions = len(set(session for log in logs for session in log.get("session_ids", [])))
            lines.append(
                "| %d | %s | %s | %s | %s | %d | %.2f%% | %.1f%% | %.1f ms | %.1f ms | %d | %d | %s |"
                % (
                    trial["trial"],
                    result["case"],
                    "PASS" if result["contract_pass"] else "FAIL",
                    "PASS" if result["safety_pass"] else "UNSAFE BASELINE",
                    completion,
                    collisions,
                    stale_rate * 100.0,
                    tracker_rate * 100.0,
                    mean_latency,
                    max_latency,
                    modes,
                    sessions,
                    ", ".join(result["failures"]) or "none",
                )
            )
    (output_base / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_analyze(args, root=None):
    selected = _selected_cases(args)
    trials = []
    if getattr(args, "live", False):
        trial_roots = [(1, LIVE_ROOT)]
        output_base = Path(args.output).resolve() if args.output else DEFAULT_ARCHIVE_ROOT / "latest_live_analysis"
    else:
        root = Path(root or args.root).expanduser().resolve()
        trial_roots = []
        for path in sorted(root.glob("trial_*")):
            try:
                number = int(path.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            trial_roots.append((number, path))
        output_base = Path(args.output).resolve() if args.output else root
    if not trial_roots:
        raise SystemExit("No experiment trials found.")
    for number, trial_root in trial_roots:
        results = []
        for name in selected:
            case_root = trial_root if getattr(args, "live", False) else trial_root / name
            result = analyze_case(name, CASES[name], case_root)
            results.append(result)
            print("trial=%d %-18s contract=%s safety=%s %s" % (number, name, "PASS" if result["contract_pass"] else "FAIL", "PASS" if result["safety_pass"] else "FAIL", ",".join(result["failures"]) or "none"))
        trials.append({"trial": number, "results": results})
    report = {"suite": args.suite, "passed": all(item["contract_pass"] for trial in trials for item in trial["results"]), "trials": trials}
    _write_reports(report, output_base)
    print("Reports: %s" % output_base)
    return 0 if report["passed"] else 1


def command_list():
    for suite, names in SUITES.items():
        print("%-12s %s" % (suite, ", ".join(names)))
    print("\nCases:")
    for name, contract in CASES.items():
        print("  %-18s %-48s %s" % (name, contract["scenario"], ", ".join(contract["dirs"])))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    subparsers.add_parser("list")
    subparsers.add_parser("preflight")
    for command in ("run", "analyze"):
        child = subparsers.add_parser(command)
        child.add_argument("--suite", choices=list(SUITES), default="cp_af")
        child.add_argument("--cases", default="", help="comma-separated case names overriding --suite")
        child.add_argument("--output", default="")
        if command == "run":
            child.add_argument("--trials", type=int, default=3)
            child.add_argument("--python", default=sys.executable)
            child.add_argument("--carla-version", default="0.9.12")
            child.add_argument("--output-root", default=str(DEFAULT_ARCHIVE_ROOT))
            child.add_argument("--continue-on-failure", action="store_true")
        else:
            child.add_argument("--root", default="")
            child.add_argument("--live", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "list":
        command_list()
        return 0
    if args.command == "preflight":
        return 0 if preflight() else 1
    if args.command == "run":
        return command_run(args)
    if not args.live and not args.root:
        parser.error("analyze requires --live or --root")
    return command_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
