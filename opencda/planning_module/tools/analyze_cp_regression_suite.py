#!/usr/bin/env python3
"""Deterministic acceptance report for CP ON/OFF CARLA runs.

The tool intentionally reads planner outputs only; it does not know scenario
implementation details or tune planner thresholds.  A non-zero exit status
means at least one required closed-loop contract failed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = {
    "vru_off": {
        "directory": "debug_cp_vru_crossing_off",
        "configs": ("cpx_cp_vru_crossing.yaml", "cpx_cp_vru_crossing_off.yaml"),
        "cp_expected": False,
        "route_completion_expected": True,
        "dependencies": (),
    },
    "vru_on": {
        "directory": "debug_cp_vru_crossing_on",
        "configs": ("cpx_cp_vru_crossing.yaml",),
        "cp_expected": True,
        "route_completion_expected": True,
        "risk": "VRU_CONFLICT",
        "action": "YIELD_STOP",
        "dependencies": (
            "opencda/planning_module/pipeline/behavior_risk.py",
            "opencda/planning_module/pipeline/behavior_stage.py",
            "opencda/planning_module/opencda_bridge/cp_provider.py",
        ),
    },
    "roadway_object_off": {
        "directory": "debug_cp_roadway_object_off",
        "configs": (
            "cpx_cp_roadway_object.yaml", "cpx_cp_roadway_object_off.yaml"
        ),
        "cp_expected": False,
        # This is the control arm: without the shared observation the ego is
        # expected to stop safely behind the lane blockage, not complete the
        # route within the evaluation window.
        "route_completion_expected": False,
        "dependencies": (),
    },
    "roadway_object_on": {
        "directory": "debug_cp_roadway_object_on",
        "configs": ("cpx_cp_roadway_object.yaml",),
        "cp_expected": True,
        "route_completion_expected": True,
        "risk": "LANE_BLOCKAGE",
        "action": "PREPARE_LANE_CHANGE",
        "decision": "lane_change_left",
        "dependencies": (
            "opencda/planning_module/pipeline/behavior_risk.py",
            "opencda/planning_module/pipeline/behavior_stage.py",
            "opencda/planning_module/pipeline/static_obstacle_stage.py",
            "opencda/planning_module/opencda_bridge/cp_provider.py",
            "opencda/scenario_testing/cpx_mature_runner.py",
        ),
    },
    # A/B/E route through cav_conflict_enabled's classify -> assign -> corridor
    # chain instead of behavior_risk.py's front-obstacle path.  A/B's hazard
    # is a non-cooperative actor (red-light violator / oncoming car around a
    # bend) that never broadcasts a claim, so cooperative_arbitration.py never
    # assigns it a proceed/yield/make_gap role (cav_arbitration stays
    # "ego_claim_inactive") -- confirmed against a real run where
    # cav_conflict_tags correctly showed CROSSING/ONCOMING but
    # cav_conflict_roles was empty every frame.  The corridor still binds and
    # caps speed (cav_total_qp_row_count > 0), so that -- not "role" -- is
    # what these two check.  E's two CAVs are both cooperative, so its roles
    # are checked once a real run confirms which actor gets which role.
    "red_light_on": {
        "directory": "debug_cp_red_light_on",
        "configs": ("cpx_cp_red_light_violator.yaml",),
        "cp_expected": True,
        "route_completion_expected": True,
        "requires_corridor_binding": True,
        "dependencies": (
            "opencda/planning_module/pipeline/conflict_classifier.py",
            "opencda/planning_module/pipeline/cav_conflict_pipeline.py",
            "opencda/planning_module/pipeline/cooperative_arbitration.py",
            "opencda/planning_module/opencda_bridge/cp_provider.py",
        ),
    },
    "red_light_off": {
        "directory": "debug_cp_red_light_off",
        "configs": (
            "cpx_cp_red_light_violator.yaml", "cpx_cp_red_light_violator_off.yaml"
        ),
        "cp_expected": False,
        "route_completion_expected": True,
        "dependencies": (),
    },
    "bend_oncoming_on": {
        "directory": "debug_cp_bend_oncoming_on",
        "configs": ("cpx_cp_bend_oncoming.yaml",),
        "cp_expected": True,
        "route_completion_expected": True,
        "requires_corridor_binding": True,
        "dependencies": (
            "opencda/planning_module/pipeline/conflict_classifier.py",
            "opencda/planning_module/pipeline/cav_conflict_pipeline.py",
            "opencda/planning_module/pipeline/cooperative_arbitration.py",
            "opencda/planning_module/pipeline/spatiotemporal_corridor.py",
            "opencda/planning_module/opencda_bridge/cp_provider.py",
        ),
    },
    "bend_oncoming_off": {
        "directory": "debug_cp_bend_oncoming_off",
        "configs": (
            "cpx_cp_bend_oncoming.yaml", "cpx_cp_bend_oncoming_off.yaml"
        ),
        "cp_expected": False,
        "route_completion_expected": True,
        "dependencies": (),
    },
    # Two actors, so one directory each.  ON vs. OFF here is
    # cav_conflict_enabled, not CP obstacle sharing (confirmed: the two
    # configs differ only in that flag and their debug dirs), so no
    # cp_expected -- that check is opt-in and skipped when the key is
    # absent.  Which specific role (yield/make_gap) lands on which actor id
    # is deliberately left unpinned: confirm against the first real run
    # before tightening this to a specific vehicle.
    "two_cav_merge_on_cav1": {
        "directory": "debug_two_cav_merge_cav1",
        "configs": ("cpx_two_cav_merge_conflict.yaml",),
        "route_completion_expected": True,
        "dependencies": (
            "opencda/planning_module/pipeline/cav_conflict_pipeline.py",
            "opencda/planning_module/pipeline/cooperative_arbitration.py",
            "opencda/planning_module/pipeline/spatiotemporal_corridor.py",
        ),
    },
    "two_cav_merge_on_cav2": {
        "directory": "debug_two_cav_merge_cav2",
        "configs": ("cpx_two_cav_merge_conflict.yaml",),
        "route_completion_expected": True,
        "dependencies": (
            "opencda/planning_module/pipeline/cav_conflict_pipeline.py",
            "opencda/planning_module/pipeline/cooperative_arbitration.py",
            "opencda/planning_module/pipeline/spatiotemporal_corridor.py",
        ),
    },
    "two_cav_merge_off_cav1": {
        "directory": "debug_two_cav_merge_off_cav1",
        "configs": ("cpx_two_cav_merge_conflict_baseline.yaml",),
        # Confirmed against a real run: without cav_conflict_enabled neither
        # CAV yields, so both stall a few meters short of the merge with no
        # collision rather than resolving -- this is the control arm's whole
        # point, same shape as roadway_object_off.
        "route_completion_expected": False,
        "dependencies": (),
    },
    "two_cav_merge_off_cav2": {
        "directory": "debug_two_cav_merge_off_cav2",
        "configs": ("cpx_two_cav_merge_conflict_baseline.yaml",),
        "route_completion_expected": False,
        "dependencies": (),
    },
}

REPO_ROOT = Path(__file__).resolve().parents[3]


def _rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count(rows, key, value):
    return sum(row.get(key) == value for row in rows)


def _role_frames(rows, role):
    if not role:
        return 0
    return sum(
        str(role) in dict(row.get("cav_conflict_roles", {}) or {}).values()
        for row in rows
    )


def _min_make_gap_gate_margin_m(rows):
    # Informational only, never a pass/fail signal: how close (meters, at
    # closest approach) a make-gap peer's track ever came to opening the
    # corridor cap. A clean cooperative merge can legitimately stay
    # positive the whole run (see test_spatiotemporal_corridor.py's
    # test_gradual_real_world_merge_convergence_stays_below_corridor_gate)
    # -- this exists so that fact is visible in the report instead of
    # requiring the same log archaeology that first surfaced it.
    margins = [
        float(value)
        for row in rows
        for value in dict(row.get("cav_make_gap_gate_margin_m", {}) or {}).values()
    ]
    return min(margins) if margins else None


def _corridor_binding_frames(rows):
    # cav_conflict_roles (proceed/yield/make_gap) is peer-negotiation only --
    # it stays empty against a non-cooperative hazard (a red-light violator
    # or an oncoming car around a bend never broadcasts a claim/intent, so
    # cav_arbitration reads "ego_claim_inactive").  For those cases the only
    # observable evidence that classify->corridor actually engaged is the
    # corridor's own QP row count.
    return sum(int(row.get("cav_total_qp_row_count", 0) or 0) > 0 for row in rows)


def analyze_case(root, name, contract):
    path = root / str(contract["directory"]) / "opencda_planner_debug.jsonl"
    if not path.is_file():
        return {"case": name, "passed": False, "failures": ["missing_log"]}
    rows = _rows(path)
    if not rows:
        return {"case": name, "passed": False, "failures": ["empty_log"]}

    last = rows[-1]
    collision_count = max(int(row.get("collision_count", 0) or 0) for row in rows)
    cp_frames = sum(int(row.get("cp_obstacle_count", 0) or 0) > 0 for row in rows)
    infeasible_frames = sum(
        "infeasible" in str(row.get("mpc_status", "")).lower()
        or str(row.get("mpc_status", "")) == "bounded_safe_stop"
        for row in rows
    )
    risk_frames = _count(rows, "semantic_risk_kind", contract.get("risk"))
    action_frames = _count(
        rows, "semantic_behavior_action", contract.get("action")
    )
    decision_frames = _count(rows, "behavior_decision", contract.get("decision"))
    role_frames = _role_frames(rows, contract.get("role"))
    corridor_binding_frames = _corridor_binding_frames(rows)
    min_make_gap_gate_margin_m = _min_make_gap_gate_margin_m(rows)

    failures = []
    config_paths = tuple(
        REPO_ROOT / "opencda/scenario_testing/config_yaml" / str(name)
        for name in contract["configs"]
    )
    code_paths = tuple(
        REPO_ROOT / str(name) for name in contract.get("dependencies", ())
    )
    dependencies = code_paths + config_paths
    newest_input_mtime = max(
        dependency.stat().st_mtime
        for dependency in dependencies
        if dependency.is_file()
    )
    if path.stat().st_mtime < newest_input_mtime:
        failures.append("stale_log_after_code_or_config_change")
    if collision_count:
        failures.append("collision")
    if (
        bool(contract.get("route_completion_expected", True))
        and not bool(last.get("route_reached_destination", False))
    ):
        failures.append("route_not_completed")
    if infeasible_frames:
        failures.append("mpc_infeasible_or_safe_stop")
    # cp_expected is opt-in: absent means "not applicable to this case"
    # (e.g. the two-CAV scenarios gate on cav_conflict_enabled, not CP
    # obstacle sharing), not "OFF arm, must see zero CP frames".
    if "cp_expected" in contract:
        if bool(contract["cp_expected"]) and cp_frames == 0:
            failures.append("cp_not_received")
        if not bool(contract["cp_expected"]) and cp_frames != 0:
            failures.append("cp_leaked_into_off_arm")
    if contract.get("risk") and risk_frames == 0:
        failures.append("expected_semantic_risk_missing")
    if contract.get("action") and action_frames == 0:
        failures.append("expected_behavior_action_missing")
    if contract.get("decision") and decision_frames == 0:
        failures.append("expected_maneuver_missing")
    if contract.get("role") and role_frames == 0:
        failures.append("expected_cav_role_missing")
    if bool(contract.get("requires_corridor_binding")) and corridor_binding_frames == 0:
        failures.append("expected_corridor_binding_missing")

    return {
        "case": name,
        "passed": not failures,
        "failures": failures,
        "frames": len(rows),
        "sim_time_s": float(last.get("sim_time_s", 0.0) or 0.0),
        "collision_count": collision_count,
        "route_completed": bool(last.get("route_reached_destination", False)),
        "remaining_distance_m": float(
            last.get("route_remaining_distance_m", float("inf"))
        ),
        "cp_frames": cp_frames,
        "semantic_risk_frames": risk_frames,
        "semantic_action_frames": action_frames,
        "expected_decision_frames": decision_frames,
        "expected_role_frames": role_frames,
        "corridor_binding_frames": corridor_binding_frames,
        "min_make_gap_gate_margin_m": min_make_gap_gate_margin_m,
        "mpc_failure_frames": infeasible_frames,
        "log": str(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="opencda/planning_module/opencda_bridge",
        help="directory containing debug_cp_* run folders",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("cases", nargs="*", metavar="CASE")
    args = parser.parse_args()
    unknown = sorted(set(args.cases) - set(CASES))
    if unknown:
        parser.error(
            "unknown case(s): %s; choices: %s"
            % (", ".join(unknown), ", ".join(sorted(CASES)))
        )
    selected = args.cases or list(CASES)
    results = [
        analyze_case(Path(args.root), name, CASES[name]) for name in selected
    ]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        failures = ",".join(result["failures"]) or "none"
        print("%-22s %-4s %s" % (result["case"], status, failures))
    report = {
        "passed": all(result["passed"] for result in results),
        "results": results,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
