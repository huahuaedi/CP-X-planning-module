"""Offline acceptance audit for one CP-X OpenCDA planning run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _number(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _lane_sequence(row):
    return tuple(
        int(value)
        for value in str(row.get("local_map_route_lane_sequence", "")).split(";")
        if str(value).strip()
    )


def audit_run(rows, run_status=None):
    samples = [dict(row) for row in list(rows or [])]
    violations = []
    metrics = {"frame_count": len(samples)}
    if not samples:
        return ("run_has_no_frames",), metrics

    collision_frames = sum(
        str(row.get("collision_event_this_frame", "")).lower() == "true"
        for row in samples
    )
    breach_frames = sum(
        str(row.get("road_boundary_breach", "")).lower() == "true"
        for row in samples
    )
    invalid_map_frames = sum(
        str(row.get("local_map_valid", "")).lower() != "true"
        for row in samples
    )
    # A local lane-change scenario intentionally has no global route.  An
    # empty topology is invalid only when the run actually advertises a route
    # signature; otherwise this audit used to reject a healthy local test.
    topology_expected_frames = sum(
        bool(str(row.get("route_topology_signature", "")).strip())
        for row in samples
    )
    topology_invalid_frames = sum(
        bool(str(row.get("route_topology_signature", "")).strip())
        and str(row.get("route_topology_valid", "")).lower() != "true"
        for row in samples
    )
    speed_ceiling_frames = sum(
        bool(str(row.get("pid_target_velocity_mps", "")).strip())
        and _number(row, "pid_target_velocity_mps")
        > _number(row, "nominal_speed_ref_mps", _number(row, "target_speed_mps"))
        + 1.0e-3
        for row in samples
    )
    pipeline_error_frames = sum(
        bool(str(row.get("pipeline_error", "")).strip()) for row in samples
    )
    defer_safe_stop_frames = sum(
        "route_required_candidate_infeasible_defer"
        in str(row.get("candidate_pipeline_selected_reason", ""))
        and str(row.get("candidate_pipeline_selected", "")).strip().lower()
        == "bounded_safe_stop"
        for row in samples
    )
    route_direction_mismatch_frames = sum(
        str(row.get("lane_change_authorization_direction", "")).strip().lower()
        in {"left", "right"}
        and "lane_change_" in str(row.get("candidate_evaluation_summary", ""))
        and (
            "lane_change_" + str(
                row.get("lane_change_authorization_direction", "")
            ).strip().lower()
        ) not in str(row.get("candidate_evaluation_summary", ""))
        for row in samples
    )
    max_infeasible_run = 0
    infeasible_run = 0
    max_hard_gate_run = 0
    hard_gate_run = 0
    for row in samples:
        if "infeasible" in str(row.get("mpc_status", "")).lower():
            infeasible_run += 1
            max_infeasible_run = max(max_infeasible_run, infeasible_run)
        else:
            infeasible_run = 0
        hard_gate_active = (
            str(row.get("candidate_pipeline_selected_status", "")).strip().lower()
            == "candidate_hard_gate"
            or str(row.get("candidate_pipeline_selected", "")).strip().lower()
            == "bounded_safe_stop"
            or str(row.get("fallback_active", "")).strip().lower() == "true"
        )
        if hard_gate_active:
            hard_gate_run += 1
            max_hard_gate_run = max(max_hard_gate_run, hard_gate_run)
        else:
            hard_gate_run = 0

    metrics.update({
        "collision_frames": int(collision_frames),
        "road_boundary_breach_frames": int(breach_frames),
        "invalid_local_map_frames": int(invalid_map_frames),
        "invalid_route_topology_frames": int(topology_invalid_frames),
        "route_topology_expected_frames": int(topology_expected_frames),
        "pid_above_nominal_frames": int(speed_ceiling_frames),
        "pipeline_error_frames": int(pipeline_error_frames),
        "route_defer_safe_stop_frames": int(defer_safe_stop_frames),
        "route_direction_mismatch_frames": int(route_direction_mismatch_frames),
        "max_consecutive_mpc_infeasible_frames": int(max_infeasible_run),
        "max_consecutive_hard_gate_frames": int(max_hard_gate_run),
        "final_speed_mps": _number(samples[-1], "speed_mps"),
    })

    turn_indices = [
        index for index, row in enumerate(samples)
        if str(row.get("behavior_decision", "")).startswith("intersection_turn_")
    ]
    turn_exit_route_consistent = True
    turn_exit_lane_id = 0
    turn_completed = False
    if turn_indices:
        last_turn_index = turn_indices[-1]
        turn_lane_id = int(
            _number(samples[last_turn_index], "current_lane_id", 0.0)
        )
        exit_rows = [
            row for row in samples[last_turn_index + 1:]
            if str(row.get("behavior_decision", "")) == "lane_follow"
        ]
        if exit_rows:
            turn_completed = True
            # Validate the settled post-turn lane, not merely the first frame
            # after the FSM changes state.  That first frame normally remains
            # on the connector and previously hid a later drift into an
            # adjacent non-route lane.
            exit_row = exit_rows[-1]
            turn_exit_lane_id = int(_number(exit_row, "current_lane_id", 0.0))
            sequence = _lane_sequence(exit_row) or _lane_sequence(
                samples[last_turn_index]
            )
            turn_exit_route_consistent = bool(
                int(turn_exit_lane_id) != 0
                and (
                    int(turn_exit_lane_id) == int(turn_lane_id)
                    or int(turn_exit_lane_id) in sequence
                )
            )
    metrics["turn_exit_lane_id"] = int(turn_exit_lane_id)
    metrics["turn_exit_route_consistent"] = bool(turn_exit_route_consistent)
    metrics["turn_started"] = bool(turn_indices)
    metrics["turn_completed"] = bool(turn_completed)
    if collision_frames:
        violations.append("collision_detected")
    if breach_frames:
        violations.append("road_boundary_breach")
    if invalid_map_frames:
        violations.append("local_map_invalid")
    if topology_invalid_frames:
        violations.append("route_topology_invalid")
    if speed_ceiling_frames:
        violations.append("pid_target_above_nominal")
    if pipeline_error_frames:
        violations.append("planning_pipeline_exception")
    if defer_safe_stop_frames:
        violations.append("route_lane_change_defer_became_safe_stop")
    if route_direction_mismatch_frames:
        violations.append("route_lane_change_direction_mismatch")
    if max_infeasible_run > 10:
        violations.append("mpc_infeasible_run_exceeds_10_frames")
    if max_hard_gate_run > 10:
        violations.append("candidate_hard_gate_run_exceeds_10_frames")
    if not bool(turn_exit_route_consistent):
        violations.append("turn_exit_lane_not_on_route")
    if turn_indices and not bool(turn_completed):
        violations.append("turn_not_completed")

    status = dict(run_status or {})
    termination_reason = str(status.get("termination_reason", ""))
    metrics["termination_reason"] = termination_reason
    cav_states = list(status.get("cav_states", []) or [])
    terminal_speed_mps = (
        _number(cav_states[0], "speed_mps")
        if cav_states else metrics["final_speed_mps"]
    )
    metrics["terminal_speed_mps"] = terminal_speed_mps
    if termination_reason in {"destination_reached", "route_destination_stopped"}:
        if terminal_speed_mps > 0.5:
            violations.append("destination_terminated_while_moving")
    if termination_reason in {"scenario_exception", "max_ticks"}:
        violations.append("scenario_did_not_complete_normally")
    return tuple(dict.fromkeys(violations)), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--run-status", type=Path)
    args = parser.parse_args()
    with args.csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    status = {}
    if args.run_status and args.run_status.exists():
        status = json.loads(args.run_status.read_text())
    violations, metrics = audit_run(rows, status)
    print(json.dumps({
        "accepted": not violations,
        "violations": list(violations),
        "metrics": metrics,
    }, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
