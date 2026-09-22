# -*- coding: utf-8 -*-
"""Canonical scenario-run completion artifacts for CP-X regressions.

The planner owns planning metrics.  Scenario runners own simulator truth and
task completion.  This module joins those two read-only views at teardown so
every runner emits the same small ``run_status.json`` contract without
feeding evaluator state back into planning.
"""

import json
import math
import os
from pathlib import Path


def _distance_to_destination(vehicle, destination):
    if vehicle is None or destination is None:
        return None
    location = vehicle.get_location()
    return math.hypot(
        float(location.x) - float(destination[0]),
        float(location.y) - float(destination[1]),
    )


def _debug_output_dir(vehicle_manager, fallback_dir):
    planner = getattr(vehicle_manager, "cpx_planner", None)
    resolve = getattr(planner, "_resolved_debug_output_dir", None)
    if callable(resolve):
        return Path(resolve())
    return Path(str(fallback_dir))


def _planner_metrics(vehicle_manager):
    planner = getattr(vehicle_manager, "cpx_planner", None)
    recorder = getattr(planner, "evaluation_metrics", None)
    summary = getattr(recorder, "summary", None)
    if not callable(summary):
        return {}
    return dict(summary())


def build_cav_run_state(index, vehicle_manager, destination=None):
    """Capture one CAV's terminal simulator and planner-metric state."""

    vehicle = getattr(vehicle_manager, "vehicle", None)
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    metrics = _planner_metrics(vehicle_manager)
    distance = _distance_to_destination(vehicle, destination)
    return {
        "index": int(index),
        "vehicle_id": int(getattr(vehicle, "id", -1)),
        "planner": (
            "cpx_mpc"
            if getattr(vehicle_manager, "cpx_planner", None) is not None
            else "opencda_behavior_agent"
        ),
        "x_m": float(transform.location.x),
        "y_m": float(transform.location.y),
        "speed_mps": float(math.sqrt(
            velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
        )),
        "distance_to_destination_m": distance,
        "collision_count": int(metrics.get("collision_count", 0) or 0),
        "road_boundary_breach_count": int(
            metrics.get("road_boundary_breach_count", 0) or 0
        ),
        "road_boundary_sample_count": int(
            metrics.get("road_boundary_sample_count", 0) or 0
        ),
        "road_boundary_breach_rate": metrics.get(
            "road_boundary_breach_rate", None
        ),
        "mpc_plan_attempts": int(metrics.get("mpc_plan_attempts", 0) or 0),
        "mpc_plan_successes": int(metrics.get("mpc_plan_successes", 0) or 0),
        "mpc_plan_success_rate": float(
            metrics.get("mpc_plan_success_rate", 0.0) or 0.0
        ),
        "agent_finished": bool(getattr(
            vehicle_manager, "_opencda_agent_finished", False
        )),
    }


def write_run_status(
    *,
    vehicle_managers,
    vehicle_configs,
    termination_reason,
    completed_ticks,
    scenario_name,
    exception="",
    completion_mode="first_cav",
    fallback_debug_output_dir="artifacts",
):
    """Write one canonical status payload beside every CAV's planner log.

    A multi-CAV run may configure one debug directory per vehicle.  The same
    scenario-level payload is written to each directory, while
    ``subject_cav_index`` identifies which CAV owns that directory.
    """

    managers = list(vehicle_managers or [])
    configs = list(vehicle_configs or [])
    cav_states = []
    for index, manager in enumerate(managers):
        destination = (
            configs[index].get("destination") if index < len(configs) else None
        )
        cav_states.append(build_cav_run_state(index, manager, destination))

    payload = {
        "scenario_name": str(scenario_name),
        "termination_reason": str(termination_reason),
        "exception": str(exception or ""),
        "completed_ticks": int(completed_ticks),
        "completion_mode": str(completion_mode),
        "distance_to_destination_m": (
            cav_states[0].get("distance_to_destination_m")
            if cav_states else None
        ),
        "cav_count": int(len(cav_states)),
        "all_cavs_finished": bool(cav_states) and all(
            bool(state.get("agent_finished", False)) for state in cav_states
        ),
        "collision_count": sum(
            int(state.get("collision_count", 0) or 0) for state in cav_states
        ),
        "road_boundary_breach_count": sum(
            int(state.get("road_boundary_breach_count", 0) or 0)
            for state in cav_states
        ),
        "cav_states": cav_states,
    }

    written = []
    seen_dirs = set()
    for index, manager in enumerate(managers):
        output_dir = _debug_output_dir(manager, fallback_debug_output_dir)
        key = os.path.abspath(str(output_dir))
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        output_dir.mkdir(parents=True, exist_ok=True)
        per_dir_payload = dict(payload)
        per_dir_payload["subject_cav_index"] = int(index)
        status_path = output_dir / "run_status.json"
        with status_path.open("w", encoding="utf-8") as status_file:
            json.dump(per_dir_payload, status_file, indent=2, sort_keys=True)
        written.append(str(status_path))
    return tuple(written)
