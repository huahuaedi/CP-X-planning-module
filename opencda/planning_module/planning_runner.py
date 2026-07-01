"""
Shared CARLA scenario runner.
"""

from __future__ import annotations

import csv
import struct
import zlib
import importlib
import inspect
import json
import math
import os
import queue
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from MPC import (
    MPC,
    build_route_reference_samples,
    compute_lane_lookahead_distance,
)
from behavior_planner import (
    LaneSafetyScorer,
    RuleBasedBehaviorPlanner,
    build_reference_samples,
    find_relevant_signal_context,
    find_stop_target_from_ego,
    evaluate_intersection_obstacle_response,
    is_fixed_stop_decision,
    is_stop_decision,
    is_emergency_brake_decision,
    intersection_route_follow_maneuver,
    normalize_behavior_decision,
    normalize_macro_maneuver,
    compute_temp_destination_mode,
    compute_temp_destination,
    compute_ego_lane_offset,
)
from behavior_planner.reroute import reroute_from_lane_closure_messages
from utility import (
    AStarGlobalPlanner,
    CP_MESSAGE_PATH,
    EvaluationMetricsRecorder,
    Tracker,
    build_lane_center_waypoints,
    load_lane_closure_messages,
    load_obstacle_snapshots,
    load_yaml_file,
    reset_cp_message_payload,
    write_planning_metrics_artifacts,
)
from utility import canonical_lane_id_for_waypoint, canonical_lane_waypoint_for_lane_id
from pipeline import build_prediction_frame, evaluate_behavior_candidates
from behavior_planner.car_follow import idm_acceleration
from utility.speed_profile import trapezoidal_stop_profile

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]


# `planning_runner.py` lives directly under `planning_module/`, unlike the
# legacy runner under `carla_scenario/`. Keep config paths rooted at the
# planning-module directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MPC_CONFIG_PATH = os.path.join(PROJECT_ROOT, "MPC", "mpc.yaml")
TRACKER_CONFIG_PATH = os.path.join(PROJECT_ROOT, "utility", "tracker.yaml")

# CARLA simulation tick [s]. Fixed at 20 Hz — do NOT change at runtime.
# MPC prediction step (plan_dt_s) is independent and configured in mpc.yaml.
CARLA_FIXED_DELTA_SECONDS: float = 0.05
WORLD_DEBUG_ROUTE_REFRESH_PERIOD_S: float = 0.10
WORLD_DEBUG_ROUTE_LIFE_TIME_S: float = 0.15
REQUIRED_CONSTRAINT_KEYS = (
    "min_velocity_mps",
    "max_velocity_mps",
    "min_acceleration_mps2",
    "max_acceleration_mps2",
    "max_jerk_mps3",
    "min_steer_rad",
    "max_steer_rad",
    "min_steer_rate_rps",
    "max_steer_rate_rps",
    "enforce_terminal_velocity_constraint",
    "terminal_velocity_mps",
)


def _scenario_artifact_dir(scenario_cfg: Mapping[str, object]) -> str:
    scenario_dir = str(scenario_cfg.get("_scenario_dir", "")).strip()
    if scenario_dir:
        return scenario_dir

    scenario_path = str(scenario_cfg.get("_scenario_path", "")).strip()
    if scenario_path:
        return os.path.dirname(scenario_path)

    runner_module_name = str(scenario_cfg.get("runner_module", "")).strip()
    base_dir = os.path.join(
        PROJECT_ROOT,
        "opencda_scenario" if runner_module_name == "opencda_scenario.runner" else "carla_scenario",
    )
    scenario_name = str(scenario_cfg.get("name", "scenario")).strip() or "scenario"
    direct_dir = os.path.join(base_dir, scenario_name)
    if os.path.isdir(direct_dir):
        return direct_dir

    for root, _, filenames in os.walk(base_dir):
        if f"{scenario_name}.yaml" in filenames or f"{scenario_name}.yml" in filenames:
            return root
    return direct_dir


def _mpc_cost_total(cost_terms: Mapping[str, object]) -> float:
    cost_road_boundary = float(
        cost_terms.get("Cost_RoadBoundary", cost_terms.get("Cost_LaneBoundary", 0.0))
    )
    return float(
        float(cost_terms.get("Cost_ref", 0.0))
        + float(cost_terms.get("Cost_LaneCenter", 0.0))
        + float(cost_road_boundary)
        + float(cost_terms.get("Cost_Repulsive", 0.0))
        + float(cost_terms.get("Cost_Control", 0.0))
    )


def _safe_plot_float(value: object, default: float = 0.0) -> float:
    try:
        numeric_value = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(float(numeric_value)):
        return float(default)
    return float(numeric_value)


def _append_mpc_cost_sample(cost_history: List[Dict[str, object]], sim_time_s: float, mpc: MPC) -> None:
    cost_terms = dict(mpc.get_last_cost_terms())
    runtime_status = dict(mpc.get_runtime_status())
    cost_road_boundary = float(
        cost_terms.get("Cost_RoadBoundary", cost_terms.get("Cost_LaneBoundary", 0.0))
    )
    sample: Dict[str, object] = {
        "sample_index": int(len(cost_history)),
        "sim_time_s": float(sim_time_s),
        "Cost_ref": float(cost_terms.get("Cost_ref", 0.0)),
        "Cost_LaneCenter": float(cost_terms.get("Cost_LaneCenter", 0.0)),
        "Cost_RoadBoundary": float(cost_road_boundary),
        "Cost_LaneBoundary": float(cost_road_boundary),
        "Cost_Lane": float(cost_terms.get("Cost_Lane", 0.0)),
        "Cost_Repulsive_Safe": float(cost_terms.get("Cost_Repulsive_Safe", 0.0)),
        "Cost_Repulsive_Collision": float(cost_terms.get("Cost_Repulsive_Collision", 0.0)),
        "Cost_Repulsive": float(cost_terms.get("Cost_Repulsive", 0.0)),
        "Cost_Control": float(cost_terms.get("Cost_Control", 0.0)),
        "Cost_Total": float(_mpc_cost_total(cost_terms)),
        "solver_status": str(runtime_status.get("solver_status", "unknown")),
        "solve_time_ms": float(runtime_status.get("solve_time_ms", 0.0)),
    }
    cost_history.append(sample)


def _write_mpc_cost_csv(csv_path: str, cost_history: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "sample_index",
        "sim_time_s",
        "Cost_Total",
        "Cost_ref",
        "Cost_Lane",
        "Cost_LaneCenter",
        "Cost_RoadBoundary",
        "Cost_LaneBoundary",
        "Cost_Repulsive",
        "Cost_Repulsive_Safe",
        "Cost_Repulsive_Collision",
        "Cost_Control",
        "solver_status",
        "solve_time_ms",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in cost_history:
            row = {name: sample.get(name, "") for name in fieldnames}
            if row.get("Cost_RoadBoundary", "") == "":
                row["Cost_RoadBoundary"] = sample.get("Cost_LaneBoundary", "")
            if row.get("Cost_LaneBoundary", "") == "":
                row["Cost_LaneBoundary"] = sample.get("Cost_RoadBoundary", "")
            writer.writerow(row)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _write_rgb_png(path: str, width_px: int, height_px: int, pixels: bytearray) -> None:
    raw_rows = bytearray()
    stride = int(width_px) * 3
    for y_idx in range(int(height_px)):
        raw_rows.append(0)
        row_start = int(y_idx) * int(stride)
        raw_rows.extend(pixels[row_start: row_start + stride])
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", int(width_px), int(height_px), 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=6))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png_data)


def _draw_line_rgb(
    pixels: bytearray,
    width_px: int,
    height_px: int,
    start_xy: Tuple[int, int],
    end_xy: Tuple[int, int],
    color_rgb: Tuple[int, int, int],
) -> None:
    x0, y0 = int(start_xy[0]), int(start_xy[1])
    x1, y1 = int(end_xy[0]), int(end_xy[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < int(width_px) and 0 <= y0 < int(height_px):
            idx = (y0 * int(width_px) + x0) * 3
            pixels[idx: idx + 3] = bytes(color_rgb)
        if x0 == x1 and y0 == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x0 += sx
        if err2 <= dx:
            err += dx
            y0 += sy


def _write_mpc_cost_plot_fallback_png(
    plot_path: str,
    x_values: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float], Tuple[int, int, int]]],
) -> None:
    width_px = 1200
    height_px = 1320
    pixels = bytearray([255] * int(width_px) * int(height_px) * 3)
    left_pad = 80
    right_pad = 40
    top_pad = 30
    bottom_pad = 40
    panel_gap = 22
    panel_count = max(1, len(series))
    panel_height = max(80, (height_px - top_pad - bottom_pad - panel_gap * (panel_count - 1)) // panel_count)

    x_min = min(x_values) if len(x_values) > 0 else 0.0
    x_max = max(x_values) if len(x_values) > 0 else 1.0
    if abs(float(x_max) - float(x_min)) <= 1e-9:
        x_max = float(x_min) + 1.0

    for panel_idx, (_label, values, color_rgb) in enumerate(series):
        y_top = top_pad + panel_idx * (panel_height + panel_gap)
        y_bottom = y_top + panel_height
        x_left = left_pad
        x_right = width_px - right_pad
        for x_pos in range(x_left, x_right + 1):
            for y_pos in (y_top, y_bottom):
                idx = (y_pos * width_px + x_pos) * 3
                pixels[idx: idx + 3] = b"\xDD\xDD\xDD"
        for y_pos in range(y_top, y_bottom + 1):
            for x_pos in (x_left, x_right):
                idx = (y_pos * width_px + x_pos) * 3
                pixels[idx: idx + 3] = b"\xDD\xDD\xDD"

        safe_values = [_safe_plot_float(value, 0.0) for value in values]
        if len(safe_values) == 0:
            continue
        y_min = min(safe_values)
        y_max = max(safe_values)
        if abs(float(y_max) - float(y_min)) <= 1e-9:
            y_max = float(y_min) + 1.0

        points: List[Tuple[int, int]] = []
        for sample_idx, value in enumerate(safe_values):
            x_value = float(x_values[sample_idx]) if sample_idx < len(x_values) else float(sample_idx)
            x_norm = (float(x_value) - float(x_min)) / max(1e-9, float(x_max) - float(x_min))
            y_norm = (_safe_plot_float(value, 0.0) - float(y_min)) / max(1e-9, float(y_max) - float(y_min))
            x_px = int(round(x_left + x_norm * (x_right - x_left)))
            y_px = int(round(y_bottom - y_norm * (y_bottom - y_top)))
            points.append((x_px, y_px))
        for first, second in zip(points, points[1:]):
            _draw_line_rgb(pixels, width_px, height_px, first, second, color_rgb)

    _write_rgb_png(plot_path, width_px, height_px, pixels)


def _write_mpc_cost_plot(
    plot_path: str,
    cost_history: Sequence[Mapping[str, object]],
    scenario_name: str,
    *,
    prefer_fast_fallback: bool = False,
) -> None:
    x_values = [_safe_plot_float(sample.get("sim_time_s", index), float(index)) for index, sample in enumerate(cost_history)]
    total_cost = [_safe_plot_float(sample.get("Cost_Total", 0.0)) for sample in cost_history]
    cost_ref = [_safe_plot_float(sample.get("Cost_ref", 0.0)) for sample in cost_history]
    cost_lane = [_safe_plot_float(sample.get("Cost_LaneCenter", 0.0)) for sample in cost_history]
    cost_road_boundary = [
        _safe_plot_float(sample.get("Cost_RoadBoundary", sample.get("Cost_LaneBoundary", 0.0)))
        for sample in cost_history
    ]
    cost_control = [_safe_plot_float(sample.get("Cost_Control", 0.0)) for sample in cost_history]
    cost_collision = [_safe_plot_float(sample.get("Cost_Repulsive_Collision", 0.0)) for sample in cost_history]

    series = [
        ("Cost_Total", total_cost, (0, 0, 0), "black"),
        ("Cost_ref", cost_ref, (31, 119, 180), "tab:blue"),
        ("Cost_LaneCenter", cost_lane, (44, 160, 44), "tab:green"),
        ("Cost_RoadBoundary", cost_road_boundary, (214, 39, 40), "tab:red"),
        ("Cost_Repulsive_Collision", cost_collision, (148, 103, 189), "tab:purple"),
        ("Cost_Control", cost_control, (255, 127, 14), "tab:orange"),
    ]
    fallback_series = [
        (label, values, fallback_color)
        for label, values, fallback_color, _color in series
    ]
    _write_mpc_cost_plot_fallback_png(plot_path, x_values, fallback_series)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(series), 1, figsize=(12.0, 13.0), sharex=True)
        for ax, (label, values, _fallback_color, color) in zip(axes, series):
            ax.plot(x_values, values, label=label, color=color, linewidth=1.6)
            ax.set_ylabel("Cost")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        axes[0].set_title(f"MPC Cost History: {scenario_name}")
        axes[-1].set_xlabel("Simulation Time [s]")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[MPC] Matplotlib cost plot failed; keeping fallback PNG: {exc}")



def _write_mpc_cost_artifacts(
    cost_history: Sequence[Mapping[str, object]],
    scenario_cfg: Mapping[str, object],
    *,
    prefer_fast_plot: bool = False,
) -> Dict[str, str]:
    if len(cost_history) == 0:
        return {}

    artifact_dir = _scenario_artifact_dir(scenario_cfg)
    os.makedirs(artifact_dir, exist_ok=True)
    csv_path = os.path.join(artifact_dir, "mpc_cost_history.csv")
    plot_path = os.path.join(artifact_dir, "mpc_cost_plot.png")
    _write_mpc_cost_csv(csv_path, cost_history)
    _write_mpc_cost_plot(
        plot_path,
        cost_history,
        str(scenario_cfg.get("name", "scenario")),
        prefer_fast_fallback=bool(prefer_fast_plot),
    )
    return {"csv_path": csv_path, "plot_path": plot_path}


def _best_partial_match(candidates: List[tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _find_environment_object_by_name(world, carla, name: str):
    name_lower = str(name).lower()
    partial_candidates = []
    for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
        env_name = str(env_obj.name).lower()
        if env_name == name_lower:
            return env_obj
        if name_lower in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_partial_match(partial_candidates)


def _find_environment_objects_by_prefix(world, carla, prefix: str) -> List[Any]:
    prefix_lower = str(prefix).lower()
    matches: List[Any] = []
    for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
        env_name = str(env_obj.name).lower()
        if env_name.startswith(prefix_lower):
            matches.append(env_obj)
    matches.sort(key=lambda item: str(getattr(item, "name", "")).lower())
    return matches


def _find_actor_by_name(world, name: str):
    name_lower = str(name).lower()
    partial_candidates = []
    for actor in world.get_actors():
        attr_name = str(actor.attributes.get("name", "")).lower()
        role_name = str(actor.attributes.get("role_name", "")).lower()
        type_id = str(actor.type_id).lower()
        if attr_name == name_lower or role_name == name_lower or type_id.endswith(name_lower):
            return actor
        if name_lower in attr_name:
            partial_candidates.append((len(attr_name), actor))
        if name_lower in role_name:
            partial_candidates.append((len(role_name), actor))
        if name_lower in type_id:
            partial_candidates.append((len(type_id), actor))
    return _best_partial_match(partial_candidates)


def _resolve_anchor_transform(world, carla, name: str):
    transform = _find_anchor_transform(world, carla, name)
    if transform is not None:
        return transform

    raise RuntimeError(
        f"Could not find an EnvironmentObject or Actor named '{name}'. "
        "Make sure the cube exists in the Unreal level and its name contains the keyword."
    )


def _find_anchor_transform(world, carla, name: str):
    env_obj = _find_environment_object_by_name(world, carla, name)
    if env_obj is not None:
        return env_obj.transform

    actor = _find_actor_by_name(world, name)
    if actor is not None:
        return actor.get_transform()

    return None


def _get_location_from_anchor(world, carla, name: str):
    """Check EnvironmentObjects first, then Actors, and return a CARLA location."""
    transform = _find_anchor_transform(world, carla, name)
    if transform is not None:
        return transform.location

    print(f"[CARLA GLOBAL ROUTE OUTPUT] Anchor '{name}' was not found in the loaded world.")
    return None


def _location_distance_sq(location_a, location_b) -> float:
    return (
        (float(location_a.x) - float(location_b.x)) ** 2
        + (float(location_a.y) - float(location_b.y)) ** 2
        + (float(location_a.z) - float(location_b.z)) ** 2
    )


def _fallback_route_anchor_transforms_from_spawn_points(world_map) -> Tuple[Any | None, Any | None]:
    spawn_points = list(world_map.get_spawn_points() or [])
    if len(spawn_points) < 2:
        return None, None

    best_start = spawn_points[0]
    best_goal = spawn_points[1]
    best_distance_sq = _location_distance_sq(best_start.location, best_goal.location)
    for start_transform in spawn_points:
        for goal_transform in spawn_points:
            if start_transform is goal_transform:
                continue
            distance_sq = _location_distance_sq(start_transform.location, goal_transform.location)
            if distance_sq > best_distance_sq:
                best_distance_sq = distance_sq
                best_start = start_transform
                best_goal = goal_transform
    return best_start, best_goal


def _resolve_route_anchor_transforms(world, carla, world_map, anchors_cfg: Mapping[str, object]):
    ego_anchor_name = str(anchors_cfg.get("ego_spawn", "cav_spawn"))
    destination_anchor_name = str(anchors_cfg.get("final_destination", "final_destination"))
    spawn_anchor = _find_anchor_transform(world, carla, ego_anchor_name)
    destination_anchor = _find_anchor_transform(world, carla, destination_anchor_name)
    if spawn_anchor is not None and destination_anchor is not None:
        return spawn_anchor, destination_anchor, ego_anchor_name, destination_anchor_name

    if spawn_anchor is None:
        print(f"[CARLA GLOBAL ROUTE OUTPUT] Anchor '{ego_anchor_name}' was not found in the loaded world.")
    if destination_anchor is None:
        print(f"[CARLA GLOBAL ROUTE OUTPUT] Anchor '{destination_anchor_name}' was not found in the loaded world.")

    fallback_start, fallback_goal = _fallback_route_anchor_transforms_from_spawn_points(world_map)
    if fallback_start is None or fallback_goal is None:
        missing_anchor_names = []
        if spawn_anchor is None:
            missing_anchor_names.append(f"ego_spawn='{ego_anchor_name}'")
        if destination_anchor is None:
            missing_anchor_names.append(f"final_destination='{destination_anchor_name}'")
        raise RuntimeError(
            "Could not find the required global-route anchors and the CARLA map "
            f"does not expose enough spawn points for fallback routing. {', '.join(missing_anchor_names)}."
        )

    if spawn_anchor is None:
        spawn_anchor = fallback_start
    if destination_anchor is None:
        destination_anchor = fallback_goal
    print(
        "[CARLA GLOBAL ROUTE OUTPUT] Falling back to CARLA map spawn points for missing route anchors: "
        f"start=({float(spawn_anchor.location.x):.3f}, {float(spawn_anchor.location.y):.3f}, {float(spawn_anchor.location.z):.3f}), "
        f"goal=({float(destination_anchor.location.x):.3f}, {float(destination_anchor.location.y):.3f}, {float(destination_anchor.location.z):.3f})."
    )
    return spawn_anchor, destination_anchor, ego_anchor_name, destination_anchor_name


def _print_anchor_lookup(world, carla, name: str) -> None:
    env_obj = _find_environment_object_by_name(world, carla, name)
    if env_obj is not None:
        transform = env_obj.transform
        print(
            f"[CARLA SCENARIO] Found anchor '{name}' as EnvironmentObject "
            f"'{env_obj.name}' at "
            f"({transform.location.x:.3f}, {transform.location.y:.3f}, {transform.location.z:.3f}) "
            f"yaw={transform.rotation.yaw:.3f}"
        )
        return

    actor = _find_actor_by_name(world, name)
    if actor is not None:
        transform = actor.get_transform()
        print(
            f"[CARLA SCENARIO] Found anchor '{name}' as Actor "
            f"'{actor.type_id}' at "
            f"({transform.location.x:.3f}, {transform.location.y:.3f}, {transform.location.z:.3f}) "
            f"yaw={transform.rotation.yaw:.3f}"
        )
        return

    print(f"[CARLA SCENARIO] Anchor '{name}' was not found in the loaded world.")


def _align_transform_to_lane(map_obj, carla, transform):
    waypoint = map_obj.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return transform, None
    return waypoint.transform, waypoint


def _vehicle_speed_mps(vehicle) -> float:
    velocity = vehicle.get_velocity()
    return float(math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z))


def _signal_state_requires_stop(signal_state: object) -> bool:
    return str(signal_state or "").strip().lower() in {"red", "yellow", "amber"}


def _fallback_signal_stop_target_from_ego(
    *,
    world_map: Any,
    carla: Any,
    ego_transform: Any,
    signal_context: Mapping[str, object] | None,
    search_distance_m: float,
    stop_buffer_m: float,
) -> Dict[str, object] | None:
    if not isinstance(signal_context, Mapping):
        return None
    if not _signal_state_requires_stop(signal_context.get("signal_state", "")):
        return None
    try:
        signal_forward_m = float(signal_context.get("signal_forward_m", float("inf")))
    except Exception:
        signal_forward_m = float("inf")
    if math.isfinite(float(signal_forward_m)) and float(signal_forward_m) < -2.0:
        return None
    try:
        signal_lateral_m = float(signal_context.get("signal_lateral_m", 0.0))
    except Exception:
        signal_lateral_m = 0.0
    if math.isfinite(float(signal_lateral_m)) and abs(float(signal_lateral_m)) > 8.0:
        return None
    try:
        signal_distance_m = float(signal_context.get("signal_distance_m", search_distance_m))
    except Exception:
        signal_distance_m = float(search_distance_m)
    if not math.isfinite(float(signal_distance_m)):
        signal_distance_m = float(search_distance_m)
    if float(signal_distance_m) <= 0.0:
        return None

    raw_stop_distance_m = float(signal_distance_m) - max(0.0, float(stop_buffer_m))
    if float(raw_stop_distance_m) <= 1.0:
        return None
    forward_distance_m = max(
        3.0,
        min(float(search_distance_m), float(raw_stop_distance_m)),
    )
    if float(forward_distance_m) <= 0.0:
        return None
    try:
        ego_waypoint = world_map.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_waypoint is None:
            return None
        next_waypoints = ego_waypoint.next(float(forward_distance_m))
        stop_waypoint = next_waypoints[0] if next_waypoints else ego_waypoint
        stop_transform = stop_waypoint.transform
        return {
            "x_m": float(stop_transform.location.x),
            "y_m": float(stop_transform.location.y),
            "heading_rad": float(math.radians(stop_transform.rotation.yaw)),
            "lane_id": int(getattr(stop_waypoint, "lane_id", 0)),
            "road_id": int(getattr(stop_waypoint, "road_id", 0)),
            "section_id": int(getattr(stop_waypoint, "section_id", 0)),
            "distance_m": float(forward_distance_m),
            "source": "fallback_signal_stop_target",
            "signal_distance_m": float(signal_distance_m),
        }
    except Exception:
        return None


def _solver_speed_upper_bound_mps(traffic_rule_vmax_mps: float, margin_mps: float = 1.0) -> float:
    return max(0.0, float(traffic_rule_vmax_mps)) + max(0.0, float(margin_mps))


def _should_force_stationary_release_replan(
    *,
    ego_speed_mps: float,
    zero_speed_threshold_mps: float,
    current_behavior: str,
    target_v_ref_mps: float,
    current_acceleration_mps2: float = 0.0,
    cached_step_acceleration_mps2: float | None = None,
) -> bool:
    zero_speed_threshold_mps = max(0.0, float(zero_speed_threshold_mps))
    release_speed_threshold_mps = max(float(zero_speed_threshold_mps), 1.0)
    target_v_ref_mps = float(target_v_ref_mps)
    ego_speed_mps = float(ego_speed_mps)
    stale_braking_active = (
        float(current_acceleration_mps2) < -0.05
        or (
            cached_step_acceleration_mps2 is not None
            and float(cached_step_acceleration_mps2) < -0.05
        )
    )
    return (
        not bool(is_stop_decision(current_behavior))
        and float(target_v_ref_mps) > float(zero_speed_threshold_mps)
        and (
            float(ego_speed_mps) <= float(zero_speed_threshold_mps)
            or (
                float(ego_speed_mps) <= float(release_speed_threshold_mps)
                and bool(stale_braking_active)
            )
        )
    )


def _current_cached_step_acceleration_mps2(
    cached_control_sequence: np.ndarray | None,
    cached_control_step_idx: int,
) -> float | None:
    if cached_control_sequence is None or len(cached_control_sequence) == 0:
        return None
    try:
        control_step_idx = min(
            int(max(0, int(cached_control_step_idx))),
            int(len(cached_control_sequence) - 1),
        )
        return float(cached_control_sequence[control_step_idx][0])
    except Exception:
        return None


def _allowed_lane_ids_from_context(
    local_context: Mapping[str, object] | None,
    fallback_lane_count: int,
    fallback_lane_id: int | None = None,
) -> list[int]:
    local_context = dict(local_context or {})
    lane_ids = [
        int(lane_id)
        for lane_id in list(local_context.get("lane_ids", []))
        if int(lane_id) != 0
    ]
    if len(lane_ids) > 0:
        return list(dict.fromkeys(lane_ids))

    context_lane_id = int(local_context.get("lane_id", 0))
    if int(context_lane_id) != 0:
        return [int(context_lane_id)]
    if fallback_lane_id is not None and int(fallback_lane_id) != 0:
        return [int(fallback_lane_id)]
    del fallback_lane_count
    return []


def _clamp_optional_lane_id_to_allowed(
    raw_lane_id: int | None,
    allowed_lane_ids: Sequence[int],
) -> int:
    if raw_lane_id is None or int(raw_lane_id) == 0:
        return 0
    return _clamp_lane_id_to_allowed(int(raw_lane_id), allowed_lane_ids)


def _clamp_lane_id_to_allowed(raw_lane_id: int | None, allowed_lane_ids: Sequence[int]) -> int:
    if len(allowed_lane_ids) == 0:
        return int(raw_lane_id or 0)
    lane_id = int(raw_lane_id if raw_lane_id is not None else allowed_lane_ids[0])
    if lane_id in allowed_lane_ids:
        return int(lane_id)
    return int(
        min(
            allowed_lane_ids,
            key=lambda candidate_lane_id: abs(int(candidate_lane_id) - int(lane_id)),
        )
    )


def _selected_lane_id_for_behavior_step(
    *,
    planner_output: Mapping[str, object] | None,
    current_lane_id: int,
    allowed_lane_ids: Sequence[int],
    reroute_lane_override: int | None = None,
) -> int:
    if reroute_lane_override is not None and int(reroute_lane_override) != 0:
        candidate_lane_id = int(reroute_lane_override)
    else:
        normalized_output = dict(planner_output or {})
        normalized_decision = normalize_behavior_decision(
            normalized_output.get("decision", "lane_follow")
        )
        candidate_lane_id = int(
            normalized_output.get(
                "selected_lane_id",
                normalized_output.get("target_lane_id", int(current_lane_id)),
            )
        )
        if (
            str(normalized_decision) in {"lane_change_left", "lane_change_right"}
            and int(candidate_lane_id) != 0
        ):
            return int(candidate_lane_id)
    return _clamp_lane_id_to_allowed(
        int(candidate_lane_id),
        allowed_lane_ids,
    )


def _route_tracking_target_lane_after_reroute(
    *,
    current_behavior: str,
    planner_selected_lane_id: int,
    route_optimal_lane_id: int,
    reroute_route_follow_latched: bool,
) -> tuple[int, bool, bool]:
    """
    Keep the blue-dot and MPC reference attached to the updated route.

    Right after a reroute, the behavior planner's selected lane can still
    reflect the pre-reroute route for a short time. When that happens, keep
    the rolling target on the rerouted route until the planner-selected lane
    catches up to the new route lane.
    """
    normalized_behavior = normalize_behavior_decision(current_behavior)
    selected_lane_id = int(planner_selected_lane_id)
    route_lane_id = int(route_optimal_lane_id)
    has_route_lane = int(route_lane_id) != 0

    if str(normalized_behavior) == "reroute":
        if bool(has_route_lane):
            return int(route_lane_id), True, True
        return int(selected_lane_id), True, False

    if not bool(has_route_lane):
        return int(selected_lane_id), False, False

    if int(selected_lane_id) == int(route_lane_id):
        return int(route_lane_id), True, False

    if bool(reroute_route_follow_latched) and str(normalized_behavior) == "lane_follow":
        return int(route_lane_id), True, True

    return int(selected_lane_id), False, bool(reroute_route_follow_latched)


def _reference_lane_from_blue_dot(
    *,
    planner_reference_lane_id: int,
    temporary_destination_state: Sequence[float] | None,
    allowed_lane_ids: Sequence[int],
    route_optimal_lane_id: int,
    should_follow_global_route_lane: bool,
) -> tuple[int, bool]:
    reference_lane_id = int(planner_reference_lane_id)
    follow_route_lane = bool(should_follow_global_route_lane)

    if temporary_destination_state is not None and len(temporary_destination_state) >= 5:
        try:
            blue_dot_lane_id = _clamp_optional_lane_id_to_allowed(
                int(round(float(temporary_destination_state[4]))),
                allowed_lane_ids,
            )
        except Exception:
            blue_dot_lane_id = 0
        if int(blue_dot_lane_id) != 0:
            reference_lane_id = int(blue_dot_lane_id)
            if int(reference_lane_id) != int(route_optimal_lane_id):
                follow_route_lane = False

    return int(reference_lane_id), bool(follow_route_lane)


def _has_pending_lane_closure_reroute_request(message_path: str | None) -> bool:
    normalized_message_path = str(message_path or "").strip() or CP_MESSAGE_PATH
    try:
        return len(load_lane_closure_messages(message_path=normalized_message_path)) > 0
    except Exception:
        return False


def _behavior_runtime_value(
    behavior_runtime_cfg: Mapping[str, object] | None,
    legacy_rule_based_cfg: Mapping[str, object] | None,
    *,
    key: str,
    legacy_key: str | None = None,
    default: object,
) -> object:
    runtime_cfg = dict(behavior_runtime_cfg or {})
    legacy_cfg = dict(legacy_rule_based_cfg or {})

    if key in runtime_cfg:
        return runtime_cfg[key]
    if legacy_key is not None and legacy_key in legacy_cfg:
        return legacy_cfg[legacy_key]
    if key in legacy_cfg:
        return legacy_cfg[key]
    return default


def _world_state_from_vehicle(vehicle, zero_speed_threshold_mps: float = 0.0) -> List[float]:
    transform = vehicle.get_transform()
    location = transform.location
    speed_mps = _vehicle_speed_mps(vehicle)
    if float(speed_mps) < max(0.0, float(zero_speed_threshold_mps)):
        speed_mps = 0.0
    yaw_rad = math.radians(float(transform.rotation.yaw))
    return [float(location.x), float(location.y), float(speed_mps), float(yaw_rad)]


def _clone_transform(transform, carla):
    return carla.Transform(
        carla.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def _destroy_stale_role_vehicles(world, role_name: str) -> int:
    destroyed_count = 0
    for actor in world.get_actors().filter("vehicle.*"):
        actor_role_name = str(getattr(actor, "attributes", {}).get("role_name", "")).strip().lower()
        if actor_role_name != str(role_name).strip().lower():
            continue
        try:
            actor.destroy()
            destroyed_count += 1
        except Exception:
            pass
    return destroyed_count


def _nearby_vehicle_descriptions(world, location, radius_m: float = 8.0) -> List[str]:
    descriptions: List[str] = []
    radius_sq = float(radius_m) * float(radius_m)
    for actor in world.get_actors().filter("vehicle.*"):
        actor_location = actor.get_location()
        dx = float(actor_location.x) - float(location.x)
        dy = float(actor_location.y) - float(location.y)
        dz = float(actor_location.z) - float(location.z)
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq > radius_sq:
            continue
        descriptions.append(
            f"{actor.type_id} role={str(actor.attributes.get('role_name', ''))!r} "
            f"at ({float(actor_location.x):.2f}, {float(actor_location.y):.2f}, {float(actor_location.z):.2f}) "
            f"d={math.sqrt(dist_sq):.2f}m"
        )
    descriptions.sort()
    return descriptions


def _spawn_vehicle(
    world,
    blueprint_library,
    scenario_cfg: Mapping[str, object],
    *,
    carla,
    lane_transform,
    anchor_transform=None,
):
    ego_cfg = dict(scenario_cfg.get("ego", {}))
    blueprint_filter = str(ego_cfg.get("blueprint", "vehicle.tesla.model3"))
    role_name = str(ego_cfg.get("role_name", "ego"))
    z_offset_m = float(ego_cfg.get("spawn_z_offset_m", 1.0))

    blueprints = blueprint_library.filter(blueprint_filter)
    if not blueprints:
        raise RuntimeError(f"No CARLA vehicle blueprint matched '{blueprint_filter}'.")
    blueprint = blueprints[0]
    blueprint.set_attribute("role_name", role_name)
    color_rgb = ego_cfg.get("color_rgb", None)
    if color_rgb and blueprint.has_attribute("color"):
        blueprint.set_attribute("color", str(color_rgb))

    destroyed_stale_count = _destroy_stale_role_vehicles(world, role_name)
    if destroyed_stale_count > 0:
        print(
            f"[CARLA SCENARIO] Removed {destroyed_stale_count} stale vehicle(s) "
            f"with role_name='{role_name}' before spawning the ego vehicle."
        )

    candidate_transforms = []
    if lane_transform is not None:
        candidate_transforms.append(_clone_transform(lane_transform, carla))
    if anchor_transform is not None:
        candidate_transforms.append(_clone_transform(anchor_transform, carla))
    if anchor_transform is not None and lane_transform is not None:
        mixed_transform = _clone_transform(anchor_transform, carla)
        mixed_transform.rotation = carla.Rotation(
            pitch=float(lane_transform.rotation.pitch),
            yaw=float(lane_transform.rotation.yaw),
            roll=float(lane_transform.rotation.roll),
        )
        candidate_transforms.append(mixed_transform)

    spawn_candidates = [0.0, z_offset_m, z_offset_m + 0.5, z_offset_m + 1.0, z_offset_m + 2.0, z_offset_m + 3.0]
    for base_transform in candidate_transforms:
        for extra_z_m in spawn_candidates:
            attempt_transform = _clone_transform(base_transform, carla)
            attempt_transform.location.z += float(extra_z_m)
            vehicle = world.try_spawn_actor(blueprint, attempt_transform)
            if vehicle is not None:
                return vehicle

    diagnostic_location = None
    if lane_transform is not None:
        diagnostic_location = lane_transform.location
    elif anchor_transform is not None:
        diagnostic_location = anchor_transform.location
    nearby_vehicles = (
        _nearby_vehicle_descriptions(world, diagnostic_location)
        if diagnostic_location is not None
        else []
    )
    if diagnostic_location is not None:
        print(
            "[CARLA SCENARIO] Ego spawn failed near "
            f"({float(diagnostic_location.x):.3f}, {float(diagnostic_location.y):.3f}, {float(diagnostic_location.z):.3f})."
        )
    if lane_transform is not None:
        print(
            "[CARLA SCENARIO] Lane-aligned ego spawn candidate "
            f"({float(lane_transform.location.x):.3f}, {float(lane_transform.location.y):.3f}, {float(lane_transform.location.z):.3f}) "
            f"yaw={float(lane_transform.rotation.yaw):.3f}"
        )
    if anchor_transform is not None:
        print(
            "[CARLA SCENARIO] Anchor ego spawn candidate "
            f"({float(anchor_transform.location.x):.3f}, {float(anchor_transform.location.y):.3f}, {float(anchor_transform.location.z):.3f}) "
            f"yaw={float(anchor_transform.rotation.yaw):.3f}"
        )
    for description in nearby_vehicles[:10]:
        print(f"[CARLA SCENARIO] Nearby spawn blocker: {description}")
    raise RuntimeError("Failed to spawn the ego vehicle at the requested anchor.")


def _spawn_fallback_npc_traffic(
    *,
    world,
    world_map,
    blueprint_library,
    carla,
    ego_vehicle,
    traffic_manager_port: int,
    count: int,
    min_distance_m: float,
    max_distance_m: float,
) -> List[Any]:
    if int(count) <= 0 or ego_vehicle is None:
        return []
    try:
        ego_location = ego_vehicle.get_transform().location
    except Exception:
        return []
    vehicle_blueprints = list(blueprint_library.filter("vehicle.*"))
    vehicle_blueprints = [
        blueprint for blueprint in vehicle_blueprints
        if not str(getattr(blueprint, "id", "")).startswith("vehicle.carlamotors.firetruck")
    ] or list(blueprint_library.filter("vehicle.tesla.model3"))
    if not vehicle_blueprints:
        return []

    spawn_points = list(world_map.get_spawn_points() or [])
    ranked_spawn_points = []
    min_distance_sq = float(min_distance_m) * float(min_distance_m)
    max_distance_sq = float(max_distance_m) * float(max_distance_m)
    for spawn_transform in spawn_points:
        distance_sq = _location_distance_sq(spawn_transform.location, ego_location)
        if distance_sq < min_distance_sq or distance_sq > max_distance_sq:
            continue
        ranked_spawn_points.append((float(distance_sq), spawn_transform))
    ranked_spawn_points.sort(key=lambda item: float(item[0]))

    spawned_actors: List[Any] = []
    for spawn_index, (_distance_sq, spawn_transform) in enumerate(ranked_spawn_points):
        if len(spawned_actors) >= int(count):
            break
        blueprint = vehicle_blueprints[spawn_index % len(vehicle_blueprints)]
        try:
            blueprint = blueprint_library.find(str(getattr(blueprint, "id", "vehicle.tesla.model3")))
        except Exception:
            pass
        if hasattr(blueprint, "has_attribute") and blueprint.has_attribute("role_name"):
            try:
                blueprint.set_attribute("role_name", f"fallback_npc_{len(spawned_actors)}")
            except Exception:
                pass
        if hasattr(blueprint, "has_attribute") and blueprint.has_attribute("color"):
            try:
                colors = blueprint.get_attribute("color").recommended_values
                if colors:
                    blueprint.set_attribute("color", colors[spawn_index % len(colors)])
            except Exception:
                pass
        spawn_attempt = _clone_transform(spawn_transform, carla)
        spawn_attempt.location.z += 0.25
        actor = world.try_spawn_actor(blueprint, spawn_attempt)
        if actor is None:
            continue
        try:
            actor.set_autopilot(True, int(traffic_manager_port))
        except Exception:
            try:
                actor.set_autopilot(True)
            except Exception:
                pass
        spawned_actors.append(actor)

    if spawned_actors:
        print(
            "[CARLA SCENARIO] Spawned "
            f"{len(spawned_actors)} fallback NPC vehicle(s) near ego because "
            "the loaded map did not provide scenario vehicle markers."
        )
    return spawned_actors


def _camera_blueprint(world, width_px: int, height_px: int, fov_deg: float):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(int(width_px)))
    blueprint.set_attribute("image_size_y", str(int(height_px)))
    blueprint.set_attribute("fov", str(float(fov_deg)))
    return blueprint


def _spawn_camera(world, carla, blueprint, transform, parent=None):
    if parent is None:
        sensor = world.spawn_actor(
            blueprint,
            transform,
        )
    else:
        sensor = world.spawn_actor(
            blueprint,
            transform,
            attach_to=parent,
            attachment_type=carla.AttachmentType.Rigid,
        )
    image_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)

    def _on_image(image) -> None:
        if image_queue.full():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                pass
        image_queue.put(image)

    sensor.listen(_on_image)
    return sensor, image_queue


def _spawn_collision_sensor(world, carla, ego_vehicle, metrics_recorder: EvaluationMetricsRecorder):
    blueprint = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(
        blueprint,
        carla.Transform(),
        attach_to=ego_vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    def _on_collision(event) -> None:
        other_actor = getattr(event, "other_actor", None)
        other_actor_id = getattr(other_actor, "id", "")
        frame_id = getattr(event, "frame", "")

        # simulation time from event timestamp
        ts = getattr(event, "timestamp", None)
        sim_time_s = float(getattr(ts, "elapsed_seconds", 0.0)) if ts is not None else None

        # ego state at collision time
        try:
            loc = ego_vehicle.get_location()
            vel = ego_vehicle.get_velocity()
            ego_x = float(loc.x)
            ego_y = float(loc.y)
            ego_speed_mps = float((vel.x**2 + vel.y**2 + vel.z**2) ** 0.5)
        except Exception:
            ego_x = ego_y = ego_speed_mps = None

        # actor type: vehicle / pedestrian / static / unknown
        type_id = str(getattr(other_actor, "type_id", "unknown"))
        if "vehicle" in type_id:
            actor_type = "vehicle"
        elif "walker" in type_id or "pedestrian" in type_id:
            actor_type = "pedestrian"
        elif "static" in type_id or "prop" in type_id:
            actor_type = "static"
        else:
            actor_type = type_id[:40]

        # impulse magnitude
        impulse = getattr(event, "normal_impulse", None)
        if impulse is not None:
            impulse_mag = float((impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5)
        else:
            impulse_mag = None

        metrics_recorder.record_collision(
            event_id=f"{frame_id}:{other_actor_id}",
            sim_time_s=sim_time_s,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_speed_mps=ego_speed_mps,
            other_actor_type=actor_type,
            impulse_magnitude=impulse_mag,
        )

    sensor.listen(_on_collision)
    return sensor


def _camera_calibration_matrix(width_px: int, height_px: int, fov_deg: float) -> np.ndarray:
    focal_length_px = float(width_px) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    calibration_matrix = np.identity(3)
    calibration_matrix[0, 0] = focal_length_px
    calibration_matrix[1, 1] = focal_length_px
    calibration_matrix[0, 2] = float(width_px) / 2.0
    calibration_matrix[1, 2] = float(height_px) / 2.0
    return calibration_matrix


def _world_fixed_topdown_transform(
    carla,
    focus_points_xy: Sequence[Sequence[float]],
    image_width_px: int,
    image_height_px: int,
    fov_deg: float,
    min_height_m: float,
    padding_m: float,
):
    valid_points = [
        (float(point[0]), float(point[1]))
        for point in focus_points_xy
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if len(valid_points) == 0:
        return carla.Transform(
            carla.Location(x=0.0, y=0.0, z=float(min_height_m)),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

    x_values_m = [float(point[0]) for point in valid_points]
    y_values_m = [float(point[1]) for point in valid_points]
    center_x_m = 0.5 * (min(x_values_m) + max(x_values_m))
    center_y_m = 0.5 * (min(y_values_m) + max(y_values_m))
    half_span_x_m = 0.5 * max(1.0, max(x_values_m) - min(x_values_m)) + max(0.0, float(padding_m))
    half_span_y_m = 0.5 * max(1.0, max(y_values_m) - min(y_values_m)) + max(0.0, float(padding_m))

    horizontal_fov_rad = math.radians(float(fov_deg))
    vertical_fov_rad = 2.0 * math.atan(
        math.tan(0.5 * horizontal_fov_rad) * float(image_height_px) / max(1.0, float(image_width_px))
    )
    required_height_x_m = half_span_x_m / max(1e-6, math.tan(0.5 * horizontal_fov_rad))
    required_height_y_m = half_span_y_m / max(1e-6, math.tan(0.5 * vertical_fov_rad))
    camera_height_m = max(float(min_height_m), float(required_height_x_m), float(required_height_y_m))

    return carla.Transform(
        carla.Location(x=float(center_x_m), y=float(center_y_m), z=float(camera_height_m)),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )


def _project_world_to_image(
    camera_transform,
    calibration_matrix: np.ndarray,
    world_xyz: Sequence[float],
    image_width_px: int,
    image_height_px: int,
) -> tuple[int, int] | None:
    world_point = np.array(
        [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0],
        dtype=np.float64,
    )
    world_to_camera = np.array(camera_transform.get_inverse_matrix(), dtype=np.float64)
    point_camera = np.dot(world_to_camera, world_point)
    point_camera = np.array(
        [float(point_camera[1]), -float(point_camera[2]), float(point_camera[0])],
        dtype=np.float64,
    )
    if float(point_camera[2]) <= 1e-6:
        return None

    image_point = np.dot(calibration_matrix, point_camera)
    pixel_x = int(round(float(image_point[0] / image_point[2])))
    pixel_y = int(round(float(image_point[1] / image_point[2])))
    if pixel_x < 0 or pixel_x >= int(image_width_px) or pixel_y < 0 or pixel_y >= int(image_height_px):
        return None
    return pixel_x, pixel_y


def _draw_dotted_polyline(surface, points_px: Sequence[tuple[int, int]], color_rgb=(35, 210, 70), dot_spacing_px: int = 12, radius_px: int = 3) -> None:
    if pygame is None or len(points_px) < 2:
        return

    spacing_px = max(2, int(dot_spacing_px))
    radius_px = max(1, int(radius_px))
    for idx in range(len(points_px) - 1):
        x0_px, y0_px = points_px[idx]
        x1_px, y1_px = points_px[idx + 1]
        dx_px = float(x1_px - x0_px)
        dy_px = float(y1_px - y0_px)
        segment_length_px = math.hypot(dx_px, dy_px)
        if segment_length_px <= 1e-6:
            pygame.draw.circle(surface, color_rgb, (int(x0_px), int(y0_px)), radius_px)
            continue
        steps = max(1, int(segment_length_px / float(spacing_px)))
        for step_idx in range(steps + 1):
            alpha = float(step_idx) / float(steps)
            dot_x_px = int(round(float(x0_px) + alpha * dx_px))
            dot_y_px = int(round(float(y0_px) + alpha * dy_px))
            pygame.draw.circle(surface, color_rgb, (dot_x_px, dot_y_px), radius_px)


def _split_projected_polyline_segments(
    projected_points: Sequence[tuple[int, int] | None],
) -> List[List[tuple[int, int]]]:
    segments: List[List[tuple[int, int]]] = []
    current_segment: List[tuple[int, int]] = []
    for point_px in projected_points:
        if point_px is None:
            if len(current_segment) >= 2:
                segments.append(list(current_segment))
            current_segment = []
            continue
        current_segment.append((int(point_px[0]), int(point_px[1])))
    if len(current_segment) >= 2:
        segments.append(list(current_segment))
    return segments


def _split_route_world_segments(
    route_points: Sequence[Sequence[float]],
    *,
    max_gap_m: float = 12.0,
) -> List[List[tuple[float, float]]]:
    segments: List[List[tuple[float, float]]] = []
    current_segment: List[tuple[float, float]] = []
    previous_point_xy: tuple[float, float] | None = None
    gap_threshold_m = max(0.5, float(max_gap_m))

    for point_xy in route_points:
        if len(point_xy) < 2:
            continue
        current_point_xy = (float(point_xy[0]), float(point_xy[1]))
        if (
            previous_point_xy is not None
            and math.hypot(
                current_point_xy[0] - previous_point_xy[0],
                current_point_xy[1] - previous_point_xy[1],
            ) > gap_threshold_m
        ):
            if len(current_segment) >= 2:
                segments.append(list(current_segment))
            current_segment = []
        current_segment.append(current_point_xy)
        previous_point_xy = current_point_xy

    if len(current_segment) >= 2:
        segments.append(list(current_segment))
    return segments


def _draw_world_debug_route(world, carla, route_points: Sequence[Sequence[float]], life_time_s: float = 60.0) -> None:
    if len(route_points) == 0:
        return
    debug = getattr(world, "debug", None)
    if debug is None:
        return
    yellow = carla.Color(255, 255, 0)
    for segment_world_points in _split_route_world_segments(route_points):
        elevated_points = [
            carla.Location(x=float(point_xy[0]), y=float(point_xy[1]), z=0.5)
            for point_xy in segment_world_points
        ]
        for idx, point in enumerate(elevated_points):
            debug.draw_point(
                point,
                size=0.15,
                color=yellow,
                life_time=float(life_time_s),
            )
            if idx == 0:
                continue
            debug.draw_line(
                elevated_points[idx - 1],
                point,
                thickness=0.1,
                color=yellow,
                life_time=float(life_time_s),
            )


def _route_points_for_visualization(
    route_points: Sequence[Sequence[float]] | None,
    *,
    enabled: bool,
) -> List[List[float]] | None:
    if not bool(enabled) or route_points is None:
        return None
    return [
        [float(point[0]), float(point[1])]
        for point in route_points
        if len(point) >= 2
    ]


def _draw_planning_overlay(
    *,
    surface,
    camera_transform,
    calibration_matrix: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    overlay_z_m: float,
    global_route_points: Sequence[Sequence[float]] | None,
    temporary_destination_state: Sequence[float] | None,
    planned_trajectory_states: Sequence[Sequence[float]] | None,
    predicted_obstacle_trajectories: Mapping[str, Sequence[Mapping[str, object]]] | None,
    obstacle_field_contours: Sequence[Mapping[str, object]] | None,
    obstacle_risk_ids: set | None = None,
) -> None:
    if pygame is None:
        return

    if global_route_points is not None and len(global_route_points) >= 2:
        for route_segment_world in _split_route_world_segments(global_route_points):
            projected_points_px = [
                _project_world_to_image(
                    camera_transform=camera_transform,
                    calibration_matrix=calibration_matrix,
                    world_xyz=[float(point_xy[0]), float(point_xy[1]), float(overlay_z_m)],
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                )
                for point_xy in route_segment_world
            ]
            for route_points_px in _split_projected_polyline_segments(projected_points_px):
                if len(route_points_px) < 2:
                    continue
                _draw_dotted_polyline(
                    surface,
                    route_points_px,
                    color_rgb=(255, 220, 60),
                    dot_spacing_px=12,
                    radius_px=4,
                )

    if temporary_destination_state is not None and len(temporary_destination_state) >= 2:
        temp_destination_pixel = _project_world_to_image(
            camera_transform=camera_transform,
            calibration_matrix=calibration_matrix,
            world_xyz=[
                float(temporary_destination_state[0]),
                float(temporary_destination_state[1]),
                float(overlay_z_m),
            ],
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )
        if temp_destination_pixel is not None:
            pygame.draw.circle(surface, (40, 170, 255), temp_destination_pixel, 7)
            pygame.draw.circle(surface, (20, 20, 20), temp_destination_pixel, 7, width=1)

    if planned_trajectory_states is not None and len(planned_trajectory_states) >= 2:
        trajectory_points_px: List[tuple[int, int]] = []
        for state in planned_trajectory_states:
            if len(state) < 2:
                continue
            pixel = _project_world_to_image(
                camera_transform=camera_transform,
                calibration_matrix=calibration_matrix,
                world_xyz=[float(state[0]), float(state[1]), float(overlay_z_m)],
                image_width_px=image_width_px,
                image_height_px=image_height_px,
            )
            if pixel is not None:
                trajectory_points_px.append(pixel)
        _draw_dotted_polyline(surface, trajectory_points_px)

    if predicted_obstacle_trajectories is not None:
        risky_ids = set(obstacle_risk_ids or [])
        for obs_id, predicted_points in list(predicted_obstacle_trajectories.items())[:12]:
            is_risky = str(obs_id) in risky_ids
            # Risky obstacles: bright red-orange; normal: muted orange
            traj_color = (255, 60, 30) if is_risky else (255, 145, 40)
            dot_radius = 3 if is_risky else 2
            predicted_points_px: List[tuple[int, int]] = []
            for point in list(predicted_points or [])[:20]:
                if not isinstance(point, Mapping):
                    continue
                try:
                    point_x_m = float(point.get("x", 0.0))
                    point_y_m = float(point.get("y", 0.0))
                except Exception:
                    continue
                pixel = _project_world_to_image(
                    camera_transform=camera_transform,
                    calibration_matrix=calibration_matrix,
                    world_xyz=[point_x_m, point_y_m, float(overlay_z_m)],
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                )
                if pixel is not None:
                    predicted_points_px.append(pixel)
            if len(predicted_points_px) >= 2:
                _draw_dotted_polyline(
                    surface,
                    predicted_points_px,
                    color_rgb=traj_color,
                    dot_spacing_px=10,
                    radius_px=dot_radius,
                )

    if obstacle_field_contours is not None:
        for contour in obstacle_field_contours:
            collision_points_px: List[tuple[int, int]] = []
            for world_xyz in contour.get("collision_points_world", []) or []:
                pixel = _project_world_to_image(
                    camera_transform=camera_transform,
                    calibration_matrix=calibration_matrix,
                    world_xyz=world_xyz,
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                )
                if pixel is not None:
                    collision_points_px.append(pixel)
            if len(collision_points_px) >= 3:
                pygame.draw.lines(surface, (255, 80, 80), True, collision_points_px, width=2)


def _draw_hud_lines(
    surface,
    font,
    lines: Sequence[str],
    top_left_margin_px: tuple[int, int],
    max_width_px: int | None = None,
) -> None:
    if pygame is None or font is None or len(lines) == 0:
        return

    x0_px, y0_px = int(top_left_margin_px[0]), int(top_left_margin_px[1])
    line_height_px = int(font.get_linesize())
    padding_px = 6
    text_width_limit_px = None
    if max_width_px is not None:
        text_width_limit_px = max(40, int(max_width_px) - 2 * padding_px)

    rendered_lines: List[str] = []
    for line in lines:
        rendered_line = str(line)
        if text_width_limit_px is not None and font.size(rendered_line)[0] > text_width_limit_px:
            while len(rendered_line) > 4 and font.size(rendered_line + "...")[0] > text_width_limit_px:
                rendered_line = rendered_line[:-1]
            rendered_line = rendered_line.rstrip() + "..."
        rendered_lines.append(rendered_line)

    text_surfaces = [font.render(line, True, (255, 255, 255)) for line in rendered_lines]
    box_width_px = max(text_surface.get_width() for text_surface in text_surfaces) + 2 * padding_px
    if max_width_px is not None:
        box_width_px = min(int(box_width_px), int(max_width_px))
    box_height_px = len(text_surfaces) * line_height_px + 2 * padding_px
    x0_px = max(0, min(int(x0_px), int(surface.get_width()) - int(box_width_px)))
    box_surface = pygame.Surface((box_width_px, box_height_px), pygame.SRCALPHA)
    box_surface.fill((0, 0, 0, 120))
    surface.blit(box_surface, (x0_px, y0_px))

    for idx, text_surface in enumerate(text_surfaces):
        surface.blit(
            text_surface,
            (
                x0_px + padding_px,
                y0_px + padding_px + idx * line_height_px,
            ),
        )


def _render_camera_pair(
    display,
    left_image,
    right_image,
    topdown_overlay: Mapping[str, object] | None = None,
    hud_lines: Sequence[str] | None = None,
    hud_font=None,
    hud_panel_width_px: int = 0,
):
    if pygame is None:
        return
    camera_pair_width_px = 0
    if left_image is not None:
        left_array = np.frombuffer(left_image.raw_data, dtype=np.uint8)
        left_array = left_array.reshape((left_image.height, left_image.width, 4))
        left_rgb = left_array[:, :, :3][:, :, ::-1]
        left_surface = pygame.surfarray.make_surface(left_rgb.swapaxes(0, 1))
        if isinstance(topdown_overlay, Mapping):
            _draw_planning_overlay(
                surface=left_surface,
                camera_transform=topdown_overlay["camera_transform"],
                calibration_matrix=topdown_overlay["calibration_matrix"],
                image_width_px=int(topdown_overlay["image_width_px"]),
                image_height_px=int(topdown_overlay["image_height_px"]),
                overlay_z_m=float(topdown_overlay["overlay_z_m"]),
                global_route_points=topdown_overlay.get("global_route_points", None),
                temporary_destination_state=topdown_overlay.get("temporary_destination_state", None),
                planned_trajectory_states=topdown_overlay.get("planned_trajectory_states", None),
                predicted_obstacle_trajectories=topdown_overlay.get("predicted_obstacle_trajectories", None),
                obstacle_field_contours=topdown_overlay.get("obstacle_field_contours", None),
                obstacle_risk_ids=topdown_overlay.get("obstacle_risk_ids", None),
            )
        display.blit(left_surface, (0, 0))
        camera_pair_width_px = max(int(camera_pair_width_px), int(left_image.width))

    if right_image is not None:
        right_array = np.frombuffer(right_image.raw_data, dtype=np.uint8)
        right_array = right_array.reshape((right_image.height, right_image.width, 4))
        right_rgb = right_array[:, :, :3][:, :, ::-1]
        right_surface = pygame.surfarray.make_surface(right_rgb.swapaxes(0, 1))
        display.blit(right_surface, (right_image.width, 0))
        camera_pair_width_px = max(int(camera_pair_width_px), int(right_image.width) * 2)

    if hud_lines:
        panel_width_px = max(0, int(hud_panel_width_px))
        if panel_width_px > 0:
            panel_x_px = int(camera_pair_width_px)
            panel_rect = pygame.Rect(
                int(panel_x_px),
                0,
                min(int(panel_width_px), max(0, int(display.get_width()) - int(panel_x_px))),
                int(display.get_height()),
            )
            pygame.draw.rect(display, (18, 18, 18), panel_rect)
            pygame.draw.line(display, (70, 70, 70), (panel_x_px, 0), (panel_x_px, display.get_height()), 1)
            _draw_hud_lines(
                display,
                hud_font,
                hud_lines,
                (panel_x_px + 10, 10),
                max_width_px=max(40, int(panel_rect.width) - 20),
            )
        else:
            _draw_hud_lines(display, hud_font, hud_lines, (16, 16))

    pygame.display.flip()

def _merge_tracker_predictions(
    object_snapshots: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Sequence[Mapping[str, float]]],
) -> List[dict]:
    merged: List[dict] = []
    for snapshot in object_snapshots:
        snapshot_copy = dict(snapshot)
        prediction = predictions.get(str(snapshot_copy.get("vehicle_id", "")), [])
        snapshot_copy["predicted_trajectory"] = [
            [
                float(item.get("x", 0.0)),
                float(item.get("y", 0.0)),
                float(item.get("v", 0.0)),
                float(item.get("psi", 0.0)),
            ]
            for item in prediction
        ]
        merged.append(snapshot_copy)
    return merged


def _call_with_supported_kwargs(func, **kwargs):
    signature = inspect.signature(func)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return func(**kwargs)
    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return func(**supported_kwargs)


def _load_optional_module(*, module_name: str, purpose: str):
    normalized_module_name = str(module_name).strip()
    if not normalized_module_name:
        return None
    try:
        return importlib.import_module(normalized_module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import {purpose} module '{normalized_module_name}': {exc}"
        ) from exc


def _initialize_scenario_runtime_state(
    *,
    module,
    world,
    world_map,
    carla,
    scenario_cfg: Mapping[str, object],
    traffic_manager_port: int | None = None,
    tracker_cfg: Mapping[str, object] | None = None,
    obstacle_filter_cfg: Mapping[str, object] | None = None,
    prediction_dt_s: float | None = None,
    prediction_horizon_s: float | None = None,
):
    if module is None:
        return None
    initialize_fn = getattr(module, "initialize_runtime", None)
    if not callable(initialize_fn):
        return None
    return _call_with_supported_kwargs(
        initialize_fn,
        world=world,
        world_map=world_map,
        carla=carla,
        scenario_cfg=scenario_cfg,
        traffic_manager_port=traffic_manager_port,
        tracker_cfg=tracker_cfg,
        obstacle_filter_cfg=obstacle_filter_cfg,
        prediction_dt_s=prediction_dt_s,
        prediction_horizon_s=prediction_horizon_s,
    )


def _apply_scenario_dynamic_obstacle_filter(
    *,
    module,
    runtime_state,
    world,
    world_map,
    carla,
    scenario_cfg: Mapping[str, object],
    object_snapshots: Sequence[Mapping[str, object]],
    sim_time_s: float,
    wall_time_s: float | None = None,
) -> Tuple[List[dict], Any]:
    snapshots_copy = [dict(snapshot) for snapshot in list(object_snapshots)]
    if module is None:
        return snapshots_copy, runtime_state

    filter_fn = getattr(module, "filter_dynamic_obstacle_snapshots", None)
    if not callable(filter_fn):
        return snapshots_copy, runtime_state

    filter_result = _call_with_supported_kwargs(
        filter_fn,
        world=world,
        world_map=world_map,
        carla=carla,
        scenario_cfg=scenario_cfg,
        runtime_state=runtime_state,
        object_snapshots=snapshots_copy,
        sim_time_s=float(sim_time_s),
        wall_time_s=None if wall_time_s is None else float(wall_time_s),
    )
    if isinstance(filter_result, tuple) and len(filter_result) == 2:
        filtered_snapshots, updated_runtime_state = filter_result
    else:
        filtered_snapshots = filter_result
        updated_runtime_state = runtime_state
    filtered_list = [dict(snapshot) for snapshot in list(filtered_snapshots or [])]
    return filtered_list, updated_runtime_state


def _maybe_apply_scenario_global_route_update(
    *,
    module,
    runtime_state,
    world,
    world_map,
    carla,
    scenario_cfg: Mapping[str, object],
    global_planner: AStarGlobalPlanner,
    ego_vehicle,
    sumo_bridge,
    ego_transform,
    goal_location,
    object_snapshots: Sequence[Mapping[str, object]],
    current_route_summary,
    active_global_route_points: Sequence[Sequence[float]],
    sim_time_s: float,
    wall_time_s: float | None = None,
) -> Tuple[object | None, List[List[float]] | None, Any]:
    if module is None:
        return None, None, runtime_state

    update_fn = getattr(module, "maybe_replan_global_route", None)
    if not callable(update_fn):
        return None, None, runtime_state

    update_result = _call_with_supported_kwargs(
        update_fn,
        world=world,
        world_map=world_map,
        carla=carla,
        scenario_cfg=scenario_cfg,
        runtime_state=runtime_state,
        global_planner=global_planner,
        ego_vehicle=ego_vehicle,
        sumo_bridge=sumo_bridge,
        ego_transform=ego_transform,
        goal_location=goal_location,
        object_snapshots=[dict(snapshot) for snapshot in list(object_snapshots or [])],
        current_route_summary=current_route_summary,
        active_global_route_points=[list(point) for point in list(active_global_route_points or [])],
        sim_time_s=float(sim_time_s),
        wall_time_s=None if wall_time_s is None else float(wall_time_s),
    )
    if isinstance(update_result, tuple):
        if len(update_result) == 3:
            route_summary, route_points, updated_runtime_state = update_result
            return route_summary, route_points, updated_runtime_state
        if len(update_result) == 2:
            route_summary, route_points = update_result
            return route_summary, route_points, runtime_state
    if isinstance(update_result, Mapping):
        return (
            update_result.get("route_summary", None),
            update_result.get("route_points", None),
            update_result.get("runtime_state", runtime_state),
        )
    return None, None, runtime_state


def _filter_obstacle_snapshots_by_vertical_overlap(
    *,
    ego_z_m: float,
    ego_height_m: float,
    object_snapshots: Sequence[Mapping[str, object]],
    vertical_clearance_margin_m: float,
    default_obstacle_height_m: float,
) -> List[dict]:
    filtered_snapshots: List[dict] = []
    ego_height_value_m = max(0.5, float(ego_height_m))
    margin_m = max(0.0, float(vertical_clearance_margin_m))
    fallback_height_m = max(0.5, float(default_obstacle_height_m))

    for snapshot in object_snapshots:
        snapshot_copy = dict(snapshot)
        obstacle_z_raw = snapshot_copy.get("z", None)
        if obstacle_z_raw is None:
            filtered_snapshots.append(snapshot_copy)
            continue

        obstacle_height_m = max(
            0.5,
            float(snapshot_copy.get("height_m", fallback_height_m)),
        )
        vertical_gap_m = abs(float(obstacle_z_raw) - float(ego_z_m))
        overlap_threshold_m = 0.5 * (ego_height_value_m + obstacle_height_m) + margin_m
        if vertical_gap_m <= overlap_threshold_m:
            filtered_snapshots.append(snapshot_copy)

    return filtered_snapshots


def _road_numeric_id_from_context(local_context: Mapping[str, object] | None) -> int:
    if not isinstance(local_context, Mapping):
        return -1
    raw_numeric_id = local_context.get("road_numeric_id", None)
    if raw_numeric_id is not None:
        try:
            return int(raw_numeric_id)
        except Exception:
            pass
    raw_road_id = str(local_context.get("road_id", "") or "").strip()
    if ":" in raw_road_id:
        raw_road_id = raw_road_id.split(":", 1)[0]
    try:
        return int(raw_road_id)
    except Exception:
        return -1


def _same_lane_safety_corridor(
    ego_lane_context: Mapping[str, object] | None,
    obstacle_lane_context: Mapping[str, object] | None,
) -> bool:
    if not isinstance(ego_lane_context, Mapping) or not isinstance(obstacle_lane_context, Mapping):
        return False
    ego_direction = str(ego_lane_context.get("direction", "") or "").strip().lower()
    obstacle_direction = str(obstacle_lane_context.get("direction", "") or "").strip().lower()
    if ego_direction and obstacle_direction and ego_direction != obstacle_direction:
        return False

    ego_road_numeric_id = _road_numeric_id_from_context(ego_lane_context)
    obstacle_road_numeric_id = _road_numeric_id_from_context(obstacle_lane_context)
    if ego_road_numeric_id > 0 and obstacle_road_numeric_id > 0:
        if int(ego_road_numeric_id) == int(obstacle_road_numeric_id):
            return True

    ego_road_id = str(ego_lane_context.get("road_id", "") or "").strip()
    obstacle_road_id = str(obstacle_lane_context.get("road_id", "") or "").strip()
    if ego_road_id and obstacle_road_id:
        if str(ego_road_id) == str(obstacle_road_id):
            return True

    ego_is_intersection = bool(ego_lane_context.get("is_intersection", False))
    obstacle_is_intersection = bool(obstacle_lane_context.get("is_intersection", False))
    if not ego_is_intersection and not obstacle_is_intersection:
        return False

    ego_heading_rad = ego_lane_context.get("heading_rad", None)
    obstacle_heading_rad = obstacle_lane_context.get("heading_rad", None)
    try:
        if ego_heading_rad is None or obstacle_heading_rad is None:
            return False
        heading_error_rad = abs(
            math.atan2(
                math.sin(float(ego_heading_rad) - float(obstacle_heading_rad)),
                math.cos(float(ego_heading_rad) - float(obstacle_heading_rad)),
            )
        )
    except Exception:
        return False
    return float(heading_error_rad) <= math.radians(60.0)


def _lane_safety_assignment_for_obstacle(
    *,
    ego_lane_context: Mapping[str, object] | None,
    obstacle_lane_context: Mapping[str, object] | None,
    ego_lane_id: int,
    ego_in_junction: bool,
    available_lane_ids: Sequence[int] | None = None,
    ego_snapshot: Mapping[str, float] | None = None,
    obstacle_snapshot: Mapping[str, object] | None = None,
) -> int:
    obstacle_lane_context = (
        dict(obstacle_lane_context)
        if isinstance(obstacle_lane_context, Mapping)
        else {}
    )
    ego_lane_context = (
        dict(ego_lane_context)
        if isinstance(ego_lane_context, Mapping)
        else {}
    )
    same_corridor = _same_lane_safety_corridor(ego_lane_context, obstacle_lane_context)
    obstacle_is_intersection = bool(obstacle_lane_context.get("is_intersection", False))
    ordered_lane_ids = [
        int(lane_id)
        for lane_id in list(available_lane_ids or [])
        if int(lane_id) != 0
    ]
    obstacle_lane_id = int(obstacle_lane_context.get("lane_id", 0) or 0)

    relative_lane_id = 0
    lateral_offset_m = None
    lane_width_m = None
    obstacle_heading_rad = None
    if (
        len(ordered_lane_ids) > 0
        and int(ego_lane_id) in ordered_lane_ids
        and isinstance(ego_snapshot, Mapping)
        and isinstance(obstacle_snapshot, Mapping)
    ):
        ego_heading_rad = ego_snapshot.get("psi", None)
        ego_x_m = ego_snapshot.get("x", None)
        ego_y_m = ego_snapshot.get("y", None)
        obstacle_x_m = obstacle_snapshot.get("x", None)
        obstacle_y_m = obstacle_snapshot.get("y", None)
        try:
            ego_heading_rad = float(ego_heading_rad)
            ego_x_m = float(ego_x_m)
            ego_y_m = float(ego_y_m)
            obstacle_x_m = float(obstacle_x_m)
            obstacle_y_m = float(obstacle_y_m)
            if all(
                math.isfinite(value)
                for value in (
                    ego_heading_rad,
                    ego_x_m,
                    ego_y_m,
                    obstacle_x_m,
                    obstacle_y_m,
                )
            ):
                dx_m = float(obstacle_x_m) - float(ego_x_m)
                dy_m = float(obstacle_y_m) - float(ego_y_m)
                lateral_offset_m = float(
                    -math.sin(float(ego_heading_rad)) * float(dx_m)
                    + math.cos(float(ego_heading_rad)) * float(dy_m)
                )
                lane_width_m = max(
                    2.5,
                    float(ego_lane_context.get("lane_width_m", 0.0) or 0.0),
                    float(obstacle_lane_context.get("lane_width_m", 0.0) or 0.0),
                )
                obstacle_heading_rad = float(
                    obstacle_snapshot.get(
                        "psi",
                        obstacle_lane_context.get("heading_rad", ego_heading_rad),
                    )
                )
                if float(lane_width_m) > 1.0e-3:
                    ego_lane_index = ordered_lane_ids.index(int(ego_lane_id))
                    lane_index_shift = int(
                        round(float(lateral_offset_m) / float(lane_width_m))
                    )
                    target_lane_index = max(
                        0,
                        min(
                            len(ordered_lane_ids) - 1,
                            int(ego_lane_index) + int(lane_index_shift),
                        ),
                    )
                    relative_lane_id = int(ordered_lane_ids[int(target_lane_index)])
        except Exception:
            relative_lane_id = 0
            lateral_offset_m = None
            lane_width_m = None
            obstacle_heading_rad = None

    if not bool(same_corridor):
        same_direction = True
        ego_direction = str(ego_lane_context.get("direction", "") or "").strip().lower()
        obstacle_direction = str(obstacle_lane_context.get("direction", "") or "").strip().lower()
        if ego_direction and obstacle_direction and ego_direction != obstacle_direction:
            same_direction = False
        heading_aligned = False
        if obstacle_heading_rad is not None and isinstance(ego_snapshot, Mapping):
            try:
                ego_heading_rad = float(ego_snapshot.get("psi", 0.0))
                heading_error_rad = abs(
                    math.atan2(
                        math.sin(float(ego_heading_rad) - float(obstacle_heading_rad)),
                        math.cos(float(ego_heading_rad) - float(obstacle_heading_rad)),
                    )
                )
                heading_aligned = float(heading_error_rad) <= math.radians(30.0)
            except Exception:
                heading_aligned = False
        within_local_lane_band = False
        if (
            lateral_offset_m is not None
            and lane_width_m is not None
            and len(ordered_lane_ids) > 0
        ):
            within_local_lane_band = abs(float(lateral_offset_m)) <= (
                float(lane_width_m) * max(1.0, float(len(ordered_lane_ids)) - 0.5) + 0.5
            )
        if not (
            bool(same_direction)
            and bool(heading_aligned)
            and bool(within_local_lane_band)
            and int(relative_lane_id) != 0
        ):
            return 0

    if int(obstacle_lane_id) == 0:
        return int(relative_lane_id)
    if len(ordered_lane_ids) > 0 and int(obstacle_lane_id) not in ordered_lane_ids:
        if int(relative_lane_id) != 0:
            return int(relative_lane_id)
        return 0

    if (
        bool(ego_in_junction)
        and int(ego_lane_id) != 0
        and len(ordered_lane_ids) > 1
        and int(relative_lane_id) != 0
    ):
        return int(relative_lane_id)
    return int(obstacle_lane_id)


def _sanitize_lane_score(value: object, *, default: float = 1.0) -> float:
    try:
        numeric_value = float(value)
    except Exception:
        numeric_value = float(default)
    if not math.isfinite(float(numeric_value)):
        numeric_value = float(default)
    return float(max(0.0, min(1.0, float(numeric_value))))


def _nearest_front_obstacle_by_lane(
    *,
    ego_snapshot: Mapping[str, float],
    obstacle_snapshots: Sequence[Mapping[str, object]],
    lane_assignments: Mapping[str, int],
    available_lane_ids: Sequence[int],
) -> Dict[int, dict]:
    ego_x_m = float(ego_snapshot.get("x", 0.0))
    ego_y_m = float(ego_snapshot.get("y", 0.0))
    ego_psi_rad = float(ego_snapshot.get("psi", 0.0))
    cos_heading = math.cos(float(ego_psi_rad))
    sin_heading = math.sin(float(ego_psi_rad))

    nearest_obstacles: Dict[int, dict] = {}
    valid_lane_ids = {int(lane_id) for lane_id in list(available_lane_ids or [])}

    for obstacle_snapshot in obstacle_snapshots:
        obstacle_id = str(obstacle_snapshot.get("vehicle_id", ""))
        lane_id = int(lane_assignments.get(obstacle_id, -1))
        if lane_id not in valid_lane_ids:
            continue

        dx_m = float(obstacle_snapshot.get("x", 0.0)) - float(ego_x_m)
        dy_m = float(obstacle_snapshot.get("y", 0.0)) - float(ego_y_m)
        longitudinal_m = float(cos_heading) * float(dx_m) + float(sin_heading) * float(dy_m)
        if float(longitudinal_m) < 0.0:
            continue

        previous_distance_m = float(
            nearest_obstacles.get(int(lane_id), {}).get("front_distance_m", float("inf"))
        )
        if float(longitudinal_m) < float(previous_distance_m):
            obstacle_copy = dict(obstacle_snapshot)
            obstacle_copy["front_distance_m"] = float(longitudinal_m)
            nearest_obstacles[int(lane_id)] = obstacle_copy

    return nearest_obstacles


def _nearest_front_obstacle_distance_by_lane(
    *,
    ego_snapshot: Mapping[str, float],
    obstacle_snapshots: Sequence[Mapping[str, object]],
    lane_assignments: Mapping[str, int],
    available_lane_ids: Sequence[int],
) -> Dict[int, float]:
    nearest_obstacles = _nearest_front_obstacle_by_lane(
        ego_snapshot=ego_snapshot,
        obstacle_snapshots=obstacle_snapshots,
        lane_assignments=lane_assignments,
        available_lane_ids=available_lane_ids,
    )
    return {
        int(lane_id): float(obstacle.get("front_distance_m", float("inf")))
        for lane_id, obstacle in nearest_obstacles.items()
    }


def _should_force_intersection_reroute(
    *,
    mode: str,
    ego_lane_id: int,
    lane_safety_scores: Mapping[int, float],
    nearest_front_obstacles_by_lane: Mapping[int, Mapping[str, object]],
    safety_threshold: float,
) -> bool:
    if str(mode or "NORMAL").strip().upper() != "INTERSECTION":
        return False
    if int(ego_lane_id) not in nearest_front_obstacles_by_lane:
        return False
    return float(lane_safety_scores.get(int(ego_lane_id), 1.0)) < float(safety_threshold)


def _static_obstacle_replan_candidate_lane_ids(
    *,
    current_lane_id: int,
    available_lane_ids: Sequence[int],
    lane_safety_scores: Mapping[int, float],
) -> List[int]:
    lane_ids = [int(lane_id) for lane_id in list(available_lane_ids or [])]
    if len(lane_ids) == 0:
        return []

    normalized_current_lane_id = int(current_lane_id)
    if normalized_current_lane_id not in lane_ids:
        normalized_current_lane_id = min(
            lane_ids,
            key=lambda lane_id: abs(int(lane_id) - int(current_lane_id)),
        )

    alternative_lane_ids = [
        lane_id
        for lane_id in lane_ids
        if int(lane_id) != int(normalized_current_lane_id)
    ]
    alternative_lane_ids.sort(
        key=lambda lane_id: (
            -float(lane_safety_scores.get(int(lane_id), 0.0)),
            abs(int(lane_id) - int(normalized_current_lane_id)),
            int(lane_id),
        )
    )
    return list(alternative_lane_ids) + [int(normalized_current_lane_id)]


def _replan_route_around_static_intersection_obstacle(
    *,
    global_planner: AStarGlobalPlanner,
    ego_transform,
    goal_location,
    blocked_obstacle_snapshot: Mapping[str, object] | None,
    blocked_lane_id: int,
) -> tuple[object | None, List[List[float]]]:
    if blocked_obstacle_snapshot is None:
        return None, []

    start_xy = [
        float(ego_transform.location.x),
        float(ego_transform.location.y),
    ]
    goal_xy = [
        float(goal_location.x),
        float(goal_location.y),
    ]
    blocked_point_xy = [
        float(blocked_obstacle_snapshot.get("x", 0.0)),
        float(blocked_obstacle_snapshot.get("y", 0.0)),
    ]
    obstacle_length_m = max(0.0, float(blocked_obstacle_snapshot.get("length_m", 4.5)))
    obstacle_width_m = max(0.0, float(blocked_obstacle_snapshot.get("width_m", 2.0)))
    block_radius_m = max(
        6.0,
        0.5 * max(float(obstacle_length_m), float(obstacle_width_m)) + 4.0,
    )
    route_summary = global_planner.plan_route_astar_avoiding_points(
        start_xy=start_xy,
        goal_xy=goal_xy,
        blocked_points_xy=[blocked_point_xy],
        blocked_lane_ids=[int(blocked_lane_id)],
        block_radius_m=float(block_radius_m),
        replace_stored_route=True,
    )
    if not bool(getattr(route_summary, "route_found", False)):
        return None, []

    route_points = [
        [float(item[0]), float(item[1])]
        for item in list(getattr(route_summary, "route_waypoints", []) or [])
        if isinstance(item, Sequence) and len(item) >= 2
    ]
    if len(route_points) == 0:
        return None, []
    return route_summary, route_points


def _apply_final_destination_snap(
    *,
    temporary_destination_state: Sequence[float] | None,
    final_destination_state: Sequence[float],
    ego_state: Sequence[float],
    lock_to_final_distance_m: float,
    original_max_velocity_mps: float,
    speed_taper_distance_m: float | None = None,
) -> tuple[List[float] | None, float]:
    if temporary_destination_state is None:
        return None, float(original_max_velocity_mps)
    if len(temporary_destination_state) < 4 or len(final_destination_state) < 4:
        return list(temporary_destination_state), float(original_max_velocity_mps)

    snap_distance_threshold_m = max(0.0, float(lock_to_final_distance_m))
    if len(ego_state) < 2:
        return list(temporary_destination_state), float(original_max_velocity_mps)

    distance_ego_to_final_m = math.hypot(
        float(ego_state[0]) - float(final_destination_state[0]),
        float(ego_state[1]) - float(final_destination_state[1]),
    )
    if distance_ego_to_final_m > float(snap_distance_threshold_m):
        return list(temporary_destination_state), float(original_max_velocity_mps)

    snapped_destination_state = list(temporary_destination_state)
    snapped_destination_state[0] = float(final_destination_state[0])
    snapped_destination_state[1] = float(final_destination_state[1])
    snapped_destination_state[3] = float(final_destination_state[3])

    taper_distance_threshold_m = (
        float(snap_distance_threshold_m)
        if speed_taper_distance_m is None
        else max(0.0, float(speed_taper_distance_m))
    )

    if float(taper_distance_threshold_m) <= 1e-6:
        active_max_velocity_mps = 0.0
    else:
        stop_speed_scale = max(
            0.0,
            min(1.0, float(distance_ego_to_final_m) / float(taper_distance_threshold_m)),
        )
        active_max_velocity_mps = float(original_max_velocity_mps) * float(stop_speed_scale)

    if len(snapped_destination_state) >= 3:
        snapped_destination_state[2] = float(active_max_velocity_mps)

    return snapped_destination_state, float(active_max_velocity_mps)


def _apply_stop_target_speed_cap(
    *,
    temporary_destination_state: Sequence[float] | None,
    ego_state: Sequence[float],
    stop_target_distance_m: float | None,
    original_max_velocity_mps: float,
    braking_deceleration_mps2: float,
    stop_buffer_m: float,
) -> tuple[List[float] | None, float]:
    if temporary_destination_state is None:
        return None, float(original_max_velocity_mps)

    shaped_destination_state = list(temporary_destination_state)
    if len(shaped_destination_state) < 3:
        return shaped_destination_state, float(original_max_velocity_mps)

    if stop_target_distance_m is None:
        if len(ego_state) >= 2 and len(shaped_destination_state) >= 2:
            remaining_distance_m = math.hypot(
                float(shaped_destination_state[0]) - float(ego_state[0]),
                float(shaped_destination_state[1]) - float(ego_state[1]),
            )
        else:
            remaining_distance_m = 0.0
    else:
        remaining_distance_m = max(0.0, float(stop_target_distance_m))

    effective_braking_deceleration_mps2 = max(1.0e-6, float(braking_deceleration_mps2))
    remaining_brake_distance_m = max(
        0.0,
        float(remaining_distance_m) - max(0.0, float(stop_buffer_m)),
    )
    speed_cap_mps = min(
        float(original_max_velocity_mps),
        math.sqrt(2.0 * float(effective_braking_deceleration_mps2) * float(remaining_brake_distance_m)),
    )
    shaped_destination_state[2] = float(speed_cap_mps)
    return shaped_destination_state, float(speed_cap_mps)


def _apply_exact_stop_target_snap(
    *,
    temporary_destination_state: Sequence[float] | None,
    stop_target_state: Sequence[float] | None,
    ego_state: Sequence[float],
    lock_to_stop_distance_m: float,
    original_max_velocity_mps: float,
) -> tuple[List[float] | None, float]:
    if stop_target_state is None:
        return (
            None if temporary_destination_state is None else list(temporary_destination_state),
            float(original_max_velocity_mps),
        )
    return _apply_final_destination_snap(
        temporary_destination_state=temporary_destination_state,
        final_destination_state=stop_target_state,
        ego_state=ego_state,
        lock_to_final_distance_m=float(lock_to_stop_distance_m),
        original_max_velocity_mps=float(original_max_velocity_mps),
    )


def _ensure_rolling_destination_speed(
    *,
    temporary_destination_state: Sequence[float] | None,
    active_plan_max_velocity_mps: float,
    current_target_v_mps: float,
    current_behavior: str,
    final_goal_stop_active: bool,
    stop_speed_threshold_mps: float = 0.05,
) -> tuple[List[float] | None, float]:
    if temporary_destination_state is None:
        return None, float(active_plan_max_velocity_mps)
    destination_state = list(temporary_destination_state)
    if len(destination_state) < 3:
        return destination_state, float(active_plan_max_velocity_mps)
    if bool(final_goal_stop_active) or bool(is_fixed_stop_decision(current_behavior)):
        return destination_state, float(active_plan_max_velocity_mps)

    desired_speed_mps = max(
        float(active_plan_max_velocity_mps),
        float(current_target_v_mps),
        0.0,
    )
    if float(destination_state[2]) <= float(stop_speed_threshold_mps) and float(desired_speed_mps) > float(stop_speed_threshold_mps):
        destination_state[2] = float(desired_speed_mps)
        active_plan_max_velocity_mps = max(
            float(active_plan_max_velocity_mps),
            float(desired_speed_mps),
        )
    return destination_state, float(active_plan_max_velocity_mps)


def _stabilize_temporary_destination(
    *,
    temporary_destination_state: Sequence[float] | None,
    previous_destination_state: Sequence[float] | None,
    current_behavior: str,
    enabled: bool,
    blend_alpha: float,
    max_smooth_jump_m: float,
    max_lane_change_smooth_jump_m: float,
    final_goal_stop_active: bool,
) -> List[float] | None:
    if temporary_destination_state is None:
        return None
    destination_state = list(temporary_destination_state)
    if (
        not bool(enabled)
        or previous_destination_state is None
        or len(destination_state) < 4
        or len(previous_destination_state) < 4
        or bool(final_goal_stop_active)
        or bool(is_fixed_stop_decision(current_behavior))
    ):
        return destination_state

    if str(normalize_behavior_decision(current_behavior)) in {
        "lane_change_left",
        "lane_change_right",
    }:
        return destination_state

    previous_state = list(previous_destination_state)
    current_lane_id = int(destination_state[4]) if len(destination_state) >= 5 else 0
    previous_lane_id = int(previous_state[4]) if len(previous_state) >= 5 else 0
    current_mode = float(destination_state[5]) if len(destination_state) >= 6 else 0.0
    previous_mode = float(previous_state[5]) if len(previous_state) >= 6 else 0.0
    current_road_id = int(destination_state[6]) if len(destination_state) >= 7 else -1
    previous_road_id = int(previous_state[6]) if len(previous_state) >= 7 else -1

    same_route_context = (
        int(current_lane_id) == int(previous_lane_id)
        and int(current_road_id) == int(previous_road_id)
        and abs(float(current_mode) - float(previous_mode)) <= 0.5
    )
    if not bool(same_route_context):
        return destination_state

    jump_distance_m = math.hypot(
        float(destination_state[0]) - float(previous_state[0]),
        float(destination_state[1]) - float(previous_state[1]),
    )
    allowed_jump_m = float(max_smooth_jump_m)
    if float(jump_distance_m) <= 1.0e-6 or float(jump_distance_m) > max(0.0, float(allowed_jump_m)):
        return destination_state

    alpha = max(0.0, min(1.0, float(blend_alpha)))
    destination_state[0] = (1.0 - alpha) * float(previous_state[0]) + alpha * float(destination_state[0])
    destination_state[1] = (1.0 - alpha) * float(previous_state[1]) + alpha * float(destination_state[1])
    previous_heading = float(previous_state[3])
    heading_delta = math.atan2(
        math.sin(float(destination_state[3]) - float(previous_heading)),
        math.cos(float(destination_state[3]) - float(previous_heading)),
    )
    destination_state[3] = float(previous_heading) + float(alpha) * float(heading_delta)
    return destination_state


def _destination_matches_target_point(
    *,
    destination_state: Sequence[float] | None,
    target_state: Sequence[float],
    tolerance_m: float = 1.0e-3,
) -> bool:
    if destination_state is None or len(destination_state) < 2 or len(target_state) < 2:
        return False
    return math.hypot(
        float(destination_state[0]) - float(target_state[0]),
        float(destination_state[1]) - float(target_state[1]),
    ) <= max(0.0, float(tolerance_m))


def _final_destination_stop_target_state(
    *,
    destination_state: Sequence[float] | None,
    final_destination_state: Sequence[float],
) -> List[float] | None:
    if destination_state is None or len(destination_state) < 4 or len(final_destination_state) < 4:
        return None
    lane_id = int(destination_state[4]) if len(destination_state) >= 5 else 0
    mode_value = float(destination_state[5]) if len(destination_state) >= 6 else 0.0
    road_id = float(destination_state[6]) if len(destination_state) >= 7 else -1.0
    entered_intersection = float(destination_state[7]) if len(destination_state) >= 8 else 0.0
    return [
        float(final_destination_state[0]),
        float(final_destination_state[1]),
        0.0,
        float(final_destination_state[3]),
        float(lane_id),
        float(mode_value),
        float(road_id),
        float(entered_intersection),
    ]


def _stop_target_state_from_behavior_output(
    *,
    world_map: Any,
    carla: Any,
    ego_transform: Any,
    stop_target: Mapping[str, object] | None,
    target_lane_id: object = None,
) -> List[float] | None:
    if not isinstance(stop_target, Mapping):
        return None
    try:
        stop_x_m = float(stop_target.get("x_m", 0.0))
        stop_y_m = float(stop_target.get("y_m", 0.0))
        stop_heading_rad = float(stop_target.get("heading_rad", 0.0))
        stop_lane_id = int(stop_target.get("lane_id", 0))
        stop_road_id = float(stop_target.get("road_id", -1))
        desired_lane_id = None
        if target_lane_id is not None:
            desired_lane_id = int(target_lane_id)
        ego_z_m = float(getattr(ego_transform.location, "z", 0.0))
        stop_waypoint = world_map.get_waypoint(
            carla.Location(
                x=float(stop_x_m),
                y=float(stop_y_m),
                z=float(ego_z_m),
            ),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if stop_waypoint is not None:
            if desired_lane_id is not None:
                lane_waypoint = canonical_lane_waypoint_for_lane_id(
                    stop_waypoint,
                    int(desired_lane_id),
                )
                if lane_waypoint is not None:
                    stop_waypoint = lane_waypoint
            stop_x_m = float(stop_waypoint.transform.location.x)
            stop_y_m = float(stop_waypoint.transform.location.y)
            stop_heading_rad = float(math.radians(stop_waypoint.transform.rotation.yaw))
            stop_lane_id = int(canonical_lane_id_for_waypoint(stop_waypoint))
            stop_road_id = float(getattr(stop_waypoint, "road_id", int(stop_road_id)))
        elif desired_lane_id is not None:
            stop_lane_id = int(desired_lane_id)
        return [
            float(stop_x_m),
            float(stop_y_m),
            0.0,
            float(stop_heading_rad),
            float(stop_lane_id),
            0.0,
            float(stop_road_id),
            0.0,
        ]
    except Exception:
        return None


def _follow_target_state_from_behavior_output(
    *,
    world_map: Any,
    carla: Any,
    ego_transform: Any,
    follow_target: Mapping[str, object] | None,
) -> List[float] | None:
    if not isinstance(follow_target, Mapping):
        return None
    try:
        follow_x_m = float(follow_target.get("x_m", 0.0))
        follow_y_m = float(follow_target.get("y_m", 0.0))
        follow_heading_rad = float(follow_target.get("heading_rad", 0.0))
        follow_lane_id = float(follow_target.get("lane_id", 0))
        follow_road_id = float(follow_target.get("road_id", -1))
        ego_z_m = float(getattr(ego_transform.location, "z", 0.0))
        follow_waypoint = world_map.get_waypoint(
            carla.Location(
                x=float(follow_x_m),
                y=float(follow_y_m),
                z=float(ego_z_m),
            ),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if follow_waypoint is not None:
            follow_x_m = float(follow_waypoint.transform.location.x)
            follow_y_m = float(follow_waypoint.transform.location.y)
            follow_heading_rad = float(math.radians(follow_waypoint.transform.rotation.yaw))
            follow_lane_id = float(canonical_lane_id_for_waypoint(follow_waypoint))
            follow_road_id = float(getattr(follow_waypoint, "road_id", int(follow_road_id)))
        return [
            float(follow_x_m),
            float(follow_y_m),
            max(0.0, float(follow_target.get("target_v_mps", 0.0))),
            float(follow_heading_rad),
            float(follow_lane_id),
            0.0,
            float(follow_road_id),
            0.0,
        ]
    except Exception:
        return None


def _sample_superellipse_contour_world(
    *,
    center_x_m: float,
    center_y_m: float,
    center_z_m: float,
    heading_rad: float,
    half_length_m: float,
    half_width_m: float,
    shape_exponent: float,
    num_points: int = 72,
) -> List[List[float]]:
    half_length_m = max(1e-6, float(half_length_m))
    half_width_m = max(1e-6, float(half_width_m))
    shape_exponent = max(2.0, float(shape_exponent))
    cos_heading = math.cos(float(heading_rad))
    sin_heading = math.sin(float(heading_rad))
    contour_world: List[List[float]] = []
    exponent = 2.0 / float(shape_exponent)
    for idx in range(max(8, int(num_points))):
        theta_rad = 2.0 * math.pi * float(idx) / float(max(8, int(num_points)))
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        local_x_m = float(half_length_m) * math.copysign(abs(cos_theta) ** exponent, cos_theta)
        local_y_m = float(half_width_m) * math.copysign(abs(sin_theta) ** exponent, sin_theta)
        world_x_m = float(center_x_m) + local_x_m * cos_heading - local_y_m * sin_heading
        world_y_m = float(center_y_m) + local_x_m * sin_heading + local_y_m * cos_heading
        contour_world.append([float(world_x_m), float(world_y_m), float(center_z_m)])
    return contour_world


def _static_obstacle_prediction(snapshot: Mapping[str, object], horizon_steps: int) -> List[List[float]]:
    state = [
        float(snapshot.get("x", 0.0)),
        float(snapshot.get("y", 0.0)),
        float(snapshot.get("v", 0.0)),
        float(snapshot.get("psi", 0.0)),
    ]
    return [list(state) for _ in range(max(1, int(horizon_steps)))]


def _filter_snapshots_by_distance(
    *,
    ego_x_m: float,
    ego_y_m: float,
    object_snapshots: Sequence[Mapping[str, object]],
    max_distance_m: float | None = None,
    max_snapshots: int | None = None,
) -> List[dict]:
    if len(object_snapshots) == 0:
        return []

    max_distance_sq = None
    if max_distance_m is not None and float(max_distance_m) > 0.0:
        max_distance_sq = float(max_distance_m) * float(max_distance_m)

    ranked_snapshots: List[tuple[float, dict]] = []
    for snapshot in object_snapshots:
        dx_m = float(snapshot.get("x", 0.0)) - float(ego_x_m)
        dy_m = float(snapshot.get("y", 0.0)) - float(ego_y_m)
        distance_sq = float(dx_m) * float(dx_m) + float(dy_m) * float(dy_m)
        if max_distance_sq is not None and float(distance_sq) > float(max_distance_sq):
            continue
        ranked_snapshots.append((float(distance_sq), dict(snapshot)))

    if len(ranked_snapshots) == 0:
        return []

    ranked_snapshots.sort(key=lambda item: float(item[0]))
    if max_snapshots is not None and int(max_snapshots) > 0:
        ranked_snapshots = ranked_snapshots[: int(max_snapshots)]
    return [snapshot for _distance_sq, snapshot in ranked_snapshots]


def _collect_vehicle_snapshots(
    world,
    ego_vehicle,
    sumo_bridge=None,
    *,
    max_distance_m: float | None = None,
    max_snapshots: int | None = None,
) -> List[dict]:
    ego_transform = ego_vehicle.get_transform()
    ego_x_m = float(ego_transform.location.x)
    ego_y_m = float(ego_transform.location.y)
    max_distance_sq = None
    if max_distance_m is not None and float(max_distance_m) > 0.0:
        max_distance_sq = float(max_distance_m) * float(max_distance_m)

    ranked_snapshots: List[tuple[float, dict]] = []
    for actor in world.get_actors().filter("vehicle.*"):
        if int(actor.id) == int(ego_vehicle.id):
            continue
        transform = actor.get_transform()
        dx_m = float(transform.location.x) - float(ego_x_m)
        dy_m = float(transform.location.y) - float(ego_y_m)
        distance_sq = float(dx_m) * float(dx_m) + float(dy_m) * float(dy_m)
        if max_distance_sq is not None and float(distance_sq) > float(max_distance_sq):
            continue

        velocity = actor.get_velocity()
        speed_mps = math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
        yaw_rad = float(math.radians(transform.rotation.yaw))
        role_name = str(getattr(actor, "attributes", {}).get("role_name", "")).strip()
        vehicle_id = role_name or str(actor.id)
        snapshot_override = None
        if sumo_bridge is not None and hasattr(sumo_bridge, "get_actor_snapshot_override"):
            snapshot_override = sumo_bridge.get_actor_snapshot_override(int(actor.id))
        if isinstance(snapshot_override, Mapping):
            override_vehicle_id = str(snapshot_override.get("vehicle_id", "")).strip()
            if override_vehicle_id:
                vehicle_id = override_vehicle_id
            try:
                speed_mps = float(snapshot_override.get("v", speed_mps))
            except Exception:
                pass
            try:
                yaw_rad = float(snapshot_override.get("psi", yaw_rad))
            except Exception:
                pass
        ranked_snapshots.append(
            (
                float(distance_sq),
                {
                    "vehicle_id": str(vehicle_id),
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                    "v": float(speed_mps),
                    "psi": float(yaw_rad),
                    "length_m": float(actor.bounding_box.extent.x * 2.0),
                    "width_m": float(actor.bounding_box.extent.y * 2.0),
                    "height_m": float(actor.bounding_box.extent.z * 2.0),
                },
            )
        )

    ranked_snapshots.sort(key=lambda item: float(item[0]))
    if max_snapshots is not None and int(max_snapshots) > 0:
        ranked_snapshots = ranked_snapshots[: int(max_snapshots)]
    return [snapshot for _distance_sq, snapshot in ranked_snapshots]


def _collect_environment_obstacle_snapshots(world, map_obj, carla, obstacle_prefix: str) -> List[dict]:
    snapshots: List[dict] = []
    if not str(obstacle_prefix).strip():
        return snapshots

    for env_obj in _find_environment_objects_by_prefix(world, carla, obstacle_prefix):
        transform = env_obj.transform
        location = transform.location
        nearest_waypoint = map_obj.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        heading_rad = float(math.radians(transform.rotation.yaw))
        if nearest_waypoint is not None:
            heading_rad = float(math.radians(nearest_waypoint.transform.rotation.yaw))

        half_length_m = 2.25
        half_width_m = 1.0
        half_height_m = 1.0
        bbox = getattr(env_obj, "bounding_box", None)
        if bbox is not None:
            half_length_m = max(0.05, float(getattr(bbox.extent, "x", half_length_m)))
            half_width_m = max(0.05, float(getattr(bbox.extent, "y", half_width_m)))
            half_height_m = max(0.05, float(getattr(bbox.extent, "z", half_height_m)))

        snapshots.append(
            {
                "vehicle_id": str(getattr(env_obj, "name", f"{obstacle_prefix}_{len(snapshots)+1}")),
                "x": float(location.x),
                "y": float(location.y),
                "z": float(location.z),
                "v": 0.0,
                "psi": float(heading_rad),
                "length_m": 2.0 * float(half_length_m),
                "width_m": 2.0 * float(half_width_m),
                "height_m": 2.0 * float(half_height_m),
                "type": "static_obstacle",
                "predicted_trajectory": _static_obstacle_prediction(
                    {
                        "x": float(location.x),
                        "y": float(location.y),
                        "v": 0.0,
                        "psi": float(heading_rad),
                    },
                    horizon_steps=1,
                ),
            }
        )
    return snapshots


def _build_obstacle_field_contours(
    *,
    mpc: MPC,
    ego_state: Sequence[float],
    object_snapshots: Sequence[Mapping[str, object]],
) -> List[dict]:
    contours: List[dict] = []
    for snapshot in object_snapshots:
        obstacle_length_m = float(snapshot.get("length_m", 4.5))
        obstacle_width_m = float(snapshot.get("width_m", 2.0))
        obstacle_state = [
            float(snapshot.get("x", 0.0)),
            float(snapshot.get("y", 0.0)),
            float(snapshot.get("v", 0.0)),
            float(snapshot.get("psi", 0.0)),
        ]
        geometry = mpc._superellipsoid_zone_geometry(
            ego_state=ego_state,
            obstacle_state=obstacle_state,
            obstacle_length_m=obstacle_length_m,
            obstacle_width_m=obstacle_width_m,
        )
        obstacle_z_m = float(snapshot.get("z", 0.0))
        contours.append(
            {
                "label": str(snapshot.get("vehicle_id", "")),
                "collision_points_world": _sample_superellipse_contour_world(
                    center_x_m=float(geometry["obstacle_x_m"]),
                    center_y_m=float(geometry["obstacle_y_m"]),
                    center_z_m=obstacle_z_m,
                    heading_rad=float(geometry["obstacle_psi_rad"]),
                    half_length_m=float(geometry["xc_m"]),
                    half_width_m=float(geometry["yc_m"]),
                    shape_exponent=float(geometry["shape_exponent"]),
                ),
            }
        )
    return contours


def _curvature_from_points_xy(points_xy: Sequence[Sequence[float]]) -> float:
    normalized_points: List[tuple[float, float]] = []
    for point in list(points_xy or [])[:3]:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        try:
            normalized_points.append((float(point[0]), float(point[1])))
        except Exception:
            continue
    if len(normalized_points) < 3:
        return 0.0

    (x1, y1), (x2, y2), (x3, y3) = normalized_points[:3]
    a_m = math.hypot(float(x2) - float(x1), float(y2) - float(y1))
    b_m = math.hypot(float(x3) - float(x2), float(y3) - float(y2))
    c_m = math.hypot(float(x3) - float(x1), float(y3) - float(y1))
    denom = float(a_m) * float(b_m) * float(c_m)
    if float(denom) <= 1.0e-6:
        return 0.0
    signed_area_2x = (float(x2) - float(x1)) * (float(y3) - float(y1)) - (float(y2) - float(y1)) * (float(x3) - float(x1))
    return float(2.0 * float(signed_area_2x) / float(denom))


def _reference_curvature_abs(
    reference_samples: Sequence[Mapping[str, object]] | None,
    sample_count: int = 6,
) -> float:
    if reference_samples is None:
        return 0.0
    points_xy: List[list[float]] = []
    for sample in list(reference_samples or [])[:max(3, int(sample_count))]:
        if not isinstance(sample, Mapping):
            continue
        try:
            points_xy.append([
                float(sample.get("x_ref_m", sample.get("x", 0.0))),
                float(sample.get("y_ref_m", sample.get("y", 0.0))),
            ])
        except Exception:
            continue
    if len(points_xy) < 3:
        return 0.0
    max_curvature = 0.0
    for idx in range(0, len(points_xy) - 2):
        max_curvature = max(
            float(max_curvature),
            abs(float(_curvature_from_points_xy(points_xy[idx:idx + 3]))),
        )
    return float(max_curvature)


def _is_lane_change_reference_decision(decision: object) -> bool:
    normalized = str(decision or "").strip().lower()
    return normalized in {
        "lane_change_left",
        "lane_change_right",
        "reroute",
        "prepare_lane_change_left",
        "prepare_lane_change_right",
        "execute_lane_change_left",
        "execute_lane_change_right",
    }


def _reference_first_sample_jump_m(
    previous_reference: Sequence[Mapping[str, object]] | None,
    current_reference: Sequence[Mapping[str, object]] | None,
) -> float:
    if not previous_reference or not current_reference:
        return 0.0
    try:
        previous_sample = previous_reference[0]
        current_sample = current_reference[0]
        return float(math.hypot(
            float(current_sample.get("x_ref_m", current_sample.get("x", 0.0)))
            - float(previous_sample.get("x_ref_m", previous_sample.get("x", 0.0))),
            float(current_sample.get("y_ref_m", current_sample.get("y", 0.0)))
            - float(previous_sample.get("y_ref_m", previous_sample.get("y", 0.0))),
        ))
    except Exception:
        return 0.0


def _stabilize_lane_reference_samples(
    current_reference: Sequence[Mapping[str, object]] | None,
    previous_reference: Sequence[Mapping[str, object]] | None,
    *,
    decision: object,
    max_non_lc_first_sample_jump_m: float = 2.25,
) -> tuple[List[Dict[str, object]], bool, float]:
    """Report lane-reference side jumps without freezing the active reference."""
    current_samples = [dict(sample) for sample in list(current_reference or [])]
    previous_samples = [dict(sample) for sample in list(previous_reference or [])]
    if not current_samples:
        return current_samples, False, 0.0
    jump_m = _reference_first_sample_jump_m(previous_samples, current_samples)
    jump_detected = (
        bool(previous_samples)
        and not _is_lane_change_reference_decision(decision)
        and float(jump_m) > float(max_non_lc_first_sample_jump_m)
    )
    return current_samples, bool(jump_detected), float(jump_m)


def _angle_diff_abs_rad(a_rad: float, b_rad: float) -> float:
    return abs(math.atan2(math.sin(float(a_rad) - float(b_rad)), math.cos(float(a_rad) - float(b_rad))))


def _reference_first_sample_invalid_reason(
    *,
    ego_state: Sequence[float],
    current_reference: Sequence[Mapping[str, object]] | None,
    previous_reference: Sequence[Mapping[str, object]] | None,
    decision: object,
    max_non_lc_jump_m: float = 4.0,
    max_heading_error_rad: float = 2.2,
    max_backward_m: float = 1.0,
) -> str:
    samples = list(current_reference or [])
    if not samples:
        return "empty_reference"
    if len(ego_state) < 4:
        return ""
    first = dict(samples[0])
    try:
        ref_x = float(first.get("x_ref_m", first.get("x", 0.0)))
        ref_y = float(first.get("y_ref_m", first.get("y", 0.0)))
        ref_heading = float(first.get("heading_rad", float(ego_state[3])))
        ego_x = float(ego_state[0])
        ego_y = float(ego_state[1])
        ego_yaw = float(ego_state[3])
    except Exception:
        return "invalid_reference_values"

    dx_m = ref_x - ego_x
    dy_m = ref_y - ego_y
    forward_m = math.cos(ego_yaw) * dx_m + math.sin(ego_yaw) * dy_m
    lateral_m = -math.sin(ego_yaw) * dx_m + math.cos(ego_yaw) * dy_m
    if float(forward_m) < -float(max_backward_m) and math.hypot(dx_m, dy_m) > 3.0:
        return "first_sample_behind_ego"
    if abs(float(lateral_m)) > 8.0 and float(forward_m) < 2.0:
        return "first_sample_lateral_jump"
    if _angle_diff_abs_rad(ref_heading, ego_yaw) > float(max_heading_error_rad):
        return "first_sample_heading_reversed"

    jump_m = _reference_first_sample_jump_m(previous_reference, current_reference)
    if (
        bool(previous_reference)
        and not _is_lane_change_reference_decision(decision)
        and float(jump_m) > float(max_non_lc_jump_m)
    ):
        return "first_sample_discontinuous"
    return ""


def _route_reference_fallback_samples(
    *,
    ego_state: Sequence[float],
    global_route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    target_lane_id: int,
) -> List[Dict[str, object]]:
    if len(global_route_points or []) < 2 or len(ego_state) < 4:
        return []
    ego_snapshot = {
        "x": float(ego_state[0]),
        "y": float(ego_state[1]),
        "psi": float(ego_state[3]),
    }
    route_samples = build_route_reference_samples(
        ego_snapshot=ego_snapshot,
        route_points=global_route_points,
        horizon_steps=int(horizon_steps),
        step_distance_m=float(step_distance_m),
        target_lane_id=int(target_lane_id),
    )
    return [dict(sample) for sample in route_samples]


def _reference_with_route_fallback(
    *,
    ego_state: Sequence[float],
    current_reference: Sequence[Mapping[str, object]] | None,
    previous_reference: Sequence[Mapping[str, object]] | None,
    decision: object,
    global_route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    target_lane_id: int,
) -> tuple[List[Dict[str, object]], str]:
    reason = _reference_first_sample_invalid_reason(
        ego_state=ego_state,
        current_reference=current_reference,
        previous_reference=previous_reference,
        decision=decision,
    )
    if not reason:
        return [dict(sample) for sample in list(current_reference or [])], ""
    fallback_samples = _route_reference_fallback_samples(
        ego_state=ego_state,
        global_route_points=global_route_points,
        horizon_steps=int(horizon_steps),
        step_distance_m=float(step_distance_m),
        target_lane_id=int(target_lane_id),
    )
    if fallback_samples:
        return fallback_samples, str(reason)
    return [dict(sample) for sample in list(current_reference or [])], ""


def _control_from_mpc(mpc: MPC, carla, acceleration_mps2: float, steering_angle_rad: float):
    max_accel_mps2 = max(1e-6, float(mpc.constraints.max_acceleration_mps2))
    max_brake_mps2 = max(1e-6, abs(float(mpc.constraints.min_acceleration_mps2)))
    max_steer_rad = max(1e-6, float(mpc.constraints.max_steer_rad))

    throttle = 0.0
    brake = 0.0
    if float(acceleration_mps2) >= 0.0:
        throttle = min(1.0, max(0.0, float(acceleration_mps2) / max_accel_mps2))
    else:
        brake = min(1.0, max(0.0, abs(float(acceleration_mps2)) / max_brake_mps2))

    steer = min(1.0, max(-1.0, float(steering_angle_rad) / max_steer_rad))
    return carla.VehicleControl(
        throttle=float(throttle),
        brake=float(brake),
        steer=float(steer),
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
    )


def _destroy_actors(actors: Iterable[Any]) -> None:
    for actor in actors:
        if actor is None:
            continue
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except RuntimeError:
            pass
        try:
            actor.destroy()
        except RuntimeError:
            pass


def _spawn_scenario_obstacles_from_module(
    *,
    client,
    world,
    map_obj,
    carla,
    blueprint_library,
    traffic_manager,
    traffic_manager_port: int,
    scenario_cfg: Mapping[str, object],
    route_summary,
    route_points: Sequence[Sequence[float]],
) -> List[Any]:
    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    module_name = str(obstacle_cfg.get("spawner_module", "")).strip()
    if not module_name:
        return []

    print(
        "[CARLA SCENARIO] Spawning scenario obstacles after startup global route generation "
        f"using module '{module_name}'."
    )
    module = _load_optional_module(
        module_name=str(module_name),
        purpose="scenario obstacle spawner",
    )

    spawn_fn = getattr(module, "spawn_obstacles", None)
    if not callable(spawn_fn):
        raise RuntimeError(
            f"Scenario obstacle spawner module '{module_name}' does not expose "
            "spawn_obstacles(...)."
        )

    spawned_actors = _call_with_supported_kwargs(
        spawn_fn,
        client=client,
        world=world,
        world_map=map_obj,
        carla=carla,
        blueprint_library=blueprint_library,
        traffic_manager=traffic_manager,
        traffic_manager_port=int(traffic_manager_port),
        scenario_cfg=scenario_cfg,
        route_summary=route_summary,
        route_points=route_points,
    )
    if spawned_actors is None:
        return []
    actors = [actor for actor in list(spawned_actors) if actor is not None]
    print(
        f"[CARLA SCENARIO] Scenario obstacle spawner '{module_name}' "
        f"spawned {len(actors)} actor(s)."
    )
    return actors


def _scenario_constraints_cfg(scenario_cfg: Mapping[str, object]) -> Dict[str, object]:
    constraints_cfg = dict(scenario_cfg.get("constraints", {}))
    missing_keys = [key for key in REQUIRED_CONSTRAINT_KEYS if key not in constraints_cfg]
    if len(missing_keys) > 0:
        raise RuntimeError(
            "Scenario constraints are incomplete. "
            f"Missing keys in scenario '{scenario_cfg.get('name', '<unknown>')}': {missing_keys}"
        )
    return constraints_cfg


def run_loaded_world(client, world, scenario_cfg: Mapping[str, object], carla) -> int:
    scenario_cfg = dict(scenario_cfg or {})
    carla_cfg = dict(scenario_cfg.get("carla", {}))
    sumo_cfg = dict(scenario_cfg.get("sumo", {}))
    anchors_cfg = dict(scenario_cfg.get("anchors", {}))
    planning_cfg = dict(scenario_cfg.get("planning", {}))
    camera_cfg = dict(scenario_cfg.get("camera", {}))
    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    traffic_manager_cfg = dict(scenario_cfg.get("traffic_manager", {}))

    if pygame is None and bool(camera_cfg.get("enabled", True)):
        raise RuntimeError("pygame is required for the two-camera CARLA scenario window.")

    world_map = world.get_map()
    blueprint_library = world.get_blueprint_library()
    mpc_payload = load_yaml_file(MPC_CONFIG_PATH)
    tracker_payload = load_yaml_file(TRACKER_CONFIG_PATH)
    mpc_cfg = dict(mpc_payload.get("mpc", mpc_payload))
    scenario_constraints_cfg = _scenario_constraints_cfg(scenario_cfg)
    mpc_cfg["constraints"] = dict(scenario_constraints_cfg)
    tracker_cfg = dict(tracker_payload.get("tracker", tracker_payload))
    behavior_runtime_cfg = dict(mpc_cfg.get("behavior_planner_runtime", {}))
    local_goal_cfg = dict(mpc_cfg.get("local_goal", {}))
    mpc_constraints_cfg = dict(mpc_cfg.get("constraints", {}))
    obstacle_filter_cfg = dict(mpc_cfg.get("obstacle_filter", {}))
    visualization_cfg = dict(mpc_cfg.get("visualization", {}))
    global_route_visualization_enabled = bool(
        visualization_cfg.get("show_global_route", True)
    )

    sample_distance_m = float(planning_cfg.get("waypoint_sample_distance_m", 2.0))
    lane_center_waypoints, road_cfg = build_lane_center_waypoints(
        map_obj=world_map,
        carla=carla,
        sample_distance_m=float(sample_distance_m),
    )
    global_planner = AStarGlobalPlanner(
        lane_center_waypoints=lane_center_waypoints,
        world_map=world_map,
        route_sample_distance_m=float(sample_distance_m),
    )

    spawn_anchor, destination_anchor, ego_anchor_name, destination_anchor_name = _resolve_route_anchor_transforms(
        world=world,
        carla=carla,
        world_map=world_map,
        anchors_cfg=anchors_cfg,
    )
    global_route_start_location = spawn_anchor.location
    global_route_goal_location = destination_anchor.location
    aligned_spawn_transform, spawn_waypoint = _align_transform_to_lane(world_map, carla, spawn_anchor)
    aligned_destination_transform, destination_waypoint = _align_transform_to_lane(world_map, carla, destination_anchor)
    if spawn_waypoint is None or destination_waypoint is None:
        raise RuntimeError("Could not align the spawn or destination anchors to a driving lane.")

    initial_global_route_summary = global_planner.plan_route_from_locations(
        start_location=global_route_start_location,
        goal_location=global_route_goal_location,
        fallback_start_xy=[
            float(global_route_start_location.x),
            float(global_route_start_location.y),
        ],
        fallback_goal_xy=[
            float(global_route_goal_location.x),
            float(global_route_goal_location.y),
        ],
    )
    initial_route_points: List[List[float]] = []
    if bool(initial_global_route_summary.route_found):
        initial_route_points = [
            [float(item[0]), float(item[1])]
            for item in initial_global_route_summary.route_waypoints
        ]
    topdown_focus_points: List[List[float]] = list(initial_route_points)
    if len(topdown_focus_points) == 0:
        topdown_focus_points = [
            [float(global_route_start_location.x), float(global_route_start_location.y)],
            [float(global_route_goal_location.x), float(global_route_goal_location.y)],
        ]

    _print_anchor_lookup(world, carla, ego_anchor_name)
    ego_vehicle = _spawn_vehicle(
        world=world,
        blueprint_library=blueprint_library,
        scenario_cfg=scenario_cfg,
        carla=carla,
        lane_transform=aligned_spawn_transform,
        anchor_transform=spawn_anchor,
    )

    traffic_manager = None
    traffic_manager_port = int(traffic_manager_cfg.get("port", 8000))
    if bool(traffic_manager_cfg.get("enabled", False)):
        traffic_manager_bind_error = None
        for candidate_port in range(int(traffic_manager_port), int(traffic_manager_port) + 21):
            try:
                traffic_manager = client.get_trafficmanager(int(candidate_port))
                if int(candidate_port) != int(traffic_manager_port):
                    print(
                        f"[CARLA SCENARIO] Traffic Manager port {traffic_manager_port} was busy; "
                        f"using {candidate_port} instead."
                    )
                traffic_manager_port = int(candidate_port)
                traffic_manager_bind_error = None
                break
            except RuntimeError as exc:
                message = str(exc).lower()
                if "traffic manager" in message and "bind" in message:
                    traffic_manager_bind_error = exc
                    continue
                raise
        if traffic_manager is None and traffic_manager_bind_error is not None:
            raise traffic_manager_bind_error

    actors_to_destroy = [ego_vehicle]
    actors_to_destroy.extend(
        _spawn_scenario_obstacles_from_module(
            client=client,
            world=world,
            map_obj=world_map,
            carla=carla,
            blueprint_library=blueprint_library,
            traffic_manager=traffic_manager,
            traffic_manager_port=int(traffic_manager_port),
            scenario_cfg=scenario_cfg,
            route_summary=initial_global_route_summary,
            route_points=initial_route_points,
        )
    )
    previous_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = bool(carla_cfg.get("synchronous_mode", True))
    settings.fixed_delta_seconds = CARLA_FIXED_DELTA_SECONDS
    world.apply_settings(settings)
    if traffic_manager is not None:
        try:
            traffic_manager.set_synchronous_mode(
                bool(traffic_manager_cfg.get("synchronous_mode", settings.synchronous_mode))
            )
        except Exception:
            pass
    sumo_bridge = None
    if bool(sumo_cfg.get("enabled", False)):
        from opencda_scenario.sumo_assets import ensure_sumo_assets
        from opencda_scenario.sumo_bridge import OpenCDASumoBridge

        sumo_asset_dir = ensure_sumo_assets(
            scenario_cfg=scenario_cfg,
            sumo_cfg=sumo_cfg,
        )
        sumo_bridge = OpenCDASumoBridge(
            client=client,
            world=world,
            sumo_cfg=sumo_cfg,
            sumo_asset_dir=sumo_asset_dir,
        )
        print(
            "[CARLA SCENARIO] OpenCDA SUMO co-simulation enabled "
            f"(assets={sumo_asset_dir})."
        )
    print(
        f"[CARLA SCENARIO] Simulation tick = {CARLA_FIXED_DELTA_SECONDS:.3f}s "
        f"({1.0 / CARLA_FIXED_DELTA_SECONDS:.0f} Hz), "
        f"MPC prediction dt = {float(mpc_cfg.get('plan_dt_s', 0.2)):.3f}s"
    )
    realtime_pacing_enabled = bool(carla_cfg.get("realtime_pacing_enabled", False))
    realtime_pacing_factor = max(1e-3, float(carla_cfg.get("realtime_pacing_factor", 1.0)))
    realtime_loop_period_s = 0.0
    if realtime_pacing_enabled:
        realtime_loop_period_s = CARLA_FIXED_DELTA_SECONDS / float(realtime_pacing_factor)
        print(
            "[CARLA SCENARIO] Real-time pacing enabled "
            f"(wall_period={float(realtime_loop_period_s):.3f}s sim_dt={CARLA_FIXED_DELTA_SECONDS:.3f}s factor={float(realtime_pacing_factor):.3f})"
        )

    image_width_px = int(camera_cfg.get("image_size_x", 960))
    image_height_px = int(camera_cfg.get("image_size_y", 540))
    hud_panel_width_px = max(0, int(camera_cfg.get("hud_panel_width_px", 320)))
    camera_fov_deg = float(camera_cfg.get("fov", 90.0))
    camera_blueprint = _camera_blueprint(
        world=world,
        width_px=image_width_px,
        height_px=image_height_px,
        fov_deg=float(camera_fov_deg),
    )
    topdown_calibration_matrix = _camera_calibration_matrix(
        width_px=image_width_px,
        height_px=image_height_px,
        fov_deg=float(camera_fov_deg),
    )

    topdown_camera = None
    topdown_queue = None
    chase_camera = None
    chase_queue = None
    display = None

    if bool(camera_cfg.get("enabled", True)):
        pygame.init()
        pygame.font.init()
        display = pygame.display.set_mode((int(image_width_px * 2 + hud_panel_width_px), int(image_height_px)))
        pygame.display.set_caption(f"CARLA {scenario_cfg.get('name', 'scenario')} - Topdown | Chase")

        topdown_cfg = dict(camera_cfg.get("topdown", {}))
        chase_cfg = dict(camera_cfg.get("chase", {}))
        topdown_world_fixed = bool(topdown_cfg.get("world_fixed", False))
        if topdown_world_fixed:
            topdown_transform = _world_fixed_topdown_transform(
                carla=carla,
                focus_points_xy=topdown_focus_points,
                image_width_px=int(image_width_px),
                image_height_px=int(image_height_px),
                fov_deg=float(camera_fov_deg),
                min_height_m=float(topdown_cfg.get("height", 65.0)),
                padding_m=float(topdown_cfg.get("padding_m", 20.0)),
            )
        else:
            topdown_transform = carla.Transform(
                carla.Location(x=0.0, y=0.0, z=float(topdown_cfg.get("height", 65.0))),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
        chase_transform = carla.Transform(
            carla.Location(
                x=-float(chase_cfg.get("back", 8.0)),
                y=0.0,
                z=float(chase_cfg.get("height", 2.8)),
            ),
            carla.Rotation(
                pitch=float(chase_cfg.get("pitch", -10.0)),
                yaw=0.0,
                roll=0.0,
            ),
        )
        topdown_camera, topdown_queue = _spawn_camera(
            world,
            carla,
            camera_blueprint,
            topdown_transform,
            parent=None if topdown_world_fixed else ego_vehicle,
        )
        chase_camera, chase_queue = _spawn_camera(
            world,
            carla,
            camera_blueprint,
            chase_transform,
            parent=ego_vehicle,
        )
        actors_to_destroy.extend([topdown_camera, chase_camera])
    hud_font = pygame.font.SysFont("monospace", 14) if display is not None else None
    hud_mode = str(camera_cfg.get("hud_mode", "compact")).strip().lower()
    if hud_mode in {"debug", "detailed"}:
        hud_mode = "full"
    if hud_mode not in {"compact", "full", "off"}:
        hud_mode = "compact"

    tracker = Tracker(tracker_cfg=tracker_cfg)
    mpc = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)
    mpc_cost_history: List[Dict[str, object]] = []
    lane_ref_history: List[Dict[str, object]] = []
    control_history: List[Dict[str, object]] = []
    temp_destination_history: List[Dict[str, object]] = []
    scenario_timeout_s = max(0.0, float(runtime_cfg.get("timeout_s", 0.0)))
    run_start_wall_time_s = float(time.perf_counter())
    run_start_sim_time_s: float | None = None
    last_status_ego_state: List[float] | None = None
    last_status_sim_time_s: float | None = None
    run_status: Dict[str, object] = {
        "finished": False,
        "reason": "not_finished",
        "scenario_name": str(scenario_cfg.get("name", "scenario")),
        "timeout_s": float(scenario_timeout_s),
        "destination_reached_threshold_m": float(mpc.destination_reached_threshold_m),
    }
    metrics_cfg = dict(scenario_cfg.get("metrics", {}))
    last_cost_artifact_write_wall_time_s = 0.0
    obstacle_height_filter_enabled = bool(obstacle_filter_cfg.get("enable_height_filter", True))
    obstacle_vertical_clearance_margin_m = float(
        obstacle_filter_cfg.get("vertical_clearance_margin_m", 1.0)
    )
    default_obstacle_height_m = float(
        obstacle_filter_cfg.get("default_obstacle_height_m", 2.0)
    )
    obstacle_tracking_distance_m = max(
        0.0,
        float(obstacle_filter_cfg.get("tracking_distance_m", 30.0)),
    )
    max_dynamic_obstacles = max(
        0,
        int(obstacle_filter_cfg.get("max_dynamic_obstacles", 64)),
    )
    max_planning_obstacles = max(
        0,
        int(obstacle_filter_cfg.get("max_planning_obstacles", max_dynamic_obstacles)),
    )
    ego_bbox = getattr(ego_vehicle, "bounding_box", None)
    ego_bbox_extent = getattr(ego_bbox, "extent", None)
    ego_length_m = max(
        0.5,
        float(getattr(ego_bbox_extent, "x", 2.25)) * 2.0,
    )
    ego_height_m = max(
        0.5,
        float(getattr(ego_bbox_extent, "z", 0.9)) * 2.0,
    )
    metrics_recorder = EvaluationMetricsRecorder(
        ego_length_m=float(ego_length_m),
        lateral_conflict_width_m=float(metrics_cfg.get("lateral_conflict_width_m", 3.5)),
        pet_conflict_radius_m=float(metrics_cfg.get("pet_conflict_radius_m", 3.0)),
        pet_bin_size_m=float(metrics_cfg.get("pet_bin_size_m", 3.0)),
    )
    if bool(metrics_cfg.get("collision_sensor_enabled", True)):
        try:
            collision_sensor = _spawn_collision_sensor(
                world=world,
                carla=carla,
                ego_vehicle=ego_vehicle,
                metrics_recorder=metrics_recorder,
            )
            actors_to_destroy.append(collision_sensor)
        except Exception as exc:
            print(f"[Metrics] Collision sensor disabled: {exc}")

    final_destination_state = [
        float(aligned_destination_transform.location.x),
        float(aligned_destination_transform.location.y),
        0.0,
        float(math.radians(aligned_destination_transform.rotation.yaw)),
    ]
    initial_lane_context = global_planner.get_local_lane_context(
        x_m=float(aligned_spawn_transform.location.x),
        y_m=float(aligned_spawn_transform.location.y),
        heading_rad=float(math.radians(aligned_spawn_transform.rotation.yaw)),
        z_m=float(aligned_spawn_transform.location.z),
    )
    initial_lane_count = int(initial_lane_context.get("lane_count", 0))
    if initial_lane_count > 0:
        road_cfg["lane_count"] = int(initial_lane_count)
    lane_count = max(1, int(road_cfg.get("lane_count", 1)))
    initial_allowed_lane_ids = _allowed_lane_ids_from_context(
        local_context=initial_lane_context,
        fallback_lane_count=int(lane_count),
        fallback_lane_id=int(initial_lane_context.get("lane_id", 0)),
    )
    selected_lane_id = _clamp_lane_id_to_allowed(
        int(initial_lane_context.get("lane_id", 0)),
        initial_allowed_lane_ids,
    )
    current_applied_behavior = "lane_follow"
    previous_lane_center_reference: List[Dict[str, object]] = []
    last_reference_stabilized = False
    last_reference_jump_m = 0.0
    last_reference_fallback_reason = ""
    stop_sign_watchdog_key: str | None = None
    stop_sign_watchdog_start_sim_time_s: float | None = None
    original_max_velocity_mps = float(mpc.constraints.max_velocity_mps)
    current_target_v_mps = float(original_max_velocity_mps)
    active_plan_max_velocity_mps = float(original_max_velocity_mps)
    lane_scores: Dict[int, float] = {
        int(lane_id): 1.0 for lane_id in initial_allowed_lane_ids
    }
    current_route_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
        getattr(initial_global_route_summary, "optimal_lane_id", 0),
        initial_allowed_lane_ids,
    )
    display_reference_maneuver = normalize_macro_maneuver(
        getattr(initial_global_route_summary, "next_macro_maneuver", "straight")
    )
    traffic_light_debug: Dict[str, object] = {
        "signal_state": "unknown",
        "signal_distance_m": None,
        "signal_found": False,
        "should_stop_now": False,
        "stop_latched": False,
        "stop_decision_active": False,
        "signal_actor_name": "",
    }

    current_acceleration_mps2 = 0.0
    current_steering_rad = 0.0
    _hud_idm_state: str = "IDM: free"
    _mpc_fail_count: int = 0
    active_global_route_points: List[List[float]] = list(initial_route_points)
    current_route_summary = initial_global_route_summary
    temporary_route_summary = initial_global_route_summary
    active_reference_maneuver = normalize_macro_maneuver(
        getattr(initial_global_route_summary, "next_macro_maneuver", "straight")
    )
    temporary_destination_state: List[float] | None = None
    reroute_route_follow_latched = False
    # True for exactly one replan tick after the planner first outputs
    # "reroute".  Forces one immediate MPC replan to execute the reroute,
    # then clears — preventing 20 Hz forced replanning while reroute is active.
    _reroute_execution_pending: bool = False
    planned_trajectory: List[List[float]] = []
    _cached_blocking_obstacle_id: str = ""
    _cached_decision_reason: str = ""
    _mpc_replan_count: int = 0
    # Wall-clock time [s] when ego speed first dropped below 0.1 m/s in the
    # current standstill episode; None while the vehicle is moving.
    _standstill_start_wall_time_s: float | None = None
    cached_control_sequence: np.ndarray | None = None
    cached_control_step_idx = 0
    raw_current_lane_id = int(initial_lane_context.get("lane_id", 0))
    last_world_debug_route_draw_time_s = -float("inf")
    last_temp_destination_mismatch_log_time_s = -float("inf")

    # ---- Rule-based behavior planner -------------------------------- #
    rule_planner_cfg = dict(behavior_runtime_cfg.get("rule_based", {}))
    lane_safety_scorer = LaneSafetyScorer(
        d_safe_m=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_distance_safe_m",
                legacy_key="d_safe_m",
                default=12.0,
            )
        ),
        rear_d_safe_m=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_rear_distance_safe_m",
                legacy_key="rear_d_safe_m",
                default=_behavior_runtime_value(
                    behavior_runtime_cfg,
                    rule_planner_cfg,
                    key="lane_safety_distance_safe_m",
                    legacy_key="d_safe_m",
                    default=5.0,
                ),
            )
        ),
        ttc_safe_s=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_ttc_safe_s",
                legacy_key="ttc_safe_s",
                default=3.0,
            )
        ),
        rear_ttc_safe_s=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_rear_ttc_safe_s",
                legacy_key="rear_ttc_safe_s",
                default=_behavior_runtime_value(
                    behavior_runtime_cfg,
                    rule_planner_cfg,
                    key="lane_safety_ttc_safe_s",
                    legacy_key="ttc_safe_s",
                    default=2.0,
                ),
            )
        ),
        sigmoid_k=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_ttc_sigmoid_gain",
                legacy_key="sigmoid_k",
                default=2.0,
            )
        ),
        ttc_history_size=int(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_ttc_history_length",
                legacy_key="ttc_history_size",
                default=8,
            )
        ),
        ttc_epsilon_mps=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_ttc_epsilon_mps",
                legacy_key="ttc_epsilon_mps",
                default=0.1,
            )
        ),
        infinite_ttc_cap_s=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_safety_infinite_ttc_cap_s",
                legacy_key="infinite_ttc_cap_s",
                default=15.0,
            )
        ),
    )
    cooperative_message_path = str(
        rule_planner_cfg.get(
            "cooperative_message_path",
            behavior_runtime_cfg.get("cooperative_message_path", CP_MESSAGE_PATH),
        )
    ).strip() or CP_MESSAGE_PATH
    reset_cp_message_payload(message_path=cooperative_message_path)

    rule_planner = RuleBasedBehaviorPlanner(
        hysteresis_delta=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="decision_hysteresis_delta",
                legacy_key="hysteresis_delta",
                default=0.10,
            )
        ),
        lateral_complete_m=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_change_completion_lateral_threshold_m",
                legacy_key="lateral_complete_m",
                default=0.75,
            )
        ),
        heading_complete_rad=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_change_completion_heading_threshold_rad",
                legacy_key="heading_complete_rad",
                default=0.20,
            )
        ),
        lane_change_target_safety_threshold=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_change_target_safety_threshold",
                legacy_key="intersection_lane_change_safety_threshold",
                default=0.10,
            )
        ),
        lane_change_abort_safety_threshold=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_change_abort_safety_threshold",
                legacy_key="lane_change_abort_safety_threshold",
                default=0.50,
            )
        ),
        optimal_lane_unsafe_threshold=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="optimal_lane_unsafe_threshold",
                default=0.50,
            )
        ),
        cooperative_message_check_frequency_hz=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="cooperative_message_check_frequency_hz",
                legacy_key="cooperative_message_check_frequency_hz",
                default=1.0,
            )
        ),
        stop_sign_wait_duration_s=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="stop_sign_wait_duration_s",
                default=3.0,
            )
        ),
        emergency_brake_follow_buffer_m=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="emergency_brake_follow_buffer_m",
                legacy_key="emergence_stop_follow_buffer_m",
                default=1.0,
            )
        ),
        prepare_lane_change_min_hold_s=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="prepare_lane_change_min_hold_s",
                default=0.5,
            )
        ),
        execute_lane_change_min_hold_s=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="execute_lane_change_min_hold_s",
                default=1.0,
            )
        ),
        lane_keep_min_hold_s=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="lane_keep_min_hold_s",
                default=0.3,
            )
        ),
        candidate_route_deviation_weight=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="candidate_route_deviation_weight",
                default=0.15,
            )
        ),
        candidate_lane_change_weight=float(
            _behavior_runtime_value(
                behavior_runtime_cfg,
                rule_planner_cfg,
                key="candidate_lane_change_weight",
                default=0.20,
            )
        ),
        cp_message_path=str(cooperative_message_path),
    )
    moving_obstacle_speed_threshold_mps = float(
        _behavior_runtime_value(
            behavior_runtime_cfg,
            rule_planner_cfg,
            key="intersection_obstacle_moving_speed_threshold_mps",
            legacy_key="intersection_obstacle_moving_speed_threshold_mps",
            default=0.5,
        )
    )
    static_obstacle_replan_lane_safety_threshold = float(
        _behavior_runtime_value(
            behavior_runtime_cfg,
            rule_planner_cfg,
            key="intersection_static_obstacle_replan_lane_safety_threshold",
            legacy_key="intersection_static_obstacle_replan_lane_safety_threshold",
            default=0.5,
        )
    )
    static_obstacle_replan_cooldown_s = float(
        _behavior_runtime_value(
            behavior_runtime_cfg,
            rule_planner_cfg,
            key="intersection_static_obstacle_replan_cooldown_s",
            legacy_key="intersection_static_obstacle_replan_cooldown_s",
            default=1.0,
        )
    )
    traffic_light_stop_cfg = dict(
        behavior_runtime_cfg.get(
            "traffic_light_stop",
            rule_planner_cfg.get("traffic_light_stop", {}),
        )
    )
    traffic_light_stop_enabled = bool(traffic_light_stop_cfg.get("enabled", False))
    traffic_light_stop_search_distance_m = max(
        1.0,
        float(traffic_light_stop_cfg.get("search_distance_m", 100.0)),
    )
    traffic_light_stop_buffer_m = max(
        0.0,
        float(traffic_light_stop_cfg.get("stop_buffer_m", 2.0)),
    )
    traffic_light_unknown_release_s = max(
        0.0,
        float(traffic_light_stop_cfg.get("unknown_release_s", 6.0)),
    )
    post_stop_lane_change_freeze_s = max(
        0.0,
        float(behavior_runtime_cfg.get("post_stop_lane_change_freeze_s", 3.0)),
    )
    stop_release_temp_lookahead_smooth_s = max(
        0.0,
        float(behavior_runtime_cfg.get("stop_release_temp_lookahead_smooth_s", 1.5)),
    )
    lane_change_commit_cooldown_s = max(
        0.0,
        float(behavior_runtime_cfg.get("lane_change_commit_cooldown_s", 4.0)),
    )
    lane_change_abort_cooldown_s = max(
        0.0,
        float(behavior_runtime_cfg.get("lane_change_abort_cooldown_s", 3.0)),
    )
    last_static_intersection_replan_time_s = -float("inf")
    post_stop_lane_change_freeze_until_sim_time_s = -float("inf")
    stop_release_temp_smooth_until_sim_time_s = -float("inf")
    lane_change_commit_cooldown_until_sim_time_s = -float("inf")
    lane_change_abort_cooldown_until_sim_time_s = -float("inf")
    traffic_signal_unknown_start_sim_time_s: float | None = None
    traffic_signal_last_known_state = "unknown"
    traffic_signal_last_known_source = "none"
    traffic_signal_last_known_actor_id = ""
    traffic_signal_last_known_actor_name = ""
    traffic_signal_last_known_distance_m: float | None = None
    traffic_signal_unknown_duration_s = 0.0
    traffic_signal_unknown_release_active = False
    traffic_signal_unknown_release_reason = ""
    print("[CARLA SCENARIO] Rule-based behavior planner initialized.")
    scenario_runtime_module_name = str(runtime_cfg.get("module", "")).strip()
    scenario_runtime_module = _load_optional_module(
        module_name=scenario_runtime_module_name,
        purpose="scenario runtime",
    )
    use_cp_obstacle_pipeline = bool(
        scenario_runtime_module_name.startswith("opencda_scenario.")
    )
    scenario_runtime_state = _initialize_scenario_runtime_state(
        module=scenario_runtime_module,
        world=world,
        world_map=world_map,
        carla=carla,
        scenario_cfg=scenario_cfg,
        traffic_manager_port=int(traffic_manager_port),
        tracker_cfg=tracker_cfg,
        obstacle_filter_cfg=obstacle_filter_cfg,
        prediction_dt_s=float(mpc.dt_s),
        prediction_horizon_s=float(mpc.horizon_s),
    )
    runtime_vehicle_markers = (
        list(scenario_runtime_state.get("vehicle_markers", []) or [])
        if isinstance(scenario_runtime_state, Mapping)
        else []
    )
    runtime_spawned_vehicle_ids = (
        list(scenario_runtime_state.get("manual_vehicle_actor_ids", []) or [])
        if isinstance(scenario_runtime_state, Mapping)
        else []
    )
    should_spawn_fallback_npc = (
        bool(runtime_cfg.get("spawn_fallback_npc_when_no_markers", True))
        and len(runtime_vehicle_markers) == 0
        and len(runtime_spawned_vehicle_ids) == 0
    )
    if bool(should_spawn_fallback_npc):
        fallback_npc_actors = _spawn_fallback_npc_traffic(
            world=world,
            world_map=world_map,
            blueprint_library=blueprint_library,
            carla=carla,
            ego_vehicle=ego_vehicle,
            traffic_manager_port=int(traffic_manager_port),
            count=int(runtime_cfg.get("fallback_npc_count", 16)),
            min_distance_m=float(runtime_cfg.get("fallback_npc_min_distance_m", 12.0)),
            max_distance_m=float(runtime_cfg.get("fallback_npc_max_distance_m", 120.0)),
        )
        actors_to_destroy.extend(list(fallback_npc_actors))

    try:
        obstacle_prefix = str(obstacle_cfg.get("environment_name_prefix", "")).strip()
        cached_static_environment_obstacles: List[dict] = []
        if obstacle_prefix and not bool(use_cp_obstacle_pipeline):
            cached_static_environment_obstacles = _collect_environment_obstacle_snapshots(
                world=world,
                map_obj=world_map,
                carla=carla,
                obstacle_prefix=obstacle_prefix,
            )
            print(
                f"[CARLA SCENARIO] Found {len(cached_static_environment_obstacles)} static environment obstacles "
                f"with prefix '{obstacle_prefix}'."
            )
            for snapshot in cached_static_environment_obstacles:
                print(
                    "[CARLA SCENARIO] Obstacle "
                    f"{snapshot['vehicle_id']} at ({snapshot['x']:.3f}, {snapshot['y']:.3f}) "
                    f"size=({snapshot['length_m']:.2f}m, {snapshot['width_m']:.2f}m)"
                )
        next_tick_wall_time_s = time.monotonic()
        # Cached planning state (updated at MPC replan rate, not every tick)
        rolling_target_distance_m = float(local_goal_cfg.get("dynamic_lookahead_min_distance_m", 20.0))
        ego_snapshot: Dict[str, float] = {
            "x": float(aligned_spawn_transform.location.x),
            "y": float(aligned_spawn_transform.location.y),
            "v": 0.0,
            "psi": float(math.radians(aligned_spawn_transform.rotation.yaw)),
        }
        base_destination_state: List[float] = list(final_destination_state) + [0.0]
        target_distance_for_destination_m = float(rolling_target_distance_m)
        local_allowed_lane_ids = list(initial_allowed_lane_ids)
        # Last known good lane values from a non-junction road segment.
        # Inside a junction, canonical_lane_id_for_waypoint() always returns 1
        # for every connector (unique road_ids, no lane siblings).  We freeze
        # these values at junction entry so ego_lane_id, selected_lane_id, and
        # optimal_lane_id stay coherent throughout the intersection.
        _pre_junction_lane_id: int = 0
        _pre_junction_allowed_lane_ids: List[int] = list(initial_allowed_lane_ids)
        _pre_junction_optimal_lane_id: int = 0
        cached_obstacle_contours: List[dict] = []
        cached_predicted_obstacle_trajectories: Dict[str, List[dict]] = {}
        lane_prediction_risks: Dict[int, Dict[str, object]] = {}
        cached_prediction_risk_summary = "clear"
        cached_candidate_summary = "keep_lane->L0 cost=0.00 ok"
        cached_candidate_detail_summary = ""
        cached_planner_decision = "lane_follow"
        cached_planner_lc_state = rule_planner.lc_state
        cached_planner_target_lane_id = int(selected_lane_id)
        cached_planner_selected_lane_id = int(selected_lane_id)
        cached_motion_decision = "lane_follow"
        cached_temp_destination_decision = "lane_follow"
        cached_reference_target_lane_id = int(selected_lane_id)
        cached_mpc_max_velocity_mps = float(original_max_velocity_mps)
        cached_hud_temp_lane_prompt = 0
        render_tick_counter = 0
        while True:
            if realtime_loop_period_s > 0.0:
                now_wall_time_s = time.monotonic()
                sleep_duration_s = float(next_tick_wall_time_s) - float(now_wall_time_s)
                if sleep_duration_s > 0.0:
                    time.sleep(float(sleep_duration_s))
                else:
                    next_tick_wall_time_s = float(now_wall_time_s)
            if sumo_bridge is not None:
                sumo_bridge.tick()
            else:
                world.tick()
            if realtime_loop_period_s > 0.0:
                next_tick_wall_time_s += float(realtime_loop_period_s)
            for event in pygame.event.get() if display is not None else []:
                if event.type == pygame.QUIT:
                    run_status.update({"finished": True, "reason": "user_closed_window"})
                    return 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                    hud_mode = {
                        "compact": "full",
                        "full": "off",
                        "off": "compact",
                    }.get(str(hud_mode), "compact")

            ego_state = _world_state_from_vehicle(ego_vehicle)
            last_status_ego_state = list(ego_state)
            if math.hypot(
                float(ego_state[0]) - float(final_destination_state[0]),
                float(ego_state[1]) - float(final_destination_state[1]),
            ) <= float(mpc.destination_reached_threshold_m):
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                destination_distance_m = math.hypot(
                    float(ego_state[0]) - float(final_destination_state[0]),
                    float(ego_state[1]) - float(final_destination_state[1]),
                )
                run_status.update({
                    "finished": True,
                    "reason": "destination_reached",
                    "destination_distance_m": float(destination_distance_m),
                })
                print("[SCENARIO FINISHED] reason=destination_reached")
                return 0

            sim_time_s = float(world.get_snapshot().timestamp.elapsed_seconds)
            wall_time_s = float(time.perf_counter())
            last_status_sim_time_s = float(sim_time_s)
            if run_start_sim_time_s is None:
                run_start_sim_time_s = float(sim_time_s)
            if float(scenario_timeout_s) > 0.0 and float(sim_time_s) - float(run_start_sim_time_s) >= float(scenario_timeout_s):
                run_status.update({
                    "finished": True,
                    "reason": "timeout",
                    "sim_elapsed_s": float(sim_time_s) - float(run_start_sim_time_s),
                })
                print(f"[SCENARIO FINISHED] reason=timeout elapsed={float(sim_time_s) - float(run_start_sim_time_s):.2f}s")
                return 0

            dynamic_object_snapshots: List[dict] = []
            if bool(use_cp_obstacle_pipeline):
                _unused_snapshots, scenario_runtime_state = _apply_scenario_dynamic_obstacle_filter(
                    module=scenario_runtime_module,
                    runtime_state=scenario_runtime_state,
                    world=world,
                    world_map=world_map,
                    carla=carla,
                    scenario_cfg=scenario_cfg,
                    object_snapshots=[],
                    sim_time_s=float(sim_time_s),
                    wall_time_s=float(wall_time_s),
                )
                dynamic_object_snapshots = []
            else:
                dynamic_object_snapshots, scenario_runtime_state = _apply_scenario_dynamic_obstacle_filter(
                    module=scenario_runtime_module,
                    runtime_state=scenario_runtime_state,
                    world=world,
                    world_map=world_map,
                    carla=carla,
                    scenario_cfg=scenario_cfg,
                    object_snapshots=_collect_vehicle_snapshots(
                        world,
                        ego_vehicle,
                        sumo_bridge=sumo_bridge,
                        max_distance_m=float(obstacle_tracking_distance_m),
                        max_snapshots=(int(max_dynamic_obstacles) if int(max_dynamic_obstacles) > 0 else None),
                    ),
                    sim_time_s=float(sim_time_s),
                    wall_time_s=float(wall_time_s),
                )
                dynamic_object_snapshots = _filter_snapshots_by_distance(
                    ego_x_m=float(ego_state[0]),
                    ego_y_m=float(ego_state[1]),
                    object_snapshots=dynamic_object_snapshots,
                    max_distance_m=float(obstacle_tracking_distance_m),
                    max_snapshots=(int(max_dynamic_obstacles) if int(max_dynamic_obstacles) > 0 else None),
                )
            metrics_obstacle_snapshots = (
                load_obstacle_snapshots(message_path=cooperative_message_path)
                if bool(use_cp_obstacle_pipeline)
                else list(dynamic_object_snapshots) + list(cached_static_environment_obstacles)
            )
            metrics_recorder.update(
                ego_state=ego_state,
                obstacle_snapshots=metrics_obstacle_snapshots,
                sim_time_s=float(sim_time_s),
                behavior_decision=str(current_applied_behavior),
                fsm_state=str(cached_planner_lc_state),
                blocking_obstacle_id=str(_cached_blocking_obstacle_id),
                decision_reason=str(_cached_decision_reason),
            )
            scenario_route_summary, scenario_route_points, scenario_runtime_state = _maybe_apply_scenario_global_route_update(
                module=scenario_runtime_module,
                runtime_state=scenario_runtime_state,
                world=world,
                world_map=world_map,
                carla=carla,
                scenario_cfg=scenario_cfg,
                global_planner=global_planner,
                ego_vehicle=ego_vehicle,
                sumo_bridge=sumo_bridge,
                ego_transform=ego_vehicle.get_transform(),
                goal_location=aligned_destination_transform.location,
                object_snapshots=dynamic_object_snapshots,
                current_route_summary=current_route_summary,
                active_global_route_points=active_global_route_points,
                sim_time_s=float(sim_time_s),
                wall_time_s=float(wall_time_s),
            )
            if scenario_route_summary is not None and scenario_route_points is not None:
                if len(scenario_route_points) > 0:
                    active_global_route_points = [
                        [float(point[0]), float(point[1])]
                        for point in list(scenario_route_points)
                        if isinstance(point, Sequence) and len(point) >= 2
                    ]
                current_route_summary = scenario_route_summary
            traffic_signal_context: Dict[str, object] = {}
            traffic_stop_target = None
            traffic_signal_state = "unknown"
            tick_traffic_stop_target = None
            tick_traffic_signal_context = None
            tick_ego_in_junction = False
            if bool(traffic_light_stop_enabled):
                tick_ego_transform = ego_vehicle.get_transform()
                tick_ego_waypoint = world_map.get_waypoint(
                    tick_ego_transform.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                tick_ego_in_junction = bool(getattr(tick_ego_waypoint, "is_junction", False))
                if not bool(tick_ego_in_junction):
                    tick_traffic_stop_target = find_stop_target_from_ego(
                        world_map=world_map,
                        carla=carla,
                        ego_transform=tick_ego_transform,
                        global_route_points=active_global_route_points,
                        search_distance_m=float(traffic_light_stop_search_distance_m),
                        query_key="ego",
                    )
                tick_traffic_signal_context = find_relevant_signal_context(
                    world=world,
                    ego_vehicle=ego_vehicle,
                    ego_transform=tick_ego_transform,
                    stop_target=tick_traffic_stop_target,
                    max_actor_position_match_distance_m=float(
                        traffic_light_stop_search_distance_m
                    ),
                )
                if (
                    not bool(tick_ego_in_junction)
                    and tick_traffic_stop_target is None
                    and _signal_state_requires_stop(
                        dict(tick_traffic_signal_context or {}).get("signal_state", "")
                    )
                ):
                    tick_traffic_stop_target = _fallback_signal_stop_target_from_ego(
                        world_map=world_map,
                        carla=carla,
                        ego_transform=tick_ego_transform,
                        signal_context=tick_traffic_signal_context,
                        search_distance_m=float(traffic_light_stop_search_distance_m),
                        stop_buffer_m=float(traffic_light_stop_buffer_m),
                    )
                    if isinstance(tick_traffic_stop_target, Mapping):
                        tick_traffic_signal_context = dict(tick_traffic_signal_context or {})
                        tick_traffic_signal_context["fallback_stop_target"] = True
                        tick_traffic_signal_context["stop_target_source"] = "fallback_signal_stop_target"
            if bool(use_cp_obstacle_pipeline):
                traffic_signal_context = dict(tick_traffic_signal_context or {})
                traffic_stop_target = (
                    dict(tick_traffic_stop_target)
                    if isinstance(tick_traffic_stop_target, Mapping)
                    else None
                )
                traffic_signal_state = str(
                    traffic_signal_context.get("signal_state", "unknown")
                )
            if not bool(use_cp_obstacle_pipeline):
                tracker.update(
                    obstacle_snapshots=dynamic_object_snapshots,
                    timestamp_s=sim_time_s,
                    next_signal_context=tick_traffic_signal_context,
                    next_stop_target=tick_traffic_stop_target,
                )
                tracked_signal_context = tracker.get_next_signal_context()
                traffic_signal_context = dict(tracked_signal_context or {})
                tracked_stop_target = traffic_signal_context.get("stop_target", None)
                traffic_stop_target = (
                    dict(tracked_stop_target)
                    if isinstance(tracked_stop_target, Mapping)
                    else None
                )
                traffic_signal_state = str(
                    traffic_signal_context.get("signal_state", "unknown")
                )

            normalized_tick_signal_state = str(traffic_signal_state or "unknown").strip().lower()
            traffic_signal_unknown_release_active = False
            traffic_signal_unknown_release_reason = ""
            if normalized_tick_signal_state in {"red", "yellow", "green"}:
                traffic_signal_unknown_start_sim_time_s = None
                traffic_signal_unknown_duration_s = 0.0
                traffic_signal_last_known_state = str(normalized_tick_signal_state)
                traffic_signal_last_known_source = str(
                    traffic_signal_context.get("signal_source", "none")
                )
                traffic_signal_last_known_actor_id = str(
                    traffic_signal_context.get("signal_actor_id", "") or ""
                )
                traffic_signal_last_known_actor_name = str(
                    traffic_signal_context.get("signal_actor_name", "") or ""
                )
                try:
                    traffic_signal_last_known_distance_m = float(
                        traffic_signal_context.get("signal_distance_m", "")
                    )
                except Exception:
                    traffic_signal_last_known_distance_m = None
            else:
                if traffic_signal_unknown_start_sim_time_s is None:
                    traffic_signal_unknown_start_sim_time_s = float(sim_time_s)
                traffic_signal_unknown_duration_s = max(
                    0.0,
                    float(sim_time_s) - float(traffic_signal_unknown_start_sim_time_s),
                )
            if isinstance(traffic_signal_context, Mapping):
                traffic_signal_context["signal_unknown_duration_s"] = float(
                    traffic_signal_unknown_duration_s
                )
                traffic_signal_context["last_known_signal_state"] = str(
                    traffic_signal_last_known_state
                )
                traffic_signal_context["last_known_signal_source"] = str(
                    traffic_signal_last_known_source
                )
                traffic_signal_context["last_known_signal_actor_id"] = str(
                    traffic_signal_last_known_actor_id
                )
                traffic_signal_context["last_known_signal_actor_name"] = str(
                    traffic_signal_last_known_actor_name
                )
                traffic_signal_context["last_known_signal_distance_m"] = (
                    ""
                    if traffic_signal_last_known_distance_m is None
                    else float(traffic_signal_last_known_distance_m)
                )
            if (
                float(traffic_light_unknown_release_s) > 0.0
                and normalized_tick_signal_state == "unknown"
                and float(traffic_signal_unknown_duration_s) >= float(traffic_light_unknown_release_s)
                and bool(getattr(rule_planner, "stop", False))
                and str(normalize_behavior_decision(current_applied_behavior)) == "stop_at_intersection"
                and not bool(tick_ego_in_junction)
                and not isinstance(traffic_stop_target, Mapping)
            ):
                if hasattr(rule_planner, "_clear_stop_state"):
                    rule_planner._clear_stop_state()
                traffic_signal_unknown_release_active = True
                traffic_signal_unknown_release_reason = "traffic_signal_lost_timeout"
                traffic_signal_context["unknown_signal_release"] = True
                traffic_signal_context["unknown_signal_release_reason"] = str(
                    traffic_signal_unknown_release_reason
                )
                cached_control_sequence = None
                cached_control_step_idx = 0
                current_acceleration_mps2 = 0.0
                current_steering_rad = 0.0
                if hasattr(mpc, "clear_previous_solution_seed"):
                    mpc.clear_previous_solution_seed()

            # --- Standstill timer ---
            if float(ego_state[2]) < 0.1:
                if _standstill_start_wall_time_s is None:
                    _standstill_start_wall_time_s = float(wall_time_s)
            else:
                _standstill_start_wall_time_s = None

            force_behavior_replan = False
            if _has_pending_lane_closure_reroute_request(cooperative_message_path):
                force_behavior_replan = True
            if bool(_reroute_execution_pending):
                force_behavior_replan = True
            cached_step_acceleration_mps2 = _current_cached_step_acceleration_mps2(
                cached_control_sequence=cached_control_sequence,
                cached_control_step_idx=int(cached_control_step_idx),
            )
            if bool(is_fixed_stop_decision(current_applied_behavior)):
                if float(ego_state[2]) <= 0.25:
                    force_behavior_replan = True
            elif (
                not bool(is_emergency_brake_decision(current_applied_behavior))
                and _should_force_stationary_release_replan(
                    ego_speed_mps=float(ego_state[2]),
                    zero_speed_threshold_mps=0.0,
                    current_behavior=str(current_applied_behavior),
                    target_v_ref_mps=float(current_target_v_mps),
                    current_acceleration_mps2=float(current_acceleration_mps2),
                    cached_step_acceleration_mps2=cached_step_acceleration_mps2,
                )
            ):
                force_behavior_replan = True

            # --- Involuntary standstill recovery ---
            # If the vehicle has been stopped for longer than the timeout while
            # NOT in a voluntary stop (stop_sign / traffic-light stop), flush
            # the MPC seed and force a fresh behavior replan.  A longer timeout
            # applies for emergency_brake so a genuine obstacle-following stop
            # is not interrupted too early.
            if _standstill_start_wall_time_s is not None and float(current_target_v_mps) > 0.1:
                _standstill_elapsed_s = float(wall_time_s) - float(_standstill_start_wall_time_s)
                _is_voluntary_stop = bool(is_fixed_stop_decision(current_applied_behavior))
                _is_eb = bool(is_emergency_brake_decision(current_applied_behavior))
                _recovery_timeout_s = (
                    None if _is_voluntary_stop
                    else float(
                        behavior_runtime_cfg.get(
                            "standstill_emergency_brake_recovery_timeout_s", 30.0
                        )
                        if _is_eb
                        else behavior_runtime_cfg.get("standstill_recovery_timeout_s", 15.0)
                    )
                )
                if (
                    _recovery_timeout_s is not None
                    and _standstill_elapsed_s >= float(_recovery_timeout_s)
                ):
                    print(
                        f"[RECOVERY] Standstill {_standstill_elapsed_s:.0f}s"
                        f" (behavior={current_applied_behavior}): flushing MPC seed + forcing replan"
                    )
                    current_acceleration_mps2 = 0.0
                    current_steering_rad = 0.0
                    cached_control_sequence = None
                    cached_control_step_idx = 0
                    if hasattr(mpc, "clear_previous_solution_seed"):
                        mpc.clear_previous_solution_seed()
                    force_behavior_replan = True
                    # Reset timer so the recovery fires at most once per timeout
                    # window; it will fire again if the vehicle remains stuck.
                    _standstill_start_wall_time_s = float(wall_time_s)

            # --- Heavy planning: only at MPC replan rate ---
            if mpc.should_replan(sim_time_s) or bool(force_behavior_replan):
                if bool(use_cp_obstacle_pipeline):
                    predicted_snapshots = load_obstacle_snapshots(
                        message_path=cooperative_message_path,
                    )
                    static_object_snapshots = []
                else:
                    static_object_snapshots = list(cached_static_environment_obstacles)
                    predicted_snapshots = _merge_tracker_predictions(
                        object_snapshots=dynamic_object_snapshots,
                        predictions=tracker.predict(
                            step_dt_s=float(mpc.dt_s),
                            horizon_s=float(mpc.horizon_s),
                        ),
                    )
                    predicted_snapshots.extend(list(static_object_snapshots))
                ego_transform = ego_vehicle.get_transform()
                ego_z_m = float(ego_transform.location.z)
                # Determine junction status early — before the lane-context
                # call — so we can freeze lane IDs while traversing an
                # intersection (junction connectors have unique road_ids with
                # no lane siblings, so canonical IDs would collapse to 1).
                ego_waypoint = world_map.get_waypoint(
                    ego_transform.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                ego_in_junction = bool(getattr(ego_waypoint, "is_junction", False))
                if obstacle_height_filter_enabled:
                    predicted_snapshots = _filter_obstacle_snapshots_by_vertical_overlap(
                        ego_z_m=float(ego_z_m),
                        ego_height_m=float(ego_height_m),
                        object_snapshots=predicted_snapshots,
                        vertical_clearance_margin_m=float(obstacle_vertical_clearance_margin_m),
                        default_obstacle_height_m=float(default_obstacle_height_m),
                    )
                predicted_snapshots = _filter_snapshots_by_distance(
                    ego_x_m=float(ego_state[0]),
                    ego_y_m=float(ego_state[1]),
                    object_snapshots=predicted_snapshots,
                    max_distance_m=float(obstacle_tracking_distance_m),
                    max_snapshots=(int(max_planning_obstacles) if int(max_planning_obstacles) > 0 else None),
                )

                # Cheap O(N) lookup on stored initial route (no A* replan)
                current_route_summary = global_planner.get_current_route_info(
                    x_m=float(ego_state[0]),
                    y_m=float(ego_state[1]),
                    query_key="ego",
                )

                current_lane_context = global_planner.get_local_lane_context(
                    x_m=float(ego_state[0]),
                    y_m=float(ego_state[1]),
                    heading_rad=float(ego_state[3]),
                    z_m=float(ego_z_m),
                )
                raw_current_lane_id = int(
                    current_lane_context.get("lane_id", selected_lane_id)
                )
                if ego_in_junction and int(_pre_junction_lane_id) != 0:
                    # Junction connectors have unique road_ids with no lane
                    # siblings → canonical_lane_id_for_waypoint() returns 1
                    # for every connector regardless of the approach lane.
                    # Use the last known pre-junction values so ego_lane_id,
                    # selected_lane_id, and optimal_lane_id stay coherent
                    # throughout the intersection and lane-change completion
                    # can be checked correctly on exit.
                    current_lane_id = int(_pre_junction_lane_id)
                    local_allowed_lane_ids = list(_pre_junction_allowed_lane_ids)
                else:
                    current_lane_id = int(current_lane_context.get("lane_id", selected_lane_id))
                    local_allowed_lane_ids = _allowed_lane_ids_from_context(
                        local_context=current_lane_context,
                        fallback_lane_count=int(lane_count),
                        fallback_lane_id=int(selected_lane_id),
                    )
                    if int(current_lane_id) == 0:
                        current_lane_id = _clamp_lane_id_to_allowed(selected_lane_id, local_allowed_lane_ids)
                    # Persist pre-junction values while on a normal road segment.
                    if int(current_lane_id) != 0:
                        _pre_junction_lane_id = int(current_lane_id)
                    if len(local_allowed_lane_ids) > 0:
                        _pre_junction_allowed_lane_ids = list(local_allowed_lane_ids)
                local_lane_count = len(local_allowed_lane_ids)
                selected_lane_id = _clamp_lane_id_to_allowed(selected_lane_id, local_allowed_lane_ids)
                startup_elapsed_s = (
                    0.0
                    if run_start_sim_time_s is None
                    else max(0.0, float(sim_time_s) - float(run_start_sim_time_s))
                )
                startup_lane_lock_active = (
                    float(startup_elapsed_s)
                    <= max(0.0, float(behavior_runtime_cfg.get("startup_lane_lock_duration_s", 3.0)))
                    and int(current_lane_id) in [int(lane_id) for lane_id in local_allowed_lane_ids]
                )
                if bool(startup_lane_lock_active):
                    # The spawn alignment and the first CARLA lane-context query can
                    # disagree for one or two ticks.  Do not let a stale selected lane
                    # or the global-route lane pull the blue dot/MPC reference sideways
                    # before the behavior FSM has explicitly requested a lane change.
                    selected_lane_id = int(current_lane_id)

                # Use selected_lane_id (target) for lookahead distance so that
                # during a lane change the lookahead is computed for the
                # destination lane, not the lane the ego is currently straddling.
                base_target_lane_id = (
                    int(selected_lane_id)
                    if int(selected_lane_id) != 0
                    else int(current_lane_id)
                )
                base_lookahead_distance_raw_m = compute_lane_lookahead_distance(
                    ego_state=ego_state,
                    lane_center_waypoints=lane_center_waypoints,
                    target_lane_id=int(base_target_lane_id),
                    local_goal_cfg=local_goal_cfg,
                )
                rolling_target_distance_m = float(
                    max(
                        1.0,
                        round(
                            float(
                                base_lookahead_distance_raw_m
                                if base_lookahead_distance_raw_m is not None
                                else local_goal_cfg.get("dynamic_lookahead_min_distance_m", 20.0)
                            )
                        ),
                    )
                )
                ego_snapshot = {
                    "x": float(ego_state[0]),
                    "y": float(ego_state[1]),
                    "v": float(ego_state[2]),
                    "psi": float(ego_state[3]),
                }
                # ---- Assign obstacles to lanes (for safety thread) ---- #
                lane_assignments: Dict[str, int] = {}
                lane_safety_obstacle_snapshots: List[dict] = []
                for obs in predicted_snapshots:
                    obs_id = str(obs.get("vehicle_id", ""))
                    obs_ctx = global_planner.get_local_lane_context(
                        x_m=float(obs.get("x", 0.0)),
                        y_m=float(obs.get("y", 0.0)),
                        heading_rad=float(obs.get("psi", 0.0)),
                        z_m=float(obs.get("z", 0.0)) if obs.get("z", None) is not None else None,
                    )
                    assigned_lane_id = _lane_safety_assignment_for_obstacle(
                        ego_lane_context=current_lane_context,
                        obstacle_lane_context=obs_ctx,
                        ego_lane_id=int(current_lane_id),
                        ego_in_junction=bool(ego_in_junction),
                        available_lane_ids=local_allowed_lane_ids,
                        ego_snapshot=ego_snapshot,
                        obstacle_snapshot=obs,
                    )
                    if int(assigned_lane_id) != 0:
                        lane_assignments[obs_id] = int(assigned_lane_id)
                        lane_safety_obstacle_snapshots.append(dict(obs))
                    else:
                        lane_assignments[obs_id] = 0

                raw_lane_scores = lane_safety_scorer.compute_lane_scores(
                    ego_snapshot=ego_snapshot,
                    obstacle_snapshots=lane_safety_obstacle_snapshots,
                    lane_assignments=lane_assignments,
                    ego_lane_id=int(current_lane_id),
                    available_lane_ids=local_allowed_lane_ids,
                    timestamp_s=sim_time_s,
                )
                lane_safety_scorer.cleanup_stale_obstacles(
                    {
                        str(snapshot.get("vehicle_id", ""))
                        for snapshot in lane_safety_obstacle_snapshots
                    }
                )
                lane_scores = {
                    int(lane_id): _sanitize_lane_score(raw_lane_scores.get(lane_id, 1.0))
                    for lane_id in local_allowed_lane_ids
                }
                nearest_front_obstacles_by_lane = _nearest_front_obstacle_by_lane(
                    ego_snapshot=ego_snapshot,
                    obstacle_snapshots=lane_safety_obstacle_snapshots,
                    lane_assignments=lane_assignments,
                    available_lane_ids=local_allowed_lane_ids,
                )
                front_obstacle_distance_by_lane = _nearest_front_obstacle_distance_by_lane(
                    ego_snapshot=ego_snapshot,
                    obstacle_snapshots=lane_safety_obstacle_snapshots,
                    lane_assignments=lane_assignments,
                    available_lane_ids=local_allowed_lane_ids,
                )
                prediction_frame = build_prediction_frame(
                    ego_snapshot=ego_snapshot,
                    obstacle_snapshots=lane_safety_obstacle_snapshots,
                    lane_assignments=lane_assignments,
                    available_lane_ids=local_allowed_lane_ids,
                    horizon_s=float(
                        behavior_runtime_cfg.get(
                            "lane_change_prediction_horizon_s",
                            min(3.0, float(mpc.horizon_s)),
                        )
                    ),
                    dt_s=float(
                        behavior_runtime_cfg.get(
                            "lane_change_prediction_dt_s",
                            max(0.1, float(mpc.dt_s)),
                        )
                    ),
                    min_front_gap_m=float(
                        behavior_runtime_cfg.get("lane_change_min_future_front_gap_m", 12.0)
                    ),
                    min_rear_gap_m=float(
                        behavior_runtime_cfg.get("lane_change_min_future_rear_gap_m", 8.0)
                    ),
                    min_ttc_s=float(
                        behavior_runtime_cfg.get("lane_change_min_future_ttc_s", 2.5)
                    ),
                )
                lane_prediction_risks = dict(prediction_frame.lane_prediction_risks)
                cached_predicted_obstacle_trajectories = dict(
                    prediction_frame.obstacle_future_trajectories
                )
                risky_lane_summaries = []
                for risk_lane_id, risk_info in sorted(lane_prediction_risks.items()):
                    if not bool(dict(risk_info).get("risk", False)):
                        continue
                    risk_reason = str(dict(risk_info).get("reason", "risk") or "risk")
                    risk_obstacle_id = str(dict(risk_info).get("risky_obstacle_id", "") or "")
                    if risk_obstacle_id:
                        risky_lane_summaries.append(
                            f"L{int(risk_lane_id)}:{risk_reason}:{risk_obstacle_id}"
                        )
                    else:
                        risky_lane_summaries.append(f"L{int(risk_lane_id)}:{risk_reason}")
                cached_prediction_risk_summary = (
                    "; ".join(risky_lane_summaries[:3])
                    if len(risky_lane_summaries) > 0
                    else "clear"
                )

                # ---- Ego lane offset for lane-change completion ------ #
                # ego_waypoint and ego_in_junction were already computed above
                # (before the lane-context call) so they are not repeated here.
                ego_offset_info = compute_ego_lane_offset(world_map, carla, ego_transform)

                # ---- Blue-dot route context and mode ----------------- #
                previous_temporary_destination_state = (
                    list(temporary_destination_state)
                    if temporary_destination_state is not None
                    else None
                )
                _prev_dest_xy = (
                    (float(temporary_destination_state[0]), float(temporary_destination_state[1]))
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 2
                    else None
                )
                _prev_mode = (
                    float(temporary_destination_state[5])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 6
                    else None
                )
                _prev_road_id = (
                    int(temporary_destination_state[6])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 7
                    else None
                )
                _prev_entered_intersection = (
                    float(temporary_destination_state[7]) > 0.5
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 8
                    else False
                )
                planning_temporary_route_summary = current_route_summary
                if _prev_dest_xy is not None:
                    planning_temporary_route_summary = global_planner.get_current_route_info(
                        x_m=float(_prev_dest_xy[0]),
                        y_m=float(_prev_dest_xy[1]),
                        query_key="blue_dot",
                    )
                planning_next_maneuver = normalize_macro_maneuver(
                    getattr(planning_temporary_route_summary, "next_macro_maneuver", "straight")
                )
                _raw_planning_optimal = _clamp_optional_lane_id_to_allowed(
                    getattr(planning_temporary_route_summary, "optimal_lane_id", 0),
                    local_allowed_lane_ids,
                )
                if ego_in_junction and int(_pre_junction_optimal_lane_id) != 0:
                    # Freeze the route-optimal lane at the pre-junction value.
                    # Inside a junction, route waypoints are connector segments
                    # with unreliable lane IDs; using the preparatory approach
                    # lane keeps the behavior planner stable through the turn.
                    planning_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
                        int(_pre_junction_optimal_lane_id), local_allowed_lane_ids
                    )
                else:
                    planning_optimal_lane_id = int(_raw_planning_optimal)
                    if int(planning_optimal_lane_id) != 0:
                        _pre_junction_optimal_lane_id = int(planning_optimal_lane_id)
                raw_temp_mode_value, _temp_mode_road_id, _temp_mode_entered_intersection = compute_temp_destination_mode(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=ego_transform,
                    mode_reference_xy=_prev_dest_xy,
                    prev_mode=_prev_mode,
                    prev_road_id=_prev_road_id,
                    prev_entered_intersection=bool(_prev_entered_intersection),
                    next_macro_maneuver=str(planning_next_maneuver),
                    intersection_threshold_m=float(
                        _behavior_runtime_value(
                            behavior_runtime_cfg,
                            rule_planner_cfg,
                            key="intersection_distance_threshold_m",
                            legacy_key="intersection_threshold_m",
                            default=30.0,
                        )
                    ),
                )
                temp_mode_str = (
                    "INTERSECTION" if float(raw_temp_mode_value) > 0.5 else "NORMAL"
                )
                candidate_frame = evaluate_behavior_candidates(
                    lane_safety_scores=lane_scores,
                    lane_prediction_risks=lane_prediction_risks,
                    ego_lane_id=int(current_lane_id),
                    selected_lane_id=int(selected_lane_id),
                    available_lane_ids=local_allowed_lane_ids,
                    route_optimal_lane_id=int(planning_optimal_lane_id),
                    mode=str(temp_mode_str),
                    safety_weight=float(
                        behavior_runtime_cfg.get("candidate_safety_weight", 10.0)
                    ),
                    prediction_risk_weight=float(
                        behavior_runtime_cfg.get("candidate_prediction_risk_weight", 100.0)
                    ),
                    route_deviation_weight=float(
                        behavior_runtime_cfg.get("candidate_route_deviation_weight", 2.0)
                    ),
                    lane_change_weight=float(
                        behavior_runtime_cfg.get("candidate_lane_change_weight", 1.0)
                    ),
                )
                cached_candidate_summary = candidate_frame.summary()
                candidate_preferred_target_lane_id = (
                    int(candidate_frame.selected.target_lane_id)
                    if bool(candidate_frame.selected.feasible)
                    else None
                )
                cached_candidate_detail_summary = " | ".join(
                    f"{candidate.name}:L{int(candidate.target_lane_id)}={float(candidate.total_cost):.1f}"
                    for candidate in candidate_frame.candidates[:4]
                )
                lane_change_hold_reason = ""
                if (
                    candidate_preferred_target_lane_id is not None
                    and int(candidate_preferred_target_lane_id) != int(selected_lane_id)
                ):
                    if float(sim_time_s) < float(post_stop_lane_change_freeze_until_sim_time_s):
                        lane_change_hold_reason = "post_stop_release_lane_freeze"
                    elif float(sim_time_s) < float(lane_change_commit_cooldown_until_sim_time_s):
                        lane_change_hold_reason = "lane_change_commit_cooldown"
                    elif float(sim_time_s) < float(lane_change_abort_cooldown_until_sim_time_s):
                        lane_change_hold_reason = "lane_change_abort_cooldown"
                if lane_change_hold_reason:
                    candidate_preferred_target_lane_id = int(selected_lane_id)
                    cached_candidate_summary = (
                        f"{cached_candidate_summary} | hold:{lane_change_hold_reason}"
                    )
                    cached_candidate_detail_summary = (
                        f"{cached_candidate_detail_summary} | hold:{lane_change_hold_reason}"
                        if cached_candidate_detail_summary
                        else f"hold:{lane_change_hold_reason}"
                    )
                intersection_front_obstacle = nearest_front_obstacles_by_lane.get(
                    int(current_lane_id),
                    None,
                )
                current_lane_safety = float(
                    lane_scores.get(int(current_lane_id), 1.0)
                )
                intersection_obstacle_response = evaluate_intersection_obstacle_response(
                    mode=str(temp_mode_str),
                    front_obstacle_speed_mps=(
                        None
                        if intersection_front_obstacle is None
                        else float(intersection_front_obstacle.get("v", 0.0))
                    ),
                    original_max_velocity_mps=float(original_max_velocity_mps),
                    moving_obstacle_speed_threshold_mps=float(
                        moving_obstacle_speed_threshold_mps
                    ),
                    route_lane_safety_score=float(current_lane_safety),
                    static_obstacle_replan_lane_safety_threshold=float(
                        static_obstacle_replan_lane_safety_threshold
                    ),
                )
                behavior_target_max_velocity_mps = float(
                    intersection_obstacle_response.get(
                        "speed_cap_mps",
                        float(original_max_velocity_mps),
                    )
                )
                current_target_v_mps = float(behavior_target_max_velocity_mps)

                # ---- IDM car-following: compute desired accel toward lead -- #
                _idm_lead_obstacle = nearest_front_obstacles_by_lane.get(int(current_lane_id))
                if _idm_lead_obstacle is not None:
                    _idm_gap_m = float(_idm_lead_obstacle.get("front_distance_m", 999.0))
                    _idm_v_lead = float(_idm_lead_obstacle.get("v", 0.0))
                    _idm_accel = idm_acceleration(
                        v=float(ego_state[2]),
                        v_lead=_idm_v_lead,
                        gap_m=_idm_gap_m,
                        v_desired=float(current_target_v_mps),
                    )
                    _hud_idm_state = (
                        f"IDM gap={_idm_gap_m:.0f}m "
                        f"vL={_idm_v_lead:.1f}m/s "
                        f"a={_idm_accel:+.2f}m/s²"
                    )
                else:
                    _idm_accel = None
                    _hud_idm_state = "IDM: free"

                if _should_force_stationary_release_replan(
                    ego_speed_mps=float(ego_state[2]),
                    zero_speed_threshold_mps=0.0,
                    current_behavior=str(current_applied_behavior),
                    target_v_ref_mps=float(current_target_v_mps),
                    current_acceleration_mps2=float(current_acceleration_mps2),
                    cached_step_acceleration_mps2=cached_step_acceleration_mps2,
                ):
                    current_acceleration_mps2 = 0.0
                    current_steering_rad = 0.0
                    cached_control_sequence = None
                    cached_control_step_idx = 0
                    if hasattr(mpc, "clear_previous_solution_seed"):
                        mpc.clear_previous_solution_seed()

                # ---- Rule-based behavior planner (synchronous) ------- #
                _prev_planner_lc_state_for_cooldown = str(cached_planner_lc_state).upper()
                planner_output = rule_planner.update(
                    lane_safety_scores=lane_scores,
                    ego_lane_id=int(current_lane_id),
                    selected_lane_id=int(selected_lane_id),
                    ego_lateral_offset_m=float(ego_offset_info.get("lateral_offset_m", 0.0)),
                    ego_heading_error_rad=float(ego_offset_info.get("heading_error_rad", 0.0)),
                    mode=str(temp_mode_str),
                    route_optimal_lane_id=int(planning_optimal_lane_id),
                    next_macro_maneuver=str(planning_next_maneuver),
                    front_obstacle_distance_by_lane=front_obstacle_distance_by_lane,
                    current_time_s=float(sim_time_s),
                    wall_time_s=float(wall_time_s),
                    traffic_signal_state=str(traffic_signal_state),
                    traffic_stop_target=traffic_stop_target,
                    traffic_signal_context=traffic_signal_context,
                    ego_speed_mps=float(ego_state[2]),
                    ego_max_deceleration_mps2=abs(float(mpc.constraints.min_acceleration_mps2)),
                    ego_in_junction=bool(ego_in_junction),
                    ego_position_xy=[float(ego_state[0]), float(ego_state[1])],
                    global_route_points=active_global_route_points,
                    nearest_front_obstacles_by_lane=nearest_front_obstacles_by_lane,
                    lane_prediction_risks=lane_prediction_risks,
                    preferred_target_lane_id=candidate_preferred_target_lane_id,
                )
                cached_planner_decision = str(planner_output.get("decision", "lane_follow"))
                cached_planner_lc_state = str(planner_output.get("lc_state", rule_planner.lc_state))
                _cached_blocking_obstacle_id = str(planner_output.get("blocking_obstacle_id", ""))
                _cached_decision_reason = str(planner_output.get("decision_reason", ""))
                try:
                    cached_planner_target_lane_id = int(planner_output.get("target_lane_id", selected_lane_id))
                except Exception:
                    cached_planner_target_lane_id = int(selected_lane_id)
                try:
                    cached_planner_selected_lane_id = int(planner_output.get("selected_lane_id", selected_lane_id))
                except Exception:
                    cached_planner_selected_lane_id = int(selected_lane_id)
                _prev_applied_behavior = str(current_applied_behavior)
                current_applied_behavior = normalize_behavior_decision(
                    planner_output.get("decision", "lane_follow")
                )
                if (
                    str(current_applied_behavior) == "reroute"
                    and str(_prev_applied_behavior) != "reroute"
                ):
                    _reroute_execution_pending = True
                # Detect stop→resume transition: behavior just changed from
                # "stop" to a non-stop decision while ego is still near-zero
                # speed.  The previous MPC plan was a braking trajectory with
                # v=0 everywhere and possibly a large negative acceleration.
                # Both the v=0 seed and the negative current_acceleration_mps2
                # prevent the QP from planning re-acceleration on this tick:
                #   1. Degenerate linearisation at v=0 (weak cost gradients).
                #   2. Jerk constraint clamps a[0] to [a_brake ± jerk_limit],
                #      forcing continued braking for the first several steps.
                # Reset the control state and flush the MPC seed so that the
                # fresh rollout starts from rest with a clean linearisation.
                if (
                    bool(is_stop_decision(_prev_applied_behavior))
                    and not bool(is_stop_decision(current_applied_behavior))
                    and float(ego_state[2]) <= 1.0
                ):
                    post_stop_lane_change_freeze_until_sim_time_s = max(
                        float(post_stop_lane_change_freeze_until_sim_time_s),
                        float(sim_time_s) + float(post_stop_lane_change_freeze_s),
                    )
                    stop_release_temp_smooth_until_sim_time_s = max(
                        float(stop_release_temp_smooth_until_sim_time_s),
                        float(sim_time_s) + float(stop_release_temp_lookahead_smooth_s),
                    )
                    current_acceleration_mps2 = 0.0
                    current_steering_rad = 0.0
                    cached_control_sequence = None
                    cached_control_step_idx = 0
                    if hasattr(mpc, "clear_previous_solution_seed"):
                        mpc.clear_previous_solution_seed()
                # Detect emergency_brake→resume: same stale-seed problem as
                # the stop→resume case above.  The outer stationary-release
                # replan is suppressed for emergency_brake (to avoid 20 Hz
                # replanning), so flush MPC state here on the transition.
                if (
                    bool(is_emergency_brake_decision(_prev_applied_behavior))
                    and not bool(is_emergency_brake_decision(current_applied_behavior))
                    and float(ego_state[2]) <= 1.0
                ):
                    current_acceleration_mps2 = 0.0
                    current_steering_rad = 0.0
                    cached_control_sequence = None
                    cached_control_step_idx = 0
                    if hasattr(mpc, "clear_previous_solution_seed"):
                        mpc.clear_previous_solution_seed()
                planner_mode_override = str(
                    planner_output.get("mode_override", temp_mode_str) or temp_mode_str
                ).strip().upper()
                if str(planner_mode_override) not in {"NORMAL", "INTERSECTION"}:
                    planner_mode_override = str(temp_mode_str)
                blue_dot_rolling = bool(
                    planner_output.get(
                        "blue_dot_rolling",
                        not bool(is_stop_decision(current_applied_behavior)),
                    )
                )
                traffic_light_debug = dict(
                    planner_output.get("traffic_light_debug", {}) or {}
                )
                if isinstance(traffic_signal_context, Mapping):
                    if bool(traffic_signal_context.get("fallback_stop_target", False)):
                        traffic_light_debug["fallback_stop_target"] = True
                    if traffic_signal_context.get("stop_target_source", ""):
                        traffic_light_debug["stop_target_source"] = str(
                            traffic_signal_context.get("stop_target_source", "")
                        )
                    traffic_light_debug["signal_unknown_duration_s"] = float(
                        traffic_signal_context.get("signal_unknown_duration_s", traffic_signal_unknown_duration_s) or 0.0
                    )
                    traffic_light_debug["last_known_signal_state"] = str(
                        traffic_signal_context.get("last_known_signal_state", traffic_signal_last_known_state)
                    )
                    traffic_light_debug["last_known_signal_source"] = str(
                        traffic_signal_context.get("last_known_signal_source", traffic_signal_last_known_source)
                    )
                    traffic_light_debug["last_known_signal_actor_id"] = str(
                        traffic_signal_context.get("last_known_signal_actor_id", traffic_signal_last_known_actor_id)
                    )
                    traffic_light_debug["last_known_signal_actor_name"] = str(
                        traffic_signal_context.get("last_known_signal_actor_name", traffic_signal_last_known_actor_name)
                    )
                    traffic_light_debug["last_known_signal_distance_m"] = traffic_signal_context.get(
                        "last_known_signal_distance_m",
                        "" if traffic_signal_last_known_distance_m is None else float(traffic_signal_last_known_distance_m),
                    )
                    if bool(traffic_signal_context.get("unknown_signal_release", False)):
                        traffic_light_debug["unknown_signal_release"] = True
                        traffic_light_debug["unknown_signal_release_reason"] = str(
                            traffic_signal_context.get("unknown_signal_release_reason", "")
                        )
                if bool(traffic_signal_unknown_release_active):
                    traffic_light_debug["unknown_signal_release"] = True
                    traffic_light_debug["unknown_signal_release_reason"] = str(
                        traffic_signal_unknown_release_reason
                    )
                _new_planner_lc_state_for_cooldown = str(cached_planner_lc_state).upper()
                if (
                    _prev_planner_lc_state_for_cooldown.startswith("EXECUTE_LANE_CHANGE")
                    and _new_planner_lc_state_for_cooldown in {"LANE_KEEP", "IDLE"}
                ):
                    lane_change_commit_cooldown_until_sim_time_s = max(
                        float(lane_change_commit_cooldown_until_sim_time_s),
                        float(sim_time_s) + float(lane_change_commit_cooldown_s),
                    )
                if (
                    _new_planner_lc_state_for_cooldown == "ABORT_LANE_CHANGE"
                    or bool(traffic_light_debug.get("lane_change_aborted", False))
                ):
                    lane_change_abort_cooldown_until_sim_time_s = max(
                        float(lane_change_abort_cooldown_until_sim_time_s),
                        float(sim_time_s) + float(lane_change_abort_cooldown_s),
                    )
                lane_change_hold_reason = ""
                if float(sim_time_s) < float(post_stop_lane_change_freeze_until_sim_time_s):
                    lane_change_hold_reason = "post_stop_release_lane_freeze"
                elif float(sim_time_s) < float(lane_change_commit_cooldown_until_sim_time_s):
                    lane_change_hold_reason = "lane_change_commit_cooldown"
                elif float(sim_time_s) < float(lane_change_abort_cooldown_until_sim_time_s):
                    lane_change_hold_reason = "lane_change_abort_cooldown"
                if (
                    lane_change_hold_reason
                    and not bool(is_fixed_stop_decision(current_applied_behavior))
                    and (
                        str(normalize_behavior_decision(current_applied_behavior))
                        in {"lane_change_left", "lane_change_right"}
                        or str(cached_planner_lc_state).upper()
                        in {
                            "PREPARE_LANE_CHANGE_LEFT",
                            "PREPARE_LANE_CHANGE_RIGHT",
                            "EXECUTE_LANE_CHANGE_LEFT",
                            "EXECUTE_LANE_CHANGE_RIGHT",
                        }
                    )
                ):
                    if hasattr(rule_planner, "_reset_lane_change_state"):
                        rule_planner._reset_lane_change_state(reason=str(lane_change_hold_reason))
                    current_applied_behavior = "lane_follow"
                    cached_planner_decision = "lane_follow"
                    cached_planner_lc_state = str(getattr(rule_planner, "lc_state", "LANE_KEEP"))
                    cached_planner_target_lane_id = int(current_lane_id)
                    cached_planner_selected_lane_id = int(current_lane_id)
                    planner_output["decision"] = "lane_follow"
                    planner_output["target_lane_id"] = int(current_lane_id)
                    planner_output["selected_lane_id"] = int(current_lane_id)
                    planner_output["lc_state"] = str(cached_planner_lc_state)
                    traffic_light_debug["lane_change_hold_active"] = True
                    traffic_light_debug["lane_change_hold_reason"] = str(lane_change_hold_reason)
                    planner_output["traffic_light_debug"] = dict(traffic_light_debug)
                stale_intersection_override = (
                    str(planner_mode_override) == "INTERSECTION"
                    and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
                    and not bool(is_fixed_stop_decision(current_applied_behavior))
                    and not bool(ego_in_junction)
                    and not bool(traffic_light_debug.get("stop_latched", False))
                    and not bool(traffic_light_debug.get("stop_decision_active", False))
                )
                if bool(stale_intersection_override):
                    planner_mode_override = "NORMAL"
                    traffic_light_debug["intersection_override_cleared"] = True
                    traffic_light_debug["intersection_override_clear_reason"] = "no_active_stop_or_junction"
                    planner_output["mode_override"] = "NORMAL"
                    planner_output["traffic_light_debug"] = dict(traffic_light_debug)
                startup_lane_lock_active = (
                    bool(startup_lane_lock_active)
                    and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
                    and str(cached_planner_lc_state).upper() in {"IDLE", "LANE_KEEP"}
                )
                if bool(startup_lane_lock_active):
                    selected_lane_id = int(current_lane_id)
                if str(current_applied_behavior) == "stop_sign" and float(ego_state[2]) <= 0.15:
                    _stop_key = str(traffic_light_debug.get("control_message_id", "") or "").strip()
                    if not _stop_key:
                        _raw_stop_target = planner_output.get("stop_target", {})
                        if isinstance(_raw_stop_target, Mapping):
                            _stop_key = (
                                f"xy:{float(_raw_stop_target.get('x', ego_state[0])):.1f}:"
                                f"{float(_raw_stop_target.get('y', ego_state[1])):.1f}"
                            )
                        else:
                            _stop_key = "stop_sign"
                    if str(stop_sign_watchdog_key) != str(_stop_key):
                        stop_sign_watchdog_key = str(_stop_key)
                        stop_sign_watchdog_start_sim_time_s = float(sim_time_s)
                    _stop_watchdog_elapsed_s = (
                        0.0
                        if stop_sign_watchdog_start_sim_time_s is None
                        else float(sim_time_s) - float(stop_sign_watchdog_start_sim_time_s)
                    )
                    _stop_watchdog_limit_s = max(
                        1.0,
                        float(getattr(rule_planner, "_stop_sign_wait_duration_s", 2.0)) + float(behavior_runtime_cfg.get("stop_sign_watchdog_buffer_s", 2.0)),
                    )
                    if float(_stop_watchdog_elapsed_s) >= float(_stop_watchdog_limit_s):
                        if str(_stop_key).strip() and hasattr(rule_planner, "_completed_stop_sign_message_ids"):
                            rule_planner._completed_stop_sign_message_ids.add(str(_stop_key).strip())
                        if hasattr(rule_planner, "_clear_stop_state"):
                            rule_planner._clear_stop_state()
                        current_applied_behavior = "lane_follow"
                        cached_motion_decision = "lane_follow"
                        planner_mode_override = "NORMAL"
                        traffic_light_debug["stop_watchdog_released"] = True
                        traffic_light_debug["stop_watchdog_elapsed_s"] = float(_stop_watchdog_elapsed_s)
                        traffic_light_debug["stop_watchdog_key"] = str(_stop_key)
                        stop_sign_watchdog_key = None
                        stop_sign_watchdog_start_sim_time_s = None
                else:
                    stop_sign_watchdog_key = None
                    stop_sign_watchdog_start_sim_time_s = None
                stop_target_state = _stop_target_state_from_behavior_output(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=ego_transform,
                    stop_target=(
                        None
                        if str(current_applied_behavior) == "lane_follow" and bool(traffic_light_debug.get("stop_watchdog_released", False))
                        else planner_output.get("stop_target", None)
                    ),
                    target_lane_id=(
                        int(current_lane_id)
                        if bool(is_fixed_stop_decision(current_applied_behavior))
                        else None
                    ),
                )
                follow_target_state = _follow_target_state_from_behavior_output(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=ego_transform,
                    follow_target=planner_output.get("follow_target", None),
                )
                stop_target_distance_m = None
                if stop_target_state is not None:
                    try:
                        stop_target_distance_m = float(
                            planner_output.get("stop_target", {}).get("distance_m", 0.0)
                        )
                    except Exception:
                        stop_target_distance_m = None
                reroute_lane_override = None
                if str(current_applied_behavior) == "reroute":
                    print("[BEHAVIOR] executing reroute request via global planner.")
                    reroute_messages = (
                        load_lane_closure_messages(message_path=cooperative_message_path)
                        if cooperative_message_path
                        else planner_output.get("reroute_messages", [])
                    )
                    reroute_result = reroute_from_lane_closure_messages(
                        messages=reroute_messages,
                        world_map=world_map,
                        carla=carla,
                        global_planner=global_planner,
                        ego_transform=ego_transform,
                        goal_location=global_route_goal_location,
                        current_route_points=active_global_route_points,
                    )
                    rerouted_route_summary = reroute_result.get("route_summary", None)
                    rerouted_route_points = list(reroute_result.get("route_points", []))
                    if rerouted_route_summary is not None and len(rerouted_route_points) > 0:
                        print(
                            "[BEHAVIOR] reroute completed with a new global route "
                            f"({len(rerouted_route_points)} route points)."
                        )
                        current_acceleration_mps2 = 0.0
                        current_steering_rad = 0.0
                        cached_control_sequence = None
                        cached_control_step_idx = 0
                        planned_trajectory = []
                        if hasattr(mpc, "clear_previous_solution_seed"):
                            mpc.clear_previous_solution_seed()
                        handled_reroute_ids = reroute_result.get("handled_message_ids", [])
                        if hasattr(rule_planner, "acknowledge_reroute_success"):
                            rule_planner.acknowledge_reroute_success(handled_reroute_ids)
                        # A successful reroute changes the road topology the ego
                        # will follow.  The pre-junction frozen lane ID now refers
                        # to the old route's road — reset it so the first non-
                        # junction segment after the reroute re-establishes the
                        # freeze from fresh lane-context data.
                        _pre_junction_lane_id = 0
                        _pre_junction_allowed_lane_ids = list(local_allowed_lane_ids)
                        _pre_junction_optimal_lane_id = 0
                        active_global_route_points = [
                            [float(point[0]), float(point[1])]
                            for point in rerouted_route_points
                            if isinstance(point, Sequence) and len(point) >= 2
                        ]
                        last_world_debug_route_draw_time_s = -float("inf")
                        current_route_summary = global_planner.get_current_route_info(
                            x_m=float(ego_state[0]),
                            y_m=float(ego_state[1]),
                            query_key="ego",
                        )
                        planning_temporary_route_summary = current_route_summary
                        _prev_dest_xy = None
                        _prev_mode = None
                        _prev_road_id = None
                        _prev_entered_intersection = False
                        planning_next_maneuver = normalize_macro_maneuver(
                            getattr(planning_temporary_route_summary, "next_macro_maneuver", "straight")
                        )
                        planning_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
                            getattr(planning_temporary_route_summary, "optimal_lane_id", 0),
                            local_allowed_lane_ids,
                        )
                        if int(planning_optimal_lane_id) != 0:
                            reroute_lane_override = int(planning_optimal_lane_id)
                            reroute_route_follow_latched = True
                        else:
                            reroute_route_follow_latched = False
                    else:
                        print("[BEHAVIOR] reroute requested, but global planner did not return a new route.")
                    # Clear the one-shot flag regardless of success/failure so
                    # normal MPC timing resumes after this execution attempt.
                    _reroute_execution_pending = False
                motion_behavior_decision = (
                    "lane_follow" if str(current_applied_behavior) == "reroute" else str(current_applied_behavior)
                )
                cached_motion_decision = str(motion_behavior_decision)
                planner_selected_lane_id = _selected_lane_id_for_behavior_step(
                    planner_output=planner_output,
                    current_lane_id=int(current_lane_id),
                    allowed_lane_ids=local_allowed_lane_ids,
                    reroute_lane_override=reroute_lane_override,
                )
                if (
                    bool(startup_lane_lock_active)
                    and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
                ):
                    planner_selected_lane_id = int(current_lane_id)

                # ---- Temporary destination via CARLA waypoints ------- #
                temp_destination_decision = (
                    str(current_applied_behavior)
                    if str(current_applied_behavior) == "reroute"
                    else str(motion_behavior_decision)
                )
                cached_temp_destination_decision = str(temp_destination_decision)
                selected_lane_id, should_follow_global_route_lane, reroute_route_follow_latched = (
                    _route_tracking_target_lane_after_reroute(
                        current_behavior=str(temp_destination_decision),
                        planner_selected_lane_id=int(planner_selected_lane_id),
                        route_optimal_lane_id=int(planning_optimal_lane_id),
                        reroute_route_follow_latched=bool(reroute_route_follow_latched),
                    )
                )
                strict_lane_follow_current_lane = (
                    str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
                    and str(cached_planner_lc_state).upper() in {"IDLE", "LANE_KEEP"}
                )
                if bool(strict_lane_follow_current_lane):
                    selected_lane_id = int(current_lane_id)
                    planner_selected_lane_id = int(current_lane_id)
                    should_follow_global_route_lane = False
                    reroute_route_follow_latched = False
                # Clear the reroute latch when the planner is actively
                # executing a lane change toward a lane that is NOT the route
                # optimal lane.  strict_lane_follow_current_lane only fires
                # in LANE_KEEP state, so without this extra check the latch
                # stays True during EXECUTE_LC-toward-detour, causing the blue
                # dot to target the (possibly obstacle-blocked) route lane
                # while the MPC is simultaneously trying to change away from it.
                elif (
                    bool(reroute_route_follow_latched)
                    and str(cached_planner_lc_state).upper()
                    in {"EXECUTE_LANE_CHANGE_LEFT", "EXECUTE_LANE_CHANGE_RIGHT"}
                    and int(planner_selected_lane_id) != int(planning_optimal_lane_id)
                ):
                    reroute_route_follow_latched = False
                if str(normalize_behavior_decision(current_applied_behavior)) in {
                    "lane_change_left",
                    "lane_change_right",
                }:
                    selected_lane_id = int(planner_selected_lane_id)
                    should_follow_global_route_lane = False
                # During PREPARE_LC the vehicle has not yet committed to the
                # lateral move.  Keep the blue dot and MPC reference on the
                # current lane so the MPC does not begin tracking the target
                # lane prematurely; the lateral pull begins only when EXECUTE_LC
                # starts.
                if str(cached_planner_lc_state).upper() in {
                    "PREPARE_LANE_CHANGE_LEFT",
                    "PREPARE_LANE_CHANGE_RIGHT",
                }:
                    temp_destination_decision = "lane_follow"
                    selected_lane_id = int(current_lane_id)
                    should_follow_global_route_lane = False
                temp_mode_reference_xy = _prev_dest_xy
                temp_prev_mode = _prev_mode
                temp_prev_road_id = _prev_road_id
                temp_prev_entered_intersection = bool(_prev_entered_intersection)
                if (
                    bool(traffic_light_debug.get("intersection_approach_lane_lock", False))
                    and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
                    and not bool(is_fixed_stop_decision(current_applied_behavior))
                    and not bool(ego_in_junction)
                ):
                    planner_mode_override = "NORMAL"
                    temp_destination_decision = "lane_follow"
                    selected_lane_id = int(current_lane_id)
                    planner_selected_lane_id = int(current_lane_id)
                    should_follow_global_route_lane = False
                    reroute_route_follow_latched = False
                    temp_mode_reference_xy = None
                    temp_prev_mode = None
                    temp_prev_road_id = None
                    temp_prev_entered_intersection = False
                active_planning_maneuver = intersection_route_follow_maneuver(
                    mode=str(planner_mode_override),
                    next_macro_maneuver=str(planning_next_maneuver),
                    decision=str(motion_behavior_decision),
                    target_lane_id=int(selected_lane_id),
                    available_lane_ids=local_allowed_lane_ids,
                    current_road_option=str(
                        getattr(planning_temporary_route_summary, "current_road_option", "")
                    ),
                )
                effective_rolling_target_distance_m = float(rolling_target_distance_m)
                if (
                    float(sim_time_s) < float(stop_release_temp_smooth_until_sim_time_s)
                    and not bool(is_fixed_stop_decision(current_applied_behavior))
                ):
                    effective_rolling_target_distance_m = min(
                        float(effective_rolling_target_distance_m),
                        max(5.0, float(ego_state[2]) * 1.0 + 5.0),
                    )
                temporary_destination_state = compute_temp_destination(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=ego_transform,
                    target_lane_id=int(selected_lane_id),
                    decision=str(temp_destination_decision),
                    lookahead_m=float(effective_rolling_target_distance_m),
                    target_v_mps=float(current_target_v_mps),
                    global_route_points=active_global_route_points,
                    mode_reference_xy=temp_mode_reference_xy,
                    prev_mode=temp_prev_mode,
                    prev_road_id=temp_prev_road_id,
                    prev_entered_intersection=bool(temp_prev_entered_intersection),
                    next_macro_maneuver=str(active_planning_maneuver),
                    mode_override=str(planner_mode_override),
                    stop_target_state=stop_target_state,
                    follow_target_state=follow_target_state,
                    follow_global_route_lane=bool(should_follow_global_route_lane),
                )
                if (
                    bool(is_fixed_stop_decision(current_applied_behavior))
                    and stop_target_state is not None
                ):
                    temporary_destination_state, active_plan_max_velocity_mps = _apply_exact_stop_target_snap(
                        temporary_destination_state=temporary_destination_state,
                        stop_target_state=stop_target_state,
                        ego_state=ego_state,
                        lock_to_stop_distance_m=float(
                            local_goal_cfg.get("lock_to_final_distance_m", 5.0)
                        ),
                        original_max_velocity_mps=float(behavior_target_max_velocity_mps),
                    )
                    temporary_destination_state, active_plan_max_velocity_mps = _apply_stop_target_speed_cap(
                        temporary_destination_state=temporary_destination_state,
                        ego_state=ego_state,
                        stop_target_distance_m=stop_target_distance_m,
                        original_max_velocity_mps=float(active_plan_max_velocity_mps),
                        braking_deceleration_mps2=abs(float(mpc.constraints.min_acceleration_mps2)),
                        stop_buffer_m=0.0,
                    )
                else:
                    active_plan_max_velocity_mps = float(behavior_target_max_velocity_mps)
                temporary_destination_state, active_plan_max_velocity_mps = _apply_final_destination_snap(
                    temporary_destination_state=temporary_destination_state,
                    final_destination_state=final_destination_state,
                    ego_state=ego_state,
                    lock_to_final_distance_m=float(
                        local_goal_cfg.get("lock_to_final_distance_m", 5.0)
                    ),
                    original_max_velocity_mps=float(active_plan_max_velocity_mps),
                    speed_taper_distance_m=float(
                        local_goal_cfg.get(
                            "final_speed_taper_distance_m",
                            min(
                                float(local_goal_cfg.get("lock_to_final_distance_m", 5.0)),
                                5.0,
                            ),
                        )
                    ),
                )
                final_goal_stop_active = _destination_matches_target_point(
                    destination_state=temporary_destination_state,
                    target_state=final_destination_state,
                )
                final_goal_stop_target_state = (
                    _final_destination_stop_target_state(
                        destination_state=temporary_destination_state,
                        final_destination_state=final_destination_state,
                    )
                    if bool(final_goal_stop_active)
                    else None
                )
                temporary_destination_state, active_plan_max_velocity_mps = _ensure_rolling_destination_speed(
                    temporary_destination_state=temporary_destination_state,
                    active_plan_max_velocity_mps=float(active_plan_max_velocity_mps),
                    current_target_v_mps=float(current_target_v_mps),
                    current_behavior=str(current_applied_behavior),
                    final_goal_stop_active=bool(final_goal_stop_active),
                    stop_speed_threshold_mps=float(getattr(mpc, "final_stop_speed_cap_activation_threshold_mps", 0.05)),
                )
                if bool(final_goal_stop_active):
                    final_stop_distance_m = math.hypot(
                        float(ego_state[0]) - float(final_destination_state[0]),
                        float(ego_state[1]) - float(final_destination_state[1]),
                    )
                    current_applied_behavior = "stop_at_intersection"
                    planner_mode_override = "INTERSECTION"
                    stop_target_state = list(final_goal_stop_target_state or [])
                    stop_target_distance_m = float(final_stop_distance_m)
                    temporary_destination_state = compute_temp_destination(
                        world_map=world_map,
                        carla=carla,
                        ego_transform=ego_transform,
                        target_lane_id=int(selected_lane_id),
                        decision=str(current_applied_behavior),
                        lookahead_m=float(rolling_target_distance_m),
                        target_v_mps=float(current_target_v_mps),
                        global_route_points=active_global_route_points,
                        mode_reference_xy=_prev_dest_xy,
                        prev_mode=_prev_mode,
                        prev_road_id=_prev_road_id,
                        prev_entered_intersection=bool(_prev_entered_intersection),
                        next_macro_maneuver=str(active_planning_maneuver),
                        mode_override=str(planner_mode_override),
                        stop_target_state=stop_target_state,
                        follow_target_state=follow_target_state,
                        follow_global_route_lane=bool(should_follow_global_route_lane),
                    )
                    temporary_destination_state, active_plan_max_velocity_mps = _apply_exact_stop_target_snap(
                        temporary_destination_state=temporary_destination_state,
                        stop_target_state=stop_target_state,
                        ego_state=ego_state,
                        lock_to_stop_distance_m=float(
                            local_goal_cfg.get("lock_to_final_distance_m", 5.0)
                        ),
                        original_max_velocity_mps=float(behavior_target_max_velocity_mps),
                    )
                    temporary_destination_state, active_plan_max_velocity_mps = _apply_stop_target_speed_cap(
                        temporary_destination_state=temporary_destination_state,
                        ego_state=ego_state,
                        stop_target_distance_m=float(final_stop_distance_m),
                        original_max_velocity_mps=float(active_plan_max_velocity_mps),
                        braking_deceleration_mps2=abs(float(mpc.constraints.min_acceleration_mps2)),
                        stop_buffer_m=0.0,
                    )

                temporary_destination_state = _stabilize_temporary_destination(
                    temporary_destination_state=temporary_destination_state,
                    previous_destination_state=previous_temporary_destination_state,
                    current_behavior=str(current_applied_behavior),
                    enabled=bool(behavior_runtime_cfg.get("temp_destination_smoothing_enabled", True)),
                    blend_alpha=float(behavior_runtime_cfg.get("temp_destination_smoothing_alpha", 0.45)),
                    max_smooth_jump_m=float(behavior_runtime_cfg.get("temp_destination_max_smooth_jump_m", 8.0)),
                    max_lane_change_smooth_jump_m=float(
                        behavior_runtime_cfg.get("temp_destination_lane_change_max_smooth_jump_m", 4.0)
                    ),
                    final_goal_stop_active=bool(final_goal_stop_active),
                )

                if (
                    temporary_destination_state is not None
                    and len(temporary_destination_state) >= 7
                    and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
                    and not bool(ego_in_junction)
                ):
                    try:
                        ego_road_id_for_temp = int(current_lane_context.get("road_id", 0))
                    except Exception:
                        ego_road_id_for_temp = 0
                    try:
                        temp_road_id_for_log = int(temporary_destination_state[6])
                    except Exception:
                        temp_road_id_for_log = 0
                    if (
                        int(ego_road_id_for_temp) != 0
                        and int(temp_road_id_for_log) != 0
                        and int(ego_road_id_for_temp) != int(temp_road_id_for_log)
                        and float(sim_time_s) - float(last_temp_destination_mismatch_log_time_s) >= 1.0
                    ):
                        last_temp_destination_mismatch_log_time_s = float(sim_time_s)
                        print(
                            "[TEMP_DES] road mismatch "
                            f"ego_road={int(ego_road_id_for_temp)} "
                            f"temp_road={int(temp_road_id_for_log)} "
                            f"ego_lane={int(current_lane_id)} "
                            f"selected_lane={int(selected_lane_id)} "
                            f"planner_lane={int(planner_selected_lane_id)} "
                            f"route_lane={int(planning_optimal_lane_id)} "
                            f"decision={str(current_applied_behavior)} "
                            f"lc_state={str(cached_planner_lc_state)} "
                            f"follow_route={bool(should_follow_global_route_lane)} "
                            f"temp_xy=({float(temporary_destination_state[0]):.2f},"
                            f"{float(temporary_destination_state[1]):.2f})"
                        )

                temporary_route_summary = planning_temporary_route_summary
                if temporary_destination_state is not None and len(temporary_destination_state) >= 2:
                    temporary_route_summary = global_planner.get_current_route_info(
                        x_m=float(temporary_destination_state[0]),
                        y_m=float(temporary_destination_state[1]),
                        query_key="blue_dot",
                    )
                current_temp_mode_str = (
                    "INTERSECTION"
                    if (
                        temporary_destination_state is not None
                        and len(temporary_destination_state) >= 6
                        and float(temporary_destination_state[5]) > 0.5
                    )
                    else "NORMAL"
                )
                current_temp_reference_xy = (
                    (float(temporary_destination_state[0]), float(temporary_destination_state[1]))
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 2
                    else _prev_dest_xy
                )
                current_temp_mode_value = (
                    float(temporary_destination_state[5])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 6
                    else raw_temp_mode_value
                )
                current_temp_road_id = (
                    int(temporary_destination_state[6])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 7
                    else _temp_mode_road_id
                )
                current_temp_entered_intersection = (
                    float(temporary_destination_state[7]) > 0.5
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 8
                    else bool(_temp_mode_entered_intersection)
                )
                _temp_x_m = (
                    float(temporary_destination_state[0])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 1
                    else ""
                )
                _temp_y_m = (
                    float(temporary_destination_state[1])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 2
                    else ""
                )
                _temp_v_mps = (
                    float(temporary_destination_state[2])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 3
                    else ""
                )
                _temp_heading_rad = (
                    float(temporary_destination_state[3])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 4
                    else ""
                )
                _temp_lane_id = (
                    int(temporary_destination_state[4])
                    if temporary_destination_state is not None
                    and len(temporary_destination_state) >= 5
                    else ""
                )
                _stop_target_x_m = (
                    float(stop_target_state[0])
                    if stop_target_state is not None
                    and len(stop_target_state) >= 1
                    else ""
                )
                _stop_target_y_m = (
                    float(stop_target_state[1])
                    if stop_target_state is not None
                    and len(stop_target_state) >= 2
                    else ""
                )
                _stop_target_actual_distance_m = ""
                if _stop_target_x_m != "" and _stop_target_y_m != "":
                    try:
                        _stop_target_actual_distance_m = float(
                            math.hypot(
                                float(_stop_target_x_m) - float(ego_state[0]),
                                float(_stop_target_y_m) - float(ego_state[1]),
                            )
                        )
                    except Exception:
                        _stop_target_actual_distance_m = ""
                _stop_target_lane_id = (
                    int(float(stop_target_state[4]))
                    if stop_target_state is not None
                    and len(stop_target_state) >= 5
                    else ""
                )
                _stop_target_road_id = (
                    int(float(stop_target_state[6]))
                    if stop_target_state is not None
                    and len(stop_target_state) >= 7
                    else ""
                )
                try:
                    _ego_road_id_for_temp_trace = int(current_lane_context.get("road_id", 0))
                except Exception:
                    _ego_road_id_for_temp_trace = 0
                try:
                    _temp_road_id_for_trace = int(current_temp_road_id)
                except Exception:
                    _temp_road_id_for_trace = 0
                try:
                    _prev_road_id_for_trace = (
                        "" if _prev_road_id is None else int(_prev_road_id)
                    )
                except Exception:
                    _prev_road_id_for_trace = ""
                _temp_destination_jump_m = ""
                if _prev_dest_xy is not None and _temp_x_m != "" and _temp_y_m != "":
                    try:
                        _temp_destination_jump_m = float(
                            math.hypot(
                                float(_temp_x_m) - float(_prev_dest_xy[0]),
                                float(_temp_y_m) - float(_prev_dest_xy[1]),
                            )
                        )
                    except Exception:
                        _temp_destination_jump_m = ""
                try:
                    _ego_section_id_for_temp_trace = int(current_lane_context.get("section_id", 0))
                except Exception:
                    _ego_section_id_for_temp_trace = 0
                temp_destination_history.append({
                    "sim_time_s": float(sim_time_s),
                    "ego_x": float(ego_state[0]),
                    "ego_y": float(ego_state[1]),
                    "ego_speed_mps": float(ego_state[2]),
                    "ego_road_id": int(_ego_road_id_for_temp_trace),
                    "ego_section_id": int(_ego_section_id_for_temp_trace),
                    "ego_lane_id": int(current_lane_id),
                    "ego_in_junction": int(bool(ego_in_junction)),
                    "temp_x": _temp_x_m,
                    "temp_y": _temp_y_m,
                    "temp_v_mps": _temp_v_mps,
                    "temp_heading_rad": _temp_heading_rad,
                    "temp_lane_id": _temp_lane_id,
                    "temp_mode": float(current_temp_mode_value),
                    "temp_road_id": int(_temp_road_id_for_trace),
                    "temp_entered_intersection": int(bool(current_temp_entered_intersection)),
                    "previous_temp_x": "" if _prev_dest_xy is None else float(_prev_dest_xy[0]),
                    "previous_temp_y": "" if _prev_dest_xy is None else float(_prev_dest_xy[1]),
                    "previous_temp_mode": "" if _prev_mode is None else float(_prev_mode),
                    "previous_temp_road_id": _prev_road_id_for_trace,
                    "temp_destination_jump_m": _temp_destination_jump_m,
                    "current_behavior": str(current_applied_behavior),
                    "temp_destination_decision": str(temp_destination_decision),
                    "planner_lc_state": str(cached_planner_lc_state),
                    "planner_mode_override": str(planner_mode_override),
                    "raw_temp_mode": float(raw_temp_mode_value),
                    "temp_mode_road_id": int(_temp_mode_road_id),
                    "temp_mode_entered_intersection": int(bool(_temp_mode_entered_intersection)),
                    "selected_lane_id": int(selected_lane_id),
                    "planner_selected_lane_id": int(planner_selected_lane_id),
                    "planning_optimal_lane_id": int(planning_optimal_lane_id),
                    "should_follow_global_route_lane": int(bool(should_follow_global_route_lane)),
                    "active_planning_maneuver": str(active_planning_maneuver),
                    "planning_next_maneuver": str(planning_next_maneuver),
                    "candidate_summary": str(cached_candidate_summary),
                    "candidate_detail_summary": str(cached_candidate_detail_summary),
                    "prediction_risk_summary": str(cached_prediction_risk_summary),
                    "traffic_signal_state": str(traffic_signal_state),
                    "traffic_debug_signal_state": str(traffic_light_debug.get("signal_state", "")),
                    "traffic_signal_found": int(bool(traffic_light_debug.get("signal_found", False))),
                    "traffic_should_stop_now": int(bool(traffic_light_debug.get("should_stop_now", False))),
                    "traffic_stop_latched": int(bool(traffic_light_debug.get("stop_latched", False))),
                    "traffic_stop_decision_active": int(bool(traffic_light_debug.get("stop_decision_active", False))),
                    "intersection_approach_lane_lock": int(bool(traffic_light_debug.get("intersection_approach_lane_lock", False))),
                    "traffic_signal_distance_m": traffic_light_debug.get("signal_distance_m", ""),
                    "traffic_signal_forward_m": traffic_light_debug.get("signal_forward_m", ""),
                    "traffic_signal_lateral_m": traffic_light_debug.get("signal_lateral_m", ""),
                    "traffic_signal_source": str(traffic_light_debug.get("signal_source", "")),
                    "traffic_signal_actor_id": str(traffic_light_debug.get("signal_actor_id", "")),
                    "traffic_signal_actor_name": str(traffic_light_debug.get("signal_actor_name", "")),
                    "traffic_signal_unknown_duration_s": float(
                        traffic_light_debug.get("signal_unknown_duration_s", traffic_signal_unknown_duration_s) or 0.0
                    ),
                    "traffic_signal_last_known_state": str(traffic_light_debug.get("last_known_signal_state", traffic_signal_last_known_state)),
                    "traffic_signal_last_known_source": str(traffic_light_debug.get("last_known_signal_source", traffic_signal_last_known_source)),
                    "traffic_signal_last_known_actor_id": str(traffic_light_debug.get("last_known_signal_actor_id", traffic_signal_last_known_actor_id)),
                    "traffic_signal_last_known_actor_name": str(traffic_light_debug.get("last_known_signal_actor_name", traffic_signal_last_known_actor_name)),
                    "traffic_signal_last_known_distance_m": traffic_light_debug.get("last_known_signal_distance_m", ""),
                    "traffic_signal_unknown_release": int(bool(traffic_light_debug.get("unknown_signal_release", False))),
                    "traffic_signal_unknown_release_reason": str(traffic_light_debug.get("unknown_signal_release_reason", "")),
                    "fallback_stop_target": int(bool(traffic_light_debug.get("fallback_stop_target", False))),
                    "lane_change_hold_active": int(bool(traffic_light_debug.get("lane_change_hold_active", False))),
                    "lane_change_hold_reason": str(traffic_light_debug.get("lane_change_hold_reason", lane_change_hold_reason)),
                    "intersection_override_cleared": int(bool(traffic_light_debug.get("intersection_override_cleared", False))),
                    "intersection_override_clear_reason": str(traffic_light_debug.get("intersection_override_clear_reason", "")),
                    "stop_target_source": str(traffic_light_debug.get("stop_target_source", "")),
                    "effective_lookahead_m": float(effective_rolling_target_distance_m),
                    "stop_release_temp_smooth_active": int(
                        float(sim_time_s) < float(stop_release_temp_smooth_until_sim_time_s)
                    ),
                    "stop_target_x": _stop_target_x_m,
                    "stop_target_y": _stop_target_y_m,
                    "stop_target_lane_id": _stop_target_lane_id,
                    "stop_target_road_id": _stop_target_road_id,
                    "stop_target_distance_m": "" if stop_target_distance_m is None else float(stop_target_distance_m),
                    "stop_target_actual_distance_m": _stop_target_actual_distance_m,
                    "planner_stop_latched": int(bool(getattr(rule_planner, "stop", False))),
                    "final_goal_stop_active": int(bool(final_goal_stop_active)),
                    "active_plan_max_velocity_mps": float(active_plan_max_velocity_mps),
                    "mpc_constraint_max_velocity_mps": float(mpc.constraints.max_velocity_mps),
                    "current_target_v_mps": float(current_target_v_mps),
                    "reference_jump_m": float(last_reference_jump_m),
                    "reference_stabilized": int(bool(last_reference_stabilized)),
                })
                reference_next_maneuver = normalize_macro_maneuver(
                    getattr(temporary_route_summary, "next_macro_maneuver", planning_next_maneuver)
                )
                if ego_in_junction and int(_pre_junction_optimal_lane_id) != 0:
                    current_route_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
                        int(_pre_junction_optimal_lane_id), local_allowed_lane_ids
                    )
                else:
                    current_route_optimal_lane_id = _clamp_optional_lane_id_to_allowed(
                        getattr(temporary_route_summary, "optimal_lane_id", 0),
                        local_allowed_lane_ids,
                    )
                reference_target_lane_id, should_follow_global_route_lane_for_reference, reroute_route_follow_latched = (
                    _route_tracking_target_lane_after_reroute(
                        current_behavior=str(current_applied_behavior),
                        planner_selected_lane_id=int(planner_selected_lane_id),
                        route_optimal_lane_id=int(current_route_optimal_lane_id),
                        reroute_route_follow_latched=bool(reroute_route_follow_latched),
                    )
                )
                reference_target_lane_id, should_follow_global_route_lane_for_reference = (
                    _reference_lane_from_blue_dot(
                        planner_reference_lane_id=int(reference_target_lane_id),
                        temporary_destination_state=temporary_destination_state,
                        allowed_lane_ids=local_allowed_lane_ids,
                        route_optimal_lane_id=int(current_route_optimal_lane_id),
                        should_follow_global_route_lane=bool(should_follow_global_route_lane_for_reference),
                    )
                )
                if str(normalize_behavior_decision(current_applied_behavior)) in {
                    "lane_change_left",
                    "lane_change_right",
                }:
                    reference_target_lane_id = int(planner_selected_lane_id)
                    should_follow_global_route_lane_for_reference = False
                if (
                    bool(startup_lane_lock_active)
                    and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
                ):
                    reference_target_lane_id = int(current_lane_id)
                    should_follow_global_route_lane_for_reference = False
                if bool(strict_lane_follow_current_lane):
                    reference_target_lane_id = int(current_lane_id)
                    should_follow_global_route_lane_for_reference = False
                cached_reference_target_lane_id = int(reference_target_lane_id)
                display_reference_maneuver = str(reference_next_maneuver)
                active_reference_maneuver = intersection_route_follow_maneuver(
                    mode=str(current_temp_mode_str),
                    next_macro_maneuver=str(reference_next_maneuver),
                    decision=str(current_applied_behavior),
                    target_lane_id=int(reference_target_lane_id),
                    available_lane_ids=local_allowed_lane_ids,
                    current_road_option=str(
                        getattr(temporary_route_summary, "current_road_option", "")
                    ),
                )

                # ---- MPC solve to the blue dot with path shaping ----- #
                mpc.constraints.max_velocity_mps = float(active_plan_max_velocity_mps)

                # IDM: cap MPC max velocity when following too closely
                if _idm_accel is not None and _idm_accel < -0.3:
                    _idm_gap_for_cap_m = max(0.0, float(_idm_gap_m))
                    _idm_stop_decision_active = bool(
                        traffic_light_debug.get(
                            "stop_decision_active",
                            traffic_light_debug.get("stop_latched", False),
                        )
                    )
                    _idm_stop_context = bool(is_fixed_stop_decision(current_applied_behavior)) or (
                        bool(_idm_stop_decision_active)
                        and str(traffic_light_debug.get("signal_state", "unknown")).strip().lower() in {"red", "yellow"}
                    )
                    _idm_buffer_m = (
                        6.0
                        if bool(_idm_stop_context)
                        else float(behavior_runtime_cfg.get("idm_non_stop_buffer_m", 3.0))
                    )
                    _idm_brake_mps2 = max(
                        0.5,
                        float(behavior_runtime_cfg.get("idm_speed_cap_braking_deceleration_mps2", 2.5)),
                    )
                    _idm_v_cap = math.sqrt(
                        max(
                            0.0,
                            2.0
                            * float(_idm_brake_mps2)
                            * max(0.0, float(_idm_gap_for_cap_m) - float(_idm_buffer_m)),
                        )
                    )
                    if not bool(_idm_stop_context):
                        _idm_min_release_speed_mps = float(
                            behavior_runtime_cfg.get("idm_non_stop_min_speed_cap_mps", 2.0)
                        )
                        if float(_idm_gap_for_cap_m) >= float(_idm_buffer_m) + 2.0:
                            _idm_v_cap = max(float(_idm_min_release_speed_mps), float(_idm_v_cap))
                        _idm_v_cap = max(float(_idm_v_cap), min(float(current_target_v_mps), float(_idm_v_lead) + 1.5))
                    if _idm_v_cap < mpc.constraints.max_velocity_mps:
                        mpc.constraints.max_velocity_mps = float(_idm_v_cap)
                        _hud_idm_state += f" →cap={_idm_v_cap:.1f}m/s"

                # Trapezoidal stop profile: tighten speed cap only when a stop is active.
                if (
                    bool(is_fixed_stop_decision(current_applied_behavior))
                    and stop_target_distance_m is not None
                    and float(stop_target_distance_m) < 60.0
                ):
                    _stop_profile = trapezoidal_stop_profile(
                        current_v=float(ego_state[2]),
                        distance_to_stop_m=float(stop_target_distance_m),
                        a_decel=max(1.0, abs(float(mpc.constraints.min_acceleration_mps2))),
                        stop_buffer_m=1.5,
                    )
                    if _stop_profile and float(_stop_profile[0]) < mpc.constraints.max_velocity_mps:
                        mpc.constraints.max_velocity_mps = float(_stop_profile[0])
                cached_mpc_max_velocity_mps = float(mpc.constraints.max_velocity_mps)
                lane_reference_speed_mps = max(
                    1.0,
                    float(ego_state[2]),
                    abs(float(temporary_destination_state[2])) if len(temporary_destination_state) >= 3 else 0.0,
                )
                raw_lane_center_reference = build_reference_samples(
                    world_map=world_map,
                    carla=carla,
                    ego_transform=ego_transform,
                    target_lane_id=int(reference_target_lane_id),
                    decision=str(current_applied_behavior),
                    horizon_steps=int(mpc.horizon_steps),
                    step_distance_m=max(0.5, float(lane_reference_speed_mps) * float(mpc.dt_s)),
                    global_route_points=active_global_route_points,
                    mode_reference_xy=current_temp_reference_xy,
                    prev_mode=current_temp_mode_value,
                    prev_road_id=current_temp_road_id,
                    prev_entered_intersection=bool(current_temp_entered_intersection),
                    next_macro_maneuver=str(active_reference_maneuver),
                    mode_override=str(current_temp_mode_str),
                    stop_target_state=stop_target_state,
                    follow_target_state=follow_target_state,
                    follow_global_route_lane=bool(should_follow_global_route_lane_for_reference),
                    force_stop_reference=False,
                )
                reference_input_samples, last_reference_fallback_reason = _reference_with_route_fallback(
                    ego_state=ego_state,
                    current_reference=raw_lane_center_reference,
                    previous_reference=previous_lane_center_reference,
                    decision=str(current_applied_behavior),
                    global_route_points=active_global_route_points,
                    horizon_steps=int(mpc.horizon_steps),
                    step_distance_m=max(0.5, float(lane_reference_speed_mps) * float(mpc.dt_s)),
                    target_lane_id=int(reference_target_lane_id),
                )
                local_lane_center_reference, last_reference_stabilized, last_reference_jump_m = (
                    _stabilize_lane_reference_samples(
                        reference_input_samples,
                        previous_lane_center_reference,
                        decision=str(current_applied_behavior),
                    )
                )
                if local_lane_center_reference:
                    previous_lane_center_reference = [dict(sample) for sample in local_lane_center_reference]
                    _s0 = dict(local_lane_center_reference[0])
                    lane_ref_history.append({
                        "sim_time_s": float(sim_time_s),
                        "ref_x": float(_s0.get("x_ref_m", 0.0)),
                        "ref_y": float(_s0.get("y_ref_m", 0.0)),
                        "heading_rad": float(_s0.get("heading_rad", 0.0)),
                        "lane_id": int(_s0.get("lane_id", reference_target_lane_id)),
                        "current_lane_id": int(current_lane_id),
                        "selected_lane_id": int(selected_lane_id),
                        "planner_selected_lane_id": int(planner_selected_lane_id),
                        "reference_target_lane_id": int(reference_target_lane_id),
                        "route_optimal_lane_id": int(current_route_optimal_lane_id),
                        "startup_lane_lock_active": int(bool(startup_lane_lock_active)),
                        "strict_lane_follow_current_lane": int(bool(strict_lane_follow_current_lane)),
                        "follow_global_route_lane": int(bool(should_follow_global_route_lane_for_reference)),
                        "current_behavior": str(current_applied_behavior),
                        "planner_lc_state": str(cached_planner_lc_state),
                        "reference_jump_m": float(last_reference_jump_m),
                        "reference_stabilized": int(bool(last_reference_stabilized)),
                        "reference_fallback_reason": str(last_reference_fallback_reason),
                        "road_left_width_m": float(_s0.get("road_left_width_m", 0.0)),
                        "road_right_width_m": float(_s0.get("road_right_width_m", 0.0)),
                        "road_center_offset_m": float(_s0.get("road_center_offset_m", 0.0)),
                    })
                curve_curvature_abs = _reference_curvature_abs(
                    local_lane_center_reference,
                    sample_count=int(
                        behavior_runtime_cfg.get("curve_speed_cap_sample_count", 8)
                    ),
                )
                curve_min_curvature = max(
                    0.0,
                    float(behavior_runtime_cfg.get("curve_speed_cap_min_curvature", 0.015)),
                )
                curve_speed_cap_enabled = (
                    float(ego_state[2]) > 1.0
                )
                if bool(curve_speed_cap_enabled) and float(curve_curvature_abs) > float(curve_min_curvature):
                    curve_lateral_accel_limit = max(
                        0.1,
                        float(behavior_runtime_cfg.get("curve_lateral_accel_limit_mps2", 1.3)),
                    )
                    curve_speed_cap_mps = math.sqrt(
                        float(curve_lateral_accel_limit) / max(1.0e-6, float(curve_curvature_abs))
                    )
                    if float(curve_speed_cap_mps) < float(mpc.constraints.max_velocity_mps):
                        mpc.constraints.max_velocity_mps = max(
                            1.5,
                            float(curve_speed_cap_mps),
                        )
                        cached_mpc_max_velocity_mps = float(mpc.constraints.max_velocity_mps)
                new_planned_trajectory = mpc.plan_trajectory(
                    current_state=ego_state,
                    destination_state=temporary_destination_state,
                    object_snapshots=predicted_snapshots,
                    current_acceleration_mps2=float(current_acceleration_mps2),
                    current_steering_rad=float(current_steering_rad),
                    lane_center_waypoints=[],
                    lane_center_reference_samples=local_lane_center_reference,
                    stop_goal_active=bool(is_fixed_stop_decision(current_applied_behavior)),
                )
                _append_mpc_cost_sample(
                    cost_history=mpc_cost_history,
                    sim_time_s=float(sim_time_s),
                    mpc=mpc,
                )
                metrics_recorder.record_mpc_status(mpc.get_runtime_status())
                metrics_recorder.record_mpc_extras(
                    lateral_offset_m=mpc.get_current_lateral_offset_m(),
                    heading_error_rad=mpc.get_current_heading_error_rad(),
                    cost_terms=mpc.get_last_cost_terms(),
                )
                if new_planned_trajectory:
                    _mpc_replan_count += 1
                    metrics_recorder.record_planned_trajectory(
                        sim_time_s=float(sim_time_s),
                        replan_index=int(_mpc_replan_count),
                        trajectory=list(new_planned_trajectory),
                    )
                current_cost_artifact_write_wall_time_s = float(time.perf_counter())
                if (
                    len(mpc_cost_history) == 1
                    or current_cost_artifact_write_wall_time_s
                    - float(last_cost_artifact_write_wall_time_s)
                    >= 5.0
                ):
                    try:
                        _write_mpc_cost_artifacts(
                            mpc_cost_history,
                            scenario_cfg,
                            prefer_fast_plot=True,
                        )
                        last_cost_artifact_write_wall_time_s = float(
                            current_cost_artifact_write_wall_time_s
                        )
                    except Exception as exc:
                        print(f"[MPC] Failed to refresh rolling cost artifacts: {exc}")
                new_control_sequence = getattr(mpc, "_last_u_solution", None)
                mpc.mark_replanned(sim_time_s)

                if new_planned_trajectory:
                    planned_trajectory = list(new_planned_trajectory)
                    _mpc_fail_count = 0
                else:
                    _mpc_fail_count += 1
                if new_control_sequence is not None and len(new_control_sequence) > 0:
                    cached_control_sequence = np.asarray(new_control_sequence, dtype=float)
                    cached_control_step_idx = 0

                # Cache obstacle contours and HUD lane for rendering
                if display is not None:
                    contour_snapshots = _filter_snapshots_by_distance(
                        ego_x_m=float(ego_state[0]),
                        ego_y_m=float(ego_state[1]),
                        object_snapshots=predicted_snapshots,
                        max_distance_m=float(obstacle_tracking_distance_m),
                        max_snapshots=(int(max_planning_obstacles) if int(max_planning_obstacles) > 0 else None),
                    )
                    cached_obstacle_contours = _build_obstacle_field_contours(
                        mpc=mpc,
                        ego_state=ego_state,
                        object_snapshots=contour_snapshots,
                    )
                else:
                    cached_obstacle_contours = []
                if len(temporary_destination_state) >= 5:
                    cached_hud_temp_lane_prompt = int(temporary_destination_state[4])
            # --- End heavy planning block ---

            applied_control_reason = "mpc"
            applied_throttle = 0.0
            applied_brake = 0.0
            applied_steer = 0.0
            if not planned_trajectory:
                applied_control_reason = "no_planned_trajectory"
                applied_brake = 1.0
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            elif cached_control_sequence is None or len(cached_control_sequence) == 0:
                applied_control_reason = "empty_control_sequence"
                applied_brake = 0.3
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.3, steer=0.0))
            else:
                control_step_idx = min(int(cached_control_step_idx), int(len(cached_control_sequence) - 1))
                current_acceleration_mps2 = float(cached_control_sequence[control_step_idx][0])
                current_steering_rad = float(cached_control_sequence[control_step_idx][1])
                vehicle_control = _control_from_mpc(
                    mpc=mpc,
                    carla=carla,
                    acceleration_mps2=float(current_acceleration_mps2),
                    steering_angle_rad=float(current_steering_rad),
                )
                applied_throttle = float(getattr(vehicle_control, "throttle", 0.0))
                applied_brake = float(getattr(vehicle_control, "brake", 0.0))
                applied_steer = float(getattr(vehicle_control, "steer", 0.0))
                ego_vehicle.apply_control(vehicle_control)
                if cached_control_step_idx < int(len(cached_control_sequence) - 1):
                    cached_control_step_idx += 1
            _temporary_destination_v_mps = float(temporary_destination_state[2]) if len(temporary_destination_state) >= 3 else 0.0
            control_history.append({
                "sim_time_s": float(sim_time_s),
                "ego_speed_mps": float(ego_state[2]),
                "mpc_vmax_mps": float(cached_mpc_max_velocity_mps),
                "active_plan_max_velocity_mps": float(active_plan_max_velocity_mps),
                "temporary_destination_v_mps": float(_temporary_destination_v_mps),
                "terminal_stop_constraint_candidate": int(
                    bool(getattr(mpc.constraints, "enforce_terminal_velocity_constraint", False))
                    and abs(float(_temporary_destination_v_mps)) <= float(getattr(mpc, "final_stop_speed_cap_activation_threshold_mps", 0.05))
                ),
                "acceleration_mps2": float(current_acceleration_mps2),
                "steering_rad": float(current_steering_rad),
                "throttle": float(applied_throttle),
                "brake": float(applied_brake),
                "steer": float(applied_steer),
                "control_reason": str(applied_control_reason),
                "planned_trajectory_len": int(len(planned_trajectory or [])),
                "control_sequence_len": int(len(cached_control_sequence)) if cached_control_sequence is not None else 0,
            })

            if display is not None and topdown_queue is not None and chase_queue is not None:
                render_tick_counter += 1
                topdown_image = None
                chase_image = None
                try:
                    topdown_image = topdown_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    chase_image = chase_queue.get_nowait()
                except queue.Empty:
                    pass
                if topdown_image is None and chase_image is None:
                    continue

                # ---- HUD text (rule-based planner info) -------------- #
                # Safety scores as compact string
                safety_str = " ".join(
                    f"L{int(lid)}={float(lane_scores.get(int(lid), 1.0)):.2f}"
                    for lid in list(local_allowed_lane_ids)
                )
                allowed_lanes_str = ",".join(str(int(lane_id)) for lane_id in list(local_allowed_lane_ids))
                # Mode from temp destination (6th element)
                td_mode_str = "INTERSECTION" if (len(temporary_destination_state) >= 6 and float(temporary_destination_state[5]) > 0.5) else "NORMAL"

                terminal_planned_velocity_mps = float("nan")
                if planned_trajectory and len(planned_trajectory[-1]) >= 3:
                    terminal_planned_velocity_mps = float(planned_trajectory[-1][2])
                hud_signal_distance_value = traffic_light_debug.get("signal_distance_m", None)
                hud_signal_distance_str = (
                    "n/a"
                    if hud_signal_distance_value is None
                    else f"{float(hud_signal_distance_value):.1f}m"
                )
                hud_stop_wait_remaining_s = traffic_light_debug.get("stop_wait_remaining_s", None)
                hud_stop_wait_started = bool(traffic_light_debug.get("stop_wait_started", False))
                if hud_stop_wait_remaining_s is None:
                    hud_stop_wait_str = (
                        "armed"
                        if str(current_applied_behavior) == "stop_sign" and not bool(hud_stop_wait_started)
                        else "n/a"
                    )
                else:
                    hud_stop_wait_str = f"{float(hud_stop_wait_remaining_s):.1f}s"
                hud_stop_decision = bool(
                    traffic_light_debug.get(
                        "stop_decision_active",
                        traffic_light_debug.get("stop_latched", False),
                    )
                )
                _hud_mpc_status = "OK" if _mpc_fail_count == 0 else f"FAIL×{_mpc_fail_count}"
                _hud_stop_dist_str = (
                    f"{float(stop_target_distance_m):.0f}m"
                    if stop_target_distance_m is not None
                    else "n/a"
                )
                full_hud_lines = [
                    "── MOTION ──────────────────────────────────────",
                    f" behavior={cached_motion_decision}  fsm={cached_planner_lc_state}",
                    f" v={float(ego_state[2]):.2f}  v_ref={float(temporary_destination_state[2]):.2f}  mpc_vmax={float(cached_mpc_max_velocity_mps):.2f} m/s",
                    f" mpc={_hud_mpc_status}  traj_v_end={float(terminal_planned_velocity_mps):.2f}  look={int(round(float(rolling_target_distance_m)))}m",
                    "── CAR FOLLOWING (IDM) ─────────────────────────",
                    f" {_hud_idm_state}",
                    "── STOP PROFILE ────────────────────────────────",
                    f" stop_dist={_hud_stop_dist_str}  signal={str(traffic_light_debug.get('signal_state', 'unk'))}  dist={hud_signal_distance_str}",
                    f" stop_active={hud_stop_decision}  wait={hud_stop_wait_str}",
                    "── LANES / RISK ────────────────────────────────",
                    f" ego={int(current_lane_id)}  sel={int(selected_lane_id)}  route={int(current_route_optimal_lane_id)}  [{allowed_lanes_str}]",
                    f" safety: {safety_str}",
                    f" risk: {cached_prediction_risk_summary}",
                    f" candidate: {cached_candidate_summary}",
                    "── ROUTE ───────────────────────────────────────",
                    f" mode={td_mode_str}  maneuver={str(display_reference_maneuver)}",
                    f" mpc_lane={int(cached_reference_target_lane_id)}  blue_lane={int(cached_hud_temp_lane_prompt)}",
                ]
                compact_hud_lines = [
                    f"{cached_motion_decision}  fsm={cached_planner_lc_state}  H=HUD",
                    f"v={float(ego_state[2]):.1f}  ref={float(temporary_destination_state[2]):.1f}  vmax={float(cached_mpc_max_velocity_mps):.1f}  mpc={_hud_mpc_status}",
                    f"stop={hud_stop_decision}  signal={str(traffic_light_debug.get('signal_state', 'unk'))}  wait={hud_stop_wait_str}",
                    f"lane ego={int(current_lane_id)} sel={int(selected_lane_id)} route={int(current_route_optimal_lane_id)}",
                ]
                if str(hud_mode) == "off":
                    hud_lines = []
                elif str(hud_mode) == "full":
                    hud_lines = full_hud_lines
                else:
                    hud_lines = compact_hud_lines
                topdown_overlay = None
                if topdown_camera is not None:
                    topdown_camera_transform = (
                        getattr(topdown_image, "transform", None)
                        if topdown_image is not None
                        else None
                    )
                    if topdown_camera_transform is None:
                        topdown_camera_transform = topdown_camera.get_transform()
                    _overlay_risk_ids = {
                        str(dict(risk_info).get("risky_obstacle_id", ""))
                        for risk_info in lane_prediction_risks.values()
                        if bool(dict(risk_info).get("risk", False))
                        and str(dict(risk_info).get("risky_obstacle_id", ""))
                    }
                    topdown_overlay = {
                        "camera_transform": topdown_camera_transform,
                        "calibration_matrix": topdown_calibration_matrix,
                        "image_width_px": int(image_width_px),
                        "image_height_px": int(image_height_px),
                        "overlay_z_m": float(ego_vehicle.get_location().z),
                        "global_route_points": _route_points_for_visualization(
                            active_global_route_points,
                            enabled=bool(global_route_visualization_enabled),
                        ),
                        "temporary_destination_state": list(temporary_destination_state),
                        "planned_trajectory_states": list(
                            (planned_trajectory or [])[int(cached_control_step_idx):]
                        ),
                        "predicted_obstacle_trajectories": dict(cached_predicted_obstacle_trajectories),
                        "obstacle_field_contours": cached_obstacle_contours,
                        "obstacle_risk_ids": _overlay_risk_ids,
                    }
                _render_camera_pair(
                    display,
                    topdown_image,
                    chase_image,
                    topdown_overlay=topdown_overlay,
                    hud_lines=hud_lines,
                    hud_font=hud_font,
                    hud_panel_width_px=int(hud_panel_width_px),
                )
    finally:
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
        if sumo_bridge is not None:
            try:
                sumo_bridge.close()
            except Exception:
                pass
        try:
            cost_artifacts = _write_mpc_cost_artifacts(mpc_cost_history, scenario_cfg)
            if cost_artifacts:
                print(
                    "[MPC] Saved cost history to "
                    f"{cost_artifacts.get('csv_path', '<unknown>')} and plot to "
                    f"{cost_artifacts.get('plot_path', '<unknown>')}"
                )
        except Exception as exc:
            print(f"[MPC] Failed to save cost artifacts: {exc}")
        try:
            metric_artifacts = write_planning_metrics_artifacts(
                artifact_dir=_scenario_artifact_dir(scenario_cfg),
                recorder=metrics_recorder,
                scenario_name=str(scenario_cfg.get("name", "scenario")),
            )
            print(
                "[Metrics] Saved planning metrics to "
                f"{metric_artifacts.get('json_path', '<unknown>')} and "
                f"{metric_artifacts.get('csv_path', '<unknown>')}"
            )
        except Exception as exc:
            print(f"[Metrics] Failed to save planning metrics: {exc}")
        try:
            if control_history:
                _control_csv = os.path.join(_scenario_artifact_dir(scenario_cfg), "control_timeseries.csv")
                _control_fields = [
                    "sim_time_s", "ego_speed_mps", "mpc_vmax_mps",
                    "active_plan_max_velocity_mps", "temporary_destination_v_mps",
                    "terminal_stop_constraint_candidate",
                    "acceleration_mps2", "steering_rad", "throttle", "brake", "steer",
                    "control_reason", "planned_trajectory_len", "control_sequence_len",
                ]
                with open(_control_csv, "w", encoding="utf-8", newline="") as _fh:
                    _writer = csv.DictWriter(_fh, fieldnames=_control_fields)
                    _writer.writeheader()
                    for _row in control_history:
                        _writer.writerow({k: _row.get(k, "") for k in _control_fields})
                print(f"[Metrics] Saved control trace to {_control_csv}")
        except Exception as exc:
            print(f"[Metrics] Failed to save control trace: {exc}")
        try:
            if temp_destination_history:
                _temp_destination_csv = os.path.join(
                    _scenario_artifact_dir(scenario_cfg),
                    "temporary_destination_timeseries.csv",
                )
                _temp_destination_fields = [
                    "sim_time_s",
                    "ego_x", "ego_y", "ego_speed_mps",
                    "ego_road_id", "ego_section_id", "ego_lane_id", "ego_in_junction",
                    "temp_x", "temp_y", "temp_v_mps", "temp_heading_rad",
                    "temp_lane_id", "temp_mode", "temp_road_id",
                    "temp_entered_intersection",
                    "previous_temp_x", "previous_temp_y",
                    "previous_temp_mode", "previous_temp_road_id",
                    "temp_destination_jump_m",
                    "current_behavior", "temp_destination_decision",
                    "planner_lc_state",
                    "planner_mode_override", "raw_temp_mode",
                    "temp_mode_road_id", "temp_mode_entered_intersection",
                    "selected_lane_id", "planner_selected_lane_id",
                    "planning_optimal_lane_id",
                    "should_follow_global_route_lane",
                    "active_planning_maneuver",
                    "planning_next_maneuver",
                    "candidate_summary", "candidate_detail_summary",
                    "prediction_risk_summary",
                    "traffic_signal_state", "traffic_debug_signal_state",
                    "traffic_signal_found", "traffic_should_stop_now",
                    "traffic_stop_latched", "traffic_stop_decision_active",
                    "intersection_approach_lane_lock",
                    "traffic_signal_distance_m", "traffic_signal_forward_m",
                    "traffic_signal_lateral_m",
                    "traffic_signal_source",
                    "traffic_signal_actor_id", "traffic_signal_actor_name",
                    "traffic_signal_unknown_duration_s",
                    "traffic_signal_last_known_state", "traffic_signal_last_known_source",
                    "traffic_signal_last_known_actor_id", "traffic_signal_last_known_actor_name",
                    "traffic_signal_last_known_distance_m",
                    "traffic_signal_unknown_release", "traffic_signal_unknown_release_reason",
                    "fallback_stop_target",
                    "lane_change_hold_active", "lane_change_hold_reason",
                    "intersection_override_cleared", "intersection_override_clear_reason",
                    "stop_target_source",
                    "effective_lookahead_m", "stop_release_temp_smooth_active",
                    "stop_target_x", "stop_target_y",
                    "stop_target_lane_id", "stop_target_road_id",
                    "stop_target_distance_m", "stop_target_actual_distance_m",
                    "planner_stop_latched",
                    "final_goal_stop_active",
                    "active_plan_max_velocity_mps",
                    "mpc_constraint_max_velocity_mps",
                    "current_target_v_mps",
                    "reference_jump_m", "reference_stabilized",
                ]
                with open(_temp_destination_csv, "w", encoding="utf-8", newline="") as _fh:
                    _writer = csv.DictWriter(_fh, fieldnames=_temp_destination_fields)
                    _writer.writeheader()
                    for _row in temp_destination_history:
                        _writer.writerow({
                            k: _row.get(k, "")
                            for k in _temp_destination_fields
                        })
                print(f"[Metrics] Saved temporary destination trace to {_temp_destination_csv}")
        except Exception as exc:
            print(f"[Metrics] Failed to save temporary destination trace: {exc}")
        try:
            _transition_events = rule_planner.transition_events
            if _transition_events:
                _transition_csv = os.path.join(_scenario_artifact_dir(scenario_cfg), "fsm_transition_log.csv")
                _transition_fields = [
                    "sim_time_s", "old_state", "new_state", "reason", "decision",
                    "target_lane_id", "selected_lane_id", "source_lane_id",
                    "state_elapsed_s", "state_min_hold_s", "target_lane_safety",
                    "lane_change_cancel_reason", "lane_change_abort_reason",
                    "candidate_target_lane_id",
                ]
                with open(_transition_csv, "w", encoding="utf-8", newline="") as _fh:
                    _writer = csv.DictWriter(_fh, fieldnames=_transition_fields)
                    _writer.writeheader()
                    for _row in _transition_events:
                        _writer.writerow({k: _row.get(k, "") for k in _transition_fields})
                print(f"[Metrics] Saved FSM transition log to {_transition_csv}")
        except Exception as exc:
            print(f"[Metrics] Failed to save FSM transition log: {exc}")
        try:
            if lane_ref_history:
                _lane_ref_csv = os.path.join(_scenario_artifact_dir(scenario_cfg), "lane_reference_timeseries.csv")
                _lane_ref_fields = [
                    "sim_time_s", "ref_x", "ref_y", "heading_rad", "lane_id",
                    "current_lane_id", "selected_lane_id", "planner_selected_lane_id",
                    "reference_target_lane_id", "route_optimal_lane_id",
                    "startup_lane_lock_active", "strict_lane_follow_current_lane",
                    "follow_global_route_lane",
                    "current_behavior", "planner_lc_state",
                    "reference_jump_m", "reference_stabilized", "reference_fallback_reason",
                    "road_left_width_m", "road_right_width_m", "road_center_offset_m",
                ]
                with open(_lane_ref_csv, "w", encoding="utf-8", newline="") as _fh:
                    _writer = csv.DictWriter(_fh, fieldnames=_lane_ref_fields)
                    _writer.writeheader()
                    for _row in lane_ref_history:
                        _writer.writerow({k: _row.get(k, "") for k in _lane_ref_fields})
                print(f"[Metrics] Saved lane reference to {_lane_ref_csv}")
        except Exception as exc:
            print(f"[Metrics] Failed to save lane reference: {exc}")
        try:
            _artifact_dir = _scenario_artifact_dir(scenario_cfg)
            _ego_state = last_status_ego_state
            _destination_distance_m = None
            if _ego_state is not None and len(_ego_state) >= 2:
                _destination_distance_m = math.hypot(
                    float(_ego_state[0]) - float(final_destination_state[0]),
                    float(_ego_state[1]) - float(final_destination_state[1]),
                )
            run_status.update({
                "artifacts_written": True,
                "wall_elapsed_s": float(time.perf_counter()) - float(run_start_wall_time_s),
                "sim_time_s": "" if last_status_sim_time_s is None else float(last_status_sim_time_s),
                "sim_elapsed_s": (
                    ""
                    if last_status_sim_time_s is None or run_start_sim_time_s is None
                    else float(last_status_sim_time_s) - float(run_start_sim_time_s)
                ),
                "destination_distance_m": (
                    "" if _destination_distance_m is None else float(_destination_distance_m)
                ),
                "mpc_plan_attempts": int(len(mpc_cost_history)),
                "mpc_consecutive_failures": int(_mpc_fail_count),
                "control_samples": int(len(control_history)),
                "temporary_destination_samples": int(len(temp_destination_history)),
                "lane_reference_samples": int(len(lane_ref_history)),
                "fsm_transition_count": int(len(rule_planner.transition_events)),
                "final_behavior": str(cached_motion_decision),
                "final_fsm_state": str(cached_planner_lc_state),
            })
            if not bool(run_status.get("finished", False)):
                run_status["reason"] = str(run_status.get("reason", "interrupted_or_exception"))
            _status_path = os.path.join(_artifact_dir, "run_status.json")
            with open(_status_path, "w", encoding="utf-8") as _fh:
                json.dump(run_status, _fh, indent=2, sort_keys=True)
                _fh.write("\n")
            print(
                "[SCENARIO STATUS] "
                f"finished={bool(run_status.get('finished', False))} "
                f"reason={run_status.get('reason', 'unknown')} "
                f"status={_status_path}"
            )
        except Exception as exc:
            print(f"[Metrics] Failed to save run status: {exc}")
        world.apply_settings(previous_settings)
        _destroy_actors(actors_to_destroy)
        if pygame is not None:
            pygame.quit()
