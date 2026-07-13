"""Temporary destination and lane references from the custom Global_Planner."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

from .planner import is_fixed_stop_decision, normalize_behavior_decision
from utility.custom_planner import (
    behavior_lane_id,
    behavior_lane_waypoints,
    waypoint_for_behavior_lane,
    waypoint_position,
    world_point,
)


DEFAULT_STEP_M = 2.0
INTERSECTION_THRESHOLD_M = 30.0


def _route_cum_dists(route_points: Sequence[Sequence[float]]) -> List[float]:
    if not route_points:
        return []
    cumulative = [0.0]
    for first, second in zip(route_points[:-1], route_points[1:]):
        cumulative.append(
            cumulative[-1]
            + math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1]),
            )
        )
    return cumulative


def project_ego_to_route(
    ego_x: float,
    ego_y: float,
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
) -> float:
    if len(route_points) < 2:
        return 0.0
    best_distance = float("inf")
    best_progress = 0.0
    for index, (first, second) in enumerate(zip(route_points[:-1], route_points[1:])):
        ax, ay = float(first[0]), float(first[1])
        bx, by = float(second[0]), float(second[1])
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        alpha = 0.0 if length_sq <= 1.0e-12 else max(
            0.0,
            min(1.0, ((float(ego_x) - ax) * dx + (float(ego_y) - ay) * dy) / length_sq),
        )
        px, py = ax + alpha * dx, ay + alpha * dy
        distance = math.hypot(float(ego_x) - px, float(ego_y) - py)
        if distance < best_distance:
            best_distance = distance
            best_progress = float(cum_dists[index]) + alpha * math.sqrt(max(length_sq, 0.0))
    return best_progress


def _sample_route(
    route_points: Sequence[Sequence[float]],
    cumulative: Sequence[float],
    progress_m: float,
) -> Tuple[float, float, float]:
    if not route_points:
        return 0.0, 0.0, 0.0
    if len(route_points) == 1:
        return float(route_points[0][0]), float(route_points[0][1]), 0.0
    target = max(0.0, min(float(progress_m), float(cumulative[-1])))
    for index in range(len(route_points) - 1):
        start_s = float(cumulative[index])
        end_s = float(cumulative[index + 1])
        if target > end_s and index < len(route_points) - 2:
            continue
        first, second = route_points[index], route_points[index + 1]
        length = max(1.0e-9, end_s - start_s)
        alpha = max(0.0, min(1.0, (target - start_s) / length))
        x_m = float(first[0]) + alpha * (float(second[0]) - float(first[0]))
        y_m = float(first[1]) + alpha * (float(second[1]) - float(first[1]))
        heading = math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))
        return x_m, y_m, heading
    last = route_points[-1]
    previous = route_points[-2]
    return (
        float(last[0]),
        float(last[1]),
        math.atan2(float(last[1]) - float(previous[1]), float(last[0]) - float(previous[0])),
    )


def get_lookahead_route_point(
    ego_x: float,
    ego_y: float,
    route_points: Sequence[Sequence[float]],
    lookahead_m: float,
) -> Tuple[float, float, float]:
    cumulative = _route_cum_dists(route_points)
    progress = project_ego_to_route(ego_x, ego_y, route_points, cumulative)
    return _sample_route(route_points, cumulative, progress + max(0.0, float(lookahead_m)))


def _active_route_points(global_planner, fallback: Sequence[Sequence[float]] | None) -> List[List[float]]:
    route = getattr(global_planner, "active_route", None)
    if route is not None:
        points = []
        for waypoint in list(route.sampled_waypoints or []):
            position = waypoint_position(waypoint)
            if position is not None:
                points.append([position["x"], position["y"]])
        if points:
            return points
    return [
        [float(point[0]), float(point[1])]
        for point in list(fallback or [])
        if isinstance(point, Sequence) and len(point) >= 2
    ]


def _lane_reference_waypoint(global_planner, x_m: float, y_m: float, z_m: float, target_lane_id: int):
    waypoint = global_planner.get_waypoint({"x": x_m, "y": y_m, "z": z_m})
    if waypoint is None:
        return None
    if bool(waypoint.is_intersection):
        return waypoint
    return waypoint_for_behavior_lane(waypoint, int(target_lane_id))


def _road_fields(waypoint, lane_id: int) -> Dict[str, float]:
    width = max(0.1, float(getattr(waypoint, "lane_width_m", 3.5) or 3.5))
    lanes = behavior_lane_waypoints(waypoint)
    count = max(1, len(lanes))
    index = max(1, min(int(lane_id), count))
    right_width = (float(index) - 0.5) * width
    left_width = (float(count - index) + 0.5) * width
    return {
        "lane_width_m": width,
        "road_center_offset_m": 0.5 * (right_width - left_width),
        "road_left_width_m": left_width,
        "road_right_width_m": right_width,
    }


def compute_temp_destination_mode(
    *,
    global_planner,
    ego_position,
    mode_reference_xy: Sequence[float] | None,
    prev_mode: float | None,
    prev_road_id: int | None,
    prev_entered_intersection: bool,
    next_macro_maneuver: str,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
) -> Tuple[float, int, bool]:
    del next_macro_maneuver
    query = world_point(mode_reference_xy if mode_reference_xy is not None else ego_position)
    if query is None:
        return float(prev_mode or 0.0), int(prev_road_id or -1), bool(prev_entered_intersection)
    waypoint = global_planner.get_waypoint(query)
    if waypoint is None:
        return float(prev_mode or 0.0), int(prev_road_id or -1), bool(prev_entered_intersection)
    in_intersection = bool(waypoint.is_intersection)
    if not in_intersection:
        route_points = _active_route_points(global_planner, None)
        if len(route_points) >= 2:
            cumulative = _route_cum_dists(route_points)
            progress = project_ego_to_route(query["x"], query["y"], route_points, cumulative)
            probe_x, probe_y, _ = _sample_route(
                route_points,
                cumulative,
                progress + max(0.0, float(intersection_threshold_m)),
            )
            probe = global_planner.get_waypoint({"x": probe_x, "y": probe_y, "z": query["z"]})
            in_intersection = bool(probe is not None and probe.is_intersection)
    road_id = int(waypoint.road_id or prev_road_id or -1)
    entered = bool(prev_entered_intersection or waypoint.is_intersection)
    return (1.0 if in_intersection else 0.0), road_id, entered


def compute_temp_destination(
    *,
    global_planner,
    ego_position,
    target_lane_id: int,
    decision: str,
    lookahead_m: float,
    target_v_mps: float,
    global_route_points: Sequence[Sequence[float]],
    mode_reference_xy: Sequence[float] | None = None,
    prev_mode: float | None = None,
    prev_road_id: int | None = None,
    prev_entered_intersection: bool = False,
    next_macro_maneuver: str = "straight",
    mode_override: str | None = None,
    stop_target_state: Sequence[float] | None = None,
    follow_target_state: Sequence[float] | None = None,
    follow_global_route_lane: bool = False,
) -> List[float] | None:
    ego = world_point(ego_position)
    if ego is None:
        return None
    normalized_decision = normalize_behavior_decision(decision)
    if is_fixed_stop_decision(normalized_decision) and stop_target_state is not None:
        return [float(value) for value in list(stop_target_state)]
    if normalized_decision == "car_follow" and follow_target_state is not None:
        return [float(value) for value in list(follow_target_state)]

    route_points = _active_route_points(global_planner, global_route_points)
    if len(route_points) < 2:
        return None
    x_m, y_m, route_heading = get_lookahead_route_point(
        ego["x"], ego["y"], route_points, float(lookahead_m)
    )
    waypoint = _lane_reference_waypoint(global_planner, x_m, y_m, ego["z"], target_lane_id)
    if waypoint is not None and not bool(follow_global_route_lane):
        position = waypoint_position(waypoint)
        if position is not None:
            x_m, y_m = position["x"], position["y"]
        route_heading = float(waypoint.heading or route_heading)
        target_lane_id = behavior_lane_id(waypoint)

    mode_value, road_id, entered = compute_temp_destination_mode(
        global_planner=global_planner,
        ego_position=ego,
        mode_reference_xy=mode_reference_xy,
        prev_mode=prev_mode,
        prev_road_id=prev_road_id,
        prev_entered_intersection=prev_entered_intersection,
        next_macro_maneuver=next_macro_maneuver,
    )
    if str(mode_override or "").strip().upper() == "INTERSECTION":
        mode_value = 1.0
    return [
        float(x_m),
        float(y_m),
        max(0.0, float(target_v_mps)),
        float(route_heading),
        float(target_lane_id),
        float(mode_value),
        float(road_id),
        1.0 if entered else 0.0,
    ]


def build_reference_samples(
    *,
    global_planner,
    ego_position,
    target_lane_id: int,
    decision: str,
    horizon_steps: int,
    step_distance_m: float,
    global_route_points: Sequence[Sequence[float]],
    mode_reference_xy: Sequence[float] | None = None,
    prev_mode: float | None = None,
    prev_road_id: int | None = None,
    prev_entered_intersection: bool = False,
    next_macro_maneuver: str = "straight",
    mode_override: str | None = None,
    stop_target_state: Sequence[float] | None = None,
    follow_target_state: Sequence[float] | None = None,
    follow_global_route_lane: bool = False,
    force_stop_reference: bool = False,
) -> List[Dict[str, float]]:
    del mode_reference_xy, prev_mode, prev_road_id, prev_entered_intersection, next_macro_maneuver, mode_override
    ego = world_point(ego_position)
    if ego is None or int(horizon_steps) <= 0:
        return []
    route_points = _active_route_points(global_planner, global_route_points)
    if len(route_points) < 2:
        return []
    cumulative = _route_cum_dists(route_points)
    start_progress = project_ego_to_route(ego["x"], ego["y"], route_points, cumulative)
    normalized_decision = normalize_behavior_decision(decision)
    fixed_target = None
    if (force_stop_reference or is_fixed_stop_decision(normalized_decision)) and stop_target_state is not None:
        fixed_target = [float(stop_target_state[0]), float(stop_target_state[1])]
    elif normalized_decision == "car_follow" and follow_target_state is not None:
        fixed_target = [float(follow_target_state[0]), float(follow_target_state[1])]

    samples: List[Dict[str, float]] = []
    for stage in range(int(horizon_steps) + 1):
        progress = start_progress + float(stage) * max(0.1, float(step_distance_m))
        x_m, y_m, heading = _sample_route(route_points, cumulative, progress)
        if fixed_target is not None:
            target_progress = project_ego_to_route(fixed_target[0], fixed_target[1], route_points, cumulative)
            x_m, y_m, heading = _sample_route(route_points, cumulative, min(progress, target_progress))
        waypoint = _lane_reference_waypoint(global_planner, x_m, y_m, ego["z"], target_lane_id)
        lane_id = int(target_lane_id)
        if waypoint is not None:
            if not bool(follow_global_route_lane):
                position = waypoint_position(waypoint)
                if position is not None:
                    x_m, y_m = position["x"], position["y"]
            heading = float(waypoint.heading or heading)
            lane_id = behavior_lane_id(waypoint)
            road_fields = _road_fields(waypoint, lane_id)
        else:
            road_fields = {
                "lane_width_m": 3.5,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 1.75,
                "road_right_width_m": 1.75,
            }
        samples.append(
            {
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "heading_rad": float(heading),
                "lane_id": int(lane_id),
                **road_fields,
            }
        )
    return samples


def compute_ego_lane_offset(
    *,
    global_planner,
    ego_position,
    ego_heading_rad: float,
) -> Dict[str, float]:
    ego = world_point(ego_position)
    if ego is None:
        return {"offset_m": 0.0, "lane_id": 0, "lane_width_m": 0.0}
    waypoint = global_planner.get_waypoint(ego)
    position = waypoint_position(waypoint)
    if waypoint is None or position is None:
        return {"offset_m": 0.0, "lane_id": 0, "lane_width_m": 0.0}
    lane_heading = float(waypoint.heading or ego_heading_rad)
    dx = ego["x"] - position["x"]
    dy = ego["y"] - position["y"]
    offset = -math.sin(lane_heading) * dx + math.cos(lane_heading) * dy
    return {
        "offset_m": float(offset),
        "lane_id": int(behavior_lane_id(waypoint)),
        "lane_width_m": float(waypoint.lane_width_m or 0.0),
    }


__all__ = [
    "build_reference_samples",
    "compute_ego_lane_offset",
    "compute_temp_destination",
    "compute_temp_destination_mode",
    "get_lookahead_route_point",
    "project_ego_to_route",
]
