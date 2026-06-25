"""
Behavior-planner reroute helpers.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from utility.carla_lane_graph import (
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoints,
    direction_key,
    is_driving_waypoint,
    raw_carla_lane_id_for_waypoint,
)
from utility.cp_messages import (
    CP_MESSAGE_PATH,
    control_messages,
    ensure_cp_message_file_exists,
    lane_closure_messages,
    load_control_messages,
    load_cp_message_payload,
    load_cp_messages,
    load_lane_closure_messages,
    pop_lane_closure_messages,
    remove_cp_messages_by_id,
    reset_cp_message_payload,
    write_cp_message_payload,
    write_cp_messages,
)
from utility.global_planner import RoutePlanSummary


DEFAULT_REROUTE_PENALTY = 1.0e9


def _coerce_position_xy(raw_position: object) -> List[float] | None:
    if not isinstance(raw_position, (list, tuple)) or len(raw_position) < 2:
        return None
    return [float(raw_position[0]), float(raw_position[1])]


def _coerce_lane_id_from_message(message: Mapping[str, object]) -> tuple[int | None, str | None]:
    direct_lane_id = message.get("lane_id", None)
    if direct_lane_id is not None:
        try:
            return int(direct_lane_id), "lane_id"
        except Exception:
            return None, "lane_id"
    raw_lane_ids = message.get("lane_ids", None)
    if isinstance(raw_lane_ids, Sequence) and not isinstance(raw_lane_ids, (str, bytes, bytearray)):
        for raw_lane_id in list(raw_lane_ids):
            try:
                return int(raw_lane_id), "lane_ids"
            except Exception:
                continue
    return None, None


def _road_numeric_id(raw_road_id: object) -> int | None:
    if raw_road_id is None:
        return None
    raw_text = str(raw_road_id).strip()
    if not raw_text:
        return None
    if ":" in raw_text:
        raw_text = raw_text.split(":", 1)[0]
    try:
        return int(raw_text)
    except Exception:
        return None


def _section_numeric_id(raw_section_id: object) -> int | None:
    if raw_section_id is None:
        return None
    raw_text = str(raw_section_id).strip()
    if not raw_text:
        return None
    try:
        return int(raw_text)
    except Exception:
        return None


def _road_segment_key_from_waypoint(waypoint) -> str:
    return f"{int(getattr(waypoint, 'road_id', 0))}:{int(getattr(waypoint, 'section_id', 0))}"


def _route_points_from_summary(route_summary: object) -> List[List[float]]:
    return [
        [float(item[0]), float(item[1])]
        for item in list(getattr(route_summary, "route_waypoints", []) or [])
        if isinstance(item, Sequence) and len(item) >= 2
    ]


def _distance_point_to_segment_m(
    point_xy: Sequence[float],
    segment_start_xy: Sequence[float],
    segment_end_xy: Sequence[float],
) -> float:
    px_m = float(point_xy[0])
    py_m = float(point_xy[1])
    ax_m = float(segment_start_xy[0])
    ay_m = float(segment_start_xy[1])
    bx_m = float(segment_end_xy[0])
    by_m = float(segment_end_xy[1])
    dx_m = float(bx_m) - float(ax_m)
    dy_m = float(by_m) - float(ay_m)
    segment_len_sq = dx_m * dx_m + dy_m * dy_m
    if float(segment_len_sq) <= 1.0e-9:
        return float(math.hypot(float(px_m) - float(ax_m), float(py_m) - float(ay_m)))
    projection = (
        (float(px_m) - float(ax_m)) * float(dx_m)
        + (float(py_m) - float(ay_m)) * float(dy_m)
    ) / float(segment_len_sq)
    projection = max(0.0, min(1.0, float(projection)))
    closest_x_m = float(ax_m) + float(projection) * float(dx_m)
    closest_y_m = float(ay_m) + float(projection) * float(dy_m)
    return float(math.hypot(float(px_m) - float(closest_x_m), float(py_m) - float(closest_y_m)))


def _route_min_distance_to_point_m(
    route_points: Sequence[Sequence[float]],
    point_xy: Sequence[float],
) -> float:
    normalized_route_points = [
        [float(route_point[0]), float(route_point[1])]
        for route_point in list(route_points or [])
        if isinstance(route_point, Sequence) and len(route_point) >= 2
    ]
    if len(normalized_route_points) == 0:
        return float("inf")
    if len(normalized_route_points) == 1:
        return float(
            math.hypot(
                float(point_xy[0]) - float(normalized_route_points[0][0]),
                float(point_xy[1]) - float(normalized_route_points[0][1]),
            )
        )
    return min(
        _distance_point_to_segment_m(
            point_xy=point_xy,
            segment_start_xy=segment_start_xy,
            segment_end_xy=segment_end_xy,
        )
        for segment_start_xy, segment_end_xy in zip(
            normalized_route_points[:-1],
            normalized_route_points[1:],
        )
    )


def _route_overlaps_blocked_positions(
    route_points: Sequence[Sequence[float]],
    blocked_positions_xy: Sequence[Sequence[float]],
    *,
    clearance_m: float,
) -> bool:
    for blocked_position_xy in list(blocked_positions_xy or []):
        if not isinstance(blocked_position_xy, Sequence) or len(blocked_position_xy) < 2:
            continue
        if _route_min_distance_to_point_m(route_points, blocked_position_xy) <= float(clearance_m):
            return True
    return False


def _expand_specific_lane_segments(
    *,
    global_planner,
    road_id: object,
    lane_id: int,
) -> List[Tuple[str, int]]:
    if hasattr(global_planner, "segment_keys_for_road_and_lane"):
        return list(
            global_planner.segment_keys_for_road_and_lane(
                road_id=road_id,
                lane_id=int(lane_id),
            )
        )
    return []


def _expand_same_direction_road_segments(
    *,
    global_planner,
    road_id: object,
    ego_direction: str,
) -> List[Tuple[str, int]]:
    if hasattr(global_planner, "segment_keys_for_road"):
        return list(
            global_planner.segment_keys_for_road(
                road_id=road_id,
                direction=str(ego_direction),
            )
        )
    return []


def _select_reroute_start_waypoint(
    *,
    start_waypoint,
    blocked_segments: Sequence[Tuple[object, int]] | None,
):
    if start_waypoint is None:
        return None
    blocked_segment_keys = {
        (str(segment_key[0]), int(segment_key[1]))
        for segment_key in list(blocked_segments or [])
        if isinstance(segment_key, (list, tuple)) and len(segment_key) >= 2
    }
    if len(blocked_segment_keys) == 0:
        return start_waypoint

    candidate_waypoints = canonical_lane_waypoints(start_waypoint)
    if len(candidate_waypoints) == 0:
        candidate_waypoints = [start_waypoint]
    current_lane_id = int(canonical_lane_id_for_waypoint(start_waypoint))

    def _candidate_sort_key(candidate_waypoint) -> Tuple[int, int]:
        candidate_lane_id = int(canonical_lane_id_for_waypoint(candidate_waypoint))
        is_blocked = (
            str(_road_segment_key_from_waypoint(candidate_waypoint)),
            int(candidate_lane_id),
        ) in blocked_segment_keys
        return (
            1 if bool(is_blocked) else 0,
            abs(int(candidate_lane_id) - int(current_lane_id)),
        )

    for candidate_waypoint in sorted(candidate_waypoints, key=_candidate_sort_key):
        candidate_segment_key = (
            str(_road_segment_key_from_waypoint(candidate_waypoint)),
            int(canonical_lane_id_for_waypoint(candidate_waypoint)),
        )
        if candidate_segment_key not in blocked_segment_keys:
            return candidate_waypoint
    return start_waypoint


def _waypoint_identity(waypoint) -> Tuple[int, int, int, float, float] | None:
    if waypoint is None:
        return None
    location = getattr(getattr(waypoint, "transform", None), "location", None)
    return (
        int(getattr(waypoint, "road_id", 0)),
        int(getattr(waypoint, "section_id", 0)),
        int(getattr(waypoint, "lane_id", 0)),
        float(getattr(location, "x", 0.0)),
        float(getattr(location, "y", 0.0)),
    )


def _select_adjacent_reroute_start_waypoint(
    *,
    start_waypoint,
    blocked_segments: Sequence[Tuple[object, int]] | None,
):
    if start_waypoint is None:
        return None
    blocked_segment_keys = {
        (str(segment_key[0]), int(segment_key[1]))
        for segment_key in list(blocked_segments or [])
        if isinstance(segment_key, (list, tuple)) and len(segment_key) >= 2
    }
    candidate_waypoints = canonical_lane_waypoints(start_waypoint)
    if len(candidate_waypoints) == 0:
        candidate_waypoints = [start_waypoint]
    current_lane_id = int(canonical_lane_id_for_waypoint(start_waypoint))
    adjacent_candidates = []
    for candidate_waypoint in list(candidate_waypoints):
        candidate_lane_id = int(canonical_lane_id_for_waypoint(candidate_waypoint))
        if int(candidate_lane_id) == int(current_lane_id):
            continue
        candidate_segment_key = (
            str(_road_segment_key_from_waypoint(candidate_waypoint)),
            int(candidate_lane_id),
        )
        if candidate_segment_key in blocked_segment_keys:
            continue
        adjacent_candidates.append(candidate_waypoint)
    if len(adjacent_candidates) == 0:
        return None
    adjacent_candidates.sort(
        key=lambda candidate_waypoint: abs(
            int(canonical_lane_id_for_waypoint(candidate_waypoint)) - int(current_lane_id)
        )
    )
    return adjacent_candidates[0]


def _select_reroute_goal_waypoint(
    *,
    goal_waypoint,
    blocked_segments: Sequence[Tuple[object, int]] | None,
):
    if goal_waypoint is None:
        return None
    blocked_segment_keys = {
        (str(segment_key[0]), int(segment_key[1]))
        for segment_key in list(blocked_segments or [])
        if isinstance(segment_key, (list, tuple)) and len(segment_key) >= 2
    }
    if len(blocked_segment_keys) == 0:
        return goal_waypoint

    candidate_waypoints = canonical_lane_waypoints(goal_waypoint)
    if len(candidate_waypoints) == 0:
        candidate_waypoints = [goal_waypoint]
    current_lane_id = int(canonical_lane_id_for_waypoint(goal_waypoint))

    def _candidate_sort_key(candidate_waypoint) -> Tuple[int, int]:
        candidate_lane_id = int(canonical_lane_id_for_waypoint(candidate_waypoint))
        is_blocked = (
            str(_road_segment_key_from_waypoint(candidate_waypoint)),
            int(candidate_lane_id),
        ) in blocked_segment_keys
        return (
            1 if bool(is_blocked) else 0,
            abs(int(candidate_lane_id) - int(current_lane_id)),
        )

    for candidate_waypoint in sorted(candidate_waypoints, key=_candidate_sort_key):
        candidate_segment_key = (
            str(_road_segment_key_from_waypoint(candidate_waypoint)),
            int(canonical_lane_id_for_waypoint(candidate_waypoint)),
        )
        if candidate_segment_key not in blocked_segment_keys:
            return candidate_waypoint
    return goal_waypoint


def _route_cumulative_distances(route_points: Sequence[Sequence[float]]) -> List[float]:
    if len(route_points) == 0:
        return []
    cumulative = [0.0]
    for idx in range(1, len(route_points)):
        prev_point = route_points[idx - 1]
        current_point = route_points[idx]
        cumulative.append(
            float(cumulative[-1])
            + float(
                math.hypot(
                    float(current_point[0]) - float(prev_point[0]),
                    float(current_point[1]) - float(prev_point[1]),
                )
            )
        )
    return cumulative


def _sample_route_points(route_points: Sequence[Sequence[float]], *, sample_count: int = 40) -> List[List[float]]:
    valid_route_points = [
        [float(point[0]), float(point[1])]
        for point in list(route_points or [])
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if len(valid_route_points) <= 1:
        return list(valid_route_points)
    cumulative = _route_cumulative_distances(valid_route_points)
    total_length_m = float(cumulative[-1])
    if total_length_m <= 1.0e-6:
        return [list(valid_route_points[0])]

    sample_points: List[List[float]] = []
    requested_samples = max(2, int(sample_count))
    for sample_idx in range(requested_samples):
        target_arc_m = float(sample_idx) * float(total_length_m) / float(requested_samples - 1)
        upper_index = 1
        while upper_index < len(cumulative) and float(cumulative[upper_index]) < float(target_arc_m):
            upper_index += 1
        if upper_index >= len(cumulative):
            sample_points.append(list(valid_route_points[-1]))
            continue
        lower_index = max(0, int(upper_index) - 1)
        lower_arc_m = float(cumulative[lower_index])
        upper_arc_m = float(cumulative[upper_index])
        if upper_arc_m <= lower_arc_m + 1.0e-9:
            sample_points.append(list(valid_route_points[upper_index]))
            continue
        alpha = float(target_arc_m - lower_arc_m) / float(upper_arc_m - lower_arc_m)
        lower_point = valid_route_points[lower_index]
        upper_point = valid_route_points[upper_index]
        sample_points.append(
            [
                float(lower_point[0]) + float(alpha) * (float(upper_point[0]) - float(lower_point[0])),
                float(lower_point[1]) + float(alpha) * (float(upper_point[1]) - float(lower_point[1])),
            ]
        )
    return sample_points


def _routes_effectively_same(
    reference_route_points: Sequence[Sequence[float]] | None,
    candidate_route_points: Sequence[Sequence[float]] | None,
    *,
    mean_distance_threshold_m: float = 0.35,
    max_distance_threshold_m: float = 1.0,
) -> bool:
    reference_samples = _sample_route_points(reference_route_points or [])
    candidate_samples = _sample_route_points(candidate_route_points or [])
    if len(reference_samples) < 2 or len(candidate_samples) < 2:
        return False
    sample_count = min(len(reference_samples), len(candidate_samples))
    if sample_count < 2:
        return False

    sample_distances_m: List[float] = []
    for idx in range(sample_count):
        ref_point = reference_samples[idx]
        cand_point = candidate_samples[idx]
        sample_distances_m.append(
            float(
                math.hypot(
                    float(cand_point[0]) - float(ref_point[0]),
                    float(cand_point[1]) - float(ref_point[1]),
                )
            )
        )
    if len(sample_distances_m) == 0:
        return False
    return (
        float(sum(sample_distances_m) / len(sample_distances_m)) <= float(mean_distance_threshold_m)
        and float(max(sample_distances_m)) <= float(max_distance_threshold_m)
    )


def _advance_waypoint_forward(waypoint, *, distance_m: float, step_m: float = 2.0):
    current_waypoint = waypoint
    remaining_distance_m = max(0.0, float(distance_m))
    while current_waypoint is not None and remaining_distance_m > 1.0e-3:
        next_fn = getattr(current_waypoint, "next", None)
        if not callable(next_fn):
            break
        try:
            next_candidates = list(next_fn(min(float(step_m), float(remaining_distance_m))) or [])
        except Exception:
            break
        if len(next_candidates) == 0:
            break
        current_lane_id = int(getattr(current_waypoint, "lane_id", 0))
        preferred_candidates = [
            candidate
            for candidate in next_candidates
            if int(getattr(candidate, "lane_id", 0)) == int(current_lane_id)
        ]
        selection_pool = preferred_candidates if len(preferred_candidates) > 0 else next_candidates
        current_waypoint = selection_pool[0]
        remaining_distance_m -= min(float(step_m), float(remaining_distance_m))
    return current_waypoint if current_waypoint is not None else waypoint


def _neighbor_waypoint_along_lane(
    waypoint,
    *,
    step_distance_m: float,
    use_previous: bool,
):
    if waypoint is None:
        return None
    neighbor_fn = getattr(waypoint, "previous" if bool(use_previous) else "next", None)
    if not callable(neighbor_fn):
        return None
    try:
        candidates = list(neighbor_fn(float(step_distance_m)) or [])
    except Exception:
        return None
    if len(candidates) == 0:
        return None

    current_raw_lane_id = int(raw_carla_lane_id_for_waypoint(waypoint))
    preferred_candidates = [
        candidate
        for candidate in list(candidates)
        if _is_valid_astar_waypoint(candidate)
        and int(raw_carla_lane_id_for_waypoint(candidate)) == int(current_raw_lane_id)
    ]
    if len(preferred_candidates) > 0:
        return preferred_candidates[0]

    same_direction_candidates = [
        candidate
        for candidate in list(candidates)
        if _is_valid_astar_waypoint(candidate)
        and _same_direction_waypoints(waypoint, candidate)
    ]
    if len(same_direction_candidates) > 0:
        return same_direction_candidates[0]
    return None


def _expanded_hazard_waypoint_keys(
    *,
    hazard_waypoint,
    forward_count: int = 2,
    backward_count: int = 2,
    step_distance_m: float = 2.0,
) -> List[tuple]:
    expanded_keys: List[tuple] = []
    seen_keys: set[tuple] = set()

    def _append_waypoint_key(candidate_waypoint) -> None:
        candidate_key = _waypoint_astar_key(candidate_waypoint)
        if candidate_key is None:
            return
        normalized_key = tuple(candidate_key)
        if normalized_key in seen_keys:
            return
        seen_keys.add(normalized_key)
        expanded_keys.append(normalized_key)

    _append_waypoint_key(hazard_waypoint)

    current_waypoint = hazard_waypoint
    for _ in range(max(0, int(forward_count))):
        current_waypoint = _neighbor_waypoint_along_lane(
            current_waypoint,
            step_distance_m=float(step_distance_m),
            use_previous=False,
        )
        if current_waypoint is None:
            break
        _append_waypoint_key(current_waypoint)

    current_waypoint = hazard_waypoint
    for _ in range(max(0, int(backward_count))):
        current_waypoint = _neighbor_waypoint_along_lane(
            current_waypoint,
            step_distance_m=float(step_distance_m),
            use_previous=True,
        )
        if current_waypoint is None:
            break
        _append_waypoint_key(current_waypoint)

    return list(expanded_keys)


def _select_bypass_waypoint(
    *,
    resolved_messages: Sequence[Mapping[str, object]],
    blocked_segment_keys: Sequence[Tuple[object, int]],
    world_map,
    carla,
):
    blocked_segment_key_set = {
        (str(segment_key[0]), int(segment_key[1]))
        for segment_key in list(blocked_segment_keys or [])
        if isinstance(segment_key, (list, tuple)) and len(segment_key) >= 2
    }
    for message in list(resolved_messages or []):
        position_xy = _coerce_position_xy(message.get("position", None))
        if position_xy is None:
            continue
        blocked_waypoint = world_map.get_waypoint(
            carla.Location(x=float(position_xy[0]), y=float(position_xy[1]), z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if blocked_waypoint is None:
            continue
        blocked_lane_id = int(canonical_lane_id_for_waypoint(blocked_waypoint))
        lane_candidates = list(canonical_lane_waypoints(blocked_waypoint))
        if len(lane_candidates) == 0:
            lane_candidates = [blocked_waypoint]
        unblocked_candidates = [
            candidate_waypoint
            for candidate_waypoint in lane_candidates
            if (
                str(_road_segment_key_from_waypoint(candidate_waypoint)),
                int(canonical_lane_id_for_waypoint(candidate_waypoint)),
            ) not in blocked_segment_key_set
            and int(canonical_lane_id_for_waypoint(candidate_waypoint)) != int(blocked_lane_id)
        ]
        if len(unblocked_candidates) == 0:
            continue
        unblocked_candidates.sort(
            key=lambda candidate_waypoint: abs(
                int(canonical_lane_id_for_waypoint(candidate_waypoint)) - int(blocked_lane_id)
            )
        )
        return _advance_waypoint_forward(
            unblocked_candidates[0],
            distance_m=15.0,
            step_m=2.0,
        )
    return None


def _expanded_blocked_positions_for_whole_road_messages(
    *,
    messages: Sequence[Mapping[str, object]],
    world_map,
    carla,
) -> List[List[float]]:
    expanded_positions_xy: List[List[float]] = []
    seen_position_keys: set[Tuple[float, float]] = set()

    def _append_position(position_xy: Sequence[float] | None) -> None:
        if not isinstance(position_xy, Sequence) or len(position_xy) < 2:
            return
        position_key = (
            round(float(position_xy[0]), 3),
            round(float(position_xy[1]), 3),
        )
        if position_key in seen_position_keys:
            return
        seen_position_keys.add(position_key)
        expanded_positions_xy.append([float(position_xy[0]), float(position_xy[1])])

    for message in list(messages or []):
        position_xy = _coerce_position_xy(message.get("position", None))
        _append_position(position_xy)
        if not bool(message.get("block_entire_road", False)):
            continue
        if position_xy is None:
            continue
        blocked_waypoint = world_map.get_waypoint(
            carla.Location(x=float(position_xy[0]), y=float(position_xy[1]), z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        blocked_location = getattr(getattr(blocked_waypoint, "transform", None), "location", None)
        if blocked_location is None:
            continue
        if (
            float(
                math.hypot(
                    float(getattr(blocked_location, "x", 0.0)) - float(position_xy[0]),
                    float(getattr(blocked_location, "y", 0.0)) - float(position_xy[1]),
                )
            ) > 6.0
        ):
            continue
        lane_waypoints = list(canonical_lane_waypoints(blocked_waypoint))
        if len(lane_waypoints) == 0:
            lane_waypoints = [blocked_waypoint]
        for lane_waypoint in lane_waypoints:
            lane_location = getattr(getattr(lane_waypoint, "transform", None), "location", None)
            if lane_location is not None:
                _append_position(
                    [
                        float(getattr(lane_location, "x", 0.0)),
                        float(getattr(lane_location, "y", 0.0)),
                    ]
                )
            advanced_waypoint = _advance_waypoint_forward(
                lane_waypoint,
                distance_m=6.0,
                step_m=2.0,
            )
            advanced_location = getattr(getattr(advanced_waypoint, "transform", None), "location", None)
            if advanced_location is not None:
                _append_position(
                    [
                        float(getattr(advanced_location, "x", 0.0)),
                        float(getattr(advanced_location, "y", 0.0)),
                    ]
                )
    return expanded_positions_xy


def _snapped_hazard_waypoint_resolution(
    *,
    raw_position_xy: Sequence[float] | None,
    world_map,
    carla,
    global_planner,
) -> Dict[str, object] | None:
    del global_planner
    if raw_position_xy is None:
        return None

    inferred_waypoint = world_map.get_waypoint(
        carla.Location(x=float(raw_position_xy[0]), y=float(raw_position_xy[1]), z=0.0),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if inferred_waypoint is not None:
        return {
            "waypoint": inferred_waypoint,
            "segment_key": _road_segment_key_from_waypoint(inferred_waypoint),
            "road_id": int(getattr(inferred_waypoint, "road_id", 0)),
            "section_id": int(getattr(inferred_waypoint, "section_id", 0)),
            "lane_id": int(canonical_lane_id_for_waypoint(inferred_waypoint)),
            "carla_lane_id": int(raw_carla_lane_id_for_waypoint(inferred_waypoint)),
            "source": "map_waypoint",
        }
    return None


def _resolved_message_waypoint(
    *,
    resolved_message: Mapping[str, object],
    world_map,
    carla,
):
    direct_waypoint = resolved_message.get("waypoint", None)
    if direct_waypoint is not None:
        return direct_waypoint
    raw_position_xy = _coerce_position_xy(resolved_message.get("position", None))
    if raw_position_xy is None:
        return None
    return world_map.get_waypoint(
        carla.Location(x=float(raw_position_xy[0]), y=float(raw_position_xy[1]), z=0.0),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def _blocked_waypoints_from_resolved_messages(
    *,
    resolved_messages: Sequence[Mapping[str, object]],
    world_map,
    carla,
) -> List[object]:
    blocked_waypoints: List[object] = []
    seen_waypoint_ids: set[Tuple[int, int, int, float, float] | None] = set()
    for resolved_message in list(resolved_messages or []):
        waypoint = _resolved_message_waypoint(
            resolved_message=resolved_message,
            world_map=world_map,
            carla=carla,
        )
        waypoint_id = _waypoint_identity(waypoint)
        if waypoint is None or waypoint_id in seen_waypoint_ids:
            continue
        seen_waypoint_ids.add(waypoint_id)
        blocked_waypoints.append(waypoint)
    return blocked_waypoints


def _safe_int(raw_value: object) -> int | None:
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except Exception:
        return None


def _waypoint_location_xy(waypoint) -> List[float] | None:
    if waypoint is None:
        return None
    location = getattr(getattr(waypoint, "transform", None), "location", None)
    if location is None:
        return None
    return [float(getattr(location, "x", 0.0)), float(getattr(location, "y", 0.0))]


def _append_unique_route_point(route_points: List[List[float]], point_xy: Sequence[float] | None) -> None:
    if not isinstance(point_xy, Sequence) or len(point_xy) < 2:
        return
    normalized_point = [float(point_xy[0]), float(point_xy[1])]
    if len(route_points) == 0:
        route_points.append(normalized_point)
        return
    if (
        math.hypot(
            float(route_points[-1][0]) - float(normalized_point[0]),
            float(route_points[-1][1]) - float(normalized_point[1]),
        )
        > 1.0e-3
    ):
        route_points.append(normalized_point)


def _route_points_from_waypoint_path(
    *,
    origin_location,
    goal_location,
    waypoint_path: Sequence[object],
) -> List[List[float]]:
    route_points: List[List[float]] = []
    if origin_location is not None:
        _append_unique_route_point(
            route_points,
            [float(getattr(origin_location, "x", 0.0)), float(getattr(origin_location, "y", 0.0))],
        )
    for waypoint in list(waypoint_path or []):
        _append_unique_route_point(route_points, _waypoint_location_xy(waypoint))
    if goal_location is not None:
        _append_unique_route_point(
            route_points,
            [float(getattr(goal_location, "x", 0.0)), float(getattr(goal_location, "y", 0.0))],
        )
    return route_points


def _route_length_m(route_points: Sequence[Sequence[float]]) -> float:
    total_distance_m = 0.0
    for start_xy, end_xy in zip(list(route_points or [])[:-1], list(route_points or [])[1:]):
        if not isinstance(start_xy, Sequence) or len(start_xy) < 2:
            continue
        if not isinstance(end_xy, Sequence) or len(end_xy) < 2:
            continue
        total_distance_m += float(
            math.hypot(
                float(end_xy[0]) - float(start_xy[0]),
                float(end_xy[1]) - float(start_xy[1]),
            )
        )
    return float(total_distance_m)


def _waypoint_astar_key(waypoint, *, s_resolution: float = 1.0):
    if waypoint is None:
        return None
    longitudinal_s = getattr(waypoint, "s", None)
    if longitudinal_s is not None:
        try:
            return (
                int(getattr(waypoint, "road_id", 0)),
                int(getattr(waypoint, "section_id", 0)),
                int(raw_carla_lane_id_for_waypoint(waypoint)),
                round(float(longitudinal_s) / float(s_resolution)) * float(s_resolution),
            )
        except Exception:
            pass
    waypoint_xy = _waypoint_location_xy(waypoint) or [0.0, 0.0]
    return (
        int(getattr(waypoint, "road_id", 0)),
        int(getattr(waypoint, "section_id", 0)),
        int(raw_carla_lane_id_for_waypoint(waypoint)),
        round(float(waypoint_xy[0]), 1),
        round(float(waypoint_xy[1]), 1),
    )


def _waypoint_distance_m(waypoint_a, waypoint_b) -> float:
    waypoint_a_xy = _waypoint_location_xy(waypoint_a)
    waypoint_b_xy = _waypoint_location_xy(waypoint_b)
    if waypoint_a_xy is None or waypoint_b_xy is None:
        return float("inf")
    return float(
        math.hypot(
            float(waypoint_b_xy[0]) - float(waypoint_a_xy[0]),
            float(waypoint_b_xy[1]) - float(waypoint_a_xy[1]),
        )
    )


def _same_direction_waypoints(waypoint_a, waypoint_b) -> bool:
    raw_lane_a = int(raw_carla_lane_id_for_waypoint(waypoint_a))
    raw_lane_b = int(raw_carla_lane_id_for_waypoint(waypoint_b))
    return int(raw_lane_a) != 0 and int(raw_lane_b) != 0 and int(raw_lane_a) * int(raw_lane_b) > 0


def _is_valid_astar_waypoint(waypoint) -> bool:
    return bool(
        waypoint is not None
        and is_driving_waypoint(waypoint)
        and int(raw_carla_lane_id_for_waypoint(waypoint)) != 0
    )


def _astar_neighbors(
    *,
    waypoint,
    step_distance_m: float,
) -> List[Tuple[object, float]]:
    neighbors: List[Tuple[object, float]] = []
    seen_neighbor_keys = set()

    try:
        next_candidates = list(waypoint.next(float(step_distance_m)) or [])
    except Exception:
        next_candidates = []
    for next_waypoint in next_candidates:
        neighbor_key = _waypoint_astar_key(next_waypoint)
        if not _is_valid_astar_waypoint(next_waypoint) or neighbor_key in seen_neighbor_keys:
            continue
        seen_neighbor_keys.add(neighbor_key)
        neighbors.append((next_waypoint, float(step_distance_m)))

    for adjacent_waypoint in (
        getattr(waypoint, "get_left_lane", lambda: None)(),
        getattr(waypoint, "get_right_lane", lambda: None)(),
    ):
        neighbor_key = _waypoint_astar_key(adjacent_waypoint)
        if (
            not _is_valid_astar_waypoint(adjacent_waypoint)
            or not _same_direction_waypoints(waypoint, adjacent_waypoint)
            or neighbor_key in seen_neighbor_keys
        ):
            continue
        seen_neighbor_keys.add(neighbor_key)
        neighbors.append((adjacent_waypoint, float(step_distance_m) * 2.5))
    return neighbors


def _resolved_message_block_profile(resolved_message: Mapping[str, object]) -> Dict[str, object]:
    return {
        "id": str(resolved_message.get("id", "")).strip(),
        "blocked_waypoint_key": tuple(resolved_message.get("blocked_waypoint_key", ()) or ()),
        "blocked_waypoint_keys": [
            tuple(blocked_key)
            for blocked_key in list(resolved_message.get("blocked_waypoint_keys", []) or [])
            if isinstance(blocked_key, Sequence)
        ],
        "road_id": _road_numeric_id(resolved_message.get("road_id", None)),
        "section_id": _section_numeric_id(resolved_message.get("section_id", None)),
        "lane_id": _safe_int(resolved_message.get("lane_id", None)),
        "carla_lane_id": _safe_int(resolved_message.get("carla_lane_id", None)),
        "position": _coerce_position_xy(resolved_message.get("position", None)),
        "block_entire_road": bool(resolved_message.get("block_entire_road", False)),
        "position_only": bool(resolved_message.get("position_only", False)),
        "block_radius_m": 0.0,
    }


def _block_profile_matches_waypoint(
    *,
    block_profile: Mapping[str, object],
    waypoint,
) -> bool:
    if waypoint is None:
        return False
    raw_road_id = _road_numeric_id(block_profile.get("road_id", None))
    raw_section_id = _section_numeric_id(block_profile.get("section_id", None))
    raw_lane_id = _safe_int(block_profile.get("carla_lane_id", None))
    raw_waypoint_lane_id = int(raw_carla_lane_id_for_waypoint(waypoint))
    waypoint_road_id = int(getattr(waypoint, "road_id", 0))
    waypoint_section_id = int(getattr(waypoint, "section_id", 0))

    if raw_road_id is not None and int(waypoint_road_id) != int(raw_road_id):
        return False
    if raw_section_id is not None and int(waypoint_section_id) != int(raw_section_id):
        return False
    if bool(block_profile.get("block_entire_road", False)):
        if raw_lane_id is None or int(raw_lane_id) == 0:
            return True
        return int(raw_waypoint_lane_id) * int(raw_lane_id) > 0
    if raw_lane_id is None:
        return False
    return int(raw_waypoint_lane_id) == int(raw_lane_id)


def _waypoint_is_blocked(
    *,
    waypoint,
    block_profiles: Sequence[Mapping[str, object]],
) -> bool:
    waypoint_key = _waypoint_astar_key(waypoint)
    waypoint_xy = _waypoint_location_xy(waypoint)
    for block_profile in list(block_profiles or []):
        if not isinstance(block_profile, Mapping):
            continue
        blocked_waypoint_keys = {
            tuple(blocked_key)
            for blocked_key in list(block_profile.get("blocked_waypoint_keys", []) or [])
            if isinstance(blocked_key, Sequence)
        }
        if len(blocked_waypoint_keys) > 0:
            if waypoint_key in blocked_waypoint_keys:
                return True
            continue
        blocked_waypoint_key = tuple(block_profile.get("blocked_waypoint_key", ()) or ())
        if len(blocked_waypoint_key) > 0:
            if waypoint_key == blocked_waypoint_key:
                return True
            continue
        if not bool(block_profile.get("position_only", False)) and _block_profile_matches_waypoint(
            block_profile=block_profile,
            waypoint=waypoint,
        ):
            return True
        position_xy = _coerce_position_xy(block_profile.get("position", None))
        if waypoint_xy is None or position_xy is None:
            continue
        if float(
            math.hypot(
                float(waypoint_xy[0]) - float(position_xy[0]),
                float(waypoint_xy[1]) - float(position_xy[1]),
            )
        ) > float(block_profile.get("block_radius_m", 4.0)):
            continue
        if bool(block_profile.get("position_only", False)):
            return True
        if _block_profile_matches_waypoint(block_profile=block_profile, waypoint=waypoint):
            return True
    return False


def _reconstruct_waypoint_path(
    *,
    came_from: Mapping[object, object],
    current_key,
    waypoint_lookup: Mapping[object, object],
) -> List[object]:
    waypoint_path: List[object] = []
    while current_key in waypoint_lookup:
        waypoint_path.append(waypoint_lookup[current_key])
        if current_key not in came_from:
            break
        current_key = came_from[current_key]
    waypoint_path.reverse()
    return waypoint_path


def _astar_route_avoiding_lane_closures(
    *,
    start_waypoint,
    goal_waypoint,
    block_profiles: Sequence[Mapping[str, object]],
    step_distance_m: float = 2.0,
    max_iterations: int = 40000,
) -> List[object]:
    if not _is_valid_astar_waypoint(start_waypoint) or not _is_valid_astar_waypoint(goal_waypoint):
        return []

    start_key = _waypoint_astar_key(start_waypoint)
    goal_distance_threshold_m = max(2.0, float(step_distance_m) * 1.5)
    open_heap: List[Tuple[float, object, object]] = []
    heapq.heappush(open_heap, (0.0, start_key, start_waypoint))
    came_from: Dict[object, object] = {}
    waypoint_lookup: Dict[object, object] = {start_key: start_waypoint}
    g_score: Dict[object, float] = {start_key: 0.0}
    visited = set()
    iteration_count = 0

    while len(open_heap) > 0 and int(iteration_count) < int(max_iterations):
        iteration_count += 1
        _, current_key, current_waypoint = heapq.heappop(open_heap)
        if current_key in visited:
            continue
        visited.add(current_key)

        if _waypoint_distance_m(current_waypoint, goal_waypoint) <= float(goal_distance_threshold_m):
            return _reconstruct_waypoint_path(
                came_from=came_from,
                current_key=current_key,
                waypoint_lookup=waypoint_lookup,
            )

        for next_waypoint, move_cost_m in _astar_neighbors(
            waypoint=current_waypoint,
            step_distance_m=float(step_distance_m),
        ):
            next_key = _waypoint_astar_key(next_waypoint)
            if next_key in visited:
                continue
            if _waypoint_is_blocked(waypoint=next_waypoint, block_profiles=block_profiles):
                continue

            tentative_g_score = float(g_score[current_key]) + float(move_cost_m)
            if tentative_g_score >= float(g_score.get(next_key, float("inf"))):
                continue

            came_from[next_key] = current_key
            waypoint_lookup[next_key] = next_waypoint
            g_score[next_key] = float(tentative_g_score)
            priority = float(tentative_g_score) + float(_waypoint_distance_m(next_waypoint, goal_waypoint))
            heapq.heappush(open_heap, (priority, next_key, next_waypoint))

    return []


def _adjacent_unblocked_waypoints(
    *,
    waypoint,
    block_profiles: Sequence[Mapping[str, object]],
) -> List[object]:
    adjacent_candidates: List[object] = []
    for adjacent_waypoint in (
        getattr(waypoint, "get_left_lane", lambda: None)(),
        getattr(waypoint, "get_right_lane", lambda: None)(),
    ):
        if (
            not _is_valid_astar_waypoint(adjacent_waypoint)
            or not _same_direction_waypoints(waypoint, adjacent_waypoint)
            or _waypoint_is_blocked(waypoint=adjacent_waypoint, block_profiles=block_profiles)
        ):
            continue
        adjacent_candidates.append(adjacent_waypoint)
    return adjacent_candidates


def _select_unblocked_goal_waypoint(
    *,
    goal_waypoint,
    block_profiles: Sequence[Mapping[str, object]],
):
    if goal_waypoint is None:
        return None
    goal_lane_candidates = list(canonical_lane_waypoints(goal_waypoint))
    if len(goal_lane_candidates) == 0:
        goal_lane_candidates = [goal_waypoint]
    current_lane_id = int(canonical_lane_id_for_waypoint(goal_waypoint))
    goal_lane_candidates.sort(
        key=lambda candidate_waypoint: abs(
            int(canonical_lane_id_for_waypoint(candidate_waypoint)) - int(current_lane_id)
        )
    )
    for candidate_waypoint in goal_lane_candidates:
        if not _waypoint_is_blocked(waypoint=candidate_waypoint, block_profiles=block_profiles):
            return candidate_waypoint
    return goal_waypoint


def _build_reroute_summary(
    *,
    route_points: Sequence[Sequence[float]],
    start_waypoint,
    goal_waypoint,
) -> RoutePlanSummary:
    return RoutePlanSummary(
        route_found=True,
        start_road_id="unknown_road" if start_waypoint is None else _road_segment_key_from_waypoint(start_waypoint),
        start_lane_id=0 if start_waypoint is None else int(canonical_lane_id_for_waypoint(start_waypoint)),
        goal_road_id="unknown_road" if goal_waypoint is None else _road_segment_key_from_waypoint(goal_waypoint),
        goal_lane_id=0 if goal_waypoint is None else int(canonical_lane_id_for_waypoint(goal_waypoint)),
        optimal_lane_id=0 if goal_waypoint is None else int(canonical_lane_id_for_waypoint(goal_waypoint)),
        distance_to_destination_m=_route_length_m(route_points),
        next_macro_maneuver="Continue Straight",
        route_waypoints=[
            [float(point_xy[0]), float(point_xy[1])]
            for point_xy in list(route_points or [])
            if isinstance(point_xy, Sequence) and len(point_xy) >= 2
        ],
    )


def _store_reroute_summary(
    *,
    route_summary: RoutePlanSummary | None,
    route_points: Sequence[Sequence[float]],
    world_map,
    carla,
    global_planner,
) -> None:
    if (
        route_summary is None
        or not bool(getattr(route_summary, "route_found", False))
        or not hasattr(global_planner, "replace_stored_route")
    ):
        return

    per_waypoint_options: List[str] = []
    per_waypoint_lane_ids: List[int] = []
    if hasattr(global_planner, "_internal_metadata_from_route_waypoints"):
        try:
            metadata = global_planner._internal_metadata_from_route_waypoints(route_points)
        except Exception:
            metadata = None
        if (
            isinstance(metadata, tuple)
            and len(metadata) == 2
            and isinstance(metadata[0], Sequence)
            and isinstance(metadata[1], Sequence)
        ):
            per_waypoint_options = [str(option) for option in list(metadata[0])]
            per_waypoint_lane_ids = [int(lane_id) for lane_id in list(metadata[1])]

    if len(per_waypoint_options) != len(route_points) or len(per_waypoint_lane_ids) != len(route_points):
        per_waypoint_options = []
        per_waypoint_lane_ids = []
        for point_xy in list(route_points or []):
            if not isinstance(point_xy, Sequence) or len(point_xy) < 2:
                per_waypoint_options.append("LANEFOLLOW")
                per_waypoint_lane_ids.append(0)
                continue
            try:
                waypoint = world_map.get_waypoint(
                    carla.Location(x=float(point_xy[0]), y=float(point_xy[1]), z=0.0),
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
            except Exception:
                waypoint = None
            per_waypoint_options.append("LANEFOLLOW")
            per_waypoint_lane_ids.append(
                0 if waypoint is None else int(canonical_lane_id_for_waypoint(waypoint))
            )

    global_planner.replace_stored_route(
        summary=route_summary,
        per_waypoint_options=list(per_waypoint_options),
        per_waypoint_lane_ids=[int(lane_id) for lane_id in list(per_waypoint_lane_ids)],
    )


def resolve_lane_closure_segments(
    *,
    messages: Sequence[Mapping[str, object]],
    world_map,
    carla,
    global_planner,
    ego_transform,
) -> Dict[str, object]:
    ego_waypoint = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    blocked_waypoint_keys: List[tuple] = []
    seen_blocked_waypoint_keys: set[tuple] = set()
    resolved_messages: List[dict] = []
    block_profiles: List[dict] = []

    for message in lane_closure_messages(messages):
        raw_position_xy = _coerce_position_xy(message.get("position", None))
        snapped_resolution = _snapped_hazard_waypoint_resolution(
            raw_position_xy=raw_position_xy,
            world_map=world_map,
            carla=carla,
            global_planner=global_planner,
        )
        snapped_waypoint = None if snapped_resolution is None else snapped_resolution.get("waypoint", None)
        snapped_road_id = None if snapped_resolution is None else _safe_int(snapped_resolution.get("road_id", None))
        snapped_section_id = None if snapped_resolution is None else _safe_int(snapped_resolution.get("section_id", None))
        snapped_lane_id = None if snapped_resolution is None else _safe_int(snapped_resolution.get("lane_id", None))
        snapped_carla_lane_id = None if snapped_resolution is None else _safe_int(
            snapped_resolution.get("carla_lane_id", None)
        )

        message_road_id = _road_numeric_id(message.get("road_id", None))
        message_section_id = _section_numeric_id(message.get("section_id", None))
        message_lane_id, _ = _coerce_lane_id_from_message(message)
        message_carla_lane_id = _safe_int(message.get("carla_lane_id", None))
        explicit_lane_matches_raw_carla = (
            snapped_carla_lane_id is not None
            and message_lane_id is not None
            and int(message_lane_id) == int(snapped_carla_lane_id)
        )
        explicit_lane_matches_canonical = (
            snapped_lane_id is not None
            and message_lane_id is not None
            and int(message_lane_id) == int(snapped_lane_id)
        )
        road_ids_match_carla = (
            snapped_road_id is None
            or message_road_id is None
            or int(message_road_id) == int(snapped_road_id)
        )
        section_ids_match_carla = (
            snapped_section_id is None
            or message_section_id is None
            or int(message_section_id) == int(snapped_section_id)
        )
        if message_carla_lane_id is not None and snapped_carla_lane_id is not None:
            lane_ids_match_carla = int(message_carla_lane_id) == int(snapped_carla_lane_id)
        elif message_lane_id is not None and snapped_carla_lane_id is not None:
            lane_ids_match_carla = bool(explicit_lane_matches_raw_carla)
        else:
            lane_ids_match_carla = False

        resolved_road_id = snapped_road_id if snapped_road_id is not None else message_road_id
        resolved_section_id = snapped_section_id if snapped_section_id is not None else message_section_id
        resolved_lane_id = snapped_lane_id if snapped_lane_id is not None else _safe_int(message_lane_id)
        resolved_carla_lane_id = (
            snapped_carla_lane_id
            if snapped_carla_lane_id is not None
            else (
                message_carla_lane_id
                if message_carla_lane_id is not None
                else (
                    _safe_int(message_lane_id)
                    if message_lane_id is not None and not bool(explicit_lane_matches_canonical)
                    else None
                )
            )
        )
        resolved_position_xy = raw_position_xy
        if resolved_position_xy is None and snapped_waypoint is not None:
            resolved_position_xy = _waypoint_location_xy(snapped_waypoint)
        blocked_waypoint_key = _waypoint_astar_key(snapped_waypoint)
        if blocked_waypoint_key is None:
            continue
        expanded_blocked_waypoint_keys = _expanded_hazard_waypoint_keys(
            hazard_waypoint=snapped_waypoint,
            forward_count=2,
            backward_count=2,
            step_distance_m=2.0,
        )
        if len(expanded_blocked_waypoint_keys) == 0:
            continue
        normalized_blocked_waypoint_key = tuple(blocked_waypoint_key)
        for expanded_blocked_waypoint_key in list(expanded_blocked_waypoint_keys):
            if expanded_blocked_waypoint_key in seen_blocked_waypoint_keys:
                continue
            blocked_waypoint_keys.append(tuple(expanded_blocked_waypoint_key))
            seen_blocked_waypoint_keys.add(tuple(expanded_blocked_waypoint_key))

        resolved_message = {
            "id": str(message.get("id", "")).strip(),
            "type": "lane_closure",
            "position": None
            if resolved_position_xy is None
            else [float(resolved_position_xy[0]), float(resolved_position_xy[1])],
            "waypoint": snapped_waypoint,
            "road_id": resolved_road_id,
            "section_id": resolved_section_id,
            "lane_id": resolved_lane_id,
            "carla_lane_id": resolved_carla_lane_id,
            "blocked_waypoint_key": list(blocked_waypoint_key),
            "blocked_waypoint_keys": [list(blocked_key) for blocked_key in list(expanded_blocked_waypoint_keys)],
            "block_entire_road": bool(message.get("block_entire_road", False)),
            "road_ids_match_carla": bool(road_ids_match_carla and section_ids_match_carla),
            "lane_ids_match_carla": bool(lane_ids_match_carla),
            "lane_id_matches_canonical": bool(explicit_lane_matches_canonical),
            "normalized_from_position": bool(
                snapped_waypoint is not None
                and (
                    not bool(road_ids_match_carla and section_ids_match_carla)
                    or (
                        message_carla_lane_id is not None
                        and not bool(lane_ids_match_carla)
                    )
                    or (
                        message_carla_lane_id is None
                        and message_lane_id is not None
                        and not bool(explicit_lane_matches_raw_carla)
                    )
                    or message_road_id is None
                    or (message_carla_lane_id is None and message_lane_id is None)
                )
            ),
            "position_only": bool(
                resolved_road_id is None and resolved_carla_lane_id is None and resolved_position_xy is not None
            ),
        }
        resolved_messages.append(resolved_message)
        block_profiles.append(_resolved_message_block_profile(resolved_message))

    return {
        "ego_road_id": None if ego_waypoint is None else _road_segment_key_from_waypoint(ego_waypoint),
        "ego_lane_id": None if ego_waypoint is None else int(canonical_lane_id_for_waypoint(ego_waypoint)),
        "blocked_waypoint_keys": [list(blocked_key) for blocked_key in list(blocked_waypoint_keys)],
        "resolved_messages": list(resolved_messages),
        "block_profiles": list(block_profiles),
    }


def reroute_from_lane_closure_messages(
    *,
    messages: Sequence[Mapping[str, object]],
    world_map,
    carla,
    global_planner,
    ego_transform,
    goal_location,
    penalty_value: float = DEFAULT_REROUTE_PENALTY,
    current_route_points: Sequence[Sequence[float]] | None = None,
) -> Dict[str, object]:
    del penalty_value
    resolution = resolve_lane_closure_segments(
        messages=messages,
        world_map=world_map,
        carla=carla,
        global_planner=global_planner,
        ego_transform=ego_transform,
    )
    resolved_messages = [
        dict(resolved_message)
        for resolved_message in list(resolution.get("resolved_messages", []))
        if isinstance(resolved_message, Mapping)
    ]
    block_profiles = [
        dict(block_profile)
        for block_profile in list(resolution.get("block_profiles", []))
        if isinstance(block_profile, Mapping)
    ]
    if len(resolved_messages) == 0 or len(block_profiles) == 0:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_waypoint_keys": list(resolution.get("blocked_waypoint_keys", [])),
            "resolved_messages": resolved_messages,
            "ego_road_id": resolution.get("ego_road_id", None),
            "ego_lane_id": resolution.get("ego_lane_id", None),
        }
    ego_start_waypoint = world_map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    goal_waypoint = world_map.get_waypoint(
        goal_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    goal_waypoint = _select_unblocked_goal_waypoint(
        goal_waypoint=goal_waypoint,
        block_profiles=block_profiles,
    )
    goal_location_for_route = getattr(getattr(goal_waypoint, "transform", None), "location", None)
    if goal_location_for_route is None:
        goal_location_for_route = goal_location
    route_waypoint_path: List[object] = []
    attempt_start_waypoints: List[object] = []
    if ego_start_waypoint is not None:
        attempt_start_waypoints.append(ego_start_waypoint)
        for adjacent_waypoint in _adjacent_unblocked_waypoints(
            waypoint=ego_start_waypoint,
            block_profiles=block_profiles,
        ):
            if _waypoint_identity(adjacent_waypoint) == _waypoint_identity(ego_start_waypoint):
                continue
            attempt_start_waypoints.append(adjacent_waypoint)

    for attempt_index, attempt_start_waypoint in enumerate(attempt_start_waypoints):
        route_waypoint_path = _astar_route_avoiding_lane_closures(
            start_waypoint=attempt_start_waypoint,
            goal_waypoint=goal_waypoint,
            block_profiles=block_profiles,
        )
        if len(route_waypoint_path) > 0:
            if int(attempt_index) > 0:
                print("[BEHAVIOR] reroute retry: attempting adjacent ego lane as reroute start.")
            break

    route_points = (
        _route_points_from_waypoint_path(
            origin_location=getattr(ego_transform, "location", None),
            goal_location=goal_location_for_route,
            waypoint_path=route_waypoint_path,
        )
        if len(route_waypoint_path) > 0
        else []
    )
    route_summary = None
    if len(route_waypoint_path) > 0 and len(route_points) > 1:
        route_summary = _build_reroute_summary(
            route_points=route_points,
            start_waypoint=route_waypoint_path[0],
            goal_waypoint=goal_waypoint,
        )

    if (
        route_summary is not None
        and current_route_points is not None
        and _routes_effectively_same(current_route_points, route_points)
    ):
        print("[BEHAVIOR] reroute warning: generated route is effectively unchanged from the current route.")

    if not bool(getattr(route_summary, "route_found", False)):
        print(
            "[BEHAVIOR] reroute failed: "
            f"{str(getattr(route_summary, 'debug_reason', '')).strip() or 'no A* route found around the blocked hazard waypoint'}"
        )
    else:
        _store_reroute_summary(
            route_summary=route_summary,
            route_points=route_points,
            world_map=world_map,
            carla=carla,
            global_planner=global_planner,
        )

    return {
        "route_summary": route_summary if bool(getattr(route_summary, "route_found", False)) else None,
        "route_points": route_points,
        "handled_message_ids": [
            str(message.get("id", "")).strip()
            for message in list(resolved_messages)
            if str(message.get("id", "")).strip()
        ],
        "blocked_waypoint_keys": list(resolution.get("blocked_waypoint_keys", [])),
        "resolved_messages": list(resolved_messages),
        "ego_road_id": resolution.get("ego_road_id", None),
        "ego_lane_id": resolution.get("ego_lane_id", None),
    }
