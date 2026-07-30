"""Lane-closure rerouting through the custom global planner."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Mapping, Sequence

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
from utility.global_planner import RoutePlanSummary, canonical_lane_id_for_waypoint
def _message_position(message: Mapping[str, object]) -> Dict[str, float] | None:
    raw = message.get("position", None)
    try:
        if isinstance(raw, Mapping) and "x" in raw and "y" in raw:
            return {
                "x": float(raw["x"]),
                "y": float(raw["y"]),
                "z": float(raw.get("z", 0.0)),
            }
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
            return {
                "x": float(raw[0]),
                "y": float(raw[1]),
                "z": float(raw[2]) if len(raw) >= 3 else 0.0,
            }
    except (TypeError, ValueError):
        return None
    return None


def _resolve_blocked_ad_lane_id(
    message: Mapping[str, object],
    global_planner,
) -> int | None:
    position = _message_position(message)
    if position is not None:
        if hasattr(global_planner, "block_lane_at_position"):
            try:
                return global_planner.block_lane_at_position(position)
            except Exception:
                pass
        waypoint = global_planner.get_waypoint(position)
        if waypoint is None:
            return None
        ad_lane_id = getattr(waypoint, "ad_lane_id", None)
        if ad_lane_id is not None:
            return int(ad_lane_id)
        return int(getattr(waypoint, "lane_id", 0) or 0)

    raw_ad_lane_id = message.get("ad_lane_id", None)
    if raw_ad_lane_id is None:
        return None
    try:
        ad_lane_id = int(raw_ad_lane_id)
        core = getattr(global_planner, "core", None)
        if core is not None:
            core.get_lane_centerline(ad_lane_id)
        return ad_lane_id
    except Exception:
        return None


def _reroute_ad_map(
    *,
    messages: Sequence[Mapping[str, object]],
    global_planner,
    ego_position: Mapping[str, object] | Sequence[object],
    goal_position: Mapping[str, object] | Sequence[object],
    current_route_points: Sequence[Sequence[float]],
) -> Dict[str, object]:
    del current_route_points
    handled_ids: List[str] = []
    blocked_ad_lane_ids: List[int] = []

    for raw_message in list(messages or []):
        if not isinstance(raw_message, Mapping):
            continue
        message = dict(raw_message)
        message_id = str(message.get("id", "") or "").strip()
        if not message_id or str(message.get("type", "")).strip().lower() != "lane_closure":
            continue
        ad_lane_id = _resolve_blocked_ad_lane_id(message, global_planner)
        if ad_lane_id is None:
            continue
        block_ad_lane = getattr(global_planner, "block_ad_lane_id", None)
        if callable(block_ad_lane):
            try:
                block_ad_lane(ad_lane_id)
            except Exception:
                continue
        if ad_lane_id not in blocked_ad_lane_ids:
            blocked_ad_lane_ids.append(ad_lane_id)
        if message_id not in handled_ids:
            handled_ids.append(message_id)

    if not blocked_ad_lane_ids:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_ad_lane_ids": [],
            "debug_reason": "No lane-closure message resolved to an AD lane.",
        }

    route_summary = global_planner.trace_route(
        ego_position,
        goal_position,
        replace_stored_route=True,
    )
    if not route_summary.route_found:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_ad_lane_ids": blocked_ad_lane_ids,
            "debug_reason": route_summary.debug_reason,
        }

    return {
        "route_summary": route_summary,
        "route_points": [list(point) for point in route_summary.route_waypoints],
        "handled_message_ids": handled_ids,
        "blocked_ad_lane_ids": blocked_ad_lane_ids,
        "debug_reason": "",
    }


def _carla_waypoint_key(waypoint) -> tuple[int, int, int, float]:
    return (
        int(getattr(waypoint, "road_id", 0) or 0),
        int(getattr(waypoint, "section_id", 0) or 0),
        int(getattr(waypoint, "lane_id", 0) or 0),
        float(getattr(waypoint, "s", 0.0) or 0.0),
    )


def _neighbor_waypoints(waypoint) -> List[object]:
    neighbors: List[object] = []
    for method_name in ("get_left_lane", "get_right_lane"):
        method = getattr(waypoint, method_name, None)
        if callable(method):
            candidate = method()
            if candidate is not None:
                neighbors.append(candidate)
    next_method = getattr(waypoint, "next", None)
    if callable(next_method):
        neighbors.extend(list(next_method(5.0) or []))
    return neighbors


def _blocked_waypoints_around(hazard_waypoint) -> List[object]:
    blocked = [hazard_waypoint]
    for method_name in ("next", "previous"):
        current = hazard_waypoint
        for _ in range(2):
            method = getattr(current, method_name, None)
            candidates = list(method(5.0) or []) if callable(method) else []
            if not candidates:
                break
            current = candidates[0]
            blocked.append(current)
    return blocked


def _location_point(location) -> Dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(getattr(location, "z", 0.0)),
    }


def _legacy_carla_reroute(
    *,
    messages,
    world_map,
    carla,
    global_planner,
    ego_transform,
    goal_location,
) -> Dict[str, object]:
    handled_ids: List[str] = []
    blocked_waypoints: List[object] = []
    resolved_messages: List[Dict[str, object]] = []

    for raw_message in list(messages or []):
        if not isinstance(raw_message, Mapping):
            continue
        message = dict(raw_message)
        message_id = str(message.get("id", "") or "").strip()
        if not message_id or str(message.get("type", "")).strip().lower() != "lane_closure":
            continue
        position = _message_position(message)
        if position is None:
            continue
        location = carla.Location(**position)
        hazard_waypoint = world_map.get_waypoint(location)
        if hazard_waypoint is None:
            continue
        local_blocked = _blocked_waypoints_around(hazard_waypoint)
        blocked_waypoints.extend(local_blocked)
        hazard_key = _carla_waypoint_key(hazard_waypoint)
        canonical_lane_id = int(canonical_lane_id_for_waypoint(hazard_waypoint))
        resolved = dict(message)
        resolved.update(
            {
                "road_id": int(hazard_key[0]),
                "section_id": int(hazard_key[1]),
                "lane_id": int(canonical_lane_id),
                "carla_lane_id": int(hazard_key[2]),
                "normalized_from_position": True,
                "lane_id_matches_canonical": int(canonical_lane_id)
                == int(resolved.get("lane_id", canonical_lane_id)),
                "lane_ids_match_carla": int(hazard_key[2])
                == int(resolved.get("lane_id", hazard_key[2])),
                "blocked_waypoint_key": list(hazard_key),
                "blocked_waypoint_keys": [
                    list(_carla_waypoint_key(waypoint)) for waypoint in local_blocked
                ],
            }
        )
        resolved_messages.append(resolved)
        handled_ids.append(message_id)

    unique_blocked: List[object] = []
    blocked_keys: List[tuple[int, int, int, float]] = []
    for waypoint in blocked_waypoints:
        key = _carla_waypoint_key(waypoint)
        if key not in blocked_keys:
            blocked_keys.append(key)
            unique_blocked.append(waypoint)

    if not handled_ids:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_waypoint_keys": [],
            "resolved_messages": [],
            "debug_reason": "No lane-closure message resolved to a CARLA waypoint.",
        }

    start_location = getattr(ego_transform, "location", None)
    start_waypoint = None if start_location is None else world_map.get_waypoint(start_location)
    goal_waypoint = world_map.get_waypoint(goal_location)
    if start_waypoint is None or goal_waypoint is None:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": handled_ids,
            "blocked_waypoint_keys": [list(key) for key in blocked_keys],
            "resolved_messages": resolved_messages,
            "debug_reason": "Could not resolve reroute start or goal waypoint.",
        }

    goal_candidates = [goal_waypoint]
    for method_name in ("get_left_lane", "get_right_lane"):
        method = getattr(goal_waypoint, method_name, None)
        candidate = method() if callable(method) else None
        if candidate is not None:
            goal_candidates.append(candidate)
    goal_keys = {_carla_waypoint_key(waypoint) for waypoint in goal_candidates}
    blocked_key_set = set(blocked_keys)
    start_key = _carla_waypoint_key(start_waypoint)
    frontier = deque([start_waypoint])
    parents = {start_key: None}
    waypoints_by_key = {start_key: start_waypoint}
    reached_key = None
    while frontier:
        waypoint = frontier.popleft()
        waypoint_key = _carla_waypoint_key(waypoint)
        if waypoint_key in goal_keys and waypoint_key not in blocked_key_set:
            reached_key = waypoint_key
            break
        for neighbor in _neighbor_waypoints(waypoint):
            neighbor_key = _carla_waypoint_key(neighbor)
            if neighbor_key in parents or neighbor_key in blocked_key_set:
                continue
            parents[neighbor_key] = waypoint_key
            waypoints_by_key[neighbor_key] = neighbor
            frontier.append(neighbor)

    if reached_key is None:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": handled_ids,
            "blocked_waypoint_keys": [list(key) for key in blocked_keys],
            "resolved_messages": resolved_messages,
            "debug_reason": "No CARLA waypoint route remains around the closure.",
        }

    route_keys = []
    cursor = reached_key
    while cursor is not None:
        route_keys.append(cursor)
        cursor = parents[cursor]
    route_keys.reverse()
    route_waypoints = [waypoints_by_key[key] for key in route_keys]
    route_points = [
        [
            float(waypoint.transform.location.x),
            float(waypoint.transform.location.y),
        ]
        for waypoint in route_waypoints
    ]
    summary = RoutePlanSummary(
        route_found=True,
        start_road_id=str(start_key[0]),
        start_lane_id=int(canonical_lane_id_for_waypoint(start_waypoint)),
        goal_road_id=str(reached_key[0]),
        goal_lane_id=int(canonical_lane_id_for_waypoint(waypoints_by_key[reached_key])),
        optimal_lane_id=int(canonical_lane_id_for_waypoint(waypoints_by_key[reached_key])),
        distance_to_destination_m=sum(
            (
                (route_points[index][0] - route_points[index - 1][0]) ** 2
                + (route_points[index][1] - route_points[index - 1][1]) ** 2
            )
            ** 0.5
            for index in range(1, len(route_points))
        ),
        next_macro_maneuver="Lane Follow",
        route_waypoints=route_points,
        road_options=["LANEFOLLOW"] * len(route_points),
    )
    replace_route = getattr(global_planner, "replace_stored_route", None)
    if callable(replace_route):
        replace_route(
            summary=summary,
            per_waypoint_options=summary.road_options,
            per_waypoint_lane_ids=[
                int(canonical_lane_id_for_waypoint(waypoint))
                for waypoint in route_waypoints
            ],
        )
    return {
        "route_summary": summary,
        "route_points": route_points,
        "handled_message_ids": handled_ids,
        "blocked_waypoint_keys": [list(key) for key in blocked_keys],
        "resolved_messages": resolved_messages,
        "debug_reason": "",
    }


def reroute_from_lane_closure_messages(*, messages, global_planner, **kwargs):
    """Reroute through either the AD-map or CARLA waypoint backend."""

    world_map = kwargs.pop("world_map", None)
    if world_map is not None:
        return _legacy_carla_reroute(
            messages=messages,
            world_map=world_map,
            carla=kwargs.pop("carla"),
            global_planner=global_planner,
            ego_transform=kwargs.pop("ego_transform"),
            goal_location=kwargs.pop("goal_location"),
        )
    return _reroute_ad_map(
        messages=messages,
        global_planner=global_planner,
        ego_position=kwargs.pop("ego_position"),
        goal_position=kwargs.pop("goal_position"),
        current_route_points=kwargs.pop("current_route_points", []),
    )
