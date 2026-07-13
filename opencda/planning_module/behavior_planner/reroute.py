"""Lane-closure rerouting through the custom Global_Planner only."""

from __future__ import annotations

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
from utility.custom_planner import (
    behavior_lane_waypoints,
    route_summary,
    waypoint_position,
    world_point,
)


def _closure_position(message: Mapping[str, object]) -> Dict[str, float] | None:
    return world_point(message.get("position", None))


def _blocked_waypoints_for_message(global_planner, message: Mapping[str, object]):
    """Resolve one closure using its position and existing custom planner methods."""

    position = _closure_position(message)
    if position is None:
        print(
            "[BEHAVIOR] reroute message rejected: a closure position is required "
            "to resolve the custom Global_Planner ad_lane_id."
        )
        return []
    waypoint = global_planner.get_waypoint(position)
    if waypoint is None:
        print(
            "[BEHAVIOR] reroute message rejected: custom Global_Planner could not "
            "find a lane at the closure position."
        )
        return []
    if bool(message.get("block_entire_road", False)):
        return behavior_lane_waypoints(waypoint)
    return [waypoint]


def reroute_from_lane_closure_messages(
    *,
    messages: Sequence[Mapping[str, object]],
    global_planner,
    ego_position=None,
    goal_position=None,
    ego_transform=None,
    goal_location=None,
    current_route_points: Sequence[Sequence[float]] | None = None,
) -> Dict[str, object]:
    """Block custom planner lanes, replan, and activate the replacement route."""

    del current_route_points
    ego_point = world_point(ego_position if ego_position is not None else ego_transform)
    goal_point = world_point(goal_position if goal_position is not None else goal_location)
    if ego_point is None or goal_point is None:
        return {
            "route_summary": None,
            "route": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_ad_lane_ids": [],
            "reroute_failed": True,
            "debug_reason": "Ego or final-destination position was unavailable.",
        }

    handled_ids: List[str] = []
    blocked_ad_lane_ids: List[int] = []
    resolved_any = False
    for message in lane_closure_messages(messages):
        resolved_waypoints = _blocked_waypoints_for_message(global_planner, message)
        if not resolved_waypoints:
            continue
        resolved_any = True
        for waypoint in resolved_waypoints:
            ad_lane_id = int(waypoint.ad_lane_id)
            global_planner.add_blocked_lane(ad_lane_id)
            if ad_lane_id not in blocked_ad_lane_ids:
                blocked_ad_lane_ids.append(ad_lane_id)
        message_id = str(message.get("id", "") or "").strip()
        if message_id:
            handled_ids.append(message_id)

    if not resolved_any:
        return {
            "route_summary": None,
            "route": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_ad_lane_ids": blocked_ad_lane_ids,
            "reroute_failed": True,
            "debug_reason": "No valid closure position could be resolved.",
        }

    previous_route = global_planner.active_route
    try:
        replacement_route = global_planner.trace_route(
            ego_point,
            goal_point,
            activate_route=True,
        )
    except Exception as exc:
        # Keep the last route available for display. The runner applies a full stop.
        global_planner.active_route = previous_route
        return {
            "route_summary": None,
            "route": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_ad_lane_ids": blocked_ad_lane_ids,
            "reroute_failed": True,
            "debug_reason": str(exc),
        }

    summary = route_summary(global_planner, replacement_route, position=ego_point)
    route_points = [
        [float(position["x"]), float(position["y"])]
        for waypoint in list(replacement_route.sampled_waypoints or [])
        for position in [waypoint_position(waypoint)]
        if position is not None
    ]
    return {
        "route_summary": summary,
        "route": replacement_route,
        "route_points": route_points,
        "handled_message_ids": handled_ids,
        "blocked_ad_lane_ids": blocked_ad_lane_ids,
        "reroute_failed": False,
        "debug_reason": "",
    }


__all__ = [
    "CP_MESSAGE_PATH",
    "control_messages",
    "ensure_cp_message_file_exists",
    "lane_closure_messages",
    "load_control_messages",
    "load_cp_message_payload",
    "load_cp_messages",
    "load_lane_closure_messages",
    "pop_lane_closure_messages",
    "remove_cp_messages_by_id",
    "reroute_from_lane_closure_messages",
    "reset_cp_message_payload",
    "write_cp_message_payload",
    "write_cp_messages",
]
