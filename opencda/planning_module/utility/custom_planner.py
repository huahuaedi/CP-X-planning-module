"""CARLA-free integration helpers for the custom :mod:`Global_Planner`.

This module only converts the custom planner's public ``Route`` and ``Waypoint``
objects into the compact records consumed by behavior planning and MPC.  Map
queries and route searches always remain owned by ``GlobalPlanner``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from statistics import median
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    from Global_Planner import GlobalPlanner, Route, Waypoint
except ModuleNotFoundError:  # pragma: no cover - package import path
    from opencda.planning_module.Global_Planner import GlobalPlanner, Route, Waypoint


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBAL_PLANNER_ROOT = os.path.join(PROJECT_ROOT, "Global_Planner")
GLOBAL_PLANNER_CACHE_ROOT = os.path.join(GLOBAL_PLANNER_ROOT, "cache")
GLOBAL_PLANNER_INSTALL_ROOT = os.path.join(GLOBAL_PLANNER_ROOT, "map_repo", "install")
DEFAULT_CARLA_ROOT = "/home/umd-user/carla_source/carla"
INVALID_LANE_ID = 0


@dataclass
class RoutePlanSummary:
    """Planning-facing view of a route returned by the custom Global_Planner."""

    route_found: bool
    start_road_id: str
    start_lane_id: int
    goal_road_id: str
    goal_lane_id: int
    optimal_lane_id: int
    distance_to_destination_m: float
    next_macro_maneuver: str
    route_waypoints: List[List[float]]
    road_options: List[str] = field(default_factory=list)
    current_road_option: str = "LANEFOLLOW"
    debug_reason: str = ""


def world_point(value, *, default_z: float = 0.0) -> Dict[str, float] | None:
    """Convert a plain point or a simulator point object into numeric values."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            return None
        return {
            "x": float(value["x"]),
            "y": float(value["y"]),
            "z": float(value.get("z", default_z)),
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) < 2:
            return None
        return {
            "x": float(value[0]),
            "y": float(value[1]),
            "z": float(value[2]) if len(value) >= 3 else float(default_z),
        }
    if hasattr(value, "x") and hasattr(value, "y"):
        return {
            "x": float(getattr(value, "x", 0.0)),
            "y": float(getattr(value, "y", 0.0)),
            "z": float(getattr(value, "z", default_z)),
        }
    location = getattr(value, "location", None)
    if location is not None:
        return world_point(location, default_z=default_z)
    return None


def waypoint_position(waypoint: Waypoint | None) -> Dict[str, float] | None:
    if waypoint is None:
        return None
    return world_point(getattr(waypoint, "position", None))


def waypoint_key(waypoint: Waypoint | None) -> Tuple[int, float] | None:
    if waypoint is None:
        return None
    return int(waypoint.ad_lane_id), round(float(waypoint.parametric_offset), 6)


def direction_key(raw_lane_id: int) -> str:
    return "positive" if int(raw_lane_id) > 0 else "negative"


def behavior_lane_waypoints(waypoint: Waypoint | None) -> List[Waypoint]:
    """Return same-direction lanes ordered from right to left."""

    if waypoint is None:
        return []
    rightmost = waypoint
    visited = {int(waypoint.ad_lane_id)}
    while True:
        candidate = rightmost.right()
        if candidate is None or int(candidate.ad_lane_id) in visited:
            break
        visited.add(int(candidate.ad_lane_id))
        rightmost = candidate

    lanes = [rightmost]
    while True:
        candidate = lanes[-1].left()
        if candidate is None or int(candidate.ad_lane_id) in {
            int(item.ad_lane_id) for item in lanes
        }:
            break
        lanes.append(candidate)
    return lanes


def behavior_lane_id(waypoint: Waypoint | None) -> int:
    if waypoint is None:
        return int(INVALID_LANE_ID)
    for index, candidate in enumerate(behavior_lane_waypoints(waypoint)):
        if int(candidate.ad_lane_id) == int(waypoint.ad_lane_id):
            return int(index + 1)
    return int(INVALID_LANE_ID)


def waypoint_for_behavior_lane(
    waypoint: Waypoint | None,
    target_lane_id: int,
) -> Waypoint | None:
    lanes = behavior_lane_waypoints(waypoint)
    lane_index = int(target_lane_id) - 1
    if 0 <= lane_index < len(lanes):
        return lanes[lane_index]
    return waypoint


def lane_context(planner: GlobalPlanner, position) -> Dict[str, object]:
    point = world_point(position)
    if point is None:
        return {
            "road_id": "unknown_road",
            "road_numeric_id": -1,
            "section_id": -1,
            "direction": "unknown",
            "lane_id": int(INVALID_LANE_ID),
            "lane_ids": [],
            "lane_count": 0,
            "min_lane_id": int(INVALID_LANE_ID),
            "max_lane_id": int(INVALID_LANE_ID),
            "can_change_left": False,
            "can_change_right": False,
            "heading_rad": None,
            "is_intersection": False,
            "lane_width_m": 3.5,
        }

    raw = dict(planner.get_lane_context(point) or {})
    lane_ids = [int(item) for item in list(raw.get("lane_ids", []) or [])]
    current_lane_id = int(raw.get("lane_id", INVALID_LANE_ID) or INVALID_LANE_ID)
    current_index = lane_ids.index(current_lane_id) if current_lane_id in lane_ids else -1
    road_numeric_id = int(raw.get("road_id", -1) or -1)
    section_id = int(raw.get("section_id", -1) or -1)
    raw_lane_id = int(raw.get("opendrive_lane_id", 0) or 0)
    return {
        **raw,
        "road_id": f"{road_numeric_id}:{section_id}",
        "road_numeric_id": road_numeric_id,
        "section_id": section_id,
        "direction": direction_key(raw_lane_id),
        "lane_id": current_lane_id,
        "lane_ids": lane_ids,
        "lane_count": len(lane_ids),
        "min_lane_id": min(lane_ids) if lane_ids else int(INVALID_LANE_ID),
        "max_lane_id": max(lane_ids) if lane_ids else int(INVALID_LANE_ID),
        "can_change_left": bool(0 <= current_index < len(lane_ids) - 1),
        "can_change_right": bool(current_index > 0),
        "is_intersection": bool(raw.get("is_intersection", False)),
        "lane_width_m": max(0.1, float(raw.get("lane_width_m", 3.5) or 3.5)),
    }


def _route_points(route: Route | None) -> List[List[float]]:
    points: List[List[float]] = []
    for waypoint in list(getattr(route, "sampled_waypoints", []) or []):
        position = waypoint_position(waypoint)
        if position is None:
            continue
        point = [float(position["x"]), float(position["y"])]
        if points and math.hypot(points[-1][0] - point[0], points[-1][1] - point[1]) <= 1.0e-6:
            continue
        points.append(point)
    return points


def _route_options(route: Route | None) -> List[str]:
    waypoints = list(getattr(route, "sampled_waypoints", []) or [])
    if not waypoints:
        return []
    options = ["LANEFOLLOW"] * len(waypoints)
    for index in range(1, len(waypoints) - 1):
        if not bool(getattr(waypoints[index], "is_intersection", False)):
            continue
        before = float(getattr(waypoints[index - 1], "heading", 0.0) or 0.0)
        after = float(getattr(waypoints[index + 1], "heading", before) or before)
        delta = (after - before + math.pi) % (2.0 * math.pi) - math.pi
        if delta > math.radians(20.0):
            options[index] = "LEFT"
        elif delta < -math.radians(20.0):
            options[index] = "RIGHT"
        else:
            options[index] = "STRAIGHT"
    deduped: List[str] = []
    for option in options:
        if not deduped or option != deduped[-1]:
            deduped.append(option)
    return deduped


def _next_maneuver(options: Sequence[str]) -> str:
    for option in options:
        if option == "LEFT":
            return "Turn Left"
        if option == "RIGHT":
            return "Turn Right"
        if option == "STRAIGHT":
            return "Continue Straight"
    return "Continue Straight"


def _project_route_progress(points: Sequence[Sequence[float]], x_m: float, y_m: float) -> Tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    cumulative = [0.0]
    for first, second in zip(points[:-1], points[1:]):
        cumulative.append(cumulative[-1] + math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1])))
    best_distance = float("inf")
    best_progress = 0.0
    for index, (first, second) in enumerate(zip(points[:-1], points[1:])):
        ax, ay = float(first[0]), float(first[1])
        bx, by = float(second[0]), float(second[1])
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        alpha = 0.0 if length_sq <= 1.0e-12 else max(0.0, min(1.0, ((x_m - ax) * dx + (y_m - ay) * dy) / length_sq))
        px, py = ax + alpha * dx, ay + alpha * dy
        distance = math.hypot(x_m - px, y_m - py)
        if distance < best_distance:
            best_distance = distance
            best_progress = cumulative[index] + alpha * math.sqrt(max(length_sq, 0.0))
    return best_progress, cumulative[-1]


def route_summary(
    planner: GlobalPlanner,
    route: Route | None = None,
    *,
    position=None,
    debug_reason: str = "",
) -> RoutePlanSummary:
    active_route = route if route is not None else planner.active_route
    if active_route is None:
        return RoutePlanSummary(False, "unknown_road", 0, "unknown_road", 0, 0, 0.0, "Continue Straight", [], debug_reason=debug_reason)

    points = _route_points(active_route)
    start_wp = getattr(active_route, "resolved_start", None)
    goal_wp = getattr(active_route, "resolved_goal", None)
    start_lane = behavior_lane_id(start_wp)
    goal_lane = behavior_lane_id(goal_wp)
    remaining = float(getattr(active_route, "length_m", 0.0) or 0.0)
    current_lane = start_lane
    optimal_lane = goal_lane
    current_option = "LANEFOLLOW"
    if position is not None:
        point = world_point(position)
        if point is not None:
            progress, total = _project_route_progress(points, point["x"], point["y"])
            remaining = max(0.0, total - progress)
            context = lane_context(planner, point)
            current_lane = int(context.get("lane_id", current_lane) or current_lane)
            raw_optimal = planner.find_opt_lane(point)
            raw_to_behavior = dict(context.get("behavior_lane_by_opendrive_lane", {}) or {})
            optimal_lane = int(raw_to_behavior.get(int(raw_optimal), current_lane)) if raw_optimal is not None else current_lane

    options = _route_options(active_route)
    return RoutePlanSummary(
        route_found=bool(points),
        start_road_id=f"{int(getattr(start_wp, 'road_id', 0) or 0)}:{int(getattr(start_wp, 'section_id', 0) or 0)}",
        start_lane_id=int(current_lane),
        goal_road_id=f"{int(getattr(goal_wp, 'road_id', 0) or 0)}:{int(getattr(goal_wp, 'section_id', 0) or 0)}",
        goal_lane_id=int(goal_lane),
        optimal_lane_id=int(optimal_lane),
        distance_to_destination_m=float(remaining),
        next_macro_maneuver=_next_maneuver(options),
        route_waypoints=points,
        road_options=list(options),
        current_road_option=current_option,
        debug_reason=str(debug_reason),
    )


def build_lane_center_waypoints_from_route(route: Route) -> Tuple[List[dict], dict]:
    """Build MPC lane records from custom route waypoints and adjacent lanes."""

    records: List[dict] = []
    seen_keys: set[Tuple[int, float]] = set()
    lane_widths: List[float] = []
    max_lane_count = 1
    for route_waypoint in list(route.sampled_waypoints or []):
        lanes = behavior_lane_waypoints(route_waypoint)
        max_lane_count = max(max_lane_count, len(lanes))
        for lane_index, waypoint in enumerate(lanes, start=1):
            key = waypoint_key(waypoint)
            position = waypoint_position(waypoint)
            if key is None or position is None or key in seen_keys:
                continue
            seen_keys.add(key)
            successors = list(waypoint.next(float(route.sampling_resolution_m)) or [])
            successor_positions = []
            successor_keys = []
            for successor in successors:
                successor_position = waypoint_position(successor)
                successor_key = waypoint_key(successor)
                if successor_position is None or successor_key is None:
                    continue
                successor_positions.append([successor_position["x"], successor_position["y"]])
                successor_keys.append(successor_key)
            lane_width = max(0.1, float(waypoint.lane_width_m or 3.5))
            lane_widths.append(lane_width)
            record = {
                "planner_waypoint": waypoint,
                "planner_waypoint_key": key,
                "position": [position["x"], position["y"]],
                "heading_rad": float(waypoint.heading or 0.0),
                "lane_id": int(lane_index),
                "opendrive_lane_id": int(waypoint.lane_id or 0),
                "ad_lane_id": int(waypoint.ad_lane_id),
                "road_id": f"{int(waypoint.road_id or 0)}:{int(waypoint.section_id or 0)}",
                "direction": direction_key(int(waypoint.lane_id or 0)),
                "lane_width_m": lane_width,
                "is_intersection": bool(waypoint.is_intersection),
                "maneuver": "straight",
                "successors": successor_positions,
                "successor_keys": successor_keys,
            }
            if successor_positions:
                record["next"] = successor_positions[0]
                record["next_key"] = successor_keys[0]
            records.append(record)
    return records, {
        "lane_width_m": float(median(lane_widths)) if lane_widths else 3.5,
        "lane_count": int(max_lane_count),
    }


def _candidate_map_leaf(map_name: str) -> str:
    leaf = str(map_name or "").strip().rstrip("/").split("/")[-1]
    return leaf[:-5] if leaf.lower().endswith(".umap") else leaf


def _opendrive_path_for_map(carla_root: str, map_name: str) -> str:
    return os.path.join(carla_root, "Unreal", "CarlaUE4", "Content", "Carla", "Maps", "OpenDrive", f"{_candidate_map_leaf(map_name)}.xodr")


def _resolve_existing_path(raw_path: str, scenario_cfg: Mapping[str, object]) -> str | None:
    if not str(raw_path or "").strip():
        return None
    roots = [PROJECT_ROOT, str(scenario_cfg.get("_scenario_dir", "") or "")]
    scenario_path = str(scenario_cfg.get("_scenario_path", "") or "")
    if scenario_path:
        roots.append(os.path.dirname(scenario_path))
    if os.path.isabs(raw_path):
        return raw_path if os.path.isfile(raw_path) else None
    for root in roots:
        if not root:
            continue
        candidate = os.path.abspath(os.path.join(root, raw_path))
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_planner_xodr_path(scenario_cfg: Mapping[str, object]) -> str:
    sumo_cfg = dict(scenario_cfg.get("sumo", {}) or {})
    carla_cfg = dict(scenario_cfg.get("carla", {}) or {})
    configured = str(sumo_cfg.get("xodr_path", "") or "")
    resolved = _resolve_existing_path(configured, scenario_cfg)
    if resolved:
        return resolved
    carla_root = str(carla_cfg.get("carla_root", DEFAULT_CARLA_ROOT) or DEFAULT_CARLA_ROOT)
    map_name = str(carla_cfg.get("map", "") or "")
    candidates = [map_name]
    if _candidate_map_leaf(map_name).endswith("_Opt"):
        candidates.append(map_name[:-4])
    for candidate_name in candidates:
        candidate = _opendrive_path_for_map(carla_root, candidate_name)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError("Could not resolve the OpenDRIVE file required by custom Global_Planner.")


def build_custom_global_planner(
    *,
    scenario_cfg: Mapping[str, object],
    sample_distance_m: float,
) -> GlobalPlanner:
    planning_cfg = dict(scenario_cfg.get("planning", {}) or {})
    planner = GlobalPlanner(
        xodr_path=resolve_planner_xodr_path(scenario_cfg),
        cache_root=GLOBAL_PLANNER_CACHE_ROOT,
        centerline_spacing_m=float(sample_distance_m),
        default_search_radius_m=float(planning_cfg.get("search_radius_m", 8.0)),
        default_lane_change_penalty_m=float(planning_cfg.get("lane_change_penalty_m", 2.0)),
        lane_change_distance_m=float(planning_cfg.get("lane_change_distance_m", 8.0)),
        ad_map_install_root=GLOBAL_PLANNER_INSTALL_ROOT,
    )
    planner.load()
    planner.clear_blocked_lanes()
    return planner


__all__ = [
    "GlobalPlanner",
    "INVALID_LANE_ID",
    "RoutePlanSummary",
    "behavior_lane_id",
    "behavior_lane_waypoints",
    "build_custom_global_planner",
    "build_lane_center_waypoints_from_route",
    "direction_key",
    "lane_context",
    "resolve_planner_xodr_path",
    "route_summary",
    "waypoint_for_behavior_lane",
    "waypoint_key",
    "waypoint_position",
    "world_point",
]
