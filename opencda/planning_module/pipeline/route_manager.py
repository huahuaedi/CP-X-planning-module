"""Route lifecycle manager for the CP-X OpenCDA pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass
class RouteManagerStatus:
    route_found: bool = False
    route_point_count: int = 0
    remaining_distance_m: float = 0.0
    reached_destination: bool = False
    debug_reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "route_found": bool(self.route_found),
            "route_point_count": int(self.route_point_count),
            "remaining_distance_m": float(self.remaining_distance_m),
            "reached_destination": bool(self.reached_destination),
            "debug_reason": str(self.debug_reason),
        }


class CPXRouteManager:
    """Own destination, active global route, progress, and query diagnostics."""

    def __init__(
        self,
        *,
        global_planner: Any,
        carla_map: Any = None,
        carla_api: Any = None,
        carla_route_sampling_resolution_m: float = 1.0,
        carla_reference_smoothing_passes: int = 3,
        carla_rejoin_min_lateral_m: float = 0.35,
        carla_rejoin_max_lateral_m: float = 3.0,
        carla_rejoin_distance_m: float = 8.0,
        reached_distance_m: float = 3.0,
        stale_route_lateral_m: float = 12.0,
    ) -> None:
        self.global_planner = global_planner
        self.carla_map = carla_map
        self.carla_api = carla_api
        self.carla_route_sampling_resolution_m = max(
            0.25, float(carla_route_sampling_resolution_m)
        )
        self.carla_reference_smoothing_passes = max(
            0, int(carla_reference_smoothing_passes)
        )
        self.carla_rejoin_min_lateral_m = max(
            0.0, float(carla_rejoin_min_lateral_m)
        )
        self.carla_rejoin_max_lateral_m = max(
            self.carla_rejoin_min_lateral_m,
            float(carla_rejoin_max_lateral_m),
        )
        self.carla_rejoin_distance_m = max(
            1.0, float(carla_rejoin_distance_m)
        )
        self.reached_distance_m = max(0.1, float(reached_distance_m))
        self.stale_route_lateral_m = max(1.0, float(stale_route_lateral_m))
        self._active_route_summary = None
        self._start_point: Optional[Dict[str, float]] = None
        self._goal_point: Optional[Dict[str, float]] = None
        self._fallback_route_points: List[List[float]] = []
        self._carla_route_planner = None
        self._carla_route_entries: List[Any] = []
        self._carla_route_progress_index = 0
        self._carla_route_progress_initialized = False
        self._carla_route_projection: Optional[Tuple[int, float, float, float, float]] = None
        self._carla_route_sync_reason = "carla_route_progress_not_initialized"
        self._carla_route_debug_reason = "carla_route_not_initialized"
        self._last_status = RouteManagerStatus(debug_reason="route_not_initialized")

    def set_destination(
        self,
        *,
        start_point: Mapping[str, object],
        goal_point: Mapping[str, object],
    ) -> Any:
        self._start_point = _point_dict(start_point)
        self._goal_point = _point_dict(goal_point)
        self._active_route_summary = self.global_planner.plan_route_from_locations(
            start_location=self._start_point,
            goal_location=self._goal_point,
            replace_stored_route=True,
        )
        self._build_carla_route(
            start_point=self._start_point,
            goal_point=self._goal_point,
        )
        self._fallback_route_points = []
        if not bool(getattr(self._active_route_summary, "route_found", False)) or len(
            list(getattr(self._active_route_summary, "route_waypoints", []) or [])
        ) < 2:
            self._fallback_route_points = self._direct_fallback_route_points(
                start_point=self._start_point,
                goal_point=self._goal_point,
            )
        self._last_status = self._status_from_summary(self._active_route_summary)
        return self._active_route_summary

    def get_route_info(
        self,
        *,
        x_m: float,
        y_m: float,
        query_key: str,
        fallback_lane_id: int,
    ) -> Dict[str, object]:
        try:
            summary = self.global_planner.get_current_route_info(
                x_m=float(x_m),
                y_m=float(y_m),
                query_key=str(query_key),
            )
        except Exception as exc:
            self._last_status = RouteManagerStatus(
                route_found=False,
                route_point_count=0,
                remaining_distance_m=0.0,
                reached_destination=False,
                debug_reason=f"route_query_failed:{exc}",
            )
            return self._fallback_summary(
                fallback_lane_id=int(fallback_lane_id),
                debug_reason=str(self._last_status.debug_reason),
            )

        if summary is None:
            summary = self._active_route_summary
        if summary is None:
            self._last_status = RouteManagerStatus(debug_reason="route_missing")
            return self._fallback_summary(
                fallback_lane_id=int(fallback_lane_id),
                debug_reason="route_missing",
            )

        self._active_route_summary = summary
        self._last_status = self._status_from_summary(summary)
        if (
            not bool(getattr(summary, "route_found", False))
            and len(self._fallback_route_points) >= 2
        ):
            remaining = self._remaining_distance_on_fallback_route(
                x_m=float(x_m),
                y_m=float(y_m),
            )
            self._last_status = RouteManagerStatus(
                route_found=True,
                route_point_count=len(self._fallback_route_points),
                remaining_distance_m=float(remaining),
                reached_destination=bool(remaining <= self.reached_distance_m),
                debug_reason=self._fallback_debug_reason(summary),
            )
            return {
                "route_found": True,
                "optimal_lane_id": int(fallback_lane_id),
                "current_road_option": "FALLBACK_DIRECT",
                "next_macro_maneuver": "Continue Straight",
                "debug_reason": str(self._last_status.debug_reason),
                "remaining_distance_m": float(remaining),
                "reached_destination": bool(self._last_status.reached_destination),
            }
        lane_id = _to_int(getattr(summary, "optimal_lane_id", fallback_lane_id), fallback_lane_id)
        if int(lane_id) == 0:
            lane_id = int(fallback_lane_id)
        return {
            "route_found": bool(getattr(summary, "route_found", False)),
            "optimal_lane_id": int(lane_id),
            "current_road_option": str(getattr(summary, "current_road_option", "")),
            "next_macro_maneuver": str(
                getattr(summary, "next_macro_maneuver", "Continue Straight")
            ),
            "debug_reason": str(
                getattr(summary, "debug_reason", "planning_module_global_route")
            ),
            "remaining_distance_m": float(
                getattr(summary, "distance_to_destination_m", 0.0) or 0.0
            ),
            "reached_destination": bool(self._last_status.reached_destination),
        }

    def route_points(self, *, x_m: Optional[float] = None, y_m: Optional[float] = None, query_key: str = "") -> List[List[float]]:
        summary = None
        if x_m is not None and y_m is not None:
            try:
                summary = self.global_planner.get_current_route_info(
                    x_m=float(x_m),
                    y_m=float(y_m),
                    query_key=str(query_key or "route_points"),
                )
            except Exception:
                summary = None
        if summary is None:
            summary = self._active_route_summary
        route_waypoints = list(getattr(summary, "route_waypoints", []) or [])
        route_points = _route_points_from_waypoints(route_waypoints)
        if len(route_points) >= 2:
            return route_points
        return [list(point) for point in list(self._fallback_route_points or [])]

    def geometry_route_points(
        self,
        *,
        x_m: Optional[float] = None,
        y_m: Optional[float] = None,
        query_key: str = "",
    ) -> List[List[float]]:
        """Return the route geometry shared by reference generation and debug.

        The custom global planner remains responsible for route topology and
        maneuver semantics.  When available, CARLA GRP waypoints are the
        geometric source of truth because they follow the simulator lane
        center and include the selected junction connector.
        """

        nodes = self._carla_route_nodes()
        if len(nodes) >= 2:
            points: List[List[float]] = []
            for index, node in enumerate(nodes):
                if index + 1 < len(nodes):
                    next_node = nodes[index + 1]
                    heading_rad = math.atan2(
                        float(next_node[1]) - float(node[1]),
                        float(next_node[0]) - float(node[0]),
                    )
                elif points:
                    heading_rad = float(points[-1][3])
                else:
                    heading_rad = 0.0
                points.append([
                    float(node[0]),
                    float(node[1]),
                    float(node[2]),
                    float(heading_rad),
                ])
            return points
        return self.route_points(x_m=x_m, y_m=y_m, query_key=query_key)

    def upcoming_turn(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        lookahead_m: float,
    ) -> Tuple[str, float, str]:
        """Return the first CARLA GRP turn option within the lookahead."""

        sync_reason = self.sync_carla_route_progress(
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
        )
        nodes = self._carla_route_nodes()
        if len(nodes) < 2 or self._carla_route_projection is None:
            return "", float("inf"), str(sync_reason)

        segment_index, projection_x_m, projection_y_m, _, lateral_m = (
            self._carla_route_projection
        )
        if float(lateral_m) > float(self.stale_route_lateral_m):
            return "", float("inf"), str(sync_reason)

        distance_m = 0.0
        previous_xy = (float(projection_x_m), float(projection_y_m))
        limit_m = max(0.0, float(lookahead_m))
        for node in nodes[int(segment_index) + 1 :]:
            current_xy = (float(node[0]), float(node[1]))
            distance_m += math.hypot(
                current_xy[0] - previous_xy[0],
                current_xy[1] - previous_xy[1],
            )
            option = str(node[4] or "").strip().upper()
            if option in {"LEFT", "RIGHT"}:
                if float(distance_m) <= float(limit_m):
                    return option.lower(), float(distance_m), "carla_route_turn_ahead"
                break
            if float(distance_m) > float(limit_m):
                break
            previous_xy = current_xy
        return "", float("inf"), "carla_route_no_turn_in_lookahead"

    def carla_waypoint_reference(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        horizon_steps: int,
        step_distance_m: float,
        target_speed_mps: float,
        fallback_lane_id: int,
    ) -> Tuple[List[Dict[str, object]], str]:
        """Sample a local reference from the CARLA GRP waypoint chain.

        The chain already contains the route-selected junction connector. Route
        progress is monotonic, so a nearby crossing or adjacent connector cannot
        make the local reference jump backward to another branch.
        """

        sync_reason = self.sync_carla_route_progress(
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
        )
        nodes = self._carla_route_nodes()
        if len(nodes) < 2:
            return [], str(sync_reason or "carla_route_waypoints_empty")
        if self._carla_route_projection is None:
            return [], str(sync_reason or "carla_route_projection_failed")
        (
            segment_index,
            projection_x_m,
            projection_y_m,
            projection_ratio,
            lateral_distance_m,
        ) = self._carla_route_projection
        if float(lateral_distance_m) > float(self.stale_route_lateral_m):
            return [], str(sync_reason)
        first = nodes[segment_index]
        second = nodes[min(segment_index + 1, len(nodes) - 1)]

        polyline: List[Tuple[float, float, float, Any, str]] = [
            (
                float(projection_x_m),
                float(projection_y_m),
                float(first[2]) + float(projection_ratio) * (float(second[2]) - float(first[2])),
                second[3],
                second[4],
            )
        ]
        polyline.extend(nodes[segment_index + 1 :])
        polyline = _deduplicate_carla_nodes(polyline)
        if len(polyline) < 2:
            return [], "carla_route_remaining_polyline_too_short"

        cumulative = [0.0]
        for previous, current in zip(polyline[:-1], polyline[1:]):
            cumulative.append(
                cumulative[-1]
                + math.hypot(current[0] - previous[0], current[1] - previous[1])
            )
        step_m = max(0.25, float(step_distance_m))
        samples: List[Dict[str, object]] = []
        sample_segment = 0
        for sample_index in range(max(1, int(horizon_steps))):
            target_s_m = float(sample_index + 1) * float(step_m)
            while sample_segment + 1 < len(cumulative) and cumulative[sample_segment + 1] < target_s_m:
                sample_segment += 1
            if sample_segment + 1 < len(polyline):
                node_a = polyline[sample_segment]
                node_b = polyline[sample_segment + 1]
                segment_length_m = max(
                    1.0e-6, cumulative[sample_segment + 1] - cumulative[sample_segment]
                )
                ratio = min(
                    1.0,
                    max(0.0, (target_s_m - cumulative[sample_segment]) / segment_length_m),
                )
                x_m = node_a[0] + ratio * (node_b[0] - node_a[0])
                y_m = node_a[1] + ratio * (node_b[1] - node_a[1])
                heading_rad = math.atan2(node_b[1] - node_a[1], node_b[0] - node_a[0])
                waypoint = node_b[3]
                option = node_b[4]
            else:
                node_a = polyline[-2]
                node_b = polyline[-1]
                heading_rad = math.atan2(node_b[1] - node_a[1], node_b[0] - node_a[0])
                extra_m = max(0.0, target_s_m - cumulative[-1])
                x_m = node_b[0] + extra_m * math.cos(heading_rad)
                y_m = node_b[1] + extra_m * math.sin(heading_rad)
                waypoint = node_b[3]
                option = node_b[4]
            normalized_option = str(option or "").strip().upper().replace("_", "")
            lane_transition_kind = (
                "lateral_lane_change"
                if normalized_option in {"CHANGELANELEFT", "CHANGELANERIGHT"}
                else "longitudinal_successor"
            )
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(_canonical_carla_lane_id(waypoint, fallback_lane_id)),
                "lane_width_m": float(getattr(waypoint, "lane_width", 3.5) or 3.5),
                "road_id": int(getattr(waypoint, "road_id", 0) or 0),
                "road_option": str(option),
                "lane_transition_kind": str(lane_transition_kind),
                "speed_ref_mps": max(0.0, float(target_speed_mps)),
                "v_ref_mps": max(0.0, float(target_speed_mps)),
                "speed_mps": max(0.0, float(target_speed_mps)),
            })
        samples = _smooth_carla_reference_samples(
            samples,
            passes=int(self.carla_reference_smoothing_passes),
        )
        reason = "carla_grp_waypoint_chain_smoothed"
        if (
            float(lateral_distance_m) >= float(self.carla_rejoin_min_lateral_m)
            and float(lateral_distance_m) <= float(self.carla_rejoin_max_lateral_m)
        ):
            samples = _apply_route_rejoin_offset(
                samples,
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                projection_x_m=float(projection_x_m),
                projection_y_m=float(projection_y_m),
                step_distance_m=float(step_m),
                rejoin_distance_m=float(self.carla_rejoin_distance_m),
            )
            reason += ":route_rejoin"
        return samples, reason

    def sync_carla_route_progress(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
    ) -> str:
        """Synchronize ego progress against the CARLA waypoint route.

        The first call searches the complete route. Later calls use a bounded
        forward window and preserve monotonic progress.
        """

        nodes = self._carla_route_nodes()
        if len(nodes) < 2:
            self._carla_route_projection = None
            self._carla_route_sync_reason = str(
                self._carla_route_debug_reason or "carla_route_unavailable"
            )
            return str(self._carla_route_sync_reason)

        if not bool(self._carla_route_progress_initialized):
            lower = 0
            upper = len(nodes) - 1
            search_mode = "global_init"
        else:
            lower = max(0, int(self._carla_route_progress_index) - 5)
            upper = min(
                len(nodes) - 1,
                max(lower + 1, int(self._carla_route_progress_index) + 80),
            )
            search_mode = "local_update"

        best = _best_carla_route_projection(
            nodes=nodes,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
            lower_index=int(lower),
            upper_index=int(upper),
        )
        if best is None:
            self._carla_route_sync_reason = f"carla_route_progress_{search_mode}_failed"
            return str(self._carla_route_sync_reason)

        _, _, best_index, _, _, _ = best
        segment_index = (
            int(best_index)
            if not bool(self._carla_route_progress_initialized)
            else max(int(self._carla_route_progress_index), int(best_index))
        )
        first = nodes[segment_index]
        second = nodes[min(segment_index + 1, len(nodes) - 1)]
        projection_x_m, projection_y_m, projection_ratio, lateral_distance_m = (
            _project_to_segment(
                x_m=float(ego_x_m),
                y_m=float(ego_y_m),
                first_xy=(first[0], first[1]),
                second_xy=(second[0], second[1]),
            )
        )
        self._carla_route_progress_index = int(segment_index)
        self._carla_route_progress_initialized = True
        self._carla_route_projection = (
            int(segment_index),
            float(projection_x_m),
            float(projection_y_m),
            float(projection_ratio),
            float(lateral_distance_m),
        )
        if float(lateral_distance_m) > float(self.stale_route_lateral_m):
            self._carla_route_sync_reason = (
                f"carla_route_stale:lateral={float(lateral_distance_m):.2f}"
            )
        else:
            self._carla_route_sync_reason = (
                f"carla_route_progress_{search_mode}:index={int(segment_index)}"
            )
        return str(self._carla_route_sync_reason)

    def _carla_route_nodes(self) -> List[Tuple[float, float, float, Any, str]]:
        nodes: List[Tuple[float, float, float, Any, str]] = []
        for entry in list(self._carla_route_entries or []):
            waypoint, option = _carla_route_entry(entry)
            location = getattr(getattr(waypoint, "transform", None), "location", None)
            if location is None:
                continue
            x_m = float(location.x)
            y_m = float(location.y)
            z_m = float(getattr(location, "z", 0.0))
            if nodes and math.hypot(x_m - nodes[-1][0], y_m - nodes[-1][1]) < 1.0e-3:
                continue
            nodes.append((x_m, y_m, z_m, waypoint, _road_option_name(option)))
        return nodes

    @property
    def carla_route_debug_reason(self) -> str:
        return str(self._carla_route_debug_reason)

    @property
    def carla_route_sync_reason(self) -> str:
        return str(self._carla_route_sync_reason)

    @property
    def carla_route_progress_index(self) -> int:
        return int(self._carla_route_progress_index)

    @property
    def last_status(self) -> RouteManagerStatus:
        return self._last_status

    def _build_carla_route(
        self,
        *,
        start_point: Mapping[str, object],
        goal_point: Mapping[str, object],
    ) -> None:
        self._carla_route_entries = []
        self._carla_route_progress_index = 0
        self._carla_route_progress_initialized = False
        self._carla_route_projection = None
        self._carla_route_sync_reason = "carla_route_progress_not_initialized"
        if self.carla_map is None or self.carla_api is None:
            self._carla_route_debug_reason = "carla_route_map_unavailable"
            return
        try:
            if self._carla_route_planner is None:
                from opencda.core.plan.global_route_planner import GlobalRoutePlanner
                from opencda.core.plan.global_route_planner_dao import GlobalRoutePlannerDAO

                dao = GlobalRoutePlannerDAO(
                    self.carla_map,
                    sampling_resolution=float(self.carla_route_sampling_resolution_m),
                )
                planner = GlobalRoutePlanner(dao)
                planner.setup()
                self._carla_route_planner = planner
            start_location = self.carla_api.Location(
                x=float(start_point["x"]),
                y=float(start_point["y"]),
                z=float(start_point.get("z", 0.0)),
            )
            goal_location = self.carla_api.Location(
                x=float(goal_point["x"]),
                y=float(goal_point["y"]),
                z=float(goal_point.get("z", 0.0)),
            )
            self._carla_route_entries = list(
                self._carla_route_planner.trace_route(start_location, goal_location) or []
            )
            self._carla_route_debug_reason = (
                "carla_grp_route_ready"
                if len(self._carla_route_entries) >= 2
                else "carla_grp_route_empty"
            )
        except Exception as exc:
            self._carla_route_entries = []
            self._carla_route_debug_reason = f"carla_grp_route_failed:{exc}"

    def _status_from_summary(self, summary: Any) -> RouteManagerStatus:
        route_points = list(getattr(summary, "route_waypoints", []) or [])
        remaining = float(getattr(summary, "distance_to_destination_m", 0.0) or 0.0)
        route_found = bool(getattr(summary, "route_found", False))
        return RouteManagerStatus(
            route_found=bool(route_found) or len(self._fallback_route_points) >= 2,
            route_point_count=max(len(route_points), len(self._fallback_route_points)),
            remaining_distance_m=float(
                remaining
                if bool(route_found)
                else self._fallback_route_total_distance_m()
            ),
            reached_destination=bool(
                (bool(route_found) and remaining <= self.reached_distance_m)
                or (
                    not bool(route_found)
                    and self._fallback_route_total_distance_m() <= self.reached_distance_m
                )
            ),
            debug_reason=self._fallback_debug_reason(summary)
            if not bool(route_found)
            else str(getattr(summary, "debug_reason", "route_active")),
        )

    @staticmethod
    def _fallback_summary(*, fallback_lane_id: int, debug_reason: str) -> Dict[str, object]:
        return {
            "route_found": False,
            "optimal_lane_id": int(fallback_lane_id),
            "current_road_option": "",
            "next_macro_maneuver": "Continue Straight",
            "debug_reason": str(debug_reason),
            "remaining_distance_m": 0.0,
            "reached_destination": False,
        }

    @staticmethod
    def _direct_fallback_route_points(
        *,
        start_point: Mapping[str, object],
        goal_point: Mapping[str, object],
    ) -> List[List[float]]:
        start = _point_dict(start_point)
        goal = _point_dict(goal_point)
        heading = math.atan2(float(goal["y"]) - float(start["y"]), float(goal["x"]) - float(start["x"]))
        return [
            [float(start["x"]), float(start["y"]), float(start.get("z", 0.0)), float(heading)],
            [float(goal["x"]), float(goal["y"]), float(goal.get("z", 0.0)), float(heading)],
        ]

    def _fallback_route_total_distance_m(self) -> float:
        if len(self._fallback_route_points) < 2:
            return 0.0
        first = self._fallback_route_points[0]
        last = self._fallback_route_points[-1]
        return math.hypot(float(last[0]) - float(first[0]), float(last[1]) - float(first[1]))

    def _remaining_distance_on_fallback_route(self, *, x_m: float, y_m: float) -> float:
        if len(self._fallback_route_points) < 2:
            return 0.0
        goal = self._fallback_route_points[-1]
        return math.hypot(float(goal[0]) - float(x_m), float(goal[1]) - float(y_m))

    @staticmethod
    def _fallback_debug_reason(summary: Any) -> str:
        raw_reason = str(getattr(summary, "debug_reason", "") or "").strip()
        if raw_reason:
            return f"global_route_failed_direct_fallback:{raw_reason}"
        return "global_route_failed_direct_fallback"


def _point_dict(point: Mapping[str, object]) -> Dict[str, float]:
    return {
        "x": float(point.get("x", point.get("x_m", 0.0))),
        "y": float(point.get("y", point.get("y_m", 0.0))),
        "z": float(point.get("z", point.get("z_m", 0.0))),
    }


def _route_points_from_waypoints(route_waypoints: List[Any]) -> List[List[float]]:
    points: List[List[float]] = []
    for index, raw_point in enumerate(route_waypoints):
        try:
            x_m = float(raw_point[0])
            y_m = float(raw_point[1])
            z_m = float(raw_point[2]) if len(raw_point) >= 3 else 0.0
        except Exception:
            continue
        if index < len(route_waypoints) - 1:
            try:
                nx_m = float(route_waypoints[index + 1][0])
                ny_m = float(route_waypoints[index + 1][1])
                heading_rad = math.atan2(ny_m - y_m, nx_m - x_m)
            except Exception:
                heading_rad = points[-1][3] if points else 0.0
        else:
            heading_rad = points[-1][3] if points else 0.0
        if points and math.hypot(points[-1][0] - x_m, points[-1][1] - y_m) < 1.0e-3:
            continue
        points.append([float(x_m), float(y_m), float(z_m), float(heading_rad)])
    return points


def _carla_route_entry(entry: Any) -> Tuple[Any, Any]:
    if isinstance(entry, (list, tuple)) and entry:
        return entry[0], entry[1] if len(entry) >= 2 else None
    return entry, None


def _road_option_name(option: Any) -> str:
    if option is None:
        return ""
    name = getattr(option, "name", None)
    if name is not None:
        return str(name).strip().upper()
    text = str(option).strip()
    return text.rsplit(".", 1)[-1].upper() if "." in text else text.upper()


def _canonical_carla_lane_id(waypoint: Any, fallback_lane_id: int) -> int:
    try:
        from utility.global_planner import canonical_lane_id_for_waypoint

        lane_id = int(canonical_lane_id_for_waypoint(waypoint))
        if lane_id != 0:
            return lane_id
    except Exception:
        pass
    return int(fallback_lane_id)


def _project_to_segment(
    *,
    x_m: float,
    y_m: float,
    first_xy: Sequence[float],
    second_xy: Sequence[float],
) -> Tuple[float, float, float, float]:
    dx_m = float(second_xy[0]) - float(first_xy[0])
    dy_m = float(second_xy[1]) - float(first_xy[1])
    length_sq = dx_m * dx_m + dy_m * dy_m
    if length_sq <= 1.0e-9:
        ratio = 0.0
    else:
        ratio = (
            (float(x_m) - float(first_xy[0])) * dx_m
            + (float(y_m) - float(first_xy[1])) * dy_m
        ) / length_sq
        ratio = min(1.0, max(0.0, float(ratio)))
    px_m = float(first_xy[0]) + ratio * dx_m
    py_m = float(first_xy[1]) + ratio * dy_m
    return px_m, py_m, ratio, math.hypot(float(x_m) - px_m, float(y_m) - py_m)


def _best_carla_route_projection(
    *,
    nodes: Sequence[Tuple[float, float, float, Any, str]],
    ego_x_m: float,
    ego_y_m: float,
    ego_heading_rad: float,
    lower_index: int,
    upper_index: int,
) -> Optional[Tuple[float, float, int, float, float, float]]:
    best = None
    lower = max(0, int(lower_index))
    upper = min(len(nodes) - 1, int(upper_index))
    for index in range(lower, upper):
        first = nodes[index]
        second = nodes[index + 1]
        px_m, py_m, ratio, distance_m = _project_to_segment(
            x_m=float(ego_x_m),
            y_m=float(ego_y_m),
            first_xy=(first[0], first[1]),
            second_xy=(second[0], second[1]),
        )
        segment_heading = math.atan2(second[1] - first[1], second[0] - first[0])
        heading_error = abs(_wrap_angle(segment_heading - float(ego_heading_rad)))
        opposite_penalty = 25.0 if heading_error > 0.75 * math.pi else 0.0
        score = float(distance_m) + float(opposite_penalty) + 0.2 * float(heading_error)
        candidate = (
            float(score),
            float(distance_m),
            int(index),
            float(px_m),
            float(py_m),
            float(ratio),
        )
        if best is None or candidate < best:
            best = candidate
    return best


def _deduplicate_carla_nodes(
    nodes: Sequence[Tuple[float, float, float, Any, str]],
) -> List[Tuple[float, float, float, Any, str]]:
    result: List[Tuple[float, float, float, Any, str]] = []
    for node in nodes:
        if result and math.hypot(node[0] - result[-1][0], node[1] - result[-1][1]) < 1.0e-3:
            continue
        result.append(node)
    return result


def _smooth_carla_reference_samples(
    samples: Sequence[Mapping[str, object]],
    *,
    passes: int,
) -> List[Dict[str, object]]:
    """Smooth a resampled CARLA connector without changing its route branch."""

    result = [dict(sample) for sample in list(samples or [])]
    if len(result) < 3:
        return _recompute_reference_headings(result)

    for _ in range(max(0, int(passes))):
        previous = [dict(sample) for sample in result]
        for index in range(1, len(result) - 1):
            before = previous[index - 1]
            current = previous[index]
            after = previous[index + 1]
            result[index]["x_ref_m"] = (
                float(before["x_ref_m"])
                + 2.0 * float(current["x_ref_m"])
                + float(after["x_ref_m"])
            ) / 4.0
            result[index]["y_ref_m"] = (
                float(before["y_ref_m"])
                + 2.0 * float(current["y_ref_m"])
                + float(after["y_ref_m"])
            ) / 4.0
            result[index]["x"] = float(result[index]["x_ref_m"])
            result[index]["y"] = float(result[index]["y_ref_m"])
    return _recompute_reference_headings(result)


def _apply_route_rejoin_offset(
    samples: Sequence[Mapping[str, object]],
    *,
    ego_x_m: float,
    ego_y_m: float,
    projection_x_m: float,
    projection_y_m: float,
    step_distance_m: float,
    rejoin_distance_m: float,
) -> List[Dict[str, object]]:
    """Decay the ego-to-route offset while retaining the selected connector."""

    result = [dict(sample) for sample in list(samples or [])]
    offset_x_m = float(ego_x_m) - float(projection_x_m)
    offset_y_m = float(ego_y_m) - float(projection_y_m)
    merge_distance_m = max(1.0, float(rejoin_distance_m))
    step_m = max(0.1, float(step_distance_m))
    for index, sample in enumerate(result):
        progress = min(
            1.0,
            max(0.0, float(index + 1) * float(step_m) / float(merge_distance_m)),
        )
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        residual = 1.0 - float(smooth_progress)
        sample["x_ref_m"] = float(sample["x_ref_m"]) + residual * float(offset_x_m)
        sample["y_ref_m"] = float(sample["y_ref_m"]) + residual * float(offset_y_m)
        sample["x"] = float(sample["x_ref_m"])
        sample["y"] = float(sample["y_ref_m"])
    return _recompute_reference_headings(result)


def _recompute_reference_headings(
    samples: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    result = [dict(sample) for sample in list(samples or [])]
    for index, sample in enumerate(result):
        if len(result) < 2:
            break
        if index + 1 < len(result):
            first = sample
            second = result[index + 1]
        else:
            first = result[index - 1]
            second = sample
        dx_m = float(second["x_ref_m"]) - float(first["x_ref_m"])
        dy_m = float(second["y_ref_m"]) - float(first["y_ref_m"])
        if math.hypot(dx_m, dy_m) > 1.0e-6:
            sample["heading_rad"] = math.atan2(dy_m, dx_m)
        sample["x"] = float(sample["x_ref_m"])
        sample["y"] = float(sample["y_ref_m"])
    return result


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _to_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)
