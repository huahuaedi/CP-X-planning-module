"""Route lifecycle manager for the CP-X OpenCDA pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


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
        reached_distance_m: float = 3.0,
        stale_route_lateral_m: float = 12.0,
    ) -> None:
        self.global_planner = global_planner
        self.reached_distance_m = max(0.1, float(reached_distance_m))
        self.stale_route_lateral_m = max(1.0, float(stale_route_lateral_m))
        self._active_route_summary = None
        self._start_point: Optional[Dict[str, float]] = None
        self._goal_point: Optional[Dict[str, float]] = None
        self._fallback_route_points: List[List[float]] = []
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

    @property
    def last_status(self) -> RouteManagerStatus:
        return self._last_status

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


def _to_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)
