"""Apply cooperative route events through RouteManager's atomic lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .route_manager import LaneClosureRouteResult


@dataclass(frozen=True)
class RouteUpdateRequest:
    route_manager: Any
    ego_location: Any
    cp_payload: Optional[Mapping[str, Any]]
    stop_goal_active: bool
    lane_closure_reroute_enabled: bool
    reset_for_route_revision: Callable[..., None]


@dataclass(frozen=True)
class RouteUpdateFrame:
    lane_closure: LaneClosureRouteResult
    stop_goal_active: bool

    @property
    def route_replan_attempted(self) -> bool:
        return bool(self.lane_closure.attempted)

    @property
    def route_replan_succeeded(self) -> bool:
        return bool(
            self.lane_closure.success and self.lane_closure.route_changed
        )

    @property
    def route_replan_reason(self) -> str:
        if not self.lane_closure.attempted:
            return "route_replan_not_requested"
        return str(self.lane_closure.reason)

    def trace_fields(self) -> dict[str, object]:
        return {
            "cp_lane_closure_attempted": bool(self.lane_closure.attempted),
            "cp_lane_closure_route_changed": bool(
                self.lane_closure.route_changed
            ),
            "cp_lane_closure_reason": str(self.lane_closure.reason),
            "cp_lane_closure_handled_ids": list(
                self.lane_closure.handled_message_ids
            ),
            "cp_lane_closure_blocked_lane_ids": list(
                self.lane_closure.blocked_lane_ids
            ),
        }


class RouteUpdateStage:
    """Sequence CP route events without taking topology ownership."""

    @staticmethod
    def _no_update(reason: str) -> LaneClosureRouteResult:
        return LaneClosureRouteResult(
            attempted=False,
            success=True,
            route_changed=False,
            reason=str(reason),
        )

    def run(self, request: RouteUpdateRequest) -> RouteUpdateFrame:
        result = self._apply_lane_closures(request)
        if bool(result.route_changed):
            request.reset_for_route_revision(
                reason="cp_lane_closure_route_replanned"
            )
        stop_required = bool(
            request.stop_goal_active
            or (result.attempted and not result.success)
        )
        return RouteUpdateFrame(
            lane_closure=result,
            stop_goal_active=stop_required,
        )

    def _apply_lane_closures(
        self, request: RouteUpdateRequest
    ) -> LaneClosureRouteResult:
        if not bool(request.lane_closure_reroute_enabled):
            return self._no_update("cp_lane_closure_disabled")
        payload = dict(request.cp_payload or {})
        messages = list(
            payload.get("lane_closures", payload.get("lane_events", ())) or ()
        )
        apply_closures = getattr(
            request.route_manager, "apply_lane_closures", None
        )
        if not messages:
            return self._no_update("cp_lane_closure_no_messages")
        if not callable(apply_closures):
            return self._no_update("cp_lane_closure_unsupported")
        try:
            return apply_closures(
                messages=messages,
                start_point={
                    "x": float(request.ego_location.x),
                    "y": float(request.ego_location.y),
                    "z": float(getattr(request.ego_location, "z", 0.0)),
                },
            )
        except Exception as exc:  # RouteManager failure becomes planner data.
            return LaneClosureRouteResult(
                attempted=True,
                success=False,
                route_changed=False,
                reason="cp_lane_closure_apply_exception:" + repr(exc),
            )
