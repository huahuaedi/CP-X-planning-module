"""Typed destination-speed lifecycle stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .speed_planner import SpeedConstraint, SpeedTargetPlanner


@dataclass(frozen=True)
class DestinationSpeedStageResult:
    constraint: Optional[SpeedConstraint]
    route_found: bool
    reached_destination: bool
    approach_active: bool
    stop_latched: bool
    remaining_distance_m: float
    required_distance_m: float
    stop_buffer_m: float
    reason: str

    def trace_fields(self) -> dict[str, object]:
        return {
            "destination_stop_latched": bool(self.stop_latched),
            "destination_stop_reason": str(self.reason),
            "destination_stop_remaining_distance_m": float(
                self.remaining_distance_m
            ),
            "destination_stop_required_distance_m": float(
                self.required_distance_m
            ),
        }


class DestinationSpeedStage:
    """Own destination approach and terminal-stop state across ticks."""

    def __init__(
        self,
        *,
        config: Mapping[str, object],
        speed_planner: SpeedTargetPlanner,
    ) -> None:
        self._config = dict(config)
        self._speed_planner = speed_planner
        self._stop_latched = False
        self._route_revision = ""

    @property
    def stop_latched(self) -> bool:
        return bool(self._stop_latched)

    def evaluate(
        self,
        *,
        route_status: Any,
        route_revision: str,
        ego_speed_mps: float,
    ) -> DestinationSpeedStageResult:
        revision = str(route_revision or "")
        if revision != self._route_revision:
            self._route_revision = revision
            self._stop_latched = False
        route_found = bool(getattr(route_status, "route_found", False))
        reached = bool(
            route_status and getattr(route_status, "reached_destination", False)
        )
        remaining_m = float(
            getattr(route_status, "remaining_distance_m", float("inf"))
            if route_status is not None
            else float("inf")
        )
        deceleration_mps2 = max(
            0.5,
            float(self._config.get("fallback_safe_stop_deceleration_mps2", 2.0)),
        )
        buffer_m = max(
            0.0, float(self._config.get("destination_stop_buffer_m", 1.5))
        )
        constraint, approach_active, required_m = (
            self._speed_planner.destination_approach_constraint(
                route_revision=revision,
                route_found=route_found,
                route_reached_destination=reached,
                remaining_distance_m=remaining_m,
                ego_speed_mps=float(ego_speed_mps),
                deceleration_mps2=deceleration_mps2,
                buffer_m=buffer_m,
            )
        )
        self._stop_latched = bool(
            self._stop_latched
            or reached
            or (approach_active and remaining_m <= buffer_m)
        )
        reason = ""
        if self._stop_latched:
            reason = (
                "route_destination_reached"
                if reached
                else "route_destination_approach"
            )
        elif approach_active:
            reason = "route_destination_approach_speed_profile"
        return DestinationSpeedStageResult(
            constraint=constraint,
            route_found=route_found,
            reached_destination=reached,
            approach_active=bool(approach_active),
            stop_latched=bool(self._stop_latched),
            remaining_distance_m=remaining_m,
            required_distance_m=float(required_m),
            stop_buffer_m=buffer_m,
            reason=reason,
        )
