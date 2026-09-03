"""Typed destination-speed lifecycle stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class DestinationApplicationResult:
    stage: DestinationSpeedStageResult
    destination_state: Tuple[float, ...]
    reference_samples: Tuple[Mapping[str, Any], ...]
    behavior_stage_result: Any
    reference_debug: Mapping[str, Any]
    finished: bool

    def mutable_destination_state(self):
        return list(self.destination_state)

    def mutable_reference(self):
        return [dict(sample) for sample in self.reference_samples]

    def mutable_reference_debug(self):
        return dict(self.reference_debug)


class DestinationSpeedStage:
    """Own destination approach and terminal-stop state across ticks."""

    def __init__(
        self,
        *,
        config: Mapping[str, object],
        speed_planner: SpeedTargetPlanner,
        fallback_manager: Any = None,
        behavior_stage: Any = None,
    ) -> None:
        self._config = dict(config)
        self._speed_planner = speed_planner
        self._fallback = fallback_manager
        self._behavior = behavior_stage
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

    def apply(
        self,
        *,
        route_status: Any,
        route_revision: str,
        ego_speed_mps: float,
        current_state: Sequence[float],
        destination_state: Sequence[float],
        reference_samples: Sequence[Mapping[str, Any]],
        behavior_stage_result: Any,
        reference_debug: Mapping[str, Any],
        fallback_lane_id: int,
    ) -> DestinationApplicationResult:
        """Apply destination approach/stop exactly once after behavior planning."""

        stage = self.evaluate(
            route_status=route_status,
            route_revision=route_revision,
            ego_speed_mps=ego_speed_mps,
        )
        destination = list(destination_state or ())
        reference = [dict(sample) for sample in reference_samples or ()]
        debug = dict(reference_debug or {})
        behavior = behavior_stage_result
        if stage.approach_active and not stage.reached_destination:
            debug.update(stage.trace_fields())
        if stage.stop_latched:
            if self._fallback is None or self._behavior is None:
                raise RuntimeError("destination stop dependencies are not configured")
            stop = self._fallback.bounded_safe_stop(
                current_speed_mps=float(ego_speed_mps),
                current_reference=reference,
                reason=str(stage.reason),
            )
            stop_reference = stop.mutable_trajectory()
            if len(stop_reference) >= 2:
                reference = stop_reference
                terminal = dict(reference[-1])
                diagnostics = behavior.mutable_diagnostics()
                destination = [
                    float(terminal.get("x_ref_m", terminal.get("x", current_state[0]))),
                    float(terminal.get("y_ref_m", terminal.get("y", current_state[1]))),
                    0.0,
                    float(terminal.get("heading_rad", current_state[3])),
                    int(diagnostics.get("target_lane_id", fallback_lane_id) or 0),
                ]
            behavior = self._behavior.destination_stop(behavior)
            debug.update({
                "reference_source": "persistent_bounded_safe_stop",
                "final_reference_geometry_source": "persistent_bounded_safe_stop",
                **stage.trace_fields(),
                "destination_stop_reason": str(stop.reason),
            })
        return DestinationApplicationResult(
            stage=stage,
            destination_state=tuple(destination),
            reference_samples=tuple(dict(sample) for sample in reference),
            behavior_stage_result=behavior,
            reference_debug=debug,
            finished=bool(
                stage.stop_latched
                and float(ego_speed_mps) <= float(
                    self._config.get("destination_stop_complete_speed_mps", 0.15)
                )
            ),
        )
