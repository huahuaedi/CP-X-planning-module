"""Execute behavior/reference planning with one typed recovery path."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any, Callable, Mapping, Sequence, Tuple

from .fallback_manager import FailureReason


@dataclass(frozen=True)
class BehaviorReferenceRequest:
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    requested_speed_mps: float
    object_snapshots: Sequence[Mapping[str, Any]]
    stop_goal_active: bool
    cp_payload: Mapping[str, Any]
    current_state: Sequence[float]
    sim_time_s: float
    route_revision: str


@dataclass(frozen=True)
class BehaviorReferenceResult:
    destination_state: Sequence[float]
    reference_samples: Tuple[Mapping[str, Any], ...]
    behavior_stage_result: Any
    reference_debug: Mapping[str, Any]
    speed_plan: Any
    failure_reason: str = ""


class BehaviorReferenceExecutionStage:
    """Own the behavior/reference exception boundary and degradation policy."""

    def __init__(
        self,
        *,
        reference_provider: Any,
        fallback_manager: Any,
        behavior_stage: Any,
        lane_id_at_location: Callable[[Any], int],
    ) -> None:
        self._reference_provider = reference_provider
        self._fallback = fallback_manager
        self._behavior = behavior_stage
        self._lane_id_at_location = lane_id_at_location

    def run(
        self,
        request: BehaviorReferenceRequest,
        *,
        planner: Callable[..., tuple],
    ) -> BehaviorReferenceResult:
        try:
            destination, reference, behavior, debug, speed_plan = planner(
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                ego_speed_mps=float(request.ego_speed_mps),
                speed_ref_mps=float(request.requested_speed_mps),
                object_snapshots=request.object_snapshots,
                stop_goal_active=bool(request.stop_goal_active),
                cp_payload=request.cp_payload,
            )
            return BehaviorReferenceResult(
                destination_state=list(destination or []),
                reference_samples=tuple(dict(item) for item in reference or ()),
                behavior_stage_result=behavior,
                reference_debug=dict(debug or {}),
                speed_plan=speed_plan,
            )
        except Exception as exc:
            trace = traceback.format_exc(limit=8).strip()
            generated = self._reference_provider.lane_fallback_reference(
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                current_state=list(request.current_state),
                speed_ref_mps=float(request.requested_speed_mps),
            )
            failure = FailureReason(
                stage="behavior_reference",
                code="pipeline_exception",
                severity="degraded",
                recoverable=True,
                details=str(exc),
            )
            fallback = self._fallback.resolve(
                sim_time_s=float(request.sim_time_s),
                route_revision=str(request.route_revision),
                current_speed_mps=float(request.ego_speed_mps),
                current_reference=list(generated.samples or []),
                failure_reason=failure,
            )
            reference = fallback.mutable_trajectory()
            destination = list(generated.destination_state or [])
            lane_id = int(self._lane_id_at_location(request.ego_location))
            if reference:
                terminal = dict(reference[-1])
                destination = [
                    float(terminal.get("x_ref_m", terminal.get("x", request.current_state[0]))),
                    float(terminal.get("y_ref_m", terminal.get("y", request.current_state[1]))),
                    float(fallback.target_speed_mps),
                    float(terminal.get("heading_rad", request.current_state[3])),
                    lane_id,
                ]
            fallback_stop = bool(
                str(fallback.mode) == "bounded_safe_stop"
                or float(fallback.target_speed_mps) <= 0.0
            )
            behavior = self._behavior.finalize(
                maneuver="lane_follow",
                phase="FALLBACK",
                source_lane_id=lane_id,
                target_lane_id=0,
                requested_speed_mps=float(fallback.target_speed_mps),
                stop_required=fallback_stop,
                route_required=False,
                traffic_signal_state="unknown",
                boundary_recovery_active=False,
                stop_target=None,
                reason=str(fallback.reason),
                diagnostics={
                    "pipeline_error": str(exc),
                    "pipeline_error_traceback": trace,
                },
            )
            debug = {
                "reference_source": "fallback_manager:" + str(fallback.mode),
                "fallback_reason": str(fallback.reason),
                "pipeline_error": str(exc),
                "pipeline_error_traceback": trace,
            }
            return BehaviorReferenceResult(
                destination_state=destination,
                reference_samples=tuple(dict(item) for item in reference),
                behavior_stage_result=behavior,
                reference_debug=debug,
                speed_plan=None,
                failure_reason=str(exc),
            )
