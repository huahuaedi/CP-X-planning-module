"""Final reference validation and persistent-publication stage."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

from .behavior_decision import BehaviorDecision
from .reference_line_provider import ReferenceLineRequest
from .reference_pipeline import ReferencePipelineRequest


@dataclass(frozen=True)
class ReferencePublicationStageResult:
    destination_state: tuple
    reference_samples: tuple
    gate: Any
    stabilizer_reason: str
    debug_fields: Mapping[str, object]

    def mutable_destination(self):
        return list(self.destination_state)

    def mutable_samples(self):
        return [dict(sample) for sample in self.reference_samples]


class ReferencePublicationStage:
    """Sole bridge-facing boundary that may publish a reference to MPC."""

    def __init__(self, *, reference_pipeline: Any, reference_provider: Any,
                 config: Optional[Mapping[str, object]] = None):
        self._pipeline = reference_pipeline
        self._provider = reference_provider
        self._config = dict(config or {})

    def run(
        self,
        *,
        destination_state: Sequence[float],
        reference_samples: Sequence[Mapping[str, object]],
        current_state: Sequence[float],
        ego_location: Any,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        target_speed_mps: float,
        behavior: BehaviorDecision,
        stop_goal_active: bool,
        route_points: Sequence[Sequence[float]],
        local_map: Any,
        route_cursor: Any,
        route_revision: str,
        map_epoch: str,
        reference_source: str,
        candidate_status: str = "",
        candidate_reason: str = "",
        heading_error_rad: float = float("nan"),
    ) -> ReferencePublicationStageResult:
        destination = list(destination_state)
        samples = [dict(sample) for sample in reference_samples]
        prepublish_debug: dict[str, object] = {}
        if (
            bool(self._config.get("boundary_recovery_enabled", False))
            and bool(behavior.boundary_recovery_active)
        ):
            generated, recovery, conditioning = (
                self._provider.boundary_recovery_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    current_lane_id=int(behavior.source_lane_id),
                    base_reference_samples=samples,
                    target_speed_mps=float(target_speed_mps),
                    horizon_steps=int(self._pipeline.horizon_steps),
                    dt_s=float(self._pipeline.dt_s),
                    max_curvature_1pm=float(self._config.get(
                        "boundary_recovery_max_curvature_1pm",
                        self._config.get("reference_vehicle_max_curvature_1pm", 0.22),
                    )),
                )
            )
            prepublish_debug.update({
                "boundary_recovery_active": True,
                "boundary_recovery_generation_reason": str(generated.reason),
                "boundary_recovery_conditioning_reason": str(conditioning),
            })
            if recovery:
                samples = [dict(sample) for sample in recovery]
                terminal = samples[-1]
                destination = [
                    float(terminal.get("x_ref_m", terminal.get("x", ego_location.x))),
                    float(terminal.get("y_ref_m", terminal.get("y", ego_location.y))),
                    float(target_speed_mps),
                    float(terminal.get("heading_rad", ego_yaw_rad)),
                    int(terminal.get("lane_id", behavior.source_lane_id)
                        or behavior.source_lane_id),
                ]
                reference_source = "ego_anchored_boundary_recovery"
                prepublish_debug.update({
                    "reference_source": reference_source,
                    "final_reference_geometry_source": reference_source,
                    "stage": "boundary_recovery_reference",
                    "intent_mode": "boundary_recovery",
                })
            else:
                candidate_status = "infeasible"
                candidate_reason = (
                    "boundary_recovery_reference_generation_failed:"
                    + str(generated.reason)
                )
        pipeline_result = self._pipeline.finalize(ReferencePipelineRequest(
            destination_state=destination,
            reference_samples=samples,
            current_state=current_state,
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            target_speed_mps=float(target_speed_mps),
            behavior_decision=str(behavior.maneuver),
            behavior_fsm_state=str(behavior.phase),
            current_lane_id=int(behavior.source_lane_id),
            target_lane_id=int(behavior.target_lane_id),
            stop_goal_active=bool(stop_goal_active),
            stop_target=behavior.stop_target,
            route_points=route_points,
        ))
        destination = list(pipeline_result.destination_state)
        samples = [dict(sample) for sample in pipeline_result.reference_samples]
        gate = pipeline_result.gate
        stabilizer_reason = str(pipeline_result.conditioning_reason)
        debug = dict(pipeline_result.as_debug_fields())
        debug.update(prepublish_debug)
        publication = self._provider.publish(
            ReferenceLineRequest(
                local_map=local_map,
                route_cursor=route_cursor,
                behavior=behavior,
                route_revision=str(route_revision),
                map_epoch=str(map_epoch),
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
            ),
            samples,
            valid=bool(gate.accepted),
            validation_reason=str(gate.reason),
            build_reason=str(reference_source or "planning_reference"),
        )
        debug.update({
            "reference_provider_mode": str(publication.mode),
            "reference_provider_reason": str(publication.reason),
            "reference_provider_geometry_revision": int(
                publication.geometry_revision
            ),
        })
        if publication.accepted:
            samples = publication.mutable_samples()
        if not gate.accepted:
            gate_reason = "final_reference_gate:" + str(gate.reason)
            stabilizer_reason = ";".join(
                value for value in (stabilizer_reason, gate_reason) if value
            )
            debug["candidate_pipeline_selected_status"] = "infeasible"
            debug["candidate_pipeline_selected_reason"] = ";".join(
                value for value in (str(candidate_reason), gate_reason) if value
            )
        debug["mpc_reference_stabilizer_reason"] = stabilizer_reason
        debug["final_reference_geometry_source"] = str(
            reference_source or "unknown"
        )
        guard_reason = self._lateral_guard_reason(
            behavior=behavior,
            stop_goal_active=bool(stop_goal_active),
            destination_state=destination,
            reference_samples=samples,
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            heading_error_rad=float(heading_error_rad),
        )
        debug["reference_lateral_guard_reason"] = guard_reason
        if guard_reason:
            debug["lateral_guard_validation"] = "warning"
        return ReferencePublicationStageResult(
            destination_state=tuple(destination),
            reference_samples=tuple(samples),
            gate=gate,
            stabilizer_reason=stabilizer_reason,
            debug_fields=debug,
        )

    def _lateral_guard_reason(
        self, *, behavior: BehaviorDecision, stop_goal_active: bool,
        destination_state: Sequence[float],
        reference_samples: Sequence[Mapping[str, object]], ego_location: Any,
        ego_yaw_rad: float, heading_error_rad: float,
    ) -> str:
        decision = str(behavior.maneuver or "").strip().lower()
        phase = str(behavior.phase or "").strip().upper()
        lane_follow = decision == "lane_follow" and phase in {
            "", "IDLE", "LANE_KEEP",
        }
        stop = bool(stop_goal_active) or decision in {
            "stop_at_intersection", "stop_sign",
        }
        if not (lane_follow or stop) or bool(behavior.boundary_recovery_active):
            return ""

        prefix = "full_stop" if stop else "full_lane_follow"
        destination_limit = float(self._config.get(
            prefix + "_max_destination_lateral_m", 1.0 if stop else 1.2,
        ))
        first_limit = float(self._config.get(
            prefix + "_max_reference_first_lateral_m", 0.55 if stop else 0.65,
        ))
        heading_limit = float(self._config.get(
            prefix + "_max_heading_error_deg", 4.0,
        ))
        reasons: list[str] = []
        if len(destination_state) >= 2:
            lateral = self._body_lateral(
                ego_location, ego_yaw_rad,
                float(destination_state[0]), float(destination_state[1]),
            )
            if abs(lateral) > destination_limit:
                reasons.append("dest_lat=%.2f" % lateral)
        if reference_samples:
            first = reference_samples[0]
            lateral = self._body_lateral(
                ego_location, ego_yaw_rad,
                float(first.get("x_ref_m", first.get("x", ego_location.x))),
                float(first.get("y_ref_m", first.get("y", ego_location.y))),
            )
            if abs(lateral) > first_limit:
                reasons.append("ref_lat=%.2f" % lateral)
        if math.isfinite(heading_error_rad):
            heading_deg = math.degrees(heading_error_rad)
            if abs(heading_deg) > heading_limit:
                reasons.append("heading=%.2f" % heading_deg)
        if not reasons:
            return ""
        return ("stop" if stop else "lane_follow") + "_lateral_guard:" + ":".join(reasons)

    @staticmethod
    def _body_lateral(ego_location: Any, yaw_rad: float,
                      target_x_m: float, target_y_m: float) -> float:
        dx = float(target_x_m) - float(ego_location.x)
        dy = float(target_y_m) - float(ego_location.y)
        return -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
