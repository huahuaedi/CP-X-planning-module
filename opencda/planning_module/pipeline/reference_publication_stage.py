"""Final reference validation and persistent-publication stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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

    def __init__(self, *, reference_pipeline: Any, reference_provider: Any):
        self._pipeline = reference_pipeline
        self._provider = reference_provider

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
    ) -> ReferencePublicationStageResult:
        pipeline_result = self._pipeline.finalize(ReferencePipelineRequest(
            destination_state=destination_state,
            reference_samples=reference_samples,
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
        return ReferencePublicationStageResult(
            destination_state=tuple(destination),
            reference_samples=tuple(samples),
            gate=gate,
            stabilizer_reason=stabilizer_reason,
            debug_fields=debug,
        )
