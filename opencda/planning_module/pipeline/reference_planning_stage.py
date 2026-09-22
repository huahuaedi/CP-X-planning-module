"""Reference-line lifecycle orchestration for one planning tick.

The provider remains the sole geometry owner.  This stage owns the ordering
around that provider: read the last accepted nominal trajectory, build the
baseline reference, perform the post-turn handoff, and publish the accepted
nominal trajectory.  Platform bridges only assemble immutable requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .reference_line_provider import TURN


@dataclass(frozen=True)
class BehaviorReferenceRequest:
    map_planner: Any
    local_map: Any
    ego_pose: Mapping[str, object]
    ego_state: Sequence[float]
    route_points: Sequence[Sequence[float]]
    behavior_runtime_config: Mapping[str, object]
    decision: str
    lane_change_state: str
    target_lane_id: int
    current_lane_id: int
    route_optimal_lane_id: int
    route_reference_allowed: bool
    route_reference_gate_reason: str
    in_junction: bool
    next_macro_maneuver: str
    planner_mode: str
    lookahead_m: float
    target_speed_mps: float
    ego_speed_mps: float
    horizon_steps: int
    dt_s: float
    sim_time_s: float
    stop_release_smooth_until_s: float
    authoritative_ego_waypoint: Any


@dataclass(frozen=True)
class BehaviorReferenceFrame:
    built_reference: Any
    previous_reference: tuple
    previous_target_state: Optional[tuple]

    def mutable_previous_reference(self):
        return [dict(sample) for sample in self.previous_reference]

    def mutable_previous_target_state(self):
        if self.previous_target_state is None:
            return None
        return list(self.previous_target_state)


@dataclass(frozen=True)
class PostTurnReferenceRequest:
    maneuver_manager: Any
    decision: str
    scenario_state: str
    exit_alignment_valid: bool
    exit_lateral_error_m: float
    exit_heading_error_rad: float
    local_map: Any
    ego_x_m: float
    ego_y_m: float
    current_state: Sequence[float]
    current_lane_id: int
    target_speed_mps: float
    horizon_steps: int
    dt_s: float
    route_revision: str
    map_epoch: str
    config: Mapping[str, object]
    destination_state: Sequence[float]
    reference_samples: Sequence[Mapping[str, object]]
    debug_fields: Mapping[str, object]
    reference_freeze_count: int


class ReferencePlanningStage:
    """Single orchestrator for baseline, handoff, and nominal publication."""

    def __init__(self, *, provider: Any, nominal_trajectory_generator: Any) -> None:
        self._provider = provider
        self._nominal = nominal_trajectory_generator

    def build_behavior_reference(
        self, request: BehaviorReferenceRequest
    ) -> BehaviorReferenceFrame:
        previous = self._nominal.current
        previous_target = previous.target_state()
        previous_reference = previous.mutable_samples()
        built = self._provider.build_behavior_reference(
            map_planner=request.map_planner,
            local_map=request.local_map,
            ego_pose=request.ego_pose,
            ego_state=request.ego_state,
            route_points=request.route_points,
            previous_reference=previous_reference,
            previous_target_state=previous_target,
            behavior_runtime_config=request.behavior_runtime_config,
            decision=str(request.decision),
            lane_change_state=str(request.lane_change_state),
            target_lane_id=int(request.target_lane_id),
            current_lane_id=int(request.current_lane_id),
            route_optimal_lane_id=int(request.route_optimal_lane_id),
            route_reference_allowed=bool(request.route_reference_allowed),
            route_reference_gate_reason=str(request.route_reference_gate_reason),
            in_junction=bool(request.in_junction),
            next_macro_maneuver=str(request.next_macro_maneuver),
            planner_mode=str(request.planner_mode),
            lookahead_m=float(request.lookahead_m),
            target_speed_mps=float(request.target_speed_mps),
            ego_speed_mps=float(request.ego_speed_mps),
            horizon_steps=int(request.horizon_steps),
            dt_s=float(request.dt_s),
            reference_freeze_count=int(previous.reference_freeze_count),
            sim_time_s=float(request.sim_time_s),
            stop_release_smooth_until_s=float(request.stop_release_smooth_until_s),
            authoritative_ego_waypoint=request.authoritative_ego_waypoint,
        )
        return BehaviorReferenceFrame(
            built_reference=built,
            previous_reference=tuple(dict(row) for row in previous_reference),
            previous_target_state=(
                None if previous_target is None else tuple(previous_target)
            ),
        )

    def finalize_post_turn(
        self, request: PostTurnReferenceRequest
    ) -> Any:
        result = self._provider.resolve_post_turn_reference(
            maneuver_manager=request.maneuver_manager,
            decision=str(request.decision),
            scenario_state=str(request.scenario_state),
            exit_alignment_valid=bool(request.exit_alignment_valid),
            exit_lateral_error_m=float(request.exit_lateral_error_m),
            exit_heading_error_rad=float(request.exit_heading_error_rad),
            local_map=request.local_map,
            ego_x_m=float(request.ego_x_m),
            ego_y_m=float(request.ego_y_m),
            current_state=request.current_state,
            current_lane_id=int(request.current_lane_id),
            target_speed_mps=float(request.target_speed_mps),
            horizon_steps=int(request.horizon_steps),
            dt_s=float(request.dt_s),
            route_revision=str(request.route_revision),
            map_epoch=str(request.map_epoch),
            config=request.config,
            destination_state=request.destination_state,
            reference_samples=request.reference_samples,
            debug_fields=request.debug_fields,
        )
        if bool(result.clear_turn_reference):
            self._provider.release(TURN, event="reset")
        self._nominal.update(
            target_state=result.mutable_destination_state(),
            samples=result.mutable_samples(),
            reference_freeze_count=int(request.reference_freeze_count),
            source=str(result.debug_fields.get("reference_source", "planning_tick")),
        )
        return result
