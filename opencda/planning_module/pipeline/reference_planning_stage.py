"""Reference-line lifecycle orchestration for one planning tick.

The provider remains the sole geometry owner.  This stage owns the ordering
around that provider: read the last accepted nominal trajectory, build the
baseline reference, perform the post-turn handoff, and publish the accepted
nominal trajectory.  Platform bridges only assemble immutable requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .candidate_selection_stage import CandidateArbitrationRequest
from .reference_line_provider import CandidateReferenceBuildContext, TURN


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
class BehaviorReferencePreparationRequest:
    """Typed stage outputs needed to build the baseline reference request."""

    map_planner: Any
    local_map: Any
    planning_context: Any
    executable_behavior: Any
    speed_frame: Any
    behavior_runtime_config: Mapping[str, object]
    planner_mode: str
    lookahead_m: float
    ego_speed_mps: float
    horizon_steps: int
    dt_s: float
    sim_time_s: float
    stop_release_smooth_until_s: float
    authoritative_ego_waypoint: Any


@dataclass(frozen=True)
class PreparedBehaviorReference:
    request: BehaviorReferenceRequest
    frame: BehaviorReferenceFrame

    @property
    def built_reference(self):
        return self.frame.built_reference


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


@dataclass(frozen=True)
class CandidatePlanningRequest:
    """Inputs that join a built baseline with candidate arbitration.

    The nested baseline request/frame avoid reconstructing reference-builder
    state in the runtime bridge.  CandidateSelectionStage still owns scoring,
    probing, and commitment; this request only defines their typed boundary.
    """

    baseline_request: BehaviorReferenceRequest
    baseline_frame: BehaviorReferenceFrame
    baseline_destination_state: Sequence[float]
    baseline_reference: Sequence[Mapping[str, object]]
    baseline_debug: Mapping[str, object]
    planner_config: Mapping[str, object]
    selected_decision: str
    selected_target_lane_id: int
    current_lane_id: int
    target_speed_mps: float
    candidate_lane_ids: Sequence[int]
    lane_safety_scores: Mapping[int, float]
    lane_prediction_risks: Mapping[int, Mapping[str, object]]
    stop_goal_active: bool
    traffic_stop_active: bool
    lane_change_authorization: Any
    opportunistic_lane_change_allowed: bool
    stop_target: Any
    local_obstacle_avoidance_active: bool
    current_state: Sequence[float]
    ego_location: Any
    ego_yaw_rad: float
    object_snapshots: Sequence[Mapping[str, object]]
    prediction_trajectories: Mapping[str, object]
    current_acceleration_mps2: float
    current_steering_rad: float
    route_required: bool
    scenario_stop_required: bool
    speed_plan: Any
    turn_prepare_speed_suppressed: bool
    cooperative_lane_change_deferred: bool
    lane_change_mpc_stall_failure_count: int
    route_revision: str
    map_epoch: str
    upcoming_turn_direction: str
    upcoming_turn_distance_m: float
    lane_change_duration_s: float
    lane_change_duration_reason: str
    lane_width_m: float
    validate_contract: Callable[..., Any]


class ReferencePlanningStage:
    """Single orchestrator for baseline, handoff, and nominal publication."""

    def __init__(
        self, *, provider: Any, nominal_trajectory_generator: Any,
        candidate_selection: Any,
    ) -> None:
        self._provider = provider
        self._nominal = nominal_trajectory_generator
        self._candidate_selection = candidate_selection

    @property
    def candidate_selection(self) -> Any:
        """Candidate owner nested under the reference-planning lifecycle."""

        return self._candidate_selection

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

    def prepare_behavior_reference(
        self, request: BehaviorReferencePreparationRequest
    ) -> PreparedBehaviorReference:
        """Build a baseline from already-resolved typed planning stages."""

        planning = request.planning_context
        adapter = planning.adapter_output
        frame = planning.planner_input_frame
        executable = request.executable_behavior
        speed = request.speed_frame
        baseline_request = BehaviorReferenceRequest(
            map_planner=request.map_planner,
            local_map=request.local_map,
            ego_pose=adapter.ego_pose,
            ego_state=adapter.current_state,
            route_points=tuple(adapter.route_points),
            behavior_runtime_config=request.behavior_runtime_config,
            decision=str(executable.decision),
            lane_change_state=str(executable.phase),
            target_lane_id=int(executable.target_lane_id),
            current_lane_id=int(planning.current_lane_id),
            route_optimal_lane_id=int(adapter.route_optimal_lane_id),
            route_reference_allowed=bool(adapter.route_reference_allowed),
            route_reference_gate_reason=str(
                adapter.route_reference_gate_reason
            ),
            in_junction=bool(frame.map_lane.in_junction),
            next_macro_maneuver=str(
                frame.planning.route.next_macro_maneuver
            ),
            planner_mode=str(request.planner_mode),
            lookahead_m=float(request.lookahead_m),
            target_speed_mps=float(speed.target_speed_mps),
            ego_speed_mps=float(request.ego_speed_mps),
            horizon_steps=int(request.horizon_steps),
            dt_s=float(request.dt_s),
            sim_time_s=float(request.sim_time_s),
            stop_release_smooth_until_s=float(
                request.stop_release_smooth_until_s
            ),
            authoritative_ego_waypoint=request.authoritative_ego_waypoint,
        )
        return PreparedBehaviorReference(
            request=baseline_request,
            frame=self.build_behavior_reference(baseline_request),
        )

    def arbitrate_candidates(self, request: CandidatePlanningRequest) -> Any:
        baseline = request.baseline_request
        previous_target = request.baseline_frame.mutable_previous_target_state()
        reference_context = CandidateReferenceBuildContext(
            map_planner=baseline.map_planner,
            local_map=baseline.local_map,
            planner_config=request.planner_config,
            ego_pose=baseline.ego_pose,
            current_state=request.current_state,
            ego_location=request.ego_location,
            ego_yaw_rad=float(request.ego_yaw_rad),
            ego_speed_mps=float(baseline.ego_speed_mps),
            route_points=baseline.route_points,
            previous_reference=(
                request.baseline_frame.mutable_previous_reference()
            ),
            previous_target_state=list(previous_target or []),
            behavior_runtime_config=baseline.behavior_runtime_config,
            baseline_decision=str(request.selected_decision),
            baseline_target_lane_id=int(request.selected_target_lane_id),
            baseline_speed_mps=float(request.target_speed_mps),
            baseline_destination_state=request.baseline_destination_state,
            baseline_reference=request.baseline_reference,
            baseline_debug=request.baseline_debug,
            current_lane_id=int(request.current_lane_id),
            route_optimal_lane_id=int(baseline.route_optimal_lane_id),
            route_reference_allowed=bool(baseline.route_reference_allowed),
            route_reference_gate_reason=str(
                baseline.route_reference_gate_reason
            ),
            in_junction=bool(baseline.in_junction),
            next_macro_maneuver=str(baseline.next_macro_maneuver),
            planner_mode=str(baseline.planner_mode),
            lookahead_m=float(baseline.lookahead_m),
            horizon_steps=int(baseline.horizon_steps),
            dt_s=float(baseline.dt_s),
            reference_freeze_count=int(
                request.baseline_frame.built_reference.reference_freeze_count
            ),
            sim_time_s=float(baseline.sim_time_s),
            stop_release_smooth_until_s=float(
                baseline.stop_release_smooth_until_s
            ),
            authoritative_ego_waypoint=baseline.authoritative_ego_waypoint,
            route_revision=str(request.route_revision),
            map_epoch=str(request.map_epoch),
            upcoming_turn_direction=str(request.upcoming_turn_direction),
            upcoming_turn_distance_m=float(request.upcoming_turn_distance_m),
            lane_change_duration_s=float(request.lane_change_duration_s),
            lane_change_duration_reason=str(request.lane_change_duration_reason),
            lane_width_m=float(request.lane_width_m),
        )
        return self._candidate_selection.arbitrate(
            CandidateArbitrationRequest(
                reference_context=reference_context,
                selected_decision=str(request.selected_decision),
                selected_target_lane_id=int(request.selected_target_lane_id),
                current_lane_id=int(request.current_lane_id),
                target_speed_mps=float(request.target_speed_mps),
                candidate_lane_ids=request.candidate_lane_ids,
                lane_safety_scores=request.lane_safety_scores,
                lane_prediction_risks=request.lane_prediction_risks,
                stop_goal_active=bool(request.stop_goal_active),
                traffic_stop_active=bool(request.traffic_stop_active),
                lane_change_authorization=request.lane_change_authorization,
                opportunistic_lane_change_allowed=bool(
                    request.opportunistic_lane_change_allowed
                ),
                stop_target=request.stop_target,
                local_obstacle_avoidance_active=bool(
                    request.local_obstacle_avoidance_active
                ),
                ego_speed_mps=float(baseline.ego_speed_mps),
                lane_width_m=float(request.lane_width_m),
                baseline_lane_change_state=str(baseline.lane_change_state),
                current_state=request.current_state,
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                object_snapshots=request.object_snapshots,
                prediction_trajectories=request.prediction_trajectories,
                current_acceleration_mps2=float(
                    request.current_acceleration_mps2
                ),
                current_steering_rad=float(request.current_steering_rad),
                route_required=bool(request.route_required),
                scenario_stop_required=bool(request.scenario_stop_required),
                speed_plan=request.speed_plan,
                turn_prepare_speed_suppressed=bool(
                    request.turn_prepare_speed_suppressed
                ),
                cooperative_lane_change_deferred=bool(
                    request.cooperative_lane_change_deferred
                ),
                lane_change_mpc_stall_failure_count=int(
                    request.lane_change_mpc_stall_failure_count
                ),
            ),
            sim_time_s=float(baseline.sim_time_s),
            route_revision=str(request.route_revision),
            validate_contract=request.validate_contract,
        )

    def cooperative_conflict_reference(self, **kwargs: Any) -> Any:
        return self._candidate_selection.cooperative_conflict_reference(**kwargs)

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
