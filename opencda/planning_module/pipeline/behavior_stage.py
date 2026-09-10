"""Typed behavior-stage output boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from .behavior_decision import BehaviorDecision
from .cooperative_maneuver_proposal import CooperativeManeuverProposal
from .reference_line_provider import LANE_CHANGE
from .candidate_evaluation import evaluate_behavior_candidates
from .route_authorization import (
    RouteLaneChangeAuthorizationLatch,
    authorize_opportunistic_lane_change,
    authorize_route_lane_change,
    lane_change_target_reached,
)


@dataclass(frozen=True)
class RouteLaneChangeRequest:
    route_lane_change_allowed: bool
    current_lane_id: int
    route_required_lane_id: int
    next_macro_maneuver: str
    current_road_option: str
    remaining_distance_m: float
    available_lane_ids: Sequence[int]
    lane_safety_scores: Mapping[int, float]
    lane_prediction_risks: Mapping[int, Mapping[str, object]]
    preparation_start_distance_m: float
    latest_start_distance_m: float
    target_safety_threshold: float
    require_adjacent: bool
    explicit_lane_change_start_distance_m: float
    adjacent_lane_directions: Mapping[int, str]
    topology_current_lane_id: int
    topology_target_lane_id: int
    topology_lane_offset: int
    topology_target_in_local_frame: bool
    route_geometry_direction: str
    route_geometry_distance_m: Optional[float]


@dataclass(frozen=True)
class RouteLaneChangeContext:
    """Topology-derived lane-change facts consumed by behavior planning."""

    request: RouteLaneChangeRequest
    topology_current_lane_id: int
    topology_target_lane_id: int
    topology_lane_offset: int
    topology_target_in_local_frame: bool
    physical_target_lane_id: int
    physical_direction: str
    physical_direction_reason: str
    preparation_start_distance_m: float
    geometry_direction: str
    geometry_distance_m: float
    geometry_reason: str
    edge_id: Optional[str]


@dataclass(frozen=True)
class RouteLaneChangeStageResult:
    """Resolved route authorization and its lifecycle side effect request."""

    context: RouteLaneChangeContext
    authorization: Any
    replan_reason: str = ""


@dataclass(frozen=True)
class RouteBehaviorContextResult:
    """Route topology interpretation for one immutable planning frame."""

    context: RouteLaneChangeContext
    authorization: Any
    replan_reason: str
    route_lane_change_allowed: bool


@dataclass(frozen=True)
class TurnScenarioContext:
    """Route-derived turn facts consumed by ScenarioManager."""

    direction: str
    distance_m: float
    reason: str
    route_advanced_to_lane_change: bool
    exit_heading_error_rad: float
    exit_lateral_m: float
    exit_alignment_reason: str
    exit_alignment_valid: bool
    exit_aligned: bool


@dataclass(frozen=True)
class LateralOwnershipResult:
    """One arbitration result for lane-change versus turn ownership."""

    authorization: Any
    start_transition: Any
    handoff: Any
    reference_release_event: str
    geometry_arc_m: float
    operational_curvature_1pm: float


@dataclass(frozen=True)
class OpportunisticLaneChangeRequest:
    enabled: bool
    sim_time_s: float
    start_lock_until_s: float
    dense_traffic_lock_enabled: bool
    object_count: int
    dense_object_count: int
    lane_prediction_risks: Mapping[int, Mapping[str, object]]
    dense_risky_lane_count: int


@dataclass(frozen=True)
class ConflictResolutionRequest:
    """Frozen inputs to the lateral/cooperative arbitration seam."""

    route_authorization: Any
    opportunistic_request: OpportunisticLaneChangeRequest
    owner_state: str
    ego_speed_mps: float
    planning_speed_mps: float
    lane_change_duration_s: float
    dt_s: float
    lane_width_m: float
    distance_to_turn_m: float
    config: Mapping[str, object]
    ego_location: Any
    ego_yaw_rad: float

@dataclass(frozen=True)
class ConflictResolutionResult:
    """One authoritative answer for all lane-change ownership conflicts."""

    authorization: Any
    lateral_ownership: LateralOwnershipResult
    opportunistic_allowed: bool
    lane_change_gate_reason: str


@dataclass(frozen=True)
class BehaviorCandidateRequest:
    lane_safety_scores: Mapping[int, float]
    lane_prediction_risks: Mapping[int, Mapping[str, object]]
    ego_lane_id: int
    available_lane_ids: Sequence[int]
    route_optimal_lane_id: int
    in_junction: bool
    mpc_feedback_blocked_lane_ids: Sequence[int]
    mpc_feedback_weight: float
    nearest_front_obstacles_by_lane: Mapping[int, Mapping[str, object]]
    desired_speed_mps: float
    progress_cost_weight: float


@dataclass(frozen=True)
class BehaviorOverrideRequest:
    decision: str
    target_lane_id: int
    phase: str
    current_lane_id: int
    lane_change_authorized: bool
    opportunistic_lane_change_allowed: bool
    lane_change_gate_reason: str
    lane_change_authorization_reason: str
    prepare_reference_lock: bool
    scenario_speed_cap_active: bool
    scenario_reason: str
    scenario_override_decision: str
    scenario_override_phase: str
    scenario_stop_required: bool
    local_avoidance_active: bool
    ego_in_junction: bool
    lane_change_commitment_active: bool
    route_turn_decision: str
    route_current_road_option: str
    stop_goal_active: bool


@dataclass(frozen=True)
class BehaviorOverrideResult:
    decision: str
    target_lane_id: int
    phase: str
    stop_goal_active: bool
    reason: str
    reset_lane_change_reason: str = ""


@dataclass(frozen=True)
class BehaviorStageResult:
    decision: BehaviorDecision
    diagnostics: Mapping[str, object]

    def mutable_diagnostics(self) -> dict[str, object]:
        return dict(self.diagnostics)


@dataclass(frozen=True)
class BehaviorCommandResult:
    """BehaviorPlanner command plus the observations that produced it."""

    decision: str
    target_lane_id: int
    phase: str
    lane_alignment_valid: bool
    lane_lateral_error_m: float
    lane_heading_error_rad: float
    traffic_control_stop_active: bool
    static_obstacle_result: Any
    opportunistic_lane_change_allowed: bool
    preferred_target_lane_id: int


@dataclass(frozen=True)
class BehaviorCommandFrameRequest:
    adapter_output: Any
    ego_pose: Any
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    current_lane_id: int
    target_speed_mps: float
    sim_time_s: float
    route_optimal_lane_id: int
    route_next_macro_maneuver: str
    route_points: Sequence
    front_distance_by_lane: Mapping[int, float]
    lane_safety_scores: Mapping[int, float]
    object_snapshots: Sequence[Mapping[str, object]]
    lane_change_authorization: Any
    opportunistic_lane_change_allowed: bool
    behavior_traffic_state: str
    behavior_stop_target: Optional[Mapping[str, object]]
    signal_context: Mapping[str, object]
    scenario_stop_required: bool
    lane_change_reference_active: bool
    mpc_feedback: Mapping[str, object]
    max_deceleration_mps2: float
    config: Mapping[str, object]
    runtime_config: Mapping[str, object]


@dataclass(frozen=True)
class BehaviorCommandFrameResult:
    command: BehaviorCommandResult
    candidate_frame: Any
    nearest_front_obstacles_by_lane: Mapping[int, Mapping[str, object]]
    candidate_lane_ids: tuple
    cooperative_proposal: CooperativeManeuverProposal


class BehaviorStage:
    """Sole constructor and transition owner for BehaviorDecision."""

    def __init__(self) -> None:
        self._route_lane_change_latch = RouteLaneChangeAuthorizationLatch()

    def reset_route_lane_change_authorization(self) -> None:
        self._route_lane_change_latch.reset()

    @staticmethod
    def _cooperative_proposal_from_candidate(
        *, authorization: Any, candidate_frame: Any, current_lane_id: int,
        opportunistic_lane_change_allowed: bool,
    ) -> CooperativeManeuverProposal:
        """Preserve the pre-safety maneuver request for peer negotiation."""

        if bool(authorization.allowed):
            maneuver = str(authorization.maneuver)
            target_lane_id = int(authorization.target_lane_id)
            route_required = bool(authorization.required_by_route)
            reason = str(authorization.reason)
        elif bool(opportunistic_lane_change_allowed):
            selected = candidate_frame.selected
            maneuver = str(selected.decision)
            target_lane_id = int(selected.target_lane_id)
            route_required = False
            reason = str(selected.reason)
        else:
            maneuver = "lane_follow"
            target_lane_id = int(current_lane_id)
            route_required = False
            reason = "lane_change_not_requested"
        return CooperativeManeuverProposal.from_behavior(
            maneuver=maneuver,
            source_corridor_id=int(current_lane_id),
            target_corridor_id=int(target_lane_id),
            route_required=bool(route_required),
            maneuver_active=False,
            committed_at_s=0.0,
            reason=str(reason),
        )

    def produce_command_from_frame(
        self, request: BehaviorCommandFrameRequest, *, behavior_planner: Any,
        static_obstacle_stage: Any, reference_map: Any,
        nearest_front_obstacles: Any, attempt_replan: Any,
        object_track_id: Any,
    ) -> BehaviorCommandFrameResult:
        """Evaluate lane candidates and produce one command from a frozen frame."""

        frame = request.adapter_output.frame
        available = tuple(frame.map_lane.allowed_lane_ids)
        nearest = nearest_front_obstacles(
            ego_snapshot={
                "x": float(request.ego_location.x),
                "y": float(request.ego_location.y),
                "psi": float(request.ego_yaw_rad),
            },
            obstacle_snapshots=request.object_snapshots,
            lane_assignments=dict(frame.prediction.lane_assignments or {}),
            available_lane_ids=available,
        )
        authorization = request.lane_change_authorization
        if bool(authorization.allowed):
            candidate_lane_ids = (
                int(request.current_lane_id), int(authorization.target_lane_id)
            )
        elif bool(request.opportunistic_lane_change_allowed):
            candidate_lane_ids = available
        else:
            candidate_lane_ids = (int(request.current_lane_id),)
        candidate_frame = self.evaluate_lane_candidates(BehaviorCandidateRequest(
            lane_safety_scores=dict(request.lane_safety_scores),
            lane_prediction_risks=dict(frame.prediction.lane_prediction_risks),
            ego_lane_id=int(request.current_lane_id),
            available_lane_ids=candidate_lane_ids,
            route_optimal_lane_id=int(request.route_optimal_lane_id),
            in_junction=bool(frame.map_lane.in_junction),
            mpc_feedback_blocked_lane_ids=tuple(
                request.mpc_feedback.get("blocked_lane_ids", []) or []
            ),
            mpc_feedback_weight=float(
                request.config.get("mpc_feedback_candidate_weight", 80.0)
            ),
            nearest_front_obstacles_by_lane=dict(nearest),
            desired_speed_mps=float(request.target_speed_mps),
            progress_cost_weight=float(
                request.config.get("candidate_progress_cost_weight", 4.0)
            ),
        ))
        preferred = (
            int(authorization.target_lane_id)
            if bool(authorization.allowed)
            else int(candidate_frame.selected.target_lane_id)
            if bool(request.opportunistic_lane_change_allowed)
            else int(request.current_lane_id)
        )
        command = self.produce_command(
            behavior_planner=behavior_planner,
            static_obstacle_stage=static_obstacle_stage,
            reference_map=reference_map, ego_pose=request.ego_pose,
            ego_x_m=float(request.ego_location.x),
            ego_y_m=float(request.ego_location.y),
            ego_yaw_rad=float(request.ego_yaw_rad),
            ego_speed_mps=float(request.ego_speed_mps),
            max_deceleration_mps2=float(request.max_deceleration_mps2),
            current_lane_id=int(request.current_lane_id),
            route_optimal_lane_id=int(request.route_optimal_lane_id),
            next_macro_maneuver=str(request.route_next_macro_maneuver),
            in_junction=bool(frame.map_lane.in_junction),
            sim_time_s=float(request.sim_time_s),
            target_speed_mps=float(request.target_speed_mps),
            lane_safety_scores=request.lane_safety_scores,
            lane_prediction_risks=dict(frame.prediction.lane_prediction_risks),
            front_distance_by_lane=request.front_distance_by_lane,
            route_points=request.route_points,
            nearest_front_obstacles_by_lane=nearest,
            available_lane_ids=available,
            behavior_traffic_state=str(request.behavior_traffic_state),
            behavior_stop_target=request.behavior_stop_target,
            signal_context=request.signal_context,
            scenario_stop_required=bool(request.scenario_stop_required),
            preferred_target_lane_id=int(preferred),
            opportunistic_lane_change_allowed=bool(
                request.opportunistic_lane_change_allowed
            ),
            lane_change_reference_active=bool(
                request.lane_change_reference_active
            ),
            config=request.config, runtime_config=request.runtime_config,
            attempt_replan=attempt_replan,
            object_track_id=object_track_id,
        )
        return BehaviorCommandFrameResult(
            command=command, candidate_frame=candidate_frame,
            nearest_front_obstacles_by_lane=dict(nearest),
            candidate_lane_ids=tuple(candidate_lane_ids),
            cooperative_proposal=self._cooperative_proposal_from_candidate(
                authorization=authorization,
                candidate_frame=candidate_frame,
                current_lane_id=int(request.current_lane_id),
                opportunistic_lane_change_allowed=bool(
                    request.opportunistic_lane_change_allowed
                ),
            ),
        )

    @staticmethod
    def produce_command(
        *, behavior_planner: Any, static_obstacle_stage: Any,
        reference_map: Any, ego_pose: Any, ego_x_m: float, ego_y_m: float,
        ego_yaw_rad: float, ego_speed_mps: float, max_deceleration_mps2: float,
        current_lane_id: int, route_optimal_lane_id: int,
        next_macro_maneuver: str, in_junction: bool, sim_time_s: float,
        target_speed_mps: float, lane_safety_scores: Mapping[int, float],
        lane_prediction_risks: Mapping[int, Mapping[str, object]],
        front_distance_by_lane: Mapping[int, float], route_points: Sequence,
        nearest_front_obstacles_by_lane: Mapping[int, Mapping[str, object]],
        available_lane_ids: Sequence[int], behavior_traffic_state: str,
        behavior_stop_target: Optional[Mapping[str, object]],
        signal_context: Mapping[str, object], scenario_stop_required: bool,
        preferred_target_lane_id: int, opportunistic_lane_change_allowed: bool,
        lane_change_reference_active: bool, config: Mapping[str, object],
        runtime_config: Mapping[str, object],
        attempt_replan: Any, object_track_id: Any,
    ) -> BehaviorCommandResult:
        """Produce one behavior command through the sole obstacle arbitration path."""

        from opencda.planning_module.behavior_planner import (
            compute_ego_lane_offset,
            evaluate_intersection_obstacle_response,
        )

        try:
            alignment = compute_ego_lane_offset(reference_map, ego_pose)
        except Exception:
            alignment = {
                "lane_id": 0, "lateral_offset_m": float("inf"),
                "heading_error_rad": float("inf"),
            }
        lateral_m = float(alignment.get("lateral_offset_m", float("inf")))
        heading_rad = float(alignment.get("heading_error_rad", float("inf")))
        alignment_valid = bool(
            int(alignment.get("lane_id", 0) or 0) != 0
            and math.isfinite(lateral_m) and math.isfinite(heading_rad)
        )
        if not alignment_valid:
            lateral_m = heading_rad = float("inf")

        front_obstacle = nearest_front_obstacles_by_lane.get(int(current_lane_id))
        actual_mode = "INTERSECTION" if in_junction else "NORMAL"
        evaluation_mode = str(actual_mode)
        if evaluation_mode == "NORMAL" and bool(config.get(
            "static_obstacle_replan_normal_mode_enabled",
            runtime_config.get("static_obstacle_replan_normal_mode_enabled", True),
        )):
            evaluation_mode = "INTERSECTION"
        response = evaluate_intersection_obstacle_response(
            mode=evaluation_mode,
            front_obstacle_speed_mps=(
                None if front_obstacle is None else float(front_obstacle.get("v", 0.0))
            ),
            original_max_velocity_mps=float(target_speed_mps),
            moving_obstacle_speed_threshold_mps=float(config.get(
                "static_obstacle_speed_threshold_mps",
                runtime_config.get("static_obstacle_speed_threshold_mps",
                    runtime_config.get("intersection_obstacle_moving_speed_threshold_mps", 0.5)),
            )),
            route_lane_safety_score=float(lane_safety_scores.get(int(current_lane_id), 1.0)),
            static_obstacle_replan_lane_safety_threshold=float(config.get(
                "static_obstacle_replan_lane_safety_threshold",
                runtime_config.get("static_obstacle_replan_lane_safety_threshold",
                    runtime_config.get("intersection_static_obstacle_replan_lane_safety_threshold", 0.5)),
            )),
        )
        traffic_stop = bool(
            scenario_stop_required
            or str(behavior_traffic_state).strip().lower() in {"red", "yellow", "stop"}
        )
        requested = bool(
            config.get("static_obstacle_replan_enabled",
                runtime_config.get("static_obstacle_replan_enabled", True))
            and response.get("request_static_obstacle_replan", False)
            and not traffic_stop
        )
        obstacle_result = static_obstacle_stage.evaluate(
            requested=requested, traffic_control_stop_active=traffic_stop,
            obstacle_id="" if front_obstacle is None else object_track_id(front_obstacle),
            current_lane_id=int(current_lane_id),
            lane_change_reference_active=bool(lane_change_reference_active),
            sim_time_s=float(sim_time_s), normal_mode=actual_mode == "NORMAL",
            available_lane_ids=tuple(available_lane_ids),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
            attempt_replan=lambda: attempt_replan(dict(front_obstacle or {})),
        )
        local_avoidance = bool(obstacle_result.local_avoidance_active)
        local_target = obstacle_result.target_lane_id
        opportunistic_allowed = bool(opportunistic_lane_change_allowed or local_avoidance)
        preferred_lane = int(local_target) if local_avoidance else int(preferred_target_lane_id)
        command = behavior_planner.update(
            static_obstacle_stop_active=bool(obstacle_result.stop_active),
            lane_safety_scores=lane_safety_scores, ego_lane_id=int(current_lane_id),
            selected_lane_id=int(current_lane_id), ego_lateral_offset_m=float(lateral_m),
            ego_heading_error_rad=float(heading_rad), mode=actual_mode,
            route_optimal_lane_id=int(route_optimal_lane_id),
            next_macro_maneuver=str(next_macro_maneuver),
            front_obstacle_distance_by_lane=front_distance_by_lane,
            current_time_s=float(sim_time_s), wall_time_s=float(sim_time_s),
            traffic_signal_state=str(behavior_traffic_state),
            traffic_stop_target=(
                dict(behavior_stop_target)
                if isinstance(behavior_stop_target, Mapping) else None
            ),
            traffic_signal_context=dict(signal_context or {}),
            ego_speed_mps=float(ego_speed_mps),
            ego_max_deceleration_mps2=abs(float(max_deceleration_mps2)),
            ego_in_junction=bool(in_junction),
            ego_position_xy=(float(ego_x_m), float(ego_y_m)),
            global_route_points=list(route_points or ()),
            nearest_front_obstacles_by_lane=nearest_front_obstacles_by_lane,
            lane_prediction_risks=lane_prediction_risks,
            preferred_target_lane_id=int(preferred_lane),
            local_avoidance_target_lane_id=(
                int(local_target)
                if local_avoidance and local_target is not None else None
            ),
            lane_change_completion_allowed=not bool(lane_change_reference_active),
        )
        decision = str(command.get("decision", "lane_follow"))
        static_obstacle_stage.observe_behavior(decision, traffic_stop)
        return BehaviorCommandResult(
            decision=decision,
            target_lane_id=int(command.get("target_lane_id", current_lane_id) or current_lane_id),
            phase=str(command.get("lc_state", "LANE_KEEP")),
            lane_alignment_valid=alignment_valid,
            lane_lateral_error_m=float(lateral_m), lane_heading_error_rad=float(heading_rad),
            traffic_control_stop_active=traffic_stop,
            static_obstacle_result=obstacle_result,
            opportunistic_lane_change_allowed=opportunistic_allowed,
            preferred_target_lane_id=int(preferred_lane),
        )

    @staticmethod
    def prepare_turn_scenario_context(
        *,
        route_manager: Any,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        cruise_speed_mps: float,
        next_macro_maneuver: str,
        next_macro_distance_m: float,
        config: Mapping[str, object],
    ) -> TurnScenarioContext:
        """Resolve one authoritative topology/geometry view of the next turn."""

        from .speed_planner import turn_approach_lookahead_m

        direction, distance_m, reason = route_manager.upcoming_turn(
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
            lookahead_m=float(turn_approach_lookahead_m(
                cruise_speed_mps=float(cruise_speed_mps),
                config=dict(config),
            )),
        )
        macro_text = str(next_macro_maneuver or "").strip().lower()
        normalized = macro_text.replace("-", "_").replace(" ", "_")
        advanced_to_lane_change = normalized in {
            "lane_change_left", "lane_change_right",
            "change_lane_left", "change_lane_right",
        }
        macro_direction = (
            "left" if "turn left" in macro_text
            else "right" if "turn right" in macro_text
            else ""
        )
        if macro_direction and str(reason) != "route_geometry_turn_ahead":
            direction = macro_direction
            distance_m = float(next_macro_distance_m)
            reason = "admap_route_macro_direction_fallback"

        heading_error, lateral_m, alignment_reason = (
            route_manager.route_alignment(
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                ego_heading_rad=float(ego_heading_rad),
                heading_lookahead_m=float(
                    config.get("scenario_turn_exit_heading_lookahead_m", 5.0)
                ),
            )
        )
        valid = bool(
            math.isfinite(float(heading_error))
            and math.isfinite(float(lateral_m))
        )
        aligned = bool(
            valid
            and abs(float(heading_error)) <= float(
                config.get("scenario_turn_exit_max_heading_error_rad", 0.15)
            )
            and float(lateral_m) <= float(
                config.get("scenario_turn_exit_max_lateral_m", 0.75)
            )
        )
        return TurnScenarioContext(
            direction=str(direction),
            distance_m=float(distance_m),
            reason=str(reason),
            route_advanced_to_lane_change=advanced_to_lane_change,
            exit_heading_error_rad=float(heading_error),
            exit_lateral_m=float(lateral_m),
            exit_alignment_reason=str(alignment_reason),
            exit_alignment_valid=valid,
            exit_aligned=aligned,
        )

    @staticmethod
    def prepare_route_lane_change(
        *,
        route_manager: Any,
        local_map_snapshot: Any,
        route_summary: Mapping[str, object],
        current_lane_id: int,
        route_optimal_lane_id: int,
        route_found: bool,
        next_macro_maneuver: str,
        current_road_option: str,
        next_macro_distance_m: float,
        available_lane_ids: Sequence[int],
        lane_safety_scores: Mapping[int, float],
        lane_prediction_risks: Mapping[int, Mapping[str, object]],
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        ego_speed_mps: float,
        config: Mapping[str, object],
    ) -> RouteLaneChangeContext:
        """Interpret AD-map topology once and build the authorization request.

        This is deliberately pure with respect to maneuver state.  The bridge
        supplies one immutable map/route snapshot; BehaviorStage owns all
        interpretation of lane direction, physical adjacency and trigger
        windows.
        """

        frame_valid = int(getattr(local_map_snapshot, "frame_id", 0)) > 0
        direction = str(route_summary.get("lane_change_direction", "") or "").strip().lower()
        adjacent_directions = {}
        if direction in {"left", "right"}:
            adjacent_directions[int(route_optimal_lane_id)] = direction

        topology_current = int(
            getattr(local_map_snapshot, "ego_lane_id", 0)
            if frame_valid
            else route_summary.get("ad_current_lane_id", 0) or 0
        )
        topology_target = int(
            getattr(local_map_snapshot, "route_target_lane_id", 0)
            if frame_valid
            else route_summary.get("ad_target_lane_id", 0) or 0
        )
        topology_offset = int(
            getattr(local_map_snapshot, "route_target_offset", 0)
            if frame_valid
            else route_summary.get("lane_change_offset", 0) or 0
        )
        target_in_frame = bool(
            getattr(local_map_snapshot, "route_target_in_frame", False)
            if frame_valid
            else route_summary.get("target_in_local_frame", False)
        )
        physical_target = topology_target
        if frame_valid and topology_offset != 0:
            physical_target = int(
                local_map_snapshot.lane_at_ego_station(topology_offset) or 0
            )
            if physical_target == 0:
                physical_target = topology_target
        if target_in_frame and topology_offset != 0:
            direction = "left" if topology_offset > 0 else "right"
            adjacent_directions[int(route_optimal_lane_id)] = direction
            adjacent_directions[int(physical_target)] = direction

        preparation_m = max(
            float(config.get("route_lane_change_preparation_start_distance_m", 45.0)),
            float(ego_speed_mps)
            * float(config.get("route_lane_change_preparation_time_margin_s", 15.0)),
        )
        latest_m = max(
            float(config.get("route_lane_change_latest_start_distance_m", 12.0)),
            float(ego_speed_mps)
            * float(config.get("route_lane_change_latest_retry_time_margin_s", 9.0)),
        )
        explicit_start_m = max(
            float(config.get("route_lane_change_min_trigger_distance_m", 8.0)),
            float(ego_speed_mps)
            * float(config.get("route_tracking_lane_change_duration_s", 4.0))
            + float(config.get("route_lane_change_trigger_buffer_m", 3.0)),
        )
        geometry_direction, geometry_distance, geometry_reason = (
            route_manager.upcoming_lane_change(
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                ego_heading_rad=float(ego_heading_rad),
                lookahead_m=float(preparation_m),
            )
        )
        edge_id = route_manager.upcoming_lane_change_edge_id(
            lookahead_m=float(preparation_m)
        )

        # Maneuver naming follows the topology direction resolved so far, before
        # the arc-length geometry model gets to override physical adjacency --
        # this matches the ordering the bridge used inline.
        maneuver = str(next_macro_maneuver)
        normalized_maneuver = maneuver.strip().lower().replace("-", "_").replace(" ", "_")
        if direction in {"left", "right"} and normalized_maneuver in {
            "lane_change_left", "lane_change_right",
            "change_lane_left", "change_lane_right",
        }:
            maneuver = "lane_change_" + direction

        authorization_offset = topology_offset
        normalized_geometry_direction = str(geometry_direction or "").strip().lower()
        if normalized_geometry_direction in {"left", "right"} and frame_valid:
            adjacent_offset = 1 if normalized_geometry_direction == "left" else -1
            adjacent_lane_id = int(
                local_map_snapshot.lane_at_ego_station(adjacent_offset) or 0
            )
            if adjacent_lane_id != 0:
                physical_target = adjacent_lane_id
                authorization_offset = adjacent_offset
                direction = normalized_geometry_direction
                adjacent_directions[adjacent_lane_id] = direction

        request = RouteLaneChangeRequest(
            route_lane_change_allowed=bool(route_found),
            current_lane_id=int(current_lane_id),
            route_required_lane_id=int(physical_target),
            next_macro_maneuver=maneuver,
            current_road_option=str(current_road_option),
            remaining_distance_m=float(next_macro_distance_m),
            available_lane_ids=tuple(available_lane_ids),
            lane_safety_scores=dict(lane_safety_scores),
            lane_prediction_risks=dict(lane_prediction_risks),
            preparation_start_distance_m=float(preparation_m),
            latest_start_distance_m=float(latest_m),
            target_safety_threshold=float(config.get("route_lane_change_target_safety_threshold", 0.65)),
            require_adjacent=bool(config.get("route_lane_change_require_adjacent", True)),
            explicit_lane_change_start_distance_m=float(explicit_start_m),
            adjacent_lane_directions=adjacent_directions,
            topology_current_lane_id=int(topology_current),
            topology_target_lane_id=int(physical_target),
            topology_lane_offset=int(authorization_offset),
            topology_target_in_local_frame=bool(target_in_frame),
            route_geometry_direction=str(geometry_direction or ""),
            route_geometry_distance_m=(
                float(geometry_distance)
                if geometry_distance is not None
                and math.isfinite(float(geometry_distance))
                else None
            ),
        )
        return RouteLaneChangeContext(
            request=request,
            topology_current_lane_id=topology_current,
            topology_target_lane_id=topology_target,
            topology_lane_offset=topology_offset,
            topology_target_in_local_frame=target_in_frame,
            physical_target_lane_id=physical_target,
            physical_direction=direction,
            physical_direction_reason=(
                "admap_topology_direction"
                if direction in {"left", "right"}
                else "admap_topology_direction_missing"
            ),
            preparation_start_distance_m=preparation_m,
            geometry_direction=str(geometry_direction or ""),
            geometry_distance_m=(
                float(geometry_distance)
                if geometry_distance is not None
                else float("inf")
            ),
            geometry_reason=str(geometry_reason),
            edge_id=(str(edge_id) if edge_id is not None else None),
        )

    @staticmethod
    def authorize_opportunistic_lane_change(
        request: OpportunisticLaneChangeRequest,
    ):
        risky_lane_count = sum(
            bool(dict(risk or {}).get("risk", False))
            for risk in dict(request.lane_prediction_risks).values()
        )
        dense_traffic_active = bool(
            request.dense_traffic_lock_enabled
            and (
                int(request.object_count) >= int(request.dense_object_count)
                or int(risky_lane_count) >= int(request.dense_risky_lane_count)
            )
        )
        return authorize_opportunistic_lane_change(
            enabled=bool(request.enabled),
            start_lock_active=(
                float(request.sim_time_s) <= float(request.start_lock_until_s)
            ),
            dense_traffic_lock_active=bool(dense_traffic_active),
        )

    @staticmethod
    def evaluate_lane_candidates(request: BehaviorCandidateRequest):
        """Select a lane using route, prediction and prior MPC feedback."""

        return evaluate_behavior_candidates(
            lane_safety_scores=dict(request.lane_safety_scores),
            lane_prediction_risks=dict(request.lane_prediction_risks),
            ego_lane_id=int(request.ego_lane_id),
            selected_lane_id=int(request.ego_lane_id),
            available_lane_ids=list(request.available_lane_ids),
            route_optimal_lane_id=int(request.route_optimal_lane_id),
            mode="INTERSECTION" if request.in_junction else "NORMAL",
            mpc_feedback_blocked_lane_ids=list(
                request.mpc_feedback_blocked_lane_ids
            ),
            mpc_feedback_weight=float(request.mpc_feedback_weight),
            nearest_front_obstacles_by_lane=dict(
                request.nearest_front_obstacles_by_lane
            ),
            desired_speed_mps=float(request.desired_speed_mps),
            progress_cost_weight=float(request.progress_cost_weight),
        )

    @staticmethod
    def apply_overrides(request: BehaviorOverrideRequest) -> BehaviorOverrideResult:
        """Apply authorization and scenario priority in one deterministic order."""

        decision = str(request.decision)
        target_lane_id = int(request.target_lane_id)
        phase = str(request.phase)
        reason = ""
        reset_reason = ""
        authorized = bool(
            request.lane_change_authorized
            or request.opportunistic_lane_change_allowed
        )

        def append(value: str) -> None:
            nonlocal reason
            if str(value):
                reason = ";".join(value for value in (reason, str(value)) if value)

        if (
            request.scenario_speed_cap_active
            and decision not in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        ):
            append(request.scenario_reason)
        if not authorized and decision in {"lane_change_left", "lane_change_right"}:
            decision, target_lane_id, phase = "lane_follow", int(request.current_lane_id), "LANE_KEEP"
            append(request.lane_change_gate_reason or "lane_change_suppressed_without_valid_route")
            reset_reason = str(reason)
        if request.prepare_reference_lock and phase.upper().startswith("PREPARE_LANE_CHANGE"):
            decision, target_lane_id = "lane_follow", int(request.current_lane_id)
            append("prepare_lane_change_reference_locked_to_current_lane")
        if not authorized and decision in {"lane_change_left", "lane_change_right"}:
            decision, target_lane_id, phase = "lane_follow", int(request.current_lane_id), "LANE_KEEP"
            append("lane_change_without_authorization:" + str(request.lane_change_authorization_reason))
            reset_reason = "lane_change_without_authorization"

        scenario_override = str(request.scenario_override_decision)
        mandatory_stop = scenario_override in {
            "stop_at_intersection", "stop_sign", "emergency_brake"
        }
        suppress_scenario_override = bool(
            request.local_avoidance_active
            and decision in {"lane_change_left", "lane_change_right"}
            and not request.ego_in_junction
            and not mandatory_stop
        )
        if scenario_override and not suppress_scenario_override:
            decision = scenario_override
            target_lane_id = int(request.current_lane_id)
            phase = str(request.scenario_override_phase or "LANE_KEEP")
            append(request.scenario_reason)
        elif (
            request.route_turn_decision
            and not request.lane_change_commitment_active
            and not request.local_avoidance_active
            and decision not in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        ):
            decision = str(request.route_turn_decision)
            target_lane_id = int(request.current_lane_id)
            phase = (
                "INTERSECTION_TURN_LEFT"
                if decision.endswith("_left")
                else "INTERSECTION_TURN_RIGHT"
            )
            append("route_option_driven_behavior:" + str(request.route_current_road_option))
        return BehaviorOverrideResult(
            decision=decision,
            target_lane_id=target_lane_id,
            phase=phase,
            stop_goal_active=bool(request.stop_goal_active or request.scenario_stop_required),
            reason=reason,
            reset_lane_change_reason=reset_reason,
        )

    def authorize_route_lane_change(
        self,
        request: RouteLaneChangeRequest,
        *,
        maneuver_manager: Any,
    ):
        """Resolve and latch one route-required maneuver exactly once."""

        authorization = authorize_route_lane_change(
            route_lane_change_allowed=bool(request.route_lane_change_allowed),
            current_lane_id=int(request.current_lane_id),
            route_required_lane_id=int(request.route_required_lane_id),
            next_macro_maneuver=str(request.next_macro_maneuver),
            current_road_option=str(request.current_road_option),
            remaining_distance_m=float(request.remaining_distance_m),
            available_lane_ids=list(request.available_lane_ids),
            lane_safety_scores=dict(request.lane_safety_scores),
            lane_prediction_risks=dict(request.lane_prediction_risks),
            preparation_start_distance_m=float(request.preparation_start_distance_m),
            latest_start_distance_m=float(request.latest_start_distance_m),
            target_safety_threshold=float(request.target_safety_threshold),
            require_adjacent=bool(request.require_adjacent),
            explicit_lane_change_start_distance_m=float(
                request.explicit_lane_change_start_distance_m
            ),
            adjacent_lane_directions=dict(request.adjacent_lane_directions),
            topology_current_lane_id=int(request.topology_current_lane_id),
            topology_target_lane_id=int(request.topology_target_lane_id),
            topology_lane_offset=int(request.topology_lane_offset),
            topology_target_in_local_frame=bool(
                request.topology_target_in_local_frame
            ),
            route_geometry_direction=str(request.route_geometry_direction),
            route_geometry_distance_m=request.route_geometry_distance_m,
        )
        active = self._route_lane_change_latch.active
        target_reached = bool(
            active is not None
            and lane_change_target_reached(
                current_lane_id=int(request.current_lane_id),
                remembered_target_lane_id=int(active.target_lane_id),
                current_ad_lane_id=int(request.topology_current_lane_id),
                remembered_target_ad_lane_id=int(active.target_lane_id),
                target_in_local_frame=bool(
                    request.topology_target_in_local_frame
                ),
                target_lane_offset=int(request.topology_lane_offset),
            )
        )
        authorization = self._route_lane_change_latch.update(
            authorization,
            target_reached=bool(target_reached),
            in_turn_connector=(
                str(request.current_road_option).strip().upper()
                in {"LEFT", "RIGHT"}
            ),
        )
        if bool(maneuver_manager.route_lane_change_edge_completed):
            self._route_lane_change_latch.reset()
            authorization = replace(
                authorization,
                allowed=False,
                required_by_route=False,
                reason=(
                    "route_lane_change_edge_already_completed:"
                    + str(maneuver_manager.route_lane_change_edge_id)
                ),
            )
        return authorization

    def resolve_route_context(
        self, *, adapter_output: Any, local_map_snapshot: Any,
        route_manager: Any, maneuver_manager: Any, reference_provider: Any,
        current_lane_id: int, ego_x_m: float, ego_y_m: float,
        ego_heading_rad: float, ego_speed_mps: float,
        available_lane_ids: Sequence[int], config: Mapping[str, object],
    ) -> RouteBehaviorContextResult:
        """Prepare and authorize the route maneuver from one input frame."""

        frame = adapter_output.frame
        route = frame.planning.route
        route_allowed = bool(route.route_found)
        context = self.prepare_route_lane_change(
            route_manager=route_manager,
            local_map_snapshot=local_map_snapshot,
            route_summary=adapter_output.route_summary,
            current_lane_id=int(current_lane_id),
            route_optimal_lane_id=int(adapter_output.route_optimal_lane_id),
            route_found=bool(route_allowed),
            next_macro_maneuver=str(route.next_macro_maneuver),
            current_road_option=str(route.current_road_option),
            next_macro_distance_m=float(route.next_macro_distance_m),
            available_lane_ids=tuple(available_lane_ids),
            lane_safety_scores=dict(adapter_output.lane_safety_scores),
            lane_prediction_risks=dict(frame.prediction.lane_prediction_risks),
            ego_x_m=float(ego_x_m), ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
            ego_speed_mps=float(ego_speed_mps), config=config,
        )
        phase = str(maneuver_manager.lane_change.phase or "").strip().lower()
        execution_active = bool(
            reference_provider.snapshot(LANE_CHANGE).mutable_samples()
        ) or phase in {"executing", "target_lane_stabilization"}
        resolved = self.resolve_route_lane_change(
            context,
            maneuver_manager=maneuver_manager,
            route_cursor=route_manager.route_cursor,
            current_lane_id=int(current_lane_id),
            execution_active=bool(execution_active),
            replan_missed_lane_change=bool(
                config.get("missed_lane_change_route_replan_enabled", True)
            ),
        )
        return RouteBehaviorContextResult(
            context=context,
            authorization=resolved.authorization,
            replan_reason=str(resolved.replan_reason),
            route_lane_change_allowed=bool(route_allowed),
        )

    def resolve_route_lane_change(
        self,
        context: RouteLaneChangeContext,
        *,
        maneuver_manager: Any,
        route_cursor: Any,
        current_lane_id: int,
        execution_active: bool,
        replan_missed_lane_change: bool,
    ) -> RouteLaneChangeStageResult:
        """Own route-lane-change authorization and remembered requirement.

        Geometry execution remains in the reference stage.  This method owns
        only the behavior lifecycle: edge observation, authorization latching,
        missed-maneuver invalidation, and required-target memory.
        """

        maneuver_manager.observe_route_lane_change_edge(context.edge_id)
        authorization = self.authorize_route_lane_change(
            context.request,
            maneuver_manager=maneuver_manager,
        )
        replan_reason = ""
        if bool(getattr(route_cursor, "missed_maneuver", False)) and not bool(
            execution_active
        ):
            missed_reason = (
                "route_cursor_missed_lane_change:"
                f"s={float(getattr(route_cursor, 'route_s_m', 0.0)):.2f}:"
                "untracked_motion="
                f"{float(getattr(route_cursor, 'stalled_motion_m', 0.0)):.2f}"
            )
            self.reset_route_lane_change_authorization()
            maneuver_manager.clear_required_lane_change()
            authorization = replace(
                authorization,
                allowed=False,
                required_by_route=False,
                reason=missed_reason,
            )
            replan_reason = "turn_missed_lane_change_route_unreachable"

        if bool(authorization.required_by_route):
            raw_target = int(context.topology_target_lane_id or 0)
            maneuver_manager.remember_required_lane_change(
                int(authorization.target_lane_id),
                raw_target if raw_target != 0 else None,
            )
        elif maneuver_manager.lane_change.required_target_lane_id is not None:
            reached = lane_change_target_reached(
                current_lane_id=int(current_lane_id),
                remembered_target_lane_id=int(
                    maneuver_manager.lane_change.required_target_lane_id
                ),
                current_ad_lane_id=int(context.topology_current_lane_id or 0),
                remembered_target_ad_lane_id=int(
                    maneuver_manager.lane_change.required_target_ad_lane_id or 0
                ),
                target_in_local_frame=bool(
                    context.topology_target_in_local_frame
                ),
                target_lane_offset=int(context.topology_lane_offset),
            )
            if reached:
                maneuver_manager.clear_required_lane_change()
            elif not execution_active and bool(replan_missed_lane_change):
                maneuver_manager.clear_required_lane_change()
                replan_reason = "lane_change_missed_route_unreachable"

        return RouteLaneChangeStageResult(
            context=context,
            authorization=authorization,
            replan_reason=replan_reason,
        )

    def resolve_lateral_ownership(
        self,
        *,
        authorization: Any,
        maneuver_manager: Any,
        owner_state: str,
        ego_speed_mps: float,
        planning_speed_mps: float,
        lane_change_duration_s: float,
        dt_s: float,
        lane_width_m: float,
        distance_to_turn_m: float,
        config: Mapping[str, object],
    ) -> LateralOwnershipResult:
        """Arbitrate start feasibility and exclusive lateral ownership once."""

        from .candidate_pipeline import (
            lane_change_geometry_requirements,
            lane_change_operational_curvature_limit_1pm,
        )
        from .route_authorization import suppress_lane_change_for_lateral_owner

        operational_curvature = lane_change_operational_curvature_limit_1pm(
            planning_speed_mps=float(planning_speed_mps),
            lateral_accel_limit_mps2=float(config.get(
                "route_tracking_lane_change_lateral_accel_limit_mps2", 1.3
            )),
            vehicle_max_curvature_1pm=float(config.get(
                "reference_vehicle_max_curvature_1pm", 0.35
            )),
            minimum_speed_mps=float(config.get(
                "lane_change_min_geometry_speed_mps", 2.0
            )),
        )
        _, geometry_arc_m, _ = lane_change_geometry_requirements(
            ego_speed_mps=float(ego_speed_mps),
            target_speed_mps=float(planning_speed_mps),
            duration_s=float(lane_change_duration_s),
            dt_s=float(dt_s),
            lane_width_m=float(lane_width_m),
            max_curvature_1pm=float(operational_curvature),
            minimum_geometry_speed_mps=float(config.get(
                "lane_change_min_geometry_speed_mps", 2.0
            )),
            minimum_length_m=float(config.get("lane_change_min_length_m", 10.0)),
            acceleration_limit_mps2=float(config.get(
                "lane_change_planning_acceleration_limit_mps2", 2.0
            )),
        )
        handoff_arc_m = max(0.0, float(config.get(
            "lane_change_to_turn_reference_transition_arc_m", 10.0
        )))
        start_transition = maneuver_manager.lane_change_start_feasibility(
            authorization_allowed=bool(authorization.allowed),
            distance_to_turn_m=float(distance_to_turn_m),
            geometry_arc_m=float(geometry_arc_m),
            handoff_arc_m=float(handoff_arc_m),
        )
        if str(start_transition.action) == "deny":
            self.reset_route_lane_change_authorization()
            authorization = replace(
                authorization,
                allowed=False,
                reason=str(start_transition.reason),
            )

        handoff = maneuver_manager.transfer_lateral_ownership_to_turn(
            owner_state=str(owner_state)
        )
        if str(handoff.action) == "release":
            self.reset_route_lane_change_authorization()
            maneuver_manager.clear_required_lane_change()
        authorization = suppress_lane_change_for_lateral_owner(
            authorization,
            owner_state=str(owner_state),
        )
        return LateralOwnershipResult(
            authorization=authorization,
            start_transition=start_transition,
            handoff=handoff,
            reference_release_event=(
                "maneuver_abandoned"
                if str(handoff.action) == "release"
                else ""
            ),
            geometry_arc_m=float(geometry_arc_m),
            operational_curvature_1pm=float(operational_curvature),
        )

    def resolve_conflicts(
        self,
        request: ConflictResolutionRequest,
        *,
        maneuver_manager: Any,
    ) -> ConflictResolutionResult:
        """Resolve cooperative, opportunistic and lateral-owner conflicts.

        This is the explicit seam where prediction/cooperation may constrain a
        behavior proposal.  It returns policy data only; releasing reference
        geometry remains the caller's lifecycle side effect.
        """

        authorization = request.route_authorization
        opportunistic = self.authorize_opportunistic_lane_change(
            request.opportunistic_request
        )
        opportunistic_allowed = bool(opportunistic.allowed)
        gate_reason = (
            "" if opportunistic_allowed else str(opportunistic.reason)
        )
        ownership = self.resolve_lateral_ownership(
            authorization=authorization,
            maneuver_manager=maneuver_manager,
            owner_state=str(request.owner_state),
            ego_speed_mps=float(request.ego_speed_mps),
            planning_speed_mps=float(request.planning_speed_mps),
            lane_change_duration_s=float(request.lane_change_duration_s),
            dt_s=float(request.dt_s),
            lane_width_m=float(request.lane_width_m),
            distance_to_turn_m=float(request.distance_to_turn_m),
            config=request.config,
        )
        authorization = ownership.authorization
        if str(authorization.reason).startswith("scenario_lateral_owner:"):
            opportunistic_allowed = False
            gate_reason = (
                "opportunistic_lane_change_suppressed:"
                + str(authorization.reason)
            )

        return ConflictResolutionResult(
            authorization=authorization,
            lateral_ownership=ownership,
            opportunistic_allowed=bool(opportunistic_allowed),
            lane_change_gate_reason=str(gate_reason),
        )

    @staticmethod
    def finalize(
        *,
        maneuver: str,
        phase: str,
        source_lane_id: int,
        target_lane_id: int,
        requested_speed_mps: float,
        stop_required: bool,
        route_required: bool,
        traffic_signal_state: str,
        boundary_recovery_active: bool,
        stop_target: Optional[Mapping[str, object]],
        reason: str = "",
        diagnostics: Optional[Mapping[str, object]] = None,
    ) -> BehaviorStageResult:
        normalized = str(maneuver or "lane_follow")
        direction = (
            "left" if normalized.endswith("_left")
            else "right" if normalized.endswith("_right")
            else ""
        )
        decision = BehaviorDecision(
            maneuver=normalized,
            phase=str(phase or "LANE_KEEP"),
            source_lane_id=int(source_lane_id),
            target_lane_id=int(target_lane_id),
            target_corridor_id=int(target_lane_id),
            direction=direction,
            speed_intent="track_target",
            requested_speed_mps=max(0.0, float(requested_speed_mps)),
            stop_required=bool(stop_required),
            route_required=bool(route_required),
            traffic_signal_state=str(traffic_signal_state or "unknown"),
            boundary_recovery_active=bool(boundary_recovery_active),
            stop_target=(
                MappingProxyType(dict(stop_target))
                if isinstance(stop_target, Mapping)
                else None
            ),
            reason=str(reason or ""),
        )
        fields = dict(diagnostics or {})
        fields.update(decision.as_debug_fields())
        return BehaviorStageResult(decision, MappingProxyType(fields))

    def finalize_planning_frame(
        self, *, maneuver: str, phase: str, source_lane_id: int,
        target_lane_id: int, requested_speed_mps: float, stop_required: bool,
        route_required: bool, traffic_signal_state: str,
        boundary_recovery_active: bool,
        stop_target: Optional[Mapping[str, object]], reason: str,
        lane_safety_scores: Mapping[int, float], raw_signal_state: str,
        resolved_signal_state: str, filtered_signal_state: str,
        traffic_control_from_cp: bool, scenario_state: str,
    ) -> BehaviorStageResult:
        """Publish the one typed behavior value consumed downstream."""

        return self.finalize(
            maneuver=maneuver, phase=phase, source_lane_id=source_lane_id,
            target_lane_id=target_lane_id,
            requested_speed_mps=requested_speed_mps,
            stop_required=stop_required, route_required=route_required,
            traffic_signal_state=traffic_signal_state,
            boundary_recovery_active=boundary_recovery_active,
            stop_target=stop_target, reason=reason,
            diagnostics={
                "lane_safety_scores": dict(lane_safety_scores),
                "traffic_signal_raw_state": str(raw_signal_state),
                "traffic_signal_resolved_state": str(resolved_signal_state),
                "traffic_signal_filtered_state": str(filtered_signal_state),
                "traffic_signal_behavior_state": str(traffic_signal_state),
                "traffic_control_from_cp": bool(traffic_control_from_cp),
                "boundary_recovery_scenario_state": str(scenario_state),
            },
        )

    @staticmethod
    def destination_stop(result: BehaviorStageResult) -> BehaviorStageResult:
        decision = replace(
            result.decision,
            maneuver="destination_stop",
            phase="DESTINATION_STOP",
            direction="",
            speed_intent="stop",
            requested_speed_mps=0.0,
            stop_required=True,
            reason="route_destination_stop",
        )
        fields = result.mutable_diagnostics()
        fields.update(decision.as_debug_fields())
        return BehaviorStageResult(decision, MappingProxyType(fields))
