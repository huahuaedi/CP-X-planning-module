"""Finalize one selected behavior/reference candidate through its owners."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from .behavior_reference_execution_stage import BehaviorReferenceResult


@dataclass(frozen=True)
class MPCCostProfileRequest:
    """Selected-maneuver inputs for the downstream MPC objective profile."""

    behavior: str
    planner_lc_state: str
    planner_mode: str
    reference_tracking_mode: str
    next_macro_maneuver: str
    sim_time_s: float
    nearest_obstacle_distance_m: Optional[float]
    ego_speed_mps: float


@dataclass(frozen=True)
class BehaviorFrameFinalizationRequest:
    """Inputs for the immutable behavior decision published by this tick."""

    maneuver: str
    phase: str
    source_lane_id: int
    target_lane_id: int
    stop_required: bool
    route_required: bool
    traffic_signal_state: str
    boundary_recovery_active: bool
    stop_target: Optional[Mapping[str, object]]
    reason: str
    lane_safety_scores: Mapping[int, float]
    raw_signal_state: str
    resolved_signal_state: str
    filtered_signal_state: str
    traffic_control_from_cp: bool
    scenario_state: str


@dataclass(frozen=True)
class BehaviorReferenceFinalizationRequest:
    """Candidate-selected inputs for the final planning-owner handoff."""

    speed_plan: Any
    cooperative_frame: Any
    reference_provider: Any
    config: Mapping[str, object]
    mpc_profile: MPCCostProfileRequest
    post_turn: Any
    behavior: BehaviorFrameFinalizationRequest


class BehaviorReferenceFinalizationStage:
    """Own the downstream ordering after candidate selection.

    The algorithms remain with their domain owners.  This stage only makes
    their order explicit and atomic: cooperative speed, turn speed, MPC cost
    profile, post-turn handoff, then immutable behavior publication.
    """

    def __init__(
        self, *, speed: Any, mpc_cost_profile: Any,
        reference_planning: Any, behavior: Any,
    ) -> None:
        self._speed = speed
        self._mpc_cost_profile = mpc_cost_profile
        self._reference_planning = reference_planning
        self._behavior = behavior

    def run(
        self, request: BehaviorReferenceFinalizationRequest
    ) -> BehaviorReferenceResult:
        speed_plan = request.speed_plan
        cav_resolution = request.cooperative_frame.cav_resolution
        if cav_resolution is not None:
            speed_plan = self._speed.constrain_plan(
                speed_plan, cav_resolution.speed_constraint
            )

        turn_constraint = self._speed.turn_curvature_constraint(
            request.reference_provider.turn_master_curvature_1pm(),
            request.config,
            distance_to_turn_m=speed_plan.upcoming_turn_distance_m,
        )
        speed_plan = self._speed.constrain_plan(speed_plan, turn_constraint)

        reference_debug = dict(request.post_turn.debug_fields)
        if turn_constraint is not None:
            reference_debug.update({
                "turn_master_curvature_1pm": float(
                    request.reference_provider.turn_master_curvature_1pm()
                ),
                "turn_curvature_speed_advisory_mps": float(
                    turn_constraint.maximum_mps
                ),
                "turn_curvature_speed_source": "persistent_turn_master",
                "turn_longitudinal_authority": "SpeedPlanner",
            })

        profile = request.mpc_profile
        self._mpc_cost_profile.apply(
            behavior=str(profile.behavior),
            planner_lc_state=str(profile.planner_lc_state),
            planner_mode=str(profile.planner_mode),
            reference_tracking_mode=str(profile.reference_tracking_mode),
            next_macro_maneuver=str(profile.next_macro_maneuver),
            sim_time_s=float(profile.sim_time_s),
            nearest_obstacle_distance_m=profile.nearest_obstacle_distance_m,
            ego_speed_mps=float(profile.ego_speed_mps),
        )

        target_speed_mps = float(speed_plan.target_speed_mps)
        post_turn = self._reference_planning.finalize_post_turn(replace(
            request.post_turn,
            target_speed_mps=target_speed_mps,
            debug_fields=reference_debug,
        ))
        behavior = request.behavior
        behavior_stage_result = self._behavior.finalize_planning_frame(
            maneuver=str(behavior.maneuver),
            phase=str(behavior.phase),
            source_lane_id=int(behavior.source_lane_id),
            target_lane_id=int(behavior.target_lane_id),
            requested_speed_mps=target_speed_mps,
            stop_required=bool(behavior.stop_required),
            route_required=bool(behavior.route_required),
            traffic_signal_state=str(behavior.traffic_signal_state),
            boundary_recovery_active=bool(behavior.boundary_recovery_active),
            stop_target=(
                dict(behavior.stop_target)
                if behavior.stop_target is not None else None
            ),
            reason=str(behavior.reason),
            lane_safety_scores=behavior.lane_safety_scores,
            raw_signal_state=str(behavior.raw_signal_state),
            resolved_signal_state=str(behavior.resolved_signal_state),
            filtered_signal_state=str(behavior.filtered_signal_state),
            traffic_control_from_cp=bool(behavior.traffic_control_from_cp),
            scenario_state=str(behavior.scenario_state),
        )
        return BehaviorReferenceResult(
            destination_state=tuple(post_turn.mutable_destination_state()),
            reference_samples=tuple(
                dict(sample) for sample in post_turn.mutable_samples()
            ),
            behavior_stage_result=behavior_stage_result,
            reference_debug=dict(post_turn.debug_fields),
            speed_plan=speed_plan,
            cav_resolution=cav_resolution,
        )
