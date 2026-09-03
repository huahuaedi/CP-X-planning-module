"""Typed behavior-stage output boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from .behavior_decision import BehaviorDecision
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


class BehaviorStage:
    """Sole constructor and transition owner for BehaviorDecision."""

    def __init__(self) -> None:
        self._route_lane_change_latch = RouteLaneChangeAuthorizationLatch()

    def reset_route_lane_change_authorization(self) -> None:
        self._route_lane_change_latch.reset()

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
