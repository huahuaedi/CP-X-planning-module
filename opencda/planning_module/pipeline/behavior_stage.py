"""Typed behavior-stage output boundary."""

from __future__ import annotations

import math
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
