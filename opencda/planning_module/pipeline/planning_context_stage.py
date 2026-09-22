"""Freeze planner input and resolve route/scenario/lateral context once."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from .behavior_stage import ConflictResolutionRequest, OpportunisticLaneChangeRequest


@dataclass(frozen=True)
class PlanningContextRequest:
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    object_snapshots: Sequence[Mapping[str, Any]]
    cp_payload: Optional[Mapping[str, Any]]
    local_map_snapshot: Any
    route_manager: Any
    maneuver_manager: Any
    reference_provider: Any
    traffic_memory: Any
    opportunistic_lane_change_enabled: bool
    lane_change_start_lock_until_s: float
    dense_traffic_lock_enabled: bool
    dense_object_count: int
    dense_risky_lane_count: int
    planning_speed_mps: float
    sim_time_s: float
    cruise_speed_mps: float
    mpc_dt_s: float
    lane_width_m: float
    config: Mapping[str, object]
    boundary_recovery_request: Any = None


@dataclass(frozen=True)
class BehaviorContextFrame:
    route_behavior: Any
    scenario_observation: Any
    conflict_resolution: Any
    route_replan_attempted: bool = False
    route_replan_succeeded: bool = False
    route_replan_reason: str = ""


@dataclass(frozen=True)
class PlanningContextFrame:
    adapter_output: Any
    object_snapshots: Tuple[Mapping[str, Any], ...]
    local_map_snapshot: Any
    current_lane_id: int
    behavior_context: BehaviorContextFrame

    @property
    def planner_input_frame(self):
        return self.adapter_output.frame

    def mutable_object_snapshots(self):
        return [dict(snapshot) for snapshot in self.object_snapshots]


class PlanningContextStage:
    """Own the frozen-input -> behavior-context stage ordering."""

    def __init__(self, *, input_adapter: Any, behavior: Any, scenario: Any) -> None:
        self._input_adapter = input_adapter
        self._behavior = behavior
        self._scenario = scenario

    def run(
        self,
        request: PlanningContextRequest,
        *,
        resolve_actor_state: Callable[..., Any],
        attempt_turn_replan: Callable[[str], tuple],
        reset_lane_change: Optional[Callable[..., Any]] = None,
        observe_stage_duration: Optional[Callable[[str, float], None]] = None,
    ) -> PlanningContextFrame:
        adapter_output = self._timed(
            "sub_input_adapter_build",
            lambda: self._input_adapter.build(
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                ego_speed_mps=float(request.ego_speed_mps),
                object_snapshots=request.object_snapshots,
                cp_payload=request.cp_payload,
            ),
            observe_stage_duration,
        )
        frame = adapter_output.frame
        tracked_objects = tuple(
            dict(snapshot)
            for snapshot in frame.perception.planning_objects
        )
        local_map = request.local_map_snapshot
        current_lane_id = int(
            local_map.ego_lane_id
            if local_map.frame_id > 0 and local_map.ego_lane_id != 0
            else adapter_output.current_lane_id
        )

        route_behavior = self._timed(
            "sub_resolve_route_context",
            lambda: self._behavior.resolve_route_context(
                adapter_output=adapter_output,
                local_map_snapshot=local_map,
                route_manager=request.route_manager,
                maneuver_manager=request.maneuver_manager,
                reference_provider=request.reference_provider,
                current_lane_id=current_lane_id,
                available_lane_ids=tuple(frame.map_lane.allowed_lane_ids),
                ego_x_m=float(request.ego_location.x),
                ego_y_m=float(request.ego_location.y),
                ego_heading_rad=float(request.ego_yaw_rad),
                ego_speed_mps=float(request.ego_speed_mps),
                config=request.config,
            ),
            observe_stage_duration,
        )
        replan_attempted = False
        replan_succeeded = False
        replan_reason = ""
        if str(route_behavior.replan_reason):
            replan_attempted, replan_succeeded, replan_reason = (
                attempt_turn_replan(str(route_behavior.replan_reason))
            )

        scenario_observation = self._timed(
            "sub_observe_planning_frame",
            lambda: self._observe_scenario(
                request,
                adapter_output=adapter_output,
                current_lane_id=current_lane_id,
                resolve_actor_state=resolve_actor_state,
            ),
            observe_stage_duration,
        )
        conflict_resolution = self._timed(
            "sub_resolve_conflicts",
            lambda: self._resolve_conflict(
                request,
                adapter_output=adapter_output,
                object_count=len(tracked_objects),
                route_behavior=route_behavior,
                scenario_observation=scenario_observation,
            ),
            observe_stage_duration,
        )
        self._apply_lateral_handoff(
            conflict_resolution,
            reference_provider=request.reference_provider,
            reset_lane_change=reset_lane_change,
        )
        return PlanningContextFrame(
            adapter_output=adapter_output,
            object_snapshots=tracked_objects,
            local_map_snapshot=local_map,
            current_lane_id=current_lane_id,
            behavior_context=BehaviorContextFrame(
                route_behavior=route_behavior,
                scenario_observation=scenario_observation,
                conflict_resolution=conflict_resolution,
                route_replan_attempted=bool(replan_attempted),
                route_replan_succeeded=bool(replan_succeeded),
                route_replan_reason=str(replan_reason),
            ),
        )

    def _observe_scenario(
        self, request, *, adapter_output, current_lane_id, resolve_actor_state
    ):
        frame = adapter_output.frame
        route = frame.planning.route
        traffic = frame.planning.traffic_control
        raw_stop_target = (
            traffic.stop_target.as_dict() if traffic.stop_target.active else None
        )
        return self._scenario.observe_planning_context(
            raw_traffic_state=str(traffic.signal_state),
            raw_stop_target=raw_stop_target,
            signal_context=dict(adapter_output.signal_context),
            traffic_memory=request.traffic_memory,
            resolve_actor_state=resolve_actor_state,
            project_stop_target=lambda *, stop_target: (
                request.reference_provider.stop_target_forward(
                    ego_location=request.ego_location,
                    ego_yaw_rad=float(request.ego_yaw_rad),
                    stop_target=(
                        dict(stop_target)
                        if isinstance(stop_target, Mapping) else None
                    ),
                    fallback_destination_state=[],
                )
            ),
            prepare_turn_context=lambda: self._behavior.prepare_turn_scenario_context(
                route_manager=request.route_manager,
                ego_x_m=float(request.ego_location.x),
                ego_y_m=float(request.ego_location.y),
                ego_heading_rad=float(request.ego_yaw_rad),
                cruise_speed_mps=float(request.cruise_speed_mps),
                next_macro_maneuver=str(route.next_macro_maneuver),
                next_macro_distance_m=float(route.next_macro_distance_m),
                config=request.config,
            ),
            sim_time_s=float(request.sim_time_s),
            ego_x_m=float(request.ego_location.x),
            ego_y_m=float(request.ego_location.y),
            ego_yaw_rad=float(request.ego_yaw_rad),
            ego_speed_mps=float(request.ego_speed_mps),
            current_lane_id=int(current_lane_id),
            ego_in_junction=bool(frame.map_lane.in_junction),
            current_road_option=str(route.current_road_option),
            next_macro_maneuver=str(route.next_macro_maneuver),
            virtual_stop_distance_m=float(request.config.get(
                "full_latched_virtual_stop_distance_m", 12.0
            )),
            boundary_recovery_request=request.boundary_recovery_request,
        )

    def _resolve_conflict(
        self, request, *, adapter_output, object_count,
        route_behavior, scenario_observation,
    ):
        opportunistic_request = OpportunisticLaneChangeRequest(
            enabled=bool(request.opportunistic_lane_change_enabled),
            sim_time_s=float(request.sim_time_s),
            start_lock_until_s=float(request.lane_change_start_lock_until_s),
            dense_traffic_lock_enabled=bool(request.dense_traffic_lock_enabled),
            object_count=int(object_count),
            dense_object_count=int(request.dense_object_count),
            lane_prediction_risks=dict(
                adapter_output.frame.prediction.lane_prediction_risks
            ),
            dense_risky_lane_count=int(request.dense_risky_lane_count),
        )
        return self._behavior.resolve_conflicts(
            ConflictResolutionRequest(
                route_authorization=route_behavior.authorization,
                opportunistic_request=opportunistic_request,
                owner_state=str(scenario_observation.scenario.decision.state),
                ego_speed_mps=float(request.ego_speed_mps),
                planning_speed_mps=float(request.planning_speed_mps),
                lane_change_duration_s=max(0.1, float(request.config.get(
                    "candidate_lane_change_normal_duration_s", 4.0
                ))),
                dt_s=float(request.mpc_dt_s),
                lane_width_m=float(request.lane_width_m),
                distance_to_turn_m=float(
                    scenario_observation.turn_context.distance_m
                ),
                config=request.config,
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
            ),
            maneuver_manager=request.maneuver_manager,
        )

    @staticmethod
    def _timed(name, operation, observer):
        started_s = time.monotonic()
        result = operation()
        if observer is not None:
            observer(name, time.monotonic() - started_s)
        return result

    @staticmethod
    def _apply_lateral_handoff(
        conflict_resolution,
        *,
        reference_provider,
        reset_lane_change,
    ) -> None:
        lateral_ownership = conflict_resolution.lateral_ownership
        handoff = lateral_ownership.handoff
        if handoff.action != "release":
            return
        released, release_result = reference_provider.release(
            "lane_change",
            event=str(lateral_ownership.reference_release_event),
        )
        if not released and reference_provider.snapshot("lane_change").active:
            raise RuntimeError(
                "lane-change semantic ownership was released but its "
                "reference remained active: " + str(release_result)
            )
        if callable(reset_lane_change):
            reset_lane_change(reason=str(handoff.reason))
