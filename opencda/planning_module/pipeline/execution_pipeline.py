"""Executable, bridge-independent CP-X planning stage owner."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from .perception_stage import PerceptionStage, PerceptionStageResult
from .runtime_input_stage import RuntimeInputStage, RuntimeTickSnapshot
from .speed_planner import (
    effective_emergency_gap_m,
)
from .behavior_stage import ConflictResolutionRequest, BehaviorOverrideRequest
from .behavior_reference_finalization_stage import (
    BehaviorReferenceFinalizationRequest,
    BehaviorReferenceFinalizationStage,
)


@dataclass(frozen=True)
class PlanningCycle:
    """Immutable normalized inputs and the initial longitudinal safety intent."""

    tick: RuntimeTickSnapshot
    perception: PerceptionStageResult
    emergency_gap_m: float
    emergency_stop_required: bool
    requested_speed_mps: float

    @property
    def object_snapshots(self):
        return self.perception.fused_objects

    @property
    def mpc_object_snapshots(self):
        return self.perception.mpc_objects

    @property
    def local_object_snapshots(self):
        return self.perception.local_objects


@dataclass(frozen=True)
class TrajectoryAdmission:
    """Published reference together with its sole MPC admission decision."""

    publication: Any
    entry: Any
    control_context: Any
    mode_transition_reason: str = ""

    def trace_fields(self) -> dict[str, object]:
        fields = dict(self.publication.debug_fields)
        fields.update(self.entry.trace_fields())
        return fields


@dataclass(frozen=True)
class NominalPlanningRequest:
    """Inputs for the behavior -> destination -> speed stage chain.

    Runtime/platform adaptation is deliberately absent.  The OpenCDA bridge
    supplies an already-normalized behavior request and route state; this
    stage owns the ordering and data hand-off between the three planning
    owners.
    """

    behavior_request: Any
    route_status: Any
    route_revision: str
    ego_speed_mps: float
    current_state: Sequence[float]
    fallback_lane_id: int


@dataclass(frozen=True)
class NominalPlanningFrame:
    """Resolved nominal plan before reference publication/MPC admission."""

    destination_state: Tuple[float, ...]
    reference_samples: Tuple[Mapping[str, Any], ...]
    behavior_stage_result: Any
    behavior_debug: Mapping[str, Any]
    reference_debug: Mapping[str, Any]
    speed_target: Any
    cav_resolution: Any
    mission_finished: bool
    failure_reason: str = ""

    @property
    def behavior(self):
        return self.behavior_stage_result.decision

    def mutable_destination_state(self):
        return list(self.destination_state)

    def mutable_reference(self):
        return [dict(sample) for sample in self.reference_samples]

    def mutable_behavior_debug(self):
        return dict(self.behavior_debug)

    def mutable_reference_debug(self):
        return dict(self.reference_debug)


@dataclass(frozen=True)
class ScenarioPlanningFrameRequest:
    """Immutable adapter-frame view consumed by scenario observation."""

    adapter_output: Any
    traffic_memory: Any
    route_manager: Any
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    current_lane_id: int
    cruise_speed_mps: float
    sim_time_s: float
    config: Mapping[str, object]
    boundary_recovery_request: Any = None


@dataclass(frozen=True)
class BehaviorContextRequest:
    """Frozen inputs for route, scenario and lateral-conflict resolution."""

    adapter_output: Any
    local_map_snapshot: Any
    route_manager: Any
    maneuver_manager: Any
    reference_provider: Any
    traffic_memory: Any
    opportunistic_request: Any
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    planning_speed_mps: float
    current_lane_id: int
    sim_time_s: float
    cruise_speed_mps: float
    mpc_dt_s: float
    lane_width_m: float
    config: Mapping[str, object]
    boundary_recovery_request: Any = None


@dataclass(frozen=True)
class BehaviorContextFrame:
    """One authoritative route/scenario/conflict view for a planning tick."""

    route_behavior: Any
    scenario_observation: Any
    conflict_resolution: Any
    route_replan_attempted: bool = False
    route_replan_succeeded: bool = False
    route_replan_reason: str = ""


@dataclass(frozen=True)
class ExecutableBehaviorRequest:
    """Inputs needed to freeze a behavior command for reference generation."""

    command_request: Any
    scenario_decision: Any
    lane_change_authorized: bool
    lane_change_gate_reason: str
    lane_change_authorization_reason: str
    prepare_reference_lock: bool
    lane_change_commitment_active: bool
    route_advanced_to_lane_change: bool
    route_current_road_option: str
    ego_in_junction: bool
    stop_goal_active: bool
    cruise_speed_mps: float
    scenario_reason: str


@dataclass(frozen=True)
class ExecutableBehaviorFrame:
    """Typed, override-complete behavior consumed by geometry and speed."""

    command_frame: Any
    override: Any
    turn_prepare_speed_suppressed: bool
    scenario_speed_cap_active: bool

    @property
    def command(self):
        return self.command_frame.command

    @property
    def decision(self) -> str:
        return str(self.override.decision)

    @property
    def target_lane_id(self) -> int:
        return int(self.override.target_lane_id)

    @property
    def phase(self) -> str:
        return str(self.override.phase)

    @property
    def stop_goal_active(self) -> bool:
        return bool(self.override.stop_goal_active)


@dataclass(frozen=True)
class CooperativePlanningFrame:
    """CAV resolution and its single longitudinal handoff."""

    cav_resolution: Any
    lane_change_deferred: bool


class PlanningPipeline:
    """Own and sequence planning stages without owning OpenCDA runtime I/O."""

    def __init__(
        self,
        *,
        runtime_input: RuntimeInputStage,
        perception: PerceptionStage,
        behavior: Any,
        scenario: Any,
        static_obstacle: Any,
        control_safety: Any,
        speed: Any,
        destination_speed: Any,
        reference_publication: Any,
        mpc_entry: Any,
        mpc_execution: Any = None,
        fallback: Any = None,
        behavior_reference_execution: Any = None,
        reference_planning: Any = None,
        mpc_cost_profile: Any = None,
        cooperative: Any = None,
        control_finalization: Any = None,
    ) -> None:
        self._runtime_input = runtime_input
        self._perception = perception
        self.behavior = behavior
        self.scenario = scenario
        self.static_obstacle = static_obstacle
        self.control_safety = control_safety
        self.speed = speed
        self.destination_speed = destination_speed
        self.reference_publication = reference_publication
        self.mpc_entry = mpc_entry
        self.mpc_execution = mpc_execution
        self.fallback = fallback
        self.behavior_reference_execution = behavior_reference_execution
        self.reference_planning = reference_planning
        self.mpc_cost_profile = mpc_cost_profile
        self.cooperative = cooperative
        self.control_finalization = control_finalization
        self.behavior_reference_finalization = (
            BehaviorReferenceFinalizationStage(
                speed=speed,
                mpc_cost_profile=mpc_cost_profile,
                reference_planning=reference_planning,
                behavior=behavior,
            )
        )

    def attach_cooperative(self, cooperative: Any) -> None:
        """Complete late assembly after the route owner exists."""

        if self.cooperative is not None:
            raise RuntimeError("cooperative stage is already configured")
        self.cooperative = cooperative

    def begin_tick(
        self, *, timestamp_s: float, ego_transform: Any, ego_speed_kmh: float
    ) -> RuntimeTickSnapshot:
        return self._runtime_input.build(
            timestamp_s=float(timestamp_s),
            ego_transform=ego_transform,
            ego_speed_kmh=float(ego_speed_kmh),
        )

    def perceive(
        self,
        tick: RuntimeTickSnapshot,
        *,
        detected_objects: Any,
        cp_payload: Mapping[str, Any],
        ignore_dynamic_objects: bool,
    ) -> PerceptionStageResult:
        return self._perception.build(
            detected_objects=detected_objects,
            cp_payload=cp_payload,
            ego_location=tick.ego_location,
            ego_yaw_rad=float(tick.ego_yaw_rad),
            timestamp_s=float(tick.timestamp_s),
            ignore_dynamic_objects=bool(ignore_dynamic_objects),
        )

    def begin_cycle(
        self,
        *,
        timestamp_s: float,
        ego_transform: Any,
        ego_speed_kmh: float,
        detected_objects: Any,
        cp_payload: Mapping[str, Any],
        ignore_dynamic_objects: bool,
        cruise_speed_mps: float,
        base_emergency_gap_m: float,
        emergency_standstill_buffer_m: float,
        following_time_headway_s: float,
    ) -> PlanningCycle:
        """Create the single authoritative input snapshot for one planner tick."""

        tick = self.begin_tick(
            timestamp_s=timestamp_s,
            ego_transform=ego_transform,
            ego_speed_kmh=ego_speed_kmh,
        )
        perception = self.perceive(
            tick,
            detected_objects=detected_objects,
            cp_payload=cp_payload,
            ignore_dynamic_objects=ignore_dynamic_objects,
        )
        emergency_gap_m = effective_emergency_gap_m(
            base_emergency_gap_m=max(0.5, float(base_emergency_gap_m)),
            ego_speed_mps=float(tick.ego_speed_mps),
            front_obstacle_speed_mps=perception.front_actor_speed_mps,
            standstill_buffer_m=max(0.0, float(emergency_standstill_buffer_m)),
            time_headway_s=max(0.1, float(following_time_headway_s)),
        )
        stop_required = bool(
            perception.front_gap_m is not None
            and float(perception.front_gap_m) <= float(emergency_gap_m)
        )
        return PlanningCycle(
            tick=tick,
            perception=perception,
            emergency_gap_m=float(emergency_gap_m),
            emergency_stop_required=stop_required,
            requested_speed_mps=(
                0.0 if stop_required else max(0.0, float(cruise_speed_mps))
            ),
        )

    def front_gap(self, **kwargs):
        return self._perception.front_gap(**kwargs)

    def evaluate_destination(self, **kwargs):
        return self.destination_speed.evaluate(**kwargs)

    def apply_destination(self, **kwargs):
        return self.destination_speed.apply(**kwargs)

    def finalize_behavior(self, **kwargs):
        return self.behavior.finalize(**kwargs)

    def finalize_behavior_frame(self, **kwargs):
        return self.behavior.finalize_planning_frame(**kwargs)

    def destination_stop_behavior(self, result):
        return self.behavior.destination_stop(result)

    def authorize_route_lane_change(self, request, *, maneuver_manager):
        return self.behavior.authorize_route_lane_change(
            request, maneuver_manager=maneuver_manager
        )

    def resolve_route_lane_change(self, context, **kwargs):
        return self.behavior.resolve_route_lane_change(context, **kwargs)

    def resolve_conflicts(self, request, **kwargs):
        return self.behavior.resolve_conflicts(request, **kwargs)

    def prepare_route_lane_change(self, **kwargs):
        return self.behavior.prepare_route_lane_change(**kwargs)

    def resolve_route_context(self, **kwargs):
        return self.behavior.resolve_route_context(**kwargs)

    def observe_planning_frame(
        self, request: ScenarioPlanningFrameRequest, *,
        resolve_actor_state: Callable[..., Any],
        project_stop_target: Callable[..., Any],
    ):
        """Resolve traffic memory and turn context from one frozen frame."""

        frame = request.adapter_output.frame
        route = frame.planning.route
        traffic = frame.planning.traffic_control
        raw_stop_target = (
            traffic.stop_target.as_dict() if traffic.stop_target.active else None
        )
        return self.scenario.observe_planning_context(
            raw_traffic_state=str(traffic.signal_state),
            raw_stop_target=raw_stop_target,
            signal_context=dict(request.adapter_output.signal_context),
            traffic_memory=request.traffic_memory,
            resolve_actor_state=resolve_actor_state,
            project_stop_target=project_stop_target,
            prepare_turn_context=lambda: self.behavior.prepare_turn_scenario_context(
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
            current_lane_id=int(request.current_lane_id),
            ego_in_junction=bool(frame.map_lane.in_junction),
            current_road_option=str(route.current_road_option),
            next_macro_maneuver=str(route.next_macro_maneuver),
            virtual_stop_distance_m=float(request.config.get(
                "full_latched_virtual_stop_distance_m", 12.0
            )),
            boundary_recovery_request=request.boundary_recovery_request,
        )

    def resolve_behavior_context(
        self,
        request: BehaviorContextRequest,
        *,
        resolve_actor_state: Callable[..., Any],
        project_stop_target: Callable[..., Any],
        attempt_turn_replan: Callable[[str], tuple],
        reset_lane_change: Optional[Callable[..., Any]] = None,
        observe_stage_duration: Optional[Callable[[str, float], None]] = None,
    ) -> BehaviorContextFrame:
        """Resolve route, scenario and lateral ownership in one order.

        These stages consume the same frozen adapter frame.  Keeping their
        ordering here prevents the bridge from rebuilding route or scenario
        facts between calls, while callbacks isolate the few state-changing
        ports (reroute and behavior-planner reset).
        """

        def run_stage(name, operation):
            started_s = time.monotonic()
            result = operation()
            if observe_stage_duration is not None:
                observe_stage_duration(name, time.monotonic() - started_s)
            return result

        frame = request.adapter_output.frame
        route_behavior = run_stage(
            "sub_resolve_route_context",
            lambda: self.resolve_route_context(
                adapter_output=request.adapter_output,
                local_map_snapshot=request.local_map_snapshot,
                route_manager=request.route_manager,
                maneuver_manager=request.maneuver_manager,
                reference_provider=request.reference_provider,
                current_lane_id=int(request.current_lane_id),
                available_lane_ids=tuple(frame.map_lane.allowed_lane_ids),
                ego_x_m=float(request.ego_location.x),
                ego_y_m=float(request.ego_location.y),
                ego_heading_rad=float(request.ego_yaw_rad),
                ego_speed_mps=float(request.ego_speed_mps),
                config=request.config,
            ),
        )

        replan_attempted = False
        replan_succeeded = False
        replan_reason = ""
        if str(route_behavior.replan_reason):
            replan_attempted, replan_succeeded, replan_reason = (
                attempt_turn_replan(str(route_behavior.replan_reason))
            )

        scenario_observation = run_stage(
            "sub_observe_planning_frame",
            lambda: self.observe_planning_frame(
                ScenarioPlanningFrameRequest(
                    adapter_output=request.adapter_output,
                    traffic_memory=request.traffic_memory,
                    route_manager=request.route_manager,
                    ego_location=request.ego_location,
                    ego_yaw_rad=float(request.ego_yaw_rad),
                    ego_speed_mps=float(request.ego_speed_mps),
                    current_lane_id=int(request.current_lane_id),
                    cruise_speed_mps=float(request.cruise_speed_mps),
                    sim_time_s=float(request.sim_time_s),
                    config=request.config,
                    boundary_recovery_request=request.boundary_recovery_request,
                ),
                resolve_actor_state=resolve_actor_state,
                project_stop_target=project_stop_target,
            ),
        )
        turn_context = scenario_observation.turn_context
        conflict_resolution = run_stage(
            "sub_resolve_conflicts",
            lambda: self.resolve_conflicts(
                ConflictResolutionRequest(
                    route_authorization=route_behavior.authorization,
                    opportunistic_request=request.opportunistic_request,
                    owner_state=str(scenario_observation.scenario.decision.state),
                    ego_speed_mps=float(request.ego_speed_mps),
                    planning_speed_mps=float(request.planning_speed_mps),
                    lane_change_duration_s=max(0.1, float(request.config.get(
                        "candidate_lane_change_normal_duration_s", 4.0
                    ))),
                    dt_s=float(request.mpc_dt_s),
                    lane_width_m=float(request.lane_width_m),
                    distance_to_turn_m=float(turn_context.distance_m),
                    config=request.config,
                    ego_location=request.ego_location,
                    ego_yaw_rad=float(request.ego_yaw_rad),
                ),
                maneuver_manager=request.maneuver_manager,
            ),
        )

        lateral_ownership = conflict_resolution.lateral_ownership
        handoff = lateral_ownership.handoff
        if handoff.action == "release":
            released, release_result = request.reference_provider.release(
                "lane_change",
                event=str(lateral_ownership.reference_release_event),
            )
            if (
                not released
                and request.reference_provider.snapshot("lane_change").active
            ):
                raise RuntimeError(
                    "lane-change semantic ownership was released but its "
                    "reference remained active: " + str(release_result)
                )
            if callable(reset_lane_change):
                reset_lane_change(reason=str(handoff.reason))

        return BehaviorContextFrame(
            route_behavior=route_behavior,
            scenario_observation=scenario_observation,
            conflict_resolution=conflict_resolution,
            route_replan_attempted=bool(replan_attempted),
            route_replan_succeeded=bool(replan_succeeded),
            route_replan_reason=str(replan_reason),
        )

    def resolve_static_obstacle(self, **kwargs):
        return self.static_obstacle.evaluate(**kwargs)

    def reset_route_lane_change_authorization(self):
        self.behavior.reset_route_lane_change_authorization()

    def produce_behavior_command(self, request, **kwargs):
        return self.behavior.produce_command_from_frame(request, **kwargs)

    def apply_behavior_overrides(self, request):
        return self.behavior.apply_overrides(request)

    def resolve_executable_behavior(
        self,
        request: ExecutableBehaviorRequest,
        *,
        behavior_planner: Any,
        reference_map: Any,
        nearest_front_obstacles: Callable[..., Any],
        attempt_replan: Callable[..., Any],
        object_track_id: Callable[..., Any],
        reset_lane_change: Optional[Callable[..., Any]] = None,
        observe_stage_duration: Optional[Callable[[str, float], None]] = None,
    ) -> ExecutableBehaviorFrame:
        """Produce and override behavior once before geometry generation."""

        started_s = time.monotonic()
        command_frame = self.produce_behavior_command(
            request.command_request,
            behavior_planner=behavior_planner,
            static_obstacle_stage=self.static_obstacle,
            reference_map=reference_map,
            nearest_front_obstacles=nearest_front_obstacles,
            attempt_replan=attempt_replan,
            object_track_id=object_track_id,
        )
        if observe_stage_duration is not None:
            observe_stage_duration(
                "sub_produce_behavior_command", time.monotonic() - started_s
            )

        command = command_frame.command
        scenario = request.scenario_decision
        commitment_active = bool(request.lane_change_commitment_active)
        turn_prepare_suppressed = bool(
            commitment_active
            and str(scenario.state).strip().upper() == "PREPARE_TURN"
            and not bool(scenario.stop_goal_active)
        )
        scenario_speed_cap_active = bool(
            scenario.speed_cap_mps is not None
            and float(scenario.speed_cap_mps) < float(request.cruise_speed_mps)
            and not turn_prepare_suppressed
        )
        route_turn_decision = self._route_option_turn_decision(
            current_road_option=str(request.route_current_road_option)
        )
        if bool(request.route_advanced_to_lane_change):
            route_turn_decision = ""

        static_obstacle = command.static_obstacle_result
        override = self.apply_behavior_overrides(BehaviorOverrideRequest(
            decision=str(command.decision),
            target_lane_id=int(command.target_lane_id),
            phase=str(command.phase),
            current_lane_id=int(request.command_request.current_lane_id),
            lane_change_authorized=bool(request.lane_change_authorized),
            opportunistic_lane_change_allowed=bool(
                command.opportunistic_lane_change_allowed
            ),
            lane_change_gate_reason=str(request.lane_change_gate_reason),
            lane_change_authorization_reason=str(
                request.lane_change_authorization_reason
            ),
            prepare_reference_lock=bool(request.prepare_reference_lock),
            scenario_speed_cap_active=bool(scenario_speed_cap_active),
            scenario_reason=str(request.scenario_reason),
            scenario_override_decision=str(
                scenario.behavior_override_decision or ""
            ),
            scenario_override_phase=str(scenario.behavior_override_lc_state),
            scenario_stop_required=bool(scenario.stop_goal_active),
            local_avoidance_active=bool(static_obstacle.local_avoidance_active),
            ego_in_junction=bool(request.ego_in_junction),
            lane_change_commitment_active=commitment_active,
            route_turn_decision=str(route_turn_decision),
            route_current_road_option=str(request.route_current_road_option),
            stop_goal_active=bool(request.stop_goal_active),
        ))
        if str(override.reset_lane_change_reason) and callable(reset_lane_change):
            reset_lane_change(reason=str(override.reset_lane_change_reason))
        return ExecutableBehaviorFrame(
            command_frame=command_frame,
            override=override,
            turn_prepare_speed_suppressed=bool(turn_prepare_suppressed),
            scenario_speed_cap_active=bool(scenario_speed_cap_active),
        )

    @staticmethod
    def _route_option_turn_decision(*, current_road_option: str) -> str:
        route_option = str(current_road_option or "").strip().upper()
        if route_option == "LEFT":
            return "intersection_turn_left"
        if route_option == "RIGHT":
            return "intersection_turn_right"
        return ""

    def resolve_nominal_plan(
        self,
        request: NominalPlanningRequest,
        *,
        planner: Callable[..., Any],
        observe_stage_duration: Optional[Callable[[str, float], None]] = None,
    ) -> NominalPlanningFrame:
        """Run the nominal stage chain with one typed output contract.

        This is the only sequencing owner for behavior/reference recovery,
        destination completion, and final nominal speed resolution.  The
        optional observer preserves bridge-side profiling without giving the
        bridge ownership of the stage ordering.
        """

        def run_stage(name, operation):
            started_s = time.monotonic()
            result = operation()
            if observe_stage_duration is not None:
                observe_stage_duration(name, time.monotonic() - started_s)
            return result

        behavior_reference = run_stage(
            "execute_behavior_reference",
            lambda: self.execute_behavior_reference(
                request.behavior_request, planner=planner
            ),
        )
        destination_application = run_stage(
            "apply_destination",
            lambda: self.apply_destination(
                route_status=request.route_status,
                route_revision=str(request.route_revision),
                ego_speed_mps=float(request.ego_speed_mps),
                current_state=list(request.current_state),
                destination_state=list(behavior_reference.destination_state),
                reference_samples=[
                    dict(item) for item in behavior_reference.reference_samples
                ],
                behavior_stage_result=behavior_reference.behavior_stage_result,
                reference_debug=dict(behavior_reference.reference_debug),
                fallback_lane_id=int(request.fallback_lane_id),
            ),
        )

        behavior_stage_result = destination_application.behavior_stage_result
        behavior = behavior_stage_result.decision
        behavior_debug = behavior_stage_result.mutable_diagnostics()
        behavior_debug.update(behavior.as_debug_fields())
        reference_debug = destination_application.mutable_reference_debug()
        destination_state = destination_application.mutable_destination_state()
        reference_samples = destination_application.mutable_reference()
        destination_constraint = destination_application.stage.constraint

        speed_frame = run_stage(
            "resolve_speed",
            lambda: self.resolve_speed(
                behavior=behavior,
                speed_plan=behavior_reference.speed_plan,
                additional_constraints=(
                    ()
                    if destination_constraint is None
                    else (destination_constraint,)
                ),
                destination_state=destination_state,
                reference_samples=reference_samples,
            ),
        )
        reference_debug.update(speed_frame.trace_fields())
        return NominalPlanningFrame(
            destination_state=tuple(speed_frame.ceiling.destination_state),
            reference_samples=tuple(
                dict(item) for item in speed_frame.ceiling.reference_samples
            ),
            behavior_stage_result=behavior_stage_result,
            behavior_debug=dict(behavior_debug),
            reference_debug=dict(reference_debug),
            speed_target=speed_frame.target,
            cav_resolution=behavior_reference.cav_resolution,
            mission_finished=bool(destination_application.finished),
            failure_reason=str(behavior_reference.failure_reason or ""),
        )

    def resolve_speed(
        self, *, behavior, speed_plan, additional_constraints,
        destination_state, reference_samples,
    ):
        return self.speed.resolve_frame(
            behavior=behavior,
            speed_plan=speed_plan,
            additional_constraints=additional_constraints,
            destination_state=destination_state,
            reference_samples=reference_samples,
        )

    def constrain_speed_plan(self, speed_plan, constraint, **kwargs):
        return self.speed.constrain_plan(speed_plan, constraint, **kwargs)

    def constrain_turn_speed_from_reference(
        self, speed_plan, *, reference_provider, config
    ):
        """Apply the provider-owned persistent turn geometry to speed once."""

        constraint = self.speed.turn_curvature_constraint(
            reference_provider.turn_master_curvature_1pm(),
            config,
            distance_to_turn_m=speed_plan.upcoming_turn_distance_m,
        )
        return self.speed.constrain_plan(speed_plan, constraint), constraint

    def propose_speed(self, **kwargs):
        return self.speed.propose(**kwargs)

    @property
    def destination_stop_latched(self) -> bool:
        return bool(self.destination_speed.stop_latched)

    @property
    def destination_mission_complete(self) -> bool:
        return bool(self.destination_speed.mission_complete)

    def publish_reference(self, **kwargs):
        return self.reference_publication.run(**kwargs)

    def prepare_trajectory_execution(
        self,
        *,
        publication_kwargs,
        behavior,
        stop_goal_active,
        ego_speed_mps,
        ego_x_m,
        ego_y_m,
        ego_yaw_rad,
        front_gap_actor_id,
        candidate_status,
        candidate_name,
        candidate_reason,
        reset_control_buffer=None,
    ) -> TrajectoryAdmission:
        publication = self.reference_publication.run(**publication_kwargs)
        mode_transition_reason = self.mpc_entry.apply_behavior_mode_transition(
            behavior=behavior,
            stop_goal_active=bool(stop_goal_active),
            reset_control_buffer=reset_control_buffer,
        )
        entry = self.mpc_entry.evaluate(
            candidate_status=candidate_status,
            candidate_name=candidate_name,
            candidate_reason=candidate_reason,
            final_reference_accepted=bool(publication.gate.accepted),
            final_reference_reason=str(publication.gate.reason),
            behavior_decision=str(behavior.maneuver),
            stop_goal_active=bool(stop_goal_active),
            ego_speed_mps=float(ego_speed_mps),
        )
        control_context = self.mpc_entry.prepare_control_context(
            behavior=behavior,
            reference_source=str(
                publication.debug_fields.get("reference_source", "")
            ),
            stop_goal_active=bool(stop_goal_active),
            front_gap_actor_id=str(front_gap_actor_id),
            reference_samples=publication.mutable_samples(),
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_yaw_rad=float(ego_yaw_rad),
            mode_transition_reason=str(mode_transition_reason),
        )
        return TrajectoryAdmission(
            publication=publication,
            entry=entry,
            control_context=control_context,
            mode_transition_reason=str(mode_transition_reason),
        )

    def evaluate_mpc_entry(self, **kwargs):
        return self.mpc_entry.evaluate(**kwargs)

    def prepare_mpc_control_context(self, **kwargs):
        return self.mpc_entry.prepare_control_context(**kwargs)

    def execute_mpc(self, request, **kwargs):
        if self.mpc_execution is None:
            raise RuntimeError("MPC execution stage is not configured")
        return self.mpc_execution.run(request, **kwargs)

    def apply_control_safety(self, **kwargs):
        return self.control_safety.run(**kwargs)

    def finalize_control(self, request, **kwargs):
        if self.control_finalization is None:
            raise RuntimeError("control finalization stage is not configured")
        return self.control_finalization.run(request, **kwargs)

    @staticmethod
    def explain_decision(diagnostics):
        from .decision_record import decision_record_from_diagnostics

        return decision_record_from_diagnostics(diagnostics)

    def record_valid_trajectory(self, trajectory, **kwargs):
        if self.fallback is None:
            return False
        return self.fallback.record_valid(trajectory, **kwargs)

    def resolve_fallback(self, **kwargs):
        if self.fallback is None:
            raise RuntimeError("fallback stage is not configured")
        return self.fallback.resolve(**kwargs)

    def bounded_safe_stop(self, **kwargs):
        if self.fallback is None:
            raise RuntimeError("fallback stage is not configured")
        return self.fallback.bounded_safe_stop(**kwargs)

    def resolve_candidate_failure(self, **kwargs):
        if self.fallback is None:
            raise RuntimeError("fallback stage is not configured")
        return self.fallback.resolve_candidate_failure(**kwargs)

    def execute_behavior_reference(self, request, *, planner):
        if self.behavior_reference_execution is None:
            raise RuntimeError("behavior/reference execution stage is not configured")
        return self.behavior_reference_execution.run(request, planner=planner)

    def arbitrate_candidates(self, request, **kwargs):
        if self.reference_planning is None:
            raise RuntimeError("reference planning stage is not configured")
        return self.reference_planning.arbitrate_candidates(request, **kwargs)

    def build_behavior_reference(self, request):
        if self.reference_planning is None:
            raise RuntimeError("reference planning stage is not configured")
        return self.reference_planning.build_behavior_reference(request)

    def finalize_post_turn_reference(self, request):
        if self.reference_planning is None:
            raise RuntimeError("reference planning stage is not configured")
        return self.reference_planning.finalize_post_turn(request)

    def apply_mpc_cost_profile(self, **kwargs):
        if self.mpc_cost_profile is None:
            raise RuntimeError("MPC cost-profile stage is not configured")
        return self.mpc_cost_profile.apply(**kwargs)

    def resolve_cooperative(self, request) -> CooperativePlanningFrame:
        if self.cooperative is None:
            raise RuntimeError("cooperative arbitration stage is not configured")
        cav_resolution, deferred = self.cooperative.run(request)
        return CooperativePlanningFrame(
            cav_resolution=cav_resolution,
            lane_change_deferred=bool(deferred),
        )

    def constrain_speed_from_cooperative(self, speed_plan, frame):
        if frame.cav_resolution is None:
            return speed_plan
        return self.speed.constrain_plan(
            speed_plan, frame.cav_resolution.speed_constraint
        )

    def finalize_behavior_reference(
        self, request: BehaviorReferenceFinalizationRequest
    ):
        return self.behavior_reference_finalization.run(request)

    def cooperative_conflict_reference(self, **kwargs):
        if self.reference_planning is None:
            raise RuntimeError("reference planning stage is not configured")
        return self.reference_planning.cooperative_conflict_reference(**kwargs)
