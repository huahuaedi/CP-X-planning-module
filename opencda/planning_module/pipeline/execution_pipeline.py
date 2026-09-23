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
from .behavior_stage import BehaviorCommandFrameRequest, BehaviorOverrideRequest
from .behavior_reference_finalization_stage import (
    BehaviorReferenceFinalizationPreparationRequest,
    BehaviorReferenceFinalizationRequest,
    BehaviorReferenceFinalizationStage,
)
from .speed_planning_stage import (
    SpeedPlanningPreparationRequest,
    SpeedPlanningRequest,
    SpeedPlanningStage,
)
from .route_update_stage import RouteUpdateRequest, RouteUpdateStage
from .planning_context_stage import PlanningContextRequest, PlanningContextStage
from .behavior_reference_execution_stage import BehaviorReferenceRequest
from .control_finalization_stage import ControlFinalizationRequest
from .mpc_execution_stage import MPCExecutionRequest
from .turn_road_envelope import rolling_turn_envelope_payload_world
from .planner_diagnostics_stage import PlannerDiagnosticsStage
from .output import BehaviorCommand, PlannerDiagnostics, PlannerOutput


@dataclass(frozen=True)
class PlanningCycle:
    """Immutable normalized inputs and longitudinal safety evidence.

    ``emergency_stop_required`` records a proximity violation for the final
    safety owner.  It is deliberately not a behavior/reference request: a raw
    front gap must not create a second nominal stop planner ahead of the CAV
    corridor and ``SafetySupervisor``.
    """

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
class PlanningTickAdapters:
    """Everything ``PlanningPipeline.plan()`` needs that is not planning logic.

    Three kinds of thing live here, all owned and supplied by
    CPXMPCPlannerBridge per call (never held): stable collaborators (mpc,
    route_manager, maneuver_manager, config, ...), coordinate/actuator/
    message-format conversions (body_frame_xy, safe_stop_control,
    apply_velocity_steering, ...), and OpenCDA/CARLA side effects (CP/V2X
    publish, debug draw, the mission-finished flag).  None of it decides
    anything; see cpx_mpc_planner.py's execute_planning_pipeline() for how
    each field is built.
    """

    mpc: Any
    route_manager: Any
    maneuver_manager: Any
    config: Mapping[str, Any]
    waypoint_backend: str
    route_context: Any
    ego_vehicle: Any
    v2x_manager: Any
    safety_manager: Any
    control_buffer: Any
    safety_supervisor: Any
    cp_provider: Any
    architecture_profile: Any
    vehicle_dynamics: Any
    reference_line_provider: Any
    prediction_mode: str
    fallback_policy: str
    fallback_policy_warning: str
    global_planner_backend: str
    global_planner_backend_warning: str
    last_accel_mps2: float
    last_steer_rad: float
    reset_control_buffer: Optional[Callable[..., Any]]
    planner: Callable[..., Any]
    accum_stage_ms: Callable[[str, float], None]
    body_frame_xy: Callable[..., Tuple[float, float]]
    active_global_route_points: Callable[[], Any]
    maybe_capture_execute_mpc_frame: Callable[..., None]
    mpc_object_snapshots_with_prediction: Callable[..., Any]
    safe_stop_control: Callable[..., Any]
    pedal_control: Callable[..., Any]
    set_actuator_context: Callable[..., None]
    acceleration_from_control: Callable[..., float]
    steering_from_control: Callable[..., float]
    apply_velocity_steering: Callable[..., tuple]
    control_factory: Callable[..., Any]
    road_boundary_measure: Callable[..., Any]
    update_boundary_recovery: Callable[..., Any]
    reset_boundary_recovery: Callable[..., Any]
    warn_fallback: Callable[[str], None]
    log_behavior_reference_failure: Callable[[str], None]
    cav_conflict_enabled: bool
    cav_intent_broadcast_enabled: bool
    publish_cav_intent: Callable[..., None]
    mark_mission_finished: Callable[[], None]
    last_mpc_trajectory_points: Callable[[], Any]
    draw_world_debug_primitives: Callable[..., None]
    static_obstacle_blocked_lane_id: Callable[[], Any]
    route_replan_attempt_count: Callable[[], int]
    display_global_route_points: Callable[[], list]
    route_points_for_display: Callable[[str], list]
    perception_diagnostics: Callable[[], dict]
    cooperative_actor_evidence: Callable[..., dict]
    update_evaluation_metrics: Callable[..., dict]
    diagnostics_owner: Any


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
class ExecutableBehaviorPreparationRequest:
    """Typed planning outputs needed to produce one executable behavior."""

    planning_context: Any
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    cruise_speed_mps: float
    sim_time_s: float
    mpc_feedback: Mapping[str, object]
    lane_change_reference_active: bool
    lane_change_commitment_active: bool
    stop_goal_active: bool
    max_deceleration_mps2: float
    prepare_reference_lock: bool
    route_recovery_requested: bool
    static_obstacle_mpc_stall_failure_count: int
    config: Mapping[str, object]
    runtime_config: Mapping[str, object]


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

    @property
    def candidate_frame(self):
        return self.command_frame.candidate_frame

    @property
    def cooperative_proposal(self):
        return self.command_frame.cooperative_proposal

    @property
    def candidate_lane_ids(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self.command_frame.candidate_lane_ids)

    @property
    def lane_alignment_valid(self) -> bool:
        return bool(self.command.lane_alignment_valid)

    @property
    def lane_lateral_error_m(self) -> float:
        return float(self.command.lane_lateral_error_m)

    @property
    def lane_heading_error_rad(self) -> float:
        return float(self.command.lane_heading_error_rad)

    @property
    def static_obstacle_result(self):
        return self.command.static_obstacle_result

    @property
    def semantic_response(self):
        return self.command.semantic_response

    @property
    def opportunistic_lane_change_allowed(self) -> bool:
        return bool(self.command.opportunistic_lane_change_allowed)

    @property
    def override_reason(self) -> str:
        return str(self.override.reason)


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
        planner_input_adapter: Any = None,
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
        self.speed_planning = SpeedPlanningStage(
            perception=perception,
            speed=speed,
        )
        self.route_update = RouteUpdateStage()
        self.planning_context = PlanningContextStage(
            input_adapter=planner_input_adapter,
            behavior=behavior,
            scenario=scenario,
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

    def update_route_from_cp(self, request: RouteUpdateRequest):
        return self.route_update.run(request)

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
            requested_speed_mps=max(0.0, float(cruise_speed_mps)),
        )

    def plan(self, cycle: PlanningCycle, adapters: "PlanningTickAdapters") -> "PlannerOutput":
        """Run one planning tick: PlanningCycle -> stage sequence -> PlannerOutput.

        ``cycle`` is this tick's normalized OpenCDA/CARLA input (see
        begin_cycle).  ``adapters`` bundles everything here that is not
        planning logic -- stable collaborators (mpc, route_manager,
        maneuver_manager, config), coordinate/actuator/message conversions,
        and OpenCDA/CARLA side effects (CP/V2X publish, debug draw, the
        mission-finished flag) -- all owned and supplied by the bridge.  This
        method sequences the same stage calls CPXMPCPlannerBridge used to
        make directly; see CPXMPCPlannerBridge.run_step's docstring for the
        full control-writing precedence this tick's result feeds into.
        """

        tick = cycle.tick
        perception = cycle.perception
        sim_time_s = float(tick.timestamp_s)
        ego_transform = tick.ego_transform
        ego_location = tick.ego_location
        ego_speed_mps = float(tick.ego_speed_mps)
        ego_yaw_rad = float(tick.ego_yaw_rad)
        measured_accel_mps2 = float(tick.measured_accel_mps2)
        cp_payload = dict(perception.cp_payload)
        object_snapshots = [dict(item) for item in cycle.object_snapshots]
        mpc_object_snapshots = [dict(item) for item in cycle.mpc_object_snapshots]
        local_object_snapshots = [dict(item) for item in cycle.local_object_snapshots]
        front_gap_m = perception.front_gap_m
        requested_speed_mps = float(cycle.requested_speed_mps)
        current_state = list(tick.current_state)

        nominal_frame = self.resolve_nominal_plan(
            NominalPlanningRequest(
                behavior_request=BehaviorReferenceRequest(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    ego_speed_mps=float(ego_speed_mps),
                    requested_speed_mps=float(requested_speed_mps),
                    object_snapshots=object_snapshots,
                    # Nominal behavior owns semantic stops (signals, route
                    # completion, obstacles).  Raw proximity is safety
                    # evidence and is consumed only in final control.
                    stop_goal_active=False,
                    cp_payload=cp_payload,
                    current_state=current_state,
                    sim_time_s=float(tick.timestamp_s),
                    route_revision=str(adapters.route_manager.route_revision),
                ),
                route_status=getattr(adapters.route_manager, "last_status", None),
                route_revision=str(adapters.route_manager.route_revision),
                ego_speed_mps=float(ego_speed_mps),
                current_state=current_state,
                fallback_lane_id=int(getattr(
                    adapters.route_context.local_map_snapshot, "ego_lane_id", 0
                )),
            ),
            planner=adapters.planner,
            observe_stage_duration=adapters.accum_stage_ms,
        )
        destination_state = nominal_frame.mutable_destination_state()
        lane_center_reference = nominal_frame.mutable_reference()
        behavior_stage_result = nominal_frame.behavior_stage_result
        behavior_decision = nominal_frame.behavior
        behavior_debug = nominal_frame.mutable_behavior_debug()
        reference_debug = nominal_frame.mutable_reference_debug()
        speed_target = nominal_frame.speed_target
        speed_ref_mps = float(speed_target.target_mps)
        cav_resolution = nominal_frame.cav_resolution
        if nominal_frame.mission_finished:
            adapters.mark_mission_finished()
        if nominal_frame.failure_reason:
            adapters.log_behavior_reference_failure(str(nominal_frame.failure_reason))
        # The typed behavior decision owns the final stop state. The raw
        # front-gap threshold is only an input proposal and must not re-latch
        # stop after candidate evaluation has selected a safe route maneuver.
        mpc_stop_goal_active = bool(behavior_decision.stop_required)
        behavior_decision_normalized = str(behavior_decision.maneuver).strip().lower()
        normal_stop_requested = behavior_decision_normalized in {
            "stop_at_intersection",
            "stop_sign",
        }
        emergency_brake_requested = (
            behavior_decision_normalized == "emergency_brake"
        )
        stop_target_forward_m_debug = ""
        stop_target_debug = behavior_decision.stop_target
        if bool(mpc_stop_goal_active) and isinstance(stop_target_debug, Mapping):
            try:
                stop_target_forward_m_debug, _ = adapters.body_frame_xy(
                    origin_x_m=float(ego_location.x),
                    origin_y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    target_x_m=float(stop_target_debug.get("x_m", stop_target_debug.get("x", ego_location.x))),
                    target_y_m=float(stop_target_debug.get("y_m", stop_target_debug.get("y", ego_location.y))),
                )
            except Exception:
                stop_target_forward_m_debug = ""

        candidate_status = str(reference_debug.get(
            "candidate_pipeline_selected_status", ""
        ))
        candidate_name = str(reference_debug.get(
            "candidate_pipeline_selected", ""
        ))
        candidate_reason = str(reference_debug.get(
            "candidate_pipeline_selected_reason", ""
        ))
        _ts_stage = time.monotonic()
        admission = self.prepare_trajectory_execution(
            publication_kwargs={
                "destination_state": destination_state,
                "reference_samples": lane_center_reference,
                "current_state": current_state,
                "ego_location": ego_location,
                "ego_yaw_rad": float(ego_yaw_rad),
                "ego_speed_mps": float(ego_speed_mps),
                "target_speed_mps": float(speed_ref_mps),
                "behavior": behavior_decision,
                "stop_goal_active": bool(mpc_stop_goal_active),
                "route_points": adapters.active_global_route_points(),
                "local_map": adapters.route_context.local_map_snapshot,
                "route_cursor": adapters.route_manager.route_cursor,
                "route_revision": str(adapters.route_manager.route_revision),
                "map_epoch": str(adapters.waypoint_backend),
                "reference_source": str(reference_debug.get(
                    "reference_source", "planning_reference"
                )),
                "candidate_status": candidate_status,
                "candidate_reason": candidate_reason,
                "heading_error_rad": (
                    math.radians(float(reference_debug["behavior_lane_heading_error_deg"]))
                    if reference_debug.get("behavior_lane_heading_error_deg", "") != ""
                    else float("nan")
                ),
                "committed_lane_change_tracking_active": bool(
                    str(adapters.maneuver_manager.lane_change.phase) == "executing"
                ),
            },
            behavior=behavior_decision,
            stop_goal_active=bool(mpc_stop_goal_active),
            ego_speed_mps=float(ego_speed_mps),
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_yaw_rad=float(ego_yaw_rad),
            front_gap_actor_id=str(reference_debug.get("front_gap_actor_id", "")),
            candidate_status=candidate_status,
            candidate_name=candidate_name,
            candidate_reason=candidate_reason,
            reset_control_buffer=adapters.reset_control_buffer,
        )
        adapters.accum_stage_ms("prepare_trajectory_execution", time.monotonic() - _ts_stage)
        mode_transition_guard_reason = str(admission.mode_transition_reason)
        _ts_stage = time.monotonic()
        publication_result = admission.publication
        destination_state = publication_result.mutable_destination()
        lane_center_reference = publication_result.mutable_samples()
        final_reference_gate = publication_result.gate
        mpc_reference_stabilizer_reason = str(publication_result.stabilizer_reason)
        reference_debug.update(admission.trace_fields())
        mpc_entry = admission.entry
        candidate_hard_gate_reason = str(mpc_entry.hard_gate_reason)
        stationary_traffic_stop_hold = bool(mpc_entry.stationary_stop_hold)
        mpc_control_context = admission.control_context
        destination_forward_m, destination_lateral_m = adapters.body_frame_xy(
            origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad), target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        reference_first_forward_m = ""
        reference_first_lateral_m = ""
        if lane_center_reference:
            first_reference = lane_center_reference[0]
            reference_first_forward_m, reference_first_lateral_m = adapters.body_frame_xy(
                origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(first_reference.get("x_ref_m", first_reference.get("x", ego_location.x))),
                target_y_m=float(first_reference.get("y_ref_m", first_reference.get("y", ego_location.y))),
            )
        mpc_status = str(getattr(adapters.mpc, "_last_status", ""))
        # MPC constrains jerk between the previous control input and the new
        # acceleration sequence. Seed that constraint with the acceleration
        # command actually sent last tick, not the measured vehicle response.
        # The latter contains actuator lag and can stay strongly negative
        # after the speed target has recovered, otherwise forcing every new
        # solve to continue braking until the vehicle is almost stationary.
        mpc_jerk_seed_accel_mps2 = float(adapters.last_accel_mps2)
        road_envelope_payload_world = rolling_turn_envelope_payload_world(
            config=adapters.config,
            mpc=adapters.mpc,
            vehicle=adapters.ego_vehicle,
            behavior_decision=str(behavior_decision.maneuver),
            reference_samples=lane_center_reference,
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
        )
        cav_result = cav_resolution
        cav_constraint_rows = ()
        cav_constraint_revision = ""
        cav_diagnostics = {}
        if cav_result is not None:
            cav_constraint_rows = tuple(cav_result.mpc_rows or ())
            cav_diagnostics = dict(cav_result.diagnostics or {})
            cav_constraint_revision = self.cooperative.schedule.constraint_revision(
                cav_diagnostics
            )
        corridor_infeasible_escalate = bool(
            self.cooperative.schedule.corridor_emergency_stop_required
        )
        adapters.maybe_capture_execute_mpc_frame(
            sim_time_s=float(sim_time_s),
            current_state=current_state,
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            current_acceleration_mps2=float(mpc_jerk_seed_accel_mps2),
            current_steering_rad=float(adapters.last_steer_rad),
            destination_state=destination_state,
            target_speed_mps=float(speed_ref_mps),
            stop_goal_active=bool(mpc_stop_goal_active),
            behavior_decision=behavior_decision,
            published_reference=lane_center_reference,
            mpc_object_snapshots=mpc_object_snapshots,
            cav_constraint_rows=cav_constraint_rows,
            cav_diagnostics=cav_diagnostics,
            cav_corridor=(
                None if cav_result is None
                else cav_result.constraint_corridor
            ),
        )
        adapters.accum_stage_ms("between_admission_and_mpc", time.monotonic() - _ts_stage)
        _ts_stage = time.monotonic()
        execution_result = self.execute_mpc(
            MPCExecutionRequest(
                sim_time_s=float(sim_time_s),
                current_state=current_state,
                destination_state=destination_state,
                reference_samples=lane_center_reference,
                object_snapshots=adapters.mpc_object_snapshots_with_prediction(
                    mpc_object_snapshots,
                    prediction_trajectories=reference_debug.get(
                        "prediction_trajectories", {}
                    ),
                ),
                current_acceleration_mps2=float(mpc_jerk_seed_accel_mps2),
                current_steering_rad=float(adapters.last_steer_rad),
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(speed_ref_mps),
                stop_goal_active=bool(mpc_stop_goal_active),
                behavior_maneuver=str(behavior_decision.maneuver),
                behavior_phase=str(behavior_decision.phase),
                hard_gate_reason=str(candidate_hard_gate_reason),
                stationary_stop_hold=bool(stationary_traffic_stop_hold),
                control_context=mpc_control_context,
                road_envelope_payload_world=road_envelope_payload_world,
                speed_crossing_deadband_mps=float(adapters.config.get(
                    "control_buffer_speed_crossing_deadband_mps", 0.15,
                )),
                corridor_rows=cav_constraint_rows,
                constraint_revision=str(cav_constraint_revision),
            ),
            safe_stop_control=adapters.safe_stop_control,
        )
        adapters.accum_stage_ms("execute_mpc", time.monotonic() - _ts_stage)
        mpc_jerk_seed_accel_mps2 = float(
            execution_result.jerk_seed_acceleration_mps2
        )
        _ts_stage = time.monotonic()
        finalized_control = self.finalize_control(
            ControlFinalizationRequest(
                execution=execution_result,
                behavior=behavior_decision,
                ego_transform=ego_transform,
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(speed_ref_mps),
                destination_state=destination_state,
                reference_samples=lane_center_reference,
                stop_goal_active=bool(mpc_stop_goal_active),
                stop_target_forward_m=stop_target_forward_m_debug,
                final_reference_accepted=bool(final_reference_gate.accepted),
                candidate_status=str(reference_debug.get(
                    "candidate_pipeline_selected_status", ""
                )),
                safety_manager=adapters.safety_manager,
                make_pedal_control=adapters.pedal_control,
                sim_time_s=float(sim_time_s),
                mpc_velocity_safety_cap_active=any(
                    str(getattr(row, "slack_group", "")) == "corridor"
                    for row in cav_constraint_rows
                ),
                corridor_infeasible_escalate=bool(corridor_infeasible_escalate),
                proximity_emergency_stop_required=bool(
                    cycle.emergency_stop_required
                ),
            ),
            set_actuator_context=adapters.set_actuator_context,
            acceleration_from_control=adapters.acceleration_from_control,
            steering_from_control=adapters.steering_from_control,
            apply_velocity_steering=adapters.apply_velocity_steering,
            control_factory=adapters.control_factory,
            boundary_metrics=adapters.road_boundary_measure,
            update_boundary_recovery=adapters.update_boundary_recovery,
            reset_boundary_recovery=adapters.reset_boundary_recovery,
        )
        adapters.accum_stage_ms("finalize_control", time.monotonic() - _ts_stage)
        _ts_stage = time.monotonic()
        mpc_jerk_seed_accel_mps2 = float(
            execution_result.jerk_seed_acceleration_mps2
        )
        mpc_status = str(execution_result.status)
        fallback_reason = str(execution_result.fallback_reason)
        hard_gate_active = fallback_reason.startswith("candidate_hard_gate:")
        mpc_replan_executed = bool(execution_result.replan_executed)
        failed_replan_buffer_reused = bool(
            execution_result.failed_replan_buffer_reused
        )
        if fallback_reason:
            adapters.warn_fallback(fallback_reason)
        if adapters.cav_conflict_enabled and adapters.cav_intent_broadcast_enabled:
            adapters.publish_cav_intent(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                sim_time_s=float(sim_time_s),
            )
        control = finalized_control.control
        accel_mps2 = float(finalized_control.acceleration_mps2)
        steer_rad = float(finalized_control.steering_rad)
        pre_supervisor_accel_mps2 = float(
            finalized_control.pre_filter_acceleration_mps2
        )
        pre_supervisor_steer_rad = float(
            finalized_control.pre_filter_steering_rad
        )
        post_supervisor_accel_mps2 = accel_mps2
        post_supervisor_steer_rad = steer_rad
        control_guard_reason = str(finalized_control.control_guard_reason)
        boundary_guard_reason = str(finalized_control.boundary_guard_reason)
        boundary_snapshot = finalized_control.boundary_snapshot
        safety_supervisor_reason = str(finalized_control.supervisor_reason)
        platform_adapter_debug = dict(finalized_control.platform_debug)
        mpc_feedback_record_reason = str(finalized_control.feedback_reason)
        diagnostics = PlannerDiagnosticsStage.build(
            adapters,
            self,
            {
                "sim_time_s": float(sim_time_s),
                "accel_mps2": accel_mps2,
                "behavior_debug": behavior_debug,
                "behavior_decision": behavior_decision,
                "boundary_snapshot": boundary_snapshot,
                "cav_resolution": cav_resolution,
                "control": control,
                "control_guard_reason": control_guard_reason,
                "destination_forward_m": destination_forward_m,
                "destination_lateral_m": destination_lateral_m,
                "destination_state": destination_state,
                "ego_location": ego_location,
                "ego_speed_mps": ego_speed_mps,
                "ego_transform": ego_transform,
                "ego_yaw_rad": ego_yaw_rad,
                "emergency_brake_requested": emergency_brake_requested,
                "fallback_reason": fallback_reason,
                "front_gap_m": front_gap_m,
                "hard_gate_active": hard_gate_active,
                "lane_center_reference": lane_center_reference,
                "local_object_snapshots": local_object_snapshots,
                "measured_accel_mps2": measured_accel_mps2,
                "mode_transition_guard_reason": mode_transition_guard_reason,
                "mpc_feedback_record_reason": mpc_feedback_record_reason,
                "mpc_jerk_seed_accel_mps2": mpc_jerk_seed_accel_mps2,
                "mpc_object_snapshots": mpc_object_snapshots,
                "mpc_replan_executed": mpc_replan_executed,
                "mpc_status": mpc_status,
                "mpc_stop_goal_active": mpc_stop_goal_active,
                "normal_stop_requested": normal_stop_requested,
                "object_snapshots": object_snapshots,
                "platform_adapter_debug": platform_adapter_debug,
                "post_supervisor_accel_mps2": post_supervisor_accel_mps2,
                "post_supervisor_steer_rad": post_supervisor_steer_rad,
                "pre_supervisor_accel_mps2": pre_supervisor_accel_mps2,
                "pre_supervisor_steer_rad": pre_supervisor_steer_rad,
                "reference_debug": reference_debug,
                "reference_first_forward_m": reference_first_forward_m,
                "reference_first_lateral_m": reference_first_lateral_m,
                "safety_supervisor_reason": safety_supervisor_reason,
                "speed_ref_mps": speed_ref_mps,
                "speed_target": speed_target,
                "stationary_traffic_stop_hold": stationary_traffic_stop_hold,
                "steer_rad": steer_rad,
                "stop_target_forward_m_debug": stop_target_forward_m_debug,
            },
        )
        decision_record = self.explain_decision(diagnostics)
        diagnostics.update(decision_record.as_debug_fields())
        adapters.draw_world_debug_primitives(
            destination_state=destination_state,
            lane_center_reference=lane_center_reference,
        )
        adapters.accum_stage_ms("post_finalize_control", time.monotonic() - _ts_stage)
        return PlannerOutput(
            control=control,
            behavior_command=BehaviorCommand.from_decision(
                behavior_decision,
                target_speed_mps=float(speed_ref_mps),
                debug_reason=str(nominal_frame.failure_reason or ""),
            ),
            reference_trajectory=[dict(sample) for sample in list(lane_center_reference or [])],
            planned_trajectory=adapters.last_mpc_trajectory_points(),
            predictions=dict(reference_debug.get("prediction_trajectories", {}) or {}),
            acceleration_mps2=float(post_supervisor_accel_mps2),
            steering_rad=float(post_supervisor_steer_rad),
            diagnostics=PlannerDiagnostics(diagnostics),
        )

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

    def prepare_route_lane_change(self, **kwargs):
        return self.behavior.prepare_route_lane_change(**kwargs)

    def prepare_planning_context(
        self, request: PlanningContextRequest, **kwargs
    ):
        return self.planning_context.run(request, **kwargs)

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
        cooperative_actor_ids: frozenset = frozenset(),
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
            cooperative_actor_ids=cooperative_actor_ids,
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

    def prepare_executable_behavior(
        self,
        request: ExecutableBehaviorPreparationRequest,
        **kwargs: Any,
    ) -> ExecutableBehaviorFrame:
        """Build the behavior contract from the frozen planning context."""

        planning = request.planning_context
        adapter = planning.adapter_output
        frame = planning.planner_input_frame
        behavior = planning.behavior_context
        route_behavior = behavior.route_behavior
        scenario_observation = behavior.scenario_observation
        scenario = scenario_observation.scenario
        conflict = behavior.conflict_resolution
        return self.resolve_executable_behavior(
            ExecutableBehaviorRequest(
                command_request=BehaviorCommandFrameRequest(
                    adapter_output=adapter,
                    ego_pose=adapter.ego_pose,
                    ego_location=request.ego_location,
                    ego_yaw_rad=float(request.ego_yaw_rad),
                    ego_speed_mps=float(request.ego_speed_mps),
                    current_lane_id=int(planning.current_lane_id),
                    target_speed_mps=float(request.cruise_speed_mps),
                    sim_time_s=float(request.sim_time_s),
                    route_optimal_lane_id=int(adapter.route_optimal_lane_id),
                    route_next_macro_maneuver=str(
                        frame.planning.route.next_macro_maneuver
                    ),
                    route_points=tuple(adapter.route_points),
                    front_distance_by_lane=dict(adapter.front_distance_by_lane),
                    lane_safety_scores=dict(adapter.lane_safety_scores),
                    object_snapshots=planning.object_snapshots,
                    route_lane_change_context=route_behavior.context,
                    lane_change_authorization=conflict.authorization,
                    opportunistic_lane_change_allowed=bool(
                        conflict.opportunistic_allowed
                    ),
                    behavior_traffic_state=str(
                        scenario.behavior_traffic_state
                    ),
                    behavior_stop_target=scenario.behavior_stop_target,
                    signal_context=dict(scenario.signal_context),
                    scenario_stop_required=bool(
                        scenario.decision.stop_goal_active
                    ),
                    lane_change_reference_active=bool(
                        request.lane_change_reference_active
                    ),
                    mpc_feedback=request.mpc_feedback,
                    max_deceleration_mps2=float(
                        request.max_deceleration_mps2
                    ),
                    config=request.config,
                    runtime_config=request.runtime_config,
                    route_recovery_requested=bool(
                        request.route_recovery_requested
                    ),
                    static_obstacle_mpc_stall_failure_count=int(
                        request.static_obstacle_mpc_stall_failure_count
                    ),
                ),
                scenario_decision=scenario.decision,
                lane_change_authorized=bool(conflict.authorization.allowed),
                lane_change_gate_reason=str(conflict.lane_change_gate_reason),
                lane_change_authorization_reason=str(
                    conflict.authorization.reason
                ),
                prepare_reference_lock=bool(request.prepare_reference_lock),
                lane_change_commitment_active=bool(
                    request.lane_change_commitment_active
                ),
                route_advanced_to_lane_change=bool(
                    scenario_observation.turn_context.route_advanced_to_lane_change
                ),
                route_current_road_option=str(
                    frame.planning.route.current_road_option
                ),
                ego_in_junction=bool(frame.map_lane.in_junction),
                stop_goal_active=bool(request.stop_goal_active),
                cruise_speed_mps=float(request.cruise_speed_mps),
                scenario_reason=str(scenario.decision.reason),
            ),
            **kwargs,
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

    def plan_speed(self, request: SpeedPlanningRequest):
        return self.speed_planning.run(request)

    def plan_speed_from_stages(
        self, request: SpeedPlanningPreparationRequest
    ):
        return self.speed_planning.prepare_and_run(request)

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

    def prepare_behavior_reference(self, request):
        if self.reference_planning is None:
            raise RuntimeError("reference planning stage is not configured")
        return self.reference_planning.prepare_behavior_reference(request)

    def select_candidate_reference(self, request):
        if self.reference_planning is None:
            raise RuntimeError("reference planning stage is not configured")
        return self.reference_planning.select_candidate_reference(request)

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

    def finalize_behavior_reference_from_stages(
        self, request: BehaviorReferenceFinalizationPreparationRequest
    ):
        return self.behavior_reference_finalization.prepare_and_run(request)

    def cooperative_conflict_reference(self, **kwargs):
        if self.reference_planning is None:
            raise RuntimeError("reference planning stage is not configured")
        return self.reference_planning.cooperative_conflict_reference(**kwargs)
