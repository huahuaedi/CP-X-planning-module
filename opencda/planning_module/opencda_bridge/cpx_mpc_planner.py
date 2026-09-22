"""Bridge from native OpenCDA vehicle managers to the CP-X MPC planner.

The bridge is intentionally small: OpenCDA still owns simulation, localization,
perception, and V2X discovery. This class consumes a custom map planner and
returns a CARLA ``VehicleControl`` directly, replacing both
OpenCDA's behavior agent and PID controller when enabled.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


from opencda.planning_module.utility.carla_compat import carla
from opencda.planning_module.opencda_bridge.planner_assembly import assemble_planner
from opencda.planning_module.pipeline.safety_supervisor import pipeline_failure_stop
from opencda.planning_module.pipeline.turn_road_envelope import (
    rolling_turn_envelope_payload_world,
)
from opencda.planning_module.pipeline.cooperative_arbitration_stage import (
    CooperativeArbitrationRequest,
)

from opencda.planning_module.pipeline.local_map_snapshot import LocalMapSnapshot
from opencda.planning_module.pipeline.reference_line_provider import (
    LANE_CHANGE,
    POST_TURN,
    TURN,
    ReferenceLineProvider,
)
from opencda.planning_module.pipeline.reference_planning_stage import (
    BehaviorReferencePreparationRequest,
    CandidatePlanningPreparationRequest,
)
from opencda.planning_module.pipeline.planner_diagnostics_stage import (
    PlannerDiagnosticsStage,
    ReferenceDiagnosticsRequest,
)
from opencda.planning_module.pipeline.mpc_execution_stage import (
    MPCExecutionRequest,
)
from opencda.planning_module.pipeline.mpc_cost_profile_stage import (
    adaptive_target_horizon_s as _adaptive_target_horizon_s,
    select_profile_with_hysteresis as _select_mpc_cost_profile_with_hysteresis,
)
from opencda.planning_module.pipeline.execution_pipeline import (
    ExecutableBehaviorPreparationRequest,
    NominalPlanningRequest,
)
from opencda.planning_module.pipeline.planning_context_stage import (
    PlanningContextRequest,
)
from opencda.planning_module.pipeline.behavior_reference_finalization_stage import (
    BehaviorReferenceFinalizationPreparationRequest,
)
from opencda.planning_module.pipeline.speed_planning_stage import (
    SpeedPlanningPreparationRequest,
)
from opencda.planning_module.pipeline.route_update_stage import (
    RouteUpdateRequest,
)
from opencda.planning_module.pipeline.candidate_selection_stage import (
    CandidateSelectionStage,
)
class CPXMPCPlannerBridge:
    """Direct-control planner used inside ``VehicleManager.run_step``."""

    @property
    def reference_generator(self):
        """Compatibility view; ReferenceLineProvider owns the builder."""
        return self._stable_reference_line_provider.builder

    @reference_generator.setter
    def reference_generator(self, builder):
        provider = getattr(self, "_stable_reference_line_provider", None)
        if provider is None:
            provider = ReferenceLineProvider()
            self._stable_reference_line_provider = provider
        provider.attach_builder(builder)

    def __init__(
        self,
        vehicle_manager: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        map_planner: Any = None,
    ):
        assemble_planner(self, vehicle_manager, config, map_planner)

    def set_destination(
        self,
        *,
        start_location: Any,
        end_location: Any,
        clean: bool = False,
        end_reset: bool = True,
    ) -> None:
        """Set the CP-X global route without using OpenCDA BehaviorAgent."""

        del clean, end_reset
        start_point = self._location_to_point(start_location)
        goal_point = self._location_to_point(end_location)
        self._active_route_summary = self.route_manager.set_destination(
            start_point=start_point,
            goal_point=goal_point,
        )
        self.nominal_trajectory_generator.reset(source="destination_updated")
        self.control_buffer.reset(reason="destination_updated")
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="destination_updated")
        self.maneuver_manager.clear_turn(reason="destination_updated")
        self._clear_turn_master_reference()
        self._stable_reference_line_provider.release(
            POST_TURN, event="phase_transition"
        )

    def set_external_global_plan(self, world_plan: Sequence[Any]) -> None:
        """Accept only mission endpoints; AD-map owns the route between them."""

        locations = []
        for entry in list(world_plan or []):
            node = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
            # ``carla.Transform`` (what callers normally pass here) exposes
            # its own ``.location`` directly, so try that first -- and only
            # then fall back to ``.transform.location`` for a
            # ``carla.Waypoint``. The previous order (``.transform`` first)
            # misidentified a plain ``carla.Transform`` as a ``Waypoint``:
            # ``Transform`` also happens to define a *method* named
            # ``transform`` (coordinate transform, unrelated to this route
            # plumbing), so ``getattr(node, "transform", node)`` silently
            # picked up that bound method instead of falling back to
            # ``node``, and the method object has no ``.location`` --
            # dropping every point and leaving ``locations`` empty.
            location = getattr(node, "location", None)
            if location is None:
                transform = getattr(node, "transform", None)
                location = getattr(transform, "location", None)
            if location is not None:
                locations.append(location)
        if len(locations) < 2:
            raise ValueError("Global plan must provide at least start and goal")
        self._active_route_summary = self.route_manager.set_destination(
            start_point=self._location_to_point(locations[0]),
            goal_point=self._location_to_point(locations[-1]),
        )
        self._route_replan_last_reason = "admap_route_from_mission_endpoints"
        self._active_route_summary = None
        self.nominal_trajectory_generator.reset(
            source="external_global_plan_installed"
        )
        self.control_buffer.reset(reason="external_global_plan_installed")
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="external_global_plan_installed")

    def update_information(
        self,
        *,
        ego_transform: Any,
        ego_speed_kmh: float,
        detected_objects: Any = None,
        v2x_manager: Any = None,
        safety_manager: Any = None,
        map_manager: Any = None,
    ) -> None:
        """Receive the current OpenCDA tick snapshot from VehicleManager.update_info."""

        self._latest_opencda_update = {
            "ego_transform": ego_transform,
            "ego_speed_kmh": float(ego_speed_kmh),
            "detected_objects": detected_objects,
            "v2x_manager": v2x_manager,
            "safety_manager": safety_manager,
            "map_manager": map_manager,
            "sim_time_s": float(self._sim_time_s()),
        }

    # Ad-hoc stage-level timing: run_step's own wall time was found to be the
    # dominant cost in MDrive's per-tick loop (~150ms/call vs. mpc_solve_time_ms's
    # 3ms), so this locates which of the pipeline's stages inside that call
    # actually spends it.
    def _accum_stage_ms(self, name: str, seconds: float) -> None:
        stats = self.__dict__.setdefault("_stage_ms_stats", {})
        total, count = stats.get(name, (0.0, 0))
        stats[name] = (total + seconds * 1000.0, count + 1)

    def _print_and_reset_stage_ms(self) -> None:
        stats = self.__dict__.get("_stage_ms_stats", {})
        if stats:
            parts = [
                f"{name}: total={total:.0f}ms avg={total / max(1, count):.1f}ms"
                for name, (total, count) in stats.items()
            ]
            print("[cpx_stage_timing] last 50 calls -- " + " | ".join(parts), flush=True)
            self._stage_ms_stats = {}
        wp_hits = self.__dict__.get("_waypoint_cache_hits", 0)
        wp_misses = self.__dict__.get("_waypoint_cache_misses", 0)
        if wp_hits or wp_misses:
            wp_total = wp_hits + wp_misses
            print(
                "[cpx_stage_timing] prediction get_waypoint cache -- "
                f"last 50 ticks: hits={wp_hits} misses={wp_misses} "
                f"hit_rate={100.0 * wp_hits / max(1, wp_total):.0f}% "
                f"(native calls avoided: {wp_hits}/{wp_total})",
                flush=True,
            )
            self._waypoint_cache_hits = 0
            self._waypoint_cache_misses = 0
            self._per_tick_waypoint_repeat_pct = []

    def run_step(self) -> carla.VehicleControl:
        """Plan and return a low-level CARLA control command.

        This is the sole write point: every other safety/fallback system
        (TrajectoryFallbackManager, MPCExecutionStage's normal/safe/emergency
        stop controls, SafetySupervisor) feeds into the one ``control`` this
        method returns -- none of them hands a command to CARLA on its own.
        The full precedence, upstream to downstream, in case a future change
        needs to reason about which layer wins when several are active at
        once:

        1. TrajectoryFallbackManager.bounded_safe_stop() -- reference layer.
           Replaces the trajectory the MPC tracks, not a control command.
        2. MPCExecutionStage ("execute_mpc") -- solves the QP against that
           reference or reuses the control buffer.  Only when the solve
           fails and no valid buffered command exists does it produce a
           control: safe_stop_control (bounded decel matching the MPC's own
           acceleration/steering intent).  A stop hold or a hard gate carries
           no control.
        3. ControlFinalizationStage.run() ("finalize_control") -- decides
           whether the situation is an emergency (a hard gate that names a
           stop hazard, the emergency_brake maneuver, or a corridor
           infeasibility escalation), selects the platform target velocity,
           and has apply_velocity_steering (PID) build the raw
           carla.VehicleControl for every case except the bounded safe stop
           above; SafetySupervisor.run() is the
           last-mile filter (steer/throttle/brake rate limits, red/yellow
           signal-stop envelope, boundary-recovery hard-stop, stuck-release)
           and owns whatever control this stage returns.
        4. This method's own try/except -- the absolute last resort. Only
           reached if steps 1-3 raise; returns pipeline_failure_stop()'s
           bare hard brake unconditionally, never the platform's normal
           control path.
        """

        try:
            planner_output = self.execute_planning_pipeline()
        except Exception as exc:
            stop = pipeline_failure_stop(
                error=exc,
                fallback_policy=self.fallback_policy,
                emergency_stop_control=self.actuator_port.emergency_stop_control,
                min_acceleration_mps2=float(
                    getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0)
                ),
                sim_time_s=float(self._sim_time_s()),
                vehicle_id=int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            )
            self._last_accel_mps2 = float(stop.acceleration_mps2)
            self._last_steer_rad = float(stop.steering_rad)
            self.last_debug = dict(stop.debug)
            self._record_debug(self.last_debug)
            return stop.control
        self.last_output = planner_output
        self.last_debug = planner_output.diagnostics_dict()
        self._record_debug(self.last_debug)
        self._stage_timing_call_count = getattr(self, "_stage_timing_call_count", 0) + 1
        if self._stage_timing_call_count % 50 == 0:
            self._print_and_reset_stage_ms()
        return planner_output.control

    def _apply_velocity_steering_interface(
        self,
        *,
        target_speed_mps: float,
        target_steering_rad: float,
        actual_speed_mps: float,
        stop_goal_active: bool,
        emergency_stop: bool,
        sim_time_s: float,
    ):
        """Map planner-owned speed/steering before final safety supervision."""

        from opencda.planning_module.pipeline.velocity_steering_adapter import (
            VelocitySteeringCommand,
        )

        control, adapter_reason = self.velocity_steering_adapter.run_step(
            command=VelocitySteeringCommand(
                target_speed_mps=float(target_speed_mps),
                target_steering_rad=float(target_steering_rad),
                emergency_stop=bool(emergency_stop),
                stop_goal_active=bool(stop_goal_active),
            ),
            actual_speed_mps=float(actual_speed_mps),
            sim_time_s=float(sim_time_s),
            make_pedal_control=self.actuator_port.pedal_control,
        )
        applied_steer_rad = (
            float(getattr(control, "steer", 0.0))
            * float(self.vehicle_dynamics.actuator_max_steer_rad)
        )
        debug = {
            "control_interface": "planner_velocity_steering",
            "platform_target_speed_mps": float(target_speed_mps),
            "platform_target_steer_rad": float(target_steering_rad),
            "platform_adapter_steer_rad": float(applied_steer_rad),
            "platform_actuator_max_steer_rad": float(
                self.vehicle_dynamics.actuator_max_steer_rad
            ),
            "platform_wheelbase_m": float(self.vehicle_dynamics.wheelbase_m),
            "mpc_kinematic_steering_effectiveness": float(
                self.mpc.kinematic_steering_effectiveness
            ),
            "reference_vehicle_curvature_safety_factor": float(
                self.config.get(
                    "reference_vehicle_curvature_safety_factor", 0.90
                )
            ),
            "reference_vehicle_max_curvature_1pm": float(
                self.config["reference_vehicle_max_curvature_1pm"]
            ),
            "platform_vehicle_dynamics_source": str(
                self.vehicle_dynamics.source
            ),
            "platform_actual_speed_mps": float(actual_speed_mps),
            "platform_adapter_reason": str(adapter_reason),
            "platform_adapter_throttle": float(
                getattr(control, "throttle", 0.0)
            ),
            "platform_adapter_brake": float(getattr(control, "brake", 0.0)),
            "platform_adapter_steer": float(getattr(control, "steer", 0.0)),
        }
        return (
            control,
            float(self._accel_from_control(control)),
            float(applied_steer_rad),
            debug,
        )

    @property
    def _local_map_snapshot(self):
        return self._route_context.local_map_snapshot

    def execute_planning_pipeline(self):
        """Public OpenCDA bridge port for one full CP-X planning tick."""

        return self._run_full_cpx_pipeline_step()

    def _run_full_cpx_pipeline_step(self):
        """Run OpenCDAPlanningAdapter -> PlanningPipeline -> PlannerOutput."""

        from opencda.planning_module.pipeline.output import (
            BehaviorCommand,
            PlannerDiagnostics,
            PlannerOutput,
        )
        latest_update = dict(getattr(self, "_latest_opencda_update", {}) or {})
        if self.cp_provider is not None:
            try:
                self.cp_provider.publish(
                    world=self.vehicle_manager.vehicle.get_world(),
                    map_planner=self.map_planner,
                    ego_vehicle=self.vehicle_manager.vehicle,
                    sim_time_s=self._sim_time_s(),
                    vehicle_manager=self.vehicle_manager,
                )
            except Exception as exc:
                if self.debug:
                    print(f"[CP-X OpenCDA Bridge] native CP publish failed: {exc}")
        _ts_stage = time.monotonic()
        cycle = self.pipeline.begin_cycle(
            timestamp_s=float(self._sim_time_s()),
            ego_transform=(
                latest_update.get("ego_transform")
                or self.vehicle_manager.localizer.get_ego_pos()
            ),
            ego_speed_kmh=float(latest_update.get(
                "ego_speed_kmh", self.vehicle_manager.localizer.get_ego_spd()
            )),
            detected_objects=(
                latest_update.get("detected_objects")
                if latest_update.get("detected_objects") is not None
                else getattr(
                    self.vehicle_manager.perception_manager, "objects", {}
                ) or {}
            ),
            cp_payload=self._load_cp_message_payload(),
            ignore_dynamic_objects=bool(
                self.functional_test_ignore_dynamic_objects
            ),
            cruise_speed_mps=float(self.target_speed_mps),
            base_emergency_gap_m=float(
                self.config.get("following_emergency_gap_m", 3.0)
            ),
            emergency_standstill_buffer_m=float(
                self.config.get("following_emergency_standstill_buffer_m", 1.0)
            ),
            following_time_headway_s=float(
                self.config.get("following_time_headway_s", 1.5)
            ),
        )
        self._accum_stage_ms("begin_cycle", time.monotonic() - _ts_stage)
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
        emergency_front_gap_m = float(cycle.emergency_gap_m)
        stop_goal_active = bool(cycle.emergency_stop_required)
        requested_speed_mps = float(cycle.requested_speed_mps)
        current_state = list(tick.current_state)

        from opencda.planning_module.pipeline.behavior_reference_execution_stage import (
            BehaviorReferenceRequest,
        )
        nominal_frame = self.pipeline.resolve_nominal_plan(
            NominalPlanningRequest(
                behavior_request=BehaviorReferenceRequest(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    ego_speed_mps=float(ego_speed_mps),
                    requested_speed_mps=float(requested_speed_mps),
                    object_snapshots=object_snapshots,
                    stop_goal_active=bool(stop_goal_active),
                    cp_payload=cp_payload,
                    current_state=current_state,
                    sim_time_s=float(tick.timestamp_s),
                    route_revision=str(self.route_manager.route_revision),
                ),
                route_status=getattr(self.route_manager, "last_status", None),
                route_revision=str(self.route_manager.route_revision),
                ego_speed_mps=float(ego_speed_mps),
                current_state=current_state,
                fallback_lane_id=int(getattr(
                    self._local_map_snapshot, "ego_lane_id", 0
                )),
            ),
            planner=self._plan_behavior_and_reference,
            observe_stage_duration=self._accum_stage_ms,
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
            setattr(self.vehicle_manager, "_opencda_agent_finished", True)
        if nominal_frame.failure_reason and self.debug:
            print(
                "[CP-X OpenCDA Bridge] behavior/reference pipeline failed: "
                + str(nominal_frame.failure_reason)
            )
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
                stop_target_forward_m_debug, _ = self._body_frame_xy(
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
        admission = self.pipeline.prepare_trajectory_execution(
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
                "route_points": self._active_global_route_points(),
                "local_map": self._local_map_snapshot,
                "route_cursor": self.route_manager.route_cursor,
                "route_revision": str(self.route_manager.route_revision),
                "map_epoch": str(getattr(self, "waypoint_backend", "admap") or "admap"),
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
                    str(self.maneuver_manager.lane_change.phase) == "executing"
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
            reset_control_buffer=getattr(self.control_buffer, "reset", None),
        )
        self._accum_stage_ms("prepare_trajectory_execution", time.monotonic() - _ts_stage)
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
        destination_forward_m, destination_lateral_m = self._body_frame_xy(
            origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad), target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        reference_first_forward_m = ""
        reference_first_lateral_m = ""
        if lane_center_reference:
            first_reference = lane_center_reference[0]
            reference_first_forward_m, reference_first_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(first_reference.get("x_ref_m", first_reference.get("x", ego_location.x))),
                target_y_m=float(first_reference.get("y_ref_m", first_reference.get("y", ego_location.y))),
            )
        mpc_status = str(getattr(self.mpc, "_last_status", ""))
        # MPC constrains jerk between the previous control input and the new
        # acceleration sequence. Seed that constraint with the acceleration
        # command actually sent last tick, not the measured vehicle response.
        # The latter contains actuator lag and can stay strongly negative
        # after the speed target has recovered, otherwise forcing every new
        # solve to continue braking until the vehicle is almost stationary.
        mpc_jerk_seed_accel_mps2 = float(self._last_accel_mps2)
        road_envelope_payload_world = rolling_turn_envelope_payload_world(
            config=self.config,
            mpc=self.mpc,
            vehicle=getattr(getattr(self, "vehicle_manager", None), "vehicle", None),
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
            cav_constraint_revision = self.pipeline.cooperative.schedule.constraint_revision(
                cav_diagnostics
            )
        corridor_infeasible_escalate = bool(
            self.pipeline.cooperative.schedule.corridor_emergency_stop_required
        )
        self._maybe_capture_execute_mpc_frame(
            sim_time_s=float(sim_time_s),
            current_state=current_state,
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            current_acceleration_mps2=float(mpc_jerk_seed_accel_mps2),
            current_steering_rad=float(self._last_steer_rad),
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
        self._accum_stage_ms("between_admission_and_mpc", time.monotonic() - _ts_stage)
        _ts_stage = time.monotonic()
        execution_result = self.pipeline.execute_mpc(
            MPCExecutionRequest(
                sim_time_s=float(sim_time_s),
                current_state=current_state,
                destination_state=destination_state,
                reference_samples=lane_center_reference,
                object_snapshots=self._mpc_object_snapshots_with_prediction(
                    mpc_object_snapshots,
                    prediction_trajectories=reference_debug.get(
                        "prediction_trajectories", {}
                    ),
                ),
                current_acceleration_mps2=float(mpc_jerk_seed_accel_mps2),
                current_steering_rad=float(self._last_steer_rad),
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(speed_ref_mps),
                stop_goal_active=bool(mpc_stop_goal_active),
                behavior_maneuver=str(behavior_decision.maneuver),
                behavior_phase=str(behavior_decision.phase),
                hard_gate_reason=str(candidate_hard_gate_reason),
                stationary_stop_hold=bool(stationary_traffic_stop_hold),
                control_context=mpc_control_context,
                road_envelope_payload_world=road_envelope_payload_world,
                speed_crossing_deadband_mps=float(self.config.get(
                    "control_buffer_speed_crossing_deadband_mps", 0.15,
                )),
                corridor_rows=cav_constraint_rows,
                constraint_revision=str(cav_constraint_revision),
            ),
            safe_stop_control=self.actuator_port.safe_stop_control,
        )
        self._accum_stage_ms("execute_mpc", time.monotonic() - _ts_stage)
        mpc_jerk_seed_accel_mps2 = float(
            execution_result.jerk_seed_acceleration_mps2
        )
        from opencda.planning_module.pipeline.control_finalization_stage import (
            ControlFinalizationRequest,
        )
        _ts_stage = time.monotonic()
        finalized_control = self.pipeline.finalize_control(
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
                safety_manager=latest_update.get("safety_manager"),
                make_pedal_control=self.actuator_port.pedal_control,
                sim_time_s=float(sim_time_s),
                mpc_velocity_safety_cap_active=any(
                    str(getattr(row, "slack_group", "")) == "corridor"
                    for row in cav_constraint_rows
                ),
                corridor_infeasible_escalate=bool(corridor_infeasible_escalate),
            ),
            set_actuator_context=self._set_actuator_context,
            acceleration_from_control=self._accel_from_control,
            steering_from_control=self._steer_rad_from_control,
            apply_velocity_steering=self._apply_velocity_steering_interface,
            control_factory=self._control_from_mpc,
            boundary_metrics=self._road_boundary.measure,
            update_boundary_recovery=self._boundary_recovery.update,
            reset_boundary_recovery=self._boundary_recovery.reset,
        )
        self._accum_stage_ms("finalize_control", time.monotonic() - _ts_stage)
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
        if fallback_reason and not self._warned:
            print("[CP-X OpenCDA Bridge] MPC fallback active: " + fallback_reason)
            self._warned = True
        if self._cav_conflict_enabled and self._cav_intent_broadcast_enabled:
            self._publish_cav_intent(
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
        self._last_accel_mps2 = post_supervisor_accel_mps2
        self._last_steer_rad = post_supervisor_steer_rad
        diagnostics = PlannerDiagnosticsStage.build(
            self,
            {
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
        decision_record = self.pipeline.explain_decision(diagnostics)
        diagnostics.update(decision_record.as_debug_fields())
        self._draw_world_debug_primitives(
            destination_state=destination_state,
            lane_center_reference=lane_center_reference,
        )
        self._accum_stage_ms("post_finalize_control", time.monotonic() - _ts_stage)
        return PlannerOutput(
            control=control,
            behavior_command=BehaviorCommand.from_debug(
                behavior_debug=behavior_debug,
                target_speed_mps=float(speed_ref_mps),
            ),
            reference_trajectory=[dict(sample) for sample in list(lane_center_reference or [])],
            planned_trajectory=self._last_mpc_trajectory_points(),
            predictions=dict(reference_debug.get("prediction_trajectories", {}) or {}),
            acceleration_mps2=float(post_supervisor_accel_mps2),
            steering_rad=float(post_supervisor_steer_rad),
            diagnostics=PlannerDiagnostics(diagnostics),
        )

    def _resolve_full_traffic_state_from_carla_actor(
        self,
        *,
        raw_state: str,
        signal_context: Mapping[str, object] | None,
    ) -> tuple[str, str]:
        """Resolve a temporarily missing CP signal from its latched CARLA actor."""

        state = str(raw_state or "unknown").strip().lower()
        context = dict(signal_context or {})
        actor_id = str(
            context.get(
                "signal_actor_id",
                context.get("control_id", context.get("cp_control_id", "")),
            )
            or ""
        ).strip()
        if actor_id and state in {"red", "yellow", "green"}:
            self._full_signal_actor_id = str(actor_id)

        should_query_latched_actor = (
            state == "unknown"
            and bool(self._full_signal_actor_id)
            and (
                self._full_traffic_memory.latched_stop_target is not None
                or self._full_traffic_memory.latched_stop_state
                in {"red", "yellow"}
            )
        )
        if not bool(should_query_latched_actor):
            return str(state), ""

        try:
            numeric_actor_id = int(float(self._full_signal_actor_id))
            world = self.vehicle_manager.vehicle.get_world()
            actor = world.get_actor(numeric_actor_id)
            if actor is None:
                return str(state), (
                    f"latched_carla_signal_actor_missing:{numeric_actor_id}"
                )
            actor_state = actor.get_state()
            live_state = str(
                getattr(actor_state, "name", actor_state) or "unknown"
            ).split(".")[-1].strip().lower()
            if live_state in {"red", "yellow", "green"}:
                return (
                    str(live_state),
                    f"latched_carla_signal_actor:{numeric_actor_id}:{live_state}",
                )
            return str(state), (
                f"latched_carla_signal_actor_invalid:{numeric_actor_id}:{live_state}"
            )
        except Exception as exc:
            return str(state), (
                "latched_carla_signal_actor_error:"
                f"{type(exc).__name__}"
            )

    def _maybe_capture_execute_mpc_frame(self, *, sim_time_s, current_state,
                                         ego_location, ego_yaw_rad, ego_speed_mps,
                                         current_acceleration_mps2,
                                         current_steering_rad, destination_state,
                                         target_speed_mps, stop_goal_active,
                                         behavior_decision, published_reference,
                                         mpc_object_snapshots, cav_constraint_rows,
                                         cav_diagnostics, cav_corridor=None):
        """Serialize one tick for tools/frame_replay. No-op unless the
        ``frame_capture`` config block is present and the sim-time window
        (if any) contains this tick. Debug-only; never raises into the
        planning path."""

        raw = self.config.get("frame_capture")
        if not isinstance(raw, Mapping):
            return
        try:
            from opencda.planning_module.tools.frame_capture_hook import (
                FrameCaptureConfig,
                dump_execute_mpc_frame,
            )
            cfg = getattr(self, "_frame_capture_cfg", None)
            if cfg is None:
                cfg = FrameCaptureConfig(raw)
                self._frame_capture_cfg = cfg
            if not cfg.wants(float(sim_time_s)):
                return
            pre_reference = getattr(self, "_frame_capture_pre_reference", None)
            dump_execute_mpc_frame(
                cfg,
                sim_time_s=float(sim_time_s),
                current_state=list(current_state),
                ego_origin_xy=(float(ego_location.x), float(ego_location.y)),
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                current_acceleration_mps2=float(current_acceleration_mps2),
                current_steering_rad=float(current_steering_rad),
                destination_state=list(destination_state),
                target_speed_mps=float(target_speed_mps),
                stop_goal_active=bool(stop_goal_active),
                behavior_maneuver=str(getattr(behavior_decision, "maneuver", "")),
                behavior_phase=str(getattr(behavior_decision, "phase", "")),
                pre_publication_reference=list(pre_reference or published_reference),
                published_reference=list(published_reference or []),
                mpc_object_snapshots=list(mpc_object_snapshots or []),
                mpc_rows=list(cav_constraint_rows or ()),
                cav_diagnostics=dict(cav_diagnostics or {}),
                corridor=cav_corridor,
                prev_x_solution=getattr(self.mpc, "_last_x_solution", None),
                prev_u_solution=getattr(self.mpc, "_last_u_solution", None),
                mpc_runtime_state={
                    **(
                        self.mpc._capture_mode_cost_state()
                        if hasattr(self.mpc, "_capture_mode_cost_state")
                        else {}
                    ),
                    "horizon_steps": int(getattr(self.mpc, "horizon_steps", 0)),
                    "active_cost_profile_name": str(getattr(
                        self.mpc, "active_cost_profile_name", "base"
                    )),
                },
                mpc_config_path=str(self.config.get("mpc_config_path", "") or "")
                or None,
            )
        except Exception as exc:  # debug-only path; must not break planning
            if self.debug:
                print(f"[CP-X OpenCDA Bridge] frame capture failed: {exc}")

    def _resolved_debug_output_dir(self) -> Path:
        """Configured debug dir, with the prediction-ablation mode appended as
        a safety net so the ``cv`` / ``blind`` / ``oracle`` runs of one
        scenario don't clobber each other's ``opencda_planner_debug.csv`` when
        the config forgets to give them distinct ``debug_output_dir`` values.
        Idempotent: never appends a suffix that is already there."""

        base = Path(
            self.config.get(
                "debug_output_dir",
                Path(__file__).resolve().parent / "debug",
            )
        )
        mode = str(getattr(self, "_prediction_mode", "cv"))
        if mode and mode != "cv" and not base.name.endswith(f"_{mode}"):
            return base.with_name(f"{base.name}_{mode}")
        return base

    def _record_debug(self, payload: Mapping[str, Any]) -> None:
        if not bool(self.config.get("record_debug", True)):
            return
        try:
            debug_dir = self._resolved_debug_output_dir()
            debug_dir.mkdir(parents=True, exist_ok=True)
            configured_formats = self.config.get("debug_log_formats", ("jsonl",))
            if isinstance(configured_formats, str):
                configured_formats = (configured_formats,)
            formats = {
                str(item).strip().lower() for item in configured_formats or ()
            }
            unsupported = formats.difference({"csv", "jsonl"})
            if unsupported:
                raise ValueError(
                    "unsupported debug_log_formats: %s"
                    % ",".join(sorted(unsupported))
                )
            if "csv" in formats and self._debug_writer is None:
                self._debug_csv_file = open(
                    debug_dir / "opencda_planner_debug.csv",
                    "w",
                    newline="",
                    encoding="utf-8",
                )
                self._debug_writer = csv.DictWriter(
                    self._debug_csv_file,
                    fieldnames=PlannerDiagnosticsStage.fieldnames(payload),
                    extrasaction="ignore",
                )
                self._debug_writer.writeheader()
            if "jsonl" in formats and self._debug_jsonl_file is None:
                self._debug_jsonl_file = open(
                    debug_dir / "opencda_planner_debug.jsonl",
                    "w",
                    encoding="utf-8",
                )
            if self._debug_writer is not None:
                row = {
                    name: payload.get(name, "")
                    for name in self._debug_writer.fieldnames
                }
                self._debug_writer.writerow(row)
                self._debug_csv_file.flush()
            if self._debug_jsonl_file is not None:
                self._debug_jsonl_file.write(json.dumps(dict(payload), default=str) + "\n")
                self._debug_jsonl_file.flush()
        except Exception as exc:
            # Diagnostic plumbing failing silently on the branch being debugged
            # is its own trap -- always surface it.
            print(f"[CP-X OpenCDA Bridge] debug record failed: {type(exc).__name__}: {exc}")
            if not getattr(self, "_debug_record_traceback_printed", False):
                import traceback
                traceback.print_exc()
                self._debug_record_traceback_printed = True


    def _update_evaluation_metrics(
        self,
        *,
        ego_location: Any,
        ego_speed_mps: float,
        ego_yaw_rad: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        behavior_decision: str,
        behavior_fsm_state: str,
        mpc_replan_executed: bool,
        cp_summary: Mapping[str, Any],
        reference_samples: Sequence[Mapping[str, Any]] = (),
        boundary_snapshot: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Update run metrics and return fields for the unified debug row."""

        if not bool(self.config.get("record_evaluation_metrics", True)):
            return {"evaluation_metrics_available": False}
        sim_time_s = float(self._sim_time_s())
        self.evaluation_metrics.update(
            ego_state={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "v": float(ego_speed_mps),
                "psi": float(ego_yaw_rad),
            },
            obstacle_snapshots=list(object_snapshots or []),
            sim_time_s=sim_time_s,
            behavior_decision=str(behavior_decision),
            fsm_state=str(behavior_fsm_state),
            cp_provider_source=str(cp_summary.get("provider_source", "")),
            native_opencda_available=bool(
                cp_summary.get("native_opencda_available", False)
            ),
            native_opencda_required=bool(
                cp_summary.get("native_opencda_required", False)
            ),
            cp_obstacle_count=int(cp_summary.get("obstacle_count", 0) or 0),
        )
        if bool(mpc_replan_executed):
            self.evaluation_metrics.record_mpc_status(self.mpc.get_runtime_status())
            self.evaluation_metrics.record_mpc_extras(
                lateral_offset_m=None,
                heading_error_rad=None,
                cost_terms=self.mpc.get_last_cost_terms(),
            )
        sample = dict(self.evaluation_metrics.samples[-1])
        boundary = (
            dict(boundary_snapshot)
            if boundary_snapshot is not None
            else self._road_boundary.measure(
                ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                reference_samples=reference_samples,
            )
        )
        self.evaluation_metrics.record_road_boundary(
            sample_valid=bool(boundary["road_boundary_sample_valid"]),
            lateral_offset_m=(
                boundary["road_boundary_lateral_offset_m"]
                if boundary["road_boundary_lateral_offset_m"] != ""
                else None
            ),
            lane_width_m=(
                boundary["road_boundary_lane_width_m"]
                if boundary["road_boundary_lane_width_m"] != ""
                else None
            ),
            ego_half_width_m=(
                boundary["road_boundary_ego_half_width_m"]
                if boundary["road_boundary_ego_half_width_m"] != ""
                else None
            ),
            clearance_m=(
                boundary["road_boundary_clearance_m"]
                if boundary["road_boundary_clearance_m"] != ""
                else None
            ),
            breach=bool(boundary["road_boundary_breach"]),
        )
        summary = self.evaluation_metrics.summary()
        boundary_sample_count = int(self._road_boundary.sample_count)
        cost_terms = dict(self.mpc.get_last_cost_terms())
        return {
            "evaluation_metrics_available": True,
            # Ground truth for these two comes only from whatever calls
            # self.evaluation_metrics.record_collision() -- the planner
            # itself has no CARLA collision sensor and never calls it; the
            # scenario runner's own collision sensor (scenario-evaluator
            # side, not planner side) is what feeds this.
            "collision_count": int(summary.get("collision_count", 0)),
            "collision_rate_per_km": float(summary.get("collision_rate_per_km", 0.0)),
            # ON/OFF ablation credibility: proves CP actually bought a head
            # start on some obstacle (positive lead, or cp_only with no
            # local sighting at all) rather than just adding a redundant
            # observation source both runs would have gotten anyway.
            "max_anticipation_lead_s": summary.get("max_anticipation_lead_s", ""),
            "cp_only_obstacle_count": int(summary.get("cp_only_obstacle_count", 0)),
            "nearest_ttc_s": sample.get("nearest_ttc_s", ""),
            "min_ttc_s": summary.get("min_ttc_s", ""),
            "nearest_ttc_obstacle_id": sample.get(
                "nearest_ttc_obstacle_id", ""
            ),
            "nearest_ttc_reason": sample.get("nearest_ttc_reason", ""),
            "nearest_ttc_longitudinal_gap_m": sample.get(
                "nearest_ttc_longitudinal_gap_m", ""
            ),
            "nearest_ttc_lateral_gap_m": sample.get(
                "nearest_ttc_lateral_gap_m", ""
            ),
            "nearest_ttc_bumper_gap_m": sample.get(
                "nearest_ttc_bumper_gap_m", ""
            ),
            "nearest_ttc_closing_speed_mps": sample.get(
                "nearest_ttc_closing_speed_mps", ""
            ),
            "tick_max_drac_mps2": sample.get("max_drac_mps2", ""),
            "max_drac_mps2": summary.get("max_drac_mps2", ""),
            "min_pet_s": summary.get("min_pet_s", ""),
            "distance_traveled_m": summary.get("distance_traveled_m", ""),
            **boundary,
            "road_boundary_breach_count": int(
                self._road_boundary.breach_count
            ),
            "road_boundary_sample_count": boundary_sample_count,
            "road_boundary_breach_rate": (
                float(self._road_boundary.breach_count)
                / float(boundary_sample_count)
                if boundary_sample_count > 0
                else ""
            ),
            "Cost_RoadBoundary": cost_terms.get("Cost_RoadBoundary", ""),
            "mpc_road_boundary_peak": (
                self.mpc.get_last_road_boundary_peak_diagnostic()
            ),
            "mpc_heading_tracking": (
                self.mpc.get_last_heading_tracking_diagnostic()
            ),
            "Cost_Repulsive": cost_terms.get("Cost_Repulsive", ""),
            "Cost_Repulsive_Safe": cost_terms.get("Cost_Repulsive_Safe", ""),
            "Cost_Repulsive_Collision": cost_terms.get(
                "Cost_Repulsive_Collision", ""
            ),
            "Cost_Repulsive_LogBarrier": cost_terms.get(
                "Cost_Repulsive_LogBarrier", ""
            ),
            "Cost_ref": cost_terms.get("Cost_ref", ""),
            "Cost_LaneCenter": cost_terms.get("Cost_LaneCenter", ""),
            "Cost_Control": cost_terms.get("Cost_Control", ""),
            "Cost_VelocitySlack": cost_terms.get("Cost_VelocitySlack", ""),
            "prediction_lane_step_resolved_count": int(
                self._prediction_lane_step_resolved_count
            ),
            "prediction_lane_step_none_count": int(
                self._prediction_lane_step_none_count
            ),
        }

    def destroy(self) -> None:
        if bool(self.config.get("record_evaluation_metrics", True)):
            try:
                debug_dir = self._resolved_debug_output_dir()
                self._write_planning_metrics_artifacts(
                    artifact_dir=str(debug_dir),
                    recorder=self.evaluation_metrics,
                    scenario_name=str(
                        self.config.get("scenario_name", "opencda_scenario")
                    ),
                )
            except Exception as exc:
                if self.debug:
                    print(
                        "[CP-X OpenCDA Bridge] metrics artifact write failed: "
                        f"{exc}"
                    )
        for handle_name in ("_debug_csv_file", "_debug_jsonl_file"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_name, None)
        self._debug_writer = None

    def _plan_behavior_and_reference(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        speed_ref_mps: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        stop_goal_active: bool,
        cp_payload: Mapping[str, Any] | None = None,
    ):
        sim_time_s = self._sim_time_s()
        route_update = self.pipeline.update_route_from_cp(
            RouteUpdateRequest(
                route_manager=self.route_manager,
                ego_location=ego_location,
                cp_payload=cp_payload,
                stop_goal_active=bool(stop_goal_active),
                lane_closure_reroute_enabled=bool(
                    self.config.get("cp_lane_closure_reroute_enabled", True)
                ),
                reset_for_route_revision=self._reset_pipeline_for_route_revision,
            )
        )
        stop_goal_active = route_update.stop_goal_active
        local_map_snapshot = getattr(
            self, "_local_map_snapshot", LocalMapSnapshot()
        )
        reset_lane_change = getattr(
            self.behavior_planner, "_reset_lane_change_state", None
        )
        planning_context = self.pipeline.prepare_planning_context(
            PlanningContextRequest(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                object_snapshots=object_snapshots,
                cp_payload=cp_payload,
                local_map_snapshot=local_map_snapshot,
                route_manager=self.route_manager,
                maneuver_manager=self.maneuver_manager,
                reference_provider=self._stable_reference_line_provider,
                traffic_memory=self._full_traffic_memory,
                opportunistic_lane_change_enabled=bool(
                    self.full_allow_opportunistic_lane_change
                ),
                lane_change_start_lock_until_s=float(
                    self.full_lane_change_start_lock_s
                ),
                dense_traffic_lock_enabled=bool(
                    self.full_dense_traffic_lane_change_lock_enabled
                ),
                dense_object_count=int(self.full_dense_traffic_object_count),
                dense_risky_lane_count=int(
                    self.full_dense_traffic_risky_lane_count
                ),
                planning_speed_mps=float(speed_ref_mps),
                sim_time_s=float(sim_time_s),
                cruise_speed_mps=float(self.target_speed_mps),
                mpc_dt_s=float(self.mpc.dt_s),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                config=self.config,
                boundary_recovery_request=(
                    getattr(getattr(self, "_boundary_recovery", None), "request", None)
                    if bool(self.config.get("boundary_recovery_enabled", False))
                    else None
                ),
            ),
            resolve_actor_state=self._resolve_full_traffic_state_from_carla_actor,
            attempt_turn_replan=lambda trigger_reason: self._attempt_turn_route_replan(
                ego_location=ego_location,
                trigger_reason=str(trigger_reason),
            ),
            reset_lane_change=reset_lane_change,
            observe_stage_duration=self._accum_stage_ms,
        )
        adapter_output = planning_context.adapter_output
        planner_input_frame = planning_context.planner_input_frame
        object_snapshots = planning_context.mutable_object_snapshots()
        local_map_snapshot = planning_context.local_map_snapshot
        current_state = adapter_output.current_state
        current_lane_id = int(planning_context.current_lane_id)
        behavior_context = planning_context.behavior_context
        route_lane_change_required = bool(
            behavior_context.route_behavior.authorization.required_by_route
        )
        scenario_observation = behavior_context.scenario_observation
        turn_context = scenario_observation.turn_context
        scenario_result = scenario_observation.scenario
        upcoming_turn_direction = str(turn_context.direction)
        upcoming_turn_distance_m = float(turn_context.distance_m)
        scenario_decision = scenario_result.decision
        conflict_resolution = behavior_context.conflict_resolution
        lane_change_authorization = conflict_resolution.authorization
        behavior_stop_target = scenario_result.behavior_stop_target
        mpc_feedback = self.mpc_feedback.candidate_feedback(
            current_time_s=float(sim_time_s)
        )
        lane_change_commitment_pending_stabilization = bool(
            self._stable_reference_line_provider.snapshot(
                LANE_CHANGE
            ).mutable_samples()
        )
        executable_behavior = self.pipeline.prepare_executable_behavior(
            ExecutableBehaviorPreparationRequest(
                planning_context=planning_context,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                cruise_speed_mps=float(self.target_speed_mps),
                sim_time_s=float(sim_time_s),
                mpc_feedback=mpc_feedback,
                lane_change_reference_active=bool(
                    self._stable_reference_line_provider.snapshot(
                        LANE_CHANGE
                    ).active
                ),
                lane_change_commitment_active=bool(
                    lane_change_commitment_pending_stabilization
                ),
                stop_goal_active=bool(stop_goal_active),
                max_deceleration_mps2=float(
                    self.mpc.constraints.min_acceleration_mps2
                ),
                prepare_reference_lock=bool(
                    self.full_prepare_lane_change_reference_lock
                ),
                route_recovery_requested=bool(
                    self.maneuver_manager.route_recovery_pending
                ),
                static_obstacle_mpc_stall_failure_count=int(
                    self._static_obstacle_mpc_stall_failure_count()
                ),
                config=self.config,
                runtime_config=self.behavior_runtime_cfg,
            ),
            behavior_planner=self.behavior_planner,
            reference_map=self.reference_map,
            nearest_front_obstacles=self._nearest_front_obstacle_by_lane,
            attempt_replan=lambda obstacle: self._attempt_static_obstacle_route_replan(
                ego_location=ego_location, obstacle=obstacle,
            ),
            object_track_id=self._object_track_id,
            reset_lane_change=getattr(
                self.behavior_planner, "_reset_lane_change_state", None
            ),
            observe_stage_duration=self._accum_stage_ms,
        )
        stop_goal_active = executable_behavior.stop_goal_active
        turn_prepare_speed_suppressed_by_lane_change = bool(
            executable_behavior.turn_prepare_speed_suppressed
        )
        speed_frame = self.pipeline.plan_speed_from_stages(
            SpeedPlanningPreparationRequest(
                planning_context=planning_context,
                executable_behavior=executable_behavior,
                maneuver_manager=self.maneuver_manager,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                requested_speed_mps=float(speed_ref_mps),
                lane_change_commitment_active=bool(
                    lane_change_commitment_pending_stabilization
                ),
                config=self.config,
            )
        )
        speed_plan = speed_frame.speed_plan
        front_gap_m = speed_frame.front_gap_m
        front_gap_actor_id = speed_frame.front_actor_id
        front_gap_obstacle_speed_mps = speed_frame.front_obstacle_speed_mps
        front_obstacle_lane_id = speed_frame.front_obstacle_lane_id
        front_obstacle_is_source_lane = (
            speed_frame.front_obstacle_is_source_lane
        )
        planned_speed_mps = speed_frame.target_speed_mps
        stop_goal_active = bool(stop_goal_active or speed_frame.stop_goal_active)
        planner_mode = "INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL"

        prepared_reference = self.pipeline.prepare_behavior_reference(
            BehaviorReferencePreparationRequest(
                map_planner=self.reference_map,
                local_map=local_map_snapshot,
                planning_context=planning_context,
                executable_behavior=executable_behavior,
                speed_frame=speed_frame,
                behavior_runtime_config=self.behavior_runtime_cfg,
                planner_mode=str(planner_mode),
                lookahead_m=float(self.lookahead_m),
                ego_speed_mps=float(ego_speed_mps),
                horizon_steps=int(self.mpc.horizon_steps),
                dt_s=float(self.mpc.dt_s),
                sim_time_s=float(sim_time_s),
                stop_release_smooth_until_s=float(
                    self._stop_release_temp_smooth_until_sim_time_s
                ),
                authoritative_ego_waypoint=(
                    self._route_context.authoritative_ego_waypoint
                ),
            )
        )
        built_reference = prepared_reference.built_reference
        local_lane_center_reference = built_reference.mutable_samples()
        # Stash the pre-publication reference (the one Stage C/D build corridor
        # rows on) so the offline frame-replay hook can compare it against the
        # post-publication reference the MPC tracks. No-op unless armed.
        if isinstance(self.config.get("frame_capture"), Mapping):
            self._frame_capture_pre_reference = list(local_lane_center_reference)
        nominal_destination_state = built_reference.mutable_destination_state()
        nominal_freeze_count = int(built_reference.reference_freeze_count)
        reference_debug = PlannerDiagnosticsStage.build_reference_debug_from_stages(
            self,
            ReferenceDiagnosticsRequest(
                built_reference=built_reference,
                planning_context=planning_context,
                executable_behavior=executable_behavior,
                speed_frame=speed_frame,
                route_update=route_update,
                mpc_feedback=mpc_feedback,
            ),
        )
        # Cooperative arbitration precedes physical maneuver commitment.
        # The first proposal tick is deliberately deferred so both peers can
        # exchange the same proposed claims before either installs a locked
        # lane-change reference.  This is also the sole CAV resolution call
        # for the tick; its corridor rows are reused by MPC below.
        cooperative_frame = self.pipeline.resolve_cooperative(
            self._build_cooperative_request(
                cooperative_proposal=executable_behavior.cooperative_proposal,
                current_state=current_state,
                ego_location=ego_location,
                ego_yaw_rad=ego_yaw_rad,
                ego_speed_mps=ego_speed_mps,
                local_lane_center_reference=local_lane_center_reference,
                local_map_snapshot=local_map_snapshot,
                object_snapshots=object_snapshots,
                planned_speed_mps=planned_speed_mps,
                planner_input_frame=planner_input_frame,
                sim_time_s=sim_time_s,
            )
        )
        selected_reference = self.pipeline.select_candidate_reference(
            CandidatePlanningPreparationRequest(
                enabled=bool(self.full_candidate_pipeline_enabled),
                prepared_reference=prepared_reference,
                planning_context=planning_context,
                executable_behavior=executable_behavior,
                speed_frame=speed_frame,
                cooperative_frame=cooperative_frame,
                baseline_reference=local_lane_center_reference,
                baseline_destination_state=nominal_destination_state,
                baseline_debug=reference_debug,
                planner_config=self.config,
                lane_change_authorization=lane_change_authorization,
                stop_target=behavior_stop_target,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                object_snapshots=object_snapshots,
                current_acceleration_mps2=float(self._last_accel_mps2),
                current_steering_rad=float(self._last_steer_rad),
                route_required=bool(route_lane_change_required),
                traffic_stop_active=bool(scenario_decision.stop_goal_active),
                turn_prepare_speed_suppressed=bool(
                    turn_prepare_speed_suppressed_by_lane_change
                ),
                lane_change_mpc_stall_failure_count=int(
                    self._lane_change_mpc_stall_failure_count()
                ),
                route_revision=str(self.route_manager.route_revision),
                map_epoch=str(self.waypoint_backend or "admap"),
                upcoming_turn_direction=str(upcoming_turn_direction),
                upcoming_turn_distance_m=float(upcoming_turn_distance_m),
                lane_change_duration_s=float(
                    self.maneuver_manager.lane_change.resolved_duration_s
                ),
                lane_change_duration_reason=str(
                    self.maneuver_manager.lane_change.duration_comfort_reason
                ),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                validate_contract=self._validate_candidate_reference_contract,
            )
        )
        return self.pipeline.finalize_behavior_reference_from_stages(
            BehaviorReferenceFinalizationPreparationRequest(
                selected_reference=selected_reference,
                cooperative_frame=cooperative_frame,
                planning_context=planning_context,
                executable_behavior=executable_behavior,
                reference_provider=self._stable_reference_line_provider,
                maneuver_manager=self.maneuver_manager,
                config=self.config,
                planner_mode=str(planner_mode),
                sim_time_s=float(sim_time_s),
                ego_location=ego_location,
                ego_speed_mps=float(ego_speed_mps),
                front_gap_m=front_gap_m,
                horizon_steps=int(self.mpc.horizon_steps),
                dt_s=float(self.mpc.dt_s),
                route_revision=str(self.route_manager.route_revision),
                map_epoch=str(self.waypoint_backend or "admap"),
                reference_freeze_count=int(nominal_freeze_count),
            )
        )

    def _build_cooperative_request(
        self, *, cooperative_proposal, current_state, ego_location,
        ego_yaw_rad, ego_speed_mps, local_lane_center_reference,
        local_map_snapshot, object_snapshots, planned_speed_mps,
        planner_input_frame, sim_time_s,
    ):
        """Adapt OpenCDA/V2X inputs into the cooperative stage contract."""

        cav_intents = self._collect_cav_intents()
        return CooperativeArbitrationRequest(
            proposal=cooperative_proposal,
            current_state=current_state,
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            ego_speed_mps=ego_speed_mps,
            ego_actor_id=int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            local_lane_center_reference=local_lane_center_reference,
            local_map_snapshot=local_map_snapshot,
            object_snapshots=object_snapshots,
            planned_speed_mps=planned_speed_mps,
            prediction_revision=str(planner_input_frame.prediction.revision),
            predicted_objects=planner_input_frame.prediction.predicted_objects,
            sim_time_s=sim_time_s,
            cav_intents=cav_intents,
            transport_diagnostics=self._cav_transport_diagnostics,
            last_accel_mps2=self._last_accel_mps2,
        )

    def _publish_cav_intent(
        self, *, ego_location: Any, ego_yaw_rad: float,
        ego_speed_mps: float, sim_time_s: float,
    ) -> None:
        """Store this CAV's broadcast (planned trajectory + claim + pose) on
        ``self.last_cav_intent_payload`` for the V2X adapter to transport."""

        from opencda.planning_module.pipeline.cav_intent_codec import (
            build_ego_cav_intent,
            cav_intent_to_payload,
            rebase_mpc_state_plan,
        )

        states = getattr(self.mpc, "_last_x_solution", None)
        planned = rebase_mpc_state_plan(
            [] if states is None else states,
            plan_time_s=self.control_buffer.plan_time_s,
            now_s=float(sim_time_s),
            dt_s=float(self.mpc.dt_s),
            current_state=(
                float(ego_location.x), float(ego_location.y),
                float(ego_speed_mps), float(ego_yaw_rad),
            ),
        )
        self._cav_intent_sequence += 1
        extent = getattr(
            getattr(self.vehicle_manager.vehicle, "bounding_box", None),
            "extent", None,
        )
        intent = build_ego_cav_intent(
            actor_id=int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            position_xy=(float(ego_location.x), float(ego_location.y)),
            heading_rad=float(ego_yaw_rad),
            speed_mps=float(ego_speed_mps),
            claim=self._ego_cav_claim(sim_time_s=float(sim_time_s)),
            planned_states=planned,
            dt_s=float(self.mpc.dt_s),
            generated_at_s=float(sim_time_s),
            valid_for_s=float(self.config.get("cav_intent_valid_for_s", 0.5)),
            sequence=int(self._cav_intent_sequence),
            length_m=2.0 * float(getattr(extent, "x", 0.0)),
            width_m=2.0 * float(getattr(extent, "y", 0.0)),
        )
        self.last_cav_intent_payload = cav_intent_to_payload(intent)

    def _collect_cav_intents(self) -> list:
        """CavIntent list for every nearby CP-X CAV.

        The simulator adapter currently exposes nearby peers through
        ``v2x_manager.cav_nearby``.  Only their serialized intent payload is
        consumed here; peer CARLA pose and planner internals are deliberately
        not read.  The same payload boundary can therefore be replaced by a
        real V2X transport without changing planning logic.
        """

        if not self._cav_conflict_enabled:
            return []
        v2x_manager = getattr(self.vehicle_manager, "v2x_manager", None)
        cav_nearby = dict(getattr(v2x_manager, "cav_nearby", {}) or {})
        if not cav_nearby:
            return []
        from opencda.planning_module.pipeline.cav_intent_codec import collect_cav_intents

        records = []
        for cav_id, cav_manager in cav_nearby.items():
            cav_bridge = getattr(cav_manager, "cpx_planner", None)
            if cav_bridge is None:
                continue
            payload = getattr(cav_bridge, "last_cav_intent_payload", None)
            if payload is not None:
                records.append(payload)
        diagnostics = {
            "nearby_count": len(cav_nearby),
            "payload_count": len(records),
        }
        intents = collect_cav_intents(
            records,
            self_actor_id=int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            now_s=float(self._sim_time_s()),
            minimum_probability=float(
                self.config.get("cav_intent_minimum_probability", 0.05)
            ),
            diagnostics=diagnostics,
        )
        self._cav_transport_diagnostics = diagnostics
        return intents

    def _ego_cav_claim(self, *, sim_time_s: float):
        """Return the claim produced by this tick's planning stage.

        Intent publication is deliberately read-only.  Claim proposal,
        commitment, and release are owned by ``CooperativeClaimManager`` at
        the conflict-resolution call site; the V2X path must not advance or
        reconstruct that lifecycle.
        """

        del sim_time_s
        if not self._cav_conflict_enabled:
            return None
        return self.pipeline.cooperative.claims.current_claim

    def _attempt_turn_route_replan(
        self,
        *,
        ego_location: Any,
        trigger_reason: str = "turn_reference_unavailable",
    ) -> tuple[bool, bool, str]:
        """Request a bounded route rebuild while preserving stop-on-failure."""

        # Master isolation switch: while validating the route / authorization /
        # reference layers, replan must be off so a downstream failure cannot
        # feed back and reset route progress under the test.
        if not bool(self.config.get("route_replan_enabled", True)):
            self._route_replan_last_reason = "route_replan_disabled_for_isolation"
            return False, False, "route_replan_disabled_for_isolation"

        now_s = float(self._sim_time_s())
        cooldown_s = max(
            0.1,
            float(self.config.get("turn_route_replan_cooldown_s", 2.0)),
        )
        elapsed_s = float(now_s) - float(self._route_replan_last_attempt_s)
        if elapsed_s < cooldown_s:
            reason = (
                "route_replan_cooldown:"
                f"remaining={float(cooldown_s - elapsed_s):.2f}"
            )
            self._route_replan_last_reason = str(reason)
            return False, False, str(reason)

        self._route_replan_last_attempt_s = float(now_s)
        self._route_replan_attempt_count += 1
        result = self.route_manager.replan_from(
            start_point={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "z": float(getattr(ego_location, "z", 0.0)),
            },
            trigger_reason=str(trigger_reason),
        )
        self._route_replan_last_reason = str(result.reason)
        if not bool(result.success):
            return True, False, str(result.reason)

        self._reset_pipeline_for_route_revision(reason="turn_route_replanned")
        return True, True, str(result.reason)

    def _lane_change_mpc_stall_failure_count(self) -> int:
        """Consecutive MPC infeasibility count for the active lane-change target.

        Feeds LaneChangeLifecycleStage.release_completed's stall watchdog
        (see its docstring/comment). Only the committed maneuver's own target
        lane matters here -- self.mpc_feedback.active_records already keys
        failures by (decision, target_lane_id), so this just looks up the
        record for whatever the maneuver FSM currently owns, independent of
        which decision label produced it.
        """

        target_lane_id = int(self.maneuver_manager.lane_change.target_lane_id)
        if target_lane_id == 0:
            return 0
        best = 0
        for record in self.mpc_feedback.active_records:
            if int(record.get("target_lane_id", 0)) != target_lane_id:
                continue
            best = max(best, int(record.get("consecutive_failures", 0)))
        return best

    def _static_obstacle_mpc_stall_failure_count(self) -> int:
        """Consecutive MPC infeasibility count for the static-obstacle
        local-avoidance stage's own committed target lane.

        Same data source and shape as ``_lane_change_mpc_stall_failure_count``
        (self.mpc_feedback.active_records keys failures by
        (decision, target_lane_id) regardless of which FSM owns the target),
        but reads StaticObstacleStage's own target_lane_id -- borrowing a
        lane to get around a blocked lane is a separate state machine from
        the general lane_change maneuver and was not covered by that
        watchdog, so a target this stage commits to that turns out to be
        geometrically infeasible had no timeout: MPC keeps reporting
        "primal infeasible" every tick, the vehicle coasts to a stop, and
        StaticObstacleStage only ever clears target_lane_id on actually
        *reaching* it -- never on failing to.
        """

        stage = getattr(self.pipeline, "static_obstacle", None)
        target_lane_id = int(getattr(stage, "target_lane_id", 0) or 0)
        if target_lane_id == 0:
            return 0
        best = 0
        for record in self.mpc_feedback.active_records:
            if int(record.get("target_lane_id", 0)) != target_lane_id:
                continue
            best = max(best, int(record.get("consecutive_failures", 0)))
        return best

    def _lane_change_lifecycle(self):
        stage = getattr(self, "lane_change_lifecycle_stage", None)
        if stage is not None:
            return stage
        from opencda.planning_module.pipeline.lane_change_lifecycle_stage import (
            LaneChangeLifecycleStage,
        )
        stage = LaneChangeLifecycleStage(
            provider=self._stable_reference_line_provider,
            maneuver_manager=self.maneuver_manager,
            reference_generator=self.reference_generator,
            route_manager=self.route_manager,
            mpc=self.mpc,
            control_buffer=getattr(self, "control_buffer", None),
            config=self.config,
            target_speed_mps=float(getattr(self, "target_speed_mps", 3.0)),
            map_epoch=str(getattr(self, "waypoint_backend", "admap") or "admap"),
            vehicle_extent=lambda: getattr(
                getattr(
                    getattr(self, "vehicle_manager", None), "vehicle", None
                ),
                "bounding_box",
                None,
            ) and getattr(self.vehicle_manager.vehicle.bounding_box, "extent", None),
        )
        self.lane_change_lifecycle_stage = stage
        return stage

    # Transitional compatibility ports for focused tests and external tools.
    # The lifecycle implementation and state ownership live in the stage.
    def _reset_route_tracking_lane_change_reference(self) -> None:
        self._lane_change_lifecycle().reset_reference()

    def _release_completed_lane_change_commitment(
        self, *, current_lane_id: int, ego_location: Any, ego_yaw_rad: float,
        ego_speed_mps: float = 0.0,
    ) -> str:
        return self._lane_change_lifecycle().release_completed(
            current_lane_id=current_lane_id,
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            ego_speed_mps=float(ego_speed_mps),
            local_map=getattr(self, "_local_map_snapshot", None),
        )

    def _start_target_lane_stabilization(
        self, *, ego_location: Any, ego_yaw_rad: float,
    ) -> str:
        return self._lane_change_lifecycle()._start_stabilization(
            ego_location=ego_location, ego_yaw_rad=ego_yaw_rad
        )
    def _clear_turn_master_reference(self) -> None:
        provider = getattr(self, "_stable_reference_line_provider", None)
        if provider is not None:
            provider.release(TURN, event="reset")

    @staticmethod
    def _normalized_final_lc_state(
        *, decision: str, lc_state: str, lane_change_phase: str = ""
    ) -> str:
        """Compatibility wrapper; CandidateSelectionStage owns normalization."""

        if str(decision or "").strip().lower() not in {
            "lane_change_left", "lane_change_right",
        }:
            return str(lc_state or "LANE_KEEP")
        from opencda.planning_module.pipeline.candidate_selection_stage import (
            CandidateSelectionStage,
        )
        return CandidateSelectionStage.normalized_lane_change_state(
            decision=str(decision), lane_change_phase=str(lane_change_phase),
        )

    def _validate_candidate_reference_contract(
        self,
        *,
        decision: str,
        lc_state: str,
        current_lane_id: int,
        speed_ref_mps: float,
        stop_goal_active: bool,
        current_state: Sequence[float],
        destination_state: Sequence[float],
        lane_center_reference: Sequence[Mapping[str, object]],
        committed_lane_change_tracking_active: bool = False,
    ):
        from opencda.planning_module.pipeline.reference_contract import (
            contract_from_config,
            validate_reference_contract,
        )

        normalized_decision = str(decision or "").strip().lower()
        normalized_fsm = str(lc_state or "").strip().upper()
        lane_change_active = (
            normalized_decision in {"lane_change_left", "lane_change_right"}
            or normalized_fsm.startswith("EXECUTE_LANE_CHANGE")
        )
        turn_active = (
            normalized_decision in {"intersection_turn_left", "intersection_turn_right"}
            or normalized_fsm.startswith("INTERSECTION_TURN")
        )
        stop_like = bool(stop_goal_active) or normalized_decision in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        direct_target_tracking_enabled = bool(
            self.config.get(
                "route_tracking_lane_change_direct_target_tracking_enabled",
                False,
            )
        )
        contract_mode = (
            "emergency_stop"
            if normalized_decision == "emergency_brake"
            else "stop"
            if bool(stop_like)
            else "lane_change_direct"
            if bool(lane_change_active) and bool(direct_target_tracking_enabled)
            else "lane_change"
            if bool(lane_change_active)
            else "intersection_turn"
            if bool(turn_active)
            else "lane_follow"
        )
        expected_lane_id = int(current_lane_id)
        if bool(lane_change_active) and len(destination_state or []) >= 5:
            try:
                expected_lane_id = int(float(destination_state[4]))
            except (TypeError, ValueError):
                expected_lane_id = int(current_lane_id)
        contract = contract_from_config(
            mode=str(contract_mode),
            expected_lane_id=int(expected_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            config=dict(self.config),
            default_speed_mps=max(float(self.target_speed_mps), float(speed_ref_mps), 0.1),
        )
        recovery_reference_active = any(
            str(sample.get("lane_transition_kind", ""))
            == "ego_anchored_lane_recovery"
            for sample in list(lane_center_reference or [])[:2]
        )
        validation = validate_reference_contract(
            reference_samples=lane_center_reference,
            destination_state=destination_state,
            ego_state=current_state,
            contract=contract,
            check_destination_body_lateral=not bool(
                lane_change_active or turn_active or recovery_reference_active
            ),
            check_first_lateral=not bool(
                committed_lane_change_tracking_active and lane_change_active
            ),
        )
        if (
            bool(validation.valid)
            and bool(turn_active)
            and bool(
                self.config.get(
                    "reference_contract_turn_vehicle_footprint_enabled",
                    True,
                )
            )
        ):
            boundary_valid, boundary_reason = (
                self._reference_vehicle_footprint_boundary_valid(
                    reference_samples=lane_center_reference,
                )
            )
            if not bool(boundary_valid):
                validation.valid = False
                validation.violations.append(str(boundary_reason))
        return validation

    def _reference_vehicle_footprint_boundary_valid(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
    ) -> tuple[bool, str]:
        """Delegate candidate turn-corridor validation to ReferenceGenerator."""

        vehicle_manager = getattr(self, "vehicle_manager", None)
        vehicle = getattr(vehicle_manager, "vehicle", None)
        bounding_box = getattr(vehicle, "bounding_box", None)
        extent = getattr(bounding_box, "extent", None)
        ego_half_width_m = float(
            getattr(
                extent,
                "y",
                self.config.get("metrics_ego_half_width_m", 1.0),
            )
        )
        ego_half_length_m = float(
            getattr(
                extent,
                "x",
                self.config.get("reference_vehicle_half_length_m", 2.4),
            )
        )
        safety_margin_m = max(
            0.0,
            float(
                self.config.get(
                    "reference_contract_turn_boundary_margin_m",
                    0.15,
                )
            ),
        )
        max_failures = max(
            0,
            int(
                self.config.get(
                    "reference_contract_turn_max_boundary_failures",
                    1,
                )
            ),
        )
        validation = self._stable_reference_line_provider.validate_turn_swept_footprint(
            reference_samples=reference_samples,
            ego_half_width_m=max(0.1, float(ego_half_width_m)),
            ego_half_length_m=max(0.1, float(ego_half_length_m)),
            safety_margin_m=float(safety_margin_m),
            max_violations=int(max_failures),
        )
        return bool(validation.valid), (
            "" if bool(validation.valid) else str(validation.reason)
        )

    def _sim_time_s(self) -> float:
        try:
            snapshot = self.vehicle_manager.vehicle.get_world().get_snapshot()
            return float(snapshot.timestamp.elapsed_seconds)
        except Exception:
            return 0.0

    def prediction_snapshot_transform(self):
        """Return the obstacle-snapshot transform for the active
        prediction-knowledge ablation mode, or ``None`` for the default
        ``cv`` pipeline behaviour.

        ``cv``     -> None (constant-velocity/acceleration kinematic rollout).
        ``blind``  -> every obstacle frozen at its current pose ("no predicted
                      trajectory": ego knows position, assumes no motion).
        ``oracle`` -> each obstacle's recorded ground-truth future + a
                      maneuver-intent label, replayed from ``oracle_trace_path``.
        """

        if self._prediction_snapshot_transform_cached:
            return self._prediction_snapshot_transform_fn
        from opencda.planning_module.pipeline.prediction_ablation import (
            build_snapshot_transform,
        )
        self._prediction_snapshot_transform_fn = build_snapshot_transform(
            prediction_mode=self._prediction_mode,
            horizon_s=float(self.mpc.horizon_s),
            dt_s=float(self.mpc.dt_s),
            oracle_store=self._oracle_trace_store,
            freeze_unmatched_oracle=bool(
                self.config.get("oracle_freeze_unmatched", False)
            ),
            synthetic_actor_ids=tuple(
                int(value)
                for value in list(
                    self.config.get("synthetic_prediction_actor_ids", []) or []
                )
            ),
            synthetic_update_period_s=1.0 / max(
                0.1, float(self.config.get("prediction_update_hz", 5.0))
            ),
            synthetic_actor_activation=(
                self._synthetic_prediction_actor_activation
            ),
        )
        self._prediction_snapshot_transform_cached = True
        return self._prediction_snapshot_transform_fn

    def _obstacle_lane_step_fn(self):
        """Return a ``(x, y, distance_m) -> (x, y, heading_rad) | None``
        closure for lane-curve-aware obstacle prediction, or None to keep the
        old straight-line-only fallback.

        ``obstacle_future_trajectory`` (behavior_planner/trajectory_risk.py)
        only follows the lane centerline when given this closure; without
        it, every obstacle without a CP-supplied ``predicted_trajectory``
        keeps being extrapolated as a straight line at its current heading,
        which is wrong for a vehicle following a curved lane (e.g. mid-turn
        at an intersection).
        """

        if not bool(self.config.get("prediction_lane_following_enabled", True)):
            return None
        from utility.global_planner import lane_step_xy_heading

        _raw_get_waypoint_fn = self.reference_map.get_waypoint

        # Same-tick AD-map query cache. lane_step_xy_heading calls
        # get_waypoint_fn once per (object x horizon step) during prediction
        # extrapolation; measured ~50% of those queries land on a (x, y)
        # already queried earlier in the *same* tick (adjacent horizon steps
        # of a slow-moving object, or two objects predicted through the same
        # stretch of lane). This closure -- and so the cache -- is rebuilt
        # fresh every tick (called once per input_adapter.build()), so it can
        # never return a stale result: within one tick the map query for a
        # given point is a pure function of (x, y).
        _waypoint_cache: dict = {}

        def get_waypoint_fn(pose):
            key = (round(float(pose["x"]), 1), round(float(pose["y"]), 1))
            if key in _waypoint_cache:
                self._waypoint_cache_hits = getattr(self, "_waypoint_cache_hits", 0) + 1
                return _waypoint_cache[key]
            self._waypoint_cache_misses = getattr(self, "_waypoint_cache_misses", 0) + 1
            result = _raw_get_waypoint_fn(pose)
            _waypoint_cache[key] = result
            return result

        def _step(x_m: float, y_m: float, distance_m: float):
            result = lane_step_xy_heading(
                float(x_m),
                float(y_m),
                float(distance_m),
                get_waypoint_fn=get_waypoint_fn,
            )
            if result is None:
                self._prediction_lane_step_none_count += 1
            else:
                self._prediction_lane_step_resolved_count += 1
            return result

        return _step

    def _assign_obstacles_to_lanes(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        ego_waypoint: Any = None,
        ego_lane_id: int = 0,
    ) -> dict[str, int]:
        from utility.global_planner import canonical_lane_id_for_waypoint

        assignments: dict[str, int] = {}
        for snapshot in list(object_snapshots or []):
            obstacle_id = self._object_track_id(snapshot)
            if not obstacle_id:
                continue
            waypoint = self.reference_map.get_waypoint({
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "z": float(snapshot.get("z", 0.0)),
            })
            # Ego, obstacles, route and prediction all use opaque AD-map ids.
            # Cross-road continuity is handled by topology, not by rewriting
            # an obstacle into a small canonical lane index.
            lane_id = int(canonical_lane_id_for_waypoint(waypoint) or 0)
            if int(lane_id) != 0:
                assignments[obstacle_id] = int(lane_id)
        return assignments

    @staticmethod
    def _object_track_id(snapshot: Mapping[str, Any]) -> str:
        for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id"):
            value = snapshot.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        try:
            return "xy:{:.1f}:{:.1f}".format(
                float(snapshot.get("x", snapshot.get("x_m", 0.0))),
                float(snapshot.get("y", snapshot.get("y_m", 0.0))),
            )
        except Exception:
            return ""

    @staticmethod
    def _nearest_front_distance_by_lane(
        *,
        ego_snapshot: Mapping[str, object],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        lane_assignments: Mapping[str, int],
        available_lane_ids: Sequence[int],
    ) -> dict[int, float]:
        ego_x = float(ego_snapshot.get("x", 0.0))
        ego_y = float(ego_snapshot.get("y", 0.0))
        ego_psi = float(ego_snapshot.get("psi", 0.0))
        cos_h = math.cos(ego_psi)
        sin_h = math.sin(ego_psi)
        nearest: dict[int, float] = {}
        allowed = {int(lane_id) for lane_id in list(available_lane_ids or [])}
        for snapshot in list(obstacle_snapshots or []):
            obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
            lane_id = int(lane_assignments.get(obstacle_id, 0))
            if lane_id not in allowed:
                continue
            dx = float(snapshot.get("x", 0.0)) - ego_x
            dy = float(snapshot.get("y", 0.0)) - ego_y
            longitudinal = dx * cos_h + dy * sin_h
            if longitudinal <= 0.0:
                continue
            nearest[lane_id] = min(float(nearest.get(lane_id, float("inf"))), float(longitudinal))
        return {
            int(lane_id): float(distance)
            for lane_id, distance in nearest.items()
            if math.isfinite(float(distance))
        }

    @classmethod
    def _nearest_front_obstacle_by_lane(
        cls,
        *,
        ego_snapshot: Mapping[str, object],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        lane_assignments: Mapping[str, int],
        available_lane_ids: Sequence[int],
    ) -> dict[int, dict[str, Any]]:
        """Return the nearest complete front-obstacle record per lane."""

        ego_x = float(ego_snapshot.get("x", 0.0))
        ego_y = float(ego_snapshot.get("y", 0.0))
        ego_psi = float(ego_snapshot.get("psi", 0.0))
        cos_h = math.cos(ego_psi)
        sin_h = math.sin(ego_psi)
        allowed = {int(lane_id) for lane_id in list(available_lane_ids or [])}
        nearest: dict[int, dict[str, Any]] = {}
        for raw_snapshot in list(obstacle_snapshots or []):
            snapshot = dict(raw_snapshot)
            obstacle_id = cls._object_track_id(snapshot)
            lane_id = int(lane_assignments.get(str(obstacle_id), 0))
            if lane_id not in allowed:
                continue
            obstacle_x = float(snapshot.get("x", snapshot.get("x_m", 0.0)))
            obstacle_y = float(snapshot.get("y", snapshot.get("y_m", 0.0)))
            longitudinal_m = (
                (obstacle_x - ego_x) * cos_h
                + (obstacle_y - ego_y) * sin_h
            )
            if longitudinal_m <= 0.0:
                continue
            previous = nearest.get(int(lane_id))
            if previous is not None and float(
                previous.get("front_distance_m", float("inf"))
            ) <= float(longitudinal_m):
                continue
            snapshot["vehicle_id"] = str(obstacle_id)
            snapshot["x"] = float(obstacle_x)
            snapshot["y"] = float(obstacle_y)
            snapshot["v"] = max(
                0.0,
                float(snapshot.get("v", snapshot.get("speed_mps", 0.0))),
            )
            snapshot["front_distance_m"] = float(longitudinal_m)
            nearest[int(lane_id)] = snapshot
        return nearest

    def _attempt_static_obstacle_route_replan(
        self,
        *,
        ego_location: Any,
        obstacle: Mapping[str, object],
    ) -> tuple[bool, bool, str]:
        """Block the obstacle lane and atomically rebuild the active route."""

        block_fn = getattr(self.global_planner, "block_lane_at_position", None)
        if not callable(block_fn):
            reason = "static_obstacle_block_lane_unsupported"
            return True, False, str(reason)
        blocked_lane_id = block_fn({
            "x": float(obstacle.get("x", obstacle.get("x_m", 0.0))),
            "y": float(obstacle.get("y", obstacle.get("y_m", 0.0))),
            "z": float(obstacle.get("z", obstacle.get("z_m", 0.0))),
        })
        if blocked_lane_id is None:
            reason = "static_obstacle_lane_mapping_failed"
            return True, False, str(reason)
        self._static_obstacle_blocked_lane_id = blocked_lane_id

        result = self.route_manager.replan_from(
            start_point={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "z": float(getattr(ego_location, "z", 0.0)),
            },
            trigger_reason="static_obstacle",
        )
        if not bool(result.success):
            return True, False, str(result.reason)

        self._reset_pipeline_for_route_revision(
            reason="static_obstacle_route_replanned"
        )
        return True, True, str(result.reason)

    def _reset_pipeline_for_route_revision(self, *, reason: str) -> None:
        """Retire all trajectory/control state after an accepted route swap.

        Best-effort across every sub-reset: one step raising must not skip
        the rest and leave a mix of new-route and still-stale state, which
        is worse than any single subsystem staying stale on its own. This
        call only ever follows a route change RouteManager has already
        committed to, so a failure here must be loud (printed
        unconditionally, not gated on self.debug) -- silence would hide
        exactly the case where the visible route stops matching what
        RouteManager now believes is active.
        """

        def _step(name: str, fn: Callable[[], None]) -> None:
            try:
                fn()
            except Exception as exc:
                print(
                    "[CP-X OpenCDA Bridge] route-revision reset step "
                    "'%s' failed (reason=%s): %r" % (name, reason, exc)
                )

        _step(
            "active_route_summary",
            lambda: setattr(
                self,
                "_active_route_summary",
                self.route_manager.active_route_summary,
            ),
        )
        _step(
            "nominal_trajectory_generator",
            lambda: self.nominal_trajectory_generator.reset(source=str(reason)),
        )
        _step(
            "lane_change_reference",
            self._reset_route_tracking_lane_change_reference,
        )
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            _step(
                "maneuver_manager",
                lambda: maneuver_manager.reset(reason=str(reason)),
            )
        _step(
            "control_buffer",
            lambda: self.control_buffer.reset(reason=str(reason)),
        )
        mpc = getattr(self, "mpc", None)
        if mpc is not None and hasattr(mpc, "clear_previous_solution_seed"):
            _step("mpc_seed", mpc.clear_previous_solution_seed)

    def _load_cp_message_payload(self) -> dict[str, Any]:
        try:
            with open(self.cp_message_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return dict(payload or {})
        except Exception:
            return {}

    @staticmethod
    def _traffic_context_from_cp_control(
        *,
        selected_control: Mapping[str, object] | None,
        ego_location: carla.Location,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        if not isinstance(selected_control, Mapping):
            return {"signal_state": "unknown", "from_cp": False}, None
        state = str(
            selected_control.get(
                "signal_state",
                selected_control.get("state", "unknown"),
            )
            or "unknown"
        ).strip().lower()
        stop_line = selected_control.get("stop_line_position", selected_control.get("stop_line", None))
        stop_target = None
        if isinstance(stop_line, Mapping):
            x_value = stop_line.get("x", stop_line.get("x_m", None))
            y_value = stop_line.get("y", stop_line.get("y_m", None))
            if x_value is not None and y_value is not None:
                distance_m = math.hypot(float(x_value) - float(ego_location.x), float(y_value) - float(ego_location.y))
                stop_target = {
                    "x_m": float(x_value),
                    "y_m": float(y_value),
                    "lane_id": int(float(
                        selected_control.get("lane_id", stop_line.get("lane_id", 0)) or 0
                    )),
                    "road_id": int(float(
                        selected_control.get("road_id", stop_line.get("road_id", 0)) or 0
                    )),
                    "distance_m": float(distance_m),
                    "source": "opencda_cp_control",
                }
        context = {
            "signal_state": str(state),
            "signal_source": str(selected_control.get("source", "opencda_cp")),
            "source": str(selected_control.get("source", "opencda_cp")),
            "cp_control_id": str(selected_control.get("control_id", selected_control.get("id", ""))),
            "control_id": str(selected_control.get("control_id", selected_control.get("id", ""))),
            "signal_actor_id": str(
                selected_control.get(
                    "signal_actor_id",
                    selected_control.get(
                        "control_id",
                        selected_control.get("id", ""),
                    ),
                )
            ),
            "cp_provider_source": str(selected_control.get("provider_source", "")),
            "provider_source": str(selected_control.get("provider_source", "")),
            "from_cp": True,
            "traffic_control_from_cp": True,
            "confidence": float(selected_control.get("confidence", 1.0) or 0.0),
            "ego_passed_stop_line": bool(selected_control.get("ego_passed_stop_line", False)),
        }
        return context, stop_target

    def _select_relevant_traffic_control(
        self,
        *,
        traffic_controls: Sequence[Mapping[str, object]],
        ego_location: carla.Location,
        ego_heading_rad: float,
        current_lane_id: int,
        current_road_id: int,
        sim_time_s: float,
    ) -> Mapping[str, object] | None:
        best_control: Mapping[str, object] | None = None
        best_score: tuple[float, float, float] | None = None
        cos_h = math.cos(float(ego_heading_rad))
        sin_h = math.sin(float(ego_heading_rad))
        for control in list(traffic_controls or []):
            if not isinstance(control, Mapping):
                continue
            if not self._cp_message_is_fresh(control, sim_time_s=float(sim_time_s)):
                continue
            stop_line = control.get("stop_line_position", control.get("stop_line", None))
            if not isinstance(stop_line, Mapping):
                continue
            x_value = stop_line.get("x", stop_line.get("x_m", None))
            y_value = stop_line.get("y", stop_line.get("y_m", None))
            if x_value is None or y_value is None:
                continue
            dx_m = float(x_value) - float(ego_location.x)
            dy_m = float(y_value) - float(ego_location.y)
            forward_m = cos_h * dx_m + sin_h * dy_m
            lateral_m = -sin_h * dx_m + cos_h * dy_m
            if bool(control.get("ego_passed_stop_line", False)) or float(forward_m) < -1.0:
                continue
            lane_id = int(float(
                control.get(
                    "lane_id",
                    stop_line.get("lane_id", 0),
                )
                or 0
            ))
            road_id = int(float(
                control.get(
                    "road_id",
                    stop_line.get("road_id", 0),
                )
                or 0
            ))
            road_mismatch = 1.0 if road_id and current_road_id and road_id != current_road_id else 0.0
            lane_mismatch = 1.0 if lane_id and current_lane_id and lane_id != current_lane_id else 0.0
            score = (road_mismatch, lane_mismatch, abs(float(lateral_m)) + 0.01 * float(forward_m))
            if best_score is None or score < best_score:
                best_control = control
                best_score = score
        return best_control

    @staticmethod
    def _cp_message_is_fresh(message: Mapping[str, object], *, sim_time_s: float) -> bool:
        try:
            valid_until_s = float(message.get("valid_until_s", "nan"))
            if math.isfinite(valid_until_s):
                return float(sim_time_s) <= valid_until_s
        except Exception:
            pass
        try:
            timestamp_s = float(message.get("timestamp_s", sim_time_s))
            ttl_s = float(message.get("ttl_s", 0.0))
        except Exception:
            return True
        if float(ttl_s) <= 0.0:
            return True
        return float(sim_time_s) <= float(timestamp_s) + float(ttl_s)

    def _planning_module_global_route_summary(
        self,
        *,
        ego_location: carla.Location,
        ego_heading_rad: float,
        fallback_lane_id: int,
        ego_waypoint: Any = None,
    ) -> dict[str, object]:
        if getattr(self, "global_planner_backend", "") == "custom_admap_dijkstra":
            return self._route_context.build(
                global_planner=getattr(self, "global_planner", None),
                route_manager=getattr(self, "route_manager", None),
                clock=self._sim_time_s,
                ego_location=ego_location,
                ego_heading_rad=ego_heading_rad,
                fallback_lane_id=fallback_lane_id,
            )
        if not hasattr(self, "route_manager"):
            return {
                "route_found": False,
                "optimal_lane_id": int(fallback_lane_id),
                "current_road_option": "",
                "next_macro_maneuver": "Continue Straight",
                "debug_reason": "route_manager_unavailable",
            }
        return self.route_manager.get_route_info(
            x_m=float(ego_location.x),
            y_m=float(ego_location.y),
            query_key=f"vehicle_{int(getattr(self.vehicle_manager.vehicle, 'id', 0))}",
            fallback_lane_id=int(fallback_lane_id),
            ego_waypoint=ego_waypoint,
        )

    def _perception_diagnostics(self) -> dict[str, object]:
        manager = getattr(self.vehicle_manager, "perception_manager", None)
        activated = bool(getattr(manager, "activate", False))
        ml_active = bool(activated and getattr(manager, "ml_manager", None) is not None)
        return {
            "perception_mode": (
                "opencda_ml_yolov5_lidar_fusion"
                if ml_active
                else "carla_ground_truth"
            ),
            "perception_ml_active": bool(ml_active),
            "perception_camera_count": int(
                getattr(manager, "camera_num", 0) or 0
            ),
        }

    def _mpc_object_snapshots_with_prediction(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        prediction_trajectories: Mapping[str, Sequence[Mapping[str, object]]],
    ) -> list[dict[str, Any]]:
        """Attach this tick's predicted trajectory to each MPC obstacle.

        Without this, ``MPC._get_object_state_at_stage`` (MPC/mpc.py) never
        sees ``planner_input_frame.prediction.obstacle_future_trajectories``
        at all -- it only recognizes a ``predicted_trajectory`` already
        shaped as one ``[x, y, v, psi]`` entry per stage, so it silently
        falls back to its own constant-velocity extrapolation for every
        obstacle, independent of (and less accurate than) the
        constant-acceleration/CP-supplied prediction the rest of the
        pipeline already computed. ``bridge.tracker.predict()`` is already
        called with ``horizon_s=self.mpc.horizon_s, dt_s=self.mpc.dt_s`` (see
        planner_input_adapter.py), so each trajectory here already has one
        point per MPC stage -- this only needs to convert the shape and
        attach it, not resample it.
        """

        from opencda.planning_module.pipeline.prediction import (
            mpc_stage_trajectory,
            obstacle_track_id,
        )

        if not prediction_trajectories:
            return [dict(snapshot) for snapshot in list(object_snapshots or [])]
        horizon_steps = int(self.mpc.horizon_steps)
        dt_s = float(self.mpc.dt_s)
        annotated: list[dict[str, Any]] = []
        for snapshot in list(object_snapshots or []):
            if not isinstance(snapshot, Mapping):
                continue
            updated = dict(snapshot)
            points = prediction_trajectories.get(obstacle_track_id(snapshot))
            if points:
                updated["predicted_trajectory"] = mpc_stage_trajectory(
                    list(points),
                    fallback_heading_rad=float(snapshot.get("psi", snapshot.get("heading_rad", 0.0))),
                    horizon_steps=horizon_steps,
                    dt_s=dt_s,
                )
            annotated.append(updated)
        return annotated


    @staticmethod
    def _cooperative_actor_evidence(
        *,
        cp_summary: Mapping[str, Any],
        prediction_trajectories: Mapping[
            str, Sequence[Mapping[str, Any]]
        ],
        selected_reference: Sequence[Mapping[str, Any]],
        candidate_proximity_m: float = 3.0,
    ) -> dict[str, Any]:
        """Build an auditable CP-to-prediction-to-candidate evidence chain.

        ``candidate_relevant`` means that a predicted actor position enters
        the selected reference's spatial safety envelope at a corresponding
        horizon step. It deliberately does not claim that the actor changed
        the selected decision; proving that stronger counterfactual requires
        evaluating the same candidate set with that actor removed.
        """

        provenance = [
            dict(item)
            for item in list(cp_summary.get("actor_provenance", []) or [])
            if isinstance(item, Mapping)
        ]
        prediction_by_actor = {
            str(key).rsplit(":", 1)[-1]: list(points or [])
            for key, points in dict(prediction_trajectories or {}).items()
        }
        reference = list(selected_reference or [])
        prediction_used_ids: list[str] = []
        prediction_used_pedestrian_ids: list[str] = []
        candidate_relevant_ids: list[str] = []
        candidate_relevant_pedestrian_ids: list[str] = []
        evidence: list[dict[str, Any]] = []

        for actor in provenance:
            message_id = str(actor.get("actor_id", ""))
            actor_id = message_id.rsplit(":", 1)[-1]
            actor_type = str(actor.get("actor_type", "unknown"))
            predicted_points = prediction_by_actor.get(actor_id, [])
            used_by_prediction = bool(predicted_points)
            min_distance_m: float | None = None
            if used_by_prediction and reference:
                for index in range(min(len(reference), len(predicted_points))):
                    try:
                        ref = reference[index]
                        point = predicted_points[index]
                        rx = float(ref.get("x_ref_m", ref.get("x", 0.0)))
                        ry = float(ref.get("y_ref_m", ref.get("y", 0.0)))
                        px = float(point.get("x", point.get("x_m", 0.0)))
                        py = float(point.get("y", point.get("y_m", 0.0)))
                    except (TypeError, ValueError):
                        continue
                    distance_m = math.hypot(rx - px, ry - py)
                    min_distance_m = (
                        distance_m
                        if min_distance_m is None
                        else min(min_distance_m, distance_m)
                    )
            candidate_relevant = bool(
                min_distance_m is not None
                and min_distance_m <= float(candidate_proximity_m)
            )
            if used_by_prediction:
                prediction_used_ids.append(message_id)
                if actor_type == "pedestrian":
                    prediction_used_pedestrian_ids.append(message_id)
            if candidate_relevant:
                candidate_relevant_ids.append(message_id)
                if actor_type == "pedestrian":
                    candidate_relevant_pedestrian_ids.append(message_id)
            evidence.append({
                **actor,
                "used_by_prediction": bool(used_by_prediction),
                "candidate_relevant": bool(candidate_relevant),
                "candidate_min_predicted_distance_m": (
                    None if min_distance_m is None else float(min_distance_m)
                ),
            })

        pedestrian_count = sum(
            str(item.get("actor_type", "")) == "pedestrian"
            for item in provenance
        )
        blind_pedestrian_count = sum(
            str(item.get("actor_type", "")) == "pedestrian"
            and bool(item.get("blind_spot_shared", False))
            for item in provenance
        )
        return {
            "cp_actor_provenance": json.dumps(provenance, default=str),
            "cp_pedestrian_count": int(pedestrian_count),
            "cp_blind_spot_pedestrian_count": int(blind_pedestrian_count),
            "cp_prediction_used_actor_ids": ",".join(prediction_used_ids),
            "cp_prediction_used_pedestrian_ids": ",".join(
                prediction_used_pedestrian_ids
            ),
            "cp_candidate_relevant_actor_ids": ",".join(
                candidate_relevant_ids
            ),
            "cp_candidate_relevant_pedestrian_ids": ",".join(
                candidate_relevant_pedestrian_ids
            ),
            "cp_actor_evidence": json.dumps(evidence, default=str),
        }

    def _world_debug(self):
        port = getattr(self, "world_debug_port", None)
        if port is not None:
            return port
        from opencda.planning_module.opencda_bridge.platform_ports import WorldDebugPort
        port = WorldDebugPort(
            vehicle_manager=getattr(self, "vehicle_manager", None),
            carla_module=getattr(self, "carla", None),
            mpc=getattr(self, "mpc", None),
            route_points=self._active_global_route_points,
            enabled=bool(getattr(self, "draw_world_debug", False)),
            draw_destination=bool(getattr(self, "draw_world_debug_destination", False)),
            life_time_s=float(getattr(self, "world_debug_life_time_s", 0.15)),
            report_error=lambda reason: print(
                "[CP-X OpenCDA Bridge] world debug draw failed: " + reason
            ) if bool(getattr(self, "debug", False)) else None,
        )
        self.world_debug_port = port
        return port

    def _draw_world_debug_primitives(
        self, *, destination_state: Sequence[float],
        lane_center_reference: Sequence[Mapping[str, Any]],
    ) -> None:
        self._world_debug().draw(
            destination_state=destination_state,
            reference_samples=lane_center_reference,
        )

    def _last_mpc_trajectory_points(self) -> list[tuple[float, float]]:
        return self._world_debug().mpc_trajectory_points()
    def _active_global_route_points(self) -> list[list[float]]:
        """Return planner-owned route topology geometry; never display-filter it."""

        latest_update = dict(getattr(self, "_latest_opencda_update", {}) or {})
        ego_transform = latest_update.get("ego_transform")
        if ego_transform is not None:
            loc = ego_transform.location
            return self.route_manager.geometry_route_points(
                x_m=float(loc.x),
                y_m=float(loc.y),
                query_key=f"vehicle_{int(getattr(self.vehicle_manager.vehicle, 'id', 0))}_polyline",
            )
        route_points = self.route_manager.geometry_route_points()
        return [list(point) for point in route_points]

    def _display_global_route_points(self) -> list[list[float]]:
        """Return the diagnostics-only smoothed route polyline."""

        return self._world_debug().display_route_points()
    def _map_waypoint_from_location(self, location: carla.Location):
        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        port = getattr(self, "map_lookup_port", None) or MapLookupPort.from_owner(self)
        return port.waypoint(location)

    def _drivable_waypoint_from_location(
        self,
        location: carla.Location,
    ):
        """Return a waypoint only when the point is on a driving lane."""

        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        port = getattr(self, "map_lookup_port", None) or MapLookupPort.from_owner(self)
        return port.drivable_waypoint(location) or port.waypoint(location)

    def _lane_id_at_location(self, location: carla.Location) -> int:
        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        port = getattr(self, "map_lookup_port", None) or MapLookupPort.from_owner(self)
        return port.lane_id(location)

    @staticmethod
    def _location_to_point(location: Any) -> dict[str, float]:
        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        return MapLookupPort.location_to_point(location)

    @staticmethod
    def _body_frame_xy(
        *,
        origin_x_m: float,
        origin_y_m: float,
        heading_rad: float,
        target_x_m: float,
        target_y_m: float,
    ) -> tuple[float, float]:
        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        return MapLookupPort.body_frame_xy(
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            heading_rad=heading_rad,
            target_x_m=target_x_m,
            target_y_m=target_y_m,
        )

    def _set_actuator_context(
        self,
        *,
        ego_speed_mps: float,
        target_speed_mps: float,
        stop_goal_active: bool,
    ) -> None:
        self.actuator_port.set_context(
            ego_speed_mps=ego_speed_mps,
            target_speed_mps=target_speed_mps,
            stop_goal_active=stop_goal_active,
        )

    def _control_from_mpc(self, acceleration_mps2: float, steering_angle_rad: float) -> carla.VehicleControl:
        return self.actuator_port.control(acceleration_mps2, steering_angle_rad)

    def _accel_from_control(self, control: carla.VehicleControl) -> float:
        return self.actuator_port.acceleration(control)

    def _steer_rad_from_control(self, control: carla.VehicleControl) -> float:
        return self.actuator_port.steering(control)

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi



def cpx_planner_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether a vehicle config requests the CP-X planner bridge."""

    planner_cfg = dict(config.get("planner", {}) or {})
    if planner_cfg and not bool(planner_cfg.get("enabled", True)):
        return False
    planner_type = str(planner_cfg.get("type", "")).strip().lower()
    env_type = str(os.environ.get("OPENCDA_PLANNER", "")).strip().lower()
    if planner_type:
        return planner_type in {"cpx_mpc", "cp_x_mpc"}
    return env_type in {"cpx_mpc", "cp_x_mpc"}
