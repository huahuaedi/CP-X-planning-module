"""Finalize one MPC result into the sole platform-safe control command.

This is step 3 of the bridge's control-writing precedence
(TrajectoryFallbackManager reference substitution -> MPCExecutionStage's
bounded safe stop, its only control -> this stage, which alone decides
whether a hard gate is an emergency (see safety_supervisor) and builds the
platform control for every other case -> SafetySupervisor last-mile filter ->
CPXMPCPlannerBridge.run_step()'s try/except as the absolute last resort) --
see run_step()'s docstring for the full chain in one place before changing
what any single layer does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .safety_supervisor import emergency_stop_reason


@dataclass(frozen=True)
class ControlFinalizationRequest:
    execution: Any
    behavior: Any
    ego_transform: Any
    ego_speed_mps: float
    target_speed_mps: float
    destination_state: Sequence[float]
    reference_samples: Sequence[Mapping[str, Any]]
    stop_goal_active: bool
    stop_target_forward_m: Any
    final_reference_accepted: bool
    candidate_status: str
    safety_manager: Any
    make_pedal_control: Callable[..., Any]
    sim_time_s: float
    # The optimized near-term velocity is a platform safety ceiling only when
    # Stage D installed a longitudinal space-time corridor.  In an ordinary
    # single-vehicle tick, SpeedTargetPlanner is the velocity owner; feeding
    # MPC's near-term state to the downstream PID would apply acceleration
    # dynamics twice and make turns unnecessarily slow.
    mpc_velocity_safety_cap_active: bool = False
    # Set once CAVConflictSchedule reports that a CAV corridor has remained
    # infeasible (even true max braking cannot satisfy it) for its configured
    # debounce period. Escalates to
    # the same full brake + zero-steer response already reserved for a
    # confirmed collision-risk hard gate below, instead of continuing to
    # trust whatever the (slack-relaxed) QP solution produced.
    corridor_infeasible_escalate: bool = False
    # Raw proximity is safety evidence, not a nominal stop goal.  Keeping it
    # here prevents the emergency-gap check from changing BehaviorDecision,
    # reference selection, or SpeedTarget before the final safety owner.
    proximity_emergency_stop_required: bool = False


@dataclass(frozen=True)
class ControlFinalizationResult:
    control: Any
    acceleration_mps2: float
    steering_rad: float
    pre_filter_acceleration_mps2: float
    pre_filter_steering_rad: float
    control_guard_reason: str
    boundary_guard_reason: str
    boundary_snapshot: Any
    supervisor_reason: str
    platform_debug: Mapping[str, Any]
    feedback_reason: str


class ControlFinalizationStage:
    """Own platform conversion, feedback recording and final safety filtering."""

    def __init__(
        self, *, mpc: Any, command_extractor: Any, feedback: Any,
        control_safety: Any, config: Mapping[str, Any],
    ) -> None:
        self._mpc = mpc
        self._extractor = command_extractor
        self._feedback = feedback
        self._safety = control_safety
        self._config = config

    def run(
        self,
        request: ControlFinalizationRequest,
        *,
        set_actuator_context: Callable[..., None],
        acceleration_from_control: Callable[[Any], float],
        steering_from_control: Callable[[Any], float],
        apply_velocity_steering: Callable[..., tuple],
        control_factory: Callable[..., Any],
        boundary_metrics: Callable[..., Any],
        update_boundary_recovery: Callable[..., Any],
        reset_boundary_recovery: Callable[..., Any],
    ) -> ControlFinalizationResult:
        execution = request.execution
        acceleration = float(execution.acceleration_mps2)
        steering = float(execution.steering_rad)
        control = execution.control
        fallback_reason = str(execution.fallback_reason)
        set_actuator_context(
            ego_speed_mps=float(request.ego_speed_mps),
            target_speed_mps=float(request.target_speed_mps),
            stop_goal_active=bool(request.stop_goal_active),
        )
        if control is not None:
            acceleration = float(acceleration_from_control(control))
            steering = float(steering_from_control(control))
        feedback_reason = self._feedback.record_result(
            decision=str(request.behavior.maneuver),
            target_lane_id=int(request.behavior.target_lane_id),
            status=str(execution.status),
            reason=fallback_reason,
            timestamp_s=float(request.sim_time_s),
            success=not bool(fallback_reason),
        )
        platform_debug = {"control_interface": "mpc_acceleration_steering"}
        hard_gate_active = fallback_reason.startswith("candidate_hard_gate:")
        if (
            not fallback_reason
            or hard_gate_active
            or bool(execution.failed_replan_buffer_reused)
        ):
            maneuver = str(request.behavior.maneuver).strip().lower()
            solution = getattr(self._mpc, "_last_x_solution", None)
            if (
                str(execution.status).strip().lower()
                in {"solved", "solved inaccurate"}
                and solution is not None
                and len(solution) > 0
            ):
                tracking = self._extractor.extract(
                    state_solution=solution,
                    steering_rad=steering,
                    solution_dt_s=float(self._mpc.dt_s),
                    timestamp_s=float(request.sim_time_s),
                    max_velocity_mps=float(self._mpc.constraints.max_velocity_mps),
                )
            else:
                tracking = self._extractor.hold(
                    steering_rad=steering,
                    reason=(
                        "mpc_velocity_command_hold_after_failed_replan"
                        if bool(execution.failed_replan_buffer_reused)
                        else "mpc_velocity_command_hold_control_buffer"
                    ),
                )
            emergency_reason = emergency_stop_reason(
                fallback_reason=fallback_reason,
                behavior_decision=maneuver,
                stop_goal_active=bool(request.stop_goal_active),
                corridor_infeasible_escalate=bool(
                    request.corridor_infeasible_escalate
                ),
                proximity_emergency_stop_required=bool(
                    request.proximity_emergency_stop_required
                ),
            )
            emergency = bool(emergency_reason)
            platform_target = self._extractor.platform_target_velocity(
                nominal_velocity_mps=float(request.target_speed_mps),
                stop_goal_active=bool(request.stop_goal_active),
                emergency_stop=emergency,
                mpc_safety_cap_mps=(
                    float(tracking.target_velocity_mps)
                    if (
                        bool(tracking.valid)
                        and bool(request.mpc_velocity_safety_cap_active)
                    )
                    else None
                ),
            )
            control, acceleration, steering, platform_debug = (
                apply_velocity_steering(
                    target_speed_mps=float(platform_target),
                    target_steering_rad=float(tracking.target_steering_rad),
                    actual_speed_mps=float(request.ego_speed_mps),
                    stop_goal_active=bool(request.stop_goal_active),
                    emergency_stop=emergency,
                    sim_time_s=float(request.sim_time_s),
                )
            )
            platform_debug.update({
                "mpc_velocity_preview_time_s": float(tracking.velocity_preview_time_s),
                "mpc_velocity_source_index": float(tracking.velocity_source_index),
                "mpc_optimized_velocity_mps": float(tracking.target_velocity_mps),
                "nominal_speed_ref_mps": float(request.target_speed_mps),
                "pid_target_velocity_mps": float(platform_target),
                "mpc_velocity_safety_cap_active": bool(
                    request.mpc_velocity_safety_cap_active
                ),
                "corridor_infeasible_escalate": bool(
                    request.corridor_infeasible_escalate
                ),
                "proximity_emergency_stop_required": bool(
                    request.proximity_emergency_stop_required
                ),
                "emergency_stop_reason": str(emergency_reason),
                "velocity_command_source": (
                    "mpc_corridor_velocity_safety_cap"
                    if (
                        bool(tracking.valid)
                        and bool(request.mpc_velocity_safety_cap_active)
                        and float(platform_target)
                        < float(request.target_speed_mps) - 1.0e-9
                    )
                    else "speed_target_planner"
                ),
                "velocity_command_valid": bool(tracking.valid),
            })
        safety = self._safety.run(
            control=control,
            make_pedal_control=request.make_pedal_control,
            acceleration_mps2=acceleration,
            steering_rad=steering,
            ego_transform=request.ego_transform,
            ego_speed_mps=float(request.ego_speed_mps),
            destination_state=request.destination_state,
            behavior=request.behavior,
            stop_goal_active=bool(request.stop_goal_active),
            stop_target_forward_m=request.stop_target_forward_m,
            min_acceleration_mps2=float(self._mpc.constraints.min_acceleration_mps2),
            control_factory=control_factory,
            reference_samples=request.reference_samples,
            final_reference_accepted=bool(request.final_reference_accepted),
            candidate_status=str(request.candidate_status),
            safety_manager=request.safety_manager,
            sim_time_s=float(request.sim_time_s),
            boundary_metrics=boundary_metrics,
            update_boundary_recovery=update_boundary_recovery,
            reset_boundary_recovery=reset_boundary_recovery,
            acceleration_from_control=acceleration_from_control,
            steering_from_control=steering_from_control,
        )
        return ControlFinalizationResult(
            control=safety.control,
            acceleration_mps2=float(safety.acceleration_mps2),
            steering_rad=float(safety.steering_rad),
            pre_filter_acceleration_mps2=float(safety.pre_filter_acceleration_mps2),
            pre_filter_steering_rad=float(safety.pre_filter_steering_rad),
            control_guard_reason=str(safety.control_guard_reason),
            boundary_guard_reason=str(safety.boundary_guard_reason),
            boundary_snapshot=safety.boundary_snapshot,
            supervisor_reason=str(safety.supervisor_reason),
            platform_debug=dict(platform_debug),
            feedback_reason=str(feedback_reason),
        )
