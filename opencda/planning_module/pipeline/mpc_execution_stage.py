"""Single owner for MPC solve scheduling, control reuse, and solve failure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class MPCExecutionRequest:
    sim_time_s: float
    current_state: Sequence[float]
    destination_state: Sequence[float]
    reference_samples: Sequence[Mapping[str, object]]
    object_snapshots: Sequence[Mapping[str, object]]
    current_acceleration_mps2: float
    current_steering_rad: float
    ego_speed_mps: float
    target_speed_mps: float
    stop_goal_active: bool
    behavior_maneuver: str
    behavior_phase: str
    hard_gate_reason: str
    stationary_stop_hold: bool
    control_context: Any
    road_envelope_payload_world: Any = None
    speed_crossing_deadband_mps: float = 0.15
    # Stage-D interaction corridor rows (pipeline.mpc_corridor_constraints);
    # empty for a single-vehicle tick.
    corridor_rows: Sequence[Any] = ()


@dataclass(frozen=True)
class MPCExecutionResult:
    acceleration_mps2: float
    steering_rad: float
    control: Any
    status: str
    fallback_reason: str
    replan_executed: bool
    failed_replan_buffer_reused: bool
    jerk_seed_acceleration_mps2: float = 0.0


class MPCExecutionStage:
    """Execute exactly one MPC/cache path and own its degradation policy.

    The bridge supplies runtime adapters, but cannot choose between solving,
    reusing, or stopping. A failed solve may reuse one still-valid optimized
    command; otherwise it degrades to the single safe-stop control factory.
    """

    def __init__(self, *, mpc: Any, control_buffer: Any,
                 minimum_replan_speed_mps: float = 1.5) -> None:
        self._mpc = mpc
        self._buffer = control_buffer
        self._minimum_replan_speed_mps = max(
            0.0, float(minimum_replan_speed_mps)
        )
        self._planned_acceleration_mps2 = None
        self._last_command_time_s = None

    def run(
        self,
        request: MPCExecutionRequest,
        *,
        normal_stop_control: Callable[[], Any],
        safe_stop_control: Callable[[float, float], Any],
        emergency_stop_control: Callable[[], Any],
    ) -> MPCExecutionResult:
        context = request.control_context
        context_key = str(context.key)
        anchor = context.reference_anchor_relative_m
        hard_gate = bool(str(request.hard_gate_reason))
        jerk_seed_acceleration_mps2 = (
            float(request.current_acceleration_mps2)
            if self._planned_acceleration_mps2 is None
            else float(self._planned_acceleration_mps2)
        )
        elapsed_s = (
            float(self._mpc.dt_s)
            if self._last_command_time_s is None
            else max(
                1.0e-3,
                min(
                    float(self._mpc.dt_s),
                    float(request.sim_time_s) - float(self._last_command_time_s),
                ),
            )
        )
        self._last_command_time_s = float(request.sim_time_s)
        if hard_gate:
            self._buffer.reset(reason="control_buffer_reference_hard_veto")
        elif request.stationary_stop_hold:
            self._buffer.update_from_solution(
                u_solution=[[0.0, 0.0]],
                plan_time_s=float(request.sim_time_s),
                dt_s=float(self._mpc.dt_s),
                context_key=context_key,
                reference_anchor_relative_m=anchor,
            )

        fallback_reason = ""
        reused_after_failure = False
        control = None
        acceleration = float(request.current_acceleration_mps2)
        steering = float(request.current_steering_rad)
        replan = False
        try:
            if hard_gate:
                raise RuntimeError(str(request.hard_gate_reason))
            low_speed_replan = bool(
                not request.stop_goal_active
                and str(request.behavior_maneuver).strip().lower() == "lane_follow"
                and str(request.behavior_phase).strip().upper()
                in {"", "IDLE", "LANE_KEEP"}
                and float(request.ego_speed_mps) < self._minimum_replan_speed_mps
            )
            replan = bool(self._buffer.should_replan(
                sim_time_s=float(request.sim_time_s),
                force_replan=bool(
                    not request.stationary_stop_hold
                    and (bool(context.force_replan) or low_speed_replan)
                ),
                context_key=context_key,
                reference_anchor_relative_m=anchor,
                ego_speed_mps=float(request.ego_speed_mps),
                target_speed_mps=float(request.target_speed_mps),
                speed_error_crossing_deadband_mps=float(
                    request.speed_crossing_deadband_mps
                ),
            ))
            if replan:
                self._mpc.plan_trajectory(
                    current_state=request.current_state,
                    destination_state=request.destination_state,
                    object_snapshots=request.object_snapshots,
                    current_acceleration_mps2=float(
                        jerk_seed_acceleration_mps2
                    ),
                    current_steering_rad=float(request.current_steering_rad),
                    lane_center_reference_samples=request.reference_samples,
                    stop_goal_active=bool(request.stop_goal_active),
                    road_envelope_payload_world=request.road_envelope_payload_world,
                    corridor_rows=request.corridor_rows,
                )
                status = str(getattr(self._mpc, "_last_status", "")).strip().lower()
                if status and status not in {"solved", "solved inaccurate"}:
                    raise RuntimeError("MPC status=" + status)
                solution = getattr(self._mpc, "_last_u_solution", None)
                if solution is None or len(solution) == 0:
                    raise RuntimeError("MPC did not expose a control solution")
                states = getattr(self._mpc, "_last_x_solution", None)
                speeds = (
                    None if states is None or len(states) == 0
                    else [float(state[2]) for state in states]
                )
                self._buffer.update_from_solution(
                    u_solution=solution,
                    plan_time_s=float(request.sim_time_s),
                    dt_s=float(self._mpc.dt_s),
                    context_key=context_key,
                    reference_anchor_relative_m=anchor,
                    predicted_speed_sequence_mps=speeds,
                    target_speed_mps=float(request.target_speed_mps),
                )
                acceleration = float(solution[0, 0])
                steering = float(solution[0, 1])
            else:
                buffered = self._buffer.sample(
                    sim_time_s=float(request.sim_time_s),
                    context_key=context_key,
                    reference_anchor_relative_m=anchor,
                )
                if buffered is None:
                    raise RuntimeError("MPC control buffer empty")
                acceleration, steering, _ = buffered
                status = (
                    "stop_hold_direct"
                    if request.stationary_stop_hold else "buffer_reuse"
                )
            if request.stationary_stop_hold:
                control = normal_stop_control()
                acceleration = 0.0
            self._planned_acceleration_mps2 = float(acceleration)
            return MPCExecutionResult(
                float(acceleration), float(steering), control, str(status), "",
                bool(replan), False, float(jerk_seed_acceleration_mps2),
            )
        except Exception as exc:
            fallback_reason = str(exc)
            hard_gate = fallback_reason.startswith("candidate_hard_gate:")
            replan = False if hard_gate else True
            buffered = (
                self._buffer.sample(
                    sim_time_s=float(request.sim_time_s),
                    context_key=context_key,
                    reference_anchor_relative_m=anchor,
                )
                if not hard_gate and not request.stop_goal_active else None
            )
            if buffered is not None:
                acceleration, steering, _ = buffered
                reused_after_failure = True
                status = "buffer_reuse_after_failed_replan"
            else:
                constraints = self._mpc.constraints
                jerk_step_mps2 = max(
                    0.0,
                    float(constraints.max_jerk_mps3) * float(elapsed_s),
                )
                acceleration = max(
                    float(constraints.min_acceleration_mps2),
                    min(
                        0.0,
                        float(jerk_seed_acceleration_mps2)
                        - float(jerk_step_mps2),
                    ),
                )
                control = (
                    emergency_stop_control()
                    if hard_gate else safe_stop_control(
                        float(acceleration), float(steering)
                    )
                )
                steering = 0.0 if hard_gate else float(steering)
                status = "candidate_hard_gate" if hard_gate else "bounded_safe_stop"
            self._planned_acceleration_mps2 = float(acceleration)
            return MPCExecutionResult(
                float(acceleration), float(steering), control, str(status),
                fallback_reason, bool(replan), bool(reused_after_failure),
                float(jerk_seed_acceleration_mps2),
            )
