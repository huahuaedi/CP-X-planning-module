"""Ordered final-control safety arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ControlSafetyResult:
    control: Any
    acceleration_mps2: float
    steering_rad: float
    pre_filter_acceleration_mps2: float
    pre_filter_steering_rad: float
    control_guard_reason: str
    boundary_guard_reason: str
    supervisor_reason: str
    boundary_snapshot: Optional[Mapping[str, object]]


class ControlSafetyStage:
    """Apply all final control vetoes in one deterministic order."""

    def __init__(self, *, supervisor: Any, config: Mapping[str, object]) -> None:
        self.supervisor = supervisor
        self.config = dict(config)

    def run(
        self, *, control: Any, carla_module: Any, acceleration_mps2: float,
        steering_rad: float, ego_transform: Any, ego_speed_mps: float,
        destination_state: Sequence[float], behavior: Any,
        stop_goal_active: bool, stop_target_forward_m: object,
        min_acceleration_mps2: float, control_factory: Callable,
        reference_samples: Sequence[Mapping[str, object]],
        final_reference_accepted: bool, candidate_status: str,
        safety_manager: Any, sim_time_s: float,
        boundary_metrics: Callable, update_boundary_recovery: Callable,
        reset_boundary_recovery: Callable, acceleration_from_control: Callable,
        steering_from_control: Callable,
    ) -> ControlSafetyResult:
        control, accel, steer, signal_reason = self.supervisor.enforce_signal_stop(
            control=control, carla_module=carla_module,
            accel_mps2=float(acceleration_mps2), steer_rad=float(steering_rad),
            ego_transform=ego_transform, ego_speed_mps=float(ego_speed_mps),
            destination_state=destination_state,
            stop_goal_active=bool(stop_goal_active),
            traffic_signal_state=str(behavior.traffic_signal_state),
            min_acceleration_mps2=float(min_acceleration_mps2),
            control_factory=control_factory, config=self.config,
            stop_target_forward_m=stop_target_forward_m,
        )
        boundary_reason = ""
        boundary_snapshot = None
        turn_active = str(behavior.maneuver) in {
            "intersection_turn_left", "intersection_turn_right",
        }
        if turn_active:
            boundary_snapshot = boundary_metrics(
                ego_transform.location, record_sample=True,
                ego_yaw_rad=float(ego_transform.rotation.yaw) * 3.141592653589793 / 180.0,
                reference_samples=reference_samples,
            )
            if bool(self.config.get("boundary_recovery_enabled", False)):
                update_boundary_recovery(
                    boundary_snapshot=boundary_snapshot,
                    behavior_decision=str(behavior.maneuver),
                    sim_time_s=float(sim_time_s),
                    recovery_planned=bool(behavior.boundary_recovery_active),
                    recovery_reference_feasible=bool(
                        str(candidate_status).strip().lower() != "infeasible"
                        and final_reference_accepted
                    ),
                )
            else:
                reset_boundary_recovery()
            control, accel, steer, boundary_reason = (
                self.supervisor.enforce_turn_boundary(
                    control=control, carla_module=carla_module,
                    accel_mps2=float(accel), steer_rad=float(steer),
                    ego_speed_mps=float(ego_speed_mps),
                    behavior_decision=str(behavior.maneuver),
                    boundary_clearance_m=boundary_snapshot.get(
                        "road_boundary_clearance_m", ""
                    ),
                    min_acceleration_mps2=float(min_acceleration_mps2),
                    config=self.config, control_factory=control_factory,
                    boundary_recovery_planned=bool(
                        behavior.boundary_recovery_active
                    ),
                )
            )
        else:
            reset_boundary_recovery()
        guard_reason = ";".join(
            reason for reason in (str(signal_reason), str(boundary_reason))
            if reason
        )
        pre_accel, pre_steer = float(accel), float(steer)
        control, supervisor_reason = self.supervisor.filter_control(
            control=control, carla_module=carla_module,
            safety_manager=safety_manager,
            behavior_decision=str(behavior.maneuver),
            traffic_signal_state=str(behavior.traffic_signal_state),
            stop_goal_active=bool(stop_goal_active),
            planner_accel_mps2=pre_accel, sim_time_s=float(sim_time_s),
        )
        return ControlSafetyResult(
            control=control,
            acceleration_mps2=float(acceleration_from_control(control)),
            steering_rad=float(steering_from_control(control)),
            pre_filter_acceleration_mps2=pre_accel,
            pre_filter_steering_rad=pre_steer,
            control_guard_reason=guard_reason,
            boundary_guard_reason=str(boundary_reason),
            supervisor_reason=str(supervisor_reason),
            boundary_snapshot=boundary_snapshot,
        )
