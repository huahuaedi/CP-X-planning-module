"""OpenCDA control boundary for planner velocity and steering commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VelocitySteeringCommand:
    """Platform-independent command produced by the planning stack."""

    target_speed_mps: float
    target_steering_rad: float
    emergency_stop: bool = False
    stop_goal_active: bool = False


class OpenCDAVelocitySteeringAdapter:
    """Execute planner commands with OpenCDA's longitudinal PID.

    Planning owns target velocity and physical steering angle. OpenCDA owns
    the conversion from target velocity to throttle/brake. Its lateral
    waypoint PID is intentionally not called because that would overwrite the
    MPC steering command.
    """

    def __init__(self, control_manager: Any):
        if control_manager is None:
            raise ValueError("OpenCDA ControlManager is required")
        controller = getattr(control_manager, "controller", control_manager)
        if not callable(getattr(controller, "lon_run_step", None)):
            raise TypeError("OpenCDA controller must provide lon_run_step()")
        self.control_manager = control_manager
        self.controller = controller

    def run_step(
        self,
        *,
        command: VelocitySteeringCommand,
        actual_speed_mps: float,
        sim_time_s: float,
        max_steering_rad: float,
        carla_module: Any,
    ):
        del sim_time_s  # OpenCDA PID owns its configured control timestep.

        target_speed_mps = (
            0.0
            if bool(command.stop_goal_active)
            else max(0.0, float(command.target_speed_mps))
        )
        actual_speed_mps = max(0.0, float(actual_speed_mps))
        target_speed_kmh = 3.6 * target_speed_mps

        if bool(command.emergency_stop):
            throttle = 0.0
            brake = 1.0
            desired_steer = 0.0
            reason = "opencda_pid_emergency_stop"
        else:
            # VehicleManager normally updates this measurement before planning.
            # Assigning it here also makes the boundary deterministic for a
            # future external platform adapter.
            self.controller.current_speed = 3.6 * actual_speed_mps
            pid_output = float(self.controller.lon_run_step(target_speed_kmh))
            if target_speed_mps <= 0.0 and actual_speed_mps <= 0.05:
                throttle = 0.0
                brake = min(0.3, float(self.controller.max_brake))
                reason = "opencda_pid_stop_hold"
            elif pid_output >= 0.0 and target_speed_mps > 0.0:
                throttle = min(pid_output, float(self.controller.max_throttle))
                brake = 0.0
                reason = "opencda_pid_accelerate"
            else:
                throttle = 0.0
                brake = min(abs(pid_output), float(self.controller.max_brake))
                reason = (
                    "opencda_pid_stop"
                    if bool(command.stop_goal_active) or target_speed_mps <= 0.0
                    else "opencda_pid_tracking_brake"
                )
            desired_steer = self._normalized_mpc_steering(
                steering_rad=float(command.target_steering_rad),
                max_steering_rad=float(max_steering_rad),
            )

        # MPC is the sole lateral controller and already enforces physical
        # steering-angle and steering-rate constraints. OpenCDA's
        # ``max_steering``/``past_steering`` belong to its waypoint lateral
        # PID; applying them here would silently impose a second controller
        # limit (0.3 normalized == only 0.18 rad for a 0.6 rad vehicle).
        steer = max(-1.0, min(1.0, float(desired_steer)))

        control = carla_module.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer),
        )
        if hasattr(control, "hand_brake"):
            control.hand_brake = False
        if hasattr(control, "manual_gear_shift"):
            control.manual_gear_shift = False
        return control, str(reason)

    @staticmethod
    def _normalized_mpc_steering(
        *, steering_rad: float, max_steering_rad: float
    ) -> float:
        steering_scale = max(1.0e-6, abs(float(max_steering_rad)))
        return max(-1.0, min(1.0, float(steering_rad) / steering_scale))


# Import compatibility only. The implementation is OpenCDA PID-backed; no
# custom velocity PI controller remains.
CarlaVelocitySteeringAdapter = OpenCDAVelocitySteeringAdapter
