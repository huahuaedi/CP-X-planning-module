import unittest
from types import SimpleNamespace

from opencda.planning_module.opencda_bridge.platform_ports import VehicleDynamics
from opencda.planning_module.pipeline.velocity_steering_adapter import (
    OpenCDAVelocitySteeringAdapter,
    VelocitySteeringCommand,
)


class _Control:
    def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
        self.throttle = float(throttle)
        self.brake = float(brake)
        self.steer = float(steer)


class _Carla:
    VehicleControl = _Control


class _OpenCDAPID:
    def __init__(self):
        self.current_speed = 0.0
        self.max_throttle = 0.8
        self.max_brake = 0.6
        self.max_steering = 0.7
        self.past_steering = 0.0
        self.longitudinal_targets = []
        self.lateral_call_count = 0

    def lon_run_step(self, target_speed_kmh):
        self.longitudinal_targets.append(float(target_speed_kmh))
        return max(-1.0, min(1.0, 0.1 * (target_speed_kmh - self.current_speed)))

    def lat_run_step(self, _waypoint):
        self.lateral_call_count += 1
        raise AssertionError("OpenCDA lateral PID must not own MPC steering")


class _ControlManager:
    def __init__(self):
        self.controller = _OpenCDAPID()


class VelocitySteeringAdapterTest(unittest.TestCase):
    def setUp(self):
        self.manager = _ControlManager()
        self.adapter = OpenCDAVelocitySteeringAdapter(
            self.manager, actuator_max_steer_rad=1.2
        )

    def _run(self, command, actual_speed_mps=1.0):
        return self.adapter.run_step(
            command=command,
            actual_speed_mps=actual_speed_mps,
            sim_time_s=1.0,
            make_pedal_control=_Carla.VehicleControl,
        )

    def test_target_velocity_is_sent_to_opencda_pid_in_kmh(self):
        control, reason = self._run(VelocitySteeringCommand(3.0, 0.0))
        self.assertEqual(self.manager.controller.longitudinal_targets, [10.8])
        self.assertEqual(reason, "opencda_pid_accelerate")
        self.assertGreater(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)

    def test_mpc_steering_bypasses_opencda_lateral_pid(self):
        control, _ = self._run(VelocitySteeringCommand(3.0, 0.12))
        self.assertAlmostEqual(control.steer, 0.1)
        self.assertEqual(self.manager.controller.lateral_call_count, 0)

    def test_carla_physics_separates_actuator_scale_from_planning_limit(self):
        def wheel(x_cm, maximum_degrees):
            return SimpleNamespace(
                position=SimpleNamespace(x=float(x_cm)),
                max_steer_angle=float(maximum_degrees),
            )

        physics = SimpleNamespace(wheels=(
            wheel(223.62, 70.0), wheel(223.62, 70.0),
            wheel(-63.38, 0.0), wheel(-63.38, 0.0),
        ))
        vehicle = SimpleNamespace(get_physics_control=lambda: physics)
        dynamics = VehicleDynamics.from_carla_vehicle(
            vehicle,
            fallback_wheelbase_m=2.7,
            fallback_actuator_max_steer_rad=0.6,
        )

        self.assertAlmostEqual(dynamics.wheelbase_m, 2.87)
        self.assertAlmostEqual(dynamics.actuator_max_steer_rad, 1.2217304764)
        self.assertEqual(dynamics.source, "carla_physics_control")

    def test_physical_steering_round_trip_uses_actuator_full_scale(self):
        requested_rad = 0.25
        normalized = self.adapter._normalized_mpc_steering(
            steering_rad=requested_rad,
            actuator_max_steering_rad=self.adapter.actuator_max_steer_rad,
        )
        reconstructed_rad = normalized * self.adapter.actuator_max_steer_rad
        self.assertAlmostEqual(reconstructed_rad, requested_rad)

    def test_opencda_lateral_max_does_not_clip_mpc_physical_steering(self):
        self.manager.controller.max_steering = 0.3
        control, _ = self._run(VelocitySteeringCommand(3.0, 0.32))
        self.assertAlmostEqual(control.steer, 0.32 / 1.2)
        self.assertLess(control.steer, self.manager.controller.max_steering)

    def test_tracking_overspeed_uses_opencda_pid_brake(self):
        control, reason = self._run(
            VelocitySteeringCommand(2.0, 0.0), actual_speed_mps=4.0
        )
        self.assertEqual(reason, "opencda_pid_tracking_brake")
        self.assertEqual(control.throttle, 0.0)
        self.assertGreater(control.brake, 0.0)

    def test_pid_derivative_cannot_brake_while_ego_is_below_target(self):
        self.manager.controller.lon_run_step = lambda _target_speed_kmh: -0.38
        control, reason = self.adapter.run_step(
            command=VelocitySteeringCommand(
                target_speed_mps=6.57,
                target_steering_rad=0.0,
            ),
            actual_speed_mps=6.46,
            sim_time_s=1.0,
            make_pedal_control=_Carla.VehicleControl,
        )

        self.assertEqual(reason, "opencda_pid_underspeed_coast")
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)

    def test_pid_derivative_cannot_accelerate_while_ego_is_above_target(self):
        self.manager.controller.lon_run_step = lambda _target_speed_kmh: 0.38
        control, reason = self.adapter.run_step(
            command=VelocitySteeringCommand(
                target_speed_mps=6.46,
                target_steering_rad=0.0,
            ),
            actual_speed_mps=6.57,
            sim_time_s=1.0,
            make_pedal_control=_Carla.VehicleControl,
        )

        self.assertEqual(reason, "opencda_pid_overspeed_coast")
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)

    def test_small_overspeed_inside_deadband_cannot_trigger_pid_brake(self):
        adapter = OpenCDAVelocitySteeringAdapter(
            self.manager, actuator_max_steer_rad=1.2,
            speed_deadband_mps=0.15
        )
        self.manager.controller.lon_run_step = lambda _target_speed_kmh: -1.0

        control, reason = adapter.run_step(
            command=VelocitySteeringCommand(2.2, 0.0),
            actual_speed_mps=2.31,
            sim_time_s=1.0,
            make_pedal_control=_Carla.VehicleControl,
        )

        self.assertEqual(reason, "opencda_pid_deadband_coast")
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)

    def test_stop_hold_uses_opencda_brake_limit(self):
        control, reason = self._run(
            VelocitySteeringCommand(0.0, 0.0, stop_goal_active=True),
            actual_speed_mps=0.0,
        )
        self.assertEqual(reason, "opencda_pid_stop_hold")
        self.assertEqual(control.throttle, 0.0)
        self.assertAlmostEqual(control.brake, 0.3)

    def test_emergency_stop_bypasses_pid(self):
        control, reason = self._run(
            VelocitySteeringCommand(0.0, 0.2, emergency_stop=True),
            actual_speed_mps=3.0,
        )
        self.assertEqual(reason, "opencda_pid_emergency_stop")
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 1.0)
        self.assertEqual(self.manager.controller.longitudinal_targets, [])


if __name__ == "__main__":
    unittest.main()
