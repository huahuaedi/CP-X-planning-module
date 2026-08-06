import math
import unittest

from opencda.planning_module.pipeline.velocity_steering_adapter import (
    CarlaVelocitySteeringAdapter,
    VelocitySteeringCommand,
)


class _Control:
    def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
        self.throttle = float(throttle)
        self.brake = float(brake)
        self.steer = float(steer)


class _Carla:
    VehicleControl = _Control


class VelocitySteeringAdapterTest(unittest.TestCase):
    def test_below_target_never_brakes(self):
        adapter = CarlaVelocitySteeringAdapter({})
        control, _ = adapter.run_step(
            command=VelocitySteeringCommand(3.0, 0.1),
            actual_speed_mps=1.0,
            sim_time_s=1.0,
            max_steering_rad=0.6,
            carla_module=_Carla,
        )
        self.assertGreater(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)

    def test_small_overspeed_coasts_without_brake(self):
        adapter = CarlaVelocitySteeringAdapter({})
        control, _ = adapter.run_step(
            command=VelocitySteeringCommand(3.0, 0.0),
            actual_speed_mps=3.2,
            sim_time_s=1.0,
            max_steering_rad=0.6,
            carla_module=_Carla,
        )
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)

    def test_stop_and_emergency_are_separate(self):
        adapter = CarlaVelocitySteeringAdapter({})
        stop, _ = adapter.run_step(
            command=VelocitySteeringCommand(0.0, 0.0, stop_goal_active=True),
            actual_speed_mps=2.0,
            sim_time_s=1.0,
            max_steering_rad=0.6,
            carla_module=_Carla,
        )
        emergency, _ = adapter.run_step(
            command=VelocitySteeringCommand(0.0, 0.0, emergency_stop=True),
            actual_speed_mps=2.0,
            sim_time_s=1.05,
            max_steering_rad=0.6,
            carla_module=_Carla,
        )
        self.assertGreater(stop.brake, 0.0)
        self.assertLessEqual(stop.brake, 0.35)
        self.assertEqual(emergency.brake, 1.0)

    def test_stop_hold_smoothly_returns_steering_to_zero(self):
        adapter = CarlaVelocitySteeringAdapter({
            "velocity_adapter_stop_steering_decay_rate_deg_s": 12.0,
        })
        moving, _ = adapter.run_step(
            command=VelocitySteeringCommand(3.0, 0.20),
            actual_speed_mps=3.0,
            sim_time_s=1.0,
            max_steering_rad=0.6,
            carla_module=_Carla,
        )
        stopped, _ = adapter.run_step(
            command=VelocitySteeringCommand(0.0, 0.0, stop_goal_active=True),
            actual_speed_mps=0.0,
            sim_time_s=1.05,
            max_steering_rad=0.6,
            carla_module=_Carla,
        )

        maximum_normalized_delta = math.radians(12.0) * 0.05 / 0.6
        self.assertGreater(stopped.steer, 0.0)
        self.assertLess(stopped.steer, moving.steer)
        self.assertLessEqual(
            abs(stopped.steer - moving.steer),
            maximum_normalized_delta + 1.0e-9,
        )

    def test_emergency_brake_is_immediate_but_steering_is_rate_limited(self):
        adapter = CarlaVelocitySteeringAdapter({
            "velocity_adapter_emergency_steering_decay_rate_deg_s": 25.0,
        })
        moving, _ = adapter.run_step(
            command=VelocitySteeringCommand(3.0, -0.20),
            actual_speed_mps=3.0,
            sim_time_s=1.0,
            max_steering_rad=0.6,
            carla_module=_Carla,
        )
        emergency, reason = adapter.run_step(
            command=VelocitySteeringCommand(0.0, 0.0, emergency_stop=True),
            actual_speed_mps=2.0,
            sim_time_s=1.05,
            max_steering_rad=0.6,
            carla_module=_Carla,
        )

        self.assertEqual(reason, "velocity_adapter_emergency_stop")
        self.assertEqual(emergency.throttle, 0.0)
        self.assertEqual(emergency.brake, 1.0)
        self.assertLess(emergency.steer, 0.0)
        self.assertGreater(emergency.steer, moving.steer)


if __name__ == "__main__":
    unittest.main()
