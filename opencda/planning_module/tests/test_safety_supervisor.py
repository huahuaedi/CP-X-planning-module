import unittest
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "safety_supervisor_under_test",
    ROOT / "pipeline" / "safety_supervisor.py",
)
safety_supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = safety_supervisor
SPEC.loader.exec_module(safety_supervisor)
SafetySupervisor = safety_supervisor.SafetySupervisor


class _Control:
    def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
        self.throttle = float(throttle)
        self.brake = float(brake)
        self.steer = float(steer)


class _Carla:
    VehicleControl = _Control


class _SafetyManager:
    def __init__(self, status):
        self.status_queue = [(0.0, dict(status))]


class SafetySupervisorTest(unittest.TestCase):
    def test_collision_always_stops(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.5),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"collision": True, "stuck": True}),
            behavior_decision="lane_follow",
            traffic_signal_state="green",
            stop_goal_active=False,
            planner_accel_mps2=1.0,
        )
        self.assertEqual(reason, "safety_supervisor_emergency_stop:collision")
        self.assertEqual(control.brake, 1.0)

    def test_stuck_holds_during_stop(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.5),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="stop_at_intersection",
            traffic_signal_state="red",
            stop_goal_active=True,
            planner_accel_mps2=1.0,
        )
        self.assertEqual(reason, "safety_supervisor_emergency_stop:stuck")
        self.assertEqual(control.brake, 1.0)

    def test_stuck_releases_on_green_lane_follow(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.5),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="lane_follow",
            traffic_signal_state="green",
            stop_goal_active=False,
            planner_accel_mps2=1.0,
        )
        self.assertEqual(reason, "safety_supervisor_release:stuck")
        self.assertEqual(control.throttle, 0.5)
        self.assertEqual(control.brake, 0.0)

    def test_stuck_releases_on_unknown_lane_follow(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.5),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="lane_follow",
            traffic_signal_state="unknown",
            stop_goal_active=False,
            planner_accel_mps2=1.0,
        )
        self.assertEqual(reason, "safety_supervisor_release:stuck")
        self.assertEqual(control.throttle, 0.5)
        self.assertEqual(control.brake, 0.0)

    def test_stuck_holds_on_red_lane_follow(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.5),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="lane_follow",
            traffic_signal_state="red",
            stop_goal_active=False,
            planner_accel_mps2=1.0,
        )
        self.assertEqual(reason, "safety_supervisor_emergency_stop:stuck")
        self.assertEqual(control.brake, 1.0)

    def test_stuck_releases_on_unknown_intersection_turn(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.35),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="intersection_turn_left",
            traffic_signal_state="unknown",
            stop_goal_active=False,
            planner_accel_mps2=0.8,
        )
        self.assertEqual(reason, "safety_supervisor_release:stuck")
        self.assertEqual(control.throttle, 0.35)
        self.assertEqual(control.brake, 0.0)

    def test_stale_ran_light_releases_after_lane_follow_resume(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.4),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"ran_light": True}),
            behavior_decision="lane_follow",
            traffic_signal_state="unknown",
            stop_goal_active=False,
            planner_accel_mps2=1.0,
        )
        self.assertEqual(reason, "safety_supervisor_release:ran_light")
        self.assertEqual(control.throttle, 0.4)
        self.assertEqual(control.brake, 0.0)

    def test_ran_light_still_stops_during_active_red_stop(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.4),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"ran_light": True}),
            behavior_decision="stop_at_intersection",
            traffic_signal_state="red",
            stop_goal_active=True,
            planner_accel_mps2=0.5,
        )
        self.assertEqual(reason, "safety_supervisor_emergency_stop:ran_light")
        self.assertEqual(control.brake, 1.0)

    def test_rate_limit_does_not_mix_throttle_and_brake(self):
        supervisor = SafetySupervisor(max_throttle_delta=0.45)
        supervisor._last_control = _Control(throttle=0.5)

        control, reason = supervisor.filter_control(
            control=_Control(brake=0.2),
            carla_module=_Carla,
        )

        self.assertEqual(reason, "safety_supervisor_rate_limit")
        self.assertEqual(control.throttle, 0.0)
        self.assertAlmostEqual(control.brake, 0.2)


if __name__ == "__main__":
    unittest.main()
