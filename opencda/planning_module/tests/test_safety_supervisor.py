import unittest
import importlib.util
import pathlib
import sys
from types import SimpleNamespace


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
    @staticmethod
    def _control_factory(accel_mps2, steer_rad):
        return _Control(
            throttle=max(0.0, float(accel_mps2)),
            brake=max(0.0, -float(accel_mps2)),
            steer=float(steer_rad),
        )

    def test_red_stop_envelope_never_returns_positive_acceleration(self):
        supervisor = SafetySupervisor()
        control, accel, steer, reason = supervisor.enforce_signal_stop(
            control=_Control(throttle=0.5),
            carla_module=_Carla,
            accel_mps2=1.0,
            steer_rad=0.1,
            ego_transform=SimpleNamespace(
                location=SimpleNamespace(x=0.0, y=0.0),
                rotation=SimpleNamespace(yaw=0.0),
            ),
            ego_speed_mps=3.0,
            destination_state=[4.0, 0.0],
            stop_goal_active=True,
            traffic_signal_state="red",
            min_acceleration_mps2=-3.0,
            control_factory=self._control_factory,
            config={},
        )

        self.assertLessEqual(accel, 0.0)
        self.assertAlmostEqual(steer, 0.1)
        self.assertIn("red_yellow_stop", reason)
        self.assertEqual(control.throttle, 0.0)

    def test_green_signal_is_not_modified_by_stop_envelope(self):
        supervisor = SafetySupervisor()
        original = _Control(throttle=0.4, steer=0.1)
        control, accel, steer, reason = supervisor.enforce_signal_stop(
            control=original,
            carla_module=_Carla,
            accel_mps2=0.8,
            steer_rad=0.1,
            ego_transform=SimpleNamespace(),
            ego_speed_mps=2.0,
            destination_state=[4.0, 0.0],
            stop_goal_active=True,
            traffic_signal_state="green",
            min_acceleration_mps2=-3.0,
            control_factory=self._control_factory,
            config={},
        )

        self.assertIs(control, original)
        self.assertAlmostEqual(accel, 0.8)
        self.assertEqual(reason, "")

    def test_committed_red_stop_never_accelerates_at_low_speed(self):
        supervisor = SafetySupervisor()
        control, accel, _, reason = supervisor.enforce_signal_stop(
            control=_Control(throttle=0.5),
            carla_module=_Carla,
            accel_mps2=1.0,
            steer_rad=0.1,
            ego_transform=SimpleNamespace(
                location=SimpleNamespace(x=0.0, y=0.0),
                rotation=SimpleNamespace(yaw=0.0),
            ),
            ego_speed_mps=0.3,
            destination_state=[8.0, 0.0],
            stop_goal_active=True,
            traffic_signal_state="red",
            min_acceleration_mps2=-3.0,
            control_factory=self._control_factory,
            config={},
            stop_target_forward_m=4.8,
        )

        self.assertLessEqual(accel, 0.0)
        self.assertEqual(control.throttle, 0.0)
        self.assertIn("red_yellow_stop", reason)

    def test_red_stop_overshoot_enters_brake_hold(self):
        supervisor = SafetySupervisor()
        control, accel, steer, reason = supervisor.enforce_signal_stop(
            control=_Control(throttle=0.5),
            carla_module=_Carla,
            accel_mps2=1.0,
            steer_rad=0.1,
            ego_transform=SimpleNamespace(
                location=SimpleNamespace(x=0.0, y=0.0),
                rotation=SimpleNamespace(yaw=0.0),
            ),
            ego_speed_mps=0.3,
            destination_state=[8.0, 0.0],
            stop_goal_active=True,
            traffic_signal_state="red",
            min_acceleration_mps2=-3.0,
            control_factory=self._control_factory,
            config={},
            stop_target_forward_m=-0.1,
        )

        self.assertLess(accel, 0.0)
        self.assertEqual(control.throttle, 0.0)
        self.assertGreater(control.brake, 0.0)
        self.assertEqual(steer, 0.0)
        self.assertEqual(reason, "red_yellow_stop_overshoot_hold")

    def test_turn_boundary_warning_uses_speed_envelope_without_braking(self):
        supervisor = SafetySupervisor()
        control, accel, _, reason = supervisor.enforce_turn_boundary(
            control=_Control(throttle=0.4),
            carla_module=_Carla,
            accel_mps2=0.8,
            steer_rad=0.1,
            ego_speed_mps=0.5,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=0.05,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
        )

        self.assertGreater(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)
        self.assertGreater(accel, 0.0)
        self.assertLessEqual(accel, 0.8)
        self.assertIn("turn_road_boundary_continuous_recovery", reason)

    def test_turn_boundary_warning_does_not_lock_stationary_vehicle(self):
        supervisor = SafetySupervisor()
        control, accel, _, reason = supervisor.enforce_turn_boundary(
            control=_Control(throttle=0.4),
            carla_module=_Carla,
            accel_mps2=0.8,
            steer_rad=0.1,
            ego_speed_mps=0.0,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=0.05,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
        )

        self.assertIsInstance(control, _Control)
        self.assertAlmostEqual(accel, 0.8)
        self.assertEqual(control.brake, 0.0)
        self.assertIn("turn_road_boundary_continuous_recovery", reason)

    def test_minor_turn_boundary_breach_keeps_forward_recovery(self):
        supervisor = SafetySupervisor()
        control, accel, _, reason = supervisor.enforce_turn_boundary(
            control=_Control(throttle=0.4),
            carla_module=_Carla,
            accel_mps2=0.8,
            steer_rad=0.1,
            ego_speed_mps=0.0,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=-0.10,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
        )

        self.assertGreater(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)
        self.assertGreater(accel, 0.0)
        self.assertIn("turn_road_boundary_continuous_recovery", reason)

    def test_minor_turn_boundary_breach_never_transitions_through_hard_stop(self):
        supervisor = SafetySupervisor()
        for _ in range(8):
            control, accel, _, reason = supervisor.enforce_turn_boundary(
                control=_Control(throttle=0.4),
                carla_module=_Carla,
                accel_mps2=0.8,
                steer_rad=0.2,
                ego_speed_mps=0.0,
                behavior_decision="intersection_turn_right",
                boundary_clearance_m=-0.06,
                min_acceleration_mps2=-3.0,
                config={},
                control_factory=self._control_factory,
            )
            self.assertGreater(control.throttle, 0.0)
            self.assertEqual(control.brake, 0.0)
            self.assertGreater(accel, 0.0)
            self.assertIn("turn_road_boundary_continuous_recovery", reason)
        self.assertTrue(supervisor.turn_boundary_recovery_active)
        self.assertEqual(
            supervisor.turn_boundary_recovery_phase,
            "continuous_recovery",
        )

    def test_turn_boundary_recovery_uses_release_hysteresis(self):
        supervisor = SafetySupervisor()
        supervisor.enforce_turn_boundary(
            control=_Control(throttle=0.4),
            carla_module=_Carla,
            accel_mps2=0.8,
            steer_rad=0.2,
            ego_speed_mps=0.5,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=-0.02,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
        )
        _, _, _, active_reason = supervisor.enforce_turn_boundary(
            control=_Control(throttle=0.4),
            carla_module=_Carla,
            accel_mps2=0.8,
            steer_rad=0.2,
            ego_speed_mps=0.5,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=0.17,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
        )
        _, _, _, released_reason = supervisor.enforce_turn_boundary(
            control=_Control(throttle=0.4),
            carla_module=_Carla,
            accel_mps2=0.8,
            steer_rad=0.2,
            ego_speed_mps=0.5,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=0.21,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
        )

        self.assertIn("continuous_recovery", active_reason)
        self.assertEqual(released_reason, "")
        self.assertFalse(supervisor.turn_boundary_recovery_active)

    def test_critical_turn_boundary_breach_keeps_steered_creep(self):
        supervisor = SafetySupervisor()
        result = None
        for _ in range(8):
            result = supervisor.enforce_turn_boundary(
                control=_Control(brake=0.3),
                carla_module=_Carla,
                accel_mps2=-0.9,
                steer_rad=0.2,
                ego_speed_mps=0.0,
                behavior_decision="intersection_turn_right",
                boundary_clearance_m=-0.35,
                min_acceleration_mps2=-3.0,
                config={},
                control_factory=self._control_factory,
            )

        control, accel, _, reason = result
        self.assertGreater(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)
        self.assertGreater(accel, 0.0)
        self.assertAlmostEqual(control.steer, 0.2)
        self.assertIn("turn_road_boundary_critical_recovery", reason)
        self.assertEqual(
            supervisor.turn_boundary_recovery_phase,
            "critical_recovery",
        )

    def test_stuck_does_not_veto_active_turn_boundary_recovery(self):
        supervisor = SafetySupervisor()
        supervisor.enforce_turn_boundary(
            control=_Control(throttle=0.0, brake=0.3, steer=0.2),
            carla_module=_Carla,
            accel_mps2=-0.9,
            steer_rad=0.2,
            ego_speed_mps=0.0,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=-0.42,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
        )

        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.3, steer=0.2),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="intersection_turn_right",
            traffic_signal_state="unknown",
            stop_goal_active=False,
            planner_accel_mps2=-0.9,
        )

        self.assertEqual(reason, "safety_supervisor_release:stuck")
        self.assertGreater(control.throttle, 0.0)
        self.assertAlmostEqual(control.steer, 0.2)

    def test_offroad_diagnostic_does_not_veto_boundary_recovery(self):
        supervisor = SafetySupervisor()
        supervisor.enforce_turn_boundary(
            control=_Control(throttle=0.3, steer=0.2),
            carla_module=_Carla,
            accel_mps2=0.3,
            steer_rad=0.2,
            ego_speed_mps=0.1,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=-0.35,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
        )

        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.2, steer=0.2),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"offroad": True}),
            behavior_decision="intersection_turn_right",
            traffic_signal_state="unknown",
            stop_goal_active=False,
            planner_accel_mps2=0.2,
        )

        self.assertEqual(reason, "safety_supervisor_release:offroad")
        self.assertGreater(control.throttle, 0.0)

    def test_planned_boundary_recovery_is_not_rescaled_after_mpc(self):
        supervisor = SafetySupervisor()
        original = _Control(throttle=0.25, steer=0.2)

        control, accel, steer, reason = supervisor.enforce_turn_boundary(
            control=original,
            carla_module=_Carla,
            accel_mps2=0.75,
            steer_rad=0.2,
            ego_speed_mps=0.2,
            behavior_decision="intersection_turn_right",
            boundary_clearance_m=-0.4,
            min_acceleration_mps2=-3.0,
            config={},
            control_factory=self._control_factory,
            boundary_recovery_planned=True,
        )

        self.assertIs(control, original)
        self.assertAlmostEqual(accel, 0.75)
        self.assertAlmostEqual(steer, 0.2)
        self.assertIn("planned_recovery_monitor", reason)
        self.assertEqual(
            supervisor.turn_boundary_recovery_phase,
            "planned_recovery",
        )

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

    def test_stuck_releases_small_positive_turn_accel_after_green(self):
        supervisor = SafetySupervisor(stuck_release_min_accel_mps2=0.01)
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.013),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="intersection_turn_right",
            traffic_signal_state="green",
            stop_goal_active=False,
            planner_accel_mps2=0.04,
        )
        self.assertEqual(reason, "safety_supervisor_release:stuck")
        self.assertAlmostEqual(control.throttle, 0.013)
        self.assertEqual(control.brake, 0.0)

    def test_stuck_releases_during_route_recovery(self):
        supervisor = SafetySupervisor()
        control, reason = supervisor.filter_control(
            control=_Control(throttle=0.25),
            carla_module=_Carla,
            safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="route_recovery",
            traffic_signal_state="unknown",
            stop_goal_active=False,
            planner_accel_mps2=0.5,
        )
        self.assertEqual(reason, "safety_supervisor_release:stuck")
        self.assertEqual(control.throttle, 0.25)
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
