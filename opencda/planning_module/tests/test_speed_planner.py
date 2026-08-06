import importlib.util
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "speed_planner_under_test",
    ROOT / "pipeline" / "speed_planner.py",
)
speed_planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = speed_planner
SPEC.loader.exec_module(speed_planner)
build_speed_plan = speed_planner.build_speed_plan
enforce_speed_ceiling = speed_planner.enforce_speed_ceiling


class SpeedPlannerTest(unittest.TestCase):
    @staticmethod
    def _legacy_result(
        *, scenario_cap, scenario_stop, decision, requested, ego_speed, front_gap
    ):
        """Frozen pre-instrumentation behavior used as a regression oracle."""
        cap = requested if scenario_cap is None else min(requested, max(0.0, scenario_cap))
        stop_goal = scenario_stop or decision in {
            "stop_at_intersection", "stop_sign", "emergency_brake"
        }
        if decision in {"intersection_turn_left", "intersection_turn_right"}:
            cap = min(cap, 2.2)
        following_active = False
        if front_gap is not None and math.isfinite(front_gap) and not stop_goal:
            desired_gap = 5.0 + 1.5 * max(0.0, ego_speed)
            free_gap = desired_gap + 6.0
            if front_gap <= 3.0:
                cap = 0.0
                stop_goal = True
            elif front_gap < free_gap:
                ratio = (front_gap - 3.0) / max(1.0e-6, free_gap - 3.0)
                smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
                follow_cap = 0.35 + smooth_ratio * max(0.0, cap - 0.35)
                cap = min(cap, follow_cap)
                following_active = True
        if stop_goal:
            cap = 0.0
        if not math.isfinite(cap):
            cap = 0.0
        return cap, stop_goal, following_active

    @staticmethod
    def _plan(front_gap_m):
        return build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=3.0,
            ego_speed_mps=2.0,
            config={},
            front_gap_m=front_gap_m,
        )

    def test_clear_road_keeps_requested_speed(self):
        plan = self._plan(30.0)
        self.assertAlmostEqual(plan.target_speed_mps, 3.0)
        self.assertFalse(plan.stop_goal_active)
        self.assertEqual(plan.limiting_owner, "behavior_request")

    def test_following_gap_reduces_speed_without_stop_latch(self):
        plan = self._plan(7.9)
        self.assertGreater(plan.target_speed_mps, 0.0)
        self.assertLess(plan.target_speed_mps, 3.0)
        self.assertFalse(plan.stop_goal_active)
        self.assertIn("speed_plan_continuous_following", plan.reason)
        self.assertEqual(plan.limiting_owner, "following_cap")
        self.assertIn("following_cap", plan.active_constraints)

    def test_emergency_gap_requests_stop(self):
        plan = self._plan(2.5)
        self.assertEqual(plan.target_speed_mps, 0.0)
        self.assertTrue(plan.stop_goal_active)
        self.assertIn("speed_plan_obstacle_emergency_stop", plan.reason)
        self.assertEqual(plan.limiting_owner, "emergency_stop")

    def test_scenario_cap_records_longitudinal_authority(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=1.5,
                stop_goal_active=False,
                reason="traffic_light_approach",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=3.0,
            ego_speed_mps=2.0,
            config={},
            front_gap_m=None,
        )
        self.assertEqual(plan.target_speed_mps, 1.5)
        self.assertEqual(plan.limiting_owner, "scenario_cap")
        self.assertEqual(plan.as_debug_fields()["speed_owner_scenario_cap_mps"], 1.5)

    def test_turn_cap_records_longitudinal_authority(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="intersection_turn_right",
            requested_speed_mps=4.0,
            ego_speed_mps=2.0,
            config={"full_intersection_turn_speed_cap_mps": 2.2},
            front_gap_m=None,
        )
        self.assertEqual(plan.target_speed_mps, 2.2)
        self.assertEqual(plan.limiting_owner, "turn_cap")
        self.assertIn("turn_cap", plan.active_constraints)

    def test_instrumentation_is_behaviorally_equivalent_to_legacy_speed_logic(self):
        decisions = [
            "lane_follow",
            "intersection_turn_left",
            "intersection_turn_right",
            "stop_at_intersection",
            "emergency_brake",
        ]
        for decision in decisions:
            for scenario_cap in (None, 1.5, 4.0):
                for scenario_stop in (False, True):
                    for front_gap in (None, 2.5, 7.9, 30.0):
                        expected = self._legacy_result(
                            scenario_cap=scenario_cap,
                            scenario_stop=scenario_stop,
                            decision=decision,
                            requested=3.0,
                            ego_speed=2.0,
                            front_gap=front_gap,
                        )
                        plan = build_speed_plan(
                            scenario_decision=SimpleNamespace(
                                speed_cap_mps=scenario_cap,
                                stop_goal_active=scenario_stop,
                                reason="",
                            ),
                            behavior_decision=decision,
                            requested_speed_mps=3.0,
                            ego_speed_mps=2.0,
                            config={},
                            front_gap_m=front_gap,
                        )
                        actual = (
                            plan.target_speed_mps,
                            plan.stop_goal_active,
                            plan.continuous_following_active,
                        )
                        self.assertEqual(actual, expected)

    def test_speed_ceiling_prevents_candidate_from_raising_speed(self):
        result = enforce_speed_ceiling(
            proposed_target_mps=3.0,
            ceiling_mps=2.5,
            destination_state=[1.0, 2.0, 3.0, 0.0, 1],
            reference_samples=[
                {"x_ref_m": 1.0, "y_ref_m": 2.0, "v_ref_mps": 3.0},
                {"x_ref_m": 2.0, "y_ref_m": 2.0, "speed_ref_mps": 2.8},
            ],
        )
        self.assertEqual(result.target_speed_mps, 2.5)
        self.assertEqual(result.destination_state[2], 2.5)
        self.assertEqual(result.reference_samples[0]["v_ref_mps"], 2.5)
        self.assertEqual(result.reference_samples[1]["speed_ref_mps"], 2.5)
        self.assertTrue(result.applied)

    def test_speed_ceiling_preserves_candidate_slowdown_and_geometry(self):
        result = enforce_speed_ceiling(
            proposed_target_mps=1.5,
            ceiling_mps=3.0,
            destination_state=[4.0, 5.0, 1.5, 0.2, 1],
            reference_samples=[
                {"x_ref_m": 4.0, "y_ref_m": 5.0, "v_ref_mps": 1.5},
            ],
        )
        self.assertEqual(result.target_speed_mps, 1.5)
        self.assertEqual(result.destination_state[:2], [4.0, 5.0])
        self.assertEqual(result.reference_samples[0]["x_ref_m"], 4.0)
        self.assertFalse(result.applied)

    def test_zero_speed_ceiling_remains_zero_through_reference(self):
        result = enforce_speed_ceiling(
            proposed_target_mps=2.0,
            ceiling_mps=0.0,
            destination_state=[1.0, 2.0, 2.0],
            reference_samples=[{"v_ref_mps": 2.0, "speed_mps": 2.0}],
        )
        self.assertEqual(result.target_speed_mps, 0.0)
        self.assertEqual(result.destination_state[2], 0.0)
        self.assertEqual(result.reference_samples[0]["v_ref_mps"], 0.0)
        self.assertEqual(result.reference_samples[0]["speed_mps"], 0.0)


if __name__ == "__main__":
    unittest.main()
