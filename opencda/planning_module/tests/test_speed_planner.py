import importlib.util
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


class SpeedPlannerTest(unittest.TestCase):
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

    def test_following_gap_reduces_speed_without_stop_latch(self):
        plan = self._plan(7.9)
        self.assertGreater(plan.target_speed_mps, 0.0)
        self.assertLess(plan.target_speed_mps, 3.0)
        self.assertFalse(plan.stop_goal_active)
        self.assertIn("speed_plan_continuous_following", plan.reason)

    def test_emergency_gap_requests_stop(self):
        plan = self._plan(2.5)
        self.assertEqual(plan.target_speed_mps, 0.0)
        self.assertTrue(plan.stop_goal_active)
        self.assertIn("speed_plan_obstacle_emergency_stop", plan.reason)


if __name__ == "__main__":
    unittest.main()
