import os
import sys
import unittest


PLANNING_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLANNING_MODULE_ROOT not in sys.path:
    sys.path.insert(0, PLANNING_MODULE_ROOT)

from behavior_planner.planner import RuleBasedBehaviorPlanner
from pipeline import build_prediction_frame, evaluate_behavior_candidates


class PlanningPipelineTests(unittest.TestCase):
    def test_prediction_frame_uses_cp_trajectory_for_lane_risk(self):
        frame = build_prediction_frame(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 8.0, "psi": 0.0},
            obstacle_snapshots=[
                {
                    "vehicle_id": "rear_fast",
                    "x": -12.0,
                    "y": 3.5,
                    "v": 18.0,
                    "psi": 0.0,
                    "predicted_trajectory": [
                        {"x": -12.0, "y": 3.5, "t": 0.0},
                        {"x": 6.0, "y": 3.5, "t": 1.0},
                    ],
                }
            ],
            lane_assignments={"rear_fast": 2},
            available_lane_ids=[1, 2],
            horizon_s=3.0,
            dt_s=0.2,
            min_front_gap_m=12.0,
            min_rear_gap_m=10.0,
            min_ttc_s=2.5,
        )

        self.assertIn("rear_fast", frame.obstacle_future_trajectories)
        self.assertTrue(frame.risk_for_lane(2)["risk"])
        self.assertFalse(frame.risk_for_lane(1)["risk"])

    def test_prediction_frame_falls_back_to_constant_velocity(self):
        frame = build_prediction_frame(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 4.0, "psi": 0.0},
            obstacle_snapshots=[
                {
                    "vehicle_id": "front",
                    "x": 10.0,
                    "y": 0.0,
                    "v": 1.0,
                    "psi": 0.0,
                }
            ],
            lane_assignments={"front": 1},
            available_lane_ids=[1],
            horizon_s=1.0,
            dt_s=0.5,
            min_front_gap_m=12.0,
            min_rear_gap_m=8.0,
            min_ttc_s=2.5,
        )

        trajectory = frame.obstacle_future_trajectories["front"]
        self.assertGreaterEqual(len(trajectory), 2)
        self.assertAlmostEqual(trajectory[0]["x"], 10.5)
        self.assertTrue(frame.risk_for_lane(1)["risk"])

    def test_candidate_evaluation_prefers_safe_route_lane(self):
        frame = evaluate_behavior_candidates(
            lane_safety_scores={1: 0.3, 2: 0.95},
            lane_prediction_risks={1: {"risk": False}, 2: {"risk": False}},
            ego_lane_id=1,
            selected_lane_id=1,
            available_lane_ids=[1, 2],
            route_optimal_lane_id=2,
        )

        self.assertEqual(frame.selected.target_lane_id, 2)
        self.assertEqual(frame.selected.decision, "lane_change_left")
        self.assertTrue(frame.selected.feasible)

    def test_candidate_evaluation_blocks_prediction_risk_lane(self):
        frame = evaluate_behavior_candidates(
            lane_safety_scores={1: 0.5, 2: 1.0},
            lane_prediction_risks={
                1: {"risk": False},
                2: {"risk": True, "reason": "ttc", "risky_obstacle_id": "rear_fast"},
            },
            ego_lane_id=1,
            selected_lane_id=1,
            available_lane_ids=[1, 2],
            route_optimal_lane_id=2,
        )

        self.assertEqual(frame.selected.target_lane_id, 1)
        self.assertEqual(frame.selected.decision, "lane_follow")
        self.assertTrue(frame.selected.feasible)
        risky_candidate = [candidate for candidate in frame.candidates if candidate.target_lane_id == 2][0]
        self.assertFalse(risky_candidate.feasible)

    def test_candidate_preferred_lane_feeds_behavior_fsm(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        first = planner.update(
            lane_safety_scores={1: 0.8, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": False}},
            preferred_target_lane_id=2,
        )
        self.assertEqual(first["decision"], "lane_follow")
        self.assertEqual(first["lc_state"], "PREPARE_LANE_CHANGE_LEFT")

        second = planner.update(
            lane_safety_scores={1: 0.8, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": False}},
            preferred_target_lane_id=2,
        )
        self.assertEqual(second["decision"], "lane_change_left")
        self.assertEqual(second["target_lane_id"], 2)
        self.assertEqual(second["lc_state"], "EXECUTE_LANE_CHANGE_LEFT")

    def test_candidate_preferred_lane_still_blocked_by_prediction_risk(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 0.8, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": True, "reason": "ttc"}},
            preferred_target_lane_id=2,
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(result["target_lane_id"], 1)
        self.assertTrue(result["traffic_light_debug"]["lane_change_blocked_by_prediction"])


if __name__ == "__main__":
    unittest.main()
