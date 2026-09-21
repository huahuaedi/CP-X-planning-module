import math
import unittest

from behavior_planner.lane_safety import LaneSafetyScorer


class LaneSafetyRuntimeTests(unittest.TestCase):
    def test_empty_lane_scores_default_to_safe(self):
        scorer = LaneSafetyScorer()

        scores = scorer.compute_lane_scores(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0},
            obstacle_snapshots=[],
            lane_assignments={},
            ego_lane_id=1,
            available_lane_ids=[1, 2],
            timestamp_s=0.0,
        )

        self.assertEqual(scores, {1: 1.0, 2: 1.0})

    def test_lane_safety_score_drops_for_close_obstacle_in_ego_lane(self):
        scorer = LaneSafetyScorer()

        scores = scorer.compute_lane_scores(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0},
            obstacle_snapshots=[
                {"vehicle_id": "front_close", "x": 6.0, "y": 0.0, "v": 0.0, "psi": 0.0}
            ],
            lane_assignments={"front_close": 1},
            ego_lane_id=1,
            available_lane_ids=[1, 2],
            timestamp_s=0.0,
        )

        self.assertLess(float(scores[1]), 0.2)
        self.assertEqual(scores[2], 1.0)

    def test_ego_lane_safety_ignores_rear_obstacle(self):
        scorer = LaneSafetyScorer()

        scores = scorer.compute_lane_scores(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
            obstacle_snapshots=[
                {"vehicle_id": "ego_front_far", "x": 20.0, "y": 0.0, "v": 0.0, "psi": 0.0},
                {"vehicle_id": "ego_rear_close", "x": -1.0, "y": 0.0, "v": 20.0, "psi": 0.0},
                {"vehicle_id": "adjacent_front_far", "x": 20.0, "y": 3.5, "v": 0.0, "psi": 0.0},
                {"vehicle_id": "adjacent_rear_close", "x": -1.0, "y": 3.5, "v": 20.0, "psi": 0.0},
            ],
            lane_assignments={
                "ego_front_far": 1,
                "ego_rear_close": 1,
                "adjacent_front_far": 2,
                "adjacent_rear_close": 2,
            },
            ego_lane_id=1,
            available_lane_ids=[1, 2],
            timestamp_s=0.0,
        )

        self.assertGreater(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)
        self.assertNotIn("ego_rear_close", scorer._ttc_history)
        self.assertIn("adjacent_rear_close", scorer._ttc_history)

    def test_far_non_closing_front_obstacle_keeps_lane_score_high_but_not_exactly_one(self):
        scorer = LaneSafetyScorer(d_safe_m=5.0, ttc_safe_s=2.0)

        scores = scorer.compute_lane_scores(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0},
            obstacle_snapshots=[
                {"vehicle_id": "front_far_same_speed", "x": 25.0, "y": 0.0, "v": 5.0, "psi": 0.0}
            ],
            lane_assignments={"front_far_same_speed": 1},
            ego_lane_id=1,
            available_lane_ids=[1],
            timestamp_s=0.0,
        )

        self.assertGreater(float(scores[1]), 0.9)
        self.assertLess(float(scores[1]), 1.0)

    def test_rear_lane_uses_rear_specific_thresholds(self):
        scorer = LaneSafetyScorer(
            d_safe_m=5.0,
            rear_d_safe_m=3.0,
            ttc_safe_s=2.0,
            rear_ttc_safe_s=1.0,
        )

        scores = scorer.compute_lane_scores(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0},
            obstacle_snapshots=[
                {"vehicle_id": "rear_adjacent", "x": -4.0, "y": 3.5, "v": 5.0, "psi": 0.0}
            ],
            lane_assignments={"rear_adjacent": 2},
            ego_lane_id=1,
            available_lane_ids=[1, 2],
            timestamp_s=0.0,
        )

        self.assertEqual(float(scores[1]), 1.0)
        self.assertGreater(float(scores[2]), 0.5)

    def test_lane_safety_considers_all_front_obstacles_not_only_the_nearest_one(self):
        scorer = LaneSafetyScorer(d_safe_m=5.0, ttc_safe_s=2.0)

        scores = scorer.compute_lane_scores(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 8.0, "psi": 0.0},
            obstacle_snapshots=[
                {"vehicle_id": "front_nearest_same_speed", "x": 8.0, "y": 3.5, "v": 8.0, "psi": 0.0},
                {"vehicle_id": "front_far_slower", "x": 20.0, "y": 3.5, "v": 0.0, "psi": 0.0},
            ],
            lane_assignments={
                "front_nearest_same_speed": 2,
                "front_far_slower": 2,
            },
            ego_lane_id=1,
            available_lane_ids=[1, 2],
            timestamp_s=0.0,
        )

        self.assertLess(float(scores[2]), 0.8)

    def test_lane_safety_score_stays_finite_for_nan_obstacle_speed(self):
        scorer = LaneSafetyScorer()

        scores = scorer.compute_lane_scores(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0},
            obstacle_snapshots=[
                {"vehicle_id": "front_nan_v", "x": 6.0, "y": 0.0, "v": float("nan"), "psi": 0.0}
            ],
            lane_assignments={"front_nan_v": 1},
            ego_lane_id=1,
            available_lane_ids=[1],
            timestamp_s=0.0,
        )

        self.assertTrue(math.isfinite(float(scores[1])))
        self.assertGreaterEqual(float(scores[1]), 0.0)
        self.assertLessEqual(float(scores[1]), 1.0)


if __name__ == "__main__":
    unittest.main()
