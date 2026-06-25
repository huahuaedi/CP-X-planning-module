import math
import unittest

from behavior_planner.lane_safety import LaneSafetyScorer
from carla_scenario.runner import _lane_safety_assignment_for_obstacle, _same_lane_safety_corridor


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

    def test_lane_safety_corridor_filter_rejects_different_road(self):
        ego_context = {
            "road_id": "12:0",
            "road_numeric_id": 12,
            "direction": "positive",
            "lane_id": 1,
        }
        obstacle_context = {
            "road_id": "34:0",
            "road_numeric_id": 34,
            "direction": "positive",
            "lane_id": 1,
        }

        self.assertFalse(_same_lane_safety_corridor(ego_context, obstacle_context))

    def test_lane_safety_corridor_filter_accepts_same_road_and_direction(self):
        ego_context = {
            "road_id": "12:0",
            "road_numeric_id": 12,
            "direction": "positive",
            "lane_id": 1,
        }
        obstacle_context = {
            "road_id": "12:1",
            "road_numeric_id": 12,
            "direction": "positive",
            "lane_id": 2,
        }

        self.assertTrue(_same_lane_safety_corridor(ego_context, obstacle_context))

    def test_lane_safety_corridor_filter_accepts_aligned_intersection_connector(self):
        ego_context = {
            "road_id": "12:0",
            "road_numeric_id": 12,
            "direction": "positive",
            "lane_id": 2,
            "heading_rad": 0.0,
            "is_intersection": True,
        }
        obstacle_context = {
            "road_id": "34:0",
            "road_numeric_id": 34,
            "direction": "positive",
            "lane_id": 1,
            "heading_rad": 0.2,
            "is_intersection": True,
        }

        self.assertTrue(_same_lane_safety_corridor(ego_context, obstacle_context))

    def test_lane_safety_corridor_filter_rejects_cross_traffic_inside_intersection(self):
        ego_context = {
            "road_id": "12:0",
            "road_numeric_id": 12,
            "direction": "positive",
            "lane_id": 2,
            "heading_rad": 0.0,
            "is_intersection": True,
        }
        obstacle_context = {
            "road_id": "34:0",
            "road_numeric_id": 34,
            "direction": "positive",
            "lane_id": 1,
            "heading_rad": 1.8,
            "is_intersection": True,
        }

        self.assertFalse(_same_lane_safety_corridor(ego_context, obstacle_context))

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

    def test_lane_safety_assignment_maps_intersection_connector_to_ego_lane(self):
        assigned_lane_id = _lane_safety_assignment_for_obstacle(
            ego_lane_context={
                "road_id": "12:0",
                "road_numeric_id": 12,
                "direction": "positive",
                "lane_id": 2,
                "heading_rad": 0.0,
                "is_intersection": True,
                "lane_width_m": 3.5,
            },
            obstacle_lane_context={
                "road_id": "34:0",
                "road_numeric_id": 34,
                "direction": "positive",
                "lane_id": 1,
                "heading_rad": 0.1,
                "is_intersection": True,
                "lane_width_m": 3.5,
            },
            ego_lane_id=2,
            ego_in_junction=True,
            available_lane_ids=[1, 2],
            ego_snapshot={"x": 0.0, "y": 0.0, "psi": 0.0},
            obstacle_snapshot={"x": 0.0, "y": 0.2},
        )

        self.assertEqual(assigned_lane_id, 2)

    def test_lane_safety_assignment_maps_left_connector_obstacle_to_left_lane(self):
        assigned_lane_id = _lane_safety_assignment_for_obstacle(
            ego_lane_context={
                "road_id": "12:0",
                "road_numeric_id": 12,
                "direction": "positive",
                "lane_id": 1,
                "heading_rad": 0.0,
                "is_intersection": True,
                "lane_width_m": 3.5,
            },
            obstacle_lane_context={
                "road_id": "34:0",
                "road_numeric_id": 34,
                "direction": "positive",
                "lane_id": 1,
                "heading_rad": 0.05,
                "is_intersection": True,
                "lane_width_m": 3.5,
            },
            ego_lane_id=1,
            ego_in_junction=True,
            available_lane_ids=[1, 2],
            ego_snapshot={"x": 0.0, "y": 0.0, "psi": 0.0},
            obstacle_snapshot={"x": 0.0, "y": 3.6},
        )

        self.assertEqual(assigned_lane_id, 2)

    def test_lane_safety_assignment_prefers_relative_lane_while_exiting_junction(self):
        assigned_lane_id = _lane_safety_assignment_for_obstacle(
            ego_lane_context={
                "road_id": "50:0",
                "road_numeric_id": 50,
                "direction": "positive",
                "lane_id": 2,
                "heading_rad": 0.0,
                "is_intersection": True,
                "lane_width_m": 3.5,
            },
            obstacle_lane_context={
                "road_id": "50:0",
                "road_numeric_id": 50,
                "direction": "positive",
                "lane_id": 1,
                "heading_rad": 0.0,
                "is_intersection": False,
                "lane_width_m": 3.5,
            },
            ego_lane_id=2,
            ego_in_junction=True,
            available_lane_ids=[1, 2],
            ego_snapshot={"x": 0.0, "y": 0.0, "psi": 0.0},
            obstacle_snapshot={"x": 8.0, "y": 0.2, "psi": 0.0},
        )

        self.assertEqual(assigned_lane_id, 2)


if __name__ == "__main__":
    unittest.main()
