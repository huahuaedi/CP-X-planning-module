import math
import unittest

from opencda.planning_module.pipeline.prediction import (
    build_prediction_frame,
    mpc_stage_trajectory,
    obstacle_track_id,
)


class PredictionFrameTest(unittest.TestCase):
    def test_track_id_generates_future_trajectory(self):
        frame = build_prediction_frame(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
            obstacle_snapshots=[
                {"track_id": "veh-1", "x": 5.0, "y": 0.0, "v": 1.0, "psi": 0.0}
            ],
            lane_assignments={"veh-1": 1},
            available_lane_ids=[1],
            horizon_s=1.0,
            dt_s=0.5,
            min_front_gap_m=2.0,
            min_rear_gap_m=2.0,
            min_ttc_s=1.0,
        )
        self.assertIn("veh-1", frame.obstacle_future_trajectories)
        self.assertGreater(len(frame.obstacle_future_trajectories["veh-1"]), 0)
        self.assertEqual(frame.predicted_objects["veh-1"].primary.probability, 1.0)

    def test_v2x_multimodal_probabilities_are_normalized_and_primary_is_compatible(self):
        frame = build_prediction_frame(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
            obstacle_snapshots=[{
                "track_id": "cav-2", "x": 5.0, "y": 0.0, "v": 3.0,
                "prediction_source": "v2x_plan", "plan_revision": "plan-4",
                "trajectory_hypotheses": [
                    {"maneuver": "lane_keep", "probability": 3.0,
                     "points": [{"x": 6.0, "y": 0.0, "t": 0.2, "v": 3.0}]},
                    {"maneuver": "yield", "probability": 1.0,
                     "points": [{"x": 5.5, "y": 0.0, "t": 0.2, "v": 2.0}]},
                ],
            }],
            lane_assignments={"cav-2": 1}, available_lane_ids=[1],
            horizon_s=1.0, dt_s=0.2, min_front_gap_m=2.0,
            min_rear_gap_m=2.0, min_ttc_s=1.0, timestamp_s=10.0,
        )
        predicted = frame.predicted_objects["cav-2"]
        self.assertEqual(predicted.plan_revision, "plan-4")
        self.assertAlmostEqual(sum(x.probability for x in predicted.hypotheses), 1.0)
        self.assertEqual(predicted.primary.maneuver, "lane_keep")
        self.assertEqual(frame.obstacle_future_trajectories["cav-2"][0]["x"], 6.0)

    def test_multimodal_conflict_probability_is_exposed_to_behavior(self):
        frame = build_prediction_frame(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0},
            obstacle_snapshots=[{
                "track_id": "cav-2", "prediction_source": "v2x_plan",
                "trajectory_hypotheses": [
                    {"maneuver": "yield", "probability": 0.8,
                     "points": [{"x": 30.0, "y": 0.0, "t": 1.0, "v": 2.0}]},
                    {"maneuver": "merge", "probability": 0.2,
                     "points": [{"x": 3.0, "y": 0.0, "t": 1.0, "v": 2.0}]},
                ],
            }],
            lane_assignments={"cav-2": 1}, available_lane_ids=[1],
            horizon_s=1.0, dt_s=1.0, min_front_gap_m=5.0,
            min_rear_gap_m=5.0, min_ttc_s=2.0,
        )
        risk = frame.lane_prediction_risks[1]
        self.assertAlmostEqual(risk["collision_probability"], 0.2)
        self.assertTrue(risk["risk"])
        self.assertEqual(risk["hypothesis_count"], 2)


class ObstacleTrackIdTest(unittest.TestCase):
    def test_matches_prediction_frame_keys_for_the_same_snapshot(self):
        snapshot = {"track_id": "veh-1", "x": 5.0, "y": 0.0, "v": 1.0, "psi": 0.0}
        frame = build_prediction_frame(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
            obstacle_snapshots=[snapshot],
            lane_assignments={"veh-1": 1},
            available_lane_ids=[1],
            horizon_s=1.0,
            dt_s=0.5,
            min_front_gap_m=2.0,
            min_rear_gap_m=2.0,
            min_ttc_s=1.0,
        )

        self.assertIn(obstacle_track_id(snapshot), frame.obstacle_future_trajectories)

    def test_falls_back_to_rounded_position_without_an_id_field(self):
        snapshot = {"x": 5.04, "y": -1.02}
        self.assertEqual(obstacle_track_id(snapshot), "xy:5.0:-1.0")


class MpcStageTrajectoryTest(unittest.TestCase):
    def test_converts_xyt_points_into_one_xyvpsi_entry_per_stage(self):
        points = [
            {"x": 1.0, "y": 0.0, "t": 0.5, "v": 2.0},
            {"x": 2.0, "y": 0.0, "t": 1.0, "v": 2.0},
        ]

        stages = mpc_stage_trajectory(
            points,
            fallback_heading_rad=0.0,
            horizon_steps=2,
            dt_s=0.5,
        )

        self.assertEqual(stages, [[1.0, 0.0, 2.0, 0.0], [2.0, 0.0, 2.0, 0.0]])

    def test_extrapolates_straight_line_past_a_shorter_than_horizon_trajectory(self):
        points = [{"x": 1.0, "y": 0.0, "t": 0.5, "v": 2.0}]

        stages = mpc_stage_trajectory(
            points,
            fallback_heading_rad=0.0,
            horizon_steps=3,
            dt_s=0.5,
        )

        self.assertEqual(len(stages), 3)
        self.assertEqual(stages[0], [1.0, 0.0, 2.0, 0.0])
        # Extrapolated stages keep moving at the last known speed/heading.
        self.assertAlmostEqual(stages[1][0], 2.0)
        self.assertAlmostEqual(stages[2][0], 3.0)
        for stage in stages[1:]:
            self.assertAlmostEqual(stage[2], 2.0)
            self.assertAlmostEqual(stage[3], 0.0)

    def test_empty_points_with_no_fallback_heading_still_returns_empty(self):
        self.assertEqual(
            mpc_stage_trajectory([], fallback_heading_rad=0.0, horizon_steps=3, dt_s=0.2),
            [],
        )


if __name__ == "__main__":
    unittest.main()
