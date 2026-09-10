import math
import unittest

from opencda.planning_module.pipeline.tracker import CPXObstacleTracker


class CPXObstacleTrackerTest(unittest.TestCase):
    def test_external_prediction_revision_is_not_forged_from_planner_tick(self):
        tracker = CPXObstacleTracker(max_acceleration_mps2=12.0)
        snapshot = {
            "vehicle_id": "peer", "x": 10.0, "y": 0.0,
            "v": 2.0, "psi": 0.0,
            "prediction_timestamp_s": 1.0,
            "plan_revision": "peer-plan-7",
            "predicted_trajectory": ({"x": 10.4, "y": 0.0, "t": 0.2},),
        }
        tracker.update(obstacle_snapshots=[snapshot], timestamp_s=1.0)
        first = tracker.predict(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 2.0, "psi": 0.0},
            lane_assignments={"peer": 1}, available_lane_ids=[1],
            horizon_s=2.0, dt_s=0.2, min_front_gap_m=4.0,
            min_rear_gap_m=3.0, min_ttc_s=2.0,
        )
        tracker.update(obstacle_snapshots=[snapshot], timestamp_s=1.05)
        second = tracker.predict(
            ego_snapshot={"x": 0.1, "y": 0.0, "v": 2.0, "psi": 0.0},
            lane_assignments={"peer": 1}, available_lane_ids=[1],
            horizon_s=2.0, dt_s=0.2, min_front_gap_m=4.0,
            min_rear_gap_m=3.0, min_ttc_s=2.0,
        )
        self.assertEqual(first.revision, second.revision)
        self.assertIn("peer-plan-7", first.revision)

    def test_recovers_missing_velocity_from_consecutive_positions(self):
        tracker = CPXObstacleTracker(max_acceleration_mps2=12.0)
        tracker.update(
            obstacle_snapshots=[{"id": "cut", "x": 0.0, "y": 3.5, "v": 0.0}],
            timestamp_s=1.0,
        )
        tracked = tracker.update(
            obstacle_snapshots=[{"id": "cut", "x": 0.4, "y": 3.45, "v": 0.0}],
            timestamp_s=1.05,
        )[0]
        self.assertAlmostEqual(float(tracked["v"]), math.hypot(0.4, 0.05) / 0.05)
        self.assertLess(float(tracked["psi"]), 0.0)
        self.assertEqual(tracked["kinematics_source"], "position_finite_difference")

    def test_rejected_observation_holds_previous_track_for_prediction(self):
        tracker = CPXObstacleTracker(
            max_stale_s=0.5,
            max_speed_mps=45.0,
            max_acceleration_mps2=2.0,
            max_position_jump_m=12.0,
        )
        tracker.update(
            obstacle_snapshots=[
                {"vehicle_id": "slow", "x": 10.0, "y": 0.0, "v": 0.0, "psi": 0.0}
            ],
            timestamp_s=1.0,
        )

        tracked = tracker.update(
            obstacle_snapshots=[
                {"vehicle_id": "slow", "x": 10.1, "y": 0.0, "v": 10.0, "psi": 0.0}
            ],
            timestamp_s=1.1,
        )

        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["track_id"], "slow")
        self.assertTrue(tracked[0]["track_stale"])
        self.assertEqual(
            tracked[0]["prediction_validity_reason"],
            "held_by_tracker_ttl",
        )
        self.assertIn(
            "tracker_rejected_held_by_ttl:1:acceleration_gate",
            tracker.diagnostics["prediction_validity_reason"],
        )
        prediction = tracker.predict(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 2.0, "psi": 0.0},
            lane_assignments={"slow": 1},
            available_lane_ids=[1, 2],
            horizon_s=2.0,
            dt_s=0.2,
            min_front_gap_m=4.0,
            min_rear_gap_m=3.0,
            min_ttc_s=2.0,
        )
        self.assertIn("slow", prediction.obstacle_future_trajectories)


if __name__ == "__main__":
    unittest.main()
