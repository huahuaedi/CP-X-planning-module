import unittest

from opencda.planning_module.pipeline.prediction import build_prediction_frame


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


if __name__ == "__main__":
    unittest.main()
