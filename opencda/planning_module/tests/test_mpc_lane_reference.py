import unittest
import numpy as np

from MPC.mpc import MPC


class MPCLaneReferenceTests(unittest.TestCase):
    def test_query_aware_lane_reference_stays_local_to_stage_window(self):
        mpc = object.__new__(MPC)
        mpc.lane_center_reference_local_window = 1

        lane_center_reference = [
            {"x_ref_m": 0.0, "y_ref_m": 0.0, "heading_rad": 0.0},
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "heading_rad": 0.0},
            {"x_ref_m": 1.0, "y_ref_m": 1.0, "heading_rad": 1.57},
            {"x_ref_m": 0.0, "y_ref_m": 1.0, "heading_rad": 3.14},
        ]

        ref = mpc._get_lane_center_stage_ref(
            lane_center_reference=lane_center_reference,
            stage_index=0,
            query_x_m=0.1,
            query_y_m=0.9,
        )

        self.assertIsNotNone(ref)
        self.assertAlmostEqual(float(ref[0]), 0.0)
        self.assertAlmostEqual(float(ref[1]), 0.0)

    def test_query_aware_lane_reference_can_move_with_stage_progress(self):
        mpc = object.__new__(MPC)
        mpc.lane_center_reference_local_window = 1

        lane_center_reference = [
            {"x_ref_m": 0.0, "y_ref_m": 0.0, "heading_rad": 0.0},
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "heading_rad": 0.0},
            {"x_ref_m": 1.0, "y_ref_m": 1.0, "heading_rad": 1.57},
            {"x_ref_m": 0.0, "y_ref_m": 1.0, "heading_rad": 3.14},
        ]

        ref = mpc._get_lane_center_stage_ref(
            lane_center_reference=lane_center_reference,
            stage_index=3,
            query_x_m=0.1,
            query_y_m=0.9,
        )

        self.assertIsNotNone(ref)
        self.assertAlmostEqual(float(ref[0]), 0.0)
        self.assertAlmostEqual(float(ref[1]), 1.0)

    def test_clear_previous_solution_seed_drops_warm_start(self):
        mpc = object.__new__(MPC)
        mpc._previous_x_solution = np.zeros((2, 4), dtype=float)
        mpc._previous_u_solution = np.zeros((1, 2), dtype=float)

        mpc.clear_previous_solution_seed()

        self.assertIsNone(mpc._previous_x_solution)
        self.assertIsNone(mpc._previous_u_solution)

    def test_tracking_reference_uses_matching_rollout_stage(self):
        rollout = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.2, 0.0, 1.5, 0.0],
            [0.5, 0.0, 2.0, 0.0],
            [0.9, 0.0, 2.5, 0.0],
        ])

        stage_one = MPC._tracking_reference_at_stage(
            x_ref_rollout=rollout,
            stage_index=1,
        )
        terminal = MPC._tracking_reference_at_stage(
            x_ref_rollout=rollout,
            stage_index=99,
        )

        np.testing.assert_allclose(stage_one, rollout[1])
        np.testing.assert_allclose(terminal, rollout[-1])


if __name__ == "__main__":
    unittest.main()
