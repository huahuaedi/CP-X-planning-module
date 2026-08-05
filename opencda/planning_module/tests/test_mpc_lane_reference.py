import math
import unittest
import numpy as np

from MPC.lane_keep import LaneKeepingStageReference, signed_longitudinal_progress_affine_form
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

    def test_build_route_reference_tags_monotonic_progress_matching_arc_length(self):
        mpc = object.__new__(MPC)
        mpc.horizon_steps = 3
        mpc.dt_s = 0.5
        mpc.lane_width_m = 3.5

        # Quarter-circle route of radius 10, sampled coarsely -- arc length
        # along the polyline (chord-based) is what _nearest_progress_along_route
        # /this function actually accumulate, not the true circular arc length.
        radius = 10.0
        route_points = [
            (radius * math.sin(theta), radius * (1.0 - math.cos(theta)))
            for theta in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
        ]

        stage_reference = mpc._build_route_reference(
            current_state=np.array([0.0, 0.0, 2.0, 0.0], dtype=float),
            destination_state=np.array([route_points[-1][0], route_points[-1][1], 0.0, 0.0], dtype=float),
            route_reference_points=route_points,
        )

        self.assertTrue(all("progress_m" in sample for sample in stage_reference))
        progress_values = [float(sample["progress_m"]) for sample in stage_reference]
        for earlier, later in zip(progress_values, progress_values[1:]):
            self.assertLessEqual(earlier, later + 1e-9)
        # First stage's progress should match _nearest_progress_along_route's
        # own computation from the same start state (self-consistency check).
        expected_start_progress_m, _ = mpc._nearest_progress_along_route(
            route_points=route_points, xy=[0.0, 0.0],
        )
        self.assertAlmostEqual(progress_values[0], expected_start_progress_m, places=3)

    def test_progress_based_lookup_interpolates_between_bracketing_samples(self):
        mpc = object.__new__(MPC)
        lane_center_reference = [
            {"x_ref_m": 0.0, "y_ref_m": 0.0, "heading_rad": 0.0, "progress_m": 0.0},
            {"x_ref_m": 10.0, "y_ref_m": 0.0, "heading_rad": 0.0, "progress_m": 10.0},
        ]

        sample = mpc._get_lane_center_stage_sample_by_progress(
            lane_center_reference=lane_center_reference,
            query_progress_m=2.5,
        )

        self.assertIsNotNone(sample)
        self.assertAlmostEqual(float(sample["x_ref_m"]), 2.5)
        self.assertAlmostEqual(float(sample["y_ref_m"]), 0.0)

    def test_progress_based_lookup_falls_back_to_none_when_progress_missing(self):
        mpc = object.__new__(MPC)
        lane_center_reference = [
            {"x_ref_m": 0.0, "y_ref_m": 0.0, "heading_rad": 0.0, "progress_m": 0.0},
            {"x_ref_m": 10.0, "y_ref_m": 0.0, "heading_rad": 0.0},  # no progress_m tag
        ]

        sample = mpc._get_lane_center_stage_sample_by_progress(
            lane_center_reference=lane_center_reference,
            query_progress_m=2.5,
        )

        self.assertIsNone(sample)

    def test_progress_and_index_lookup_agree_on_a_straight_reference(self):
        mpc = object.__new__(MPC)
        mpc.lane_center_reference_local_window = 0
        # A straight reference built at exactly the nominal step distance --
        # array position and along-path distance coincide here by
        # construction, so both lookup methods must agree.
        lane_center_reference = [
            {"x_ref_m": float(i) * 2.0, "y_ref_m": 0.0, "heading_rad": 0.0, "progress_m": float(i) * 2.0}
            for i in range(6)
        ]

        index_ref = mpc._get_lane_center_stage_ref(
            lane_center_reference=lane_center_reference, stage_index=3,
        )
        progress_ref = mpc._get_lane_center_stage_ref_by_progress(
            lane_center_reference=lane_center_reference, query_progress_m=6.0,
        )

        self.assertIsNotNone(index_ref)
        self.assertIsNotNone(progress_ref)
        self.assertAlmostEqual(index_ref[0], progress_ref[0])
        self.assertAlmostEqual(index_ref[1], progress_ref[1])

    def test_progress_lookup_differs_from_index_lookup_on_a_tight_curve(self):
        mpc = object.__new__(MPC)
        mpc.lane_center_reference_local_window = 0
        # A 90-degree turn where the vehicle traveled slower than the
        # reference's own nominal step assumption -- true traveled progress
        # (3.0 m) sits well before array index 3's position, so index-based
        # and progress-based lookup must diverge, with the progress-based
        # one landing closer to the true along-path point at progress=3.0.
        lane_center_reference = [
            {"x_ref_m": 0.0, "y_ref_m": 0.0, "heading_rad": 0.0, "progress_m": 0.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "heading_rad": 0.0, "progress_m": 2.0},
            {"x_ref_m": 4.0, "y_ref_m": 0.0, "heading_rad": 1.5708, "progress_m": 4.0},
            {"x_ref_m": 4.0, "y_ref_m": 2.0, "heading_rad": 1.5708, "progress_m": 6.0},
        ]

        index_ref = mpc._get_lane_center_stage_ref(
            lane_center_reference=lane_center_reference, stage_index=3,
        )
        progress_ref = mpc._get_lane_center_stage_ref_by_progress(
            lane_center_reference=lane_center_reference, query_progress_m=3.0,
        )

        self.assertIsNotNone(index_ref)
        self.assertIsNotNone(progress_ref)
        # Index lookup jumps straight to the last sample (4.0, 2.0).
        self.assertAlmostEqual(index_ref[0], 4.0)
        self.assertAlmostEqual(index_ref[1], 2.0)
        # Progress lookup correctly lands halfway along the 2.0->4.0 segment
        # (progress 2.0 to 4.0), i.e. at (3.0, 0.0) -- materially different
        # from, and geometrically closer to where 3.0m of travel actually
        # is, than the index-based result.
        self.assertAlmostEqual(progress_ref[0], 3.0)
        self.assertAlmostEqual(progress_ref[1], 0.0)
        self.assertNotAlmostEqual(index_ref[0], progress_ref[0])


class SignedLongitudinalProgressAffineFormTests(unittest.TestCase):
    def test_zero_at_the_reference_point_itself(self):
        reference = LaneKeepingStageReference(
            x_center_m=10.0, y_center_m=5.0, heading_rad=math.pi / 2.0, lane_width_m=4.0,
        )
        affine = signed_longitudinal_progress_affine_form(reference)

        self.assertAlmostEqual(affine.evaluate(x_m=10.0, y_m=5.0), 0.0)

    def test_positive_ahead_along_heading_direction(self):
        # heading = 0 (facing +x): a point 3m further along +x is "ahead".
        reference = LaneKeepingStageReference(
            x_center_m=0.0, y_center_m=0.0, heading_rad=0.0, lane_width_m=4.0,
        )
        affine = signed_longitudinal_progress_affine_form(reference)

        self.assertAlmostEqual(affine.evaluate(x_m=3.0, y_m=0.0), 3.0)
        self.assertAlmostEqual(affine.evaluate(x_m=-2.0, y_m=0.0), -2.0)
        # Purely lateral offset (y) contributes zero longitudinal error at heading=0.
        self.assertAlmostEqual(affine.evaluate(x_m=0.0, y_m=5.0), 0.0)


if __name__ == "__main__":
    unittest.main()
