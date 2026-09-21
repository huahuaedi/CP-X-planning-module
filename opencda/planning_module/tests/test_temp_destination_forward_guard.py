import unittest

from behavior_planner.reference_pipeline import (
    reference_forward_trim,
    reference_with_route_fallback,
    stabilize_lane_reference_samples,
)


class TempDestinationForwardGuardTests(unittest.TestCase):
    def test_persistent_maneuver_reference_consumes_behind_prefix(self):
        samples = reference_forward_trim(
            [
                {"x_ref_m": -2.0, "y_ref_m": 0.0, "heading_rad": 0.0},
                {"x_ref_m": 0.5, "y_ref_m": 0.2, "heading_rad": 0.1},
                {"x_ref_m": 2.0, "y_ref_m": 0.5, "heading_rad": 0.2},
                {"x_ref_m": 4.0, "y_ref_m": 1.0, "heading_rad": 0.3},
            ],
            ego_state=[0.0, 0.0, 3.0, 0.0],
            min_first_forward_m=1.0,
            step_distance_m=2.0,
        )

        self.assertEqual(len(samples), 4)
        self.assertAlmostEqual(float(samples[0]["x_ref_m"]), 2.0)
        self.assertAlmostEqual(float(samples[0]["y_ref_m"]), 0.5)
        self.assertAlmostEqual(float(samples[0]["heading_rad"]), 0.2)

    def test_consumed_reference_is_not_returned_behind_ego(self):
        samples = reference_forward_trim(
            [
                {"x_ref_m": -3.0, "y_ref_m": 0.0, "heading_rad": 0.0},
                {"x_ref_m": -1.0, "y_ref_m": 0.0, "heading_rad": 0.0},
            ],
            ego_state=[0.0, 0.0, 3.0, 0.0],
            min_first_forward_m=1.0,
            step_distance_m=2.0,
        )

        self.assertEqual(samples, [])

    def test_reference_fallback_generates_forward_heading_samples_when_route_is_bad(self):
        samples, reason = reference_with_route_fallback(
            ego_state=[0.0, 0.0, 2.0, 0.0],
            current_reference=[
                {"x_ref_m": -10.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1}
            ],
            previous_reference=None,
            decision="lane_follow",
            global_route_points=[],
            horizon_steps=3,
            step_distance_m=2.0,
            target_lane_id=1,
        )

        self.assertIn("heading_fallback", reason)
        self.assertEqual(len(samples), 3)
        self.assertGreater(float(samples[0]["x_ref_m"]), 0.0)
        self.assertAlmostEqual(float(samples[0]["y_ref_m"]), 0.0)

    def test_reference_fallback_rejects_non_lane_change_lane_mismatch(self):
        samples, reason = reference_with_route_fallback(
            ego_state=[0.0, 0.0, 2.0, 0.0],
            current_reference=[
                {"x_ref_m": 2.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1}
            ],
            previous_reference=None,
            decision="stop_sign",
            global_route_points=[[0.0, 0.0], [10.0, 0.0]],
            horizon_steps=3,
            step_distance_m=2.0,
            target_lane_id=2,
            expected_lane_id=2,
            allow_route_fallback=False,
        )

        self.assertIn("first_sample_lane_mismatch", reason)
        self.assertIn("heading_fallback_no_route", reason)
        self.assertEqual(len(samples), 3)
        self.assertEqual(int(samples[0]["lane_id"]), 2)
        self.assertGreater(float(samples[0]["x_ref_m"]), 0.0)

    def test_persistent_lane_change_is_not_replaced_when_prefix_is_behind(self):
        reference = [
            {"x_ref_m": -4.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1},
            {"x_ref_m": 2.0, "y_ref_m": 1.0, "heading_rad": 0.1, "lane_id": 2},
        ]

        samples, reason = reference_with_route_fallback(
            ego_state=[0.0, 0.0, 4.0, 0.0],
            current_reference=reference,
            previous_reference=None,
            decision="lane_change_right",
            global_route_points=[],
            horizon_steps=2,
            step_distance_m=2.0,
            target_lane_id=2,
            allow_route_fallback=False,
        )

        self.assertEqual(reason, "")
        self.assertEqual(samples, reference)

    def test_reference_stabilizer_freezes_non_lane_change_jump(self):
        previous = [
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1},
            {"x_ref_m": 4.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1},
        ]
        current = [
            {"x_ref_m": 2.0, "y_ref_m": 5.0, "heading_rad": 0.0, "lane_id": 1},
            {"x_ref_m": 4.0, "y_ref_m": 5.0, "heading_rad": 0.0, "lane_id": 1},
        ]

        samples, stabilized, jump_m = stabilize_lane_reference_samples(
            current,
            previous,
            decision="lane_follow",
            max_non_lc_first_sample_jump_m=2.25,
        )

        self.assertTrue(stabilized)
        self.assertGreater(jump_m, 2.25)
        self.assertEqual(samples, previous)

    def test_reference_stabilizer_allows_lane_change_jump(self):
        previous = [
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1}
        ]
        current = [
            {"x_ref_m": 2.0, "y_ref_m": 5.0, "heading_rad": 0.0, "lane_id": 2}
        ]

        samples, stabilized, _jump_m = stabilize_lane_reference_samples(
            current,
            previous,
            decision="lane_change_left",
            max_non_lc_first_sample_jump_m=2.25,
        )

        self.assertFalse(stabilized)
        self.assertEqual(samples, current)


if __name__ == "__main__":
    unittest.main()
