import math
import unittest

from pipeline.stage_contracts import evaluate_lane_change_completion


class LaneChangeCompletionTests(unittest.TestCase):
    @staticmethod
    def _terminal_reference():
        return [
            {
                "x_ref_m": float(index),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_change_progress": 0.90 + 0.01 * index,
            }
            for index in range(10)
        ]

    def test_lane_id_is_diagnostic_but_footprint_remains_required(self):
        reference = self._terminal_reference()

        wrong_lane = evaluate_lane_change_completion(
            reference_samples=reference,
            ego_x_m=9.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            progress=1.0,
            previous_stable_frames=4,
            target_lane_matches=False,
            footprint_clearance_m=0.2,
            required_stable_frames=5,
        )
        outside_corridor = evaluate_lane_change_completion(
            reference_samples=reference,
            ego_x_m=9.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            progress=1.0,
            previous_stable_frames=4,
            target_lane_matches=True,
            footprint_clearance_m=-0.01,
            required_stable_frames=5,
        )

        # The locked target reference proves physical arrival even if map
        # matching re-anchors into another lane-ID namespace.
        self.assertTrue(wrong_lane.complete)
        self.assertIn("lane_id_mismatch", wrong_lane.reason)
        self.assertFalse(outside_corridor.complete)
        self.assertEqual(wrong_lane.stable_frames, 5)
        self.assertEqual(outside_corridor.stable_frames, 0)

    def test_releases_after_geometric_convergence_is_stable(self):
        result = evaluate_lane_change_completion(
            reference_samples=self._terminal_reference(),
            ego_x_m=9.0,
            ego_y_m=0.05,
            ego_heading_rad=math.radians(1.0),
            progress=1.0,
            previous_stable_frames=4,
            target_lane_matches=True,
            footprint_clearance_m=0.2,
            required_stable_frames=5,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.reason, "lane_change_geometrically_complete")

    def test_early_lane_id_flip_cannot_complete_without_progress_and_alignment(self):
        result = evaluate_lane_change_completion(
            reference_samples=self._terminal_reference(),
            ego_x_m=4.0,
            ego_y_m=1.0,
            ego_heading_rad=math.radians(20.0),
            progress=0.55,
            previous_stable_frames=4,
            target_lane_matches=True,
            footprint_clearance_m=0.2,
            required_stable_frames=5,
        )

        self.assertFalse(result.complete)
        self.assertEqual(result.stable_frames, 0)


if __name__ == "__main__":
    unittest.main()
