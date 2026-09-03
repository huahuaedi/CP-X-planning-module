import math
import unittest

from pipeline.stage_contracts import (
    LaneChangeContract,
    authorize_mpc_entry,
    evaluate_lane_change_completion,
)


class LaneChangeCompletionTests(unittest.TestCase):
    def test_committed_reference_probe_failure_reaches_authoritative_solve(self):
        result = authorize_mpc_entry(
            candidate_status="mpc_probe_infeasible",
            candidate_name="committed_lane_change_continuation",
            candidate_reason="mpc_probe:primal infeasible",
            final_reference_accepted=True,
            final_reference_reason="",
            behavior_decision="lane_change_left",
        )

        self.assertTrue(result.allowed)
        self.assertEqual(
            result.status, "authorized_committed_after_probe_failure"
        )

    def test_contract_parses_and_clamps_one_configuration_source(self):
        contract = LaneChangeContract.from_config({
            "lane_change_completion_min_progress": 1.5,
            "lane_change_completion_max_lateral_error_m": 0.2,
            "lane_change_completion_max_heading_error_deg": 5.0,
            "lane_change_completion_stable_frames": 0,
        })

        self.assertEqual(contract.min_progress, 1.0)
        self.assertEqual(contract.max_lateral_error_m, 0.2)
        self.assertAlmostEqual(contract.max_heading_error_rad, math.radians(5.0))
        self.assertEqual(contract.required_stable_frames, 1)
        self.assertTrue(contract.convergence_ready(
            progress=1.0,
            lateral_error_m=0.2,
            heading_error_rad=math.radians(5.0),
        ))

    def test_stabilization_handoff_precedes_final_convergence(self):
        contract = LaneChangeContract()

        self.assertTrue(contract.stabilization_handoff_ready(
            progress=0.98,
            lateral_error_m=0.8,
            heading_error_rad=math.radians(9.0),
            lane_width_m=3.5,
        ))
        self.assertFalse(contract.convergence_ready(
            progress=0.98,
            lateral_error_m=0.8,
            heading_error_rad=math.radians(9.0),
        ))
        self.assertFalse(contract.stabilization_handoff_ready(
            progress=0.98,
            lateral_error_m=1.4,
            heading_error_rad=math.radians(9.0),
            lane_width_m=3.5,
        ))

    def test_completion_accepts_central_contract(self):
        contract = LaneChangeContract(
            min_progress=0.95,
            max_lateral_error_m=0.1,
            max_heading_error_rad=math.radians(2.0),
            required_stable_frames=2,
        )
        result = evaluate_lane_change_completion(
            reference_samples=self._terminal_reference(),
            ego_x_m=9.0,
            ego_y_m=0.05,
            ego_heading_rad=math.radians(1.0),
            progress=1.0,
            previous_stable_frames=1,
            contract=contract,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.stable_frames, 2)

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
