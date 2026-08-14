import unittest
import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "reference_contract.py"
_SPEC = importlib.util.spec_from_file_location("reference_contract", str(_MODULE_PATH))
reference_contract = importlib.util.module_from_spec(_SPEC)
sys.modules["reference_contract"] = reference_contract
_SPEC.loader.exec_module(reference_contract)

contract_from_config = reference_contract.contract_from_config
validate_reference_contract = reference_contract.validate_reference_contract


class ReferenceContractTest(unittest.TestCase):
    def _lane_follow_contract(self):
        return contract_from_config(
            mode="lane_follow",
            expected_lane_id=1,
            horizon_steps=3,
            config={},
            default_speed_mps=3.0,
        )

    def test_lane_follow_reference_passes_contract(self):
        contract = self._lane_follow_contract()
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {"x_ref_m": 3.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
        ]
        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[3.0, 0.0, 2.0, 0.0, 1],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=True,
        )
        self.assertTrue(result.valid, result.reason())

    def test_destination_lane_error_blocks_lane_follow(self):
        contract = self._lane_follow_contract()
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {"x_ref_m": 3.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
        ]
        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[3.0, 3.0, 2.0, 0.0, 1],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=True,
        )
        self.assertFalse(result.valid)
        self.assertIn("destination_lane_error_out_of_contract", result.violations)

    def test_lane_follow_allows_longitudinal_successor_lane_renumbering(self):
        contract = self._lane_follow_contract()
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {
                "x_ref_m": 2.0,
                "y_ref_m": 0.0,
                "lane_id": 2,
                "lane_transition_kind": "longitudinal_successor",
                "speed_ref_mps": 2.0,
            },
            {
                "x_ref_m": 3.0,
                "y_ref_m": 0.0,
                "lane_id": 2,
                "lane_transition_kind": "longitudinal_successor",
                "speed_ref_mps": 2.0,
            },
        ]
        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[3.0, 0.0, 2.0, 0.0, 2],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=True,
        )
        self.assertTrue(result.valid, result.reason())

    def test_lane_follow_still_blocks_unmarked_lateral_lane_transition(self):
        contract = self._lane_follow_contract()
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "lane_id": 2, "speed_ref_mps": 2.0},
            {"x_ref_m": 3.0, "y_ref_m": 0.0, "lane_id": 2, "speed_ref_mps": 2.0},
        ]
        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[3.0, 0.0, 2.0, 0.0, 2],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=True,
        )
        self.assertFalse(result.valid)
        self.assertIn("lane_id_transition_not_allowed", result.violations)

    def test_stop_requires_zero_terminal_speed(self):
        contract = contract_from_config(
            mode="stop",
            expected_lane_id=1,
            horizon_steps=3,
            config={},
            default_speed_mps=2.0,
        )
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.5},
            {"x_ref_m": 3.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.2},
        ]
        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[3.0, 0.0, 0.0, 0.0, 1],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=True,
        )
        self.assertFalse(result.valid)
        self.assertIn("terminal_speed_not_zero", result.violations)

    def test_lane_change_direct_mode_allows_full_lane_width_first_sample(self):
        # Under direct target-lane tracking, MPC is handed the target lane's
        # own (unblended) centerline -- its first sample legitimately sits
        # close to a full lane width from ego at lock time. The standard
        # "lane_change" mode's 1.25m limit (sized for an already-ramping
        # blend) must reject that; the new "lane_change_direct" mode must
        # accept it.
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 3.2, "lane_id": 2, "speed_ref_mps": 2.0},
            {"x_ref_m": 2.0, "y_ref_m": 3.2, "lane_id": 2, "speed_ref_mps": 2.0},
            {"x_ref_m": 3.0, "y_ref_m": 3.2, "lane_id": 2, "speed_ref_mps": 2.0},
        ]
        lane_change_contract = contract_from_config(
            mode="lane_change",
            expected_lane_id=1,
            horizon_steps=3,
            config={},
            default_speed_mps=3.0,
        )
        blocked = validate_reference_contract(
            reference_samples=reference,
            destination_state=[3.0, 3.2, 2.0, 0.0, 2],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=lane_change_contract,
            check_destination_body_lateral=False,
        )
        self.assertFalse(blocked.valid)
        self.assertIn("first_lateral_out_of_contract", blocked.violations)

        direct_contract = contract_from_config(
            mode="lane_change_direct",
            expected_lane_id=1,
            horizon_steps=3,
            config={},
            default_speed_mps=3.0,
        )
        self.assertGreater(
            direct_contract.max_first_lateral_abs_m,
            lane_change_contract.max_first_lateral_abs_m,
        )
        self.assertTrue(direct_contract.allow_lane_transition)
        allowed = validate_reference_contract(
            reference_samples=reference,
            destination_state=[3.0, 3.2, 2.0, 0.0, 2],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=direct_contract,
            check_destination_body_lateral=False,
        )
        self.assertTrue(allowed.valid, allowed.reason())

    def test_lane_change_destination_accepts_its_transition_segment(self):
        contract = contract_from_config(
            mode="lane_change",
            expected_lane_id=2,
            horizon_steps=4,
            config={},
            default_speed_mps=4.0,
        )
        reference = [
            {
                "x_ref_m": 1.0,
                "y_ref_m": 0.02,
                "lane_id": 1,
                "lane_transition_kind": "lateral_lane_change",
                "speed_ref_mps": 3.0,
            },
            {
                "x_ref_m": 2.0,
                "y_ref_m": 0.08,
                "lane_id": 1,
                "lane_transition_kind": "lateral_lane_change",
                "speed_ref_mps": 3.0,
            },
            {
                "x_ref_m": 3.0,
                "y_ref_m": 0.18,
                "lane_id": 2,
                "lane_transition_kind": "lateral_lane_change",
                "speed_ref_mps": 3.0,
            },
            {
                "x_ref_m": 4.0,
                "y_ref_m": 0.32,
                "lane_id": 2,
                "lane_transition_kind": "lateral_lane_change",
                "speed_ref_mps": 3.0,
            },
        ]

        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[2.0, 0.08, 3.0, 0.0, 2],
            ego_state=[0.0, 0.0, 3.0, 0.0],
            contract=contract,
            check_destination_body_lateral=False,
        )

        self.assertTrue(result.valid, result.reason())
        self.assertAlmostEqual(result.destination_lane_error_m, 0.0)

    def test_lane_follow_does_not_accept_lateral_transition_marker(self):
        contract = self._lane_follow_contract()
        reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 0.0,
                "lane_id": 1,
                "lane_transition_kind": "lateral_lane_change",
                "speed_ref_mps": 2.0,
            }
            for index in range(3)
        ]

        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[2.0, 2.0, 2.0, 0.0, 2],
            ego_state=[0.0, 0.0, 2.0, 0.0],
            contract=contract,
            check_destination_body_lateral=False,
        )

        self.assertFalse(result.valid)
        self.assertIn("destination_lane_error_out_of_contract", result.violations)

    def test_intersection_turn_allows_route_branch_body_lateral(self):
        contract = contract_from_config(
            mode="intersection_turn",
            expected_lane_id=1,
            horizon_steps=3,
            config={},
            default_speed_mps=3.0,
        )
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {"x_ref_m": 3.0, "y_ref_m": 1.0, "lane_id": 2, "speed_ref_mps": 2.0},
            {"x_ref_m": 4.0, "y_ref_m": 3.0, "lane_id": 2, "speed_ref_mps": 2.0},
        ]
        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[4.0, 3.0, 2.0, 1.10, 2],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=False,
        )
        self.assertTrue(result.valid, result.reason())

    def test_intersection_turn_uses_route_progress_not_ego_body_progress(self):
        contract = contract_from_config(
            mode="intersection_turn",
            expected_lane_id=1,
            horizon_steps=4,
            config={"reference_contract_intersection_turn_max_curvature_1pm": 2.0},
            default_speed_mps=3.0,
        )
        self.assertFalse(contract.require_monotonic_progress)
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.5, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 2.2, "y_ref_m": 1.5, "lane_id": 2, "speed_ref_mps": 1.0},
            {"x_ref_m": 1.8, "y_ref_m": 2.5, "lane_id": 2, "speed_ref_mps": 1.0},
        ]
        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[1.8, 2.5, 1.0, 1.9, 2],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=False,
        )
        self.assertTrue(result.valid, result.reason())

    def test_vehicle_curvature_limit_tightens_turn_contract(self):
        contract = contract_from_config(
            mode="intersection_turn",
            expected_lane_id=1,
            horizon_steps=4,
            config={
                "reference_contract_intersection_turn_max_curvature_1pm": 0.55,
                "reference_vehicle_max_curvature_1pm": 0.20,
            },
            default_speed_mps=2.0,
        )

        self.assertAlmostEqual(contract.max_curvature_1pm, 0.20)

    def test_validation_reports_vehicle_curvature_margin(self):
        contract = contract_from_config(
            mode="intersection_turn",
            expected_lane_id=1,
            horizon_steps=4,
            config={"reference_vehicle_max_curvature_1pm": 0.10},
            default_speed_mps=2.0,
        )
        reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0},
            {"x_ref_m": 2.98, "y_ref_m": 0.20},
            {"x_ref_m": 3.88, "y_ref_m": 0.63},
        ]
        result = validate_reference_contract(
            reference_samples=reference,
            destination_state=[3.88, 0.63, 1.0, 0.0, 1],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=False,
        )

        self.assertFalse(result.valid)
        self.assertIn("curvature_out_of_contract", result.violations)
        self.assertLess(result.curvature_margin_1pm, 0.0)


if __name__ == "__main__":
    unittest.main()
