import unittest

from pipeline.reference_gate import FinalReferenceGate


def _straight_reference(lane_id=1, terminal_speed=2.0):
    samples = []
    for index in range(4):
        samples.append(
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": int(lane_id),
                "speed_ref_mps": (
                    float(terminal_speed) if index == 3 else 2.0
                ),
            }
        )
    return samples


class FinalReferenceGateTests(unittest.TestCase):
    def test_lane_follow_body_lateral_check_is_disabled_for_curved_reference(self):
        gate = FinalReferenceGate({})
        curved = _straight_reference()
        for sample, heading in zip(curved, [0.0, 0.08, 0.18, 0.30]):
            sample["heading_rad"] = float(heading)

        self.assertFalse(gate.check_destination_body_lateral(
            mode="lane_follow",
            reference_samples=curved,
        ))
        self.assertTrue(gate.check_destination_body_lateral(
            mode="lane_follow",
            reference_samples=_straight_reference(),
        ))

    def test_accepts_valid_lane_follow_reference(self):
        gate = FinalReferenceGate({})
        result = gate.validate(
            reference_samples=_straight_reference(),
            destination_state=[4.0, 0.0, 2.0],
            ego_state=[0.0, 0.0, 2.0, 0.0],
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            current_lane_id=1,
            target_lane_id=1,
            stop_goal_active=False,
            horizon_steps=4,
            default_speed_mps=3.0,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.mode, "lane_follow")

    def test_rejects_reference_that_starts_behind_ego(self):
        reference = _straight_reference()
        reference[0]["x_ref_m"] = -1.0
        gate = FinalReferenceGate({})
        result = gate.validate(
            reference_samples=reference,
            destination_state=[4.0, 0.0, 2.0],
            ego_state=[0.0, 0.0, 2.0, 0.0],
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            current_lane_id=1,
            target_lane_id=1,
            stop_goal_active=False,
            horizon_steps=4,
            default_speed_mps=3.0,
        )

        self.assertFalse(result.accepted)
        self.assertIn("first_forward_before_contract", result.reason)

    def test_stop_requires_zero_terminal_speed(self):
        gate = FinalReferenceGate({})
        result = gate.validate(
            reference_samples=_straight_reference(terminal_speed=1.0),
            destination_state=[4.0, 0.0, 0.0],
            ego_state=[0.0, 0.0, 2.0, 0.0],
            behavior_decision="stop_at_intersection",
            behavior_fsm_state="STOP",
            current_lane_id=1,
            target_lane_id=1,
            stop_goal_active=True,
            horizon_steps=4,
            default_speed_mps=3.0,
        )

        self.assertFalse(result.accepted)
        self.assertIn("terminal_speed_not_zero", result.reason)

    def test_lane_change_contract_uses_target_lane(self):
        gate = FinalReferenceGate({})
        result = gate.validate(
            reference_samples=_straight_reference(lane_id=2),
            destination_state=[4.0, 0.0, 2.0],
            ego_state=[0.0, 0.0, 2.0, 0.0],
            behavior_decision="lane_change_left",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_LEFT",
            current_lane_id=1,
            target_lane_id=2,
            stop_goal_active=False,
            horizon_steps=4,
            default_speed_mps=3.0,
        )

        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
