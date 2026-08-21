import unittest

from behavior_planner.planner import RuleBasedBehaviorPlanner


class BehaviorCommandInvariantEnforcementTests(unittest.TestCase):
    """`_make_result` must correct, not just log, commands that violate the
    architecture proposal's Behavior Layer invariants -- Path/MPC have no
    safe way to recover from a self-contradictory command once it leaves the
    behavior layer.
    """

    def _corrected(self, decision: str, target_lane_id: int, ego_lane_id: int):
        planner = RuleBasedBehaviorPlanner()
        planner._current_ego_lane_id = int(ego_lane_id)
        result = {
            "decision": decision,
            "target_lane_id": int(target_lane_id),
            "selected_lane_id": int(target_lane_id),
        }
        planner._enforce_behavior_command_invariants(result)
        return result

    def test_stop_at_intersection_target_lane_is_forced_to_ego_lane(self):
        result = self._corrected("stop_at_intersection", target_lane_id=5, ego_lane_id=2)

        self.assertEqual(result["decision"], "stop_at_intersection")
        self.assertEqual(int(result["target_lane_id"]), 2)
        self.assertEqual(int(result["selected_lane_id"]), 2)
        self.assertIn("invariant_violations", result)

    def test_stop_sign_target_lane_is_forced_to_ego_lane(self):
        result = self._corrected("stop_sign", target_lane_id=7, ego_lane_id=3)

        self.assertEqual(int(result["target_lane_id"]), 3)
        self.assertEqual(int(result["selected_lane_id"]), 3)

    def test_emergency_brake_target_lane_is_forced_to_ego_lane(self):
        result = self._corrected("emergency_brake", target_lane_id=9, ego_lane_id=1)

        self.assertEqual(result["decision"], "emergency_brake")
        self.assertEqual(int(result["target_lane_id"]), 1)
        self.assertEqual(int(result["selected_lane_id"]), 1)

    def test_static_obstacle_stop_target_lane_is_forced_to_ego_lane(self):
        result = self._corrected(
            "static_obstacle_stop", target_lane_id=9, ego_lane_id=4
        )

        self.assertEqual(result["decision"], "static_obstacle_stop")
        self.assertEqual(int(result["target_lane_id"]), 4)
        self.assertEqual(int(result["selected_lane_id"]), 4)

    def test_already_consistent_stop_command_is_left_untouched(self):
        result = self._corrected("stop_at_intersection", target_lane_id=2, ego_lane_id=2)

        self.assertNotIn("invariant_violations", result)
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_lane_change_target_equal_to_ego_lane_is_not_flagged(self):
        # A reversing/aborted lane change legitimately reports its target as
        # the (still current) ego lane before the vehicle has moved -- see
        # test_ongoing_lane_change_reverses_when_target_lane_becomes_worse_than_previous_lane
        # in test_intersection_behavior.py. This must not be "corrected".
        result = self._corrected("lane_change_right", target_lane_id=1, ego_lane_id=1)

        self.assertNotIn("invariant_violations", result)
        self.assertEqual(result["decision"], "lane_change_right")
        self.assertEqual(int(result["target_lane_id"]), 1)


if __name__ == "__main__":
    unittest.main()
