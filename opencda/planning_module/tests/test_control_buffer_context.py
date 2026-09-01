import unittest

from pipeline.control_buffer import MPCControlBuffer


class MPCControlBufferContextTests(unittest.TestCase):
    @staticmethod
    def _buffer():
        buffer = MPCControlBuffer(
            replan_period_s=1.0,
            max_reuse_s=1.0,
            max_reference_anchor_jump_m=0.75,
        )
        buffer.update_from_solution(
            u_solution=[[0.5, 0.1], [0.4, 0.1]],
            plan_time_s=1.0,
            dt_s=0.1,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.0, 2.0),
        )
        return buffer

    def test_same_context_and_nearby_anchor_can_reuse(self):
        buffer = self._buffer()

        self.assertFalse(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.2, 2.0),
        ))
        self.assertIsNotNone(buffer.sample(
            sim_time_s=1.1,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.2, 2.0),
        ))

    def test_phase_change_invalidates_buffer(self):
        buffer = self._buffer()

        self.assertTrue(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_change_right|TARGET_LANE_STABILIZATION|1",
            reference_anchor_relative_m=(5.2, 2.0),
        ))
        self.assertEqual(
            buffer.last_reason,
            "control_buffer_context_changed",
        )

    def test_relative_reference_lateral_jump_invalidates_buffer(self):
        buffer = self._buffer()

        self.assertTrue(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.0, 3.0),
        ))
        self.assertEqual(
            buffer.last_reason,
            "control_buffer_reference_anchor_jump",
        )

    def test_sample_rejects_stale_context_even_without_replan_check(self):
        buffer = self._buffer()

        self.assertIsNone(buffer.sample(
            sim_time_s=1.1,
            context_key="intersection_turn_right|TURN|1",
            reference_anchor_relative_m=(5.0, 2.0),
        ))
        self.assertEqual(buffer.last_reason, "control_buffer_context_changed")

    def test_entering_target_speed_band_invalidates_acceleration_plan(self):
        buffer = self._buffer()
        self.assertFalse(buffer.should_replan(
            sim_time_s=1.05,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.0, 2.0),
            ego_speed_mps=2.0,
            target_speed_mps=3.0,
        ))
        self.assertTrue(buffer.should_replan(
            sim_time_s=1.10,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.0, 2.0),
            ego_speed_mps=2.9,
            target_speed_mps=3.0,
        ))
        self.assertEqual(
            buffer.last_reason,
            "control_buffer_speed_target_band_entered",
        )

    def test_crossing_target_speed_invalidates_acceleration_plan(self):
        buffer = self._buffer()
        buffer.should_replan(
            sim_time_s=1.05,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.0, 2.0),
            ego_speed_mps=2.5,
            target_speed_mps=3.0,
        )
        self.assertTrue(buffer.should_replan(
            sim_time_s=1.10,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.0, 2.0),
            ego_speed_mps=3.1,
            target_speed_mps=3.0,
        ))
        self.assertEqual(
            buffer.last_reason,
            "control_buffer_speed_target_crossed",
        )

    def test_predicted_speed_divergence_forces_replan(self):
        # The plan's own state trajectory predicted a gentle decline
        # (3.0 -> 2.9 -> 2.8 m/s), but reality has already fallen to 1.5
        # m/s by the time sample() would use index 1 -- an open-loop
        # buffer would otherwise keep playing back the rest of a plan
        # that's no longer tracking reality.
        buffer = MPCControlBuffer(
            replan_period_s=1.0,
            max_reuse_s=1.0,
            max_reference_anchor_jump_m=0.75,
            max_predicted_speed_error_mps=0.75,
        )
        buffer.update_from_solution(
            u_solution=[[-0.5, 0.1], [-0.5, 0.1], [-0.5, 0.1]],
            plan_time_s=1.0,
            dt_s=0.1,
            context_key="lane_change_left|EXECUTE_LANE_CHANGE_LEFT|2",
            reference_anchor_relative_m=(5.0, 2.0),
            predicted_speed_sequence_mps=[3.0, 2.9, 2.8, 2.7],
        )

        self.assertTrue(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_change_left|EXECUTE_LANE_CHANGE_LEFT|2",
            reference_anchor_relative_m=(5.0, 2.0),
            ego_speed_mps=1.5,
        ))
        self.assertEqual(
            buffer.last_reason,
            "control_buffer_predicted_speed_diverged",
        )

    def test_predicted_speed_within_tolerance_still_reuses(self):
        buffer = MPCControlBuffer(
            replan_period_s=1.0,
            max_reuse_s=1.0,
            max_reference_anchor_jump_m=0.75,
            max_predicted_speed_error_mps=0.75,
        )
        buffer.update_from_solution(
            u_solution=[[-0.5, 0.1], [-0.5, 0.1], [-0.5, 0.1]],
            plan_time_s=1.0,
            dt_s=0.1,
            context_key="lane_change_left|EXECUTE_LANE_CHANGE_LEFT|2",
            reference_anchor_relative_m=(5.0, 2.0),
            predicted_speed_sequence_mps=[3.0, 2.9, 2.8, 2.7],
        )

        self.assertFalse(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_change_left|EXECUTE_LANE_CHANGE_LEFT|2",
            reference_anchor_relative_m=(5.0, 2.0),
            ego_speed_mps=2.5,
        ))
        self.assertEqual(buffer.last_reason, "control_buffer_reuse")

    def test_missing_predicted_speed_sequence_does_not_force_replan(self):
        # Backward compatibility: callers that don't pass
        # predicted_speed_sequence_mps (or a solve that exposed no state
        # trajectory) must not be affected by this check at all.
        buffer = self._buffer()

        self.assertFalse(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_relative_m=(5.2, 2.0),
            ego_speed_mps=0.0,
        ))

    def test_target_speed_jump_forces_replan_without_crossing(self):
        # Ego stays below the target both before and after the jump (2.05
        # -> 5.2 m/s), so the crossing/deadband check in
        # _longitudinal_replan_reason never fires -- this is exactly the
        # gap this new check closes.
        buffer = MPCControlBuffer(
            replan_period_s=1.0,
            max_reuse_s=1.0,
            max_target_speed_jump_mps=1.0,
        )
        buffer.update_from_solution(
            u_solution=[[0.4, 0.0], [0.4, 0.0]],
            plan_time_s=1.0,
            dt_s=0.1,
            context_key="lane_change_left|EXECUTE_LANE_CHANGE_LEFT|2",
            target_speed_mps=2.05,
        )

        self.assertTrue(buffer.should_replan(
            sim_time_s=1.05,
            context_key="lane_change_left|EXECUTE_LANE_CHANGE_LEFT|2",
            ego_speed_mps=2.3,
            target_speed_mps=5.2,
        ))
        self.assertEqual(buffer.last_reason, "control_buffer_target_speed_jumped")

    def test_small_target_speed_change_still_reuses(self):
        buffer = MPCControlBuffer(
            replan_period_s=1.0,
            max_reuse_s=1.0,
            max_target_speed_jump_mps=1.0,
        )
        buffer.update_from_solution(
            u_solution=[[0.4, 0.0], [0.4, 0.0]],
            plan_time_s=1.0,
            dt_s=0.1,
            context_key="lane_follow|LANE_KEEP|1",
            target_speed_mps=2.9,
        )

        self.assertFalse(buffer.should_replan(
            sim_time_s=1.05,
            context_key="lane_follow|LANE_KEEP|1",
            ego_speed_mps=2.6,
            target_speed_mps=3.0,
        ))
        self.assertEqual(buffer.last_reason, "control_buffer_reuse")


if __name__ == "__main__":
    unittest.main()
