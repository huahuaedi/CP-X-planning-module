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
            reference_anchor_xy=(5.0, 2.0),
        )
        return buffer

    def test_same_context_and_nearby_anchor_can_reuse(self):
        buffer = self._buffer()

        self.assertFalse(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_xy=(5.2, 2.0),
        ))
        self.assertIsNotNone(buffer.sample(
            sim_time_s=1.1,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_xy=(5.2, 2.0),
        ))

    def test_phase_change_invalidates_buffer(self):
        buffer = self._buffer()

        self.assertTrue(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_change_right|TARGET_LANE_STABILIZATION|1",
            reference_anchor_xy=(5.2, 2.0),
        ))
        self.assertEqual(
            buffer.last_reason,
            "control_buffer_context_changed",
        )

    def test_reference_anchor_jump_invalidates_buffer(self):
        buffer = self._buffer()

        self.assertTrue(buffer.should_replan(
            sim_time_s=1.1,
            context_key="lane_follow|LANE_KEEP|1",
            reference_anchor_xy=(6.0, 2.0),
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
            reference_anchor_xy=(5.0, 2.0),
        ))
        self.assertEqual(buffer.last_reason, "control_buffer_context_changed")


if __name__ == "__main__":
    unittest.main()
