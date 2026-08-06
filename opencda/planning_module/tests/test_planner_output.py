import unittest

from pipeline.output import BehaviorCommand


class BehaviorCommandStopSemanticsTests(unittest.TestCase):
    def _command(self, decision):
        return BehaviorCommand.from_debug(
            behavior_debug={"decision": decision, "target_lane_id": 1},
            target_speed_mps=0.0,
        )

    def test_intersection_stop_is_normal_not_emergency(self):
        command = self._command("stop_at_intersection")

        self.assertTrue(command.normal_stop)
        self.assertTrue(command.stop_requested)
        self.assertFalse(command.emergency_brake)

    def test_stop_sign_is_normal_not_emergency(self):
        command = self._command("stop_sign")

        self.assertTrue(command.normal_stop)
        self.assertTrue(command.stop_requested)
        self.assertFalse(command.emergency_brake)

    def test_emergency_brake_is_not_normal_stop(self):
        command = self._command("emergency_brake")

        self.assertFalse(command.normal_stop)
        self.assertTrue(command.stop_requested)
        self.assertTrue(command.emergency_brake)

    def test_lane_follow_has_no_stop_request(self):
        command = self._command("lane_follow")

        self.assertFalse(command.normal_stop)
        self.assertFalse(command.stop_requested)
        self.assertFalse(command.emergency_brake)


if __name__ == "__main__":
    unittest.main()
