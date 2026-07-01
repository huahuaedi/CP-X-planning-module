import math
import types
import unittest

from planning_runner import (
    _fallback_signal_stop_target_from_ego,
    _keep_temporary_destination_ahead,
    _lock_junction_lane_follow_destination,
    _reference_with_route_fallback,
)


class TempDestinationForwardGuardTests(unittest.TestCase):
    def test_uses_previous_destination_when_new_destination_is_behind(self):
        repaired = _keep_temporary_destination_ahead(
            temporary_destination_state=[-5.0, 0.0, 4.0, 0.0, 2, 0.0, 10, 0.0],
            previous_destination_state=[12.0, 1.0, 3.0, 0.1, 2, 0.0, 10, 0.0],
            ego_state=[0.0, 0.0, 2.0, 0.0],
            current_behavior="lane_follow",
            final_goal_stop_active=False,
        )

        self.assertIsNotNone(repaired)
        self.assertAlmostEqual(float(repaired[0]), 12.0)
        self.assertAlmostEqual(float(repaired[1]), 1.0)
        self.assertAlmostEqual(float(repaired[2]), 4.0)

    def test_generates_forward_fallback_when_no_previous_destination_is_safe(self):
        repaired = _keep_temporary_destination_ahead(
            temporary_destination_state=[-5.0, 0.0, 4.0, 0.0, 2, 0.0, 10, 0.0],
            previous_destination_state=[-2.0, 0.0, 3.0, 0.0, 2, 0.0, 10, 0.0],
            ego_state=[1.0, 2.0, 2.0, math.pi / 2.0],
            current_behavior="lane_follow",
            final_goal_stop_active=False,
            fallback_forward_m=8.0,
        )

        self.assertIsNotNone(repaired)
        self.assertAlmostEqual(float(repaired[0]), 1.0, places=6)
        self.assertAlmostEqual(float(repaired[1]), 10.0, places=6)
        self.assertAlmostEqual(float(repaired[3]), math.pi / 2.0, places=6)

    def test_does_not_modify_fixed_stop_destination(self):
        destination = [0.0, -5.0, 0.0, 0.0, 2, 1.0, 10, 1.0]
        repaired = _keep_temporary_destination_ahead(
            temporary_destination_state=destination,
            previous_destination_state=[10.0, 0.0, 2.0, 0.0, 2, 1.0, 10, 1.0],
            ego_state=[0.0, 0.0, 2.0, 0.0],
            current_behavior="stop_at_intersection",
            final_goal_stop_active=False,
        )

        self.assertEqual(repaired, destination)

    def test_actor_position_signal_does_not_create_far_fallback_stop_target(self):
        stop_target = _fallback_signal_stop_target_from_ego(
            world_map=types.SimpleNamespace(),
            carla=types.SimpleNamespace(),
            ego_transform=types.SimpleNamespace(),
            signal_context={
                "signal_state": "red",
                "signal_source": "actor_position_match",
                "signal_forward_m": 55.0,
                "signal_lateral_m": 3.0,
                "signal_distance_m": 55.0,
            },
            search_distance_m=100.0,
            stop_buffer_m=2.0,
        )

        self.assertIsNone(stop_target)

    def test_reference_fallback_generates_forward_heading_samples_when_route_is_bad(self):
        samples, reason = _reference_with_route_fallback(
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

    def test_junction_lane_follow_reuses_previous_destination_when_lane_jumps(self):
        repaired = _lock_junction_lane_follow_destination(
            temporary_destination_state=[20.0, 5.0, 6.0, 0.0, 1, 0.0, 100, 1.0],
            previous_destination_state=[8.0, 0.0, 4.0, 0.0, 2, 0.0, 100, 1.0],
            ego_state=[0.0, 0.0, 3.0, 0.0],
            ego_in_junction=True,
            current_behavior="lane_follow",
            planner_lc_state="IDLE",
            selected_lane_id=2,
        )

        self.assertIsNotNone(repaired)
        self.assertAlmostEqual(float(repaired[0]), 8.0)
        self.assertAlmostEqual(float(repaired[1]), 0.0)
        self.assertAlmostEqual(float(repaired[2]), 6.0)
        self.assertEqual(int(repaired[4]), 2)

    def test_junction_lane_lock_does_not_block_active_lane_change(self):
        destination = [20.0, 5.0, 6.0, 0.0, 1, 0.0, 100, 1.0]
        repaired = _lock_junction_lane_follow_destination(
            temporary_destination_state=destination,
            previous_destination_state=[8.0, 0.0, 4.0, 0.0, 2, 0.0, 100, 1.0],
            ego_state=[0.0, 0.0, 3.0, 0.0],
            ego_in_junction=True,
            current_behavior="lane_change_right",
            planner_lc_state="EXECUTE_LANE_CHANGE_RIGHT",
            selected_lane_id=1,
        )

        self.assertEqual(repaired, destination)


if __name__ == "__main__":
    unittest.main()
