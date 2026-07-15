import math
import types
import unittest

from planning_runner import (
    _fallback_signal_stop_target_from_ego,
    _keep_temporary_destination_ahead,
    _limit_destination_xy_step,
    _lock_junction_lane_follow_destination,
    _select_mpc_cost_profile_with_hysteresis,
)
from behavior_planner.reference_pipeline import (
    reference_with_route_fallback,
    stabilize_lane_reference_samples,
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

    def test_mpc_cost_profile_hysteresis_holds_non_safety_switch(self):
        profile, since_s, reason = _select_mpc_cost_profile_with_hysteresis(
            requested_profile="intersection_turn",
            active_profile="lane_follow",
            sim_time_s=10.4,
            active_since_s=10.0,
            min_hold_s=1.5,
        )

        self.assertEqual(profile, "lane_follow")
        self.assertAlmostEqual(since_s, 10.0)
        self.assertEqual(reason, "min_hold")

    def test_mpc_cost_profile_hysteresis_allows_stop_preempt(self):
        profile, since_s, reason = _select_mpc_cost_profile_with_hysteresis(
            requested_profile="stop",
            active_profile="lane_follow",
            sim_time_s=10.4,
            active_since_s=10.0,
            min_hold_s=1.5,
        )

        self.assertEqual(profile, "stop")
        self.assertAlmostEqual(since_s, 10.4)
        self.assertEqual(reason, "safety_preempt")

    def test_mpc_cost_profile_hysteresis_switches_after_hold(self):
        profile, since_s, reason = _select_mpc_cost_profile_with_hysteresis(
            requested_profile="intersection_turn",
            active_profile="lane_follow",
            sim_time_s=12.0,
            active_since_s=10.0,
            min_hold_s=1.5,
        )

        self.assertEqual(profile, "intersection_turn")
        self.assertAlmostEqual(since_s, 12.0)
        self.assertEqual(reason, "switched")

    def test_stop_destination_step_limiter_caps_large_snap(self):
        limited = _limit_destination_xy_step(
            destination_state=[20.0, 0.0, 0.0, 0.0, 1],
            previous_destination_state=[0.0, 0.0, 5.0, 0.0, 1],
            max_step_m=5.0,
        )

        self.assertIsNotNone(limited)
        self.assertAlmostEqual(float(limited[0]), 5.0)
        self.assertAlmostEqual(float(limited[1]), 0.0)
        self.assertAlmostEqual(float(limited[2]), 0.0)

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
