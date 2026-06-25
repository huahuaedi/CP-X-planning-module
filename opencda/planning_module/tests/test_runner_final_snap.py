import json
import math
import os
import tempfile
import types
import unittest

from carla_scenario.runner import (
    _apply_exact_stop_target_snap,
    _apply_final_destination_snap,
    _destination_matches_target_point,
    _final_destination_stop_target_state,
    _follow_target_state_from_behavior_output,
    _has_pending_lane_closure_reroute_request,
    _apply_stop_target_speed_cap,
    _reference_lane_from_blue_dot,
    _route_tracking_target_lane_after_reroute,
    _selected_lane_id_for_behavior_step,
    _should_force_stationary_release_replan,
    _world_state_from_vehicle,
)


class RunnerFinalDestinationSnapTests(unittest.TestCase):
    def test_keeps_temporary_destination_when_not_within_snap_distance(self):
        temporary_destination_state, active_v_max = _apply_final_destination_snap(
            temporary_destination_state=[0.0, 0.0, 5.0, 0.0, 1],
            final_destination_state=[10.0, 0.0, 0.0, 0.5],
            ego_state=[0.0, 0.0, 4.0, 0.0],
            lock_to_final_distance_m=5.0,
            original_max_velocity_mps=7.0,
        )

        self.assertEqual(temporary_destination_state, [0.0, 0.0, 5.0, 0.0, 1])
        self.assertAlmostEqual(active_v_max, 7.0)

    def test_snaps_blue_dot_to_final_destination_and_tapers_speed(self):
        temporary_destination_state, active_v_max = _apply_final_destination_snap(
            temporary_destination_state=[9.5, 0.2, 5.0, 0.0, 2, 0.0],
            final_destination_state=[10.0, 0.0, 0.0, 1.2],
            ego_state=[8.0, 0.0, 4.0, 0.0],
            lock_to_final_distance_m=5.0,
            original_max_velocity_mps=7.0,
        )

        self.assertIsNotNone(temporary_destination_state)
        self.assertAlmostEqual(float(temporary_destination_state[0]), 10.0)
        self.assertAlmostEqual(float(temporary_destination_state[1]), 0.0)
        self.assertAlmostEqual(float(temporary_destination_state[2]), 7.0 * (2.0 / 5.0))
        self.assertAlmostEqual(float(temporary_destination_state[3]), 1.2)
        self.assertAlmostEqual(active_v_max, 7.0 * (2.0 / 5.0))

    def test_final_destination_snap_can_taper_speed_over_shorter_distance_than_snap_window(self):
        temporary_destination_state, active_v_max = _apply_final_destination_snap(
            temporary_destination_state=[9.5, 0.2, 5.0, 0.0, 2, 0.0],
            final_destination_state=[10.0, 0.0, 0.0, 1.2],
            ego_state=[8.0, 0.0, 4.0, 0.0],
            lock_to_final_distance_m=20.0,
            original_max_velocity_mps=7.0,
            speed_taper_distance_m=5.0,
        )

        self.assertIsNotNone(temporary_destination_state)
        self.assertAlmostEqual(float(temporary_destination_state[0]), 10.0)
        self.assertAlmostEqual(float(temporary_destination_state[1]), 0.0)
        self.assertAlmostEqual(float(temporary_destination_state[2]), 7.0 * (2.0 / 5.0))
        self.assertAlmostEqual(float(temporary_destination_state[3]), 1.2)
        self.assertAlmostEqual(active_v_max, 7.0 * (2.0 / 5.0))

    def test_snapped_final_destination_can_be_reused_as_stop_target(self):
        snapped_destination_state = [10.0, 0.0, 1.0, 1.2, 2, 0.0, 9.0, 0.0]
        final_destination_state = [10.0, 0.0, 0.0, 1.2]

        self.assertTrue(
            _destination_matches_target_point(
                destination_state=snapped_destination_state,
                target_state=final_destination_state,
            )
        )

        stop_target_state = _final_destination_stop_target_state(
            destination_state=snapped_destination_state,
            final_destination_state=final_destination_state,
        )

        self.assertIsNotNone(stop_target_state)
        self.assertAlmostEqual(float(stop_target_state[0]), 10.0)
        self.assertAlmostEqual(float(stop_target_state[1]), 0.0)
        self.assertAlmostEqual(float(stop_target_state[2]), 0.0)
        self.assertAlmostEqual(float(stop_target_state[3]), 1.2)
        self.assertEqual(int(stop_target_state[4]), 2)

    def test_stop_target_speed_cap_shapes_speed_toward_stop_point_without_zeroing_immediately(self):
        temporary_destination_state, active_v_max = _apply_stop_target_speed_cap(
            temporary_destination_state=[0.0, 20.0, 0.0, 1.57, 1],
            ego_state=[0.0, 0.0, 6.0, 1.57],
            stop_target_distance_m=20.0,
            original_max_velocity_mps=7.0,
            braking_deceleration_mps2=4.0,
            stop_buffer_m=2.0,
        )

        self.assertIsNotNone(temporary_destination_state)
        self.assertGreater(float(temporary_destination_state[2]), 0.0)
        self.assertAlmostEqual(float(temporary_destination_state[2]), float(active_v_max))
        self.assertLessEqual(float(active_v_max), 7.0)

    def test_exact_stop_target_snap_uses_final_destination_stop_method(self):
        temporary_destination_state, active_v_max = _apply_exact_stop_target_snap(
            temporary_destination_state=[10.0, 0.0, 5.0, 0.0, 1, 1.0],
            stop_target_state=[10.0, 0.0, 0.0, 1.2, 1, 1.0, 10.0, 0.0],
            ego_state=[8.0, 0.0, 4.0, 0.0],
            lock_to_stop_distance_m=5.0,
            original_max_velocity_mps=7.0,
        )

        self.assertIsNotNone(temporary_destination_state)
        self.assertAlmostEqual(float(temporary_destination_state[0]), 10.0)
        self.assertAlmostEqual(float(temporary_destination_state[1]), 0.0)
        self.assertAlmostEqual(float(temporary_destination_state[2]), 7.0 * (2.0 / 5.0))
        self.assertAlmostEqual(float(temporary_destination_state[3]), 1.2)
        self.assertAlmostEqual(active_v_max, 7.0 * (2.0 / 5.0))

    def test_world_state_from_vehicle_zeroes_small_speed(self):
        class _Velocity:
            def __init__(self, x, y, z):
                self.x = float(x)
                self.y = float(y)
                self.z = float(z)

        class _Transform:
            def __init__(self):
                self.location = type("Location", (), {"x": 1.0, "y": 2.0, "z": 0.0})()
                self.rotation = type("Rotation", (), {"yaw": 90.0})()

        class _Vehicle:
            def get_transform(self):
                return _Transform()

            def get_velocity(self):
                return _Velocity(0.1, 0.0, 0.0)

        ego_state = _world_state_from_vehicle(_Vehicle(), zero_speed_threshold_mps=0.3)
        self.assertEqual(float(ego_state[2]), 0.0)

    def test_stationary_release_replan_triggers_for_non_stop_behavior(self):
        self.assertTrue(
            _should_force_stationary_release_replan(
                ego_speed_mps=0.0,
                zero_speed_threshold_mps=0.3,
                current_behavior="lane_follow",
                target_v_ref_mps=4.0,
            )
        )

    def test_stationary_release_replan_does_not_trigger_for_stop_behavior(self):
        self.assertFalse(
            _should_force_stationary_release_replan(
                ego_speed_mps=0.0,
                zero_speed_threshold_mps=0.3,
                current_behavior="stop",
                target_v_ref_mps=4.0,
            )
        )

    def test_stationary_release_replan_triggers_for_near_stop_stale_braking(self):
        self.assertTrue(
            _should_force_stationary_release_replan(
                ego_speed_mps=0.8,
                zero_speed_threshold_mps=0.3,
                current_behavior="lane_follow",
                target_v_ref_mps=5.0,
                current_acceleration_mps2=-1.0,
            )
        )

    def test_stationary_release_replan_does_not_trigger_while_cruising(self):
        self.assertFalse(
            _should_force_stationary_release_replan(
                ego_speed_mps=3.0,
                zero_speed_threshold_mps=0.3,
                current_behavior="lane_follow",
                target_v_ref_mps=5.0,
                current_acceleration_mps2=-1.0,
            )
        )

    def test_follow_target_state_snaps_to_leading_vehicle_waypoint(self):
        class _DummyWaypoint:
            def __init__(self):
                self.road_id = 7
                self.section_id = 1
                self.lane_id = 1
                self.transform = types.SimpleNamespace(
                    location=types.SimpleNamespace(x=10.0, y=2.0, z=0.0),
                    rotation=types.SimpleNamespace(yaw=45.0),
                )

            def get_left_lane(self):
                return None

            def get_right_lane(self):
                return None

        class _DummyMap:
            def get_waypoint(self, location, project_to_road=True, lane_type=None):
                del location
                del project_to_road
                del lane_type
                return _DummyWaypoint()

        class _DummyCarla:
            class Location:
                def __init__(self, x, y, z):
                    self.x = float(x)
                    self.y = float(y)
                    self.z = float(z)

            class LaneType:
                Driving = "Driving"

        ego_transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0, z=0.5)
        )
        follow_target_state = _follow_target_state_from_behavior_output(
            world_map=_DummyMap(),
            carla=_DummyCarla,
            ego_transform=ego_transform,
            follow_target={
                "x_m": 9.7,
                "y_m": 1.9,
                "target_v_mps": 2.5,
                "heading_rad": 0.0,
                "lane_id": 99,
                "road_id": 42,
            },
        )

        self.assertIsNotNone(follow_target_state)
        self.assertAlmostEqual(float(follow_target_state[0]), 10.0)
        self.assertAlmostEqual(float(follow_target_state[1]), 2.0)
        self.assertAlmostEqual(float(follow_target_state[2]), 2.5)
        self.assertAlmostEqual(float(follow_target_state[3]), math.radians(45.0))
        self.assertAlmostEqual(float(follow_target_state[4]), 1.0)
        self.assertAlmostEqual(float(follow_target_state[6]), 7.0)

    def test_selected_lane_id_prefers_reroute_override(self):
        selected_lane_id = _selected_lane_id_for_behavior_step(
            planner_output={"selected_lane_id": 1, "target_lane_id": 1},
            current_lane_id=1,
            allowed_lane_ids=[1, 2],
            reroute_lane_override=2,
        )

        self.assertEqual(int(selected_lane_id), 2)

    def test_selected_lane_id_falls_back_to_planner_output_without_reroute_override(self):
        selected_lane_id = _selected_lane_id_for_behavior_step(
            planner_output={"selected_lane_id": 2, "target_lane_id": 1},
            current_lane_id=1,
            allowed_lane_ids=[1, 2],
            reroute_lane_override=None,
        )

        self.assertEqual(int(selected_lane_id), 2)

    def test_selected_lane_id_keeps_lane_change_target_even_if_local_context_is_still_single_lane(self):
        selected_lane_id = _selected_lane_id_for_behavior_step(
            planner_output={
                "decision": "lane_change_left",
                "selected_lane_id": 2,
                "target_lane_id": 2,
            },
            current_lane_id=1,
            allowed_lane_ids=[1],
            reroute_lane_override=None,
        )

        self.assertEqual(int(selected_lane_id), 2)

    def test_route_tracking_helper_keeps_lane_follow_on_rerouted_route_until_planner_catches_up(self):
        target_lane_id, should_follow_route_lane, latch_state = (
            _route_tracking_target_lane_after_reroute(
                current_behavior="lane_follow",
                planner_selected_lane_id=1,
                route_optimal_lane_id=2,
                reroute_route_follow_latched=True,
            )
        )

        self.assertEqual(int(target_lane_id), 2)
        self.assertTrue(bool(should_follow_route_lane))
        self.assertTrue(bool(latch_state))

    def test_route_tracking_helper_clears_latch_once_planner_matches_rerouted_route_lane(self):
        target_lane_id, should_follow_route_lane, latch_state = (
            _route_tracking_target_lane_after_reroute(
                current_behavior="lane_follow",
                planner_selected_lane_id=2,
                route_optimal_lane_id=2,
                reroute_route_follow_latched=True,
            )
        )

        self.assertEqual(int(target_lane_id), 2)
        self.assertTrue(bool(should_follow_route_lane))
        self.assertFalse(bool(latch_state))

    def test_route_tracking_helper_respects_lane_change_decision_while_preserving_latch(self):
        target_lane_id, should_follow_route_lane, latch_state = (
            _route_tracking_target_lane_after_reroute(
                current_behavior="lane_change_left",
                planner_selected_lane_id=1,
                route_optimal_lane_id=2,
                reroute_route_follow_latched=True,
            )
        )

        self.assertEqual(int(target_lane_id), 1)
        self.assertFalse(bool(should_follow_route_lane))
        self.assertTrue(bool(latch_state))

    def test_reference_lane_follows_blue_dot_lane_instead_of_planner_target_lane(self):
        target_lane_id, should_follow_route_lane = _reference_lane_from_blue_dot(
            planner_reference_lane_id=2,
            temporary_destination_state=[10.0, 0.0, 0.0, 0.0, 1],
            allowed_lane_ids=[1, 2],
            route_optimal_lane_id=2,
            should_follow_global_route_lane=False,
        )

        self.assertEqual(int(target_lane_id), 1)
        self.assertFalse(bool(should_follow_route_lane))

    def test_reference_lane_clears_route_follow_when_blue_dot_lane_differs_from_route_lane(self):
        target_lane_id, should_follow_route_lane = _reference_lane_from_blue_dot(
            planner_reference_lane_id=2,
            temporary_destination_state=[10.0, 0.0, 0.0, 0.0, 1],
            allowed_lane_ids=[1, 2],
            route_optimal_lane_id=2,
            should_follow_global_route_lane=True,
        )

        self.assertEqual(int(target_lane_id), 1)
        self.assertFalse(bool(should_follow_route_lane))

    def test_reference_lane_preserves_route_follow_when_blue_dot_lane_matches_route_lane(self):
        target_lane_id, should_follow_route_lane = _reference_lane_from_blue_dot(
            planner_reference_lane_id=1,
            temporary_destination_state=[10.0, 0.0, 0.0, 0.0, 2],
            allowed_lane_ids=[1, 2],
            route_optimal_lane_id=2,
            should_follow_global_route_lane=True,
        )

        self.assertEqual(int(target_lane_id), 2)
        self.assertTrue(bool(should_follow_route_lane))

    def test_detects_pending_lane_closure_reroute_request(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "lane_events": [
                            {
                                "id": "closure_1",
                                "type": "lane_closure",
                                "position": [10.0, 20.0],
                            }
                        ],
                        "control": [],
                    },
                    message_file,
                )

            self.assertTrue(_has_pending_lane_closure_reroute_request(message_path))

    def test_ignores_non_lane_closure_messages_for_pending_reroute_detection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "lane_events": [
                            {
                                "id": "speed_note",
                                "type": "speed_limit",
                                "position": [10.0, 20.0],
                            }
                        ],
                        "control": [],
                    },
                    message_file,
                )

            self.assertFalse(_has_pending_lane_closure_reroute_request(message_path))


if __name__ == "__main__":
    unittest.main()
