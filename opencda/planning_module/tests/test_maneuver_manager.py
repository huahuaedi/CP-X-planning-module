import unittest
import sys
import types


if "carla" not in sys.modules:
    fake_carla = types.ModuleType("carla")

    class _Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class _VehicleControl:
        def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
            self.throttle = throttle
            self.brake = brake
            self.steer = steer

    fake_carla.Location = _Location
    fake_carla.VehicleControl = _VehicleControl
    sys.modules["carla"] = fake_carla

from pipeline.maneuver_manager import ManeuverManager


def _reference(y_offset=0.0, speed=2.0):
    return [
        {
            "x_ref_m": float(index),
            "y_ref_m": float(y_offset),
            "heading_rad": 0.0,
            "speed_ref_mps": float(speed),
        }
        for index in range(1, 21)
    ]


class ManeuverManagerTests(unittest.TestCase):
    def test_plain_route_lane_change_is_not_classified_as_turn_chained(self):
        manager = ManeuverManager()

        result = manager.update(
            reference_samples=_reference(0.0, 2.0),
            destination_state=[20.0, 0.0, 2.0, 0.0, 2],
            decision="lane_change_right",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            current_lane_id=1,
            target_lane_id=2,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="locked_quintic_lane_change_reference",
            route_current_option="CHANGELANERIGHT",
            route_next_maneuver="Lane Change Right",
        )

        self.assertEqual(result.debug["maneuver_geometry_type"], "lane_change")

    def test_extend_geometry_does_not_append_the_same_path_back_to_its_start(self):
        geometry = ManeuverManager._extend_geometry(_reference(), _reference())

        xs = [float(sample["x_ref_m"]) for sample in geometry]
        self.assertEqual(len(xs), 20)
        self.assertTrue(all(second > first for first, second in zip(xs, xs[1:])))

    def test_keeps_one_owner_across_lane_change_stabilization_and_turn(self):
        manager = ManeuverManager()
        lane_change = manager.update(
            reference_samples=_reference(0.0, 2.1),
            destination_state=[20.0, 0.0, 2.1, 0.0, 2],
            decision="lane_change_right",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            current_lane_id=1,
            target_lane_id=2,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="locked_quintic_lane_change_reference",
            route_next_maneuver="RIGHT",
        )
        maneuver_id = lane_change.debug["maneuver_geometry_id"]

        stabilization = manager.update(
            reference_samples=_reference(1.0, 2.1),
            destination_state=[20.0, 1.0, 2.1, 0.0, 2],
            decision="lane_change_right",
            behavior_fsm_state="TARGET_LANE_STABILIZATION",
            current_lane_id=2,
            target_lane_id=2,
            ego_x_m=1.0,
            ego_y_m=0.2,
            reference_source="target_lane_stabilization_reference",
            route_next_maneuver="RIGHT",
        )
        turn = manager.update(
            reference_samples=_reference(2.0, 1.8),
            destination_state=[20.0, 2.0, 1.8, 0.0, 2],
            decision="intersection_turn_right",
            behavior_fsm_state="INTERSECTION_TURN_RIGHT",
            current_lane_id=2,
            target_lane_id=2,
            ego_x_m=2.0,
            ego_y_m=0.5,
            reference_source="carla_grp_waypoint_turn",
            route_current_option="RIGHT",
        )

        self.assertEqual(stabilization.debug["maneuver_geometry_id"], maneuver_id)
        self.assertEqual(turn.debug["maneuver_geometry_id"], maneuver_id)
        self.assertEqual(turn.debug["maneuver_geometry_owner"], "ManeuverManager")
        self.assertLess(
            abs(stabilization.reference_samples[0]["y_ref_m"]),
            0.5,
        )

    def test_stop_changes_velocity_without_replacing_geometry(self):
        manager = ManeuverManager()
        active = manager.update(
            reference_samples=_reference(0.0, 2.0),
            destination_state=[20.0, 0.0, 2.0, 0.0, 2],
            decision="lane_change_right",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            current_lane_id=1,
            target_lane_id=2,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="lane_change",
            route_next_maneuver="RIGHT",
        )
        stopped = manager.update(
            reference_samples=_reference(3.0, 0.0),
            destination_state=[5.0, 3.0, 0.0, 0.0, 2],
            decision="stop_at_intersection",
            behavior_fsm_state="LANE_KEEP",
            current_lane_id=2,
            target_lane_id=2,
            ego_x_m=1.0,
            ego_y_m=0.0,
            reference_source="independent_stop_reference",
            route_next_maneuver="RIGHT",
            stop_goal_active=True,
        )

        self.assertEqual(
            stopped.debug["maneuver_geometry_id"],
            active.debug["maneuver_geometry_id"],
        )
        self.assertTrue(all(
            sample["speed_ref_mps"] == 0.0
            for sample in stopped.reference_samples
        ))
        self.assertLess(abs(stopped.reference_samples[0]["y_ref_m"]), 0.1)

    def test_releases_after_route_maneuver_is_complete(self):
        manager = ManeuverManager()
        manager.update(
            reference_samples=_reference(),
            destination_state=[20.0, 0.0, 2.0, 0.0, 2],
            decision="intersection_turn_right",
            behavior_fsm_state="INTERSECTION_TURN_RIGHT",
            current_lane_id=2,
            target_lane_id=2,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="turn",
            route_current_option="RIGHT",
        )
        released = manager.update(
            reference_samples=_reference(),
            destination_state=[20.0, 0.0, 2.0, 0.0, 2],
            decision="lane_follow",
            behavior_fsm_state="LANE_KEEP",
            current_lane_id=2,
            target_lane_id=2,
            ego_x_m=10.0,
            ego_y_m=0.0,
            reference_source="lane_follow",
            route_current_option="LANEFOLLOW",
            route_next_maneuver="",
        )

        self.assertFalse(released.debug["maneuver_geometry_active"])
        self.assertEqual(manager.active_plan, None)

    def test_lane_change_macro_releases_stale_intersection_geometry(self):
        manager = ManeuverManager()
        turn = manager.update(
            reference_samples=_reference(),
            destination_state=[20.0, 0.0, 2.0, 0.0, 1],
            decision="intersection_turn_left",
            behavior_fsm_state="INTERSECTION_TURN_LEFT",
            current_lane_id=1,
            target_lane_id=1,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="turn",
            route_current_option="LEFT",
            route_next_maneuver="Turn Left",
        )
        released = manager.update(
            reference_samples=_reference(1.0),
            destination_state=[20.0, 1.0, 2.0, 0.0, 1],
            decision="lane_follow",
            behavior_fsm_state="LANE_KEEP",
            current_lane_id=1,
            target_lane_id=1,
            ego_x_m=5.0,
            ego_y_m=0.0,
            reference_source="lane_follow",
            # CARLA can still report LEFT while AD-map has advanced.
            route_current_option="LEFT",
            route_next_maneuver="Lane Change Right",
        )

        self.assertTrue(turn.debug["maneuver_geometry_active"])
        self.assertFalse(released.debug["maneuver_geometry_active"])
        self.assertEqual(
            released.debug["maneuver_geometry_release_reason"],
            "route_advanced_to_lane_change",
        )
        self.assertIsNone(manager.active_plan)

    def test_exit_stabilization_keeps_turn_geometry_after_route_advances(self):
        manager = ManeuverManager()
        turn = manager.update(
            reference_samples=_reference(),
            destination_state=[20.0, 0.0, 2.0, 0.0, 1],
            decision="intersection_turn_left",
            behavior_fsm_state="INTERSECTION_TURN_LEFT",
            current_lane_id=1,
            target_lane_id=1,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="turn",
            route_current_option="LEFT",
            route_next_maneuver="Turn Left",
        )
        held = manager.update(
            reference_samples=_reference(0.2),
            destination_state=[20.0, 0.2, 2.0, 0.0, 1],
            decision="intersection_turn_left",
            behavior_fsm_state="INTERSECTION_TURN_LEFT",
            current_lane_id=1,
            target_lane_id=1,
            ego_x_m=5.0,
            ego_y_m=0.0,
            reference_source="turn_exit_stabilization",
            route_current_option="LaneFollow",
            route_next_maneuver="Lane Change Right",
        )

        self.assertTrue(turn.debug["maneuver_geometry_active"])
        self.assertTrue(held.debug["maneuver_geometry_active"])
        self.assertEqual(
            held.debug["maneuver_geometry_id"],
            turn.debug["maneuver_geometry_id"],
        )
        self.assertIsNotNone(manager.active_plan)

    def test_retained_turn_continuation_uses_owned_geometry_and_new_speed(self):
        manager = ManeuverManager()
        manager.update(
            reference_samples=_reference(0.5, 2.0),
            destination_state=[20.0, 0.5, 2.0, 0.0, 1],
            decision="intersection_turn_left",
            behavior_fsm_state="INTERSECTION_TURN_LEFT",
            current_lane_id=1,
            target_lane_id=1,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="turn",
            route_current_option="LEFT",
        )

        continuation = manager.retained_turn_continuation(
            ego_x_m=5.0,
            ego_y_m=0.5,
            target_speed_mps=5.0,
            count=8,
        )

        self.assertTrue(continuation)
        self.assertLessEqual(len(continuation), 8)
        self.assertTrue(all(
            float(sample["speed_ref_mps"]) == 5.0
            for sample in continuation
        ))
        self.assertGreaterEqual(float(continuation[0]["x_ref_m"]), 5.0)

    def test_retained_turn_continuation_rejects_non_turn_owner(self):
        manager = ManeuverManager()
        manager.update(
            reference_samples=_reference(0.0, 2.0),
            destination_state=[20.0, 0.0, 2.0, 0.0, 2],
            decision="lane_change_right",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            current_lane_id=1,
            target_lane_id=2,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="lane_change",
        )

        self.assertEqual(
            manager.retained_turn_continuation(
                ego_x_m=2.0,
                ego_y_m=0.0,
                target_speed_mps=3.0,
                count=8,
            ),
            [],
        )

    def test_completed_commitment_releases_lane_change_even_if_macro_lags(self):
        manager = ManeuverManager()
        manager.update(
            reference_samples=_reference(),
            destination_state=[20.0, 0.0, 2.0, 0.0, 1],
            decision="lane_change_right",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            current_lane_id=1,
            target_lane_id=1,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="locked_lane_change",
            route_next_maneuver="Lane Change Right",
            lane_change_commitment_active=True,
        )
        released = manager.update(
            reference_samples=_reference(),
            destination_state=[20.0, 0.0, 2.0, 0.0, 1],
            decision="lane_follow",
            behavior_fsm_state="LANE_KEEP",
            current_lane_id=1,
            target_lane_id=1,
            ego_x_m=10.0,
            ego_y_m=0.0,
            reference_source="lane_follow",
            route_next_maneuver="Lane Change Right",
            lane_change_commitment_active=False,
        )

        self.assertFalse(released.debug["maneuver_geometry_active"])
        self.assertEqual(
            released.debug["maneuver_geometry_release_reason"],
            "lane_change_commitment_complete",
        )


class ApplyVelocityProfileTests(unittest.TestCase):
    """`geometry` (the persisted maneuver window) and `incoming` (this
    tick's freshly planned speed profile) are very often different
    lengths. Speed must follow each geometry sample's own
    lane_change_progress, not the array offset it happens to sit at."""

    @staticmethod
    def _geometry(progress_values):
        return [
            {
                "x_ref_m": float(index),
                "y_ref_m": 0.0,
                "lane_change_progress": float(progress),
            }
            for index, progress in enumerate(progress_values)
        ]

    @staticmethod
    def _speed_source(progress_speed_pairs):
        return [
            {"lane_change_progress": float(progress), "speed_ref_mps": float(speed)}
            for progress, speed in progress_speed_pairs
        ]

    def test_matches_by_progress_when_incoming_is_shorter_than_geometry(self):
        geometry = self._geometry([0.0, 0.25, 0.5, 0.75, 1.0])
        incoming = self._speed_source([(0.0, 1.0), (1.0, 9.0)])

        result = ManeuverManager._apply_velocity_profile(geometry, incoming)

        speeds = [float(sample["speed_ref_mps"]) for sample in result]
        # Early-maneuver points (progress 0.25, 0.5) must stay near the
        # early speed (1.0) -- plain index alignment would instead clamp
        # to incoming[-1] (9.0, the terminal speed) starting at index 1.
        self.assertEqual(speeds, [1.0, 1.0, 1.0, 9.0, 9.0])

    def test_falls_back_to_index_alignment_without_progress_tags(self):
        geometry = [
            {"x_ref_m": float(index), "y_ref_m": 0.0} for index in range(4)
        ]
        incoming = [{"speed_ref_mps": 3.0}, {"speed_ref_mps": 6.0}]

        result = ManeuverManager._apply_velocity_profile(geometry, incoming)

        speeds = [float(sample["speed_ref_mps"]) for sample in result]
        self.assertEqual(speeds, [3.0, 6.0, 6.0, 6.0])

    def test_empty_incoming_zeros_speed(self):
        geometry = self._geometry([0.0, 0.5, 1.0])

        result = ManeuverManager._apply_velocity_profile(geometry, [])

        self.assertTrue(all(float(s["speed_ref_mps"]) == 0.0 for s in result))


if __name__ == "__main__":
    unittest.main()
