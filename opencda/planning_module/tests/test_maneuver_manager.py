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


if __name__ == "__main__":
    unittest.main()
