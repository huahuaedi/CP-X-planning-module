import math
import unittest

from opencda.planning_module.pipeline.route_manager import CPXRouteManager


class _Location:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Transform:
    def __init__(self, x, y, z=0.0):
        self.location = _Location(x, y, z)


class _Waypoint:
    def __init__(self, x, y, lane_id=1, road_id=1):
        self.transform = _Transform(x, y)
        self.lane_id = int(lane_id)
        self.road_id = int(road_id)
        self.lane_width = 3.5


class RouteManagerCarlaReferenceTest(unittest.TestCase):
    def test_samples_selected_turn_connector_without_duplicate_points(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(2.0, 0.0), "LEFT"),
            (_Waypoint(4.0, 1.0), "LEFT"),
            (_Waypoint(5.0, 3.0), "LEFT"),
            (_Waypoint(5.0, 5.0), "LANEFOLLOW"),
        ]
        manager._carla_route_debug_reason = "carla_grp_route_ready"

        reference, reason = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=12,
            step_distance_m=0.5,
            target_speed_mps=1.2,
            fallback_lane_id=1,
        )

        self.assertEqual(reason, "carla_grp_waypoint_chain")
        self.assertEqual(len(reference), 12)
        self.assertGreater(reference[0]["x_ref_m"], 0.0)
        self.assertTrue(any(sample["y_ref_m"] > 0.5 for sample in reference))
        for previous, current in zip(reference[:-1], reference[1:]):
            distance_m = math.hypot(
                current["x_ref_m"] - previous["x_ref_m"],
                current["y_ref_m"] - previous["y_ref_m"],
            )
            self.assertGreater(distance_m, 1.0e-3)

    def test_route_progress_never_moves_backward(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW")
            for index in range(12)
        ]
        manager._carla_route_debug_reason = "carla_grp_route_ready"

        manager.carla_waypoint_reference(
            ego_x_m=6.2,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=4,
            step_distance_m=0.5,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )
        progressed_index = manager._carla_route_progress_index
        manager.carla_waypoint_reference(
            ego_x_m=3.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=4,
            step_distance_m=0.5,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )

        self.assertGreaterEqual(manager._carla_route_progress_index, progressed_index)

    def test_first_sync_searches_entire_long_route(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW")
            for index in range(300)
        ]
        manager._carla_route_debug_reason = "carla_grp_route_ready"

        reason = manager.sync_carla_route_progress(
            ego_x_m=240.2,
            ego_y_m=0.1,
            ego_heading_rad=0.0,
        )
        reference, reference_reason = manager.carla_waypoint_reference(
            ego_x_m=240.2,
            ego_y_m=0.1,
            ego_heading_rad=0.0,
            horizon_steps=6,
            step_distance_m=0.5,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )

        self.assertIn("carla_route_progress_global_init", reason)
        self.assertGreaterEqual(manager.carla_route_progress_index, 239)
        self.assertEqual(reference_reason, "carla_grp_waypoint_chain")
        self.assertEqual(len(reference), 6)
        self.assertLess(abs(reference[0]["x_ref_m"] - 240.7), 0.2)


if __name__ == "__main__":
    unittest.main()
