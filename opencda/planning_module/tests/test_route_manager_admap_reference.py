import math
import types
import unittest

from opencda.planning_module.pipeline.route_manager import CPXRouteManager


class _Location:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Transform:
    def __init__(self, x, y, z=0.0, yaw=0.0):
        self.location = _Location(x, y, z)
        self.rotation = types.SimpleNamespace(yaw=float(yaw))


class _ADMapWaypoint:
    def __init__(self, x, y, lane_id=1, lane_width=3.5):
        self.transform = _Transform(x, y)
        self.ad_lane_id = int(lane_id)
        self.lane_id = int(lane_id)
        self.lane_width = float(lane_width)


class _ADMapPlanner:
    def __init__(self, entries):
        self.entries = list(entries)
        self.plan_calls = []

    def plan_route_from_locations(self, **kwargs):
        self.plan_calls.append(dict(kwargs))
        points = [
            (
                item["waypoint"].transform.location.x,
                item["waypoint"].transform.location.y,
                item["waypoint"].transform.location.z,
            )
            for item in self.entries
        ]
        return types.SimpleNamespace(
            route_found=len(points) >= 2,
            route_waypoints=points,
            road_options=[item["road_option"] for item in self.entries],
            distance_to_destination_m=max(0.0, float(len(points) - 1)),
            debug_reason="admap_route_ready",
        )

    def get_dense_route_entries(self):
        return list(self.entries)


def _planner(points_and_options):
    return _ADMapPlanner([
        {
            "waypoint": _ADMapWaypoint(x, y, lane_id),
            "road_option": option,
        }
        for x, y, lane_id, option in points_and_options
    ])


class RouteManagerADMapReferenceTest(unittest.TestCase):
    def test_destination_installs_only_admap_dense_route(self):
        planner = _planner([
            (0.0, 0.0, 1, "LANEFOLLOW"),
            (2.0, 0.0, 1, "LANEFOLLOW"),
            (4.0, 1.0, 2, "CHANGELANELEFT"),
        ])
        manager = CPXRouteManager(global_planner=planner)

        summary = manager.set_destination(
            start_point={"x": 0.0, "y": 0.0},
            goal_point={"x": 4.0, "y": 1.0},
        )

        self.assertTrue(summary.route_found)
        self.assertEqual(len(planner.plan_calls), 1)
        self.assertEqual(len(manager._route_nodes()), 3)
        self.assertEqual(manager.route_debug_reason, "admap_route_ready")

    def test_route_reference_samples_admap_waypoints(self):
        planner = _planner([
            (0.0, 0.0, 1, "LANEFOLLOW"),
            (1.0, 0.0, 1, "LANEFOLLOW"),
            (2.0, 0.2, 1, "RIGHT"),
            (3.0, 0.8, 1, "RIGHT"),
        ])
        manager = CPXRouteManager(global_planner=planner)
        manager.set_destination(
            start_point={"x": 0.0, "y": 0.0},
            goal_point={"x": 3.0, "y": 0.8},
        )

        reference, reason = manager.route_reference(
            ego_x_m=0.1,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            target_speed_mps=2.0,
            fallback_lane_id=1,
            horizon_steps=4,
            step_distance_m=0.8,
        )

        self.assertTrue(reference, reason)
        self.assertTrue(all(sample["lane_id"] == 1 for sample in reference))
        self.assertGreater(reference[-1]["x_ref_m"], reference[0]["x_ref_m"])

    def test_route_progress_is_monotonic_on_admap_route(self):
        planner = _planner([
            (float(index), 0.0, 1, "LANEFOLLOW")
            for index in range(20)
        ])
        manager = CPXRouteManager(global_planner=planner)
        manager.set_destination(
            start_point={"x": 0.0, "y": 0.0},
            goal_point={"x": 19.0, "y": 0.0},
        )

        manager.sync_route_progress(ego_x_m=12.2, ego_y_m=0.0, ego_heading_rad=0.0)
        progressed = manager.route_progress_index
        manager.sync_route_progress(ego_x_m=3.0, ego_y_m=0.0, ego_heading_rad=0.0)

        self.assertGreaterEqual(manager.route_progress_index, progressed)

    def test_route_arc_progress_and_lane_change_distance_are_monotonic(self):
        points = []
        for index in range(24):
            option = (
                "CHANGELANERIGHT"
                if 10 <= index <= 11
                else "RIGHT"
                if 18 <= index <= 20
                else "LANEFOLLOW"
            )
            lane_id = 500144 if index >= 12 else 500145
            points.append((float(index), 0.0, lane_id, option))
        manager = CPXRouteManager(global_planner=_planner(points))
        manager.set_destination(
            start_point={"x": 0.0, "y": 0.0},
            goal_point={"x": 23.0, "y": 0.0},
        )

        progress_samples = []
        distance_samples = []
        for ego_x_m in (1.0, 3.5, 6.0, 8.0):
            manager.sync_route_progress(
                ego_x_m=ego_x_m,
                ego_y_m=0.0,
                ego_heading_rad=0.0,
            )
            progress_samples.append(manager.route_progress_s_m)
            direction, distance_m, _ = manager.upcoming_lane_change(
                ego_x_m=ego_x_m,
                ego_y_m=0.0,
                ego_heading_rad=0.0,
                lookahead_m=50.0,
            )
            self.assertEqual(direction, "right")
            distance_samples.append(distance_m)

        self.assertEqual(progress_samples, sorted(progress_samples))
        self.assertEqual(distance_samples, sorted(distance_samples, reverse=True))
        self.assertTrue(manager.route_topology_validation.valid)
        self.assertEqual(
            manager.route_topology_validation.signature,
            (
                "lane_follow",
                "lane_change:right",
                "lane_follow",
                "junction_turn:right",
                "lane_follow",
            ),
        )

        direction, _, reason = manager.upcoming_lane_change(
            ego_x_m=13.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            lookahead_m=50.0,
        )
        self.assertEqual(direction, "")
        self.assertEqual(reason, "route_geometry_no_lane_change_in_lookahead")

    def test_replan_replaces_route_with_new_admap_entries(self):
        planner = _planner([
            (0.0, 0.0, 1, "LANEFOLLOW"),
            (3.0, 0.0, 1, "LANEFOLLOW"),
        ])
        manager = CPXRouteManager(global_planner=planner)
        manager.set_destination(
            start_point={"x": 0.0, "y": 0.0},
            goal_point={"x": 3.0, "y": 0.0},
        )
        planner.entries = [
            {"waypoint": _ADMapWaypoint(1.0, 0.0, 1), "road_option": "LANEFOLLOW"},
            {"waypoint": _ADMapWaypoint(2.0, 1.0, 2), "road_option": "CHANGELANELEFT"},
            {"waypoint": _ADMapWaypoint(3.0, 1.0, 2), "road_option": "LANEFOLLOW"},
        ]

        result = manager.replan_from(
            start_point={"x": 1.0, "y": 0.0},
            trigger_reason="static_obstacle",
        )

        self.assertTrue(result.success, result.reason)
        self.assertEqual(len(manager._route_nodes()), 3)
        self.assertIn("blocked_summary_route_replanned", result.reason)

    def test_turn_replan_rejects_long_macro_changing_disconnected_route(self):
        planner = _planner([
            (0.0, 0.0, 10, "LANEFOLLOW"),
            (10.0, 0.0, 10, "LANEFOLLOW"),
            (20.0, -5.0, 20, "RIGHT"),
            (20.0, -15.0, 20, "RIGHT"),
        ])
        manager = CPXRouteManager(global_planner=planner)
        manager.set_destination(
            start_point={"x": 0.0, "y": 0.0},
            goal_point={"x": 20.0, "y": -15.0},
        )
        old_nodes = manager._route_nodes()
        planner.entries = [
            {"waypoint": _ADMapWaypoint(0.0, 0.0, 99), "road_option": "LANEFOLLOW"},
            {"waypoint": _ADMapWaypoint(-100.0, 0.0, 99), "road_option": "CHANGELANELEFT"},
            {"waypoint": _ADMapWaypoint(-200.0, 0.0, 98), "road_option": "LEFT"},
            {"waypoint": _ADMapWaypoint(-300.0, 0.0, 98), "road_option": "LANEFOLLOW"},
        ]

        result = manager.replan_from(
            start_point={"x": 0.0, "y": 0.0},
            trigger_reason="turn_reference_unavailable",
        )

        self.assertFalse(result.success)
        self.assertIn("turn_replan_rejected", result.reason)
        self.assertEqual(
            [(node[0], node[1], node[3].ad_lane_id) for node in manager._route_nodes()],
            [(node[0], node[1], node[3].ad_lane_id) for node in old_nodes],
        )

    def test_alignment_uses_admap_route_geometry(self):
        planner = _planner([
            (0.0, 0.0, 1, "LANEFOLLOW"),
            (5.0, 0.0, 1, "LANEFOLLOW"),
        ])
        manager = CPXRouteManager(global_planner=planner)
        manager.set_destination(
            start_point={"x": 0.0, "y": 0.0},
            goal_point={"x": 5.0, "y": 0.0},
        )

        heading_error, lateral_m, reason = manager.route_alignment(
            ego_x_m=1.0,
            ego_y_m=0.5,
            ego_heading_rad=0.0,
        )

        self.assertAlmostEqual(heading_error, 0.0, places=6)
        self.assertAlmostEqual(lateral_m, 0.5, places=6)
        self.assertIn("route_alignment", reason)


if __name__ == "__main__":
    unittest.main()
