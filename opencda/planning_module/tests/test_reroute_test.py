import unittest
from types import SimpleNamespace

from opencda.planning_module.opencda_scenario.loader import (
    list_available_scenarios as list_opencda_scenarios,
    load_carla_scenario as load_opencda_scenario,
)
from opencda.planning_module.opencda_scenario.reroute_test import runner as reroute_test_runner
from opencda.planning_module.opencda_scenario.reroute_test.reroute_test import (
    MarkerWaypointState,
    _plan_initial_route,
    _blocked_segments_for_close_waypoint,
    _camera_height_for_centered_focus_points,
    _map_matches_town10,
    _marker_snapshot_signature,
    _normalize_route_points_with_endpoints,
    _runtime_cfg,
)


class _FakeWaypoint:
    def __init__(self, road_id=20, section_id=0, lane_id=-2, s=10.0, x_m=0.0, y_m=0.0, z_m=0.0):
        self.road_id = road_id
        self.section_id = section_id
        self.lane_id = lane_id
        self.s = s
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x_m), y=float(y_m), z=float(z_m))
        )

    def get_left_lane(self):
        return None

    def get_right_lane(self):
        return None


class _FakePlanner:
    def __init__(
        self,
        raw_segments=None,
        fallback_segments=None,
        route_found=True,
        carla_blocked_edges=None,
    ):
        self.raw_segments = list(raw_segments or [])
        self.fallback_segments = list(fallback_segments or [])
        self.route_found = bool(route_found)
        self.carla_blocked_edges = list(carla_blocked_edges or [])
        self.raw_calls = []
        self.fallback_calls = []
        self.nearest_calls = []
        self.route_calls = []
        self.carla_blocked_edge_calls = []
        self.carla_route_calls = []

    def segment_keys_for_raw_carla_lane(self, **kwargs):
        self.raw_calls.append(dict(kwargs))
        return list(self.raw_segments)

    def segment_keys_for_road_and_lane(self, **kwargs):
        self.fallback_calls.append(dict(kwargs))
        return list(self.fallback_segments)

    def blocked_carla_graph_edges_for_waypoints(self, blocked_waypoints):
        self.carla_blocked_edge_calls.append(list(blocked_waypoints or []))
        return list(self.carla_blocked_edges)

    def nearest_waypoint_query(self, *, x_m, y_m):
        self.nearest_calls.append({"x_m": float(x_m), "y_m": float(y_m)})
        query_index = int(round(float(x_m)))
        return SimpleNamespace(
            index=query_index,
            distance_m=0.0,
            x_m=float(x_m),
            y_m=float(y_m),
            road_id="20:0",
            lane_id=1,
            direction="negative",
        )

    def _route_with_endpoint_candidates(self, **kwargs):
        self.route_calls.append(dict(kwargs))
        return type(
            "RouteSummary",
            (),
            {
                "route_found": bool(self.route_found),
                "route_waypoints": [[1.0, 1.0], [2.0, 2.0]] if self.route_found else [],
                "debug_reason": "blocked close waypoint test",
            },
        )()

    def plan_route_from_locations_with_blocked_carla_waypoints(self, **kwargs):
        self.carla_route_calls.append(dict(kwargs))
        return type(
            "RouteSummary",
            (),
            {
                "route_found": bool(self.route_found),
                "route_waypoints": [[0.0, 0.0], [4.0, 3.5], [10.0, 0.0]] if self.route_found else [],
                "debug_reason": "blocked close carla edge test",
            },
        )()


class RerouteTestHelpers(unittest.TestCase):
    def test_reroute_test_scenario_is_available_via_opencda_loader(self):
        self.assertIn("reroute_test", list_opencda_scenarios())

        scenario_cfg = load_opencda_scenario("reroute_test")

        self.assertEqual(
            str(scenario_cfg.get("runner_module", "")),
            "opencda_scenario.reroute_test.runner",
        )
        self.assertEqual(
            str(scenario_cfg.get("carla", {}).get("map", "")),
            "/Game/Carla/Maps/Town10HD_Opt",
        )
        self.assertFalse(bool(scenario_cfg.get("carla", {}).get("synchronous_mode", True)))
        self.assertTrue(bool(scenario_cfg.get("camera", {}).get("enabled", False)))
        self.assertTrue(callable(getattr(reroute_test_runner, "run_loaded_world", None)))

    def test_runtime_cfg_reads_camera_settings(self):
        runtime_cfg = _runtime_cfg(
            {
                "planning": {"waypoint_sample_distance_m": 3.0},
                "camera": {
                    "enabled": True,
                    "image_size_x": 900,
                    "image_size_y": 700,
                    "fov": 100.0,
                    "topdown": {"height": 42.0},
                },
                "runtime": {
                    "start_marker": "alpha",
                    "end_marker": "omega",
                    "close_marker": "block",
                },
            }
        )

        self.assertEqual(runtime_cfg["start_marker"], "alpha")
        self.assertEqual(runtime_cfg["end_marker"], "omega")
        self.assertEqual(runtime_cfg["close_marker"], "block")
        self.assertEqual(runtime_cfg["sample_distance_m"], 3.0)
        self.assertTrue(runtime_cfg["camera_enabled"])
        self.assertEqual(runtime_cfg["camera_image_width_px"], 900)
        self.assertEqual(runtime_cfg["camera_image_height_px"], 700)
        self.assertEqual(runtime_cfg["camera_fov_deg"], 100.0)
        self.assertEqual(runtime_cfg["camera_height_m"], 42.0)
        self.assertGreater(runtime_cfg["route_debug_life_s"], 100.0)

    def test_normalize_route_points_keeps_start_and_goal(self):
        route_points = _normalize_route_points_with_endpoints(
            route_points=[[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]],
            start_xy=[0.0, 0.0],
            goal_xy=[12.0, 12.0],
        )

        self.assertEqual(route_points[0], [0.0, 0.0])
        self.assertEqual(route_points[-1], [12.0, 12.0])
        self.assertIn([5.0, 5.0], route_points)

    def test_camera_height_grows_with_far_route_points(self):
        camera_height_m = _camera_height_for_centered_focus_points(
            center_xy=[0.0, 0.0],
            focus_points_xy=[[0.0, 0.0], [120.0, 0.0], [0.0, 80.0]],
            image_width_px=840,
            image_height_px=680,
            fov_deg=90.0,
            min_height_m=55.0,
            padding_m=20.0,
        )

        self.assertGreater(camera_height_m, 55.0)

    def test_map_matches_town10_variants(self):
        self.assertTrue(_map_matches_town10("Town10HD_Opt"))
        self.assertTrue(_map_matches_town10("/Game/Carla/Maps/Town10HD_Opt"))
        self.assertTrue(_map_matches_town10("Town10"))
        self.assertFalse(_map_matches_town10("Town06"))

    def test_marker_snapshot_signature_changes_with_marker_position(self):
        first_state = MarkerWaypointState(
            marker_name="start",
            matched_object_name="start",
            marker_position_xy=(10.0, 20.0),
            waypoint_position_xy=(11.0, 21.0),
            road_id=1,
            section_id=0,
            carla_lane_id=-1,
            waypoint_key=(1, 0, -1, 10.0),
            waypoint=None,
        )
        moved_state = MarkerWaypointState(
            marker_name="start",
            matched_object_name="start",
            marker_position_xy=(10.5, 20.0),
            waypoint_position_xy=(11.0, 21.0),
            road_id=1,
            section_id=0,
            carla_lane_id=-1,
            waypoint_key=(1, 0, -1, 10.0),
            waypoint=None,
        )
        self.assertNotEqual(
            _marker_snapshot_signature([first_state]),
            _marker_snapshot_signature([moved_state]),
        )

    def test_blocked_segments_use_raw_carla_lane_first(self):
        planner = _FakePlanner(raw_segments=[("20:0", 1)])
        waypoint = _FakeWaypoint(road_id=20, section_id=0, lane_id=-2, s=30.0)

        blocked_segments = _blocked_segments_for_close_waypoint(planner, waypoint)

        self.assertEqual(blocked_segments, [("20:0", 1)])
        self.assertEqual(
            planner.raw_calls,
            [{"road_id": 20, "section_id": 0, "lane_id": -2}],
        )
        self.assertEqual(planner.fallback_calls, [])

    def test_blocked_segments_fall_back_to_internal_lane(self):
        planner = _FakePlanner(raw_segments=[], fallback_segments=[("20:0", 1)])
        waypoint = _FakeWaypoint(road_id=20, section_id=0, lane_id=-2, s=30.0)

        blocked_segments = _blocked_segments_for_close_waypoint(planner, waypoint)

        self.assertEqual(blocked_segments, [("20:0", 1)])
        self.assertEqual(
            planner.fallback_calls,
            [{"road_id": "20:0", "lane_id": 1}],
        )

    def test_plan_initial_route_prefers_carla_graph_edge_blocking(self):
        planner = _FakePlanner(
            raw_segments=[("20:0", 1)],
            route_found=True,
            carla_blocked_edges=[(7, 8)],
        )
        start_state = MarkerWaypointState(
            marker_name="start",
            matched_object_name="start",
            marker_position_xy=(0.0, 0.0),
            waypoint_position_xy=(0.0, 0.0),
            road_id=20,
            section_id=0,
            carla_lane_id=-2,
            waypoint_key=(20, 0, -2, 0.0),
            waypoint=_FakeWaypoint(road_id=20, section_id=0, lane_id=-2, s=0.0, x_m=0.0, y_m=0.0),
        )
        end_state = MarkerWaypointState(
            marker_name="end",
            matched_object_name="end",
            marker_position_xy=(10.0, 0.0),
            waypoint_position_xy=(10.0, 0.0),
            road_id=20,
            section_id=0,
            carla_lane_id=-2,
            waypoint_key=(20, 0, -2, 10.0),
            waypoint=_FakeWaypoint(road_id=20, section_id=0, lane_id=-2, s=10.0, x_m=10.0, y_m=0.0),
        )
        close_state = MarkerWaypointState(
            marker_name="close",
            matched_object_name="close",
            marker_position_xy=(5.0, 0.0),
            waypoint_position_xy=(5.0, 0.0),
            road_id=20,
            section_id=0,
            carla_lane_id=-2,
            waypoint_key=(20, 0, -2, 5.0),
            waypoint=_FakeWaypoint(road_id=20, section_id=0, lane_id=-2, s=5.0, x_m=5.0, y_m=0.0),
        )

        route_summary, blocked_segments, route_method = _plan_initial_route(
            planner=planner,
            start_state=start_state,
            end_state=end_state,
            close_state=close_state,
        )

        self.assertTrue(bool(route_summary.route_found))
        self.assertEqual(route_method, "blocked_close_carla_edge")
        self.assertEqual(blocked_segments, ["edge:7->8"])
        self.assertEqual(len(planner.carla_blocked_edge_calls), 1)
        self.assertEqual(len(planner.carla_route_calls), 1)
        self.assertEqual(len(planner.nearest_calls), 0)
        self.assertEqual(len(planner.route_calls), 0)


if __name__ == "__main__":
    unittest.main()
