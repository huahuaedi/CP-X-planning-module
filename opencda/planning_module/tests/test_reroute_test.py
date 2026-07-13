import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opencda.planning_module.opencda_scenario.loader import (
    list_available_scenarios as list_opencda_scenarios,
)
from opencda.planning_module.opencda_scenario.reroute_test.reroute_test import (
    MarkerWaypointState,
    _blocked_segments_for_close_waypoint,
    _camera_height_for_centered_focus_points,
    _map_matches_town10,
    _normalize_route_points_with_endpoints,
    _plan_initial_route,
)


class _Planner:
    def __init__(self):
        self.blocked = []
        self.trace_calls = []
        self.route = SimpleNamespace(sampled_waypoints=[])

    def add_blocked_lane(self, lane_id):
        self.blocked.append(int(lane_id))

    def trace_route(self, start, goal, activate_route=True):
        self.trace_calls.append((start, goal, bool(activate_route)))
        return self.route


def _state(name, x, y, waypoint):
    return MarkerWaypointState(
        marker_name=name,
        matched_object_name=name,
        marker_position_xy=(float(x), float(y)),
        waypoint_position_xy=(float(x), float(y)),
        road_id=1,
        section_id=0,
        opendrive_lane_id=-1,
        waypoint_key=(1, 0, int(waypoint.ad_lane_id), float(x)),
        waypoint=waypoint,
    )


class RerouteTestHelpers(unittest.TestCase):
    def test_blocked_segments_use_custom_ad_lane_id(self):
        waypoint = SimpleNamespace(ad_lane_id=77)
        self.assertEqual(
            _blocked_segments_for_close_waypoint(_Planner(), waypoint),
            [("ad_lane", 77)],
        )

    def test_initial_route_blocks_lane_and_uses_custom_trace_route(self):
        planner = _Planner()
        start = _state("start", 0.0, 0.0, SimpleNamespace(ad_lane_id=10))
        end = _state("end", 20.0, 0.0, SimpleNamespace(ad_lane_id=10))
        close = _state("close", 10.0, 0.0, SimpleNamespace(ad_lane_id=99))
        summary = SimpleNamespace(route_found=True)

        with patch(
            "opencda.planning_module.opencda_scenario.reroute_test.reroute_test.route_summary",
            return_value=summary,
        ):
            returned_summary, blocked, source = _plan_initial_route(
                planner=planner,
                start_state=start,
                end_state=end,
                close_state=close,
            )

        self.assertIs(returned_summary, summary)
        self.assertEqual(planner.blocked, [99])
        self.assertEqual(blocked, ["ad_lane:99"])
        self.assertEqual(source, "custom_Global_Planner_blocked_lane")
        self.assertTrue(planner.trace_calls[0][2])

    def test_route_normalization_keeps_start_and_goal(self):
        points = _normalize_route_points_with_endpoints(
            route_points=[[2.0, 0.0], [8.0, 0.0]],
            start_xy=(0.0, 0.0),
            goal_xy=(10.0, 0.0),
        )
        self.assertEqual(points[0], [0.0, 0.0])
        self.assertEqual(points[-1], [10.0, 0.0])

    def test_camera_height_grows_with_route_span(self):
        near = _camera_height_for_centered_focus_points(
            focus_points_xy=[[0.0, 0.0], [10.0, 0.0]],
            center_xy=(5.0, 0.0),
            image_width_px=800,
            image_height_px=600,
            fov_deg=90.0,
            padding_m=10.0,
            min_height_m=20.0,
        )
        far = _camera_height_for_centered_focus_points(
            focus_points_xy=[[0.0, 0.0], [100.0, 0.0]],
            center_xy=(50.0, 0.0),
            image_width_px=800,
            image_height_px=600,
            fov_deg=90.0,
            padding_m=10.0,
            min_height_m=20.0,
        )
        self.assertGreater(far, near)

    def test_map_and_scenario_registration(self):
        self.assertTrue(_map_matches_town10("/Game/Carla/Maps/Town10HD_Opt"))
        self.assertIn("reroute_test", list_opencda_scenarios())


if __name__ == "__main__":
    unittest.main()
