"""Contracts for the AD-map-only Town10 reroute diagnostic."""

from types import SimpleNamespace

from opencda.planning_module.opencda_scenario.loader import (
    list_available_scenarios,
    load_carla_scenario,
)
from opencda.planning_module.opencda_scenario.reroute_test import runner
from opencda.planning_module.opencda_scenario.reroute_test.reroute_test import (
    MarkerWaypointState,
    _camera_height_for_centered_focus_points,
    _map_matches_town10,
    _normalize_route_points_with_endpoints,
    _plan_initial_route,
    _runtime_cfg,
)


class _Planner:
    def __init__(self):
        self.blocked_lane_ids = []
        self.trace_requests = []

    def block_ad_lane_id(self, lane_id):
        self.blocked_lane_ids.append(int(lane_id))

    def trace_route(self, start, goal, replace_stored_route=False):
        self.trace_requests.append((dict(start), dict(goal), replace_stored_route))
        return SimpleNamespace(route_found=True, route_waypoints=[[0.0, 0.0]])


def _marker(name, x_m, ad_lane_id):
    waypoint = SimpleNamespace(position={"x": float(x_m), "y": 0.0, "z": 0.3})
    return MarkerWaypointState(
        marker_name=name,
        matched_object_name=name,
        marker_position_xy=(float(x_m), 0.0),
        waypoint_position_xy=(float(x_m), 0.0),
        road_id=20,
        section_id=0,
        opendrive_lane_id=-2,
        ad_lane_id=int(ad_lane_id),
        waypoint_key=(20, 0, -2, float(x_m)),
        waypoint=waypoint,
    )


def test_reroute_scenario_is_loadable():
    assert "reroute_test" in list_available_scenarios()
    config = load_carla_scenario("reroute_test")
    assert config["runner_module"] == "opencda_scenario.reroute_test.runner"
    assert config["carla"]["map"] == "/Game/Carla/Maps/Town10HD_Opt"
    assert callable(runner.run_loaded_world)


def test_runtime_config_contains_only_runtime_and_camera_contracts():
    result = _runtime_cfg({
        "planning": {"waypoint_sample_distance_m": 3.0},
        "runtime": {"start_marker": "a", "end_marker": "b", "close_marker": "c"},
        "camera": {"enabled": False},
    })
    assert result["sample_distance_m"] == 3.0
    assert (result["start_marker"], result["end_marker"], result["close_marker"]) == (
        "a", "b", "c",
    )
    assert result["camera_enabled"] is False
    assert "planner_mode" not in result


def test_initial_route_blocks_one_ad_lane_then_traces():
    planner = _Planner()
    route, blocked, method = _plan_initial_route(
        planner=planner,
        start_state=_marker("start", 0.0, 101),
        end_state=_marker("end", 20.0, 103),
        close_state=_marker("close", 10.0, 102),
    )
    assert route.route_found
    assert planner.blocked_lane_ids == [102]
    assert blocked == ["ad_lane:102"]
    assert method == "blocked_close_ad_lane"
    assert planner.trace_requests[0][2] is True


def test_route_points_keep_explicit_endpoints():
    result = _normalize_route_points_with_endpoints(
        route_points=[[0.0, 0.0], [5.0, 1.0]],
        start_xy=[-1.0, 0.0],
        goal_xy=[10.0, 0.0],
    )
    assert result[0] == [-1.0, 0.0]
    assert result[-1] == [10.0, 0.0]


def test_town10_name_and_camera_extent_helpers():
    assert _map_matches_town10("/Game/Carla/Maps/Town10HD_Opt")
    assert not _map_matches_town10("Town06")
    height = _camera_height_for_centered_focus_points(
        center_xy=[0.0, 0.0],
        focus_points_xy=[[120.0, 0.0], [0.0, 80.0]],
        image_width_px=840,
        image_height_px=680,
        fov_deg=90.0,
        min_height_m=55.0,
        padding_m=20.0,
    )
    assert height > 55.0
