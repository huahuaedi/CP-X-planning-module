from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import pytest

from cpx_planning.Global_Planner.global_planner.waypoint import Waypoint
from cpx_planning.utility.global_planner import CustomGlobalPlannerAdapter


def _planner_core(path):
    core = mock.Mock()
    core.xodr_path = Path(path).resolve()
    core.centerline_spacing_m = 2.0
    core.route_engine = "custom"
    core.default_search_radius_m = 8.0
    core.default_lane_change_penalty_m = 2.0
    core.lane_change_distance_m = 8.0
    core.overlap_margin = 0.05
    core.blocked_lanes = []
    return core


def test_adapter_exposes_complete_reusable_map_contract(tmp_path):
    xodr_path = tmp_path / "map_a.xodr"
    xodr_path.write_text("<OpenDRIVE/>", encoding="utf-8")
    core = _planner_core(xodr_path)
    waypoint = object()
    core.get_lane_centerline.return_value = [waypoint]

    with mock.patch(
        "cpx_planning.utility.global_planner.GlobalPlanner",
        return_value=core,
    ):
        adapter = CustomGlobalPlannerAdapter(
            xodr_path=str(xodr_path),
            cache_root=str(tmp_path / "cache"),
        )

    assert adapter.name == "map_a.xodr"
    assert adapter.xodr_path == xodr_path.resolve()
    assert adapter.get_lane_centerline(123) == [waypoint]
    core.get_lane_centerline.assert_called_once_with(123)


def test_dense_route_entries_preserve_waypoint_order_and_options(tmp_path):
    xodr_path = tmp_path / "map_a.xodr"
    xodr_path.write_text("<OpenDRIVE/>", encoding="utf-8")
    core = _planner_core(xodr_path)

    with mock.patch(
        "cpx_planning.utility.global_planner.GlobalPlanner",
        return_value=core,
    ):
        adapter = CustomGlobalPlannerAdapter(
            xodr_path=str(xodr_path),
            cache_root=str(tmp_path / "cache"),
        )

    first = object()
    second = object()
    adapter._stored_route_waypoints = [first, second]
    adapter._stored_route_options = ["LANEFOLLOW", "RIGHT"]

    assert adapter.get_dense_route_entries() == [
        {"waypoint": first, "road_option": "LANEFOLLOW"},
        {"waypoint": second, "road_option": "RIGHT"},
    ]


def test_admap_waypoint_exposes_shared_route_geometry_contract():
    waypoint = Waypoint(
        position={"x": 12.5, "y": -7.25, "z": 0.3},
        enu_position=(12.5, 7.25, 0.3),
        ad_lane_id=123,
        road_id=4,
        section_id=0,
        lane_id=-1,
        parametric_offset=0.5,
        heading=0.25,
        lane_length_m=50.0,
        lane_width_m=3.5,
        is_intersection=True,
        _planner=mock.Mock(),
    )

    assert waypoint.transform.location.x == pytest.approx(12.5)
    assert waypoint.transform.location.y == pytest.approx(-7.25)
    assert waypoint.transform.location.z == pytest.approx(0.3)
    assert waypoint.transform.rotation.yaw == pytest.approx(
        -0.25 * 180.0 / 3.141592653589793
    )
    assert waypoint.is_junction is True


def test_carla_world_positive_heading_change_is_right_turn():
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.2],
        [5.0, 0.8],
        [6.0, 1.8],
        [6.5, 3.0],
        [6.7, 4.0],
    ]

    options = CustomGlobalPlannerAdapter._geometric_road_options(points)

    assert "RIGHT" in options
    assert "LEFT" not in options


def test_non_intersection_curve_is_not_promoted_to_junction_turn():
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.2],
        [5.0, 0.8],
        [6.0, 1.8],
        [6.5, 3.0],
        [6.7, 4.0],
    ]
    waypoints = [
        SimpleNamespace(is_intersection=False, ad_lane_id=1)
        for _ in points
    ]

    options = CustomGlobalPlannerAdapter._lane_aware_road_options(
        points, waypoints
    )

    assert set(options) == {"LANEFOLLOW"}


def test_activate_map_replaces_owner_and_clears_lane_id_state(tmp_path):
    first_path = tmp_path / "map_a.xodr"
    second_path = tmp_path / "map_b.xodr"
    first_path.write_text("<OpenDRIVE/>", encoding="utf-8")
    second_path.write_text("<OpenDRIVE/>", encoding="utf-8")
    first_core = _planner_core(first_path)
    second_core = _planner_core(second_path)

    with mock.patch(
        "cpx_planning.utility.global_planner.GlobalPlanner",
        side_effect=[first_core, second_core],
    ):
        adapter = CustomGlobalPlannerAdapter(
            xodr_path=str(first_path),
            cache_root=str(tmp_path / "cache"),
        )
        adapter._stored_route_summary = object()
        adapter._stored_route_xy = object()
        adapter._stored_route_waypoints = [object()]
        adapter._lane_context_cache = (1.0, 2.0, 0.0, {})

        changed = adapter.activate_map(str(second_path))

    assert changed is True
    assert adapter.core is second_core
    assert adapter.xodr_path == second_path.resolve()
    second_core.load.assert_called_once_with(force_rebuild=False)
    assert adapter._stored_route_summary is None
    assert adapter._stored_route_xy is None
    assert adapter._stored_route_waypoints == []
    assert adapter._lane_context_cache is None


def test_activate_same_map_reuses_existing_store(tmp_path):
    xodr_path = tmp_path / "map_a.xodr"
    xodr_path.write_text("<OpenDRIVE/>", encoding="utf-8")
    core = _planner_core(xodr_path)

    with mock.patch(
        "cpx_planning.utility.global_planner.GlobalPlanner",
        return_value=core,
    ):
        adapter = CustomGlobalPlannerAdapter(
            xodr_path=str(xodr_path),
            cache_root=str(tmp_path / "cache"),
        )
        changed = adapter.activate_map(str(xodr_path))

    assert changed is False
    core.load.assert_not_called()


def test_reset_dynamic_state_keeps_map_but_clears_run_owned_state(tmp_path):
    xodr_path = tmp_path / "map_a.xodr"
    xodr_path.write_text("<OpenDRIVE/>", encoding="utf-8")
    core = _planner_core(xodr_path)

    with mock.patch(
        "cpx_planning.utility.global_planner.GlobalPlanner",
        return_value=core,
    ):
        adapter = CustomGlobalPlannerAdapter(
            xodr_path=str(xodr_path),
            cache_root=str(tmp_path / "cache"),
        )

    core.blocked_lanes.extend([630148, 630149])
    adapter._stored_route_summary = object()
    adapter._stored_route_waypoints = [object()]
    adapter._query_indices["route"] = 4
    adapter._lane_context_cache = (1.0, 2.0, 0.0, {})
    adapter._waypoint_query_cache["point"] = object()

    adapter.reset_dynamic_state()

    assert core.blocked_lanes == []
    assert adapter._stored_route_summary is None
    assert adapter._stored_route_waypoints == []
    assert adapter._query_indices == {}
    assert adapter._lane_context_cache is None
    assert adapter._waypoint_query_cache == {}
    core.load.assert_not_called()


def test_route_contexts_share_map_but_isolate_stored_routes(tmp_path):
    xodr_path = tmp_path / "map_a.xodr"
    xodr_path.write_text("<OpenDRIVE/>", encoding="utf-8")
    core = _planner_core(xodr_path)

    with mock.patch(
        "cpx_planning.utility.global_planner.GlobalPlanner",
        return_value=core,
    ):
        owner = CustomGlobalPlannerAdapter(
            xodr_path=str(xodr_path),
            cache_root=str(tmp_path / "cache"),
        )

    first = owner.fork_route_context()
    second = owner.fork_route_context()
    first._stored_route_waypoints = ["cav-1-route"]
    first._query_indices["route"] = 12
    second._stored_route_waypoints = ["cav-2-route"]
    second._query_indices["route"] = 3

    assert first.core is owner.core
    assert second.core is owner.core
    assert first._stored_route_waypoints == ["cav-1-route"]
    assert second._stored_route_waypoints == ["cav-2-route"]
    assert owner._stored_route_waypoints == []
    assert first._query_indices == {"route": 12}
    assert second._query_indices == {"route": 3}

    first.close()
    core.close.assert_not_called()
    owner.close()
    core.close.assert_called_once_with()
