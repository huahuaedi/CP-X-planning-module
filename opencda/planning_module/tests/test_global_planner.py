"""AD-map-only tests for the planning-facing global-planner adapter."""

import math
import threading
from collections import OrderedDict
from types import SimpleNamespace

from utility.global_planner import (
    INVALID_LANE_ID,
    CustomGlobalPlannerAdapter,
    lane_step_xy_heading,
)
from Global_Planner.global_planner.planner import GlobalPlanner
from Global_Planner.global_planner import planner as planner_module


class _Waypoint:
    def __init__(self, x_m, y_m, yaw_deg, *, successors=()):
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x_m), y=float(y_m)),
            rotation=SimpleNamespace(yaw=float(yaw_deg)),
        )
        self._successors = list(successors)

    def next(self, distance_m):
        del distance_m
        return list(self._successors)

    def previous(self, distance_m):
        del distance_m
        return []


def test_public_planner_surface_is_admap_only():
    assert INVALID_LANE_ID == 0
    assert CustomGlobalPlannerAdapter.__name__ == "CustomGlobalPlannerAdapter"


def test_longitudinal_neighbors_exclude_non_drivable_admap_links():
    planner = object.__new__(GlobalPlanner)
    planner._lane_cache = {
        10: {
            "forward_lane_ids": [20, 630148],
            "backward_lane_ids": [30, 660153],
        },
        20: {},
        30: {},
    }

    assert planner._get_longitudinal_neighbors(10, True) == [20]
    assert planner._get_longitudinal_neighbors(10, False) == [30]
    assert planner._get_longitudinal_neighbors(630148, True) == []


def test_occupancy_waypoint_skips_unused_lane_boundaries(monkeypatch):
    planner = object.__new__(GlobalPlanner)
    planner._lane_cache = {
        10: {
            "road_id": 1,
            "section_id": 0,
            "lane_id": -1,
            "length_m": 20.0,
            "is_intersection": True,
        },
    }
    boundary_calls = []
    monkeypatch.setattr(
        planner_module.backend,
        "get_lane_boundaries_enu",
        lambda lane_id, offset: boundary_calls.append((lane_id, offset)),
    )
    monkeypatch.setattr(
        planner_module.backend,
        "get_lane_heading",
        lambda lane_id, offset: 0.0,
    )
    monkeypatch.setattr(
        planner_module.backend,
        "get_lane_width_m",
        lambda lane_id, offset: 3.5,
    )

    waypoint = planner._make_waypoint_from_lane_offset(
        10,
        0.5,
        enu_position=(1.0, 2.0, 0.0),
        include_boundaries=False,
    )

    assert waypoint.position == {"x": 1.0, "y": -2.0, "z": 0.0}
    assert waypoint.left_boundary_position is None
    assert waypoint.right_boundary_position is None
    assert boundary_calls == []


def test_lane_step_uses_the_resolved_waypoint_successor():
    successor = _Waypoint(10.0, 2.0, 15.0)
    current = _Waypoint(0.0, 0.0, 0.0, successors=[successor])

    result = lane_step_xy_heading(
        0.0, 0.0, 5.0, get_waypoint_fn=lambda pose: current,
    )

    assert result is not None
    assert result[0] == 10.0
    assert result[1] == 2.0
    assert math.isclose(result[2], math.radians(15.0))


def test_lane_step_returns_none_without_admap_successor():
    current = _Waypoint(0.0, 0.0, 0.0)
    assert lane_step_xy_heading(
        0.0, 0.0, 5.0, get_waypoint_fn=lambda pose: current,
    ) is None
    assert lane_step_xy_heading(
        0.0, 0.0, 5.0, get_waypoint_fn=lambda pose: None,
    ) is None


def _adapter_with_query_core(core):
    adapter = object.__new__(CustomGlobalPlannerAdapter)
    adapter.core = core
    adapter._waypoint_query_lock = threading.Lock()
    adapter._waypoint_query_cache = OrderedDict()
    adapter._drivable_waypoint_cache = OrderedDict()
    adapter._waypoint_candidates_cache = OrderedDict()
    return adapter


def test_static_waypoint_queries_are_reused_until_cache_invalidation():
    waypoint = object()
    core = SimpleNamespace(
        waypoint_calls=0,
        candidate_calls=0,
    )

    def get_waypoint(position, search_radius_m=None):
        del position, search_radius_m
        core.waypoint_calls += 1
        return waypoint

    def get_waypoint_candidates(position, search_radius_m=None):
        del position, search_radius_m
        core.candidate_calls += 1
        return [{"waypoint": waypoint, "is_in_lane": True}]

    core.get_waypoint = get_waypoint
    core.get_waypoint_candidates = get_waypoint_candidates
    adapter = _adapter_with_query_core(core)
    position = {"x": 1.0, "y": 2.0, "z": 0.0}

    assert adapter.get_waypoint(position) is waypoint
    assert adapter.get_waypoint(position) is waypoint
    first = adapter.get_waypoint_candidates(position)
    first[0]["is_in_lane"] = False
    second = adapter.get_waypoint_candidates(position)

    assert core.waypoint_calls == 1
    assert core.candidate_calls == 1
    assert second[0]["is_in_lane"] is True

    adapter._clear_waypoint_query_caches()
    adapter.get_waypoint(position)
    adapter.get_waypoint_candidates(position)
    assert core.waypoint_calls == 2
    assert core.candidate_calls == 2


def test_drivable_waypoint_reuses_cached_candidate_projection():
    waypoint = object()
    core = SimpleNamespace(candidate_calls=0)

    def get_waypoint_candidates(position, search_radius_m=None):
        del position, search_radius_m
        core.candidate_calls += 1
        return [{"waypoint": waypoint, "is_in_lane": True}]

    core.get_waypoint_candidates = get_waypoint_candidates
    adapter = _adapter_with_query_core(core)
    position = (3.0, 4.0, 0.0)

    assert adapter.get_drivable_waypoint(position) is waypoint
    assert adapter.get_drivable_waypoint(position) is waypoint
    assert core.candidate_calls == 1


def test_drivable_waypoint_prefers_direct_core_query():
    waypoint = object()
    core = SimpleNamespace(direct_calls=0)

    def get_drivable_waypoint(position, search_radius_m=None):
        assert position == (3.0, 4.0, 0.0)
        assert search_radius_m == 0.75
        core.direct_calls += 1
        return waypoint

    def get_waypoint_candidates(position, search_radius_m=None):
        del position, search_radius_m
        raise AssertionError("candidate construction must not run")

    core.get_drivable_waypoint = get_drivable_waypoint
    core.get_waypoint_candidates = get_waypoint_candidates
    adapter = _adapter_with_query_core(core)

    assert adapter.get_drivable_waypoint(
        (3.0, 4.0, 0.0), search_radius_m=0.75,
    ) is waypoint
    assert adapter.get_drivable_waypoint(
        (3.0, 4.0, 0.0), search_radius_m=0.75,
    ) is waypoint
    assert core.direct_calls == 1
