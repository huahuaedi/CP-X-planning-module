"""Regression tests for conservative drivable-lane fast queries."""

from types import SimpleNamespace

from cpx_planning.Global_Planner.global_planner import planner as planner_module


def _planner(*, intersection=False, ambiguous=False):
    planner = object.__new__(planner_module.GlobalPlanner)
    first = (1, 0.0, 1.0, 0.0, 0.0, 10.0, 0.0)
    segments = [first]
    lane_cache = {1: {"is_intersection": bool(intersection)}}
    if ambiguous:
        segments.append((2, 0.0, 1.0, 0.0, 0.2, 10.0, 0.2))
        lane_cache[2] = {"is_intersection": False}
    planner._drivable_segment_grid = {(0, 0): segments}
    planner._drivable_segment_grid_cell_m = 20.0
    planner._drivable_interior_guard_m = 0.10
    planner._drivable_fast_hits = 0
    planner._drivable_exact_fallbacks = 0
    planner.default_search_radius_m = 8.0
    planner._loaded = True
    planner.to_enu = lambda position: tuple(position)
    planner._lane_cache = lane_cache
    planner._make_waypoint_from_lane_offset = (
        lambda lane_id, offset, enu_position=None, **_kwargs: SimpleNamespace(
            ad_lane_id=int(lane_id),
            parametric_offset=float(offset),
            enu_position=enu_position,
        )
    )
    return planner


def test_unique_non_intersection_interior_uses_fast_query(monkeypatch):
    planner = _planner()

    monkeypatch.setattr(
        planner_module.backend,
        "get_lane_width_m",
        lambda lane_id, offset: 4.0,
    )
    monkeypatch.setattr(
        planner_module.backend,
        "sample_lane_center_at_offset",
        lambda lane_id, offset: (10.0 * float(offset), 0.0, 0.0),
    )

    waypoint = planner._fast_interior_drivable_waypoint((5.0, 0.5, 0.0))

    assert waypoint is not None
    assert waypoint.ad_lane_id == 1


def test_intersection_lane_always_falls_back_to_exact_matcher(monkeypatch):
    planner = _planner(intersection=True)
    monkeypatch.setattr(
        planner_module.backend,
        "get_lane_width_m",
        lambda lane_id, offset: 4.0,
    )
    monkeypatch.setattr(
        planner_module.backend,
        "sample_lane_center_at_offset",
        lambda lane_id, offset: (10.0 * float(offset), 0.0, 0.0),
    )

    assert planner._fast_interior_drivable_waypoint((5.0, 0.5, 0.0)) is None


def test_intersection_union_occupancy_uses_verified_fast_query(monkeypatch):
    planner = _planner(intersection=True)
    monkeypatch.setattr(
        planner_module.backend,
        "get_lane_width_m",
        lambda lane_id, offset: 4.0,
    )
    monkeypatch.setattr(
        planner_module.backend,
        "sample_lane_center_at_offset",
        lambda lane_id, offset: (10.0 * float(offset), 0.0, 0.0),
    )
    planner._exact_drivable_waypoint = lambda *args: (
        (_ for _ in ()).throw(AssertionError("unexpected exact lookup"))
    )

    waypoint = planner.get_drivable_occupancy_waypoint((5.0, 0.5, 0.0))

    assert waypoint is not None
    assert waypoint.ad_lane_id == 1
    assert planner._drivable_fast_hits == 1
    assert planner._drivable_exact_fallbacks == 0


def test_overlapping_lane_interiors_fall_back_to_exact_matcher(monkeypatch):
    planner = _planner(ambiguous=True)
    monkeypatch.setattr(
        planner_module.backend,
        "get_lane_width_m",
        lambda lane_id, offset: 4.0,
    )
    monkeypatch.setattr(
        planner_module.backend,
        "sample_lane_center_at_offset",
        lambda lane_id, offset: (
            10.0 * float(offset),
            0.0 if int(lane_id) == 1 else 0.2,
            0.0,
        ),
    )

    assert planner._fast_interior_drivable_waypoint((5.0, 0.1, 0.0)) is None


def test_overlapping_lane_union_accepts_any_verified_interior(monkeypatch):
    planner = _planner(ambiguous=True)
    monkeypatch.setattr(
        planner_module.backend,
        "get_lane_width_m",
        lambda lane_id, offset: 4.0,
    )
    monkeypatch.setattr(
        planner_module.backend,
        "sample_lane_center_at_offset",
        lambda lane_id, offset: (
            10.0 * float(offset),
            0.0 if int(lane_id) == 1 else 0.2,
            0.0,
        ),
    )

    waypoint = planner._fast_interior_drivable_waypoint(
        (5.0, 0.1, 0.0),
        allow_intersection_union=True,
    )

    assert waypoint is not None
    assert waypoint.ad_lane_id in {1, 2}

def test_exact_matcher_uses_segment_candidates_at_intersection(monkeypatch):
    planner = _planner(intersection=True)
    planner._nearby_lane_ids = lambda position, radius: {1, 2, 3}
    captured = {}

    monkeypatch.setattr(
        planner_module.backend,
        "create_enu_point",
        lambda position: tuple(position),
    )
    monkeypatch.setattr(
        planner_module.backend,
        "get_map_matches",
        lambda point, radius, relevant_lane_ids=None: (
            captured.__setitem__(
                "lane_ids", set(relevant_lane_ids or ())
            ),
            [],
        )[1],
    )
    monkeypatch.setattr(
        planner_module.backend,
        "enu_point_to_tuple",
        lambda point: tuple(point),
    )

    assert planner._exact_drivable_waypoint((5.0, 0.5, 0.0), 8.0) is None
    assert captured["lane_ids"] == {1}


def test_exact_matcher_falls_back_when_segment_cell_is_empty(monkeypatch):
    planner = _planner()
    planner._drivable_segment_grid = {}
    planner._nearby_lane_ids = lambda position, radius: {7, 8}
    captured = {}

    monkeypatch.setattr(
        planner_module.backend,
        "create_enu_point",
        lambda position: tuple(position),
    )
    monkeypatch.setattr(
        planner_module.backend,
        "get_map_matches",
        lambda point, radius, relevant_lane_ids=None: (
            captured.__setitem__(
                "lane_ids", set(relevant_lane_ids or ())
            ),
            [],
        )[1],
    )
    monkeypatch.setattr(
        planner_module.backend,
        "enu_point_to_tuple",
        lambda point: tuple(point),
    )

    assert planner._exact_drivable_waypoint((50.0, 50.0, 0.0), 8.0) is None
    assert captured["lane_ids"] == {7, 8}
