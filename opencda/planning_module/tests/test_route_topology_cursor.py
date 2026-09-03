import numpy as np

from opencda.planning_module.pipeline.route_manager import RouteCursorSnapshot
from utility.global_planner import CustomGlobalPlannerAdapter


def _planner():
    planner = object.__new__(CustomGlobalPlannerAdapter)
    planner._stored_route_xy = np.asarray([
        [0.0, 0.0],
        [5.0, 0.0],
        [10.0, 0.0],
        [5.0, 0.2],  # spatially nearer, but belongs to another AD lane
        [10.0, 0.2],
    ], dtype=float)
    planner._stored_route_lane_ids = [101, 101, 101, 202, 202]
    planner._stored_route_cum_dists = planner._route_cumulative_distances(
        planner._stored_route_xy
    )
    return planner


def test_route_cursor_cannot_jump_to_spatially_nearer_different_lane():
    planner = _planner()

    index = planner._topology_stored_route_index(
        x_m=5.0,
        y_m=0.19,
        query_key="ego",
        current_lane_id=101,
    )

    assert index == 1


def test_topology_lookup_uses_current_continuous_match_without_own_cursor():
    planner = _planner()

    index = planner._topology_stored_route_index(
        x_m=5.0,
        y_m=0.19,
        query_key="ego",
        current_lane_id=202,
    )

    assert index == 3
    assert not hasattr(planner, "_query_indices")


def test_route_cursor_snapshot_is_an_explicit_immutable_contract():
    cursor = RouteCursorSnapshot(
        route_revision="route-4",
        segment_index=8,
        route_s_m=12.5,
        lane_index=-1,
        current_lane_id=202,
        projection_x_m=4.0,
        projection_y_m=5.0,
        projection_ratio=0.25,
        lateral_distance_m=0.1,
        valid=True,
        reason="route_progress_local_update",
    )

    assert cursor.route_revision == "route-4"
    assert cursor.current_lane_id == 202
