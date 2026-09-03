import dataclasses

import pytest

from pipeline.local_map_snapshot import audit_local_map_rows, build_local_map_snapshot


def _snapshot(**overrides):
    values = {
        "frame_id": 7,
        "timestamp_s": 12.5,
        "match": {
            "valid": True,
            "ad_lane_id": 10,
            "road_id": 1,
            "section_id": 2,
            "lane_width_m": 3.5,
            "confidence": 0.9,
            "match_reason": "pose_geometry_topology_history",
        },
        "local_graph": {
            "forward_distance_m": 100.0,
            "backward_distance_m": 100.0,
            "corridors": {0: [11, 10], 1: [20], -1: [30]},
            "lane_to_offset": {10: 0, 11: 0, 20: 1, 30: -1},
            "cache_reused": True,
            "generation_reason": "position_within_2m_same_matched_lane",
            "route_lane_sequence": [10, 10, 11],
            "lane_centerlines": {
                10: [
                    {"x_m": 0.0, "y_m": 0.0, "lane_width_m": 3.5},
                    {"x_m": 1.0, "y_m": 0.0, "lane_width_m": 3.5},
                    {"x_m": 2.0, "y_m": 0.2, "lane_width_m": 3.5},
                ],
                11: [
                    {"x_m": 2.0, "y_m": 0.2, "lane_width_m": 3.5},
                    {"x_m": 3.0, "y_m": 0.4, "lane_width_m": 3.5},
                ],
            },
        },
        "route_target_lane_id": 30,
        "invariant_violations": (),
    }
    values.update(overrides)
    return build_local_map_snapshot(**values)


def test_snapshot_is_immutable_and_normalized():
    snapshot = _snapshot()
    assert snapshot.valid
    assert snapshot.ego_lane_id == 10
    assert snapshot.corridor_lane_ids(0) == (10, 11)
    assert snapshot.offset_for_lane(30) == -1
    assert snapshot.route_target_in_frame
    assert snapshot.route_target_offset == -1
    assert snapshot.route_lane_sequence == (10, 11)
    assert snapshot.route_successor_lane(10) == 11
    assert snapshot.route_successor_lane(11) is None
    geometry = snapshot.geometry_for_lane(10)
    assert geometry is not None
    assert geometry.length_m > 2.0
    assert geometry.centerline[0].s_m == 0.0
    assert all(
        second.s_m > first.s_m
        for first, second in zip(
            geometry.centerline, geometry.centerline[1:]
        )
    )
    assert geometry.centerline[-1].heading_rad > 0.0
    assert tuple(
        geometry.lane_id for geometry in snapshot.route_lane_geometries
    ) == (10, 11)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.ego_lane_id = 99
    with pytest.raises(TypeError):
        snapshot.lane_to_offset[30] = 1


def test_snapshot_invalid_when_contract_has_violation():
    snapshot = _snapshot(invariant_violations=["matched_lane_missing_from_current_corridor"])
    assert not snapshot.valid
    assert snapshot.invariant_violations == (
        "matched_lane_missing_from_current_corridor",
    )


def test_legacy_export_returns_detached_mutable_copy():
    snapshot = _snapshot()
    legacy = snapshot.as_legacy_dict()
    legacy["corridors"][0].append(999)
    assert 999 not in snapshot.corridor_lane_ids(0)
    assert legacy["corridor_geometry_lane_ids"][0] == [10, 11]
    assert legacy["route_lane_sequence"] == [10, 11]


def test_route_sequence_requires_local_geometry_contract():
    snapshot = _snapshot(local_graph={
        "corridors": {0: [10, 11]},
        "lane_to_offset": {10: 0, 11: 0},
        "route_lane_sequence": [10, 11],
        "lane_centerlines": {
            10: [{"x_m": 0.0, "y_m": 0.0}, {"x_m": 1.0, "y_m": 0.0}],
        },
    })
    assert not snapshot.valid
    assert "route_lane_centerline_missing" in snapshot.invariant_violations


def test_centerline_geometry_is_deeply_immutable():
    snapshot = _snapshot()
    geometry = snapshot.geometry_for_lane(10)
    assert geometry is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        geometry.centerline[0].x_m = 99.0


def test_admap_boundaries_are_preserved_instead_of_reconstructed_from_width():
    snapshot = _snapshot(local_graph={
        "corridors": {0: [10]},
        "lane_to_offset": {10: 0},
        "lane_centerlines": {
            10: [
                {
                    "x_m": 0.0, "y_m": 0.0, "lane_width_m": 4.0,
                    "left_boundary_x_m": 0.0, "left_boundary_y_m": -1.2,
                    "right_boundary_x_m": 0.0, "right_boundary_y_m": 2.8,
                },
                {
                    "x_m": 1.0, "y_m": 0.0, "lane_width_m": 4.0,
                    "left_boundary_x_m": 1.0, "left_boundary_y_m": -1.2,
                    "right_boundary_x_m": 1.0, "right_boundary_y_m": 2.8,
                },
            ],
        },
    })
    point = snapshot.geometry_for_lane(10).centerline[0]
    assert point.boundary_source == "admap_border"
    assert point.left_boundary_y_m == pytest.approx(-1.2)
    assert point.right_boundary_y_m == pytest.approx(2.8)


def test_missing_admap_boundaries_use_explicit_width_fallback():
    snapshot = _snapshot()
    point = snapshot.geometry_for_lane(10).centerline[0]
    assert point.boundary_source == "centerline_width_fallback"


def test_invalid_or_duplicate_centerline_samples_are_removed():
    snapshot = _snapshot(local_graph={
        "corridors": {0: [10]},
        "lane_to_offset": {10: 0},
        "lane_centerlines": {
            10: [
                {"x_m": 0.0, "y_m": 0.0},
                {"x_m": 0.0, "y_m": 0.0},
                {"x_m": 1.0, "y_m": 0.0},
            ]
        },
    })
    geometry = snapshot.geometry_for_lane(10)
    assert geometry is not None
    assert len(geometry.centerline) == 2
    assert geometry.length_m == pytest.approx(1.0)


def test_corridor_offset_resolves_lane_segment_beside_ego_not_future_target():
    snapshot = _snapshot(
        route_target_lane_id=500144,
        local_graph={
            "corridors": {0: [11640145], -1: [500144, 11640144]},
            "lane_to_offset": {11640145: 0, 500144: -1, 11640144: -1},
            "route_lane_sequence": [11640145, 500144],
            "lane_centerlines": {
                11640145: [
                    {"x_m": 130.0, "y_m": 0.0},
                    {"x_m": 140.0, "y_m": 0.0},
                ],
                11640144: [
                    {"x_m": 130.0, "y_m": 3.5},
                    {"x_m": 140.0, "y_m": 3.5},
                ],
                500144: [
                    {"x_m": 20.0, "y_m": 3.5},
                    {"x_m": 30.0, "y_m": 3.5},
                ],
            },
        },
        match={
            "valid": True,
            "ad_lane_id": 11640145,
            "center_x_m": 132.0,
            "center_y_m": 0.0,
        },
    )

    assert snapshot.route_target_lane_id == 500144
    assert snapshot.route_target_offset == -1
    assert snapshot.lane_at_ego_station(-1) == 11640144


def test_snapshot_removes_duplicate_lane_from_noncanonical_corridor():
    snapshot = _snapshot(local_graph={
        "corridors": {-1: [20], 0: [10], 1: [20]},
        "lane_to_offset": {10: 0, 20: -1},
        "lane_centerlines": {
            10: [{"x_m": 0.0, "y_m": 0.0}, {"x_m": 1.0, "y_m": 0.0}],
            20: [{"x_m": 0.0, "y_m": -3.5}, {"x_m": 1.0, "y_m": -3.5}],
        },
    })

    assert snapshot.corridor_lane_ids(-1) == (20,)
    assert snapshot.corridor_lane_ids(1) == ()
    assert snapshot.offset_for_lane(20) == -1


def test_local_map_audit_accepts_monotonic_continuous_match():
    rows = [
        {
            "local_map_frame_id": index + 1,
            "local_map_valid": True,
            "map_match_valid": True,
            "local_map_ego_lane_id": lane_id,
            "local_map_invariant_violations": "",
            "local_lane_frame_invariant_violations": "",
        }
        for index, lane_id in enumerate([10] * 10 + [11] * 10 + [12] * 10)
    ]
    assert audit_local_map_rows(rows) == ()


def test_local_map_audit_rejects_short_lane_identity_flip_flop():
    lanes = [10] * 10 + [20] * 3 + [10] * 10
    rows = [
        {
            "local_map_frame_id": index + 1,
            "local_map_valid": True,
            "map_match_valid": True,
            "local_map_ego_lane_id": lane_id,
            "local_map_invariant_violations": "",
            "local_lane_frame_invariant_violations": "",
        }
        for index, lane_id in enumerate(lanes)
    ]
    assert "map_match_lane_identity_flip_flop" in audit_local_map_rows(rows)
