import dataclasses

import pytest

from pipeline.local_map_snapshot import build_local_map_snapshot


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
