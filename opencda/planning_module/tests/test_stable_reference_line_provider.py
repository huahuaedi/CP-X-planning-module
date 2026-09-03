import math

from pipeline.stable_reference_line_provider import StableReferenceLineProvider
from pipeline.local_map_snapshot import build_local_map_snapshot


def _line(count=30, step=1.2):
    return [{"x_ref_m": index * step,
             "y_ref_m": 0.01 * (index * step) ** 2,
             "heading_rad": math.atan(0.02 * index * step),
             "lane_change_progress": index / max(1, count - 1),
             "lane_id": 10 if index < count // 2 else 20}
            for index in range(count)]


def test_window_uses_interpolated_arc_station_not_discrete_master_index():
    window = StableReferenceLineProvider().window_from_reference(
        _line(), ego_x_m=5.55, ego_y_m=0.31, lower_s_m=0.0,
        first_forward_m=0.2, spacing_m=1.1, count=12)
    assert len(window.samples) == 12
    assert window.reason == "arc_length_projection_stitched"
    assert window.samples[0]["reference_global_s_m"] == window.start_s_m
    assert 5.6 < window.start_s_m < 6.1
    assert all(abs((b["reference_global_s_m"] - a["reference_global_s_m"]) - 1.1) < 1e-6
               for a, b in zip(window.samples, window.samples[1:]))


def test_projection_is_monotonic_with_lower_arc_bound():
    provider = StableReferenceLineProvider()
    first = provider.window_from_reference(
        _line(), ego_x_m=8.0, ego_y_m=0.6, lower_s_m=0.0,
        first_forward_m=0.2, spacing_m=1.0, count=10)
    second = provider.window_from_reference(
        _line(), ego_x_m=7.0, ego_y_m=0.5, lower_s_m=first.projection_s_m,
        first_forward_m=0.2, spacing_m=1.0, count=10)
    assert second.projection_s_m >= first.projection_s_m


def test_window_near_master_end_never_repeats_terminal_point():
    window = StableReferenceLineProvider().window_from_reference(
        [
            {"x_ref_m": 0.0, "y_ref_m": 0.0},
            {"x_ref_m": 1.0, "y_ref_m": 0.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0},
        ],
        ego_x_m=1.4,
        ego_y_m=0.0,
        lower_s_m=0.0,
        first_forward_m=0.2,
        spacing_m=0.25,
        count=10,
    )

    assert 1 <= len(window.samples) < 10
    points = [
        (sample["x_ref_m"], sample["y_ref_m"])
        for sample in window.samples
    ]
    assert all(first != second for first, second in zip(points, points[1:]))


def test_projection_cannot_jump_across_future_overlapping_segment():
    # Two nearby/overlapping branches make an unconstrained nearest-point
    # projection jump far ahead in arc length.  A persistent provider must
    # advance only within the local station neighbourhood of the last frame.
    reference = (
        [{"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(11)]
        + [{"x_ref_m": float(x), "y_ref_m": 0.1} for x in range(10, -1, -1)]
    )
    window = StableReferenceLineProvider().window_from_reference(
        reference,
        ego_x_m=1.0,
        ego_y_m=0.1,
        lower_s_m=0.0,
        first_forward_m=0.0,
        spacing_m=0.5,
        count=8,
        max_projection_advance_m=2.0,
    )

    assert window.projection_s_m <= 2.0 + 1.0e-6
    assert window.reason == "arc_length_projection_stitched_bounded"


def _snapshot(second_lane_start_x=2.0):
    return build_local_map_snapshot(
        frame_id=1,
        timestamp_s=1.0,
        match={"valid": True, "ad_lane_id": 10},
        local_graph={
            "corridors": {0: [10, 11]},
            "lane_to_offset": {10: 0, 11: 0},
            "route_lane_sequence": [10, 11],
            "lane_centerlines": {
                10: [
                    {"x_m": 0.0, "y_m": 0.0},
                    {"x_m": 2.0, "y_m": 0.0},
                ],
                11: [
                    {"x_m": second_lane_start_x, "y_m": 0.0},
                    {"x_m": second_lane_start_x + 2.0, "y_m": 1.0},
                ],
            },
        },
    )


def test_reference_master_uses_local_map_topology_order():
    reference, reason = StableReferenceLineProvider().reference_from_local_map(
        _snapshot(), start_lane_id=10, target_speed_mps=3.0
    )
    assert [sample["lane_id"] for sample in reference] == [10, 10, 11]
    assert all(
        sample["reference_geometry_owner"] == "local_map_snapshot"
        for sample in reference
    )
    assert reason == "local_map_snapshot_route:10>11"
    assert [sample["lane_transition_kind"] for sample in reference] == [
        "lane_center", "lane_center", "longitudinal_successor"
    ]


def test_reference_master_does_not_bridge_disconnected_successor():
    reference, reason = StableReferenceLineProvider().reference_from_local_map(
        _snapshot(second_lane_start_x=20.0), start_lane_id=10
    )
    assert len(reference) == 2
    assert {sample["lane_id"] for sample in reference} == {10}
    assert reason == "local_map_snapshot_route:10"


def test_single_lane_window_preserves_offset_to_snapshot_centerline():
    provider = StableReferenceLineProvider()
    reference, reason = provider.reference_for_lane(
        _snapshot(), lane_id=10, target_speed_mps=6.0
    )
    window = provider.window_from_reference(
        reference,
        ego_x_m=0.5,
        ego_y_m=0.4,
        lower_s_m=0.0,
        first_forward_m=0.5,
        spacing_m=0.5,
        count=3,
    )

    assert reason == "local_map_snapshot_lane:10"
    assert len(window.samples) == 3
    assert all(abs(sample["y_ref_m"]) < 1.0e-9 for sample in window.samples)
    assert window.samples[0]["x_ref_m"] == 1.0
    assert window.samples[0]["lane_id"] == 10
    assert window.samples[0]["speed_ref_mps"] == 6.0


def test_lane_chain_joins_physical_adjacent_segment_to_route_successor():
    snapshot = build_local_map_snapshot(
        frame_id=1,
        timestamp_s=1.0,
        match={"valid": True, "ad_lane_id": 10},
        local_graph={
            "corridors": {0: [10], -1: [20, 21]},
            "lane_to_offset": {10: 0, 20: -1, 21: -1},
            "route_lane_sequence": [10, 21],
            "lane_centerlines": {
                10: [{"x_m": 0.0, "y_m": 0.0}, {"x_m": 4.0, "y_m": 0.0}],
                20: [{"x_m": 0.0, "y_m": 3.5}, {"x_m": 2.0, "y_m": 3.5}],
                21: [{"x_m": 2.0, "y_m": 3.5}, {"x_m": 5.0, "y_m": 3.5}],
            },
        },
        route_target_lane_id=21,
    )
    reference, reason = StableReferenceLineProvider().reference_for_lane_chain(
        snapshot, lane_ids=(20, 21), target_speed_mps=8.0)

    assert [sample["lane_id"] for sample in reference] == [20, 20, 21]
    assert all(abs(sample["y_ref_m"] - 3.5) < 1e-9 for sample in reference)
    assert reason == "local_map_snapshot_lane_chain:20>21"


def test_corridor_follows_connected_successor_from_current_segment():
    snapshot = _snapshot()
    reference, reason = StableReferenceLineProvider().reference_for_corridor(
        snapshot, offset=0, start_lane_id=10, target_speed_mps=4.0)
    assert [sample["lane_id"] for sample in reference] == [10, 10, 11]
    assert reason == "local_map_snapshot_lane_chain:10>11"


def test_lane_change_nominal_is_owned_by_provider_and_uses_frenet_progress():
    snapshot = build_local_map_snapshot(
        frame_id=1,
        timestamp_s=1.0,
        match={"valid": True, "ad_lane_id": 10},
        local_graph={
            "corridors": {0: [10], 1: [20]},
            "lane_to_offset": {10: 0, 20: 1},
            "route_lane_sequence": [10],
            "lane_centerlines": {
                10: [{"x_m": float(i), "y_m": 0.0} for i in range(81)],
                20: [{"x_m": float(i), "y_m": -3.5} for i in range(81)],
            },
        },
        route_target_lane_id=20,
    )
    target = [
        {
            "x_ref_m": float(index),
            "y_ref_m": -3.5,
            "heading_rad": 0.0,
            "lane_id": 20,
            "lane_width_m": 3.5,
        }
        for index in range(1, 81)
    ]

    reference, debug = StableReferenceLineProvider().lane_change_nominal(
        snapshot,
        ego_x_m=0.0,
        ego_y_m=0.0,
        ego_heading_rad=0.0,
        current_lane_id=10,
        target_lane_id=20,
        target_reference=target,
        fallback_source_reference=[],
        target_speed_mps=10.0,
        geometry_speed_mps=10.0,
        geometry_length_m=40.0,
        transition_duration_s=4.0,
        spacing_m=1.0,
        horizon_steps=20,
        lane_width_m=3.5,
    )

    assert reference
    assert debug["source_corridor_reason"] == "local_map_snapshot_lane_chain:10"
    assert debug["lateral_offset_m"] < -3.0
    assert reference[0]["lane_change_progress"] == 0.0
    assert reference[-1]["lane_change_progress"] == 1.0
    assert reference[-1]["lane_id"] == 20


def test_lane_change_completion_reference_never_appends_source_route_lane():
    snapshot = build_local_map_snapshot(
        frame_id=1,
        timestamp_s=1.0,
        match={"valid": True, "ad_lane_id": 10},
        local_graph={
            "corridors": {0: [10], 1: [20]},
            "lane_to_offset": {10: 0, 20: 1},
            "route_lane_sequence": [10],
            "lane_centerlines": {
                10: [{"x_m": 0.0, "y_m": 0.0}, {"x_m": 20.0, "y_m": 0.0}],
                20: [{"x_m": 0.0, "y_m": -3.5}, {"x_m": 20.0, "y_m": -3.5}],
            },
        },
        route_target_lane_id=10,
    )

    reference, reason = (
        StableReferenceLineProvider().lane_change_completion_reference(
            snapshot,
            target_lane_id=20,
            target_speed_mps=8.0,
        )
    )

    assert reference
    assert {sample["lane_id"] for sample in reference} == {20}
    assert all(abs(sample["y_ref_m"] + 3.5) < 1.0e-9 for sample in reference)
    assert reason == "local_map_snapshot_lane_chain:20"


def test_target_corridor_excludes_next_lateral_route_edge():
    snapshot = build_local_map_snapshot(
        frame_id=1,
        timestamp_s=1.0,
        match={"valid": True, "ad_lane_id": 10},
        local_graph={
            "corridors": {0: [10, 30], -1: [20, 21]},
            "lane_to_offset": {10: 0, 30: 0, 20: -1, 21: -1},
            "route_lane_sequence": [10, 20, 30],
            "lane_centerlines": {
                10: [{"x_m": 0.0, "y_m": 0.0}, {"x_m": 10.0, "y_m": 0.0}],
                20: [{"x_m": 0.0, "y_m": -3.5}, {"x_m": 10.0, "y_m": -3.5}],
                21: [{"x_m": 10.0, "y_m": -3.5}, {"x_m": 20.0, "y_m": -3.5}],
                30: [{"x_m": 10.0, "y_m": -7.0}, {"x_m": 20.0, "y_m": -7.0}],
            },
        },
        route_target_lane_id=30,
    )

    reference, _ = StableReferenceLineProvider().lane_change_target_reference(
        snapshot,
        target_lane_id=20,
        target_speed_mps=8.0,
    )

    assert {sample["lane_id"] for sample in reference} == {20, 21}
    assert all(abs(sample["y_ref_m"] + 3.5) < 1.0e-9 for sample in reference)
