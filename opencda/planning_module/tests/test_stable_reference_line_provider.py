import math

import pytest

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


def test_coarse_curve_body_anchor_does_not_jump_one_spacing_interval():
    radius_m = 10.0
    angle_step_rad = 0.12  # about 1.2 m between source samples
    reference = [
        {
            "x_ref_m": radius_m * math.cos(index * angle_step_rad),
            "y_ref_m": radius_m * math.sin(index * angle_step_rad),
            "heading_rad": index * angle_step_rad + 0.5 * math.pi,
        }
        for index in range(50)
    ]
    ego_angle_rad = 1.2
    ego_x_m = radius_m * math.cos(ego_angle_rad)
    ego_y_m = radius_m * math.sin(ego_angle_rad)
    # Model a vehicle whose heading trails the local turn tangent by 0.2 rad.
    ego_heading_rad = ego_angle_rad + 0.5 * math.pi - 0.2

    window = StableReferenceLineProvider().window_from_reference(
        reference,
        ego_x_m=ego_x_m,
        ego_y_m=ego_y_m,
        lower_s_m=0.0,
        first_forward_m=0.2,
        spacing_m=1.2,
        count=24,
        ego_heading_rad=ego_heading_rad,
        min_body_forward_m=0.2,
    )

    first = window.samples[0]
    dx_m = float(first["x_ref_m"]) - ego_x_m
    dy_m = float(first["y_ref_m"]) - ego_y_m
    body_forward_m = (
        dx_m * math.cos(ego_heading_rad)
        + dy_m * math.sin(ego_heading_rad)
    )
    assert body_forward_m == pytest.approx(0.2, abs=1.0e-5)
    assert body_forward_m < 0.25
    assert len(window.samples) == 24
    assert "body_forward_anchored" in window.reason


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


def test_reference_master_conditions_lane_join_to_continuous_tangent():
    snapshot = build_local_map_snapshot(
        frame_id=1,
        timestamp_s=1.0,
        match={"valid": True, "ad_lane_id": 10},
        local_graph={
            "corridors": {0: [10, 11]},
            "lane_to_offset": {10: 0, 11: 0},
            "route_lane_sequence": [10, 11],
            "lane_centerlines": {
                10: [
                    {"x_m": float(x), "y_m": 0.0}
                    for x in range(0, 7)
                ],
                11: [
                    {"x_m": 6.0 + float(i), "y_m": 0.45 * float(i)}
                    for i in range(0, 9)
                ],
            },
        },
    )

    reference, _ = StableReferenceLineProvider().reference_from_local_map(
        snapshot,
        start_lane_id=10,
        maximum_join_curvature_1pm=0.20,
    )
    headings = [
        math.atan2(
            float(second["y_ref_m"]) - float(first["y_ref_m"]),
            float(second["x_ref_m"]) - float(first["x_ref_m"]),
        )
        for first, second in zip(reference, reference[1:])
    ]
    jumps = [
        abs(math.atan2(math.sin(b - a), math.cos(b - a)))
        for a, b in zip(headings, headings[1:])
    ]

    assert max(jumps) < math.radians(12.0)
    assert any(
        sample.get("reference_join_conditioning") == "c1_hermite"
        for sample in reference
    )
    assert list(dict.fromkeys(int(sample["lane_id"]) for sample in reference)) == [10, 11]


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


def test_preturn_lane_stops_at_route_topology_boundary_then_extends_tangent():
    snapshot = build_local_map_snapshot(
        frame_id=1,
        timestamp_s=1.0,
        match={"valid": True, "ad_lane_id": 10},
        local_graph={
            "corridors": {0: [10]},
            "lane_to_offset": {10: 0},
            "route_lane_sequence": [10],
            # The map lane contains geometry beyond the route connector.  It
            # bends there, but PREPARE_TURN must not consume that bend before
            # the explicit connector handoff.
            "lane_centerlines": {
                10: [
                    {"x_m": 0.0, "y_m": 0.0},
                    {"x_m": 10.0, "y_m": 0.0},
                    {"x_m": 14.0, "y_m": 4.0},
                ],
            },
        },
    )

    reference, reason = StableReferenceLineProvider().preturn_lane_reference(
        snapshot,
        lane_id=10,
        ego_x_m=0.0,
        ego_y_m=0.0,
        target_speed_mps=6.0,
        first_forward_m=2.0,
        spacing_m=2.0,
        horizon_steps=10,
        topology_forward_limit_m=9.0,
    )

    assert reason == "preturn_lane:local_map_snapshot_lane:10"
    assert len(reference) == 10
    assert all(abs(float(sample["y_ref_m"])) < 1.0e-9 for sample in reference)
    assert all(
        sample.get("lane_transition_kind") == "current_ad_lane_terminal_tangent"
        for sample in reference[4:]
    )


def test_preturn_lane_never_restores_uncropped_reference_at_boundary():
    snapshot = build_local_map_snapshot(
        frame_id=1,
        timestamp_s=1.0,
        match={"valid": True, "ad_lane_id": 10},
        local_graph={
            "corridors": {0: [10]},
            "lane_to_offset": {10: 0},
            "route_lane_sequence": [10],
            "lane_centerlines": {
                10: [
                    {"x_m": 0.0, "y_m": 0.0},
                    {"x_m": 5.0, "y_m": 0.0},
                    {"x_m": 8.0, "y_m": 3.0},
                ],
            },
        },
    )

    reference, reason = StableReferenceLineProvider().preturn_lane_reference(
        snapshot,
        lane_id=10,
        ego_x_m=0.0,
        ego_y_m=0.0,
        target_speed_mps=4.0,
        first_forward_m=2.0,
        spacing_m=2.0,
        horizon_steps=8,
        topology_forward_limit_m=1.0,
    )

    assert reference == []
    assert reason.startswith("preturn_topology_boundary_before_reference_anchor")


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
