import pytest
from types import SimpleNamespace

from pipeline.behavior_decision import BehaviorDecision

from pipeline.reference_line_provider import (
    CONNECTOR,
    LANE_CHANGE,
    LANE_FOLLOW,
    POST_TURN,
    TURN,
    ReferenceLineProvider,
    ReferenceLineRequest,
    TurnReferenceRequest,
)
from pipeline.stable_reference_line_provider import StableReferenceLineProvider


def _line(y_m=0.0):
    return [
        {"x_ref_m": float(index), "y_ref_m": float(y_m)}
        for index in range(30)
    ]


def _local_geometry(lane_id, points):
    centerline = tuple(
        SimpleNamespace(
            x_m=float(x_m), y_m=float(y_m), heading_rad=0.0,
            curvature_1pm=0.0, lane_width_m=3.5,
            left_boundary_x_m=float(x_m), left_boundary_y_m=float(y_m + 1.75),
            right_boundary_x_m=float(x_m), right_boundary_y_m=float(y_m - 1.75),
            boundary_source="admap_border",
        )
        for x_m, y_m in points
    )
    return SimpleNamespace(lane_id=int(lane_id), centerline=centerline)


def test_local_route_reference_prefixes_matched_lane_before_connector():
    geometries = {
        10: _local_geometry(10, [(0.0, 0.0), (1.0, 0.0)]),
        20: _local_geometry(20, [(1.0, 0.0), (2.0, 1.0)]),
        30: _local_geometry(30, [(2.0, 1.0), (2.0, 2.0)]),
    }
    snapshot = SimpleNamespace(
        valid=True,
        # RouteCursor has entered the connector while the continuous matcher
        # still owns the incoming lane beneath ego.
        route_lane_sequence=(20, 30),
        geometry_for_lane=lambda lane_id: geometries.get(int(lane_id)),
    )

    samples, reason = StableReferenceLineProvider().reference_from_local_map(
        snapshot,
        start_lane_id=10,
        target_speed_mps=2.0,
        maximum_join_distance_m=2.0,
    )

    assert [10, 20, 30] == list(dict.fromkeys(
        int(sample["lane_id"]) for sample in samples
    ))
    assert reason == "local_map_snapshot_route:10>20>30"


def test_turn_reference_keeps_valid_master_until_local_map_revision_catches_up():
    provider = ReferenceLineProvider()
    provider.attach_builder(SimpleNamespace(
        curvature_feasible_turn_samples=lambda reference_samples, **_kwargs: (
            list(reference_samples), ""
        )
    ))
    provider.install(
        TURN,
        [
            {
                "x_ref_m": float(index), "y_ref_m": 0.0,
                "heading_rad": 0.0, "lane_id": 10,
            }
            for index in range(20)
        ],
        route_revision="route-1",
        map_epoch="admap",
        event="maneuver_started",
        source_lane_id=10,
        target_lane_id=20,
        maneuver_direction="left",
    )
    stale_local_map = SimpleNamespace(
        valid=True,
        route_revision="route-1",
        route_lane_sequence=(20,),
        geometry_for_lane=lambda _lane_id: None,
    )

    samples, _destination, reason = provider.turn_reference(
        TurnReferenceRequest(
            local_map=stale_local_map,
            config={},
            horizon_steps=8,
            dt_s=0.1,
            ego_location=SimpleNamespace(x=2.0, y=0.0),
            ego_yaw_rad=0.0,
            current_state=[2.0, 0.0, 2.0, 0.0],
            current_lane_id=10,
            target_lane_id=20,
            target_speed_mps=2.0,
            lock_master=True,
            turn_direction="left",
            route_revision="route-2",
            map_epoch="admap",
        )
    )

    assert samples
    assert provider.snapshot(TURN).route_revision == "route-1"
    assert "turn_local_map_route_revision_mismatch:route-1!=route-2" in reason
    assert "turn_master_window" in reason


@pytest.mark.parametrize(
    "mode", [LANE_FOLLOW, LANE_CHANGE, CONNECTOR, TURN, POST_TURN]
)
def test_all_modes_use_same_install_window_release_contract(mode):
    provider = ReferenceLineProvider()
    installed, _ = provider.install(
        mode,
        _line(),
        route_revision="route-1",
        map_epoch="town06",
        event="maneuver_started" if mode != LANE_FOLLOW else "initial_route",
        source_lane_id=10,
        target_lane_id=11,
        ego_x_m=5.0,
        ego_y_m=0.2,
    )
    assert installed
    initial = provider.snapshot(mode)
    assert initial.active
    assert initial.activation_s_m == pytest.approx(5.0)

    window = provider.window(
        mode,
        ego_x_m=7.0,
        ego_y_m=0.2,
        first_forward_m=0.5,
        spacing_m=0.5,
        count=10,
        max_projection_advance_m=3.0,
    )
    assert len(window.samples) == 10
    assert provider.snapshot(mode).progress_s_m == pytest.approx(7.0)
    assert provider.snapshot(mode).travelled_s_m == pytest.approx(2.0)

    released, _ = provider.release(mode, event="phase_transition")
    assert released
    assert not provider.snapshot(mode).active


def test_contract_failure_records_diagnostic_without_rebuilding_geometry():
    provider = ReferenceLineProvider()
    provider.install(
        TURN,
        _line(),
        route_revision="route-1",
        map_epoch="town06",
        event="maneuver_started",
    )
    revision = provider.snapshot(TURN).geometry_revision
    samples = provider.snapshot(TURN).samples

    installed, reason = provider.install(
        TURN,
        _line(4.0),
        route_revision="route-1",
        map_epoch="town06",
        event="contract_failure",
    )
    provider.mark_validation_failure(TURN, "curvature")

    assert not installed
    assert reason == "reference_install_event_forbidden:contract_failure"
    assert provider.snapshot(TURN).geometry_revision == revision
    assert provider.snapshot(TURN).samples == samples
    assert provider.snapshot(TURN).last_validation_failure == "curvature"


def test_window_anchor_is_measured_in_ego_body_frame():
    provider = ReferenceLineProvider()
    provider.install(
        LANE_CHANGE,
        _line(),
        route_revision="route-1",
        map_epoch="town06",
        event="maneuver_started",
        source_lane_id=10,
        target_lane_id=11,
        ego_x_m=5.0,
        ego_y_m=0.0,
    )

    # With a 60 degree ego heading, 0.25 m of master arc is only 0.125 m
    # forward in the body frame.  The provider must advance to the next
    # sample instead of delivering a contract-invalid first point.
    window = provider.window(
        LANE_CHANGE,
        ego_x_m=7.0,
        ego_y_m=0.0,
        ego_heading_rad=1.0471975512,
        first_forward_m=0.25,
        min_body_forward_m=0.2,
        spacing_m=0.4,
        count=10,
    )

    first = window.samples[0]
    forward_m = (float(first["x_ref_m"]) - 7.0) * 0.5
    assert forward_m >= 0.2
    assert "body_forward_anchored" in window.reason


def test_locked_lane_change_window_owns_padding_and_speed_tags():
    provider = ReferenceLineProvider()
    provider.install(
        LANE_CHANGE,
        _line()[:5],
        route_revision="route-1",
        map_epoch="town06",
        event="maneuver_started",
        source_lane_id=10,
        target_lane_id=11,
        ego_x_m=2.0,
        ego_y_m=0.0,
    )

    result = provider.locked_lane_change_window(
        ego_x_m=3.0,
        ego_y_m=0.0,
        ego_heading_rad=0.0,
        target_lane_id=11,
        target_speed_mps=6.0,
        spacing_m=0.5,
        count=8,
        min_first_forward_m=0.2,
    )

    assert len(result.samples) == 8
    assert result.samples[-1]["lane_id"] == 11
    assert all(row["speed_ref_mps"] == 6.0 for row in result.samples)


def test_provider_owns_transient_lane_fallback_generation():
    calls = []
    generated = SimpleNamespace(samples=_line(), destination_state=[1, 2, 3, 4])
    builder = SimpleNamespace(
        build_lane_fallback=lambda **kwargs: calls.append(kwargs) or generated
    )
    provider = ReferenceLineProvider()
    provider.attach_builder(builder)

    result = provider.lane_fallback_reference(
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.1,
        current_state=[0.0, 0.0, 2.0, 0.1],
        speed_ref_mps=3.0,
    )

    assert result is generated
    assert calls[0]["speed_ref_mps"] == pytest.approx(3.0)
    assert not provider.snapshot(LANE_FOLLOW).active


def test_provider_owns_boundary_recovery_conditioning_and_speed_tags():
    generated = SimpleNamespace(samples=_line(), destination_state=[], reason="built")
    builder = SimpleNamespace(
        build_boundary_recovery=lambda **kwargs: generated,
        curvature_feasible_samples=lambda **kwargs: (
            list(kwargs["reference_samples"]),
            "conditioned",
        ),
    )
    provider = ReferenceLineProvider()
    provider.attach_builder(builder)

    result, samples, reason = provider.boundary_recovery_reference(
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.0,
        current_lane_id=7,
        base_reference_samples=_line(),
        target_speed_mps=2.5,
        horizon_steps=10,
        dt_s=0.1,
        max_curvature_1pm=0.2,
    )

    assert result is generated
    assert reason == "conditioned"
    assert samples
    assert all(row["speed_ref_mps"] == pytest.approx(2.5) for row in samples)
    assert all(row["reference_mode"] == "boundary_recovery" for row in samples)


def test_preturn_candidate_keeps_current_lane_outside_transition_arc():
    provider = ReferenceLineProvider()
    provider.preturn_lane_reference = lambda *_args, **_kwargs: (
        _line(1.0)[:8], "current_lane"
    )
    request = TurnReferenceRequest(
        local_map=SimpleNamespace(valid=True),
        config={"lane_follow_to_turn_reference_transition_arc_m": 12.0},
        horizon_steps=8,
        dt_s=0.1,
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.0,
        current_state=[0.0, 0.0, 5.0, 0.0],
        current_lane_id=10,
        target_lane_id=20,
        target_speed_mps=5.0,
        destination_state=[8.0, 1.0, 5.0, 0.0, 10],
        turn_direction="right",
    )

    result = provider.preturn_candidate(
        request, upcoming_turn_distance_m=20.0, first_forward_m=0.5
    )

    assert result.samples[0]["y_ref_m"] == pytest.approx(1.0)
    assert result.diagnostics["reference_source"] == (
        "admap_current_lane_center_preturn"
    )
    assert "lane_follow_turn_geometry_hold_reason" not in result.diagnostics


def test_preturn_candidate_locks_connector_and_preserves_speed_owner():
    provider = ReferenceLineProvider()
    provider.preturn_lane_reference = lambda *_args, **_kwargs: (
        _line(1.0)[:8], "current_lane"
    )
    provider.turn_reference = lambda request: (
        _line(-1.0)[:8], [8.0, -1.0, 2.2, 0.0, 20], "turn_locked"
    )
    request = TurnReferenceRequest(
        local_map=SimpleNamespace(valid=True),
        config={"lane_follow_to_turn_reference_transition_arc_m": 12.0},
        horizon_steps=8,
        dt_s=0.1,
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.0,
        current_state=[0.0, 0.0, 5.0, 0.0],
        current_lane_id=10,
        target_lane_id=20,
        target_speed_mps=5.0,
        destination_state=[8.0, 1.0, 5.0, 0.0, 10],
        turn_direction="right",
    )

    result = provider.preturn_candidate(
        request, upcoming_turn_distance_m=8.0, first_forward_m=0.5
    )

    assert result.diagnostics["reference_source"] == (
        "admap_preturn_connector_transition"
    )
    assert all(row["speed_ref_mps"] == pytest.approx(5.0) for row in result.samples)
    assert result.destination_state[2] == pytest.approx(5.0)


def test_modes_do_not_overwrite_each_other():
    provider = ReferenceLineProvider()
    provider.install(
        LANE_CHANGE,
        _line(1.0),
        route_revision="route-1",
        map_epoch="town06",
        event="maneuver_started",
    )
    provider.install(
        TURN,
        _line(2.0),
        route_revision="route-1",
        map_epoch="town06",
        event="maneuver_started",
    )

    assert provider.snapshot(LANE_CHANGE).samples[0]["y_ref_m"] == 1.0
    assert provider.snapshot(TURN).samples[0]["y_ref_m"] == 2.0


def test_lane_follow_rebuild_requires_a_named_lifecycle_event():
    provider = ReferenceLineProvider()
    provider.install(
        LANE_FOLLOW,
        _line(),
        route_revision="route-1",
        map_epoch="town06",
        event="initial_route",
    )
    revision = provider.snapshot(LANE_FOLLOW).geometry_revision

    installed, reason = provider.install(
        LANE_FOLLOW,
        _line(3.5),
        route_revision="route-1",
        map_epoch="town06",
        event="validation_failure",
    )

    assert not installed
    assert reason == "reference_install_event_forbidden:validation_failure"
    assert provider.snapshot(LANE_FOLLOW).geometry_revision == revision


def test_lane_follow_trace_uses_existing_persistent_diagnostic_schema():
    provider = ReferenceLineProvider()
    provider.install(
        LANE_FOLLOW,
        _line(),
        route_revision="route-1",
        map_epoch="town06",
        event="initial_route",
    )
    provider.mark_validation_failure(LANE_FOLLOW, "curvature")

    trace = provider.snapshot(LANE_FOLLOW).trace_fields()
    assert trace["persistent_reference_active"]
    assert trace["persistent_reference_state"] == "DEGRADED"
    assert trace["persistent_reference_last_rejected_trigger"] == (
        "validation_failure:curvature"
    )


def test_same_route_extension_cannot_change_route_owner():
    provider = ReferenceLineProvider()
    provider.install(
        LANE_FOLLOW,
        _line(),
        route_revision="route-1",
        map_epoch="town06",
        event="initial_route",
    )

    installed, reason = provider.install(
        LANE_FOLLOW,
        _line(),
        route_revision="route-2",
        map_epoch="town06",
        event="same_route_extension",
    )

    assert not installed
    assert reason == "extension_owner_mismatch"
    assert provider.snapshot(LANE_FOLLOW).route_revision == "route-1"


def test_publish_consumes_typed_planning_inputs_and_owns_acceptance():
    provider = ReferenceLineProvider()
    behavior = BehaviorDecision.from_mapping(
        {
            "decision": "lane_change_right",
            "lc_state": "EXECUTE_LANE_CHANGE_RIGHT",
            "current_lane_id": 10,
            "target_lane_id": 11,
            "target_speed_mps": 5.0,
        },
        default_speed_mps=5.0,
    )
    result = provider.publish(
        ReferenceLineRequest(
            local_map=SimpleNamespace(ego_lane_id=10),
            route_cursor=SimpleNamespace(segment_kind="lane_change"),
            behavior=behavior,
            route_revision="route-1",
            map_epoch="town06",
            ego_x_m=0.0,
            ego_y_m=0.0,
        ),
        _line(y_m=-1.0),
        valid=True,
        build_reason="typed_test",
    )
    assert result.accepted
    assert result.mode == LANE_CHANGE
    assert provider.snapshot(LANE_CHANGE).target_lane_id == 11
