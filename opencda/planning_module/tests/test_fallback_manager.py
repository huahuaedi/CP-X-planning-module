from pipeline.fallback_manager import (
    FailureReason,
    FallbackRequest,
    TrajectoryFallbackManager,
)


def _line(speed=4.0):
    return [
        {"x_ref_m": float(index), "y_ref_m": 0.0, "speed_ref_mps": speed}
        for index in range(6)
    ]


def test_recent_trajectory_on_same_route_is_held():
    manager = TrajectoryFallbackManager(max_hold_age_s=0.4)
    assert manager.record_valid(_line(), sim_time_s=1.0, route_revision="r1")
    result = manager.resolve(
        sim_time_s=1.2,
        route_revision="r1",
        current_speed_mps=3.5,
        current_reference=_line(),
        failure_reason="solver_failure",
    )
    assert result.mode == "hold_last_valid"
    assert result.target_speed_mps == 4.0


def test_route_change_forbids_hold_and_uses_bounded_safe_stop():
    manager = TrajectoryFallbackManager(
        max_hold_age_s=0.4, safe_stop_deceleration_mps2=2.0
    )
    manager.record_valid(_line(), sim_time_s=1.0, route_revision="r1")
    result = manager.resolve(
        sim_time_s=1.1,
        route_revision="r2",
        current_speed_mps=4.0,
        current_reference=_line(),
        failure_reason="route_changed",
    )
    speeds = [sample["speed_ref_mps"] for sample in result.trajectory]
    assert result.mode == "bounded_safe_stop"
    assert result.target_speed_mps == 0.0
    assert speeds[-1] == 0.0
    assert all(second <= first for first, second in zip(speeds, speeds[1:]))


def test_expired_last_valid_uses_safe_stop_without_new_geometry():
    reference = _line()
    manager = TrajectoryFallbackManager(max_hold_age_s=0.1)
    manager.record_valid(reference, sim_time_s=1.0, route_revision="r1")
    result = manager.resolve(
        sim_time_s=2.0,
        route_revision="r1",
        current_speed_mps=2.0,
        current_reference=reference,
        failure_reason="contract_rejected",
    )
    assert result.mode == "bounded_safe_stop"
    assert [sample["x_ref_m"] for sample in result.trajectory] == [
        sample["x_ref_m"] for sample in reference
    ]


def test_explicit_destination_stop_never_holds_last_valid_motion():
    manager = TrajectoryFallbackManager(max_hold_age_s=10.0)
    manager.record_valid(_line(speed=6.0), sim_time_s=1.0, route_revision="r1")
    result = manager.bounded_safe_stop(
        current_speed_mps=6.0,
        current_reference=_line(speed=6.0),
        reason="route_destination_reached",
    )
    assert result.mode == "bounded_safe_stop"
    assert result.target_speed_mps == 0.0
    assert result.trajectory[-1]["speed_ref_mps"] == 0.0
    assert result.reason == "bounded_safe_stop:route_destination_reached"


def test_typed_failure_is_data_and_manager_selects_policy():
    manager = TrajectoryFallbackManager(max_hold_age_s=0.5)
    reference = _line(speed=2.0)
    manager.record_valid(reference, sim_time_s=1.0, route_revision="r1")
    result = manager.arbitrate(FallbackRequest(
        sim_time_s=1.1,
        route_revision="r1",
        current_speed_mps=2.0,
        current_reference=tuple(reference),
        failure=FailureReason(stage="mpc", code="infeasible"),
    ))
    assert result.mode == "hold_last_valid"
    assert "mpc:infeasible" in result.reason
