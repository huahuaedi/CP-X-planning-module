from pipeline.static_obstacle_stage import StaticObstacleStage


def _evaluate(stage, **overrides):
    values = dict(
        requested=True, traffic_control_stop_active=False,
        obstacle_id="car-1", current_lane_id=10,
        lane_change_reference_active=False, sim_time_s=2.0,
        normal_mode=True, available_lane_ids=(10, 11),
        lane_safety_scores={10: 1.0, 11: 1.0},
        lane_prediction_risks={},
        attempt_replan=lambda: (True, True, "replanned"),
        # AD-map lane ids are opaque; these fixtures use consecutive small
        # integers purely for readability, so pin down that 11 really is
        # one real lane over from 10 rather than relying on that being
        # true by coincidence of the id values chosen.
        lane_to_offset={10: 0, 11: 1},
    )
    values.update(overrides)
    return stage.evaluate(**values)


def test_stage_confirms_then_owns_local_lane_borrow():
    stage = StaticObstacleStage({"static_obstacle_blocked_confirm_s": 1.0})
    first = _evaluate(stage, sim_time_s=1.0)
    ready = _evaluate(stage, sim_time_s=2.1)
    assert first.status == "confirming"
    assert ready.local_avoidance_active
    assert ready.target_lane_id == 11
    assert not ready.stop_active


def test_committed_lane_borrow_is_not_reselected_during_execution():
    stage = StaticObstacleStage({"static_obstacle_blocked_confirm_s": 0.0})
    ready = _evaluate(stage, sim_time_s=1.0)
    executing = _evaluate(
        stage,
        sim_time_s=1.1,
        lane_change_reference_active=True,
        lane_safety_scores={10: 1.0, 11: 0.0},
        lane_prediction_risks={11: {"risk": True}},
    )

    assert ready.target_lane_id == 11
    assert executing.target_lane_id == 11
    assert executing.status == "local_avoidance_executing"
    assert executing.local_avoidance_active
    assert not executing.stop_active


def test_stalled_local_avoidance_abandons_target_and_tries_next_lane():
    # Regression: a target_lane_id that turns out to be geometrically
    # infeasible (MPC reports "primal infeasible" every tick) used to be
    # held forever -- target_lane_id only ever cleared on *reaching* it.
    # A stall past the failure-count threshold must abandon it and pick a
    # different safe lane on the very same tick, not wait for another
    # confirm cycle.
    stage = StaticObstacleStage({
        "static_obstacle_blocked_confirm_s": 0.0,
        "static_obstacle_mpc_stall_timeout_failures": 30,
    })
    ready = _evaluate(
        stage,
        available_lane_ids=(10, 11, 12),
        lane_safety_scores={10: 1.0, 11: 1.0, 12: 1.0},
        lane_to_offset={10: 0, 11: 1, 12: 2},
    )
    assert ready.target_lane_id == 11

    stalled = _evaluate(
        stage,
        available_lane_ids=(10, 11, 12),
        lane_safety_scores={10: 1.0, 11: 1.0, 12: 1.0},
        lane_to_offset={10: 0, 11: 1, 12: 2},
        lane_change_reference_active=True,
        mpc_stall_failure_count=30,
    )
    assert stalled.target_lane_id == 12

    # Both candidates now stalled out (still stuck on the same current
    # lane) -- nothing left to try, so it must not silently re-pick 11.
    stalled_again = _evaluate(
        stage,
        available_lane_ids=(10, 11, 12),
        lane_safety_scores={10: 1.0, 11: 1.0, 12: 1.0},
        lane_to_offset={10: 0, 11: 1, 12: 2},
        lane_change_reference_active=True,
        mpc_stall_failure_count=30,
    )
    assert stalled_again.target_lane_id is None


def test_lane_borrow_ignores_lane_id_proximity_when_not_actually_adjacent():
    # Regression: AD-map lane ids are opaque and do not encode lateral
    # position. Lane 12 is numerically closer to current (10) than lane 30,
    # but its real corridor offset (+3, three lanes over -- e.g. a
    # different road segment reachable via the same local frame) makes it a
    # far worse local-avoidance target than lane 30, which is genuinely one
    # lane over. The old id-subtraction heuristic picked 12 every time;
    # this must pick 30 instead.
    stage = StaticObstacleStage({"static_obstacle_blocked_confirm_s": 0.0})
    ready = _evaluate(
        stage,
        available_lane_ids=(10, 12, 30),
        lane_safety_scores={10: 1.0, 12: 1.0, 30: 1.0},
        lane_to_offset={10: 0, 12: 3, 30: 1},
    )
    assert ready.target_lane_id == 30


def test_lane_borrow_releases_after_stable_target_lane_match():
    stage = StaticObstacleStage({
        "static_obstacle_blocked_confirm_s": 0.0,
        "static_obstacle_target_lane_release_frames": 3,
    })
    ready = _evaluate(stage, sim_time_s=1.0)
    assert ready.target_lane_id == 11

    first = _evaluate(
        stage, requested=False, current_lane_id=11,
        lane_change_reference_active=True, sim_time_s=1.1,
    )
    second = _evaluate(
        stage, requested=False, current_lane_id=11,
        lane_change_reference_active=True, sim_time_s=1.2,
    )
    released = _evaluate(
        stage, requested=False, current_lane_id=11,
        lane_change_reference_active=True, sim_time_s=1.3,
    )

    assert first.local_avoidance_active
    assert second.local_avoidance_active
    assert not released.local_avoidance_active
    assert released.target_lane_id is None
    assert released.status == "idle"


def test_stage_stops_when_no_safe_avoidance_exists():
    stage = StaticObstacleStage({"static_obstacle_blocked_confirm_s": 0.0})
    result = _evaluate(stage, lane_prediction_risks={11: {"risk": True}})
    assert result.status == "local_avoidance_unavailable_stop"
    assert result.stop_active


def test_stage_owns_replan_cooldown_and_transition_hold():
    stage = StaticObstacleStage({
        "static_obstacle_blocked_confirm_s": 0.0,
        "static_obstacle_local_avoidance_enabled": False,
        "static_obstacle_global_replan_enabled": True,
        "static_obstacle_replan_cooldown_s": 2.0,
    })
    first = _evaluate(stage, sim_time_s=3.0)
    second = _evaluate(stage, sim_time_s=3.5)
    assert first.status == "succeeded"
    assert first.stop_active
    assert second.status == "cooldown_route_transition"
    assert not second.stop_active


def test_stage_stays_stopped_when_no_lane_and_no_route_around_closure():
    # A closure spanning every lane with no detour must fail safe: hold the
    # stop rather than release control with nowhere for the ego to go.
    stage = StaticObstacleStage({
        "static_obstacle_blocked_confirm_s": 0.0,
        "static_obstacle_local_avoidance_enabled": False,
        "static_obstacle_global_replan_enabled": True,
        "static_obstacle_replan_cooldown_s": 2.0,
    })
    result = _evaluate(
        stage, sim_time_s=3.0,
        attempt_replan=lambda: (True, False, "route_replan_no_route_around_blocked_lane"),
    )
    assert result.status == "failed_stop"
    assert result.stop_active


def test_stage_retries_replan_after_cooldown_following_a_failed_attempt():
    attempts = []

    def _attempt_replan():
        attempts.append(True)
        return True, False, "route_replan_no_route_around_blocked_lane"

    stage = StaticObstacleStage({
        "static_obstacle_blocked_confirm_s": 0.0,
        "static_obstacle_local_avoidance_enabled": False,
        "static_obstacle_global_replan_enabled": True,
        "static_obstacle_replan_cooldown_s": 2.0,
    })
    _evaluate(stage, sim_time_s=3.0, attempt_replan=_attempt_replan)
    during_cooldown = _evaluate(stage, sim_time_s=3.5, attempt_replan=_attempt_replan)
    after_cooldown = _evaluate(stage, sim_time_s=5.5, attempt_replan=_attempt_replan)

    assert len(attempts) == 2
    assert during_cooldown.status == "cooldown_stop"
    assert during_cooldown.stop_active
    assert after_cooldown.status == "failed_stop"
    assert after_cooldown.stop_active
