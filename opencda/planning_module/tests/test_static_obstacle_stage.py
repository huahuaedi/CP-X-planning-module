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
