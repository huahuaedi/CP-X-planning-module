from tools.audit_planning_run import audit_run


def _row(**overrides):
    row = {
        "collision_event_this_frame": "False",
        "road_boundary_breach": "False",
        "local_map_valid": "True",
        "route_topology_valid": "True",
        "pid_target_velocity_mps": "4.0",
        "nominal_speed_ref_mps": "5.0",
        "target_speed_mps": "5.0",
        "mpc_status": "solved",
        "speed_mps": "0.1",
    }
    row.update(overrides)
    return row


def test_accepts_safe_stopped_completed_run():
    violations, metrics = audit_run(
        [_row(), _row()],
        {
            "termination_reason": "route_destination_stopped",
            "cav_states": [{"speed_mps": 0.1}],
        },
    )
    assert violations == ()
    assert metrics["frame_count"] == 2


def test_rejects_moving_destination_and_pid_overspeed():
    violations, _ = audit_run(
        [_row(pid_target_velocity_mps="6.0")],
        {
            "termination_reason": "destination_reached",
            "cav_states": [{"speed_mps": 4.9}],
        },
    )
    assert "pid_target_above_nominal" in violations
    assert "destination_terminated_while_moving" in violations


def test_rejects_long_infeasible_sequence_and_safety_events():
    rows = [
        _row(
            mpc_status="primal infeasible",
            road_boundary_breach="True" if index == 0 else "False",
            collision_event_this_frame="True" if index == 1 else "False",
        )
        for index in range(11)
    ]
    violations, metrics = audit_run(rows)
    assert "collision_detected" in violations
    assert "road_boundary_breach" in violations
    assert "mpc_infeasible_run_exceeds_10_frames" in violations
    assert metrics["max_consecutive_mpc_infeasible_frames"] == 11


def test_local_scenario_without_global_route_does_not_fail_topology_audit():
    violations, metrics = audit_run([
        _row(route_topology_valid="False", route_topology_signature="")
    ])

    assert "route_topology_invalid" not in violations
    assert metrics["route_topology_expected_frames"] == 0


def test_rejects_persistent_candidate_hard_gate():
    rows = [
        _row(candidate_pipeline_selected_status="candidate_hard_gate")
        for _ in range(11)
    ]

    violations, metrics = audit_run(rows)

    assert "candidate_hard_gate_run_exceeds_10_frames" in violations
    assert metrics["max_consecutive_hard_gate_frames"] == 11


def test_rejects_pipeline_exception_and_route_defer_safe_stop():
    violations, metrics = audit_run([
        _row(
            pipeline_error="name 'legacy_state' is not defined",
            lane_change_authorization_direction="right",
            candidate_evaluation_summary="lane_change_left->L20",
            candidate_pipeline_selected="bounded_safe_stop",
            candidate_pipeline_selected_reason=(
                "route_required_candidate_infeasible_defer"
            ),
        )
    ])

    assert "planning_pipeline_exception" in violations
    assert "route_lane_change_defer_became_safe_stop" in violations
    assert "route_lane_change_direction_mismatch" in violations
    assert metrics["pipeline_error_frames"] == 1


def test_turn_exit_must_remain_on_admap_route_successor():
    route = "500144;5960149;540156"
    rows = [
        _row(
            behavior_decision="intersection_turn_right",
            current_lane_id="5960149",
            local_map_route_lane_sequence=route,
        ),
        _row(
            behavior_decision="lane_follow",
            current_lane_id="540155",
            local_map_route_lane_sequence=route,
        ),
    ]

    violations, metrics = audit_run(rows)

    assert "turn_exit_lane_not_on_route" in violations
    assert metrics["turn_exit_lane_id"] == 540155
    assert metrics["turn_exit_route_consistent"] is False


def test_turn_exit_accepts_connector_lane_and_longitudinal_successor():
    route = "500144;5960149;540156"
    rows = [
        _row(
            behavior_decision="intersection_turn_right",
            current_lane_id="5960149",
            local_map_route_lane_sequence=route,
        ),
        _row(
            behavior_decision="lane_follow",
            current_lane_id="5960149",
            local_map_route_lane_sequence=route,
        ),
        _row(
            behavior_decision="lane_follow",
            current_lane_id="540156",
            local_map_route_lane_sequence=route,
        ),
    ]

    violations, metrics = audit_run(rows)

    assert "turn_exit_lane_not_on_route" not in violations
    assert metrics["turn_exit_lane_id"] == 540156
    assert metrics["turn_exit_route_consistent"] is True


def test_turn_exit_rejects_late_drift_after_touching_route_successor():
    route = "500144;5960149;540156"
    rows = [
        _row(
            behavior_decision="intersection_turn_right",
            current_lane_id="5960149",
            local_map_route_lane_sequence=route,
        ),
        _row(
            behavior_decision="lane_follow",
            current_lane_id="5960149",
            local_map_route_lane_sequence=route,
        ),
        _row(
            behavior_decision="lane_follow",
            current_lane_id="540156",
            local_map_route_lane_sequence=route,
        ),
        _row(
            behavior_decision="lane_follow",
            current_lane_id="540155",
            local_map_route_lane_sequence=route,
        ),
    ]

    violations, metrics = audit_run(rows)

    assert "turn_exit_lane_not_on_route" in violations
    assert metrics["turn_exit_lane_id"] == 540155
    assert metrics["turn_exit_route_consistent"] is False


def test_rejects_run_that_stops_during_unfinished_turn():
    violations, metrics = audit_run([
        _row(
            behavior_decision="intersection_turn_left",
            current_lane_id="10040151",
            local_map_route_lane_sequence="340154;10040151;390145",
        )
    ])

    assert "turn_not_completed" in violations
    assert metrics["turn_started"] is True
    assert metrics["turn_completed"] is False
