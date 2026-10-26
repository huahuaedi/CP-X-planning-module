import json

from tools.ros_integration_suite import analyze_case
from tools.ros_integration_suite import _is_complete
from tools.ros_integration_suite import _subject_cav_state


def _write_case(root, name, rows, *, completed=True, collisions=0):
    directory = root / name
    directory.mkdir(parents=True)
    status = {
        "termination_reason": (
            "route_destination_stopped" if completed else "max_ticks_reached"
        ),
        "completed_ticks": len(rows),
        "collision_count": collisions,
        "road_boundary_breach_count": 0,
        "distance_to_destination_m": 2.0 if completed else 20.0,
        "exception": "",
        "subject_cav_index": 0,
        "cav_states": [{"agent_finished": completed}],
    }
    (directory / "run_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    (directory / "ros_planner_control_output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _row(session_id="run:1", **updates):
    row = {
        "source": "ros",
        "ros_control_state": "active",
        "session_id": session_id,
        "input_to_output_time_ms": 30.0,
        "planning_cycle_time_ms": 20.0,
        "prediction_bridge_attached_mode_count": 0,
        "prediction_ros_revision": -1,
        "cav_conflict_summary": "",
        "route_physical_target_lane_id": 630148,
    }
    row.update(updates)
    return row


def test_destination_reached_termination_is_complete():
    assert _is_complete({
        "termination_reason": "destination_reached",
        "cav_states": [{"agent_finished": False}],
    }) is True


def test_subject_cav_state_uses_archived_log_owner():
    status = {
        "subject_cav_index": 1,
        "cav_states": [
            {"index": 0, "distance_to_destination_m": 0.5},
            {"index": 1, "distance_to_destination_m": 40.0},
        ],
    }

    assert _subject_cav_state(status)["distance_to_destination_m"] == 40.0


def test_six_mode_ros_contract_accepts_fresh_advancing_prediction(tmp_path):
    _write_case(
        tmp_path,
        "ego",
        [
            _row(
                prediction_bridge_attached_mode_count=6,
                prediction_ros_revision=10,
                prediction_ros_frame_age_s=0.2,
            ),
            _row(
                prediction_bridge_attached_mode_count=6,
                prediction_ros_revision=11,
                prediction_ros_frame_age_s=0.3,
            ),
        ],
    )

    result = analyze_case(
        "prediction",
        {"scenario": "test", "dirs": ["ego"], "completion": True, "modes": 6},
        tmp_path,
    )

    assert result["contract_pass"] is True
    assert result["safety_pass"] is True


def test_multi_cav_tracker_contract_reports_reused_execution_rate(tmp_path):
    rows = [
        _row(control_reused=False),
        _row(
            control_reused=True,
            lateral_execution_tracker_active=True,
            lateral_execution_tracker_reason="latest_completed_mpc_path",
        ),
        _row(
            control_reused=True,
            lateral_execution_tracker_active=True,
            lateral_execution_tracker_reason="latest_completed_mpc_path",
        ),
    ]
    _write_case(tmp_path, "cav1", rows)

    result = analyze_case(
        "multi",
        {
            "scenario": "test",
            "dirs": ["cav1"],
            "completion": True,
            "require_lateral_tracker": True,
        },
        tmp_path,
    )

    log = result["logs"][0]
    assert result["contract_pass"] is True
    assert log["reused_active_frames"] == 2
    assert log["lateral_tracker_frames"] == 2
    assert log["lateral_tracker_rate"] == 1.0


def test_stale_control_fallback_is_counted_and_rejected(tmp_path):
    rows = [_row() for _ in range(49)]
    rows.append(
        _row(
            ros_control_state="stale_control",
            emergency_brake_control_active=True,
        )
    )
    rows.append(
        _row(
            ros_control_state="stale_control",
            emergency_brake_control_active=True,
        )
    )
    _write_case(tmp_path, "ego", rows)

    result = analyze_case(
        "stale_control",
        {"scenario": "test", "dirs": ["ego"], "completion": True},
        tmp_path,
    )

    log = result["logs"][0]
    assert log["stale_frames"] == 2
    assert log["stale_rate"] == 2.0 / 51.0
    assert "ego:stale_control_rate_above_2_percent" in result["failures"]


def test_prediction_contract_rejects_stale_attached_frame(tmp_path):
    _write_case(
        tmp_path,
        "ego",
        [
            _row(
                prediction_bridge_attached_mode_count=6,
                prediction_ros_revision=10,
                prediction_ros_frame_age_s=0.2,
            ),
            _row(
                prediction_bridge_attached_mode_count=6,
                prediction_ros_revision=11,
                prediction_ros_frame_age_s=0.7,
            ),
        ],
    )

    result = analyze_case(
        "prediction",
        {"scenario": "test", "dirs": ["ego"], "completion": True, "modes": 6},
        tmp_path,
    )

    assert "attached_prediction_older_than_0_5s" in result["failures"]


def test_multi_cav_contract_rejects_shared_session_identity(tmp_path):
    _write_case(tmp_path, "cav1", [_row("same-run:same-cav")])
    _write_case(tmp_path, "cav2", [_row("same-run:same-cav")])

    result = analyze_case(
        "multi",
        {"scenario": "test", "dirs": ["cav1", "cav2"], "completion": True},
        tmp_path,
    )

    assert "multi_cav_session_not_isolated" in result["failures"]


def test_longitudinal_contract_rejects_lane_change_decision(tmp_path):
    _write_case(
        tmp_path,
        "lead",
        [_row(behavior_decision="lane_follow")],
    )
    _write_case(
        tmp_path,
        "follower",
        [_row("run:2", behavior_decision="lane_change_left")],
    )

    result = analyze_case(
        "lead_braking",
        {
            "scenario": "test",
            "dirs": ["lead", "follower"],
            "completion": True,
            "forbid_lane_change": True,
        },
        tmp_path,
    )

    assert "unexpected_lane_change_decision" in result["failures"]


def test_lane_closure_off_rejects_state_leak_from_on_run(tmp_path):
    _write_case(
        tmp_path,
        "off",
        [_row(route_physical_target_lane_id=630149)],
        completed=False,
    )

    result = analyze_case(
        "F_off",
        {
            "scenario": "test",
            "dirs": ["off"],
            "completion": False,
            "forbidden_lane": 630149,
        },
        tmp_path,
    )

    assert "forbidden_target_lane_present" in result["failures"]
