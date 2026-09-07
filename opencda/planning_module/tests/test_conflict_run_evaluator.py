import json

from opencda.scenario_testing.evaluate_conflict_run import analyze_conflict_run
from opencda.scenario_testing.planner_debug_records import load_planner_records


def _row(tick, tag="LEAD_BRAKE", rows=4, **extra):
    value = {
        "sim_time_s": tick * 0.1,
        "cav_conflict_tags": {"peer": tag},
        "cav_conflict_roles": {"peer": "yield"},
        "cav_total_qp_row_count": rows,
        "measured_accel_mps2": -1.0,
        "front_gap_m": 12.0,
        "collision_count": 0,
        "fallback_active": False,
        "mpc_status": "solved",
    }
    value.update(extra)
    return value


def test_evaluator_reports_valid_safe_end_to_end_run():
    result = analyze_conflict_run([_row(0), _row(1)], expected_tags=["LEAD_BRAKE"])
    assert result["verdict"] == "PASS"
    assert result["constraint_ticks"] == 2
    assert result["collision_count"] == 0


def test_evaluator_does_not_call_missing_conflict_a_pass():
    result = analyze_conflict_run([_row(0, tag="IGNORE", rows=0)],
                                  expected_tags=["CUT_IN"])
    assert result["verdict"] == "INVALID_SCENARIO"


def test_evaluator_requires_requested_cooperative_role():
    missing = analyze_conflict_run(
        [_row(0)], expected_tags=["LEAD_BRAKE"], expected_roles=["make_gap"]
    )
    present = analyze_conflict_run(
        [_row(0, cav_conflict_roles={"peer": "make_gap"})],
        expected_tags=["LEAD_BRAKE"], expected_roles=["make_gap"],
    )
    assert missing["verdict"] == "INVALID_SCENARIO"
    assert present["verdict"] == "PASS"


def test_evaluator_distinguishes_collision_and_planner_failure():
    collision = analyze_conflict_run(
        [_row(0, collision_count=1)], expected_tags=["LEAD_BRAKE"]
    )
    failure = analyze_conflict_run(
        [_row(0, fallback_active=True)], expected_tags=["LEAD_BRAKE"]
    )
    assert collision["verdict"] == "UNSAFE_COLLISION"
    assert failure["verdict"] == "PLANNER_FAILURE"


def test_canonical_reader_prefers_jsonl_over_legacy_csv(tmp_path):
    (tmp_path / "opencda_planner_debug.csv").write_text(
        "sim_time_s,speed_mps\n0.0,99.0\n", encoding="utf-8"
    )
    (tmp_path / "opencda_planner_debug.jsonl").write_text(
        json.dumps({"sim_time_s": 0.0, "speed_mps": 7.0}) + "\n",
        encoding="utf-8",
    )
    assert load_planner_records(tmp_path)[0]["speed_mps"] == 7.0
