"""The MPC-path emergency decision: what is an emergency and what is not.

An emergency means full braking with straight wheels.  A hard gate that only
says "this reference is unusable" is a bounded degradation instead, and the
distinction has real history: an empty reference during a turn was once
treated as harmless and the ego drove on into static scene geometry.
"""

import pytest

from pipeline.safety_supervisor import (
    emergency_stop_reason,
    hard_gate_requires_emergency_stop,
    pipeline_failure_action,
    pipeline_failure_stop,
)

GATE = "candidate_hard_gate:"


def _hard_gate(reason, decision="lane_follow", stop_goal=False):
    return hard_gate_requires_emergency_stop(
        fallback_reason=reason, behavior_decision=decision, stop_goal_active=stop_goal,
    )


# ---- what makes a hard gate an emergency -----------------------------------

@pytest.mark.parametrize("token", [
    "collision_risk", "emergency_brake_direct_control", "stop_missing_target_hard_lock",
    "empty_reference", "no_corridor_geometry", "too_few_forward_samples",
])
def test_hazard_tokens_make_a_hard_gate_an_emergency(token):
    assert _hard_gate(GATE + "lane_change_left:" + token) is True


def test_the_collision_risk_reason_from_the_candidate_pipeline_is_an_emergency():
    assert _hard_gate(GATE + "lane_change_left:candidate_prediction_collision_risk",
                      decision="lane_change_left") is True


def test_a_geometry_or_continuity_veto_is_not_an_emergency():
    assert _hard_gate(
        GATE + "lane_change_left:final_reference_gate:destination_lane_error_out_of_contract",
        decision="lane_change_left") is False
    assert _hard_gate(GATE + "turn_swept_footprint:corridor_violation") is False


@pytest.mark.parametrize("decision", ["emergency_brake", "stop_at_intersection", "stop_sign"])
def test_any_hard_gate_during_a_stop_like_maneuver_is_an_emergency(decision):
    assert _hard_gate(GATE + "some_geometry_veto", decision=decision) is True


def test_any_hard_gate_with_a_stop_goal_is_an_emergency():
    assert _hard_gate(GATE + "some_geometry_veto", stop_goal=True) is True


@pytest.mark.parametrize("reason", [
    "", None, "MPC status=primal infeasible", "collision_risk",   # not a hard gate: the prefix is required
    "hard_gate:collision_risk",
])
def test_only_a_hard_gate_can_be_an_emergency_by_this_rule(reason):
    assert _hard_gate(reason, decision="emergency_brake", stop_goal=True) is False


def test_matching_ignores_case_and_surrounding_whitespace():
    assert _hard_gate("  CANDIDATE_HARD_GATE:Collision_Risk ") is True
    assert _hard_gate(GATE + "veto", decision="  STOP_SIGN ") is True


# ---- the combined decision and its reason -----------------------------------

def _reason(fallback="", decision="lane_follow", stop_goal=False, escalate=False):
    return emergency_stop_reason(
        fallback_reason=fallback, behavior_decision=decision,
        stop_goal_active=stop_goal, corridor_infeasible_escalate=escalate,
    )


def test_an_ordinary_tick_is_not_an_emergency():
    assert _reason() == ""


def test_each_source_reports_its_own_reason():
    assert _reason(fallback=GATE + "collision_risk") == "hard_gate_stop_hazard"
    assert _reason(decision="emergency_brake") == "emergency_brake_maneuver"
    assert _reason(escalate=True) == "corridor_infeasible_escalation"
    assert emergency_stop_reason(
        fallback_reason="",
        behavior_decision="lane_follow",
        stop_goal_active=False,
        corridor_infeasible_escalate=False,
        proximity_emergency_stop_required=True,
    ) == "proximity_emergency_gap"


def test_the_emergency_brake_maneuver_needs_no_hard_gate():
    assert _reason(fallback="", decision="Emergency_Brake") == "emergency_brake_maneuver"


def test_a_hard_gate_hazard_is_reported_ahead_of_the_other_sources():
    assert _reason(fallback=GATE + "collision_risk", decision="emergency_brake", escalate=True) == "hard_gate_stop_hazard"


def test_a_geometry_veto_stays_a_non_emergency_even_with_no_other_source():
    assert _reason(fallback=GATE + "turn_swept_footprint") == ""


# ---- what to do when the pipeline itself raises ----------------------------

@pytest.mark.parametrize("policy, action", [
    ("emergency_stop", "emergency_stop"),
    ("raise", "raise"),
    ("opencda", "raise"),
    ("", "emergency_stop"),
    ("anything-else", "emergency_stop"),
])
def test_pipeline_failure_action(policy, action):
    assert pipeline_failure_action(policy) == action


# ---- the bridge's last-resort stop when the pipeline raises ----------------

def _bridge_whose_pipeline_raises(policy):
    from types import SimpleNamespace
    from opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge

    bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
    bridge.fallback_policy = policy
    bridge.vehicle_manager = SimpleNamespace(vehicle=SimpleNamespace(id=5))
    bridge.mpc = SimpleNamespace(constraints=SimpleNamespace(min_acceleration_mps2=-3.0))
    bridge._sim_time_s = lambda: 1.0
    bridge._record_debug = lambda payload: None
    bridge.actuator_port = SimpleNamespace(
        emergency_stop_control=lambda: "EMERGENCY-CONTROL"
    )

    def boom():
        raise RuntimeError("pipeline blew up")

    bridge.execute_planning_pipeline = boom
    return bridge


def test_run_step_falls_back_to_an_emergency_stop_by_default():
    bridge = _bridge_whose_pipeline_raises("emergency_stop")
    assert bridge.run_step() == "EMERGENCY-CONTROL"
    assert bridge.last_debug["fallback_active"] is True
    assert bridge.last_debug["control_guard_reason"] == "fallback_policy_emergency_stop"
    assert bridge.last_debug["fallback_reason"] == "pipeline blew up"


def test_run_step_lets_the_exception_through_under_the_raise_policy():
    with pytest.raises(RuntimeError, match="pipeline blew up"):
        _bridge_whose_pipeline_raises("raise").run_step()


def test_run_step_records_the_hard_brake_as_the_last_command():
    bridge = _bridge_whose_pipeline_raises("emergency_stop")
    bridge.run_step()
    assert bridge._last_accel_mps2 == -3.0
    assert bridge._last_steer_rad == 0.0


def test_raise_policy_leaves_the_last_command_alone():
    bridge = _bridge_whose_pipeline_raises("raise")
    with pytest.raises(RuntimeError):
        bridge.run_step()
    assert not hasattr(bridge, "_last_accel_mps2")


# ---- pipeline_failure_stop itself -------------------------------------------

def _failure(policy, error=None, control=lambda: "BRAKE"):
    return pipeline_failure_stop(
        error=error or RuntimeError("boom"),
        fallback_policy=policy,
        emergency_stop_control=control,
        min_acceleration_mps2=-4.5,
        sim_time_s=12.0,
        vehicle_id=7,
    )


def test_failure_stop_is_a_hard_brake_with_the_mpc_deceleration_limit():
    stop = _failure("emergency_stop")
    assert stop.control == "BRAKE"
    assert stop.acceleration_mps2 == -4.5
    assert stop.steering_rad == 0.0


def test_failure_stop_debug_payload_is_exact():
    assert dict(_failure("emergency_stop").debug) == {
        "sim_time_s": 12.0,
        "vehicle_id": 7,
        "planner": "cpx_mpc",
        "planner_requested": True,
        "planner_executed": False,
        "fallback_active": True,
        "fallback_reason": "boom",
        "mpc_fallback_reason": "boom",
        "control_guard_reason": "fallback_policy_emergency_stop",
        "accel_cmd_mps2": -4.5,
        "steer_cmd_rad": 0.0,
    }


@pytest.mark.parametrize("policy", ["raise", "opencda"])
def test_failure_stop_reraises_the_same_error_without_building_a_control(policy):
    error = RuntimeError("original")
    built = []

    def control():
        built.append(1)
        return "BRAKE"

    with pytest.raises(RuntimeError) as caught:
        _failure(policy, error=error, control=control)
    assert caught.value is error
    assert built == []
