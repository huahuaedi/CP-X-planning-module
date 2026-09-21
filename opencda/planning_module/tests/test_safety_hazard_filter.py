"""SafetySupervisor's hazard gate: which OpenCDA safety flags stop the car.

OpenCDA's safety manager reports boolean flags (collision, ran_light, stuck,
offroad, ...).  SafetySupervisor.filter_control turns them into either a full
brake or a *release* -- "this flag is expected right now, keep driving" --
depending on the signal state, the maneuver, a stop goal, the planner's own
acceleration and whether a turn-boundary recovery is active.

The existing tests in test_safety_supervisor.py pin a handful of those cases.
This file states the rules once as an independent model (``expected``) and
compares the implementation to it over the whole grid, then pins the edges the
grid cannot express: flag parsing, the acceleration threshold, the disabled
supervisor, and the exact reason strings.

Only the FIXTURE section knows how the supervisor is built and called.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

from pipeline.safety_supervisor import SafetySupervisor

# =============================== FIXTURE ===============================


class _Control:
    def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
        self.throttle, self.brake, self.steer = float(throttle), float(brake), float(steer)


class _SafetyManager:
    def __init__(self, status):
        self.status_queue = [(0.0, dict(status))]


_THRESHOLD = 0.01
_UNSET = object()
_INPUT = _Control(throttle=0.3, brake=0.0, steer=0.1)


def gate(status, *, behavior="lane_follow", signal="green", stop_goal=False,
         accel=0.5, recovery=False, enabled=True, safety_manager=_UNSET):
    """Run the hazard gate; returns (control_out, reason)."""

    supervisor = SafetySupervisor(enabled=enabled, stuck_release_min_accel_mps2=_THRESHOLD)
    supervisor._turn_boundary_recovery_active = bool(recovery)
    return supervisor.filter_control(
        control=_INPUT,
        make_pedal_control=_Control,
        safety_manager=_SafetyManager(status) if safety_manager is _UNSET else safety_manager,
        behavior_decision=behavior,
        traffic_signal_state=signal,
        stop_goal_active=stop_goal,
        planner_accel_mps2=accel,
    )


# =========================== independent rule model ===========================

_STOP_LIKE = {"stop_at_intersection", "stop_sign", "emergency_brake", "static_obstacle_stop"}
_TURNS = {"intersection_turn_left", "intersection_turn_right"}
_DRIVING_RECOVERY = {"lane_change_left", "lane_change_right", "route_recovery"}


def expected(flags, *, behavior, signal, stop_goal, accel, recovery):
    """The hazard the gate should still see after releases; '' means release."""

    remaining = set(flags)
    if "collision" in remaining:
        return "collision"                                   # never released, checked first
    stop_like = stop_goal or behavior in _STOP_LIKE
    turn = behavior in _TURNS
    driving = behavior == "lane_follow" or turn or behavior in _DRIVING_RECOVERY
    clear_light = signal in {"green", "unknown"}
    red_or_yellow = signal in {"red", "yellow"}

    if "ran_light" in remaining:
        if clear_light and driving and not stop_like:
            remaining.discard("ran_light")                   # a stale flag once driving resumed
        elif stop_like or red_or_yellow:
            return "ran_light"                               # decided immediately, ahead of the rest
    if "stuck" in remaining:
        if clear_light and driving and not stop_like and (accel >= _THRESHOLD or (turn and recovery)):
            remaining.discard("stuck")
        elif stop_like:
            return "stuck"
        elif red_or_yellow and accel >= _THRESHOLD:
            remaining.discard("stuck")                       # creeping toward a distant light
    if "offroad" in remaining and turn and recovery and clear_light and not stop_like:
        remaining.discard("offroad")                         # the recovery itself leaves the lane
    return ",".join(sorted(remaining))


_FLAG_SETS = [
    ("collision",), ("ran_light",), ("stuck",), ("offroad",), ("imu",),
    ("ran_light", "stuck"), ("stuck", "offroad"), ("collision", "stuck"),
    ("ran_light", "offroad"), ("ran_light", "stuck", "offroad"),
]
_SIGNALS = ["green", "unknown", "red", "yellow", "", "RED", "  Green "]
_BEHAVIORS = [
    "lane_follow", "intersection_turn_left", "intersection_turn_right", "lane_change_left",
    "lane_change_right", "route_recovery", "stop_at_intersection", "stop_sign",
    "emergency_brake", "static_obstacle_stop", "yield_slow_down", "",
]


def _norm(text):
    return str(text).strip().lower()


def test_the_gate_matches_the_rule_model_over_the_whole_grid():
    mismatches = []
    checked = 0
    for flags, signal, behavior, stop_goal, accel, recovery in itertools.product(
        _FLAG_SETS, _SIGNALS, _BEHAVIORS, (False, True), (0.0, 0.5), (False, True)
    ):
        control, reason = gate(
            {flag: True for flag in flags}, behavior=behavior, signal=signal,
            stop_goal=stop_goal, accel=accel, recovery=recovery)
        want = expected(flags, behavior=_norm(behavior), signal=_norm(signal),
                        stop_goal=stop_goal, accel=accel, recovery=recovery)
        if want:
            ok = (reason == "safety_supervisor_emergency_stop:" + want
                  and (control.throttle, control.brake, control.steer) == (0.0, 1.0, 0.0))
        else:
            ok = reason.startswith("safety_supervisor_release:") and control is _INPUT
        checked += 1
        if not ok:
            mismatches.append((flags, signal, behavior, stop_goal, accel, recovery, reason, want))
    assert checked == len(_FLAG_SETS) * len(_SIGNALS) * len(_BEHAVIORS) * 2 * 2 * 2
    assert not mismatches, f"{len(mismatches)} of {checked} differ; first: {mismatches[:3]}"


# ================================ named rules ================================
# The same behaviour, one sentence each, so a failure reads as a broken rule
# rather than as one of thousands of grid cells.


def _emergency(reason):
    return reason.startswith("safety_supervisor_emergency_stop:")


def test_collision_always_brakes_whatever_else_is_true():
    for kw in [dict(), dict(behavior="lane_follow", signal="green", accel=9.0, recovery=True),
               dict(behavior="intersection_turn_left", recovery=True, signal="unknown")]:
        control, reason = gate({"collision": True, "stuck": True, "ran_light": True}, **kw)
        assert reason == "safety_supervisor_emergency_stop:collision"
        assert (control.throttle, control.brake, control.steer) == (0.0, 1.0, 0.0)


def test_a_stale_ran_light_flag_is_released_once_driving_resumes_on_a_clear_light():
    for behavior in ["lane_follow", "intersection_turn_left", "lane_change_right", "route_recovery"]:
        for signal in ["green", "unknown"]:
            control, reason = gate({"ran_light": True}, behavior=behavior, signal=signal)
            assert reason == "safety_supervisor_release:ran_light" and control is _INPUT


def test_ran_light_stops_on_red_or_yellow_and_during_any_stop():
    for signal in ["red", "yellow"]:
        assert gate({"ran_light": True}, signal=signal)[1] == "safety_supervisor_emergency_stop:ran_light"
    assert gate({"ran_light": True}, signal="green", stop_goal=True)[1] == "safety_supervisor_emergency_stop:ran_light"
    assert gate({"ran_light": True}, signal="green", behavior="stop_sign")[1] == "safety_supervisor_emergency_stop:ran_light"


def test_ran_light_on_a_clear_light_with_an_unclassified_maneuver_is_not_released():
    # Neither a driving maneuver nor a stop: the flag survives the filter.
    assert gate({"ran_light": True}, signal="green", behavior="yield_slow_down")[1] == \
        "safety_supervisor_emergency_stop:ran_light"


def test_stuck_is_released_when_driving_on_a_clear_light_and_the_planner_wants_to_move():
    assert gate({"stuck": True}, accel=0.5)[1] == "safety_supervisor_release:stuck"
    assert gate({"stuck": True}, accel=0.0)[1] == "safety_supervisor_emergency_stop:stuck"


def test_stuck_release_threshold_is_inclusive():
    assert gate({"stuck": True}, accel=_THRESHOLD)[1] == "safety_supervisor_release:stuck"
    assert gate({"stuck": True}, accel=_THRESHOLD - 1e-6)[1] == "safety_supervisor_emergency_stop:stuck"


def test_a_turn_in_boundary_recovery_releases_stuck_without_planner_acceleration():
    assert gate({"stuck": True}, behavior="intersection_turn_left", accel=0.0, recovery=True)[1] == \
        "safety_supervisor_release:stuck"
    assert gate({"stuck": True}, behavior="intersection_turn_left", accel=0.0, recovery=False)[1] == \
        "safety_supervisor_emergency_stop:stuck"
    assert gate({"stuck": True}, behavior="lane_follow", accel=0.0, recovery=True)[1] == \
        "safety_supervisor_emergency_stop:stuck", "recovery only excuses a turn"


def test_stuck_always_stops_during_a_stop_maneuver_or_stop_goal():
    for behavior in ["stop_at_intersection", "stop_sign", "emergency_brake", "static_obstacle_stop"]:
        assert gate({"stuck": True}, behavior=behavior, accel=9.0)[1] == "safety_supervisor_emergency_stop:stuck"
    assert gate({"stuck": True}, stop_goal=True, accel=9.0)[1] == "safety_supervisor_emergency_stop:stuck"


def test_stuck_on_red_or_yellow_is_released_only_while_the_planner_is_creeping_forward():
    for signal in ["red", "yellow"]:
        assert gate({"stuck": True}, signal=signal, accel=0.5)[1] == "safety_supervisor_release:stuck"
        assert gate({"stuck": True}, signal=signal, accel=0.0)[1] == "safety_supervisor_emergency_stop:stuck"


def test_offroad_is_excused_only_for_a_turn_that_is_recovering_on_a_clear_light():
    turn = dict(behavior="intersection_turn_right", signal="green", recovery=True)
    assert gate({"offroad": True}, **turn)[1] == "safety_supervisor_release:offroad"
    for change in [dict(recovery=False), dict(behavior="lane_follow"), dict(signal="red"),
                   dict(stop_goal=True), dict(behavior="stop_sign")]:
        assert gate({"offroad": True}, **{**turn, **change})[1] == "safety_supervisor_emergency_stop:offroad"


def test_an_unknown_flag_is_never_released():
    assert gate({"imu": True})[1] == "safety_supervisor_emergency_stop:imu"
    assert gate({"mystery": True}, behavior="intersection_turn_left", recovery=True)[1] == \
        "safety_supervisor_emergency_stop:mystery"


def test_ran_light_is_decided_before_the_other_flags_are_looked_at():
    # ran_light returns at once, so the reason names it alone even with stuck also set.
    assert gate({"ran_light": True, "stuck": True}, signal="red", accel=0.0)[1] == \
        "safety_supervisor_emergency_stop:ran_light"


def test_when_several_flags_survive_the_reason_lists_them_sorted():
    assert gate({"stuck": True, "offroad": True}, accel=0.0)[1] == \
        "safety_supervisor_emergency_stop:offroad,stuck"


def test_matching_ignores_case_and_surrounding_whitespace():
    assert gate({"stuck": True}, behavior="  LANE_FOLLOW ", signal="  GREEN ", accel=0.5)[1] == \
        "safety_supervisor_release:stuck"
    assert gate({"stuck": True}, behavior=" Stop_Sign ", accel=9.0)[1] == "safety_supervisor_emergency_stop:stuck"


def test_a_missing_signal_or_behavior_is_treated_as_unclassified_not_as_clear():
    assert gate({"ran_light": True}, signal=None, behavior=None)[1] == "safety_supervisor_emergency_stop:ran_light"


# ============================= flag parsing / edges =============================


def test_the_release_reason_lists_the_active_flags_in_report_order():
    control, reason = gate({"stuck": True, "ran_light": True}, accel=0.5)
    assert reason == "safety_supervisor_release:stuck,ran_light"
    assert control is _INPUT


def test_flags_that_are_false_are_not_hazards():
    control, reason = gate({"collision": False, "stuck": False, "offroad": 0, "imu": ""})
    assert not reason.startswith("safety_supervisor_")


def test_truthy_non_boolean_values_count_as_active():
    assert gate({"collision": 1})[1] == "safety_supervisor_emergency_stop:collision"
    assert gate({"collision": "yes"})[1] == "safety_supervisor_emergency_stop:collision"


@pytest.mark.parametrize("manager", [
    None,
    SimpleNamespace(),
    SimpleNamespace(status_queue=[]),
    SimpleNamespace(status_queue=None),
    SimpleNamespace(status_queue=[(0.0, "not-a-mapping")]),
    SimpleNamespace(status_queue=[(0.0, None)]),
    SimpleNamespace(status_queue=["garbage"]),
    SimpleNamespace(status_queue=[(0.0,)]),
], ids=["no-manager", "no-queue-attr", "empty-queue", "none-queue", "non-mapping-status",
        "none-status", "malformed-entry", "short-entry"])
def test_a_missing_or_malformed_status_is_no_hazard(manager):
    control, reason = gate(None, safety_manager=manager)
    assert not reason.startswith("safety_supervisor_")


def test_only_the_latest_status_is_read():
    manager = SimpleNamespace(status_queue=[(0.0, {"collision": True}), (1.0, {"collision": False})])
    assert not gate(None, safety_manager=manager)[1].startswith("safety_supervisor_")
    manager = SimpleNamespace(status_queue=[(0.0, {"collision": False}), (1.0, {"collision": True})])
    assert gate(None, safety_manager=manager)[1] == "safety_supervisor_emergency_stop:collision"


def test_a_disabled_supervisor_ignores_every_hazard():
    control, reason = gate({"collision": True, "offroad": True}, enabled=False)
    assert reason == "" and control is _INPUT


def test_the_stuck_release_threshold_is_configurable_and_never_negative():
    for configured, accel, want_release in [(0.5, 0.4, False), (0.5, 0.5, True), (-3.0, 0.0, True)]:
        supervisor = SafetySupervisor(stuck_release_min_accel_mps2=configured)
        _, reason = supervisor.filter_control(
            control=_INPUT, make_pedal_control=_Control, safety_manager=_SafetyManager({"stuck": True}),
            behavior_decision="lane_follow", traffic_signal_state="green", planner_accel_mps2=accel)
        assert reason.startswith("safety_supervisor_release:") is want_release, (configured, accel)


def test_the_emergency_control_is_a_fresh_full_brake_and_becomes_the_rate_limit_baseline():
    supervisor = SafetySupervisor(stuck_release_min_accel_mps2=_THRESHOLD)
    control, _ = supervisor.filter_control(
        control=_INPUT, make_pedal_control=_Control, safety_manager=_SafetyManager({"collision": True}),
        behavior_decision="lane_follow", traffic_signal_state="green")
    assert control is not _INPUT and control.brake == 1.0
    assert supervisor._last_control is control
