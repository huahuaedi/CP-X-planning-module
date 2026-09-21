from types import SimpleNamespace

from pipeline.control_finalization_stage import (
    ControlFinalizationRequest,
    ControlFinalizationStage,
)


class _TrackingExtractor:
    def __init__(self, optimized_velocity_mps):
        self.optimized_velocity_mps = float(optimized_velocity_mps)
        self.received_cap = "not-called"

    def extract(self, **_kwargs):
        return SimpleNamespace(
            target_velocity_mps=self.optimized_velocity_mps,
            target_steering_rad=0.1,
            velocity_preview_time_s=0.6,
            velocity_source_index=6.0,
            valid=True,
        )

    def platform_target_velocity(
        self, *, nominal_velocity_mps, stop_goal_active, emergency_stop,
        mpc_safety_cap_mps=None,
    ):
        self.received_cap = mpc_safety_cap_mps
        if stop_goal_active or emergency_stop:
            return 0.0
        if mpc_safety_cap_mps is None:
            return float(nominal_velocity_mps)
        return min(float(nominal_velocity_mps), float(mpc_safety_cap_mps))


class _AcceptingSafety:
    def run(self, **kwargs):
        return SimpleNamespace(
            control=kwargs["control"],
            acceleration_mps2=0.0,
            steering_rad=0.1,
            pre_filter_acceleration_mps2=0.0,
            pre_filter_steering_rad=0.1,
            control_guard_reason="",
            boundary_guard_reason="",
            boundary_snapshot=None,
            supervisor_reason="accepted",
        )


class _Feedback:
    def record_result(self, **_kwargs):
        return "feedback:success"


def _run_solved_finalization(*, safety_cap_active):
    extractor = _TrackingExtractor(optimized_velocity_mps=1.5)
    mpc = SimpleNamespace(
        constraints=SimpleNamespace(
            min_acceleration_mps2=-3.0,
            max_velocity_mps=20.0,
        ),
        dt_s=0.1,
        _last_x_solution=[[0.0, 0.0, 1.5, 0.0]],
    )
    stage = ControlFinalizationStage(
        mpc=mpc,
        command_extractor=extractor,
        feedback=_Feedback(),
        control_safety=_AcceptingSafety(),
        config={},
    )
    request = ControlFinalizationRequest(
        execution=SimpleNamespace(
            acceleration_mps2=0.0,
            steering_rad=0.1,
            control=None,
            fallback_reason="",
            status="solved",
            failed_replan_buffer_reused=False,
        ),
        behavior=SimpleNamespace(maneuver="lane_follow", target_lane_id=1),
        ego_transform=object(),
        ego_speed_mps=1.0,
        target_speed_mps=3.0,
        destination_state=(),
        reference_samples=(),
        stop_goal_active=False,
        stop_target_forward_m="",
        final_reference_accepted=True,
        candidate_status="feasible",
        safety_manager=None,
        make_pedal_control=lambda **pedals: SimpleNamespace(**pedals),
        sim_time_s=1.0,
        mpc_velocity_safety_cap_active=bool(safety_cap_active),
    )
    applied = {}

    def apply_velocity_steering(**kwargs):
        applied.update(kwargs)
        return "pid-control", 0.0, 0.1, {}

    result = stage.run(
        request,
        set_actuator_context=lambda **_kwargs: None,
        acceleration_from_control=lambda _control: 0.0,
        steering_from_control=lambda _control: 0.1,
        apply_velocity_steering=apply_velocity_steering,
        control_factory=lambda *_args: None,
        boundary_metrics=lambda *_args: {},
        update_boundary_recovery=lambda *_args: None,
        reset_boundary_recovery=lambda *_args: None,
    )
    return extractor, applied, result


def test_clear_scene_pid_tracks_speed_planner_without_mpc_preview_cap():
    extractor, applied, result = _run_solved_finalization(
        safety_cap_active=False
    )

    assert extractor.received_cap is None
    assert applied["target_speed_mps"] == 3.0
    assert result.platform_debug["velocity_command_source"] == (
        "speed_target_planner"
    )


def test_longitudinal_corridor_may_apply_mpc_preview_safety_cap():
    extractor, applied, result = _run_solved_finalization(
        safety_cap_active=True
    )

    assert extractor.received_cap == 1.5
    assert applied["target_speed_mps"] == 1.5
    assert result.platform_debug["velocity_command_source"] == (
        "mpc_corridor_velocity_safety_cap"
    )


def _run_solved_finalization_with_corridor_escalation(*, corridor_infeasible_escalate):
    extractor = _TrackingExtractor(optimized_velocity_mps=1.5)
    mpc = SimpleNamespace(
        constraints=SimpleNamespace(min_acceleration_mps2=-3.0, max_velocity_mps=20.0),
        dt_s=0.1,
        _last_x_solution=[[0.0, 0.0, 1.5, 0.0]],
    )
    stage = ControlFinalizationStage(
        mpc=mpc,
        command_extractor=extractor,
        feedback=_Feedback(),
        control_safety=_AcceptingSafety(),
        config={},
        # The escalation must fire on its own signal, not by piggybacking on
        # the hard-gate path -- this always returns False so a passing test
        # can't be explained by that other trigger.
    )
    request = ControlFinalizationRequest(
        execution=SimpleNamespace(
            acceleration_mps2=0.0, steering_rad=0.1, control=None,
            fallback_reason="", status="solved",
            failed_replan_buffer_reused=False,
        ),
        # A non-emergency maneuver -- the escalation must not depend on the
        # behavior FSM having already recognized the hazard.
        behavior=SimpleNamespace(maneuver="lane_follow", target_lane_id=1),
        ego_transform=object(), ego_speed_mps=1.0, target_speed_mps=3.0,
        destination_state=(), reference_samples=(), stop_goal_active=False,
        stop_target_forward_m="", final_reference_accepted=True,
        candidate_status="feasible", safety_manager=None,
        make_pedal_control=lambda **pedals: SimpleNamespace(**pedals), sim_time_s=1.0,
        corridor_infeasible_escalate=bool(corridor_infeasible_escalate),
    )
    applied = {}

    def apply_velocity_steering(**kwargs):
        applied.update(kwargs)
        return "pid-control", 0.0, 0.1, {}

    result = stage.run(
        request,
        set_actuator_context=lambda **_kwargs: None,
        acceleration_from_control=lambda _control: 0.0,
        steering_from_control=lambda _control: 0.1,
        apply_velocity_steering=apply_velocity_steering,
        control_factory=lambda *_args: None,
        boundary_metrics=lambda *_args: {},
        update_boundary_recovery=lambda *_args: None,
        reset_boundary_recovery=lambda *_args: None,
    )
    return applied, result


def test_corridor_infeasible_escalation_forces_full_emergency_stop():
    applied, result = _run_solved_finalization_with_corridor_escalation(
        corridor_infeasible_escalate=True
    )

    assert applied["emergency_stop"] is True
    assert result.platform_debug["corridor_infeasible_escalate"] is True


def test_no_corridor_escalation_leaves_normal_pid_tracking_untouched():
    applied, result = _run_solved_finalization_with_corridor_escalation(
        corridor_infeasible_escalate=False
    )

    assert applied["emergency_stop"] is False
    assert applied["target_speed_mps"] == 3.0
    assert result.platform_debug["corridor_infeasible_escalate"] is False


def test_failed_mpc_control_goes_through_one_safety_exit_without_pid_remap():
    feedback_calls = []

    class Feedback:
        def record_result(self, **kwargs):
            feedback_calls.append(kwargs)
            return "feedback:failure"

    class Safety:
        def run(self, **kwargs):
            assert kwargs["control"] == "bounded-stop"
            return SimpleNamespace(
                control="safe-bounded-stop",
                acceleration_mps2=-0.5,
                steering_rad=0.1,
                pre_filter_acceleration_mps2=-0.5,
                pre_filter_steering_rad=0.1,
                control_guard_reason="",
                boundary_guard_reason="",
                boundary_snapshot=None,
                supervisor_reason="accepted",
            )

    mpc = SimpleNamespace(
        constraints=SimpleNamespace(
            min_acceleration_mps2=-3.0,
            max_velocity_mps=20.0,
        ),
        dt_s=0.1,
    )
    stage = ControlFinalizationStage(
        mpc=mpc,
        command_extractor=object(),
        feedback=Feedback(),
        control_safety=Safety(),
        config={},
    )
    request = ControlFinalizationRequest(
        execution=SimpleNamespace(
            acceleration_mps2=-0.5,
            steering_rad=0.1,
            control="bounded-stop",
            fallback_reason="MPC status=primal infeasible",
            status="bounded_safe_stop",
            failed_replan_buffer_reused=False,
        ),
        behavior=SimpleNamespace(
            maneuver="lane_follow", target_lane_id=1
        ),
        ego_transform=object(),
        ego_speed_mps=2.0,
        target_speed_mps=5.0,
        destination_state=(),
        reference_samples=(),
        stop_goal_active=False,
        stop_target_forward_m="",
        final_reference_accepted=True,
        candidate_status="feasible",
        safety_manager=None,
        make_pedal_control=lambda **pedals: SimpleNamespace(**pedals),
        sim_time_s=1.0,
    )
    result = stage.run(
        request,
        set_actuator_context=lambda **_kwargs: None,
        acceleration_from_control=lambda _control: -0.5,
        steering_from_control=lambda _control: 0.1,
        apply_velocity_steering=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("PID remap must not run for bounded fallback")
        ),
        control_factory=lambda *_args: None,
        boundary_metrics=lambda *_args: {},
        update_boundary_recovery=lambda *_args: None,
        reset_boundary_recovery=lambda *_args: None,
    )
    assert result.control == "safe-bounded-stop"
    assert result.feedback_reason == "feedback:failure"
    assert len(feedback_calls) == 1


# --- which execution results can put a control on the plant ------------------
#
# ControlFinalizationStage keeps ``execution.control`` only for a bounded safe
# stop.  In every other case (clean solve, stop hold, hard gate, buffer reuse)
# the platform PID path builds the control and ``execution.control`` is thrown
# away, so the emergency decision belongs to this stage alone.  These pin that:
# the final control must not depend on what the execution stage put there.

import pytest


class _HoldingExtractor(_TrackingExtractor):
    def hold(self, **_kwargs):
        return SimpleNamespace(
            target_velocity_mps=0.0, target_steering_rad=0.0,
            velocity_preview_time_s=0.0, velocity_source_index=0.0, valid=False,
        )


def _finalize(*, status, fallback_reason, control, buffer_reused=False, stop_goal=False,
              maneuver="lane_follow"):
    mpc = SimpleNamespace(
        constraints=SimpleNamespace(min_acceleration_mps2=-3.0, max_velocity_mps=20.0),
        dt_s=0.1, _last_x_solution=None,
    )
    stage = ControlFinalizationStage(
        mpc=mpc, command_extractor=_HoldingExtractor(1.5), feedback=_Feedback(),
        control_safety=_AcceptingSafety(), config={},
    )
    request = ControlFinalizationRequest(
        execution=SimpleNamespace(
            acceleration_mps2=-1.0, steering_rad=0.0, control=control,
            fallback_reason=fallback_reason, status=status,
            failed_replan_buffer_reused=buffer_reused,
        ),
        behavior=SimpleNamespace(maneuver=maneuver, target_lane_id=1),
        ego_transform=object(), ego_speed_mps=3.0, target_speed_mps=3.0,
        destination_state=(), reference_samples=(), stop_goal_active=stop_goal,
        stop_target_forward_m="", final_reference_accepted=True,
        candidate_status="ok", safety_manager=None,
        make_pedal_control=lambda **pedals: SimpleNamespace(**pedals), sim_time_s=1.0,
    )
    applied = {}

    def apply_velocity_steering(**kwargs):
        applied.update(kwargs)
        return "pid-control", 0.0, 0.0, {}

    result = stage.run(
        request,
        set_actuator_context=lambda **_kwargs: None,
        acceleration_from_control=lambda _control: -3.0,
        steering_from_control=lambda _control: 0.0,
        apply_velocity_steering=apply_velocity_steering,
        control_factory=lambda *_args: None,
        boundary_metrics=lambda *_args: {},
        update_boundary_recovery=lambda *_args: None,
        reset_boundary_recovery=lambda *_args: None,
    )
    return result, applied


_PID_CASES = [
    pytest.param(dict(status="stop_hold_direct", fallback_reason=""), id="stationary-stop-hold"),
    pytest.param(dict(status="candidate_hard_gate",
                      fallback_reason="candidate_hard_gate:turn_swept_footprint"),
                 id="hard-gate-geometry-veto"),
    pytest.param(dict(status="candidate_hard_gate",
                      fallback_reason="candidate_hard_gate:collision_risk"),
                 id="hard-gate-collision-risk"),
    pytest.param(dict(status="buffer_reuse_after_failed_replan",
                      fallback_reason="MPC status=infeasible", buffer_reused=True),
                 id="buffer-reused-after-failure"),
]


@pytest.mark.parametrize("case", _PID_CASES)
def test_the_platform_control_ignores_whatever_the_execution_stage_produced(case):
    with_control, applied_with = _finalize(control="STOP-CONTROL-FROM-EXECUTION", **case)
    without_control, applied_without = _finalize(control=None, **case)

    assert with_control.control == "pid-control" == without_control.control
    assert with_control.acceleration_mps2 == without_control.acceleration_mps2
    assert with_control.steering_rad == without_control.steering_rad
    assert applied_with == applied_without, "the PID request must not depend on execution.control"
    assert with_control.platform_debug == without_control.platform_debug


def test_emergency_is_decided_here_and_only_by_the_hazard_rule():
    geometry, applied_geometry = _finalize(
        control="EMERGENCY", status="candidate_hard_gate",
        fallback_reason="candidate_hard_gate:turn_swept_footprint")
    collision, applied_collision = _finalize(
        control=None, status="candidate_hard_gate",
        fallback_reason="candidate_hard_gate:collision_risk")

    assert applied_geometry["emergency_stop"] is False, "an emergency control upstream must not make a veto an emergency"
    assert applied_collision["emergency_stop"] is True
    assert geometry.control == collision.control == "pid-control"


# --- the finalization stage must hand the stop policy the real tick state -------

_VETO = "candidate_hard_gate:turn_swept_footprint"     # not a hazard on its own


def test_a_veto_becomes_an_emergency_when_the_maneuver_is_stop_like():
    _, applied = _finalize(control=None, status="candidate_hard_gate", fallback_reason=_VETO,
                           maneuver="stop_sign")
    assert applied["emergency_stop"] is True


def test_a_veto_becomes_an_emergency_when_a_stop_goal_is_active():
    _, applied = _finalize(control=None, status="candidate_hard_gate", fallback_reason=_VETO,
                           stop_goal=True)
    assert applied["emergency_stop"] is True


def test_a_veto_on_an_ordinary_maneuver_stays_a_non_emergency():
    _, applied = _finalize(control=None, status="candidate_hard_gate", fallback_reason=_VETO)
    assert applied["emergency_stop"] is False


@pytest.mark.parametrize("kwargs, reason", [
    (dict(status="candidate_hard_gate", fallback_reason="candidate_hard_gate:collision_risk"), "hard_gate_stop_hazard"),
    (dict(status="solved", fallback_reason="", maneuver="emergency_brake"), "emergency_brake_maneuver"),
    (dict(status="candidate_hard_gate", fallback_reason=_VETO), ""),
    (dict(status="stop_hold_direct", fallback_reason=""), ""),
])
def test_the_emergency_reason_is_reported_in_the_platform_debug(kwargs, reason):
    result, _ = _finalize(control=None, **kwargs)
    assert result.platform_debug["emergency_stop_reason"] == reason
