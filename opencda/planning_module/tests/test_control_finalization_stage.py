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
        hard_gate_requires_emergency_stop=lambda **_kwargs: False,
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
        carla_module=object(),
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
        hard_gate_requires_emergency_stop=lambda **_kwargs: False,
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
        carla_module=object(), sim_time_s=1.0,
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
        hard_gate_requires_emergency_stop=lambda **_kwargs: False,
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
        carla_module=object(),
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
