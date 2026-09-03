from types import SimpleNamespace

from pipeline.control_finalization_stage import (
    ControlFinalizationRequest,
    ControlFinalizationStage,
)


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
