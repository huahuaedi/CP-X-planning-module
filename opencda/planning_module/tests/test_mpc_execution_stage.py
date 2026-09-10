from types import SimpleNamespace

import numpy as np

from pipeline.mpc_execution_stage import (
    MPCExecutionRequest,
    MPCExecutionStage,
)


class _Buffer:
    def __init__(self, *, replan=True, sample=None):
        self.replan = replan
        self.sample_value = sample
        self.reset_reason = ""
        self.updated = False
        self.last_replan_request = {}

    def reset(self, *, reason):
        self.reset_reason = reason

    def should_replan(self, **kwargs):
        self.last_replan_request = dict(kwargs)
        return bool(self.replan or kwargs.get("force_replan", False))

    def update_from_solution(self, **kwargs):
        self.updated = True

    def sample(self, **kwargs):
        return self.sample_value


class _MPC:
    dt_s = 0.1

    def __init__(self, *, status="solved"):
        self._last_status = status
        self._last_u_solution = np.array([[0.4, 0.1]])
        self._last_x_solution = np.array([[0.0, 0.0, 3.0, 0.0]])
        self.constraints = SimpleNamespace(
            min_acceleration_mps2=-4.0,
            max_jerk_mps3=10.0,
        )

    def plan_trajectory(self, **kwargs):
        return None


def _request(**overrides):
    values = dict(
        sim_time_s=1.0,
        current_state=[0.0, 0.0, 2.0, 0.0],
        destination_state=[5.0, 0.0, 4.0, 0.0, 1],
        reference_samples=[{"x_ref_m": 1.0, "y_ref_m": 0.0}],
        object_snapshots=[],
        current_acceleration_mps2=0.0,
        current_steering_rad=0.0,
        ego_speed_mps=2.0,
        target_speed_mps=4.0,
        stop_goal_active=False,
        behavior_maneuver="lane_follow",
        behavior_phase="LANE_KEEP",
        hard_gate_reason="",
        stationary_stop_hold=False,
        control_context=SimpleNamespace(
            key="lane_follow", reference_anchor_relative_m=(1.0, 0.0),
            force_replan=False,
        ),
    )
    values.update(overrides)
    return MPCExecutionRequest(**values)


def _run(stage, request):
    return stage.run(
        request,
        normal_stop_control=lambda: "hold",
        safe_stop_control=lambda acceleration, steering: (
            "safe-stop", acceleration, steering,
        ),
        emergency_stop_control=lambda: "emergency-stop",
    )


def test_solved_plan_updates_buffer_and_returns_first_control():
    buffer = _Buffer(replan=True)
    result = _run(MPCExecutionStage(mpc=_MPC(), control_buffer=buffer), _request())
    assert result.status == "solved"
    assert result.acceleration_mps2 == 0.4
    assert result.steering_rad == 0.1
    assert buffer.updated


def test_constraint_revision_change_forces_immediate_resolve():
    buffer = _Buffer(replan=False, sample=(0.0, 0.0, "cached"))
    stage = MPCExecutionStage(mpc=_MPC(), control_buffer=buffer)
    first = _run(stage, _request(constraint_revision="corridor:1"))
    assert first.replan_executed
    assert buffer.last_replan_request["force_replan"]
    second = _run(stage, _request(
        sim_time_s=1.05, constraint_revision="corridor:1"
    ))
    assert not second.replan_executed


def test_failed_solve_reuses_only_valid_buffered_control():
    buffer = _Buffer(replan=True, sample=(0.2, -0.05, "valid"))
    result = _run(
        MPCExecutionStage(mpc=_MPC(status="primal infeasible"), control_buffer=buffer),
        _request(),
    )
    assert result.status == "buffer_reuse_after_failed_replan"
    assert result.failed_replan_buffer_reused
    assert result.control is None


def test_failed_solve_without_buffer_uses_bounded_safe_stop():
    result = _run(
        MPCExecutionStage(
            mpc=_MPC(status="primal infeasible"), control_buffer=_Buffer(replan=True)
        ),
        _request(),
    )
    assert result.status == "bounded_safe_stop"
    assert result.control == ("safe-stop", -1.0, 0.0)
    assert result.acceleration_mps2 == -1.0


def test_hard_gate_resets_buffer_and_uses_emergency_stop():
    buffer = _Buffer(replan=False, sample=(0.2, 0.1, "stale"))
    result = _run(
        MPCExecutionStage(mpc=_MPC(), control_buffer=buffer),
        _request(hard_gate_reason="candidate_hard_gate:reference_invalid"),
    )
    assert result.status == "candidate_hard_gate"
    assert result.control == "emergency-stop"
    assert buffer.reset_reason == "control_buffer_reference_hard_veto"


def test_fallback_jerk_uses_last_planned_acceleration_and_tick_elapsed_time():
    buffer = _Buffer(replan=True)
    mpc = _MPC(status="solved")
    stage = MPCExecutionStage(mpc=mpc, control_buffer=buffer)
    solved = _run(stage, _request(sim_time_s=1.0))
    assert solved.acceleration_mps2 == 0.4

    mpc._last_status = "primal infeasible"
    failed = _run(stage, _request(
        sim_time_s=1.05,
        current_acceleration_mps2=-4.0,
    ))
    # The OpenCDA PID's pedal-equivalent -4.0 input is not MPC memory.
    # With j_max=10 and a 0.05 s tick, 0.4 may only fall to -0.1.
    assert abs(failed.jerk_seed_acceleration_mps2 - 0.4) < 1.0e-9
    assert abs(failed.acceleration_mps2 - (-0.1)) < 1.0e-9
