import pytest

from pipeline.mpc_cost_profile_stage import (
    MPCCostProfileStage,
    adaptive_target_horizon_s,
    select_profile_with_hysteresis,
)


class _MPC:
    adaptive_horizon_enabled = True

    def __init__(self):
        self.applied = []
        self.cleared = 0
        self.horizons = []

    def apply_mode_cost_profile(self, profile):
        self.applied.append(profile)
        return profile

    def clear_previous_solution_seed(self):
        self.cleared += 1

    def blend_toward_horizon_s(self, horizon_s):
        self.horizons.append(float(horizon_s))


def test_stage_owns_profile_hysteresis_seed_and_horizon_lifecycle():
    mpc = _MPC()
    stage = MPCCostProfileStage(
        mpc=mpc,
        config={"adaptive_horizon_profile_s": {"lane_follow": 3.0, "stop": 2.0}},
        behavior_runtime_config={"mpc_cost_profile_min_hold_s": 1.5},
    )

    stopped = stage.apply(
        behavior="stop_at_intersection", planner_lc_state="LANE_KEEP",
        planner_mode="NORMAL", next_macro_maneuver="lane_follow",
        reference_tracking_mode="", sim_time_s=0.1,
        nearest_obstacle_distance_m=2.0, ego_speed_mps=4.0,
    )
    held = stage.apply(
        behavior="lane_follow", planner_lc_state="LANE_KEEP",
        planner_mode="NORMAL", next_macro_maneuver="lane_follow",
        reference_tracking_mode="", sim_time_s=0.2,
        nearest_obstacle_distance_m=None, ego_speed_mps=0.0,
    )

    assert stopped.active == "stop"
    assert stopped.switch_reason == "safety_preempt"
    assert held.active == "stop"
    assert held.requested == "lane_follow"
    assert held.switch_reason == "hold_safety_profile"
    assert mpc.cleared == 1
    assert mpc.horizons == pytest.approx([2.0, 2.0])


def test_profile_helpers_keep_existing_adaptive_and_preemption_semantics():
    assert adaptive_target_horizon_s(
        mpc_cost_profile="lane_follow", nearest_obstacle_distance_m=3.0,
        ego_speed_mps=2.0, profile_horizon_s={"lane_follow": 4.0},
        obstacle_reference_speed_mps=2.0,
        obstacle_comfortable_decel_mps2=2.0,
    ) == pytest.approx(1.5)
    assert select_profile_with_hysteresis(
        requested_profile="recovery", active_profile="lane_follow",
        sim_time_s=0.1, active_since_s=0.0, min_hold_s=5.0,
    ) == ("recovery", 0.1, "safety_preempt")
