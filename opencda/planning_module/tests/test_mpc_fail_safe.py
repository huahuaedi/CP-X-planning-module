import numpy as np
import unittest

from MPC.mpc import MPC, MPCComfortCostSpec, MPCConstraintSpec, MPCSafetyCostSpec


def _bare_mpc(
    *,
    consecutive_failures: int,
    gentle_brake_mps2: float = 2.0,
    emergency_threshold: int = 5,
    min_acceleration_mps2: float = -6.0,
) -> MPC:
    mpc = object.__new__(MPC)
    mpc.constraints = MPCConstraintSpec(
        min_velocity_mps=0.0,
        max_velocity_mps=20.0,
        min_acceleration_mps2=min_acceleration_mps2,
        max_acceleration_mps2=3.0,
        max_jerk_mps3=8.0,
        min_steer_rad=-0.6,
        max_steer_rad=0.6,
        min_steer_rate_rps=-3.0,
        max_steer_rate_rps=3.0,
        enforce_terminal_velocity_constraint=False,
        terminal_velocity_mps=0.0,
    )
    mpc.dt_s = 0.1
    mpc.horizon_steps = 10
    mpc.fail_safe_gentle_brake_deceleration_mps2 = float(gentle_brake_mps2)
    mpc.fail_safe_emergency_stop_failure_threshold = int(emergency_threshold)
    mpc._consecutive_solver_failure_count = int(consecutive_failures)
    return mpc


class MinimumReachableSpeedProfileBrakingMagnitudeTests(unittest.TestCase):
    def test_default_uses_full_min_acceleration_magnitude(self):
        mpc = _bare_mpc(consecutive_failures=0, min_acceleration_mps2=-6.0)

        profile = mpc._minimum_reachable_speed_profile_mps(
            current_speed_mps=10.0,
            current_acceleration_mps2=0.0,
        )

        # Jerk-limited ramp-up to full braking (-6 m/s^2) takes a few steps;
        # by the end of the 1s horizon the speed drop must reflect that full
        # braking magnitude, not a comfort brake.
        self.assertLess(profile[-1], 10.0 - 3.0)

    def test_custom_braking_magnitude_is_milder_than_default(self):
        mpc = _bare_mpc(consecutive_failures=0, min_acceleration_mps2=-6.0)

        gentle_profile = mpc._minimum_reachable_speed_profile_mps(
            current_speed_mps=10.0,
            current_acceleration_mps2=0.0,
            braking_deceleration_mps2=2.0,
        )
        full_profile = mpc._minimum_reachable_speed_profile_mps(
            current_speed_mps=10.0,
            current_acceleration_mps2=0.0,
            braking_deceleration_mps2=None,
        )

        # The gentle profile must never brake harder (lower speed) than the
        # full-braking profile at any horizon step.
        for gentle_v, full_v in zip(gentle_profile, full_profile):
            self.assertGreaterEqual(gentle_v + 1e-9, full_v)
        self.assertGreater(gentle_profile[-1], full_profile[-1])


class FailSafeFallbackTrajectoryTests(unittest.TestCase):
    def _rollout(self, horizon_steps: int, speed_mps: float = 10.0):
        x = np.zeros((horizon_steps + 1, 4), dtype=float)
        for k in range(horizon_steps + 1):
            x[k, 0] = float(k) * 1.0
            x[k, 1] = 2.5
            x[k, 2] = float(speed_mps)
            x[k, 3] = 0.3
        u = np.zeros((horizon_steps, 2), dtype=float)
        u[:, 1] = 0.05
        return x, u

    def test_below_threshold_brakes_gently_and_holds_rollout_path(self):
        mpc = _bare_mpc(consecutive_failures=1, gentle_brake_mps2=2.0, emergency_threshold=5)
        rollout_x, rollout_u = self._rollout(mpc.horizon_steps)
        x0 = np.array([0.0, 2.5, 10.0, 0.3])

        x_solution, u_solution = mpc._fail_safe_fallback_trajectory(
            x0=x0,
            rollout_x=rollout_x,
            rollout_u=rollout_u,
            current_acceleration_mps2=0.0,
        )

        # Path (x, y, heading) is held from the rollout unchanged.
        np.testing.assert_allclose(x_solution[:, 0], rollout_x[:, 0])
        np.testing.assert_allclose(x_solution[:, 1], rollout_x[:, 1])
        np.testing.assert_allclose(x_solution[:, 3], rollout_x[:, 3])
        # Speed is overridden to a decelerating profile, not left at 10 m/s.
        self.assertLess(x_solution[1, 2], 10.0)
        self.assertLess(x_solution[-1, 2], x_solution[1, 2])
        # Commanded acceleration is braking but within the gentle magnitude,
        # not slammed to min_acceleration_mps2.
        self.assertTrue(np.all(u_solution[:, 0] < 0.0))
        self.assertTrue(np.all(u_solution[:, 0] >= -2.0 - 1e-6))

    def test_at_or_above_threshold_applies_full_emergency_braking(self):
        mpc = _bare_mpc(
            consecutive_failures=5,
            gentle_brake_mps2=2.0,
            emergency_threshold=5,
            min_acceleration_mps2=-6.0,
        )
        rollout_x, rollout_u = self._rollout(mpc.horizon_steps)
        x0 = np.array([0.0, 2.5, 10.0, 0.3])

        x_solution, u_solution = mpc._fail_safe_fallback_trajectory(
            x0=x0,
            rollout_x=rollout_x,
            rollout_u=rollout_u,
            current_acceleration_mps2=0.0,
        )

        # Braking ramps up under the jerk limit, but by the end of the
        # horizon the commanded deceleration must reach the hard limit,
        # clearly stronger than the 2 m/s^2 gentle-brake case would allow.
        self.assertLess(float(u_solution[-1, 0]), -2.5)
        self.assertLess(float(x_solution[-1, 2]), 10.0 - 3.0)

    def test_commanded_acceleration_stays_within_hard_constraints(self):
        mpc = _bare_mpc(
            consecutive_failures=10,
            gentle_brake_mps2=2.0,
            emergency_threshold=5,
            min_acceleration_mps2=-6.0,
        )
        rollout_x, rollout_u = self._rollout(mpc.horizon_steps)
        x0 = np.array([0.0, 2.5, 10.0, 0.3])

        _, u_solution = mpc._fail_safe_fallback_trajectory(
            x0=x0,
            rollout_x=rollout_x,
            rollout_u=rollout_u,
            current_acceleration_mps2=0.0,
        )

        self.assertTrue(np.all(u_solution[:, 0] >= mpc.constraints.min_acceleration_mps2 - 1e-6))
        self.assertTrue(np.all(u_solution[:, 0] <= mpc.constraints.max_acceleration_mps2 + 1e-6))


class ModeCostProfileTests(unittest.TestCase):
    def test_apply_mode_cost_profile_updates_objective_weights(self):
        mpc = object.__new__(MPC)
        mpc.safety_cost = MPCSafetyCostSpec(w_safe=1.0)
        mpc.comfort_cost = MPCComfortCostSpec(
            w_comf=5.0,
            qx=2.0,
            qy=0.0,
            qv=2.0,
            qpsi=0.0,
            qa=5.0,
            qdelta=100.0,
        )
        mpc.lane_center_follow_weight = 20.0
        mpc.lane_center_follow_qpsi = 2.0
        mpc.road_boundary_weight = 10000.0
        mpc.road_boundary_margin_m = 0.5
        mpc.road_boundary_max_slack_m = 0.1
        mpc.lane_keep_boundary_weight = 10000.0
        mpc.mode_cost_profiles = {
            "intersection_turn": {
                "lane_center_w0": 36.0,
                "lane_center_q_psi": 3.0,
                "road_boundary_w": 16000.0,
                "w_control": 8.0,
            }
        }
        mpc.mode_cost_profile_blend_alpha = 1.0
        mpc._base_mode_cost_state = mpc._capture_mode_cost_state()

        applied = mpc.apply_mode_cost_profile("intersection_turn", blend_alpha=1.0)

        self.assertEqual(applied, "intersection_turn")
        self.assertAlmostEqual(mpc.lane_center_follow_weight, 36.0)
        self.assertAlmostEqual(mpc.lane_center_follow_qpsi, 3.0)
        self.assertAlmostEqual(mpc.road_boundary_weight, 16000.0)
        self.assertAlmostEqual(mpc.lane_keep_boundary_weight, 16000.0)
        self.assertAlmostEqual(mpc.comfort_cost.w_comf, 8.0)


if __name__ == "__main__":
    unittest.main()
