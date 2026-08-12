import numpy as np
import unittest

from MPC.mpc import MPC


def _bare_mpc(
    *,
    horizon_steps: int,
    use_seed: bool = True,
    search_steps: int = 5,
    max_position_error_m: float = 5.0,
    max_heading_error_rad: float = 1.0,
    max_speed_error_mps: float = 5.0,
) -> MPC:
    mpc = object.__new__(MPC)
    mpc.nx = 4
    mpc.nu = 2
    mpc.horizon_steps = int(horizon_steps)
    mpc.reference_use_previous_solution_seed = bool(use_seed)
    mpc.reference_previous_solution_search_steps = int(search_steps)
    mpc.reference_previous_solution_max_position_error_m = float(max_position_error_m)
    mpc.reference_previous_solution_max_heading_error_rad = float(max_heading_error_rad)
    mpc.reference_previous_solution_max_speed_error_mps = float(max_speed_error_mps)
    mpc._previous_x_solution = None
    mpc._previous_u_solution = None
    return mpc


def _linear_x_solution(step_count: int, *, x0: float = 0.0, dx: float = 1.0, v: float = 5.0) -> np.ndarray:
    # [x, y, v, psi] per row, x advancing by dx each step -- a simple
    # cruising-straight trajectory, good enough to exercise the
    # nearest-stage search and index-clamped extrapolation.
    traj = np.zeros((step_count, 4), dtype=float)
    for k in range(step_count):
        traj[k] = [x0 + dx * k, 0.0, v, 0.0]
    return traj


def _constant_u_solution(step_count: int, *, accel: float = 0.5, steer: float = 0.0) -> np.ndarray:
    traj = np.zeros((step_count, 2), dtype=float)
    traj[:, 0] = accel
    traj[:, 1] = steer
    return traj


class MPCWarmStartSeedHorizonChangeTests(unittest.TestCase):
    def test_same_horizon_length_still_seeds(self):
        mpc = _bare_mpc(horizon_steps=10)
        mpc._previous_x_solution = _linear_x_solution(11)
        mpc._previous_u_solution = _constant_u_solution(10)

        seed = mpc._build_shifted_previous_solution_seed(x0=np.array([0.0, 0.0, 5.0, 0.0]))

        self.assertIsNotNone(seed)
        x_seed, u_seed = seed
        self.assertEqual(x_seed.shape, (11, 4))
        self.assertEqual(u_seed.shape, (10, 2))

    def test_grown_horizon_reuses_seed_by_repeating_last_stage(self):
        # Previous solve was 11 steps (horizon_steps=10); this tick's
        # horizon_steps grew to 15 (e.g. blend_toward_horizon_s ramping
        # lane_follow's ~3s toward execute_lane_change's ~4.5s). Before the
        # fix this returned None outright (a cold start); now it should
        # reuse the previous trajectory and pad the extra steps by
        # repeating its last stage.
        mpc = _bare_mpc(horizon_steps=15)
        mpc._previous_x_solution = _linear_x_solution(11, dx=1.0, v=5.0)
        mpc._previous_u_solution = _constant_u_solution(10, accel=0.5)

        seed = mpc._build_shifted_previous_solution_seed(x0=np.array([0.0, 0.0, 5.0, 0.0]))

        self.assertIsNotNone(seed)
        x_seed, u_seed = seed
        self.assertEqual(x_seed.shape, (16, 4))
        self.assertEqual(u_seed.shape, (15, 2))
        # Steps beyond the old horizon repeat the previous solution's last
        # available stage rather than falling back to zeros.
        last_available_x = mpc._previous_x_solution[-1]
        last_available_u = mpc._previous_u_solution[-1]
        np.testing.assert_allclose(x_seed[-1, :3], last_available_x[:3])
        np.testing.assert_allclose(u_seed[-1], last_available_u)
        self.assertFalse(np.allclose(x_seed[-1], 0.0))
        self.assertFalse(np.allclose(u_seed[-1], 0.0))

    def test_shrunk_horizon_reuses_seed_with_prefix(self):
        mpc = _bare_mpc(horizon_steps=5)
        mpc._previous_x_solution = _linear_x_solution(11, dx=1.0, v=5.0)
        mpc._previous_u_solution = _constant_u_solution(10, accel=0.5)

        seed = mpc._build_shifted_previous_solution_seed(x0=np.array([0.0, 0.0, 5.0, 0.0]))

        self.assertIsNotNone(seed)
        x_seed, u_seed = seed
        self.assertEqual(x_seed.shape, (6, 4))
        self.assertEqual(u_seed.shape, (5, 2))

    def test_mismatched_state_dimensionality_still_rejected(self):
        mpc = _bare_mpc(horizon_steps=10)
        mpc._previous_x_solution = np.zeros((11, 3), dtype=float)  # wrong nx
        mpc._previous_u_solution = _constant_u_solution(10)

        seed = mpc._build_shifted_previous_solution_seed(x0=np.array([0.0, 0.0, 5.0, 0.0]))

        self.assertIsNone(seed)

    def test_mismatched_control_dimensionality_still_rejected(self):
        mpc = _bare_mpc(horizon_steps=10)
        mpc._previous_x_solution = _linear_x_solution(11)
        mpc._previous_u_solution = np.zeros((10, 3), dtype=float)  # wrong nu

        seed = mpc._build_shifted_previous_solution_seed(x0=np.array([0.0, 0.0, 5.0, 0.0]))

        self.assertIsNone(seed)

    def test_disabled_flag_returns_none_regardless_of_shape(self):
        mpc = _bare_mpc(horizon_steps=10, use_seed=False)
        mpc._previous_x_solution = _linear_x_solution(11)
        mpc._previous_u_solution = _constant_u_solution(10)

        seed = mpc._build_shifted_previous_solution_seed(x0=np.array([0.0, 0.0, 5.0, 0.0]))

        self.assertIsNone(seed)

    def test_no_previous_solution_returns_none(self):
        mpc = _bare_mpc(horizon_steps=10)

        seed = mpc._build_shifted_previous_solution_seed(x0=np.array([0.0, 0.0, 5.0, 0.0]))

        self.assertIsNone(seed)


if __name__ == "__main__":
    unittest.main()
