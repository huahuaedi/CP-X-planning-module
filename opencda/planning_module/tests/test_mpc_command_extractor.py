import unittest

from pipeline.mpc_command_extractor import MPCCommandExtractor


class MPCCommandExtractorTest(unittest.TestCase):
    def test_preview_is_safe_default_not_terminal_state(self):
        extractor = MPCCommandExtractor(preview_time_s=0.2)
        command = extractor.extract(
            state_solution=[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 12.0, 0.0],
            ],
            steering_rad=0.0,
            solution_dt_s=0.1,
            timestamp_s=1.0,
            max_velocity_mps=15.0,
        )
        self.assertAlmostEqual(command.target_velocity_mps, 2.0)
        self.assertEqual(command.reason, "mpc_optimized_velocity_preview")

    def test_interpolates_velocity_at_preview_time(self):
        extractor = MPCCommandExtractor(
            preview_time_s=0.25, velocity_source="preview"
        )
        command = extractor.extract(
            state_solution=[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ],
            steering_rad=0.1,
            solution_dt_s=0.1,
            timestamp_s=1.0,
            max_velocity_mps=10.0,
        )
        self.assertAlmostEqual(command.target_velocity_mps, 2.5)
        self.assertAlmostEqual(command.velocity_source_index, 2.5)
        self.assertAlmostEqual(command.target_steering_rad, 0.1)

    def test_does_not_use_immediate_next_state_as_launch_setpoint(self):
        extractor = MPCCommandExtractor(
            preview_time_s=0.6, velocity_source="preview"
        )
        solution = [[0.0, 0.0, 0.3 * index, 0.0] for index in range(11)]
        command = extractor.extract(
            state_solution=solution,
            steering_rad=0.0,
            solution_dt_s=0.1,
            timestamp_s=1.0,
            max_velocity_mps=10.0,
        )
        self.assertAlmostEqual(command.target_velocity_mps, 1.8)
        self.assertGreater(command.target_velocity_mps, solution[1][2])

    def test_command_change_respects_mpc_acceleration_limits(self):
        extractor = MPCCommandExtractor(
            preview_time_s=0.2,
            min_acceleration_mps2=-2.0,
            max_acceleration_mps2=3.0,
        )
        extractor.extract(
            state_solution=[[0.0, 0.0, 2.0, 0.0]] * 4,
            steering_rad=0.0,
            solution_dt_s=0.1,
            timestamp_s=1.0,
            max_velocity_mps=10.0,
        )
        command = extractor.extract(
            state_solution=[[0.0, 0.0, 8.0, 0.0]] * 4,
            steering_rad=0.0,
            solution_dt_s=0.1,
            timestamp_s=1.1,
            max_velocity_mps=10.0,
        )
        self.assertAlmostEqual(command.target_velocity_mps, 2.3)

    def test_horizon_velocity_is_default_platform_target(self):
        extractor = MPCCommandExtractor(
            preview_time_s=0.2, velocity_source="horizon"
        )
        command = extractor.extract(
            state_solution=[
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 9.0, 0.0],
                [0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 12.0, 0.0],
            ],
            steering_rad=0.0,
            solution_dt_s=0.1,
            timestamp_s=1.0,
            max_velocity_mps=15.0,
        )
        self.assertAlmostEqual(command.target_velocity_mps, 12.0)
        self.assertAlmostEqual(command.velocity_source_index, 3.0)
        self.assertEqual(command.reason, "mpc_optimized_horizon_velocity")

    def test_failed_cycle_holds_last_valid_velocity(self):
        extractor = MPCCommandExtractor(preview_time_s=0.6)
        extractor.extract(
            state_solution=[[0.0, 0.0, 3.0, 0.0]] * 8,
            steering_rad=0.0,
            solution_dt_s=0.1,
            timestamp_s=1.0,
            max_velocity_mps=10.0,
        )
        held = extractor.hold(
            steering_rad=0.2,
            reason="mpc_command_hold_buffer_reuse",
        )
        self.assertTrue(held.valid)
        self.assertAlmostEqual(held.target_velocity_mps, 3.0)
        self.assertAlmostEqual(held.target_steering_rad, 0.2)

    def test_platform_pid_uses_nominal_not_near_term_mpc_state(self):
        target = MPCCommandExtractor.platform_target_velocity(
            nominal_velocity_mps=3.0,
            stop_goal_active=False,
            emergency_stop=False,
        )

        self.assertEqual(target, 3.0)

    def test_explicit_mpc_safety_cap_may_only_lower_nominal(self):
        self.assertEqual(
            MPCCommandExtractor.platform_target_velocity(
                nominal_velocity_mps=3.0,
                stop_goal_active=False,
                emergency_stop=False,
                mpc_safety_cap_mps=1.25,
            ),
            1.25,
        )
        self.assertEqual(
            MPCCommandExtractor.platform_target_velocity(
                nominal_velocity_mps=3.0,
                stop_goal_active=False,
                emergency_stop=False,
                mpc_safety_cap_mps=5.0,
            ),
            3.0,
        )

    def test_stop_contract_overrides_all_velocity_owners(self):
        self.assertEqual(
            MPCCommandExtractor.platform_target_velocity(
                nominal_velocity_mps=3.0,
                stop_goal_active=True,
                emergency_stop=False,
                mpc_safety_cap_mps=2.0,
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
