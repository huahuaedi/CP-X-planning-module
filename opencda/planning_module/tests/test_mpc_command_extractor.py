import unittest

from pipeline.mpc_command_extractor import MPCCommandExtractor


class MPCCommandExtractorTest(unittest.TestCase):
    def test_interpolates_velocity_at_preview_time(self):
        extractor = MPCCommandExtractor(preview_time_s=0.25)
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
        extractor = MPCCommandExtractor(preview_time_s=0.6)
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


if __name__ == "__main__":
    unittest.main()
