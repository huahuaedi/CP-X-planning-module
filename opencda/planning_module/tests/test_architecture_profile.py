import unittest

from pipeline.architecture_profile import normalize_architecture_config


class ArchitectureProfileTests(unittest.TestCase):
    def test_full_pipeline_has_single_reference_and_control_memory_owner(self):
        config, profile = normalize_architecture_config(
            {
                "mode": "full_cpx_mpc",
                "opencda_style_reference_conditioning_enabled": True,
                "low_speed_lateral_recovery_enabled": True,
                "lane_follow_speed_recovery_enabled": True,
                "lane_follow_negative_accel_release_enabled": True,
                "full_dense_traffic_lane_change_lock_enabled": True,
                "overspeed_guard_enabled": True,
                "full_low_speed_launch_enabled": True,
                "full_low_speed_launch_ramp_enabled": True,
                "full_low_speed_straight_steer_guard_enabled": True,
                "strict_reference_validator_veto_enabled": True,
            }
        )

        self.assertEqual(profile.name, "unified_velocity_steering_v1")
        self.assertTrue(config["velocity_steering_interface_enabled"])
        self.assertTrue(config["full_mpc_reference_stabilizer_enabled"])
        self.assertTrue(config["control_buffer_enabled"])
        self.assertTrue(config["safety_supervisor_enabled"])
        self.assertFalse(config["opencda_style_reference_conditioning_enabled"])
        self.assertFalse(config["low_speed_lateral_recovery_enabled"])
        self.assertFalse(config["lane_follow_speed_recovery_enabled"])
        self.assertFalse(config["lane_follow_negative_accel_release_enabled"])
        self.assertFalse(config["full_dense_traffic_lane_change_lock_enabled"])
        self.assertFalse(config["overspeed_guard_enabled"])
        self.assertFalse(config["full_low_speed_launch_enabled"])
        self.assertFalse(config["full_low_speed_launch_ramp_enabled"])
        self.assertFalse(config["full_low_speed_straight_steer_guard_enabled"])
        self.assertFalse(config["strict_reference_validator_veto_enabled"])
        self.assertFalse(config["boundary_recovery_enabled"])
        self.assertFalse(config["turn_road_boundary_speed_guard_enabled"])
        self.assertEqual(len(profile.normalized_overrides), 10)

    def test_removed_legacy_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            normalize_architecture_config({"mode": "opencda_reference_mpc"})


if __name__ == "__main__":
    unittest.main()
