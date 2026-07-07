import math
import unittest

from utility.speed_profile import (
    apply_sequential_speed_caps,
    curvature_speed_cap_mps,
    idm_following_speed_cap_mps,
    rate_limit_speed_cap_rise_mps,
    reference_jump_speed_cap_mps,
    stop_profile_speed_cap_mps,
    trapezoidal_stop_profile,
)


class SpeedProfileTests(unittest.TestCase):
    def test_stop_profile_does_not_self_lock_when_current_speed_is_near_zero(self):
        profile = trapezoidal_stop_profile(
            current_v=0.0,
            distance_to_stop_m=11.4,
            a_decel=2.5,
            stop_buffer_m=1.5,
            n_steps=3,
            dt_s=0.1,
        )

        self.assertGreater(profile[0], 6.0)
        self.assertGreater(profile[1], 6.0)

    def test_stop_profile_caps_overspeed_when_close_to_stop_line(self):
        profile = trapezoidal_stop_profile(
            current_v=10.0,
            distance_to_stop_m=3.0,
            a_decel=2.0,
            stop_buffer_m=1.5,
            n_steps=1,
            dt_s=0.1,
        )

        self.assertLess(profile[0], 3.0)


class IdmFollowingSpeedCapTests(unittest.TestCase):
    def test_returns_none_when_idm_is_not_braking(self):
        self.assertIsNone(
            idm_following_speed_cap_mps(
                idm_accel_mps2=None,
                idm_gap_m=10.0,
                is_fixed_stop=False,
                stop_decision_active=False,
                signal_state="unknown",
            )
        )
        self.assertIsNone(
            idm_following_speed_cap_mps(
                idm_accel_mps2=-0.2,
                idm_gap_m=10.0,
                is_fixed_stop=False,
                stop_decision_active=False,
                signal_state="unknown",
            )
        )

    def test_stop_context_uses_six_meter_buffer_and_skips_release_speed_floor(self):
        cap = idm_following_speed_cap_mps(
            idm_accel_mps2=-1.0,
            idm_gap_m=20.0,
            is_fixed_stop=True,
            stop_decision_active=False,
            signal_state="unknown",
            idm_speed_cap_braking_deceleration_mps2=2.5,
        )

        self.assertAlmostEqual(cap, math.sqrt(2.0 * 2.5 * (20.0 - 6.0)))

    def test_non_stop_context_applies_release_speed_and_lead_speed_floor(self):
        cap = idm_following_speed_cap_mps(
            idm_accel_mps2=-1.0,
            idm_gap_m=20.0,
            is_fixed_stop=False,
            stop_decision_active=False,
            signal_state="unknown",
            idm_non_stop_buffer_m=3.0,
            idm_speed_cap_braking_deceleration_mps2=2.5,
            idm_non_stop_min_speed_cap_mps=2.0,
            current_target_v_mps=5.0,
            idm_lead_v_mps=3.0,
        )

        expected = math.sqrt(2.0 * 2.5 * (20.0 - 3.0))
        self.assertAlmostEqual(cap, expected)


class StopProfileSpeedCapTests(unittest.TestCase):
    def test_returns_none_when_not_a_fixed_stop_or_no_target(self):
        self.assertIsNone(
            stop_profile_speed_cap_mps(
                is_fixed_stop=False,
                stop_target_distance_m=20.0,
                current_speed_mps=5.0,
                min_acceleration_mps2=-3.0,
            )
        )
        self.assertIsNone(
            stop_profile_speed_cap_mps(
                is_fixed_stop=True,
                stop_target_distance_m=None,
                current_speed_mps=5.0,
                min_acceleration_mps2=-3.0,
            )
        )

    def test_returns_none_when_stop_target_beyond_max_distance(self):
        self.assertIsNone(
            stop_profile_speed_cap_mps(
                is_fixed_stop=True,
                stop_target_distance_m=60.0,
                current_speed_mps=5.0,
                min_acceleration_mps2=-3.0,
                max_stop_target_distance_m=60.0,
            )
        )

    def test_matches_trapezoidal_profile_first_step(self):
        cap = stop_profile_speed_cap_mps(
            is_fixed_stop=True,
            stop_target_distance_m=20.0,
            current_speed_mps=5.0,
            min_acceleration_mps2=-3.0,
            stop_buffer_m=1.5,
        )

        self.assertAlmostEqual(cap, math.sqrt(2.0 * 3.0 * (20.0 - 1.5)))


class CurvatureSpeedCapTests(unittest.TestCase):
    def test_returns_none_when_too_slow_or_too_straight(self):
        self.assertIsNone(
            curvature_speed_cap_mps(
                curve_curvature_abs=0.05,
                curve_min_curvature=0.015,
                current_speed_mps=0.5,
            )
        )
        self.assertIsNone(
            curvature_speed_cap_mps(
                curve_curvature_abs=0.01,
                curve_min_curvature=0.015,
                current_speed_mps=5.0,
            )
        )

    def test_matches_lateral_accel_formula(self):
        cap = curvature_speed_cap_mps(
            curve_curvature_abs=0.05,
            curve_min_curvature=0.015,
            current_speed_mps=5.0,
            curve_lateral_accel_limit_mps2=1.3,
        )

        self.assertAlmostEqual(cap, math.sqrt(1.3 / 0.05))


class ReferenceJumpSpeedCapTests(unittest.TestCase):
    def test_returns_none_when_fixed_stop_or_jump_within_threshold(self):
        self.assertIsNone(
            reference_jump_speed_cap_mps(
                is_fixed_stop=True,
                last_reference_jump_m=5.0,
            )
        )
        self.assertIsNone(
            reference_jump_speed_cap_mps(
                is_fixed_stop=False,
                last_reference_jump_m=1.0,
                reference_jump_speed_cap_threshold_m=1.5,
            )
        )

    def test_configured_cap_is_floored_at_1_5_mps(self):
        cap = reference_jump_speed_cap_mps(
            is_fixed_stop=False,
            last_reference_jump_m=3.0,
            reference_jump_speed_cap_threshold_m=1.5,
            configured_cap_mps=1.0,
        )

        self.assertAlmostEqual(cap, 1.5)


class ApplySequentialSpeedCapsTests(unittest.TestCase):
    def test_only_strictly_tighter_candidates_apply(self):
        capped, applied = apply_sequential_speed_caps(
            10.0,
            [("a", 8.0, None), ("b", None, None), ("c", 12.0, None)],
        )

        self.assertAlmostEqual(capped, 8.0)
        self.assertEqual(applied, ["a"])

    def test_candidates_compare_against_running_value_not_base(self):
        capped, applied = apply_sequential_speed_caps(
            10.0,
            [("a", 6.0, None), ("b", 8.0, None)],
        )

        # "b" (8.0) is tighter than the original base (10.0) but not tighter
        # than the running value after "a" already tightened it to 6.0, so
        # it must not apply.
        self.assertAlmostEqual(capped, 6.0)
        self.assertEqual(applied, ["a"])

    def test_floor_is_applied_after_the_tightening_decision(self):
        # This reproduces the pre-existing curvature-cap quirk: a candidate
        # that is tighter than the running value can still raise it back up
        # once its floor is applied. Locking this down intentionally so a
        # future refactor doesn't "fix" it into a silent behavior change.
        capped, applied = apply_sequential_speed_caps(
            1.0,
            [("curvature", 0.3, 1.5)],
        )

        self.assertAlmostEqual(capped, 1.5)
        self.assertEqual(applied, ["curvature"])


class RateLimitSpeedCapRiseTests(unittest.TestCase):
    def test_drop_is_never_limited(self):
        # A tighter new cap (obstacle, tighter stop, emergency brake) must
        # always apply immediately regardless of dt or rise rate.
        capped = rate_limit_speed_cap_rise_mps(
            previous_cap_mps=7.7,
            new_cap_mps=2.0,
            dt_s=0.0,
            max_rise_mps2=2.0,
        )

        self.assertAlmostEqual(capped, 2.0)

    def test_rise_is_limited_by_rate_and_dt(self):
        # town10_scenario_6: cap crashed 7.7 -> 2.0, then the very next 0.25s
        # tick tried to jump back up to 3.26. Without rate limiting the MPC
        # sees an immediate re-acceleration invitation right after a hard
        # brake; this should instead ease back up by at most rate * dt.
        capped = rate_limit_speed_cap_rise_mps(
            previous_cap_mps=2.0,
            new_cap_mps=3.26,
            dt_s=0.25,
            max_rise_mps2=2.0,
        )

        self.assertAlmostEqual(capped, 2.5)

    def test_rise_within_budget_passes_through_unchanged(self):
        capped = rate_limit_speed_cap_rise_mps(
            previous_cap_mps=5.0,
            new_cap_mps=5.2,
            dt_s=0.25,
            max_rise_mps2=2.0,
        )

        self.assertAlmostEqual(capped, 5.2)


if __name__ == "__main__":
    unittest.main()
