import unittest

from utility.speed_profile import trapezoidal_stop_profile


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


if __name__ == "__main__":
    unittest.main()
