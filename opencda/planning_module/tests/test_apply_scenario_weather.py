import os
import sys
import unittest

PLANNING_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLANNING_MODULE_ROOT not in sys.path:
    sys.path.insert(0, PLANNING_MODULE_ROOT)

from planning_runner import _apply_scenario_weather


class _FakeWeatherParameters:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _FakeCarlaModule:
    WeatherParameters = _FakeWeatherParameters


class _FakeWorld:
    def __init__(self):
        self.applied_weather = None

    def set_weather(self, weather_parameters):
        self.applied_weather = weather_parameters


class ApplyScenarioWeatherTests(unittest.TestCase):
    def test_applies_recognized_weather_keys(self):
        world = _FakeWorld()
        weather_cfg = {
            "cloudiness": 20.0,
            "sun_altitude_angle": -15.0,
            "fog_density": 5.0,
        }

        _apply_scenario_weather(world, _FakeCarlaModule(), weather_cfg)

        self.assertIsNotNone(world.applied_weather)
        self.assertEqual(world.applied_weather.kwargs["cloudiness"], 20.0)
        self.assertEqual(world.applied_weather.kwargs["sun_altitude_angle"], -15.0)
        self.assertEqual(world.applied_weather.kwargs["fog_density"], 5.0)

    def test_ignores_unrecognized_keys(self):
        world = _FakeWorld()
        weather_cfg = {"cloudiness": 20.0, "id": "night", "not_a_real_field": 1.0}

        _apply_scenario_weather(world, _FakeCarlaModule(), weather_cfg)

        self.assertNotIn("id", world.applied_weather.kwargs)
        self.assertNotIn("not_a_real_field", world.applied_weather.kwargs)

    def test_noop_when_weather_cfg_is_none(self):
        world = _FakeWorld()

        _apply_scenario_weather(world, _FakeCarlaModule(), None)

        self.assertIsNone(world.applied_weather)

    def test_noop_when_weather_cfg_is_empty(self):
        world = _FakeWorld()

        _apply_scenario_weather(world, _FakeCarlaModule(), {})

        self.assertIsNone(world.applied_weather)

    def test_does_not_raise_when_set_weather_fails(self):
        class _BrokenWorld:
            def set_weather(self, _weather_parameters):
                raise RuntimeError("boom")

        # Should print a warning and return, not propagate the exception.
        _apply_scenario_weather(_BrokenWorld(), _FakeCarlaModule(), {"cloudiness": 20.0})


if __name__ == "__main__":
    unittest.main()
