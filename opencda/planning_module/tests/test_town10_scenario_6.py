import unittest

from main import list_available_scenarios, load_any_scenario


class Town10Scenario6Tests(unittest.TestCase):
    def test_scenario_is_available_and_uses_town10_scenario_6_modules(self):
        self.assertIn("town10_scenario_6", list_available_scenarios())

        scenario_cfg = load_any_scenario("town10_scenario_6")

        self.assertEqual(
            str(scenario_cfg.get("name", "")),
            "town10_scenario_6",
        )
        self.assertEqual(
            str(scenario_cfg.get("runner_module", "")),
            "opencda_scenario.town10_scenario_6.runner",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "opencda_scenario.town10_scenario_6.scenario",
        )
        self.assertTrue(
            str(scenario_cfg.get("_scenario_path", "")).endswith(
                "/opencda_scenario/town10_scenario_6/town10_scenario_6.yaml"
            )
        )


if __name__ == "__main__":
    unittest.main()
