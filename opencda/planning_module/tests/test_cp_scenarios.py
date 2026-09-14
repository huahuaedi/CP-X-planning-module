import unittest

from opencda_scenario import (
    list_available_scenarios,
    load_carla_scenario as load_any_scenario,
)


class CpScenarioConfigTests(unittest.TestCase):
    def test_cp_scenarios_are_available(self):
        available = set(list_available_scenarios())
        self.assertIn("town10_cp_roadway_object", available)
        self.assertIn("town10_cp_red_light_violator", available)
        self.assertIn("town10_cp_passing_merge", available)

    def test_cp_scenarios_use_config_driven_runtime(self):
        expected = {
            "town10_cp_roadway_object": ("D", "opencda_scenario.cp_scenarios"),
            "town10_cp_red_light_violator": (
                "A", "utility.coordinate_obstacle_spawner"
            ),
            "town10_cp_passing_merge": (
                "E", "utility.coordinate_obstacle_spawner"
            ),
        }
        for scenario_name, (expected_use_case, runtime_module) in expected.items():
            with self.subTest(scenario_name=scenario_name):
                scenario_cfg = load_any_scenario(scenario_name)
                self.assertEqual(
                    "opencda_scenario.runner",
                    str(scenario_cfg.get("runner_module", "")),
                )
                self.assertEqual(
                    runtime_module,
                    str(scenario_cfg.get("runtime", {}).get("module", "")),
                )
                self.assertEqual(
                    expected_use_case,
                    str(scenario_cfg.get("cp_scenario", {}).get("use_case", "")),
                )
                self.assertTrue(bool(scenario_cfg.get("anchors", {}).get("ego_spawn_xyz_yaw")))
                self.assertTrue(bool(scenario_cfg.get("anchors", {}).get("final_destination_xyz_yaw")))

    def test_cp_scenarios_define_relevant_objects(self):
        roadway_object = load_any_scenario("town10_cp_roadway_object")
        self.assertGreaterEqual(
            len(roadway_object.get("cp_scenario", {}).get("hazards", [])),
            1,
        )

        red_light = load_any_scenario("town10_cp_red_light_violator")
        self.assertEqual(
            len(red_light.get("obstacles", {}).get("npc_vehicles", [])), 1
        )

        passing_merge = load_any_scenario("town10_cp_passing_merge")
        self.assertEqual(
            len(passing_merge.get("obstacles", {}).get("npc_vehicles", [])), 2
        )


if __name__ == "__main__":
    unittest.main()
