import json
import os
import shutil
import sys
import tempfile
import unittest

PLANNING_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLANNING_MODULE_ROOT not in sys.path:
    sys.path.insert(0, PLANNING_MODULE_ROOT)

import import_mdrive_scenario as mdrive_import
from planning_runner import REQUIRED_CONSTRAINT_KEYS


EGO_ROUTE_XML = """<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="07_placement" town="Town05" role="ego" speed="8.0">
    <waypoint x="-100.0" y="50.0" z="0.0" yaw="0.0" />
    <waypoint x="-90.0" y="50.0" z="0.0" yaw="0.0" />
    <waypoint x="-80.0" y="50.0" z="0.0" yaw="0.0" />
  </route>
</routes>
"""

STATIC_ROUTE_XML_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<routes><route id="07_placement" model="vehicle.tesla.cybertruck" role="static" speed="0.00" town="Town05">
  <waypoint x="{x}" y="{y}" yaw="{yaw}" z="0.000000" />
</route></routes>
"""


class MDriveScenarioConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="mdrive_fixture_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.static_dir = os.path.join(self.tmp_dir, "actors", "static")
        os.makedirs(self.static_dir, exist_ok=True)

        with open(os.path.join(self.tmp_dir, "town05_vehicle_1_0.xml"), "w", encoding="utf-8") as file:
            file.write(EGO_ROUTE_XML)
        with open(os.path.join(self.static_dir, "entity_1.xml"), "w", encoding="utf-8") as file:
            file.write(STATIC_ROUTE_XML_TEMPLATE.format(x=-95.0, y=53.5, yaw=-179.9))
        with open(os.path.join(self.static_dir, "entity_2.xml"), "w", encoding="utf-8") as file:
            file.write(STATIC_ROUTE_XML_TEMPLATE.format(x=-92.0, y=53.5, yaw=-179.9))

        manifest = {
            "bicycle": [],
            "ego": [
                {
                    "file": "town05_vehicle_1_0.xml",
                    "kind": "ego",
                    "name": "Vehicle 1",
                    "route_id": "07_placement",
                    "speed": 8.0,
                    "town": "Town05",
                    "model": "vehicle.lincoln.mkz2017",
                    "target_speed": 8.0,
                },
                {
                    "file": "town05_vehicle_1_0.xml",
                    "kind": "ego",
                    "name": "Vehicle 2",
                    "route_id": "07_placement",
                    "speed": 8.0,
                    "town": "Town05",
                    "model": "vehicle.lincoln.mkz2017",
                    "target_speed": 8.0,
                },
            ],
            "npc": [
                {
                    "file": "town05_npc_1.xml",
                    "kind": "npc",
                    "name": "NPC 1",
                    "route_id": "07_placement",
                    "town": "Town05",
                }
            ],
            "pedestrian": [],
            "static": [
                {
                    "file": "actors/static/entity_1.xml",
                    "kind": "static",
                    "model": "vehicle.tesla.cybertruck",
                    "name": "entity_1",
                    "route_id": "07_placement",
                    "speed": 0.0,
                    "town": "Town05",
                    "target_speed": 0.0,
                },
                {
                    "file": "actors/static/entity_2.xml",
                    "kind": "static",
                    "model": "vehicle.tesla.cybertruck",
                    "name": "entity_2",
                    "route_id": "07_placement",
                    "speed": 0.0,
                    "town": "Town05",
                    "target_speed": 0.0,
                },
            ],
        }
        with open(os.path.join(self.tmp_dir, "actors_manifest.json"), "w", encoding="utf-8") as file:
            json.dump(manifest, file)

    def test_convert_scenario_extracts_ego_anchors_and_static_actors(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        config = result["config"]

        self.assertEqual(config["anchors"]["ego_spawn_xyz_yaw"], [-100.0, 50.0, 0.0, 0.0])
        self.assertEqual(config["anchors"]["final_destination_xyz_yaw"], [-80.0, 50.0, 0.0, 0.0])
        self.assertEqual(config["carla"]["map"], "/Game/Carla/Maps/Town05")

        static_actors = config["obstacles"]["static_actors"]
        self.assertEqual(len(static_actors), 2)
        self.assertEqual(static_actors[0]["blueprint"], "vehicle.tesla.cybertruck")
        self.assertEqual(static_actors[0]["role_name"], "entity_1")
        self.assertAlmostEqual(static_actors[0]["x"], -95.0)
        self.assertAlmostEqual(static_actors[0]["y"], 53.5)
        self.assertAlmostEqual(static_actors[0]["yaw"], -179.9)
        self.assertEqual(config["obstacles"]["spawner_module"], "utility.coordinate_obstacle_spawner")

    def test_convert_scenario_wires_cp_hazard_publisher_when_static_actors_present(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        config = result["config"]

        self.assertEqual(config["runtime"]["module"], "utility.coordinate_obstacle_spawner")
        self.assertIn("cooperative_message_trigger_distance_m", config["obstacles"])

    def test_convert_scenario_reports_skipped_actors(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        skipped = result["skipped"]

        self.assertEqual(skipped["extra_ego"], ["Vehicle 2"])
        self.assertEqual(skipped["npc"], ["NPC 1"])
        self.assertEqual(skipped["pedestrian"], [])
        self.assertEqual(skipped["bicycle"], [])

    def test_generated_config_has_all_required_constraint_keys(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        constraints = result["config"]["constraints"]

        for key in REQUIRED_CONSTRAINT_KEYS:
            self.assertIn(key, constraints)

    def test_write_scenario_creates_expected_yaml_file(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane_write")
        output_dir = tempfile.mkdtemp(prefix="scenario_out_")
        self.addCleanup(shutil.rmtree, output_dir, ignore_errors=True)
        original_scenario_dir = mdrive_import.CARLA_SCENARIO_DIR
        mdrive_import.CARLA_SCENARIO_DIR = output_dir
        try:
            output_path = mdrive_import.write_scenario("test_blocked_lane_write", result["config"])
        finally:
            mdrive_import.CARLA_SCENARIO_DIR = original_scenario_dir

        self.assertTrue(os.path.isfile(output_path))
        loaded = mdrive_import.load_yaml_file(output_path)
        self.assertEqual(loaded["name"], "test_blocked_lane_write")
        self.assertEqual(loaded["runner_module"], "planning_runner")

    def test_scenario_with_no_static_actors_omits_obstacles_block(self):
        manifest_path = os.path.join(self.tmp_dir, "actors_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["static"] = []
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_no_obstacles")
        self.assertNotIn("obstacles", result["config"])
        self.assertEqual(result["static_actor_count"], 0)
        self.assertNotIn("module", result["config"]["runtime"])


if __name__ == "__main__":
    unittest.main()
