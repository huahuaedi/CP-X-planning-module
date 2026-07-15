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
  <weather id="night" cloudiness="20.0" precipitation="0.0" precipitation_deposits="0.0" wind_intensity="5.0" sun_azimuth_angle="300.0" sun_altitude_angle="-15.0" wetness="10.0" fog_distance="80.0" fog_density="5.0" fog_falloff="2.0" />
</route>
</routes>
"""

EGO_ROUTE_XML_NO_WEATHER = """<?xml version='1.0' encoding='utf-8'?>
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

# Mirrors a real MDrive v2xpnp npc entry, which has no speed/target_speed
# field on the manifest entry itself -- exercises the importer's default
# speed fallback.
NPC_ROUTE_XML = """<?xml version="1.0" encoding="utf-8"?>
<routes>
  <route id="07_placement" town="Town05" role="npc">
    <waypoint x="-98.0" y="51.0" z="0.0" yaw="0.0" />
    <waypoint x="-88.0" y="51.0" z="0.0" yaw="0.0" />
  </route>
</routes>
"""

PEDESTRIAN_ROUTE_XML = """<?xml version="1.0" encoding="utf-8"?>
<routes>
  <route id="07_placement" town="Town05" role="pedestrian" speed="1.2">
    <waypoint x="-95.0" y="55.0" z="0.0" yaw="90.0" />
    <waypoint x="-95.0" y="50.0" z="0.0" yaw="90.0" />
    <waypoint x="-95.0" y="45.0" z="0.0" yaw="90.0" />
  </route>
</routes>
"""


class MDriveScenarioConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="mdrive_fixture_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.static_dir = os.path.join(self.tmp_dir, "actors", "static")
        os.makedirs(self.static_dir, exist_ok=True)

        with open(os.path.join(self.tmp_dir, "town05_vehicle_1_0.xml"), "w", encoding="utf-8") as file:
            file.write(EGO_ROUTE_XML)
        with open(os.path.join(self.tmp_dir, "town05_npc_1.xml"), "w", encoding="utf-8") as file:
            file.write(NPC_ROUTE_XML)
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

    def test_convert_scenario_uses_mdrive_ego_model_and_speed_with_margin(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        config = result["config"]

        self.assertEqual(config["ego"]["blueprint"], "vehicle.lincoln.mkz2017")
        # 1.25x margin over MDrive's own 8.0 m/s cruising speed.
        self.assertAlmostEqual(config["constraints"]["max_velocity_mps"], 10.0)

    def test_convert_scenario_extracts_weather_from_ego_route(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        config = result["config"]

        self.assertTrue(result["weather_converted"])
        weather = config["weather"]
        self.assertAlmostEqual(weather["cloudiness"], 20.0)
        self.assertAlmostEqual(weather["sun_altitude_angle"], -15.0)
        self.assertAlmostEqual(weather["fog_density"], 5.0)
        self.assertNotIn("id", weather)

    def test_convert_scenario_omits_weather_when_route_has_none(self):
        with open(os.path.join(self.tmp_dir, "town05_vehicle_1_0.xml"), "w", encoding="utf-8") as file:
            file.write(EGO_ROUTE_XML_NO_WEATHER)

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")

        self.assertFalse(result["weather_converted"])
        self.assertNotIn("weather", result["config"])

    def test_route_proximity_warning_empty_when_actors_are_close(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")

        self.assertEqual(result["route_proximity_warnings"], [])

    def test_route_proximity_warning_fires_for_far_static_actor(self):
        far_static_path = os.path.join(self.static_dir, "entity_far.xml")
        with open(far_static_path, "w", encoding="utf-8") as file:
            file.write(STATIC_ROUTE_XML_TEMPLATE.format(x=-95.0, y=150.0, yaw=-179.9))
        manifest_path = os.path.join(self.tmp_dir, "actors_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["static"].append(
            {
                "file": "actors/static/entity_far.xml",
                "kind": "static",
                "model": "vehicle.tesla.cybertruck",
                "name": "entity_far",
                "route_id": "07_placement",
                "speed": 0.0,
                "town": "Town05",
                "target_speed": 0.0,
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")

        warnings = result["route_proximity_warnings"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("entity_far", warnings[0])

    def _add_pedestrian_to_manifest(self, route_xml: str, *, file_name: str = "pedestrian_1.xml", name: str = "pedestrian_1"):
        with open(os.path.join(self.tmp_dir, file_name), "w", encoding="utf-8") as file:
            file.write(route_xml)
        manifest_path = os.path.join(self.tmp_dir, "actors_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["pedestrian"].append(
            {
                "file": file_name,
                "kind": "pedestrian",
                "model": "walker.pedestrian.0001",
                "name": name,
                "route_id": "07_placement",
                "speed": 1.2,
                "target_speed": 1.2,
                "town": "Town05",
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)

    def test_convert_scenario_converts_pedestrian_with_full_waypoint_list(self):
        self._add_pedestrian_to_manifest(PEDESTRIAN_ROUTE_XML)

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        config = result["config"]

        self.assertEqual(result["pedestrian_count"], 1)
        self.assertNotIn("pedestrian", result["skipped"])
        pedestrians = config["obstacles"]["pedestrians"]
        self.assertEqual(len(pedestrians), 1)
        self.assertEqual(pedestrians[0]["role_name"], "pedestrian_1")
        self.assertEqual(pedestrians[0]["blueprint"], "walker.pedestrian.0001")
        self.assertAlmostEqual(pedestrians[0]["speed_mps"], 1.2)
        self.assertEqual(len(pedestrians[0]["waypoints"]), 3)
        self.assertEqual(pedestrians[0]["waypoints"][0], [-95.0, 55.0, 0.0, 90.0])
        self.assertEqual(pedestrians[0]["waypoints"][-1], [-95.0, 45.0, 0.0, 90.0])
        # This pedestrian's path crosses the ego route (y=50) directly.
        self.assertEqual(result["route_proximity_warnings"], [])

    def test_convert_scenario_flags_pedestrian_route_that_never_nears_ego(self):
        far_pedestrian_xml = PEDESTRIAN_ROUTE_XML.replace('x="-95.0"', 'x="500.0"')
        self._add_pedestrian_to_manifest(far_pedestrian_xml, file_name="pedestrian_far.xml", name="pedestrian_far")

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")

        warnings = result["route_proximity_warnings"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("pedestrian_far", warnings[0])

    def test_obstacles_block_created_for_pedestrian_only_scenario(self):
        manifest_path = os.path.join(self.tmp_dir, "actors_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["static"] = []
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)
        self._add_pedestrian_to_manifest(PEDESTRIAN_ROUTE_XML)

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")

        self.assertIn("obstacles", result["config"])
        self.assertEqual(result["config"]["obstacles"]["spawner_module"], "utility.coordinate_obstacle_spawner")
        self.assertNotIn("static_actors", result["config"]["obstacles"])
        self.assertEqual(result["config"]["runtime"]["module"], "utility.coordinate_obstacle_spawner")

    def _add_bicycle_to_manifest(self, route_xml: str, *, file_name: str = "bicycle_1.xml", name: str = "bicycle_1"):
        with open(os.path.join(self.tmp_dir, file_name), "w", encoding="utf-8") as file:
            file.write(route_xml)
        manifest_path = os.path.join(self.tmp_dir, "actors_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["bicycle"].append(
            {
                "file": file_name,
                "kind": "bicycle",
                "model": "vehicle.bh.crossbike",
                "name": name,
                "route_id": "07_placement",
                "speed": 4.0,
                "target_speed": 4.0,
                "town": "Town05",
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)

    def test_convert_scenario_converts_bicycle_into_npc_vehicles_list(self):
        self._add_bicycle_to_manifest(NPC_ROUTE_XML)

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        config = result["config"]

        self.assertEqual(result["bicycle_count"], 1)
        self.assertNotIn("bicycle", result["skipped"])
        npc_vehicles = config["obstacles"]["npc_vehicles"]
        bicycle_entries = [entry for entry in npc_vehicles if entry["role_name"] == "bicycle_1"]
        self.assertEqual(len(bicycle_entries), 1)
        self.assertEqual(bicycle_entries[0]["blueprint"], "vehicle.bh.crossbike")
        self.assertEqual(bicycle_entries[0]["follow_mode"], "autopilot")
        self.assertAlmostEqual(bicycle_entries[0]["speed_mps"], 4.0)

    def test_convert_scenario_skips_nothing_in_the_interaction_bucket_shape(self):
        # extra ego / npc / pedestrian / bicycle are all converted (demoted
        # to autopilot where they can't run this project's own planner) --
        # nothing in a normal `interaction`-bucket manifest is dropped.
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")

        self.assertEqual(result["skipped"], {})

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

    def test_scenario_with_no_convertible_actors_omits_obstacles_block(self):
        manifest_path = os.path.join(self.tmp_dir, "actors_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["static"] = []
        manifest["npc"] = []
        manifest["ego"] = manifest["ego"][:1]
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file)

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_no_obstacles")
        self.assertNotIn("obstacles", result["config"])
        self.assertEqual(result["static_actor_count"], 0)
        self.assertEqual(result["npc_vehicle_count"], 0)
        self.assertNotIn("module", result["config"]["runtime"])

    def test_convert_scenario_converts_npc_with_default_speed_fallback(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        config = result["config"]

        npc_vehicles = config["obstacles"]["npc_vehicles"]
        npc_entries = [entry for entry in npc_vehicles if entry["role_name"] == "NPC 1"]
        self.assertEqual(len(npc_entries), 1)
        self.assertEqual(npc_entries[0]["blueprint"], "vehicle.tesla.model3")
        self.assertEqual(npc_entries[0]["follow_mode"], "autopilot")
        # NPC 1's manifest entry has neither speed nor target_speed -- falls
        # back to the importer's default.
        self.assertAlmostEqual(npc_entries[0]["speed_mps"], 8.0)
        self.assertEqual(len(npc_entries[0]["waypoints"]), 2)

    def test_convert_scenario_demotes_extra_ego_to_autopilot_npc(self):
        result = mdrive_import.convert_scenario(self.tmp_dir, "test_blocked_lane")
        config = result["config"]

        self.assertEqual(result["demoted_ego_count"], 1)
        npc_vehicles = config["obstacles"]["npc_vehicles"]
        demoted_entries = [entry for entry in npc_vehicles if entry["role_name"] == "Vehicle 2"]
        self.assertEqual(len(demoted_entries), 1)
        self.assertEqual(demoted_entries[0]["blueprint"], "vehicle.lincoln.mkz2017")
        self.assertEqual(demoted_entries[0]["follow_mode"], "autopilot")
        self.assertNotIn("extra_ego", result["skipped"])


class PrimaryEgoAutoSelectionTests(unittest.TestCase):
    """Covers `_select_primary_ego_index`: MDrive's `ego[0]` is not always
    the actor the scenario is actually built around (e.g. real
    `precrash/D`/`precrash/L` scenarios where a hazard/NPC sits right next
    to `ego[1]`'s route but tens of meters from `ego[0]`'s) -- the importer
    should pick whichever ego route sits closest overall to the other
    actors instead of blindly taking the manifest's first entry."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="mdrive_ego_select_fixture_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _write_ego_route(self, filename: str, *, y: float) -> None:
        with open(os.path.join(self.tmp_dir, filename), "w", encoding="utf-8") as file:
            file.write(
                f"""<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="07_placement" town="Town05" role="ego" speed="8.0">
    <waypoint x="-100.0" y="{y}" z="0.0" yaw="0.0" />
    <waypoint x="-90.0" y="{y}" z="0.0" yaw="0.0" />
    <waypoint x="-80.0" y="{y}" z="0.0" yaw="0.0" />
  </route>
</routes>
"""
            )

    def _write_manifest(self, ego_entries) -> None:
        with open(os.path.join(self.tmp_dir, "actors_manifest.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "bicycle": [],
                    "ego": ego_entries,
                    "npc": [
                        {
                            "file": "npc_1.xml",
                            "kind": "npc",
                            "name": "NPC 1",
                            "route_id": "07_placement",
                            "town": "Town05",
                        }
                    ],
                    "pedestrian": [],
                    "static": [],
                },
                file,
            )

    def test_selects_ego_whose_route_is_closest_to_other_actors(self):
        # ego[0]'s route is 40m away from the npc's route; ego[1]'s route
        # runs right alongside it -- the importer should auto-select ego[1],
        # mirroring the real precrash/L bug (npc_brake sits 0.0m from
        # ego[1] but 13m from ego[0]).
        self._write_ego_route("ego_0.xml", y=90.0)
        self._write_ego_route("ego_1.xml", y=50.0)
        with open(os.path.join(self.tmp_dir, "npc_1.xml"), "w", encoding="utf-8") as file:
            file.write(
                """<?xml version="1.0" encoding="utf-8"?>
<routes>
  <route id="07_placement" town="Town05" role="npc">
    <waypoint x="-98.0" y="50.0" z="0.0" yaw="0.0" />
    <waypoint x="-88.0" y="50.0" z="0.0" yaw="0.0" />
  </route>
</routes>
"""
            )
        self._write_manifest(
            [
                {
                    "file": "ego_0.xml",
                    "kind": "ego",
                    "name": "Vehicle 1",
                    "route_id": "07_placement",
                    "speed": 8.0,
                    "town": "Town05",
                    "model": "vehicle.lincoln.mkz2017",
                    "target_speed": 8.0,
                },
                {
                    "file": "ego_1.xml",
                    "kind": "ego",
                    "name": "Vehicle 2",
                    "route_id": "07_placement",
                    "speed": 8.0,
                    "town": "Town05",
                    "model": "vehicle.lincoln.mkz2017",
                    "target_speed": 8.0,
                },
            ]
        )

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_ego_select")

        self.assertTrue(result["auto_selected_ego"])
        self.assertEqual(result["primary_ego_name"], "Vehicle 2")
        self.assertEqual(result["config"]["planning"]["imported_route_waypoints"][0][1], 50.0)
        npc_vehicles = result["config"]["obstacles"]["npc_vehicles"]
        demoted_entries = [entry for entry in npc_vehicles if entry["role_name"] == "Vehicle 1"]
        self.assertEqual(len(demoted_entries), 1)

    def test_keeps_ego_zero_when_it_is_already_the_closest(self):
        self._write_ego_route("ego_0.xml", y=50.0)
        self._write_ego_route("ego_1.xml", y=90.0)
        with open(os.path.join(self.tmp_dir, "npc_1.xml"), "w", encoding="utf-8") as file:
            file.write(
                """<?xml version="1.0" encoding="utf-8"?>
<routes>
  <route id="07_placement" town="Town05" role="npc">
    <waypoint x="-98.0" y="50.0" z="0.0" yaw="0.0" />
    <waypoint x="-88.0" y="50.0" z="0.0" yaw="0.0" />
  </route>
</routes>
"""
            )
        self._write_manifest(
            [
                {
                    "file": "ego_0.xml",
                    "kind": "ego",
                    "name": "Vehicle 1",
                    "route_id": "07_placement",
                    "speed": 8.0,
                    "town": "Town05",
                    "model": "vehicle.lincoln.mkz2017",
                    "target_speed": 8.0,
                },
                {
                    "file": "ego_1.xml",
                    "kind": "ego",
                    "name": "Vehicle 2",
                    "route_id": "07_placement",
                    "speed": 8.0,
                    "town": "Town05",
                    "model": "vehicle.lincoln.mkz2017",
                    "target_speed": 8.0,
                },
            ]
        )

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_ego_select")

        self.assertFalse(result["auto_selected_ego"])
        self.assertEqual(result["primary_ego_name"], "Vehicle 1")

    def test_noop_with_a_single_ego_candidate(self):
        self._write_ego_route("ego_0.xml", y=50.0)
        with open(os.path.join(self.tmp_dir, "npc_1.xml"), "w", encoding="utf-8") as file:
            file.write(
                """<?xml version="1.0" encoding="utf-8"?>
<routes>
  <route id="07_placement" town="Town05" role="npc">
    <waypoint x="-98.0" y="90.0" z="0.0" yaw="0.0" />
    <waypoint x="-88.0" y="90.0" z="0.0" yaw="0.0" />
  </route>
</routes>
"""
            )
        self._write_manifest(
            [
                {
                    "file": "ego_0.xml",
                    "kind": "ego",
                    "name": "Vehicle 1",
                    "route_id": "07_placement",
                    "speed": 8.0,
                    "town": "Town05",
                    "model": "vehicle.lincoln.mkz2017",
                    "target_speed": 8.0,
                }
            ]
        )

        result = mdrive_import.convert_scenario(self.tmp_dir, "test_ego_select")

        self.assertFalse(result["auto_selected_ego"])
        self.assertEqual(result["primary_ego_name"], "Vehicle 1")
        self.assertEqual(result["demoted_ego_count"], 0)


def _write_minimal_scenario(scenario_dir: str, *, town: str = "Town05") -> None:
    os.makedirs(scenario_dir, exist_ok=True)
    with open(os.path.join(scenario_dir, "ego_route.xml"), "w", encoding="utf-8") as file:
        file.write(
            f"""<?xml version='1.0' encoding='utf-8'?>
<routes>
  <route id="07_placement" town="{town}" role="ego" speed="8.0">
    <waypoint x="0.0" y="0.0" z="0.0" yaw="0.0" />
    <waypoint x="10.0" y="0.0" z="0.0" yaw="0.0" />
  </route>
</routes>
"""
        )
    manifest = {
        "bicycle": [],
        "ego": [
            {
                "file": "ego_route.xml",
                "kind": "ego",
                "name": "Vehicle 1",
                "route_id": "07_placement",
                "speed": 8.0,
                "town": town,
                "model": "vehicle.lincoln.mkz2017",
                "target_speed": 8.0,
            }
        ],
        "npc": [],
        "pedestrian": [],
        "static": [],
    }
    with open(os.path.join(scenario_dir, "actors_manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file)


class BatchConversionTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mdrive_batch_root_")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        _write_minimal_scenario(os.path.join(self.root, "interaction", "Blocked_Lane_Obstacle", "1"))
        _write_minimal_scenario(os.path.join(self.root, "interaction", "Unprotected_Left_Turn", "1"))
        _write_minimal_scenario(os.path.join(self.root, "v2xpnp", "scenario_abc"), town="Town03")

    def test_discover_finds_scenarios_at_any_nesting_depth(self):
        discovered = mdrive_import._discover_mdrive_scenario_dirs(self.root)

        self.assertEqual(len(discovered), 3)
        self.assertTrue(any(path.endswith(os.path.join("Blocked_Lane_Obstacle", "1")) for path in discovered))
        self.assertTrue(any(path.endswith("scenario_abc") for path in discovered))

    def test_scenario_name_from_path_is_sanitized(self):
        scenario_dir = os.path.join(self.root, "interaction", "Blocked_Lane_Obstacle", "1")

        name = mdrive_import._scenario_name_from_path(self.root, scenario_dir)

        self.assertEqual(name, "mdrive_interaction_blocked_lane_obstacle_1")

    def test_dry_run_reports_without_writing_files(self):
        original_scenario_dir = mdrive_import.CARLA_SCENARIO_DIR
        output_dir = tempfile.mkdtemp(prefix="mdrive_batch_out_")
        self.addCleanup(shutil.rmtree, output_dir, ignore_errors=True)
        mdrive_import.CARLA_SCENARIO_DIR = output_dir
        try:
            summary = mdrive_import.run_batch_conversion(self.root, dry_run=True)
        finally:
            mdrive_import.CARLA_SCENARIO_DIR = original_scenario_dir

        self.assertEqual(len(summary["converted"]), 3)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(os.listdir(output_dir), [])

    def test_real_run_writes_a_yaml_per_scenario(self):
        original_scenario_dir = mdrive_import.CARLA_SCENARIO_DIR
        output_dir = tempfile.mkdtemp(prefix="mdrive_batch_out_")
        self.addCleanup(shutil.rmtree, output_dir, ignore_errors=True)
        mdrive_import.CARLA_SCENARIO_DIR = output_dir
        try:
            summary = mdrive_import.run_batch_conversion(self.root, dry_run=False)
        finally:
            mdrive_import.CARLA_SCENARIO_DIR = original_scenario_dir

        self.assertEqual(len(summary["converted"]), 3)
        written = sorted(os.listdir(output_dir))
        self.assertEqual(
            written,
            [
                "mdrive_interaction_blocked_lane_obstacle_1",
                "mdrive_interaction_unprotected_left_turn_1",
                "mdrive_v2xpnp_scenario_abc",
            ],
        )

    def test_batch_continues_past_a_broken_scenario(self):
        broken_dir = os.path.join(self.root, "interaction", "Broken_Scenario", "1")
        os.makedirs(broken_dir, exist_ok=True)
        with open(os.path.join(broken_dir, "actors_manifest.json"), "w", encoding="utf-8") as file:
            json.dump({"ego": []}, file)

        original_scenario_dir = mdrive_import.CARLA_SCENARIO_DIR
        output_dir = tempfile.mkdtemp(prefix="mdrive_batch_out_")
        self.addCleanup(shutil.rmtree, output_dir, ignore_errors=True)
        mdrive_import.CARLA_SCENARIO_DIR = output_dir
        try:
            summary = mdrive_import.run_batch_conversion(self.root, dry_run=True)
        finally:
            mdrive_import.CARLA_SCENARIO_DIR = original_scenario_dir

        self.assertEqual(len(summary["converted"]), 3)
        self.assertEqual(len(summary["errors"]), 1)
        self.assertIn("broken_scenario", summary["errors"][0].lower())


if __name__ == "__main__":
    unittest.main()
