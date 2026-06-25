import os
import tempfile
import unittest
from unittest import mock

from main import list_available_scenarios, load_any_scenario
from opencda_scenario.runner import _strip_static_obstacle_spawner, run_loaded_world
from opencda_scenario.sumo_assets import ensure_sumo_assets, resolve_xodr_path


class OpenCDASumoScenarioTests(unittest.TestCase):
    def test_town10_sumo_scenario_is_available(self):
        self.assertIn("town10_sumo", list_available_scenarios())

        scenario_cfg = load_any_scenario("town10_sumo")

        self.assertEqual(str(scenario_cfg.get("runner_module", "")), "opencda_scenario.runner")
        self.assertTrue(bool(scenario_cfg.get("sumo", {}).get("enabled", False)))
        self.assertEqual(
            str(scenario_cfg.get("carla", {}).get("map", "")),
            "/Game/Carla/Maps/Town10HD_Opt",
        )

    def test_town10_sumo_scenario_resolves_local_xodr(self):
        scenario_cfg = load_any_scenario("town10_sumo")

        xodr_path = resolve_xodr_path(
            scenario_cfg=scenario_cfg,
            sumo_cfg=scenario_cfg.get("sumo", {}),
        )

        self.assertTrue(os.path.isfile(xodr_path))
        self.assertTrue(xodr_path.endswith("Town10HD_Opt.xodr"))

    def test_roadway_hazard_opencda_scenario_is_available(self):
        self.assertIn("roadway_hazard", list_available_scenarios())

        scenario_cfg = load_any_scenario("roadway_hazard")

        self.assertEqual(str(scenario_cfg.get("runner_module", "")), "opencda_scenario.runner")
        self.assertTrue(bool(scenario_cfg.get("sumo", {}).get("enabled", False)))
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "carla_scenario.roadway_hazard.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("anchors", {}).get("ego_spawn", "")),
            "ego_high_level_path",
        )

    def test_ensure_sumo_assets_accepts_prebuilt_asset_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_dir = os.path.join(tmp_dir, "Town10HD_Opt")
            os.makedirs(asset_dir, exist_ok=True)
            for suffix in (".sumocfg", ".net.xml", ".rou.xml"):
                with open(
                    os.path.join(asset_dir, f"Town10HD_Opt{suffix}"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    handle.write("placeholder")

            scenario_cfg = {
                "carla": {
                    "map": "/Game/Carla/Maps/Town10HD_Opt",
                }
            }
            sumo_cfg = {
                "enabled": True,
                "map_basename": "Town10HD_Opt",
                "asset_root": tmp_dir,
                "auto_generate_assets": False,
            }

            resolved = ensure_sumo_assets(
                scenario_cfg=scenario_cfg,
                sumo_cfg=sumo_cfg,
            )

            self.assertEqual(resolved, asset_dir)

    def test_open_cda_runner_strips_static_obstacle_spawner(self):
        scenario_cfg = load_any_scenario("town10_sumo")
        normalized_cfg = _strip_static_obstacle_spawner(scenario_cfg)

        self.assertEqual(
            str(scenario_cfg.get("obstacles", {}).get("spawner_module", "")),
            "carla_scenario.town10.obstacle_spawner",
        )
        self.assertNotIn("spawner_module", normalized_cfg.get("obstacles", {}))

    def test_open_cda_runner_delegates_with_obstacle_spawner_removed(self):
        scenario_cfg = load_any_scenario("high_level_route_planning_sumo")

        with mock.patch("opencda_scenario.runner._run_loaded_world", return_value=7) as run_mock:
            result = run_loaded_world(
                client=object(),
                world=object(),
                scenario_cfg=scenario_cfg,
                carla=object(),
            )

        self.assertEqual(result, 7)
        forwarded_cfg = run_mock.call_args.kwargs["scenario_cfg"]
        self.assertNotIn("spawner_module", forwarded_cfg.get("obstacles", {}))
        self.assertEqual(
            str(forwarded_cfg.get("runtime", {}).get("module", "")),
            "carla_scenario.high_level_route_planning.scenario",
        )

    def test_ensure_sumo_assets_regenerates_routes_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            asset_dir = os.path.join(tmp_dir, "Town10HD_Opt")
            os.makedirs(asset_dir, exist_ok=True)
            sumocfg_path = os.path.join(asset_dir, "Town10HD_Opt.sumocfg")
            net_path = os.path.join(asset_dir, "Town10HD_Opt.net.xml")
            route_path = os.path.join(asset_dir, "Town10HD_Opt.rou.xml")
            trips_path = os.path.join(asset_dir, "Town10HD_Opt.trips.xml")
            xodr_path = os.path.join(tmp_dir, "Town10HD_Opt.xodr")
            for file_path in (sumocfg_path, net_path, route_path, trips_path, xodr_path):
                with open(file_path, "w", encoding="utf-8") as handle:
                    handle.write("placeholder")

            scenario_cfg = {
                "carla": {
                    "map": "/Game/Carla/Maps/Town10HD_Opt",
                }
            }
            sumo_cfg = {
                "enabled": True,
                "map_basename": "Town10HD_Opt",
                "asset_root": tmp_dir,
                "auto_generate_assets": True,
                "route_generation": {
                    "force_regenerate": True,
                },
            }

            def _fake_run_subprocess(command, cwd=None, scenario_cfg=None):
                with open(route_path, "w", encoding="utf-8") as handle:
                    handle.write("regenerated-route")
                with open(trips_path, "w", encoding="utf-8") as handle:
                    handle.write("regenerated-trips")

            with mock.patch("opencda_scenario.sumo_assets.resolve_xodr_path", return_value=xodr_path):
                with mock.patch("opencda_scenario.sumo_assets._run_subprocess", side_effect=_fake_run_subprocess) as run_mock:
                    resolved = ensure_sumo_assets(
                        scenario_cfg=scenario_cfg,
                        sumo_cfg=sumo_cfg,
                    )

            self.assertEqual(resolved, asset_dir)
            self.assertGreaterEqual(run_mock.call_count, 1)
            with open(route_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "regenerated-route")
            with open(trips_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "regenerated-trips")


if __name__ == "__main__":
    unittest.main()
