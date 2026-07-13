import os
import unittest
from unittest import mock

from main import (
    _build_map_load_candidates,
    _is_retriable_world_ready_error,
    _safe_world_map_name,
    launch_carla_server,
)
from planning_runner import MPC_CONFIG_PATH, TRACKER_CONFIG_PATH


class MainStartupTests(unittest.TestCase):
    def test_runtime_error_is_retriable(self):
        self.assertTrue(_is_retriable_world_ready_error(RuntimeError("world not ready")))

    def test_known_carla_color_overflow_value_error_is_retriable(self):
        self.assertTrue(
            _is_retriable_world_ready_error(
                ValueError("color: integer overflow in color channel")
            )
        )

    def test_other_value_error_is_not_retriable(self):
        self.assertFalse(_is_retriable_world_ready_error(ValueError("unexpected payload")))

    def test_build_map_load_candidates_skips_invalid_parent_folder_candidate(self):
        self.assertEqual(
            _build_map_load_candidates("/Game/Carla/Maps/roadway_hazard_scenario"),
            [
                "/Game/Carla/Maps/roadway_hazard_scenario",
                "/Game/Carla/Maps/roadway_hazard_scenario/roadway_hazard_scenario",
            ],
        )

    def test_build_map_load_candidates_supports_absolute_umap_paths(self):
        self.assertEqual(
            _build_map_load_candidates(
                "/home/umd-user/carla_source/carla/Unreal/CarlaUE4/Content/Carla/Maps/Town06.umap"
            ),
            [
                "/home/umd-user/carla_source/carla/Unreal/CarlaUE4/Content/Carla/Maps/Town06.umap",
                "Town06.umap",
                "Town06",
                "/Game/Carla/Maps/Town06",
            ],
        )

    def test_build_map_load_candidates_prefers_base_map_when_opt_xodr_is_missing(self):
        def _fake_isfile(path):
            if str(path).endswith("Town10HD_Opt.xodr"):
                return False
            if str(path).endswith("Town10HD.xodr"):
                return True
            return False

        with mock.patch("main.os.path.isfile", side_effect=_fake_isfile):
            self.assertEqual(
                _build_map_load_candidates(
                    "/Game/Carla/Maps/Town10HD_Opt",
                    carla_root="/fake/carla",
                ),
                [
                    "/Game/Carla/Maps/Town10HD",
                    "/Game/Carla/Maps/Town10HD_Opt",
                    "/Game/Carla/Maps/Town10HD_Opt/Town10HD_Opt",
                ],
            )

    def test_launch_carla_server_prefers_base_map_when_opt_xodr_is_missing(self):
        def _fake_isfile(path):
            if str(path).endswith("Town10HD_Opt.xodr"):
                return False
            if str(path).endswith("Town10HD.xodr"):
                return True
            return False

        cfg = {
            "carla_root": "/fake/carla",
            "launch_mode": "ue4editor_map",
            "map": "/Game/Carla/Maps/Town10HD_Opt",
            "rhi": "-vulkan",
            "launch_extra_args": [],
        }

        with mock.patch.dict("main.os.environ", {"UE4_ROOT": "/fake/ue4"}, clear=False):
            with mock.patch("main.os.path.isfile", side_effect=_fake_isfile):
                with mock.patch("main.open", mock.mock_open()):
                    with mock.patch("main.subprocess.Popen") as popen_mock:
                        launch_carla_server(cfg)

        command = popen_mock.call_args.kwargs["args"] if "args" in popen_mock.call_args.kwargs else popen_mock.call_args.args[0]
        self.assertIn("/Game/Carla/Maps/Town10HD", command)
        self.assertNotIn("/Game/Carla/Maps/Town10HD_Opt", command)

    def test_safe_world_map_name_returns_empty_string_when_world_map_fails(self):
        class _BrokenWorld:
            def get_map(self):
                raise RuntimeError("failed to generate map")

        self.assertEqual(_safe_world_map_name(_BrokenWorld()), "")

    def test_planning_runner_config_paths_resolve_inside_planning_module(self):
        self.assertTrue(os.path.exists(MPC_CONFIG_PATH), MPC_CONFIG_PATH)
        self.assertTrue(os.path.exists(TRACKER_CONFIG_PATH), TRACKER_CONFIG_PATH)


if __name__ == "__main__":
    unittest.main()
