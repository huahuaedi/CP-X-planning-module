import os
import unittest

from main import (
    _build_map_load_candidates,
    _is_retriable_world_ready_error,
    _safe_world_map_name,
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
