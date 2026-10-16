from types import SimpleNamespace

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf

from opencda.scenario_testing.run_status import mark_destination_finished


def test_mark_destination_finished_respects_completion_mode():
    managers = [
        SimpleNamespace(_opencda_agent_finished=False),
        SimpleNamespace(_opencda_agent_finished=False),
    ]

    mark_destination_finished(managers, "first_cav")
    assert managers[0]._opencda_agent_finished is True
    assert managers[1]._opencda_agent_finished is False

    mark_destination_finished(managers, "all_cavs")
    assert all(manager._opencda_agent_finished for manager in managers)


def test_isolated_left_turn_preserves_vehicle_and_owns_debug_directory():
    repository_root = Path(__file__).resolve().parents[3]
    entry_path = repository_root / "opencda.py"
    spec = importlib.util.spec_from_file_location("opencda_entry", entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    scenario_path = repository_root / (
        "opencda/scenario_testing/config_yaml/"
        "cpx_single_left_lane_turn_isolated.yaml"
    )
    config = entry._load_scenario_config(str(scenario_path))
    vehicle = OmegaConf.merge(
        config.vehicle_base, config.scenario.single_cav_list[0]
    )

    assert list(vehicle.spawn_position[:2]) == [-10.75, 162.52]
    assert list(vehicle.destination[:2]) == [64.38, 248.05]
    assert vehicle.planner.ignore_traffic_control is True
    assert vehicle.planner.functional_test_ignore_dynamic_objects is True
    assert vehicle.planner.max_mpc_obstacles == 0
    assert vehicle.planner.debug_output_dir == (
        "artifacts/debug/single_left_lane_turn_isolated"
    )
