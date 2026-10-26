from types import SimpleNamespace

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf

from opencda.scenario_testing.cpx_mature_runner import (
    _update_completion_checkpoints,
)
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


def test_completion_checkpoints_latch_after_vehicle_continues_downstream():
    class Vehicle(object):
        def __init__(self, x, y):
            self.location = SimpleNamespace(x=x, y=y)

        def get_location(self):
            return self.location

    managers = [
        SimpleNamespace(vehicle=Vehicle(-70.0, 4.8)),
        SimpleNamespace(vehicle=Vehicle(-100.0, 8.3)),
    ]
    configs = [
        {
            "destination": [-20.0, 4.8, 0.3],
            "completion_destination": [-70.0, 4.8, 0.3],
        },
        {
            "destination": [-20.0, 8.3, 0.3],
            "completion_destination": [-76.0, 8.3, 0.3],
        },
    ]
    reached = set()

    assert _update_completion_checkpoints(
        managers, configs, 3.0, reached
    ) is False
    assert reached == {0}
    assert managers[0]._opencda_completion_checkpoint_distance_m == 0.0

    # CAV 1 has moved far beyond its checkpoint. Its completion remains
    # latched while CAV 2 reaches its own checkpoint later.
    managers[0].vehicle.location.x = -40.0
    managers[1].vehicle.location.x = -76.0
    assert _update_completion_checkpoints(
        managers, configs, 3.0, reached
    ) is True
    assert reached == {0, 1}
    assert managers[1]._opencda_completion_checkpoint_distance_m == 0.0


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
