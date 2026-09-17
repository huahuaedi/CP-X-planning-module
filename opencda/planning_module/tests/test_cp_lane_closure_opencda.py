import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from opencda.planning_module.utility.cp_messages import load_cp_message_payload
from opencda.scenario_testing.cpx_mature_runner import (
    _LaneClosureStimulus,
    _primary_cp_message_path,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "opencda" / "scenario_testing" / "config_yaml"


class _Vehicle:
    def __init__(self, x, y):
        self.location = SimpleNamespace(x=float(x), y=float(y))

    def get_location(self):
        return self.location


def _manager(x, y):
    return SimpleNamespace(vehicle=_Vehicle(x, y))


def test_lane_closure_stimulus_publishes_one_durable_lane_event(tmp_path):
    message_path = tmp_path / "cp_message.json"
    stimulus = _LaneClosureStimulus({
        "enabled": True,
        "trigger_position": {
            "cav_index": 0,
            "axis": "x",
            "value": -265.0,
            "direction": "increasing",
        },
        "message": {
            "id": "workzone",
            "position": [-180.0, 8.3, 0.0],
        },
    }, message_path=str(message_path))
    manager = _manager(-270.0, 8.3)

    assert stimulus.publish_if_due(
        tick=1, sim_time_s=0.05, vehicle_managers=[manager]
    ) is False
    manager.vehicle.location.x = -264.0
    assert stimulus.publish_if_due(
        tick=2, sim_time_s=0.10, vehicle_managers=[manager]
    ) is True
    assert stimulus.publish_if_due(
        tick=3, sim_time_s=0.15, vehicle_managers=[manager]
    ) is False

    payload = load_cp_message_payload(str(message_path))
    assert payload["lane_events"] == [{
        "id": "workzone",
        "position": [-180.0, 8.3, 0.0],
        "type": "lane_closure",
        "source": "scenario_cp_fixture",
    }]


def test_cp_f_config_uses_one_native_entry_and_real_lane_event():
    config = OmegaConf.load(CONFIG_DIR / "cpx_cp_lane_closure.yaml")
    fixture = OmegaConf.to_container(
        config.cpx_mature.cp_lane_closure, resolve=True
    )

    assert fixture["enabled"] is True
    assert fixture["message"]["type"] == "lane_closure"
    assert fixture["message"]["position"] == [-180.0, 8.3, 0.0]
    assert config.cpx_mature.map_mode == "2lane_freeway_simplified"


def test_primary_cp_path_prefers_ego_override_then_vehicle_base(tmp_path):
    base_path = tmp_path / "base.json"
    ego_path = tmp_path / "ego.json"
    params = {
        "vehicle_base": {"planner": {"cp_message_path": str(base_path)}},
        "scenario": {"single_cav_list": [{"planner": {}}]},
    }
    assert _primary_cp_message_path(params) == str(base_path)
    params["scenario"]["single_cav_list"][0]["planner"] = {
        "cp_message_path": str(ego_path)
    }
    assert _primary_cp_message_path(params) == str(ego_path)
