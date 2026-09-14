import json
from pathlib import Path

from omegaconf import OmegaConf

import opencda.scenario_testing.cpx_mature_runner as mature_runner


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "opencda" / "scenario_testing" / "config_yaml"


def _load_config(name):
    path = CONFIG_DIR / (name + ".yaml")
    value = OmegaConf.load(path)
    base = value.pop("base_config", None)
    if not base:
        return value
    return OmegaConf.merge(_load_config(Path(str(base)).stem), value)


def test_roadway_object_ablation_changes_only_cp_transport_and_debug_path():
    on_cfg = OmegaConf.to_container(
        _load_config("cpx_cp_roadway_object"), resolve=True
    )
    off_cfg = OmegaConf.to_container(
        _load_config("cpx_cp_roadway_object_off"), resolve=True
    )

    assert on_cfg["scenario"]["scripted_actors"] == off_cfg["scenario"]["scripted_actors"]
    assert on_cfg["scenario"]["single_cav_list"][1] == off_cfg["scenario"]["single_cav_list"][1]

    on_ego = dict(on_cfg["scenario"]["single_cav_list"][0])
    off_ego = dict(off_cfg["scenario"]["single_cav_list"][0])
    on_planner = dict(on_ego.pop("planner"))
    off_planner = dict(off_ego.pop("planner"))
    assert on_ego == off_ego
    assert off_planner.pop("publish_cp_message") is False
    assert off_planner.pop("require_native_opencda_cp") is False
    on_planner.pop("debug_output_dir")
    off_planner.pop("debug_output_dir")
    assert on_planner == off_planner


def test_mature_runner_resets_default_and_configured_cp_payloads(tmp_path, monkeypatch):
    default_path = tmp_path / "default.json"
    custom_path = tmp_path / "custom.json"
    monkeypatch.setattr(mature_runner, "CP_MESSAGE_PATH", str(default_path))
    for path in (default_path, custom_path):
        path.write_text('{"obstacles":[{"id":"stale"}]}')

    mature_runner._reset_cooperative_payloads({
        "vehicle_base": {"planner": {"cp_message_path": str(custom_path)}},
        "scenario": {"single_cav_list": []},
    })

    for path in (default_path, custom_path):
        payload = json.loads(path.read_text())
        assert payload["obstacles"] == []
        assert payload["lane_events"] == []
        assert payload["control"] == []
        assert payload["sequence"] == 0
