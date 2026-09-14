import json
from pathlib import Path

from omegaconf import OmegaConf

import opencda.scenario_testing.cpx_mature_runner as mature_runner


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "opencda" / "scenario_testing" / "config_yaml"


class _OwnedActor:
    def __init__(self, actor_id, role_name, type_id="vehicle.test", x=0.0, y=0.0):
        self.id = actor_id
        self.type_id = type_id
        self.attributes = {"role_name": role_name}
        self.destroyed = False
        self._location = type("Location", (), {"x": x, "y": y})()

    def destroy(self):
        self.destroyed = True

    def get_location(self):
        return self._location


class _World:
    def __init__(self, actors):
        self.actors = actors

    def get_actors(self):
        return self.actors


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


def test_mature_runner_reclaims_only_actors_owned_by_same_scenario():
    params = {
        "scenario": {
            "single_cav_list": [{"name": "ego"}, {"name": "observer"}],
            "scripted_actors": [{}, {}],
        }
    }
    roles = mature_runner._assign_scenario_actor_roles(
        params, "cpx_cp_bend_oncoming"
    )
    assert roles == (
        "opencda_cpx_cpx_cp_bend_oncoming_cav_ego",
        "opencda_cpx_cpx_cp_bend_oncoming_cav_observer",
        "opencda_cpx_cpx_cp_bend_oncoming_scripted_0",
        "opencda_cpx_cpx_cp_bend_oncoming_scripted_1",
    )
    assert params["scenario"]["single_cav_list"][0]["role_name"] == roles[0]
    assert params["scenario"]["scripted_actors"][1]["role_name"] == roles[3]

    stale = _OwnedActor(10, roles[0])
    unrelated = _OwnedActor(11, "another_scenario_ego")
    unowned = _OwnedActor(12, "")
    destroyed = mature_runner._destroy_stale_scenario_actors(
        _World([stale, unrelated, unowned]), roles
    )

    assert destroyed == [10]
    assert stale.destroyed is True
    assert unrelated.destroyed is False
    assert unowned.destroyed is False


def test_isolated_runner_reports_foreign_dynamic_actors_without_deleting_them():
    owned = _OwnedActor(20, "opencda_cpx_case_cav_ego")
    stale = _OwnedActor(21, "", x=12.25, y=-3.5)
    sensor = _OwnedActor(22, "", type_id="sensor.camera.rgb")

    foreign = mature_runner._foreign_dynamic_actors(
        _World([owned, stale, sensor]),
        ("opencda_cpx_case_cav_ego",),
    )

    assert foreign == [{
        "id": 21,
        "type_id": "vehicle.test",
        "role_name": "",
        "x": 12.25,
        "y": -3.5,
    }]
    assert stale.destroyed is False
