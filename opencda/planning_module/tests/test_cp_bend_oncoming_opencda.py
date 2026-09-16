from pathlib import Path

from omegaconf import OmegaConf



ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "opencda" / "scenario_testing" / "config_yaml"


def _load_config(name):
    value = OmegaConf.load(CONFIG_DIR / (name + ".yaml"))
    base = value.pop("base_config", None)
    if not base:
        return value
    return OmegaConf.merge(_load_config(Path(str(base)).stem), value)


def test_bend_oncoming_ablation_changes_only_cp_transport_and_debug_path():
    on_cfg = OmegaConf.to_container(
        _load_config("cpx_cp_bend_oncoming"), resolve=True
    )
    off_cfg = OmegaConf.to_container(
        _load_config("cpx_cp_bend_oncoming_off"), resolve=True
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


def test_oncoming_path_reverses_the_ego_bend_and_is_event_aligned():
    cfg = OmegaConf.to_container(
        _load_config("cpx_cp_bend_oncoming"), resolve=True
    )
    actor = cfg["scenario"]["scripted_actors"][0]
    path = actor["path"]
    assert actor["trigger"] == {
        "axis": "x", "cross": 70.0, "from": "above"
    }
    assert float(path[0][1]) < float(path[1][1])
    assert float(path[-1][0]) > float(path[-2][0])
    assert abs(float(path[-1][1]) - float(path[-2][1])) < 1.0e-9
    assert float(path[-1][1]) > float(path[3][1])
    assert cfg["cpx_mature"]["town"] == "Town06"
    assert cfg["cpx_mature"]["require_isolated_world"] is True
    assert cfg["cpx_mature"]["max_ticks"] >= 1800
    assert cfg["vehicle_base"]["planner"]["cav_conflict_enabled"] is True
    assert cfg["vehicle_base"]["planner"]["global_planner_mode"] == "dij"
    assert cfg["vehicle_base"]["planner"]["ignore_traffic_control"] is True
    assert cfg["vehicle_base"]["planner"]["route_reached_distance_m"] == 1.5
    assert (
        cfg["vehicle_base"]["sensing"]["perception"]
        ["deactivated_detection_range_m"]
        == 20.0
    )
    assert cfg["vehicle_base"]["sensing"]["perception"]["activate"] is False
    observer = cfg["scenario"]["single_cav_list"][1]
    assert observer["behavior"]["max_speed"] == 0
    assert observer["sensing"]["perception"][
        "deactivated_detection_range_m"
    ] == 60.0
    assert observer["sensing"]["perception"]["activate"] is False



def test_bend_oncoming_scenarios_use_root_entrypoint_modules():
    for name in ("cpx_cp_bend_oncoming", "cpx_cp_bend_oncoming_off"):
        assert (ROOT / "opencda" / "scenario_testing" / (name + ".py")).is_file()
        assert (CONFIG_DIR / (name + ".yaml")).is_file()
