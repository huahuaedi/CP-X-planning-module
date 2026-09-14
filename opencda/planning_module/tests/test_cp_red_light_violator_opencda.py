from pathlib import Path

from omegaconf import OmegaConf


CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "scenario_testing" / "config_yaml"
)


def _load(name):
    value = OmegaConf.load(CONFIG_DIR / (name + ".yaml"))
    base = value.get("base_config")
    if base:
        return OmegaConf.merge(_load(Path(str(base)).stem), value)
    return value


def test_red_light_ablation_changes_only_cp_and_artifact_output():
    on = OmegaConf.to_container(_load("cpx_cp_red_light_violator"), resolve=True)
    off = OmegaConf.to_container(
        _load("cpx_cp_red_light_violator_off"), resolve=True
    )
    on_scene = on["scenario"]
    off_scene = off["scenario"]

    assert on_scene["scripted_actors"] == off_scene["scripted_actors"]
    assert [v["spawn_position"] for v in on_scene["single_cav_list"]] == [
        v["spawn_position"] for v in off_scene["single_cav_list"]
    ]
    assert [v["destination"] for v in on_scene["single_cav_list"]] == [
        v["destination"] for v in off_scene["single_cav_list"]
    ]
    assert on["vehicle_base"]["planner"]["publish_cp_message"] is True
    assert off["vehicle_base"]["planner"]["publish_cp_message"] is False
    assert on["vehicle_base"]["sensing"]["perception"][
        "deactivated_detection_range_m"
    ] == 20.0


def test_observer_sees_violator_before_range_limited_ego():
    cfg = OmegaConf.to_container(
        _load("cpx_cp_red_light_violator"), resolve=True
    )
    ego, observer = cfg["scenario"]["single_cav_list"]
    target = cfg["scenario"]["scripted_actors"][0]["path"][0]
    sensing_range = cfg["vehicle_base"]["planner"]["communication_range_m"]

    def distance(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    assert distance(observer["spawn_position"], target) < sensing_range
    assert distance(ego["spawn_position"], target) > sensing_range
    assert ego["v2x"]["communication_range"] > distance(
        ego["spawn_position"], observer["spawn_position"]
    )
