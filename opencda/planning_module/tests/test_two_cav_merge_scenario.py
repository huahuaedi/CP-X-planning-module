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


def test_cooperative_merge_ablation_has_one_opposing_lane_exchange_scene():
    on = OmegaConf.to_container(
        _load("cpx_two_cav_merge_conflict"), resolve=True
    )
    off = OmegaConf.to_container(
        _load("cpx_two_cav_merge_conflict_baseline"), resolve=True
    )
    on_vehicles = on["scenario"]["single_cav_list"]
    off_vehicles = off["scenario"]["single_cav_list"]

    assert [v["spawn_position"] for v in on_vehicles] == [
        v["spawn_position"] for v in off_vehicles
    ]
    assert [v["destination"] for v in on_vehicles] == [
        v["destination"] for v in off_vehicles
    ]
    assert on_vehicles[0]["spawn_position"][1] == on_vehicles[1]["destination"][1]
    assert on_vehicles[1]["spawn_position"][1] == on_vehicles[0]["destination"][1]
    assert on["vehicle_base"]["planner"]["cav_conflict_enabled"] is True
    assert off["vehicle_base"]["planner"]["cav_conflict_enabled"] is False
