from pathlib import Path
from types import SimpleNamespace
import math
import sys

from omegaconf import OmegaConf

from opencda.scenario_testing.scripted_actor import ScriptedActor

PLANNING_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(PLANNING_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANNING_MODULE_ROOT))
from opencda_bridge.cp_provider import OpenCDACPProvider


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "opencda" / "scenario_testing" / "config_yaml"


def _load_config(name):
    value = OmegaConf.load(CONFIG_DIR / (name + ".yaml"))
    base = value.pop("base_config", None)
    if not base:
        return value
    return OmegaConf.merge(_load_config(Path(str(base)).stem), value)


def test_vru_ablation_changes_only_cp_transport_and_debug_path():
    on_cfg = OmegaConf.to_container(
        _load_config("cpx_cp_vru_crossing"), resolve=True
    )
    off_cfg = OmegaConf.to_container(
        _load_config("cpx_cp_vru_crossing_off"), resolve=True
    )

    assert on_cfg["scenario"]["scripted_actors"] == off_cfg["scenario"]["scripted_actors"]
    assert on_cfg["vehicle_base"]["sensing"]["perception"]["activate"] is False
    assert off_cfg["vehicle_base"]["sensing"]["perception"]["activate"] is False
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


def test_vru_crossing_is_event_aligned_and_crosses_ego_path():
    cfg = OmegaConf.to_container(
        _load_config("cpx_cp_vru_crossing"), resolve=True
    )
    ego = cfg["scenario"]["single_cav_list"][0]
    walker = cfg["scenario"]["scripted_actors"][0]
    assert walker["blueprint"].startswith("walker.")
    assert walker["trigger"] == {
        "axis": "x", "cross": -244.0, "from": "below"
    }
    ego_y = float(ego["spawn_position"][1])
    assert float(walker["path"][0][1]) > ego_y > float(walker["path"][-1][1])
    assert float(ego["spawn_position"][0]) < float(walker["path"][0][0]) < float(ego["destination"][0])


def test_vru_on_and_off_scenarios_use_root_entrypoint_modules():
    for name in ("cpx_cp_vru_crossing", "cpx_cp_vru_crossing_off"):
        assert (ROOT / "opencda" / "scenario_testing" / (name + ".py")).is_file()
        assert (CONFIG_DIR / (name + ".yaml")).is_file()


class _Walker:
    type_id = "walker.pedestrian.0001"

    def __init__(self):
        self.controls = []

    def set_transform(self, transform):
        self.transform = transform

    def apply_control(self, control):
        self.controls.append(control)


def test_scripted_walker_reports_motion_through_walker_control():
    walker = _Walker()
    actor = ScriptedActor(
        walker,
        path_xy=[(0.0, 4.0), (0.0, 0.0)],
        speed_mps=1.2,
    )

    actor.step(0.05)

    assert abs(float(walker.controls[-1].speed) - 1.2) < 1.0e-6
    assert abs(float(walker.controls[-1].direction.x)) < 1.0e-6
    assert abs(float(walker.controls[-1].direction.y) + 1.0) < 1.0e-6


def test_cp_pedestrian_speed_comes_from_observed_positions_not_teleport_velocity():
    provider = OpenCDACPProvider(message_path="unused")
    first = provider._observed_motion(
        actor_id="walker-1", x_m=-180.0, y_m=16.0,
        timestamp_s=0.0, heading_rad=0.0,
    )
    held = provider._observed_motion(
        actor_id="walker-1", x_m=-180.0, y_m=16.0,
        timestamp_s=0.05, heading_rad=-1.5707963268,
    )
    crossing = provider._observed_motion(
        actor_id="walker-1", x_m=-180.0, y_m=15.94,
        timestamp_s=0.10, heading_rad=-1.5707963268,
    )

    assert first[0] == 0.0
    assert held[0] == 0.0
    assert abs(crossing[0] - 1.2) < 1.0e-6
    assert abs(crossing[1] + 1.5707963268) < 1.0e-6


def test_cp_pedestrian_message_ignores_bogus_carla_velocity():
    class Location:
        x, y, z = -180.0, 16.0, 0.5

        def distance(self, other):
            return math.hypot(self.x - other.x, self.y - other.y)

    class Walker:
        id = 7
        bounding_box = SimpleNamespace(extent=SimpleNamespace(x=0.3, y=0.3, z=0.9))

        def __init__(self):
            self.location = Location()

        def get_transform(self):
            return SimpleNamespace(
                location=self.location, rotation=SimpleNamespace(yaw=-90.0)
            )

        def get_velocity(self):
            return SimpleNamespace(x=40.0, y=0.0, z=0.0)

    provider = OpenCDACPProvider(message_path="unused")
    walker = Walker()
    arguments = dict(
        obj=walker, map_planner=None, ego_location=Location(),
        source="test", provider_source="test", fallback_id="7",
        object_type="pedestrian", skip_range_filter=True,
    )
    held = provider._object_to_cp_message(sim_time_s=0.0, **arguments)
    walker.location.y = 15.94
    crossing = provider._object_to_cp_message(sim_time_s=0.05, **arguments)

    assert held["state"][2] == 0.0
    assert abs(crossing["state"][2] - 1.2) < 1.0e-6
    assert abs(crossing["trajectory"][-1][0] + 180.0) < 1.0e-5
