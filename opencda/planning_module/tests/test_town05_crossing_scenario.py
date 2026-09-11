from pathlib import Path

from omegaconf import OmegaConf

from opencda.scenario_testing.cpx_mature_runner import (
    _configure_synthetic_multimodal_prediction,
)
from opencda.scenario_testing.scripted_actor import ScriptedActor


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "opencda"
    / "scenario_testing"
    / "config_yaml"
    / "cpx_town05_two_vehicle_crossing.yaml"
)
OFF_CONFIG = (
    ROOT
    / "opencda"
    / "scenario_testing"
    / "config_yaml"
    / "cpx_town05_crossing_late_conflict_off.yaml"
)
CUT_IN_MULTIMODAL_CONFIG = (
    ROOT
    / "opencda"
    / "scenario_testing"
    / "config_yaml"
    / "cpx_two_vehicle_cut_in_multimodal.yaml"
)


class _Actor:
    def __init__(self, actor_id):
        self.id = actor_id


class _KinematicActor:
    def __init__(self):
        self.transforms = []
        self.velocities = []

    def set_transform(self, transform):
        self.transforms.append(transform)

    def set_target_velocity(self, velocity):
        self.velocities.append(velocity)


def _speed(vector):
    return (float(vector.x) ** 2 + float(vector.y) ** 2) ** 0.5


class _Planner:
    def __init__(self):
        self.config = {}
        self._prediction_mode = "cv"
        self._prediction_snapshot_transform_cached = True
        self._prediction_snapshot_transform_fn = object()
        self._cav_intent_broadcast_enabled = True


class _Manager:
    def __init__(self, actor_id):
        self.vehicle = _Actor(actor_id)
        self.cpx_planner = _Planner()


class _Scripted:
    def __init__(self, actor_id):
        self.vehicle = _Actor(actor_id)
        self.prediction_active = False


def test_town05_crossing_uses_multimodal_prediction_without_cv_arm():
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    assert cfg["cpx_mature"]["town"] == "Town05"
    assert cfg["vehicle_base"]["planner"]["global_planner_xodr_path"].endswith(
        "Town05.xodr"
    )
    assert cfg["vehicle_base"]["planner"]["prediction_mode"] == (
        "synthetic_multimodal"
    )
    assert cfg["vehicle_base"]["planner"]["ignore_traffic_control"] is True
    assert cfg["cpx_mature"]["synthetic_multimodal_prediction"] == {
        "enabled": True,
        "prediction_mode": "synthetic_multimodal",
        "ego_cav_index": 0,
        "target_scripted_actor_index": 0,
    }


def test_town05_crossing_paths_share_the_configured_conflict_point():
    cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    ego = cfg["scenario"]["single_cav_list"][0]
    target_path = cfg["scenario"]["scripted_actors"][0]["path"]
    conflict_x, conflict_y = target_path[1][:2]
    ego_x0, ego_y = ego["spawn_position"][:2]
    ego_x1 = ego["destination"][0]

    assert ego_x0 < conflict_x < ego_x1
    assert abs(float(ego_y) - float(conflict_y)) < 1.0e-6
    assert target_path[0][1] > conflict_y > target_path[-1][1]
    assert cfg["carla_traffic_manager"]["vehicle_list"] == []
    assert cfg["scenario"]["scripted_actors"][0]["trigger"] == {
        "axis": "x",
        "cross": -150.0,
        "from": "below",
    }


def test_scripted_actor_reports_zero_speed_until_position_trigger():
    vehicle = _KinematicActor()
    actor = ScriptedActor(
        vehicle,
        path_xy=[(0.0, 10.0), (0.0, 0.0)],
        speed_mps=6.0,
        trigger={"axis": "x", "cross": -5.0, "from": "below"},
    )

    actor.step(0.05, ego_xy=(-6.0, 0.0))
    assert actor.prediction_active is False
    assert abs(float(vehicle.velocities[-1].x)) < 1.0e-9
    assert abs(float(vehicle.velocities[-1].y)) < 1.0e-9

    actor.step(0.05, ego_xy=(-4.9, 0.0))
    assert actor.prediction_active is True
    assert abs(float(vehicle.velocities[-1].y) + 6.0) < 1.0e-6


def test_scripted_actor_optional_acceleration_limit_avoids_velocity_step():
    vehicle = _KinematicActor()
    actor = ScriptedActor(
        vehicle,
        path_xy=[(0.0, 0.0), (20.0, 0.0)],
        speed_mps=8.0,
        acceleration_mps2=4.0,
    )

    actor.step(0.05)
    assert abs(_speed(vehicle.velocities[-1]) - 0.2) < 1.0e-6
    actor.step(0.05)
    assert abs(_speed(vehicle.velocities[-1]) - 0.4) < 1.0e-6


def test_runner_binds_scripted_actor_id_to_ego_predictor():
    ego = _Manager(11)
    scripted = _Scripted(42)
    params = {
        "cpx_mature": {
            "synthetic_multimodal_prediction": {
                "enabled": True,
                "prediction_mode": "synthetic_multimodal",
                "ego_cav_index": 0,
                "target_scripted_actor_index": 0,
            }
        }
    }

    _configure_synthetic_multimodal_prediction(
        scenario_params=params,
        single_cav_list=[ego],
        scripted_actor_list=[scripted],
    )

    assert ego.cpx_planner._prediction_mode == "synthetic_multimodal"
    assert ego.cpx_planner.config["synthetic_prediction_actor_ids"] == [42]
    assert ego.cpx_planner._prediction_snapshot_transform_cached is False
    assert ego.cpx_planner._prediction_snapshot_transform_fn is None
    assert ego.cpx_planner._cav_intent_broadcast_enabled is True


def test_runner_preserves_existing_cav_target_binding():
    target = _Manager(21)
    ego = _Manager(22)
    params = {
        "cpx_mature": {
            "synthetic_multimodal_prediction": {
                "enabled": True,
                "ego_cav_index": 1,
                "target_cav_index": 0,
            }
        }
    }

    _configure_synthetic_multimodal_prediction(
        scenario_params=params,
        single_cav_list=[target, ego],
        scripted_actor_list=[],
    )

    assert ego.cpx_planner.config["synthetic_prediction_actor_ids"] == [21]
    assert target.cpx_planner._cav_intent_broadcast_enabled is False


def test_late_crossing_off_arm_changes_only_conflict_pipeline_and_output():
    def _load(path):
        value = OmegaConf.load(path)
        base = value.pop("base_config", None)
        if not base:
            return value
        return OmegaConf.merge(_load(Path(path).parent / str(base)), value)

    on_cfg = OmegaConf.to_container(
        _load(CONFIG.parent / "cpx_town05_crossing_late_conflict.yaml"),
        resolve=True,
    )
    off_cfg = OmegaConf.to_container(
        _load(OFF_CONFIG),
        resolve=True,
    )

    assert on_cfg["vehicle_base"]["planner"]["cav_conflict_enabled"] is True
    assert off_cfg["vehicle_base"]["planner"]["cav_conflict_enabled"] is False
    assert on_cfg["scenario"]["scripted_actors"] == off_cfg["scenario"]["scripted_actors"]
    on_ego = dict(on_cfg["scenario"]["single_cav_list"][0])
    off_ego = dict(off_cfg["scenario"]["single_cav_list"][0])
    on_debug = on_ego["planner"].pop("debug_output_dir")
    off_debug = off_ego["planner"].pop("debug_output_dir")
    # Frame capture is instrumentation, not a planning-policy difference.
    on_ego["planner"].pop("frame_capture", None)
    off_ego["planner"].pop("frame_capture", None)
    assert on_ego == off_ego
    assert on_debug != off_debug


def test_cut_in_multimodal_uses_prediction_and_position_trigger():
    def _load(path):
        value = OmegaConf.load(path)
        base = value.pop("base_config", None)
        if not base:
            return value
        return OmegaConf.merge(_load(Path(path).parent / str(base)), value)

    cfg = OmegaConf.to_container(_load(CUT_IN_MULTIMODAL_CONFIG), resolve=True)
    planner = cfg["vehicle_base"]["planner"]
    synthetic = cfg["cpx_mature"]["synthetic_multimodal_prediction"]
    actor = cfg["scenario"]["scripted_actors"][0]

    assert planner["cav_conflict_enabled"] is True
    assert planner["prediction_mode"] == "synthetic_multimodal"
    assert planner["prediction_update_hz"] == 5.0
    assert cfg["cpx_mature"]["max_ticks"] == 850
    assert synthetic["target_scripted_actor_index"] == 0
    assert "start_tick" not in actor
    assert actor["trigger"] == {
        "axis": "x", "cross": -280.0, "from": "below"
    }
    assert float(actor["path"][-1][0]) > float(
        cfg["scenario"]["single_cav_list"][0]["destination"][0]
    )
