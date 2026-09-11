# -*- coding: utf-8 -*-
"""Run CP-X planner on mature OpenCDA scenario layouts."""

import math
import os
from collections.abc import Mapping

import carla

import opencda.scenario_testing.utils.customized_map_api as map_api
import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.planning_module.opencda_bridge.debug_viewer import OpenCDADebugViewer
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.scripted_actor import spawn_scripted_actors
from opencda.scenario_testing.utils.yaml_utils import add_current_time


def _set_spectator_transform(spectator, ego_vehicle):
    transform = ego_vehicle.get_transform()
    mode = str(os.environ.get("OPENCDA_SPECTATOR_VIEW", "planner")).strip().lower()
    if mode in {"topdown", "bird", "birdview"}:
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=70.0),
            carla.Rotation(pitch=-90.0, yaw=float(transform.rotation.yaw)),
        ))
        return
    yaw_rad = math.radians(float(transform.rotation.yaw))
    distance_m = float(os.environ.get("OPENCDA_SPECTATOR_DISTANCE_M", "10.0"))
    height_m = float(os.environ.get("OPENCDA_SPECTATOR_HEIGHT_M", "4.5"))
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(
            x=-distance_m * math.cos(yaw_rad),
            y=-distance_m * math.sin(yaw_rad),
            z=height_m,
        ),
        carla.Rotation(
            pitch=float(os.environ.get("OPENCDA_SPECTATOR_PITCH_DEG", "-15.0")),
            yaw=float(transform.rotation.yaw),
            roll=0.0,
        ),
    ))


def _distance_to_destination(vehicle, destination):
    loc = vehicle.get_location()
    return math.hypot(float(loc.x) - float(destination[0]), float(loc.y) - float(destination[1]))


def _destinations_reached(vehicle_managers, vehicle_configs, tolerance_m):
    """Return true only when every configured CAV reached its own goal."""

    if not vehicle_managers or len(vehicle_managers) != len(vehicle_configs):
        return False
    return all(
        _distance_to_destination(manager.vehicle, config["destination"])
        <= float(tolerance_m)
        for manager, config in zip(vehicle_managers, vehicle_configs)
    )


class _TargetBrakeStimulus(object):
    """Deterministic, state-triggered brake stimulus for prediction A/B runs.

    A fixed tick does not define an interaction: vehicle spawn settling and
    longitudinal control transients move the actual encounter by seconds.
    When trigger conditions are configured, arm once the ego is closing on
    the target inside the requested range.  Fixed ``start_tick`` remains the
    fallback for existing fixtures without trigger conditions.
    """

    def __init__(self, config):
        self.config = dict(config or {})
        self.started_tick = None

    @staticmethod
    def _speed_mps(vehicle):
        velocity = vehicle.get_velocity()
        return math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )

    def _triggered(self, tick, vehicle_managers):
        cfg = self.config
        target_position = cfg.get("trigger_target_position")
        if isinstance(target_position, Mapping):
            target_index = int(cfg.get("target_cav_index", 0))
            if not 0 <= target_index < len(vehicle_managers):
                return False
            axis = str(target_position.get("axis", "x")).strip().lower()
            if axis not in ("x", "y"):
                raise ValueError("trigger_target_position.axis must be x or y")
            location = vehicle_managers[target_index].vehicle.get_location()
            position = float(getattr(location, axis))
            threshold = float(target_position["value"])
            direction = str(
                target_position.get("direction", "increasing")
            ).strip().lower()
            if direction == "increasing":
                crossed = position >= threshold
            elif direction == "decreasing":
                crossed = position <= threshold
            else:
                raise ValueError(
                    "trigger_target_position.direction must be increasing or decreasing"
                )
            return bool(
                int(tick) >= max(0, int(cfg.get("minimum_start_tick", 0)))
                and crossed
            )

        trigger_gap = cfg.get("trigger_gap_m")
        if trigger_gap is None:
            return int(tick) >= max(0, int(cfg.get("start_tick", 0)))

        target_index = int(cfg.get("target_cav_index", 0))
        ego_index = int(cfg.get("ego_cav_index", 1))
        if not (
            0 <= target_index < len(vehicle_managers)
            and 0 <= ego_index < len(vehicle_managers)
            and target_index != ego_index
        ):
            return False
        target = vehicle_managers[target_index].vehicle
        ego = vehicle_managers[ego_index].vehicle
        target_location = target.get_location()
        ego_location = ego.get_location()
        gap_m = math.hypot(
            float(target_location.x) - float(ego_location.x),
            float(target_location.y) - float(ego_location.y),
        )
        ego_speed = self._speed_mps(ego)
        closing_speed = ego_speed - self._speed_mps(target)
        return bool(
            int(tick) >= max(0, int(cfg.get("minimum_start_tick", 0)))
            and gap_m <= float(trigger_gap)
            and ego_speed >= float(cfg.get("minimum_ego_speed_mps", 0.0))
            and closing_speed >= float(cfg.get("minimum_closing_speed_mps", 0.0))
        )

    def apply(self, control, *, tick, cav_index, vehicle_managers):
        cfg = self.config
        if not bool(cfg.get("enabled", False)):
            return control
        if int(cav_index) != int(cfg.get("target_cav_index", 0)):
            return control
        if self.started_tick is None and self._triggered(tick, vehicle_managers):
            self.started_tick = int(tick)
            print("[CP-X stimulus] target brake triggered at tick %d" % int(tick))
        duration = max(0, int(cfg.get("duration_ticks", 0)))
        if self.started_tick is None or not (
            self.started_tick <= int(tick) < self.started_tick + duration
        ):
            return control
        control.throttle = 0.0
        control.brake = max(float(control.brake), float(cfg.get("brake", 0.65)))
        return control


def _scenario_manager_kwargs(scenario_params):
    mature_cfg = scenario_params.get("cpx_mature", {})
    map_mode = str(mature_cfg.get("map_mode", "town")).strip().lower()
    if map_mode == "2lane_freeway_simplified":
        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(
            current_path,
            "../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr",
        )
        return {"xodr_path": xodr_path}, map_api.spawn_helper_2lanefree
    town = str(mature_cfg.get("town", "Town06"))
    return {"town": town}, None


def _configure_synthetic_multimodal_prediction(
    *, scenario_params, single_cav_list, scripted_actor_list,
):
    """Bind a scenario's synthetic predictor to one physical target actor.

    The planner consumes actor ids, while scenario YAML uses stable list
    indices because CARLA assigns actor ids only after spawn.  A target may
    be either another managed CAV or a deterministic scripted road user; the
    prediction pipeline is identical after this one-time binding.
    """

    multimodal_cfg = dict(
        scenario_params.get("cpx_mature", {}).get(
            "synthetic_multimodal_prediction", {}
        ) or {}
    )
    if not bool(multimodal_cfg.get("enabled", False)):
        return None

    ego_index = int(multimodal_cfg.get("ego_cav_index", 0))
    if not 0 <= ego_index < len(single_cav_list):
        raise IndexError("synthetic multimodal ego_cav_index is out of range")

    target_planner = None
    target_scripted_actor = None
    target_label = ""
    if "target_scripted_actor_index" in multimodal_cfg:
        target_index = int(multimodal_cfg["target_scripted_actor_index"])
        if not 0 <= target_index < len(scripted_actor_list):
            raise IndexError(
                "synthetic multimodal target_scripted_actor_index is out of range"
            )
        target_actor = scripted_actor_list[target_index].vehicle
        target_scripted_actor = scripted_actor_list[target_index]
        target_label = "scripted[%d]" % target_index
    else:
        target_index = int(multimodal_cfg.get("target_cav_index", 0))
        if not 0 <= target_index < len(single_cav_list):
            raise IndexError("synthetic multimodal target_cav_index is out of range")
        if target_index == ego_index:
            raise ValueError("synthetic multimodal ego and target must differ")
        target_manager = single_cav_list[target_index]
        target_actor = target_manager.vehicle
        target_planner = getattr(target_manager, "cpx_planner", None)
        target_label = "cav[%d]" % target_index

    ego_planner = getattr(single_cav_list[ego_index], "cpx_planner", None)
    if ego_planner is None:
        raise RuntimeError("synthetic multimodal ego has no CP-X planner")

    mode_override = str(
        multimodal_cfg.get("prediction_mode", "synthetic_multimodal") or ""
    ).strip().lower()
    if mode_override:
        ego_planner._prediction_mode = mode_override
    target_actor_id = int(target_actor.id)
    activation_state = {str(target_actor_id): True}
    if target_scripted_actor is not None:
        activation_state[str(target_actor_id)] = bool(
            target_scripted_actor.prediction_active
        )
    ego_planner._synthetic_prediction_actor_activation = activation_state
    ego_planner.config["synthetic_prediction_actor_ids"] = [target_actor_id]
    ego_planner._prediction_snapshot_transform_cached = False
    ego_planner._prediction_snapshot_transform_fn = None
    if target_planner is not None:
        # The experiment must consume prediction hypotheses instead of the
        # target CAV's exact broadcast MPC solution.
        target_planner._cav_intent_broadcast_enabled = False
    print(
        "[CP-X multimodal] ego cav[%d] predicts %s actor=%d"
        % (ego_index, target_label, target_actor_id)
    )
    return target_scripted_actor, activation_state, str(target_actor_id)


def run_mature_scenario(opt, scenario_params, *, script_name):
    scenario_manager = None
    eval_manager = None
    debug_viewer = None
    single_cav_list = []
    bg_veh_list = []
    scripted_actor_list = []
    try:
        scenario_params = add_current_time(scenario_params)
        cav_world = CavWorld(opt.apply_ml)
        manager_kwargs, map_helper = _scenario_manager_kwargs(scenario_params)
        scenario_manager = sim_api.ScenarioManager(
            scenario_params,
            opt.apply_ml,
            opt.version,
            cav_world=cav_world,
            **manager_kwargs,
        )
        if opt.record:
            scenario_manager.client.start_recorder("%s.log" % script_name, True)

        single_cav_list = scenario_manager.create_vehicle_manager(
            application=["single"],
            map_helper=map_helper,
        )
        _, bg_veh_list = scenario_manager.create_traffic_carla()
        scripted_actor_list = spawn_scripted_actors(
            scenario_manager.world,
            scenario_params.get("scenario", {}).get("scripted_actors", []),
        )
        synthetic_prediction_runtime = _configure_synthetic_multimodal_prediction(
            scenario_params=scenario_params,
            single_cav_list=single_cav_list,
            scripted_actor_list=scripted_actor_list,
        )

        eval_manager = EvaluationManager(
            scenario_manager.cav_world,
            script_name=str(script_name),
            current_time=scenario_params["current_time"],
        )
        viewer_enabled = bool(
            scenario_params.get("debug_viewer", {}).get("enabled", False)
        )
        if OpenCDADebugViewer.enabled_from_env(viewer_enabled) and single_cav_list:
            debug_viewer = OpenCDADebugViewer(
                world=scenario_manager.world,
                carla_module=carla,
                ego_vehicle=single_cav_list[0].vehicle,
            )

        runtime_cfg = scenario_params.get("cpx_mature", {})
        max_ticks = int(runtime_cfg.get("max_ticks", 1200))
        destination_tolerance_m = float(runtime_cfg.get("destination_tolerance_m", 8.0))
        vehicle_configs = scenario_params["scenario"]["single_cav_list"]
        completion_mode = str(runtime_cfg.get("completion_mode", "first_cav"))
        scripted_brake_cfg = dict(
            runtime_cfg.get("scripted_target_brake", {}) or {}
        )
        target_brake_stimulus = _TargetBrakeStimulus(scripted_brake_cfg)
        fixed_dt_s = float(
            scenario_params.get("world", {}).get("fixed_delta_seconds", 0.05)
        )
        spectator = scenario_manager.world.get_spectator()
        for tick_index in range(max(1, max_ticks)):
            scenario_manager.tick()
            ego_vehicle = single_cav_list[0].vehicle
            _ego_loc = ego_vehicle.get_location()
            _ego_xy = (float(_ego_loc.x), float(_ego_loc.y))
            for scripted_actor in scripted_actor_list:
                scripted_actor.step(fixed_dt_s, ego_xy=_ego_xy)
            if synthetic_prediction_runtime is not None:
                scripted_target, activation_state, actor_id = (
                    synthetic_prediction_runtime
                )
                if scripted_target is not None:
                    activation_state[actor_id] = bool(
                        scripted_target.prediction_active
                    )
            _set_spectator_transform(spectator, ego_vehicle)
            if completion_mode == "all_cavs":
                reached_destination = _destinations_reached(
                    single_cav_list, vehicle_configs, destination_tolerance_m
                )
            else:
                reached_destination = _distance_to_destination(
                    ego_vehicle, vehicle_configs[0]["destination"]
                ) <= destination_tolerance_m
            if reached_destination:
                print("CP-X mature scenario reached the configured destination.")
                break
            for cav_index, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                try:
                    control = single_cav.run_step()
                except SystemExit as exc:
                    if int(getattr(exc, "code", 0) or 0) == 0:
                        print("CP-X mature scenario stopped by OpenCDA destination condition.")
                        return
                    raise
                control = target_brake_stimulus.apply(
                    control,
                    tick=tick_index,
                    cav_index=cav_index,
                    vehicle_managers=single_cav_list,
                )
                single_cav.vehicle.apply_control(control)
            if debug_viewer is not None:
                debug_viewer.render(single_cav_list)

    finally:
        if eval_manager is not None and bool(
            scenario_params.get("cpx_mature", {}).get("run_opencda_evaluation", False)
        ):
            eval_manager.evaluate()
        if opt.record and scenario_manager is not None:
            scenario_manager.client.stop_recorder()
        # Actors and their sensors must be destroyed while the scenario's
        # synchronous world is still active.  Restoring world settings first
        # can leave the previous run's ego alive at its destination, where a
        # following A/B run observes it as a real stopped lead vehicle.
        if debug_viewer is not None:
            debug_viewer.destroy()
        for vehicle_manager in single_cav_list:
            vehicle_manager.destroy()
        for vehicle in bg_veh_list:
            vehicle.destroy()
        for scripted_actor in scripted_actor_list:
            scripted_actor.vehicle.destroy()
        if scenario_manager is not None:
            scenario_manager.close()
