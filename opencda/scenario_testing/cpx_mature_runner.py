# -*- coding: utf-8 -*-
"""Run CP-X planner on mature OpenCDA scenario layouts."""

import math
import os

import carla

import opencda.scenario_testing.utils.customized_map_api as map_api
import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.planning_module.opencda_bridge.debug_viewer import OpenCDADebugViewer
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
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


def run_mature_scenario(opt, scenario_params, *, script_name):
    scenario_manager = None
    eval_manager = None
    debug_viewer = None
    single_cav_list = []
    bg_veh_list = []
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
        multimodal_cfg = dict(
            scenario_params.get("cpx_mature", {}).get(
                "synthetic_multimodal_prediction", {}
            ) or {}
        )
        if bool(multimodal_cfg.get("enabled", False)):
            ego_index = int(multimodal_cfg.get("ego_cav_index", 1))
            target_index = int(multimodal_cfg.get("target_cav_index", 0))
            if (
                0 <= ego_index < len(single_cav_list)
                and 0 <= target_index < len(single_cav_list)
                and ego_index != target_index
            ):
                ego_planner = getattr(single_cav_list[ego_index], "cpx_planner", None)
                target_planner = getattr(
                    single_cav_list[target_index], "cpx_planner", None
                )
                target_actor_id = int(single_cav_list[target_index].vehicle.id)
                if ego_planner is not None:
                    mode_override = str(
                        multimodal_cfg.get("prediction_mode", "") or ""
                    ).strip().lower()
                    if mode_override:
                        ego_planner._prediction_mode = mode_override
                    ego_planner.config["synthetic_prediction_actor_ids"] = [
                        target_actor_id
                    ]
                    ego_planner._prediction_snapshot_transform_cached = False
                    ego_planner._prediction_snapshot_transform_fn = None
                if target_planner is not None:
                    target_planner._cav_intent_broadcast_enabled = False
                print(
                    "[CP-X multimodal] ego cav[%d] predicts cav[%d] actor=%d; "
                    "target planned_path broadcast disabled."
                    % (ego_index, target_index, target_actor_id)
                )
        _, bg_veh_list = scenario_manager.create_traffic_carla()

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
        spectator = scenario_manager.world.get_spectator()
        for _ in range(max(1, max_ticks)):
            scenario_manager.tick()
            ego_vehicle = single_cav_list[0].vehicle
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
            for single_cav in single_cav_list:
                single_cav.update_info()
                try:
                    control = single_cav.run_step()
                except SystemExit as exc:
                    if int(getattr(exc, "code", 0) or 0) == 0:
                        print("CP-X mature scenario stopped by OpenCDA destination condition.")
                        return
                    raise
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
        if debug_viewer is not None:
            debug_viewer.destroy()
        if scenario_manager is not None:
            scenario_manager.close()
        for vehicle_manager in single_cav_list:
            vehicle_manager.destroy()
        for vehicle in bg_veh_list:
            vehicle.destroy()
