# -*- coding: utf-8 -*-
"""Small deterministic CP-X/OpenCDA smoke-test runner."""

import math
import os

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.planning_module.opencda_bridge.debug_viewer import OpenCDADebugViewer
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time


def _spectator_mode():
    mode = str(os.environ.get("OPENCDA_SPECTATOR_VIEW", "planner")).strip().lower()
    if mode in {"topdown", "bird", "birdview"}:
        return "topdown"
    return "planner"


def _set_spectator_transform(spectator, ego_vehicle):
    transform = ego_vehicle.get_transform()
    if _spectator_mode() == "topdown":
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=70.0),
            carla.Rotation(pitch=-90.0, yaw=float(transform.rotation.yaw)),
        ))
        return

    yaw_rad = math.radians(float(transform.rotation.yaw))
    follow_distance_m = float(os.environ.get("OPENCDA_SPECTATOR_DISTANCE_M", "10.0"))
    follow_height_m = float(os.environ.get("OPENCDA_SPECTATOR_HEIGHT_M", "4.5"))
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(
            x=-follow_distance_m * math.cos(yaw_rad),
            y=-follow_distance_m * math.sin(yaw_rad),
            z=follow_height_m,
        ),
        carla.Rotation(
            pitch=float(os.environ.get("OPENCDA_SPECTATOR_PITCH_DEG", "-15.0")),
            yaw=float(transform.rotation.yaw),
            roll=0.0,
        ),
    ))


def _distance_to_destination(vehicle, destination):
    location = vehicle.get_location()
    return math.hypot(
        float(location.x) - float(destination[0]),
        float(location.y) - float(destination[1]),
    )


def _traffic_light_state(name):
    normalized = str(name or "").strip().lower()
    if normalized == "red":
        return carla.TrafficLightState.Red
    if normalized == "yellow":
        return carla.TrafficLightState.Yellow
    if normalized == "green":
        return carla.TrafficLightState.Green
    return None


def _apply_traffic_light_schedule(world, runtime_cfg, tick_index):
    red_until_tick = int(runtime_cfg.get("force_red_until_tick", -1))
    green_after_tick = int(runtime_cfg.get("force_green_after_tick", red_until_tick + 1))
    forced_state = str(runtime_cfg.get("force_traffic_light_state", "")).strip().lower()
    state = None
    if forced_state:
        state = _traffic_light_state(forced_state)
    elif red_until_tick >= 0 and int(tick_index) <= int(red_until_tick):
        state = carla.TrafficLightState.Red
    elif green_after_tick >= 0 and int(tick_index) >= int(green_after_tick):
        state = carla.TrafficLightState.Green
    if state is None:
        return
    for actor in world.get_actors():
        if not str(getattr(actor, "type_id", "")).startswith("traffic.traffic_light"):
            continue
        try:
            actor.set_state(state)
            actor.freeze(True)
        except Exception:
            continue


def run_smoke_scenario(opt, scenario_params, *, town, script_name):
    scenario_manager = None
    eval_manager = None
    debug_viewer = None
    single_cav_list = []
    bg_veh_list = []
    try:
        scenario_params = add_current_time(scenario_params)
        cav_world = CavWorld(opt.apply_ml)
        scenario_manager = sim_api.ScenarioManager(
            scenario_params,
            opt.apply_ml,
            opt.version,
            town=str(town),
            cav_world=cav_world,
        )
        if opt.record:
            scenario_manager.client.start_recorder("%s.log" % script_name, True)

        single_cav_list = scenario_manager.create_vehicle_manager(application=["single"])
        traffic_cfg = scenario_params.get("carla_traffic_manager", {})
        if traffic_cfg.get("vehicle_list", []) or traffic_cfg.get("range", []):
            _, bg_veh_list = scenario_manager.create_traffic_carla()

        eval_manager = EvaluationManager(
            scenario_manager.cav_world,
            script_name=str(script_name),
            current_time=scenario_params["current_time"],
        )
        if OpenCDADebugViewer.enabled_from_env() and single_cav_list:
            debug_viewer = OpenCDADebugViewer(
                world=scenario_manager.world,
                carla_module=carla,
                ego_vehicle=single_cav_list[0].vehicle,
            )

        runtime_cfg = scenario_params.get("cpx_smoke", {})
        max_ticks = int(runtime_cfg.get("max_ticks", 900))
        destination_tolerance_m = float(runtime_cfg.get("destination_tolerance_m", 5.0))
        destination = scenario_params["scenario"]["single_cav_list"][0]["destination"]
        spectator = scenario_manager.world.get_spectator()
        termination_reason = "max_ticks_reached"
        completed_ticks = 0

        for tick_index in range(max(1, max_ticks)):
            completed_ticks = int(tick_index) + 1
            _apply_traffic_light_schedule(scenario_manager.world, runtime_cfg, tick_index)
            scenario_manager.tick()
            ego_vehicle = single_cav_list[0].vehicle
            _set_spectator_transform(spectator, ego_vehicle)
            for single_cav in single_cav_list:
                single_cav.update_info()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)
            if debug_viewer is not None:
                debug_viewer.render(single_cav_list)
            if _distance_to_destination(ego_vehicle, destination) <= destination_tolerance_m:
                termination_reason = "destination_reached"
                break
        print(
            "[CP-X smoke] Scenario finished: reason=%s ticks=%d/%d "
            "distance_to_destination_m=%.2f"
            % (
                str(termination_reason),
                int(completed_ticks),
                int(max_ticks),
                float(_distance_to_destination(ego_vehicle, destination)),
            )
        )

    finally:
        if eval_manager is not None and bool(
            scenario_params.get("cpx_smoke", {}).get("run_opencda_evaluation", False)
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
