# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math
import os

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.planning_module.opencda_bridge.debug_viewer import OpenCDADebugViewer
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time


def _set_spectator_transform(spectator, ego_vehicle):
    transform = ego_vehicle.get_transform()
    mode = str(os.environ.get(
        "OPENCDA_SPECTATOR_VIEW", "planner")).strip().lower()
    if mode in {"topdown", "bird", "birdview", "opencda"}:
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=70),
            carla.Rotation(pitch=-90)))
        return

    yaw_rad = math.radians(float(transform.rotation.yaw))
    distance_m = float(os.environ.get(
        "OPENCDA_SPECTATOR_DISTANCE_M", "10.0"))
    height_m = float(os.environ.get(
        "OPENCDA_SPECTATOR_HEIGHT_M", "4.5"))
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(
            x=-distance_m * math.cos(yaw_rad),
            y=-distance_m * math.sin(yaw_rad),
            z=height_m),
        carla.Rotation(
            pitch=float(os.environ.get(
                "OPENCDA_SPECTATOR_PITCH_DEG", "-15.0")),
            yaw=float(transform.rotation.yaw))))


def run_scenario(opt, scenario_params):
    scenario_manager = None
    eval_manager = None
    debug_viewer = None
    single_cav_list = []
    bg_veh_list = []
    try:
        scenario_params = add_current_time(scenario_params)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("single_town06_carla.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='single_2lanefree_carla',
                              current_time=scenario_params['current_time'])
        if OpenCDADebugViewer.enabled_from_env() and single_cav_list:
            debug_viewer = OpenCDADebugViewer(
                world=scenario_manager.world,
                carla_module=carla,
                ego_vehicle=single_cav_list[0].vehicle,
            )

        spectator = scenario_manager.world.get_spectator()
        # run steps
        while True:
            scenario_manager.tick()
            _set_spectator_transform(
                spectator, single_cav_list[0].vehicle)

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)

            if debug_viewer is not None:
                debug_viewer.render(single_cav_list)

    finally:
        if eval_manager is not None:
            eval_manager.evaluate()

        if opt.record and scenario_manager is not None:
            scenario_manager.client.stop_recorder()

        if debug_viewer is not None:
            debug_viewer.destroy()

        if scenario_manager is not None:
            scenario_manager.close()

        for v in single_cav_list:
            v.destroy()
        for v in bg_veh_list:
            v.destroy()
