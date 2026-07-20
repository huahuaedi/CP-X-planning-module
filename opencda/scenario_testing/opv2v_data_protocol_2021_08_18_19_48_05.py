# -*- coding: utf-8 -*-
"""
Scenario testing: merging vehicle joining a platoon in the
customized 2-lane freeway simplified map sorely with carla
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import os
import math

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.planning_module.opencda_bridge.debug_viewer import OpenCDADebugViewer
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time


def _spectator_view_mode():
    """Return the requested OpenCDA spectator camera mode."""

    mode = str(os.environ.get("OPENCDA_SPECTATOR_VIEW", "planner")).strip().lower()
    if mode in {"planner", "chase", "follow", "third_person"}:
        return "planner"
    if mode in {"topdown", "bird", "birdview", "opencda"}:
        return "topdown"
    return "planner"


def _set_spectator_transform(spectator, ego_vehicle):
    """Use the CP-X/planning-module style follow view by default."""

    transform = ego_vehicle.get_transform()
    mode = _spectator_view_mode()
    if mode == "topdown":
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=70),
            carla.Rotation(pitch=-90)))
        return

    yaw_rad = math.radians(float(transform.rotation.yaw))
    follow_distance_m = float(os.environ.get("OPENCDA_SPECTATOR_DISTANCE_M", "10.0"))
    follow_height_m = float(os.environ.get("OPENCDA_SPECTATOR_HEIGHT_M", "4.5"))
    location = transform.location + carla.Location(
        x=-follow_distance_m * math.cos(yaw_rad),
        y=-follow_distance_m * math.sin(yaw_rad),
        z=follow_height_m,
    )
    spectator.set_transform(carla.Transform(
        location,
        carla.Rotation(
            pitch=float(os.environ.get("OPENCDA_SPECTATOR_PITCH_DEG", "-15.0")),
            yaw=float(transform.rotation.yaw),
            roll=0.0,
        )
    ))


def run_scenario(opt, config_yaml):
    cav_world = None
    scenario_manager = None
    eval_manager = None
    debug_viewer = None
    single_cav_list = []
    bg_veh_list = []
    try:
        scenario_params = add_current_time(config_yaml)

        # create CAV world
        cav_world = CavWorld(apply_ml=opt.apply_ml,
                             apply_coperception=True,
                             coperception_params=scenario_params['coperception'])

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("opv2v_data_protocol_2021_08_18_19_48_05.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single', 'cooperative'],
                                                    data_dump=False)
        # single_cav_list = \
        #     scenario_manager.create_vehicle_manager(application=['single'],
        #                                             data_dump=False)
        # rsu_list = \
        #     scenario_manager.create_rsu_manager(data_dump=False)

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='coop_town06',
                              current_time=scenario_params['current_time'])
        if OpenCDADebugViewer.enabled_from_env() and single_cav_list:
            debug_viewer = OpenCDADebugViewer(
                world=scenario_manager.world,
                carla_module=carla,
                ego_vehicle=single_cav_list[0].vehicle,
            )

        spectator = scenario_manager.world.get_spectator()
        while True:
            scenario_manager.tick()
            cav_world.tick()
            _set_spectator_transform(spectator, single_cav_list[0].vehicle)

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                if bool(getattr(single_cav, "_opencda_agent_finished", False)):
                    control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                else:
                    try:
                        control = single_cav.run_step()
                    except SystemExit:
                        if getattr(single_cav, "cpx_planner", None) is not None:
                            raise
                        single_cav._opencda_agent_finished = True
                        control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                single_cav.vehicle.apply_control(control)

            if debug_viewer is not None:
                debug_viewer.render(single_cav_list)

            # for rsu in rsu_list:
            #     rsu.update_info()
            #     rsu.run_step()

    finally:
        if eval_manager is not None:
            eval_manager.evaluate()
        if cav_world is not None and cav_world.ml_manager is not None:
            cav_world.ml_manager.evaluate_final_average_precision()

        if opt.record and scenario_manager is not None:
            scenario_manager.client.stop_recorder()

        if debug_viewer is not None:
            debug_viewer.destroy()

        if scenario_manager is not None:
            scenario_manager.close()

        for v in single_cav_list:
            v.destroy()
        # for r in rsu_list:
        #     r.destroy()
        for v in bg_veh_list:
            v.destroy()
