# -*- coding: utf-8 -*-
"""
Basic class of CAV
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import uuid

import carla

from opencda.core.actuation.control_manager \
    import ControlManager
from opencda.core.application.platooning.platoon_behavior_agent\
    import PlatooningBehaviorAgent
from opencda.core.common.v2x_manager \
    import V2XManager
from opencda.core.sensing.localization.localization_manager \
    import LocalizationManager
from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager
from opencda.core.safety.safety_manager import SafetyManager
from opencda.core.plan.behavior_agent \
    import BehaviorAgent
from opencda.core.map.map_manager import MapManager
from opencda.core.common.data_dumper import DataDumper
from opencda.planning_module.opencda_bridge.cpx_mpc_planner import (
    CPXMPCPlannerBridge,
    cpx_planner_enabled,
)


DEFAULT_SAFETY_MANAGER_CONFIG = {
    'print_message': True,
    'collision_sensor': {
        'history_size': 30,
        'col_thresh': 1,
    },
    'stuck_dector': {
        'len_thresh': 500,
        'speed_thresh': 0.5,
    },
    'offroad_dector': [],
    'traffic_light_detector': {
        'light_dist_thresh': 20,
    },
}


class VehicleManager(object):
    """
    A class manager to embed different modules with vehicle together.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle. We need this class to spawn our gnss and imu sensor.

    config_yaml : dict
        The configuration dictionary of this CAV.

    application : list
        The application category, currently support:['single','platoon'].

    carla_map : carla.Map
        The CARLA simulation map.

    cav_world : opencda object
        CAV World. This is used for V2X communication simulation.

    current_time : str
        Timestamp of the simulation beginning, used for data dumping.

    data_dumping : bool
        Indicates whether to dump sensor data during simulation.

    Attributes
    ----------
    v2x_manager : opencda object
        The current V2X manager.

    localizer : opencda object
        The current localization manager.

    perception_manager : opencda object
        The current V2X perception manager.

    agent : opencda object
        The current carla agent that handles the basic behavior
         planning of ego vehicle.

    controller : opencda object
        The current control manager.

    data_dumper : opencda object
        Used for dumping sensor data.
    """

    def __init__(
            self,
            vehicle,
            config_yaml,
            application,
            carla_map,
            cav_world,
            current_time='',
            data_dumping=False):

        # an unique uuid for this vehicle
        self.vid = str(uuid.uuid1())
        self.vehicle = vehicle
        self.carla_map = carla_map

        # retrieve the configure for different modules
        sensing_config = config_yaml['sensing']
        map_config = config_yaml['map_manager']
        behavior_config = config_yaml['behavior']
        control_config = config_yaml['controller']
        v2x_config = config_yaml['v2x']

        # v2x module
        self.v2x_manager = V2XManager(cav_world, v2x_config, self.vid)
        # localization module
        self.localizer = LocalizationManager(
            vehicle, sensing_config['localization'], carla_map)
        # perception module
        self.perception_manager = PerceptionManager(
            vehicle, sensing_config['perception'], cav_world,
            data_dumping)
        # map manager
        self.map_manager = MapManager(vehicle,
                                      carla_map,
                                      map_config)
        # safety manager
        safety_config = config_yaml.get(
            'safety_manager', DEFAULT_SAFETY_MANAGER_CONFIG)
        self.safety_manager = SafetyManager(cav_world=cav_world,
                                            vehicle=vehicle,
                                            params=safety_config)
        cpx_requested = cpx_planner_enabled(config_yaml)
        cpx_mode = str(
            config_yaml.get('planner', {}).get('mode', 'full_cpx_mpc')
        ).strip().lower()
        cpx_full_pipeline_requested = bool(cpx_requested) and cpx_mode not in {
            'opencda_reference_mpc',
            'opencda_ref_mpc',
            'mode2',
        }
        cpx_single_cav_supported = bool(cpx_full_pipeline_requested) and 'platooning' not in application

        # behavior agent / CP-X planner are mutually exclusive in full CP-X mode.
        self.agent = None
        self.cpx_planner = None
        self.controller = None
        if bool(cpx_single_cav_supported):
            self.cpx_planner = CPXMPCPlannerBridge(
                vehicle_manager=self,
                config=config_yaml.get('planner', {}),
                map_planner=carla_map,
            )
            print(
                "[OpenCDA VehicleManager] CP-X MPC planner bridge enabled "
                f"for vehicle {self.vehicle.id}."
            )
        elif 'platooning' in application:
            platoon_config = config_yaml['platoon']
            self.agent = PlatooningBehaviorAgent(
                vehicle,
                self,
                self.v2x_manager,
                behavior_config,
                platoon_config,
                carla_map)
        else:
            self.agent = BehaviorAgent(vehicle, carla_map, behavior_config)

        if (
                bool(cpx_requested)
                and not bool(cpx_full_pipeline_requested)
                and 'platooning' not in application):
            self.cpx_planner = CPXMPCPlannerBridge(
                vehicle_manager=self,
                config=config_yaml.get('planner', {}),
                map_planner=carla_map,
            )
            print(
                "[OpenCDA VehicleManager] CP-X legacy OpenCDA-reference "
                f"planner bridge enabled for vehicle {self.vehicle.id}."
            )
        elif bool(cpx_full_pipeline_requested) and not bool(cpx_single_cav_supported):
            print(
                "[OpenCDA VehicleManager] CP-X full pipeline is currently "
                "limited to single-CAV scenarios; using the OpenCDA planner "
                f"for vehicle {self.vehicle.id}."
            )

        # Control module. CP-X full pipeline returns carla.VehicleControl
        # directly, so OpenCDA PID/ControlManager is only needed for legacy mode.
        if self.agent is not None:
            self.controller = ControlManager(control_config)

        if data_dumping:
            self.data_dumper = DataDumper(self.perception_manager,
                                          vehicle.id,
                                          save_time=current_time)
        else:
            self.data_dumper = None

        cav_world.update_vehicle_manager(self)

    def set_destination(
            self,
            start_location,
            end_location,
            clean=False,
            end_reset=True):
        """
        Set global route.

        Parameters
        ----------
        start_location : carla.location
            The CAV start location.

        end_location : carla.location
            The CAV destination.

        clean : bool
             Indicator of whether clean waypoint queue.

        end_reset : bool
            Indicator of whether reset the end location.

        Returns
        -------
        """

        cpx_mode = str(getattr(self.cpx_planner, 'mode', '')).strip().lower()
        cpx_full_pipeline_active = (
            self.cpx_planner is not None
            and cpx_mode not in {'opencda_reference_mpc', 'opencda_ref_mpc', 'mode2'}
        )
        if bool(cpx_full_pipeline_active):
            self.cpx_planner.set_destination(
                start_location=start_location,
                end_location=end_location,
                clean=clean,
                end_reset=end_reset,
            )
            return

        self.agent.set_destination(
            start_location, end_location, clean, end_reset)

    def update_info(self):
        """
        Call perception and localization module to
        retrieve surrounding info an ego position.
        """
        # localization
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()
        ego_spd = self.localizer.get_ego_spd()

        # object detection
        objects = self.perception_manager.detect(ego_pos)

        # update the ego pose for map manager
        self.map_manager.update_information(ego_pos)

        # this is required by safety manager
        safety_input = {
            'ego_pos': ego_pos,
            'ego_speed': ego_spd,
            'objects': objects,
            'carla_map': self.carla_map,
            'world': self.vehicle.get_world(),
            'static_bev': self.map_manager.static_bev,
            'vis_bev': self.map_manager.vis_bev
        }
        self.safety_manager.update_info(safety_input)

        # update ego position and speed to v2x manager,
        # and then v2x manager will search the nearby cavs
        self.v2x_manager.update_info(ego_pos, ego_spd)

        if self.cpx_planner is not None:
            self.cpx_planner.update_information(
                ego_transform=ego_pos,
                ego_speed_kmh=ego_spd,
                detected_objects=objects,
                v2x_manager=self.v2x_manager,
                safety_manager=self.safety_manager,
                map_manager=self.map_manager,
            )

        cpx_mode = str(getattr(self.cpx_planner, 'mode', '')).strip().lower()
        cpx_full_pipeline_active = (
            self.cpx_planner is not None
            and cpx_mode not in {'opencda_reference_mpc', 'opencda_ref_mpc', 'mode2'}
        )
        if not bool(cpx_full_pipeline_active):
            self.agent.update_information(ego_pos, ego_spd, objects)
            # pass position and speed info to controller
            self.controller.update_info(ego_pos, ego_spd)

    def run_step(self, target_speed=None):
        """
        Execute one step of navigation.
        """
        # visualize the bev map if needed
        #print('=============================================running steps==========================================================')
        self.map_manager.run_step()
        if self.cpx_planner is not None:
            try:
                return self.cpx_planner.run_step()
            except Exception as exc:
                print(
                    "[OpenCDA VehicleManager] CP-X MPC planner bridge failed; "
                    f"fallback_policy={getattr(self.cpx_planner, 'fallback_policy', '')}: {exc}"
                )
                if getattr(self.cpx_planner, 'fallback_policy', '') == 'raise':
                    raise
                return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        #print('=============================================running steps1==========================================================')
        target_speed, target_pos = self.agent.run_step(target_speed)
        #print('=============================================running steps2==========================================================')
        control = self.controller.run_step(target_speed, target_pos)
        #print('=============================================running steps3==========================================================')

        # dump data
        if self.data_dumper:
            #print('=============================================running steps4==========================================================')
            self.data_dumper.run_step(self.perception_manager,
                                      self.localizer,
                                      self.agent)
            
        #print('=============================================running steps5==========================================================')

        return control

    def destroy(self):
        """
        Destroy the actor vehicle
        """
        if getattr(self, 'cpx_planner', None):
            destroy = getattr(self.cpx_planner, 'destroy', None)
            if callable(destroy):
                destroy()
        if getattr(self, 'perception_manager', None):
            self.perception_manager.destroy()
        if getattr(self, 'localizer', None):
            self.localizer.destroy()
        if getattr(self, 'safety_manager', None):
            self.safety_manager.destroy()
        if getattr(self, 'map_manager', None):
            self.map_manager.destroy()
        if getattr(self, 'vehicle', None):
            self.vehicle.destroy()
