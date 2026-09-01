# -*- coding: utf-8 -*-
"""
Basic class of CAV
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import json
import math
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
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
from opencda.planning_module.utility.cp_messages import load_cp_message_payload
from opencda.planning_module.opencda_bridge.cp_provider import OpenCDACPProvider
from opencda.planning_module.pipeline.actuator_mapper import CarlaActuatorMapper
from opencda.planning_module.utility.config_loader import load_yaml_file
from opencda import data_transmitter
from opencda.data_receiver import DataReceiver


# Run exactly one planner. True selects the ROS planner; False selects the local CP-X planner.
USE_ROS_COMMAND = False

# Save the large planner input/debug records only when an input-frame comparison is needed.
store_input_frame = False

# Save detailed transport and planner-stage timing only when performance debugging is needed.
debug_time = True


def _planner_input_json_safe(value):
    """Convert the complete adapter output into JSON without changing planner values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value.__class__.__name__ == "Waypoint":
        if callable(getattr(value, "to_dict", None)):
            return _planner_input_json_safe(value.to_dict())
        transform = getattr(value, "transform", None)
        location = getattr(transform, "location", None)
        rotation = getattr(transform, "rotation", None)
        return {
            "position": {"x": float(getattr(location, "x", 0.0)), "y": float(getattr(location, "y", 0.0)), "z": float(getattr(location, "z", 0.0))},
            "road_id": int(getattr(value, "road_id", 0) or 0),
            "section_id": int(getattr(value, "section_id", 0) or 0),
            "lane_id": int(getattr(value, "lane_id", 0) or 0),
            "parametric_offset": float(getattr(value, "s", 0.0) or 0.0),
            "heading": math.radians(float(getattr(rotation, "yaw", 0.0) or 0.0)),
            "lane_width_m": float(getattr(value, "lane_width", 0.0) or 0.0),
            "is_intersection": bool(getattr(value, "is_junction", False)),
        }
    if is_dataclass(value):
        return {field.name: _planner_input_json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _planner_input_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_planner_input_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _planner_input_json_safe(item_method())
        except Exception:
            pass
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return _planner_input_json_safe(list_method())
        except Exception:
            pass
    return str(value)


def _flatten_planner_input(value, path="$", output=None):
    """Flatten nested adapter data so each compared field has one readable path."""
    if output is None:
        output = {}
    if isinstance(value, dict):
        if not value:
            output[path] = {}
        for key in sorted(value.keys(), key=str):
            _flatten_planner_input(value[key], "{}.{}".format(path, key), output)
        return output
    if isinstance(value, list):
        output["{}.__length__".format(path)] = len(value)
        if not value:
            output[path] = []
        for index, item in enumerate(value):
            _flatten_planner_input(item, "{}[{}]".format(path, index), output)
        return output
    output[path] = value
    return output


def _compare_planner_input_values(opencda_value, ros_value, numeric_tolerance=0.1):
    """Compare the two complete input contracts field by field."""
    missing = "<missing>"
    opencda_fields = _flatten_planner_input(opencda_value)
    ros_fields = _flatten_planner_input(ros_value)
    comparisons = {}
    matching_count = 0
    for path in sorted(set(opencda_fields) | set(ros_fields)):
        left = opencda_fields.get(path, missing)
        right = ros_fields.get(path, missing)
        if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            difference = abs(float(left) - float(right))
            result = {"opencda": left, "ros": right, "difference": difference, "tolerance": float(numeric_tolerance), "match": difference <= float(numeric_tolerance)}
        else:
            result = {"opencda": left, "ros": right, "match": left == right}
        comparisons[path] = result
        matching_count += int(bool(result["match"]))
    return {"numeric_tolerance": float(numeric_tolerance), "field_count": len(comparisons), "matching_field_count": matching_count, "different_field_count": len(comparisons) - matching_count, "match": matching_count == len(comparisons), "fields": comparisons}


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
        self._ros_final_destination = None
        self._latest_ros_input_snapshot = None
        self.ros_receiver = None
        self.ros_planner_enabled = False
        self.ros_cp_provider = None
        self.ros_actuator_mapper = None

        # retrieve the configure for different modules
        sensing_config = config_yaml['sensing']
        map_config = config_yaml['map_manager']
        behavior_config = config_yaml['behavior']
        control_config = config_yaml['controller']
        v2x_config = config_yaml['v2x']
        planner_config = dict(config_yaml.get('planner', {}) or {})
        self.store_input_frame = bool(store_input_frame)
        self.debug_time = bool(debug_time)
        # `store_input_frame` force-enables the heavy records; it must not
        # silently disable `record_debug` / `record_evaluation_metrics` when the
        # scenario YAML asked for them (the per-tick planner debug CSV/JSONL is
        # the primary diagnostic surface).
        planner_config["record_debug"] = bool(
            planner_config.get("record_debug", False)
        ) or bool(self.store_input_frame)
        planner_config["record_evaluation_metrics"] = bool(
            planner_config.get("record_evaluation_metrics", False)
        ) or bool(self.store_input_frame)
        planner_config["draw_world_debug"] = False
        debug_output_dir = Path(str(planner_config.get("debug_output_dir", "opencda/planning_module/opencda_bridge/debug")))
        self.shadow_comparison_timeout_s = max(0.0, float(planner_config.get("shadow_comparison_timeout_s", 5.0)))
        self.shadow_comparison_output_path = debug_output_dir / "planner_comparison.jsonl"
        self.opencda_planner_input_output_path = debug_output_dir / "opencda_planner_input_adapter_output.jsonl"
        self.ros_planner_control_output_path = debug_output_dir / "ros_planner_control_output.jsonl"
        self.opencda_planner_control_output_path = debug_output_dir / "opencda_planner_control_output.jsonl"
        self.opencda_planner_timing_summary_path = debug_output_dir / "opencda_planner_timing_summary.json"
        self.ros_command_timing_summary_path = debug_output_dir / "ros_command_timing_summary.json"
        self._shadow_cycle_time_s = None
        self._ros_cycle_sent = False
        self._ros_retry_after_monotonic = 0.0
        self._latest_ros_output = None
        self.selected_planner_visualization_enabled = False
        self.selected_planner_trajectory_life_time_s = max(0.05, float(planner_config.get("selected_planner_trajectory_life_time_s", 0.20)))
        self.selected_planner_route_refresh_s = max(0.05, float(planner_config.get("selected_planner_route_refresh_s", 0.50)))
        self.selected_planner_route_life_time_s = max(self.selected_planner_route_refresh_s + 0.10, float(planner_config.get("selected_planner_route_life_time_s", 0.75)))
        self._last_planner_route_draw_time_s = -1.0e9
        self._last_visualized_planner_source = ""
        self._planner_input_ready_monotonic = None
        self._timing_statistics = {}
        self._planner_stage_profiler = None
        self.shadow_comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
        if bool(USE_ROS_COMMAND):
            self.ros_planner_control_output_path.write_text("", encoding="utf-8")
        else:
            self.opencda_planner_control_output_path.write_text("", encoding="utf-8")
        if self.debug_time:
            active_timing_path = self.ros_command_timing_summary_path if bool(USE_ROS_COMMAND) else self.opencda_planner_timing_summary_path
            active_timing_path.write_text("{}\n", encoding="utf-8")

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
        cpx_full_pipeline_requested = bool(cpx_requested)
        cpx_single_cav_supported = bool(cpx_full_pipeline_requested) and 'platooning' not in application
        self.ros_planner_enabled = bool(cpx_single_cav_supported) and bool(USE_ROS_COMMAND)

        # behavior agent / CP-X planner are mutually exclusive in full CP-X mode.
        self.agent = None
        self.cpx_planner = None
        self.controller = None
        if bool(cpx_single_cav_supported) and not bool(self.ros_planner_enabled):
            # CP-X owns planning and lateral steering, while OpenCDA remains
            # the platform-level longitudinal controller.  Construct the
            # normal ControlManager before the bridge so the planner can hand
            # it target velocity commands without owning throttle/brake.
            self.controller = ControlManager(control_config)
            self.cpx_planner = CPXMPCPlannerBridge(
                vehicle_manager=self,
                config=planner_config,
                map_planner=carla_map,
            )
            if self.debug_time:
                from opencda.planning_module.pipeline.performance_profiler import PlannerStageProfiler

                self._planner_stage_profiler = PlannerStageProfiler(enabled=True)
                self._planner_stage_profiler.instrument_global_planner(getattr(self.cpx_planner, "global_planner", None))
                self._planner_stage_profiler.instrument_reference_map(getattr(self.cpx_planner, "reference_map", None))
                self._planner_stage_profiler.instrument_route_manager(getattr(self.cpx_planner, "route_manager", None))
                self._planner_stage_profiler.instrument_bridge(self.cpx_planner)
                self._planner_stage_profiler.instrument_mpc(getattr(self.cpx_planner, "mpc", None))
            print(
                "[OpenCDA VehicleManager] CP-X MPC planner bridge enabled "
                f"for vehicle {self.vehicle.id}."
            )
        elif bool(self.ros_planner_enabled):
            self._initialize_ros_planner_boundary(planner_config)
            print("[OpenCDA VehicleManager] ROS CP-X planner boundary enabled for vehicle {}.".format(self.vehicle.id))
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

        if bool(cpx_full_pipeline_requested) and not bool(cpx_single_cav_supported):
            print(
                "[OpenCDA VehicleManager] CP-X full pipeline is currently "
                "limited to single-CAV scenarios; using the OpenCDA planner "
                f"for vehicle {self.vehicle.id}."
            )

        # CP-X returns carla.VehicleControl directly. OpenCDA's controller is
        # only constructed for vehicles using OpenCDA's BehaviorAgent.
        if self.agent is not None:
            self.controller = ControlManager(control_config)

        if data_dumping:
            self.data_dumper = DataDumper(self.perception_manager,
                                          vehicle.id,
                                          save_time=current_time)
        else:
            self.data_dumper = None

        cav_world.update_vehicle_manager(self)

    def _initialize_ros_planner_boundary(self, planner_config):
        """Create only the OpenCDA-to-ROS boundary objects, not a local planner."""
        planning_module_root = Path(__file__).resolve().parents[2] / "planning_module"
        mpc_path = Path(str(planner_config.get("mpc_config_path", planning_module_root / "MPC" / "mpc.yaml"))).expanduser()
        if not mpc_path.is_absolute():
            mpc_path = planning_module_root / mpc_path
        mpc_payload = load_yaml_file(str(mpc_path))
        mpc_config = dict(mpc_payload.get("mpc", mpc_payload) or {})
        constraints = dict(mpc_config.get("constraints", {}) or {})
        self._ros_max_acceleration_mps2 = float(constraints.get("max_acceleration_mps2", 3.0))
        self._ros_min_acceleration_mps2 = float(constraints.get("min_acceleration_mps2", -3.0))
        self._ros_max_steer_rad = max(1.0e-6, float(constraints.get("max_steer_rad", 0.6)))
        self.ros_actuator_mapper = CarlaActuatorMapper(planner_config)
        cp_path = Path(str(planner_config.get("cp_message_path", planning_module_root / "behavior_planner" / "cp_message.json"))).expanduser()
        if not cp_path.is_absolute():
            cp_path = planning_module_root / cp_path
        self.ros_cp_message_path = str(cp_path)
        if bool(planner_config.get("publish_cp_message", True)):
            self.ros_cp_provider = OpenCDACPProvider(
                message_path=self.ros_cp_message_path,
                schema_version=1,
                communication_range_m=float(planner_config.get("communication_range_m", 80.0)),
                prediction_horizon_s=float(mpc_config.get("horizon_s", 4.5)),
                prediction_dt_s=float(mpc_config.get("plan_dt_s", 0.1)),
                source="native_opencda",
                require_native_opencda=bool(planner_config.get("require_native_opencda_cp", True)),
                visibility_filter_enabled=bool(planner_config.get("cp_visibility_filter_enabled", False)),
                visibility_backend=str(planner_config.get("cp_visibility_backend", "actor_geometry")),
                visibility_sensor_height_m=float(planner_config.get("cp_visibility_sensor_height_m", 1.6)),
                visibility_target_tolerance_m=float(planner_config.get("cp_visibility_target_tolerance_m", 0.75)),
            )
        self.ros_receiver = DataReceiver().start()

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

        self._ros_final_destination = {"x": float(end_location.x), "y": float(end_location.y), "z": float(getattr(end_location, "z", 0.0))}

        if bool(self.ros_planner_enabled):
            return

        cpx_full_pipeline_active = self.cpx_planner is not None
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

        if bool(self.ros_planner_enabled) and self.ros_actuator_mapper is not None:
            self.ros_actuator_mapper.update_measurement(speed_mps=float(ego_spd) / 3.6, timestamp_s=float(self.vehicle.get_world().get_snapshot().timestamp.elapsed_seconds))

        if self.cpx_planner is not None:
            # Keep OpenCDA's longitudinal PID measurement current even though
            # its BehaviorAgent and lateral waypoint PID are bypassed.
            self.controller.update_info(ego_pos, ego_spd)
            self.cpx_planner.update_information(
                ego_transform=ego_pos,
                ego_speed_kmh=ego_spd,
                detected_objects=objects,
                v2x_manager=self.v2x_manager,
                safety_manager=self.safety_manager,
                map_manager=self.map_manager,
            )
            self._planner_input_ready_monotonic = time.perf_counter()

        if bool(self.ros_planner_enabled):
            self._latest_ros_input_snapshot = {
                "ego_pos": ego_pos,
                "ego_spd": float(ego_spd),
                "objects": objects,
                "sim_time_s": float(self.vehicle.get_world().get_snapshot().timestamp.elapsed_seconds),
            }
            self._planner_input_ready_monotonic = time.perf_counter()

        cpx_full_pipeline_active = self.cpx_planner is not None
        if not bool(cpx_full_pipeline_active) and not bool(self.ros_planner_enabled):
            self.agent.update_information(ego_pos, ego_spd, objects)
            # pass position and speed info to controller
            self.controller.update_info(ego_pos, ego_spd)

    def _record_timing_summary(self, source, planning_cycle_time_ms, input_to_output_time_ms):
        """Store average planner timing in the file for the currently selected planner."""
        if not self.debug_time:
            return
        source = str(source)
        statistics = self._timing_statistics.setdefault(source, {"sample_count": 0, "planning_total_ms": 0.0, "planning_min_ms": None, "planning_max_ms": 0.0, "input_output_total_ms": 0.0, "input_output_min_ms": None, "input_output_max_ms": 0.0})
        planning_time = max(0.0, float(planning_cycle_time_ms))
        input_output_time = max(0.0, float(input_to_output_time_ms))
        statistics["sample_count"] += 1
        statistics["planning_total_ms"] += planning_time
        statistics["planning_min_ms"] = planning_time if statistics["planning_min_ms"] is None else min(float(statistics["planning_min_ms"]), planning_time)
        statistics["planning_max_ms"] = max(float(statistics["planning_max_ms"]), planning_time)
        statistics["input_output_total_ms"] += input_output_time
        statistics["input_output_min_ms"] = input_output_time if statistics["input_output_min_ms"] is None else min(float(statistics["input_output_min_ms"]), input_output_time)
        statistics["input_output_max_ms"] = max(float(statistics["input_output_max_ms"]), input_output_time)
        count = max(1, int(statistics["sample_count"]))
        use_ros = source == "ros"
        output_path = self.ros_command_timing_summary_path if use_ros else self.opencda_planner_timing_summary_path
        summary = {
            "debug_time": True,
            "use_ros_command": bool(use_ros),
            "planner_source": "ros_planner" if use_ros else "opencda_planner",
            "sample_count": count,
            "planning_cycle_time_ms": {"average": float(statistics["planning_total_ms"]) / count, "minimum": float(statistics["planning_min_ms"] or 0.0), "maximum": float(statistics["planning_max_ms"])},
            "input_to_output_time_ms": {"average": float(statistics["input_output_total_ms"]) / count, "minimum": float(statistics["input_output_min_ms"] or 0.0), "maximum": float(statistics["input_output_max_ms"])},
            "definitions": {
                "planning_cycle_time_ms": "Time used by the selected planner to generate one planner output after its complete input is ready.",
                "input_to_output_time_ms": "OpenCDA planner: input-ready to local output. ROS planner: OpenCDA TCP send start to ROS control received back in OpenCDA.",
                "planner_operation_breakdown": "Nested method timings are diagnostic and must not be added together. A parent method can include its child method time.",
            },
        }
        if not use_ros and self._planner_stage_profiler is not None:
            summary["planner_operation_breakdown"] = self._planner_stage_profiler.summary(count)
        output_path.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _exact_pair(opencda_value, ros_value):
        """Place matching text or integer values beside each other."""
        return {"opencda": opencda_value, "ros": ros_value, "match": opencda_value == ros_value}

    @staticmethod
    def _numeric_pair(opencda_value, ros_value, tolerance):
        """Place numeric values beside each other and calculate their difference."""
        opencda_number = float(opencda_value)
        ros_number = float(ros_value)
        difference = abs(opencda_number - ros_number)
        return {"opencda": opencda_number, "ros": ros_number, "difference": difference, "tolerance": float(tolerance), "match": difference <= float(tolerance)}

    def _send_current_cycle_to_ros(self):
        """Send one raw module cycle to ROS without running the local planner."""
        snapshot = dict(self._latest_ros_input_snapshot or {})
        if not bool(self.ros_planner_enabled) or not snapshot or self._ros_final_destination is None:
            return False
        if time.monotonic() < float(self._ros_retry_after_monotonic):
            return False
        if self.ros_cp_provider is not None:
            try:
                self.ros_cp_provider.publish(world=self.vehicle.get_world(), map_planner=self.carla_map, ego_vehicle=self.vehicle, sim_time_s=float(snapshot["sim_time_s"]), vehicle_manager=self)
            except Exception as exc:
                print("[OpenCDA VehicleManager] Native CP publish failed: {}".format(exc))
        cp_payload = load_cp_message_payload(self.ros_cp_message_path)
        results = data_transmitter.send(
            localization_transform=snapshot["ego_pos"],
            localization_speed_kmh=float(snapshot["ego_spd"]),
            perception_objects=snapshot["objects"],
            v2x_manager=self.v2x_manager,
            safety_manager=self.safety_manager,
            final_destination=self._ros_final_destination,
            cp_payload=cp_payload,
            cooperative_payload=cp_payload,
            timestamp_s=float(snapshot["sim_time_s"]),
            timeout_s=0.05,
        )
        self._shadow_cycle_time_s = float(snapshot["sim_time_s"])
        self._ros_cycle_sent = bool(results) and all(bool(value) for value in results.values())
        if not self._ros_cycle_sent:
            self._ros_retry_after_monotonic = time.monotonic() + 2.0
        return bool(self._ros_cycle_sent)

    def _write_planner_input_adapter_output(self, adapter_output, cycle_time_s):
        """Save every OpenCDA input-contract field for the same-cycle comparison."""
        if not bool(self.store_input_frame) or adapter_output is None:
            return
        record = {"schema_version": 1, "source": "opencda", "cycle_time_s": float(cycle_time_s), "planner_input_adapter_output": _planner_input_json_safe(adapter_output)}
        with open(str(self.opencda_planner_input_output_path), "a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def _compare_planner_outputs(self, local_output, local_adapter_output, ros_output, cycle_time_s):
        """Save the same input, behavior, reference, and MPC comparisons used previously."""
        if not bool(self.store_input_frame) or local_output is None or local_adapter_output is None or not isinstance(ros_output, dict):
            return
        local_frame = local_adapter_output.frame
        local_behavior = local_output.behavior_command.as_dict()
        local_diagnostics = local_output.diagnostics_dict()
        ros_input = dict(ros_output.get("input", {}) or {})
        ros_behavior = dict(ros_output.get("behavior", {}) or {})
        ros_mpc = dict(ros_output.get("mpc", {}) or {})
        ros_ego_state = list(ros_input.get("ego_state", [0.0, 0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0, 0.0])
        while len(ros_ego_state) < 4:
            ros_ego_state.append(0.0)
        adapter_comparison = _compare_planner_input_values(_planner_input_json_safe(local_adapter_output), ros_output.get("planner_input_adapter_output"), numeric_tolerance=0.1)
        local_reference = [[float(dict(sample).get("x_ref_m", dict(sample).get("x", 0.0))), float(dict(sample).get("y_ref_m", dict(sample).get("y", 0.0)))] for sample in list(local_output.reference_trajectory or []) if isinstance(sample, dict)]
        ros_reference = list(ros_output.get("reference_xy", []) or [])
        local_trajectory = [[float(value) for value in list(state)[:4]] for state in list(local_output.planned_trajectory or [])]
        ros_trajectory = list(ros_output.get("planned_trajectory", []) or [])
        comparison = {
            "cycle_time_s": float(cycle_time_s),
            "input": {
                "ego_x_m": self._numeric_pair(local_frame.planning.ego.x_m, ros_ego_state[0], 0.1),
                "ego_y_m": self._numeric_pair(local_frame.planning.ego.y_m, ros_ego_state[1], 0.1),
                "ego_speed_mps": self._numeric_pair(local_frame.planning.ego.speed_mps, ros_ego_state[2], 0.1),
                "ego_heading_rad": self._numeric_pair(local_frame.planning.ego.heading_rad, ros_ego_state[3], 0.1),
                "current_lane_id": self._exact_pair(int(local_frame.map_lane.lane_id), int(ros_input.get("current_lane_id", 0))),
                "object_count": self._exact_pair(int(local_frame.perception.planning_count), int(ros_input.get("object_count", 0))),
                "predicted_object_count": self._exact_pair(int(local_frame.prediction.predicted_object_count), int(ros_input.get("predicted_object_count", 0))),
                "v2x_obstacle_count": self._exact_pair(int(local_frame.cp_messages.obstacle_count), int(ros_input.get("v2x_obstacle_count", 0))),
                "traffic_signal_state": self._exact_pair(str(local_frame.planning.traffic_control.signal_state), str(ros_input.get("traffic_signal_state", ""))),
                "route_point_count": self._exact_pair(len(local_adapter_output.route_points), int(ros_input.get("route_point_count", 0))),
            },
            "behavior": {
                "decision": self._exact_pair(str(local_behavior.get("decision", "")), str(ros_behavior.get("decision", ""))),
                "target_lane_id": self._exact_pair(int(local_behavior.get("target_lane_id", 0)), int(ros_behavior.get("target_lane_id", 0))),
                "target_speed_mps": self._numeric_pair(local_behavior.get("target_speed_mps", 0.0), ros_behavior.get("target_speed_mps", 0.0), 1.0),
                "fsm_state": self._exact_pair(str(local_behavior.get("fsm_state", "")), str(ros_behavior.get("fsm_state", ""))),
                "reroute_requested": self._exact_pair(bool(local_behavior.get("reroute_requested", False)), bool(ros_behavior.get("reroute_requested", False))),
                "emergency_brake": self._exact_pair(bool(local_behavior.get("emergency_brake", False)), bool(ros_behavior.get("emergency_brake", False))),
            },
            "mpc": {
                "acceleration_mps2": self._numeric_pair(local_output.acceleration_mps2, ros_mpc.get("acceleration_mps2", 0.0), 0.5),
                "steering_rad": self._numeric_pair(local_output.steering_rad, ros_mpc.get("steering_rad", 0.0), 0.1),
                "status": self._exact_pair(str(local_diagnostics.get("mpc_status", "")), str(ros_mpc.get("status", ""))),
                "fallback_reason": self._exact_pair(str(local_diagnostics.get("mpc_fallback_reason", "")), str(ros_mpc.get("fallback_reason", ""))),
                "replan_executed": self._exact_pair(bool(local_diagnostics.get("mpc_replan_executed", False)), bool(ros_mpc.get("replan_executed", False))),
            },
            "reference": {"point_count": self._exact_pair(len(local_reference), len(ros_reference))},
            "planned_trajectory": {"point_count": self._exact_pair(len(local_trajectory), len(ros_trajectory))},
            "planner_input_adapter_output": adapter_comparison,
        }
        comparison["summary"] = {
            "input_match": all(field.get("match", False) for field in comparison["input"].values()),
            "behavior_match": all(field.get("match", False) for field in comparison["behavior"].values()),
            "mpc_match": all(field.get("match", False) for field in comparison["mpc"].values()),
            "planner_input_adapter_output_match": bool(adapter_comparison["match"]),
        }
        with open(str(self.shadow_comparison_output_path), "a", encoding="utf-8") as comparison_file:
            comparison_file.write(json.dumps(comparison, allow_nan=False, default=str) + "\n")

    @staticmethod
    def _planner_visualization_xy_points(raw_points):
        """Convert a route or trajectory into finite XY pairs for CARLA drawing."""
        points_xy = []
        for raw_point in list(raw_points or []):
            try:
                if isinstance(raw_point, dict):
                    x_value = raw_point.get("x", raw_point.get("x_m", raw_point.get("x_ref_m")))
                    y_value = raw_point.get("y", raw_point.get("y_m", raw_point.get("y_ref_m")))
                else:
                    x_value, y_value = raw_point[0], raw_point[1]
                x_m, y_m = float(x_value), float(y_value)
                if math.isfinite(x_m) and math.isfinite(y_m):
                    points_xy.append((x_m, y_m))
            except (IndexError, KeyError, TypeError, ValueError):
                continue
        return points_xy

    def _selected_planner_visualization_data(self, ros_output):
        """Select route and trajectory from the planner whose command is applied."""
        if USE_ROS_COMMAND:
            if not isinstance(ros_output, dict):
                return [], [], "ROS"
            adapter_output = ros_output.get("planner_input_adapter_output", {})
            adapter_output = adapter_output if isinstance(adapter_output, dict) else {}
            return list(ros_output.get("planned_trajectory", []) or []), list(adapter_output.get("route_points", []) or []), "ROS"
        local_output = getattr(self.cpx_planner, "last_output", None)
        local_adapter_output = getattr(self.cpx_planner, "last_adapter_output", None)
        diagnostics = (
            dict(local_output.diagnostics_dict())
            if local_output is not None and callable(getattr(local_output, "diagnostics_dict", None))
            else {}
        )
        display_route = list(diagnostics.get("global_route_points", []) or [])
        if not display_route:
            display_route = list(getattr(local_adapter_output, "route_points", []) or [])
        return list(getattr(local_output, "planned_trajectory", []) or []), display_route, "OpenCDA"

    def _draw_selected_planner_visualization(self, ros_output):
        """Draw the selected trajectory in green and selected global route in yellow."""
        if not self.selected_planner_visualization_enabled:
            return
        trajectory, route_points, source = self._selected_planner_visualization_data(ros_output)
        trajectory_xy = self._planner_visualization_xy_points(trajectory)
        route_xy = self._planner_visualization_xy_points(route_points)
        if not trajectory_xy and not route_xy:
            return
        world = self.vehicle.get_world()
        debug = getattr(world, "debug", None)
        if debug is None:
            return
        drawing_z = float(self.vehicle.get_location().z) + 0.40
        simulation_time_s = float(world.get_snapshot().timestamp.elapsed_seconds)
        green = carla.Color(0, 255, 0)
        for x_m, y_m in trajectory_xy:
            debug.draw_point(carla.Location(x=x_m, y=y_m, z=drawing_z + 0.20), size=0.08, color=green, life_time=self.selected_planner_trajectory_life_time_s, persistent_lines=False)
        source_changed = source != self._last_visualized_planner_source
        route_refresh_due = simulation_time_s - self._last_planner_route_draw_time_s >= self.selected_planner_route_refresh_s
        if route_xy and (source_changed or route_refresh_due):
            yellow = carla.Color(255, 210, 20)
            stride = max(1, int(math.ceil(len(route_xy) / 150.0)))
            displayed_route = route_xy[::stride]
            if displayed_route and displayed_route[-1] != route_xy[-1]:
                displayed_route.append(route_xy[-1])
            for x_m, y_m in displayed_route:
                debug.draw_point(carla.Location(x=x_m, y=y_m, z=drawing_z), size=0.07, color=yellow, life_time=self.selected_planner_route_life_time_s, persistent_lines=False)
            self._last_planner_route_draw_time_s = simulation_time_s
        self._last_visualized_planner_source = source

    def _control_from_ros_planner(self, ros_control):
        """Convert the ROS planner's numeric output at the CARLA boundary only."""
        target_speed_mps = max(0.0, float(ros_control.get("target_speed_mps", 0.0) or 0.0))
        acceleration_mps2 = float(ros_control["acceleration_mps2"])
        steering_rad = float(ros_control["steering_rad"])
        ego_speed_mps = float(dict(self._latest_ros_input_snapshot or {}).get("ego_spd", 0.0)) / 3.6
        stop_goal_active = bool(target_speed_mps <= 1.0e-6)
        pedals = self.ros_actuator_mapper.map_acceleration(acceleration_mps2=acceleration_mps2, max_acceleration_mps2=float(self._ros_max_acceleration_mps2), min_acceleration_mps2=float(self._ros_min_acceleration_mps2), ego_speed_mps=ego_speed_mps, target_speed_mps=target_speed_mps, stop_goal_active=stop_goal_active)
        steer = min(1.0, max(-1.0, steering_rad / float(self._ros_max_steer_rad)))
        return carla.VehicleControl(throttle=float(pedals.throttle), brake=float(pedals.brake), steer=float(steer))

    def _write_selected_planner_output(self, source, cycle_time_s, acceleration_mps2, steering_rad, planning_cycle_time_ms, input_to_output_time_ms):
        """Store the current mode's control values for a later offline comparison."""
        output_path = self.ros_planner_control_output_path if str(source) == "ros" else self.opencda_planner_control_output_path
        velocity = self.vehicle.get_velocity()
        velocity_mps = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2)
        record = {"cycle_time_s": float(cycle_time_s), "source": str(source), "acceleration_mps2": float(acceleration_mps2), "steering_rad": float(steering_rad), "velocity_mps": float(velocity_mps), "planning_cycle_time_ms": float(planning_cycle_time_ms), "input_to_output_time_ms": float(input_to_output_time_ms)}
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def run_step(self, target_speed=None):
        """
        Execute one step of navigation.
        """
        # visualize the bev map if needed
        #print('=============================================running steps==========================================================')
        self.map_manager.run_step()
        if bool(self.ros_planner_enabled):
            if not self._send_current_cycle_to_ros():
                print("[OpenCDA VehicleManager] ROS input boundary is unavailable; applying emergency brake.")
                return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
            ros_control = self.ros_receiver.get_planner_control(self._shadow_cycle_time_s, timeout=self.shadow_comparison_timeout_s)
            if ros_control is None:
                print("[OpenCDA VehicleManager] No same-cycle ROS planner control received; applying emergency brake.")
                return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
            output_ready_monotonic = time.perf_counter()
            input_ready_monotonic = self._planner_input_ready_monotonic if self._planner_input_ready_monotonic is not None else output_ready_monotonic
            local_wait_input_to_output_ms = (output_ready_monotonic - float(input_ready_monotonic)) * 1000.0
            input_to_output_time_ms = float(ros_control.get("input_to_output_time_ms", 0.0) or 0.0)
            if input_to_output_time_ms <= 0.0:
                input_to_output_time_ms = local_wait_input_to_output_ms
            planning_cycle_time_ms = float(ros_control.get("planning_cycle_time_ms", 0.0) or 0.0)
            self._write_selected_planner_output("ros", self._shadow_cycle_time_s, ros_control["acceleration_mps2"], ros_control["steering_rad"], planning_cycle_time_ms, input_to_output_time_ms)
            self._record_timing_summary("ros", planning_cycle_time_ms, input_to_output_time_ms)
            return self._control_from_ros_planner(ros_control)
        if self.cpx_planner is not None:
            planning_started_monotonic = time.perf_counter()
            try:
                local_control = self.cpx_planner.run_step()
            except Exception as exc:
                print(
                    "[OpenCDA VehicleManager] CP-X MPC planner bridge failed; "
                    f"fallback_policy={getattr(self.cpx_planner, 'fallback_policy', '')}: {exc}"
                )
                if getattr(self.cpx_planner, 'fallback_policy', '') == 'raise':
                    raise
                return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)

            output_ready_monotonic = time.perf_counter()
            planning_cycle_time_ms = (output_ready_monotonic - planning_started_monotonic) * 1000.0
            input_ready_monotonic = self._planner_input_ready_monotonic if self._planner_input_ready_monotonic is not None else planning_started_monotonic
            input_to_output_time_ms = (output_ready_monotonic - float(input_ready_monotonic)) * 1000.0
            local_output = getattr(self.cpx_planner, "last_output", None)
            if local_output is not None:
                cycle_time_s = float(self.vehicle.get_world().get_snapshot().timestamp.elapsed_seconds)
                self._write_selected_planner_output("opencda", cycle_time_s, local_output.acceleration_mps2, local_output.steering_rad, planning_cycle_time_ms, input_to_output_time_ms)
                self._record_timing_summary("opencda", planning_cycle_time_ms, input_to_output_time_ms)
            return local_control
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
        if getattr(self, 'ros_receiver', None) is not None:
            self.ros_receiver.close()
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
