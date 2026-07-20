"""OpenCDA cooperative-perception provider for the planning module.

The provider can publish CP messages from two sources:

* native OpenCDA ``VehicleManager`` output:
  ``perception_manager.objects`` and ``v2x_manager.cav_nearby``.
* fallback CARLA actors, used only by standalone planning-module scenarios that
  do not run the native OpenCDA stack.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from utility.cp_messages import replace_cp_list
from utility.global_planner import canonical_lane_id_for_waypoint


class OpenCDACPProvider:
    """Publish CARLA actor observations as OpenCDA-style CP obstacle messages."""

    def __init__(
        self,
        *,
        message_path: str,
        schema_version: int = 1,
        communication_range_m: float = 80.0,
        prediction_horizon_s: float = 3.0,
        prediction_dt_s: float = 0.2,
        source: str = "opencda_cp_provider",
        require_native_opencda: bool = False,
    ) -> None:
        self.message_path = str(message_path)
        self.schema_version = int(schema_version)
        self.communication_range_m = max(0.0, float(communication_range_m))
        self.prediction_horizon_s = max(0.0, float(prediction_horizon_s))
        self.prediction_dt_s = max(1.0e-3, float(prediction_dt_s))
        self.source = str(source)
        self.require_native_opencda = bool(require_native_opencda)
        self.last_publish_summary: dict[str, object] = {}

    def publish(
        self,
        *,
        world: Any,
        map_planner: Any,
        ego_vehicle: Any,
        sim_time_s: float,
        vehicle_manager: Any = None,
    ) -> list[dict]:
        native_available = vehicle_manager is not None
        if native_available:
            messages = self._native_opencda_messages(
                vehicle_manager=vehicle_manager,
                map_planner=map_planner,
                sim_time_s=float(sim_time_s),
            )
            controls = self._native_opencda_traffic_controls(
                vehicle_manager=vehicle_manager,
                map_planner=map_planner,
                ego_vehicle=ego_vehicle,
                sim_time_s=float(sim_time_s),
            )
            provider_source = "native_opencda"
        else:
            if bool(self.require_native_opencda):
                messages = []
            else:
                messages = self._fallback_carla_actor_messages(
                    world=world,
                    map_planner=map_planner,
                    ego_vehicle=ego_vehicle,
                    sim_time_s=float(sim_time_s),
                )
            controls = []
            provider_source = (
                "native_opencda_missing"
                if bool(self.require_native_opencda)
                else "carla_actor_fallback"
            )

        replace_cp_list(
            message_path=self.message_path,
            schema_version=self.schema_version,
            list_name="obstacles",
            items=messages,
            timestamp_s=float(sim_time_s),
        )
        replace_cp_list(
            message_path=self.message_path,
            schema_version=self.schema_version,
            list_name="control",
            items=controls,
            timestamp_s=float(sim_time_s),
        )
        self.last_publish_summary = {
            "source": self.source,
            "provider_source": provider_source,
            "native_opencda_required": bool(self.require_native_opencda),
            "native_opencda_available": bool(native_available),
            "obstacle_count": int(len(messages)),
            "control_count": int(len(controls)),
            "communication_range_m": float(self.communication_range_m),
            "timestamp_s": float(sim_time_s),
        }
        return messages

    def _native_opencda_traffic_controls(
        self,
        *,
        vehicle_manager: Any,
        map_planner: Any,
        ego_vehicle: Any,
        sim_time_s: float,
    ) -> list[dict]:
        perception_manager = getattr(vehicle_manager, "perception_manager", None)
        perception_objects = getattr(perception_manager, "objects", {}) or {}
        traffic_lights = list(perception_objects.get("traffic_lights", []) or [])
        if not traffic_lights:
            return []

        ego_transform = self._ego_transform(vehicle_manager=vehicle_manager, ego_vehicle=ego_vehicle)
        if ego_transform is None:
            return []
        ego_location = ego_transform.location
        ego_heading_rad = math.radians(float(ego_transform.rotation.yaw))

        controls: list[dict] = []
        for index, traffic_light in enumerate(traffic_lights):
            control = self._traffic_light_to_control_message(
                traffic_light=traffic_light,
                map_planner=map_planner,
                ego_location=ego_location,
                ego_heading_rad=float(ego_heading_rad),
                sim_time_s=float(sim_time_s),
                fallback_id=f"tl:{index}",
            )
            if isinstance(control, Mapping):
                controls.append(dict(control))
        return controls

    def _traffic_light_to_control_message(
        self,
        *,
        traffic_light: Any,
        map_planner: Any,
        ego_location: Any,
        ego_heading_rad: float,
        sim_time_s: float,
        fallback_id: str,
    ) -> Optional[dict]:
        location = self._traffic_light_location(traffic_light)
        if location is None:
            return None
        try:
            dx_m = float(location.x) - float(ego_location.x)
            dy_m = float(location.y) - float(ego_location.y)
            distance_m = math.hypot(dx_m, dy_m)
        except Exception:
            return None
        if self.communication_range_m > 0.0 and float(distance_m) > self.communication_range_m:
            return None

        forward_m = math.cos(float(ego_heading_rad)) * dx_m + math.sin(float(ego_heading_rad)) * dy_m
        lateral_m = -math.sin(float(ego_heading_rad)) * dx_m + math.cos(float(ego_heading_rad)) * dy_m
        state = self._normalize_signal_state(self._traffic_light_state(traffic_light))
        actor = getattr(traffic_light, "actor", None)
        control_id = str(getattr(actor, "id", fallback_id))
        waypoint = None
        waypoint = self._map_waypoint_from_location(
            map_planner=map_planner,
            location=location,
        )

        lane_id = 0
        road_id = 0
        section_id = 0
        heading_rad = float(ego_heading_rad)
        try:
            lane_id = int(canonical_lane_id_for_waypoint(waypoint))
            road_id = int(getattr(waypoint, "road_id", 0) or 0)
            section_id = int(getattr(waypoint, "section_id", 0) or 0)
        except Exception:
            pass

        stop_line = {
            "x_m": float(location.x),
            "y_m": float(location.y),
            "heading_rad": float(heading_rad),
            "lane_id": int(lane_id),
            "road_id": int(road_id),
            "section_id": int(section_id),
        }
        ttl_s = max(0.2, 2.0 * float(self.prediction_dt_s))
        confidence = 1.0 if state in {"red", "yellow", "green"} else 0.3
        return {
            "type": "traffic_light",
            "id": f"native_opencda_tl:{control_id}",
            "state": str(state),
            "signal_state": str(state),
            "control_id": str(control_id),
            "timestamp_s": float(sim_time_s),
            "ttl_s": float(ttl_s),
            "valid_until_s": float(sim_time_s) + float(ttl_s),
            "confidence": float(confidence),
            "distance_m": float(distance_m),
            "stop_line": dict(stop_line),
            "stop_line_position": dict(stop_line),
            "valid_range": {
                "search_distance_m": float(self.communication_range_m),
                "forward_m": float(forward_m),
                "lateral_m": float(lateral_m),
                "signal_forward_m": float(forward_m),
                "signal_lateral_m": float(lateral_m),
                "road_id": int(road_id),
                "section_id": int(section_id),
                "lane_id": int(lane_id),
            },
            "ego_passed_stop_line": bool(float(forward_m) < -1.0),
            "source": "native_opencda",
            "provider_source": "native_opencda_traffic_light",
            "signal_actor_id": str(control_id),
            "signal_actor_name": str(getattr(actor, "type_id", "")),
            "signal_actor_raw_state": str(self._traffic_light_state(traffic_light)),
            "signal_distance_m": float(distance_m),
            "signal_forward_m": float(forward_m),
            "signal_lateral_m": float(lateral_m),
        }

    def _native_opencda_messages(
        self,
        *,
        vehicle_manager: Any,
        map_planner: Any,
        sim_time_s: float,
    ) -> list[dict]:
        ego_vehicle = getattr(vehicle_manager, "vehicle", None)
        if ego_vehicle is None:
            return []
        try:
            ego_location = ego_vehicle.get_location()
        except RuntimeError:
            return []

        messages: list[dict] = []
        seen_actor_ids: set[str] = set()

        perception_manager = getattr(vehicle_manager, "perception_manager", None)
        perception_objects = getattr(perception_manager, "objects", {}) or {}
        for index, obj in enumerate(list(perception_objects.get("vehicles", []) or [])):
            actor_id = self._object_actor_id(obj, fallback=f"perception:{index}")
            if actor_id in seen_actor_ids:
                continue
            message = self._object_to_cp_message(
                obj=obj,
                map_planner=map_planner,
                ego_location=ego_location,
                sim_time_s=float(sim_time_s),
                source="opencda_perception",
                provider_source="native_opencda_perception",
                fallback_id=actor_id,
            )
            if isinstance(message, Mapping):
                seen_actor_ids.add(actor_id)
                messages.append(dict(message))

        v2x_manager = getattr(vehicle_manager, "v2x_manager", None)
        cav_nearby = getattr(v2x_manager, "cav_nearby", {}) or {}
        for vid, nearby_vm in dict(cav_nearby).items():
            actor = getattr(nearby_vm, "vehicle", None)
            actor_id = self._object_actor_id(actor, fallback=f"v2x:{vid}")
            if actor_id in seen_actor_ids:
                continue
            v2x_obj = actor if actor is not None else nearby_vm
            message = self._object_to_cp_message(
                obj=v2x_obj,
                map_planner=map_planner,
                ego_location=ego_location,
                sim_time_s=float(sim_time_s),
                source="opencda_v2x",
                provider_source="native_opencda_v2x",
                fallback_id=actor_id,
                speed_kmh=self._nearby_vm_speed_kmh(nearby_vm),
                transform_override=self._nearby_vm_transform(nearby_vm),
            )
            if isinstance(message, Mapping):
                seen_actor_ids.add(actor_id)
                messages.append(dict(message))

        return messages

    def _fallback_carla_actor_messages(
        self,
        *,
        world: Any,
        map_planner: Any,
        ego_vehicle: Any,
        sim_time_s: float,
    ) -> list[dict]:
        ego_location = ego_vehicle.get_location()
        messages: list[dict] = []
        for actor in world.get_actors().filter("*vehicle*"):
            if int(getattr(actor, "id", -1)) == int(getattr(ego_vehicle, "id", -2)):
                continue
            message = self._object_to_cp_message(
                obj=actor,
                map_planner=map_planner,
                ego_location=ego_location,
                sim_time_s=float(sim_time_s),
                source=self.source,
                provider_source="carla_actor_fallback",
                fallback_id=str(getattr(actor, "id", "")),
            )
            if isinstance(message, Mapping):
                messages.append(dict(message))
        return messages

    def _object_to_cp_message(
        self,
        *,
        obj: Any,
        map_planner: Any,
        ego_location: Any,
        sim_time_s: float,
        source: str,
        provider_source: str,
        fallback_id: str,
        speed_kmh: float | None = None,
        transform_override: Any = None,
    ) -> Optional[dict]:
        try:
            transform = transform_override or self._object_transform(obj)
            if transform is None:
                return None
            location = transform.location
            rotation = transform.rotation
            velocity = self._object_velocity(obj)
            bbox = getattr(obj, "bounding_box", None)
            extent = getattr(bbox, "extent", None)
        except RuntimeError:
            return None
        except AttributeError:
            return None

        try:
            distance_m = float(location.distance(ego_location))
        except Exception:
            distance_m = 0.0
        if self.communication_range_m > 0.0 and distance_m > self.communication_range_m:
            return None

        if speed_kmh is None:
            if velocity is None:
                speed_mps = 0.0
            else:
                speed_mps = math.sqrt(
                    float(getattr(velocity, "x", 0.0)) ** 2
                    + float(getattr(velocity, "y", 0.0)) ** 2
                    + float(getattr(velocity, "z", 0.0)) ** 2
                )
        else:
            speed_mps = max(0.0, float(speed_kmh) / 3.6)
        heading_rad = math.radians(float(rotation.yaw))
        lane_id = 0
        road_id = -1
        try:
            waypoint = self._map_waypoint_from_location(
                map_planner=map_planner,
                location=location,
            )
            lane_id = int(canonical_lane_id_for_waypoint(waypoint))
            road_id = int(getattr(waypoint, "road_id", -1) or -1)
        except Exception:
            pass

        trajectory = self._constant_velocity_trajectory(
            x_m=float(location.x),
            y_m=float(location.y),
            speed_mps=float(speed_mps),
            heading_rad=float(heading_rad),
        )
        actor_id = self._object_actor_id(obj, fallback=fallback_id)
        return {
            "id": f"{provider_source}:{actor_id}",
            "type": "vehicle",
            "source": str(source),
            "provider_source": str(provider_source),
            "timestamp_s": float(sim_time_s),
            "ttl_s": max(0.2, 2.0 * float(self.prediction_dt_s)),
            "confidence": 1.0,
            "distance_m": float(distance_m),
            "state": [
                float(location.x),
                float(location.y),
                float(speed_mps),
                float(heading_rad),
            ],
            "z": float(location.z),
            "shape": {
                "length_m": 2.0 * float(getattr(extent, "x", 2.25)),
                "width_m": 2.0 * float(getattr(extent, "y", 1.0)),
                "height_m": 2.0 * float(getattr(extent, "z", 0.9)),
            },
            "road_id": int(road_id),
            "lane_id": int(lane_id),
            "trajectory": trajectory,
        }

    @staticmethod
    def _object_actor_id(obj: Any, fallback: str) -> str:
        for attr_name in ("id", "carla_id", "vid"):
            value = getattr(obj, attr_name, None)
            if value is not None:
                return str(value)
        return str(fallback)

    @staticmethod
    def _map_waypoint_from_location(*, map_planner: Any, location: Any):
        if map_planner is None or location is None:
            return None
        get_waypoint = getattr(map_planner, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        point = {
            "x": float(getattr(location, "x", 0.0)),
            "y": float(getattr(location, "y", 0.0)),
            "z": float(getattr(location, "z", 0.0)),
        }
        try:
            return get_waypoint(point)
        except Exception:
            pass
        try:
            return get_waypoint(location)
        except Exception:
            return None

    @staticmethod
    def _object_transform(obj: Any):
        get_transform = getattr(obj, "get_transform", None)
        if callable(get_transform):
            return get_transform()
        return getattr(obj, "transform", None)

    @staticmethod
    def _object_velocity(obj: Any):
        get_velocity = getattr(obj, "get_velocity", None)
        if callable(get_velocity):
            return get_velocity()
        return getattr(obj, "velocity", None)

    @staticmethod
    def _nearby_vm_transform(vehicle_manager: Any):
        v2x_manager = getattr(vehicle_manager, "v2x_manager", None)
        get_ego_pos = getattr(v2x_manager, "get_ego_pos", None)
        if callable(get_ego_pos):
            return get_ego_pos()
        localizer = getattr(vehicle_manager, "localizer", None)
        get_ego_pos = getattr(localizer, "get_ego_pos", None)
        return get_ego_pos() if callable(get_ego_pos) else None

    @staticmethod
    def _nearby_vm_speed_kmh(vehicle_manager: Any) -> float | None:
        v2x_manager = getattr(vehicle_manager, "v2x_manager", None)
        get_ego_speed = getattr(v2x_manager, "get_ego_speed", None)
        if callable(get_ego_speed):
            speed = get_ego_speed()
            return None if speed is None else float(speed)
        localizer = getattr(vehicle_manager, "localizer", None)
        get_ego_spd = getattr(localizer, "get_ego_spd", None)
        if callable(get_ego_spd):
            speed = get_ego_spd()
            return None if speed is None else float(speed)
        return None

    @staticmethod
    def _ego_transform(*, vehicle_manager: Any, ego_vehicle: Any):
        localizer = getattr(vehicle_manager, "localizer", None)
        get_ego_pos = getattr(localizer, "get_ego_pos", None)
        if callable(get_ego_pos):
            transform = get_ego_pos()
            if transform is not None:
                return transform
        get_transform = getattr(ego_vehicle, "get_transform", None)
        return get_transform() if callable(get_transform) else None

    @staticmethod
    def _traffic_light_location(traffic_light: Any):
        get_location = getattr(traffic_light, "get_location", None)
        if callable(get_location):
            return get_location()
        return getattr(traffic_light, "location", None)

    @staticmethod
    def _traffic_light_state(traffic_light: Any):
        get_state = getattr(traffic_light, "get_state", None)
        if callable(get_state):
            return get_state()
        return getattr(traffic_light, "state", None)

    @staticmethod
    def _normalize_signal_state(signal_state: object) -> str:
        raw_name = (str(signal_state) if signal_state is not None else "").strip().upper()
        if "." in raw_name:
            raw_name = raw_name.rsplit(".", 1)[-1]
        if raw_name in {"2", "GREEN", "GO"}:
            return "green"
        if raw_name in {"1", "YELLOW", "AMBER"}:
            return "yellow"
        if raw_name in {"0", "RED", "STOP"}:
            return "red"
        return "unknown"

    def _constant_velocity_trajectory(
        self,
        *,
        x_m: float,
        y_m: float,
        speed_mps: float,
        heading_rad: float,
    ) -> list[list[float]]:
        steps = max(1, int(round(self.prediction_horizon_s / self.prediction_dt_s)))
        cos_h = math.cos(float(heading_rad))
        sin_h = math.sin(float(heading_rad))
        trajectory: list[list[float]] = []
        for idx in range(steps + 1):
            t_s = float(idx) * float(self.prediction_dt_s)
            trajectory.append([
                float(x_m + speed_mps * cos_h * t_s),
                float(y_m + speed_mps * sin_h * t_s),
                float(speed_mps),
                float(heading_rad),
            ])
        return trajectory
