"""OpenCDA cooperative-perception provider for the planning module.

Obstacle messages come from a **multi-vantage-point CARLA actor scan**, not
from ``vehicle_manager.perception_manager.objects``: for the publishing CAV
and every nearby CAV in ``v2x_manager.cav_nearby`` (population itself is
distance-gated by ``v2x.communication_range``, unrelated to whether that CAV
runs the CP-X planner), this scans every ``*vehicle*``/``*walker*`` CARLA
actor within ``communication_range_m`` of *that* vantage point and fuses
(deduplicates) the union into one obstacle list. This sidesteps two
limitations of OpenCDA's own perception path: its ground-truth "deactivated"
mode hard-codes a 50 m detection radius (``perception_manager.py::deactivate_mode``'s
``thresh = 50``), and it never tracks pedestrians at all (only
``{"vehicles", "traffic_lights"}``). Nearby CAVs only need to exist and be
tracked -- they do not need their own CP-X planner/provider running, so
plain OpenCDA-default-``BehaviorAgent`` traffic works as an observation
point too.

Traffic-light control messages are unaffected: they still come from
``perception_manager.objects["traffic_lights"]`` via the native OpenCDA
path.

Fallback CARLA actor messages (vehicles only, single vantage point) remain
available for standalone planning-module scenarios that do not run the
native OpenCDA stack at all (no ``vehicle_manager``).
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
        visibility_filter_enabled: bool = False,
        visibility_backend: str = "actor_geometry",
        visibility_sensor_height_m: float = 1.6,
        visibility_target_tolerance_m: float = 0.75,
    ) -> None:
        self.message_path = str(message_path)
        self.schema_version = int(schema_version)
        self.communication_range_m = max(0.0, float(communication_range_m))
        self.prediction_horizon_s = max(0.0, float(prediction_horizon_s))
        self.prediction_dt_s = max(1.0e-3, float(prediction_dt_s))
        self.source = str(source)
        self.require_native_opencda = bool(require_native_opencda)
        self.visibility_filter_enabled = bool(visibility_filter_enabled)
        self.visibility_backend = str(visibility_backend).strip().lower()
        self.visibility_sensor_height_m = max(
            0.1, float(visibility_sensor_height_m)
        )
        self.visibility_target_tolerance_m = max(
            0.1, float(visibility_target_tolerance_m)
        )
        self.last_publish_summary: dict[str, object] = {}
        self._last_observer_cav_ids: list[str] = []

    def publish(
        self,
        *,
        world: Any,
        map_planner: Any,
        ego_vehicle: Any,
        sim_time_s: float,
        vehicle_manager: Any = None,
    ) -> list[dict]:
        self._last_observer_cav_ids = []
        native_available = vehicle_manager is not None
        if native_available:
            messages = self._native_opencda_messages(
                world=world,
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
            "observer_cav_ids": list(self._last_observer_cav_ids),
            "multi_observer_obstacle_count": int(sum(
                len(set(str(item) for item in list(
                    message.get("observed_by_cav_ids", []) or []
                ))) >= 2
                for message in messages
            )),
            "blind_spot_shared_count": int(sum(
                bool(message.get("blind_spot_shared", False))
                for message in messages
            )),
            "blind_spot_shared_actor_ids": sorted(
                str(message.get("id", ""))
                for message in messages
                if bool(message.get("blind_spot_shared", False))
            ),
            "actor_provenance": [
                {
                    "actor_id": str(message.get("id", "")),
                    "actor_type": str(message.get("type", "unknown")),
                    "observer_cav_ids": list(
                        message.get("observed_by_cav_ids", []) or []
                    ),
                    "not_observed_by_cav_ids": list(
                        message.get("not_observed_by_cav_ids", []) or []
                    ),
                    "visible_to_ego": bool(
                        str(getattr(ego_vehicle, "id", ""))
                        in list(message.get("observed_by_cav_ids", []) or [])
                    ),
                    "visible_to_auxiliary": bool(
                        any(
                            str(observer_id)
                            != str(getattr(ego_vehicle, "id", ""))
                            for observer_id in list(
                                message.get("observed_by_cav_ids", []) or []
                            )
                        )
                    ),
                    "blind_spot_shared": bool(
                        message.get("blind_spot_shared", False)
                    ),
                    "distance_to_ego_m": float(
                        message.get("distance_m", 0.0) or 0.0
                    ),
                }
                for message in messages
            ],
            "visibility_filter_enabled": bool(self.visibility_filter_enabled),
            "visibility_backend": str(self.visibility_backend),
        }
        self.last_publish_summary["observer_cav_count"] = int(
            len(self.last_publish_summary["observer_cav_ids"])
        )
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
        world: Any,
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
        ego_id = int(getattr(ego_vehicle, "id", -1))

        # Every vantage point this CAV can borrow: itself, plus every nearby
        # CAV's tracked position (`v2x_manager.cav_nearby`'s own population
        # is already distance-gated by `v2x.communication_range` -- a
        # separate, independent check from `communication_range_m` below).
        # A nearby CAV only needs to be a tracked VehicleManager; it does not
        # need its own CP-X planner running.
        vantage_points: list[tuple[str, Any]] = [(str(ego_id), ego_location)]
        v2x_manager = getattr(vehicle_manager, "v2x_manager", None)
        cav_nearby = getattr(v2x_manager, "cav_nearby", {}) or {}
        for nearby_vm in dict(cav_nearby).values():
            nearby_vehicle = getattr(nearby_vm, "vehicle", None)
            get_location = getattr(nearby_vehicle, "get_location", None)
            if not callable(get_location):
                continue
            try:
                vantage_points.append((
                    str(getattr(nearby_vehicle, "id", "")),
                    get_location(),
                ))
            except RuntimeError:
                continue
        self._last_observer_cav_ids = sorted(set(
            str(observer_id)
            for observer_id, _ in vantage_points
            if str(observer_id)
        ))

        messages: list[dict] = []
        seen_actor_ids: set[str] = set()
        candidate_actors = list(world.get_actors().filter("*vehicle*")) + list(
            world.get_actors().filter("*walker*")
        )
        for actor in candidate_actors:
            actor_id_int = int(getattr(actor, "id", -1))
            if actor_id_int == ego_id:
                continue
            try:
                actor_location = actor.get_location()
            except RuntimeError:
                continue
            # In range of *some* vantage point (ego or a relaying nearby
            # CAV) -- this is the actual cooperative-perception range check.
            # `_object_to_cp_message`'s own internal range check is
            # ego-centric and would incorrectly reject things a nearby CAV
            # sees but that sit beyond ego's own communication_range_m, so
            # it is skipped below via `skip_range_filter=True`.
            in_range_observer_ids: list[str] = []
            observed_by_cav_ids: list[str] = []
            visibility_by_cav_id: dict[str, str] = {}
            for observer_id, vantage in vantage_points:
                if actor_location.distance(vantage) > self.communication_range_m:
                    continue
                observer_id = str(observer_id)
                in_range_observer_ids.append(observer_id)
                visible, reason = self._line_of_sight_visible(
                    world=world,
                    observer_location=vantage,
                    target_actor=actor,
                    target_location=actor_location,
                    occluder_actors=candidate_actors,
                )
                visibility_by_cav_id[observer_id] = str(reason)
                if bool(visible):
                    observed_by_cav_ids.append(observer_id)
            if not observed_by_cav_ids:
                continue

            actor_id = self._object_actor_id(actor, fallback=str(actor_id_int))
            if actor_id in seen_actor_ids:
                continue
            type_id = str(getattr(actor, "type_id", ""))
            object_type = "pedestrian" if type_id.startswith("walker.") else "vehicle"
            message = self._object_to_cp_message(
                obj=actor,
                map_planner=map_planner,
                ego_location=ego_location,
                sim_time_s=float(sim_time_s),
                source="opencda_multi_vantage",
                provider_source="native_opencda_multi_vantage",
                fallback_id=actor_id,
                object_type=object_type,
                skip_range_filter=True,
                observed_by_cav_ids=observed_by_cav_ids,
                not_observed_by_cav_ids=[
                    observer_id
                    for observer_id in in_range_observer_ids
                    if observer_id not in observed_by_cav_ids
                ],
                visibility_by_cav_id=visibility_by_cav_id,
            )
            if isinstance(message, Mapping):
                message = dict(message)
                message["blind_spot_shared"] = bool(
                    str(ego_id) not in observed_by_cav_ids
                    and any(
                        observer_id != str(ego_id)
                        for observer_id in observed_by_cav_ids
                    )
                )
                seen_actor_ids.add(actor_id)
                messages.append(message)

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
        object_type: str = "vehicle",
        skip_range_filter: bool = False,
        observed_by_cav_ids: list[str] | None = None,
        not_observed_by_cav_ids: list[str] | None = None,
        visibility_by_cav_id: Mapping[str, str] | None = None,
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
        # Callers that already ran their own (possibly multi-vantage-point)
        # range check pass `skip_range_filter=True` -- this ego-centric
        # check is only correct for a single-vantage-point caller.
        if (
            not bool(skip_range_filter)
            and self.communication_range_m > 0.0
            and distance_m > self.communication_range_m
        ):
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
            "type": str(object_type),
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
            "observed_by_cav_ids": sorted(set(
                str(item) for item in list(observed_by_cav_ids or [])
                if str(item)
            )),
            "not_observed_by_cav_ids": sorted(set(
                str(item) for item in list(not_observed_by_cav_ids or [])
                if str(item)
            )),
            "visibility_by_cav_id": dict(visibility_by_cav_id or {}),
        }

    def _line_of_sight_visible(
        self,
        *,
        world: Any,
        observer_location: Any,
        target_actor: Any,
        target_location: Any,
        occluder_actors: list[Any] | None = None,
    ) -> tuple[bool, str]:
        """Approximate sensor visibility without blocking the simulator tick."""

        if not bool(self.visibility_filter_enabled):
            return True, "visibility_filter_disabled"
        if str(getattr(self, "visibility_backend", "ray_cast")) == "actor_geometry":
            return self._actor_geometry_visible(
                observer_location=observer_location,
                target_actor=target_actor,
                target_location=target_location,
                occluder_actors=list(occluder_actors or []),
            )
        cast_ray = getattr(world, "cast_ray", None)
        if not callable(cast_ray):
            return True, "cast_ray_unavailable_fallback_visible"

        start = self._location_with_height(
            observer_location,
            float(getattr(observer_location, "z", 0.0))
            + self.visibility_sensor_height_m,
        )
        bbox = getattr(target_actor, "bounding_box", None)
        extent = getattr(bbox, "extent", None)
        target_height = max(0.5, float(getattr(extent, "z", 0.8)))
        end = self._location_with_height(
            target_location,
            float(getattr(target_location, "z", 0.0)) + target_height,
        )
        ray_distance_m = self._distance_3d(start, end)
        target_radius_m = math.sqrt(
            float(getattr(extent, "x", 0.4)) ** 2
            + float(getattr(extent, "y", 0.4)) ** 2
            + float(getattr(extent, "z", 0.8)) ** 2
        )
        clear_before_m = max(
            0.0,
            ray_distance_m
            - target_radius_m
            - self.visibility_target_tolerance_m,
        )
        try:
            hits = list(cast_ray(start, end) or [])
        except Exception as exc:
            return True, "cast_ray_error_fallback_visible:%s" % type(exc).__name__
        nearest_hit_m = min(
            (
                self._distance_3d(start, getattr(hit, "location", None))
                for hit in hits
                if getattr(hit, "location", None) is not None
            ),
            default=float("inf"),
        )
        if nearest_hit_m < clear_before_m:
            return False, "occluded:hit_distance=%.2f" % float(nearest_hit_m)
        return True, "line_of_sight_clear"

    def _actor_geometry_visible(
        self,
        *,
        observer_location: Any,
        target_actor: Any,
        target_location: Any,
        occluder_actors: list[Any],
    ) -> tuple[bool, str]:
        """Use actor footprints as a non-RPC dynamic-occlusion approximation."""

        start_x = float(observer_location.x)
        start_y = float(observer_location.y)
        end_x = float(target_location.x)
        end_y = float(target_location.y)
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        segment_length_sq = segment_x ** 2 + segment_y ** 2
        if segment_length_sq <= 1.0e-6:
            return True, "actor_geometry_same_position"
        target_id = int(getattr(target_actor, "id", -1))
        for actor in list(occluder_actors or []):
            if int(getattr(actor, "id", -2)) == target_id:
                continue
            if not str(getattr(actor, "type_id", "")).startswith("vehicle."):
                continue
            try:
                location = actor.get_location()
            except (AttributeError, RuntimeError):
                continue
            rel_x = float(location.x) - start_x
            rel_y = float(location.y) - start_y
            progress = (
                rel_x * segment_x + rel_y * segment_y
            ) / segment_length_sq
            # Ignore the observer's own body and actors at/behind the target.
            if progress <= 0.05 or progress >= 0.95:
                continue
            closest_x = start_x + progress * segment_x
            closest_y = start_y + progress * segment_y
            lateral_distance_m = math.hypot(
                float(location.x) - closest_x,
                float(location.y) - closest_y,
            )
            extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
            blocking_radius_m = max(
                0.8,
                math.hypot(
                    float(getattr(extent, "x", 1.8)),
                    float(getattr(extent, "y", 0.9)),
                ),
            )
            if lateral_distance_m <= blocking_radius_m:
                return (
                    False,
                    "occluded_by_actor:%s" % str(getattr(actor, "id", "")),
                )
        return True, "actor_geometry_clear"

    @staticmethod
    def _location_with_height(location: Any, z_m: float):
        try:
            return type(location)(
                x=float(location.x),
                y=float(location.y),
                z=float(z_m),
            )
        except Exception:
            return location

    @staticmethod
    def _distance_3d(first: Any, second: Any) -> float:
        if first is None or second is None:
            return float("inf")
        return math.sqrt(
            (float(first.x) - float(second.x)) ** 2
            + (float(first.y) - float(second.y)) ** 2
            + (float(first.z) - float(second.z)) ** 2
        )

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
