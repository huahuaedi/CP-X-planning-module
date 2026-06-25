"""
Scenario-local logic for all_usecase_scenario.
"""

from __future__ import annotations

import math
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from utility import canonical_lane_id_for_waypoint, raw_carla_lane_id_for_waypoint
from utility.cp_messages import (
    empty_cp_payload,
    load_cp_message_payload,
    remove_cp_item,
    reset_cp_message_payload,
    upsert_cp_item,
    write_cp_message_payload,
)
from opencda_scenario import world_messages


SCENARIO_DIR = os.path.dirname(os.path.abspath(__file__))
PLANNING_MODULE_DIR = os.path.dirname(os.path.dirname(SCENARIO_DIR))
DEFAULT_CP_MESSAGE_PATH = os.path.join(
    PLANNING_MODULE_DIR,
    "behavior_planner",
    "cp_message.json",
)


def _info(message: str) -> None:
    print(f"[ALL USECASE] {message}")


def _warning(message: str) -> None:
    print(f"[ALL USECASE] Warning: {message}")


def _iter_world_actors(world) -> Iterable[Any]:
    if not hasattr(world, "get_actors"):
        return []
    return list(world.get_actors())


def _iter_world_traffic_lights(world) -> Iterable[Any]:
    get_actors_fn = getattr(world, "get_actors", None)
    if not callable(get_actors_fn):
        return []
    try:
        all_actors = get_actors_fn()
    except Exception:
        return []

    filter_fn = getattr(all_actors, "filter", None)
    if callable(filter_fn):
        try:
            return [
                actor
                for actor in list(filter_fn("traffic.traffic_light*"))
                if callable(getattr(actor, "get_state", None))
            ]
        except Exception:
            pass

    return [
        actor
        for actor in list(all_actors)
        if "traffic_light" in str(getattr(actor, "type_id", "")).strip().lower()
        and callable(getattr(actor, "get_state", None))
    ]


def _iter_environment_objects(world, carla) -> Iterable[Any]:
    if not hasattr(world, "get_environment_objects"):
        return []
    try:
        return list(world.get_environment_objects(carla.CityObjectLabel.Any))
    except Exception:
        return []


def _best_match(candidates: Sequence[Tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: int(item[0]))[0][1]


def _find_marker(world, carla, marker_name: str):
    requested = str(marker_name).strip().lower()
    if not requested:
        return None

    starts_with_candidates: List[Tuple[int, Any]] = []
    partial_candidates: List[Tuple[int, Any]] = []
    for env_obj in _iter_environment_objects(world, carla):
        env_name = str(getattr(env_obj, "name", "")).strip().lower()
        if env_name == requested:
            return env_obj
        if env_name.startswith(f"{requested}_"):
            starts_with_candidates.append((len(env_name), env_obj))
        elif requested in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_match(starts_with_candidates) or _best_match(partial_candidates)


def _marker_index(marker_name: str, prefix: str = "") -> int | None:
    text = str(marker_name).strip().lower()
    prefix_text = str(prefix).strip().lower()
    if prefix_text:
        match = re.search(rf"{re.escape(prefix_text)}(\d+)", text)
        if match:
            return int(match.group(1))
    match = re.search(r"(\d+)", text)
    return None if match is None else int(match.group(1))


def _find_markers_by_prefix(world, carla, prefix: str) -> List[Any]:
    prefix_text = str(prefix).strip().lower()
    if not prefix_text:
        return []

    best_matches_by_index: Dict[int, Tuple[Tuple[int, int, str], Any]] = {}
    for env_obj in _iter_environment_objects(world, carla):
        env_name = str(getattr(env_obj, "name", "")).strip().lower()
        if prefix_text not in env_name:
            continue
        marker_index = _marker_index(env_name, prefix_text)
        if marker_index is None:
            continue
        match_priority = (
            0 if env_name.startswith(prefix_text) else 1,
            len(env_name),
            env_name,
        )
        existing_match = best_matches_by_index.get(int(marker_index), None)
        if existing_match is None or match_priority < existing_match[0]:
            best_matches_by_index[int(marker_index)] = (match_priority, env_obj)
    matches = [
        env_obj
        for _, env_obj in sorted(
            best_matches_by_index.values(),
            key=lambda item: item[0],
        )
    ]
    return sorted(
        matches,
        key=lambda obj: (
            _marker_index(str(getattr(obj, "name", "")), prefix_text) or 10**9,
            str(getattr(obj, "name", "")).lower(),
        )
    )


def _find_actor_by_id(world, actor_id: object):
    try:
        normalized_actor_id = int(actor_id)
    except Exception:
        return None
    for actor in _iter_world_actors(world):
        if int(getattr(actor, "id", -1)) == normalized_actor_id:
            return actor
    return None


def _object_transform(obj: Any):
    if obj is None:
        return None
    direct_transform = getattr(obj, "transform", None)
    if direct_transform is not None:
        return direct_transform
    get_transform_fn = getattr(obj, "get_transform", None)
    if callable(get_transform_fn):
        try:
            return get_transform_fn()
        except Exception:
            return None
    return None


def _traffic_light_actor_name(actor: Any) -> str:
    if actor is None:
        return ""
    raw_attributes = getattr(actor, "attributes", None)
    if isinstance(raw_attributes, Mapping):
        for key in ("name", "role_name", "object_name"):
            value = str(raw_attributes.get(key, "")).strip()
            if value:
                return value
    return str(getattr(actor, "type_id", "")).strip()


def _transform_xy(transform: Any) -> List[float] | None:
    location = getattr(transform, "location", None)
    if location is None:
        return None
    return [float(location.x), float(location.y)]


def _distance_xy(first_xy: Sequence[object] | None, second_xy: Sequence[object] | None) -> float | None:
    if not isinstance(first_xy, Sequence) or len(first_xy) < 2:
        return None
    if not isinstance(second_xy, Sequence) or len(second_xy) < 2:
        return None
    return float(
        math.hypot(
            float(first_xy[0]) - float(second_xy[0]),
            float(first_xy[1]) - float(second_xy[1]),
        )
    )


def _normalize_signal_state(signal_state: object) -> str:
    raw_name = (str(signal_state) if signal_state is not None else "").strip().upper()
    if "." in raw_name:
        raw_name = raw_name.rsplit(".", 1)[-1]
    if raw_name in {"GREEN", "GO"}:
        return "green"
    if raw_name in {"YELLOW", "AMBER"}:
        return "yellow"
    if raw_name in {"RED", "STOP"}:
        return "red"
    return "unknown"


def _closest_driving_waypoint(world_map, carla, transform: Any):
    location = getattr(transform, "location", None)
    if location is None or world_map is None or not hasattr(world_map, "get_waypoint"):
        return None
    try:
        return world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        return None


def _record_marker(*, world_map, carla, marker_obj: Any, prefix: str = "") -> Dict[str, object]:
    transform = _object_transform(marker_obj)
    waypoint = _closest_driving_waypoint(world_map, carla, transform)
    waypoint_transform = getattr(waypoint, "transform", None) if waypoint is not None else None
    lane_id = None
    carla_lane_id = None
    if waypoint is not None:
        lane_id = int(canonical_lane_id_for_waypoint(waypoint))
        carla_lane_id = int(raw_carla_lane_id_for_waypoint(waypoint))
    return {
        "name": str(getattr(marker_obj, "name", "")).strip(),
        "index": _marker_index(str(getattr(marker_obj, "name", "")), prefix),
        "transform": transform,
        "position_xy": _transform_xy(transform) or [],
        "waypoint": waypoint,
        "waypoint_transform": waypoint_transform,
        "road_id": None if waypoint is None else int(getattr(waypoint, "road_id", 0)),
        "section_id": None if waypoint is None else int(getattr(waypoint, "section_id", 0)),
        "lane_id": lane_id,
        "carla_lane_id": carla_lane_id,
    }


def _records_by_prefix(*, world, world_map, carla, prefix: str) -> List[Dict[str, object]]:
    return [
        _record_marker(world_map=world_map, carla=carla, marker_obj=obj, prefix=prefix)
        for obj in _find_markers_by_prefix(world, carla, prefix)
    ]


def _records_by_indexed_names(*, world, world_map, carla, prefix: str) -> List[Dict[str, object]]:
    prefix_text = str(prefix).strip()
    if not prefix_text:
        return []

    marker_indices = sorted(
        {
            int(marker_index)
            for env_obj in _iter_environment_objects(world, carla)
            for env_name in [str(getattr(env_obj, "name", "")).strip().lower()]
            if prefix_text.lower() in env_name
            for marker_index in [_marker_index(env_name, prefix_text)]
            if marker_index is not None
        }
    )
    matched_records: List[Dict[str, object]] = []
    for marker_index in marker_indices:
        marker_record = _record_by_name(
            world=world,
            world_map=world_map,
            carla=carla,
            marker_name=f"{prefix_text}{int(marker_index)}",
        )
        if isinstance(marker_record, Mapping):
            matched_records.append(dict(marker_record))
    return matched_records


def _record_by_name(*, world, world_map, carla, marker_name: str) -> Dict[str, object] | None:
    marker_obj = _find_marker(world, carla, marker_name)
    if marker_obj is None:
        return None
    return _record_marker(world_map=world_map, carla=carla, marker_obj=marker_obj)


def _message_id(prefix: str, kind: str, marker: Mapping[str, object]) -> str:
    marker_index = marker.get("index", None)
    if marker_index is not None:
        suffix = str(int(marker_index))
    else:
        suffix = str(marker.get("name", kind)).strip()
    return f"{str(prefix).strip()}_{str(kind).strip()}_{suffix}"


def _cp_empty_payload(schema_version: int, timestamp_s: float = 0.0) -> Dict[str, object]:
    return empty_cp_payload(schema_version=schema_version, timestamp_s=timestamp_s)


def _normalize_cp_payload(payload: object, schema_version: int) -> Dict[str, object]:
    from utility.cp_messages import normalize_cp_payload

    return normalize_cp_payload(payload, schema_version=schema_version)


def _load_cp_payload(message_path: str, schema_version: int) -> Dict[str, object]:
    return load_cp_message_payload(
        message_path=message_path,
        schema_version=schema_version,
    )


def _write_cp_payload(message_path: str, payload: Mapping[str, object]) -> None:
    write_cp_message_payload(payload, message_path=message_path)


def _reset_cp_message_file(*, message_path: str, schema_version: int) -> None:
    reset_cp_message_payload(
        message_path=message_path,
        schema_version=schema_version,
    )


def _upsert_cp_item(
    *,
    message_path: str,
    schema_version: int,
    list_name: str,
    item: Mapping[str, object],
    timestamp_s: float,
) -> bool:
    return bool(
        upsert_cp_item(
            message_path=message_path,
            schema_version=schema_version,
            list_name=list_name,
            item=item,
            timestamp_s=timestamp_s,
        )
    )


def _remove_cp_item(
    *,
    message_path: str,
    schema_version: int,
    list_name: str,
    item_id: str,
    timestamp_s: float,
) -> bool:
    return bool(
        remove_cp_item(
            message_path=message_path,
            schema_version=schema_version,
            list_name=list_name,
            item_id=item_id,
            timestamp_s=timestamp_s,
        )
    )


def _lane_event_from_marker(prefix: str, marker: Mapping[str, object]) -> Dict[str, object]:
    event: Dict[str, object] = {
        "id": _message_id(prefix, "lane_closure", marker),
        "type": "lane_closure",
        "position": [float(value) for value in list(marker.get("position_xy", []))[:2]],
        "block_entire_road": True,
    }
    if marker.get("road_id", None) is not None:
        event["road_id"] = int(marker.get("road_id"))
    if marker.get("section_id", None) is not None:
        event["section_id"] = int(marker.get("section_id"))
    if marker.get("lane_id", None) is not None:
        event["lane_id"] = int(marker.get("lane_id"))
        event["lane_ids"] = [int(marker.get("lane_id"))]
    if marker.get("carla_lane_id", None) is not None:
        event["carla_lane_id"] = int(marker.get("carla_lane_id"))
    return event


def _stopping_point(marker: Mapping[str, object]) -> List[float]:
    waypoint_transform = marker.get("waypoint_transform", None)
    waypoint_xy = _transform_xy(waypoint_transform)
    if waypoint_xy is not None:
        return waypoint_xy
    return [float(value) for value in list(marker.get("position_xy", []))[:2]]


def _traffic_light_stop_waypoints(actor: Any) -> List[Any]:
    stop_waypoints: List[Any] = []
    for method_name in ("get_stop_waypoints", "get_affected_lane_waypoints"):
        method = getattr(actor, method_name, None)
        if not callable(method):
            continue
        try:
            returned_waypoints = list(method() or [])
        except Exception:
            continue
        for waypoint in returned_waypoints:
            if waypoint is not None:
                stop_waypoints.append(waypoint)
        if stop_waypoints:
            break
    return stop_waypoints


def _stop_waypoint_match_details(
    *,
    stop_waypoint: Any,
    marker: Mapping[str, object],
) -> Tuple[int, float] | None:
    marker_stop_xy = _stopping_point(marker)
    if len(marker_stop_xy) < 2:
        return None

    stop_location = getattr(getattr(stop_waypoint, "transform", None), "location", None)
    if stop_location is None:
        return None

    match_distance_m = float(
        math.hypot(
            float(stop_location.x) - float(marker_stop_xy[0]),
            float(stop_location.y) - float(marker_stop_xy[1]),
        )
    )

    marker_road_id = marker.get("road_id", None)
    marker_section_id = marker.get("section_id", None)
    marker_lane_id = marker.get("lane_id", None)
    stop_waypoint_road_id = int(getattr(stop_waypoint, "road_id", 0))
    stop_waypoint_section_id = int(getattr(stop_waypoint, "section_id", 0))
    stop_waypoint_lane_id = int(canonical_lane_id_for_waypoint(stop_waypoint))

    road_matches = (
        marker_road_id is not None
        and int(stop_waypoint_road_id) == int(marker_road_id)
    )
    section_matches = (
        marker_section_id is not None
        and int(stop_waypoint_section_id) == int(marker_section_id)
    )
    lane_matches = (
        marker_lane_id is not None
        and int(marker_lane_id) != 0
        and int(stop_waypoint_lane_id) == int(marker_lane_id)
    )

    if road_matches and section_matches and lane_matches:
        return 0, float(match_distance_m)
    if road_matches and lane_matches:
        return 1, float(match_distance_m)
    if road_matches and section_matches:
        return 2, float(match_distance_m)
    if road_matches:
        return 3, float(match_distance_m)
    return 4, float(match_distance_m)


def _intersection_signal_state_for_marker(
    *,
    world,
    marker: Mapping[str, object],
    stop_waypoint_match_distance_m: float,
    actor_position_match_distance_m: float,
    preferred_actor_name: str = "",
) -> Tuple[str, str]:
    del preferred_actor_name
    marker_reference_xy = marker.get("position_xy", None)
    if not isinstance(marker_reference_xy, Sequence) or len(marker_reference_xy) < 2:
        marker_reference_xy = _stopping_point(marker)
    if len(marker_reference_xy) < 2:
        return "unknown", ""

    traffic_light_actors = list(_iter_world_traffic_lights(world))
    best_candidate: Tuple[Tuple[float, float, str], str, str] | None = None
    for actor in traffic_light_actors:
        actor_name = _traffic_light_actor_name(actor)
        signal_state = "unknown"
        get_state_fn = getattr(actor, "get_state", None)
        if callable(get_state_fn):
            try:
                signal_state = _normalize_signal_state(get_state_fn())
            except Exception:
                signal_state = "unknown"

        stop_waypoint_distances_m = [
            _distance_xy(
                marker_reference_xy,
                _transform_xy(getattr(stop_waypoint, "transform", None)),
            )
            for stop_waypoint in _traffic_light_stop_waypoints(actor)
        ]
        stop_waypoint_distances_m = [
            float(distance_m)
            for distance_m in stop_waypoint_distances_m
            if distance_m is not None
        ]
        if len(stop_waypoint_distances_m) > 0:
            match_distance_m = min(stop_waypoint_distances_m)
            max_match_distance_m = max(
                float(stop_waypoint_match_distance_m),
                float(actor_position_match_distance_m),
            )
            if float(match_distance_m) <= float(max_match_distance_m):
                priority = (0.0, float(match_distance_m), str(actor_name).strip().lower())
                if best_candidate is None or priority < best_candidate[0]:
                    best_candidate = (priority, str(signal_state), str(actor_name))
                continue

        actor_xy = _transform_xy(_object_transform(actor))
        actor_distance_m = _distance_xy(marker_reference_xy, actor_xy)
        if (
            actor_distance_m is None
            or float(actor_distance_m) > float(actor_position_match_distance_m)
        ):
            continue

        priority = (1.0, float(actor_distance_m), str(actor_name).strip().lower())
        if best_candidate is None or priority < best_candidate[0]:
            best_candidate = (priority, str(signal_state), str(actor_name))

    if best_candidate is None:
        return "unknown", ""
    return str(best_candidate[1]), str(best_candidate[2])


def _intersection_signal_requires_stop(signal_state: object) -> bool:
    normalized_signal_state = _normalize_signal_state(signal_state)
    return str(normalized_signal_state) in {"red", "yellow"}


def _control_message_from_marker(
    *,
    prefix: str,
    message_type: str,
    state: str,
    marker: Mapping[str, object],
    message_id: str | None = None,
    extra_fields: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    resolved_message_id = "" if message_id is None else str(message_id).strip()
    message: Dict[str, object] = {
        "id": resolved_message_id or _message_id(prefix, message_type, marker),
        "type": str(message_type),
        "state": str(state),
        "stopping_point": _stopping_point(marker),
    }
    if marker.get("road_id", None) is not None:
        message["road_id"] = int(marker.get("road_id"))
    if marker.get("lane_id", None) is not None:
        message["lane_id"] = int(marker.get("lane_id"))
    if isinstance(extra_fields, Mapping):
        for key, value in extra_fields.items():
            if value is not None:
                message[str(key)] = value
    return message


def _clone_transform_with_pose(location_transform, rotation_transform, carla, z_m: float):
    return carla.Transform(
        carla.Location(
            x=float(location_transform.location.x),
            y=float(location_transform.location.y),
            z=float(z_m),
        ),
        carla.Rotation(
            pitch=float(rotation_transform.rotation.pitch),
            yaw=float(rotation_transform.rotation.yaw),
            roll=float(rotation_transform.rotation.roll),
        ),
    )


def _spawn_attempts_for_marker(
    marker: Mapping[str, object],
    carla,
    z_offset_m: float,
    *,
    prefer_waypoint_transform: bool = False,
) -> List[Any]:
    marker_transform = marker.get("transform", None)
    waypoint_transform = marker.get("waypoint_transform", None)
    marker_attempts: List[Any] = []
    waypoint_attempts: List[Any] = []
    rotation_source_transform = marker_transform
    ground_z_m = None if marker_transform is None else float(marker_transform.location.z)
    if waypoint_transform is not None:
        rotation_source_transform = waypoint_transform
        ground_z_m = float(waypoint_transform.location.z)

    if marker_transform is not None:
        for extra_z_m in (0.0, 0.10, 0.25, 0.50, 1.0):
            marker_attempts.append(
                _clone_transform_with_pose(
                    marker_transform,
                    rotation_source_transform or marker_transform,
                    carla,
                    float(ground_z_m if ground_z_m is not None else marker_transform.location.z)
                    + float(z_offset_m)
                    + float(extra_z_m),
                )
            )

    if waypoint_transform is not None:
        for extra_z_m in (0.0, 0.10, 0.25, 0.50, 1.0):
            waypoint_attempts.append(
                _clone_transform_with_pose(
                    waypoint_transform,
                    waypoint_transform,
                    carla,
                    float(waypoint_transform.location.z) + float(z_offset_m) + float(extra_z_m),
                )
            )
    return (
        list(waypoint_attempts) + list(marker_attempts)
        if bool(prefer_waypoint_transform)
        else list(marker_attempts) + list(waypoint_attempts)
    )


def _vehicle_blueprint(world, blueprint_id: str, role_name: str, color_rgb: str):
    blueprint_library = world.get_blueprint_library()
    blueprints = list(blueprint_library.filter(str(blueprint_id).strip() or "vehicle.*"))
    if len(blueprints) == 0:
        raise RuntimeError(f"No CARLA vehicle blueprint matched '{blueprint_id}'.")
    blueprint = blueprints[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", str(role_name))
    if str(color_rgb).strip() and blueprint.has_attribute("color"):
        blueprint.set_attribute("color", str(color_rgb).strip())
    return blueprint


def _configure_static_vehicle(vehicle, carla) -> None:
    for action in (
        lambda: vehicle.set_simulate_physics(True),
        lambda: vehicle.set_autopilot(False),
        lambda: vehicle.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0)),
        lambda: vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0)),
        lambda: vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0,
                hand_brake=True,
            )
        ),
    ):
        try:
            action()
        except Exception:
            pass


def _configure_autopilot_vehicle(vehicle, traffic_manager_port: int, carla=None) -> None:
    try:
        vehicle.set_simulate_physics(True)
    except Exception:
        pass
    if carla is not None and hasattr(carla, "VehicleControl"):
        try:
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=0.0,
                    steer=0.0,
                    hand_brake=False,
                )
            )
        except Exception:
            pass
    try:
        vehicle.set_autopilot(True, int(traffic_manager_port))
    except TypeError:
        try:
            vehicle.set_autopilot(True)
        except Exception:
            pass
    except Exception:
        pass


def _spawn_vehicle_at_marker(
    *,
    world,
    carla,
    marker: Mapping[str, object],
    role_name: str,
    blueprint_id: str,
    color_rgb: str,
    spawn_z_offset_m: float,
    autopilot_enabled: bool,
    traffic_manager_port: int,
    prefer_waypoint_transform: bool = False,
) -> Any | None:
    blueprint = _vehicle_blueprint(
        world,
        blueprint_id=blueprint_id,
        role_name=role_name,
        color_rgb=color_rgb,
    )
    spawned_vehicle = None
    for attempt_transform in _spawn_attempts_for_marker(
        marker,
        carla,
        z_offset_m=float(spawn_z_offset_m),
        prefer_waypoint_transform=bool(prefer_waypoint_transform),
    ):
        spawned_vehicle = world.try_spawn_actor(blueprint, attempt_transform)
        if spawned_vehicle is not None:
            break
    if spawned_vehicle is None:
        _warning(f"failed to spawn vehicle at marker '{marker.get('name', '')}'.")
        return None

    if bool(autopilot_enabled):
        _configure_autopilot_vehicle(
            spawned_vehicle,
            traffic_manager_port=int(traffic_manager_port),
            carla=carla,
        )
    else:
        _configure_static_vehicle(spawned_vehicle, carla)
    return spawned_vehicle


def _spawn_vru(
    *,
    world,
    carla,
    start_transform: Any,
    goal_transform: Any,
    blueprint_filter: str,
    role_name: str,
    speed_mps: float,
    spawn_z_offset_m: float,
) -> Tuple[Any | None, Any | None]:
    if start_transform is None or goal_transform is None:
        return None, None

    blueprint_library = world.get_blueprint_library()
    walker_blueprints = list(blueprint_library.filter(str(blueprint_filter).strip() or "walker.pedestrian.*"))
    if len(walker_blueprints) == 0:
        return None, None
    walker_blueprint = random.choice(walker_blueprints)
    if walker_blueprint.has_attribute("is_invincible"):
        walker_blueprint.set_attribute("is_invincible", "false")
    if walker_blueprint.has_attribute("role_name"):
        walker_blueprint.set_attribute("role_name", str(role_name))

    walker_transform = _clone_transform_with_pose(
        start_transform,
        start_transform,
        carla,
        float(start_transform.location.z) + float(spawn_z_offset_m),
    )
    walker_actor = world.try_spawn_actor(walker_blueprint, walker_transform)
    if walker_actor is None:
        return None, None

    return walker_actor, None


def _apply_vru_goal_motion(
    *,
    runtime_state: Dict[str, object],
    carla,
    walker_actor: Any,
) -> None:
    goal_marker = runtime_state.get("vru_goal_marker", None)
    if not isinstance(goal_marker, Mapping):
        return
    goal_transform = goal_marker.get("transform", None)
    goal_location = getattr(goal_transform, "location", None)
    actor_transform = _object_transform(walker_actor)
    actor_location = getattr(actor_transform, "location", None)
    if goal_location is None or actor_location is None:
        return

    dx = float(goal_location.x) - float(actor_location.x)
    dy = float(goal_location.y) - float(actor_location.y)
    distance_m = math.hypot(dx, dy)
    arrival_threshold_m = max(0.1, float(runtime_state.get("vru_arrival_threshold_m", 0.5)))
    speed_mps = 0.0
    direction = carla.Vector3D(x=0.0, y=0.0, z=0.0)
    if distance_m > arrival_threshold_m:
        speed_mps = max(0.0, float(runtime_state.get("vru_speed_mps", 1.2)))
        direction = carla.Vector3D(x=float(dx / distance_m), y=float(dy / distance_m), z=0.0)
    else:
        runtime_state["vru_reached_goal"] = True

    try:
        control = carla.WalkerControl()
        control.direction = direction
        control.speed = float(speed_mps)
        control.jump = False
        walker_actor.apply_control(control)
    except Exception:
        pass


def _destroy_actor(actor: Any) -> None:
    if actor is None:
        return
    stop_fn = getattr(actor, "stop", None)
    if callable(stop_fn):
        try:
            stop_fn()
        except Exception:
            pass
    destroy_fn = getattr(actor, "destroy", None)
    if callable(destroy_fn):
        try:
            destroy_fn()
        except Exception:
            pass


def _destroy_existing_usecase_actors(world) -> None:
    for actor in _iter_world_actors(world):
        role_name = str(getattr(actor, "attributes", {}).get("role_name", "")).strip().lower()
        type_id = str(getattr(actor, "type_id", "")).strip().lower()
        if (
            role_name.startswith("vehicle_aus_")
            or role_name.startswith("hazard_aus_")
            or role_name.startswith("all_usecase_")
            or "controller.ai.walker" in type_id
        ):
            _destroy_actor(actor)


def _actor_snapshot(actor: Any, *, actor_id: str) -> Dict[str, object] | None:
    transform = _object_transform(actor)
    location = getattr(transform, "location", None)
    rotation = getattr(transform, "rotation", None)
    if location is None:
        return None

    velocity = None
    get_velocity_fn = getattr(actor, "get_velocity", None)
    if callable(get_velocity_fn):
        try:
            velocity = get_velocity_fn()
        except Exception:
            velocity = None
    speed_mps = 0.0
    if velocity is not None:
        speed_mps = math.sqrt(
            float(getattr(velocity, "x", 0.0)) ** 2
            + float(getattr(velocity, "y", 0.0)) ** 2
            + float(getattr(velocity, "z", 0.0)) ** 2
        )

    bbox = getattr(actor, "bounding_box", None)
    extent = getattr(bbox, "extent", None)
    half_length_m = max(0.05, float(getattr(extent, "x", 0.40)))
    half_width_m = max(0.05, float(getattr(extent, "y", 0.35)))
    half_height_m = max(0.05, float(getattr(extent, "z", 0.90)))
    yaw_rad = 0.0 if rotation is None else math.radians(float(getattr(rotation, "yaw", 0.0)))
    return {
        "vehicle_id": str(actor_id),
        "x": float(location.x),
        "y": float(location.y),
        "z": float(getattr(location, "z", 0.0)),
        "v": float(speed_mps),
        "psi": float(yaw_rad),
        "length_m": float(half_length_m * 2.0),
        "width_m": float(half_width_m * 2.0),
        "height_m": float(half_height_m * 2.0),
    }


def _route_projection(route_points: Sequence[Sequence[float]], point_xy: Sequence[float]) -> Tuple[float, float] | None:
    points = [
        (float(point[0]), float(point[1]))
        for point in list(route_points or [])
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if len(points) < 2:
        return None

    px, py = float(point_xy[0]), float(point_xy[1])
    best_arc_m = 0.0
    best_distance_m = float("inf")
    cumulative_arc_m = 0.0
    for idx in range(len(points) - 1):
        ax, ay = points[idx]
        bx, by = points[idx + 1]
        dx = bx - ax
        dy = by - ay
        segment_length_m = math.hypot(dx, dy)
        if segment_length_m <= 1.0e-6:
            continue
        t = ((px - ax) * dx + (py - ay) * dy) / (segment_length_m * segment_length_m)
        t = max(0.0, min(1.0, float(t)))
        cx = ax + t * dx
        cy = ay + t * dy
        distance_m = math.hypot(px - cx, py - cy)
        if distance_m < best_distance_m:
            best_distance_m = float(distance_m)
            best_arc_m = float(cumulative_arc_m + t * segment_length_m)
        cumulative_arc_m += segment_length_m
    return best_arc_m, best_distance_m


def _marker_progress_delta_m(
    *,
    marker: Mapping[str, object],
    ego_transform,
    active_global_route_points: Sequence[Sequence[float]],
) -> float | None:
    ego_xy = _transform_xy(ego_transform)
    if ego_xy is None:
        return None

    route_ego_projection = _route_projection(active_global_route_points, ego_xy)
    if route_ego_projection is not None:
        ego_arc_m, _ego_route_error_m = route_ego_projection
        marker_xy = _stopping_point(marker)
        if len(marker_xy) >= 2:
            marker_projection = _route_projection(active_global_route_points, marker_xy)
            if marker_projection is not None:
                marker_arc_m, _marker_route_error_m = marker_projection
                return float(marker_arc_m) - float(ego_arc_m)

    heading_yaw_rad = math.radians(float(getattr(getattr(ego_transform, "rotation", None), "yaw", 0.0)))
    heading_x = math.cos(heading_yaw_rad)
    heading_y = math.sin(heading_yaw_rad)
    marker_xy = _stopping_point(marker)
    if len(marker_xy) < 2:
        return None
    dx = float(marker_xy[0]) - float(ego_xy[0])
    dy = float(marker_xy[1]) - float(ego_xy[1])
    return float(dx * heading_x + dy * heading_y)


def _marker_by_requested_name(markers: Sequence[Mapping[str, object]], requested_name: str) -> Mapping[str, object] | None:
    requested = str(requested_name).strip().lower()
    if not requested:
        return None
    starts_with_candidates: List[Tuple[int, Mapping[str, object]]] = []
    partial_candidates: List[Tuple[int, Mapping[str, object]]] = []
    for marker in list(markers or []):
        marker_name = str(marker.get("name", "")).strip().lower()
        if marker_name == requested:
            return marker
        if marker_name.startswith(f"{requested}_"):
            starts_with_candidates.append((len(marker_name), marker))
        elif requested in marker_name:
            partial_candidates.append((len(marker_name), marker))
    return _best_match(starts_with_candidates) or _best_match(partial_candidates)


def _marker_by_index(
    markers: Sequence[Mapping[str, object]],
    marker_index: object,
) -> Mapping[str, object] | None:
    try:
        normalized_index = int(marker_index)
    except Exception:
        return None
    for marker in list(markers or []):
        try:
            if int(marker.get("index", -1)) == int(normalized_index):
                return marker
        except Exception:
            continue
    return None


def initialize_runtime(
    *,
    scenario_cfg: Mapping[str, object],
    world,
    world_map=None,
    carla,
    **extras,
) -> Dict[str, object]:
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    message_prefix = str(runtime_cfg.get("cp_message_prefix", "all_usecase")).strip() or "all_usecase"
    cp_message_path = str(runtime_cfg.get("cp_message_path", DEFAULT_CP_MESSAGE_PATH)).strip() or DEFAULT_CP_MESSAGE_PATH
    cp_schema_version = int(runtime_cfg.get("cp_message_schema_version", 1))

    if bool(runtime_cfg.get("reset_cp_message_file", True)):
        _reset_cp_message_file(message_path=cp_message_path, schema_version=cp_schema_version)

    _destroy_existing_usecase_actors(world)

    hazard_marker_name = str(runtime_cfg.get("hazard_marker_name", "hazard_aus_1")).strip()
    intersection_prefix = str(runtime_cfg.get("intersection_marker_prefix", "intersection_aus_")).strip()
    stop_prefix = str(runtime_cfg.get("stop_marker_prefix", "stop_aus_")).strip()
    vehicle_prefix = str(runtime_cfg.get("vehicle_marker_prefix", "vehicle_aus_")).strip()
    vru_start_name = str(runtime_cfg.get("vru_start_marker_name", "vru_aus_1")).strip()
    vru_goal_name = str(runtime_cfg.get("vru_goal_marker_name", "vru_aus_2")).strip()

    hazard_marker = _record_by_name(
        world=world,
        world_map=world_map,
        carla=carla,
        marker_name=hazard_marker_name,
    )
    intersection_markers = _records_by_prefix(
        world=world,
        world_map=world_map,
        carla=carla,
        prefix=intersection_prefix,
    )
    stop_markers = _records_by_prefix(
        world=world,
        world_map=world_map,
        carla=carla,
        prefix=stop_prefix,
    )
    vehicle_markers = _records_by_indexed_names(
        world=world,
        world_map=world_map,
        carla=carla,
        prefix=vehicle_prefix,
    )
    vru_start_marker = _record_by_name(
        world=world,
        world_map=world_map,
        carla=carla,
        marker_name=vru_start_name,
    )
    vru_goal_marker = _record_by_name(
        world=world,
        world_map=world_map,
        carla=carla,
        marker_name=vru_goal_name,
    )

    hazard_vehicle_id = None
    if isinstance(hazard_marker, Mapping):
        hazard_vehicle = _spawn_vehicle_at_marker(
            world=world,
            carla=carla,
            marker=hazard_marker,
            role_name=str(hazard_marker_name),
            blueprint_id=str(runtime_cfg.get("hazard_vehicle_blueprint", "vehicle.tesla.model3")),
            color_rgb=str(runtime_cfg.get("hazard_vehicle_color_rgb", "40,40,40")),
            spawn_z_offset_m=float(runtime_cfg.get("vehicle_spawn_z_offset_m", 0.05)),
            autopilot_enabled=False,
            traffic_manager_port=int(dict(scenario_cfg.get("traffic_manager", {})).get("port", 8000)),
        )
        hazard_vehicle_id = None if hazard_vehicle is None else int(getattr(hazard_vehicle, "id", 0) or 0)
    else:
        _warning(f"hazard marker '{hazard_marker_name}' was not found.")

    _info(
        "markers: "
        f"intersections={len(intersection_markers)} stops={len(stop_markers)} "
        f"vehicles={len(vehicle_markers)} "
        f"vru_start={'yes' if isinstance(vru_start_marker, Mapping) else 'no'} "
        f"vru_goal={'yes' if isinstance(vru_goal_marker, Mapping) else 'no'}"
    )

    if not isinstance(vru_start_marker, Mapping):
        _warning(f"VRU start marker '{vru_start_name}' was not found.")
    if not isinstance(vru_goal_marker, Mapping):
        _warning(f"VRU goal marker '{vru_goal_name}' was not found.")

    traffic_manager_cfg = dict(scenario_cfg.get("traffic_manager", {}))
    runtime_state = {
        "message_prefix": message_prefix,
        "cp_message_path": cp_message_path,
        "cp_schema_version": cp_schema_version,
        "hazard_marker": hazard_marker,
        "hazard_vehicle_id": hazard_vehicle_id,
        "intersection_markers": intersection_markers,
        "stop_markers": stop_markers,
        "vehicle_markers": vehicle_markers,
        "vru_start_marker": vru_start_marker,
        "vru_goal_marker": vru_goal_marker,
        "lane_closure_trigger_distance_m": float(runtime_cfg.get("lane_closure_trigger_distance_m", 40.0)),
        "control_trigger_distance_m": float(runtime_cfg.get("control_trigger_distance_m", 20.0)),
        "traffic_light_stop_waypoint_match_distance_m": float(
            runtime_cfg.get("traffic_light_stop_waypoint_match_distance_m", 12.0)
        ),
        "traffic_light_actor_position_match_distance_m": float(
            runtime_cfg.get("traffic_light_actor_position_match_distance_m", 40.0)
        ),
        "manual_vehicle_trigger_distance_m": float(runtime_cfg.get("manual_vehicle_trigger_distance_m", 5.0)),
        "vru_trigger_distance_m": float(runtime_cfg.get("vru_trigger_distance_m", 20.0)),
        "manual_vehicle_trigger_marker_name": str(runtime_cfg.get("manual_vehicle_trigger_marker_name", "stop_aus_1")).strip(),
        "stop_state": str(runtime_cfg.get("stop_state", "stop")).strip() or "stop",
        "manual_vehicle_blueprint": str(runtime_cfg.get("manual_vehicle_blueprint", "vehicle.tesla.model3")).strip(),
        "manual_vehicle_color_rgb": str(runtime_cfg.get("manual_vehicle_color_rgb", "90,90,90")).strip(),
        "vehicle_spawn_z_offset_m": float(runtime_cfg.get("vehicle_spawn_z_offset_m", 0.05)),
        "autopilot_vehicle_indices": set(
            int(value)
            for value in list(runtime_cfg.get("autopilot_vehicle_indices", [1, 2, 3]))
        ),
        "delayed_autopilot_vehicle_indices": set(
            int(value)
            for value in list(runtime_cfg.get("delayed_autopilot_vehicle_indices", list(range(11, 20))))
        ),
        "delayed_autopilot_trigger_distance_m": float(
            runtime_cfg.get("delayed_autopilot_trigger_distance_m", 40.0)
        ),
        "traffic_manager_port": int(traffic_manager_cfg.get("port", 8000)),
        "vru_blueprint_filter": str(runtime_cfg.get("vru_blueprint_filter", "walker.pedestrian.*")).strip(),
        "vru_speed_mps": float(runtime_cfg.get("vru_speed_mps", 1.2)),
        "vru_spawn_z_offset_m": float(runtime_cfg.get("vru_spawn_z_offset_m", 0.0)),
        "vru_arrival_threshold_m": float(runtime_cfg.get("vru_arrival_threshold_m", 0.5)),
        "lane_closure_sent": False,
        "triggered_intersection_marker_names": set(),
        "crossed_intersection_marker_names": set(),
        "intersection_signal_states": {},
        "intersection_signal_actor_names": {},
        "stop_messages_sent": set(),
        "manual_vehicles_spawned": False,
        "manual_vehicle_actor_ids": [],
        "delayed_autopilot_vehicle_actor_ids": {},
        "delayed_autopilot_vehicle_activated_indices": set(),
        "vru_spawned": False,
        "vru_reached_goal": False,
        "vru_actor_id": None,
        "vru_controller_id": None,
        "last_runtime_event_time_s": float(time.perf_counter()),
    }
    _maybe_spawn_manual_vehicles(
        runtime_state=runtime_state,
        world=world,
        carla=carla,
        ego_xy=None,
    )
    return world_messages.initialize_runtime_state(
        runtime_state=runtime_state,
        runtime_cfg=runtime_cfg,
        tracker_cfg=extras.get("tracker_cfg", None),
        obstacle_filter_cfg=extras.get("obstacle_filter_cfg", None),
        prediction_dt_s=extras.get("prediction_dt_s", None),
        prediction_horizon_s=extras.get("prediction_horizon_s", None),
    )


def _maybe_register_lane_closure(
    *,
    runtime_state: Dict[str, object],
    ego_xy: Sequence[float] | None,
    sim_time_s: float,
) -> None:
    if bool(runtime_state.get("lane_closure_sent", False)):
        return
    marker = runtime_state.get("hazard_marker", None)
    if not isinstance(marker, Mapping):
        return
    distance_m = _distance_xy(ego_xy, marker.get("position_xy", None))
    if distance_m is None or float(distance_m) > float(runtime_state.get("lane_closure_trigger_distance_m", 40.0)):
        return
    inserted = _upsert_cp_item(
        message_path=str(runtime_state.get("cp_message_path", DEFAULT_CP_MESSAGE_PATH)),
        schema_version=int(runtime_state.get("cp_schema_version", 1)),
        list_name="lane_events",
        item=_lane_event_from_marker(str(runtime_state.get("message_prefix", "all_usecase")), marker),
        timestamp_s=float(sim_time_s),
    )
    runtime_state["lane_closure_sent"] = True
    if inserted:
        _info(f"registered lane_closure for {marker.get('name', '')}.")


def _maybe_register_stop_controls(
    *,
    runtime_state: Dict[str, object],
    ego_xy: Sequence[float] | None,
    sim_time_s: float,
) -> None:
    sent = set(runtime_state.get("stop_messages_sent", set()))
    for marker in list(runtime_state.get("stop_markers", []) or []):
        marker_name = str(marker.get("name", "")).strip()
        if not marker_name or marker_name in sent:
            continue
        distance_m = _distance_xy(ego_xy, marker.get("position_xy", None))
        if distance_m is None or float(distance_m) > float(runtime_state.get("control_trigger_distance_m", 20.0)):
            continue
        inserted = _upsert_cp_item(
            message_path=str(runtime_state.get("cp_message_path", DEFAULT_CP_MESSAGE_PATH)),
            schema_version=int(runtime_state.get("cp_schema_version", 1)),
            list_name="control",
            item=_control_message_from_marker(
                prefix=str(runtime_state.get("message_prefix", "all_usecase")),
                message_type="stop",
                state=str(runtime_state.get("stop_state", "stop")),
                marker=marker,
            ),
            timestamp_s=float(sim_time_s),
        )
        sent.add(marker_name)
        if inserted:
            _info(f"registered stop control for {marker_name}.")
    runtime_state["stop_messages_sent"] = sent


def _maybe_register_intersection_control(
    *,
    runtime_state: Dict[str, object],
    world,
    ego_transform,
    active_global_route_points: Sequence[Sequence[float]],
    sim_time_s: float,
) -> None:
    ego_xy = _transform_xy(ego_transform)
    if ego_xy is None:
        return

    triggered = set(runtime_state.get("triggered_intersection_marker_names", set()))
    crossed = set(runtime_state.get("crossed_intersection_marker_names", set()))
    signal_states = dict(runtime_state.get("intersection_signal_states", {}) or {})
    signal_actor_names = dict(runtime_state.get("intersection_signal_actor_names", {}) or {})
    trigger_distance_m = float(runtime_state.get("control_trigger_distance_m", 20.0))
    stop_waypoint_match_distance_m = float(
        runtime_state.get("traffic_light_stop_waypoint_match_distance_m", 12.0)
    )
    actor_position_match_distance_m = float(
        runtime_state.get("traffic_light_actor_position_match_distance_m", 40.0)
    )
    message_prefix = str(runtime_state.get("message_prefix", "all_usecase"))

    for marker in list(runtime_state.get("intersection_markers", []) or []):
        marker_name = str(marker.get("name", "")).strip()
        if not marker_name:
            continue
        if marker_name in crossed:
            continue

        signal_state, signal_actor_name = _intersection_signal_state_for_marker(
            world=world,
            marker=marker,
            stop_waypoint_match_distance_m=float(stop_waypoint_match_distance_m),
            actor_position_match_distance_m=float(actor_position_match_distance_m),
            preferred_actor_name=str(signal_actor_names.get(marker_name, "")),
        )

        progress_delta_m = _marker_progress_delta_m(
            marker=marker,
            ego_transform=ego_transform,
            active_global_route_points=active_global_route_points,
        )
        if progress_delta_m is not None and float(progress_delta_m) < -2.0:
            removed = _remove_cp_item(
                message_path=str(runtime_state.get("cp_message_path", DEFAULT_CP_MESSAGE_PATH)),
                schema_version=int(runtime_state.get("cp_schema_version", 1)),
                list_name="control",
                item_id=_message_id(message_prefix, "traffic_light", marker),
                timestamp_s=float(sim_time_s),
            )
            triggered.discard(marker_name)
            crossed.add(marker_name)
            signal_states.pop(marker_name, None)
            signal_actor_names.pop(marker_name, None)
            if removed:
                _info(f"removed traffic_light control for crossed {marker_name}.")
            continue

        if marker_name not in triggered:
            distance_m = _distance_xy(ego_xy, marker.get("position_xy", None))
            if distance_m is None or float(distance_m) > float(trigger_distance_m):
                continue
            triggered.add(marker_name)
        previous_signal_state = str(signal_states.get(marker_name, "")).strip()
        previous_signal_actor_name = str(signal_actor_names.get(marker_name, "")).strip()
        normalized_signal_state = str(signal_state or "unknown").strip() or "unknown"
        if (
            normalized_signal_state == "unknown"
            and not str(signal_actor_name).strip()
            and previous_signal_state
        ):
            normalized_signal_state = str(previous_signal_state)
            if previous_signal_actor_name:
                signal_actor_name = str(previous_signal_actor_name)
        updated = _upsert_cp_item(
            message_path=str(runtime_state.get("cp_message_path", DEFAULT_CP_MESSAGE_PATH)),
            schema_version=int(runtime_state.get("cp_schema_version", 1)),
            list_name="control",
            item=_control_message_from_marker(
                prefix=message_prefix,
                message_type="intersection",
                state=str(normalized_signal_state),
                marker=marker,
                message_id=_message_id(message_prefix, "traffic_light", marker),
                extra_fields={
                    "intersection_marker_name": str(marker_name),
                    "signal_actor_name": str(signal_actor_name).strip() or None,
                    "stop": bool(_intersection_signal_requires_stop(normalized_signal_state)),
                },
            ),
            timestamp_s=float(sim_time_s),
        )
        signal_states[marker_name] = str(normalized_signal_state)
        if str(signal_actor_name).strip():
            signal_actor_names[marker_name] = str(signal_actor_name)

        if updated:
            actor_suffix = f" actor={signal_actor_name}" if str(signal_actor_name).strip() else ""
            if previous_signal_state:
                _info(
                    f"updated intersection control for {marker_name}: state={normalized_signal_state} "
                    f"stop={bool(_intersection_signal_requires_stop(normalized_signal_state))}.{actor_suffix}"
                )
            else:
                _info(
                    f"registered intersection control for {marker_name}: state={normalized_signal_state} "
                    f"stop={bool(_intersection_signal_requires_stop(normalized_signal_state))}.{actor_suffix}"
                )

    runtime_state["triggered_intersection_marker_names"] = triggered
    runtime_state["crossed_intersection_marker_names"] = crossed
    runtime_state["intersection_signal_states"] = signal_states
    runtime_state["intersection_signal_actor_names"] = signal_actor_names


def _manual_vehicle_mode(
    *,
    runtime_state: Mapping[str, object],
    marker_index: object,
) -> str:
    try:
        normalized_index = int(marker_index)
    except Exception:
        return "static"
    autopilot_indices = set(runtime_state.get("autopilot_vehicle_indices", {1, 2, 3}))
    delayed_autopilot_indices = set(
        runtime_state.get("delayed_autopilot_vehicle_indices", set(range(11, 20)))
    )
    if int(normalized_index) in autopilot_indices:
        return "autopilot"
    if int(normalized_index) in delayed_autopilot_indices:
        return "delayed_autopilot"
    return "static"


def _delayed_autopilot_trigger_marker(
    *,
    runtime_state: Mapping[str, object],
    marker_index: object,
) -> Mapping[str, object] | None:
    intersection_marker = _marker_by_index(
        runtime_state.get("intersection_markers", []),
        marker_index,
    )
    if isinstance(intersection_marker, Mapping):
        return intersection_marker
    return _marker_by_index(
        runtime_state.get("stop_markers", []),
        marker_index,
    )


def _maybe_spawn_manual_vehicles(
    *,
    runtime_state: Dict[str, object],
    world,
    carla,
    ego_xy: Sequence[float] | None,
) -> None:
    del ego_xy
    if bool(runtime_state.get("manual_vehicles_spawned", False)):
        return
    vehicle_markers = list(runtime_state.get("vehicle_markers", []) or [])
    if len(vehicle_markers) == 0:
        return

    actor_ids: List[int] = []
    delayed_actor_ids: Dict[int, int] = {}
    for marker in vehicle_markers:
        marker_name = str(marker.get("name", "")).strip()
        marker_index = marker.get("index", None)
        vehicle_mode = _manual_vehicle_mode(
            runtime_state=runtime_state,
            marker_index=marker_index,
        )
        autopilot_enabled = str(vehicle_mode) == "autopilot"
        spawned_actor = _spawn_vehicle_at_marker(
            world=world,
            carla=carla,
            marker=marker,
            role_name=marker_name,
            blueprint_id=str(runtime_state.get("manual_vehicle_blueprint", "vehicle.tesla.model3")),
            color_rgb=str(runtime_state.get("manual_vehicle_color_rgb", "90,90,90")),
            spawn_z_offset_m=float(runtime_state.get("vehicle_spawn_z_offset_m", 0.05)),
            autopilot_enabled=bool(autopilot_enabled),
            traffic_manager_port=int(runtime_state.get("traffic_manager_port", 8000)),
            prefer_waypoint_transform=True,
        )
        if spawned_actor is not None:
            actor_id = int(getattr(spawned_actor, "id", 0) or 0)
            actor_ids.append(actor_id)
            if str(vehicle_mode) == "delayed_autopilot" and marker_index is not None:
                delayed_actor_ids[int(marker_index)] = int(actor_id)

    runtime_state["manual_vehicles_spawned"] = True
    runtime_state["manual_vehicle_actor_ids"] = actor_ids
    runtime_state["delayed_autopilot_vehicle_actor_ids"] = delayed_actor_ids
    runtime_state["delayed_autopilot_vehicle_activated_indices"] = set()
    _info(f"spawned {len(actor_ids)} marker vehicle(s).")


def _maybe_activate_delayed_autopilot_vehicles(
    *,
    runtime_state: Dict[str, object],
    world,
    carla,
    ego_xy: Sequence[float] | None,
) -> None:
    delayed_actor_ids = {
        int(marker_index): int(actor_id)
        for marker_index, actor_id in dict(
            runtime_state.get("delayed_autopilot_vehicle_actor_ids", {}) or {}
        ).items()
    }
    if len(delayed_actor_ids) == 0:
        return

    activated_indices = {
        int(marker_index)
        for marker_index in set(
            runtime_state.get("delayed_autopilot_vehicle_activated_indices", set())
        )
    }
    trigger_distance_m = float(
        runtime_state.get("delayed_autopilot_trigger_distance_m", 40.0)
    )
    remaining_actor_ids: Dict[int, int] = {}
    for marker_index, actor_id in delayed_actor_ids.items():
        if int(marker_index) in activated_indices:
            continue
        trigger_marker = _delayed_autopilot_trigger_marker(
            runtime_state=runtime_state,
            marker_index=int(marker_index),
        )
        trigger_distance = _distance_xy(
            ego_xy,
            _stopping_point(trigger_marker) if isinstance(trigger_marker, Mapping) else None,
        )
        if trigger_distance is None or float(trigger_distance) > float(trigger_distance_m):
            remaining_actor_ids[int(marker_index)] = int(actor_id)
            continue

        actor = _find_actor_by_id(world, actor_id)
        if actor is None:
            continue
        _configure_autopilot_vehicle(
            actor,
            traffic_manager_port=int(runtime_state.get("traffic_manager_port", 8000)),
            carla=carla,
        )
        activated_indices.add(int(marker_index))
        _info(
            f"enabled autopilot for vehicle_aus_{int(marker_index)} at "
            f"stop-distance trigger {float(trigger_distance):.2f} m."
        )

    runtime_state["delayed_autopilot_vehicle_actor_ids"] = remaining_actor_ids
    runtime_state["delayed_autopilot_vehicle_activated_indices"] = activated_indices


def _maybe_spawn_vru(
    *,
    runtime_state: Dict[str, object],
    world,
    carla,
    ego_xy: Sequence[float] | None,
) -> None:
    if bool(runtime_state.get("vru_spawned", False)):
        return
    start_marker = runtime_state.get("vru_start_marker", None)
    goal_marker = runtime_state.get("vru_goal_marker", None)
    if not isinstance(start_marker, Mapping) or not isinstance(goal_marker, Mapping):
        return
    distance_m = _distance_xy(ego_xy, start_marker.get("position_xy", None))
    if distance_m is None or float(distance_m) > float(runtime_state.get("vru_trigger_distance_m", 20.0)):
        return

    walker_actor, controller_actor = _spawn_vru(
        world=world,
        carla=carla,
        start_transform=start_marker.get("transform", None),
        goal_transform=goal_marker.get("transform", None),
        blueprint_filter=str(runtime_state.get("vru_blueprint_filter", "walker.pedestrian.*")),
        role_name="all_usecase_vru",
        speed_mps=float(runtime_state.get("vru_speed_mps", 1.2)),
        spawn_z_offset_m=float(runtime_state.get("vru_spawn_z_offset_m", 0.0)),
    )
    runtime_state["vru_spawned"] = walker_actor is not None
    runtime_state["vru_reached_goal"] = False
    runtime_state["vru_actor_id"] = None if walker_actor is None else int(getattr(walker_actor, "id", 0) or 0)
    runtime_state["vru_controller_id"] = None if controller_actor is None else int(getattr(controller_actor, "id", 0) or 0)
    if walker_actor is None:
        _warning("failed to spawn VRU.")
    else:
        _apply_vru_goal_motion(
            runtime_state=runtime_state,
            carla=carla,
            walker_actor=walker_actor,
        )
        _info(f"spawned VRU id={runtime_state['vru_actor_id']}.")


def maybe_replan_global_route(
    *,
    runtime_state,
    world,
    world_map,
    carla,
    ego_transform,
    active_global_route_points: Sequence[Sequence[float]] | None = None,
    sim_time_s: float,
    wall_time_s: float | None = None,
    **extras,
) -> Tuple[object | None, List[List[float]] | None, Dict[str, object]]:
    del wall_time_s
    next_state = dict(runtime_state or {})
    ego_xy = _transform_xy(ego_transform)

    _maybe_register_lane_closure(
        runtime_state=next_state,
        ego_xy=ego_xy,
        sim_time_s=float(sim_time_s),
    )
    _maybe_register_intersection_control(
        runtime_state=next_state,
        world=world,
        ego_transform=ego_transform,
        active_global_route_points=list(active_global_route_points or []),
        sim_time_s=float(sim_time_s),
    )
    _maybe_register_stop_controls(
        runtime_state=next_state,
        ego_xy=ego_xy,
        sim_time_s=float(sim_time_s),
    )
    _maybe_spawn_vru(
        runtime_state=next_state,
        world=world,
        carla=carla,
        ego_xy=ego_xy,
    )
    _maybe_spawn_manual_vehicles(
        runtime_state=next_state,
        world=world,
        carla=carla,
        ego_xy=ego_xy,
    )
    _maybe_activate_delayed_autopilot_vehicles(
        runtime_state=next_state,
        world=world,
        carla=carla,
        ego_xy=ego_xy,
    )
    next_state = world_messages.publish_obstacle_messages(
        runtime_state=next_state,
        world=world,
        world_map=world_map,
        carla=carla,
        ego_vehicle=extras.get("ego_vehicle", None),
        sim_time_s=float(sim_time_s),
        sumo_bridge=extras.get("sumo_bridge", None),
    )
    next_state["last_runtime_event_time_s"] = float(time.perf_counter())
    return None, None, next_state


def filter_dynamic_obstacle_snapshots(
    *,
    runtime_state,
    world,
    carla,
    object_snapshots: Sequence[Mapping[str, object]],
    **_,
) -> Tuple[List[dict], Dict[str, object]]:
    next_state = dict(runtime_state or {})
    filtered_snapshots = [dict(snapshot) for snapshot in list(object_snapshots or [])]

    vru_actor = _find_actor_by_id(world, next_state.get("vru_actor_id", None))
    if vru_actor is not None:
        if not bool(next_state.get("vru_reached_goal", False)):
            _apply_vru_goal_motion(
                runtime_state=next_state,
                carla=carla,
                walker_actor=vru_actor,
            )
        vru_snapshot = _actor_snapshot(vru_actor, actor_id="all_usecase_vru")
        if vru_snapshot is not None:
            filtered_snapshots.append(vru_snapshot)
    return filtered_snapshots, next_state
