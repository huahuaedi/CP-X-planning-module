"""
Scenario-local logic for the roadway_hazard use case.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from behavior_planner.reroute import (
    CP_MESSAGE_PATH,
    ensure_cp_message_file_exists,
    load_cp_messages,
    remove_cp_messages_by_id,
    write_cp_messages,
)


def _best_partial_match(candidates: List[Tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _find_environment_marker_by_name(world, carla, marker_name: str):
    marker_name_lower = str(marker_name).strip().lower()
    partial_candidates: List[Tuple[int, Any]] = []
    for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
        env_name = str(getattr(env_obj, "name", "")).strip().lower()
        if env_name == marker_name_lower:
            return env_obj
        if marker_name_lower and marker_name_lower in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_partial_match(partial_candidates)


def _find_actor_by_name(world, object_name: str):
    object_name_lower = str(object_name).strip().lower()
    if not object_name_lower:
        return None

    partial_candidates: List[Tuple[int, Any]] = []
    for actor in list(world.get_actors() if hasattr(world, "get_actors") else []):
        raw_attributes = getattr(actor, "attributes", {})
        attr_name = str(raw_attributes.get("name", "")).strip().lower()
        role_name = str(raw_attributes.get("role_name", "")).strip().lower()
        type_id = str(getattr(actor, "type_id", "")).strip().lower()
        if (
            attr_name == object_name_lower
            or role_name == object_name_lower
            or type_id.endswith(object_name_lower)
        ):
            return actor
        if object_name_lower in attr_name:
            partial_candidates.append((len(attr_name), actor))
        if object_name_lower in role_name:
            partial_candidates.append((len(role_name), actor))
        if object_name_lower in type_id:
            partial_candidates.append((len(type_id), actor))
    return _best_partial_match(partial_candidates)


def _object_transform(world_object: Any):
    transform = getattr(world_object, "transform", None)
    if transform is None and hasattr(world_object, "get_transform"):
        try:
            transform = world_object.get_transform()
        except RuntimeError:
            transform = None
    return transform


def _resolve_named_world_object(world, carla, object_name: str):
    marker = _find_environment_marker_by_name(world, carla, object_name)
    if marker is not None:
        return marker
    return _find_actor_by_name(world, object_name)


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


def _spawn_attempt_transforms(base_transform, waypoint_transform, carla, base_z_offset_m: float) -> List[Any]:
    attempts: List[Any] = []
    ground_z_m = float(base_transform.location.z)
    rotation_source_transform = base_transform
    if waypoint_transform is not None:
        ground_z_m = float(waypoint_transform.location.z)
        rotation_source_transform = waypoint_transform

    if waypoint_transform is not None:
        for extra_z_m in (0.0, 0.10, 0.20):
            attempts.append(
                _clone_transform_with_pose(
                    location_transform=waypoint_transform,
                    rotation_transform=rotation_source_transform,
                    carla=carla,
                    z_m=float(waypoint_transform.location.z) + float(base_z_offset_m) + float(extra_z_m),
                )
            )

    for extra_z_m in (0.0, 0.10, 0.20):
        attempts.append(
            _clone_transform_with_pose(
                location_transform=base_transform,
                rotation_transform=rotation_source_transform,
                carla=carla,
                z_m=float(ground_z_m) + float(base_z_offset_m) + float(extra_z_m),
            )
        )
    return attempts


def _configure_static_vehicle(vehicle, carla) -> None:
    try:
        vehicle.set_simulate_physics(True)
    except RuntimeError:
        pass
    try:
        vehicle.set_autopilot(False)
    except RuntimeError:
        pass
    try:
        vehicle.set_target_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
    except RuntimeError:
        pass
    try:
        vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
    except RuntimeError:
        pass
    try:
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=0.0,
                hand_brake=True,
                reverse=False,
                manual_gear_shift=False,
            )
        )
    except RuntimeError:
        pass


def _configure_autopilot_vehicle(vehicle, traffic_manager, traffic_manager_port: int) -> None:
    try:
        vehicle.set_simulate_physics(True)
    except RuntimeError:
        pass

    try:
        vehicle.set_autopilot(True, int(traffic_manager_port))
    except TypeError:
        try:
            vehicle.set_autopilot(True)
        except RuntimeError:
            pass
    except RuntimeError:
        pass

    if traffic_manager is None:
        return

    try:
        traffic_manager.auto_lane_change(vehicle, False)
    except Exception:
        pass
    try:
        traffic_manager.distance_to_leading_vehicle(vehicle, 1.0)
    except Exception:
        pass


def _lowercase_name_set(values: Sequence[object] | None, default_values: Sequence[str]) -> List[str]:
    raw_values = list(values) if values is not None else list(default_values)
    normalized: List[str] = []
    for value in raw_values:
        name = str(value).strip()
        if name:
            normalized.append(name)
    return normalized


def _transform_xy(transform: Any) -> List[float] | None:
    location = getattr(transform, "location", None)
    if location is None:
        return None
    return [float(location.x), float(location.y)]


def _distance_between_points(
    first_xy: Sequence[object] | None,
    second_xy: Sequence[object] | None,
) -> float | None:
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


def _resolved_wall_time_s(wall_time_s: float | None = None) -> float:
    if wall_time_s is not None:
        return float(wall_time_s)
    return float(time.perf_counter())


def _append_cp_message(
    *,
    message_path: str,
    message_id: str,
    message_type: str,
    position_xy: Sequence[float],
    extra_payload: Mapping[str, object] | None = None,
) -> None:
    ensure_cp_message_file_exists(message_path=message_path)
    current_messages = load_cp_messages(message_path=message_path)
    normalized_message_id = str(message_id).strip()
    retained_messages = [
        dict(message)
        for message in current_messages
        if str(message.get("id", "")).strip() != normalized_message_id
    ]
    next_message = {
        "id": normalized_message_id,
        "type": str(message_type).strip(),
        "position": [float(position_xy[0]), float(position_xy[1])],
    }
    if isinstance(extra_payload, Mapping):
        for key, value in extra_payload.items():
            next_message[str(key)] = value
    retained_messages.append(next_message)
    write_cp_messages(retained_messages, message_path=message_path)


def _destroy_actors_by_role_name(world, role_name: str) -> None:
    normalized_role_name = str(role_name).strip().lower()
    if not normalized_role_name or not hasattr(world, "get_actors"):
        return
    for actor in list(world.get_actors()):
        raw_role_name = str(getattr(actor, "attributes", {}).get("role_name", "")).strip().lower()
        if raw_role_name != normalized_role_name:
            continue
        try:
            actor.destroy()
        except RuntimeError:
            pass


def _resolve_marker_transform(world, carla, marker_name: str):
    marker = _resolve_named_world_object(world, carla, marker_name)
    if marker is None:
        return None
    return _object_transform(marker)


def _spawn_static_vehicle_at_marker(
    *,
    world,
    world_map,
    carla,
    marker_name: str,
    vehicle_blueprint_id: str,
    role_name: str,
    color_rgb: str,
    spawn_z_offset_m: float,
) -> Any | None:
    marker_transform = _resolve_marker_transform(world, carla, marker_name)
    if marker_transform is None:
        print(f"[ROADWAY HAZARD] Marker '{marker_name}' was not found.")
        return None

    blueprint_library = world.get_blueprint_library()
    try:
        blueprint = blueprint_library.find(str(vehicle_blueprint_id).strip())
    except RuntimeError as exc:
        raise RuntimeError(
            f"Obstacle vehicle blueprint '{vehicle_blueprint_id}' was not found."
        ) from exc

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", str(role_name))
    if str(color_rgb).strip() and blueprint.has_attribute("color"):
        blueprint.set_attribute("color", str(color_rgb).strip())

    waypoint_transform = None
    if world_map is not None and hasattr(world_map, "get_waypoint"):
        nearest_waypoint = world_map.get_waypoint(
            marker_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        waypoint_transform = None if nearest_waypoint is None else nearest_waypoint.transform

    spawn_vehicle = None
    for attempt_transform in _spawn_attempt_transforms(
        base_transform=marker_transform,
        waypoint_transform=waypoint_transform,
        carla=carla,
        base_z_offset_m=float(spawn_z_offset_m),
    ):
        spawn_vehicle = world.try_spawn_actor(blueprint, attempt_transform)
        if spawn_vehicle is not None:
            break

    if spawn_vehicle is None:
        print(f"[ROADWAY HAZARD] Failed to spawn vehicle at marker '{marker_name}'.")
        return None

    _configure_static_vehicle(spawn_vehicle, carla)
    transform = spawn_vehicle.get_transform()
    print(
        f"[ROADWAY HAZARD] Spawned '{role_name}' at "
        f"({float(transform.location.x):.3f}, {float(transform.location.y):.3f}, {float(transform.location.z):.3f})."
    )
    return spawn_vehicle


def spawn_obstacles(
    *,
    world,
    world_map,
    carla,
    blueprint_library,
    traffic_manager=None,
    traffic_manager_port: int = 8000,
    scenario_cfg: Mapping[str, object],
    route_summary=None,
    route_points: Sequence[Sequence[float]] | None = None,
) -> List[Any]:
    del route_summary, route_points

    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    marker_names = _lowercase_name_set(
        obstacle_cfg.get("marker_names", None),
        default_values=[f"obstacle{idx}" for idx in range(1, 7)],
    )
    autopilot_marker_names = {
        str(name).strip().lower()
        for name in _lowercase_name_set(
            obstacle_cfg.get("autopilot_marker_names", None),
            default_values=["obstacle6"],
        )
    }
    vehicle_blueprint_id = str(obstacle_cfg.get("vehicle_blueprint", "vehicle.tesla.model3")).strip()
    color_rgb = str(obstacle_cfg.get("color_rgb", "90,90,90")).strip()
    spawn_z_offset_m = float(obstacle_cfg.get("spawn_z_offset_m", 0.05))

    spawned_vehicles: List[Any] = []
    for marker_name in marker_names:
        marker = _find_environment_marker_by_name(world, carla, marker_name)
        if marker is None:
            print(f"[ROADWAY HAZARD] Marker '{marker_name}' was not found.")
            continue

        try:
            blueprint = blueprint_library.find(vehicle_blueprint_id)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Obstacle vehicle blueprint '{vehicle_blueprint_id}' was not found."
            ) from exc

        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", str(marker_name))
        if color_rgb and blueprint.has_attribute("color"):
            blueprint.set_attribute("color", color_rgb)

        nearest_waypoint = world_map.get_waypoint(
            marker.transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        waypoint_transform = None if nearest_waypoint is None else nearest_waypoint.transform
        spawn_vehicle = None
        for attempt_transform in _spawn_attempt_transforms(
            base_transform=marker.transform,
            waypoint_transform=waypoint_transform,
            carla=carla,
            base_z_offset_m=spawn_z_offset_m,
        ):
            spawn_vehicle = world.try_spawn_actor(blueprint, attempt_transform)
            if spawn_vehicle is not None:
                break

        if spawn_vehicle is None:
            print(f"[ROADWAY HAZARD] Failed to spawn vehicle at marker '{marker_name}'.")
            continue

        marker_name_lower = str(marker_name).strip().lower()
        if marker_name_lower in autopilot_marker_names:
            _configure_autopilot_vehicle(
                spawn_vehicle,
                traffic_manager=traffic_manager,
                traffic_manager_port=int(traffic_manager_port),
            )
        else:
            _configure_static_vehicle(spawn_vehicle, carla)

        spawned_vehicles.append(spawn_vehicle)
        transform = spawn_vehicle.get_transform()
        print(
            f"[ROADWAY HAZARD] Spawned '{marker_name}' at "
            f"({float(transform.location.x):.3f}, {float(transform.location.y):.3f}, {float(transform.location.z):.3f}) "
            f"autopilot={'on' if marker_name_lower in autopilot_marker_names else 'off'}"
        )

    return spawned_vehicles


def initialize_runtime(
    *,
    scenario_cfg: Mapping[str, object],
    world=None,
    world_map=None,
    carla=None,
    wall_time_s: float | None = None,
    **_,
) -> Dict[str, object]:
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    hazard_name = str(runtime_cfg.get("hazard_name", "")).strip()

    if hazard_name:
        cp_message_path = str(runtime_cfg.get("cp_message_path", CP_MESSAGE_PATH))
        cp_message_id = str(runtime_cfg.get("cp_message_id", "roadway_hazard_message")).strip()
        if cp_message_id:
            remove_cp_messages_by_id([cp_message_id], message_path=cp_message_path)
        trigger_distance_m = max(
            0.0,
            float(runtime_cfg.get("cooperative_message_trigger_distance_m", 20.0)),
        )
        hazard_vehicle_id = str(
            runtime_cfg.get(
                "hazard_vehicle_id",
                hazard_name,
            )
        ).strip().lower()
        hazard_position_xy = None
        hazard_transform = None
        hazard_marker = None
        if world is not None and carla is not None:
            hazard_marker = _resolve_named_world_object(world, carla, hazard_name)
            hazard_transform = _object_transform(hazard_marker)
            hazard_position_xy = _transform_xy(hazard_transform)

        spawned_actor_id = None
        if (
            world is not None
            and world_map is not None
            and carla is not None
            and hazard_name
            and hazard_transform is not None
        ):
            _destroy_actors_by_role_name(world, role_name=hazard_vehicle_id)
            spawned_actor = _spawn_static_vehicle_at_marker(
                world=world,
                world_map=world_map,
                carla=carla,
                marker_name=hazard_name,
                vehicle_blueprint_id=str(
                    obstacle_cfg.get("vehicle_blueprint", "vehicle.tesla.model3")
                ).strip(),
                role_name=hazard_vehicle_id,
                color_rgb=str(obstacle_cfg.get("color_rgb", "90,90,90")).strip(),
                spawn_z_offset_m=float(obstacle_cfg.get("spawn_z_offset_m", 0.05)),
            )
            if spawned_actor is not None:
                spawned_actor_id = int(getattr(spawned_actor, "id", 0) or 0)

        return {
            "mode": "hazard_cp",
            "hazard_name": str(hazard_name),
            "hazard_vehicle_id": str(hazard_vehicle_id),
            "hazard_position_xy": hazard_position_xy,
            "cooperative_message_trigger_distance_m": float(trigger_distance_m),
            "cp_message_path": str(cp_message_path),
            "cp_message_id": str(cp_message_id),
            "cp_message_inserted": False,
            "hazard_revealed": False,
            "last_hazard_distance_m": None,
            "hazard_actor_id": spawned_actor_id,
            "last_cp_message_time_s": None if wall_time_s is None else float(_resolved_wall_time_s(wall_time_s)),
        }

    hidden_obstacle_id = str(runtime_cfg.get("hidden_obstacle_id", "obstacle4")).strip().lower()
    relay_obstacle_id = str(runtime_cfg.get("relay_obstacle_id", "obstacle6")).strip().lower()
    reveal_distance_m = float(runtime_cfg.get("reveal_distance_m", 20.0))
    return {
        "mode": "legacy_hidden_obstacle",
        "hidden_obstacle_id": hidden_obstacle_id,
        "relay_obstacle_id": relay_obstacle_id,
        "reveal_distance_m": reveal_distance_m,
        "hidden_obstacle_revealed": False,
    }


def _snapshot_by_id(
    object_snapshots: Sequence[Mapping[str, object]],
) -> Dict[str, Mapping[str, object]]:
    snapshots_by_id: Dict[str, Mapping[str, object]] = {}
    for snapshot in object_snapshots:
        obstacle_id = str(snapshot.get("vehicle_id", "")).strip().lower()
        if obstacle_id:
            snapshots_by_id[obstacle_id] = snapshot
    return snapshots_by_id


def _distance_between_snapshots(
    first_snapshot: Mapping[str, object] | None,
    second_snapshot: Mapping[str, object] | None,
) -> float | None:
    if first_snapshot is None or second_snapshot is None:
        return None
    return float(
        math.hypot(
            float(first_snapshot.get("x", 0.0)) - float(second_snapshot.get("x", 0.0)),
            float(first_snapshot.get("y", 0.0)) - float(second_snapshot.get("y", 0.0)),
        )
    )


def maybe_replan_global_route(
    *,
    runtime_state,
    world,
    carla,
    ego_transform,
    sim_time_s: float,
    wall_time_s: float | None = None,
    **_,
) -> Tuple[object | None, List[List[float]] | None, Dict[str, object]]:
    del sim_time_s
    next_runtime_state = dict(runtime_state or {})
    if str(next_runtime_state.get("mode", "")).strip().lower() != "hazard_cp":
        return None, None, next_runtime_state

    if bool(next_runtime_state.get("cp_message_inserted", False)):
        return None, None, next_runtime_state

    hazard_position_xy = next_runtime_state.get("hazard_position_xy", None)
    if not isinstance(hazard_position_xy, Sequence) or len(hazard_position_xy) < 2:
        hazard_name = str(next_runtime_state.get("hazard_name", "")).strip()
        hazard_marker = _resolve_named_world_object(world, carla, hazard_name)
        hazard_transform = _object_transform(hazard_marker)
        hazard_position_xy = _transform_xy(hazard_transform)
        if isinstance(hazard_position_xy, Sequence) and len(hazard_position_xy) >= 2:
            next_runtime_state["hazard_position_xy"] = [
                float(hazard_position_xy[0]),
                float(hazard_position_xy[1]),
            ]
    ego_position_xy = _transform_xy(ego_transform)
    hazard_distance_m = _distance_between_points(ego_position_xy, hazard_position_xy)
    next_runtime_state["last_hazard_distance_m"] = (
        None if hazard_distance_m is None else float(hazard_distance_m)
    )
    if hazard_distance_m is None:
        return None, None, next_runtime_state
    trigger_distance_m = max(
        0.0,
        float(next_runtime_state.get("cooperative_message_trigger_distance_m", 20.0)),
    )
    if float(hazard_distance_m) > float(trigger_distance_m):
        return None, None, next_runtime_state

    _append_cp_message(
        message_path=str(next_runtime_state.get("cp_message_path", CP_MESSAGE_PATH)),
        message_id=str(next_runtime_state.get("cp_message_id", "roadway_hazard_message")),
        message_type="hazard",
        position_xy=[
            float(hazard_position_xy[0]),
            float(hazard_position_xy[1]),
        ],
        extra_payload={
            "hazard_vehicle_id": str(next_runtime_state.get("hazard_vehicle_id", "hazard")),
        },
    )
    next_runtime_state["cp_message_inserted"] = True
    next_runtime_state["hazard_revealed"] = True
    next_runtime_state["last_cp_message_time_s"] = float(_resolved_wall_time_s(wall_time_s))
    print(
        "[ROADWAY HAZARD] Inserted hazard cooperative message into cp_message.json "
        f"at ({float(hazard_position_xy[0]):.3f}, {float(hazard_position_xy[1]):.3f})."
    )
    return None, None, next_runtime_state


def filter_dynamic_obstacle_snapshots(
    *,
    runtime_state,
    object_snapshots: Sequence[Mapping[str, object]],
    **_,
) -> Tuple[List[dict], Dict[str, object]]:
    next_runtime_state = dict(runtime_state or {})
    if str(next_runtime_state.get("mode", "")).strip().lower() == "hazard_cp":
        hazard_vehicle_id = str(next_runtime_state.get("hazard_vehicle_id", "hazard")).strip().lower()
        hazard_revealed = bool(next_runtime_state.get("hazard_revealed", False))
        filtered_snapshots: List[dict] = []
        for snapshot in object_snapshots:
            snapshot_id = str(snapshot.get("vehicle_id", "")).strip().lower()
            if snapshot_id == hazard_vehicle_id and not hazard_revealed:
                continue
            filtered_snapshots.append(dict(snapshot))
        return filtered_snapshots, next_runtime_state

    hidden_obstacle_id = str(next_runtime_state.get("hidden_obstacle_id", "obstacle4")).strip().lower()
    relay_obstacle_id = str(next_runtime_state.get("relay_obstacle_id", "obstacle6")).strip().lower()
    reveal_distance_m = float(next_runtime_state.get("reveal_distance_m", 20.0))
    hidden_obstacle_revealed = bool(next_runtime_state.get("hidden_obstacle_revealed", False))

    snapshots_by_id = _snapshot_by_id(object_snapshots)
    hidden_snapshot = snapshots_by_id.get(hidden_obstacle_id, None)
    relay_snapshot = snapshots_by_id.get(relay_obstacle_id, None)
    obstacle_distance_m = _distance_between_snapshots(hidden_snapshot, relay_snapshot)

    if (
        not hidden_obstacle_revealed
        and obstacle_distance_m is not None
        and float(obstacle_distance_m) < float(reveal_distance_m)
    ):
        hidden_obstacle_revealed = True
        print("[ROADWAY HAZARD] Cooperative message received for obstacle4; enabling obstacle field.")

    next_runtime_state["hidden_obstacle_revealed"] = bool(hidden_obstacle_revealed)

    filtered_snapshots: List[dict] = []
    for snapshot in object_snapshots:
        snapshot_id = str(snapshot.get("vehicle_id", "")).strip().lower()
        if snapshot_id == hidden_obstacle_id and not hidden_obstacle_revealed:
            continue
        filtered_snapshots.append(dict(snapshot))

    return filtered_snapshots, next_runtime_state
