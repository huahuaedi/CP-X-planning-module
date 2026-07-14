"""
Scenario obstacle spawner for actors placed by raw world coordinates
instead of named EnvironmentObject markers.

Used by scenarios imported from external benchmarks (e.g. MDrive) that
ship absolute (x, y, z, yaw) placements rather than map-baked markers.

Also publishes a distance-triggered cooperative-perception (CP) "hazard"
message per static actor once the ego vehicle gets within range, mirroring
`carla_scenario/roadway_hazard/scenario.py`'s pattern so imported
blocked-lane-style scenarios feed the same lane-closure/reroute pathway
(`behavior_planner/reroute.py`) as the hand-authored ones, instead of the
ego only reacting once its own simulated perception detects the obstacle.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from behavior_planner.reroute import (
    CP_MESSAGE_PATH,
    ensure_cp_message_file_exists,
    load_cp_messages,
    write_cp_messages,
)


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


def spawn_obstacles(
    *,
    world,
    world_map,
    carla,
    blueprint_library,
    scenario_cfg: Mapping[str, object],
    **_ignored: object,
) -> List[Any]:
    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    static_actors_cfg = obstacle_cfg.get("static_actors", [])
    if not isinstance(static_actors_cfg, Sequence) or isinstance(static_actors_cfg, (str, bytes)):
        return []

    spawned_vehicles: List[Any] = []
    for index, actor_cfg_raw in enumerate(static_actors_cfg, start=1):
        actor_cfg = dict(actor_cfg_raw or {})
        blueprint_id = str(actor_cfg.get("blueprint", "vehicle.tesla.cybertruck")).strip()
        role_name = str(actor_cfg.get("role_name", f"static_actor_{index}")).strip() or f"static_actor_{index}"
        x_m = float(actor_cfg.get("x", 0.0))
        y_m = float(actor_cfg.get("y", 0.0))
        z_m = float(actor_cfg.get("z", 0.0))
        yaw_deg = float(actor_cfg.get("yaw", 0.0))
        base_z_offset_m = float(actor_cfg.get("spawn_z_offset_m", 0.05))
        color_rgb = str(actor_cfg.get("color_rgb", "")).strip()

        try:
            blueprint = blueprint_library.find(blueprint_id)
        except RuntimeError as exc:
            raise RuntimeError(f"Static actor blueprint '{blueprint_id}' was not found.") from exc
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)
        if color_rgb and blueprint.has_attribute("color"):
            blueprint.set_attribute("color", color_rgb)

        base_transform = carla.Transform(
            carla.Location(x=x_m, y=y_m, z=z_m),
            carla.Rotation(yaw=yaw_deg),
        )
        nearest_waypoint = world_map.get_waypoint(
            base_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        waypoint_transform = None if nearest_waypoint is None else nearest_waypoint.transform

        spawn_vehicle = None
        for attempt_transform in _spawn_attempt_transforms(
            base_transform=base_transform,
            waypoint_transform=waypoint_transform,
            carla=carla,
            base_z_offset_m=base_z_offset_m,
        ):
            spawn_vehicle = world.try_spawn_actor(blueprint, attempt_transform)
            if spawn_vehicle is not None:
                break

        if spawn_vehicle is None:
            print(f"[COORDINATE OBSTACLES] Failed to spawn actor '{role_name}' at ({x_m:.3f}, {y_m:.3f}).")
            continue

        _configure_static_vehicle(spawn_vehicle, carla)
        spawned_vehicles.append(spawn_vehicle)
        # Do not read spawn_vehicle.get_transform() here: in synchronous
        # mode a freshly spawned actor's transform is not guaranteed to be
        # populated client-side until after the next world.tick(), so an
        # immediate read commonly reports a stale (0, 0, 0). Log the
        # coordinates the actor was actually spawned at instead.
        print(
            f"[COORDINATE OBSTACLES] Spawned static actor '{role_name}' at "
            f"({x_m:.3f}, {y_m:.3f}, {z_m:.3f}) yaw={yaw_deg:.3f}"
        )

    return spawned_vehicles


def _append_hazard_cp_message(
    *,
    message_path: str,
    message_id: str,
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
    next_message: Dict[str, object] = {
        "id": normalized_message_id,
        "type": "hazard",
        "position": [float(position_xy[0]), float(position_xy[1])],
    }
    if isinstance(extra_payload, Mapping):
        for key, value in extra_payload.items():
            next_message[str(key)] = value
    retained_messages.append(next_message)
    write_cp_messages(retained_messages, message_path=message_path)


def maybe_replan_global_route(
    *,
    scenario_cfg: Mapping[str, object],
    runtime_state: object = None,
    ego_transform=None,
    **_ignored: object,
):
    """Publish a `hazard`-type CP message for each static actor once the ego
    vehicle comes within `cooperative_message_trigger_distance_m` of it, so
    `behavior_planner/reroute.py`'s lane-closure pathway can react to
    imported obstacles the same way it does for hand-authored ones.

    Never returns a route summary/points itself -- rerouting decisions stay
    with the behavior planner's own consumption of the CP message.
    """

    state: Dict[str, object] = dict(runtime_state) if isinstance(runtime_state, Mapping) else {}
    inserted_ids = set(state.get("coordinate_hazard_inserted_ids", []) or [])

    if ego_transform is None:
        return None, None, state

    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    static_actors_cfg = obstacle_cfg.get("static_actors", [])
    if not isinstance(static_actors_cfg, Sequence) or isinstance(static_actors_cfg, (str, bytes)):
        return None, None, state
    trigger_distance_m = max(
        0.0,
        float(obstacle_cfg.get("cooperative_message_trigger_distance_m", 20.0)),
    )
    message_path = str(obstacle_cfg.get("cp_message_path", CP_MESSAGE_PATH))

    ego_x_m = float(ego_transform.location.x)
    ego_y_m = float(ego_transform.location.y)

    for index, actor_cfg_raw in enumerate(static_actors_cfg, start=1):
        actor_cfg = dict(actor_cfg_raw or {})
        role_name = str(actor_cfg.get("role_name", f"static_actor_{index}")).strip() or f"static_actor_{index}"
        if role_name in inserted_ids:
            continue
        x_m = float(actor_cfg.get("x", 0.0))
        y_m = float(actor_cfg.get("y", 0.0))
        distance_m = math.hypot(x_m - ego_x_m, y_m - ego_y_m)
        if distance_m > trigger_distance_m:
            continue

        _append_hazard_cp_message(
            message_path=message_path,
            message_id=role_name,
            position_xy=[x_m, y_m],
            extra_payload={"hazard_vehicle_id": role_name},
        )
        inserted_ids.add(role_name)
        print(
            f"[COORDINATE OBSTACLES] Inserted hazard cooperative message for '{role_name}' "
            f"at ({x_m:.3f}, {y_m:.3f}) -- ego is {distance_m:.1f}m away."
        )

    state["coordinate_hazard_inserted_ids"] = list(inserted_ids)
    return None, None, state
