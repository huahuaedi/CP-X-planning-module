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


def _configure_autopilot_vehicle(vehicle, carla, traffic_manager_port) -> None:
    try:
        if traffic_manager_port is not None:
            vehicle.set_autopilot(True, int(traffic_manager_port))
        else:
            vehicle.set_autopilot(True)
    except RuntimeError:
        try:
            vehicle.set_autopilot(True)
        except RuntimeError:
            pass


def spawn_obstacles(
    *,
    world,
    world_map,
    carla,
    blueprint_library,
    scenario_cfg: Mapping[str, object],
    traffic_manager_port: int | None = None,
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

    pedestrians_cfg = obstacle_cfg.get("pedestrians", [])
    if isinstance(pedestrians_cfg, Sequence) and not isinstance(pedestrians_cfg, (str, bytes)):
        for index, pedestrian_cfg_raw in enumerate(pedestrians_cfg, start=1):
            pedestrian_cfg = dict(pedestrian_cfg_raw or {})
            role_name = str(pedestrian_cfg.get("role_name", f"pedestrian_{index}")).strip() or f"pedestrian_{index}"
            waypoints = pedestrian_cfg.get("waypoints", [])
            if not isinstance(waypoints, Sequence) or isinstance(waypoints, (str, bytes)) or len(waypoints) == 0:
                continue
            first_waypoint = list(waypoints[0])
            blueprint_id = str(pedestrian_cfg.get("blueprint", "walker.pedestrian.0001")).strip()

            actor = _spawn_pedestrian(
                world=world,
                carla=carla,
                blueprint_library=blueprint_library,
                x_m=float(first_waypoint[0]),
                y_m=float(first_waypoint[1]),
                z_m=float(first_waypoint[2]) if len(first_waypoint) > 2 else 0.0,
                yaw_deg=float(first_waypoint[3]) if len(first_waypoint) > 3 else 0.0,
                blueprint_id=blueprint_id,
                role_name=role_name,
            )
            if actor is None:
                print(f"[COORDINATE OBSTACLES] Failed to spawn pedestrian '{role_name}'.")
                continue
            spawned_vehicles.append(actor)
            print(
                f"[COORDINATE OBSTACLES] Spawned pedestrian '{role_name}' at "
                f"({float(first_waypoint[0]):.3f}, {float(first_waypoint[1]):.3f})"
            )

    npc_vehicles_cfg = obstacle_cfg.get("npc_vehicles", [])
    if isinstance(npc_vehicles_cfg, Sequence) and not isinstance(npc_vehicles_cfg, (str, bytes)):
        for index, npc_cfg_raw in enumerate(npc_vehicles_cfg, start=1):
            npc_cfg = dict(npc_cfg_raw or {})
            role_name = str(npc_cfg.get("role_name", f"npc_vehicle_{index}")).strip() or f"npc_vehicle_{index}"
            waypoints = npc_cfg.get("waypoints", [])
            if not isinstance(waypoints, Sequence) or isinstance(waypoints, (str, bytes)) or len(waypoints) == 0:
                continue
            first_waypoint = list(waypoints[0])
            blueprint_id = str(npc_cfg.get("blueprint", "vehicle.tesla.model3")).strip()
            follow_mode = str(npc_cfg.get("follow_mode", "autopilot")).strip().lower()

            try:
                blueprint = blueprint_library.find(blueprint_id)
            except RuntimeError as exc:
                raise RuntimeError(f"NPC vehicle blueprint '{blueprint_id}' was not found.") from exc
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", role_name)

            base_transform = carla.Transform(
                carla.Location(x=float(first_waypoint[0]), y=float(first_waypoint[1]), z=float(first_waypoint[2]) if len(first_waypoint) > 2 else 0.0),
                carla.Rotation(yaw=float(first_waypoint[3]) if len(first_waypoint) > 3 else 0.0),
            )
            nearest_waypoint = world_map.get_waypoint(
                base_transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            waypoint_transform = None if nearest_waypoint is None else nearest_waypoint.transform

            npc_vehicle = None
            for attempt_transform in _spawn_attempt_transforms(
                base_transform=base_transform,
                waypoint_transform=waypoint_transform,
                carla=carla,
                base_z_offset_m=float(npc_cfg.get("spawn_z_offset_m", 0.3)),
            ):
                npc_vehicle = world.try_spawn_actor(blueprint, attempt_transform)
                if npc_vehicle is not None:
                    break

            if npc_vehicle is None:
                print(f"[COORDINATE OBSTACLES] Failed to spawn NPC vehicle '{role_name}'.")
                continue

            if follow_mode == "autopilot":
                _configure_autopilot_vehicle(npc_vehicle, carla, traffic_manager_port)
            # follow_mode == "literal": leave autopilot off; maybe_replan_global_route()
            # drives it directly toward each of its own waypoints in turn.
            spawned_vehicles.append(npc_vehicle)
            print(
                f"[COORDINATE OBSTACLES] Spawned NPC vehicle '{role_name}' "
                f"(follow_mode={follow_mode}) at ({float(first_waypoint[0]):.3f}, {float(first_waypoint[1]):.3f})"
            )

    return spawned_vehicles


def _spawn_pedestrian(
    *,
    world,
    carla,
    blueprint_library,
    x_m: float,
    y_m: float,
    z_m: float,
    yaw_deg: float,
    blueprint_id: str,
    role_name: str,
):
    try:
        blueprint = blueprint_library.find(blueprint_id)
    except RuntimeError:
        candidates = blueprint_library.filter(blueprint_id)
        blueprint = candidates[0] if len(candidates) > 0 else None
    if blueprint is None:
        raise RuntimeError(f"Pedestrian blueprint '{blueprint_id}' was not found.")
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)

    spawn_actor = None
    for extra_z_m in (0.0, 0.10, 0.20, 0.5):
        transform = carla.Transform(
            carla.Location(x=float(x_m), y=float(y_m), z=float(z_m) + float(extra_z_m)),
            carla.Rotation(yaw=float(yaw_deg)),
        )
        spawn_actor = world.try_spawn_actor(blueprint, transform)
        if spawn_actor is not None:
            break
    return spawn_actor


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


def _walker_control(carla, direction_x: float, direction_y: float, speed_mps: float):
    return carla.WalkerControl(
        direction=carla.Vector3D(x=float(direction_x), y=float(direction_y), z=0.0),
        speed=float(speed_mps),
    )


def _step_pedestrian_along_waypoints(
    *,
    carla,
    actor,
    waypoints: Sequence[Sequence[float]],
    current_index: int,
    arrival_threshold_m: float,
    speed_mps: float,
) -> int:
    """Advance one pedestrian toward its current target waypoint, moving to
    the next waypoint on arrival. Mirrors the single-goal distance/direction
    math in `opencda_scenario/all_usecase_scenario/scenario.py`'s
    `_apply_vru_goal_motion`, extended to step through a full polyline
    instead of stopping at one destination."""

    current_index = max(0, min(int(current_index), len(waypoints) - 1))
    try:
        actor_transform = actor.get_transform()
    except RuntimeError:
        return current_index

    target = waypoints[current_index]
    dx = float(target[0]) - float(actor_transform.location.x)
    dy = float(target[1]) - float(actor_transform.location.y)
    distance_m = math.hypot(dx, dy)

    if distance_m <= float(arrival_threshold_m):
        if current_index < len(waypoints) - 1:
            current_index += 1
            target = waypoints[current_index]
            dx = float(target[0]) - float(actor_transform.location.x)
            dy = float(target[1]) - float(actor_transform.location.y)
            distance_m = math.hypot(dx, dy)
        else:
            try:
                actor.apply_control(_walker_control(carla, 0.0, 0.0, 0.0))
            except RuntimeError:
                pass
            return current_index

    if distance_m > 1e-6:
        try:
            actor.apply_control(_walker_control(carla, dx / distance_m, dy / distance_m, speed_mps))
        except RuntimeError:
            pass
    return current_index


def _find_actor_by_role_name(world, role_name: str):
    for candidate in world.get_actors():
        if str(candidate.attributes.get("role_name", "")) == role_name:
            return candidate
    return None


def _vehicle_control(carla, *, throttle: float, steer: float, brake: float):
    return carla.VehicleControl(
        throttle=float(max(0.0, min(1.0, throttle))),
        steer=float(max(-1.0, min(1.0, steer))),
        brake=float(max(0.0, min(1.0, brake))),
        hand_brake=False,
        reverse=False,
    )


def _step_vehicle_along_waypoints(
    *,
    carla,
    actor,
    waypoints: Sequence[Sequence[float]],
    current_index: int,
    arrival_threshold_m: float,
    target_speed_mps: float,
    steer_gain: float = 1.5,
) -> int:
    """Simple proportional (not pure-pursuit) follower: steer toward the
    heading-error to the current target waypoint, throttle/coast to hold
    target_speed_mps, advance to the next waypoint on arrival, brake to a
    stop after the last one. Lower fidelity than this project's own MPC by
    design -- NPC vehicles are background actors, not the planner under
    test."""

    current_index = max(0, min(int(current_index), len(waypoints) - 1))
    try:
        actor_transform = actor.get_transform()
        actor_velocity = actor.get_velocity()
    except RuntimeError:
        return current_index

    target = waypoints[current_index]
    dx = float(target[0]) - float(actor_transform.location.x)
    dy = float(target[1]) - float(actor_transform.location.y)
    distance_m = math.hypot(dx, dy)

    if distance_m <= float(arrival_threshold_m):
        if current_index < len(waypoints) - 1:
            current_index += 1
            target = waypoints[current_index]
            dx = float(target[0]) - float(actor_transform.location.x)
            dy = float(target[1]) - float(actor_transform.location.y)
            distance_m = math.hypot(dx, dy)
        else:
            try:
                actor.apply_control(_vehicle_control(carla, throttle=0.0, steer=0.0, brake=1.0))
            except RuntimeError:
                pass
            return current_index

    target_heading_rad = math.atan2(dy, dx)
    current_yaw_rad = math.radians(float(actor_transform.rotation.yaw))
    heading_error_rad = math.atan2(
        math.sin(target_heading_rad - current_yaw_rad),
        math.cos(target_heading_rad - current_yaw_rad),
    )
    steer = float(steer_gain) * heading_error_rad / math.pi

    current_speed_mps = math.hypot(float(actor_velocity.x), float(actor_velocity.y))
    throttle = 0.6 if current_speed_mps < float(target_speed_mps) else 0.0

    try:
        actor.apply_control(_vehicle_control(carla, throttle=throttle, steer=steer, brake=0.0))
    except RuntimeError:
        pass
    return current_index


def maybe_replan_global_route(
    *,
    scenario_cfg: Mapping[str, object],
    runtime_state: object = None,
    ego_transform=None,
    world=None,
    carla=None,
    **_ignored: object,
):
    """Per-tick hook: steps every imported pedestrian along its waypoint
    polyline, then publishes a `hazard`-type CP message for each static
    actor once the ego vehicle comes within
    `cooperative_message_trigger_distance_m` of it, so
    `behavior_planner/reroute.py`'s lane-closure pathway can react to
    imported obstacles the same way it does for hand-authored ones.

    Never returns a route summary/points itself -- rerouting decisions stay
    with the behavior planner's own consumption of the CP message.
    """

    state: Dict[str, object] = dict(runtime_state) if isinstance(runtime_state, Mapping) else {}
    inserted_ids = set(state.get("coordinate_hazard_inserted_ids", []) or [])
    pedestrian_actor_cache = dict(state.get("pedestrian_actor_cache", {}) or {})
    pedestrian_progress = dict(state.get("pedestrian_progress", {}) or {})

    obstacle_cfg = dict(scenario_cfg.get("obstacles", {}))
    pedestrians_cfg = obstacle_cfg.get("pedestrians", [])
    if (
        world is not None
        and carla is not None
        and isinstance(pedestrians_cfg, Sequence)
        and not isinstance(pedestrians_cfg, (str, bytes))
    ):
        for index, pedestrian_cfg_raw in enumerate(pedestrians_cfg, start=1):
            pedestrian_cfg = dict(pedestrian_cfg_raw or {})
            role_name = str(pedestrian_cfg.get("role_name", f"pedestrian_{index}")).strip() or f"pedestrian_{index}"
            waypoints = pedestrian_cfg.get("waypoints", [])
            if not isinstance(waypoints, Sequence) or isinstance(waypoints, (str, bytes)) or len(waypoints) == 0:
                continue

            actor = pedestrian_actor_cache.get(role_name)
            if actor is None:
                actor = _find_actor_by_role_name(world, role_name)
                if actor is not None:
                    pedestrian_actor_cache[role_name] = actor
            if actor is None:
                continue

            pedestrian_progress[role_name] = _step_pedestrian_along_waypoints(
                carla=carla,
                actor=actor,
                waypoints=waypoints,
                current_index=int(pedestrian_progress.get(role_name, 0)),
                arrival_threshold_m=float(pedestrian_cfg.get("arrival_threshold_m", 1.0)),
                speed_mps=float(pedestrian_cfg.get("speed_mps", 1.2)),
            )

    state["pedestrian_actor_cache"] = pedestrian_actor_cache
    state["pedestrian_progress"] = pedestrian_progress

    npc_actor_cache = dict(state.get("npc_vehicle_actor_cache", {}) or {})
    npc_progress = dict(state.get("npc_vehicle_progress", {}) or {})
    npc_vehicles_cfg = obstacle_cfg.get("npc_vehicles", [])
    if (
        world is not None
        and carla is not None
        and isinstance(npc_vehicles_cfg, Sequence)
        and not isinstance(npc_vehicles_cfg, (str, bytes))
    ):
        for index, npc_cfg_raw in enumerate(npc_vehicles_cfg, start=1):
            npc_cfg = dict(npc_cfg_raw or {})
            if str(npc_cfg.get("follow_mode", "autopilot")).strip().lower() != "literal":
                continue
            role_name = str(npc_cfg.get("role_name", f"npc_vehicle_{index}")).strip() or f"npc_vehicle_{index}"
            waypoints = npc_cfg.get("waypoints", [])
            if not isinstance(waypoints, Sequence) or isinstance(waypoints, (str, bytes)) or len(waypoints) == 0:
                continue

            actor = npc_actor_cache.get(role_name)
            if actor is None:
                actor = _find_actor_by_role_name(world, role_name)
                if actor is not None:
                    npc_actor_cache[role_name] = actor
            if actor is None:
                continue

            npc_progress[role_name] = _step_vehicle_along_waypoints(
                carla=carla,
                actor=actor,
                waypoints=waypoints,
                current_index=int(npc_progress.get(role_name, 0)),
                arrival_threshold_m=float(npc_cfg.get("arrival_threshold_m", 3.0)),
                target_speed_mps=float(npc_cfg.get("speed_mps", 8.0)),
            )

    state["npc_vehicle_actor_cache"] = npc_actor_cache
    state["npc_vehicle_progress"] = npc_progress

    if ego_transform is None:
        return None, None, state

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
