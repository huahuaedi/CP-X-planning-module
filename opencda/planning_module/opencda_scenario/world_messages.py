"""Scenario-side world-information publishers for cooperative messages."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from utility import Tracker, canonical_lane_id_for_waypoint
from utility.cp_messages import replace_obstacle_messages


def initialize_runtime_state(
    *,
    runtime_state: Mapping[str, object],
    runtime_cfg: Mapping[str, object],
    tracker_cfg: Mapping[str, object] | None = None,
    obstacle_filter_cfg: Mapping[str, object] | None = None,
    prediction_dt_s: float | None = None,
    prediction_horizon_s: float | None = None,
) -> Dict[str, object]:
    next_state = dict(runtime_state or {})
    obstacle_filter_cfg = dict(obstacle_filter_cfg or {})
    configured_tracker_cfg = dict(tracker_cfg or {})
    next_state["obstacle_tracker"] = Tracker(tracker_cfg=configured_tracker_cfg)
    next_state["obstacle_message_distance_m"] = max(
        0.0,
        float(
            runtime_cfg.get(
                "obstacle_message_distance_m",
                obstacle_filter_cfg.get("tracking_distance_m", 30.0),
            )
        ),
    )
    next_state["obstacle_message_max_dynamic_obstacles"] = max(
        0,
        int(
            runtime_cfg.get(
                "obstacle_message_max_dynamic_obstacles",
                obstacle_filter_cfg.get("max_dynamic_obstacles", 64),
            )
        ),
    )
    next_state["obstacle_prediction_dt_s"] = max(
        1.0e-3,
        float(runtime_cfg.get("obstacle_prediction_dt_s", prediction_dt_s or 0.05)),
    )
    next_state["obstacle_prediction_horizon_s"] = max(
        float(next_state["obstacle_prediction_dt_s"]),
        float(
            runtime_cfg.get(
                "obstacle_prediction_horizon_s",
                prediction_horizon_s or 5.0,
            )
        ),
    )
    return next_state


def _iter_world_actors(world) -> Iterable[object]:
    if not hasattr(world, "get_actors"):
        return []
    try:
        return list(world.get_actors())
    except Exception:
        return []


def _actor_role_name(actor: object) -> str:
    raw_attributes = getattr(actor, "attributes", {}) or {}
    return str(raw_attributes.get("role_name", "")).strip()


def _actor_type_name(actor: object) -> str:
    return str(getattr(actor, "type_id", "")).strip().lower()


def _is_vehicle_actor(actor: object) -> bool:
    return _actor_type_name(actor).startswith("vehicle.")


def _is_vru_actor(actor: object) -> bool:
    type_name = _actor_type_name(actor)
    return type_name.startswith("walker.") and "controller.ai.walker" not in type_name


def _object_distance_sq(first_xy: Sequence[float], second_xy: Sequence[float]) -> float:
    dx_m = float(first_xy[0]) - float(second_xy[0])
    dy_m = float(first_xy[1]) - float(second_xy[1])
    return float(dx_m * dx_m + dy_m * dy_m)


def _world_obstacle_actors(
    *,
    world,
    ego_vehicle,
    max_distance_m: float,
) -> List[Tuple[float, object, str]]:
    ego_transform = ego_vehicle.get_transform()
    ego_xy = [float(ego_transform.location.x), float(ego_transform.location.y)]
    max_distance_sq = None if float(max_distance_m) <= 0.0 else float(max_distance_m) * float(max_distance_m)
    ranked_actors: List[Tuple[float, object, str]] = []
    for actor in _iter_world_actors(world):
        actor_type = "vehicle" if _is_vehicle_actor(actor) else ("vru" if _is_vru_actor(actor) else "")
        if not actor_type:
            continue
        if int(getattr(actor, "id", -1)) == int(getattr(ego_vehicle, "id", -2)):
            continue
        get_transform_fn = getattr(actor, "get_transform", None)
        if not callable(get_transform_fn):
            continue
        try:
            transform = get_transform_fn()
        except Exception:
            continue
        location = getattr(transform, "location", None)
        if location is None:
            continue
        distance_sq = _object_distance_sq(
            ego_xy,
            [float(getattr(location, "x", 0.0)), float(getattr(location, "y", 0.0))],
        )
        if max_distance_sq is not None and float(distance_sq) > float(max_distance_sq):
            continue
        ranked_actors.append((float(distance_sq), actor, str(actor_type)))
    ranked_actors.sort(key=lambda item: float(item[0]))
    return ranked_actors


def _actor_bbox_dimensions(actor: object) -> Tuple[float, float, float]:
    bbox = getattr(actor, "bounding_box", None)
    extent = getattr(bbox, "extent", None)
    half_length_m = max(0.05, float(getattr(extent, "x", 2.25)))
    half_width_m = max(0.05, float(getattr(extent, "y", 1.0)))
    half_height_m = max(0.05, float(getattr(extent, "z", 1.0)))
    return (
        float(half_length_m * 2.0),
        float(half_width_m * 2.0),
        float(half_height_m * 2.0),
    )


def _road_lane_ids(
    *,
    world_map,
    carla,
    x_m: float,
    y_m: float,
    z_m: float,
) -> Tuple[int, int]:
    if world_map is None or not hasattr(world_map, "get_waypoint"):
        return -1, 0
    try:
        waypoint = world_map.get_waypoint(
            carla.Location(x=float(x_m), y=float(y_m), z=float(z_m)),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        return -1, 0
    if waypoint is None:
        return -1, 0
    try:
        road_id = int(getattr(waypoint, "road_id", -1))
    except Exception:
        road_id = -1
    try:
        lane_id = int(canonical_lane_id_for_waypoint(waypoint))
    except Exception:
        lane_id = 0
    return int(road_id), int(lane_id)


def _actor_snapshot(
    *,
    actor,
    actor_type: str,
    world_map,
    carla,
    sumo_bridge=None,
) -> Dict[str, object] | None:
    get_transform_fn = getattr(actor, "get_transform", None)
    if not callable(get_transform_fn):
        return None
    try:
        transform = get_transform_fn()
    except Exception:
        return None
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

    yaw_rad = 0.0 if rotation is None else math.radians(float(getattr(rotation, "yaw", 0.0)))
    tracker_id = str(getattr(actor, "id", ""))
    role_name = _actor_role_name(actor)
    if role_name:
        tracker_id = str(role_name)
    if sumo_bridge is not None and hasattr(sumo_bridge, "get_actor_snapshot_override"):
        snapshot_override = sumo_bridge.get_actor_snapshot_override(int(getattr(actor, "id", 0)))
        if isinstance(snapshot_override, Mapping):
            override_vehicle_id = str(snapshot_override.get("vehicle_id", "")).strip()
            if override_vehicle_id:
                tracker_id = override_vehicle_id
            try:
                speed_mps = float(snapshot_override.get("v", speed_mps))
            except Exception:
                pass
            try:
                yaw_rad = float(snapshot_override.get("psi", yaw_rad))
            except Exception:
                pass

    length_m, width_m, height_m = _actor_bbox_dimensions(actor)
    road_id, lane_id = _road_lane_ids(
        world_map=world_map,
        carla=carla,
        x_m=float(location.x),
        y_m=float(location.y),
        z_m=float(getattr(location, "z", 0.0)),
    )
    return {
        "message_id": int(getattr(actor, "id", 0)),
        "vehicle_id": str(tracker_id),
        "type": str(actor_type),
        "x": float(location.x),
        "y": float(location.y),
        "z": float(getattr(location, "z", 0.0)),
        "v": float(speed_mps),
        "psi": float(yaw_rad),
        "length_m": float(length_m),
        "width_m": float(width_m),
        "height_m": float(height_m),
        "road_id": int(road_id),
        "lane_id": int(lane_id),
    }


def _collect_obstacle_snapshots(
    *,
    world,
    world_map,
    carla,
    ego_vehicle,
    sumo_bridge,
    max_distance_m: float,
    max_snapshots: int,
) -> List[dict]:
    ranked_actors = _world_obstacle_actors(
        world=world,
        ego_vehicle=ego_vehicle,
        max_distance_m=float(max_distance_m),
    )
    snapshots: List[dict] = []
    for _distance_sq, actor, actor_type in ranked_actors:
        snapshot = _actor_snapshot(
            actor=actor,
            actor_type=actor_type,
            world_map=world_map,
            carla=carla,
            sumo_bridge=sumo_bridge,
        )
        if snapshot is None:
            continue
        snapshots.append(snapshot)
        if int(max_snapshots) > 0 and len(snapshots) >= int(max_snapshots):
            break
    return snapshots


def _message_from_snapshot(
    snapshot: Mapping[str, object],
    predictions: Mapping[str, Sequence[Mapping[str, float]]],
) -> Dict[str, object]:
    tracker_id = str(snapshot.get("vehicle_id", ""))
    try:
        message_id = int(snapshot.get("message_id", 0))
    except Exception:
        message_id = 0
    length_m = float(snapshot.get("length_m", 4.5))
    width_m = float(snapshot.get("width_m", 2.0))
    height_m = float(snapshot.get("height_m", 2.0))
    predicted_trajectory = [
        [
            float(item.get("x", 0.0)),
            float(item.get("y", 0.0)),
            float(item.get("v", 0.0)),
            float(item.get("psi", 0.0)),
        ]
        for item in list(predictions.get(tracker_id, []) or [])
        if isinstance(item, Mapping)
    ]
    return {
        "id": int(message_id),
        "type": str(snapshot.get("type", "vehicle")).strip().lower() or "vehicle",
        "shape": [float(length_m), float(width_m)],
        "height_m": float(height_m),
        "z": float(snapshot.get("z", 0.0)),
        "state": [
            float(snapshot.get("x", 0.0)),
            float(snapshot.get("y", 0.0)),
            float(snapshot.get("v", 0.0)),
            float(snapshot.get("psi", 0.0)),
        ],
        "trajectory": predicted_trajectory,
        "road_id": int(snapshot.get("road_id", -1) or -1),
        "lane_id": int(snapshot.get("lane_id", 0) or 0),
    }


def publish_obstacle_messages(
    *,
    runtime_state: Mapping[str, object],
    world,
    world_map,
    carla,
    ego_vehicle,
    sim_time_s: float,
    sumo_bridge=None,
) -> Dict[str, object]:
    next_state = dict(runtime_state or {})
    tracker = next_state.get("obstacle_tracker", None)
    if ego_vehicle is None or tracker is None:
        return next_state

    max_distance_m = float(next_state.get("obstacle_message_distance_m", 30.0))
    max_dynamic_obstacles = int(next_state.get("obstacle_message_max_dynamic_obstacles", 64))
    step_dt_s = float(next_state.get("obstacle_prediction_dt_s", 0.05))
    horizon_s = float(next_state.get("obstacle_prediction_horizon_s", 5.0))
    message_path = str(next_state.get("cp_message_path", "")).strip()
    schema_version = int(next_state.get("cp_schema_version", 1))
    if not message_path:
        return next_state

    dynamic_snapshots = _collect_obstacle_snapshots(
        world=world,
        world_map=world_map,
        carla=carla,
        ego_vehicle=ego_vehicle,
        sumo_bridge=sumo_bridge,
        max_distance_m=float(max_distance_m),
        max_snapshots=int(max_dynamic_obstacles),
    )
    tracker.update(
        obstacle_snapshots=dynamic_snapshots,
        timestamp_s=float(sim_time_s),
    )
    predictions = tracker.predict(
        step_dt_s=float(step_dt_s),
        horizon_s=float(horizon_s),
    )
    obstacle_messages = [
        _message_from_snapshot(snapshot, predictions=predictions)
        for snapshot in dynamic_snapshots
    ]
    replace_obstacle_messages(
        message_path=message_path,
        schema_version=int(schema_version),
        obstacles=obstacle_messages,
        timestamp_s=float(sim_time_s),
    )
    return next_state
