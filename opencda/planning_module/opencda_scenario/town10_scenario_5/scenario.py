"""
Scenario-local logic for town10_scenario_5.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from opencda_scenario import world_messages
from opencda_scenario.all_usecase_scenario import scenario as base


SCENARIO_NAME = "town10_scenario_5"
DEFAULT_CP_MESSAGE_PATH = base.DEFAULT_CP_MESSAGE_PATH


def _info(message: str) -> None:
    print(f"[TOWN10 SCENARIO 5] {message}")


def _warning(message: str) -> None:
    print(f"[TOWN10 SCENARIO 5] Warning: {message}")


def _copy_vru_state_entries(vru_states: Sequence[Mapping[str, object]] | None) -> List[Dict[str, object]]:
    return [dict(entry) for entry in list(vru_states or [])]


def _copy_marker_entries(markers: Sequence[Mapping[str, object]] | None) -> List[Dict[str, object]]:
    return [dict(entry) for entry in list(markers or [])]


def _vehicle_mode(runtime_state: Mapping[str, object], marker_index: object) -> str:
    marker_vehicle_mode = str(
        runtime_state.get("marker_vehicle_mode", "autopilot")
    ).strip().lower()
    try:
        normalized_index = int(marker_index)
    except Exception:
        return "static"
    if int(normalized_index) <= 0:
        return "static"
    if marker_vehicle_mode in {"autopilot", "autopilot_all", "moving"}:
        return "autopilot"
    if marker_vehicle_mode in {"legacy", "mixed"}:
        if int(normalized_index) % 2 == 1:
            return "autopilot"
        if int(normalized_index) > 50:
            return "delayed_autopilot"
        return "static"
    if marker_vehicle_mode in {"delayed", "delayed_autopilot"}:
        return "delayed_autopilot"
    if marker_vehicle_mode in {"static", "parked"}:
        return "static"
    if int(normalized_index) % 2 == 1:
        return "autopilot"
    if int(normalized_index) > 50:
        return "delayed_autopilot"
    return "static"


def _destroy_existing_scenario_actors(
    world,
    *,
    vehicle_role_names: Sequence[str],
    hazard_role_names: Sequence[str],
    vru_role_prefix: str,
) -> None:
    normalized_vehicle_roles = {
        str(role_name).strip().lower()
        for role_name in list(vehicle_role_names or [])
        if str(role_name).strip()
    }
    normalized_hazard_roles = {
        str(role_name).strip().lower()
        for role_name in list(hazard_role_names or [])
        if str(role_name).strip()
    }
    normalized_vru_prefix = str(vru_role_prefix).strip().lower()
    for actor in base._iter_world_actors(world):
        raw_attributes = getattr(actor, "attributes", {}) or {}
        role_name = str(raw_attributes.get("role_name", "")).strip().lower()
        if role_name in normalized_vehicle_roles:
            base._destroy_actor(actor)
            continue
        if role_name in normalized_hazard_roles:
            base._destroy_actor(actor)
            continue
        if normalized_vru_prefix and role_name.startswith(normalized_vru_prefix):
            base._destroy_actor(actor)


def _build_vru_states(
    *,
    vru_markers: Sequence[Mapping[str, object]],
    vru_role_prefix: str,
) -> List[Dict[str, object]]:
    marker_by_index = {
        int(marker.get("index")): dict(marker)
        for marker in list(vru_markers or [])
        if marker.get("index", None) is not None
    }
    vru_states: List[Dict[str, object]] = []
    for marker_index in sorted(marker_by_index):
        if int(marker_index) % 2 == 0:
            continue
        goal_marker = marker_by_index.get(int(marker_index) + 1, None)
        if not isinstance(goal_marker, Mapping):
            _warning(
                f"skipping VRU marker vru_{int(marker_index)} because vru_{int(marker_index) + 1} was not found."
            )
            continue
        start_marker = dict(marker_by_index[int(marker_index)])
        vru_states.append(
            {
                "start_index": int(marker_index),
                "goal_index": int(marker_index) + 1,
                "start_marker": start_marker,
                "goal_marker": dict(goal_marker),
                "role_name": f"{str(vru_role_prefix).strip()}{int(marker_index)}",
                "actor_id": None,
                "movement_started": False,
                "reached_goal": False,
            }
        )
    return vru_states


def _apply_static_walker_control(carla, walker_actor: Any) -> None:
    if walker_actor is None or not hasattr(carla, "WalkerControl") or not hasattr(carla, "Vector3D"):
        return
    try:
        control = carla.WalkerControl()
        control.direction = carla.Vector3D(x=0.0, y=0.0, z=0.0)
        control.speed = 0.0
        control.jump = False
        walker_actor.apply_control(control)
    except Exception:
        pass


def _spawn_configured_vrus(
    *,
    runtime_state: Dict[str, object],
    world,
    carla,
) -> None:
    vru_states = _copy_vru_state_entries(runtime_state.get("vru_states", []))
    if len(vru_states) == 0:
        runtime_state["vru_states"] = vru_states
        return

    for vru_state in vru_states:
        if vru_state.get("actor_id", None) is not None:
            continue
        walker_actor, _controller_actor = base._spawn_vru(
            world=world,
            carla=carla,
            start_transform=vru_state.get("start_marker", {}).get("transform", None),
            goal_transform=vru_state.get("goal_marker", {}).get("transform", None),
            blueprint_filter=str(runtime_state.get("vru_blueprint_filter", "walker.pedestrian.*")),
            role_name=str(vru_state.get("role_name", "town10_scenario_5_vru")),
            speed_mps=float(runtime_state.get("vru_speed_mps", 1.2)),
            spawn_z_offset_m=float(runtime_state.get("vru_spawn_z_offset_m", 0.0)),
        )
        if walker_actor is None:
            _warning(f"failed to spawn VRU for marker index {vru_state.get('start_index', '?')}.")
            continue
        vru_state["actor_id"] = int(getattr(walker_actor, "id", 0) or 0)
        _apply_static_walker_control(carla, walker_actor)

    runtime_state["vru_states"] = vru_states
    _info(f"spawned {sum(1 for entry in vru_states if entry.get('actor_id', None) is not None)} VRU(s).")


def _apply_vru_goal_motion(
    *,
    runtime_state: Dict[str, object],
    vru_state: Dict[str, object],
    carla,
    walker_actor: Any,
) -> None:
    goal_marker = vru_state.get("goal_marker", None)
    if not isinstance(goal_marker, Mapping):
        return
    goal_transform = goal_marker.get("transform", None)
    goal_location = getattr(goal_transform, "location", None)
    actor_transform = base._object_transform(walker_actor)
    actor_location = getattr(actor_transform, "location", None)
    if goal_location is None or actor_location is None:
        return

    dx = float(goal_location.x) - float(actor_location.x)
    dy = float(goal_location.y) - float(actor_location.y)
    distance_m = (float(dx) ** 2 + float(dy) ** 2) ** 0.5
    arrival_threshold_m = max(0.1, float(runtime_state.get("vru_arrival_threshold_m", 0.5)))
    speed_mps = 0.0
    direction = carla.Vector3D(x=0.0, y=0.0, z=0.0)
    if distance_m > arrival_threshold_m:
        speed_mps = max(0.0, float(runtime_state.get("vru_speed_mps", 1.2)))
        direction = carla.Vector3D(x=float(dx / distance_m), y=float(dy / distance_m), z=0.0)
    else:
        vru_state["reached_goal"] = True

    try:
        control = carla.WalkerControl()
        control.direction = direction
        control.speed = float(speed_mps)
        control.jump = False
        walker_actor.apply_control(control)
    except Exception:
        pass


def _maybe_spawn_marker_vehicles(
    *,
    runtime_state: Dict[str, object],
    world,
    carla,
) -> None:
    if bool(runtime_state.get("manual_vehicles_spawned", False)):
        return

    actor_ids: List[int] = []
    delayed_actor_ids: Dict[int, int] = {}
    for marker in list(runtime_state.get("vehicle_markers", []) or []):
        marker_name = str(marker.get("name", "")).strip()
        marker_index = marker.get("index", None)
        vehicle_mode = _vehicle_mode(runtime_state, marker_index)
        spawned_actor = base._spawn_vehicle_at_marker(
            world=world,
            carla=carla,
            marker=marker,
            role_name=marker_name,
            blueprint_id=str(runtime_state.get("vehicle_blueprint", "vehicle.tesla.model3")),
            color_rgb=str(runtime_state.get("vehicle_color_rgb", "90,90,90")),
            spawn_z_offset_m=float(runtime_state.get("vehicle_spawn_z_offset_m", 0.05)),
            autopilot_enabled=str(vehicle_mode) == "autopilot",
            traffic_manager_port=int(runtime_state.get("traffic_manager_port", 8000)),
            prefer_waypoint_transform=True,
        )
        if spawned_actor is None:
            continue
        actor_id = int(getattr(spawned_actor, "id", 0) or 0)
        actor_ids.append(actor_id)
        if str(vehicle_mode) == "delayed_autopilot" and marker_index is not None:
            delayed_actor_ids[int(marker_index)] = int(actor_id)

    runtime_state["manual_vehicles_spawned"] = True
    runtime_state["manual_vehicle_actor_ids"] = actor_ids
    runtime_state["delayed_autopilot_vehicle_actor_ids"] = delayed_actor_ids
    runtime_state["delayed_autopilot_vehicle_activated_indices"] = set()
    _info(f"spawned {len(actor_ids)} marker vehicle(s).")


def _spawn_hazard_marker_vehicles(
    *,
    runtime_state: Dict[str, object],
    world,
    carla,
) -> None:
    if bool(runtime_state.get("hazard_vehicles_spawned", False)):
        return

    actor_ids: List[int] = []
    for marker in list(runtime_state.get("hazard_markers", []) or []):
        marker_name = str(marker.get("name", "")).strip()
        spawned_actor = base._spawn_vehicle_at_marker(
            world=world,
            carla=carla,
            marker=marker,
            role_name=marker_name,
            blueprint_id=str(runtime_state.get("hazard_vehicle_blueprint", "vehicle.tesla.model3")),
            color_rgb=str(runtime_state.get("hazard_vehicle_color_rgb", "40,40,40")),
            spawn_z_offset_m=float(runtime_state.get("hazard_vehicle_spawn_z_offset_m", 0.05)),
            autopilot_enabled=False,
            traffic_manager_port=int(runtime_state.get("traffic_manager_port", 8000)),
            prefer_waypoint_transform=False,
        )
        if spawned_actor is None:
            continue
        actor_ids.append(int(getattr(spawned_actor, "id", 0) or 0))

    runtime_state["hazard_vehicles_spawned"] = True
    runtime_state["hazard_vehicle_actor_ids"] = actor_ids
    _info(f"spawned {len(actor_ids)} hazard vehicle(s).")


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
    trigger_distance_m = float(runtime_state.get("delayed_autopilot_trigger_distance_m", 20.0))
    remaining_actor_ids: Dict[int, int] = {}
    for marker_index, actor_id in delayed_actor_ids.items():
        if int(marker_index) in activated_indices:
            continue
        trigger_marker = base._marker_by_index(
            runtime_state.get("vehicle_markers", []),
            marker_index,
        )
        trigger_distance = base._distance_xy(
            ego_xy,
            trigger_marker.get("position_xy", None) if isinstance(trigger_marker, Mapping) else None,
        )
        if trigger_distance is None or float(trigger_distance) > float(trigger_distance_m):
            remaining_actor_ids[int(marker_index)] = int(actor_id)
            continue

        actor = base._find_actor_by_id(world, actor_id)
        if actor is None:
            continue
        base._configure_autopilot_vehicle(
            actor,
            traffic_manager_port=int(runtime_state.get("traffic_manager_port", 8000)),
            carla=carla,
        )
        activated_indices.add(int(marker_index))
        _info(
            f"enabled autopilot for vehicle_{int(marker_index)} at "
            f"distance {float(trigger_distance):.2f} m."
        )

    runtime_state["delayed_autopilot_vehicle_actor_ids"] = remaining_actor_ids
    runtime_state["delayed_autopilot_vehicle_activated_indices"] = activated_indices


def _maybe_trigger_vru_motion(
    *,
    runtime_state: Dict[str, object],
    world,
    carla,
    ego_xy: Sequence[float] | None,
) -> None:
    vru_states = _copy_vru_state_entries(runtime_state.get("vru_states", []))
    trigger_distance_m = float(runtime_state.get("vru_trigger_distance_m", 20.0))
    for vru_state in vru_states:
        actor_id = vru_state.get("actor_id", None)
        if actor_id is None or bool(vru_state.get("reached_goal", False)):
            continue
        walker_actor = base._find_actor_by_id(world, actor_id)
        if walker_actor is None:
            continue
        if not bool(vru_state.get("movement_started", False)):
            distance_m = base._distance_xy(
                ego_xy,
                vru_state.get("start_marker", {}).get("position_xy", None),
            )
            if distance_m is None or float(distance_m) > float(trigger_distance_m):
                continue
            vru_state["movement_started"] = True
            _info(
                f"triggered VRU movement for {vru_state.get('role_name', 'vru')} "
                f"toward vru_{vru_state.get('goal_index', '?')}."
            )
        _apply_vru_goal_motion(
            runtime_state=runtime_state,
            vru_state=vru_state,
            carla=carla,
            walker_actor=walker_actor,
        )
    runtime_state["vru_states"] = vru_states


def _maybe_register_hazard_lane_closures(
    *,
    runtime_state: Dict[str, object],
    ego_xy: Sequence[float] | None,
    sim_time_s: float,
) -> None:
    hazard_markers = _copy_marker_entries(runtime_state.get("hazard_markers", []))
    if len(hazard_markers) == 0:
        runtime_state["hazard_markers"] = hazard_markers
        return

    sent_marker_names = {
        str(marker_name).strip()
        for marker_name in set(runtime_state.get("lane_closure_sent_marker_names", set()))
        if str(marker_name).strip()
    }
    trigger_distance_m = float(runtime_state.get("hazard_trigger_distance_m", 20.0))
    for marker in hazard_markers:
        marker_name = str(marker.get("name", "")).strip()
        if marker_name in sent_marker_names:
            continue
        distance_m = base._distance_xy(ego_xy, marker.get("position_xy", None))
        if distance_m is None or float(distance_m) > float(trigger_distance_m):
            continue

        inserted = base._upsert_cp_item(
            message_path=str(runtime_state.get("cp_message_path", DEFAULT_CP_MESSAGE_PATH)),
            schema_version=int(runtime_state.get("cp_schema_version", 1)),
            list_name="lane_events",
            item=base._lane_event_from_marker(
                str(runtime_state.get("message_prefix", SCENARIO_NAME)),
                marker,
            ),
            timestamp_s=float(sim_time_s),
        )
        sent_marker_names.add(marker_name)
        if inserted:
            _info(f"registered lane_closure for {marker_name}.")

    runtime_state["hazard_markers"] = hazard_markers
    runtime_state["lane_closure_sent_marker_names"] = sent_marker_names


def initialize_runtime(
    *,
    scenario_cfg: Mapping[str, object],
    world,
    world_map=None,
    carla,
    traffic_manager_port: int | None = None,
    **extras,
) -> Dict[str, object]:
    runtime_cfg = dict(scenario_cfg.get("runtime", {}))
    message_prefix = str(runtime_cfg.get("cp_message_prefix", SCENARIO_NAME)).strip() or SCENARIO_NAME
    cp_message_path = str(runtime_cfg.get("cp_message_path", DEFAULT_CP_MESSAGE_PATH)).strip() or DEFAULT_CP_MESSAGE_PATH
    cp_schema_version = int(runtime_cfg.get("cp_message_schema_version", 1))

    if bool(runtime_cfg.get("reset_cp_message_file", True)):
        base._reset_cp_message_file(message_path=cp_message_path, schema_version=cp_schema_version)

    vehicle_prefix = str(runtime_cfg.get("vehicle_marker_prefix", "vehicle_")).strip()
    vru_prefix = str(runtime_cfg.get("vru_marker_prefix", "vru_")).strip()
    hazard_prefix = str(runtime_cfg.get("hazard_marker_prefix", "hazard_")).strip()
    stop_prefix = str(runtime_cfg.get("stop_marker_prefix", "stop_")).strip()
    intersection_prefix = str(runtime_cfg.get("intersection_marker_prefix", "intersection_")).strip()
    vru_role_prefix = str(runtime_cfg.get("vru_role_prefix", f"{SCENARIO_NAME}_vru_")).strip()

    vehicle_markers = base._records_by_indexed_names(
        world=world,
        world_map=world_map,
        carla=carla,
        prefix=vehicle_prefix,
    )
    vru_markers = base._records_by_indexed_names(
        world=world,
        world_map=world_map,
        carla=carla,
        prefix=vru_prefix,
    )
    hazard_markers = base._records_by_indexed_names(
        world=world,
        world_map=world_map,
        carla=carla,
        prefix=hazard_prefix,
    )
    stop_markers = base._records_by_prefix(
        world=world,
        world_map=world_map,
        carla=carla,
        prefix=stop_prefix,
    )
    intersection_markers = base._records_by_prefix(
        world=world,
        world_map=world_map,
        carla=carla,
        prefix=intersection_prefix,
    )
    vru_states = _build_vru_states(vru_markers=vru_markers, vru_role_prefix=vru_role_prefix)

    _destroy_existing_scenario_actors(
        world,
        vehicle_role_names=[str(marker.get("name", "")).strip() for marker in vehicle_markers],
        hazard_role_names=[str(marker.get("name", "")).strip() for marker in hazard_markers],
        vru_role_prefix=vru_role_prefix,
    )

    _info(
        "markers: "
        f"vehicles={len(vehicle_markers)} vrus={len(vru_states)} hazards={len(hazard_markers)} "
        f"stops={len(stop_markers)} intersections={len(intersection_markers)}"
    )

    traffic_manager_cfg = dict(scenario_cfg.get("traffic_manager", {}))
    try:
        resolved_traffic_manager_port = int(
            traffic_manager_port
            if traffic_manager_port is not None
            else traffic_manager_cfg.get("port", 8000)
        )
    except Exception:
        resolved_traffic_manager_port = int(traffic_manager_cfg.get("port", 8000))
    runtime_state = {
        "message_prefix": message_prefix,
        "cp_message_path": cp_message_path,
        "cp_schema_version": cp_schema_version,
        "vehicle_markers": vehicle_markers,
        "vru_states": vru_states,
        "hazard_markers": hazard_markers,
        "stop_markers": stop_markers,
        "intersection_markers": intersection_markers,
        "control_trigger_distance_m": float(runtime_cfg.get("control_trigger_distance_m", 20.0)),
        "delayed_autopilot_trigger_distance_m": float(
            runtime_cfg.get("delayed_autopilot_trigger_distance_m", 20.0)
        ),
        "hazard_trigger_distance_m": float(runtime_cfg.get("hazard_trigger_distance_m", 20.0)),
        "traffic_light_stop_waypoint_match_distance_m": float(
            runtime_cfg.get("traffic_light_stop_waypoint_match_distance_m", 12.0)
        ),
        "traffic_light_actor_position_match_distance_m": float(
            runtime_cfg.get("traffic_light_actor_position_match_distance_m", 40.0)
        ),
        "vru_trigger_distance_m": float(runtime_cfg.get("vru_trigger_distance_m", 20.0)),
        "stop_state": str(runtime_cfg.get("stop_state", "stop")).strip() or "stop",
        "marker_vehicle_mode": str(runtime_cfg.get("marker_vehicle_mode", "autopilot")).strip() or "autopilot",
        "vehicle_blueprint": str(runtime_cfg.get("vehicle_blueprint", "vehicle.tesla.model3")).strip(),
        "vehicle_color_rgb": str(runtime_cfg.get("vehicle_color_rgb", "90,90,90")).strip(),
        "vehicle_spawn_z_offset_m": float(runtime_cfg.get("vehicle_spawn_z_offset_m", 0.05)),
        "hazard_vehicle_blueprint": str(
            runtime_cfg.get("hazard_vehicle_blueprint", "vehicle.tesla.model3")
        ).strip(),
        "hazard_vehicle_color_rgb": str(
            runtime_cfg.get("hazard_vehicle_color_rgb", "40,40,40")
        ).strip(),
        "hazard_vehicle_spawn_z_offset_m": float(
            runtime_cfg.get(
                "hazard_vehicle_spawn_z_offset_m",
                runtime_cfg.get("vehicle_spawn_z_offset_m", 0.05),
            )
        ),
        "traffic_manager_port": int(resolved_traffic_manager_port),
        "vru_blueprint_filter": str(runtime_cfg.get("vru_blueprint_filter", "walker.pedestrian.*")).strip(),
        "vru_speed_mps": float(runtime_cfg.get("vru_speed_mps", 1.2)),
        "vru_spawn_z_offset_m": float(runtime_cfg.get("vru_spawn_z_offset_m", 0.0)),
        "vru_arrival_threshold_m": float(runtime_cfg.get("vru_arrival_threshold_m", 0.5)),
        "stop_messages_sent": set(),
        "triggered_intersection_marker_names": set(),
        "crossed_intersection_marker_names": set(),
        "intersection_signal_states": {},
        "intersection_signal_actor_names": {},
        "lane_closure_sent_marker_names": set(),
        "hazard_vehicles_spawned": False,
        "hazard_vehicle_actor_ids": [],
        "manual_vehicles_spawned": False,
        "manual_vehicle_actor_ids": [],
        "delayed_autopilot_vehicle_actor_ids": {},
        "delayed_autopilot_vehicle_activated_indices": set(),
        "last_runtime_event_time_s": float(time.perf_counter()),
    }
    _maybe_spawn_marker_vehicles(
        runtime_state=runtime_state,
        world=world,
        carla=carla,
    )
    _spawn_hazard_marker_vehicles(
        runtime_state=runtime_state,
        world=world,
        carla=carla,
    )
    _spawn_configured_vrus(
        runtime_state=runtime_state,
        world=world,
        carla=carla,
    )
    return world_messages.initialize_runtime_state(
        runtime_state=runtime_state,
        runtime_cfg=runtime_cfg,
        tracker_cfg=extras.get("tracker_cfg", None),
        obstacle_filter_cfg=extras.get("obstacle_filter_cfg", None),
        prediction_dt_s=extras.get("prediction_dt_s", None),
        prediction_horizon_s=extras.get("prediction_horizon_s", None),
    )


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
    next_state["vru_states"] = _copy_vru_state_entries(next_state.get("vru_states", []))
    next_state["hazard_markers"] = _copy_marker_entries(next_state.get("hazard_markers", []))
    ego_xy = base._transform_xy(ego_transform)

    base._maybe_register_intersection_control(
        runtime_state=next_state,
        world=world,
        ego_transform=ego_transform,
        active_global_route_points=list(active_global_route_points or []),
        sim_time_s=float(sim_time_s),
    )
    base._maybe_register_stop_controls(
        runtime_state=next_state,
        ego_xy=ego_xy,
        sim_time_s=float(sim_time_s),
    )
    _maybe_register_hazard_lane_closures(
        runtime_state=next_state,
        ego_xy=ego_xy,
        sim_time_s=float(sim_time_s),
    )
    _maybe_activate_delayed_autopilot_vehicles(
        runtime_state=next_state,
        world=world,
        carla=carla,
        ego_xy=ego_xy,
    )
    _maybe_trigger_vru_motion(
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
    next_state["vru_states"] = _copy_vru_state_entries(next_state.get("vru_states", []))
    filtered_snapshots = [dict(snapshot) for snapshot in list(object_snapshots or [])]

    for vru_state in next_state["vru_states"]:
        actor_id = vru_state.get("actor_id", None)
        if actor_id is None:
            continue
        walker_actor = base._find_actor_by_id(world, actor_id)
        if walker_actor is None:
            continue
        if bool(vru_state.get("movement_started", False)) and not bool(vru_state.get("reached_goal", False)):
            _apply_vru_goal_motion(
                runtime_state=next_state,
                vru_state=vru_state,
                carla=carla,
                walker_actor=walker_actor,
            )
        vru_snapshot = base._actor_snapshot(
            walker_actor,
            actor_id=str(vru_state.get("role_name", f"{SCENARIO_NAME}_vru")),
        )
        if vru_snapshot is not None:
            filtered_snapshots.append(vru_snapshot)
    return filtered_snapshots, next_state
