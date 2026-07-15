"""Config-driven OpenCDA CP scenario helpers.

These wrappers reuse the Town10 scenario runtime but allow scenario YAML files
to define extra CP-relevant vehicles and roadway hazards without requiring
pre-placed CARLA marker actors in the map.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from opencda_scenario.town10_scenario_5 import scenario as base_scenario


DEFAULT_CP_MESSAGE_PATH = base_scenario.DEFAULT_CP_MESSAGE_PATH


def _configured_transform(carla: Any, raw_value: object):
    if isinstance(raw_value, Mapping):
        x_m = float(raw_value.get("x", raw_value.get("x_m", 0.0)))
        y_m = float(raw_value.get("y", raw_value.get("y_m", 0.0)))
        z_m = float(raw_value.get("z", raw_value.get("z_m", 0.6)))
        yaw_deg = float(raw_value.get("yaw", raw_value.get("yaw_deg", 0.0)))
    elif isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        values = list(raw_value)
        if len(values) < 2:
            return None
        x_m = float(values[0])
        y_m = float(values[1])
        z_m = float(values[2]) if len(values) >= 3 else 0.6
        yaw_deg = float(values[3]) if len(values) >= 4 else 0.0
    else:
        return None

    return carla.Transform(
        carla.Location(x=float(x_m), y=float(y_m), z=float(z_m)),
        carla.Rotation(yaw=float(yaw_deg)),
    )


def _marker_from_config(
    *,
    carla: Any,
    prefix: str,
    index: int,
    raw_marker: Mapping[str, object],
) -> Dict[str, object] | None:
    raw_transform = raw_marker.get("transform", raw_marker.get("xyz_yaw", None))
    transform = _configured_transform(carla, raw_transform)
    if transform is None:
        return None
    name = str(raw_marker.get("name", f"{prefix}{int(index)}")).strip()
    location = getattr(transform, "location", None)
    return {
        "name": name,
        "index": int(index),
        "transform": transform,
        "position_xy": [
            float(getattr(location, "x", 0.0)),
            float(getattr(location, "y", 0.0)),
        ],
        "type": str(raw_marker.get("type", "")).strip(),
        "description": str(raw_marker.get("description", "")).strip(),
    }


def _configured_markers(
    *,
    carla: Any,
    prefix: str,
    raw_markers: Sequence[Mapping[str, object]] | None,
) -> list[Dict[str, object]]:
    markers: list[Dict[str, object]] = []
    for fallback_index, raw_marker in enumerate(list(raw_markers or []), start=1):
        if not isinstance(raw_marker, Mapping):
            continue
        try:
            marker_index = int(raw_marker.get("index", fallback_index))
        except Exception:
            marker_index = int(fallback_index)
        marker = _marker_from_config(
            carla=carla,
            prefix=str(prefix),
            index=int(marker_index),
            raw_marker=raw_marker,
        )
        if marker is not None:
            markers.append(marker)
    return markers


def initialize_runtime(
    *,
    scenario_cfg: Mapping[str, object],
    world,
    map_planner=None,
    carla,
    traffic_manager_port: int | None = None,
    **extras,
) -> Dict[str, object]:
    runtime_state = base_scenario.initialize_runtime(
        scenario_cfg=scenario_cfg,
        world=world,
        map_planner=map_planner,
        carla=carla,
        traffic_manager_port=traffic_manager_port,
        **extras,
    )

    cp_scenario_cfg = dict(scenario_cfg.get("cp_scenario", {}) or {})
    configured_vehicle_markers = _configured_markers(
        carla=carla,
        prefix="vehicle_",
        raw_markers=cp_scenario_cfg.get("vehicles", []),
    )
    configured_hazard_markers = _configured_markers(
        carla=carla,
        prefix="hazard_",
        raw_markers=cp_scenario_cfg.get("hazards", []),
    )

    if configured_vehicle_markers:
        runtime_state["vehicle_markers"] = list(runtime_state.get("vehicle_markers", [])) + configured_vehicle_markers
        runtime_state["manual_vehicles_spawned"] = False
        base_scenario._maybe_spawn_marker_vehicles(
            runtime_state=runtime_state,
            world=world,
            carla=carla,
        )

    if configured_hazard_markers:
        runtime_state["hazard_markers"] = list(runtime_state.get("hazard_markers", [])) + configured_hazard_markers
        runtime_state["hazard_vehicles_spawned"] = False
        base_scenario._spawn_hazard_marker_vehicles(
            runtime_state=runtime_state,
            world=world,
            carla=carla,
        )

    print(
        "[CP SCENARIO] configured "
        f"vehicles={len(configured_vehicle_markers)} hazards={len(configured_hazard_markers)}"
    )
    return runtime_state


def maybe_replan_global_route(**kwargs):
    return base_scenario.maybe_replan_global_route(**kwargs)


def filter_dynamic_obstacle_snapshots(**kwargs):
    return base_scenario.filter_dynamic_obstacle_snapshots(**kwargs)
