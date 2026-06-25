"""
Scenario-specific runner wrapper for town10_scenario_4.

This hook runs before the shared planning-module runner builds the initial
global route, so scenario-local world adjustments can happen first.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, List, Mapping, Tuple

from opencda_scenario import runner as base_runner


SCENARIO_NAME = "town10_scenario_4"
SOURCE_DESTINATION_MARKER_NAME = "final_destination"


def _info(message: str) -> None:
    print(f"[TOWN10 SCENARIO 4] {message}")


def _warning(message: str) -> None:
    print(f"[TOWN10 SCENARIO 4] Warning: {message}")


def _iter_environment_objects(world, carla) -> Iterable[Any]:
    if not hasattr(world, "get_environment_objects"):
        return []
    try:
        return list(world.get_environment_objects(carla.CityObjectLabel.Any))
    except Exception:
        return []


def _iter_world_actors(world) -> Iterable[Any]:
    if not hasattr(world, "get_actors"):
        return []
    try:
        return list(world.get_actors())
    except Exception:
        return []


def _best_partial_match(candidates: List[Tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: int(item[0]))[0][1]


def _find_environment_object(world, carla, name: str) -> Any | None:
    requested = str(name).strip().lower()
    if not requested:
        return None

    partial_candidates: List[Tuple[int, Any]] = []
    for env_obj in _iter_environment_objects(world, carla):
        env_name = str(getattr(env_obj, "name", "")).strip().lower()
        if env_name == requested:
            return env_obj
        if requested in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_partial_match(partial_candidates)


def _find_actor(world, name: str) -> Any | None:
    requested = str(name).strip().lower()
    if not requested:
        return None

    partial_candidates: List[Tuple[int, Any]] = []
    for actor in _iter_world_actors(world):
        raw_attributes = getattr(actor, "attributes", {}) or {}
        attr_name = str(raw_attributes.get("name", "")).strip().lower()
        role_name = str(raw_attributes.get("role_name", "")).strip().lower()
        type_id = str(getattr(actor, "type_id", "")).strip().lower()
        if attr_name == requested or role_name == requested or type_id.endswith(requested):
            return actor
        if requested in attr_name:
            partial_candidates.append((len(attr_name), actor))
        if requested in role_name:
            partial_candidates.append((len(role_name), actor))
        if requested in type_id:
            partial_candidates.append((len(type_id), actor))
    return _best_partial_match(partial_candidates)


def _clone_transform(transform: Any, carla):
    if transform is None:
        return None

    location = getattr(transform, "location", None)
    rotation = getattr(transform, "rotation", None)
    if location is None or rotation is None:
        return transform
    if not all(hasattr(carla, attr_name) for attr_name in ("Transform", "Location", "Rotation")):
        return transform

    return carla.Transform(
        carla.Location(
            x=float(getattr(location, "x", 0.0)),
            y=float(getattr(location, "y", 0.0)),
            z=float(getattr(location, "z", 0.0)),
        ),
        carla.Rotation(
            pitch=float(getattr(rotation, "pitch", 0.0)),
            yaw=float(getattr(rotation, "yaw", 0.0)),
            roll=float(getattr(rotation, "roll", 0.0)),
        ),
    )


def _resolve_marker_transform(world, carla, marker_name: str):
    env_obj = _find_environment_object(world, carla, marker_name)
    if env_obj is not None:
        return getattr(env_obj, "transform", None)

    actor = _find_actor(world, marker_name)
    get_transform_fn = getattr(actor, "get_transform", None)
    if actor is not None and callable(get_transform_fn):
        try:
            return get_transform_fn()
        except Exception:
            return None
    return None


def _target_destination_marker_name(scenario_cfg: Mapping[str, object]) -> str:
    anchors_cfg = dict(scenario_cfg.get("anchors", {}))
    configured_name = str(
        anchors_cfg.get("final_destination", SOURCE_DESTINATION_MARKER_NAME)
    ).strip()
    return configured_name or SOURCE_DESTINATION_MARKER_NAME


def _sync_final_destination_marker(
    world,
    carla,
    *,
    target_destination_marker_name: str,
) -> bool:
    normalized_target_name = str(target_destination_marker_name).strip()
    if not normalized_target_name:
        normalized_target_name = SOURCE_DESTINATION_MARKER_NAME
    if normalized_target_name.lower() == SOURCE_DESTINATION_MARKER_NAME.lower():
        return True

    target_transform = _resolve_marker_transform(world, carla, normalized_target_name)
    if target_transform is None:
        _warning(
            f"could not find destination marker '{normalized_target_name}' before route initialization."
        )
        return False

    moved = False
    source_env_obj = _find_environment_object(world, carla, SOURCE_DESTINATION_MARKER_NAME)
    if source_env_obj is not None and hasattr(source_env_obj, "transform"):
        try:
            source_env_obj.transform = _clone_transform(target_transform, carla)
            moved = True
            _info(
                f"aligned marker '{SOURCE_DESTINATION_MARKER_NAME}' to '{normalized_target_name}'."
            )
        except Exception as exc:
            _warning(
                f"failed to align EnvironmentObject '{SOURCE_DESTINATION_MARKER_NAME}': {exc}"
            )

    source_actor = _find_actor(world, SOURCE_DESTINATION_MARKER_NAME)
    set_transform_fn = getattr(source_actor, "set_transform", None)
    if source_actor is not None and callable(set_transform_fn):
        try:
            set_transform_fn(_clone_transform(target_transform, carla))
            moved = True
            _info(
                f"aligned actor '{SOURCE_DESTINATION_MARKER_NAME}' to '{normalized_target_name}'."
            )
        except Exception as exc:
            _warning(f"failed to align actor '{SOURCE_DESTINATION_MARKER_NAME}': {exc}")

    if not moved:
        _warning(
            f"could not move '{SOURCE_DESTINATION_MARKER_NAME}' directly; using "
            f"'{normalized_target_name}' as the route destination anchor."
        )
    return moved


def _prepare_scenario_cfg(scenario_cfg: Mapping[str, object]) -> dict:
    normalized_cfg = deepcopy(dict(scenario_cfg or {}))
    anchors_cfg = dict(normalized_cfg.get("anchors", {}))
    anchors_cfg["final_destination"] = _target_destination_marker_name(normalized_cfg)
    normalized_cfg["anchors"] = anchors_cfg
    return normalized_cfg


def run_loaded_world(client, world, scenario_cfg, carla) -> int:
    normalized_cfg = _prepare_scenario_cfg(scenario_cfg)
    _sync_final_destination_marker(
        world,
        carla,
        target_destination_marker_name=_target_destination_marker_name(normalized_cfg),
    )
    return int(
        base_runner.run_loaded_world(
            client=client,
            world=world,
            scenario_cfg=normalized_cfg,
            carla=carla,
        )
    )
