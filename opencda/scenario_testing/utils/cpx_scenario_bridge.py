# -*- coding: utf-8 -*-
"""Bridge that ports `opencda/planning_module` CARLA/SUMO scenarios onto the
native OpenCDA `python opencda.py -t <scenario_name>` entry point.

Background
----------
Before this bridge existed, scenarios living under
``opencda/planning_module/carla_scenario`` and
``opencda/planning_module/opencda_scenario`` only ran through the standalone
``opencda/planning_module/main.py`` entry point, driven by the 9000-line
``planning_runner.py`` tick loop. That loop builds its own MPC/Tracker and
applies ``carla.VehicleControl`` directly, bypassing OpenCDA's own
``VehicleManager`` / ``CPXMPCPlannerBridge`` / ``OpenCDAPlanningAdapter``
pipeline entirely.

This module lets those same scenario definitions run through the *official*
pipeline instead: ``VehicleManager`` (spawned via
``opencda.scenario_testing.utils.sim_api.ScenarioManager``) already has the
custom planner wired in as ``vehicle_base.planner.type: cpx_mpc``, and its
``CPXMPCPlannerBridge`` already derives obstacle/traffic-light awareness from
OpenCDA's own ``PerceptionManager``/``V2XManager`` -- there is no need to
replay the old pipeline's synthetic CP-message publishing for plain obstacle
awareness.

What *is* still scenario-specific and must be preserved is the world
bootstrapping performed by each scenario's own Python module: spawning marker
vehicles/hazards/VRUs, driving a scripted traffic-light state machine, and
authoring proactive ``lane_closure``/``hazard`` CP messages (which are not
derivable from passive perception, since they represent information a
scenario chooses to reveal before it is visible). Every one of the ported
scenarios already exposes that bootstrapping behind a uniform four-function
contract: ``spawn_obstacles``, ``initialize_runtime``,
``maybe_replan_global_route``, ``filter_dynamic_obstacle_snapshots``. This
module reuses those functions, and the ``planning_runner.py``/``utility``
helpers that invoke them, entirely by import -- nothing in
``planning_runner.py``, ``opencda_bridge/cpx_mpc_planner.py``, ``MPC/``,
``behavior_planner/``, or ``Global_Planner/`` is modified.

Known scope reduction versus the standalone pipeline: the two-camera pygame
HUD window and the CSV/PNG MPC-cost debug artifacts that ``planning_runner.py``
draws are not reproduced here (they are a standalone-runner visualization
convenience, not scenario behavior). ``opencda_bridge/debug_viewer.py`` remains
available for HUD-style debugging through the native OpenCDA path if needed.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import carla

from opencda.core.common.cav_world import CavWorld
from opencda.planning_module.opencda_bridge.debug_viewer import OpenCDADebugViewer
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time
import opencda.scenario_testing.utils.sim_api as sim_api


def _spectator_view_mode() -> str:
    """Return the requested OpenCDA spectator camera mode.

    Mirrors ``opencda/scenario_testing/single_intersection_town06_carla.py``'s
    helper of the same purpose so every ``cpx_*`` scenario gets the same
    ``OPENCDA_SPECTATOR_VIEW`` behavior (``planner``/chase vs. ``topdown``).
    """

    mode = str(os.environ.get("OPENCDA_SPECTATOR_VIEW", "planner")).strip().lower()
    if mode in {"planner", "chase", "follow", "third_person"}:
        return "planner"
    if mode in {"topdown", "bird", "birdview", "opencda"}:
        return "topdown"
    return "planner"


def _set_spectator_transform(spectator: Any, ego_vehicle: Any) -> None:
    """Move the CARLA spectator to follow ``ego_vehicle`` in the requested mode."""

    transform = ego_vehicle.get_transform()
    mode = _spectator_view_mode()
    if mode == "topdown":
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=70),
            carla.Rotation(pitch=-90)))
        return

    yaw_rad = math.radians(float(transform.rotation.yaw))
    follow_distance_m = float(os.environ.get("OPENCDA_SPECTATOR_DISTANCE_M", "10.0"))
    follow_height_m = float(os.environ.get("OPENCDA_SPECTATOR_HEIGHT_M", "4.5"))
    location = transform.location + carla.Location(
        x=-follow_distance_m * math.cos(yaw_rad),
        y=-follow_distance_m * math.sin(yaw_rad),
        z=follow_height_m,
    )
    spectator.set_transform(carla.Transform(
        location,
        carla.Rotation(
            pitch=float(os.environ.get("OPENCDA_SPECTATOR_PITCH_DEG", "-15.0")),
            yaw=float(transform.rotation.yaw),
            roll=0.0,
        )
    ))

PLANNING_MODULE_ROOT = Path(__file__).resolve().parents[2] / "planning_module"


def _ensure_planning_module_import_path() -> None:
    """Make the flat, non-package-qualified imports used by
    ``planning_runner.py`` and its sibling modules (``from MPC import MPC``,
    ``from utility import ...``, ``from opencda_scenario import ...``) resolve
    when this bridge is invoked from a native OpenCDA scenario script launched
    from the repository root. Mirrors the identical idiom already used by
    ``opencda_bridge/cpx_mpc_planner.py``'s
    ``_ensure_planning_module_import_path``.
    """

    root = str(PLANNING_MODULE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_planning_module_import_path()

import planning_runner  # noqa: E402  (flat import; requires the sys.path fix above)
from utility.cp_messages import reset_cp_message_payload  # noqa: E402


class LegacyScenarioContext:
    """Bundles the objects needed to call the legacy scenario hook functions."""

    def __init__(
        self,
        *,
        legacy_cfg: dict,
        global_planner: Any,
        spawn_transform: Any,
        destination_transform: Any,
        route_summary: Any,
        route_points: List[List[float]],
    ) -> None:
        self.legacy_cfg = legacy_cfg
        self.global_planner = global_planner
        self.spawn_transform = spawn_transform
        self.destination_transform = destination_transform
        self.route_summary = route_summary
        self.route_points = route_points


def load_legacy_scenario_cfg(loader_name: str, scenario_name: str) -> dict:
    """Load a scenario definition from `carla_scenario/` or `opencda_scenario/`
    using the loader package that already ships with the planning module.
    """

    _ensure_planning_module_import_path()
    if loader_name == "carla_scenario":
        from carla_scenario.loader import load_carla_scenario as _load
    elif loader_name == "opencda_scenario":
        from opencda_scenario.loader import load_carla_scenario as _load
    else:
        raise ValueError(
            f"Unknown legacy scenario loader '{loader_name}'; "
            "expected 'carla_scenario' or 'opencda_scenario'."
        )
    return dict(_load(scenario_name))


def _sync_named_destination_marker(world, legacy_cfg: dict) -> None:
    """Align the generic 'final_destination' marker onto a scenario-specific
    destination marker, when configured. Mirrors
    ``opencda_scenario/town10_scenario_1/runner.py::_sync_final_destination_marker``
    (identical across town10_scenario_1..6 and town6_scenario_1); reused as a
    small self-contained utility here rather than importing a specific
    scenario's runner module, since the behavior is generic.
    """

    anchors_cfg = dict(legacy_cfg.get("anchors", {}))
    target_name = str(anchors_cfg.get("final_destination", "final_destination")).strip()
    source_name = "final_destination"
    if not target_name or target_name.lower() == source_name.lower():
        return

    def _find_environment_object(name: str):
        requested = name.strip().lower()
        if not requested or not hasattr(world, "get_environment_objects"):
            return None
        try:
            for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
                if str(getattr(env_obj, "name", "")).strip().lower() == requested:
                    return env_obj
        except Exception:
            return None
        return None

    def _find_actor(name: str):
        requested = name.strip().lower()
        if not requested:
            return None
        for actor in world.get_actors():
            attributes = getattr(actor, "attributes", {}) or {}
            if str(attributes.get("role_name", "")).strip().lower() == requested:
                return actor
        return None

    target_env_obj = _find_environment_object(target_name)
    target_transform = getattr(target_env_obj, "transform", None)
    if target_transform is None:
        target_actor = _find_actor(target_name)
        get_transform = getattr(target_actor, "get_transform", None)
        target_transform = get_transform() if callable(get_transform) else None
    if target_transform is None:
        print(
            f"[CPX SCENARIO BRIDGE] Could not resolve destination marker "
            f"'{target_name}' before route initialization."
        )
        return

    source_actor = _find_actor(source_name)
    set_transform = getattr(source_actor, "set_transform", None)
    if source_actor is not None and callable(set_transform):
        try:
            set_transform(target_transform)
            print(
                f"[CPX SCENARIO BRIDGE] Aligned actor '{source_name}' to "
                f"'{target_name}'."
            )
        except Exception as exc:
            print(
                f"[CPX SCENARIO BRIDGE] Failed to align actor '{source_name}': {exc}"
            )


def build_legacy_context(legacy_cfg: dict, world, carla_module) -> LegacyScenarioContext:
    """Resolve the scenario's map/route context using the same Global_Planner
    utility and anchor-resolution helpers the standalone pipeline uses,
    imported unchanged from ``planning_runner``.
    """

    _sync_named_destination_marker(world, legacy_cfg)

    planning_cfg = dict(legacy_cfg.get("planning", {}))
    sumo_cfg = dict(legacy_cfg.get("sumo", {}))
    anchors_cfg = dict(legacy_cfg.get("anchors", {}))
    world_map = world.get_map()

    planner_selection = planning_runner.create_global_planner_backend(
        planning_cfg=planning_cfg,
        scenario_cfg=legacy_cfg,
        sumo_cfg=sumo_cfg,
        world_map=world_map,
        carla=carla_module,
        project_root=planning_runner.PROJECT_ROOT,
        resolve_xodr_path_fn=planning_runner.resolve_xodr_path,
    )
    global_planner = planner_selection.planner

    spawn_anchor, destination_anchor, _spawn_name, _dest_name = (
        planning_runner._resolve_route_anchor_transforms(
            world=world,
            carla=carla_module,
            world_map=world_map,
            anchors_cfg=anchors_cfg,
        )
    )
    aligned_spawn, _spawn_wp = planning_runner._align_transform_to_lane(
        global_planner, carla_module, spawn_anchor
    )
    aligned_destination, _dest_wp = planning_runner._align_transform_to_lane(
        global_planner, carla_module, destination_anchor
    )
    if aligned_spawn is None or aligned_destination is None:
        raise RuntimeError(
            "Could not align the spawn or destination anchors to a driving lane "
            f"for legacy scenario '{legacy_cfg.get('name', '<unknown>')}'."
        )

    route_summary = global_planner.plan_route_from_locations(
        start_location=aligned_spawn.location,
        goal_location=aligned_destination.location,
        replace_stored_route=True,
    )
    route_points: List[List[float]] = []
    if bool(getattr(route_summary, "route_found", False)):
        route_points = [
            [float(point[0]), float(point[1])]
            for point in route_summary.route_waypoints
        ]

    return LegacyScenarioContext(
        legacy_cfg=legacy_cfg,
        global_planner=global_planner,
        spawn_transform=aligned_spawn,
        destination_transform=aligned_destination,
        route_summary=route_summary,
        route_points=route_points,
    )


def transform_to_spawn_position(transform) -> List[float]:
    """`[x, y, z, roll, yaw, pitch]`, matching OpenCDA's `spawn_position` order
    (see `opencda/scenario_testing/utils/sim_api.py`'s `create_vehicle_manager`).
    """

    return [
        float(transform.location.x),
        float(transform.location.y),
        float(transform.location.z),
        float(transform.rotation.roll),
        float(transform.rotation.yaw),
        float(transform.rotation.pitch),
    ]


def transform_to_destination(transform) -> List[float]:
    return [
        float(transform.location.x),
        float(transform.location.y),
        float(transform.location.z),
    ]


def apply_resolved_anchors(scenario_params, cav_index: int, context: LegacyScenarioContext) -> None:
    """Overwrite the placeholder `spawn_position`/`destination` in the merged
    OpenCDA yaml config with the anchors resolved at runtime against the live
    CARLA world (marker names cannot be resolved to coordinates statically).
    """

    cav_cfg = scenario_params["scenario"]["single_cav_list"][cav_index]
    spawn_position = transform_to_spawn_position(context.spawn_transform)
    ego_cfg = dict(context.legacy_cfg.get("ego", {}) or {})
    spawn_z_offset_m = float(ego_cfg.get("spawn_z_offset_m", 0.0) or 0.0)
    if float(spawn_z_offset_m) > 0.0:
        spawn_position[2] = float(spawn_position[2]) + float(spawn_z_offset_m)
    cav_cfg["destination"] = transform_to_destination(context.destination_transform)
    cav_cfg["spawn_position"] = spawn_position


def _destroy_spawn_blockers(world: Any, spawn_transform: Any, radius_m: float = 6.0) -> int:
    """Remove stale actors left by a previous failed run near ego spawn."""

    spawn_location = getattr(spawn_transform, "location", None)
    if spawn_location is None:
        return 0
    destroyed_count = 0
    try:
        actors = list(world.get_actors())
    except Exception:
        return 0
    for actor in actors:
        try:
            type_id = str(getattr(actor, "type_id", ""))
            if not (
                type_id.startswith("vehicle.")
                or type_id.startswith("walker.")
                or type_id.startswith("sensor.")
            ):
                continue
            location = actor.get_location()
            if location.distance(spawn_location) > float(radius_m):
                continue
            actor.destroy()
            destroyed_count += 1
        except Exception:
            continue
    return destroyed_count


def _spawn_transform_from_position(carla_module: Any, spawn_position: Sequence[float]) -> Any:
    """Build a CARLA transform from OpenCDA's spawn_position list."""

    values = list(spawn_position)
    return carla_module.Transform(
        carla_module.Location(
            x=float(values[0]),
            y=float(values[1]),
            z=float(values[2]),
        ),
        carla_module.Rotation(
            roll=float(values[3]),
            yaw=float(values[4]),
            pitch=float(values[5]),
        ),
    )


def _current_cav_spawn_transform(
    scenario_params: Any,
    cav_index: int,
    carla_module: Any,
) -> Any:
    cav_cfg = scenario_params["scenario"]["single_cav_list"][cav_index]
    return _spawn_transform_from_position(carla_module, cav_cfg["spawn_position"])


def _setup_traffic_manager(scenario_manager, legacy_cfg: dict):
    traffic_manager_cfg = dict(legacy_cfg.get("traffic_manager", {}))
    if not bool(traffic_manager_cfg.get("enabled", False)):
        return None, int(traffic_manager_cfg.get("port", 8000))

    requested_port = int(traffic_manager_cfg.get("port", 8000))
    for candidate_port in range(requested_port, requested_port + 21):
        try:
            traffic_manager = scenario_manager.client.get_trafficmanager(candidate_port)
        except RuntimeError as exc:
            if "traffic manager" in str(exc).lower() and "bind" in str(exc).lower():
                continue
            raise
        if candidate_port != requested_port:
            print(
                f"[CPX SCENARIO BRIDGE] Traffic Manager port {requested_port} was busy; "
                f"using {candidate_port} instead."
            )
        try:
            traffic_manager.set_synchronous_mode(True)
        except Exception:
            pass
        return traffic_manager, candidate_port
    raise RuntimeError(f"Could not bind a Traffic Manager near port {requested_port}.")


def _setup_sumo_bridge(legacy_cfg: dict, client, world):
    sumo_cfg = dict(legacy_cfg.get("sumo", {}))
    if not bool(sumo_cfg.get("enabled", False)):
        return None

    from opencda_scenario.sumo_assets import ensure_sumo_assets
    from opencda_scenario.sumo_bridge import OpenCDASumoBridge

    sumo_asset_dir = ensure_sumo_assets(scenario_cfg=legacy_cfg, sumo_cfg=sumo_cfg)
    sumo_bridge = OpenCDASumoBridge(
        client=client,
        world=world,
        sumo_cfg=sumo_cfg,
        sumo_asset_dir=sumo_asset_dir,
    )
    print(f"[CPX SCENARIO BRIDGE] OpenCDA SUMO co-simulation enabled (assets={sumo_asset_dir}).")
    return sumo_bridge


def run_legacy_scenario_port(
    opt,
    scenario_params,
    *,
    loader_name: str,
    legacy_scenario_name: str,
    script_name: str,
    cav_index: int = 0,
) -> None:
    """Run a `carla_scenario`/`opencda_scenario` legacy scenario through the
    official OpenCDA `VehicleManager` / `CPXMPCPlannerBridge` pipeline.

    Parameters
    ----------
    opt, scenario_params:
        The same arguments every `opencda/scenario_testing/<name>.py`
        `run_scenario(opt, scenario_params)` entry point receives from
        `opencda.py`.
    loader_name:
        `"carla_scenario"` (native CARLA) or `"opencda_scenario"` (SUMO
        co-simulation variant).
    legacy_scenario_name:
        The scenario directory/name under that loader's root, e.g.
        `"town10_scenario_5"`.
    script_name:
        Used only for the evaluation manager's output folder name.
    """

    scenario_params = add_current_time(scenario_params)
    legacy_cfg = load_legacy_scenario_cfg(loader_name, legacy_scenario_name)
    reset_cp_message_payload()

    town = str(legacy_cfg.get("carla", {}).get("map", "")).strip() or None
    cav_world = CavWorld(opt.apply_ml)
    scenario_manager = sim_api.ScenarioManager(
        scenario_params,
        opt.apply_ml,
        opt.version,
        town=town,
        cav_world=cav_world,
    )
    world = scenario_manager.world
    carla_module = carla

    context = build_legacy_context(legacy_cfg, world, carla_module)
    apply_resolved_anchors(scenario_params, cav_index, context)
    current_spawn_transform = _current_cav_spawn_transform(
        scenario_params, cav_index, carla_module
    )
    cleared_spawn_blockers = _destroy_spawn_blockers(
        world,
        current_spawn_transform,
        radius_m=float(legacy_cfg.get("ego", {}).get("spawn_clear_radius_m", 12.0)),
    )
    if int(cleared_spawn_blockers) > 0:
        print(
            "[CPX SCENARIO BRIDGE] Removed "
            f"{int(cleared_spawn_blockers)} stale actor(s) near ego spawn."
        )
        try:
            world.tick()
        except Exception:
            pass

    single_cav_list = scenario_manager.create_vehicle_manager(application=["single"])
    ego_vehicle = single_cav_list[cav_index].vehicle

    traffic_manager, traffic_manager_port = _setup_traffic_manager(scenario_manager, legacy_cfg)

    obstacle_actors = planning_runner._spawn_scenario_obstacles_from_module(
        client=scenario_manager.client,
        world=world,
        map_obj=world.get_map(),
        map_planner=context.global_planner,
        carla=carla_module,
        blueprint_library=world.get_blueprint_library(),
        traffic_manager=traffic_manager,
        traffic_manager_port=int(traffic_manager_port),
        scenario_cfg=legacy_cfg,
        route_summary=context.route_summary,
        route_points=context.route_points,
    )

    sumo_bridge = _setup_sumo_bridge(legacy_cfg, scenario_manager.client, world)

    runtime_cfg = dict(legacy_cfg.get("runtime", {}))
    scenario_runtime_module = planning_runner._load_optional_module(
        module_name=str(runtime_cfg.get("module", "")).strip(),
        purpose="scenario runtime",
    )
    mpc_payload = planning_runner.load_yaml_file(planning_runner.MPC_CONFIG_PATH)
    tracker_payload = planning_runner.load_yaml_file(planning_runner.TRACKER_CONFIG_PATH)
    mpc_cfg = dict(mpc_payload.get("mpc", mpc_payload))
    tracker_cfg = dict(tracker_payload.get("tracker", tracker_payload))
    obstacle_filter_cfg = dict(mpc_cfg.get("obstacle_filter", {}))

    runtime_state = planning_runner._initialize_scenario_runtime_state(
        module=scenario_runtime_module,
        world=world,
        world_map=world.get_map(),
        map_planner=context.global_planner,
        carla=carla_module,
        scenario_cfg=legacy_cfg,
        traffic_manager_port=int(traffic_manager_port),
        tracker_cfg=tracker_cfg,
        obstacle_filter_cfg=obstacle_filter_cfg,
        prediction_dt_s=float(mpc_cfg.get("plan_dt_s", 0.2)),
        prediction_horizon_s=float(mpc_cfg.get("horizon_s", 3.0)),
    )

    eval_manager = EvaluationManager(
        scenario_manager.cav_world,
        script_name=script_name,
        current_time=scenario_params["current_time"],
    )

    debug_viewer: Optional[OpenCDADebugViewer] = None
    if OpenCDADebugViewer.enabled_from_env() and single_cav_list:
        debug_viewer = OpenCDADebugViewer(
            world=world,
            carla_module=carla_module,
            ego_vehicle=ego_vehicle,
        )

    try:
        spectator = world.get_spectator()
        while True:
            if sumo_bridge is not None:
                sumo_bridge.tick()
            else:
                scenario_manager.tick()

            sim_time_s = float(world.get_snapshot().timestamp.elapsed_seconds)
            wall_time_s = float(time.perf_counter())

            _discarded_snapshots, runtime_state = planning_runner._apply_scenario_dynamic_obstacle_filter(
                module=scenario_runtime_module,
                runtime_state=runtime_state,
                world=world,
                world_map=world.get_map(),
                map_planner=context.global_planner,
                carla=carla_module,
                scenario_cfg=legacy_cfg,
                object_snapshots=[],
                sim_time_s=sim_time_s,
                wall_time_s=wall_time_s,
            )
            _discarded_route, _discarded_points, runtime_state = (
                planning_runner._maybe_apply_scenario_global_route_update(
                    module=scenario_runtime_module,
                    runtime_state=runtime_state,
                    world=world,
                    world_map=world.get_map(),
                    carla=carla_module,
                    scenario_cfg=legacy_cfg,
                    global_planner=context.global_planner,
                    ego_vehicle=ego_vehicle,
                    sumo_bridge=sumo_bridge,
                    ego_transform=ego_vehicle.get_transform(),
                    goal_location=context.destination_transform.location,
                    object_snapshots=[],
                    current_route_summary=context.route_summary,
                    active_global_route_points=context.route_points,
                    sim_time_s=sim_time_s,
                    wall_time_s=wall_time_s,
                )
            )

            _set_spectator_transform(spectator, ego_vehicle)

            for cav in single_cav_list:
                cav.update_info()
                control = cav.run_step()
                cav.vehicle.apply_control(control)

            if debug_viewer is not None:
                debug_viewer.render(single_cav_list)
    finally:
        eval_manager.evaluate()
        if debug_viewer is not None:
            debug_viewer.destroy()
        scenario_manager.close()
        planning_runner._destroy_actors(obstacle_actors)
        if sumo_bridge is not None:
            close_fn = getattr(sumo_bridge, "close", None)
            if callable(close_fn):
                close_fn()
        for cav in single_cav_list:
            cav.destroy()
