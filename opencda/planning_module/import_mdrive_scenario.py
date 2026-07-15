"""
Convert an MDrive benchmark scenario folder (github.com/ucla-mobility/MDrive)
into a scenario directory runnable via `python main.py <name>` in this
project.

MDrive scenario folder shape:
    <folder>/actors_manifest.json
    <folder>/<ego_route>.xml            (many <waypoint> elements)
    <folder>/actors/static/<...>.xml    (single <waypoint> element)

Only the first `ego` entry is converted into this project's (single)
planner-controlled ego -- this project's runner is architecturally
single-ego, not built for MDrive's multi-agent negotiation scenarios.
Additional `ego` entries are demoted to autopilot background traffic
(same mechanism as `npc`/`bicycle` below) rather than run through the
planner or dropped entirely.

`static`, `pedestrian`, `npc`, `bicycle`, and extra `ego` actors ARE
converted:
  - static actors spawn as parked vehicles (with a distance-triggered
    cooperative "hazard" message so the behavior planner can react before
    its own perception detects them).
  - pedestrians spawn and walk their full waypoint polyline via a per-tick
    hook.
  - npc vehicles and bicycles (CARLA bicycles are vehicle-category
    blueprints, driven the same way as cars) spawn and default to generic
    Traffic Manager autopilot, matching this project's existing
    `marker_vehicle_mode: autopilot` convention -- set an entry's
    `follow_mode` to `"literal"` to instead have it drive its own MDrive
    polyline directly, for scenarios where its exact position/timing
    matters.

Usage:
    python import_mdrive_scenario.py <mdrive_scenario_dir> <new_scenario_name>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, Sequence

# A static/npc/pedestrian actor whose position sits farther than this from
# every point on the selected ego's route is very likely tied to a
# *different* MDrive vehicle's route (see the Blocked_Lane_Obstacle bug:
# ego[0] was picked, but the obstacles actually sat on ego[1]/ego[2]'s
# lane). Roughly 2 lane widths.
ROUTE_PROXIMITY_WARNING_THRESHOLD_M = 7.0

import yaml

from utility.config_loader import load_yaml_file


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CARLA_SCENARIO_DIR = os.path.join(PROJECT_ROOT, "carla_scenario")
TEMPLATE_SCENARIO_PATH = os.path.join(CARLA_SCENARIO_DIR, "town10", "town10.yaml")


def _actor_label(entry: Mapping[str, object]) -> str:
    return str(entry.get("name", entry.get("file", "<unknown>")))


def _load_manifest(scenario_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(scenario_dir, "actors_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _parse_route_waypoints(route_xml_path: str) -> List[Dict[str, float]]:
    tree = ET.parse(route_xml_path)
    root = tree.getroot()
    route_el = root.find("route")
    if route_el is None:
        raise ValueError(f"No <route> element found in {route_xml_path}")
    waypoints = [
        {
            "x": float(waypoint_el.get("x", 0.0)),
            "y": float(waypoint_el.get("y", 0.0)),
            "z": float(waypoint_el.get("z", 0.0)),
            "yaw": float(waypoint_el.get("yaw", 0.0)),
        }
        for waypoint_el in route_el.findall("waypoint")
    ]
    if not waypoints:
        raise ValueError(f"No <waypoint> elements found in {route_xml_path}")
    return waypoints


# Mirrors carla.WeatherParameters' constructor kwargs (planning_runner.py's
# _WEATHER_PARAMETER_KEYS) -- MDrive's <weather .../> attribute names match
# these directly, so no translation table is needed, just a passthrough of
# whichever of these attributes are present.
_WEATHER_ATTRIBUTE_KEYS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "fog_density",
    "fog_distance",
    "fog_falloff",
    "wetness",
)


def _parse_route_weather(route_xml_path: str) -> Dict[str, float] | None:
    tree = ET.parse(route_xml_path)
    root = tree.getroot()
    route_el = root.find("route")
    if route_el is None:
        return None
    weather_el = route_el.find("weather")
    if weather_el is None:
        return None
    weather: Dict[str, float] = {}
    for key in _WEATHER_ATTRIBUTE_KEYS:
        raw_value = weather_el.get(key)
        if raw_value is None:
            continue
        try:
            weather[key] = float(raw_value)
        except ValueError:
            continue
    return weather or None


def _xyz_yaw_list(waypoint: Mapping[str, float]) -> List[float]:
    return [float(waypoint["x"]), float(waypoint["y"]), float(waypoint["z"]), float(waypoint["yaw"])]


def _route_waypoint_list(waypoints: List[Mapping[str, float]]) -> List[List[float]]:
    return [
        [
            float(waypoint["x"]),
            float(waypoint["y"]),
            float(waypoint["z"]),
            float(waypoint["yaw"]),
        ]
        for waypoint in list(waypoints or [])
    ]


def _distance_point_to_segment(
    point_xy: Sequence[float],
    segment_start_xy: Sequence[float],
    segment_end_xy: Sequence[float],
) -> float:
    px, py = float(point_xy[0]), float(point_xy[1])
    ax, ay = float(segment_start_xy[0]), float(segment_start_xy[1])
    bx, by = float(segment_end_xy[0]), float(segment_end_xy[1])
    dx, dy = bx - ax, by - ay
    segment_len_sq = dx * dx + dy * dy
    if segment_len_sq <= 1e-9:
        return float(math.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / segment_len_sq))
    nearest_x, nearest_y = ax + t * dx, ay + t * dy
    return float(math.hypot(px - nearest_x, py - nearest_y))


def _min_distance_to_polyline(point_xy: Sequence[float], polyline_xy: Sequence[Sequence[float]]) -> float:
    points = list(polyline_xy or [])
    if len(points) == 0:
        return float("inf")
    if len(points) == 1:
        return float(math.hypot(float(point_xy[0]) - float(points[0][0]), float(point_xy[1]) - float(points[0][1])))
    return min(
        _distance_point_to_segment(point_xy, points[index], points[index + 1])
        for index in range(len(points) - 1)
    )


def _convert_moving_actor_entries(
    *,
    entries: Sequence[Mapping[str, object]],
    mdrive_scenario_dir: str,
    ego_route_xy: Sequence[Sequence[float]],
    default_blueprint: str,
    default_speed_mps: float,
    kind_label: str,
    extra_fields: Mapping[str, object] | None = None,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Parse a list of MDrive manifest entries (pedestrian/npc/bicycle) that
    each reference a full-polyline route XML, converting each into this
    project's `{blueprint, role_name, speed_mps, waypoints, ...}` shape and
    flagging any whose route never comes near the selected ego's route."""

    converted: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for entry in entries:
        route_path = os.path.join(mdrive_scenario_dir, str(entry["file"]))
        waypoints = _parse_route_waypoints(route_path)
        role_name = _actor_label(entry)
        target_speed_mps = entry.get("target_speed", entry.get("speed", default_speed_mps))
        item: Dict[str, Any] = {
            "blueprint": str(entry.get("model", default_blueprint)),
            "role_name": role_name,
            "speed_mps": float(target_speed_mps),
            "waypoints": _route_waypoint_list(waypoints),
        }
        if extra_fields:
            item.update(dict(extra_fields))
        converted.append(item)

        route_xy = [[float(wp["x"]), float(wp["y"])] for wp in waypoints]
        closest_approach_m = min(
            _min_distance_to_polyline(point_xy, ego_route_xy) for point_xy in route_xy
        )
        if closest_approach_m > ROUTE_PROXIMITY_WARNING_THRESHOLD_M:
            warnings.append(
                f"{kind_label} '{role_name}' never comes within {ROUTE_PROXIMITY_WARNING_THRESHOLD_M:.1f}m "
                f"of the selected ego's route (closest approach {closest_approach_m:.1f}m) -- it may belong "
                "to a different MDrive vehicle's route; double-check ego selection."
            )
    return converted, warnings


def _actor_route_points_xy(
    mdrive_scenario_dir: str,
    entries: Sequence[Mapping[str, object]],
) -> List[List[List[float]]]:
    """Return, for each entry, the (x, y) points of its full route -- used
    both to score ego candidates and (by the caller) to check proximity
    against whichever ego ends up selected."""

    all_points: List[List[List[float]]] = []
    for entry in entries:
        route_path = os.path.join(mdrive_scenario_dir, str(entry["file"]))
        waypoints = _parse_route_waypoints(route_path)
        all_points.append([[float(wp["x"]), float(wp["y"])] for wp in waypoints])
    return all_points


def _select_primary_ego_index(
    *,
    mdrive_scenario_dir: str,
    ego_entries: Sequence[Mapping[str, object]],
    target_points_by_actor: Sequence[Sequence[Sequence[float]]],
) -> int:
    """Pick the ego candidate whose route has the smallest total distance to
    every other (static/pedestrian/npc/bicycle) actor's route, instead of
    always taking ego[0].

    MDrive's manifest order doesn't reliably put the "relevant" vehicle
    first -- an audit of this importer's own route-proximity warnings
    across the `interaction` bucket found real cases (e.g. `precrash/L`)
    where ego[0] sat 13m from the scenario's key actor while ego[1] sat
    exactly on top of it (0.0m). With only one ego candidate, or no other
    actors to score against, this is a no-op (returns 0) -- there's nothing
    to compare.
    """

    if len(ego_entries) <= 1 or not any(target_points_by_actor):
        return 0

    best_index = 0
    best_score: float | None = None
    for index, ego_candidate in enumerate(ego_entries):
        route_path = os.path.join(mdrive_scenario_dir, str(ego_candidate["file"]))
        candidate_xy = [
            [float(wp["x"]), float(wp["y"])] for wp in _parse_route_waypoints(route_path)
        ]
        score = sum(
            min(_min_distance_to_polyline(point_xy, candidate_xy) for point_xy in actor_points)
            for actor_points in target_points_by_actor
            if actor_points
        )
        if best_score is None or score < best_score:
            best_score = score
            best_index = index
    return best_index


def convert_scenario(mdrive_scenario_dir: str, new_scenario_name: str) -> Dict[str, Any]:
    manifest = _load_manifest(mdrive_scenario_dir)

    ego_entries = list(manifest.get("ego", []) or [])
    if not ego_entries:
        raise ValueError(f"No 'ego' entries found in {mdrive_scenario_dir}/actors_manifest.json")

    other_actor_entries = [
        *(manifest.get("static", []) or []),
        *(manifest.get("pedestrian", []) or []),
        *(manifest.get("npc", []) or []),
        *(manifest.get("bicycle", []) or []),
    ]
    target_points_by_actor = _actor_route_points_xy(mdrive_scenario_dir, other_actor_entries)
    primary_ego_index = _select_primary_ego_index(
        mdrive_scenario_dir=mdrive_scenario_dir,
        ego_entries=ego_entries,
        target_points_by_actor=target_points_by_actor,
    )
    auto_selected_ego = primary_ego_index != 0

    primary_ego = ego_entries[primary_ego_index]
    ego_route_path = os.path.join(mdrive_scenario_dir, str(primary_ego["file"]))
    ego_waypoints = _parse_route_waypoints(ego_route_path)
    ego_weather = _parse_route_weather(ego_route_path)
    ego_spawn = ego_waypoints[0]
    ego_destination = ego_waypoints[-1]
    town = str(primary_ego.get("town", "Town10HD_Opt")).strip()
    ego_route_xy = [[float(waypoint["x"]), float(waypoint["y"])] for waypoint in ego_waypoints]
    ego_blueprint = str(primary_ego.get("model", "")).strip()
    ego_target_speed_mps = primary_ego.get("target_speed", primary_ego.get("speed", None))

    static_actors: List[Dict[str, Any]] = []
    route_proximity_warnings: List[str] = []
    for static_entry in manifest.get("static", []) or []:
        static_route_path = os.path.join(mdrive_scenario_dir, str(static_entry["file"]))
        static_waypoint = _parse_route_waypoints(static_route_path)[0]
        role_name = _actor_label(static_entry)
        static_actors.append(
            {
                "blueprint": str(static_entry.get("model", "vehicle.tesla.cybertruck")),
                "role_name": role_name,
                "x": float(static_waypoint["x"]),
                "y": float(static_waypoint["y"]),
                "z": float(static_waypoint["z"]),
                "yaw": float(static_waypoint["yaw"]),
            }
        )
        distance_to_route_m = _min_distance_to_polyline(
            [float(static_waypoint["x"]), float(static_waypoint["y"])],
            ego_route_xy,
        )
        if distance_to_route_m > ROUTE_PROXIMITY_WARNING_THRESHOLD_M:
            route_proximity_warnings.append(
                f"static actor '{role_name}' is {distance_to_route_m:.1f}m from the selected ego's route "
                f"(threshold {ROUTE_PROXIMITY_WARNING_THRESHOLD_M:.1f}m) -- it may belong to a different "
                "MDrive vehicle's route; double-check ego selection."
            )

    pedestrians, pedestrian_warnings = _convert_moving_actor_entries(
        entries=manifest.get("pedestrian", []) or [],
        mdrive_scenario_dir=mdrive_scenario_dir,
        ego_route_xy=ego_route_xy,
        default_blueprint="walker.pedestrian.0001",
        default_speed_mps=1.2,
        kind_label="pedestrian",
    )
    route_proximity_warnings.extend(pedestrian_warnings)

    npc_vehicles, npc_warnings = _convert_moving_actor_entries(
        entries=manifest.get("npc", []) or [],
        mdrive_scenario_dir=mdrive_scenario_dir,
        ego_route_xy=ego_route_xy,
        default_blueprint="vehicle.tesla.model3",
        default_speed_mps=8.0,
        kind_label="npc vehicle",
        extra_fields={"follow_mode": "autopilot"},
    )
    route_proximity_warnings.extend(npc_warnings)

    # CARLA bicycles are vehicle-category blueprints driven via
    # VehicleControl/set_autopilot, mechanically identical to NPC cars --
    # convert them into the same npc_vehicles list rather than a separate
    # code path in utility/coordinate_obstacle_spawner.py.
    bicycles, bicycle_warnings = _convert_moving_actor_entries(
        entries=manifest.get("bicycle", []) or [],
        mdrive_scenario_dir=mdrive_scenario_dir,
        ego_route_xy=ego_route_xy,
        default_blueprint="vehicle.bh.crossbike",
        default_speed_mps=4.0,
        kind_label="bicycle",
        extra_fields={"follow_mode": "autopilot"},
    )
    route_proximity_warnings.extend(bicycle_warnings)
    bicycle_count = len(bicycles)
    npc_vehicles.extend(bicycles)

    # Extra `ego` entries beyond the selected one aren't real multi-agent
    # negotiation (this project's planner is single-ego, architecturally),
    # but demoting them to autopilot background traffic keeps "another car
    # is present" scenario semantics instead of them vanishing entirely.
    # Their routes are deliberately NOT distance-checked against the
    # selected ego's route -- being elsewhere in the scene is normal for
    # background traffic, not a sign of a wrong-ego-selection bug.
    other_ego_entries = [
        entry for index, entry in enumerate(ego_entries) if index != primary_ego_index
    ]
    demoted_ego, _demoted_ego_warnings = _convert_moving_actor_entries(
        entries=other_ego_entries,
        mdrive_scenario_dir=mdrive_scenario_dir,
        ego_route_xy=[],
        default_blueprint="vehicle.tesla.model3",
        default_speed_mps=8.0,
        kind_label="extra ego",
        extra_fields={"follow_mode": "autopilot"},
    )
    demoted_ego_count = len(demoted_ego)
    npc_vehicles.extend(demoted_ego)

    skipped: Dict[str, List[str]] = {}

    template = load_yaml_file(TEMPLATE_SCENARIO_PATH)

    generated_cfg: Dict[str, Any] = {
        "name": new_scenario_name,
        "runner_module": "planning_runner",
        "carla": copy.deepcopy(template["carla"]),
        "anchors": {
            "ego_spawn": f"{new_scenario_name}_ego",
            "final_destination": f"{new_scenario_name}_final_destination",
            "ego_spawn_xyz_yaw": _xyz_yaw_list(ego_spawn),
            "final_destination_xyz_yaw": _xyz_yaw_list(ego_destination),
        },
        "ego": copy.deepcopy(template["ego"]),
        "constraints": copy.deepcopy(template["constraints"]),
        "planning": copy.deepcopy(template["planning"]),
        "runtime": copy.deepcopy(template.get("runtime", {})),
        "camera": copy.deepcopy(template["camera"]),
    }
    generated_cfg["planning"]["imported_route_waypoints"] = _route_waypoint_list(ego_waypoints)
    generated_cfg["carla"]["map"] = f"/Game/Carla/Maps/{town}"

    if ego_weather:
        generated_cfg["weather"] = ego_weather

    if ego_blueprint:
        generated_cfg["ego"]["blueprint"] = ego_blueprint
    if ego_target_speed_mps is not None:
        try:
            # 1.25x margin: MDrive's speed is the ego's own intended
            # cruising speed, not a hard ceiling -- capping max_velocity_mps
            # at exactly that value leaves the MPC no headroom to catch up
            # after a slowdown (e.g. yielding, a lane change) without
            # immediately saturating the constraint.
            generated_cfg["constraints"]["max_velocity_mps"] = round(float(ego_target_speed_mps) * 1.25, 3)
        except (TypeError, ValueError):
            pass

    if static_actors or pedestrians or npc_vehicles:
        obstacles_block: Dict[str, Any] = {
            "spawner_module": "utility.coordinate_obstacle_spawner",
        }
        if static_actors:
            obstacles_block["static_actors"] = static_actors
            # Distance-triggered CP "hazard" message per obstacle, consumed
            # by behavior_planner/reroute.py's lane-closure pathway --
            # mirrors carla_scenario/roadway_hazard/scenario.py so imported
            # blocked-lane scenarios feed the same mechanism as
            # hand-authored ones instead of only reacting once the ego's
            # own simulated perception detects the obstacle.
            obstacles_block["cooperative_message_trigger_distance_m"] = 20.0
        if pedestrians:
            obstacles_block["pedestrians"] = pedestrians
        if npc_vehicles:
            # follow_mode defaults to "autopilot" (generic Traffic-Manager
            # traffic, low risk, matches this project's existing
            # marker_vehicle_mode: autopilot convention) rather than
            # literally replaying MDrive's polyline -- hand-edit an entry's
            # follow_mode to "literal" for scenarios where the NPC's exact
            # position/timing matters for the interaction being tested.
            obstacles_block["npc_vehicles"] = npc_vehicles
        generated_cfg["obstacles"] = obstacles_block
        # maybe_replan_global_route() lives in the same module as
        # spawn_obstacles(); both are optional hooks planning_runner.py
        # discovers by name, so pointing runtime.module here does not
        # affect obstacle spawning, which is wired separately via
        # obstacles.spawner_module above.
        generated_cfg["runtime"]["module"] = "utility.coordinate_obstacle_spawner"

    return {
        "config": generated_cfg,
        "skipped": skipped,
        "ego_route_waypoint_count": len(ego_waypoints),
        "static_actor_count": len(static_actors),
        "pedestrian_count": len(pedestrians),
        "npc_vehicle_count": len(npc_vehicles),
        "bicycle_count": bicycle_count,
        "demoted_ego_count": demoted_ego_count,
        "auto_selected_ego": auto_selected_ego,
        "primary_ego_name": _actor_label(primary_ego),
        "town": town,
        "route_proximity_warnings": route_proximity_warnings,
        "weather_converted": bool(ego_weather),
    }


def _map_asset_exists_locally(town: str) -> bool | None:
    carla_root = os.environ.get("CARLA_ROOT", "").strip()
    if not carla_root:
        return None
    umap_path = os.path.join(carla_root, "CarlaUE4", "Content", "Carla", "Maps", f"{town}.umap")
    return os.path.isfile(umap_path)


def write_scenario(new_scenario_name: str, generated_cfg: Mapping[str, object]) -> str:
    scenario_dir = os.path.join(CARLA_SCENARIO_DIR, new_scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)
    output_path = os.path.join(scenario_dir, f"{new_scenario_name}.yaml")
    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(dict(generated_cfg), file, sort_keys=False, default_flow_style=False)
    return output_path


def _print_summary(new_scenario_name: str, result: Mapping[str, object], output_path: str) -> None:
    skipped = dict(result["skipped"])
    print(f"[MDRIVE IMPORT] Converted -> carla_scenario/{new_scenario_name}/{new_scenario_name}.yaml")
    if bool(result.get("auto_selected_ego", False)):
        print(
            f"[MDRIVE IMPORT] Auto-selected ego: '{result['primary_ego_name']}' was NOT manifest['ego'][0] -- "
            "chosen because its route sits closest overall to this scenario's other actors."
        )
    print(f"[MDRIVE IMPORT] Town: {result['town']}")
    print(f"[MDRIVE IMPORT] Ego route: {result['ego_route_waypoint_count']} waypoints (spawn = first, destination = last)")
    print(f"[MDRIVE IMPORT] Static obstacles converted: {result['static_actor_count']}")
    print(f"[MDRIVE IMPORT] Pedestrians converted: {result['pedestrian_count']}")
    npc_only_count = (
        int(result['npc_vehicle_count']) - int(result['bicycle_count']) - int(result['demoted_ego_count'])
    )
    print(
        f"[MDRIVE IMPORT] NPC vehicles converted: {npc_only_count}, bicycles: {result['bicycle_count']} "
        "(both follow_mode=autopilot by default)"
    )
    if int(result['demoted_ego_count']) > 0:
        print(
            f"[MDRIVE IMPORT] Demoted {result['demoted_ego_count']} extra ego vehicle(s) to autopilot "
            "background traffic (not run through this project's planner -- single-ego only)"
        )
    print(
        "[MDRIVE IMPORT] Weather: converted"
        if bool(result.get("weather_converted", False))
        else "[MDRIVE IMPORT] Weather: not present in source route, using CARLA default"
    )

    map_exists = _map_asset_exists_locally(str(result["town"]))
    if map_exists is None:
        print("[MDRIVE IMPORT] Could not check map availability (CARLA_ROOT is not set).")
    elif map_exists:
        print(f"[MDRIVE IMPORT] Map asset for '{result['town']}' found locally.")
    else:
        print(
            f"[MDRIVE IMPORT] WARNING: map asset for '{result['town']}' was NOT found under "
            f"$CARLA_ROOT/CarlaUE4/Content/Carla/Maps/. This scenario will fail to load until it is."
        )

    total_skipped = sum(len(v) for v in skipped.values())
    if total_skipped == 0:
        print("[MDRIVE IMPORT] Nothing skipped -- all actors in the manifest were convertible.")
    else:
        print(f"[MDRIVE IMPORT] Skipped {total_skipped} actor(s) not supported by this importer (single ego only):")
        for kind, names in skipped.items():
            if names:
                print(f"[MDRIVE IMPORT]   {kind}: {', '.join(names)}")

    route_proximity_warnings = list(result.get("route_proximity_warnings", []) or [])
    if route_proximity_warnings:
        print(
            f"[MDRIVE IMPORT] WARNING: {len(route_proximity_warnings)} static actor(s) are far from the "
            "selected ego's route -- likely a wrong-ego-selection bug, not a real blocked-lane obstacle:"
        )
        for warning in route_proximity_warnings:
            print(f"[MDRIVE IMPORT]   {warning}")

    print(f"[MDRIVE IMPORT] Wrote {output_path}")
    print(f"[MDRIVE IMPORT] Run it with: python main.py {new_scenario_name}")


def _discover_mdrive_scenario_dirs(root: str) -> List[str]:
    """Recursively find every directory under `root` containing an
    `actors_manifest.json`, regardless of nesting depth -- handles both the
    `interaction` bucket's `<category>/<n>/` shape and flatter buckets like
    `v2xpnp`'s `<scenario_id>/` shape uniformly."""

    scenario_dirs: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "actors_manifest.json" in filenames:
            scenario_dirs.append(dirpath)
    return sorted(scenario_dirs)


def _scenario_name_from_path(root: str, scenario_dir: str) -> str:
    relative_path = os.path.relpath(scenario_dir, root)
    slug = re.sub(r"[^a-z0-9]+", "_", relative_path.lower()).strip("_")
    return f"mdrive_{slug}"


def run_batch_conversion(root: str, *, dry_run: bool = False) -> Dict[str, Any]:
    scenario_dirs = _discover_mdrive_scenario_dirs(root)
    print(f"[MDRIVE IMPORT] Found {len(scenario_dirs)} scenario folder(s) under {root}")

    converted: List[str] = []
    map_missing: List[str] = []
    errors: List[str] = []
    all_route_proximity_warnings: Dict[str, List[str]] = {}
    auto_selected_egos: Dict[str, str] = {}

    for scenario_dir in scenario_dirs:
        new_scenario_name = _scenario_name_from_path(root, scenario_dir)
        try:
            result = convert_scenario(scenario_dir, new_scenario_name)
        except Exception as exc:
            errors.append(f"{new_scenario_name}: {exc}")
            print(f"[MDRIVE IMPORT] ERROR converting {scenario_dir} -> {new_scenario_name}: {exc}")
            continue

        map_exists = _map_asset_exists_locally(str(result["town"]))
        if map_exists is False:
            map_missing.append(f"{new_scenario_name} (town={result['town']})")

        if result["route_proximity_warnings"]:
            all_route_proximity_warnings[new_scenario_name] = list(result["route_proximity_warnings"])

        if result.get("auto_selected_ego"):
            auto_selected_egos[new_scenario_name] = str(result["primary_ego_name"])

        if dry_run:
            print(f"[MDRIVE IMPORT] Would convert {scenario_dir} -> {new_scenario_name}")
        else:
            output_path = write_scenario(new_scenario_name, result["config"])
            print(f"[MDRIVE IMPORT] Converted {scenario_dir} -> {output_path}")
        converted.append(new_scenario_name)

    print("\n[MDRIVE IMPORT] ==== Batch summary ====")
    print(f"[MDRIVE IMPORT] {'Would convert' if dry_run else 'Converted'}: {len(converted)}")
    print(f"[MDRIVE IMPORT] Errors: {len(errors)}")
    for error in errors:
        print(f"[MDRIVE IMPORT]   {error}")
    print(f"[MDRIVE IMPORT] Missing map assets: {len(map_missing)}")
    for entry in map_missing:
        print(f"[MDRIVE IMPORT]   {entry}")
    if all_route_proximity_warnings:
        print(f"[MDRIVE IMPORT] Scenarios with route-proximity warnings: {len(all_route_proximity_warnings)}")
        for name, warnings in all_route_proximity_warnings.items():
            print(f"[MDRIVE IMPORT]   {name}: {len(warnings)} warning(s)")
    if auto_selected_egos:
        print(f"[MDRIVE IMPORT] Scenarios with auto-selected (non-default) ego: {len(auto_selected_egos)}")
        for name, ego_name in auto_selected_egos.items():
            print(f"[MDRIVE IMPORT]   {name}: selected '{ego_name}'")

    return {
        "converted": converted,
        "errors": errors,
        "map_missing": map_missing,
        "route_proximity_warnings": all_route_proximity_warnings,
        "auto_selected_egos": auto_selected_egos,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mdrive_scenario_dir",
        help=(
            "Path to an MDrive scenario folder (contains actors_manifest.json). "
            "With --batch, this is instead a root to search recursively for scenario folders."
        ),
    )
    parser.add_argument(
        "new_scenario_name",
        nargs="?",
        default=None,
        help="Name for the generated scenario. Required unless --batch is set (names are auto-derived per scenario).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Treat mdrive_scenario_dir as a root and convert every scenario folder found under it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --batch, only report what would be converted and check map availability -- write nothing.",
    )
    args = parser.parse_args()

    if args.batch:
        run_batch_conversion(args.mdrive_scenario_dir, dry_run=bool(args.dry_run))
        return 0

    if args.new_scenario_name is None:
        parser.error("new_scenario_name is required unless --batch is set")

    result = convert_scenario(args.mdrive_scenario_dir, args.new_scenario_name)
    output_path = write_scenario(args.new_scenario_name, result["config"])
    _print_summary(args.new_scenario_name, result, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
