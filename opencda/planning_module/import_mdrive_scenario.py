"""
Convert an MDrive benchmark scenario folder (github.com/ucla-mobility/MDrive)
into a scenario directory runnable via `python main.py <name>` in this
project.

MDrive scenario folder shape:
    <folder>/actors_manifest.json
    <folder>/<ego_route>.xml            (many <waypoint> elements)
    <folder>/actors/static/<...>.xml    (single <waypoint> element)

Only the first `ego` entry is converted into this project's (single) ego
route; any additional `ego` entries plus all `npc`/`pedestrian`/`bicycle`
entries are reported as skipped, not converted -- this project's runner
supports one planner-controlled vehicle, not MDrive's multi-agent
negotiation scenarios.

Usage:
    python import_mdrive_scenario.py <mdrive_scenario_dir> <new_scenario_name>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping

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


def convert_scenario(mdrive_scenario_dir: str, new_scenario_name: str) -> Dict[str, Any]:
    manifest = _load_manifest(mdrive_scenario_dir)

    ego_entries = list(manifest.get("ego", []) or [])
    if not ego_entries:
        raise ValueError(f"No 'ego' entries found in {mdrive_scenario_dir}/actors_manifest.json")

    primary_ego = ego_entries[0]
    ego_route_path = os.path.join(mdrive_scenario_dir, str(primary_ego["file"]))
    ego_waypoints = _parse_route_waypoints(ego_route_path)
    ego_spawn = ego_waypoints[0]
    ego_destination = ego_waypoints[-1]
    town = str(primary_ego.get("town", "Town10HD_Opt")).strip()

    static_actors: List[Dict[str, Any]] = []
    for static_entry in manifest.get("static", []) or []:
        static_route_path = os.path.join(mdrive_scenario_dir, str(static_entry["file"]))
        static_waypoint = _parse_route_waypoints(static_route_path)[0]
        static_actors.append(
            {
                "blueprint": str(static_entry.get("model", "vehicle.tesla.cybertruck")),
                "role_name": _actor_label(static_entry),
                "x": float(static_waypoint["x"]),
                "y": float(static_waypoint["y"]),
                "z": float(static_waypoint["z"]),
                "yaw": float(static_waypoint["yaw"]),
            }
        )

    skipped = {
        "extra_ego": [_actor_label(entry) for entry in ego_entries[1:]],
        "npc": [_actor_label(entry) for entry in manifest.get("npc", []) or []],
        "pedestrian": [_actor_label(entry) for entry in manifest.get("pedestrian", []) or []],
        "bicycle": [_actor_label(entry) for entry in manifest.get("bicycle", []) or []],
    }

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

    if static_actors:
        generated_cfg["obstacles"] = {
            "spawner_module": "utility.coordinate_obstacle_spawner",
            "static_actors": static_actors,
            # Distance-triggered CP "hazard" message per obstacle, consumed
            # by behavior_planner/reroute.py's lane-closure pathway --
            # mirrors carla_scenario/roadway_hazard/scenario.py so imported
            # blocked-lane scenarios feed the same mechanism as
            # hand-authored ones instead of only reacting once the ego's
            # own simulated perception detects the obstacle.
            "cooperative_message_trigger_distance_m": 20.0,
        }
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
        "town": town,
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
    print(f"[MDRIVE IMPORT] Town: {result['town']}")
    print(f"[MDRIVE IMPORT] Ego route: {result['ego_route_waypoint_count']} waypoints (spawn = first, destination = last)")
    print(f"[MDRIVE IMPORT] Static obstacles converted: {result['static_actor_count']}")

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
        print(f"[MDRIVE IMPORT] Skipped {total_skipped} actor(s) not supported by this importer (v1 is single-ego, static-obstacles-only):")
        for kind, names in skipped.items():
            if names:
                print(f"[MDRIVE IMPORT]   {kind}: {', '.join(names)}")

    print(f"[MDRIVE IMPORT] Wrote {output_path}")
    print(f"[MDRIVE IMPORT] Run it with: python main.py {new_scenario_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mdrive_scenario_dir", help="Path to an MDrive scenario folder (contains actors_manifest.json)")
    parser.add_argument("new_scenario_name", help="Name for the generated scenario (used as the yaml filename and directory)")
    args = parser.parse_args()

    result = convert_scenario(args.mdrive_scenario_dir, args.new_scenario_name)
    output_path = write_scenario(args.new_scenario_name, result["config"])
    _print_summary(args.new_scenario_name, result, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
