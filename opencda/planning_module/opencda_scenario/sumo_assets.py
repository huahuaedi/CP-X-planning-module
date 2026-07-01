"""
Helpers for preparing SUMO assets for planning-module scenarios.
"""

from __future__ import annotations

import os
import glob
import platform
import shutil
import subprocess
import sys
from typing import Mapping


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidate_opencda_roots() -> list[str]:
    current_root = os.path.dirname(os.path.dirname(PROJECT_ROOT))
    roots = [
        os.environ.get("OPENCDA_ROOT", "").strip(),
        current_root,
        os.path.dirname(current_root),
        os.getcwd(),
        os.path.dirname(os.getcwd()),
        "/home/umd-user/Desktop/OpenCDA",
    ]
    return list(dict.fromkeys(os.path.abspath(root) for root in roots if root))


def _has_opencda_helpers(root: str) -> bool:
    return (
        os.path.isfile(os.path.join(root, "opencda", "__init__.py"))
        and os.path.isfile(os.path.join(root, "scripts", "netconvert_carla.py"))
    )


def _resolve_opencda_root() -> str:
    for root in _candidate_opencda_roots():
        if _has_opencda_helpers(root):
            return root
    return os.path.dirname(os.path.dirname(PROJECT_ROOT))


OPENCDA_ROOT = _resolve_opencda_root()
if OPENCDA_ROOT not in sys.path:
    sys.path.insert(0, OPENCDA_ROOT)
DEFAULT_ASSET_ROOT = os.path.join(PROJECT_ROOT, "opencda_scenario", "assets")
DEFAULT_CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/umd-user/carla_source/carla")
DEFAULT_SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")


def _map_leaf_name(map_name: str) -> str:
    return str(map_name or "").strip().rstrip("/").split("/")[-1]


def resolve_map_basename(
    scenario_cfg: Mapping[str, object] | None = None,
    sumo_cfg: Mapping[str, object] | None = None,
) -> str:
    """
    Resolve the base filename used for the SUMO config, net, and route files.
    """

    sumo_cfg = dict(sumo_cfg or {})
    scenario_cfg = dict(scenario_cfg or {})
    configured = str(sumo_cfg.get("map_basename", "")).strip()
    if configured:
        return configured
    carla_cfg = dict(scenario_cfg.get("carla", {}))
    return _map_leaf_name(str(carla_cfg.get("map", "Town10HD_Opt"))) or "Town10HD_Opt"


def resolve_asset_root(
    sumo_cfg: Mapping[str, object] | None = None,
    project_root: str = PROJECT_ROOT,
) -> str:
    """
    Resolve the asset root directory for SUMO files.
    """

    sumo_cfg = dict(sumo_cfg or {})
    configured = str(sumo_cfg.get("asset_root", "")).strip()
    if not configured:
        return DEFAULT_ASSET_ROOT
    if os.path.isabs(configured):
        return configured
    return os.path.join(project_root, configured)


def resolve_xodr_path(
    scenario_cfg: Mapping[str, object] | None = None,
    sumo_cfg: Mapping[str, object] | None = None,
) -> str:
    """
    Resolve the OpenDRIVE path for the requested CARLA map.
    """

    scenario_cfg = dict(scenario_cfg or {})
    sumo_cfg = dict(sumo_cfg or {})
    configured = str(sumo_cfg.get("xodr_path", "")).strip()
    candidates: list[str] = []
    if configured:
        candidates.append(configured)

    carla_cfg = dict(scenario_cfg.get("carla", {}))
    map_basename = resolve_map_basename(scenario_cfg=scenario_cfg, sumo_cfg=sumo_cfg)
    carla_root = str(carla_cfg.get("carla_root", DEFAULT_CARLA_ROOT)).strip() or DEFAULT_CARLA_ROOT
    open_drive_dir = os.path.join(
        carla_root,
        "Unreal",
        "CarlaUE4",
        "Content",
        "Carla",
        "Maps",
        "OpenDrive",
    )
    candidates.append(os.path.join(open_drive_dir, f"{map_basename}.xodr"))
    if map_basename.endswith("_Opt"):
        candidates.append(os.path.join(open_drive_dir, f"{map_basename[:-4]}.xodr"))

    home_dir = os.path.expanduser("~")
    cache_open_drive_dir = os.path.join(
        home_dir,
        "carlaCache",
        "Carla",
        "Maps",
        "OpenDrive",
    )
    candidates.append(os.path.join(cache_open_drive_dir, f"{map_basename}.xodr"))
    if map_basename.endswith("_Opt"):
        candidates.append(os.path.join(cache_open_drive_dir, f"{map_basename[:-4]}.xodr"))

    for fallback_root in (
        os.environ.get("CARLA_ROOT", "").strip(),
        "/home/umd-user/Downloads/MDrive/carla912",
        "/opt/carla-simulator",
    ):
        if not fallback_root:
            continue
        fallback_open_drive_dir = os.path.join(
            fallback_root,
            "CarlaUE4",
            "Content",
            "Carla",
            "Maps",
            "OpenDrive",
        )
        candidates.append(os.path.join(fallback_open_drive_dir, f"{map_basename}.xodr"))
        if map_basename.endswith("_Opt"):
            candidates.append(os.path.join(fallback_open_drive_dir, f"{map_basename[:-4]}.xodr"))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not resolve the OpenDRIVE file needed for SUMO asset generation. "
        f"Tried: {candidates}"
    )


def _get_carla_egg_glob(carla_root: str) -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        platform_tag = "linux-x86_64" if machine in {"x86_64", "amd64"} else f"linux-{machine}"
    elif sys.platform == "win32":
        platform_tag = "win-amd64"
    else:
        platform_tag = "*"
    return os.path.join(
        str(carla_root),
        "PythonAPI",
        "carla",
        "dist",
        f"carla-*{sys.version_info.major}.{sys.version_info.minor}-{platform_tag}.egg",
    )


def _sumo_env(scenario_cfg: Mapping[str, object] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("SUMO_HOME", DEFAULT_SUMO_HOME)
    scenario_cfg = dict(scenario_cfg or {})
    carla_cfg = dict(scenario_cfg.get("carla", {}))
    carla_root = str(carla_cfg.get("carla_root", DEFAULT_CARLA_ROOT)).strip() or DEFAULT_CARLA_ROOT
    python_path_entries = list(sys.path)
    for egg_path in glob.glob(_get_carla_egg_glob(carla_root)):
        if egg_path not in python_path_entries:
            python_path_entries.insert(0, egg_path)
    existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(entry for entry in python_path_entries if entry)
    return env


def _run_subprocess(
    command: list[str],
    cwd: str | None = None,
    scenario_cfg: Mapping[str, object] | None = None,
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=_sumo_env(scenario_cfg=scenario_cfg),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "<no output>"
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}. {details}"
        )


def _write_sumocfg(asset_dir: str, map_basename: str) -> str:
    sumocfg_path = os.path.join(asset_dir, f"{map_basename}.sumocfg")
    payload = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<configuration>\n"
        "  <input>\n"
        f"    <net-file value=\"{map_basename}.net.xml\"/>\n"
        f"    <route-files value=\"{map_basename}.rou.xml\"/>\n"
        "  </input>\n"
        "  <num-clients value=\"1\"/>\n"
        "</configuration>\n"
    )
    with open(sumocfg_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return sumocfg_path


def ensure_sumo_assets(
    scenario_cfg: Mapping[str, object] | None = None,
    sumo_cfg: Mapping[str, object] | None = None,
) -> str:
    """
    Ensure the requested SUMO assets exist for the scenario and return the
    asset directory.
    """

    scenario_cfg = dict(scenario_cfg or {})
    sumo_cfg = dict(sumo_cfg or {})
    map_basename = resolve_map_basename(scenario_cfg=scenario_cfg, sumo_cfg=sumo_cfg)
    asset_root = resolve_asset_root(sumo_cfg=sumo_cfg)
    asset_dir = os.path.join(asset_root, map_basename)
    os.makedirs(asset_dir, exist_ok=True)

    sumocfg_path = os.path.join(asset_dir, f"{map_basename}.sumocfg")
    net_path = os.path.join(asset_dir, f"{map_basename}.net.xml")
    route_path = os.path.join(asset_dir, f"{map_basename}.rou.xml")
    auto_generate = bool(sumo_cfg.get("auto_generate_assets", False))

    route_generation_cfg = dict(sumo_cfg.get("route_generation", {}))
    force_regenerate_routes = bool(route_generation_cfg.get("force_regenerate", False))
    if os.path.isfile(sumocfg_path) and os.path.isfile(net_path) and os.path.isfile(route_path) and not force_regenerate_routes:
        return asset_dir

    if not auto_generate:
        raise FileNotFoundError(
            "SUMO assets are missing and auto_generate_assets is disabled. "
            f"Expected files in {asset_dir}: {os.path.basename(sumocfg_path)}, "
            f"{os.path.basename(net_path)}, {os.path.basename(route_path)}"
        )

    xodr_path = resolve_xodr_path(scenario_cfg=scenario_cfg, sumo_cfg=sumo_cfg)
    netconvert_script = os.path.join(OPENCDA_ROOT, "scripts", "netconvert_carla.py")
    random_trips_script = os.path.join(
        os.environ.get("SUMO_HOME", DEFAULT_SUMO_HOME),
        "tools",
        "randomTrips.py",
    )
    typemap_source = os.path.join(
        os.environ.get("SUMO_HOME", DEFAULT_SUMO_HOME),
        "data",
        "typemap",
        "opendriveNetconvert.typ.xml",
    )
    typemap_target = os.path.join(
        OPENCDA_ROOT,
        "scripts",
        "data",
        "opendrive_netconvert.typ.xml",
    )
    if not os.path.isfile(netconvert_script):
        raise FileNotFoundError(f"OpenCDA netconvert helper was not found: {netconvert_script}")
    if not os.path.isfile(random_trips_script):
        raise FileNotFoundError(f"SUMO randomTrips.py was not found: {random_trips_script}")
    if not os.path.isfile(typemap_target):
        if not os.path.isfile(typemap_source):
            raise FileNotFoundError(
                f"SUMO OpenDRIVE typemap file was not found: {typemap_source}"
            )
        os.makedirs(os.path.dirname(typemap_target), exist_ok=True)
        shutil.copyfile(typemap_source, typemap_target)

    if not os.path.isfile(net_path):
        _run_subprocess(
            [sys.executable, netconvert_script, xodr_path, "-o", net_path],
            cwd=OPENCDA_ROOT,
            scenario_cfg=scenario_cfg,
        )

    if force_regenerate_routes:
        for candidate in (route_path, os.path.join(asset_dir, f"{map_basename}.trips.xml")):
            if os.path.isfile(candidate):
                os.remove(candidate)

    if not os.path.isfile(route_path):
        trips_path = os.path.join(asset_dir, f"{map_basename}.trips.xml")
        trip_attributes: list[str] = []
        if bool(route_generation_cfg.get("random_depart_lane", True)):
            trip_attributes.append('departLane="random"')
        if bool(route_generation_cfg.get("random_depart_speed", True)):
            trip_attributes.append('departSpeed="max"')
        command = [
            sys.executable,
            random_trips_script,
            "-n",
            net_path,
            "-o",
            trips_path,
            "-r",
            route_path,
            "--seed",
            str(int(route_generation_cfg.get("seed", 42))),
            "--period",
            str(float(route_generation_cfg.get("period_s", 2.0))),
            "--end",
            str(float(route_generation_cfg.get("end_time_s", 3600.0))),
            "--fringe-factor",
            str(float(route_generation_cfg.get("fringe_factor", 5.0))),
            "--vehicle-class",
            str(route_generation_cfg.get("vehicle_class", "passenger")),
        ]
        if len(trip_attributes) > 0:
            command.extend(["--trip-attributes", " ".join(trip_attributes)])
        if bool(route_generation_cfg.get("random_depart_pos", True)):
            command.append("--random-departpos")
        if bool(route_generation_cfg.get("random_depart", True)):
            command.append("--random-depart")
        if bool(route_generation_cfg.get("validate", True)):
            command.append("--validate")
        _run_subprocess(command, cwd=asset_dir, scenario_cfg=scenario_cfg)

    if not os.path.isfile(sumocfg_path):
        _write_sumocfg(asset_dir=asset_dir, map_basename=map_basename)

    return asset_dir
