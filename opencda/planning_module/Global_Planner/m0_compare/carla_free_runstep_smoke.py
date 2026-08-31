#!/usr/bin/env python3
"""End-to-end CARLA-free smoke: construct CPXMPCPlannerBridge with a
CustomGlobalPlannerAdapter as map_planner and a minimal fake vehicle_manager,
set a destination, push tick inputs via update_information(), and call
run_step() -> a VehicleControl. No `import carla`.

  python carla_free_runstep_smoke.py [case-substring]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
GP_DIR = HERE.parent
PM_DIR = GP_DIR.parent
REPO_ROOT = PM_DIR.parent.parent
for p in (str(GP_DIR), str(PM_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
MAPS = GP_DIR / "maps"

from utility.global_planner import CustomGlobalPlannerAdapter  # noqa: E402
from utility.carla_compat import carla, CARLA_AVAILABLE  # noqa: E402
from opencda.planning_module.opencda_bridge.cpx_mpc_planner import (  # noqa: E402
    CPXMPCPlannerBridge,
)

import json  # noqa: E402

CASES = json.loads((HERE / "cases.json").read_text())["cases"]


def _fake_vehicle_manager(ego_xy, yaw_rad, sim_time_s=0.0):
    ex, ey = ego_xy
    world = SimpleNamespace(
        get_snapshot=lambda: SimpleNamespace(
            timestamp=SimpleNamespace(elapsed_seconds=float(sim_time_s))
        ),
        debug=SimpleNamespace(
            draw_point=lambda *a, **k: None,
            draw_line=lambda *a, **k: None,
            draw_string=lambda *a, **k: None,
            draw_arrow=lambda *a, **k: None,
        ),
    )
    vehicle = SimpleNamespace(
        id=1,
        bounding_box=SimpleNamespace(
            extent=SimpleNamespace(x=2.45, y=1.02, z=0.75)
        ),
        get_world=lambda: world,
        get_transform=lambda: carla.Transform(
            carla.Location(ex, ey, 0.0), carla.Rotation(yaw=math.degrees(yaw_rad))
        ),
        get_location=lambda: carla.Location(ex, ey, 0.0),
    )
    return SimpleNamespace(
        vehicle=vehicle,
        carla_map=None,
        localizer=SimpleNamespace(
            get_ego_pos=lambda: carla.Transform(
                carla.Location(ex, ey, 0.0),
                carla.Rotation(yaw=math.degrees(yaw_rad)),
            ),
            get_ego_spd=lambda: 0.0,
        ),
        v2x_manager=None,
        perception_manager=SimpleNamespace(objects={}),
        agent=None,
    )


CONFIG = {
    "enabled": True,
    "type": "cpx_mpc",
    "mode": "full_cpx_mpc",
    "fallback_policy": "emergency_stop",
    "target_speed_mps": 8.0,
    "global_planner_mode": "dij",
}


def run_case(name, mp, start, goal):
    adapter = CustomGlobalPlannerAdapter(
        xodr_path=str(MAPS / f"{mp}.xodr"),
        cache_root=str(HERE / "_dij_cache"),
        route_sample_distance_m=1.0,
    )
    adapter.load()
    route = adapter.trace_route(
        {"x": start[0], "y": start[1], "z": start[2]},
        {"x": goal[0], "y": goal[1], "z": goal[2]},
        replace_stored_route=True,
    )
    poly = [[float(p[0]), float(p[1])] for p in route.route_waypoints]
    ex, ey = poly[0]
    nx, ny = poly[min(3, len(poly) - 1)]
    yaw = math.atan2(ny - ey, nx - ex)

    cfg = dict(CONFIG)
    cfg["global_planner_xodr_path"] = str(MAPS / f"{mp}.xodr")
    vm = _fake_vehicle_manager((ex, ey), yaw)
    bridge = CPXMPCPlannerBridge(vm, cfg, map_planner=adapter)
    bridge.set_destination(
        start_location=carla.Location(start[0], start[1], start[2]),
        end_location=carla.Location(goal[0], goal[1], goal[2]),
    )
    rm = bridge.route_manager
    route_found = bool(getattr(rm._active_route_summary, "route_found", False))
    route_dbg = rm.carla_route_debug_reason
    if not route_found:
        raise AssertionError(f"{name}: route not found (debug={route_dbg})")
    if route_dbg != "inhouse_route_ready":
        raise AssertionError(
            f"{name}: expected in-house route entries, got debug={route_dbg}"
        )

    controls = []
    fallback_ticks = 0
    dt = 0.1
    n_ticks = min(12, len(poly) - 1)
    for k in range(n_ticks):
        i = min(k, len(poly) - 2)
        ex, ey = poly[i]
        nx, ny = poly[i + 1]
        yaw = math.atan2(ny - ey, nx - ex)
        bridge.vehicle_manager = _fake_vehicle_manager((ex, ey), yaw, sim_time_s=k * dt)
        bridge.update_information(
            ego_transform=carla.Transform(
                carla.Location(ex, ey, 0.0),
                carla.Rotation(yaw=math.degrees(yaw)),
            ),
            ego_speed_kmh=5.0,
            detected_objects=[],
        )
        control = bridge.run_step()
        controls.append(control)
        for attr in ("throttle", "brake", "steer"):
            v = float(getattr(control, attr, float("nan")))
            if not math.isfinite(v):
                raise AssertionError(f"{name}: tick {k} control.{attr} not finite ({v})")
        if bool((getattr(bridge, "last_debug", {}) or {}).get("fallback_active", False)):
            fallback_ticks += 1
    last = controls[-1]
    dbg = getattr(bridge, "last_debug", {}) or {}
    return {
        "name": name,
        "ticks": len(controls),
        "fallback_ticks": fallback_ticks,
        "fallback_active": bool(dbg.get("fallback_active", False)),
        "fallback_reason": dbg.get("fallback_reason", ""),
        "last_control": {
            "throttle": round(float(getattr(last, "throttle", 0.0)), 3),
            "brake": round(float(getattr(last, "brake", 0.0)), 3),
            "steer": round(float(getattr(last, "steer", 0.0)), 3),
        },
    }


def main():
    print(f"CARLA_AVAILABLE={CARLA_AVAILABLE}")
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    cases = [c for c in CASES if not want or want in c["name"]]
    n_ok = 0
    for c in cases:
        try:
            r = run_case(c["name"], c["map"], c["start"], c["goal"])
            print(f"[ok  ] {r['name']:30s} ticks={r['ticks']} "
                  f"fallback_ticks={r['fallback_ticks']} control={r['last_control']}")
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[FAIL] {c['name']:30s} {type(exc).__name__}: {exc}")
            print("       " + "\n       ".join(traceback.format_exc().splitlines()[-8:]))
    print(f"\n{n_ok}/{len(cases)} cases construct + run_step CARLA-free")
    sys.exit(0 if n_ok == len(cases) else 1)


if __name__ == "__main__":
    main()
