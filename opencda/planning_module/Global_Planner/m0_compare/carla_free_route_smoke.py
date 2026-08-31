#!/usr/bin/env python3
"""Smoke test: run the route layer with the in-house global planner only,
CARLA map/api set to None, and check every query the trajectory pipeline
makes still returns sane values.

  python carla_free_route_smoke.py            # all cases in cases.json
  python carla_free_route_smoke.py sc6        # one case by substring

Exercises: CustomGlobalPlannerAdapter + CPXRouteManager(carla_map=None,
carla_api=None) + ReferenceGenerator with adapter-backed map callbacks.
No `import carla` anywhere in this path.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GP_DIR = HERE.parent
PM_DIR = GP_DIR.parent
REPO_ROOT = PM_DIR.parent.parent
for p in (str(GP_DIR), str(PM_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
MAPS = GP_DIR / "maps"

from utility.global_planner import (  # noqa: E402
    CustomGlobalPlannerAdapter,
    canonical_lane_id_for_waypoint,
)
from utility.carla_compat import carla as _carla, CARLA_AVAILABLE  # noqa: E402
from pipeline.route_manager import CPXRouteManager  # noqa: E402
from pipeline.reference_generator import ReferenceGenerator  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _body_frame_xy(*, origin_x_m, origin_y_m, heading_rad, target_x_m, target_y_m):
    dx = float(target_x_m) - float(origin_x_m)
    dy = float(target_y_m) - float(origin_y_m)
    ch, sh = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    return dx * ch + dy * sh, -dx * sh + dy * ch


def run_reference_case(adapter, name, poly):
    """Drive ReferenceGenerator CARLA-free: adapter as map_planner, adapter
    waypoint queries as the map callbacks."""
    def _wp(loc):
        return adapter.get_waypoint({"x": loc.x, "y": loc.y, "z": loc.z})

    def _lane_id(loc):
        w = _wp(loc)
        return int(canonical_lane_id_for_waypoint(w) or 1) if w is not None else 1

    rg = ReferenceGenerator(
        config={"lane_follow_reference_first_point_m": 2.0},
        mpc=SimpleNamespace(dt_s=0.1, horizon_steps=20),
        map_planner=adapter,
        map_waypoint_from_location=_wp,
        lane_id_at_location=_lane_id,
        body_frame_xy=_body_frame_xy,
        target_speed_mps=8.0,
        lookahead_m=40.0,
        drivable_waypoint_from_location=lambda loc: adapter.get_drivable_waypoint(
            {"x": loc.x, "y": loc.y, "z": loc.z}
        ),
    )
    problems = []
    idxs = [0, len(poly) // 3, 2 * len(poly) // 3]
    for i in idxs:
        ex, ey = poly[i]
        nx, ny = poly[min(i + 1, len(poly) - 1)]
        yaw = math.atan2(ny - ey, nx - ex)
        loc = _carla.Location(x=ex, y=ey, z=0.0)
        gr = rg.build_route_reference(ego_location=loc, ego_yaw_rad=yaw, speed_ref_mps=8.0)
        s = gr.samples
        if len(s) < 2:
            problems.append(f"i={i}: build_route_reference -> {len(s)} samples")
            continue
        d0 = math.hypot(float(s[0]["x_ref_m"]) - ex, float(s[0]["y_ref_m"]) - ey)
        if d0 > 20.0:
            problems.append(f"i={i}: first route sample {d0:.1f} m from ego")
        if not all(math.isfinite(float(row.get("heading_rad", 0.0))) for row in s):
            problems.append(f"i={i}: non-finite heading in route reference")
        lc = rg.lane_center_samples(
            start_waypoint=_wp(loc), current_lane_id=_lane_id(loc),
            horizon_steps=20, step_distance_m=2.0)
        if len(lc) < 2:
            problems.append(f"i={i}: lane_center_samples -> {len(lc)}")
    return problems


def _sane_reference(samples, ego_xy):
    """A local reference must be non-empty and start near/ahead of ego."""
    if not samples:
        return False, "empty reference"
    first = samples[0]
    fx = first.get("x_m", first.get("x"))
    fy = first.get("y_m", first.get("y"))
    if fx is None or fy is None:
        return False, f"reference sample missing xy: {list(first)[:6]}"
    d = math.hypot(float(fx) - ego_xy[0], float(fy) - ego_xy[1])
    return (d < 30.0), f"first sample {d:.1f} m from ego"


def run_case(adapter, name, start, goal):
    out = {"name": name}
    rm = CPXRouteManager(global_planner=adapter, carla_map=None, carla_api=None)
    summary = rm.set_destination(
        start_point={"x": start[0], "y": start[1], "z": start[2]},
        goal_point={"x": goal[0], "y": goal[1], "z": goal[2]},
    )
    out["route_found"] = bool(getattr(summary, "route_found", False))
    rw = list(getattr(summary, "route_waypoints", []) or [])
    out["route_points"] = len(rw)
    out["length_m"] = round(float(getattr(summary, "distance_to_destination_m", 0.0) or 0.0), 1)
    if len(rw) < 2:
        out["ok"] = False
        out["fail"] = f"route has {len(rw)} points; debug={rm.carla_route_debug_reason}"
        return out

    # walk synthetic ego poses along the planned polyline
    poly = [(float(p[0]), float(p[1])) for p in rw]
    idx_samples = list(range(0, len(poly), max(1, len(poly) // 12)))
    problems = []
    remaining_seq = []
    lane_seq = []
    for i in idx_samples:
        ex, ey = poly[i]
        nx, ny = poly[min(i + 1, len(poly) - 1)]
        heading = math.atan2(ny - ey, nx - ex)
        info = rm.get_route_info(x_m=ex, y_m=ey, query_key=f"{name}:{i}",
                                 fallback_lane_id=1)
        if not info.get("route_found"):
            problems.append(f"i={i}: route_info route_found=False ({info.get('debug_reason')})")
        remaining_seq.append(float(info.get("remaining_distance_m", 0.0) or 0.0))
        lane_seq.append(int(info.get("optimal_lane_id", 0) or 0))

        gpts = rm.geometry_route_points(x_m=ex, y_m=ey, query_key=f"g{i}")
        if len(gpts) < 2:
            problems.append(f"i={i}: geometry_route_points returned {len(gpts)}")

        try:
            ref, reason = rm.carla_waypoint_reference(
                ego_x_m=ex, ego_y_m=ey, ego_heading_rad=heading,
                horizon_steps=15, step_distance_m=2.0, target_speed_mps=8.0,
                fallback_lane_id=1)
            ok, why = _sane_reference(
                [s if isinstance(s, dict) else {"x_m": s[0], "y_m": s[1]} for s in (ref or [])],
                (ex, ey))
            if not ok and i < idx_samples[-1]:
                problems.append(f"i={i}: reference not sane ({why}; reason={reason})")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"i={i}: carla_waypoint_reference raised {type(exc).__name__}: {exc}")

        try:
            rm.upcoming_turn(ego_x_m=ex, ego_y_m=ey, ego_heading_rad=heading,
                             lookahead_m=40.0)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"i={i}: upcoming_turn raised {type(exc).__name__}: {exc}")

    # remaining distance should be (weakly) monotonically decreasing
    monotonic = all(a >= b - 5.0 for a, b in zip(remaining_seq, remaining_seq[1:]))
    if not monotonic:
        problems.append(f"remaining_distance not decreasing: {[round(r) for r in remaining_seq]}")
    out["remaining_seq"] = [round(r) for r in remaining_seq]
    out["optimal_lane_seq"] = lane_seq

    # behavior -> reference layer: ReferenceGenerator driven by the adapter
    try:
        ref_problems = run_reference_case(adapter, name, poly)
        problems.extend(ref_problems)
    except Exception as exc:  # noqa: BLE001
        import traceback
        problems.append(f"ReferenceGenerator raised {type(exc).__name__}: {exc}")
        out["ref_traceback"] = traceback.format_exc().splitlines()[-6:]

    out["ok"] = not problems
    if problems:
        out["problems"] = problems
    return out


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    if want:
        cases = [c for c in cases if want in c["name"]]
    adapters = {}
    results = []
    for c in cases:
        mp = c["map"]
        if mp not in adapters:
            a = CustomGlobalPlannerAdapter(
                xodr_path=str(MAPS / f"{mp}.xodr"),
                cache_root=str(HERE / "_dij_cache"),
                route_sample_distance_m=1.0)
            a.load()
            adapters[mp] = a
        r = run_case(adapters[mp], c["name"], c["start"], c["goal"])
        results.append(r)
        tag = "ok  " if r.get("ok") else "FAIL"
        print(f"[{tag}] {r['name']:30s} found={r.get('route_found')} "
              f"pts={r.get('route_points')} len={r.get('length_m')} "
              f"lanes={r.get('optimal_lane_seq')}")
        for p in r.get("problems", []):
            print(f"        - {p}")
        if r.get("fail"):
            print(f"        - {r['fail']}")
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(results)} cases pass CARLA-free route+reference smoke test")
    (HERE / "carla_free_route_smoke.json").write_text(json.dumps(results, indent=2))
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
