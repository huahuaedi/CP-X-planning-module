#!/usr/bin/env python3
"""M0: run ONE global-planner backend over the comparison cases and dump a
normalized route fingerprint per case as JSON.

Run this once per backend, in the environment where that backend works:

  # custom AD-map / Dijkstra backend (new Python 3.10-3.13 env)
  #   - either a built map_repo/install, or `pip install ad-map-access` + --admap-from-import
  python run_backend.py --backend dij  --out out_dij.json  [--admap-from-import]

  # legacy CARLA GlobalRoutePlanner + A* backend (old carla307 / Python 3.7 env)
  python run_backend.py --backend astar --out out_astar.json --carla-egg /path/to/carla.egg

Then diff the two result files with diff_backends.py. Neither backend is the
reference; the diff surfaces where they disagree so the AD-map topology can be
fixed before `global_planner_mode` is flipped to `dij` by default.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLOBAL_PLANNER_DIR = HERE.parent
PLANNING_MODULE_DIR = GLOBAL_PLANNER_DIR.parent
REPO_ROOT = PLANNING_MODULE_DIR.parent.parent
MAPS_DIR = GLOBAL_PLANNER_DIR / "maps"


# --------------------------------------------------------------------------- #
# route fingerprint (backend independent)
# --------------------------------------------------------------------------- #
def _dedupe_consecutive(seq):
    out = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def fingerprint_from_points(points, lane_triplets, road_options, spacing_m,
                            length_m=None, transitions=None):
    """points: list[[x,y]]; lane_triplets: list[[road,section,lane]] aligned to
    points (None entries allowed); road_options: list[str] aligned to points."""
    n = len(points)
    if n < 2:
        return {"ok": False, "error": "route has < 2 points", "n_samples": n}

    seg_len = [math.hypot(points[i + 1][0] - points[i][0],
                          points[i + 1][1] - points[i][1]) for i in range(n - 1)]
    total = float(length_m) if length_m is not None else float(sum(seg_len))
    max_gap = max(seg_len) if seg_len else 0.0
    gap_thr = 2.5 * float(spacing_m)
    gap_idx = [i for i, d in enumerate(seg_len) if d > gap_thr]

    lane_seq = _dedupe_consecutive([tuple(t) if t is not None else None
                                    for t in lane_triplets])
    opt_seq = _dedupe_consecutive([str(o).upper() for o in road_options]) if road_options else []

    # junction runs: maximal spans where lane triplet section marks intersection
    # is not available here, so approximate via road_option CHANGELANE / geometry.
    n_lane_changes = 0
    if transitions:
        n_lane_changes = sum(1 for t in transitions
                             if "CHANGE" in str(t).upper() or "LANE_CHANGE" in str(t).upper())
    if n_lane_changes == 0 and opt_seq:
        n_lane_changes = sum(1 for o in opt_seq if "CHANGELANE" in o.replace("_", ""))

    # lane-id renumber events: lane triplet changes where road_id stays the same
    # and there is NO lane change nearby (heuristic: consecutive dedup where only
    # section or lane id moves while road id is constant).
    renumber = 0
    prev = None
    for t in lane_seq:
        if t is None or prev is None:
            prev = t
            continue
        if t[0] == prev[0] and (t[1] != prev[1] or t[2] != prev[2]):
            renumber += 1
        prev = t

    # coarse polyline for eyeballing / plotting: endpoints + every ~5 m
    keep_every = max(1, int(round(5.0 / max(0.5, spacing_m))))
    coarse = [points[0]]
    acc = 0.0
    for i in range(1, n):
        acc += seg_len[i - 1]
        if i % keep_every == 0 or i == n - 1:
            coarse.append([round(points[i][0], 2), round(points[i][1], 2)])
    coarse[0] = [round(points[0][0], 2), round(points[0][1], 2)]

    return {
        "ok": True,
        "error": "",
        "length_m": round(total, 2),
        "n_samples": n,
        "spacing_m": round(float(spacing_m), 3),
        "lane_seq": [list(t) if t is not None else None for t in lane_seq],
        "lane_seq_len": len(lane_seq),
        "road_option_seq": opt_seq,
        "n_lane_changes": int(n_lane_changes),
        "max_gap_m": round(max_gap, 3),
        "gap_violations": len(gap_idx),
        "gap_violation_at_m": [round(sum(seg_len[:i]), 1) for i in gap_idx][:20],
        "lane_id_renumber_events": int(renumber),
        "coarse_polyline": coarse,
    }


# --------------------------------------------------------------------------- #
# backend: dij  (custom AD-map / OpenDRIVE planner, via the production adapter)
# --------------------------------------------------------------------------- #
def run_dij(cases, spacing_m, admap_from_import):
    # Route through CustomGlobalPlannerAdapter (what route_manager.py uses) so
    # road_options carry the same lane-aware macro labels production sees.
    # runtime.import_ad_map_access() now falls back to an importable
    # `ad_map_access` (pip wheel) on its own, so --admap-from-import is a no-op
    # kept only for explicitness.
    del admap_from_import
    sys.path.insert(0, str(GLOBAL_PLANNER_DIR))
    sys.path.insert(0, str(PLANNING_MODULE_DIR))
    from utility.global_planner import CustomGlobalPlannerAdapter

    cache_root = HERE / "_dij_cache"
    results = {}
    planners = {}
    try:
        for case in cases:
            name, mp = case["name"], case["map"]
            xodr = MAPS_DIR / f"{mp}.xodr"
            try:
                if mp not in planners:
                    a = CustomGlobalPlannerAdapter(
                        xodr_path=str(xodr), cache_root=str(cache_root),
                        route_sample_distance_m=spacing_m)
                    a.load()
                    planners[mp] = a
                a = planners[mp]
                s = {"x": case["start"][0], "y": case["start"][1], "z": case["start"][2]}
                g = {"x": case["goal"][0], "y": case["goal"][1], "z": case["goal"][2]}
                summary = a.trace_route(s, g, replace_stored_route=True)
                if not getattr(summary, "route_found", False):
                    raise RuntimeError(getattr(summary, "debug_reason", "route not found"))
                entries = a.get_dense_route_entries()
                ws = [e["waypoint"] for e in entries]
                opts = [str(e["road_option"]).upper() for e in entries]
                pts = [[w.position["x"], w.position["y"]] for w in ws]
                lanes = [[w.road_id, w.section_id, w.lane_id] for w in ws]
                fp = fingerprint_from_points(
                    pts, lanes, opts, spacing_m,
                    length_m=getattr(summary, "distance_to_destination_m", None))
                fp["n_junction_samples"] = sum(
                    1 for w in ws if getattr(w, "is_intersection", False))
                fp["optimal_lane_id"] = getattr(summary, "optimal_lane_id", None)
                fp["next_macro_maneuver"] = getattr(summary, "next_macro_maneuver", None)
                results[name] = fp
                print(f"[dij] {name:32s} ok  len={fp.get('length_m')}  "
                      f"lanes={fp.get('lane_seq_len')}  opts={fp.get('road_option_seq')}  "
                      f"gapviol={fp.get('gap_violations')}")
            except Exception as exc:  # noqa: BLE001
                results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[dij] {name:32s} FAIL  {type(exc).__name__}: {exc}")
    finally:
        for a in planners.values():
            try:
                a.close()
            except Exception:  # noqa: BLE001
                pass
    return results


# --------------------------------------------------------------------------- #
# backend: astar  (CARLA GlobalRoutePlanner + legacy A*, via the prod factory)
# --------------------------------------------------------------------------- #
def _astar_setup_diag(carla, carla_root):
    diag = {}
    diag["carla_version"] = getattr(carla, "__version__", "?")
    diag["carla_file"] = getattr(carla, "__file__", "?")
    diag["carla_root"] = str(carla_root)
    diag["sys_path_has_agents_dir"] = next(
        (p for p in sys.path if os.path.isdir(os.path.join(p, "agents"))), None)
    try:
        import networkx  # noqa: F401
        diag["networkx"] = True
    except Exception as exc:  # noqa: BLE001
        diag["networkx"] = f"MISSING: {exc}"
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: F401
        diag["agents_grp_import"] = True
    except Exception as exc:  # noqa: BLE001
        diag["agents_grp_import"] = f"FAIL: {type(exc).__name__}: {exc}"
    return diag


def run_astar(cases, spacing_m, carla_egg, carla_root, agents_path):
    import os
    import glob as _glob
    carla_root = carla_root or os.environ.get("CARLA_ROOT", "/home/umd-user/carla_source/carla")
    os.environ.setdefault("CARLA_ROOT", str(carla_root))
    if carla_egg:
        sys.path.append(carla_egg)
    # `agents.navigation.global_route_planner` ships in CARLA's PythonAPI/carla,
    # not in the egg/wheel. Try, in order: --agents-path, <carla_root>/PythonAPI/carla,
    # any PythonAPI/carla under carla_root.
    agents_candidates = []
    if agents_path:
        agents_candidates.append(str(agents_path))
    agents_candidates.append(os.path.join(str(carla_root), "PythonAPI", "carla"))
    agents_candidates += _glob.glob(os.path.join(str(carla_root), "**", "PythonAPI", "carla"),
                                    recursive=True)
    for cand in agents_candidates:
        if os.path.isdir(os.path.join(cand, "agents")) and cand not in sys.path:
            sys.path.append(cand)
            break
    import carla  # noqa: F401  (legacy env only)

    for extra in (PLANNING_MODULE_DIR, PLANNING_MODULE_DIR.parent.parent):
        sys.path.insert(0, str(extra))
    from utility.carla_lane_graph import build_lane_center_waypoints
    from utility.legacy_global_planner import (
        AStarGlobalPlanner, _get_carla_route_planner_load_error)

    results = {"_diag": _astar_setup_diag(carla, carla_root)}
    print("[astar] setup:", json.dumps(results["_diag"]))
    planners = {}
    for case in cases:
        name, mp = case["name"], case["map"]
        try:
            if mp not in planners:
                xodr = (MAPS_DIR / f"{mp}.xodr").read_text()
                cmap = carla.Map(mp, xodr)  # offline; no server needed
                gw = cmap.generate_waypoints(2.0)
                lane_wps, _road_cfg = build_lane_center_waypoints(
                    map_obj=cmap, carla=carla, sample_distance_m=spacing_m)
                pl = AStarGlobalPlanner(
                    lane_center_waypoints=lane_wps, world_map=cmap,
                    route_sample_distance_m=spacing_m)
                planners[mp] = pl
                s0 = case["start"]
                w0 = cmap.get_waypoint(carla.Location(x=s0[0], y=s0[1], z=s0[2]))
                results.setdefault("_diag", {})[f"map:{mp}"] = {
                    "generate_waypoints_2m": len(list(gw)),
                    "lane_center_waypoints": len(lane_wps),
                    "carla_route_planner_ready": pl._carla_route_planner is not None,
                    "carla_route_planner_load_error": _get_carla_route_planner_load_error(),
                    "start_get_waypoint": None if w0 is None else {
                        "road_id": int(w0.road_id), "lane_id": int(w0.lane_id),
                        "s": round(float(getattr(w0, "s", 0.0)), 1),
                        "is_junction": bool(w0.is_junction)},
                }
                print(f"[astar] map {mp}: {json.dumps(results['_diag'][f'map:{mp}'])}")
            planner = planners[mp]
            summary = planner.plan_route_from_locations(
                start_location={"x": case["start"][0], "y": case["start"][1], "z": case["start"][2]},
                goal_location={"x": case["goal"][0], "y": case["goal"][1], "z": case["goal"][2]},
                replace_stored_route=True)
            rw = list(getattr(summary, "route_waypoints", []) or [])
            pts = [[float(p[0]), float(p[1])] for p in rw]
            opts = list(getattr(summary, "road_options", []) or [])
            opts += ["LANEFOLLOW"] * (len(pts) - len(opts))
            lanes = [None] * len(pts)  # summary has no per-point lane triplet
            fp = fingerprint_from_points(pts, lanes, opts, spacing_m,
                                         length_m=getattr(summary, "distance_to_destination_m", None))
            fp["route_found"] = bool(getattr(summary, "route_found", False))
            fp["start_lane_id"] = getattr(summary, "start_lane_id", None)
            fp["goal_lane_id"] = getattr(summary, "goal_lane_id", None)
            for attr in ("debug_reason", "debug_message", "failure_reason"):
                if hasattr(summary, attr):
                    fp[attr] = getattr(summary, attr)
            if not fp["route_found"] and not fp.get("error"):
                fp["ok"] = False
                fp["error"] = fp.get("debug_reason") or fp.get("debug_message") or "route_found=false"
            results[name] = fp
            print(f"[astar] {name:32s} {'ok ' if fp.get('ok') else 'FAIL'} "
                  f"len={fp.get('length_m')} found={fp.get('route_found')} "
                  f"err={fp.get('error','')}")
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[astar] {name:32s} FAIL  {type(exc).__name__}: {exc}")
    return results


# --------------------------------------------------------------------------- #
# backend: carla_grp  (the vendored opencda.core.plan GlobalRoutePlanner --
# exactly what route_manager.py::_build_carla_route uses to drive geometry today)
# --------------------------------------------------------------------------- #
def run_carla_grp(cases, spacing_m, carla_egg):
    if carla_egg:
        sys.path.append(carla_egg)
    import carla
    sys.path.insert(0, str(REPO_ROOT))
    from opencda.core.plan.global_route_planner import GlobalRoutePlanner
    from opencda.core.plan.global_route_planner_dao import GlobalRoutePlannerDAO

    results = {"_diag": {"carla_file": getattr(carla, "__file__", "?"),
                         "carla_version": getattr(carla, "__version__", "?")}}
    print("[carla_grp] setup:", json.dumps(results["_diag"]))
    planners = {}
    for case in cases:
        name, mp = case["name"], case["map"]
        try:
            if mp not in planners:
                cmap = carla.Map(mp, (MAPS_DIR / f"{mp}.xodr").read_text())
                dao = GlobalRoutePlannerDAO(cmap, spacing_m)
                grp = GlobalRoutePlanner(dao)
                grp.setup()
                planners[mp] = grp
            grp = planners[mp]
            s, g = case["start"], case["goal"]
            route = grp.trace_route(carla.Location(x=s[0], y=s[1], z=s[2]),
                                    carla.Location(x=g[0], y=g[1], z=g[2])) or []
            pts, opts, lanes = [], [], []
            for wp, ro in route:
                loc = wp.transform.location
                pts.append([float(loc.x), float(loc.y)])
                opts.append(getattr(ro, "name", str(ro)))
                lanes.append([int(wp.road_id), int(wp.section_id), int(wp.lane_id)])
            fp = fingerprint_from_points(pts, lanes, opts, spacing_m)
            results[name] = fp
            print(f"[carla_grp] {name:32s} {'ok ' if fp.get('ok') else 'FAIL'} "
                  f"len={fp.get('length_m')} n={fp.get('n_samples')} "
                  f"LC={fp.get('n_lane_changes')} gapmax={fp.get('max_gap_m')} "
                  f"err={fp.get('error','')}")
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[carla_grp] {name:32s} FAIL  {type(exc).__name__}: {exc}")
    return results


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["dij", "astar", "carla_grp"])
    ap.add_argument("--cases", default=str(HERE / "cases.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--spacing-m", type=float, default=1.0)
    ap.add_argument("--admap-from-import", action="store_true",
                    help="dij: bypass map_repo/install and use an importable ad_map_access")
    ap.add_argument("--carla-egg", default=None, help="astar: path to append for `import carla`")
    ap.add_argument("--carla-root", default=None,
                    help="astar: CARLA_ROOT (for PythonAPI/carla -> agents.navigation). "
                         "Defaults to $CARLA_ROOT or /home/umd-user/carla_source/carla")
    ap.add_argument("--agents-path", default=None,
                    help="astar: dir that directly contains `agents/` "
                         "(CARLA's PythonAPI/carla). Overrides --carla-root discovery.")
    args = ap.parse_args()

    doc = json.loads(Path(args.cases).read_text())
    cases = doc["cases"]
    out = Path(args.out or (HERE / f"out_{args.backend}.json"))

    if args.backend == "dij":
        results = run_dij(cases, args.spacing_m, args.admap_from_import)
    elif args.backend == "carla_grp":
        results = run_carla_grp(cases, args.spacing_m, args.carla_egg)
    else:
        results = run_astar(cases, args.spacing_m, args.carla_egg,
                            args.carla_root, args.agents_path)

    payload = {
        "backend": args.backend,
        "spacing_m": args.spacing_m,
        "n_cases": len(cases),
        "n_ok": sum(1 for k, r in results.items()
                    if k != "_diag" and isinstance(r, dict) and r.get("ok")),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({payload['n_ok']}/{payload['n_cases']} ok)")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
