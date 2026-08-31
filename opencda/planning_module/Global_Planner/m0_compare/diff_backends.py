#!/usr/bin/env python3
"""M0: diff two backend result files produced by run_backend.py and emit a
markdown report ranking the cases where `dij` and `astar` disagree.

  python diff_backends.py --dij out_dij.json --astar out_astar.json --out M0_REPORT.md

Neither backend is treated as ground truth. The report is the input to fixing
AD-map junction / lane-change topology before `global_planner_mode: dij`
becomes the default.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# thresholds for the per-case verdict
LEN_PCT_MINOR = 5.0
LEN_PCT_DIVERGENT = 15.0
LATERAL_MINOR_M = 2.0
LATERAL_DIVERGENT_M = 5.0


def _resample(poly, step_m=2.0):
    if len(poly) < 2:
        return poly
    out = [poly[0]]
    carry = 0.0
    for i in range(1, len(poly)):
        ax, ay = poly[i - 1]
        bx, by = poly[i]
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-9:
            continue
        t = carry
        while t < seg:
            r = t / seg
            out.append([ax + (bx - ax) * r, ay + (by - ay) * r])
            t += step_m
        carry = t - seg
    out.append(poly[-1])
    return out


def _max_lateral(poly_a, poly_b):
    """Symmetric max nearest-point distance between two polylines (m)."""
    a = _resample(poly_a)
    b = _resample(poly_b)
    if len(a) < 2 or len(b) < 2:
        return None

    def one_way(src, dst):
        worst = 0.0
        for px, py in src:
            best = min(math.hypot(px - qx, py - qy) for qx, qy in dst)
            worst = max(worst, best)
        return worst

    return round(max(one_way(a, b), one_way(b, a)), 2)


def _junction_count(fp):
    # dij: has n_junction_samples>0 broken into runs via road_option_seq
    # astar: road_option_seq contains INTERSECTION/junction-ish tokens
    opts = fp.get("road_option_seq", []) or []
    runs = sum(1 for o in opts if "INTERSECTION" in o or "JUNCTION" in o)
    if runs:
        return runs
    if fp.get("n_junction_samples", 0):
        return 1
    return 0


def verdict(dij, astar):
    if not dij.get("ok") and not astar.get("ok"):
        return "BOTH-FAIL", "both backends failed to trace a route"
    if not dij.get("ok"):
        return "DIJ-FAIL", dij.get("error", "dij failed")
    if not astar.get("ok"):
        return "ASTAR-FAIL", astar.get("error", "astar failed")

    la = dij["length_m"]
    lb = astar["length_m"]
    len_pct = abs(la - lb) / max(1.0, min(la, lb)) * 100.0
    lateral = _max_lateral(dij.get("coarse_polyline", []), astar.get("coarse_polyline", []))
    lc_delta = abs(int(dij.get("n_lane_changes", 0)) - int(astar.get("n_lane_changes", 0)))
    dij_gapv = int(dij.get("gap_violations", 0))
    astar_gapv = int(astar.get("gap_violations", 0))

    reasons = []
    score = 0
    if lateral is not None and lateral >= LATERAL_DIVERGENT_M:
        score = max(score, 2); reasons.append(f"corridor differs by {lateral} m")
    elif lateral is not None and lateral >= LATERAL_MINOR_M:
        score = max(score, 1); reasons.append(f"corridor differs by {lateral} m")
    if len_pct >= LEN_PCT_DIVERGENT:
        score = max(score, 2); reasons.append(f"length differs {len_pct:.0f}%")
    elif len_pct >= LEN_PCT_MINOR:
        score = max(score, 1); reasons.append(f"length differs {len_pct:.0f}%")
    if lc_delta:
        score = max(score, 2 if lc_delta > 1 else 1)
        reasons.append(
            f"lane-change count differs by {lc_delta} "
            f"(dij {dij.get('n_lane_changes', 0)} vs carla {astar.get('n_lane_changes', 0)})")
    if dij_gapv:
        score = max(score, 1); reasons.append(f"{dij_gapv} dij route-point gap(s)")
    if astar_gapv:
        score = max(score, 1); reasons.append(f"{astar_gapv} carla route-point gap(s)")

    label = ["MATCH", "MINOR", "DIVERGENT"][score]
    return label, "; ".join(reasons) if reasons else "routes agree within thresholds"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dij", default=str(HERE / "out_dij.json"))
    ap.add_argument("--astar", default=str(HERE / "out_astar.json"))
    ap.add_argument("--out", default=str(HERE / "M0_REPORT.md"))
    args = ap.parse_args()

    dij = json.loads(Path(args.dij).read_text())["results"]
    astar = json.loads(Path(args.astar).read_text())["results"]
    astar_diag = astar.get("_diag", {})
    dij = {k: v for k, v in dij.items() if not k.startswith("_")}
    astar = {k: v for k, v in astar.items() if not k.startswith("_")}
    names = list(dict.fromkeys(list(dij) + list(astar)))

    rows = []
    for name in names:
        d = dij.get(name, {"ok": False, "error": "missing from dij run"})
        a = astar.get(name, {"ok": False, "error": "missing from astar run"})
        label, why = verdict(d, a)
        rows.append((name, label, why, d, a))

    order = {"DIVERGENT": 0, "DIJ-FAIL": 0, "ASTAR-FAIL": 0, "BOTH-FAIL": 0,
             "MINOR": 1, "MATCH": 2}
    rows.sort(key=lambda r: (order.get(r[1], 3), r[0]))

    out = []
    out.append("# M0 — dij vs astar global-route comparison\n")
    if astar_diag:
        out.append("## astar setup diagnostics\n")
        out.append("```json")
        out.append(json.dumps(astar_diag, indent=2))
        out.append("```\n")
    tally = {}
    for _, label, *_ in rows:
        tally[label] = tally.get(label, 0) + 1
    out.append("| verdict | count |")
    out.append("| --- | --- |")
    for label in ("DIVERGENT", "DIJ-FAIL", "ASTAR-FAIL", "BOTH-FAIL", "MINOR", "MATCH"):
        if tally.get(label):
            out.append(f"| {label} | {tally[label]} |")
    out.append("")
    out.append("| case | verdict | detail | dij len | astar len | dij LC | astar LC | dij gapv |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, label, why, d, a in rows:
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, label, why,
            d.get("length_m", "-"), a.get("length_m", "-"),
            d.get("n_lane_changes", "-"), a.get("n_lane_changes", "-"),
            d.get("gap_violations", "-")))
    out.append("")
    out.append("## Per-case detail (non-MATCH)\n")
    for name, label, why, d, a in rows:
        if label == "MATCH":
            continue
        out.append(f"### {name} — {label}")
        out.append(f"- {why}")
        out.append(f"- dij   road_option_seq: `{d.get('road_option_seq')}`")
        out.append(f"- carla road_option_seq: `{a.get('road_option_seq')}`")
        out.append(f"- dij   lane_seq: `{d.get('lane_seq')}`")
        out.append(f"- carla lane_seq: `{a.get('lane_seq')}`")
        out.append("")

    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
