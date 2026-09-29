#!/usr/bin/env python3
"""Tabulate the prediction-knowledge ablation runs side by side.

Reads the canonical planner JSONL (or legacy CSV) plus ``run_status.json`` from
each arm's debug directory and prints one metrics table.

    python opencda/scenario_testing/compare_prediction_ablation.py
    python opencda/scenario_testing/compare_prediction_ablation.py \
        --dir blind=/abs/debug_intersection_cross_blind \
        --dir oracle=/abs/debug_intersection_cross_oracle

With no --dir, looks for these under opencda/planning_module/opencda_bridge/:
    cv      -> debug_intersection_cross
    blind   -> debug_intersection_cross_blind
    oracle  -> debug_intersection_cross_oracle
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from opencda.scenario_testing.planner_debug_records import load_planner_records
except ModuleNotFoundError:
    from planner_debug_records import load_planner_records

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASE = _REPO_ROOT / "opencda" / "planning_module" / "opencda_bridge"
_DEFAULT_ARMS = {
    "cv": "debug_intersection_cross",
    "blind": "debug_intersection_cross_blind",
    "oracle": "debug_intersection_cross_oracle",
}
_HARD_BRAKE = 0.6           # applied_brake fraction
_HARD_DECEL_MPS2 = -3.0     # measured_accel_mps2


def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _truthy(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in ("true", "1", "1.0", "yes")


def _load_rows(debug_dir: Path) -> List[dict]:
    return load_planner_records(debug_dir)


def _run_status(debug_dir: Path) -> dict:
    p = debug_dir / "run_status.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _metrics(rows: List[dict], status: dict) -> Dict[str, object]:
    if not rows:
        return {"ticks": 0}
    n = len(rows)
    speeds = [s for s in (_f(r, "speed_mps") for r in rows) if s is not None]
    accels = [(float(r.get("sim_time_s", i) or i), _f(r, "measured_accel_mps2"))
              for i, r in enumerate(rows)]
    accels = [(t, a) for t, a in accels if a is not None]

    ttc = [v for v in (_f(r, "nearest_ttc_s") for r in rows)
           if v is not None and v > 0.0]
    gap = [v for v in (_f(r, "nearest_ttc_bumper_gap_m") for r in rows)
           if v is not None]

    jerk_max = 0.0
    for (t0, a0), (t1, a1) in zip(accels, accels[1:]):
        dt = t1 - t0
        if dt > 1e-3:
            jerk_max = max(jerk_max, abs(a1 - a0) / dt)

    fsm = [str(r.get("behavior_fsm_state", "")) for r in rows]
    ticks_to_goal = next(
        (i for i, s in enumerate(fsm) if s == "DESTINATION_STOP"), None
    )

    collisions = 0
    for cs in status.get("cav_states", []) or []:
        collisions = max(collisions, int(cs.get("collision_count", 0) or 0))
    col_col = max((int(_f(r, "collision_count") or 0) for r in rows), default=0)
    collisions = max(collisions, col_col)

    return {
        "ticks": n,
        "mode_col": next((r.get("prediction_mode", "") for r in rows), ""),
        "termination": status.get("termination_reason", ""),
        "dist_to_goal_m": status.get("distance_to_destination_m", None),
        "collisions": collisions,
        "min_ttc_s": round(min(ttc), 2) if ttc else None,
        "min_bumper_gap_m": round(min(gap), 2) if gap else None,
        "emergency_brake_ticks": sum(
            _truthy(r, "emergency_brake_control_active") for r in rows
        ),
        "hard_brake_ticks": sum(
            1 for r in rows if (_f(r, "applied_brake") or 0.0) >= _HARD_BRAKE
        ),
        "hard_decel_ticks": sum(
            1 for _, a in accels if a <= _HARD_DECEL_MPS2
        ),
        "max_decel_mps2": round(min((a for _, a in accels), default=0.0), 2),
        "max_abs_jerk_mps3": round(jerk_max, 1),
        "fallback_ticks": sum(_truthy(r, "fallback_active") for r in rows),
        "mpc_infeasible_ticks": sum(
            1 for r in rows if "infeasible" in str(r.get("mpc_fallback_reason", ""))
        ),
        "ticks_to_goal": ticks_to_goal if ticks_to_goal is not None else n,
        "mean_speed_mps": round(sum(speeds) / len(speeds), 2) if speeds else None,
    }


_ROW_ORDER = [
    ("ticks", "ticks logged", None),
    ("mode_col", "prediction_mode (from CSV)", None),
    ("termination", "termination_reason", None),
    ("dist_to_goal_m", "distance to goal (m)", "lower"),
    ("collisions", "collisions", "lower"),
    ("min_ttc_s", "min TTC (s)", "higher"),
    ("min_bumper_gap_m", "min bumper gap (m)", "higher"),
    ("emergency_brake_ticks", "emergency-brake ticks", "lower"),
    ("hard_brake_ticks", "hard-brake ticks (brake>=%.1f)" % _HARD_BRAKE, "lower"),
    ("hard_decel_ticks", "hard-decel ticks (a<=%.0f)" % _HARD_DECEL_MPS2, "lower"),
    ("max_decel_mps2", "peak decel (m/s^2)", "higher"),
    ("max_abs_jerk_mps3", "peak |jerk| (m/s^3)", "lower"),
    ("fallback_ticks", "MPC-fallback ticks", "lower"),
    ("mpc_infeasible_ticks", "MPC-infeasible ticks", "lower"),
    ("ticks_to_goal", "ticks to DESTINATION_STOP", "lower"),
    ("mean_speed_mps", "mean speed (m/s)", None),
]


def _fmt(v: object, width: int = 0) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        s = f"{v:g}"
    else:
        s = str(v)
    if width and len(s) > width - 1:
        s = s[: width - 2] + "…"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(_DEFAULT_BASE),
                    help="dir holding the default arm debug folders")
    ap.add_argument("--dir", action="append", default=[],
                    metavar="name=PATH", help="explicit arm=path (repeatable)")
    args = ap.parse_args()

    arms: Dict[str, Path] = {}
    if args.dir:
        for spec in args.dir:
            name, _, path = spec.partition("=")
            arms[name.strip()] = Path(path.strip())
    else:
        for name, sub in _DEFAULT_ARMS.items():
            arms[name] = Path(args.base) / sub

    present = {k: v for k, v in arms.items() if _load_rows(v)}
    if not present:
        print("no arm debug CSVs found. looked in:")
        for k, v in arms.items():
            print(f"  {k:8s} {v}")
        return 1
    for k in arms:
        if k not in present:
            print(f"[skip] {k}: no CSV at {arms[k]}")

    results = {
        name: _metrics(_load_rows(path), _run_status(path))
        for name, path in present.items()
    }
    names = list(present.keys())

    w0 = max(len(lbl) for _, lbl, _ in _ROW_ORDER) + 2
    wc = max(12, max(len(n) for n in names) + 2)
    print()
    print("prediction-knowledge ablation".center(w0 + wc * len(names)))
    print("-" * (w0 + wc * len(names)))
    hdr = "metric".ljust(w0) + "".join(n.rjust(wc) for n in names)
    print(hdr + "   better")
    print("-" * (w0 + wc * len(names)))
    for key, label, better in _ROW_ORDER:
        line = label.ljust(w0) + "".join(
            _fmt(results[n].get(key), wc).rjust(wc) for n in names
        )
        print(line + (f"   {better}" if better else ""))
    print("-" * (w0 + wc * len(names)))

    if "blind" in results and "oracle" in results:
        print("\nblind -> oracle delta (oracle minus blind):")
        for key, label, better in _ROW_ORDER:
            b, o = results["blind"].get(key), results["oracle"].get(key)
            if isinstance(b, (int, float)) and isinstance(o, (int, float)):
                d = o - b
                arrow = ""
                if better == "higher":
                    arrow = "  (oracle better)" if d > 0 else ("  (oracle worse)" if d < 0 else "")
                elif better == "lower":
                    arrow = "  (oracle better)" if d < 0 else ("  (oracle worse)" if d > 0 else "")
                print(f"  {label:<{w0}} {d:+g}{arrow}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
