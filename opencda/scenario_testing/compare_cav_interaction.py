#!/usr/bin/env python3
"""Show whether the multi-CAV interaction pipeline is effective.

Reads the two CAVs' debug CSVs for the ON run (cpx_two_cav_merge_conflict)
and the OFF run (..._baseline) and prints a side-by-side metrics table plus
the inter-CAV gap time series.

    python opencda/scenario_testing/compare_cav_interaction.py

Override dirs:
    --on-cav1 DIR --on-cav2 DIR --off-cav1 DIR --off-cav2 DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_DBG = _REPO / "opencda" / "planning_module" / "opencda_bridge"
_DEFAULTS = {
    "on_cav1": _DBG / "debug_two_cav_merge_cav1",
    "on_cav2": _DBG / "debug_two_cav_merge_cav2",
    "off_cav1": _DBG / "debug_two_cav_merge_off_cav1",
    "off_cav2": _DBG / "debug_two_cav_merge_off_cav2",
}
_HARD_BRAKE = 0.6
_HARD_DECEL = -3.0
_STALL_SPEED = 0.3
_STALL_SECONDS = 3.0


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


def _rows(d: Path) -> List[dict]:
    p = d / "opencda_planner_debug.csv"
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _status(d: Path) -> dict:
    p = d / "run_status.json"
    csv_path = d / "opencda_planner_debug.csv"
    if not p.is_file():
        return {}
    # An interrupted/new run truncates the CSV before its final status is
    # written.  Never combine that CSV with a status left by an older run.
    try:
        if csv_path.is_file() and p.stat().st_mtime_ns < csv_path.stat().st_mtime_ns:
            return {}
        value = json.loads(p.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _inter_cav_gap(rows_a: List[dict], rows_b: List[dict]) -> List[tuple]:
    """(t, euclidean_gap_m) joined on the nearest sim_time_s."""
    b_by_t = {}
    for r in rows_b:
        t = _f(r, "sim_time_s")
        if t is not None:
            b_by_t[round(t, 2)] = r
    out = []
    for r in rows_a:
        t = _f(r, "sim_time_s")
        if t is None:
            continue
        rb = b_by_t.get(round(t, 2))
        if rb is None:
            continue
        ax, ay = _f(r, "x_m"), _f(r, "y_m")
        bx, by = _f(rb, "x_m"), _f(rb, "y_m")
        if None in (ax, ay, bx, by):
            continue
        out.append((t, math.hypot(ax - bx, ay - by)))
    return out


def _stalled_seconds(rows: List[dict]) -> float:
    """Longest run of consecutive ticks with speed < _STALL_SPEED, before the
    vehicle first reaches DESTINATION_STOP (a real stall, not the goal)."""
    worst = run_s = 0.0
    prev_t = None
    for r in rows:
        if r.get("behavior_fsm_state", "") == "DESTINATION_STOP":
            break
        t, v = _f(r, "sim_time_s"), _f(r, "speed_mps")
        if t is None or v is None:
            continue
        dt = 0.0 if prev_t is None else max(0.0, t - prev_t)
        prev_t = t
        if v < _STALL_SPEED:
            run_s += dt
            worst = max(worst, run_s)
        else:
            run_s = 0.0
    return worst


def _one_cav(rows: List[dict], status: dict) -> Dict[str, object]:
    if not rows:
        return {"ticks": 0}
    n = len(rows)
    t = [_f(r, "sim_time_s") or i for i, r in enumerate(rows)]
    acc = [(t[i], _f(r, "post_supervisor_accel_cmd_mps2"))
           for i, r in enumerate(rows)]
    acc = [(tt, a) for tt, a in acc if a is not None]
    jerk = max(
        (abs(acc[i + 1][1] - acc[i][1]) / max(1e-3, acc[i + 1][0] - acc[i][0])
         for i in range(len(acc) - 1)),
        default=0.0,
    )
    ttc = [v for v in (_f(r, "nearest_ttc_s") for r in rows) if v and v > 0.0]
    bump = [v for v in (_f(r, "nearest_ttc_bumper_gap_m") for r in rows)
            if v is not None and v >= 0.0]
    # --- MPC-internal signals: what the peer trajectory does to the solve ---
    # raw MPC output before the safety supervisor clamp = true solution chatter
    racc = [(t[i], _f(r, "pre_supervisor_accel_cmd_mps2"))
            for i, r in enumerate(rows)]
    racc = [(tt, a) for tt, a in racc if a is not None]
    raw_jerk = max(
        (abs(racc[i + 1][1] - racc[i][1]) / max(1e-3, racc[i + 1][0] - racc[i][0])
         for i in range(len(racc) - 1)), default=0.0)
    raw_flips = sum(
        1 for i in range(len(racc) - 1)
        if racc[i][1] * racc[i + 1][1] < 0.0
        and abs(racc[i + 1][1] - racc[i][1]) > 0.5)
    solve_ms = sorted(v for v in (_f(r, "mpc_solve_time_ms") for r in rows)
                      if v is not None and v >= 0.0)
    n_stat = sum(1 for r in rows if r.get("mpc_feasibility_status", ""))
    n_solved = sum(1 for r in rows
                   if r.get("mpc_feasibility_status", "") == "solved")
    repcost = [v for v in (_f(r, "Cost_Repulsive") for r in rows) if v is not None]
    # Prefer the CSV's own collision_count. run_status.json is only written by
    # some runners and a stale one from an earlier scenario contaminates this.
    has_coll_col = any("collision_count" in r for r in rows[:1])
    if has_coll_col:
        coll = max((int(_f(r, "collision_count") or 0) for r in rows), default=0)
    else:
        coll = max((int(cs.get("collision_count", 0) or 0)
                    for cs in status.get("cav_states", []) or []), default=0)
    fsm = [r.get("behavior_fsm_state", "") for r in rows]
    return {
        "ticks": n,
        "termination": status.get("termination_reason", ""),
        "reached_goal": bool(status.get("cav_states", [{}])[0].get("agent_finished", False))
        if status.get("cav_states") else ("DESTINATION_STOP" in fsm),
        "collisions": coll,
        "min_ttc_s": round(min(ttc), 2) if ttc else None,
        "min_bumper_gap_m": round(min(bump), 2) if bump else None,
        "hard_brake_ticks": sum(
            1 for r in rows if (_f(r, "applied_brake") or 0.0) >= _HARD_BRAKE),
        "hard_decel_ticks": sum(1 for _, a in acc if a <= _HARD_DECEL),
        "peak_decel_mps2": round(min((a for _, a in acc), default=0.0), 2),
        "peak_jerk_mps3": round(jerk, 1),
        "fallback_ticks": sum(_truthy(r, "fallback_active") for r in rows),
        "infeasible_ticks": sum(
            1 for r in rows if "infeasible" in str(r.get("mpc_feasibility_reason", ""))),
        "lc_completion_lat_err_m": next(
            (round(_f(r, "lane_change_completion_lateral_error_m"), 3)
             for r in reversed(rows)
             if r.get("lane_change_completion_reason", "")
             and _f(r, "lane_change_completion_lateral_error_m") is not None),
            None,
        ),
        "stalled_s": round(_stalled_seconds(rows), 1),
        "mpc_solved_frac": round(n_solved / n_stat, 3) if n_stat else None,
        "mpc_replan_ticks": sum(_truthy(r, "mpc_replan_executed") for r in rows),
        "mpc_mean_solve_ms": round(sum(solve_ms) / len(solve_ms), 2) if solve_ms else None,
        "mpc_p95_solve_ms": round(solve_ms[int(0.95 * (len(solve_ms) - 1))], 2)
        if solve_ms else None,
        "raw_accel_sign_flips": raw_flips,
        "raw_peak_jerk_mps3": round(raw_jerk, 1),
        "repulsive_cost_mean": round(sum(repcost) / len(repcost), 3) if repcost else None,
        "repulsive_cost_max": round(max(repcost), 2) if repcost else None,
        "cav_roles_seen": ";".join(sorted({
            r.get("cav_conflict_summary", "") for r in rows
            if r.get("cav_conflict_summary", "") not in ("", "no_conflict")
        })) or "-",
        "cav_role_flips": _role_flips(rows),
    }


def _role_flips(rows: List[dict]) -> int:
    """How many times cav1's own role (make_gap/proceed/yield) changed."""
    def role(r):
        s = r.get("cav_conflict_summary", "")
        for part in s.split(";"):
            if part.startswith("cav") and "=" in part:
                return part.split("=", 1)[1]
        return ""
    seq = [role(r) for r in rows]
    seq = [x for x in seq if x]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


_ROWS = [
    ("reached_goal", "both reach goal", "yes"),
    ("collisions", "collisions", "0"),
    ("min_ttc_s", "min TTC to nearest obstacle (s)", "higher"),
    ("min_bumper_gap_m", "min bumper gap to nearest (m)", "higher"),
    ("stalled_s", "longest stall (s, pre-goal)", "0"),
    ("hard_brake_ticks", "hard-brake ticks (>=0.6)", "lower"),
    ("hard_decel_ticks", "hard-decel ticks (<=-3)", "lower"),
    ("peak_decel_mps2", "peak decel (m/s^2)", "higher"),
    ("peak_jerk_mps3", "peak |jerk| (m/s^3)", "lower"),
    ("fallback_ticks", "MPC-fallback ticks", "lower"),
    ("infeasible_ticks", "MPC-infeasible ticks", "lower"),
    ("lc_completion_lat_err_m", "LC completion lat err (m)", "lower"),
    ("cav_role_flips", "cav1 role flips", "lower (0)"),
    ("cav_roles_seen", "cav_conflict_summary seen", None),
]

# What adding the peer's predicted trajectory does to the MPC solve itself.
_MPC_ROWS = [
    ("mpc_solved_frac", "MPC solved fraction", "higher (->1)"),
    ("infeasible_ticks", "MPC primal-infeasible ticks", "lower"),
    ("fallback_ticks", "MPC fallback ticks", "lower"),
    ("mpc_replan_ticks", "MPC replan ticks", "lower"),
    ("mpc_mean_solve_ms", "mean QP solve (ms)", "lower/stable"),
    ("mpc_p95_solve_ms", "p95 QP solve (ms)", "lower/stable"),
    ("raw_accel_sign_flips", "raw accel-cmd sign flips", "lower"),
    ("raw_peak_jerk_mps3", "raw |jerk| pre-supervisor", "lower"),
    ("repulsive_cost_mean", "Cost_Repulsive mean", "lower (->0)"),
    ("repulsive_cost_max", "Cost_Repulsive max", "lower"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in _DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=str(v))
    a = ap.parse_args()
    dirs = {k: Path(getattr(a, k)) for k in _DEFAULTS}

    data = {}
    for arm in ("on", "off"):
        for cav in ("cav1", "cav2"):
            d = dirs[f"{arm}_{cav}"]
            data[(arm, cav)] = _one_cav(_rows(d), _status(d))

    present = {k for k, v in data.items() if v.get("ticks", 0)}
    if not present:
        print("no debug CSVs found; looked in:")
        for k, d in dirs.items():
            print(f"  {k:10s} {d}")
        return 1

    cols = [(arm, cav) for arm in ("off", "on") for cav in ("cav1", "cav2")
            if (arm, cav) in present]
    w0, wc = 32, 14
    print()
    print("multi-CAV interaction: OFF (baseline) vs ON".center(w0 + wc * len(cols)))
    print("-" * (w0 + wc * len(cols)))
    print("metric".ljust(w0) + "".join(f"{arm}/{cav}".rjust(wc) for arm, cav in cols)
          + "   better")
    print("-" * (w0 + wc * len(cols)))
    for key, label, better in _ROWS:
        line = label.ljust(w0) + "".join(
            _fmt(data[c].get(key))[: wc - 1].rjust(wc) for c in cols)
        print(line + (f"   {better}" if better else ""))
    print("-" * (w0 + wc * len(cols)))

    # MPC solve quality -- the direct "what the peer trajectory buys the MPC"
    print()
    print("effect of the peer trajectory on the MPC solve".center(
        w0 + wc * len(cols)))
    print("-" * (w0 + wc * len(cols)))
    print("metric".ljust(w0) + "".join(f"{arm}/{cav}".rjust(wc) for arm, cav in cols)
          + "   better")
    print("-" * (w0 + wc * len(cols)))
    for key, label, better in _MPC_ROWS:
        line = label.ljust(w0) + "".join(
            _fmt(data[c].get(key))[: wc - 1].rjust(wc) for c in cols)
        print(line + (f"   {better}" if better else ""))
    print("-" * (w0 + wc * len(cols)))

    # inter-CAV gap
    for arm in ("off", "on"):
        ga = _inter_cav_gap(_rows(dirs[f"{arm}_cav1"]), _rows(dirs[f"{arm}_cav2"]))
        if not ga:
            continue
        gaps = [g for _, g in ga]
        tmin = min(ga, key=lambda x: x[1])
        print(f"\n[{arm}] inter-CAV gap (m): min={min(gaps):.2f} @ t={tmin[0]:.1f}s"
              f"  mean={sum(gaps) / len(gaps):.2f}  end={gaps[-1]:.2f}")
    print()
    print("Effective if, ON vs OFF: min TTC / min gap not worse (ideally better),"
          " 0 collisions, 0 stall, fewer hard-brake/infeasible ticks in the merge,"
          " cav1 role stable (0-1 flips) at make_gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
