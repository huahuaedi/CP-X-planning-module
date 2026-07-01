"""Post-run analysis script for the planning module.

Usage:
    python analyze_run.py                          # auto-finds the newest artifact dir
    python analyze_run.py /path/to/scenario_dir   # explicit path
    python analyze_run.py --list                   # list all available runs

Outputs (written next to the CSV/JSON files):
    analysis_report.txt   - plain-text summary
    analysis_plots.png       - safety/MPC summary figure
    tracking_dashboard.png   - actual/reference/error/control/FSM figure
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _load_run(artifact_dir: str) -> Dict[str, object]:
    metrics_json_path  = os.path.join(artifact_dir, "planning_metrics.json")
    metrics_csv_path   = os.path.join(artifact_dir, "planning_metrics_timeseries.csv")
    cost_csv_path      = os.path.join(artifact_dir, "mpc_cost_history.csv")
    lane_ref_csv_path  = os.path.join(artifact_dir, "lane_reference_timeseries.csv")
    control_csv_path   = os.path.join(artifact_dir, "control_timeseries.csv")
    fsm_csv_path       = os.path.join(artifact_dir, "fsm_transition_log.csv")
    planned_csv_path   = os.path.join(artifact_dir, "planned_trajectory.csv")

    summary: Dict[str, object] = {}
    scenario_name = os.path.basename(artifact_dir.rstrip("/\\"))
    if os.path.isfile(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        summary = dict(payload.get("summary", {}))
        scenario_name = str(payload.get("scenario_name", scenario_name))

    metrics_rows     = _load_csv(metrics_csv_path)
    cost_rows        = _load_csv(cost_csv_path)
    lane_ref_rows    = _load_csv(lane_ref_csv_path)
    control_rows     = _load_csv(control_csv_path)
    fsm_rows         = _load_csv(fsm_csv_path)
    planned_rows     = _load_csv(planned_csv_path)
    collision_csv    = os.path.join(artifact_dir, "collision_events.csv")
    collision_events = _load_csv(collision_csv)

    return {
        "artifact_dir":    artifact_dir,
        "scenario_name":   scenario_name,
        "summary":         summary,
        "metrics_rows":    metrics_rows,
        "cost_rows":       cost_rows,
        "lane_ref_rows":   lane_ref_rows,
        "control_rows":    control_rows,
        "fsm_rows":        fsm_rows,
        "planned_rows":    planned_rows,
        "collision_events": collision_events,
    }


# ---------------------------------------------------------------------------
# Artifact directory discovery
# ---------------------------------------------------------------------------

_PLANNING_ROOT = os.path.dirname(os.path.abspath(__file__))


def _candidate_dirs() -> List[str]:
    candidates: List[str] = []
    for sub in ("carla_scenario", "opencda_scenario"):
        pattern = os.path.join(_PLANNING_ROOT, sub, "**", "planning_metrics.json")
        for match in glob.glob(pattern, recursive=True):
            candidates.append(os.path.dirname(match))
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates


def _newest_artifact_dir() -> Optional[str]:
    dirs = _candidate_dirs()
    return dirs[0] if dirs else None


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _grade(value: float, low: float, high: float, invert: bool = False) -> str:
    """Return PASS / WARN / FAIL based on thresholds."""
    if math.isnan(value):
        return "N/A "
    ok = value >= low and value <= high if not invert else value <= low
    if invert:
        if value <= low:
            return "PASS"
        if value <= high:
            return "WARN"
        return "FAIL"
    else:
        if value >= high:
            return "PASS"
        if value >= low:
            return "WARN"
        return "FAIL"


def _build_report(run: Dict[str, object]) -> str:
    s = run["summary"]
    scenario_name   = str(run["scenario_name"])
    collision_count = int(s.get("collision_count", -1))
    distance_m      = _safe_float(s.get("distance_traveled_m", float("nan")))
    col_per_km      = _safe_float(s.get("collision_rate_per_km", float("nan")))
    min_ttc         = _safe_float(s.get("min_ttc_s", float("nan")))
    max_drac        = _safe_float(s.get("max_drac_mps2", float("nan")))
    min_pet         = _safe_float(s.get("min_pet_s", float("nan")))
    plan_attempts   = int(s.get("mpc_plan_attempts", 0))
    plan_successes  = int(s.get("mpc_plan_successes", 0))
    plan_rate       = _safe_float(s.get("mpc_plan_success_rate", float("nan")))

    metrics_rows: List[Dict[str, str]] = run["metrics_rows"]  # type: ignore[assignment]
    cost_rows:    List[Dict[str, str]] = run["cost_rows"]  # type: ignore[assignment]

    # speed stats
    speeds = [_safe_float(r.get("ego_speed_mps", "nan")) for r in metrics_rows]
    speeds = [v for v in speeds if math.isfinite(v)]
    avg_speed = sum(speeds) / len(speeds) if speeds else float("nan")
    max_speed = max(speeds) if speeds else float("nan")

    # solve time stats
    solve_times = [_safe_float(r.get("solve_time_ms", "nan")) for r in cost_rows]
    solve_times = [v for v in solve_times if math.isfinite(v) and v > 0]
    avg_solve_ms = sum(solve_times) / len(solve_times) if solve_times else float("nan")
    max_solve_ms = max(solve_times) if solve_times else float("nan")
    pct95_solve  = sorted(solve_times)[int(len(solve_times) * 0.95)] if len(solve_times) > 20 else float("nan")

    # MPC solver failures
    solver_fail = sum(
        1 for r in cost_rows
        if "solved" not in str(r.get("solver_status", "")).lower()
    )

    # boundary cost breaches (Cost_RoadBoundary > 0.1 counts as a boundary press)
    boundary_breach_count = sum(
        1 for r in cost_rows
        if _safe_float(r.get("Cost_RoadBoundary", r.get("Cost_LaneBoundary", "0"))) > 0.1
    )
    total_cost_samples = max(1, len(cost_rows))
    boundary_breach_pct = 100.0 * boundary_breach_count / total_cost_samples

    lines = [
        "=" * 60,
        f"  Planning Run Analysis: {scenario_name}",
        "=" * 60,
        "",
        "── SAFETY ─────────────────────────────────────────────────",
        f"  Collisions          : {collision_count}  [{_grade(float(collision_count), 0, 0, invert=True)}]",
        f"  Collision / km      : {col_per_km:.3f}  [{_grade(col_per_km, 0, 0.5, invert=True)}]",
        f"  Min TTC (s)         : {min_ttc:.2f}  [{_grade(min_ttc, 3.0, 5.0)}]",
        f"  Max DRAC (m/s²)     : {max_drac:.2f}  [{_grade(max_drac, 0, 3.0, invert=True)}]",
        f"  Min PET (s)         : {_fmt(min_pet)}",
        f"  Boundary breaches   : {boundary_breach_count}/{total_cost_samples} ticks"
        f" ({boundary_breach_pct:.1f}%)  [{_grade(boundary_breach_pct, 0, 5.0, invert=True)}]",
        "",
        "── EFFICIENCY ──────────────────────────────────────────────",
        f"  Distance traveled   : {distance_m:.1f} m",
        f"  Avg speed           : {avg_speed:.2f} m/s  ({avg_speed*3.6:.1f} km/h)",
        f"  Max speed           : {max_speed:.2f} m/s  ({max_speed*3.6:.1f} km/h)",
        "",
        "── MPC PLANNER HEALTH ──────────────────────────────────────",
        f"  Plan attempts       : {plan_attempts}",
        f"  Plan success rate   : {plan_rate*100:.1f}%  [{_grade(plan_rate, 0.90, 0.97)}]",
        f"  Solver failures     : {solver_fail}",
        f"  Avg solve time      : {avg_solve_ms:.1f} ms",
        f"  P95 solve time      : {_fmt(pct95_solve)} ms",
        f"  Max solve time      : {max_solve_ms:.1f} ms  [{_grade(max_solve_ms, 0, 50, invert=True)}]",
        "",
        "── LEGEND ──────────────────────────────────────────────────",
        "  PASS  within target range",
        "  WARN  marginal — monitor closely",
        "  FAIL  outside acceptable range",
        "=" * 60,
        "",
        "── COLLISION DETAILS ───────────────────────────────────────",
        _collision_report_section(run),
        "=" * 60,
    ]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.2f}"


def _collision_report_section(run: Dict[str, object]) -> str:
    """Detailed per-collision breakdown with pre-collision context."""
    events: List[Dict[str, str]] = run.get("collision_events", [])  # type: ignore[assignment]
    metrics_rows: List[Dict[str, str]] = run["metrics_rows"]  # type: ignore[assignment]
    cost_rows: List[Dict[str, str]] = run["cost_rows"]  # type: ignore[assignment]

    if not events:
        return "  No collision events recorded.\n"

    m_t = [_safe_float(r.get("sim_time_s", "nan")) for r in metrics_rows]
    c_t = [_safe_float(r.get("sim_time_s", "nan")) for r in cost_rows]

    lines = []
    for ev in events:
        t = _safe_float(ev.get("sim_time_s", "nan"))
        idx = int(ev.get("collision_index", 0)) + 1
        actor = str(ev.get("other_actor_type", "unknown"))
        speed = _safe_float(ev.get("ego_speed_mps", "nan"))
        imp   = _safe_float(ev.get("impulse_magnitude", "nan"))
        ex    = _safe_float(ev.get("ego_x", "nan"))
        ey    = _safe_float(ev.get("ego_y", "nan"))

        lines.append(f"  Collision #{idx}")
        lines.append(f"    Time          : {_fmt(t)} s")
        lines.append(f"    Position      : ({_fmt(ex)}, {_fmt(ey)}) m")
        lines.append(f"    Ego speed     : {_fmt(speed)} m/s  ({speed*3.6:.1f} km/h)" if math.isfinite(speed) else "    Ego speed     : N/A")
        lines.append(f"    Other actor   : {actor}")
        lines.append(f"    Impulse       : {_fmt(imp)} N·s")

        # pre-collision context from timeseries (5 s window)
        if math.isfinite(t) and m_t:
            PRE = 5.0
            window_m = [(i, r) for i, r in enumerate(metrics_rows)
                        if math.isfinite(m_t[i]) and t - PRE <= m_t[i] <= t]
            if window_m:
                ttcs   = [_safe_float(r.get("nearest_ttc_s",  "nan")) for _, r in window_m]
                dracs  = [_safe_float(r.get("max_drac_mps2",  "nan")) for _, r in window_m]
                speeds = [_safe_float(r.get("ego_speed_mps",  "nan")) for _, r in window_m]
                finite = lambda vs: [v for v in vs if math.isfinite(v)]
                min_ttc_pre  = min(finite(ttcs),   default=float("nan"))
                max_drac_pre = max(finite(dracs),  default=float("nan"))
                avg_spd_pre  = sum(finite(speeds)) / max(1, len(finite(speeds)))
                lines.append(f"    ── Pre-collision {PRE:.0f}s window ──")
                lines.append(f"    Min TTC       : {_fmt(min_ttc_pre)} s  {'[CRITICAL]' if min_ttc_pre < 2 else ''}")
                lines.append(f"    Max DRAC      : {_fmt(max_drac_pre)} m/s²")
                lines.append(f"    Avg speed     : {avg_spd_pre:.1f} m/s")

        if math.isfinite(t) and c_t:
            PRE = 5.0
            window_c = [(i, r) for i, r in enumerate(cost_rows)
                        if math.isfinite(c_t[i]) and t - PRE <= c_t[i] <= t]
            if window_c:
                fails    = sum(1 for _, r in window_c if "solved" not in str(r.get("solver_status", "")).lower())
                bc_vals  = [_safe_float(r.get("Cost_RoadBoundary", r.get("Cost_LaneBoundary", "0"))) for _, r in window_c]
                rep_vals = [_safe_float(r.get("Cost_Repulsive_Collision", "0")) for _, r in window_c]
                finite   = lambda vs: [v for v in vs if math.isfinite(v)]
                lines.append(f"    MPC failures  : {fails}/{len(window_c)} in window")
                lines.append(f"    Max boundary  : {max(finite(bc_vals), default=0.0):.3f}  {'[HIGH]' if max(finite(bc_vals), default=0) > 1.0 else ''}")
                lines.append(f"    Max repulsive : {max(finite(rep_vals), default=0.0):.3f}")

                # infer likely cause
                causes = []
                if fails > len(window_c) * 0.5:
                    causes.append("MPC solver failure (>50% fails → stale control)")
                if max(finite(bc_vals), default=0) > 1.0:
                    causes.append("persistent boundary breach before collision")
                if math.isfinite(min_ttc_pre) and min_ttc_pre < 2.0:  # type: ignore[possibly-undefined]
                    causes.append(f"TTC critical ({min_ttc_pre:.1f}s < 2s)")
                if math.isfinite(speed) and speed > 8.0:
                    causes.append(f"high impact speed ({speed:.1f} m/s)")
                lines.append(f"    Likely cause  : {'; '.join(causes) if causes else 'undetermined'}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _make_plots(run: Dict[str, object], out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[analyze] matplotlib not installed — skipping plots.")
        return False

    metrics_rows:  List[Dict[str, str]] = run["metrics_rows"]          # type: ignore[assignment]
    cost_rows:     List[Dict[str, str]] = run["cost_rows"]              # type: ignore[assignment]
    lane_ref_rows: List[Dict[str, str]] = run.get("lane_ref_rows", []) # type: ignore[assignment]

    def col(rows: List[Dict[str, str]], key: str) -> List[float]:
        return [_safe_float(r.get(key, "nan")) for r in rows]

    m_t    = col(metrics_rows, "sim_time_s")
    m_v    = col(metrics_rows, "ego_speed_mps")
    m_ttc  = [min(v, 15.0) for v in col(metrics_rows, "nearest_ttc_s")]
    m_drac = col(metrics_rows, "max_drac_mps2")
    m_x    = col(metrics_rows, "ego_x")
    m_y    = col(metrics_rows, "ego_y")

    c_t        = col(cost_rows, "sim_time_s")
    c_total    = col(cost_rows, "Cost_Total")
    c_lane     = col(cost_rows, "Cost_LaneCenter")
    c_boundary = [_safe_float(r.get("Cost_RoadBoundary", r.get("Cost_LaneBoundary", "0")))
                  for r in cost_rows]
    c_repulsive= col(cost_rows, "Cost_Repulsive_Collision")
    c_control  = col(cost_rows, "Cost_Control")
    c_solve_ms = col(cost_rows, "solve_time_ms")

    # --- merge: for each cost sample find nearest position by timestamp ---
    def _nearest_idx(haystack: List[float], needle: float) -> int:
        return min(range(len(haystack)), key=lambda i: abs(haystack[i] - needle))

    merged_x, merged_y, merged_bc, merged_t = [], [], [], []
    if m_t and c_t:
        for i, ct in enumerate(c_t):
            if not math.isfinite(ct):
                continue
            mi = _nearest_idx(m_t, ct)
            mx = _safe_float(metrics_rows[mi].get("ego_x", "nan"))
            my = _safe_float(metrics_rows[mi].get("ego_y", "nan"))
            if math.isfinite(mx) and math.isfinite(my):
                merged_x.append(mx)
                merged_y.append(my)
                merged_bc.append(c_boundary[i])
                merged_t.append(ct)

    # --- curvature from ego positions (finite differences) ---
    def _curvature(xs: List[float], ys: List[float]) -> List[float]:
        n = len(xs)
        if n < 3:
            return [0.0] * n
        kappa = [0.0]
        for i in range(1, n - 1):
            dx1, dy1 = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
            dx2, dy2 = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
            cross = dx1 * dy2 - dy1 * dx2
            mag = math.hypot(dx1, dy1) * math.hypot(dx2, dy2)
            kappa.append(abs(cross) / max(mag, 1e-6))
        kappa.append(kappa[-1])
        return kappa

    m_kappa = _curvature(m_x, m_y)

    solver_statuses = [str(r.get("solver_status", "unknown")).lower() for r in cost_rows]
    solved   = solver_statuses.count("solved")
    unsolved = len(solver_statuses) - solved

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Planning Analysis — {run['scenario_name']}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1: speed
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(m_t, [v * 3.6 for v in m_v], color="#2196F3", linewidth=1.0)
    ax1.set_title("Speed (km/h)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("km/h")
    ax1.grid(True, alpha=0.3)

    # 2: TTC
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(m_t, m_ttc, color="#F44336", linewidth=1.0)
    ax2.axhline(y=3.0, color="#FF9800", linestyle="--", linewidth=0.8, label="3s threshold")
    ax2.set_title("Nearest TTC (s, clipped at 15)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("s")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 3: DRAC
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(m_t, m_drac, color="#9C27B0", linewidth=1.0)
    ax3.axhline(y=3.0, color="#FF9800", linestyle="--", linewidth=0.8, label="3 m/s² warn")
    ax3.set_title("Max DRAC (m/s²)")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("m/s²")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4: MPC cost components
    ax4 = fig.add_subplot(gs[1, :2])
    if c_t:
        ax4.plot(c_t, c_total,    label="Total",        linewidth=1.2, color="#212121")
        ax4.plot(c_t, c_lane,     label="LaneCenter",   linewidth=1.0, color="#4CAF50")
        ax4.plot(c_t, c_boundary, label="RoadBoundary", linewidth=1.0, color="#F44336")
        ax4.plot(c_t, c_repulsive,label="Collision",    linewidth=1.0, color="#FF9800")
        ax4.plot(c_t, c_control,  label="Control",      linewidth=1.0, color="#2196F3")
    ax4.set_title("MPC Cost Terms")
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("cost")
    ax4.legend(fontsize=7, ncol=3)
    ax4.grid(True, alpha=0.3)

    # 5: solver status pie
    ax5 = fig.add_subplot(gs[1, 2])
    if solved + unsolved > 0:
        colors = ["#4CAF50", "#F44336"]
        ax5.pie(
            [solved, unsolved],
            labels=[f"Solved\n{solved}", f"Failed\n{unsolved}"],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9},
        )
    ax5.set_title("MPC Solver Status")

    # 6: solve time distribution
    ax6 = fig.add_subplot(gs[2, 0])
    valid_solve = [v for v in c_solve_ms if math.isfinite(v) and v > 0]
    if valid_solve:
        ax6.hist(valid_solve, bins=30, color="#607D8B", edgecolor="white", linewidth=0.5)
        ax6.axvline(x=50, color="#F44336", linestyle="--", linewidth=0.8, label="50ms limit")
        ax6.legend(fontsize=7)
    ax6.set_title("Solve Time Distribution (ms)")
    ax6.set_xlabel("ms")
    ax6.set_ylabel("count")
    ax6.grid(True, alpha=0.3)

    # 7: boundary cost + curvature on twin axes (proves breaches happen at curves)
    ax7 = fig.add_subplot(gs[2, 1])
    BREACH_THRESH = 0.1
    if c_t:
        ax7.fill_between(c_t, c_boundary, alpha=0.25, color="#F44336")
        ax7.plot(c_t, c_boundary, linewidth=0.9, color="#F44336", label="Boundary cost")
        ax7.axhline(y=BREACH_THRESH, color="#FF9800", linestyle="--",
                    linewidth=0.8, label=f"breach >{BREACH_THRESH}")
        # shade breach intervals
        in_breach = False
        t_start = 0.0
        for i, (t, bc) in enumerate(zip(c_t, c_boundary)):
            if bc > BREACH_THRESH and not in_breach:
                t_start = t; in_breach = True
            elif bc <= BREACH_THRESH and in_breach:
                ax7.axvspan(t_start, t, alpha=0.15, color="#F44336")
                in_breach = False
        if in_breach:
            ax7.axvspan(t_start, c_t[-1], alpha=0.15, color="#F44336")
        ax7.set_ylabel("boundary cost", color="#F44336", fontsize=8)
        ax7.tick_params(axis="y", labelcolor="#F44336", labelsize=7)
        # curvature on secondary axis
        if len(m_t) > 2:
            ax7b = ax7.twinx()
            ax7b.plot(m_t, m_kappa, linewidth=0.7, color="#1565C0", alpha=0.7, label="curvature")
            ax7b.set_ylabel("curvature κ", color="#1565C0", fontsize=8)
            ax7b.tick_params(axis="y", labelcolor="#1565C0", labelsize=7)
            ax7b.set_ylim(bottom=0)
            lines7b, labels7b = ax7b.get_legend_handles_labels()
            lines7, labels7 = ax7.get_legend_handles_labels()
            ax7b.legend(lines7 + lines7b, labels7 + labels7b, fontsize=6, loc="upper right")
    ax7.set_title("Boundary Cost vs Curvature\n(breaches at curves = lane pressing)", fontsize=8)
    ax7.set_xlabel("time (s)", fontsize=8)
    ax7.grid(True, alpha=0.25)

    # 8: trajectory map with lane lines + breach markers
    ax8 = fig.add_subplot(gs[2, 2])

    # draw lane boundaries reconstructed from reference samples
    if lane_ref_rows:
        left_bx, left_by, right_bx, right_by = [], [], [], []
        for lr in lane_ref_rows:
            rx   = _safe_float(lr.get("ref_x", "nan"))
            ry   = _safe_float(lr.get("ref_y", "nan"))
            hdg  = _safe_float(lr.get("heading_rad", "nan"))
            lw   = _safe_float(lr.get("road_left_width_m", "nan"))
            rw   = _safe_float(lr.get("road_right_width_m", "nan"))
            coff = _safe_float(lr.get("road_center_offset_m", "0"), 0.0)
            if not all(math.isfinite(v) for v in (rx, ry, hdg, lw, rw)):
                continue
            # normal vector (perpendicular-left of heading in CARLA coords)
            nx = -math.sin(hdg)
            ny =  math.cos(hdg)
            # road centre = ref + center_offset * normal
            cx = rx + coff * nx
            cy = ry + coff * ny
            left_bx.append(cx + lw * nx)
            left_by.append(cy + lw * ny)
            right_bx.append(cx - rw * nx)
            right_by.append(cy - rw * ny)
        if left_bx:
            ax8.plot(left_bx,  left_by,  color="#FF6F00", linewidth=1.2,
                     linestyle="--", label="left boundary", zorder=2)
            ax8.plot(right_bx, right_by, color="#FF6F00", linewidth=1.2,
                     linestyle="--", label="right boundary", zorder=2)
            # lane centre
            ax8.plot(
                [(l + r) / 2 for l, r in zip(left_bx, right_bx)],
                [(l + r) / 2 for l, r in zip(left_by, right_by)],
                color="#BDBDBD", linewidth=0.7, linestyle=":", zorder=1,
            )

    # ego trajectory coloured by boundary cost
    if merged_x and merged_y:
        import numpy as np
        bc_arr = np.array(merged_bc, dtype=float)
        sc = ax8.scatter(merged_x, merged_y, c=bc_arr, cmap="RdYlGn_r",
                         s=8, vmin=0.0, vmax=max(0.5, float(bc_arr.max())),
                         zorder=3)
        cb = plt.colorbar(sc, ax=ax8)
        cb.set_label("boundary cost", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        # red × at every breach position
        bx = [merged_x[i] for i, v in enumerate(merged_bc) if v > BREACH_THRESH]
        by = [merged_y[i] for i, v in enumerate(merged_bc) if v > BREACH_THRESH]
        if bx:
            ax8.scatter(bx, by, marker="x", s=30, color="#D32F2F", linewidths=1.5,
                        zorder=5, label=f"breach ({len(bx)} pts)")
    elif m_x and m_y:
        ax8.plot(m_x, m_y, color="#90A4AE", linewidth=0.8, zorder=3)

    ax8.legend(fontsize=6, loc="upper left")
    ax8.set_title("Trajectory + lane boundaries\n(× = crossed lane line)", fontsize=8)
    ax8.set_xlabel("x (m)", fontsize=8)
    ax8.set_ylabel("y (m)", fontsize=8)
    ax8.set_aspect("equal", adjustable="datalim")
    ax8.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _list_runs() -> None:
    dirs = _candidate_dirs()
    if not dirs:
        print("No runs found.")
        return
    print(f"Found {len(dirs)} run(s):")
    for i, d in enumerate(dirs):
        mtime = os.path.getmtime(d)
        import datetime
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{i}] {ts}  {d}")



# ---------------------------------------------------------------------------
# Tracking / control / FSM dashboard
# ---------------------------------------------------------------------------

def _make_tracking_dashboard(run: Dict[str, object], out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        print("[analyze] matplotlib/numpy not installed — skipping tracking dashboard.")
        return False

    metrics_rows: List[Dict[str, str]] = run.get("metrics_rows", [])  # type: ignore[assignment]
    lane_ref_rows: List[Dict[str, str]] = run.get("lane_ref_rows", [])  # type: ignore[assignment]
    control_rows: List[Dict[str, str]] = run.get("control_rows", [])  # type: ignore[assignment]
    cost_rows: List[Dict[str, str]] = run.get("cost_rows", [])  # type: ignore[assignment]
    fsm_rows: List[Dict[str, str]] = run.get("fsm_rows", [])  # type: ignore[assignment]

    if not metrics_rows and not lane_ref_rows and not control_rows and not fsm_rows and not cost_rows:
        return False

    def col(rows: List[Dict[str, str]], key: str, default: float = float("nan")) -> List[float]:
        return [_safe_float(r.get(key, default), default) for r in rows]

    def nearest_idx(times: List[float], t: float) -> Optional[int]:
        finite = [(i, abs(float(tt) - float(t))) for i, tt in enumerate(times) if math.isfinite(tt)]
        if not finite:
            return None
        return min(finite, key=lambda item: item[1])[0]

    m_t = col(metrics_rows, "sim_time_s")
    m_x = col(metrics_rows, "ego_x")
    m_y = col(metrics_rows, "ego_y")
    m_v = col(metrics_rows, "ego_speed_mps")

    r_t = col(lane_ref_rows, "sim_time_s")
    r_x = col(lane_ref_rows, "ref_x")
    r_y = col(lane_ref_rows, "ref_y")
    r_h = col(lane_ref_rows, "heading_rad")
    r_jump = col(lane_ref_rows, "reference_jump_m", 0.0)
    r_lane = [str(r.get("lane_id", "")) for r in lane_ref_rows]

    ctl_t = col(control_rows, "sim_time_s")
    ctl_v = col(control_rows, "ego_speed_mps")
    ctl_acc = col(control_rows, "acceleration_mps2", 0.0)
    ctl_thr = col(control_rows, "throttle", 0.0)
    ctl_brk = col(control_rows, "brake", 0.0)
    ctl_vmax = col(control_rows, "mpc_vmax_mps", float("nan"))

    c_t = col(cost_rows, "sim_time_s")
    c_boundary = [_safe_float(r.get("Cost_RoadBoundary", r.get("Cost_LaneBoundary", "0")), 0.0) for r in cost_rows]
    c_lane = col(cost_rows, "Cost_LaneCenter", 0.0)
    c_status = [str(r.get("solver_status", "")).lower() for r in cost_rows]

    # Tracking errors: compare each ego sample with nearest lane-reference sample.
    err_t, pos_err, lat_err, matched_rx, matched_ry = [], [], [], [], []
    if metrics_rows and lane_ref_rows:
        for t, x, y in zip(m_t, m_x, m_y):
            if not all(math.isfinite(v) for v in (t, x, y)):
                continue
            idx = nearest_idx(r_t, t)
            if idx is None:
                continue
            rx, ry, hdg = r_x[idx], r_y[idx], r_h[idx]
            if not all(math.isfinite(v) for v in (rx, ry, hdg)):
                continue
            dx = float(x) - float(rx)
            dy = float(y) - float(ry)
            nx = -math.sin(float(hdg))
            ny = math.cos(float(hdg))
            err_t.append(float(t))
            pos_err.append(math.hypot(dx, dy))
            lat_err.append(dx * nx + dy * ny)
            matched_rx.append(float(rx))
            matched_ry.append(float(ry))

    fig = plt.figure(figsize=(17, 13))
    fig.suptitle(f"Tracking / Control / FSM Dashboard — {run['scenario_name']}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)

    # 1. Actual vs reference path.
    ax1 = fig.add_subplot(gs[0, 0])
    if m_x and m_y:
        ax1.plot(m_x, m_y, color="#1565C0", linewidth=1.4, label="actual ego path")
        ax1.scatter([m_x[0]], [m_y[0]], color="#2E7D32", s=45, marker="o", label="start", zorder=5)
        ax1.scatter([m_x[-1]], [m_y[-1]], color="#C62828", s=45, marker="x", label="end", zorder=5)
    if r_x and r_y:
        ax1.plot(r_x, r_y, color="#FF9800", linewidth=1.0, linestyle="--", label="reference first point")
        jump_x = [r_x[i] for i, v in enumerate(r_jump) if math.isfinite(v) and v > 2.25 and i < len(r_x)]
        jump_y = [r_y[i] for i, v in enumerate(r_jump) if math.isfinite(v) and v > 2.25 and i < len(r_y)]
        if jump_x:
            ax1.scatter(jump_x, jump_y, color="#D81B60", s=55, marker="*", label="reference jump >2.25m", zorder=6)
    ax1.set_title("Actual Path vs Reference")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)

    # 2. Tracking errors.
    ax2 = fig.add_subplot(gs[0, 1])
    if err_t:
        ax2.plot(err_t, pos_err, color="#455A64", linewidth=1.0, label="position error")
        ax2.plot(err_t, lat_err, color="#E53935", linewidth=1.0, label="signed lateral error")
        ax2.axhline(y=0.0, color="black", linewidth=0.6)
        ax2.axhline(y=1.0, color="#FB8C00", linestyle="--", linewidth=0.8, label="|lat|=1m")
        ax2.axhline(y=-1.0, color="#FB8C00", linestyle="--", linewidth=0.8)
        if pos_err:
            ax2.text(0.01, 0.98,
                     f"mean |lat|={sum(abs(v) for v in lat_err)/max(1,len(lat_err)):.2f}m\n"
                     f"max |lat|={max(abs(v) for v in lat_err):.2f}m\n"
                     f"max pos={max(pos_err):.2f}m",
                     transform=ax2.transAxes, va="top", fontsize=8,
                     bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#B0BEC5"})
    ax2.set_title("Tracking Error to Lane Reference")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("error (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7)

    # 3. Speed / acceleration / mpc vmax.
    ax3 = fig.add_subplot(gs[1, 0])
    if m_t and m_v:
        ax3.plot(m_t, [v * 3.6 for v in m_v], color="#1E88E5", linewidth=1.0, label="ego speed km/h")
    if ctl_t and ctl_vmax:
        ax3.plot(ctl_t, [v * 3.6 for v in ctl_vmax], color="#43A047", linewidth=0.9, linestyle="--", label="MPC vmax km/h")
    ax3b = ax3.twinx()
    if ctl_t and ctl_acc:
        ax3b.plot(ctl_t, ctl_acc, color="#8E24AA", linewidth=0.8, alpha=0.8, label="accel m/s²")
        ax3b.set_ylabel("accel (m/s²)", color="#8E24AA")
        ax3b.tick_params(axis="y", labelcolor="#8E24AA")
    ax3.set_title("Vehicle Speed / Acceleration")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("speed (km/h)")
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4. Control commands.
    ax4 = fig.add_subplot(gs[1, 1])
    if ctl_t:
        ax4.plot(ctl_t, ctl_thr, color="#2E7D32", linewidth=1.0, label="throttle")
        ax4.plot(ctl_t, ctl_brk, color="#C62828", linewidth=1.0, label="brake")
        ax4.plot(ctl_t, ctl_acc, color="#6A1B9A", linewidth=0.8, label="accel cmd")
    ax4.set_title("Applied Control Commands")
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("command / accel")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=7)

    # 5. FSM timeline and reference lane/jump.
    ax5 = fig.add_subplot(gs[2, 0])
    state_names = []
    for row in fsm_rows:
        for key in ("old_state", "new_state"):
            name = str(row.get(key, ""))
            if name and name not in state_names:
                state_names.append(name)
    state_to_y = {name: i for i, name in enumerate(state_names)}
    if fsm_rows and state_to_y:
        times = [_safe_float(r.get("sim_time_s", "nan")) for r in fsm_rows]
        ys = [state_to_y.get(str(r.get("new_state", "")), 0) for r in fsm_rows]
        ax5.step(times, ys, where="post", color="#3949AB", linewidth=1.2, label="FSM state")
        ax5.scatter(times, ys, color="#3949AB", s=18)
        for t, y, row in zip(times, ys, fsm_rows):
            reason = str(row.get("reason", ""))[:18]
            if math.isfinite(t):
                ax5.text(t, y + 0.06, reason, fontsize=6, rotation=30, alpha=0.75)
        ax5.set_yticks(list(state_to_y.values()))
        ax5.set_yticklabels(list(state_to_y.keys()), fontsize=7)
    else:
        ax5.text(0.5, 0.6, "No fsm_transition_log.csv found", ha="center", transform=ax5.transAxes)
    ax5b = ax5.twinx()
    if r_t and r_jump:
        ax5b.plot(r_t, r_jump, color="#D81B60", linewidth=0.8, alpha=0.8, label="reference jump m")
        ax5b.axhline(y=2.25, color="#D81B60", linestyle="--", linewidth=0.7)
        ax5b.set_ylabel("ref jump (m)", color="#D81B60")
        ax5b.tick_params(axis="y", labelcolor="#D81B60")
    ax5.set_title("FSM Transitions + Reference Jumps")
    ax5.set_xlabel("time (s)")
    ax5.grid(True, alpha=0.3)

    # 6. MPC status/cost with boundary and lane-center.
    ax6 = fig.add_subplot(gs[2, 1])
    if c_t:
        ax6.plot(c_t, c_lane, color="#43A047", linewidth=0.9, label="LaneCenter cost")
        ax6.plot(c_t, c_boundary, color="#E53935", linewidth=0.9, label="RoadBoundary cost")
        failed_t = [t for t, s in zip(c_t, c_status) if "solved" not in s]
        if failed_t:
            ymax = max([v for v in c_boundary + c_lane if math.isfinite(v)] + [1.0])
            ax6.scatter(failed_t, [ymax] * len(failed_t), color="#000000", marker="x", s=28, label="solver non-solved")
    ax6.set_title("MPC Lane/Boundary Cost + Solver Status")
    ax6.set_xlabel("time (s)")
    ax6.set_ylabel("cost")
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True

# ---------------------------------------------------------------------------
# Collision analysis figure
# ---------------------------------------------------------------------------

def _make_collision_plots(run: Dict[str, object], out_path: str) -> bool:
    """One figure per collision: pre-collision 10-second window of key signals."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return False

    events: List[Dict[str, str]] = run.get("collision_events", [])  # type: ignore[assignment]
    metrics_rows: List[Dict[str, str]] = run["metrics_rows"]  # type: ignore[assignment]
    cost_rows:    List[Dict[str, str]] = run["cost_rows"]      # type: ignore[assignment]
    lane_ref_rows: List[Dict[str, str]] = run.get("lane_ref_rows", [])  # type: ignore[assignment]

    if not events:
        return False

    m_t    = [_safe_float(r.get("sim_time_s",     "nan")) for r in metrics_rows]
    m_x    = [_safe_float(r.get("ego_x",          "nan")) for r in metrics_rows]
    m_y    = [_safe_float(r.get("ego_y",          "nan")) for r in metrics_rows]
    m_v    = [_safe_float(r.get("ego_speed_mps",  "nan")) for r in metrics_rows]
    m_ttc  = [min(_safe_float(r.get("nearest_ttc_s",  "nan"), 15.0), 15.0) for r in metrics_rows]
    m_drac = [_safe_float(r.get("max_drac_mps2",  "nan"), 0.0) for r in metrics_rows]
    c_t    = [_safe_float(r.get("sim_time_s",     "nan")) for r in cost_rows]
    c_bc   = [_safe_float(r.get("Cost_RoadBoundary", r.get("Cost_LaneBoundary", "0"))) for r in cost_rows]
    c_rep  = [_safe_float(r.get("Cost_Repulsive_Collision", "0")) for r in cost_rows]
    c_fail = [0 if "solved" in str(r.get("solver_status", "")).lower() else 1 for r in cost_rows]

    PRE, POST = 8.0, 2.0
    n_col = len(events)
    fig = plt.figure(figsize=(6 * n_col, 14))
    fig.suptitle(f"Collision Analysis — {run['scenario_name']}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(4, n_col, figure=fig, hspace=0.5, wspace=0.35)

    for ci, ev in enumerate(events):
        t_col = _safe_float(ev.get("sim_time_s", "nan"))
        actor = str(ev.get("other_actor_type", "?"))
        speed = _safe_float(ev.get("ego_speed_mps", "nan"))
        imp   = _safe_float(ev.get("impulse_magnitude", "nan"))
        cx    = _safe_float(ev.get("ego_x", "nan"))
        cy    = _safe_float(ev.get("ego_y", "nan"))
        col_title = (f"#{ci+1}  t={_fmt(t_col)}s\n"
                     f"{actor}  {speed*3.6:.0f}km/h  imp={_fmt(imp)}N·s"
                     if math.isfinite(speed) else f"#{ci+1}  t={_fmt(t_col)}s\n{actor}")

        def _win(ts, vals):
            pairs = [(t, v) for t, v in zip(ts, vals)
                     if math.isfinite(t) and math.isfinite(t_col)
                     and t_col - PRE <= t <= t_col + POST]
            if not pairs:
                return [], []
            tt, vv = zip(*pairs)
            return list(tt), list(vv)

        # row 0: speed + TTC
        ax = fig.add_subplot(gs[0, ci])
        wt, wv = _win(m_t, [v * 3.6 for v in m_v])
        wt2, wttc = _win(m_t, m_ttc)
        if wt:
            ax.plot(wt, wv, color="#2196F3", linewidth=1.2, label="speed km/h")
        ax2 = ax.twinx()
        if wt2:
            ax2.plot(wt2, wttc, color="#F44336", linewidth=1.0, linestyle="--", label="TTC s")
            ax2.axhline(y=3.0, color="#F44336", linewidth=0.6, linestyle=":")
            ax2.set_ylabel("TTC (s)", color="#F44336", fontsize=7)
        if math.isfinite(t_col):
            ax.axvline(x=t_col, color="black", linewidth=1.5, linestyle="-", label="collision")
        ax.set_title(col_title, fontsize=8, fontweight="bold")
        ax.set_ylabel("speed (km/h)", fontsize=7)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.3)

        # row 1: boundary cost + MPC failures
        ax = fig.add_subplot(gs[1, ci])
        wt_bc, wbc = _win(c_t, c_bc)
        wt_f,  wf  = _win(c_t, c_fail)
        if wt_bc:
            ax.fill_between(wt_bc, wbc, alpha=0.3, color="#F44336")
            ax.plot(wt_bc, wbc, color="#F44336", linewidth=1.0, label="boundary cost")
            ax.axhline(y=0.1, color="#FF9800", linestyle="--", linewidth=0.7)
        ax3 = ax.twinx()
        if wt_f:
            ax3.bar(wt_f, wf, width=0.08, color="#9C27B0", alpha=0.6, label="MPC fail")
            ax3.set_ylabel("MPC fail", color="#9C27B0", fontsize=7)
            ax3.set_ylim(0, 1.5)
        if math.isfinite(t_col):
            ax.axvline(x=t_col, color="black", linewidth=1.5)
        ax.set_ylabel("boundary cost", fontsize=7)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_title("Boundary cost + MPC failures", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # row 2: DRAC + repulsive cost
        ax = fig.add_subplot(gs[2, ci])
        wt_d, wd   = _win(m_t, m_drac)
        wt_r, wrep = _win(c_t, c_rep)
        if wt_d:
            ax.plot(wt_d, wd, color="#9C27B0", linewidth=1.0, label="DRAC m/s²")
            ax.axhline(y=3.0, color="#FF9800", linestyle="--", linewidth=0.7, label="3 m/s² warn")
        ax4 = ax.twinx()
        if wt_r:
            ax4.plot(wt_r, wrep, color="#FF9800", linewidth=0.9, linestyle="--", label="repulsive cost")
            ax4.set_ylabel("repulsive cost", color="#FF9800", fontsize=7)
        if math.isfinite(t_col):
            ax.axvline(x=t_col, color="black", linewidth=1.5)
        ax.set_ylabel("DRAC (m/s²)", fontsize=7)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_title("DRAC + obstacle repulsive cost", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

        # row 3: local trajectory map around collision point
        ax = fig.add_subplot(gs[3, ci])
        RADIUS = 40.0
        # full trajectory in gray
        near_x = [x for x, t in zip(m_x, m_t) if math.isfinite(t) and math.isfinite(cx)
                  and abs(x - cx) < RADIUS * 2]
        near_y = [y for y, t in zip(m_y, m_t) if math.isfinite(t) and math.isfinite(cy)
                  and abs(y - cy) < RADIUS * 2]
        if near_x:
            ax.plot(near_x, near_y, color="#CFD8DC", linewidth=1.2, zorder=1)
        # lane boundaries
        if lane_ref_rows:
            lbx, lby, rbx, rby = [], [], [], []
            for lr in lane_ref_rows:
                rx  = _safe_float(lr.get("ref_x",  "nan"))
                ry  = _safe_float(lr.get("ref_y",  "nan"))
                hdg = _safe_float(lr.get("heading_rad", "nan"))
                lw  = _safe_float(lr.get("road_left_width_m",  "nan"))
                rw  = _safe_float(lr.get("road_right_width_m", "nan"))
                cof = _safe_float(lr.get("road_center_offset_m", "0"), 0.0)
                if not all(math.isfinite(v) for v in (rx, ry, hdg, lw, rw)):
                    continue
                if math.isfinite(cx) and math.hypot(rx - cx, ry - cy) > RADIUS:
                    continue
                nx = -math.sin(hdg); ny = math.cos(hdg)
                ccx = rx + cof * nx; ccy = ry + cof * ny
                lbx.append(ccx + lw * nx); lby.append(ccy + lw * ny)
                rbx.append(ccx - rw * nx); rby.append(ccy - rw * ny)
            if lbx:
                ax.plot(lbx, lby, "--", color="#FF6F00", linewidth=1.2, label="lane boundary", zorder=2)
                ax.plot(rbx, rby, "--", color="#FF6F00", linewidth=1.2, zorder=2)
        # pre-collision path coloured by speed
        wt_loc, _ = _win(m_t, m_t)
        px = [m_x[i] for i, t in enumerate(m_t) if math.isfinite(t) and math.isfinite(t_col)
              and t_col - PRE <= t <= t_col + POST]
        py = [m_y[i] for i, t in enumerate(m_t) if math.isfinite(t) and math.isfinite(t_col)
              and t_col - PRE <= t <= t_col + POST]
        pv = [m_v[i] for i, t in enumerate(m_t) if math.isfinite(t) and math.isfinite(t_col)
              and t_col - PRE <= t <= t_col + POST]
        if px:
            try:
                import numpy as np
                sc = ax.scatter(px, py, c=pv, cmap="RdYlGn_r", s=10,
                                vmin=0, vmax=max(pv + [1]), zorder=3)
                plt.colorbar(sc, ax=ax, label="speed m/s", pad=0.01).ax.tick_params(labelsize=6)
            except Exception:
                ax.plot(px, py, color="#F44336", linewidth=1.5, zorder=3)
        # collision star marker
        if math.isfinite(cx) and math.isfinite(cy):
            ax.scatter([cx], [cy], marker="*", s=200, color="#D32F2F",
                       zorder=6, label="collision point")
        ax.legend(fontsize=6)
        ax.set_title("Local map (pre-collision path)", fontsize=8)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Multi-run comparison
# ---------------------------------------------------------------------------

def _compare_runs(runs: List[Dict[str, object]]) -> str:
    """Side-by-side comparison table for multiple runs."""
    metrics_keys = [
        ("collision_count",       "Collisions",         "{:.0f}"),
        ("collision_rate_per_km", "Collision/km",       "{:.3f}"),
        ("min_ttc_s",             "Min TTC (s)",        "{:.2f}"),
        ("max_drac_mps2",         "Max DRAC (m/s²)",   "{:.2f}"),
        ("mpc_plan_success_rate", "MPC success rate",   "{:.1%}"),
        ("distance_traveled_m",   "Distance (m)",       "{:.0f}"),
    ]

    name_col_w = max(20, max(len(str(r["scenario_name"])) for r in runs) + 2)
    val_w = 14

    header_parts = [f"{'Metric':<25}"]
    for r in runs:
        header_parts.append(f"{str(r['scenario_name'])[:val_w]:>{val_w}}")
    header = "  ".join(header_parts)
    sep = "-" * len(header)

    lines = ["", "=" * len(header), "  MULTI-SCENARIO COMPARISON", "=" * len(header), header, sep]

    for key, label, fmt in metrics_keys:
        row_parts = [f"{label:<25}"]
        for r in runs:
            v = r["summary"].get(key)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                cell = "N/A"
            else:
                try:
                    cell = fmt.format(float(v))
                except Exception:
                    cell = str(v)
            row_parts.append(f"{cell:>{val_w}}")
        lines.append("  ".join(row_parts))

    # boundary breach row (from cost CSV)
    row_parts = [f"{'Boundary breach %':<25}"]
    for r in runs:
        cost_rows = r["cost_rows"]
        if cost_rows:
            breaches = sum(
                1 for row in cost_rows
                if _safe_float(row.get("Cost_RoadBoundary", row.get("Cost_LaneBoundary", "0"))) > 0.1
            )
            pct = 100.0 * breaches / len(cost_rows)
            row_parts.append(f"{pct:>{val_w}.1f}%")
        else:
            row_parts.append(f"{'N/A':>{val_w}}")
    lines.append("  ".join(row_parts))

    lines += [sep, ""]
    return "\n".join(lines)


def _make_comparison_plot(runs: List[Dict[str, object]], out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[analyze] matplotlib/numpy not installed — skipping comparison plot.")
        return False

    names = [str(r["scenario_name"]) for r in runs]

    def _sv(run: Dict[str, object], key: str, default: float = 0.0) -> float:
        v = run["summary"].get(key, default)
        if v is None:
            return default
        return _safe_float(v, default)

    collisions      = [_sv(r, "collision_count") for r in runs]
    min_ttc         = [min(_sv(r, "min_ttc_s", 15.0), 15.0) for r in runs]
    max_drac        = [_sv(r, "max_drac_mps2") for r in runs]
    plan_rate       = [_sv(r, "mpc_plan_success_rate", 1.0) * 100 for r in runs]
    boundary_pct    = []
    for r in runs:
        cost_rows = r["cost_rows"]
        if cost_rows:
            b = sum(1 for row in cost_rows
                    if _safe_float(row.get("Cost_RoadBoundary", row.get("Cost_LaneBoundary", "0"))) > 0.1)
            boundary_pct.append(100.0 * b / len(cost_rows))
        else:
            boundary_pct.append(0.0)

    x = np.arange(len(names))
    width = 0.55

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle("Scenario Comparison", fontsize=13, fontweight="bold")

    def _bar(ax, values, title, ylabel, color, threshold=None, threshold_label=None):
        bars = ax.bar(x, values, width, color=color, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values + [1]),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        if threshold is not None:
            ax.axhline(y=threshold, color="#F44336", linestyle="--", linewidth=0.9,
                       label=threshold_label or f"{threshold}")
            ax.legend(fontsize=6)

    _bar(axes[0], collisions,   "Collisions",           "count",  "#F44336")
    _bar(axes[1], min_ttc,      "Min TTC (s)",          "s",      "#4CAF50", threshold=3.0, threshold_label="3s safety")
    _bar(axes[2], max_drac,     "Max DRAC (m/s²)",     "m/s²",  "#FF9800", threshold=3.0, threshold_label="3 m/s² warn")
    _bar(axes[3], boundary_pct, "Boundary Breach %",    "%",      "#9C27B0", threshold=5.0, threshold_label="5% warn")
    _bar(axes[4], plan_rate,    "MPC Success Rate (%)", "%",      "#2196F3", threshold=95.0, threshold_label="95% target")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _nearest_row_by_time(rows: List[Dict[str, str]], time_key: str, query_time_s: float) -> Optional[Dict[str, str]]:
    best_row = None
    best_dt = float("inf")
    for row in rows:
        row_time_s = _safe_float(row.get(time_key, "nan"))
        if not math.isfinite(row_time_s):
            continue
        dt_s = abs(float(row_time_s) - float(query_time_s))
        if dt_s < best_dt:
            best_dt = float(dt_s)
            best_row = row
    return best_row


def _tracking_error_rows(run: Dict[str, object]) -> List[Dict[str, object]]:
    metrics_rows: List[Dict[str, str]] = run.get("metrics_rows", [])  # type: ignore[assignment]
    lane_ref_rows: List[Dict[str, str]] = run.get("lane_ref_rows", [])  # type: ignore[assignment]
    control_rows: List[Dict[str, str]] = run.get("control_rows", [])  # type: ignore[assignment]
    if not metrics_rows or not lane_ref_rows:
        return []

    rows: List[Dict[str, object]] = []
    for metric in metrics_rows:
        sim_time_s = _safe_float(metric.get("sim_time_s", "nan"))
        ego_x = _safe_float(metric.get("ego_x", "nan"))
        ego_y = _safe_float(metric.get("ego_y", "nan"))
        if not all(math.isfinite(value) for value in (sim_time_s, ego_x, ego_y)):
            continue
        ref = _nearest_row_by_time(lane_ref_rows, "sim_time_s", float(sim_time_s))
        if ref is None:
            continue
        ref_x = _safe_float(ref.get("ref_x", "nan"))
        ref_y = _safe_float(ref.get("ref_y", "nan"))
        ref_heading = _safe_float(ref.get("heading_rad", "nan"))
        if not all(math.isfinite(value) for value in (ref_x, ref_y, ref_heading)):
            continue
        dx_m = float(ego_x) - float(ref_x)
        dy_m = float(ego_y) - float(ref_y)
        normal_x = -math.sin(float(ref_heading))
        normal_y = math.cos(float(ref_heading))
        tangent_x = math.cos(float(ref_heading))
        tangent_y = math.sin(float(ref_heading))
        control = _nearest_row_by_time(control_rows, "sim_time_s", float(sim_time_s)) if control_rows else None
        rows.append(
            {
                "sim_time_s": float(sim_time_s),
                "ego_x": float(ego_x),
                "ego_y": float(ego_y),
                "ref_x": float(ref_x),
                "ref_y": float(ref_y),
                "position_error_m": float(math.hypot(dx_m, dy_m)),
                "lateral_error_m": float(dx_m * normal_x + dy_m * normal_y),
                "longitudinal_error_m": float(dx_m * tangent_x + dy_m * tangent_y),
                "ego_speed_mps": _safe_float(metric.get("ego_speed_mps", "nan")),
                "mpc_vmax_mps": _safe_float((control or {}).get("mpc_vmax_mps", "nan")),
                "acceleration_mps2": _safe_float((control or {}).get("acceleration_mps2", "nan")),
                "throttle": _safe_float((control or {}).get("throttle", "nan")),
                "brake": _safe_float((control or {}).get("brake", "nan")),
                "behavior_decision": str(metric.get("behavior_decision", "")),
                "fsm_state": str(metric.get("fsm_state", "")),
                "reference_lane_id": str(ref.get("lane_id", "")),
                "reference_jump_m": _safe_float(ref.get("reference_jump_m", "nan")),
                "reference_stabilized": str(ref.get("reference_stabilized", "")),
            }
        )
    return rows


def _write_tracking_error_csv(run: Dict[str, object], out_path: str) -> bool:
    rows = _tracking_error_rows(run)
    if not rows:
        return False
    fieldnames = [
        "sim_time_s", "ego_x", "ego_y", "ref_x", "ref_y",
        "position_error_m", "lateral_error_m", "longitudinal_error_m",
        "ego_speed_mps", "mpc_vmax_mps", "acceleration_mps2",
        "throttle", "brake", "behavior_decision", "fsm_state",
        "reference_lane_id", "reference_jump_m", "reference_stabilized",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return True


def _write_mpc_diagnostics_json(run: Dict[str, object], out_path: str) -> bool:
    cost_rows: List[Dict[str, str]] = run.get("cost_rows", [])  # type: ignore[assignment]
    control_rows: List[Dict[str, str]] = run.get("control_rows", [])  # type: ignore[assignment]
    fsm_rows: List[Dict[str, str]] = run.get("fsm_rows", [])  # type: ignore[assignment]
    tracking_rows = _tracking_error_rows(run)

    def finite_values(rows: List[Dict[str, str]], key: str) -> List[float]:
        values = [_safe_float(row.get(key, "nan")) for row in rows]
        return [value for value in values if math.isfinite(value)]

    def stats(values: List[float]) -> Dict[str, object]:
        if not values:
            return {"count": 0, "mean": None, "max": None, "p95": None}
        sorted_values = sorted(values)
        p95_index = min(len(sorted_values) - 1, int(round(0.95 * (len(sorted_values) - 1))))
        return {
            "count": int(len(values)),
            "mean": float(sum(values) / len(values)),
            "max": float(max(values)),
            "p95": float(sorted_values[p95_index]),
        }

    solver_failures = [
        row for row in cost_rows
        if "solved" not in str(row.get("solver_status", "")).strip().lower()
    ]
    lateral_errors = [
        abs(float(row["lateral_error_m"]))
        for row in tracking_rows
        if math.isfinite(float(row.get("lateral_error_m", float("nan"))))
    ]
    position_errors = [
        float(row["position_error_m"])
        for row in tracking_rows
        if math.isfinite(float(row.get("position_error_m", float("nan"))))
    ]
    boundary_values = [
        _safe_float(row.get("Cost_RoadBoundary", row.get("Cost_LaneBoundary", "0")), 0.0)
        for row in cost_rows
    ]
    repulsive_collision = finite_values(cost_rows, "Cost_Repulsive_Collision")
    payload = {
        "scenario_name": str(run.get("scenario_name", "")),
        "samples": {
            "metrics": int(len(run.get("metrics_rows", []))),  # type: ignore[arg-type]
            "control": int(len(control_rows)),
            "lane_reference": int(len(run.get("lane_ref_rows", []))),  # type: ignore[arg-type]
            "mpc_cost": int(len(cost_rows)),
            "fsm_transitions": int(len(fsm_rows)),
            "tracking_error": int(len(tracking_rows)),
        },
        "tracking_error": {
            "position_error_m": stats(position_errors),
            "abs_lateral_error_m": stats(lateral_errors),
            "samples_abs_lateral_error_gt_1m": int(sum(1 for value in lateral_errors if value > 1.0)),
        },
        "control": {
            "ego_speed_mps": stats(finite_values(control_rows, "ego_speed_mps")),
            "acceleration_mps2": stats([abs(value) for value in finite_values(control_rows, "acceleration_mps2")]),
            "mpc_vmax_mps": stats(finite_values(control_rows, "mpc_vmax_mps")),
        },
        "mpc": {
            "solver_failure_count": int(len(solver_failures)),
            "solve_time_ms": stats(finite_values(cost_rows, "solve_time_ms")),
            "road_boundary_cost": stats([value for value in boundary_values if math.isfinite(value)]),
            "road_boundary_cost_gt_0_1_count": int(sum(1 for value in boundary_values if math.isfinite(value) and value > 0.1)),
            "repulsive_collision_cost": stats(repulsive_collision),
        },
        "fsm": {
            "transitions": [
                {
                    "sim_time_s": _safe_float(row.get("sim_time_s", "nan")),
                    "old_state": str(row.get("old_state", "")),
                    "new_state": str(row.get("new_state", "")),
                    "reason": str(row.get("reason", "")),
                    "decision": str(row.get("decision", "")),
                    "target_lane_id": str(row.get("target_lane_id", "")),
                }
                for row in fsm_rows
            ],
        },
        "data_quality": {
            "has_lateral_offset_column": bool(
                run.get("metrics_rows")
                and "lateral_offset_m" in dict(list(run.get("metrics_rows", []))[0]).keys()  # type: ignore[arg-type]
            ),
            "has_planned_trajectory": bool(run.get("planned_rows")),
            "note": "Run the simulation again if fields are missing; older artifacts were written before diagnostics were expanded.",
        },
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a planning module run.")
    parser.add_argument(
        "artifact_dir", nargs="?", default=None,
        help="Path to the scenario artifact directory. Omit to auto-use the newest run.",
    )
    parser.add_argument("--list", action="store_true", help="List all available runs and exit.")
    parser.add_argument(
        "--compare", nargs="+", metavar="DIR",
        help="Compare multiple artifact directories side by side.",
    )
    args = parser.parse_args()

    if args.list:
        _list_runs()
        return

    if args.compare:
        runs = []
        for d in args.compare:
            d = os.path.abspath(d)
            if not os.path.isdir(d):
                print(f"[analyze] Directory not found: {d}")
                continue
            runs.append(_load_run(d))
        if len(runs) < 2:
            print("[analyze] Need at least 2 valid directories for comparison.")
            sys.exit(1)
        table = _compare_runs(runs)
        print(table)
        # save table next to the first run
        table_path = os.path.join(runs[0]["artifact_dir"], "comparison_report.txt")
        with open(table_path, "w", encoding="utf-8") as fh:
            fh.write(table + "\n")
        print(f"[analyze] Comparison saved to {table_path}")
        plot_path = os.path.join(runs[0]["artifact_dir"], "comparison_plots.png")
        if _make_comparison_plot(runs, plot_path):
            print(f"[analyze] Comparison plot saved to {plot_path}")
        return

    artifact_dir = args.artifact_dir
    if artifact_dir is None:
        artifact_dir = _newest_artifact_dir()
        if artifact_dir is None:
            print("[analyze] No artifact directory found. Run the simulation first.")
            sys.exit(1)
        print(f"[analyze] Auto-selected newest run: {artifact_dir}")

    artifact_dir = os.path.abspath(artifact_dir)
    if not os.path.isdir(artifact_dir):
        print(f"[analyze] Directory not found: {artifact_dir}")
        sys.exit(1)

    run = _load_run(artifact_dir)

    report = _build_report(run)
    print(report)
    report_path = os.path.join(artifact_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[analyze] Report saved to {report_path}")

    plot_path = os.path.join(artifact_dir, "analysis_plots.png")
    if _make_plots(run, plot_path):
        print(f"[analyze] Plots saved to {plot_path}")

    tracking_plot_path = os.path.join(artifact_dir, "tracking_dashboard.png")
    if _make_tracking_dashboard(run, tracking_plot_path):
        print(f"[analyze] Tracking dashboard saved to {tracking_plot_path}")

    tracking_error_path = os.path.join(artifact_dir, "tracking_error_timeseries.csv")
    if _write_tracking_error_csv(run, tracking_error_path):
        print(f"[analyze] Tracking error CSV saved to {tracking_error_path}")

    diagnostics_path = os.path.join(artifact_dir, "mpc_diagnostics.json")
    if _write_mpc_diagnostics_json(run, diagnostics_path):
        print(f"[analyze] MPC diagnostics saved to {diagnostics_path}")

    collision_plot_path = os.path.join(artifact_dir, "collision_analysis.png")
    if _make_collision_plots(run, collision_plot_path):
        print(f"[analyze] Collision analysis saved to {collision_plot_path}")
    elif run.get("collision_events"):
        print("[analyze] Collision events found but matplotlib unavailable for collision plot.")


if __name__ == "__main__":
    main()
