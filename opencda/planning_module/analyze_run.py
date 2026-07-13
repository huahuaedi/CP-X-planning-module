"""Post-run analysis script for the planning module.

Usage:
    python analyze_run.py                          # auto-finds the newest artifact dir
    python analyze_run.py /path/to/scenario_dir   # explicit path
    python analyze_run.py --list                   # list all available runs

Outputs (written next to the CSV/JSON files):
    analysis_report.txt   - plain-text summary
    analysis_plots.png    - 6-panel figure
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
    metrics_json_path = os.path.join(artifact_dir, "planning_metrics.json")
    metrics_csv_path  = os.path.join(artifact_dir, "planning_metrics_timeseries.csv")
    cost_csv_path     = os.path.join(artifact_dir, "mpc_cost_history.csv")

    summary: Dict[str, object] = {}
    scenario_name = os.path.basename(artifact_dir.rstrip("/\\"))
    if os.path.isfile(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        summary = dict(payload.get("summary", {}))
        scenario_name = str(payload.get("scenario_name", scenario_name))

    metrics_rows = _load_csv(metrics_csv_path)
    cost_rows    = _load_csv(cost_csv_path)

    return {
        "artifact_dir":  artifact_dir,
        "scenario_name": scenario_name,
        "summary":       summary,
        "metrics_rows":  metrics_rows,
        "cost_rows":     cost_rows,
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
    ]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.2f}"


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

    metrics_rows: List[Dict[str, str]] = run["metrics_rows"]  # type: ignore[assignment]
    cost_rows:    List[Dict[str, str]] = run["cost_rows"]  # type: ignore[assignment]

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

    # 7: road boundary cost over time (lane boundary press indicator)
    ax7 = fig.add_subplot(gs[2, 1])
    if c_t:
        breach_mask = [v > 0.1 for v in c_boundary]
        ax7.fill_between(c_t, c_boundary, alpha=0.4, color="#F44336")
        ax7.plot(c_t, c_boundary, linewidth=0.8, color="#F44336")
        ax7.axhline(y=0.1, color="#FF9800", linestyle="--", linewidth=0.8, label="breach threshold")
        ax7.legend(fontsize=7)
    ax7.set_title("Road Boundary Cost (lane press indicator)")
    ax7.set_xlabel("time (s)")
    ax7.set_ylabel("cost")
    ax7.grid(True, alpha=0.3)

    # 8: ego trajectory
    ax8 = fig.add_subplot(gs[2, 2])
    if m_x and m_y:
        sc = ax8.scatter(m_x, m_y, c=m_v, cmap="plasma", s=2, vmin=0)
        plt.colorbar(sc, ax=ax8, label="speed m/s")
    ax8.set_title("Ego Trajectory (colored by speed)")
    ax8.set_xlabel("x (m)")
    ax8.set_ylabel("y (m)")
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


if __name__ == "__main__":
    main()
