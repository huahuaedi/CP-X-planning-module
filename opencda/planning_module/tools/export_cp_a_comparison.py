"""Compare the CP-A OFF/ON runs using road position rather than sim clock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(directory: Path):
    rows = [json.loads(line) for line in (directory / "opencda_planner_debug.jsonl").open()
            if line.strip()]
    metrics = json.loads((directory / "planning_metrics.json").read_text())["summary"]
    if not rows:
        raise ValueError(f"No planner rows in {directory}")
    return rows, metrics


def numeric(rows, key):
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key)))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values)


def first_qp_x(rows):
    return next((float(row["x_m"]) for row in rows
                 if float(row.get("cav_total_qp_row_count") or 0) > 0), None)


def export(off_dir: Path, on_dir: Path, output_dir: Path):
    off_rows, off_metrics = load_run(off_dir)
    on_rows, on_metrics = load_run(on_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    styles = (("CP OFF", off_rows, off_metrics, "#d97706"),
              ("CP ON", on_rows, on_metrics, "#177e89"))
    for label, rows, metrics, color in styles:
        x = numeric(rows, "x_m")
        axes[0].plot(x, numeric(rows, "speed_mps"), color=color, lw=1.8,
                     label=label)
        axes[1].step(x, numeric(rows, "cav_total_qp_row_count"), where="post",
                     color=color, lw=1.5, label=label)
        event_x = first_qp_x(rows)
        if event_x is not None:
            axes[1].axvline(event_x, color=color, ls="--", lw=1.1,
                            label=f"{label} first QP at x={event_x:.1f} m")

    for ax in axes:
        ax.axvline(-124.475, color="#475569", ls=":", lw=1,
                   label="Cross-traffic path x=-124.5 m")
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8)
    axes[0].set(ylabel="Ego speed (m/s)", title="CP-A: Ego response to cross-traffic")
    axes[1].set(xlabel="Ego road position x (m)",
                ylabel="Active MPC conflict rows")
    axes[1].set_xlim(-155, -120)
    fig.text(0.02, 0.005,
             "Runs aligned by ego position. Traffic-signal state was UNKNOWN in both logs; "
             "this plot does not verify a red-light violation.", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"cp_a_off_vs_on.{extension}", dpi=180)
    plt.close(fig)

    summary = {
        label: {
            "collision_count": metrics["collision_count"],
            "distance_traveled_m": metrics["distance_traveled_m"],
            "min_ttc_s": metrics["min_ttc_s"],
            "max_drac_mps2": metrics["max_drac_mps2"],
            "mpc_plan_success_rate": metrics["mpc_plan_success_rate"],
            "first_qp_x_m": first_qp_x(rows),
            "destination_reached": any(bool(r.get("route_reached_destination")) for r in rows),
            "traffic_signal_states": sorted({str(r.get("traffic_signal_state")) for r in rows}),
        }
        for label, rows, metrics, _ in styles
    }
    (output_dir / "cp_a_comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-dir", type=Path, required=True)
    parser.add_argument("--on-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.off_dir, args.on_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
