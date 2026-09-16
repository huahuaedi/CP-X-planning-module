"""Plot CP-B ON/OFF interaction using ego road position as the common axis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _run(directory: Path):
    rows = [json.loads(line) for line in
            (directory / "opencda_planner_debug.jsonl").open()
            if line.strip()]
    with (directory / "planning_metrics_timeseries.csv").open() as stream:
        metrics_rows = list(csv.DictReader(stream))
    summary = json.loads((directory / "planning_metrics.json").read_text())["summary"]
    target_id = next(
        actor_id for row in rows
        for actor_id, tag in (row.get("cav_conflict_tags") or {}).items()
        if tag in {"CROSSING", "ONCOMING"}
    )
    return rows, metrics_rows, summary, target_id


def _target_ttc(metrics_rows, target_id):
    samples = []
    for row in metrics_rows:
        if row.get("nearest_ttc_obstacle_id") != target_id:
            continue
        try:
            samples.append((float(row["ego_x"]), float(row["nearest_ttc_s"])))
        except (TypeError, ValueError):
            continue
    return np.asarray(samples)


def export(off_dir: Path, on_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "CP OFF": _run(off_dir),
        "CP ON": _run(on_dir),
    }
    colors = {"CP OFF": "#d97706", "CP ON": "#177e89"}
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    report = {}

    for label, (rows, metrics_rows, summary, target_id) in runs.items():
        color = colors[label]
        approach = [row for row in rows
                    if 10.0 <= float(row["x_m"]) <= 55.0
                    and float(row["y_m"]) > -40.0]
        x = np.asarray([float(row["x_m"]) for row in approach])
        speed = np.asarray([float(row["speed_mps"]) for row in approach])
        brake = np.asarray([float(row.get("applied_brake") or 0.0)
                            for row in approach])
        qp = np.asarray([int(row.get("cav_total_qp_row_count") or 0)
                         for row in approach])
        axes[0].plot(x, speed, color=color, lw=1.7, label=label)
        axes[1].plot(x, qp, color=color, lw=1.5, label=label)
        axes[2].plot(x, brake, color=color, lw=1.4, label=label)

        first_qp = next((row for row in approach
                         if int(row.get("cav_total_qp_row_count") or 0) > 0), None)
        first_qp_x = float(first_qp["x_m"]) if first_qp else None
        if first_qp_x is not None:
            axes[1].axvline(first_qp_x, color=color, lw=1.0, ls="--",
                            label=f"{label} first QP: x={first_qp_x:.1f} m")

        ttc = _target_ttc(metrics_rows, target_id)
        ttc = ttc[(ttc[:, 0] >= 10) & (ttc[:, 0] <= 55) &
                  (ttc[:, 1] >= 0) & (ttc[:, 1] <= 10)] if ttc.size else ttc
        if ttc.size:
            min_ttc_index = int(np.argmin(ttc[:, 1]))
            min_ttc = float(ttc[min_ttc_index, 1])
            min_ttc_x = float(ttc[min_ttc_index, 0])
        else:
            min_ttc = min_ttc_x = None

        report[label] = {
            "ego_vehicle_id": rows[0].get("vehicle_id"),
            "target_actor_id": target_id,
            "row_count": len(rows),
            "first_conflict_qp_x_m": first_qp_x,
            "conflict_qp_ticks": sum(
                int(row.get("cav_total_qp_row_count") or 0) > 0 for row in rows
            ),
            "target_specific_min_ttc_s": min_ttc,
            "target_specific_min_ttc_x_m": min_ttc_x,
            "collision_count": summary["collision_count"],
            "destination_reached": any(
                bool(row.get("route_reached_destination")) for row in rows
            ),
            "mpc_plan_success_rate": summary["mpc_plan_success_rate"],
            "max_drac_mps2": summary["max_drac_mps2"],
        }

    axes[0].set(title="CP-B: Ego response to the oncoming connector vehicle",
                ylabel="Ego speed (m/s)")
    axes[1].set(ylabel="Active conflict rows in ego MPC")
    axes[2].set(xlabel="Ego road position x (m); travel runs left to right in this plot",
                ylabel="Applied brake command", ylim=(-0.05, 1.05))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        axis.set_xlim(55, 10)
    figure.text(0.02, 0.005,
                "Runs are aligned by ego position, not simulator clock. "
                "Target-specific TTC minima are reported in the companion JSON summary.",
                fontsize=8)
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"cp_b_off_vs_on.{suffix}", dpi=180)
    plt.close(figure)
    (output_dir / "cp_b_comparison_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off-dir", type=Path, required=True)
    parser.add_argument("--on-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.off_dir, args.on_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
