"""Export an evidence-based CP-D roadway-object ON/OFF comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OBJECT_X_M = -180.0
OBJECT_Y_M = 8.3
ADJACENT_LANE_Y_M = 4.8


def _load(directory: Path):
    path = directory / "opencda_planner_debug.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise ValueError("No diagnostics in %s" % path)
    front_ids = Counter(str(row.get("front_gap_actor_id") or "") for row in rows)
    front_ids.pop("", None)
    target_id = front_ids.most_common(1)[0][0] if front_ids else ""
    return rows, target_id


def _first(rows, condition):
    return next((row for row in rows if condition(row)), None)


def _event(rows, condition):
    row = _first(rows, condition)
    if row is None:
        return None
    return {
        "sim_time_s": float(row.get("sim_time_s") or 0.0),
        "x_m": float(row.get("x_m") or 0.0),
        "y_m": float(row.get("y_m") or 0.0),
        "speed_mps": float(row.get("speed_mps") or 0.0),
    }


def _summarize(rows, target_id):
    source_lane = int(rows[0].get("current_lane_id") or 0)
    cp_seen = _event(rows, lambda row: int(row.get("cp_obstacle_count") or 0) > 0)
    target_used = _event(
        rows, lambda row: str(row.get("front_gap_actor_id") or "") == target_id
    )
    blockage = _event(
        rows, lambda row: str(row.get("semantic_risk_kind") or "") == "LANE_BLOCKAGE"
    )
    prepare = _event(
        rows,
        lambda row: str(row.get("semantic_behavior_action") or "")
        == "PREPARE_LANE_CHANGE",
    )
    execute = _event(
        rows,
        lambda row: str(row.get("behavior_decision") or "") == "lane_change_left",
    )
    lane_changed = _event(
        rows, lambda row: int(row.get("current_lane_id") or 0) != source_lane
    )
    mission = _event(
        rows,
        lambda row: bool(
            row.get("destination_mission_complete", row.get("mission_complete", False))
        ),
    )
    approach_start = target_used or blockage
    approach_end = lane_changed or execute
    approach_rows = []
    if approach_start and approach_end:
        approach_rows = [
            row for row in rows
            if approach_start["sim_time_s"]
            <= float(row.get("sim_time_s") or 0.0)
            <= approach_end["sim_time_s"]
        ]
    speed = [float(row.get("speed_mps") or 0.0) for row in rows]
    measured_accel = [float(row.get("measured_accel_mps2") or 0.0) for row in rows]
    center_distances = [
        math.hypot(
            float(row.get("x_m") or 0.0) - OBJECT_X_M,
            float(row.get("y_m") or 0.0) - OBJECT_Y_M,
        ) for row in rows
    ]
    return {
        "ego_actor_id": rows[0].get("vehicle_id"),
        "stationary_object_actor_id": target_id,
        "cp_first_seen": cp_seen,
        "target_first_used": target_used,
        "lane_blockage_first_detected": blockage,
        "prepare_lane_change_first": prepare,
        "lane_change_execution_first": execute,
        "target_lane_first_matched": lane_changed,
        "mission_complete": mission is not None,
        "mission_complete_event": mission,
        "run_duration_s": float(rows[-1]["sim_time_s"])
        - float(rows[0]["sim_time_s"]),
        "minimum_approach_speed_mps": (
            min(float(row.get("speed_mps") or 0.0) for row in approach_rows)
            if approach_rows else None
        ),
        "mean_speed_mps": float(np.mean(speed)),
        "minimum_center_distance_to_object_m": min(center_distances),
        "maximum_abs_measured_acceleration_mps2": max(map(abs, measured_accel)),
        "mpc_infeasible_or_safe_stop_ticks": sum(
            "infeasible" in str(row.get("mpc_status") or "").lower()
            or str(row.get("mpc_status") or "") == "bounded_safe_stop"
            for row in rows
        ),
        "fallback_ticks": sum(bool(row.get("fallback_active")) for row in rows),
        "lane_change_left_ticks": sum(
            str(row.get("behavior_decision") or "") == "lane_change_left"
            for row in rows
        ),
    }


def export(off_dir: Path, on_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"CP OFF": _load(off_dir), "CP ON": _load(on_dir)}
    colors = {"CP OFF": "#d97706", "CP ON": "#177e89"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    report = {}

    for label, (rows, target_id) in runs.items():
        color = colors[label]
        road_x = np.asarray([float(row["x_m"]) for row in rows])
        road_y = np.asarray([float(row["y_m"]) for row in rows])
        speed = np.asarray([float(row.get("speed_mps") or 0.0) for row in rows])
        summary = _summarize(rows, target_id)
        report[label] = summary

        axes[0].plot(road_x, speed, color=color, lw=1.9, label=label)
        axes[1].plot(road_x, road_y, color=color, lw=1.8, label=label)

        execute = summary["lane_change_execution_first"]
        if execute is not None:
            axes[0].scatter(
                [execute["x_m"]], [execute["speed_mps"]], color=color,
                edgecolor="white", linewidth=0.7, s=55, zorder=4,
                label="%s lane-change start" % label,
            )

    for axis in axes:
        axis.axvline(OBJECT_X_M, color="#b91c1c", lw=1.1, ls="--")
        axis.grid(alpha=0.22)
        axis.set_xlim(-295, -80)
        axis.legend(fontsize=8, loc="best")

    axes[0].set_title("Ego Speed")
    axes[0].set_ylabel("Speed (m/s)")
    axes[1].set_title("Ego Lateral Motion")
    axes[1].set_ylabel("Road y (m)")
    axes[1].axhline(OBJECT_Y_M, color="#475569", lw=1.0, ls=":",
                    label="Blocked lane center")
    axes[1].axhline(ADJACENT_LANE_Y_M, color="#94a3b8", lw=1.0, ls=":",
                    label="Adjacent lane center")
    axes[1].scatter([OBJECT_X_M], [OBJECT_Y_M], marker="X", s=100,
                    color="#b91c1c", zorder=5, label="Roadway object")
    axes[1].legend(fontsize=8, loc="best")
    axes[0].set_xlabel("Ego road position x (m)")
    axes[1].set_xlabel("Ego road position x (m)")
    figure.suptitle(
        "CP-D Roadway Object: Shared Perception Enables Earlier Avoidance",
        fontsize=14,
    )
    figure.text(
        0.5, 0.01,
        "Dashed vertical line: roadway-object position. Curves are aligned by road position, not wall-clock time.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / ("cp_d_off_vs_on.%s" % suffix), dpi=180)
    plt.close(figure)

    (output_dir / "cp_d_comparison_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    flat_rows = []
    for label, summary in report.items():
        flat_rows.append({
            "condition": label,
            "blockage_detected_x_m": (
                summary["lane_blockage_first_detected"] or {}
            ).get("x_m"),
            "lane_change_start_x_m": (
                summary["lane_change_execution_first"] or {}
            ).get("x_m"),
            "minimum_approach_speed_mps": summary["minimum_approach_speed_mps"],
            "run_duration_s": summary["run_duration_s"],
            "mission_complete": summary["mission_complete"],
            "mpc_failure_ticks": summary["mpc_infeasible_or_safe_stop_ticks"],
            "fallback_ticks": summary["fallback_ticks"],
        })
    with (output_dir / "cp_d_comparison_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
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
