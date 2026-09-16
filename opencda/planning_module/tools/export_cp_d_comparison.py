"""Compare CP-D roadway-object awareness with the resulting ego maneuver."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OBJECT_X_M = -180.0
OBJECT_Y_M = 8.3


def _load(directory: Path):
    path = directory / "opencda_planner_debug.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise ValueError(f"No diagnostics in {path}")
    front_ids = Counter(str(row.get("front_gap_actor_id") or "") for row in rows)
    front_ids.pop("", None)
    target_id = front_ids.most_common(1)[0][0]
    return rows, target_id


def _first(rows, condition):
    return next((row for row in rows if condition(row)), None)


def _first_slowdown(rows, limit_mps: float):
    cruising = False
    for row in rows:
        speed = float(row.get("speed_mps") or 0.0)
        cruising |= speed > 7.5
        if cruising and speed < limit_mps:
            return float(row["x_m"])
    return None


def export(off_dir: Path, on_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {"CP OFF": _load(off_dir), "CP ON": _load(on_dir)}
    colors = {"CP OFF": "#d97706", "CP ON": "#177e89"}
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    report = {}

    for label, (rows, target_id) in runs.items():
        color = colors[label]
        road_x = np.asarray([float(row["x_m"]) for row in rows])
        road_y = np.asarray([float(row["y_m"]) for row in rows])
        speed = np.asarray([float(row.get("speed_mps") or 0.0) for row in rows])
        front_gap = np.asarray([
            float(row.get("speed_plan_front_gap_m"))
            if str(row.get("front_gap_actor_id") or "") == target_id
            and row.get("speed_plan_front_gap_m") is not None
            else np.nan
            for row in rows
        ])
        axes[0].plot(road_x, speed, color=color, lw=1.8, label=label)
        axes[1].plot(road_x, front_gap, color=color, lw=1.6, label=label)
        axes[2].plot(road_x, road_y, color=color, lw=1.5, label=label)

        first_target = _first(rows, lambda row:
                              str(row.get("front_gap_actor_id") or "") == target_id)
        first_local = _first(rows, lambda row:
                             int(row.get("local_object_count") or 0) >= 2)
        first_x = float(first_target["x_m"]) if first_target else None
        local_x = float(first_local["x_m"]) if first_local else None
        if first_x is not None:
            axes[0].axvline(first_x, color=color, lw=1.1, ls="--",
                            label=f"{label} target first used: x={first_x:.1f} m")
        report[label] = {
            "ego_actor_id": rows[0].get("vehicle_id"),
            "stationary_truck_actor_id": target_id,
            "first_target_used_x_m": first_x,
            "first_local_target_x_m": local_x,
            "first_speed_below_7_x_m": _first_slowdown(rows, 7.0),
            "first_speed_below_5_x_m": _first_slowdown(rows, 5.0),
            "last_x_m": float(rows[-1]["x_m"]),
            "last_y_m": float(rows[-1]["y_m"]),
            "last_speed_mps": float(rows[-1]["speed_mps"]),
            "destination_reached": any(bool(row.get("route_reached_destination"))
                                       for row in rows),
            "lane_change_decision_ticks": sum(
                "lane_change" in str(row.get("decision_behavior") or "").lower()
                for row in rows
            ),
            "opportunistic_lane_change_allowed_ticks": sum(
                bool(row.get("opportunistic_lane_change_allowed")) for row in rows
            ),
        }

    axes[0].set(title="CP-D: Earlier shared awareness, unchanged lane-follow decision",
                ylabel="Ego speed (m/s)")
    axes[1].set(ylabel="Tracked gap to stopped truck (m)")
    axes[2].axhline(OBJECT_Y_M, color="#475569", lw=1.0, ls=":",
                    label="Truck lane center: y=8.3 m")
    axes[2].axhline(4.8, color="#94a3b8", lw=1.0, ls=":",
                    label="Adjacent lane center: y=4.8 m")
    axes[2].scatter([OBJECT_X_M], [OBJECT_Y_M], marker="X", s=110,
                    color="#b91c1c", label="Stationary truck")
    axes[2].set(xlabel="Ego road position x (m); travel runs left to right",
                ylabel="Ego lateral position y (m)", ylim=(3.5, 10.0))
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
        axis.set_xlim(-295, -175)
    figure.text(0.02, 0.005,
                "Target first-used markers refer to the stationary truck, not the observer CAV. "
                "Neither run reached the destination within 650 ticks.", fontsize=8)
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"cp_d_off_vs_on.{suffix}", dpi=180)
    plt.close(figure)
    (output_dir / "cp_d_comparison_summary.json").write_text(
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
