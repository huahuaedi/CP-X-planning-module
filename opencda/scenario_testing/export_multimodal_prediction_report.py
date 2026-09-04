#!/usr/bin/env python3
"""Export plots and metrics for CV / primary-only / multimodal runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEBUG = ROOT / "opencda" / "planning_module" / "opencda_bridge"
ARMS = {
    "CV": DEBUG / "debug_prediction_cv_ego",
    "Primary only": DEBUG / "debug_prediction_primary_ego",
    "Multimodal": DEBUG / "debug_multimodal_ego",
}
OUTPUT = ROOT / "artifacts" / "multimodal_prediction_report"


def _number(row, key, default=float("nan")):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _rows(path):
    with (path / "opencda_planner_debug.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def _series(rows, key):
    values = []
    for row in rows:
        value = _number(row, key)
        if math.isfinite(value):
            values.append(value)
    return values


def _aligned(rows, key):
    return [_number(row, key) for row in rows]


def _metrics(rows):
    times = _series(rows, "sim_time_s")
    speeds = _series(rows, "speed_mps")
    accels = _series(rows, "measured_accel_mps2")
    gaps = [v for v in _series(rows, "front_gap_m") if v >= 0.0]
    jerk = []
    for index in range(1, min(len(times), len(accels))):
        dt_s = times[index] - times[index - 1]
        if dt_s > 1.0e-4:
            jerk.append(abs(accels[index] - accels[index - 1]) / dt_s)
    return {
        "ticks": len(rows),
        "duration_s": round(times[-1] - times[0], 3) if len(times) > 1 else 0.0,
        "mean_speed_mps": round(sum(speeds) / len(speeds), 3) if speeds else None,
        "peak_decel_mps2": round(min(accels), 3) if accels else None,
        "peak_abs_jerk_mps3": round(max(jerk), 3) if jerk else None,
        "min_front_gap_m": round(min(gaps), 3) if gaps else None,
        "fallback_ticks": sum(
            str(row.get("fallback_active", "")).lower() == "true" for row in rows
        ),
        "mpc_infeasible_ticks": sum(
            "infeasible" in str(row.get("mpc_status", "")).lower() for row in rows
        ),
        "multimodal_ticks": sum(
            _number(row, "cav_multimodal_agent_count", 0.0) > 0.0 for row in rows
        ),
        "constraint_ticks": sum(
            _number(row, "cav_total_qp_row_count", 0.0) > 0.0 for row in rows
        ),
    }


def main():
    available = {
        name: _rows(path) for name, path in ARMS.items()
        if (path / "opencda_planner_debug.csv").is_file()
    }
    if not available:
        raise SystemExit("No experiment logs found; run the three scenarios first.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = {name: _metrics(rows) for name, rows in available.items()}
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for name, rows in available.items():
        time = _series(rows, "sim_time_s")
        if not time:
            continue
        time = [value - time[0] for value in time]
        axes[0].plot(time, _aligned(rows, "speed_mps")[:len(time)], label=name)
        axes[1].plot(
            time, _aligned(rows, "measured_accel_mps2")[:len(time)], label=name
        )
        axes[2].plot(
            time, _aligned(rows, "cav_total_qp_row_count")[:len(time)], label=name
        )
    axes[0].set_ylabel("Speed (m/s)")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[2].set_ylabel("Prediction QP rows")
    axes[2].set_xlabel("Relative simulation time (s)")
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "prediction_fusion_comparison.png", dpi=180)
    print("wrote", OUTPUT / "metrics.json")
    print("wrote", OUTPUT / "prediction_fusion_comparison.png")


if __name__ == "__main__":
    main()
