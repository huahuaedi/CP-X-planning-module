#!/usr/bin/env python3
"""Export a compact report proving prediction-to-MPC integration."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _number(row, key, default=np.nan):
    try:
        value = row.get(key, default)
        return float(value) if value not in (None, "") else float(default)
    except (TypeError, ValueError):
        return float(default)


def _contiguous_regions(mask, time_s):
    regions = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(mask) - 1):
            end = index if active and index == len(mask) - 1 else index - 1
            regions.append((time_s[start], time_s[end]))
            start = None
    return regions


def _shade(ax, regions, label=None):
    first = True
    for start, end in regions:
        ax.axvspan(
            start,
            end,
            color="#f2a900",
            alpha=0.16,
            linewidth=0,
            label=label if first else None,
        )
        first = False


def export(log_path, output_dir):
    rows = _read_rows(log_path)
    if not rows:
        raise RuntimeError("planner debug log is empty: %s" % log_path)

    time_s = np.asarray([_number(row, "sim_time_s", 0.0) for row in rows])
    time_s -= time_s[0]
    speed = np.asarray([_number(row, "speed_mps") for row in rows])
    target = np.asarray([
        _number(row, "speed_owner_selected_target_mps") for row in rows
    ])
    accel = np.asarray([_number(row, "accel_cmd_mps2") for row in rows])
    qp_rows = np.asarray([
        _number(row, "cav_total_qp_row_count", 0.0) for row in rows
    ])
    latency_ms = np.asarray([
        _number(row, "prediction_bridge_latency_ms") for row in rows
    ])
    mtr_attached = np.asarray([
        row.get("planner_input_prediction_model") == "mtr_http"
        and _number(row, "prediction_bridge_attached_count", 0.0) > 0.0
        for row in rows
    ])
    attached_mode_count = np.asarray([
        _number(row, "prediction_bridge_attached_mode_count", 0.0)
        for row in rows
    ])
    retained_mode_count = np.asarray([
        _number(row, "cav_retained_prediction_mode_count", 0.0)
        for row in rows
    ])
    credible_veto = np.asarray([
        _number(row, "cav_credible_mode_veto_count", 0.0)
        for row in rows
    ])
    corridor_active = qp_rows > 0.0
    active_regions = _contiguous_regions(corridor_active, time_s)

    figure, axes = plt.subplots(3, 1, figsize=(11.0, 8.2), sharex=True)
    figure.suptitle("MTR Prediction Integration into Ego MPC", fontsize=15)

    axes[0].plot(time_s, speed, color="#0068b5", linewidth=1.8, label="Ego speed")
    axes[0].plot(
        time_s, target, color="#8c1d40", linewidth=1.5,
        linestyle="--", label="Planner speed target",
    )
    _shade(axes[0], active_regions, "MTR-derived corridor active")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(loc="best", frameon=False, ncol=3)

    axes[1].plot(time_s, accel, color="#3a923a", linewidth=1.4)
    axes[1].axhline(0.0, color="#888888", linewidth=0.8)
    _shade(axes[1], active_regions)
    axes[1].set_ylabel("Acceleration (m/s²)")

    axes[2].step(
        time_s, qp_rows, where="post", color="#7a3e9d", linewidth=1.7,
        label="MPC constraint rows",
    )
    axes[2].set_ylabel("QP rows")
    axes[2].set_xlabel("Elapsed simulation time (s)")
    mode_axis = axes[2].twinx()
    mode_axis.step(
        time_s, retained_mode_count, where="post", color="#00a6a6",
        linewidth=1.5, label="Retained MTR modes",
    )
    veto_indices = np.flatnonzero(credible_veto > 0.0)
    if len(veto_indices):
        mode_axis.scatter(
            time_s[veto_indices], retained_mode_count[veto_indices],
            color="#d1495b", marker="x", s=24,
            label="Credible-mode veto",
        )
    mode_axis.set_ylabel("Prediction modes")
    mode_axis.set_ylim(-0.15, max(3.4, float(np.nanmax(attached_mode_count)) + 0.4))
    handles_left, labels_left = axes[2].get_legend_handles_labels()
    handles_right, labels_right = mode_axis.get_legend_handles_labels()
    axes[2].legend(
        handles_left + handles_right, labels_left + labels_right,
        loc="best", frameon=False, ncol=3,
    )

    for axis in axes:
        axis.grid(True, alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    mode_axis.grid(False)
    mode_axis.spines["top"].set_visible(False)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(
            output_dir / ("mtr_prediction_to_mpc.%s" % suffix),
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(figure)

    mtr_rows = [row for row in rows if row.get("planner_input_prediction_model") == "mtr_http"]
    summary = {
        "tick_count": len(rows),
        "mtr_attached_tick_count": int(np.count_nonzero(mtr_attached)),
        "maximum_attached_mode_count": int(np.nanmax(attached_mode_count)),
        "multimodal_tick_count": int(np.count_nonzero(retained_mode_count > 1.0)),
        "maximum_retained_mode_count": int(np.nanmax(retained_mode_count)),
        "credible_mode_veto_tick_count": int(np.count_nonzero(credible_veto > 0.0)),
        "mtr_request_count": int(max(_number(row, "prediction_bridge_request_count", 0.0) for row in rows)),
        "mtr_success_count": int(max(_number(row, "prediction_bridge_success_count", 0.0) for row in rows)),
        "mtr_error_values": sorted({
            str(row.get("prediction_bridge_error", ""))
            for row in rows if row.get("prediction_bridge_error", "")
        }),
        "mtr_latency_ms_mean": (
            float(np.nanmean([_number(row, "prediction_bridge_latency_ms") for row in mtr_rows]))
            if mtr_rows else None
        ),
        "mtr_latency_ms_max": (
            float(np.nanmax([_number(row, "prediction_bridge_latency_ms") for row in mtr_rows]))
            if mtr_rows else None
        ),
        "corridor_active_tick_count": int(np.count_nonzero(corridor_active)),
        "mtr_corridor_active_tick_count": int(np.count_nonzero(corridor_active & mtr_attached)),
        "maximum_mpc_constraint_rows": int(np.nanmax(qp_rows)),
        "first_corridor_active_time_s": (
            float(time_s[np.flatnonzero(corridor_active)[0]])
            if np.any(corridor_active) else None
        ),
        "last_corridor_active_time_s": (
            float(time_s[np.flatnonzero(corridor_active)[-1]])
            if np.any(corridor_active) else None
        ),
    }
    summary_path = output_dir / "mtr_prediction_to_mpc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = export(args.log, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
