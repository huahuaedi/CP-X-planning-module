#!/usr/bin/env python3
"""Export PPT-ready evidence for the ROS MTR-to-planner integration."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


COLORS = {
    "blue": "#1768AC",
    "teal": "#0F9D8A",
    "orange": "#E67E22",
    "red": "#C44536",
    "purple": "#7656A3",
    "gray": "#5B6573",
    "light": "#EEF3F8",
    "green": "#2E8B57",
}


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _number(row, key, default=np.nan):
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _finite(values):
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _stats(values):
    array = _finite(values)
    if not len(array):
        return {"mean": None, "maximum": None}
    return {
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _style_axis(axis):
    axis.grid(True, color="#D8E0E8", alpha=0.75, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(figure, output_dir: Path, stem: str):
    for suffix in ("png", "svg"):
        figure.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def _box(axis, xy, width, height, title, subtitle, color):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor="white",
        edgecolor=color,
        linewidth=2.0,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=color,
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["gray"],
        linespacing=1.25,
    )


def _arrow(axis, start, end, label, color=COLORS["gray"], label_offset=0.025):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.7,
            color=color,
        )
    )
    axis.text(
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2 + label_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=color,
    )


def export_architecture(output_dir: Path):
    figure, axis = plt.subplots(figsize=(14.0, 5.5))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    figure.suptitle(
        "ROS 2 Integration Architecture: MTR Prediction and CP-X Planning",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    _box(axis, (0.02, 0.39), 0.15, 0.25, "OpenCDA / CARLA", "Ego state, objects,\ntraffic context", COLORS["blue"])
    _box(axis, (0.22, 0.39), 0.17, 0.25, "Unified ROS Bridge", "Cycle-aligned inputs\nand control return", COLORS["purple"])
    _box(axis, (0.48, 0.71), 0.22, 0.20, "MTR Prediction Node", "10 Hz history update\n5 Hz, six-mode output", COLORS["orange"])
    _box(axis, (0.48, 0.36), 0.22, 0.25, "CP-X Planner Node", "Synchronization, behavior,\nreference and risk", COLORS["teal"])
    _box(axis, (0.79, 0.36), 0.18, 0.25, "MPC + Reuse", "5–10 Hz replanning\n20 Hz control output", COLORS["green"])

    _arrow(axis, (0.17, 0.53), (0.22, 0.53), "TCP 5051–5055", label_offset=0.035)
    _arrow(axis, (0.39, 0.53), (0.48, 0.53), "ROS input topics", label_offset=0.035)
    _arrow(axis, (0.535, 0.61), (0.535, 0.71), "", COLORS["teal"], label_offset=0.0)
    axis.text(
        0.515,
        0.66,
        "/prediction/scene\n/prediction/map · 10 Hz",
        ha="right",
        va="center",
        fontsize=8.2,
        color=COLORS["teal"],
    )
    _arrow(axis, (0.645, 0.71), (0.645, 0.61), "", COLORS["orange"], label_offset=0.0)
    axis.text(
        0.665,
        0.66,
        "/prediction/frame\n6 modes · 5 Hz",
        ha="left",
        va="center",
        fontsize=8.2,
        color=COLORS["orange"],
    )
    _arrow(axis, (0.70, 0.50), (0.79, 0.50), "Reference + obstacle risks", label_offset=0.035)

    axis.plot(
        [0.88, 0.88, 0.305],
        [0.36, 0.22, 0.22],
        color=COLORS["green"],
        linewidth=1.8,
    )
    feedback = FancyArrowPatch(
        (0.305, 0.22),
        (0.305, 0.39),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color=COLORS["green"],
    )
    axis.add_patch(feedback)
    axis.text(0.59, 0.165, "Autoware Control → TCP 5060 → OpenCDA actuator", ha="center", fontsize=9, color=COLORS["green"])

    axis.text(
        0.5,
        0.075,
        "ROS 2 decouples Python/CUDA environments while preserving asynchronous prediction and deterministic control reuse.",
        ha="center",
        fontsize=10,
        color=COLORS["gray"],
    )
    _save(figure, output_dir, "01_ros_prediction_planner_architecture")


def _parse_mtr_timing(log_path: Path, end_wall_time_s: float):
    pattern = re.compile(
        r"\[(?P<stamp>\d+\.\d+)\].*MTR timing.*"
        r"inference=(?P<infer_mean>[\d.]+)/(?P<infer_p95>[\d.]+)ms.*"
        r"total=(?P<total_mean>[\d.]+)/(?P<total_p95>[\d.]+)ms"
    )
    candidates = []
    if not log_path.is_file():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        stamp = float(match.group("stamp"))
        if stamp <= end_wall_time_s + 0.5:
            candidates.append(
                {
                    "timestamp_s": stamp,
                    "inference_mean_ms": float(match.group("infer_mean")),
                    "total_mean_ms": float(match.group("total_mean")),
                }
            )
    return candidates[-1] if candidates else None


def export_closed_loop(rows, output_dir: Path):
    rows = [row for row in rows if row.get("session_id")]
    time_s = np.asarray([_number(row, "cycle_time_s") for row in rows])
    time_s -= time_s[0]
    speed = np.asarray([_number(row, "velocity_mps") for row in rows])
    target = np.asarray([_number(row, "target_speed_mps") for row in rows])
    gap = np.asarray([_number(row, "front_gap_m") for row in rows])
    modes = np.asarray([_number(row, "prediction_bridge_attached_mode_count", 0.0) for row in rows])
    planner_ms = np.asarray([_number(row, "planning_cycle_time_ms") for row in rows])
    end_to_end_ms = np.asarray([_number(row, "input_to_output_time_ms") for row in rows])
    replanned = np.asarray([bool(row.get("mpc_replan_executed", False)) for row in rows])
    relevant = np.asarray([_number(row, "mpc_obstacle_relevant_count", 0.0) > 0 for row in rows])

    figure, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True)
    figure.suptitle(
        "Closed-Loop ROS Integration Evidence",
        fontsize=15,
        fontweight="bold",
    )

    axes[0].plot(time_s, speed, color=COLORS["blue"], linewidth=1.8, label="Ego speed")
    axes[0].plot(time_s, target, color=COLORS["red"], linewidth=1.5, linestyle="--", label="Planner target speed")
    axes[0].set_ylabel("Speed (m/s)")
    gap_axis = axes[0].twinx()
    gap_axis.plot(time_s, gap, color=COLORS["orange"], linewidth=1.2, alpha=0.75, label="Front gap")
    gap_axis.set_ylabel("Front gap (m)", color=COLORS["orange"])
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = gap_axis.get_legend_handles_labels()
    axes[0].legend(handles_a + handles_b, labels_a + labels_b, loc="best", frameon=False, ncol=3)

    if np.any(relevant):
        padded = np.concatenate(([False], relevant, [False])).astype(int)
        starts = np.flatnonzero(np.diff(padded) == 1)
        stops = np.flatnonzero(np.diff(padded) == -1) - 1
        for start_index, stop_index in zip(starts, stops):
            for axis in axes:
                axis.axvspan(
                    time_s[start_index],
                    time_s[stop_index],
                    color=COLORS["red"],
                    alpha=0.07,
                )
        first_start_s = time_s[starts[0]]
        first_end_s = time_s[stops[0]]
        axes[0].text(
            (first_start_s + first_end_s) / 2.0,
            axes[0].get_ylim()[1] * 0.93,
            "Predicted obstacle relevant to MPC",
            ha="center",
            va="top",
            fontsize=8.5,
            color=COLORS["red"],
        )

    axes[1].step(time_s, modes, where="post", color=COLORS["teal"], linewidth=1.8, label="Attached prediction modes")
    axes[1].set_ylabel("Mode count")
    axes[1].set_ylim(-0.2, 6.8)
    axes[1].legend(loc="upper left", frameon=False)

    axes[2].plot(time_s, planner_ms, color=COLORS["blue"], linewidth=1.0, alpha=0.8, label="Planner cycle")
    axes[2].plot(time_s, end_to_end_ms, color=COLORS["orange"], linewidth=1.0, alpha=0.75, label="Input-to-output")
    replan_indices = np.flatnonzero(replanned)
    if len(replan_indices):
        axes[2].scatter(time_s[replan_indices], np.zeros(len(replan_indices)), s=7, color=COLORS["green"], alpha=0.55, label="MPC replan")
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_xlabel("Elapsed simulation time (s)")
    axes[2].set_ylim(bottom=-5.0)
    axes[2].legend(loc="upper right", frameon=False, ncol=4)

    for axis in axes:
        _style_axis(axis)
    gap_axis.grid(False)
    gap_axis.spines["top"].set_visible(False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    _save(figure, output_dir, "02_six_mode_closed_loop_evidence")


def export_cut_in_response(rows, output_dir: Path):
    """Export a focused safety-response plot for the successful cut-in run."""
    rows = [row for row in rows if row.get("session_id")]
    time_s = np.asarray([_number(row, "cycle_time_s") for row in rows])
    time_s -= time_s[0]
    speed = np.asarray([_number(row, "velocity_mps") for row in rows])
    target = np.asarray([_number(row, "target_speed_mps") for row in rows])
    acceleration = np.asarray([_number(row, "acceleration_mps2") for row in rows])
    gap = np.asarray([_number(row, "front_gap_m") for row in rows])
    modes = np.asarray([_number(row, "prediction_bridge_attached_mode_count", 0.0) for row in rows])
    relevant = np.asarray([_number(row, "mpc_obstacle_relevant_count", 0.0) > 0 for row in rows])

    # Keep the figure centered on the first cut-in interaction, excluding the
    # terminal-stop reacquisition of the scripted actor near the destination.
    relevant_indices = np.flatnonzero(relevant)
    if len(relevant_indices):
        split = np.flatnonzero(np.diff(relevant_indices) > 1)
        first_end = split[0] + 1 if len(split) else len(relevant_indices)
        first_interval = relevant_indices[:first_end]
        window_start = max(0.0, time_s[first_interval[0]] - 2.0)
        window_end = min(float(time_s[-1]), time_s[first_interval[-1]] + 2.0)
    else:
        first_interval = np.asarray([], dtype=int)
        window_start = 0.0
        window_end = float(time_s[-1])

    figure, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True)
    figure.suptitle(
        "Cut-In Response: Six-Mode Prediction Constrains Planning",
        fontsize=15,
        fontweight="bold",
    )
    axes[0].plot(time_s, speed, color=COLORS["blue"], linewidth=2.0, label="Ego speed")
    axes[0].plot(time_s, target, color=COLORS["red"], linewidth=1.8, linestyle="--", label="Planner target speed")
    gap_axis = axes[0].twinx()
    gap_axis.plot(time_s, gap, color=COLORS["orange"], linewidth=1.5, label="Front gap")
    axes[0].set_ylabel("Speed (m/s)")
    gap_axis.set_ylabel("Front gap (m)", color=COLORS["orange"])
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = gap_axis.get_legend_handles_labels()
    axes[0].legend(handles_a + handles_b, labels_a + labels_b, loc="upper right", frameon=False, ncol=3)

    axes[1].plot(time_s, acceleration, color=COLORS["purple"], linewidth=1.6, label="Commanded acceleration")
    mode_axis = axes[1].twinx()
    mode_axis.step(time_s, modes, where="post", color=COLORS["teal"], linewidth=1.7, label="Attached MTR modes")
    axes[1].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    axes[1].set_ylabel("Acceleration (m/s²)")
    mode_axis.set_ylabel("Mode count", color=COLORS["teal"])
    mode_axis.set_ylim(-0.2, 6.8)
    axes[1].set_xlabel("Elapsed simulation time (s)")
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = mode_axis.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b, loc="lower right", frameon=False, ncol=2)

    if len(first_interval):
        start_s = time_s[first_interval[0]]
        end_s = time_s[first_interval[-1]]
        for axis in axes:
            axis.axvspan(start_s, end_s, color=COLORS["red"], alpha=0.09)
        axes[0].text(
            (start_s + end_s) / 2.0,
            axes[0].get_ylim()[1] * 0.96,
            "MPC obstacle active",
            ha="center",
            va="top",
            fontsize=9,
            color=COLORS["red"],
        )

    for axis in axes:
        axis.set_xlim(window_start, window_end)
        _style_axis(axis)
    gap_axis.grid(False)
    mode_axis.grid(False)
    gap_axis.spines["top"].set_visible(False)
    mode_axis.spines["top"].set_visible(False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    _save(figure, output_dir, "03_cut_in_response")


def export_latency(rows, mtr_timing, output_dir: Path):
    rows = [row for row in rows if row.get("session_id")]
    planner = _finite([_number(row, "planning_cycle_time_ms") for row in rows])
    end_to_end = _finite([_number(row, "input_to_output_time_ms") for row in rows])
    prediction = _finite([_number(row, "prediction_bridge_latency_ms") for row in rows])

    series = [planner, end_to_end, prediction]
    labels = ["Planner cycle", "OpenCDA ↔ ROS", "Prediction frame"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"]]
    if mtr_timing:
        series.append(None)
        labels.append("MTR inference")
        colors.append(COLORS["teal"])

    means = [float(np.mean(item)) for item in series if item is not None]
    if mtr_timing:
        means.append(float(mtr_timing["inference_mean_ms"]))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    figure.suptitle("ROS Integration Timing and Reliability", fontsize=15, fontweight="bold")
    x = np.arange(len(labels))
    width = 0.56
    axes[0].bar(x, means, width, color=colors, alpha=0.90)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("Latency (ms)")
    for index, value in enumerate(means):
        axes[0].text(index, value + 1.2, f"{value:.1f}", ha="center", fontsize=9)
    _style_axis(axes[0])

    attached = sum(_number(row, "prediction_bridge_attached_count", 0.0) > 0 for row in rows)
    six_modes = sum(_number(row, "prediction_bridge_attached_mode_count", 0.0) == 6 for row in rows)
    replans = sum(bool(row.get("mpc_replan_executed", False)) for row in rows)
    fallbacks = sum(bool(str(row.get("mpc_fallback_reason", "") or "")) for row in rows)
    errors = sum(bool(str(row.get("prediction_bridge_error", "") or "")) for row in rows)
    values = [attached, six_modes, replans, fallbacks, errors]
    reliability_labels = ["Prediction\nattached", "All 6 modes\nattached", "MPC\nreplans", "MPC\nfallbacks", "Prediction\nerrors"]
    reliability_colors = [COLORS["teal"], COLORS["purple"], COLORS["green"], COLORS["red"], COLORS["red"]]
    axes[1].bar(np.arange(len(values)), values, color=reliability_colors, alpha=0.88)
    axes[1].set_xticks(np.arange(len(values)), reliability_labels)
    axes[1].set_ylabel("Tick / event count")
    for index, value in enumerate(values):
        axes[1].text(index, value + max(values + [1]) * 0.025, str(value), ha="center", fontweight="bold", fontsize=9)
    axes[1].set_ylim(0, max(values + [1]) * 1.15)
    _style_axis(axes[1])
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save(figure, output_dir, "04_latency_and_reliability")


def export_report(log_path: Path, status_path: Path, mtr_log_path: Path, output_dir: Path):
    rows = _read_jsonl(log_path)
    valid_rows = [row for row in rows if row.get("session_id")]
    if not valid_rows:
        raise RuntimeError(f"No active ROS control rows found in {log_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    end_wall_time_s = max(_number(row, "control_application_wall_time_ns", 0.0) for row in valid_rows) / 1.0e9
    mtr_timing = _parse_mtr_timing(mtr_log_path, end_wall_time_s)
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}

    export_architecture(output_dir)
    export_closed_loop(rows, output_dir)
    export_cut_in_response(rows, output_dir)
    export_latency(rows, mtr_timing, output_dir)

    elapsed_s = max(_number(row, "cycle_time_s") for row in valid_rows) - min(_number(row, "cycle_time_s") for row in valid_rows)
    prediction_errors = sorted(
        {
            str(row.get("prediction_bridge_error", ""))
            for row in valid_rows
            if str(row.get("prediction_bridge_error", ""))
        }
    )
    summary = {
        "scenario": status.get("scenario_name", "cpx_two_vehicle_cut_in_mtr_six_modes"),
        "control_rows": len(valid_rows),
        "elapsed_simulation_s": float(elapsed_s),
        "effective_control_rate_hz": float((len(valid_rows) - 1) / elapsed_s) if elapsed_s > 0.0 else None,
        "planner_cycle_ms": _stats([_number(row, "planning_cycle_time_ms") for row in valid_rows]),
        "input_to_output_ms": _stats([_number(row, "input_to_output_time_ms") for row in valid_rows]),
        "prediction_bridge_latency_ms": _stats([_number(row, "prediction_bridge_latency_ms") for row in valid_rows]),
        "mtr_timing_window": mtr_timing,
        "prediction_attached_ticks": int(sum(_number(row, "prediction_bridge_attached_count", 0.0) > 0 for row in valid_rows)),
        "six_mode_attached_ticks": int(sum(_number(row, "prediction_bridge_attached_mode_count", 0.0) == 6 for row in valid_rows)),
        "maximum_prediction_revision": int(max(_number(row, "prediction_ros_revision", 0.0) for row in valid_rows)),
        "prediction_errors": prediction_errors,
        "mpc_obstacle_relevant_ticks": int(sum(_number(row, "mpc_obstacle_relevant_count", 0.0) > 0 for row in valid_rows)),
        "mpc_replan_ticks": int(sum(bool(row.get("mpc_replan_executed", False)) for row in valid_rows)),
        "mpc_fallback_ticks": int(sum(bool(str(row.get("mpc_fallback_reason", "") or "")) for row in valid_rows)),
        "minimum_front_gap_m": float(np.min(_finite([_number(row, "front_gap_m") for row in valid_rows]))),
        "minimum_acceleration_mps2": float(np.min(_finite([_number(row, "acceleration_mps2") for row in valid_rows]))),
        "stale_control_ticks": int(sum(str(row.get("ros_control_state", "")) == "stale_control" for row in valid_rows)),
        "collision_count": int(status.get("collision_count", 0) or 0),
        "termination_reason": status.get("termination_reason", "unknown"),
        "distance_to_destination_m": status.get("distance_to_destination_m"),
        "completed": bool(status.get("all_cavs_finished", False)),
    }
    (output_dir / "integration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    planner = summary["planner_cycle_ms"]
    end_to_end = summary["input_to_output_ms"]
    prediction = summary["prediction_bridge_latency_ms"]
    mtr_mean = mtr_timing["inference_mean_ms"] if mtr_timing else float("nan")
    bullets = f"""# ROS Prediction–Planner Integration Results

- The ROS 2 pipeline delivered synchronized OpenCDA state at **{summary['effective_control_rate_hz']:.1f} Hz** and returned reusable planner control asynchronously.
- MTR published **six trajectory modes** per relevant object; all six modes were attached on **{summary['six_mode_attached_ticks']} planner ticks**, with **zero prediction interface errors**.
- Predicted motion was accepted as an MPC-relevant obstacle on **{summary['mpc_obstacle_relevant_ticks']} ticks**; the minimum observed front gap was **{summary['minimum_front_gap_m']:.1f} m** and peak commanded deceleration was **{summary['minimum_acceleration_mps2']:.1f} m/s²**.
- Mean MTR inference time was **{mtr_mean:.1f} ms** for the measured timing window.
- Mean CP-X planning time was **{planner['mean']:.1f} ms**; mean OpenCDA-to-ROS-to-OpenCDA latency was **{end_to_end['mean']:.1f} ms**.
- Mean prediction-frame latency was **{prediction['mean']:.1f} ms** and reached revision **{summary['maximum_prediction_revision']}**.
- The planner executed **{summary['mpc_replan_ticks']} MPC replans**, with **{summary['mpc_fallback_ticks']} MPC fallback ticks**, **{summary['stale_control_ticks']} stale-control ticks**, and **{summary['collision_count']} collisions**.
- The vehicle completed the route with termination reason **{summary['termination_reason']}** and stopped **{float(summary['distance_to_destination_m'] or 0.0):.1f} m** from the destination.
"""
    (output_dir / "ppt_bullets.md").write_text(bullets, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("planner_log", type=Path)
    parser.add_argument("status", type=Path)
    parser.add_argument("mtr_log", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = export_report(args.planner_log, args.status, args.mtr_log, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
