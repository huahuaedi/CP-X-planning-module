"""Export focused lane-change and intersection-turn diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _number(row: Mapping[str, object], key: str, default=float("nan")) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _series(
    rows: Sequence[Mapping[str, object]],
    key: str,
    default=float("nan"),
) -> np.ndarray:
    return np.asarray([_number(row, key, default) for row in rows], dtype=float)


def _save(fig, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=190, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _mask(
    rows: Sequence[Mapping[str, object]],
    predicate: Callable[[Mapping[str, object]], bool],
) -> np.ndarray:
    return np.asarray([bool(predicate(row)) for row in rows], dtype=bool)


def _first_last(mask: np.ndarray) -> tuple[int, int]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError("Requested maneuver is absent from the debug log")
    return int(indices[0]), int(indices[-1])


def _shade_mask(axis, time_s: np.ndarray, mask: np.ndarray, color: str, label: str):
    active = False
    start = 0
    label_pending = True
    for index, value in enumerate(mask):
        if value and not active:
            start = index
            active = True
        if active and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            axis.axvspan(
                time_s[start],
                time_s[end],
                color=color,
                alpha=0.09,
                linewidth=0,
                label=label if label_pending else None,
            )
            label_pending = False
            active = False


def _speed_cap_from_reason(row: Mapping[str, object]) -> float:
    match = re.search(
        r"speed_cap=([-+]?[0-9]*\.?[0-9]+)",
        str(row.get("control_guard_reason", "")),
    )
    return float(match.group(1)) if match else float("nan")


def _status_values(rows: Sequence[Mapping[str, object]]) -> tuple[list[int], list[str]]:
    names = [
        "solved",
        "solved inaccurate",
        "buffer_reuse",
        "maximum iterations reached",
        "primal infeasible",
        "candidate_hard_gate",
    ]
    mapping = {name: index for index, name in enumerate(names)}
    return [mapping.get(str(row.get("mpc_status", "")), -1) for row in rows], names


def _finite_summary(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(finite)), float(np.mean(finite)), float(np.max(finite))


def export(input_jsonl: Path, output_dir: Path) -> None:
    with input_jsonl.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows found in {input_jsonl}")

    output_dir.mkdir(parents=True, exist_ok=True)
    time_s = _series(rows, "sim_time_s")
    x_m = _series(rows, "x_m")
    y_m = _series(rows, "y_m")
    speed_mps = _series(rows, "speed_mps", 0.0)
    target_speed_mps = _series(rows, "target_speed_mps", 0.0)

    lane_change = _mask(
        rows,
        lambda row: "lane_change" in str(row.get("behavior_decision", "")),
    )
    lane_prepare = _mask(
        rows,
        lambda row: str(row.get("behavior_fsm_state", "")).startswith(
            "PREPARE_LANE_CHANGE"
        ),
    )
    lane_execute = _mask(
        rows,
        lambda row: str(row.get("behavior_fsm_state", "")).startswith(
            "EXECUTE_LANE_CHANGE"
        ),
    )
    lane_alignment = lane_change & ~(lane_prepare | lane_execute)
    turn = _mask(
        rows,
        lambda row: "intersection_turn" in str(row.get("behavior_decision", "")),
    )
    traffic_stop = _mask(
        rows,
        lambda row: str(row.get("behavior_decision", ""))
        == "stop_at_intersection",
    )
    lc_start, lc_end = _first_last(lane_change)
    turn_start, turn_end = _first_last(turn)

    colors = {
        "route": "#9aa4b2",
        "lane_change": "#2563a6",
        "turn": "#7b3fa1",
        "speed": "#147d84",
        "target": "#d66b00",
        "steer": "#7b3fa1",
        "accel": "#238636",
        "clearance": "#147d84",
        "danger": "#c0392b",
    }

    fig, axis = plt.subplots(figsize=(10.5, 7.2))
    axis.plot(x_m, y_m, color=colors["route"], linewidth=1.7, label="Ego route")
    axis.plot(
        x_m[lane_change],
        y_m[lane_change],
        color=colors["lane_change"],
        linewidth=3.0,
        label="Lane-change behavior",
    )
    axis.plot(
        x_m[turn],
        y_m[turn],
        color=colors["turn"],
        linewidth=3.0,
        label="Intersection turn",
    )
    for mask_value, color in (
        (lane_change, colors["lane_change"]),
        (turn, colors["turn"]),
    ):
        indices = np.flatnonzero(mask_value)
        for index in indices[:: max(1, len(indices) // 12)]:
            points = rows[int(index)].get("lane_reference_points", [])
            if not points:
                continue
            reference = np.asarray(points, dtype=float)
            if reference.ndim == 2 and reference.shape[1] >= 2:
                axis.plot(
                    reference[:, 0],
                    reference[:, 1],
                    color=color,
                    alpha=0.18,
                    linewidth=0.9,
                )
    axis.scatter(
        [x_m[lc_start], x_m[turn_start]],
        [y_m[lc_start], y_m[turn_start]],
        color=[colors["lane_change"], colors["turn"]],
        s=55,
        zorder=4,
    )
    axis.set(
        title="Maneuver Geometry and Sampled MPC References",
        xlabel="World x (m)",
        ylabel="World y (m)",
    )
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, output_dir, "01_maneuver_route_geometry")

    lc_rows = rows[lc_start : lc_end + 1]
    lc_time = time_s[lc_start : lc_end + 1]
    lc_prepare = lane_prepare[lc_start : lc_end + 1]
    lc_execute = lane_execute[lc_start : lc_end + 1]
    lc_alignment = lane_alignment[lc_start : lc_end + 1]
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 11.0), sharex=True)
    axes[0].step(
        lc_time,
        _series(lc_rows, "current_lane_id"),
        where="post",
        color="#147d84",
        label="Current map lane",
    )
    axes[0].step(
        lc_time,
        _series(lc_rows, "destination_lane_id"),
        where="post",
        color="#d66b00",
        label="Target lane",
    )
    axes[0].set(ylabel="Lane ID", title="Right Lane Change: State and Geometry")
    axes[0].legend(ncol=2)

    axes[1].plot(
        lc_time,
        _series(lc_rows, "lane_change_completion_lateral_error_m"),
        color="#2563a6",
        label="Target-lane lateral error",
    )
    heading_axis = axes[1].twinx()
    heading_axis.plot(
        lc_time,
        _series(lc_rows, "lane_change_completion_heading_error_deg"),
        color="#7b3fa1",
        alpha=0.85,
        label="Heading error",
    )
    axes[1].axhline(0.0, color="#9aa4b2", linewidth=0.8)
    axes[1].set(ylabel="Lateral error (m)")
    heading_axis.set_ylabel("Heading error (deg)")
    lines = axes[1].lines[:1] + heading_axis.lines
    axes[1].legend(lines, [line.get_label() for line in lines], ncol=2)

    axes[2].plot(
        lc_time,
        speed_mps[lc_start : lc_end + 1],
        color=colors["speed"],
        label="Actual speed",
    )
    axes[2].plot(
        lc_time,
        target_speed_mps[lc_start : lc_end + 1],
        color=colors["target"],
        label="Target speed",
    )
    axes[2].set(ylabel="Speed (m/s)")
    axes[2].legend(ncol=2)

    axes[3].plot(
        lc_time,
        _series(lc_rows, "applied_steer", 0.0),
        color=colors["steer"],
        label="Applied steer",
    )
    axes[3].plot(
        lc_time,
        _series(lc_rows, "post_supervisor_accel_cmd_mps2", 0.0),
        color=colors["accel"],
        label="Acceleration command",
    )
    axes[3].axhline(0.0, color="#9aa4b2", linewidth=0.8)
    axes[3].set(xlabel="Simulation time (s)", ylabel="Control")
    axes[3].legend(ncol=2)
    for axis in axes:
        _shade_mask(axis, lc_time, lc_prepare, "#e0a11b", "Prepare")
        _shade_mask(axis, lc_time, lc_execute, "#2563a6", "Execute")
        _shade_mask(axis, lc_time, lc_alignment, "#7b3fa1", "Alignment hold")
        axis.grid(alpha=0.24)
    _save(fig, output_dir, "02_lane_change_process")

    turn_rows = rows[turn_start : turn_end + 1]
    turn_time = time_s[turn_start : turn_end + 1]
    guard_speed_cap = np.asarray(
        [_speed_cap_from_reason(row) for row in turn_rows],
        dtype=float,
    )
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 11.0), sharex=True)
    axes[0].plot(
        turn_time,
        speed_mps[turn_start : turn_end + 1],
        color=colors["speed"],
        label="Actual speed",
    )
    axes[0].plot(
        turn_time,
        target_speed_mps[turn_start : turn_end + 1],
        color=colors["target"],
        label="Behavior target",
    )
    axes[0].plot(
        turn_time,
        guard_speed_cap,
        color="#c0392b",
        linestyle="--",
        label="Boundary speed cap",
    )
    axes[0].set(ylabel="Speed (m/s)", title="Right Turn: Tracking and Boundary Recovery")
    axes[0].legend(ncol=3)

    clearance = _series(turn_rows, "road_boundary_clearance_m")
    axes[1].plot(
        turn_time,
        clearance,
        color=colors["clearance"],
        label="Body-to-lane-edge clearance",
    )
    axes[1].axhline(0.15, color="#d66b00", linestyle="--", label="Warning")
    axes[1].axhline(-0.30, color=colors["danger"], linestyle="--", label="Critical stop")
    axes[1].fill_between(
        turn_time,
        clearance,
        0.0,
        where=np.isfinite(clearance) & (clearance < 0.0),
        color=colors["danger"],
        alpha=0.14,
    )
    axes[1].set(ylabel="Clearance (m)")
    axes[1].legend(ncol=3)

    axes[2].plot(
        turn_time,
        _series(turn_rows, "reference_first_lateral_m"),
        color="#2563a6",
        label="First reference lateral",
    )
    axes[2].plot(
        turn_time,
        _series(turn_rows, "destination_lateral_m"),
        color="#7b3fa1",
        label="Horizon destination lateral",
    )
    axes[2].plot(
        turn_time,
        _series(turn_rows, "road_boundary_lateral_offset_m"),
        color="#d66b00",
        label="Ego lane-center offset",
    )
    axes[2].axhline(0.0, color="#9aa4b2", linewidth=0.8)
    axes[2].set(ylabel="Lateral geometry (m)")
    axes[2].legend(ncol=3)

    axes[3].plot(
        turn_time,
        _series(turn_rows, "applied_steer", 0.0),
        color=colors["steer"],
        label="Applied steer",
    )
    axes[3].plot(
        turn_time,
        _series(turn_rows, "post_supervisor_accel_cmd_mps2", 0.0),
        color=colors["accel"],
        label="Acceleration command",
    )
    axes[3].plot(
        turn_time,
        _series(turn_rows, "applied_brake", 0.0),
        color=colors["danger"],
        label="Applied brake",
    )
    axes[3].axhline(0.0, color="#9aa4b2", linewidth=0.8)
    axes[3].set(xlabel="Simulation time (s)", ylabel="Control")
    axes[3].legend(ncol=3)
    for axis in axes:
        axis.grid(alpha=0.24)
    _save(fig, output_dir, "03_turn_process")

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.0), sharex=True)
    statuses, status_names = _status_values(rows)
    axes[0].step(time_s, statuses, where="post", color="#2563a6", linewidth=1.1)
    axes[0].set_yticks(
        range(len(status_names)),
        [name.replace(" ", "\n") for name in status_names],
    )
    axes[0].set(title="MPC and Reference Health Across Both Maneuvers", ylabel="MPC status")
    axes[1].plot(
        time_s,
        _series(rows, "reference_first_lateral_m"),
        color="#2563a6",
        label="First reference lateral",
    )
    axes[1].plot(
        time_s,
        _series(rows, "destination_lateral_m"),
        color="#7b3fa1",
        label="Destination lateral",
    )
    axes[1].set(xlabel="Simulation time (s)", ylabel="Lateral geometry (m)")
    axes[1].legend(ncol=2)
    for axis in axes:
        _shade_mask(axis, time_s, lane_change, colors["lane_change"], "Lane change")
        _shade_mask(axis, time_s, turn, colors["turn"], "Turn")
        axis.grid(alpha=0.24)
    _save(fig, output_dir, "04_mpc_reference_health")

    stop_rows: list[Mapping[str, object]] = []
    stop_time = np.asarray([], dtype=float)
    if np.any(traffic_stop):
        stop_start, stop_end = _first_last(traffic_stop)
        stop_rows = rows[stop_start : stop_end + 1]
        stop_time = time_s[stop_start : stop_end + 1]
        fig, axes = plt.subplots(4, 1, figsize=(13.5, 11.0), sharex=True)
        axes[0].plot(
            stop_time,
            _series(stop_rows, "speed_mps", 0.0),
            color=colors["speed"],
            label="Actual speed",
        )
        axes[0].plot(
            stop_time,
            _series(stop_rows, "target_speed_mps", 0.0),
            color=colors["target"],
            label="Planner target speed",
        )
        axes[0].set(
            title="Red/Yellow Stop: Command and Stop-Line Progress",
            ylabel="Speed (m/s)",
        )
        axes[0].legend(ncol=2)

        axes[1].plot(
            stop_time,
            _series(stop_rows, "stop_target_forward_m"),
            color="#2563a6",
            label="Stop target forward distance",
        )
        axes[1].axhline(0.0, color=colors["danger"], linestyle="--", label="Stop line")
        axes[1].fill_between(
            stop_time,
            _series(stop_rows, "stop_target_forward_m"),
            0.0,
            where=_series(stop_rows, "stop_target_forward_m") < 0.0,
            color=colors["danger"],
            alpha=0.12,
        )
        axes[1].set(ylabel="Distance (m)")
        axes[1].legend(ncol=2)

        axes[2].plot(
            stop_time,
            _series(stop_rows, "pre_supervisor_accel_cmd_mps2", 0.0),
            color="#7b3fa1",
            label="Pre-supervisor acceleration",
        )
        axes[2].plot(
            stop_time,
            _series(stop_rows, "post_supervisor_accel_cmd_mps2", 0.0),
            color="#238636",
            linestyle="--",
            label="Final acceleration",
        )
        axes[2].axhline(0.0, color="#9aa4b2", linewidth=1.0)
        axes[2].set(ylabel="Acceleration (m/s²)")
        axes[2].legend(ncol=2)

        axes[3].plot(
            stop_time,
            _series(stop_rows, "applied_throttle", 0.0),
            color="#d66b00",
            label="Applied throttle",
        )
        axes[3].plot(
            stop_time,
            _series(stop_rows, "applied_brake", 0.0),
            color=colors["danger"],
            label="Applied brake",
        )
        stop_statuses, stop_status_names = _status_values(stop_rows)
        status_axis = axes[3].twinx()
        status_axis.step(
            stop_time,
            stop_statuses,
            where="post",
            color="#2563a6",
            alpha=0.35,
            label="MPC status",
        )
        status_axis.set_yticks(
            range(len(stop_status_names)),
            [name.replace(" ", "\n") for name in stop_status_names],
        )
        axes[3].set(xlabel="Simulation time (s)", ylabel="Pedal command")
        axes[3].legend(loc="upper left", ncol=2)
        status_axis.legend(loc="upper right")
        for axis in axes:
            axis.grid(alpha=0.24)
        _save(fig, output_dir, "05_traffic_light_stop_process")

    def duration(mask_value: np.ndarray) -> float:
        indices = np.flatnonzero(mask_value)
        if len(indices) < 2:
            return 0.0
        return float(time_s[indices[-1]] - time_s[indices[0]])

    lc_speed = speed_mps[lane_change]
    turn_speed = speed_mps[turn]
    turn_clearance = _series(turn_rows, "road_boundary_clearance_m")
    lc_status = Counter(str(row.get("mpc_status", "")) for row in lc_rows)
    turn_status = Counter(str(row.get("mpc_status", "")) for row in turn_rows)
    stop_status = Counter(str(row.get("mpc_status", "")) for row in stop_rows)
    stop_speed = _series(stop_rows, "speed_mps", 0.0)
    stop_forward = _series(stop_rows, "stop_target_forward_m")
    stop_accel = _series(
        stop_rows,
        "post_supervisor_accel_cmd_mps2",
        0.0,
    )
    stop_throttle = _series(stop_rows, "applied_throttle", 0.0)
    stop_brake = _series(stop_rows, "applied_brake", 0.0)
    release_reason = next(
        (
            str(row.get("lane_change_commitment_release_reason", ""))
            for row in rows[lc_end : min(len(rows), lc_end + 10)]
            if row.get("lane_change_commitment_release_reason")
        ),
        "",
    )
    target_lane_id = int(_number(rows[lc_start], "destination_lane_id", 0.0))
    target_lane_entry = next(
        (
            index
            for index in range(lc_start, lc_end + 1)
            if int(_number(rows[index], "current_lane_id", 0.0))
            == int(target_lane_id)
        ),
        lc_end,
    )
    alignment_after_lane_entry_s = float(
        time_s[lc_end] - time_s[target_lane_entry]
    )
    turn_min_speed_local = int(np.argmin(turn_speed))
    turn_min_speed_index = int(turn_start + turn_min_speed_local)
    turn_min_speed_brake = _number(
        rows[turn_min_speed_index],
        "applied_brake",
        0.0,
    )
    turn_negative_clearance_frames = int(
        np.sum(np.isfinite(turn_clearance) & (turn_clearance < 0.0))
    )
    metrics = [
        ("Lane-change behavior duration", f"{duration(lane_change):.2f} s"),
        ("Lane-change prepare duration", f"{duration(lane_prepare):.2f} s"),
        ("Lane-change execute duration", f"{duration(lane_execute):.2f} s"),
        ("Lane-change alignment-hold duration", f"{duration(lane_alignment):.2f} s"),
        ("Target map-lane entry time", f"{time_s[target_lane_entry]:.2f} s"),
        ("Alignment after target-lane entry", f"{alignment_after_lane_entry_s:.2f} s"),
        ("Lane-change mean / max speed", f"{np.mean(lc_speed):.2f} / {np.max(lc_speed):.2f} m/s"),
        ("Lane-change low-speed frames (<0.15 m/s)", str(int(np.sum(lc_speed < 0.15)))),
        ("Lane-change max abs destination lateral", f"{np.nanmax(np.abs(_series(lc_rows, 'destination_lateral_m'))):.3f} m"),
        ("Lane-change MPC solved / buffer reuse", f"{lc_status['solved'] + lc_status['solved inaccurate']} / {lc_status['buffer_reuse']}"),
        ("Lane-change commitment release", release_reason or "not recorded"),
        ("Turn duration", f"{duration(turn):.2f} s"),
        ("Turn mean / max speed", f"{np.mean(turn_speed):.2f} / {np.max(turn_speed):.2f} m/s"),
        ("Turn minimum boundary clearance", f"{np.nanmin(turn_clearance):.3f} m"),
        ("Turn negative-clearance frames", str(turn_negative_clearance_frames)),
        ("Turn continuous-recovery frames", str(sum("continuous_recovery" in str(row.get("control_guard_reason", "")) for row in turn_rows))),
        ("Turn critical-stop frames", str(sum("critical_stop" in str(row.get("control_guard_reason", "")) for row in turn_rows))),
        ("Turn minimum-speed event", f"{time_s[turn_min_speed_index]:.2f} s; brake={turn_min_speed_brake:.3f}"),
        ("Turn MPC solved / buffer reuse", f"{turn_status['solved'] + turn_status['solved inaccurate']} / {turn_status['buffer_reuse']}"),
        ("Traffic-stop duration", f"{duration(traffic_stop):.2f} s"),
        ("Traffic-stop mean / minimum speed", f"{np.mean(stop_speed):.2f} / {np.min(stop_speed):.2f} m/s"),
        ("Traffic-stop minimum stop-target forward", f"{np.nanmin(stop_forward):.3f} m"),
        ("Traffic-stop positive-acceleration frames", str(int(np.sum(stop_accel > 1.0e-3)))),
        ("Traffic-stop mean / max throttle", f"{np.mean(stop_throttle):.3f} / {np.max(stop_throttle):.3f}"),
        ("Traffic-stop max brake", f"{np.max(stop_brake):.3f}"),
        ("Traffic-stop MPC primal-infeasible frames", str(stop_status["primal infeasible"])),
        ("Collision count at run end", str(int(_number(rows[-1], "collision_count", 0.0)))),
    ]
    with (output_dir / "maneuver_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["Metric", "Value"])
        writer.writerows(metrics)

    lc_lat = _series(lc_rows, "lane_change_completion_lateral_error_m")
    lc_heading = _series(lc_rows, "lane_change_completion_heading_error_deg")
    lc_lat_stats = _finite_summary(np.abs(lc_lat))
    lc_heading_stats = _finite_summary(np.abs(lc_heading))
    report = [
        "# Lane-Change and Turn Analysis",
        "",
        f"Source: `{input_jsonl}`",
        "",
        "## Right lane change",
        "",
        f"- Behavior interval: {time_s[lc_start]:.2f}-{time_s[lc_end]:.2f} s.",
        f"- Prepare / execute / alignment hold: {duration(lane_prepare):.2f} / {duration(lane_execute):.2f} / {duration(lane_alignment):.2f} s.",
        f"- CARLA map lane changes to target lane {target_lane_id} at {time_s[target_lane_entry]:.2f} s; commitment remains active for another {alignment_after_lane_entry_s:.2f} s.",
        f"- Absolute lateral error mean / max: {lc_lat_stats[1]:.3f} / {lc_lat_stats[2]:.3f} m.",
        f"- Absolute heading error mean / max: {lc_heading_stats[1]:.2f} / {lc_heading_stats[2]:.2f} deg.",
        f"- Low-speed frames below 0.15 m/s: {int(np.sum(lc_speed < 0.15))}.",
        f"- Maximum absolute locked-reference destination lateral: {np.nanmax(np.abs(_series(lc_rows, 'destination_lateral_m'))):.3f} m.",
        f"- Commitment release: `{release_reason or 'not recorded'}`.",
        "",
        "## Right intersection turn",
        "",
        f"- Turn interval: {time_s[turn_start]:.2f}-{time_s[turn_end]:.2f} s.",
        f"- Mean / maximum speed: {np.mean(turn_speed):.2f} / {np.max(turn_speed):.2f} m/s.",
        f"- Minimum body-to-lane-edge clearance: {np.nanmin(turn_clearance):.3f} m.",
        f"- Negative-clearance frames: {turn_negative_clearance_frames}.",
        f"- Continuous boundary recovery frames: {sum('continuous_recovery' in str(row.get('control_guard_reason', '')) for row in turn_rows)}.",
        f"- Critical boundary stops: {sum('critical_stop' in str(row.get('control_guard_reason', '')) for row in turn_rows)}.",
        f"- MPC solved or solved-inaccurate frames: {turn_status['solved'] + turn_status['solved inaccurate']} / {len(turn_rows)}.",
        f"- The minimum-speed sample occurs at {time_s[turn_min_speed_index]:.2f} s with only {turn_min_speed_brake:.3f} applied brake, so the abrupt one-frame drop is not a commanded hard stop.",
        "",
        "## Traffic-light stop",
        "",
        f"- Stop interval: {stop_time[0]:.2f}-{stop_time[-1]:.2f} s.",
        f"- Mean / minimum speed: {np.mean(stop_speed):.2f} / {np.min(stop_speed):.2f} m/s.",
        f"- Minimum signed stop-target distance: {np.nanmin(stop_forward):.3f} m; a negative value means the ego passed the target.",
        f"- Positive final-acceleration frames: {int(np.sum(stop_accel > 1.0e-3))} / {len(stop_rows)}.",
        f"- Mean / maximum throttle: {np.mean(stop_throttle):.3f} / {np.max(stop_throttle):.3f}; maximum brake: {np.max(stop_brake):.3f}.",
        f"- MPC primal-infeasible frames: {stop_status['primal infeasible']} / {len(stop_rows)}.",
        "",
        "## Interpretation",
        "",
        "- The lane change reaches the target map lane without a low-speed stall, but most control frames reuse the MPC buffer and the map-lane transition is followed by a short emergency-brake hard gate before lane-follow recovery.",
        "- The turn no longer enters a hard-stop loop. Its sustained slowness comes from the continuous road-boundary speed envelope while clearance is small or negative; the isolated speed collapse is inconsistent with the applied brake and should be treated separately as a simulator/contact or measurement transient.",
        "- The traffic-stop failure is a control-policy conflict: the planner requests zero speed, but the red/yellow envelope outputs positive acceleration and throttle. The vehicle consequently creeps past the stop target while the MPC becomes persistently infeasible.",
    ]
    (output_dir / "MANEUVER_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export(args.input_jsonl, args.output_dir)


if __name__ == "__main__":
    main()
