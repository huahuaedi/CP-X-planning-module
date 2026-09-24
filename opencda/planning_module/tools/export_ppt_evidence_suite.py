#!/usr/bin/env python3
"""Export evidence-backed English figures for the CP-X presentation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEBUG = ROOT / "opencda" / "planning_module" / "opencda_bridge"


def _load(name: str) -> List[dict]:
    path = DEBUG / name / "opencda_planner_debug.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except (TypeError, ValueError):
                continue
    if not rows:
        raise RuntimeError("no planner records: %s" % path)
    return rows


def _number(row: Mapping[str, object], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _time(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    values = np.asarray([_number(row, "sim_time_s") for row in rows], dtype=float)
    return values - values[0]


def _values(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    return np.asarray([_number(row, key) for row in rows], dtype=float)


def _rolling_mean(values: np.ndarray, samples: int = 11) -> np.ndarray:
    if values.size == 0 or samples <= 1:
        return values
    finite = np.where(np.isfinite(values), values, 0.0)
    weight = np.where(np.isfinite(values), 1.0, 0.0)
    kernel = np.ones(int(samples), dtype=float)
    numerator = np.convolve(finite, kernel, mode="same")
    denominator = np.convolve(weight, kernel, mode="same")
    return numerator / np.maximum(denominator, 1.0)


def _true_spans(mask: np.ndarray) -> Iterable[tuple]:
    """Yield half-open index ranges for contiguous true samples."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return
    start = previous = int(indices[0])
    for current_value in indices[1:]:
        current = int(current_value)
        if current != previous + 1:
            yield start, previous + 1
            start = current
        previous = current
    yield start, previous + 1


def _complete(rows: Sequence[Mapping[str, object]]) -> bool:
    return any(bool(row.get("mission_complete")) for row in rows)


def _safety(rows: Sequence[Mapping[str, object]]) -> dict:
    return {
        "complete": _complete(rows),
        "collision_count": max(int(row.get("collision_count") or 0) for row in rows),
        "boundary_breach_count": max(
            int(row.get("road_boundary_breach_count") or 0) for row in rows
        ),
        "fallback_ticks": sum(bool(row.get("fallback_active")) for row in rows),
        "infeasible_ticks": sum(
            "infeasible" in str(row.get("mpc_status", "")).lower()
            for row in rows
        ),
    }


def _save(fig, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / (stem + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output / (stem + ".svg"), bbox_inches="tight")
    plt.close(fig)


def _single_vehicle_figure(output: Path, rows: Sequence[Mapping[str, object]]) -> None:
    t = _time(rows)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)
    axes[0].plot(t, _values(rows, "speed_mps"), color="#1769aa", linewidth=2)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Single-Vehicle Closed-Loop Baseline")
    lateral = np.abs(_values(rows, "executed_reference_lateral_error_m"))
    heading = np.abs(_values(rows, "executed_reference_heading_error_deg"))
    axes[1].plot(t, lateral, color="#2a9d8f", label="Lateral error (m)")
    axes[1].plot(t, heading, color="#d1495b", label="Heading error (deg)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Absolute tracking error")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    _save(fig, output, "01_single_vehicle_baseline")


def _merge_figure(output: Path, data: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    colors = {"cav1": "#1769aa", "cav2": "#d1495b"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for arm, style in (("off", "--"), ("on", "-")):
        for cav in ("cav1", "cav2"):
            rows = data[arm + "_" + cav]
            label = "%s %s%s" % (
                arm.upper(), cav.upper(), " (complete)" if _complete(rows) else " (incomplete)"
            )
            axes[0].plot(
                _values(rows, "x_m"), _values(rows, "y_m"), style,
                color=colors[cav], linewidth=2, label=label,
            )
            progress = _values(rows, "x_m") - _number(rows[0], "x_m", 0.0)
            axes[1].plot(
                _time(rows), progress, style, color=colors[cav],
                linewidth=2, label="%s %s" % (arm.upper(), cav.upper()),
            )
    axes[0].set_title("Cooperative Intent Arbitration Enables Lane Exchange")
    axes[0].set_xlabel("Road x (m)")
    axes[0].set_ylabel("Road y (m)")
    axes[1].set_title("Longitudinal Mission Progress")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Progress from spawn (m)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    _save(fig, output, "02_cooperative_merge")


def _lead_braking_figure(
    output: Path,
    ego: Sequence[Mapping[str, object]],
    lead: Sequence[Mapping[str, object]],
) -> None:
    ego_t = _time(ego)
    lead_t = _time(lead)
    lead_brake = np.asarray([
        str((row.get("cav_conflict_tags") or {}).values()).find("LEAD_BRAKE") >= 0
        for row in ego
    ], dtype=bool)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    axes[0].plot(lead_t, _values(lead, "speed_mps"), color="#d1495b", label="Lead CAV")
    axes[0].plot(ego_t, _values(ego, "speed_mps"), color="#1769aa", label="Ego CAV")
    axes[0].set_title("MTR-Informed Response to Lead-Vehicle Braking")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()
    axes[1].plot(
        ego_t,
        _rolling_mean(_values(ego, "measured_accel_mps2")),
        color="#2a9d8f",
        label="Ego acceleration",
    )
    if np.any(lead_brake):
        label_pending = True
        for start, stop in _true_spans(lead_brake):
            right = min(stop, ego_t.size - 1)
            for axis in axes:
                axis.axvspan(
                    ego_t[start], ego_t[right], color="#f4a261", alpha=0.16,
                    label="LEAD_BRAKE classified" if label_pending and axis is axes[1] else None,
                )
            label_pending = False
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Acceleration (m/s²)")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    _save(fig, output, "03_mtr_lead_braking")


def _finite_replan_values(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    values = [
        _number(row, key)
        for row in rows
        if bool(row.get("mpc_replan_executed")) and math.isfinite(_number(row, key))
    ]
    return np.asarray(values, dtype=float)


def _four_cav_figure(output: Path, runs: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    colors = ["#1769aa", "#d1495b", "#2a9d8f", "#e9c46a"]
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    for index, (name, rows) in enumerate(runs.items()):
        t = _time(rows)
        axes[0, 0].plot(t, _values(rows, "speed_mps"), color=colors[index], label=name.upper())
        axes[0, 1].plot(
            t,
            _values(rows, "cav_retained_prediction_mode_count"),
            color=colors[index], label=name.upper(),
        )
    axes[0, 0].set_title("Four-CAV Closed-Loop Speed Profiles")
    axes[0, 0].set_xlabel("Simulation time (s)")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 1].set_title("Retained Prediction Modes")
    axes[0, 1].set_xlabel("Simulation time (s)")
    axes[0, 1].set_ylabel("Mode count")

    timing_keys = [
        ("mpc_timing_qp_build_ms", "QP construction"),
        ("mpc_timing_qp_solve_wall_ms", "OSQP solve"),
        ("mpc_timing_reference_ms", "Reference"),
        ("mpc_timing_cost_evaluation_ms", "Cost evaluation"),
        ("mpc_timing_finalize_ms", "Finalization"),
        ("mpc_timing_prepare_ms", "Preparation"),
    ]
    bottoms = np.zeros(len(runs), dtype=float)
    x = np.arange(len(runs))
    palette = ["#457b9d", "#e76f51", "#2a9d8f", "#f4a261", "#8d99ae", "#6d597a"]
    for (key, label), color in zip(timing_keys, palette):
        means = np.asarray([
            np.mean(_finite_replan_values(rows, key)) for rows in runs.values()
        ])
        axes[1, 0].bar(x, means, bottom=bottoms, label=label, color=color)
        bottoms += means
    axes[1, 0].axhline(200.0, color="#333333", linestyle="--", label="5 Hz budget (200 ms)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([name.upper() for name in runs])
    axes[1, 0].set_ylabel("Mean time per replan (ms)")
    axes[1, 0].set_title("MPC Computation Breakdown")
    axes[1, 0].legend(fontsize=7, ncol=2)

    mean_total = []
    p95_total = []
    for rows in runs.values():
        values = _finite_replan_values(rows, "mpc_timing_total_ms")
        mean_total.append(float(np.mean(values)))
        p95_total.append(float(np.percentile(values, 95)))
    width = 0.36
    axes[1, 1].bar(x - width / 2, mean_total, width, label="Mean", color="#1769aa")
    axes[1, 1].bar(x + width / 2, p95_total, width, label="P95", color="#d1495b")
    axes[1, 1].axhline(200.0, color="#333333", linestyle="--", label="5 Hz budget (200 ms)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([name.upper() for name in runs])
    axes[1, 1].set_ylabel("MPC replan time (ms)")
    axes[1, 1].set_title("Real-Time Replanning Budget")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
    axes[0, 0].legend(ncol=2)
    axes[0, 1].legend(ncol=2)
    _save(fig, output, "04_four_cav_mtr_scalability")


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    single = _load("debug_right_turn_highspeed")
    merge = {
        "off_cav1": _load("debug_two_cav_merge_off_cav1"),
        "off_cav2": _load("debug_two_cav_merge_off_cav2"),
        "on_cav1": _load("debug_two_cav_merge_cav1"),
        "on_cav2": _load("debug_two_cav_merge_cav2"),
    }
    lead_ego = _load("debug_lead_braking_mtr_ego")
    lead = _load("debug_lead_braking_mtr_lead")
    four = {
        "cav1": _load("debug_four_cav_merge_mtr_cav1"),
        "cav2": _load("debug_four_cav_merge_mtr_cav2"),
        "cav3": _load("debug_four_cav_merge_mtr_cav3"),
        "cav4": _load("debug_four_cav_merge_mtr_cav4"),
    }

    _single_vehicle_figure(args.output, single)
    _merge_figure(args.output, merge)
    _lead_braking_figure(args.output, lead_ego, lead)
    _four_cav_figure(args.output, four)

    summary = {
        "single_vehicle": _safety(single),
        "cooperative_merge": {
            name: _safety(rows) for name, rows in merge.items()
        },
        "mtr_lead_braking": {
            "ego": _safety(lead_ego),
            "lead": _safety(lead),
            "lead_brake_classification_ticks": sum(
                "LEAD_BRAKE" in (row.get("cav_conflict_tags") or {}).values()
                for row in lead_ego
            ),
        },
        "four_cav_mtr": {
            name: {
                **_safety(rows),
                "mean_mpc_replan_ms": round(float(np.mean(
                    _finite_replan_values(rows, "mpc_timing_total_ms")
                )), 3),
                "p95_mpc_replan_ms": round(_percentile(
                    _finite_replan_values(rows, "mpc_timing_total_ms"), 95
                ), 3),
                "max_retained_mode_count": int(max(
                    _number(row, "cav_retained_prediction_mode_count", 0.0)
                    for row in rows
                )),
            }
            for name, rows in four.items()
        },
    }
    (args.output / "evidence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
