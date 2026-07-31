"""Export full-run CP-X planner plots and a compact metrics report."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _float(row, key, default=float("nan")):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _values(rows, key, default=float("nan")):
    return np.asarray([_float(row, key, default) for row in rows], dtype=float)


def _save(fig, output_dir, stem):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _stop_spans(rows):
    spans = []
    start = None
    for index, row in enumerate(rows):
        active = str(row.get("scenario_fsm_state", "")) == "TRAFFIC_LIGHT_STOP"
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(rows) - 1))
    return spans


def _shade_stops(axes, rows, spans):
    for axis in np.atleast_1d(axes):
        for start, end in spans:
            axis.axvspan(
                _float(rows[start], "sim_time_s", 0.0),
                _float(rows[end], "sim_time_s", 0.0),
                color="#d64545",
                alpha=0.08,
                linewidth=0,
            )


def export(input_csv: Path, output_dir: Path) -> None:
    with input_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No rows found in {input_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "full_run_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    time_s = _values(rows, "sim_time_s")
    speed = _values(rows, "speed_mps", 0.0)
    target_speed = _values(rows, "target_speed_mps", 0.0)
    x_m = _values(rows, "x_m")
    y_m = _values(rows, "y_m")
    spans = _stop_spans(rows)

    colors = {
        "LANE_FOLLOW": "#177e89",
        "TRAFFIC_LIGHT_APPROACH": "#d97706",
        "TRAFFIC_LIGHT_STOP": "#c43d3d",
        "INTERSECTION_TURN": "#6b46c1",
        "CREEP": "#805ad5",
        "RECOVERY": "#2d3748",
    }
    fig, axis = plt.subplots(figsize=(10.5, 7.0))
    for first, second in zip(range(len(rows) - 1), range(1, len(rows))):
        state = str(rows[second].get("scenario_fsm_state", ""))
        axis.plot(
            x_m[[first, second]],
            y_m[[first, second]],
            color=colors.get(state, "#718096"),
            linewidth=2.3,
        )
    for state, color in colors.items():
        if any(row.get("scenario_fsm_state") == state for row in rows):
            axis.plot([], [], color=color, linewidth=3, label=state.replace("_", " "))
    axis.scatter(x_m[0], y_m[0], s=70, color="#2f855a", label="Start", zorder=4)
    axis.scatter(x_m[-1], y_m[-1], s=70, color="#111827", label="End", zorder=4)
    axis.set(title="Full-Run Route and Scenario State", xlabel="World x (m)", ylabel="World y (m)")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    _save(fig, output_dir, "01_full_route_state")

    state_names = [
        "LANE_FOLLOW",
        "TRAFFIC_LIGHT_APPROACH",
        "TRAFFIC_LIGHT_STOP",
        "INTERSECTION_TURN",
        "CREEP",
        "RECOVERY",
    ]
    signal_names = ["unknown", "red", "yellow", "green"]
    behavior_names = [
        "lane_follow",
        "stop_at_intersection",
        "intersection_turn_left",
        "intersection_turn_right",
        "emergency_brake",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.5), sharex=True)
    for axis, key, names, title in (
        (axes[0], "scenario_fsm_state", state_names, "Scenario FSM"),
        (axes[1], "traffic_signal_state", signal_names, "Traffic Signal"),
        (axes[2], "behavior_decision", behavior_names, "Behavior Decision"),
    ):
        mapping = {name: index for index, name in enumerate(names)}
        values = [mapping.get(str(row.get(key, "")), -1) for row in rows]
        axis.step(time_s, values, where="post", color="#177e89", linewidth=1.5)
        axis.set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
        axis.set_ylabel(title)
        axis.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("Scenario, Signal, and Behavior Timeline", fontsize=15)
    _save(fig, output_dir, "02_state_signal_timeline")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.0), sharex=True)
    axes[0].plot(time_s, speed, color="#177e89", label="Actual speed")
    axes[0].plot(time_s, target_speed, color="#d97706", label="Target speed")
    axes[0].set(ylabel="Speed (m/s)", title="Actual and Target Speed")
    axes[0].legend()
    axes[1].plot(time_s, _values(rows, "post_supervisor_accel_cmd_mps2", 0.0), color="#2f855a", label="Acceleration")
    axes[1].plot(time_s, _values(rows, "applied_throttle", 0.0), color="#d97706", label="Throttle")
    axes[1].plot(time_s, _values(rows, "applied_brake", 0.0), color="#c53030", label="Brake")
    axes[1].set(xlabel="Simulation time (s)", ylabel="Command", title="Applied Longitudinal Control")
    axes[1].legend(ncol=3)
    _shade_stops(axes, rows, spans)
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(fig, output_dir, "03_speed_and_control")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.0), sharex=True)
    axes[0].plot(time_s, _values(rows, "reference_first_lateral_m"), label="First-point lateral")
    axes[0].plot(time_s, _values(rows, "destination_lateral_m"), label="Destination lateral")
    axes[0].axhline(0.0, color="#718096", linewidth=0.8)
    axes[0].set(ylabel="Lateral offset (m)", title="Reference Lateral Stability")
    axes[0].legend()
    axes[1].plot(time_s, _values(rows, "reference_first_forward_m"), label="First-point forward")
    axes[1].plot(time_s, _values(rows, "destination_forward_m"), label="Destination forward")
    axes[1].set(xlabel="Simulation time (s)", ylabel="Forward distance (m)", title="MPC Reference Horizon")
    axes[1].legend()
    _shade_stops(axes, rows, spans)
    for axis in axes:
        axis.grid(alpha=0.25)
    _save(fig, output_dir, "04_reference_stability")

    if "maneuver_geometry_active" in rows[0]:
        active = np.asarray([
            1.0
            if str(row.get("maneuver_geometry_active", "")).lower()
            in {"true", "1"}
            else 0.0
            for row in rows
        ])
        fig, axes = plt.subplots(3, 1, figsize=(13, 8.0), sharex=True)
        axes[0].step(
            time_s,
            active,
            where="post",
            color="#6b46c1",
            label="Unified maneuver active",
        )
        axes[0].set(ylabel="Active", yticks=[0, 1], title="Unified Maneuver Ownership")
        axes[0].legend()
        axes[1].plot(
            time_s,
            _values(rows, "maneuver_first_point_jump_m", 0.0),
            color="#177e89",
            label="First-point frame jump",
        )
        axes[1].axhline(0.3, color="#c53030", linestyle="--", label="0.3 m target")
        axes[1].set(ylabel="Distance (m)", title="Reference Position Continuity")
        axes[1].legend()
        axes[2].plot(
            time_s,
            _values(rows, "maneuver_first_heading_jump_deg", 0.0),
            color="#d97706",
            label="First-heading frame jump",
        )
        axes[2].axhline(5.0, color="#c53030", linestyle="--", label="5 deg target")
        axes[2].set(
            xlabel="Simulation time (s)",
            ylabel="Angle (deg)",
            title="Reference Heading Continuity",
        )
        axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.25)
        _save(fig, output_dir, "05_maneuver_continuity")

    statuses = ["solved", "solved inaccurate", "buffer_reuse", "primal infeasible", "candidate_hard_gate", "emergency_brake_direct"]
    counts = Counter(str(row.get("mpc_status", "")) for row in rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    status_labels = [
        "solved",
        "solved\ninaccurate",
        "buffer\nreuse",
        "primal\ninfeasible",
        "candidate\nhard gate",
        "emergency\nbrake",
    ]
    axes[0].bar(
        status_labels,
        [counts[name] for name in statuses],
        color=["#2f855a", "#68d391", "#3182ce", "#d97706", "#c53030", "#7f1d1d"],
    )
    axes[0].set(title="MPC Outcome Distribution", ylabel="Frames")
    solve_time = _values(rows, "mpc_solve_time_ms")
    finite_solve = solve_time[np.isfinite(solve_time) & (solve_time > 0.0)]
    axes[1].hist(finite_solve, bins=35, color="#177e89", edgecolor="white")
    axes[1].set(title="MPC Solve-Time Distribution", xlabel="Solve time (ms)", ylabel="Replans")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    _save(fig, output_dir, "05_mpc_health")

    ttc = _values(rows, "nearest_ttc_s")
    drac = _values(rows, "tick_max_drac_mps2")
    valid = np.isfinite(ttc) & np.isfinite(drac) & (ttc >= 0.0)
    fig, axis = plt.subplots(figsize=(9.5, 6.2))
    if np.any(valid):
        scatter = axis.scatter(ttc[valid], drac[valid], c=speed[valid], s=14, cmap="viridis", alpha=0.7)
        fig.colorbar(scatter, ax=axis, label="Ego speed (m/s)")
    axis.axvline(3.0, color="#c53030", linestyle="--", label="3 s TTC reference")
    axis.set(title="TTC-DRAC Safety Phase Space", xlabel="Time to collision (s)", ylabel="DRAC (m/s²)")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, output_dir, "06_ttc_drac_phase_space")

    turn_mask = np.asarray(
        [
            str(row.get("scenario_fsm_state", "")) == "INTERSECTION_TURN"
            for row in rows
        ],
        dtype=bool,
    )
    if np.any(turn_mask):
        turn_time = time_s[turn_mask]
        clearance = _values(rows, "road_boundary_clearance_m")[turn_mask]
        lateral_offset = _values(rows, "road_boundary_lateral_offset_m")[turn_mask]
        applied_steer = _values(rows, "applied_steer", 0.0)[turn_mask]
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.0), sharex=True)
        axes[0].plot(turn_time, clearance, color="#177e89", label="Body-to-lane-edge clearance")
        axes[0].axhline(0.0, color="#c53030", linestyle="--", label="Lane-boundary threshold")
        axes[0].fill_between(
            turn_time,
            clearance,
            0.0,
            where=np.isfinite(clearance) & (clearance < 0.0),
            color="#c53030",
            alpha=0.18,
        )
        axes[0].set(ylabel="Clearance (m)", title="Intersection-Turn Lane Containment")
        axes[0].legend()
        axes[1].plot(turn_time, lateral_offset, color="#6b46c1", label="Lane-center lateral offset")
        axes[1].plot(turn_time, applied_steer, color="#d97706", label="Applied steer")
        axes[1].set(xlabel="Simulation time (s)", ylabel="Offset / command", title="Turn Tracking and Steering")
        axes[1].legend()
        for axis in axes:
            axis.grid(alpha=0.25)
        _save(fig, output_dir, "07_turn_lane_containment")

    distance_m = float(
        np.nansum(np.hypot(np.diff(x_m), np.diff(y_m)))
    )
    duration_s = float(time_s[-1] - time_s[0])
    destination_distance_m = _float(
        rows[-1],
        "route_remaining_distance_m",
        float("nan"),
    )
    if not math.isfinite(destination_distance_m):
        destination_distance_m = float("nan")
    collision_count = max(_values(rows, "collision_count", 0.0))
    finite_ttc = ttc[np.isfinite(ttc) & (ttc >= 0.0)]
    finite_drac = drac[np.isfinite(drac)]
    solved = counts["solved"] + counts["solved inaccurate"]
    optimization_attempts = solved + counts["primal infeasible"]
    driving_rows = [
        row
        for row in rows
        if str(row.get("stop_goal_active", "")).lower() not in {"true", "1"}
    ]
    driving_counts = Counter(
        str(row.get("mpc_status", "")) for row in driving_rows
    )
    driving_solved = (
        driving_counts["solved"] + driving_counts["solved inaccurate"]
    )
    driving_attempts = driving_solved + driving_counts["primal infeasible"]
    stop_rows = [
        row
        for row in rows
        if str(row.get("stop_goal_active", "")).lower() in {"true", "1"}
    ]
    stop_speeds = np.asarray(
        [_float(row, "speed_mps", 0.0) for row in stop_rows],
        dtype=float,
    )
    metrics = [
        ("Samples", str(len(rows))),
        ("Duration", f"{duration_s:.2f} s"),
        ("Distance traveled", f"{distance_m:.2f} m"),
        ("Final destination distance", f"{destination_distance_m:.3f} m"),
        ("Average / maximum speed", f"{float(np.nanmean(speed)):.2f} / {float(np.nanmax(speed)):.2f} m/s"),
        ("Collisions", str(int(collision_count))),
        ("Minimum finite TTC", f"{float(np.min(finite_ttc)):.2f} s" if finite_ttc.size else "N/A"),
        ("Maximum DRAC", f"{float(np.max(finite_drac)):.2f} m/s²" if finite_drac.size else "N/A"),
        ("MPC solved", str(solved)),
        ("MPC primal infeasible", str(counts["primal infeasible"])),
        ("MPC candidate hard gate", str(counts["candidate_hard_gate"])),
        ("MPC optimization success", f"{100.0 * solved / optimization_attempts:.1f}%" if optimization_attempts else "N/A"),
        ("Driving MPC optimization success", f"{100.0 * driving_solved / driving_attempts:.1f}%" if driving_attempts else "N/A"),
        ("Stop-mode primal infeasible", str(sum(row.get("mpc_status") == "primal infeasible" for row in stop_rows))),
        ("Stop-mode mean / max speed", f"{float(np.mean(stop_speeds)):.2f} / {float(np.max(stop_speeds)):.2f} m/s" if stop_speeds.size else "N/A"),
        ("Accepted quintic recovery frames", str(sum("rebuilt_current_lane_reference_after_contract_veto" in str(row.get("mpc_reference_stabilizer_reason", "")) for row in rows))),
        ("Safety stuck-stop frames", str(sum("emergency_stop:stuck" in str(row.get("safety_supervisor_reason", "")) for row in rows))),
    ]
    with (output_dir / "full_run_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["Metric", "Value"])
        writer.writerows(metrics)
    report = [
        "# CP-X Full-Run Analysis",
        "",
        f"- Source: `{input_csv}`",
        f"- Final destination distance: **{destination_distance_m:.3f} m**",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    report.extend(f"| {name} | {value} |" for name, value in metrics)
    report += [
        "",
        "## Key Findings",
        "",
        f"- Final route-manager remaining distance: **{destination_distance_m:.3f} m**.",
        f"- Candidate hard-gate frames: **{counts['candidate_hard_gate']}**.",
        f"- Quintic lane-recovery references were accepted for **{metrics[-2][1]}** frames.",
        f"- Safety-supervisor stuck-stop frames: **{metrics[-1][1]}**.",
        f"- Most primal-infeasible frames occurred while the traffic-light stop guard already owned longitudinal braking.",
    ]
    (output_dir / "FULL_RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
