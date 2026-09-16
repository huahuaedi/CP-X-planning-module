"""Visualize CP-B scripted-vehicle transform height around its bend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


def export(debug_jsonl: Path, scene_yaml: Path, output_dir: Path) -> dict:
    rows = [json.loads(line) for line in debug_jsonl.open() if line.strip()]
    if not rows:
        raise ValueError("No planner diagnostics")
    actor_path = np.asarray(OmegaConf.load(scene_yaml).scenario.scripted_actors[0].path,
                            dtype=float)
    target_id = next(
        str(actor_id) for row in rows
        for actor_id, tag in (row.get("cav_conflict_tags") or {}).items()
        if tag in {"CROSSING", "ONCOMING"}
    )

    target_samples = []
    observer_samples = []
    target_xy = []
    target_heading = []
    turn_times = []
    start_time = float(rows[0]["sim_time_s"])
    for row in rows:
        t_s = float(row["sim_time_s"]) - start_time
        provenance = row.get("cp_actor_provenance") or "[]"
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        for actor in provenance:
            sample = (t_s, float(actor.get("actor_z_m", float("nan"))))
            if str(actor.get("actor_id", "")).endswith(":" + target_id):
                target_samples.append(sample)
            else:
                observer_samples.append(sample)
        state = (row.get("cav_conflict_agent_states") or {}).get(target_id)
        if state:
            x_m, y_m = float(state["x"]), float(state["y"])
            target_xy.append((x_m, y_m))
            target_heading.append((t_s, np.degrees(float(state["psi"]))))
            # The two connector segments lie between the vertical approach
            # and the outgoing horizontal road in the scenario's fixed path.
            if y_m >= -35.0 and x_m <= 21.0:
                turn_times.append(t_s)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    xy = np.asarray(target_xy)
    axes[0].plot(actor_path[:, 0], actor_path[:, 1], color="#64748b",
                 lw=1.8, ls="--", label="Configured path")
    axes[0].plot(xy[:, 0], xy[:, 1], color="#177e89", lw=2.1,
                 label="Observed scripted vehicle")
    axes[0].plot(actor_path[1:5, 0], actor_path[1:5, 1],
                 color="#c2410c", lw=2.5, label="Connector turn")
    axes[0].scatter([actor_path[0, 0]], [actor_path[0, 1]],
                    color="#177e89", s=45, label="Start")
    axes[0].scatter([6.3], [-80.0], color="#d97706", s=45,
                    label="Stationary observer CAV")
    axes[0].set(title="Vehicle route through the Town06 connector (axes not to scale)",
                xlabel="World x (m)", ylabel="World y (m)")
    axes[0].set_xlim(0, 45)
    axes[0].set_ylim(-85, -15)
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    for label, samples, color in (
        ("Scripted oncoming vehicle", target_samples, "#177e89"),
        ("Stationary observer CAV", observer_samples, "#d97706"),
    ):
        data = np.asarray(samples)
        if data.size:
            axes[1].plot(data[:, 0], data[:, 1], color=color,
                         lw=1.5, label=label)
    axes[1].axhline(0.0, color="#475569", lw=1.0, ls=":",
                    label="Town06 road elevation (approximately 0 m)")
    if turn_times:
        axes[1].axvspan(min(turn_times), max(turn_times), color="#c2410c",
                        alpha=0.12, label="Connector turn interval")
    axes[1].set(title="CARLA actor transform height, not tire clearance",
                xlabel="Time since run start (s)", ylabel="Actor transform Z (m)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)

    heading = np.asarray(target_heading)
    axes[2].plot(heading[:, 0], heading[:, 1], color="#6b46c1", lw=1.4)
    if turn_times:
        axes[2].axvspan(min(turn_times), max(turn_times), color="#c2410c",
                        alpha=0.12)
    axes[2].set(title="Tracked XY motion direction changes at path corners",
                xlabel="Time since run start (s)", ylabel="Motion direction (deg)",
                xlim=(25, 36))
    axes[2].grid(alpha=0.2)
    fig.text(0.02, 0.005,
             "The scripted actor uses set_transform with physics disabled; "
             "motion direction comes from tracked XY changes, not rendered yaw.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"cp_b_vehicle_height.{suffix}", dpi=180)
    plt.close(fig)
    summary = {
        "target_id": target_id,
        "target_z_min_m": min(z for _, z in target_samples),
        "target_z_max_m": max(z for _, z in target_samples),
        "target_z_samples": len(target_samples),
        "observer_z_min_m": min(z for _, z in observer_samples),
        "observer_z_max_m": max(z for _, z in observer_samples),
        "turn_time_range_s": [min(turn_times), max(turn_times)] if turn_times else None,
    }
    (output_dir / "cp_b_height_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-jsonl", type=Path, required=True)
    parser.add_argument("--scene-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.debug_jsonl, args.scene_yaml, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
