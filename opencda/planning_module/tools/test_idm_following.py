"""Standalone longitudinal closed-loop test for CP-X IDM following.

This intentionally excludes CARLA, map matching, behavior FSM, trajectory
geometry, and MPC steering so failures can be assigned to the longitudinal
following policy itself.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


_SPEED_PLANNER_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "speed_planner.py"
_SPEED_PLANNER_SPEC = importlib.util.spec_from_file_location(
    "standalone_idm_speed_planner", _SPEED_PLANNER_PATH
)
if _SPEED_PLANNER_SPEC is None or _SPEED_PLANNER_SPEC.loader is None:
    raise ImportError(f"cannot load speed planner from {_SPEED_PLANNER_PATH}")
_SPEED_PLANNER_MODULE = importlib.util.module_from_spec(_SPEED_PLANNER_SPEC)
sys.modules[_SPEED_PLANNER_SPEC.name] = _SPEED_PLANNER_MODULE
_SPEED_PLANNER_SPEC.loader.exec_module(_SPEED_PLANNER_MODULE)
build_speed_plan = _SPEED_PLANNER_MODULE.build_speed_plan


def _lead_speed(case: str, time_s: float) -> float:
    if case == "constant_lead":
        return 12.0
    if case == "lead_brakes":
        if time_s < 15.0:
            return 12.0
        if time_s < 18.0:
            return 12.0 - 2.0 * (time_s - 15.0)
        return 6.0
    raise ValueError(f"unknown case: {case}")


def simulate(case: str, duration_s: float, dt_s: float) -> tuple[list[dict], dict]:
    ego_speed_mps = 10.0
    ego_position_m = 0.0
    lead_position_m = 50.0
    requested_speed_mps = 12.0
    rows: list[dict] = []
    scenario = SimpleNamespace(speed_cap_mps=None, stop_goal_active=False, reason="")
    previous_idm_acceleration_mps2 = None
    steps = int(round(duration_s / dt_s)) + 1

    for step in range(steps):
        time_s = step * dt_s
        lead_speed_mps = _lead_speed(case, time_s)
        gap_m = max(0.0, lead_position_m - ego_position_m)
        plan = build_speed_plan(
            scenario_decision=scenario,
            behavior_decision="lane_follow",
            requested_speed_mps=requested_speed_mps,
            ego_speed_mps=ego_speed_mps,
            config={
                "following_idm_enabled": True,
                "following_standstill_gap_m": 5.0,
                "following_time_headway_s": 1.5,
                "following_gap_speed_gain_per_s": 0.20,
                "following_max_catchup_delta_mps": 2.0,
                "following_max_spacing_release_delta_mps": 3.0,
                "following_idm_target_horizon_s": 0.20,
                "following_idm_free_speed_headroom_mps": 10.0,
            },
            front_gap_m=gap_m,
            front_obstacle_speed_mps=lead_speed_mps,
            previous_idm_acceleration_mps2=previous_idm_acceleration_mps2,
        )
        previous_idm_acceleration_mps2 = plan.idm_acceleration_mps2
        desired_gap_m = float(plan.desired_follow_gap_m or 0.0)
        rows.append(
            {
                "time_s": time_s,
                "ego_speed_mps": ego_speed_mps,
                "lead_speed_mps": lead_speed_mps,
                "target_speed_mps": float(plan.target_speed_mps),
                "gap_m": gap_m,
                "desired_gap_m": desired_gap_m,
                "gap_error_m": gap_m - desired_gap_m,
                "idm_acceleration_mps2": float(plan.idm_acceleration_mps2 or 0.0),
                "limiting_owner": str(plan.limiting_owner),
                "stop_goal_active": bool(plan.stop_goal_active),
            }
        )

        # Idealized longitudinal actuator used only to close the policy loop.
        # It deliberately does not reuse IDM acceleration: this verifies that
        # the speed target exported to MPC is sufficient to produce following.
        commanded_accel_mps2 = max(
            -4.0,
            min(2.5, (float(plan.target_speed_mps) - ego_speed_mps) / 0.6),
        )
        ego_speed_mps = max(0.0, ego_speed_mps + commanded_accel_mps2 * dt_s)
        ego_position_m += ego_speed_mps * dt_s
        lead_position_m += lead_speed_mps * dt_s

    tail = rows[-max(1, int(round(5.0 / dt_s))):]
    maximum_ego_speed_mps = max(float(row["ego_speed_mps"]) for row in rows)
    minimum_gap_m = min(float(row["gap_m"]) for row in rows)
    mean_tail_gap_error_m = sum(float(row["gap_error_m"]) for row in tail) / len(tail)
    mean_tail_speed_error_mps = sum(
        float(row["ego_speed_mps"]) - float(row["lead_speed_mps"]) for row in tail
    ) / len(tail)
    target_rates_mps2 = [
        (float(rows[index]["target_speed_mps"]) - float(rows[index - 1]["target_speed_mps"]))
        / float(dt_s)
        for index in range(1, len(rows))
    ]
    target_rate_sign_flips = sum(
        1
        for index in range(1, len(target_rates_mps2))
        if target_rates_mps2[index] * target_rates_mps2[index - 1] < 0.0
    )
    maximum_abs_target_rate_mps2 = max(abs(value) for value in target_rates_mps2)
    passed = bool(
        minimum_gap_m > 1.0
        and abs(mean_tail_speed_error_mps) < 0.35
        and abs(mean_tail_gap_error_m) < 2.5
        and maximum_abs_target_rate_mps2 <= 4.0
        and target_rate_sign_flips <= 10
    )
    if case == "constant_lead":
        passed = bool(passed and maximum_ego_speed_mps > requested_speed_mps + 0.1)
    return rows, {
        "case": case,
        "passed": passed,
        "maximum_ego_speed_mps": maximum_ego_speed_mps,
        "minimum_gap_m": minimum_gap_m,
        "final_gap_m": float(rows[-1]["gap_m"]),
        "final_desired_gap_m": float(rows[-1]["desired_gap_m"]),
        "mean_tail_gap_error_m": mean_tail_gap_error_m,
        "mean_tail_speed_error_mps": mean_tail_speed_error_mps,
        "maximum_abs_target_rate_mps2": maximum_abs_target_rate_mps2,
        "target_rate_sign_flips": target_rate_sign_flips,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: list[dict], title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    time_s = [float(row["time_s"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(time_s, [row["ego_speed_mps"] for row in rows], label="Ego speed")
    axes[0].plot(time_s, [row["lead_speed_mps"] for row in rows], label="Lead speed")
    axes[0].plot(time_s, [row["target_speed_mps"] for row in rows], "--", label="IDM target")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time_s, [row["gap_m"] for row in rows], label="Actual gap")
    axes[1].plot(time_s, [row["desired_gap_m"] for row in rows], "--", label="Desired gap")
    axes[1].set_ylabel("Gap (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(time_s, [row["idm_acceleration_mps2"] for row in rows])
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("IDM accel (m/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="opencda/planning_module/opencda_bridge/idm_standalone")
    parser.add_argument("--duration-s", type=float, default=45.0)
    parser.add_argument("--dt-s", type=float, default=0.05)
    parser.add_argument("--fail-on-violations", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for case in ("constant_lead", "lead_brakes"):
        rows, summary = simulate(case, args.duration_s, args.dt_s)
        _write_csv(output_dir / f"{case}.csv", rows)
        summary["plot_written"] = _write_plot(
            output_dir / f"{case}.png", rows, f"Standalone IDM: {case}"
        )
        summaries.append(summary)
    result = {
        "suite_passed": all(item["passed"] for item in summaries),
        "cases": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if args.fail_on_violations and not result["suite_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
