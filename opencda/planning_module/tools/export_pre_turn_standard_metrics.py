"""Export documented metrics using only samples recorded before the first turn."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from export_pre_turn_report import (
    _chart_panel,
    _episode_spans,
    _float,
    _percentile,
    _svg_text,
    _write_csv,
    _write_kpi_svg,
)


def _write_standard_overview(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    stop_spans: Sequence[Tuple[int, int]],
) -> None:
    width, height = 1600, 1040
    time_min = _float(rows[0], "sim_time_s")
    time_max = _float(rows[-1], "sim_time_s")
    bands = [
        (
            _float(rows[start], "sim_time_s"),
            _float(rows[end], "sim_time_s"),
            "#d64545",
            0.10,
        )
        for start, end in stop_spans
    ]
    replan_rows = [
        dict(row)
        for row in rows
        if str(row.get("mpc_replan_executed", "")).strip().lower() in {"true", "1"}
    ]
    for row in replan_rows:
        status = str(row.get("mpc_status", "")).strip().lower()
        row["_solver_health"] = (
            1.0
            if status == "solved"
            else 0.75
            if status == "solved inaccurate"
            else 0.0
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(80, 56, "CP-X Planner: Pre-Turn Standard Metrics", size=30, weight=700),
        _svg_text(
            80,
            86,
            "Only samples recorded before the first intersection-turn command are included.",
            size=17,
            color="#59636e",
        ),
    ]
    elements += _chart_panel(
        rows=rows,
        x0=110,
        y0=135,
        width=1400,
        height=230,
        title="Speed",
        series=[
            ("speed_mps", "Ego speed (m/s)", "#177e89"),
            ("target_speed_mps", "Target speed (m/s)", "#d95f02"),
        ],
        y_min=0.0,
        y_max=max(4.0, max(_float(row, "speed_mps") for row in rows) * 1.1),
        time_min=time_min,
        time_max=time_max,
        bands=bands,
    )
    elements += _chart_panel(
        rows=replan_rows,
        x0=110,
        y0=440,
        width=1400,
        height=210,
        title="MPC solve time",
        series=[("mpc_solve_time_ms", "Solve time (ms)", "#3a506b")],
        y_min=0.0,
        y_max=max(
            50.0,
            max((_float(row, "mpc_solve_time_ms") for row in replan_rows), default=50.0)
            * 1.05,
        ),
        time_min=time_min,
        time_max=time_max,
        bands=bands,
    )
    elements += _chart_panel(
        rows=replan_rows,
        x0=110,
        y0=725,
        width=1400,
        height=170,
        title="MPC solver status on replan frames",
        series=[
            (
                "_solver_health",
                "Solved=1, solved inaccurate=0.75, failed=0",
                "#2f855a",
            )
        ],
        y_min=0.0,
        y_max=1.05,
        time_min=time_min,
        time_max=time_max,
        bands=bands,
    )
    elements += [
        _svg_text(810, 985, "Simulation time (s)", size=17, anchor="middle", weight=700),
        _svg_text(110, 1015, "Red shading: traffic-light stop mode", size=15, color="#59636e"),
        "</svg>",
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def export_standard_metrics(input_csv: Path, output_dir: Path) -> None:
    with input_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        all_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not all_rows:
        raise RuntimeError("The debug CSV is empty.")

    turn_index = next(
        (
            index
            for index, row in enumerate(all_rows)
            if row.get("scenario_fsm_state") == "INTERSECTION_TURN"
            or str(row.get("behavior_decision", "")).startswith("intersection_turn")
        ),
        None,
    )
    # A run that ends before the first turn is already a pre-turn-only input.
    if turn_index is None:
        turn_index = len(all_rows)
    rows = all_rows[:turn_index]
    if not rows:
        raise RuntimeError("No samples exist before the first turn command.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "pre_turn_data.csv", rows, fieldnames)

    duration_s = _float(rows[-1], "sim_time_s") - _float(rows[0], "sim_time_s")
    distance_traveled_m = sum(
        math.hypot(
            _float(current, "x_m") - _float(previous, "x_m"),
            _float(current, "y_m") - _float(previous, "y_m"),
        )
        for previous, current in zip(rows[:-1], rows[1:])
    )
    average_speed_mps = sum(_float(row, "speed_mps") for row in rows) / len(rows)
    max_speed_mps = max(_float(row, "speed_mps") for row in rows)
    replan_rows = [
        row
        for row in rows
        if str(row.get("mpc_replan_executed", "")).strip().lower() in {"true", "1"}
    ]
    successful_replans = [
        row
        for row in replan_rows
        if str(row.get("mpc_status", "")).strip().lower()
        in {"solved", "solved inaccurate"}
    ]
    solver_failures = len(replan_rows) - len(successful_replans)
    success_rate = (
        100.0 * len(successful_replans) / len(replan_rows)
        if replan_rows
        else 0.0
    )
    solve_times_ms = sorted(
        _float(row, "mpc_solve_time_ms")
        for row in replan_rows
        if str(row.get("mpc_solve_time_ms", "")).strip()
    )
    average_solve_ms = (
        sum(solve_times_ms) / len(solve_times_ms) if solve_times_ms else 0.0
    )
    p95_solve_ms = _percentile(solve_times_ms, 0.95)
    max_solve_ms = max(solve_times_ms) if solve_times_ms else 0.0
    stop_spans = _episode_spans(
        rows,
        lambda row: row.get("scenario_fsm_state") == "TRAFFIC_LIGHT_STOP",
    )
    metrics_recorded = any(
        str(row.get("evaluation_metrics_available", "")).strip().lower()
        in {"true", "1"}
        for row in rows
    )

    def finite_values(field: str) -> List[float]:
        values: List[float] = []
        for row in rows:
            raw = str(row.get(field, "")).strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    collision_values = finite_values("collision_count")
    ttc_values = finite_values("nearest_ttc_s")
    pet_values = finite_values("min_pet_s")
    drac_values = finite_values("tick_max_drac_mps2")
    boundary_rows = [
        row
        for row in rows
        if str(row.get("road_boundary_sample_valid", "")).strip().lower()
        in {"true", "1"}
    ]
    boundary_breaches = sum(
        str(row.get("road_boundary_breach", "")).strip().lower() in {"true", "1"}
        for row in boundary_rows
    )

    def unavailable_value() -> str:
        return "No finite conflict" if metrics_recorded else "N/A"

    metrics: List[Dict[str, str]] = [
        {
            "metric": "SAFETY — Collisions",
            "value": (
                str(int(max(collision_values)))
                if collision_values
                else ("0" if metrics_recorded else "N/A")
            ),
            "interpretation": (
                "Dedicated CARLA collision-sensor events in the pre-turn interval."
                if metrics_recorded
                else "Collision metrics were not enabled for this run."
            ),
        },
        {
            "metric": "SAFETY — Min TTC",
            "value": f"{min(ttc_values):.2f} s" if ttc_values else unavailable_value(),
            "interpretation": (
                "Minimum finite bumper-to-bumper TTC; README target >3 s."
                if ttc_values
                else "No finite closing conflict was observed."
                if metrics_recorded
                else "TTC metrics were not enabled for this run."
            ),
        },
        {
            "metric": "SAFETY — Min PET",
            "value": f"{min(pet_values):.2f} s" if pet_values else unavailable_value(),
            "interpretation": (
                "Minimum finite post-encroachment time in the sampled conflict grid."
                if pet_values
                else "No shared conflict-grid occupancy produced a finite PET."
                if metrics_recorded
                else "PET metrics were not enabled for this run."
            ),
        },
        {
            "metric": "SAFETY — Max DRAC",
            "value": (
                f"{max(drac_values):.2f} m/s²"
                if drac_values
                else ("0.00 m/s²" if metrics_recorded else "N/A")
            ),
            "interpretation": (
                "Maximum deceleration rate required to avoid a closing conflict; README target <3 m/s²."
                if metrics_recorded
                else "DRAC metrics were not enabled for this run."
            ),
        },
        {
            "metric": "SAFETY — Boundary breach",
            "value": (
                f"{100.0 * boundary_breaches / len(boundary_rows):.2f}%"
                if boundary_rows
                else "N/A"
            ),
            "interpretation": (
                f"{boundary_breaches}/{len(boundary_rows)} valid CARLA lane-envelope samples; README target <5%."
                if boundary_rows
                else "No valid CARLA lane-envelope samples were recorded."
            ),
        },
        {
            "metric": "EFFICIENCY — Distance traveled",
            "value": f"{distance_traveled_m:.1f} m",
            "interpretation": "Integrated from recorded ego x/y positions.",
        },
        {
            "metric": "EFFICIENCY — Average speed",
            "value": f"{average_speed_mps:.2f} m/s",
            "interpretation": "Mean speed over all pre-turn samples.",
        },
        {
            "metric": "EFFICIENCY — Max speed",
            "value": f"{max_speed_mps:.2f} m/s",
            "interpretation": "Maximum speed before the first turn command.",
        },
        {
            "metric": "MPC PLANNER — Plan attempts",
            "value": str(len(replan_rows)),
            "interpretation": "Frames where an MPC replan was executed.",
        },
        {
            "metric": "MPC PLANNER — Solver failures",
            "value": str(solver_failures),
            "interpretation": "Replans without solved/solved-inaccurate status.",
        },
        {
            "metric": "MPC PLANNER — Success rate",
            "value": f"{success_rate:.1f}%",
            "interpretation": "README target >97%; warning below 90%.",
        },
        {
            "metric": "MPC PLANNER — Avg / P95 / Max solve time",
            "value": f"{average_solve_ms:.2f} / {p95_solve_ms:.2f} / {max_solve_ms:.2f} ms",
            "interpretation": "README target <30 ms; warning above 50 ms.",
        },
    ]
    _write_csv(
        output_dir / "pre_turn_standard_metrics.csv",
        metrics,
        ["metric", "value", "interpretation"],
    )
    _write_kpi_svg(output_dir / "pre_turn_standard_metrics.svg", metrics)
    _write_standard_overview(
        output_dir / "pre_turn_standard_overview.svg",
        rows,
        stop_spans,
    )
    metric_value = {item["metric"]: item["value"] for item in metrics}

    report = f"""# CP-X Planner Pre-Turn Standard Metrics

## Scope

- Source: `{input_csv}`
- Interval: **{_float(rows[0], 'sim_time_s'):.2f}–{_float(rows[-1], 'sim_time_s'):.2f} s**
- Duration: **{duration_s:.1f} s**
- Samples: **{len(rows)}**
- This export contains no turn or post-turn samples.

## Results

| Section | Metric | Result |
|---|---|---:|
| Safety | Collisions | {metric_value["SAFETY — Collisions"]} |
| Safety | Min TTC | {metric_value["SAFETY — Min TTC"]} |
| Safety | Min PET | {metric_value["SAFETY — Min PET"]} |
| Safety | Max DRAC | {metric_value["SAFETY — Max DRAC"]} |
| Safety | Boundary breach | {metric_value["SAFETY — Boundary breach"]} |
| Efficiency | Distance traveled | {distance_traveled_m:.1f} m |
| Efficiency | Average speed | {average_speed_mps:.2f} m/s |
| Efficiency | Max speed | {max_speed_mps:.2f} m/s |
| MPC Planner | Plan attempts | {len(replan_rows)} |
| MPC Planner | Solver failures | {solver_failures} |
| MPC Planner | Success rate | {success_rate:.1f}% |
| MPC Planner | Avg / P95 / Max solve time | {average_solve_ms:.2f} / {p95_solve_ms:.2f} / {max_solve_ms:.2f} ms |

## Interpretation

The pre-turn MPC success rate is **{success_rate:.1f}%** against the documented
**97%** development target. The measured average, P95, and maximum solve times
are **{average_solve_ms:.2f} ms**, **{p95_solve_ms:.2f} ms**, and
**{max_solve_ms:.2f} ms** respectively; the documented target is 30 ms and the
warning threshold is 50 ms.

Safety values are read only from the dedicated collision sensor, pairwise
conflict evaluator, and CARLA lane-envelope measurements. A `No finite
conflict` result means collection was active but no finite event existed;
`N/A` means that metric was not available.
"""
    (output_dir / "pre_turn_standard_report.md").write_text(report, encoding="utf-8")
    print(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export_standard_metrics(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
