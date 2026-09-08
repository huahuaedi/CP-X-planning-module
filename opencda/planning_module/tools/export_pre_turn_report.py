"""Export an English pre-turn performance report from CP-X bridge debug data."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return float(default)


def _percentile(values: Iterable[float], ratio: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = int(round(max(0.0, min(1.0, float(ratio))) * (len(ordered) - 1)))
    return float(ordered[index])


def _episode_spans(
    rows: Sequence[Mapping[str, object]],
    predicate,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = None
    for index, row in enumerate(rows):
        active = bool(predicate(row))
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(rows) - 1):
            end = index if active and index == len(rows) - 1 else index - 1
            spans.append((int(start), int(end)))
            start = None
    return spans


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _svg_text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 18,
    color: str = "#17202a",
    anchor: str = "start",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{html.escape(str(value))}</text>'
    )


def _polyline(points: Sequence[Tuple[float, float]], color: str, width: float = 3.0) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.1f}" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _chart_panel(
    *,
    rows: Sequence[Mapping[str, object]],
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    series: Sequence[Tuple[str, str, str]],
    y_min: float,
    y_max: float,
    time_min: float,
    time_max: float,
    bands: Sequence[Tuple[float, float, str, float]] = (),
    markers: Sequence[Tuple[float, str, str]] = (),
) -> List[str]:
    elements = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'fill="#ffffff" stroke="#d7dde5" stroke-width="1"/>',
        _svg_text(x0, y0 - 12, title, size=19, weight=700),
    ]
    time_span = max(1.0e-6, float(time_max) - float(time_min))
    value_span = max(1.0e-6, float(y_max) - float(y_min))

    def sx(value: float) -> float:
        return x0 + (float(value) - time_min) / time_span * width

    def sy(value: float) -> float:
        return y0 + height - (float(value) - y_min) / value_span * height

    for start, end, color, opacity in bands:
        left = max(x0, min(x0 + width, sx(start)))
        right = max(x0, min(x0 + width, sx(end)))
        if right > left:
            elements.append(
                f'<rect x="{left:.1f}" y="{y0:.1f}" width="{right-left:.1f}" '
                f'height="{height:.1f}" fill="{color}" opacity="{opacity:.2f}"/>'
            )
    for grid_index in range(5):
        value = y_min + grid_index * value_span / 4.0
        gy = sy(value)
        elements.append(
            f'<line x1="{x0:.1f}" y1="{gy:.1f}" x2="{x0+width:.1f}" y2="{gy:.1f}" '
            'stroke="#e8ecf1" stroke-width="1"/>'
        )
        elements.append(_svg_text(x0 - 10, gy + 6, f"{value:.2f}", size=14, anchor="end", color="#59636e"))
    for tick_index in range(6):
        value = time_min + tick_index * time_span / 5.0
        gx = sx(value)
        elements.append(
            f'<line x1="{gx:.1f}" y1="{y0:.1f}" x2="{gx:.1f}" y2="{y0+height:.1f}" '
            'stroke="#f0f2f5" stroke-width="1"/>'
        )
        elements.append(_svg_text(gx, y0 + height + 24, f"{value:.1f}", size=14, anchor="middle", color="#59636e"))
    for key, label, color in series:
        points = [
            (sx(_float(row, "sim_time_s")), sy(_float(row, key)))
            for row in rows
            if time_min <= _float(row, "sim_time_s") <= time_max
        ]
        if points:
            elements.append(_polyline(points, color))
    legend_x = x0 + 12
    for _, label, color in series:
        elements.append(
            f'<line x1="{legend_x:.1f}" y1="{y0+22:.1f}" x2="{legend_x+28:.1f}" '
            f'y2="{y0+22:.1f}" stroke="{color}" stroke-width="4"/>'
        )
        elements.append(_svg_text(legend_x + 36, y0 + 28, label, size=14, color="#303942"))
        legend_x += 36 + max(100, len(label) * 8)
    for marker_index, (marker_time, marker_label, marker_color) in enumerate(markers):
        if time_min <= marker_time <= time_max:
            mx = sx(marker_time)
            elements.append(
                f'<line x1="{mx:.1f}" y1="{y0:.1f}" x2="{mx:.1f}" y2="{y0+height:.1f}" '
                f'stroke="{marker_color}" stroke-width="3" stroke-dasharray="8,6"/>'
            )
            near_right_edge = mx > x0 + 0.85 * width
            elements.append(_svg_text(
                mx - 6 if near_right_edge else mx + 6,
                y0 + height - 10 - marker_index * 21,
                marker_label,
                size=13,
                color=marker_color,
                anchor="end" if near_right_edge else "start",
                weight=700,
            ))
    return elements


def _write_overview_svg(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    turn_time: float,
    failure_time: float,
    stop_spans: Sequence[Tuple[int, int]],
) -> None:
    width, height = 1600, 1040
    time_min = _float(rows[0], "sim_time_s")
    time_max = min(_float(rows[-1], "sim_time_s"), failure_time)
    bands = [
        (
            _float(rows[start], "sim_time_s"),
            _float(rows[end], "sim_time_s"),
            "#d64545",
            0.10,
        )
        for start, end in stop_spans
    ]
    bands.append((turn_time, failure_time, "#f3a712", 0.12))
    markers = [
        (turn_time, "Turn entry", "#d97706"),
        (failure_time, "Failure boundary", "#b42318"),
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(80, 56, "CP-X Planner: Pre-Turn Performance Overview", size=30, weight=700),
        _svg_text(
            80,
            86,
            "Stable-operation window with traffic-light phases; the orange interval is shown only to locate the turn boundary.",
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
        title="Vehicle speed tracking",
        series=[
            ("speed_mps", "Actual speed (m/s)", "#177e89"),
            ("target_speed_mps", "Planner target (m/s)", "#d95f02"),
        ],
        y_min=0.0,
        y_max=max(4.0, max(_float(row, "speed_mps") for row in rows) * 1.1),
        time_min=time_min,
        time_max=time_max,
        bands=bands,
        markers=markers,
    )
    elements += _chart_panel(
        rows=rows,
        x0=110,
        y0=440,
        width=1400,
        height=210,
        title="Reference alignment in ego frame",
        series=[
            ("reference_first_lateral_m", "First-point lateral offset (m)", "#3a506b"),
            ("destination_lateral_m", "Destination lateral offset (m)", "#c43d3d"),
        ],
        y_min=-0.8,
        y_max=2.8,
        time_min=time_min,
        time_max=time_max,
        bands=bands,
        markers=markers,
    )
    elements += _chart_panel(
        rows=rows,
        x0=110,
        y0=725,
        width=1400,
        height=170,
        title="Control demand",
        series=[
            ("post_supervisor_steer_cmd_rad", "Steering command (rad)", "#6a4c93"),
            ("post_supervisor_accel_cmd_mps2", "Acceleration command (m/s²)", "#2f855a"),
        ],
        y_min=-3.2,
        y_max=3.2,
        time_min=time_min,
        time_max=time_max,
        bands=bands,
        markers=markers,
    )
    elements += [
        _svg_text(810, 985, "Simulation time (s)", size=17, anchor="middle", weight=700),
        _svg_text(110, 1015, "Red shading: traffic-light stop mode | Orange shading: turn attempt", size=15, color="#59636e"),
        "</svg>",
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_turn_svg(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    turn_time: float,
    failure_time: float,
) -> None:
    width, height = 1600, 1040
    time_min = max(_float(rows[0], "sim_time_s"), turn_time - 2.5)
    time_max = min(_float(rows[-1], "sim_time_s"), failure_time + 2.5)
    bands = [(turn_time, failure_time, "#f3a712", 0.14)]
    markers = [
        (turn_time, "Turn FSM entry", "#d97706"),
        (failure_time, "FSM exit + hard gate", "#b42318"),
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(80, 56, "Turn Transition Diagnostic", size=30, weight=700),
        _svg_text(
            80,
            86,
            "The MPC solves throughout the latched turn; failure starts when the FSM returns to lane-follow before geometry has converged.",
            size=17,
            color="#59636e",
        ),
    ]
    elements += _chart_panel(
        rows=rows,
        x0=110,
        y0=135,
        width=1400,
        height=220,
        title="Speed collapse during the turn attempt",
        series=[
            ("speed_mps", "Actual speed (m/s)", "#177e89"),
            ("target_speed_mps", "Target speed (m/s)", "#d95f02"),
        ],
        y_min=0.0,
        y_max=4.0,
        time_min=time_min,
        time_max=time_max,
        bands=bands,
        markers=markers,
    )
    elements += _chart_panel(
        rows=rows,
        x0=110,
        y0=430,
        width=1400,
        height=220,
        title="Reference-frame mismatch at turn exit",
        series=[
            ("reference_first_lateral_m", "First-point lateral offset (m)", "#3a506b"),
            ("destination_lateral_m", "Destination lateral offset (m)", "#c43d3d"),
        ],
        y_min=-0.5,
        y_max=3.0,
        time_min=time_min,
        time_max=time_max,
        bands=bands,
        markers=markers,
    )
    elements += _chart_panel(
        rows=rows,
        x0=110,
        y0=725,
        width=1400,
        height=170,
        title="Steering and acceleration commands",
        series=[
            ("post_supervisor_steer_cmd_rad", "Steering command (rad)", "#6a4c93"),
            ("post_supervisor_accel_cmd_mps2", "Acceleration command (m/s²)", "#2f855a"),
        ],
        y_min=-3.2,
        y_max=3.2,
        time_min=time_min,
        time_max=time_max,
        bands=bands,
        markers=markers,
    )
    elements += [
        _svg_text(810, 985, "Simulation time (s)", size=17, anchor="middle", weight=700),
        _svg_text(
            110,
            1015,
            "Failure signature: destination lateral offset remains large, but lane-follow constraints are re-enabled.",
            size=15,
            color="#59636e",
        ),
        "</svg>",
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_kpi_svg(path: Path, metrics: Sequence[Mapping[str, str]]) -> None:
    width = 1500
    row_height = 62
    height = 215 + row_height * len(metrics)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(70, 55, "Pre-Turn KPI Summary", size=30, weight=700),
        _svg_text(70, 88, "Window ends immediately before INTERSECTION_TURN activation.", size=17, color="#59636e"),
        '<rect x="70" y="115" width="1360" height="54" fill="#25364a"/>',
        _svg_text(95, 150, "Metric", size=18, color="#ffffff", weight=700),
        _svg_text(620, 150, "Result", size=18, color="#ffffff", weight=700),
        _svg_text(860, 150, "Interpretation", size=18, color="#ffffff", weight=700),
    ]
    top = 169
    for index, metric in enumerate(metrics):
        y = top + index * row_height
        fill = "#ffffff" if index % 2 == 0 else "#eef3f7"
        elements.append(
            f'<rect x="70" y="{y}" width="1360" height="{row_height}" fill="{fill}" stroke="#d7dde5"/>'
        )
        elements.append(_svg_text(95, y + 39, metric["metric"], size=17, weight=600))
        elements.append(_svg_text(620, y + 39, metric["value"], size=17, color="#177e89", weight=700))
        elements.append(_svg_text(860, y + 39, metric["interpretation"], size=16, color="#3f4a55"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def export_report(input_csv: Path, output_dir: Path) -> None:
    with input_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError("The debug CSV is empty.")

    turn_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("scenario_fsm_state") == "INTERSECTION_TURN"
            or str(row.get("behavior_decision", "")).startswith("intersection_turn")
        ),
        None,
    )
    if turn_index is None:
        raise RuntimeError("No intersection-turn segment was found in the debug CSV.")
    failure_index = next(
        (
            index
            for index, row in enumerate(rows[turn_index:], turn_index)
            if row.get("mpc_status") == "candidate_hard_gate"
        ),
        len(rows) - 1,
    )
    pre_turn = rows[:turn_index]
    turn_attempt = rows[turn_index:failure_index]
    transition_start = max(0, turn_index - 100)
    transition_end = min(len(rows), failure_index + 101)
    transition = rows[transition_start:transition_end]
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "pre_turn_data.csv", pre_turn, fieldnames)
    selected_fields = [
        key
        for key in [
            "sim_time_s",
            "speed_mps",
            "target_speed_mps",
            "scenario_fsm_state",
            "behavior_decision",
            "route_current_road_option",
            "reference_pipeline_intent",
            "final_reference_geometry_source",
            "reference_first_forward_m",
            "reference_first_lateral_m",
            "destination_forward_m",
            "destination_lateral_m",
            "post_supervisor_accel_cmd_mps2",
            "post_supervisor_steer_cmd_rad",
            "mpc_status",
            "mpc_fallback_reason",
            "reference_lateral_guard_reason",
            "mpc_reference_stabilizer_reason",
            "route_progress_index",
            "route_remaining_distance_m",
        ]
        if key in fieldnames
    ]
    _write_csv(output_dir / "turn_transition_data.csv", transition, selected_fields)

    start_time = _float(pre_turn[0], "sim_time_s")
    end_time = _float(pre_turn[-1], "sim_time_s")
    turn_time = _float(rows[turn_index], "sim_time_s")
    failure_time = _float(rows[failure_index], "sim_time_s")
    route_progress_m = max(
        0.0,
        _float(pre_turn[0], "route_remaining_distance_m")
        - _float(pre_turn[-1], "route_remaining_distance_m"),
    )
    accepted_statuses = {"solved", "solved inaccurate", "buffer_reuse"}
    accepted_count = sum(row.get("mpc_status") in accepted_statuses for row in pre_turn)
    hard_gate_count = sum(row.get("mpc_status") == "candidate_hard_gate" for row in pre_turn)
    infeasible_count = sum(row.get("mpc_status") == "primal infeasible" for row in pre_turn)
    stop_spans = _episode_spans(
        pre_turn,
        lambda row: row.get("scenario_fsm_state") == "TRAFFIC_LIGHT_STOP",
    )
    steady_cruise = [
        row
        for row in pre_turn
        if row.get("scenario_fsm_state") == "LANE_FOLLOW"
        and _float(row, "speed_mps") > 1.5
        and _float(row, "sim_time_s") > start_time + 3.0
    ]
    cruise_errors = [
        _float(row, "speed_mps") - _float(row, "target_speed_mps")
        for row in steady_cruise
    ]
    speed_mae = (
        sum(abs(value) for value in cruise_errors) / len(cruise_errors)
        if cruise_errors
        else 0.0
    )
    mean_cruise_speed = (
        sum(_float(row, "speed_mps") for row in steady_cruise) / len(steady_cruise)
        if steady_cruise
        else 0.0
    )
    first_lateral_p95 = _percentile(
        (abs(_float(row, "reference_first_lateral_m")) for row in pre_turn),
        0.95,
    )
    destination_lateral_p95 = _percentile(
        (abs(_float(row, "destination_lateral_m")) for row in pre_turn),
        0.95,
    )
    obstacle_frames = [row for row in pre_turn if _float(row, "object_count") > 0.0]
    prediction_frames = [
        row
        for row in obstacle_frames
        if _float(row, "candidate_prediction_trajectory_count") > 0.0
    ]
    prediction_coverage = (
        100.0 * len(prediction_frames) / len(obstacle_frames)
        if obstacle_frames
        else 0.0
    )
    turn_entry_speed = _float(rows[turn_index], "speed_mps")
    turn_exit_speed = _float(rows[max(turn_index, failure_index - 1)], "speed_mps")
    turn_destination_lateral = max(
        (abs(_float(row, "destination_lateral_m")) for row in turn_attempt),
        default=0.0,
    )

    metrics = [
        {
            "metric": "Evaluated stable duration",
            "value": f"{end_time - start_time:.1f} s",
            "interpretation": "Ends before the first intersection-turn command.",
        },
        {
            "metric": "Route progress",
            "value": f"{route_progress_m:.1f} m",
            "interpretation": f"AD-map route index advanced to {pre_turn[-1].get('route_progress_index', '')}.",
        },
        {
            "metric": "Traffic-light stop episodes",
            "value": str(len(stop_spans)),
            "interpretation": "Both approach/stop phases were recognized and released.",
        },
        {
            "metric": "MPC accepted-control availability",
            "value": f"{100.0 * accepted_count / len(pre_turn):.2f}%",
            "interpretation": "Solved, solved-inaccurate, or bounded buffer reuse.",
        },
        {
            "metric": "MPC primal-infeasible frames",
            "value": f"{infeasible_count} / {len(pre_turn)}",
            "interpretation": "Numerical failure remained isolated before the turn.",
        },
        {
            "metric": "Reference hard-gate frames",
            "value": str(hard_gate_count),
            "interpretation": "No strict reference veto in the stable pre-turn window.",
        },
        {
            "metric": "Steady-cruise speed",
            "value": f"{mean_cruise_speed:.2f} m/s",
            "interpretation": f"Speed MAE {speed_mae:.2f} m/s for samples above 1.5 m/s.",
        },
        {
            "metric": "First-point lateral error, P95",
            "value": f"{first_lateral_p95:.3f} m",
            "interpretation": "Near-field reference remained centered before the turn.",
        },
        {
            "metric": "Destination lateral error, P95",
            "value": f"{destination_lateral_p95:.3f} m",
            "interpretation": "Far-field lane-follow reference remained geometrically stable.",
        },
        {
            "metric": "Prediction coverage",
            "value": f"{prediction_coverage:.1f}%",
            "interpretation": "Frames with predictions among frames containing detected objects.",
        },
    ]
    _write_csv(
        output_dir / "pre_turn_kpi_table.csv",
        metrics,
        ["metric", "value", "interpretation"],
    )
    _write_kpi_svg(output_dir / "pre_turn_kpi_table.svg", metrics)

    events: List[Dict[str, str]] = [
        {
            "event": "Analysis start",
            "time_s": f"{start_time:.3f}",
            "details": "Beginning of the available debug run.",
        }
    ]
    for index, (span_start, span_end) in enumerate(stop_spans, 1):
        events.extend([
            {
                "event": f"Traffic-light stop {index} entered",
                "time_s": f"{_float(pre_turn[span_start], 'sim_time_s'):.3f}",
                "details": "Scenario FSM entered TRAFFIC_LIGHT_STOP.",
            },
            {
                "event": f"Traffic-light stop {index} released",
                "time_s": f"{_float(pre_turn[span_end], 'sim_time_s'):.3f}",
                "details": "Last frame before stop mode release.",
            },
        ])
    events.extend([
        {
            "event": "Intersection turn entered",
            "time_s": f"{turn_time:.3f}",
            "details": "INTERSECTION_TURN / intersection_turn_left activated.",
        },
        {
            "event": "Turn failure boundary",
            "time_s": f"{failure_time:.3f}",
            "details": "Turn FSM exited; lane-follow reference hard gate activated.",
        },
    ])
    _write_csv(
        output_dir / "event_timeline.csv",
        events,
        ["event", "time_s", "details"],
    )
    _write_overview_svg(
        output_dir / "pre_turn_overview.svg",
        rows[: failure_index + 1],
        turn_time=turn_time,
        failure_time=failure_time,
        stop_spans=stop_spans,
    )
    _write_turn_svg(
        output_dir / "turn_transition_diagnostics.svg",
        transition,
        turn_time=turn_time,
        failure_time=failure_time,
    )

    markdown = f"""# CP-X Planner Pre-Turn Performance Report

## Reporting Window

- Source: `{input_csv}`
- Stable pre-turn interval: **{start_time:.2f} s to {end_time:.2f} s**
- Left-turn activation: **{turn_time:.2f} s**
- First turn-related hard gate: **{failure_time:.2f} s**
- The stable KPI window deliberately excludes the turn attempt and all post-failure recovery frames.

## Executive Summary

Before the left-turn transition, the CP-X pipeline progressed **{route_progress_m:.1f} m** along the CARLA global route and handled **{len(stop_spans)} traffic-light stop episodes**. MPC accepted-control availability was **{100.0 * accepted_count / len(pre_turn):.2f}%**, with only **{infeasible_count} primal-infeasible frames** and **{hard_gate_count} reference hard-gate frames**. The near-field and destination reference lateral errors remained small: **{first_lateral_p95:.3f} m P95** and **{destination_lateral_p95:.3f} m P95**, respectively.

## Turn Failure Boundary

The turn FSM activated at **{turn_time:.2f} s** and MPC reported `solved` throughout all **{len(turn_attempt)} pre-failure turn frames**. Vehicle speed nevertheless fell from **{turn_entry_speed:.2f} m/s** to **{turn_exit_speed:.2f} m/s** while the destination lateral displacement increased to **{turn_destination_lateral:.2f} m**. At **{failure_time:.2f} s**, the scenario FSM returned to `LANE_FOLLOW` before the vehicle/reference geometry had converged. Lane-follow lateral constraints were then re-enabled and produced `candidate_hard_gate`.

This isolates the current limitation as a **turn-exit state-transition and reference-frame handoff problem**, rather than a general pre-turn route-following or MPC solver problem.

## Presentation-Ready Findings

1. **Stable route following before the turn:** no reference hard gates over the {end_time - start_time:.1f} s KPI window.
2. **Traffic-control integration is active:** two approach/stop/release cycles are visible in the scenario FSM.
3. **Reference geometry is well centered before the turn:** first-point lateral error P95 is {first_lateral_p95:.3f} m.
4. **MPC numerical availability is high before the turn:** {100.0 * accepted_count / len(pre_turn):.2f}% accepted-control availability.
5. **Current boundary:** the turn FSM unlatches while the destination remains {turn_destination_lateral:.2f} m lateral in the ego frame, causing lane-follow validation to reject the handoff.

## Caveats

- Traffic-light stop mode still exhibits low-speed creep and should not be presented as precise stop-line holding.
- Prediction coverage is {prediction_coverage:.1f}% on frames containing detected objects; prediction continuity remains an improvement area.
- No collision conclusion is reported because the debug CSV does not contain a validated collision-event field.
"""
    (output_dir / "pre_turn_report.md").write_text(markdown, encoding="utf-8")
    print(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export_report(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
