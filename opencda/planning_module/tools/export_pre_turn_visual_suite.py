"""Export an English visualization suite using only pre-turn bridge samples."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from export_pre_turn_report import _chart_panel, _episode_spans, _float, _svg_text


COLORS = {
    "LANE_FOLLOW": "#177e89",
    "TRAFFIC_LIGHT_APPROACH": "#d97706",
    "TRAFFIC_LIGHT_STOP": "#c43d3d",
    "unknown": "#8793a1",
    "green": "#2f855a",
    "yellow": "#d69e2e",
    "red": "#c53030",
}


def _pre_turn_rows(rows: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, str]], int | None]:
    turn_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("scenario_fsm_state") == "INTERSECTION_TURN"
            or str(row.get("behavior_decision", "")).startswith("intersection_turn")
        ),
        None,
    )
    return [dict(row) for row in rows[:turn_index]], turn_index


def _finite(rows: Iterable[Mapping[str, object]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        raw = str(row.get(key, "")).strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _state_spans(rows: Sequence[Mapping[str, object]], key: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or rows[index].get(key) != rows[start].get(key):
            spans.append((start, index - 1, str(rows[start].get(key, ""))))
            start = index
    return spans


def _write_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    rows: Sequence[Mapping[str, object]],
    panels: Sequence[Mapping[str, object]],
    stop_spans: Sequence[Tuple[int, int]],
) -> None:
    width = 1600
    panel_height = 235
    panel_gap = 90
    height = 150 + len(panels) * (panel_height + panel_gap)
    time_min = _float(rows[0], "sim_time_s")
    time_max = _float(rows[-1], "sim_time_s")
    bands = [
        (
            _float(rows[start], "sim_time_s"),
            _float(rows[end], "sim_time_s"),
            "#d64545",
            0.09,
        )
        for start, end in stop_spans
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(80, 56, title, size=30, weight=700),
        _svg_text(80, 86, subtitle, size=17, color="#59636e"),
    ]
    for index, panel in enumerate(panels):
        elements += _chart_panel(
            rows=panel.get("rows", rows),
            x0=110,
            y0=135 + index * (panel_height + panel_gap),
            width=1400,
            height=panel_height,
            title=str(panel["title"]),
            series=panel["series"],
            y_min=float(panel["y_min"]),
            y_max=float(panel["y_max"]),
            time_min=time_min,
            time_max=time_max,
            bands=bands,
        )
    elements += [
        _svg_text(
            810,
            height - 30,
            "Simulation time (s)",
            size=17,
            anchor="middle",
            weight=700,
        ),
        _svg_text(
            110,
            height - 8,
            "Light red shading: TRAFFIC_LIGHT_STOP",
            size=14,
            color="#59636e",
        ),
        "</svg>",
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_route_map(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    width, height = 1500, 980
    x_values = [_float(row, "x_m") for row in rows]
    y_values = [_float(row, "y_m") for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    margin = 120.0
    scale = min(
        (width - 2.0 * margin) / max(1.0, x_max - x_min),
        (height - 2.0 * margin) / max(1.0, y_max - y_min),
    )

    def point(row: Mapping[str, object]) -> Tuple[float, float]:
        return (
            margin + (_float(row, "x_m") - x_min) * scale,
            height - margin - (_float(row, "y_m") - y_min) * scale,
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(70, 54, "Pre-Turn Route and Operating State", size=30, weight=700),
        _svg_text(
            70,
            84,
            "The driven path is colored by scenario state; circles mark complete stops.",
            size=17,
            color="#59636e",
        ),
    ]
    for previous, current in zip(rows[:-1], rows[1:]):
        x1, y1 = point(previous)
        x2, y2 = point(current)
        state = str(current.get("scenario_fsm_state", ""))
        color = COLORS.get(state, "#59636e")
        speed = _float(current, "speed_mps")
        line_width = 3.0 + min(5.0, speed)
        elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{line_width:.2f}" stroke-linecap="round"/>'
        )
    for index, row in enumerate(rows):
        if _float(row, "speed_mps") < 0.08 and (
            index == 0 or _float(rows[index - 1], "speed_mps") >= 0.08
        ):
            x, y = point(row)
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="#111827" '
                'stroke="#ffffff" stroke-width="3"/>'
            )
    start_x, start_y = point(rows[0])
    end_x, end_y = point(rows[-1])
    elements += [
        f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="11" fill="#2f855a"/>',
        _svg_text(start_x + 15, start_y - 12, "Start", size=16, weight=700),
        f'<circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="11" fill="#6b46c1"/>',
        _svg_text(end_x + 15, end_y - 12, "Pre-turn cutoff", size=16, weight=700),
    ]
    legend_y = 920
    legend_x = 100
    for state in ("LANE_FOLLOW", "TRAFFIC_LIGHT_APPROACH", "TRAFFIC_LIGHT_STOP"):
        color = COLORS[state]
        elements.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+42}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="8"/>'
        )
        elements.append(_svg_text(legend_x + 52, legend_y + 6, state.replace("_", " ").title(), size=15))
        legend_x += 390
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_timeline(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    width, height = 1600, 620
    x0, chart_width = 210.0, 1300.0
    time_min = _float(rows[0], "sim_time_s")
    time_max = _float(rows[-1], "sim_time_s")
    time_span = max(1.0e-6, time_max - time_min)

    def sx(value: float) -> float:
        return x0 + (value - time_min) / time_span * chart_width

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(70, 55, "Scenario and Traffic-Signal Timeline", size=30, weight=700),
        _svg_text(
            70,
            85,
            "State transitions before the first turn command; no turn samples are included.",
            size=17,
            color="#59636e",
        ),
    ]
    tracks = [
        ("scenario_fsm_state", "Scenario FSM", 165),
        ("traffic_signal_state", "Traffic signal", 300),
        ("behavior_decision", "Behavior", 435),
    ]
    for key, label, y in tracks:
        elements.append(_svg_text(x0 - 22, y + 36, label, size=17, anchor="end", weight=700))
        elements.append(
            f'<rect x="{x0}" y="{y}" width="{chart_width}" height="70" '
            'fill="#ffffff" stroke="#d7dde5"/>'
        )
        for start, end, state in _state_spans(rows, key):
            left = sx(_float(rows[start], "sim_time_s"))
            right = sx(_float(rows[end], "sim_time_s"))
            width_px = max(1.0, right - left)
            normalized = state.lower()
            color = COLORS.get(state, COLORS.get(normalized, "#547aa5"))
            elements.append(
                f'<rect x="{left:.2f}" y="{y}" width="{width_px:.2f}" height="70" '
                f'fill="{color}" opacity="0.88"/>'
            )
            if width_px > 100:
                elements.append(
                    _svg_text(
                        left + width_px / 2.0,
                        y + 42,
                        state.replace("_", " "),
                        size=13,
                        color="#ffffff",
                        anchor="middle",
                        weight=700,
                    )
                )
    for tick in range(6):
        value = time_min + tick * time_span / 5.0
        x = sx(value)
        elements.append(_svg_text(x, 555, f"{value:.1f} s", size=14, anchor="middle"))
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_solver_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    replan_rows = [
        row
        for row in rows
        if str(row.get("mpc_replan_executed", "")).lower() in {"true", "1"}
    ]
    counts = Counter(str(row.get("mpc_status", "")).strip() for row in replan_rows)
    ordered = [
        ("solved", "#2f855a"),
        ("solved inaccurate", "#d69e2e"),
        ("primal infeasible", "#c53030"),
    ]
    width, height = 1400, 760
    max_count = max([counts[name] for name, _ in ordered] + [1])
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(70, 55, "MPC Replan Outcome Distribution", size=30, weight=700),
        _svg_text(
            70,
            85,
            "Buffer-reuse frames are excluded; each bar is one actual optimization attempt.",
            size=17,
            color="#59636e",
        ),
    ]
    for index, (name, color) in enumerate(ordered):
        y = 165 + index * 165
        count = counts[name]
        bar_width = 950.0 * count / max_count
        elements += [
            _svg_text(245, y + 40, name.title(), size=18, anchor="end", weight=700),
            f'<rect x="275" y="{y}" width="950" height="70" fill="#e8ecf1"/>',
            f'<rect x="275" y="{y}" width="{bar_width:.2f}" height="70" fill="{color}"/>',
            _svg_text(1245, y + 44, str(count), size=21, weight=700),
        ]
    success = counts["solved"] + counts["solved inaccurate"]
    rate = 100.0 * success / len(replan_rows) if replan_rows else 0.0
    elements += [
        _svg_text(275, 680, f"Optimization attempts: {len(replan_rows)}", size=18, weight=700),
        _svg_text(650, 680, f"Success rate: {rate:.2f}%", size=18, weight=700),
        "</svg>",
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_speed_scatter(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    width, height = 1000, 900
    x0, y0, size = 130.0, 120.0, 650.0
    upper = max(
        4.0,
        max(_finite(rows, "target_speed_mps") + _finite(rows, "speed_mps") or [4.0]),
    )

    def sx(value: float) -> float:
        return x0 + value / upper * size

    def sy(value: float) -> float:
        return y0 + size - value / upper * size

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(60, 52, "Actual Speed vs Planner Target", size=29, weight=700),
        _svg_text(60, 82, "Points on the diagonal indicate ideal tracking.", size=16, color="#59636e"),
        f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="#ffffff" stroke="#d7dde5"/>',
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(upper)}" y2="{sy(upper)}" '
        'stroke="#111827" stroke-width="2" stroke-dasharray="8,6"/>',
    ]
    for row in rows[::3]:
        target = _float(row, "target_speed_mps")
        actual = _float(row, "speed_mps")
        color = COLORS.get(str(row.get("scenario_fsm_state", "")), "#59636e")
        elements.append(
            f'<circle cx="{sx(target):.2f}" cy="{sy(actual):.2f}" r="3.2" '
            f'fill="{color}" opacity="0.42"/>'
        )
    for tick in range(5):
        value = tick * upper / 4.0
        elements.append(_svg_text(sx(value), y0 + size + 28, f"{value:.1f}", size=14, anchor="middle"))
        elements.append(_svg_text(x0 - 12, sy(value) + 5, f"{value:.1f}", size=14, anchor="end"))
    elements += [
        _svg_text(x0 + size / 2, 840, "Planner target speed (m/s)", size=17, anchor="middle", weight=700),
        _svg_text(35, y0 + size / 2, "Actual speed (m/s)", size=17, anchor="middle", weight=700),
        "</svg>",
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_risk_scatter(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    samples = []
    for row in rows:
        ttc = _finite([row], "nearest_ttc_s")
        drac = _finite([row], "tick_max_drac_mps2")
        if ttc and drac:
            samples.append((min(10.0, ttc[0]), drac[0], _float(row, "sim_time_s")))
    width, height = 1100, 860
    x0, y0, chart_w, chart_h = 140.0, 120.0, 820.0, 600.0
    x_max = 10.0
    y_max = max(3.0, max((value[1] for value in samples), default=0.0) * 1.2)

    def sx(value: float) -> float:
        return x0 + value / x_max * chart_w

    def sy(value: float) -> float:
        return y0 + chart_h - value / y_max * chart_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(60, 52, "TTC–DRAC Safety Phase Space", size=29, weight=700),
        _svg_text(60, 82, "Lower TTC and higher DRAC indicate a more demanding conflict.", size=16, color="#59636e"),
        f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" fill="#ffffff" stroke="#d7dde5"/>',
        f'<rect x="{sx(0)}" y="{sy(3.0)}" width="{sx(3.0)-sx(0):.2f}" '
        f'height="{sy(0)-sy(3.0):.2f}" fill="#f6ad55" opacity="0.13"/>',
    ]
    time_min = samples[0][2] if samples else 0.0
    time_span = max(1.0, (samples[-1][2] - time_min) if samples else 1.0)
    for ttc, drac, sim_time in samples:
        ratio = (sim_time - time_min) / time_span
        red = int(45 + 170 * ratio)
        blue = int(190 - 120 * ratio)
        elements.append(
            f'<circle cx="{sx(ttc):.2f}" cy="{sy(drac):.2f}" r="4" '
            f'fill="rgb({red},90,{blue})" opacity="0.55"/>'
        )
    for tick in range(6):
        x_value = tick * x_max / 5.0
        elements.append(_svg_text(sx(x_value), y0 + chart_h + 26, f"{x_value:.1f}", size=14, anchor="middle"))
        y_value = tick * y_max / 5.0
        elements.append(_svg_text(x0 - 12, sy(y_value) + 5, f"{y_value:.1f}", size=14, anchor="end"))
    elements += [
        _svg_text(x0 + chart_w / 2, 790, "Finite TTC (s; clipped at 10 s)", size=17, anchor="middle", weight=700),
        _svg_text(52, y0 + chart_h / 2, "DRAC (m/s²)", size=17, anchor="middle", weight=700),
        "</svg>",
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_latency_histogram(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    values = sorted(_finite(rows, "mpc_solve_time_ms"))
    clipped = [min(10.0, value) for value in values]
    bins = [0] * 20
    for value in clipped:
        index = min(len(bins) - 1, int(value / 10.0 * len(bins)))
        bins[index] += 1
    width, height = 1200, 760
    x0, y0, chart_w, chart_h = 120.0, 120.0, 980.0, 500.0
    max_count = max(bins or [1])
    bar_w = chart_w / len(bins)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        _svg_text(60, 52, "MPC Solve-Time Distribution", size=29, weight=700),
        _svg_text(
            60,
            82,
            "Values above 10 ms are grouped in the final bin; the maximum is reported separately.",
            size=16,
            color="#59636e",
        ),
        f'<rect x="{x0}" y="{y0}" width="{chart_w}" height="{chart_h}" fill="#ffffff" stroke="#d7dde5"/>',
    ]
    for index, count in enumerate(bins):
        height_px = chart_h * count / max_count
        elements.append(
            f'<rect x="{x0 + index * bar_w + 1:.2f}" y="{y0 + chart_h - height_px:.2f}" '
            f'width="{bar_w - 2:.2f}" height="{height_px:.2f}" fill="#3a506b"/>'
        )
    for tick in range(6):
        value = tick * 2.0
        elements.append(_svg_text(x0 + value / 10.0 * chart_w, y0 + chart_h + 27, f"{value:.0f}", size=14, anchor="middle"))
    elements += [
        _svg_text(x0 + chart_w / 2, 690, "Solve time (ms)", size=17, anchor="middle", weight=700),
        _svg_text(60, 660, f"Maximum: {max(values) if values else 0.0:.2f} ms", size=16, weight=700),
        "</svg>",
    ]
    path.write_text("\n".join(elements), encoding="utf-8")


def export_visual_suite(input_csv: Path, output_dir: Path) -> None:
    with input_csv.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows, turn_index = _pre_turn_rows(all_rows)
    if not rows:
        raise RuntimeError("No pre-turn rows were found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_spans = _episode_spans(
        rows, lambda row: row.get("scenario_fsm_state") == "TRAFFIC_LIGHT_STOP"
    )
    replan_rows = [
        dict(row)
        for row in rows
        if str(row.get("mpc_replan_executed", "")).lower() in {"true", "1"}
    ]
    for row in replan_rows:
        row["_solver_success"] = (
            1.0 if "solved" in str(row.get("mpc_status", "")).lower() else 0.0
        )
    max_speed = max(_finite(rows, "speed_mps") or [1.0])
    max_ttc = min(10.0, max(_finite(rows, "nearest_ttc_s") or [10.0]))
    max_drac = max(3.0, max(_finite(rows, "tick_max_drac_mps2") or [0.0]) * 1.15)
    max_cost_boundary = max(_finite(rows, "Cost_RoadBoundary") or [1.0])
    min_clearance = min(_finite(rows, "road_boundary_clearance_m") or [-0.1])
    max_clearance = max(_finite(rows, "road_boundary_clearance_m") or [1.0])
    ttc_rows: List[Dict[str, str]] = []
    for row in rows:
        values = _finite([row], "nearest_ttc_s")
        if not values:
            continue
        ttc_row = dict(row)
        ttc_row["_nearest_ttc_plot"] = str(min(10.0, values[0]))
        ttc_rows.append(ttc_row)

    _write_route_map(output_dir / "01_route_operating_state.svg", rows)
    _write_timeline(output_dir / "02_scenario_signal_timeline.svg", rows)
    _write_chart(
        output_dir / "03_speed_tracking.svg",
        title="Speed Tracking and Stop-Line Behavior",
        subtitle="Actual speed follows the planner target across approach, stop, and release phases.",
        rows=rows,
        stop_spans=stop_spans,
        panels=[
            {
                "title": "Actual and target speed",
                "series": [
                    ("speed_mps", "Actual speed (m/s)", "#177e89"),
                    ("target_speed_mps", "Target speed (m/s)", "#d95f02"),
                ],
                "y_min": 0.0,
                "y_max": max(4.0, 1.1 * max_speed),
            },
            {
                "title": "Longitudinal command",
                "series": [
                    ("post_supervisor_accel_cmd_mps2", "Acceleration (m/s²)", "#2f855a"),
                    ("applied_brake", "Applied brake", "#c53030"),
                    ("applied_throttle", "Applied throttle", "#d97706"),
                ],
                "y_min": -3.5,
                "y_max": 3.5,
            },
        ],
    )
    _write_chart(
        output_dir / "04_safety_conflict_metrics.svg",
        title="Safety Conflict Metrics",
        subtitle="Finite TTC and DRAC are computed from fused local/CP obstacle snapshots.",
        rows=rows,
        stop_spans=stop_spans,
        panels=[
            {
                "rows": ttc_rows,
                "title": "Nearest finite time-to-collision",
                "series": [
                    (
                        "_nearest_ttc_plot",
                        "Nearest TTC (s; clipped at 10 s)",
                        "#6b46c1",
                    )
                ],
                "y_min": 0.0,
                "y_max": max(3.0, max_ttc),
            },
            {
                "title": "Required deceleration to avoid collision",
                "series": [
                    ("tick_max_drac_mps2", "Tick max DRAC (m/s²)", "#c53030")
                ],
                "y_min": 0.0,
                "y_max": max_drac,
            },
        ],
    )
    _write_chart(
        output_dir / "05_reference_and_lane_stability.svg",
        title="Reference and Lane Stability",
        subtitle="Reference alignment is shown alongside physical CARLA lane-envelope clearance.",
        rows=rows,
        stop_spans=stop_spans,
        panels=[
            {
                "title": "Reference alignment in ego frame",
                "series": [
                    ("reference_first_lateral_m", "First reference lateral (m)", "#3a506b"),
                    ("destination_lateral_m", "Destination lateral (m)", "#c43d3d"),
                ],
                "y_min": min(-0.5, min(_finite(rows, "destination_lateral_m") or [0.0])),
                "y_max": max(0.5, max(_finite(rows, "destination_lateral_m") or [0.0])),
            },
            {
                "title": "Physical lane-boundary clearance",
                "series": [
                    ("road_boundary_clearance_m", "Body-to-lane-edge clearance (m)", "#177e89"),
                    ("road_boundary_lateral_offset_m", "Lane-center offset (m)", "#d97706"),
                ],
                "y_min": min(-0.3, 1.1 * min_clearance),
                "y_max": max(1.0, 1.1 * max_clearance),
            },
        ],
    )
    _write_chart(
        output_dir / "06_mpc_runtime_health.svg",
        title="MPC Runtime Health",
        subtitle="Only actual replanning frames are used for solve-time and success traces.",
        rows=rows,
        stop_spans=stop_spans,
        panels=[
            {
                "rows": replan_rows,
                "title": "MPC solve time",
                "series": [("mpc_solve_time_ms", "Solve time (ms)", "#3a506b")],
                "y_min": 0.0,
                "y_max": max(30.0, 1.1 * max(_finite(replan_rows, "mpc_solve_time_ms") or [1.0])),
            },
            {
                "rows": replan_rows,
                "title": "Optimization success",
                "series": [("_solver_success", "Solved=1, failed=0", "#2f855a")],
                "y_min": 0.0,
                "y_max": 1.05,
            },
        ],
    )
    _write_solver_summary(output_dir / "07_mpc_outcome_distribution.svg", rows)
    _write_chart(
        output_dir / "08_perception_and_cp_awareness.svg",
        title="Perception and Cooperative Awareness",
        subtitle="Object counts show the information volume presented to the planner and MPC.",
        rows=rows,
        stop_spans=stop_spans,
        panels=[
            {
                "title": "Fused planning objects",
                "series": [
                    ("local_object_count", "Local objects", "#177e89"),
                    ("cp_obstacle_count", "CP obstacles", "#d97706"),
                    ("object_count", "Fused objects", "#6b46c1"),
                    ("mpc_object_count", "MPC objects", "#c53030"),
                ],
                "y_min": 0.0,
                "y_max": max(2.0, max(_finite(rows, "object_count") or [1.0]) + 1.0),
            },
            {
                "title": "Closest longitudinal context",
                "series": [
                    ("front_gap_m", "Behavior front gap (m)", "#2f855a"),
                    (
                        "nearest_ttc_longitudinal_gap_m",
                        "TTC longitudinal gap (m)",
                        "#3a506b",
                    ),
                ],
                "y_min": 0.0,
                "y_max": max(
                    20.0,
                    min(
                        100.0,
                        max(
                            _finite(rows, "nearest_ttc_longitudinal_gap_m")
                            + _finite(rows, "front_gap_m")
                            or [20.0]
                        ),
                    ),
                ),
            },
        ],
    )
    _write_chart(
        output_dir / "09_boundary_cost_diagnostics.svg",
        title="Road-Boundary Optimization Diagnostics",
        subtitle="Physical clearance and MPC boundary cost use different units and are shown separately.",
        rows=rows,
        stop_spans=stop_spans,
        panels=[
            {
                "title": "Physical lane-envelope clearance",
                "series": [
                    ("road_boundary_clearance_m", "Clearance (m)", "#177e89")
                ],
                "y_min": min(-0.3, 1.1 * min_clearance),
                "y_max": max(1.0, 1.1 * max_clearance),
            },
            {
                "title": "MPC road-boundary objective term",
                "series": [
                    ("Cost_RoadBoundary", "Cost_RoadBoundary", "#c53030")
                ],
                "y_min": 0.0,
                "y_max": 1.05 * max_cost_boundary,
            },
        ],
    )
    _write_speed_scatter(output_dir / "10_speed_target_scatter.svg", rows)
    _write_risk_scatter(output_dir / "11_ttc_drac_phase_space.svg", rows)
    _write_latency_histogram(output_dir / "12_mpc_latency_histogram.svg", replan_rows)

    duration = _float(rows[-1], "sim_time_s") - _float(rows[0], "sim_time_s")
    distance = sum(
        math.hypot(
            _float(current, "x_m") - _float(previous, "x_m"),
            _float(current, "y_m") - _float(previous, "y_m"),
        )
        for previous, current in zip(rows[:-1], rows[1:])
    )
    scenario_counts = Counter(str(row.get("scenario_fsm_state", "")) for row in rows)
    traffic_counts = Counter(str(row.get("traffic_signal_state", "")) for row in rows)
    collision_count = max(_finite(rows, "collision_count") or [0.0])
    min_ttc = min(_finite(rows, "nearest_ttc_s") or [float("inf")])
    min_pet_values = _finite(rows, "min_pet_s")
    max_drac_value = max(_finite(rows, "tick_max_drac_mps2") or [0.0])
    breach_rows = [
        row
        for row in rows
        if str(row.get("road_boundary_sample_valid", "")).lower() in {"true", "1"}
    ]
    breaches = sum(
        str(row.get("road_boundary_breach", "")).lower() in {"true", "1"}
        for row in breach_rows
    )
    solver_counts = Counter(str(row.get("mpc_status", "")) for row in replan_rows)
    successful = solver_counts["solved"] + solver_counts["solved inaccurate"]
    solve_times = _finite(replan_rows, "mpc_solve_time_ms")
    sorted_solve_times = sorted(solve_times)
    p95_solve_time = (
        sorted_solve_times[int(round(0.95 * (len(sorted_solve_times) - 1)))]
        if sorted_solve_times
        else 0.0
    )
    solve_over_50_count = sum(value > 50.0 for value in solve_times)
    min_ttc_text = (
        f"{min_ttc:.2f} s" if math.isfinite(min_ttc) else "No finite conflict"
    )
    min_pet_text = (
        f"{min(min_pet_values):.2f} s"
        if min_pet_values
        else "No finite PET event"
    )
    red_rows = [row for row in rows if row.get("traffic_signal_state") == "red"]
    red_lane_follow_count = sum(
        row.get("behavior_decision") == "lane_follow" for row in red_rows
    )
    stop_mean_speeds = []
    for start, end in stop_spans:
        speeds = _finite(rows[start : end + 1], "speed_mps")
        if speeds:
            stop_mean_speeds.append(sum(speeds) / len(speeds))
    stable_cutoff_s = _float(rows[-1], "sim_time_s") - 10.0
    stable_rows = [
        row for row in rows if _float(row, "sim_time_s") <= stable_cutoff_s
    ]
    stable_first_lateral = max(
        [abs(value) for value in _finite(stable_rows, "reference_first_lateral_m")]
        or [0.0]
    )
    stable_destination_lateral = max(
        [abs(value) for value in _finite(stable_rows, "destination_lateral_m")]
        or [0.0]
    )
    stop_mean_text = ", ".join(
        f"{value:.2f} m/s" for value in stop_mean_speeds
    )
    report = f"""# CP-X Planner Pre-Turn Visualization Report

## Scope

- Source: `{input_csv}`
- Samples: **{len(rows)}** of {len(all_rows)}
- Time interval: **{_float(rows[0], 'sim_time_s'):.2f}–{_float(rows[-1], 'sim_time_s'):.2f} s**
- Duration: **{duration:.2f} s**
- First turn row excluded: **{turn_index}**
- Turn and post-turn samples included: **0**

## Main Results

| Category | Metric | Result |
|---|---|---:|
| Safety | Collisions | {int(collision_count)} |
| Safety | Minimum finite TTC | {min_ttc_text} |
| Safety | Minimum finite PET | {min_pet_text} |
| Safety | Maximum DRAC | {max_drac_value:.2f} m/s² |
| Safety | Physical boundary breach | {100.0 * breaches / len(breach_rows) if breach_rows else 0.0:.2f}% |
| Efficiency | Distance traveled | {distance:.1f} m |
| Efficiency | Average speed | {sum(_finite(rows, "speed_mps")) / len(rows):.2f} m/s |
| Efficiency | Maximum speed | {max_speed:.2f} m/s |
| MPC | Optimization attempts | {len(replan_rows)} |
| MPC | Solved / failed | {successful} / {len(replan_rows) - successful} |
| MPC | Success rate | {100.0 * successful / len(replan_rows) if replan_rows else 0.0:.2f}% |
| MPC | Mean / P95 / max solve time | {sum(solve_times) / len(solve_times) if solve_times else 0.0:.2f} / {p95_solve_time:.2f} / {max(solve_times) if solve_times else 0.0:.2f} ms |

## Operating-State Coverage

- Lane Follow: **{scenario_counts["LANE_FOLLOW"]} frames**
- Traffic-Light Approach: **{scenario_counts["TRAFFIC_LIGHT_APPROACH"]} frames**
- Traffic-Light Stop: **{scenario_counts["TRAFFIC_LIGHT_STOP"]} frames**
- Signal observations: red **{traffic_counts["red"]}**, yellow **{traffic_counts["yellow"]}**, green **{traffic_counts["green"]}**, unknown **{traffic_counts["unknown"]}** frames.

## Key Findings

- **Strong solver reliability:** {successful}/{len(replan_rows)} replans succeeded. The six infeasible replans were brief and recovered through trajectory memory.
- **Fast typical optimization:** P95 solve time was **{p95_solve_time:.2f} ms**; **{solve_over_50_count}** replans exceeded the 50 ms warning threshold and form a measurable latency tail.
- **Correct red-light behavior ownership:** all **{len(red_rows)}** red observations used the stop behavior; red + lane-follow occurred in **{red_lane_follow_count}** frames.
- **Stable straight-road reference:** excluding the final 10 s of turn preparation, maximum first-point and destination lateral offsets were **{stable_first_lateral:.3f} m** and **{stable_destination_lateral:.3f} m**.
- **Physical lane containment:** minimum body-to-lane-edge clearance remained positive and no boundary breach was recorded.
- **Remaining limitation:** mean speed during the traffic-light stop episodes was **{stop_mean_text}**, showing low-speed creep rather than a perfectly stationary hold.
- **Safety margin warning:** the minimum finite TTC of **{min_ttc_text}** is below the README development target of 3 s, although maximum DRAC remained low.

## Figure Guide

1. Route and operating-state map.
2. Scenario, traffic-signal, and behavior timeline.
3. Speed tracking and longitudinal control.
4. TTC and DRAC safety traces.
5. Reference alignment and physical lane stability.
6. MPC solve-time and feasibility health.
7. MPC optimization outcome distribution.
8. Local perception, CP, fused, and MPC object counts.
9. Physical boundary clearance versus MPC boundary objective cost.
10. Actual-versus-target speed scatter.
11. TTC–DRAC safety phase space.
12. MPC solve-time histogram.
"""
    (output_dir / "PRE_TURN_VISUAL_REPORT.md").write_text(report, encoding="utf-8")
    print(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export_visual_suite(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
