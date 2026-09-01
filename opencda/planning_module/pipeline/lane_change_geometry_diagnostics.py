"""Route-s/Frenet diagnostics for lane-change reference geometry."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _xy(sample: Mapping[str, object]) -> tuple[float, float]:
    return (
        float(sample.get("x_ref_m", sample.get("x", 0.0))),
        float(sample.get("y_ref_m", sample.get("y", 0.0))),
    )


def cumulative_s(samples: Sequence[Mapping[str, object]]) -> list[float]:
    points = [_xy(sample) for sample in samples]
    result = [0.0]
    for first, second in zip(points[:-1], points[1:]):
        result.append(
            result[-1]
            + math.hypot(second[0] - first[0], second[1] - first[1])
        )
    return result[: len(points)]


def project_to_route_s(
    reference: Sequence[Mapping[str, object]],
    samples: Sequence[Mapping[str, object]],
) -> list[dict[str, float]]:
    """Project samples onto one reference polyline and return common s/d."""

    ref = [_xy(sample) for sample in reference]
    if len(ref) < 2:
        return []
    ref_s = cumulative_s(reference)
    projected = []
    for sample in samples:
        px, py = _xy(sample)
        best = None
        for index, (first, second) in enumerate(zip(ref[:-1], ref[1:])):
            vx, vy = second[0] - first[0], second[1] - first[1]
            length_sq = vx * vx + vy * vy
            if length_sq <= 1.0e-12:
                continue
            ratio = min(
                1.0,
                max(0.0, ((px - first[0]) * vx + (py - first[1]) * vy) / length_sq),
            )
            qx, qy = first[0] + ratio * vx, first[1] + ratio * vy
            dx, dy = px - qx, py - qy
            distance_sq = dx * dx + dy * dy
            cross = vx * (py - qy) - vy * (px - qx)
            lateral_m = math.copysign(math.sqrt(distance_sq), cross) if distance_sq else 0.0
            segment_length = math.sqrt(length_sq)
            candidate = (
                distance_sq,
                float(ref_s[index]) + ratio * segment_length,
                lateral_m,
                math.atan2(vy, vx),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is not None:
            projected.append({
                "route_s_m": float(best[1]),
                "lateral_m": float(best[2]),
                "route_heading_rad": float(best[3]),
                "x_m": float(px),
                "y_m": float(py),
            })
    return projected


def build_lane_change_geometry_diagnostic(
    *,
    source_reference: Sequence[Mapping[str, object]],
    target_reference: Sequence[Mapping[str, object]],
    locked_reference: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return source, target and maneuver curves in one source route-s frame."""

    source = [dict(sample) for sample in source_reference]
    target = [dict(sample) for sample in target_reference]
    locked = [dict(sample) for sample in locked_reference]
    source_curve = project_to_route_s(source, source)
    target_curve = project_to_route_s(source, target)
    locked_curve = project_to_route_s(source, locked)
    station_mismatch = []
    source_s = cumulative_s(source)
    for index, row in enumerate(target_curve[: len(source_s)]):
        station_mismatch.append({
            "sample_index": int(index),
            "source_s_m": float(source_s[index]),
            "target_projected_s_m": float(row["route_s_m"]),
            "station_mismatch_m": float(row["route_s_m"]) - float(source_s[index]),
        })
    return {
        "coordinate_frame": "source_lane_route_s_frenet",
        "source_curve": source_curve,
        "target_curve": target_curve,
        "locked_curve": locked_curve,
        "source_target_station_mismatch": station_mismatch,
    }
