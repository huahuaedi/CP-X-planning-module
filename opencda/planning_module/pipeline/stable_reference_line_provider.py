"""Arc-length stable reference windows for receding-horizon planning."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class StableReferenceWindow:
    samples: tuple[dict[str, object], ...]
    projection_s_m: float
    start_s_m: float
    reason: str


def _xy(sample: Mapping[str, object]) -> tuple[float, float]:
    return (float(sample.get("x_ref_m", sample.get("x", 0.0))),
            float(sample.get("y_ref_m", sample.get("y", 0.0))))


def _arc(samples: Sequence[Mapping[str, object]]) -> list[float]:
    arc = [0.0]
    for first, second in zip(samples[:-1], samples[1:]):
        ax, ay = _xy(first)
        bx, by = _xy(second)
        arc.append(arc[-1] + math.hypot(bx - ax, by - ay))
    return arc


def _project_s(samples, arc, *, x_m, y_m, lower_s_m):
    best_distance_sq = float("inf")
    best_s = max(0.0, float(lower_s_m))
    for index, (first, second) in enumerate(zip(samples[:-1], samples[1:])):
        if arc[index + 1] < lower_s_m - 1.0e-6:
            continue
        ax, ay = _xy(first)
        bx, by = _xy(second)
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        ratio = 0.0 if length_sq <= 1.0e-12 else (
            ((x_m - ax) * dx + (y_m - ay) * dy) / length_sq
        )
        ratio = min(1.0, max(0.0, ratio))
        station = arc[index] + ratio * (arc[index + 1] - arc[index])
        px, py = ax + ratio * dx, ay + ratio * dy
        if station + 1.0e-6 < lower_s_m:
            continue
        distance_sq = (x_m - px) ** 2 + (y_m - py) ** 2
        if distance_sq < best_distance_sq:
            best_distance_sq, best_s = distance_sq, station
    return float(best_s)


def _interpolate(first, second, ratio, station_m):
    ratio = min(1.0, max(0.0, float(ratio)))
    result = dict(first if ratio < 0.5 else second)
    ax, ay = _xy(first)
    bx, by = _xy(second)
    result["x_ref_m"] = result["x"] = ax + ratio * (bx - ax)
    result["y_ref_m"] = result["y"] = ay + ratio * (by - ay)
    for key in ("curvature_1pm", "speed_ref_mps", "v_ref_mps", "speed_mps",
                "lane_change_progress", "frenet_d_m", "lane_width_m"):
        try:
            a = float(first.get(key, second.get(key, 0.0)))
            b = float(second.get(key, a))
            result[key] = a + ratio * (b - a)
        except (TypeError, ValueError):
            pass
    try:
        a = float(first.get("heading_rad", 0.0))
        b = float(second.get("heading_rad", a))
        delta = math.atan2(math.sin(b - a), math.cos(b - a))
        result["heading_rad"] = math.atan2(
            math.sin(a + ratio * delta), math.cos(a + ratio * delta)
        )
    except (TypeError, ValueError):
        pass
    result["reference_global_s_m"] = float(station_m)
    result["s_ref_m"] = float(station_m)
    result["progress_m"] = float(station_m)
    return result


class StableReferenceLineProvider:
    """Extract monotonic interpolated windows from an immutable master line."""

    def window_from_reference(self, reference, *, ego_x_m, ego_y_m,
                              lower_s_m, first_forward_m, spacing_m, count):
        samples = [dict(sample) for sample in list(reference or [])]
        if len(samples) < 2 or int(count) <= 0:
            return StableReferenceWindow((), float(lower_s_m), float(lower_s_m),
                                         "reference_unavailable")
        arc = _arc(samples)
        projection_s = _project_s(
            samples, arc, x_m=float(ego_x_m), y_m=float(ego_y_m),
            lower_s_m=max(0.0, float(lower_s_m)),
        )
        start_s = min(arc[-1], max(float(lower_s_m), projection_s)
                      + max(0.0, float(first_forward_m)))
        stations = [min(arc[-1], start_s + index * max(0.05, float(spacing_m)))
                    for index in range(max(1, int(count)))]
        result = []
        segment = 0
        for station in stations:
            while segment + 1 < len(arc) - 1 and arc[segment + 1] < station:
                segment += 1
            span = max(1.0e-9, arc[segment + 1] - arc[segment])
            result.append(_interpolate(samples[segment], samples[segment + 1],
                                       (station - arc[segment]) / span, station))
        return StableReferenceWindow(tuple(result), projection_s, start_s,
                                     "arc_length_projection_stitched")
