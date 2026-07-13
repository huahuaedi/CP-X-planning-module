"""CARLA-free lookahead calculation for the MPC local target."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _position(record: Mapping[str, object]):
    value = record.get("position", None)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _path_curvature(
    *,
    ego_state: Sequence[float],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    sample_distance_m: float,
) -> float:
    x_m, y_m = float(ego_state[0]), float(ego_state[1])
    heading = float(ego_state[3]) if len(ego_state) >= 4 else 0.0
    candidates = []
    for record in lane_center_waypoints:
        if int(record.get("lane_id", 0) or 0) != int(target_lane_id):
            continue
        position = _position(record)
        if position is None:
            continue
        dx, dy = position[0] - x_m, position[1] - y_m
        forward = math.cos(heading) * dx + math.sin(heading) * dy
        if forward < -1.0:
            continue
        candidates.append((forward, math.hypot(dx, dy), record))
    if len(candidates) < 3:
        return 0.0
    candidates.sort(key=lambda item: (float(item[0]), float(item[1])))

    sampled_headings = []
    next_distance = 0.0
    for forward, _distance, record in candidates:
        if float(forward) + 1.0e-6 < float(next_distance):
            continue
        sampled_headings.append((float(forward), float(record.get("heading_rad", heading))))
        next_distance = float(forward) + max(0.5, float(sample_distance_m))
        if len(sampled_headings) >= 8:
            break
    if len(sampled_headings) < 2:
        return 0.0
    curvature_values = []
    for first, second in zip(sampled_headings[:-1], sampled_headings[1:]):
        distance = max(1.0e-6, float(second[0]) - float(first[0]))
        curvature_values.append(abs(_wrap(float(second[1]) - float(first[1]))) / distance)
    return max(curvature_values) if curvature_values else 0.0


def compute_lane_lookahead_distance(
    *,
    ego_state: Sequence[float],
    lane_center_waypoints: Sequence[Mapping[str, object]],
    target_lane_id: int,
    local_goal_cfg: Mapping[str, object],
) -> float:
    """Return a speed- and curvature-based lookahead distance."""

    config = dict(local_goal_cfg or {})
    minimum = max(0.0, float(config.get("dynamic_lookahead_min_distance_m", 20.0)))
    if not bool(config.get("dynamic_lookahead_enabled", True)):
        return minimum
    maximum = max(minimum, float(config.get("dynamic_lookahead_max_distance_m", minimum)))
    speed_gain = float(config.get("dynamic_lookahead_speed_gain", 3.0))
    curvature_gain = float(config.get("dynamic_lookahead_curvature_gain", 20.0))
    spacing = max(
        0.5,
        float(config.get("dynamic_lookahead_curvature_sample_spacing_m", 5.0)),
    )
    speed = max(0.0, float(ego_state[2]) if len(ego_state) >= 3 else 0.0)
    curvature = _path_curvature(
        ego_state=ego_state,
        lane_center_waypoints=lane_center_waypoints,
        target_lane_id=int(target_lane_id),
        sample_distance_m=spacing,
    )
    distance = minimum + speed_gain * speed - curvature_gain * curvature
    return max(minimum, min(maximum, float(distance)))


__all__ = ["compute_lane_lookahead_distance"]
