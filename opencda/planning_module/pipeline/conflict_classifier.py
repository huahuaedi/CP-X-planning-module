"""Stage A: classify each agent's spatiotemporal relation to the ego plan.

Consumes the prediction output (per-agent future tracks) + the ego
reference path, and tags every agent so the downstream stages can act
discretely instead of feeding one non-convex potential:

  IGNORE       never near the ego forward path over the horizon
  FOLLOW       same lane, ahead, ~parallel
  LEAD_BRAKE   FOLLOW and decelerating
  CUT_IN       adjacent lane now, predicted onto the ego path ahead
  MERGE        adjacent lane converging, ~parallel heading (negotiable)
  CROSSING     track crosses the ego path / near-perpendicular heading
  ONCOMING     opposing heading, closing head-on

Each tag carries the conflict point (ego arc-length + time) and the
minimum gap over the horizon, for Stage B (who yields) and Stage C
(corridor bounds). Pure; no dependency on the bridge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from opencda.planning_module.pipeline.mpc_obstacle_relevance import (
    _obstacle_track_xy,
    _point_to_polyline,
    _polyline_xy,
)

XY = Tuple[float, float]

IGNORE = "IGNORE"
FOLLOW = "FOLLOW"
LEAD_BRAKE = "LEAD_BRAKE"
CUT_IN = "CUT_IN"
MERGE = "MERGE"
CROSSING = "CROSSING"
ONCOMING = "ONCOMING"


@dataclass(frozen=True)
class ClassifierParams:
    horizon_steps: int = 20
    dt_s: float = 0.1
    lane_half_width_m: float = 1.9        # "on the ego path" band
    adjacent_lane_m: float = 3.5          # centre-to-centre to the next lane
    ignore_lateral_m: float = 3.0
    ignore_longitudinal_ahead_m: float = 45.0
    ignore_longitudinal_behind_m: float = 6.0
    crossing_heading_rad: float = math.radians(50.0)
    crossing_heading_hysteresis_rad: float = math.radians(8.0)
    oncoming_heading_rad: float = math.radians(130.0)
    decel_threshold_mps2: float = -0.8


@dataclass(frozen=True)
class ConflictTag:
    agent_id: str
    tag: str
    conflict_s_m: Optional[float]      # ego arc-length at closest approach
    conflict_t_s: Optional[float]
    min_gap_m: float                  # min longitudinal gap (agent_s - ego_s)
    min_lateral_m: float              # min |lateral offset| over the horizon
    cooperative: bool
    reason: str


def _f(m: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in m and m[k] is not None:
            try:
                return float(m[k])
            except (TypeError, ValueError):
                return default
    return default


def _agent_id(m: Mapping[str, Any]) -> str:
    return str(
        m.get("id", m.get("vehicle_id", m.get("track_id", m.get("actor_id", ""))))
    )


def _agent_heading(m: Mapping[str, Any]) -> float:
    if "psi" in m or "heading_rad" in m or "yaw" in m:
        return _f(m, "psi", "heading_rad", "yaw")
    vx, vy = _f(m, "vx"), _f(m, "vy")
    return math.atan2(vy, vx) if math.hypot(vx, vy) > 0.1 else 0.0


def _cv_track(m: Mapping[str, Any], n: int, dt: float) -> List[XY]:
    x0, y0 = _f(m, "x", "x_m"), _f(m, "y", "y_m")
    v = max(0.0, _f(m, "v", "speed", "speed_mps"))
    h = _agent_heading(m)
    return [(x0 + v * math.cos(h) * dt * k, y0 + v * math.sin(h) * dt * k) for k in range(n)]


def _ego_arc_at(ego_xy: XY, poly: Sequence[XY]) -> float:
    return _point_to_polyline(ego_xy[0], ego_xy[1], poly)[1]


def _poly_length(poly: Sequence[XY]) -> float:
    return sum(math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
              for i in range(len(poly) - 1))


def classify_conflicts(
    reference_samples: Sequence[Any],
    ego_snapshot: Mapping[str, Any],
    agent_snapshots: Sequence[Mapping[str, Any]],
    p: ClassifierParams = ClassifierParams(),
    previous_tags: Optional[Mapping[str, str]] = None,
) -> List[ConflictTag]:
    """Tag every agent. ``ego_snapshot`` needs x/y/v(/psi); each agent needs
    x/y/v(/psi) and optionally ``predicted_trajectory``."""

    poly = _polyline_xy(reference_samples)
    out: List[ConflictTag] = []
    if len(poly) < 2:
        for a in agent_snapshots or []:
            out.append(ConflictTag(_agent_id(a), FOLLOW, None, None, 0.0, 0.0,
                                   bool(a.get("cooperative", False)),
                                   "no_reference"))
        return out

    n, dt = max(1, int(p.horizon_steps)), max(1e-3, float(p.dt_s))
    ego_h = _agent_heading(ego_snapshot)
    ego_s0 = _ego_arc_at((_f(ego_snapshot, "x", "x_m"), _f(ego_snapshot, "y", "y_m")), poly)
    ego_v = max(0.0, _f(ego_snapshot, "v", "speed", "speed_mps"))
    poly_len = _poly_length(poly)

    for a in list(agent_snapshots or []):
        aid = _agent_id(a)
        previous_tag = str((previous_tags or {}).get(aid, ""))
        coop = bool(a.get("cooperative", False))
        track = [
            (float(x), float(y)) for (x, y) in _obstacle_track_xy(a)
        ]
        if len(track) < n:
            track = _cv_track(a, n, dt)
        a_h = _agent_heading(a)
        d_head = abs(math.atan2(math.sin(a_h - ego_h), math.cos(a_h - ego_h)))
        a_accel = _f(a, "a", "acceleration", "acceleration_mps2")

        min_lat = math.inf
        min_gap = math.inf
        conflict_s = conflict_t = None
        lat_series: List[float] = []
        signed_lat: List[float] = []
        for k in range(min(n, len(track))):
            px, py = track[k]
            perp, along = _point_to_polyline(px, py, poly)
            # signed lateral: + is left of the ego path direction at that point
            i = max(0, min(len(poly) - 2, int(along / max(1e-6, poly_len) * (len(poly) - 1))))
            seg_h = math.atan2(poly[i + 1][1] - poly[i][1], poly[i + 1][0] - poly[i][0])
            sgn = 1.0 if (math.cos(seg_h) * (py - poly[i][1]) - math.sin(seg_h) * (px - poly[i][0])) >= 0 else -1.0
            lat_series.append(perp)
            signed_lat.append(sgn * perp)
            ego_s_k = ego_s0 + ego_v * dt * k
            gap = along - ego_s_k
            if perp < min_lat:
                min_lat = perp
            if abs(gap) < abs(min_gap):
                min_gap = gap
                conflict_s, conflict_t = along, dt * k

        ever_in_ignore_box = any(
            perp < p.ignore_lateral_m for perp in lat_series
        ) and (conflict_s is None or
               -p.ignore_longitudinal_behind_m <= (conflict_s - ego_s0) <= p.ignore_longitudinal_ahead_m)

        # ----- classification -----
        if not ever_in_ignore_box:
            tag, reason = IGNORE, f"min_lat={min_lat:.1f}>=gate"
        elif d_head >= p.oncoming_heading_rad:
            tag, reason = ONCOMING, f"dhead={math.degrees(d_head):.0f}"
        elif d_head >= (
            p.crossing_heading_rad
            - p.crossing_heading_hysteresis_rad
            if previous_tag == CROSSING
            else p.crossing_heading_rad + p.crossing_heading_hysteresis_rad
        ):
            tag, reason = CROSSING, f"dhead={math.degrees(d_head):.0f}"
        elif len(signed_lat) >= 2 and signed_lat[0] * signed_lat[-1] < 0 and min_lat < p.lane_half_width_m:
            tag, reason = CROSSING, "path_crossed"
        else:
            entered = (
                len(lat_series) >= 2
                and lat_series[0] >= p.lane_half_width_m
                and min(lat_series) < p.lane_half_width_m
            )
            converging = (
                len(lat_series) >= 2 and lat_series[-1] < lat_series[0] - 0.5
                and lat_series[0] >= p.lane_half_width_m
            )
            if entered:
                tag, reason = CUT_IN, "adjacent->path"
            elif converging:
                tag, reason = MERGE, "adjacent_converging"
            elif min_lat < p.lane_half_width_m and (min_gap if min_gap != math.inf else 0.0) >= -1.0:
                tag = LEAD_BRAKE if a_accel <= p.decel_threshold_mps2 else FOLLOW
                reason = f"same_lane_ahead:a={a_accel:.1f}"
            else:
                tag, reason = IGNORE, "in_box_but_not_on_path"

        out.append(
            ConflictTag(
                agent_id=aid, tag=tag,
                conflict_s_m=(None if tag == IGNORE else conflict_s),
                conflict_t_s=(None if tag == IGNORE else conflict_t),
                min_gap_m=(0.0 if min_gap == math.inf else float(min_gap)),
                min_lateral_m=(0.0 if min_lat == math.inf else float(min_lat)),
                cooperative=coop, reason=reason,
            )
        )
    return out
