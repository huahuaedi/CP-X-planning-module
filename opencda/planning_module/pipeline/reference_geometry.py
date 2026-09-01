"""Shared polyline geometry primitives for the reference / route layer.

One curvature estimator and one arc-length resampler, used everywhere a
reference or route polyline is measured or re-spaced. Both are **independent of
the input sample spacing** -- the previous adjacent-sample ``d(theta)/ds`` and
index-based smoothing broke whenever route sampling density changed (commit
9364563: 3 m -> 1 m).

All functions accept a polyline as either ``[(x, y), ...]`` or a sequence of
mappings carrying ``x_ref_m`` / ``y_ref_m`` (falling back to ``x`` / ``y``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

# Heading change is always measured across at least this much real arc, so a
# fixed kink reads the same curvature regardless of how many samples span it.
CURVATURE_EVAL_ARC_M = 1.5

Point = Tuple[float, float]


@dataclass(frozen=True)
class ReferenceLinePoint:
    """One point on a metric, arc-length parameterized reference line."""

    s_m: float
    x_m: float
    y_m: float
    heading_rad: float
    curvature_1pm: float
    lane_id: int = 0
    lane_width_m: float = 3.5


@dataclass(frozen=True)
class ReferenceLine:
    """Geometry consumed by Frenet planning; it is not a route/topology object."""

    points: Tuple[ReferenceLinePoint, ...]

    @property
    def valid(self) -> bool:
        return len(self.points) >= 2 and self.points[-1].s_m > 0.0

    @property
    def length_m(self) -> float:
        return self.points[-1].s_m if self.points else 0.0


def _metadata(sample: object, key: str, default: Any) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _unwrap_headings(headings: Sequence[float]) -> List[float]:
    if not headings:
        return []
    out = [float(headings[0])]
    for raw in headings[1:]:
        out.append(out[-1] + _wrap(float(raw) - out[-1]))
    return out


def build_reference_line(
    samples: Sequence[object],
    *,
    spacing_m: float = 0.5,
    default_lane_id: int = 0,
    default_lane_width_m: float = 3.5,
) -> ReferenceLine:
    """Build a stable line from lane-centre geometry.

    Samples are de-duplicated and uniformly re-sampled by *real* XY arc
    length. Heading uses centred metric differences and curvature is
    ``d(unwrapped heading)/ds``; neither depends on the provider's point index
    or sampling density.
    """
    raw = list(samples or [])
    raw_xy = to_points(raw)
    clean: List[Point] = []
    for point in raw_xy:
        if not clean or math.hypot(point[0] - clean[-1][0], point[1] - clean[-1][1]) > 1.0e-4:
            clean.append(point)
    if len(clean) < 2:
        return ReferenceLine(tuple())
    xy = resample_polyline(clean, max(0.05, float(spacing_m)))
    arc = cumulative_arc_m(xy)
    headings: List[float] = []
    for i in range(len(xy)):
        a = xy[max(0, i - 1)]
        b = xy[min(len(xy) - 1, i + 1)]
        headings.append(math.atan2(b[1] - a[1], b[0] - a[0]))
    unwrapped = _unwrap_headings(headings)
    curvature: List[float] = []
    for i in range(len(xy)):
        i0, i1 = max(0, i - 1), min(len(xy) - 1, i + 1)
        ds = arc[i1] - arc[i0]
        curvature.append(0.0 if ds <= 1.0e-6 else (unwrapped[i1] - unwrapped[i0]) / ds)

    lane_id = int(_metadata(raw[0], "lane_id", default_lane_id) or default_lane_id)
    lane_width = float(_metadata(raw[0], "lane_width_m", default_lane_width_m) or default_lane_width_m)
    return ReferenceLine(tuple(
        ReferenceLinePoint(float(s), float(p[0]), float(p[1]), _wrap(float(h)), float(k), lane_id, lane_width)
        for s, p, h, k in zip(arc, xy, headings, curvature)
    ))


def frenet_lane_change_path(
    reference_line: ReferenceLine,
    *,
    lateral_offset_m: float,
    transition_length_m: float,
    target_lane_id: int,
    target_speed_mps: float,
    initial_offset_m: float = 0.0,
    target_reference_line: Optional[ReferenceLine] = None,
) -> List[dict]:
    """Generate a quintic Frenet ``d(s)`` path and convert it back to XY.

    The smoothstep has zero first and second lateral derivatives at both ends,
    which gives MPC a continuous position, heading and curvature reference.

    When ``target_reference_line`` is supplied, its geometry owns the path
    after the transition.  This matters near junctions: an adjacent AD-map
    lane is not generally a constant normal offset of the source lane.  The
    legacy constant-offset mode remains available for callers that do not
    have a target centreline.
    """
    if not reference_line.valid:
        return []
    length = max(0.1, float(transition_length_m))
    delta = float(lateral_offset_m) - float(initial_offset_m)
    xy: List[Point] = []
    progress: List[float] = []
    offsets: List[float] = []
    target_points = (
        tuple(target_reference_line.points)
        if target_reference_line is not None and target_reference_line.valid
        else tuple()
    )

    def target_at_s(s_m: float) -> Optional[ReferenceLinePoint]:
        if not target_points:
            return None
        if s_m <= target_points[0].s_m:
            return target_points[0]
        if s_m >= target_points[-1].s_m:
            return target_points[-1]
        lo, hi = 0, len(target_points) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if target_points[mid].s_m <= s_m:
                lo = mid
            else:
                hi = mid
        a, b = target_points[lo], target_points[hi]
        span = max(1.0e-9, b.s_m - a.s_m)
        ratio = min(1.0, max(0.0, (s_m - a.s_m) / span))
        return ReferenceLinePoint(
            s_m=float(s_m),
            x_m=a.x_m + ratio * (b.x_m - a.x_m),
            y_m=a.y_m + ratio * (b.y_m - a.y_m),
            heading_rad=_wrap(a.heading_rad + ratio * _wrap(b.heading_rad - a.heading_rad)),
            curvature_1pm=a.curvature_1pm + ratio * (b.curvature_1pm - a.curvature_1pm),
            lane_id=int(target_lane_id),
            lane_width_m=b.lane_width_m,
        )

    for point in reference_line.points:
        u = min(1.0, max(0.0, point.s_m / length))
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        target_point = target_at_s(point.s_m)
        if target_point is None:
            d_m = float(initial_offset_m) + delta * blend
            x_m = point.x_m - math.sin(point.heading_rad) * d_m
            y_m = point.y_m + math.cos(point.heading_rad) * d_m
        else:
            # Station-aligned source/target interpolation is Frenet d(s) in
            # the source frame, while retaining the target lane's real
            # heading and curvature once blend reaches one.
            dx = target_point.x_m - point.x_m
            dy = target_point.y_m - point.y_m
            d_m = -math.sin(point.heading_rad) * dx + math.cos(point.heading_rad) * dy
            d_m *= blend
            x_m = point.x_m + blend * dx
            y_m = point.y_m + blend * dy
        xy.append((x_m, y_m))
        progress.append(blend)
        offsets.append(d_m)
    shaped = build_reference_line(
        xy,
        spacing_m=max(0.05, reference_line.length_m / max(1, len(xy) - 1)),
        default_lane_id=int(target_lane_id),
        default_lane_width_m=reference_line.points[0].lane_width_m,
    )
    count = min(len(reference_line.points), len(shaped.points))
    out: List[dict] = []
    for i in range(count):
        point = shaped.points[i]
        p = progress[min(i, len(progress) - 1)]
        out.append({
            "x_ref_m": point.x_m, "y_ref_m": point.y_m,
            "x": point.x_m, "y": point.y_m,
            "heading_rad": point.heading_rad,
            "curvature_1pm": point.curvature_1pm,
            "s_ref_m": point.s_m,
            "progress_m": point.s_m,
            "frenet_d_m": offsets[min(i, len(offsets) - 1)],
            "lane_change_progress": p,
            "lane_transition_kind": "lateral_lane_change",
            "lane_id": int(target_lane_id if p >= 0.5 else reference_line.points[0].lane_id),
            "lane_width_m": point.lane_width_m,
            "corridor_center_x_m": point.x_m,
            "corridor_center_y_m": point.y_m,
            "corridor_heading_rad": point.heading_rad,
            "speed_ref_mps": max(0.0, float(target_speed_mps)),
            "v_ref_mps": max(0.0, float(target_speed_mps)),
            "speed_mps": max(0.0, float(target_speed_mps)),
        })
    return out


def _xy(sample: object) -> Point | None:
    """Pull (x, y) from a tuple/list or a mapping with x_ref_m/y_ref_m/x/y."""
    if isinstance(sample, Mapping):
        try:
            x = sample.get("x_ref_m", sample.get("x"))
            y = sample.get("y_ref_m", sample.get("y"))
            return (float(x), float(y))
        except (TypeError, ValueError):
            return None
    try:
        return (float(sample[0]), float(sample[1]))
    except (TypeError, ValueError, IndexError):
        return None


def to_points(polyline: Sequence[object]) -> List[Point]:
    """Normalize any accepted polyline form to ``[(x, y), ...]``."""
    out: List[Point] = []
    for sample in list(polyline or []):
        point = _xy(sample)
        if point is not None:
            out.append(point)
    return out


def cumulative_arc_m(points: Sequence[Point]) -> List[float]:
    """Cumulative arc length; ``len == len(points)``, first entry 0.0."""
    cum = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return cum


def total_length_m(polyline: Sequence[object]) -> float:
    points = to_points(polyline)
    if len(points) < 2:
        return 0.0
    return cumulative_arc_m(points)[-1]


def _wrap(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _segment_headings(points: Sequence[Point]) -> Tuple[List[float], List[float]]:
    """Per-segment heading and length, skipping zero-length segments."""
    headings: List[float] = []
    lengths: List[float] = []
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy)
        if d <= 1.0e-6:
            continue
        headings.append(math.atan2(dy, dx))
        lengths.append(d)
    return headings, lengths


def _heading_at_arc(seg_headings: Sequence[float], seg_cum: Sequence[float], s: float) -> float:
    """Heading of the segment whose arc span contains ``s`` (clamped)."""
    s = min(max(s, 0.0), seg_cum[-1])
    lo, hi = 0, len(seg_headings) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if seg_cum[mid + 1] <= s:
            lo = mid + 1
        else:
            hi = mid
    return seg_headings[lo]


def curvature_profile_1pm(
    polyline: Sequence[object],
    *,
    eval_arc_m: float = CURVATURE_EVAL_ARC_M,
) -> List[float]:
    """Curvature (1/m) sampled every ~0.5 m, each over a centred ``eval_arc_m``
    heading window whose endpoints are located by *arc position*, so the value
    does not depend on the input's sample spacing.
    """
    points = to_points(polyline)
    seg_headings, seg_lengths = _segment_headings(points)
    if len(seg_headings) < 2:
        return []
    seg_cum = [0.0]
    for d in seg_lengths:
        seg_cum.append(seg_cum[-1] + d)
    total = seg_cum[-1]
    window_m = max(1.0e-3, float(eval_arc_m))
    half = 0.5 * window_m
    step = max(0.25, 0.5 * window_m / 3.0)
    profile: List[float] = []
    s = 0.0
    while s <= total + 1.0e-9:
        a = _heading_at_arc(seg_headings, seg_cum, s - half)
        b = _heading_at_arc(seg_headings, seg_cum, s + half)
        # span actually covered (shorter than window near the ends)
        span = min(s + half, total) - max(s - half, 0.0)
        profile.append(abs(_wrap(b - a)) / max(0.5 * window_m, span))
        s += step
    return profile


def max_curvature_1pm(
    polyline: Sequence[object],
    *,
    eval_arc_m: float = CURVATURE_EVAL_ARC_M,
) -> float:
    """Largest arc-windowed curvature anywhere on the polyline (0.0 if < 3 pts)."""
    profile = curvature_profile_1pm(polyline, eval_arc_m=eval_arc_m)
    return max(profile) if profile else 0.0


def resample_polyline(
    polyline: Sequence[object],
    spacing_m: float,
    *,
    keep_last: bool = True,
) -> List[Point]:
    """Uniform arc-length resample to ``spacing_m`` between points.

    The result's density does not depend on the input's -- feed it a route
    polyline at any sampling resolution and get back fixed-spacing points.
    """
    points = to_points(polyline)
    if len(points) < 2:
        return list(points)
    step = max(1.0e-3, float(spacing_m))
    cum = cumulative_arc_m(points)
    total = cum[-1]
    if total <= step:
        return [points[0], points[-1]]

    out: List[Point] = [points[0]]
    target = step
    seg = 0
    while target < total - 1.0e-9:
        while seg < len(cum) - 2 and cum[seg + 1] < target:
            seg += 1
        span = cum[seg + 1] - cum[seg]
        t = 0.0 if span <= 1.0e-9 else (target - cum[seg]) / span
        ax, ay = points[seg]
        bx, by = points[seg + 1]
        out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
        target += step
    if keep_last and math.hypot(points[-1][0] - out[-1][0], points[-1][1] - out[-1][1]) > 1.0e-6:
        out.append(points[-1])
    return out


def pose_at_arc(
    polyline: Sequence[object],
    s_m: float,
) -> Tuple[float, float, float]:
    """Interpolated ``(x, y, heading_rad)`` at arc length ``s_m`` (clamped)."""
    points = to_points(polyline)
    if not points:
        return (0.0, 0.0, 0.0)
    if len(points) == 1:
        return (points[0][0], points[0][1], 0.0)
    cum = cumulative_arc_m(points)
    total = cum[-1]
    s = min(max(0.0, float(s_m)), total)
    seg = 0
    while seg < len(cum) - 2 and cum[seg + 1] < s:
        seg += 1
    ax, ay = points[seg]
    bx, by = points[seg + 1]
    span = cum[seg + 1] - cum[seg]
    t = 0.0 if span <= 1.0e-9 else (s - cum[seg]) / span
    heading = math.atan2(by - ay, bx - ax)
    return (ax + (bx - ax) * t, ay + (by - ay) * t, heading)


def project_to_polyline(
    polyline: Sequence[object],
    x_m: float,
    y_m: float,
    *,
    s_lower_m: float = 0.0,
) -> Tuple[float, float]:
    """Project ``(x, y)`` onto the polyline.

    Returns ``(s_m, lateral_m)`` -- arc length of the foot point and the signed
    perpendicular offset (left of travel positive). ``s_lower_m`` restricts the
    search to arc lengths >= that value so route progress stays monotonic.
    """
    points = to_points(polyline)
    if len(points) < 2:
        return (0.0, 0.0)
    cum = cumulative_arc_m(points)
    best_s = float(s_lower_m)
    best_lat = 0.0
    best_d2 = float("inf")
    for i in range(len(points) - 1):
        if cum[i + 1] <= s_lower_m:
            continue
        ax, ay = points[i]
        bx, by = points[i + 1]
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 <= 1.0e-12:
            continue
        seg_len = math.hypot(dx, dy)
        t = ((x_m - ax) * dx + (y_m - ay) * dy) / seg2
        # keep the foot point at arc length >= s_lower_m (monotonic progress)
        t_min = max(0.0, (s_lower_m - cum[i]) / seg_len) if seg_len > 1.0e-9 else 0.0
        t = min(1.0, max(t_min, t))
        fx, fy = ax + dx * t, ay + dy * t
        d2 = (x_m - fx) ** 2 + (y_m - fy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_s = cum[i] + t * math.hypot(dx, dy)
            # signed lateral: cross product of segment dir and (point - foot)
            best_lat = (dx * (y_m - fy) - dy * (x_m - fx)) / math.hypot(dx, dy)
    return (best_s, best_lat)
