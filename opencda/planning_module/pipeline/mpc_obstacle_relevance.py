"""Split MPC obstacles into ``relevant`` / ``ignored`` before the QP.

Rationale: the MPC obstacle term (super-ellipsoid repulsive potential) is
non-convex and is re-linearized around a moving reference every tick. A
vehicle that is roughly abeam of the ego and stays in an adjacent lane can
never collide with a corridor-tracking ego, yet it keeps perturbing that
term's gradient/Hessian and drives tick-to-tick oscillation in the solved
input trajectory.

This module is a **pure function**: given the ego planned path as a
polyline and each obstacle's predicted track (or its current position when
no track is available), it drops obstacles whose entire predicted motion
stays outside a rectangle around the ego's forward path. It is deliberately
conservative -- anything ambiguous stays ``relevant``. The upstream
behavior / speed / TTC layers still see every obstacle; only the MPC
obstacle-cost set is filtered.

Not wired into the bridge yet.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Sequence, Tuple

Snapshot = Mapping[str, Any]
XY = Tuple[float, float]

# Behind this much (m) the ego MPC cannot act on the obstacle anyway.
_REAR_BAND_M = 5.0


def _f(m: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in m and m[k] is not None:
            try:
                return float(m[k])
            except (TypeError, ValueError):
                return default
    return default


def _obstacle_track_xy(snapshot: Snapshot) -> List[XY]:
    """Predicted (x, y) samples for one obstacle, or its current position."""

    for key in ("predicted_trajectory", "future_trajectory"):
        raw = snapshot.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            pts: List[XY] = []
            for st in raw:
                if isinstance(st, Mapping) and ("x" in st and "y" in st):
                    pts.append((float(st["x"]), float(st["y"])))
                elif isinstance(st, Sequence) and not isinstance(st, (str, bytes)) and len(st) >= 2:
                    pts.append((float(st[0]), float(st[1])))
            if pts:
                return pts
    return [(_f(snapshot, "x", "x_m"), _f(snapshot, "y", "y_m"))]


def _polyline_xy(reference_samples: Sequence[Any]) -> List[XY]:
    poly: List[XY] = []
    for s in list(reference_samples or []):
        if isinstance(s, Mapping):
            x = _f(s, "x_ref_m", "x", "x_m")
            y = _f(s, "y_ref_m", "y", "y_m")
            poly.append((x, y))
        elif isinstance(s, Sequence) and not isinstance(s, (str, bytes)) and len(s) >= 2:
            poly.append((float(s[0]), float(s[1])))
    return poly


def _point_to_polyline(px: float, py: float, poly: Sequence[XY]) -> Tuple[float, float]:
    """Return (perpendicular_distance_m, along_arc_length_m_from_poly_start)
    for the closest point of ``poly`` to ``(px, py)``."""

    best_perp = math.inf
    best_along = 0.0
    arc = 0.0
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        bx, by = poly[i + 1]
        seg_dx, seg_dy = bx - ax, by - ay
        seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len2 <= 1e-12:
            continue
        t = ((px - ax) * seg_dx + (py - ay) * seg_dy) / seg_len2
        t_clamped = min(1.0, max(0.0, t))
        cx = ax + t_clamped * seg_dx
        cy = ay + t_clamped * seg_dy
        perp = math.hypot(px - cx, py - cy)
        if perp < best_perp:
            best_perp = perp
            best_along = arc + t_clamped * math.sqrt(seg_len2)
        arc += math.sqrt(seg_len2)
    return best_perp, best_along


def split_relevant_mpc_obstacles(
    obstacle_snapshots: Sequence[Snapshot],
    reference_samples: Sequence[Any],
    *,
    lateral_gate_m: float = 3.0,
    longitudinal_gate_m: float = 40.0,
    keep_ids: Sequence[Any] = (),
) -> Tuple[List[dict], List[dict]]:
    """Partition obstacles into ``(relevant, ignored)`` for the MPC.

    An obstacle is IGNORED only if **every** point of its predicted track
    lies outside the rectangle ``|perp| < lateral_gate_m`` and
    ``-_REAR_BAND_M <= along <= longitudinal_gate_m`` measured on the ego
    reference polyline. With < 2 reference points, or on any parsing
    failure, nothing is filtered (all relevant).

    ``keep_ids`` are obstacle ids that must never be filtered (e.g. the id
    the speed layer selected as the front vehicle).
    """

    snapshots = [dict(s) for s in list(obstacle_snapshots or []) if isinstance(s, Mapping)]
    poly = _polyline_xy(reference_samples)
    if len(poly) < 2:
        return snapshots, []

    keep = {str(i) for i in list(keep_ids or [])}
    lat_gate = max(0.0, float(lateral_gate_m))
    lon_gate = max(0.0, float(longitudinal_gate_m))

    relevant: List[dict] = []
    ignored: List[dict] = []
    for snap in snapshots:
        oid = str(
            snap.get("vehicle_id", snap.get("id", snap.get("track_id", snap.get("object_id", ""))))
        )
        if oid and oid in keep:
            relevant.append(snap)
            continue
        enters_band = False
        for px, py in _obstacle_track_xy(snap):
            perp, along = _point_to_polyline(px, py, poly)
            if perp < lat_gate and -_REAR_BAND_M <= along <= lon_gate:
                enters_band = True
                break
        (relevant if enters_band else ignored).append(snap)
    return relevant, ignored
