"""Stage D: turn a Stage-C ``Corridor`` into linear inequalities for the MPC.

Kept independent of ``mpc.py``'s variable indexing: each row is returned in
the ego-origin plane as coefficients on ``(x_k, y_k)`` plus a
``[lower, upper]`` band. ``mpc.py`` maps stage ``k`` -> its state-variable
columns and attaches a slack column per row.

``corridor_rows`` produces one longitudinal band per stage:
      s_lo(k) - sigma <= t_k . (x_k, y_k) <= s_hi(k) + sigma
  where ``t_k`` is the unit tangent of the ego reference at stage ``k``'s
  linearization station (so ``t_k . p`` is arc-length along the path,
  first-order).

Pure. No numpy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from opencda.planning_module.pipeline.mpc_obstacle_relevance import (
    _point_to_polyline,
    _polyline_xy,
)
from opencda.planning_module.pipeline.spatiotemporal_corridor import Corridor

XY = Tuple[float, float]
_BIG = 1.0e9


@dataclass(frozen=True)
class LinearRow:
    """``lower <= a_x * x_k + a_y * y_k <= upper`` at MPC stage ``k``,
    ego-origin frame. ``slack_group`` names which slack the QP shares for
    this family of rows."""

    stage: int
    a_x: float
    a_y: float
    lower: float
    upper: float
    slack_group: str
    tag: str = ""


def _tangent_at(poly: Sequence[XY], along_m: float, total_m: float) -> Tuple[float, float]:
    if len(poly) < 2 or total_m <= 1e-6:
        return (1.0, 0.0)
    frac = min(1.0, max(0.0, along_m / total_m))
    i = max(0, min(len(poly) - 2, int(frac * (len(poly) - 1))))
    dx = poly[i + 1][0] - poly[i][0]
    dy = poly[i + 1][1] - poly[i][1]
    n = math.hypot(dx, dy)
    return (1.0, 0.0) if n < 1e-9 else (dx / n, dy / n)


def _poly_len(poly: Sequence[XY]) -> float:
    return sum(
        math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
        for i in range(len(poly) - 1)
    )


def corridor_rows(
    corridor: Corridor,
    reference_samples: Sequence[Any],
    ego_origin_xy: XY = (0.0, 0.0),
    *,
    slack_group: str = "corridor",
    max_band_m: float = _BIG,
) -> List[LinearRow]:
    """One longitudinal band per stage with a finite s_lo or s_hi.

    ``ego_origin_xy`` is the world point the MPC frame is centred on
    (``mpc.py`` shifts everything to the current ego position). Rows are in
    that shifted frame: ``a . (x_k, y_k)`` is arc-length measured from the
    reference's start, minus the origin's own arc-length offset folded into
    the band.
    """

    poly_world = _polyline_xy(reference_samples)
    if len(poly_world) < 2:
        return []
    total = _poly_len(poly_world)
    ox, oy = float(ego_origin_xy[0]), float(ego_origin_xy[1])
    poly = [(x - ox, y - oy) for (x, y) in poly_world]
    origin_along = _point_to_polyline(0.0, 0.0, poly)[1]

    rows: List[LinearRow] = []
    n_stages = len(corridor.s_hi)
    for k in range(n_stages):
        s_lo = float(corridor.s_lo[k])
        s_hi = float(corridor.s_hi[k])
        if s_lo <= -_BIG and s_hi >= _BIG:
            continue
        # Linearization station (arc-length from the reference start): use
        # the tighter finite bound, else the ego's own station.
        anchor = origin_along
        if s_hi < _BIG:
            anchor = s_hi
        elif s_lo > -_BIG:
            anchor = s_lo
        tx, ty = _tangent_at(poly, anchor, total)
        # The row measures arc-length RELATIVE to the ego origin
        # (t . (x_k, y_k) ~= s(point) - s(ego_origin)); shift the band by the
        # origin's own station so the bound is on absolute arc-length.
        lo_rel = (
            -_BIG if s_lo <= -_BIG
            else max(s_lo, anchor - float(max_band_m)) - origin_along
        )
        hi_rel = (
            _BIG if s_hi >= _BIG
            else min(s_hi, anchor + float(max_band_m)) - origin_along
        )
        lower, upper = lo_rel, hi_rel
        rows.append(
            LinearRow(
                stage=k, a_x=tx, a_y=ty, lower=float(lower), upper=float(upper),
                slack_group=slack_group,
                tag=(corridor.binding[k] if k < len(corridor.binding) else ""),
            )
        )
    return rows
