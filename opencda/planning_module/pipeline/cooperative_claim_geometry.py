"""Shared AD-map coordinates for cooperative resource claims."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from .local_map_snapshot import LocalMapSnapshot


@dataclass(frozen=True)
class ClaimStationInterval:
    corridor_id: int = 0
    s_begin_m: Optional[float] = None
    s_end_m: Optional[float] = None
    reason: str = "unavailable"

    @property
    def valid(self) -> bool:
        return bool(
            self.corridor_id
            and self.s_begin_m is not None
            and self.s_end_m is not None
        )


def project_claim_interval(
    *, local_map: LocalMapSnapshot, corridor_id: int,
    x_m: float, y_m: float, lookbehind_m: float, lookahead_m: float,
) -> ClaimStationInterval:
    """Project a pose onto one canonical AD-lane centerline.

    Every vehicle that names the same ``corridor_id`` uses the same complete
    AD-map polyline and therefore the same arc-length origin.  A missing or
    incomplete geometry yields no interval; arbitration then remains
    conservative rather than comparing unrelated route coordinates.
    """

    lane_id = int(corridor_id)
    geometry = local_map.geometry_for_lane(lane_id)
    if lane_id == 0 or geometry is None or len(geometry.centerline) < 2:
        return ClaimStationInterval(reason="target_corridor_geometry_missing")

    best_distance_sq = float("inf")
    best_station_m = 0.0
    points = geometry.centerline
    for first, second in zip(points[:-1], points[1:]):
        dx = float(second.x_m) - float(first.x_m)
        dy = float(second.y_m) - float(first.y_m)
        length_sq = dx * dx + dy * dy
        if length_sq <= 1.0e-12:
            continue
        ratio = max(0.0, min(1.0, (
            (float(x_m) - float(first.x_m)) * dx
            + (float(y_m) - float(first.y_m)) * dy
        ) / length_sq))
        projected_x = float(first.x_m) + ratio * dx
        projected_y = float(first.y_m) + ratio * dy
        distance_sq = (
            (float(x_m) - projected_x) ** 2
            + (float(y_m) - projected_y) ** 2
        )
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_station_m = float(first.s_m) + ratio * math.sqrt(length_sq)

    if not math.isfinite(best_distance_sq):
        return ClaimStationInterval(reason="target_corridor_projection_failed")
    return ClaimStationInterval(
        corridor_id=lane_id,
        s_begin_m=max(0.0, best_station_m - max(0.0, float(lookbehind_m))),
        s_end_m=min(
            float(geometry.length_m),
            best_station_m + max(0.0, float(lookahead_m)),
        ),
        reason="admap_target_corridor_projection",
    )
