"""Arc-length-parameterized route with typed segments.

Replaces the flat ``[(x, y, z, waypoint, road_option)]`` node list + monotonic
integer progress index + build-time gap bridging that the reference layer
currently reads through ``CPXRouteManager._route_nodes()``.

A ``RouteGeometry`` is built from that same node list, so it can be adopted one
consumer at a time:

    geom = RouteGeometry.from_entries(route_manager._route_entries)
    prog = geom.project(ego_x, ego_y, s_lower_m=prev_s_m)   # float arc length
    turn = geom.next_turn(prog.s_m, lookahead_m=40.0)       # ("right", 12.4) | None
    ref  = geom.sample(prog.s_m, count=20, spacing_m=1.0)   # speed-INDEPENDENT

Segment kinds:
  * ``lane_follow``   -- ordinary in-lane travel
  * ``lane_change``   -- a CHANGELANELEFT / CHANGELANERIGHT span (lateral maneuver)
  * ``junction_turn`` -- an explicit AD-map LEFT / RIGHT span
  * ``junction_connector`` -- AD-map intersection lane without a turn action
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from opencda.planning_module.pipeline.reference_geometry import (
    cumulative_arc_m,
    pose_at_arc,
    project_to_polyline,
    to_points,
)

LANE_FOLLOW = "lane_follow"
LANE_CHANGE = "lane_change"
JUNCTION_TURN = "junction_turn"
JUNCTION_CONNECTOR = "junction_connector"

_CHANGE_OPTIONS = {"CHANGELANELEFT", "CHANGELANERIGHT"}
_TURN_OPTIONS = {"LEFT", "RIGHT"}


@dataclass(frozen=True)
class RouteSegment:
    kind: str                    # lane_follow | lane_change | junction_turn | connector
    s_start_m: float
    s_end_m: float
    node_start: int              # inclusive index into the node list
    node_end: int                # inclusive
    direction: str = ""          # "left" | "right" for lane_change / junction_turn
    source_lane_id: int = 0
    target_lane_id: int = 0
    # Stable lateral identity of the corridor this segment runs in: 0 at the
    # route start, -1 per CHANGELANERIGHT, +1 per CHANGELANELEFT, unchanged
    # through lane_follow and junction_turn. Unlike the raw AD lane id it
    # does not churn where the map re-segments a lane it is not leaving.
    lane_index: int = 0

    @property
    def length_m(self) -> float:
        return max(0.0, self.s_end_m - self.s_start_m)


@dataclass(frozen=True)
class RouteProgress:
    s_m: float                   # arc length of the projected foot point
    lateral_m: float             # signed perpendicular offset (left positive)
    segment: Optional[RouteSegment]
    node_index: int              # nearest node at or before s_m
    off_route: bool              # lateral offset exceeds the stale threshold
    lane_index: int = 0          # stable lateral corridor identity at s_m


@dataclass(frozen=True)
class RoutePose:
    x_m: float
    y_m: float
    heading_rad: float
    lane_width_m: float
    lane_id: int
    road_option: str
    segment_kind: str


@dataclass(frozen=True)
class RouteDecision:
    """Pure topology decision at one route station."""

    optimal_lane_id: int
    current_road_option: str
    next_macro_maneuver: str
    next_macro_distance_m: float
    segment_kind: str
    lane_index: int


@dataclass(frozen=True)
class RouteTopologyValidation:
    valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    signature: Tuple[str, ...]


def _road_option_name(option: Any) -> str:
    if option is None:
        return ""
    name = getattr(option, "name", None)
    text = str(name if name is not None else option).strip()
    return text.rsplit(".", 1)[-1].upper() if "." in text else text.upper()


def _node_xy(entry: Any) -> Optional[Tuple[float, float, float, Any, str]]:
    """Normalize one entry to ``(x, y, z, waypoint, road_option_name)``.

    Accepts the ``_route_nodes()`` tuple shape and the
    ``(waypoint, road_option)`` entry shape.
    """
    if isinstance(entry, (list, tuple)):
        if len(entry) >= 5:
            wp = entry[3]
            return (float(entry[0]), float(entry[1]), float(entry[2]),
                    wp, _road_option_name(entry[4]))
        if len(entry) >= 1:
            wp = entry[0]
            option = _road_option_name(entry[1] if len(entry) >= 2 else "")
            loc = getattr(getattr(wp, "transform", None), "location", None)
            if loc is None:
                return None
            return (float(loc.x), float(loc.y), float(getattr(loc, "z", 0.0)),
                    wp, option)
    return None


class RouteGeometry:
    """Immutable per-route geometry. Rebuild on set_destination / replan."""

    def __init__(
        self,
        nodes: Sequence[Tuple[float, float, float, Any, str]],
        *,
        turn_min_heading_change_rad: float = math.radians(35.0),
        stale_lateral_m: float = 12.0,
        default_lane_width_m: float = 3.5,
    ) -> None:
        self._nodes = list(nodes)
        self._xy: List[Tuple[float, float]] = [(n[0], n[1]) for n in self._nodes]
        self._options: List[str] = [str(n[4] or "") for n in self._nodes]
        self._waypoints: List[Any] = [n[3] for n in self._nodes]
        self._cum_m: List[float] = (
            cumulative_arc_m(self._xy) if len(self._xy) >= 2 else [0.0] * len(self._xy)
        )
        self._turn_min_heading_change_rad = float(turn_min_heading_change_rad)
        self._stale_lateral_m = float(stale_lateral_m)
        self._default_lane_width_m = float(default_lane_width_m)
        self._segments: List[RouteSegment] = self._build_segments()

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_entries(cls, entries: Sequence[Any], **kwargs: Any) -> "RouteGeometry":
        nodes: List[Tuple[float, float, float, Any, str]] = []
        for entry in list(entries or []):
            node = _node_xy(entry)
            if node is None:
                continue
            if nodes and math.hypot(node[0] - nodes[-1][0], node[1] - nodes[-1][1]) < 1.0e-3:
                continue
            nodes.append(node)
        return cls(nodes, **kwargs)

    @property
    def valid(self) -> bool:
        return len(self._nodes) >= 2

    @property
    def total_m(self) -> float:
        return self._cum_m[-1] if self._cum_m else 0.0

    @property
    def segments(self) -> List[RouteSegment]:
        return list(self._segments)

    def _node_kind(self, index: int) -> Tuple[str, str]:
        """(kind, direction) from AD-map topology annotations only."""
        option = self._options[index]
        if option in _CHANGE_OPTIONS:
            return LANE_CHANGE, "left" if option == "CHANGELANELEFT" else "right"
        if option in _TURN_OPTIONS:
            return JUNCTION_TURN, option.lower()
        waypoint = self._waypoints[index]
        if self._waypoint_is_intersection(waypoint):
            # A straight intersection traversal is still connector topology,
            # but it is not a turn and must not trigger turn behavior.
            return JUNCTION_CONNECTOR, "straight"
        return LANE_FOLLOW, ""

    @staticmethod
    def _waypoint_is_intersection(waypoint: Any) -> bool:
        return bool(
            getattr(
                waypoint,
                "is_intersection",
                getattr(waypoint, "is_junction", False),
            )
        )

    def _lane_id(self, index: int) -> int:
        wp = self._waypoints[min(max(0, index), len(self._waypoints) - 1)]
        value = getattr(wp, "ad_lane_id", getattr(wp, "lane_id", 0))
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _build_segments(self) -> List[RouteSegment]:
        if len(self._nodes) < 2:
            return []
        raw = [self._node_kind(i) for i in range(len(self._nodes))]
        segments: List[RouteSegment] = []
        run_start = 0
        for i in range(1, len(raw) + 1):
            if i < len(raw) and raw[i] == raw[run_start]:
                continue
            kind = raw[run_start][0]
            direction = next((d for k, d in raw[run_start:i] if d), "")
            source_lane_id = self._lane_id(max(0, run_start - 1))
            target_lane_id = self._lane_id(min(len(raw) - 1, i))
            segments.append(RouteSegment(
                kind=kind,
                s_start_m=self._cum_m[run_start],
                s_end_m=self._cum_m[min(i - 1, len(self._cum_m) - 1)],
                node_start=run_start,
                node_end=i - 1,
                direction=direction,
                source_lane_id=source_lane_id,
                target_lane_id=target_lane_id,
            ))
            run_start = i
        # merge adjacent same-kind/same-direction segments the split above left
        merged: List[RouteSegment] = []
        for seg in segments:
            if (
                merged
                and merged[-1].kind == seg.kind
                and merged[-1].direction == seg.direction
            ):
                prev = merged.pop()
                seg = RouteSegment(seg.kind, prev.s_start_m, seg.s_end_m,
                                   prev.node_start, seg.node_end, seg.direction,
                                   prev.source_lane_id, seg.target_lane_id)
            merged.append(seg)
        # Stamp a stable lateral corridor index: it changes by exactly one
        # step across a lane_change segment (right -> -1, left -> +1, matching
        # the CARLA-frame sign used elsewhere) and is otherwise constant, so
        # "current lane identity" stays continuous even where the AD map
        # renumbers a lane the route never leaves.
        lane_index = 0
        stamped: List[RouteSegment] = []
        for seg in merged:
            entry_index = lane_index
            if seg.kind == LANE_CHANGE:
                if seg.direction == "right":
                    lane_index -= 1
                elif seg.direction == "left":
                    lane_index += 1
            stamped.append(RouteSegment(
                seg.kind, seg.s_start_m, seg.s_end_m, seg.node_start,
                seg.node_end, seg.direction, seg.source_lane_id,
                seg.target_lane_id, entry_index,
            ))
        return stamped

    # -- queries --------------------------------------------------------- #
    def _segment_at(self, s_m: float) -> Optional[RouteSegment]:
        for seg in self._segments:
            if seg.s_start_m - 1.0e-6 <= s_m <= seg.s_end_m + 1.0e-6:
                return seg
        return self._segments[-1] if self._segments else None

    def segment_at(self, s_m: float) -> Optional[RouteSegment]:
        """Return the typed topology segment containing route arc length."""

        return self._segment_at(float(s_m))

    def _node_index_at(self, s_m: float) -> int:
        lo, hi = 0, len(self._cum_m) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._cum_m[mid] <= s_m:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def project(
        self,
        x_m: float,
        y_m: float,
        *,
        s_lower_m: float = 0.0,
        s_upper_m: float | None = None,
    ) -> RouteProgress:
        if not self.valid:
            return RouteProgress(0.0, 0.0, None, 0, True)
        s_m, lateral_m = project_to_polyline(
            self._xy,
            x_m,
            y_m,
            s_lower_m=max(0.0, s_lower_m),
            s_upper_m=s_upper_m,
        )
        seg = self._segment_at(s_m)
        return RouteProgress(
            s_m=s_m,
            lateral_m=lateral_m,
            segment=seg,
            node_index=self._node_index_at(s_m),
            off_route=abs(lateral_m) > self._stale_lateral_m,
            lane_index=int(seg.lane_index) if seg is not None else 0,
        )

    def upcoming(self, s_m: float, lookahead_m: float) -> List[RouteSegment]:
        limit = s_m + max(0.0, lookahead_m)
        return [
            seg for seg in self._segments
            if seg.s_end_m >= s_m - 1.0e-6 and seg.s_start_m <= limit
        ]

    def next_segment_of(
        self,
        kind: str,
        s_m: float,
        lookahead_m: float,
        *,
        not_past: Tuple[str, ...] = (),
    ) -> Optional[Tuple[RouteSegment, float]]:
        """First segment of ``kind`` starting within ``lookahead_m``; (seg, dist_m).

        ``not_past`` names segment kinds that act as a barrier: if one of them
        starts ahead of ``s_m`` and before the first ``kind`` segment, the
        search stops with no hit. A lane change queued on the far side of a
        junction the vehicle has not turned through yet is not the *next*
        maneuver and must not be reported as one.
        """
        limit = s_m + max(0.0, lookahead_m)
        for seg in self._segments:
            if seg.s_end_m < s_m - 1.0e-6:
                continue
            if seg.s_start_m > limit:
                break
            if seg.kind == kind:
                return seg, max(0.0, seg.s_start_m - s_m)
            if seg.kind in not_past and seg.s_start_m > s_m + 1.0e-6:
                return None
        return None

    def _turn_zone_onset_s(self, turn_index: int) -> float:
        """Arc length where the turn *zone* begins, given the turn segment index.

        The sharp ``junction_turn`` run is usually only a few metres of a much
        longer physical intersection connector; ``_build_segments`` types the
        approach/exit of that connector as ``junction_connector``. Turn
        behaviour (speed taper, PREPARE_TURN) needs to engage at the connector
        entry, not at the sharp middle, so the zone onset is the start of any
        ``junction_connector`` run that sits directly before this turn.
        """
        onset_s = float(self._segments[turn_index].s_start_m)
        cursor = turn_index - 1
        while cursor >= 0 and self._segments[cursor].kind == JUNCTION_CONNECTOR:
            onset_s = float(self._segments[cursor].s_start_m)
            cursor -= 1
        return onset_s

    def turn_zone(
        self, s_m: float, lookahead_m: float
    ) -> Optional[Tuple[str, float, float, float]]:
        """``(direction, dist_to_onset_m, onset_s_m, zone_end_s_m)`` or ``None``.

        The zone spans the connector approach + the sharp turn + the connector
        exit -- the whole stretch over which the vehicle is inside the
        intersection for this maneuver. A turn is reported when its *zone
        onset* (connector entry) is within ``lookahead_m``, even if the sharp
        span itself is a little further.
        """
        limit = float(s_m) + max(0.0, float(lookahead_m))
        for index, seg in enumerate(self._segments):
            if seg.kind != JUNCTION_TURN:
                continue
            if seg.s_end_m < float(s_m) - 1.0e-6:
                continue
            onset_s = self._turn_zone_onset_s(index)
            if onset_s > limit:
                break
            zone_end_s = float(seg.s_end_m)
            cursor = index + 1
            while (
                cursor < len(self._segments)
                and self._segments[cursor].kind == JUNCTION_CONNECTOR
            ):
                zone_end_s = float(self._segments[cursor].s_end_m)
                cursor += 1
            return (
                seg.direction or "",
                max(0.0, onset_s - float(s_m)),
                onset_s,
                zone_end_s,
            )
        return None

    def next_turn(self, s_m: float, lookahead_m: float) -> Optional[Tuple[str, float]]:
        """``(direction, distance_m)`` to the next turn *zone* onset.

        Distance is measured to where the vehicle enters the intersection for
        the turn (the connector approach), not to the sharp middle -- see
        ``turn_zone``. A turn whose sharp span sits within ``lookahead_m`` is
        reported even when its connector onset is already behind ``s_m``
        (distance clamps to 0).
        """
        zone = self.turn_zone(s_m, lookahead_m)
        if zone is None:
            return None
        direction, dist_to_onset_m, _onset_s, _zone_end_s = zone
        return (direction, dist_to_onset_m)

    def next_lane_change(self, s_m: float, lookahead_m: float) -> Optional[Tuple[str, float]]:
        hit = self.next_segment_of(
            LANE_CHANGE, s_m, lookahead_m, not_past=(JUNCTION_TURN,)
        )
        if hit is None:
            return None
        seg, dist_m = hit
        return (seg.direction or "", dist_m)

    def next_lane_change_segment(
        self, s_m: float, lookahead_m: float
    ) -> Optional[RouteSegment]:
        """Explicit AD-map topology transition, including opaque lane IDs.

        A lane change beyond an intervening junction *turn* is not returned --
        the turn is the next maneuver in that case. A straight intersection
        traversal (``junction_connector`` with no turn) is not a maneuver and
        does not hide a lane change that follows it.
        """
        hit = self.next_segment_of(
            LANE_CHANGE, s_m, lookahead_m, not_past=(JUNCTION_TURN,)
        )
        return hit[0] if hit is not None else None

    def decision_at(
        self, s_m: float, *, current_lane_id: int = 0
    ) -> RouteDecision:
        """Derive lane/macro intent only from immutable route topology."""

        station_m = min(max(0.0, float(s_m)), float(self.total_m))
        segment = self._segment_at(station_m)
        pose = self.pose_at(station_m)
        current_option = str(pose.road_option or "LANEFOLLOW")
        if segment is not None:
            if segment.kind == LANE_CHANGE:
                current_option = (
                    "CHANGELANELEFT"
                    if segment.direction == "left"
                    else "CHANGELANERIGHT"
                )
            elif segment.kind == JUNCTION_TURN:
                current_option = str(segment.direction or "").upper()
            elif segment.kind == JUNCTION_CONNECTOR:
                current_option = "LANEFOLLOW"

        next_change = self.next_segment_of(
            LANE_CHANGE,
            station_m,
            max(0.0, self.total_m - station_m),
            not_past=(JUNCTION_TURN,),
        )
        turn = self.turn_zone(
            station_m, max(0.0, self.total_m - station_m)
        )
        candidates = []
        if next_change is not None:
            change_segment, distance_m = next_change
            candidates.append((
                float(distance_m),
                "Lane Change Left"
                if change_segment.direction == "left"
                else "Lane Change Right",
                int(change_segment.target_lane_id),
            ))
        if turn is not None:
            direction, distance_m, _onset_s, _end_s = turn
            candidates.append((
                float(distance_m),
                "Turn Left" if direction == "left" else "Turn Right",
                0,
            ))
        candidates.sort(key=lambda item: item[0])
        if candidates:
            next_distance_m, next_macro, macro_target_lane_id = candidates[0]
        else:
            next_distance_m = float("inf")
            next_macro = "Continue Straight"
            macro_target_lane_id = 0

        optimal_lane_id = int(current_lane_id or pose.lane_id or 0)
        if segment is not None and segment.kind == LANE_CHANGE:
            optimal_lane_id = int(segment.target_lane_id or optimal_lane_id)
        elif macro_target_lane_id:
            optimal_lane_id = int(macro_target_lane_id)
        return RouteDecision(
            optimal_lane_id=int(optimal_lane_id),
            current_road_option=str(current_option),
            next_macro_maneuver=str(next_macro),
            next_macro_distance_m=float(next_distance_m),
            segment_kind=(segment.kind if segment is not None else LANE_FOLLOW),
            lane_index=(int(segment.lane_index) if segment is not None else 0),
        )

    def waypoint_for_lane_id(self, lane_id: int, *, node_start: int = 0) -> Any:
        """First route waypoint carrying an exact AD lane ID; no XY inference."""
        for index in range(max(0, int(node_start)), len(self._waypoints)):
            if self._lane_id(index) == int(lane_id):
                return self._waypoints[index]
        return None

    def remaining_m(self, s_m: float) -> float:
        return max(0.0, self.total_m - float(s_m))

    def pose_at(self, s_m: float) -> RoutePose:
        x, y, heading = pose_at_arc(self._xy, s_m)
        idx = self._node_index_at(s_m)
        wp = self._waypoints[idx] if idx < len(self._waypoints) else None
        lane_width = self._default_lane_width_m
        for attr in ("lane_width_m", "lane_width"):
            v = getattr(wp, attr, None)
            if v:
                try:
                    lane_width = max(0.1, float(v))
                    break
                except (TypeError, ValueError):
                    pass
        lane_id = 0
        for attr in ("ad_lane_id", "lane_id"):
            v = getattr(wp, attr, None)
            if v is not None:
                try:
                    lane_id = int(v)
                    break
                except (TypeError, ValueError):
                    pass
        seg = self._segment_at(s_m)
        return RoutePose(
            x_m=x, y_m=y, heading_rad=heading,
            lane_width_m=lane_width, lane_id=lane_id,
            road_option=self._options[idx] if idx < len(self._options) else "",
            segment_kind=seg.kind if seg else LANE_FOLLOW,
        )

    def sample(self, s0_m: float, count: int, spacing_m: float) -> List[RoutePose]:
        """``count`` poses forward from ``s0_m`` at fixed ``spacing_m``.

        Spacing is a real arc length -- it does NOT depend on ego speed, which
        is what let the old ``step_distance_m = dt * speed`` sampler explode at
        crawl. Past the route end, poses extrapolate straight along the final
        heading.
        """
        step = max(1.0e-3, float(spacing_m))
        out: List[RoutePose] = []
        end_x, end_y, end_h = pose_at_arc(self._xy, self.total_m)
        for k in range(1, max(1, int(count)) + 1):
            s = float(s0_m) + k * step
            if s <= self.total_m:
                out.append(self.pose_at(s))
            else:
                over = s - self.total_m
                out.append(RoutePose(
                    x_m=end_x + over * math.cos(end_h),
                    y_m=end_y + over * math.sin(end_h),
                    heading_rad=end_h,
                    lane_width_m=self._default_lane_width_m,
                    lane_id=(self.pose_at(self.total_m).lane_id),
                    road_option="LANEFOLLOW",
                    segment_kind=LANE_FOLLOW,
                ))
        return out

    # -- diagnostics ---------------------------------------------------- #
    def describe(self) -> str:
        parts = [
            f"{seg.kind}"
            + (f"({seg.direction})" if seg.direction else "")
            + f"[{seg.s_start_m:.0f}-{seg.s_end_m:.0f}m]"
            for seg in self._segments
        ]
        return f"RouteGeometry {self.total_m:.0f}m: " + " -> ".join(parts)

    def validate_topology(self) -> RouteTopologyValidation:
        """Validate only route structure; never judge behavior or feasibility."""

        errors: List[str] = []
        warnings: List[str] = []
        if not self.valid:
            errors.append("route_geometry_invalid")
        if self.valid and self.total_m <= 1.0e-6:
            errors.append("route_total_arc_length_nonpositive")

        previous_start_m = -float("inf")
        previous_end_m = -float("inf")
        signature: List[str] = []
        for index, segment in enumerate(self._segments):
            label = str(segment.kind)
            if segment.direction:
                label += f":{segment.direction}"
            signature.append(label)
            if segment.s_start_m + 1.0e-6 < previous_start_m:
                errors.append(f"segment_{index}_start_not_monotonic")
            if segment.s_end_m + 1.0e-6 < segment.s_start_m:
                errors.append(f"segment_{index}_negative_arc_span")
            if segment.s_start_m + 1.0e-6 < previous_end_m:
                errors.append(f"segment_{index}_overlaps_previous")
            allowed_directions = (
                {"left", "right", "straight"}
                if segment.kind in {JUNCTION_TURN, JUNCTION_CONNECTOR}
                else {"left", "right"}
            )
            if segment.kind in {LANE_CHANGE, JUNCTION_TURN, JUNCTION_CONNECTOR} and segment.direction not in allowed_directions:
                errors.append(f"segment_{index}_{segment.kind}_direction_missing")
            if segment.kind == LANE_CHANGE:
                if int(segment.source_lane_id) == 0 or int(segment.target_lane_id) == 0:
                    warnings.append(f"segment_{index}_lane_change_lane_id_missing")
                elif int(segment.source_lane_id) == int(segment.target_lane_id):
                    warnings.append(f"segment_{index}_lane_change_lane_id_unchanged")
            previous_start_m = float(segment.s_start_m)
            previous_end_m = float(segment.s_end_m)

        return RouteTopologyValidation(
            valid=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            signature=tuple(signature),
        )
