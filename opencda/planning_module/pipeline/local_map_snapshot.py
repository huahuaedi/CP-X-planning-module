"""Immutable, per-cycle AD-map view consumed by planning.

The global map and route are long-lived.  Planning modules instead consume one
rolling snapshot so ego lane identity, adjacency and route-target relation are
resolved exactly once per cycle and cannot be reinterpreted downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LocalCenterlinePoint:
    s_m: float
    x_m: float
    y_m: float
    heading_rad: float
    curvature_1pm: float
    lane_width_m: float
    left_boundary_x_m: float
    left_boundary_y_m: float
    right_boundary_x_m: float
    right_boundary_y_m: float
    boundary_source: str


@dataclass(frozen=True)
class LocalLaneGeometry:
    lane_id: int
    centerline: tuple[LocalCenterlinePoint, ...] = ()

    @property
    def length_m(self) -> float:
        return float(self.centerline[-1].s_m) if self.centerline else 0.0


@dataclass(frozen=True)
class LocalLaneCorridor:
    offset: int
    lane_ids: tuple[int, ...] = ()
    lane_geometries: tuple[LocalLaneGeometry, ...] = ()

    def geometry_for_lane(self, lane_id: int) -> LocalLaneGeometry | None:
        return next(
            (
                geometry
                for geometry in self.lane_geometries
                if int(geometry.lane_id) == int(lane_id)
            ),
            None,
        )


@dataclass(frozen=True)
class LocalMapSnapshot:
    frame_id: int = 0
    timestamp_s: float = 0.0
    valid: bool = False
    ego_lane_id: int = 0
    road_id: int = 0
    section_id: int = 0
    lane_width_m: float = 0.0
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    heading_rad: float = 0.0
    lateral_offset_m: float = 0.0
    heading_error_rad: float = 0.0
    match_confidence: float = 0.0
    match_reason: str = "unmatched"
    forward_distance_m: float = 100.0
    backward_distance_m: float = 100.0
    corridors: tuple[LocalLaneCorridor, ...] = ()
    lane_to_offset_items: tuple[tuple[int, int], ...] = ()
    route_lane_sequence: tuple[int, ...] = ()
    route_target_lane_id: int = 0
    route_target_offset: int = 0
    route_target_in_frame: bool = False
    cache_reused: bool = False
    generation_reason: str = "unavailable"
    invariant_violations: tuple[str, ...] = ()

    @property
    def lane_to_offset(self) -> Mapping[int, int]:
        return MappingProxyType(dict(self.lane_to_offset_items))

    def corridor_lane_ids(self, offset: int) -> tuple[int, ...]:
        for corridor in self.corridors:
            if int(corridor.offset) == int(offset):
                return corridor.lane_ids
        return ()

    def lane_at_ego_station(self, offset: int) -> int | None:
        """Resolve a corridor offset to the physical lane beside ego.

        A corridor contains consecutive AD-map segment ids.  The route target
        may name a segment tens of metres ahead; lateral planning needs the
        segment whose centerline is closest to the current matched pose.
        """

        candidates = []
        for corridor in self.corridors:
            if int(corridor.offset) != int(offset):
                continue
            for geometry in corridor.lane_geometries:
                if not geometry.centerline:
                    continue
                distance_m = min(
                    math.hypot(
                        float(point.x_m) - float(self.center_x_m),
                        float(point.y_m) - float(self.center_y_m),
                    )
                    for point in geometry.centerline
                )
                candidates.append((float(distance_m), int(geometry.lane_id)))
        if not candidates:
            return None
        candidates.sort(key=lambda row: (float(row[0]), int(row[1])))
        return int(candidates[0][1])

    def offset_for_lane(self, lane_id: int) -> int | None:
        return self.lane_to_offset.get(int(lane_id))

    def geometry_for_lane(self, lane_id: int) -> LocalLaneGeometry | None:
        for corridor in self.corridors:
            geometry = corridor.geometry_for_lane(int(lane_id))
            if geometry is not None:
                return geometry
        return None

    def route_successor_lane(self, lane_id: int) -> int | None:
        """Return the next route lane without doing a new spatial match."""
        for index, route_lane_id in enumerate(self.route_lane_sequence[:-1]):
            if int(route_lane_id) == int(lane_id):
                return int(self.route_lane_sequence[index + 1])
        return None

    @property
    def route_lane_geometries(self) -> tuple[LocalLaneGeometry, ...]:
        """Topology-ordered route geometry owned by this immutable frame."""
        geometries = []
        for lane_id in self.route_lane_sequence:
            geometry = self.geometry_for_lane(int(lane_id))
            if geometry is not None:
                geometries.append(geometry)
        return tuple(geometries)

    def as_legacy_dict(self) -> dict[str, object]:
        """Compatibility export while legacy consumers migrate."""
        return {
            "frame_id": int(self.frame_id),
            "timestamp_s": float(self.timestamp_s),
            "ego_ad_lane_id": int(self.ego_lane_id),
            "forward_distance_m": float(self.forward_distance_m),
            "backward_distance_m": float(self.backward_distance_m),
            "corridors": {
                int(corridor.offset): list(corridor.lane_ids)
                for corridor in self.corridors
            },
            "corridor_geometry_lane_ids": {
                int(corridor.offset): [
                    int(geometry.lane_id)
                    for geometry in corridor.lane_geometries
                ]
                for corridor in self.corridors
            },
            "lane_to_offset": dict(self.lane_to_offset_items),
            "route_lane_sequence": list(self.route_lane_sequence),
            "route_target_ad_lane_id": int(self.route_target_lane_id),
            "route_target_offset": int(self.route_target_offset),
            "route_target_in_frame": bool(self.route_target_in_frame),
            "cache_reused": bool(self.cache_reused),
            "generation_reason": str(self.generation_reason),
            "invariant_violations": list(self.invariant_violations),
        }

    def trace_fields(self) -> dict[str, object]:
        return {
            "architecture_map_owner": "local_map_snapshot",
            "local_map_frame_id": int(self.frame_id),
            "local_map_timestamp_s": float(self.timestamp_s),
            "local_map_valid": bool(self.valid),
            "local_map_ego_lane_id": int(self.ego_lane_id),
            "local_map_route_target_lane_id": int(self.route_target_lane_id),
            "local_map_route_target_offset": int(self.route_target_offset),
            "local_map_route_target_in_frame": bool(self.route_target_in_frame),
            "local_map_cache_reused": bool(self.cache_reused),
            "local_map_invariant_violations": ";".join(self.invariant_violations),
            "local_map_centerline_lane_count": sum(
                len(corridor.lane_geometries) for corridor in self.corridors
            ),
            "local_map_admap_border_lane_count": sum(
                1
                for corridor in self.corridors
                for geometry in corridor.lane_geometries
                if geometry.centerline
                and all(
                    point.boundary_source == "admap_border"
                    for point in geometry.centerline
                )
            ),
            "local_map_route_lane_sequence": ";".join(
                str(lane_id) for lane_id in self.route_lane_sequence
            ),
        }


def _centerline_geometry(
    lane_id: int,
    samples: Sequence[Mapping[str, object]],
) -> LocalLaneGeometry:
    clean: list[tuple[float, float, float, object, object, object, object]] = []
    for sample in samples:
        try:
            x_m = float(sample.get("x_m", sample.get("x", 0.0)))
            y_m = float(sample.get("y_m", sample.get("y", 0.0)))
            lane_width_m = max(
                0.1,
                float(sample.get("lane_width_m", 3.5) or 3.5),
            )
            left_x = sample.get("left_boundary_x_m")
            left_y = sample.get("left_boundary_y_m")
            right_x = sample.get("right_boundary_x_m")
            right_y = sample.get("right_boundary_y_m")
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            continue
        if clean and math.hypot(x_m - clean[-1][0], y_m - clean[-1][1]) <= 1.0e-4:
            continue
        clean.append((x_m, y_m, lane_width_m, left_x, left_y, right_x, right_y))
    if len(clean) < 2:
        return LocalLaneGeometry(lane_id=int(lane_id))
    arc = [0.0]
    for first, second in zip(clean[:-1], clean[1:]):
        arc.append(
            arc[-1]
            + math.hypot(second[0] - first[0], second[1] - first[1])
        )
    headings: list[float] = []
    for index in range(len(clean)):
        first = clean[max(0, index - 1)]
        second = clean[min(len(clean) - 1, index + 1)]
        headings.append(math.atan2(second[1] - first[1], second[0] - first[0]))
    unwrapped = [headings[0]]
    for heading in headings[1:]:
        delta = math.atan2(
            math.sin(float(heading) - unwrapped[-1]),
            math.cos(float(heading) - unwrapped[-1]),
        )
        unwrapped.append(unwrapped[-1] + delta)
    points: list[LocalCenterlinePoint] = []
    for index, (
        (x_m, y_m, lane_width_m, left_x, left_y, right_x, right_y),
        s_m,
        heading_rad,
    ) in enumerate(
        zip(clean, arc, headings)
    ):
        first_index = max(0, index - 1)
        second_index = min(len(clean) - 1, index + 1)
        ds_m = arc[second_index] - arc[first_index]
        curvature_1pm = (
            0.0
            if ds_m <= 1.0e-6
            else (unwrapped[second_index] - unwrapped[first_index]) / ds_m
        )
        boundary_source = "admap_border"
        try:
            left_boundary_x_m = float(left_x)
            left_boundary_y_m = float(left_y)
            right_boundary_x_m = float(right_x)
            right_boundary_y_m = float(right_y)
            if not all(math.isfinite(value) for value in (
                left_boundary_x_m,
                left_boundary_y_m,
                right_boundary_x_m,
                right_boundary_y_m,
            )):
                raise ValueError("non-finite boundary")
        except (TypeError, ValueError):
            half_width_m = 0.5 * float(lane_width_m)
            # World headings use CARLA's left-handed XY convention.
            left_boundary_x_m = float(x_m) + half_width_m * math.sin(heading_rad)
            left_boundary_y_m = float(y_m) - half_width_m * math.cos(heading_rad)
            right_boundary_x_m = float(x_m) - half_width_m * math.sin(heading_rad)
            right_boundary_y_m = float(y_m) + half_width_m * math.cos(heading_rad)
            boundary_source = "centerline_width_fallback"
        points.append(LocalCenterlinePoint(
            s_m=float(s_m),
            x_m=float(x_m),
            y_m=float(y_m),
            heading_rad=float(heading_rad),
            curvature_1pm=float(curvature_1pm),
            lane_width_m=float(lane_width_m),
            left_boundary_x_m=float(left_boundary_x_m),
            left_boundary_y_m=float(left_boundary_y_m),
            right_boundary_x_m=float(right_boundary_x_m),
            right_boundary_y_m=float(right_boundary_y_m),
            boundary_source=str(boundary_source),
        ))
    return LocalLaneGeometry(
        lane_id=int(lane_id),
        centerline=tuple(points),
    )


def audit_local_map_rows(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Return M2 acceptance violations from a recorded planning run."""

    samples = [dict(row) for row in rows]
    violations: list[str] = []
    frame_ids = [int(row.get("local_map_frame_id", 0) or 0) for row in samples]
    if not frame_ids or any(
        second <= first for first, second in zip(frame_ids, frame_ids[1:])
    ):
        violations.append("local_map_frame_not_strictly_monotonic")
    if any(not bool(row.get("local_map_valid", False)) for row in samples):
        violations.append("local_map_invalid_frame")
    if any(not bool(row.get("map_match_valid", False)) for row in samples):
        violations.append("map_match_invalid_frame")
    if any(
        str(row.get("local_map_invariant_violations", "")).strip()
        or str(row.get("local_lane_frame_invariant_violations", "")).strip()
        for row in samples
    ):
        violations.append("local_map_contract_violation")

    # Reject short A->B->A identity oscillations. Longitudinal AD-map segment
    # transitions are expected, but a continuously matched lane must not
    # switch back within a few planning cycles at overlapping geometry.
    runs: list[tuple[int, int]] = []
    for row in samples:
        lane_id = int(row.get("local_map_ego_lane_id", 0) or 0)
        if runs and runs[-1][0] == lane_id:
            runs[-1] = (lane_id, runs[-1][1] + 1)
        else:
            runs.append((lane_id, 1))
    for first, middle, third in zip(runs, runs[1:], runs[2:]):
        if first[0] != 0 and first[0] == third[0] and middle[1] < 8:
            violations.append("map_match_lane_identity_flip_flop")
            break
    return tuple(violations)


def build_local_map_snapshot(
    *,
    frame_id: int,
    timestamp_s: float,
    match: Mapping[str, object] | None,
    local_graph: Mapping[str, object] | None,
    route_target_lane_id: int = 0,
    invariant_violations: Sequence[str] = (),
) -> LocalMapSnapshot:
    matched = dict(match or {})
    graph = dict(local_graph or {})
    raw_centerlines = {
        int(lane_id): list(samples or [])
        for lane_id, samples in dict(graph.get("lane_centerlines", {}) or {}).items()
    }
    # Resolve raw graph overlap once at the immutable boundary.  Longitudinal
    # and route expansion can list one AD lane in multiple lateral corridors;
    # ``lane_to_offset`` is the graph builder's canonical assignment.
    resolved_lane_offsets = {
        int(lane_id): int(offset)
        for lane_id, offset in dict(graph.get("lane_to_offset", {}) or {}).items()
    }
    normalized_corridor_values = []
    raw_corridors = sorted(
        (
            (int(key), value)
            for key, value in dict(graph.get("corridors", {}) or {}).items()
        ),
        key=lambda item: item[0],
    )
    for offset, lane_ids in raw_corridors:
        normalized_lane_ids = tuple(
            sorted({
                int(value)
                for value in list(lane_ids or [])
                if int(value) not in resolved_lane_offsets
                or int(resolved_lane_offsets[int(value)]) == int(offset)
            })
        )
        lane_geometries = []
        for lane_id in normalized_lane_ids:
            geometry = _centerline_geometry(
                int(lane_id), raw_centerlines.get(int(lane_id), [])
            )
            if geometry.centerline:
                lane_geometries.append(geometry)
        normalized_corridor_values.append(LocalLaneCorridor(
            offset=int(offset),
            lane_ids=normalized_lane_ids,
            lane_geometries=tuple(lane_geometries),
        ))
    normalized_corridors = tuple(normalized_corridor_values)
    lane_to_offset = tuple(sorted(resolved_lane_offsets.items()))
    offset_lookup = dict(lane_to_offset)
    route_lane_sequence: list[int] = []
    for raw_lane_id in list(graph.get("route_lane_sequence", []) or []):
        lane_id = int(raw_lane_id)
        if lane_id != 0 and (
            not route_lane_sequence or route_lane_sequence[-1] != lane_id
        ):
            route_lane_sequence.append(lane_id)
    target_lane_id = int(route_target_lane_id or 0)
    target_in_frame = target_lane_id != 0 and target_lane_id in offset_lookup
    target_offset = int(offset_lookup.get(target_lane_id, 0))
    ego_lane_id = int(matched.get("ad_lane_id", graph.get("ego_ad_lane_id", 0)) or 0)
    normalized_violations = [
        str(value) for value in invariant_violations if str(value)
    ]
    ego_geometry_available = any(
        corridor.geometry_for_lane(int(ego_lane_id)) is not None
        for corridor in normalized_corridors
    )
    if (
        bool(matched.get("valid", False))
        and int(ego_lane_id) != 0
        and not bool(ego_geometry_available)
    ):
        normalized_violations.append("ego_lane_centerline_missing")
    route_lanes_missing_from_frame = [
        lane_id for lane_id in route_lane_sequence if lane_id not in offset_lookup
    ]
    if route_lanes_missing_from_frame:
        normalized_violations.append("route_lane_missing_from_local_frame")
    route_lanes_missing_geometry = [
        lane_id
        for lane_id in route_lane_sequence
        if not any(
            corridor.geometry_for_lane(int(lane_id)) is not None
            for corridor in normalized_corridors
        )
    ]
    if route_lanes_missing_geometry:
        normalized_violations.append("route_lane_centerline_missing")
    violations = tuple(dict.fromkeys(normalized_violations))
    valid = bool(matched.get("valid", False)) and ego_lane_id != 0 and not violations
    return LocalMapSnapshot(
        frame_id=max(0, int(frame_id)),
        timestamp_s=float(timestamp_s),
        valid=bool(valid),
        ego_lane_id=ego_lane_id,
        road_id=int(matched.get("road_id", 0) or 0),
        section_id=int(matched.get("section_id", 0) or 0),
        lane_width_m=float(matched.get("lane_width_m", 0.0) or 0.0),
        center_x_m=float(matched.get("center_x_m", 0.0) or 0.0),
        center_y_m=float(matched.get("center_y_m", 0.0) or 0.0),
        heading_rad=float(matched.get("heading_rad", 0.0) or 0.0),
        lateral_offset_m=float(matched.get("lateral_offset_m", 0.0) or 0.0),
        heading_error_rad=float(matched.get("heading_error_rad", 0.0) or 0.0),
        match_confidence=float(matched.get("confidence", 0.0) or 0.0),
        match_reason=str(matched.get("match_reason", "unmatched")),
        forward_distance_m=float(graph.get("forward_distance_m", 100.0) or 0.0),
        backward_distance_m=float(graph.get("backward_distance_m", 100.0) or 0.0),
        corridors=normalized_corridors,
        lane_to_offset_items=lane_to_offset,
        route_lane_sequence=tuple(route_lane_sequence),
        route_target_lane_id=target_lane_id,
        route_target_offset=target_offset,
        route_target_in_frame=bool(target_in_frame),
        cache_reused=bool(graph.get("cache_reused", False)),
        generation_reason=str(graph.get("generation_reason", "unavailable")),
        invariant_violations=violations,
    )
