"""Immutable, per-cycle AD-map view consumed by planning.

The global map and route are long-lived.  Planning modules instead consume one
rolling snapshot so ego lane identity, adjacency and route-target relation are
resolved exactly once per cycle and cannot be reinterpreted downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LocalLaneCorridor:
    offset: int
    lane_ids: tuple[int, ...] = ()


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

    def offset_for_lane(self, lane_id: int) -> int | None:
        return self.lane_to_offset.get(int(lane_id))

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
            "lane_to_offset": dict(self.lane_to_offset_items),
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
        }


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
    normalized_corridors = tuple(
        LocalLaneCorridor(
            offset=int(offset),
            lane_ids=tuple(sorted({int(value) for value in list(lane_ids or [])})),
        )
        for offset, lane_ids in sorted(
            ((int(key), value) for key, value in dict(graph.get("corridors", {}) or {}).items()),
            key=lambda item: item[0],
        )
    )
    lane_to_offset = tuple(sorted(
        (int(lane_id), int(offset))
        for lane_id, offset in dict(graph.get("lane_to_offset", {}) or {}).items()
    ))
    offset_lookup = dict(lane_to_offset)
    target_lane_id = int(route_target_lane_id or 0)
    target_in_frame = target_lane_id != 0 and target_lane_id in offset_lookup
    target_offset = int(offset_lookup.get(target_lane_id, 0))
    ego_lane_id = int(matched.get("ad_lane_id", graph.get("ego_ad_lane_id", 0)) or 0)
    violations = tuple(str(value) for value in invariant_violations if str(value))
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
        route_target_lane_id=target_lane_id,
        route_target_offset=target_offset,
        route_target_in_frame=bool(target_in_frame),
        cache_reused=bool(graph.get("cache_reused", False)),
        generation_reason=str(graph.get("generation_reason", "unavailable")),
        invariant_violations=violations,
    )
