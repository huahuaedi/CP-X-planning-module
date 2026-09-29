"""Typed behavior-to-cooperation maneuver contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


_LANE_CHANGES = frozenset({"lane_change_left", "lane_change_right"})


@dataclass(frozen=True)
class CooperativeManeuverProposal:
    maneuver: str
    source_corridor_id: int
    target_corridor_id: int
    requested: bool
    committed: bool = False
    committed_at_s: float = 0.0
    route_required: bool = False
    reason: str = ""
    station_corridor_id: int = 0
    s_begin_m: Optional[float] = None
    s_end_m: Optional[float] = None

    @classmethod
    def from_behavior(
        cls, *, maneuver: str, source_corridor_id: int,
        target_corridor_id: int, route_required: bool,
        maneuver_active: bool, committed_at_s: float, reason: str = "",
    ) -> "CooperativeManeuverProposal":
        normalized = str(maneuver).strip().lower()
        source_id = int(source_corridor_id)
        target_id = int(target_corridor_id)
        requested = bool(
            normalized in _LANE_CHANGES
            and source_id != 0
            and target_id != 0
            and source_id != target_id
        )
        return cls(
            maneuver=normalized,
            source_corridor_id=source_id,
            target_corridor_id=target_id,
            requested=bool(requested or maneuver_active),
            committed=bool(maneuver_active),
            committed_at_s=float(committed_at_s),
            route_required=bool(route_required),
            reason=str(reason),
        )

    def with_station_interval(
        self, *, corridor_id: int, s_begin_m: Optional[float],
        s_end_m: Optional[float],
    ) -> "CooperativeManeuverProposal":
        return replace(
            self,
            station_corridor_id=int(corridor_id),
            s_begin_m=s_begin_m,
            s_end_m=s_end_m,
        )

    def with_commitment(
        self, *, maneuver: str, target_corridor_id: int,
        committed_at_s: float,
    ) -> "CooperativeManeuverProposal":
        """Attach the authoritative maneuver lifecycle without re-deciding it."""

        return replace(
            self,
            maneuver=str(maneuver).strip().lower(),
            target_corridor_id=int(target_corridor_id),
            requested=True,
            committed=True,
            committed_at_s=float(committed_at_s),
        )
