"""Stage C: build the per-stage longitudinal corridor from Stage-A tags and
Stage-B role assignments.

Turns the discrete conflict decisions into convex bounds the MPC can take
as linear constraints:

    s_lo(t_k) <= s_ego(t_k) <= s_hi(t_k)     for k = 0 .. N

* FOLLOW / LEAD_BRAKE            -> no row; SpeedPlanner/IDM is sole owner.
* CUT_IN                          -> s_hi capped a RSS gap behind the agent.
* CROSSING / ONCOMING, ego yields -> s_hi capped just short of the
  conflict point while the agent is near it.
* CROSSING / ONCOMING, ego proceeds -> no longitudinal upper bound; the
  Stage-D latched homotopy half-space still separates the trajectories.
* MERGE, ego opens the gap (role make_gap) -> s_hi capped a gap behind
  the cav's projected station (ego takes the slot behind).
* MERGE / anything, ego proceeds -> no bound.

Lateral motion is left to the road-boundary / spatial-envelope term and
Stage-D homotopy half-spaces; this module is longitudinal only.

Pure. Consumes ConflictTag (Stage A), optional ConflictAssignment
(Stage B), the agent's projected station over time, and RSSParams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from opencda.planning_module.pipeline.conflict_classifier import (
    CROSSING,
    CUT_IN,
    FOLLOW,
    LEAD_BRAKE,
    MERGE,
    ONCOMING,
    ConflictTag,
)
from opencda.planning_module.pipeline.cooperative_arbitration import ConflictAssignment
from opencda.planning_module.pipeline.mpc_obstacle_relevance import (
    _obstacle_track_xy,
    _point_to_polyline,
    _polyline_xy,
)
from opencda.planning_module.pipeline.rss import RSSParams, longitudinal_safe_distance

XY = Tuple[float, float]
_BIG = 1.0e9


@dataclass(frozen=True)
class CorridorParams:
    horizon_steps: int = 20
    dt_s: float = 0.1
    follow_extra_buffer_m: float = 2.0
    conflict_stop_buffer_m: float = 4.0
    crossing_clearance_time_s: float = 2.0   # cap s_hi within this of conflict_t
    ego_half_length_m: float = 2.45


@dataclass
class Corridor:
    s_lo: List[float]
    s_hi: List[float]
    binding: List[str] = field(default_factory=list)   # agent id capping each stage
    feasible: bool = True
    first_infeasible_stage: Optional[int] = None

    def clamp_and_check(self) -> None:
        for k in range(len(self.s_hi)):
            if self.s_lo[k] - self.s_hi[k] > 1e-6:
                self.feasible = False
                if self.first_infeasible_stage is None:
                    self.first_infeasible_stage = k


def _f(m: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in m and m[k] is not None:
            try:
                return float(m[k])
            except (TypeError, ValueError):
                return default
    return default


def _agent_station_series(
    agent_snapshot: Mapping[str, Any], poly: Sequence[XY], n: int
) -> List[float]:
    track = [(float(x), float(y)) for (x, y) in _obstacle_track_xy(agent_snapshot)]
    if not track:
        return []
    out: List[float] = []
    for k in range(n):
        px, py = track[min(k, len(track) - 1)]
        out.append(_point_to_polyline(px, py, poly)[1])
    return out


def build_longitudinal_corridor(
    reference_samples: Sequence[Any],
    ego_snapshot: Mapping[str, Any],
    items: Sequence[Tuple[Mapping[str, Any], ConflictTag, Optional[ConflictAssignment]]],
    p: CorridorParams = CorridorParams(),
    rss: RSSParams = RSSParams(),
) -> Corridor:
    """``items`` is ``(agent_snapshot, tag, assignment|None)`` per agent."""

    n = max(1, int(p.horizon_steps))
    dt = max(1e-3, float(p.dt_s))
    poly = _polyline_xy(reference_samples)
    cor = Corridor(s_lo=[-_BIG] * (n + 1), s_hi=[_BIG] * (n + 1), binding=[""] * (n + 1))
    if len(poly) < 2:
        return cor

    ego_v = max(0.0, _f(ego_snapshot, "v", "speed", "speed_mps"))

    def _cap(k: int, value: float, agent_id: str) -> None:
        if value < cor.s_hi[k]:
            cor.s_hi[k] = value
            cor.binding[k] = agent_id

    for agent, tag, assignment in list(items or []):
        role = str(getattr(assignment, "role", "") or "")
        proceed = role == "proceed"
        agent_v = max(0.0, _f(agent, "v", "speed", "speed_mps"))
        gap = longitudinal_safe_distance(ego_v, agent_v, rss) + p.follow_extra_buffer_m
        station = _agent_station_series(agent, poly, n + 1)

        # Ordinary car-following has exactly one longitudinal owner:
        # SpeedPlanner/IDM.  Mirroring FOLLOW/LEAD_BRAKE here used the same
        # peer a second time as a geometric QP bound.  On a lane-change
        # reference that tangent row coupled longitudinal following into the
        # lateral solution and pulled the vehicle away from lane centre.
        if tag.tag in (FOLLOW, LEAD_BRAKE):
            continue

        if tag.tag == CUT_IN:
            # A cooperative cav that lost the arbitration (role proceed) is
            # expected to yield to ego, so ego takes no bound from it.
            if proceed:
                continue
            start = 0
            if tag.conflict_t_s is not None:
                start = max(0, int(tag.conflict_t_s / dt))
            for k in range(start, n + 1):
                if k < len(station):
                    _cap(k, station[k] - gap, tag.agent_id)

        elif tag.tag in (CROSSING, ONCOMING):
            if proceed or tag.conflict_s_m is None or tag.conflict_t_s is None:
                continue
            lo_k = max(0, int((tag.conflict_t_s - p.crossing_clearance_time_s) / dt))
            hi_k = min(n, int((tag.conflict_t_s + p.crossing_clearance_time_s) / dt))
            for k in range(lo_k, hi_k + 1):
                _cap(k, float(tag.conflict_s_m) - p.conflict_stop_buffer_m, tag.agent_id)

        elif tag.tag == MERGE:
            if role != "make_gap":
                continue
            for k in range(n + 1):
                if k < len(station):
                    _cap(k, station[k] - gap, tag.agent_id)

    cor.clamp_and_check()
    return cor
