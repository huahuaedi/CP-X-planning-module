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

import math
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
    # Arbitration is an efficiency agreement, not permission to collide. If
    # the peer's latest path puts it inside this imminent horizon, the safety
    # corridor overrides a stale ``proceed`` role.
    proceed_safety_override_ttc_s: float = 1.0
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


def rebase_corridor(
    corridor: Corridor, *, source_reference: Sequence[Any],
    current_reference: Sequence[Any], current_ego_xy: XY,
    age_s: float, dt_s: float,
) -> Corridor:
    """Advance a cached corridor in time and align its station coordinates."""

    source_poly = _polyline_xy(source_reference)
    current_poly = _polyline_xy(current_reference)
    if len(source_poly) < 2 or len(current_poly) < 2:
        return Corridor(
            s_lo=list(corridor.s_lo), s_hi=list(corridor.s_hi),
            binding=list(corridor.binding), feasible=bool(corridor.feasible),
            first_infeasible_stage=corridor.first_infeasible_stage,
        )
    ego_x, ego_y = float(current_ego_xy[0]), float(current_ego_xy[1])
    source_ego_s = _point_to_polyline(ego_x, ego_y, source_poly)[1]
    current_ego_s = _point_to_polyline(ego_x, ego_y, current_poly)[1]
    station_shift = float(current_ego_s - source_ego_s)
    stage_shift = max(0, int(float(age_s) / max(1.0e-3, float(dt_s))))
    n = len(corridor.s_hi)

    def shifted(values: Sequence[float], *, infinite_sign: int) -> List[float]:
        out = []
        for stage in range(n):
            source_stage = min(n - 1, stage + stage_shift)
            value = float(values[source_stage])
            if abs(value) >= _BIG:
                out.append(float(infinite_sign) * _BIG)
            else:
                out.append(value + station_shift)
        return out

    binding = [
        str(corridor.binding[min(n - 1, stage + stage_shift)])
        for stage in range(n)
    ]
    rebased = Corridor(
        s_lo=shifted(corridor.s_lo, infinite_sign=-1),
        s_hi=shifted(corridor.s_hi, infinite_sign=1),
        binding=binding,
    )
    rebased.clamp_and_check()
    return rebased


def aggregate_mode_corridors(
    mode_corridors: Sequence[Tuple[Corridor, float, bool, str]],
    nominal_s: Sequence[float],
) -> Corridor:
    """Reduce probabilistic mode corridors to one MPC corridor.

    Expected risk provides the normal bound.  A credible mode marked
    dangerous may veto that expectation and impose its tighter bound.  The
    result has exactly the same shape consumed by Stage D; modes therefore
    never masquerade as several simultaneous physical vehicles.

    Each item is ``(corridor, probability, dangerous_veto, mode_label)``.
    ``nominal_s`` is the free-progress upper bound used in place of infinity
    while taking the expectation.
    """

    n = len(list(nominal_s or ()))
    out = Corridor(s_lo=[-_BIG] * n, s_hi=[_BIG] * n, binding=[""] * n)
    retained = [item for item in mode_corridors if float(item[1]) > 0.0]
    total_probability = sum(float(item[1]) for item in retained)
    if n == 0 or total_probability <= 1.0e-9:
        return out

    for k in range(n):
        neutral = float(nominal_s[k])
        expected = 0.0
        veto_cap = _BIG
        veto_label = ""
        for corridor, probability, dangerous, label in retained:
            weight = float(probability) / total_probability
            cap = corridor.s_hi[k] if k < len(corridor.s_hi) else _BIG
            expected += weight * min(float(cap), neutral)
            if bool(dangerous) and float(cap) < veto_cap:
                veto_cap = float(cap)
                veto_label = str(label)
        cap = min(expected, veto_cap)
        # A bound at/above nominal progress is inactive and need not add a QP
        # row.  This preserves the exact clear-scene MPC behavior.
        if veto_cap < _BIG or cap < neutral - 1.0e-6:
            out.s_hi[k] = cap
            out.binding[k] = veto_label or "multimodal_expected"
    out.clamp_and_check()
    return out


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
    ego_x = _f(ego_snapshot, "x", "x_m")
    ego_y = _f(ego_snapshot, "y", "y_m")
    ego_heading = _f(ego_snapshot, "psi", "heading_rad", "yaw")
    ego_tx, ego_ty = math.cos(ego_heading), math.sin(ego_heading)

    def _cap(k: int, value: float, agent_id: str) -> None:
        if value < cor.s_hi[k]:
            cor.s_hi[k] = value
            cor.binding[k] = agent_id

    for agent, tag, assignment in list(items or []):
        role = str(getattr(assignment, "role", "") or "")
        proceed = role == "proceed"
        imminent = (
            tag.conflict_t_s is not None
            and float(tag.conflict_t_s) <= float(p.proceed_safety_override_ttc_s)
        )
        agent_v = max(0.0, _f(agent, "v", "speed", "speed_mps"))
        gap = longitudinal_safe_distance(ego_v, agent_v, rss) + p.follow_extra_buffer_m
        station = _agent_station_series(agent, poly, n + 1)
        agent_starts_behind = (
            (_f(agent, "x", "x_m") - ego_x) * ego_tx
            + (_f(agent, "y", "y_m") - ego_y) * ego_ty
        ) < 0.0

        # A negotiated make-gap role is not ordinary following: it is the
        # cooperative longitudinal contract and therefore owns a corridor
        # cap regardless of whether Stage A labelled the peer FOLLOW or
        # MERGE.  Apply it before the generic FOLLOW/IDM hand-off.
        if role == "make_gap":
            for k in range(n + 1):
                if k < len(station):
                    _cap(k, station[k] - gap, tag.agent_id)
            continue

        # Ordinary car-following has exactly one longitudinal owner:
        # SpeedPlanner/IDM.  Mirroring FOLLOW/LEAD_BRAKE here used the same
        # peer a second time as a geometric QP bound.  On a lane-change
        # reference that tangent row coupled longitudinal following into the
        # lateral solution and pulled the vehicle away from lane centre.
        if tag.tag in (FOLLOW, LEAD_BRAKE):
            continue

        if tag.tag == CUT_IN:
            # A rear vehicle owns its approach to the lead vehicle. Giving the
            # lead vehicle an upper bound behind that rear vehicle reverses
            # longitudinal ordering and lets the QP evade the cap laterally.
            if agent_starts_behind:
                continue
            # A cooperative cav that lost the arbitration (role proceed) is
            # expected to yield to ego, so ego takes no bound from it.
            if proceed and not imminent:
                continue
            start = 0
            if tag.conflict_t_s is not None:
                start = max(0, int(tag.conflict_t_s / dt))
            for k in range(start, n + 1):
                if k < len(station):
                    _cap(k, station[k] - gap, tag.agent_id)

        elif tag.tag in (CROSSING, ONCOMING):
            if ((proceed and not imminent)
                    or tag.conflict_s_m is None or tag.conflict_t_s is None):
                continue
            lo_k = max(0, int((tag.conflict_t_s - p.crossing_clearance_time_s) / dt))
            hi_k = min(n, int((tag.conflict_t_s + p.crossing_clearance_time_s) / dt))
            for k in range(lo_k, hi_k + 1):
                _cap(k, float(tag.conflict_s_m) - p.conflict_stop_buffer_m, tag.agent_id)

        elif tag.tag == MERGE:
            if agent_starts_behind:
                continue
            # Without a valid cooperative assignment, MERGE still needs a
            # safety owner.  Cap progress behind the predicted merge station;
            # a later CUT_IN classification will naturally continue the same
            # constraint.  Explicit proceed may skip it until safety becomes
            # imminent.
            if proceed and not imminent:
                continue
            start = 0
            if tag.conflict_t_s is not None:
                start = max(0, int(tag.conflict_t_s / dt))
            for k in range(start, n + 1):
                if k < len(station):
                    _cap(k, station[k] - gap, tag.agent_id)

    cor.clamp_and_check()
    return cor
