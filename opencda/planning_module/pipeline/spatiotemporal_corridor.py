"""Stage C: build the per-stage longitudinal corridor from Stage-A tags and
Stage-B role assignments.

Turns the discrete conflict decisions into convex bounds the MPC can take
as linear constraints:

    s_lo(t_k) <= s_ego(t_k) <= s_hi(t_k)     for k = 0 .. N

* deterministic FOLLOW / LEAD_BRAKE -> no row; SpeedPlanner/IDM owns nominal
  car-following.
* prediction-mode FOLLOW / LEAD_BRAKE -> optional RSS upper bound used while
  reducing probabilistic futures to one MPC safety corridor.
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
    _polyline_xy,
    project_to_extended_polyline,
)
from opencda.planning_module.pipeline.rss import RSSParams, longitudinal_safe_distance

XY = Tuple[float, float]
_BIG = 1.0e9


@dataclass(frozen=True)
class CorridorParams:
    horizon_steps: int = 20
    dt_s: float = 0.1
    follow_extra_buffer_m: float = 2.0
    # Clearance measured forward from the ego front bumper. The longitudinal
    # crossing cap also subtracts ``ego_half_length_m`` because corridor
    # station represents the vehicle centre, not its front footprint.
    conflict_stop_buffer_m: float = 4.0
    crossing_clearance_time_s: float = 2.0   # cap s_hi within this of conflict_t
    # Arbitration is an efficiency agreement, not permission to collide. If
    # the peer's latest path puts it inside this imminent horizon, the safety
    # corridor overrides a stale ``proceed`` role.
    proceed_safety_override_ttc_s: float = 1.0
    ego_half_length_m: float = 2.45
    ego_half_width_m: float = 0.95
    # Magnitude of the MPC's own hard deceleration limit (mpc.yaml
    # constraints.min_acceleration_mps2). A cap tighter than what this lets
    # the ego reach by braking alone is not a geometry problem Stage D can
    # solve honestly: the longitudinal row only constrains the projection
    # onto the reference tangent, so an unreachable cap is a cheaper QP
    # solution via a heading rotation than via the (already maxed-out) brake
    # -- see build_longitudinal_corridor's kinematic floor below.
    max_braking_mps2: float = 3.0
    # Must match MPCConstraints.max_jerk_mps3.  Stage C and the MPC must use
    # the same reachable set; assuming instantaneous full braking here makes
    # early corridor rows longitudinally impossible under the MPC jerk bound.
    max_jerk_mps3: float = 10.0


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
    station_shift = 0.0
    if len(source_poly) >= 2 and len(current_poly) >= 2:
        ego_x, ego_y = float(current_ego_xy[0]), float(current_ego_xy[1])
        source_ego_s = project_to_extended_polyline(ego_x, ego_y, source_poly)[1]
        current_ego_s = project_to_extended_polyline(ego_x, ego_y, current_poly)[1]
        station_shift = float(current_ego_s - source_ego_s)
    # Missing geometry prevents station rebasing, not passage of time. The
    # original forecast must still age out rather than freezing indefinitely.
    stage_shift = max(0, int(
        float(age_s) / max(1.0e-3, float(dt_s)) + 1.0e-9
    ))
    n = len(corridor.s_hi)

    def shifted(values: Sequence[float], *, infinite_sign: int) -> List[float]:
        out = []
        for stage in range(n):
            source_stage = stage + stage_shift
            if source_stage >= n:
                # A bound published at the end of the old horizon has no
                # authority over stages that did not exist in that forecast.
                out.append(float(infinite_sign) * _BIG)
                continue
            value = float(values[source_stage])
            if abs(value) >= _BIG:
                out.append(float(infinite_sign) * _BIG)
            else:
                out.append(value + station_shift)
        return out

    binding = [
        (
            str(corridor.binding[stage + stage_shift])
            if stage + stage_shift < len(corridor.binding) else ""
        )
        for stage in range(n)
    ]
    rebased = Corridor(
        s_lo=shifted(corridor.s_lo, infinite_sign=-1),
        s_hi=shifted(corridor.s_hi, infinite_sign=1),
        binding=binding,
    )
    rebased.clamp_and_check()
    return rebased


def retain_pending_corridor(
    current: Corridor, previous: Optional[Corridor],
) -> Corridor:
    """Keep previously published bounds until their stages age out.

    ``previous`` must already be time/station rebased to the current tick.
    A fresh prediction may tighten a bound immediately, but an open or looser
    refresh cannot revoke a still-future constraint. The caller first removes
    bounds whose owners have been freshly observed as geometrically clear;
    missing observations keep the old bound until its forecast ages out.
    """

    if previous is None:
        return current
    # The caller caches ``current`` as the fresh prediction, whereas this
    # effective view also contains still-pending bounds from an older one.
    # Never mutate the fresh value: caching the merged view would republish
    # old bounds with a new origin time and prevent them from ever aging out.
    effective = Corridor(
        s_lo=list(current.s_lo), s_hi=list(current.s_hi),
        binding=list(current.binding), feasible=bool(current.feasible),
        first_infeasible_stage=current.first_infeasible_stage,
    )
    n = min(len(effective.s_hi), len(previous.s_hi))
    if n <= 0:
        return effective
    for k in range(n):
        previous_hi = float(previous.s_hi[k])
        if previous_hi < float(effective.s_hi[k]):
            effective.s_hi[k] = previous_hi
            effective.binding[k] = str(previous.binding[k])
        previous_lo = float(previous.s_lo[k])
        if previous_lo > float(effective.s_lo[k]):
            effective.s_lo[k] = previous_lo
    if not bool(previous.feasible):
        effective.feasible = False
        previous_stage = previous.first_infeasible_stage
        if previous_stage is not None and (
            effective.first_infeasible_stage is None
            or int(previous_stage) < int(effective.first_infeasible_stage)
        ):
            effective.first_infeasible_stage = int(previous_stage)
    effective.clamp_and_check()
    return effective


def remove_actor_bounds(
    corridor: Optional[Corridor], actor_ids: Sequence[str],
) -> Optional[Corridor]:
    """Return ``corridor`` without rows owned by the selected actors.

    Stage C uses this for two authoritative events: an actor was observed
    geometrically clear, or a newly built forecast supersedes that actor's
    previous forecast.  Bounds belonging to actors absent from the new input
    remain in the cache and age out normally.  The input is never mutated.
    """

    removed = {str(actor_id) for actor_id in actor_ids}
    if corridor is None or not removed:
        return corridor
    filtered = Corridor(
        s_lo=list(corridor.s_lo), s_hi=list(corridor.s_hi),
        binding=list(corridor.binding),
    )
    for k, owner in enumerate(filtered.binding):
        if str(owner).split("::mode", 1)[0] in removed:
            filtered.s_lo[k] = -_BIG
            filtered.s_hi[k] = _BIG
            filtered.binding[k] = ""
    filtered.clamp_and_check()
    return filtered


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

    # A per-mode corridor's own caps are already intersected with the ego's
    # physical upper reachability interval (build_longitudinal_corridor).
    # ``s_lo`` is intentionally not populated with the braking profile: that
    # profile describes feasibility of an upper safety cap, not a required
    # minimum advance by the ego vehicle.
    for corridor, _probability, _dangerous, _label in retained:
        if not corridor.feasible:
            out.feasible = False
            if corridor.first_infeasible_stage is not None and (
                out.first_infeasible_stage is None
                or corridor.first_infeasible_stage < out.first_infeasible_stage
            ):
                out.first_infeasible_stage = corridor.first_infeasible_stage

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


def _minimum_reachable_station_profile_m(
    *,
    v0_mps: float,
    current_acceleration_mps2: float,
    max_braking_mps2: float,
    max_jerk_mps3: float,
    horizon_steps: int,
    dt_s: float,
) -> List[float]:
    """Return the MPC-consistent minimum forward-station profile.

    The MPC uses forward Euler position dynamics (the current speed advances
    position before the stage acceleration changes the next speed) and limits
    ``a[k] - a[k-1]`` by ``max_jerk * dt``.  Reproduce those two contracts
    here so Stage C never publishes a cap that can only be met by turning away
    from the reference tangent.
    """

    steps = max(0, int(horizon_steps))
    dt = max(1.0e-6, float(dt_s))
    max_braking = max(1.0e-6, abs(float(max_braking_mps2)))
    jerk_step = max(0.0, abs(float(max_jerk_mps3))) * dt
    velocity = max(0.0, float(v0_mps))
    acceleration = max(-max_braking, float(current_acceleration_mps2))
    station = 0.0
    profile = [station]
    for _ in range(steps):
        acceleration = max(-max_braking, acceleration - jerk_step)
        station += velocity * dt
        velocity = max(0.0, velocity + acceleration * dt)
        profile.append(station)
    return profile


def _agent_station_series(
    agent_snapshot: Mapping[str, Any], poly: Sequence[XY], n: int
) -> List[float]:
    track = [(float(x), float(y)) for (x, y) in _obstacle_track_xy(agent_snapshot)]
    if not track:
        return []
    out: List[float] = []
    for k in range(n):
        px, py = track[min(k, len(track) - 1)]
        out.append(project_to_extended_polyline(px, py, poly)[1])
    return out


def _reference_heading_at_station(poly: Sequence[XY], station_m: float) -> float:
    """Return the local reference tangent at an arc-length station."""

    target = max(0.0, float(station_m))
    traversed = 0.0
    last_heading = 0.0
    for a, b in zip(poly[:-1], poly[1:]):
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        segment_length = math.hypot(dx, dy)
        if segment_length <= 1.0e-9:
            continue
        last_heading = math.atan2(dy, dx)
        if target <= traversed + segment_length:
            return last_heading
        traversed += segment_length
    return last_heading


def _projected_agent_half_extent_m(
    agent: Mapping[str, Any], poly: Sequence[XY], conflict_station_m: float,
) -> float:
    """Project an oriented agent rectangle onto the reference tangent.

    Stage-C station describes vehicle centres.  A crossing stop line must
    therefore reserve not only the ego front half-length but also the part of
    the other road user's body occupying the ego path.  Missing dimensions
    deliberately return zero for compatibility with legacy perception/V2X
    messages.
    """

    length_m = max(0.0, _f(agent, "length_m", "length", default=0.0))
    width_m = max(0.0, _f(agent, "width_m", "width", default=0.0))
    if length_m <= 0.0 or width_m <= 0.0:
        return 0.0
    agent_heading = _f(agent, "psi", "heading_rad", "yaw_rad", default=0.0)
    reference_heading = _reference_heading_at_station(poly, conflict_station_m)
    delta = agent_heading - reference_heading
    return (
        0.5 * length_m * abs(math.cos(delta))
        + 0.5 * width_m * abs(math.sin(delta))
    )


def make_gap_overlap_m(agent: Mapping[str, Any], p: "CorridorParams", rss: "RSSParams") -> float:
    """Lateral distance within which a make-gap peer's track opens the cap.

    Shared by ``build_longitudinal_corridor``'s own gating decision and by
    diagnostics that report how close a peer came to it, so the two can
    never drift apart.
    """

    return (
        0.5 * max(0.0, _f(
            agent, "width_m", "width", default=2.0 * float(p.ego_half_width_m),
        ))
        + max(0.0, float(p.ego_half_width_m))
        + max(0.0, float(rss.lateral_mu_m))
    )


def make_gap_gate_margin_m(
    *, agent: Mapping[str, Any], track: Sequence[XY], poly: Sequence[XY],
    p: "CorridorParams", rss: "RSSParams",
) -> Optional[float]:
    """How many more meters a make-gap peer's track must close to bind.

    Positive: the gate (see ``make_gap_overlap_m``) has not opened anywhere
    in ``track`` yet, by this many meters at closest approach.  Zero or
    negative: it has already opened (a corridor cap should be active).
    ``None`` when there is no track to evaluate.  Read-only: this never
    feeds back into the corridor itself, it only makes the same decision
    ``build_longitudinal_corridor`` already makes independently observable.
    """

    if len(poly) < 2 or not track:
        return None
    overlap_m = make_gap_overlap_m(agent, p, rss)
    closest_m = min(
        project_to_extended_polyline(float(x), float(y), poly)[0]
        for x, y in track
    )
    return float(closest_m - overlap_m)


def build_longitudinal_corridor(
    reference_samples: Sequence[Any],
    ego_snapshot: Mapping[str, Any],
    items: Sequence[Tuple[Mapping[str, Any], ConflictTag, Optional[ConflictAssignment]]],
    p: CorridorParams = CorridorParams(),
    rss: RSSParams = RSSParams(),
    *,
    constrain_follow: bool = False,
) -> Corridor:
    """Build one longitudinal safety envelope.

    ``items`` is ``(agent_snapshot, tag, assignment|None)`` per agent.
    ``constrain_follow`` is reserved for prediction-mode reduction.  The
    ordinary measured lead remains owned by SpeedPlanner/IDM; predicted
    futures need a time-indexed envelope or their probabilities cannot affect
    MPC at all.
    """

    n = max(1, int(p.horizon_steps))
    dt = max(1e-3, float(p.dt_s))
    poly = _polyline_xy(reference_samples)
    cor = Corridor(s_lo=[-_BIG] * (n + 1), s_hi=[_BIG] * (n + 1), binding=[""] * (n + 1))
    if len(poly) < 2:
        return cor

    ego_v = max(0.0, _f(ego_snapshot, "v", "speed", "speed_mps"))
    ego_a = _f(ego_snapshot, "a", "acceleration", "acceleration_mps2")
    ego_x = _f(ego_snapshot, "x", "x_m")
    ego_y = _f(ego_snapshot, "y", "y_m")
    # Per-stage floor: the least station reachable by braking at the MPC's
    # own hard limit, starting now. A cap below this is not a row Stage D
    # can honor honestly (see _minimum_reachable_station_profile_m); flooring
    # it there keeps every row solvable by slowing down, and flags the corridor
    # so a held warm-start solution is not reused across the transition.
    ego_s0 = project_to_extended_polyline(ego_x, ego_y, poly)[1]
    braking_floor = [
        ego_s0 + delta_s
        for delta_s in _minimum_reachable_station_profile_m(
            v0_mps=ego_v,
            current_acceleration_mps2=ego_a,
            max_braking_mps2=p.max_braking_mps2,
            max_jerk_mps3=p.max_jerk_mps3,
            horizon_steps=n,
            dt_s=dt,
        )
    ]

    # Ego's own unconstrained (constant-velocity) forward projection, used
    # only to test whether a currently-behind agent's predicted station has
    # overtaken ego by stage k -- the same nominal-progress approximation
    # cav_conflict_pipeline.py uses when reducing multimodal corridors.
    ego_nominal_station = [ego_s0 + ego_v * k * dt for k in range(n + 1)]

    def _cap(k: int, value: float, agent_id: str) -> None:
        floor = braking_floor[k]
        effective = max(float(value), floor)
        # ``floor`` is a reachability test for this safety *upper* bound. It
        # is not a lower control command: publishing it as s_lo forces the ego
        # to make at least the maximum-braking progress even when another
        # constraint or the plant response calls for more braking. Heading
        # and road-boundary ownership in Stage D prevent lateral escape.
        if effective < cor.s_hi[k]:
            cor.s_hi[k] = effective
            cor.binding[k] = agent_id
        if float(value) < floor - 1.0e-9:
            cor.feasible = False
            if cor.first_infeasible_stage is None or k < cor.first_infeasible_stage:
                cor.first_infeasible_stage = k

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
        # Longitudinal ordering belongs to this corridor's reference
        # coordinate.  A body-frame dot product disagrees with station near
        # turns and during lane-change heading transients, which can activate
        # a rear-agent cap before that agent actually passes ego's path.
        agent_starts_behind = bool(
            station and float(station[0]) < float(ego_s0) - 1.0e-6
        )

        # A claim decides priority, not physical occupancy.  An adjacent
        # peer whose broadcast path stays in its own lane must not be treated
        # as an already-merged lead (and must not demand an impossible RSS
        # gap at k=0).  The role remains latched; the longitudinal half-space
        # begins only when its predicted footprint enters ego's corridor.
        if role == "make_gap":
            if tag.reason == "conflicting_resource_claim":
                continue
            overlap_m = make_gap_overlap_m(agent, p, rss)
            track = list(_obstacle_track_xy(agent))
            for k in range(n + 1):
                if k < len(station) and k < len(track) and (
                    tag.tag in (FOLLOW, LEAD_BRAKE)
                    or project_to_extended_polyline(
                        float(track[k][0]), float(track[k][1]), poly
                    )[0] <= overlap_m
                ):
                    _cap(k, station[k] - gap, tag.agent_id)
            continue

        # Measured car-following remains a nominal SpeedPlanner/IDM concern.
        # During probabilistic mode reduction this branch instead creates the
        # time-indexed safety envelope that distinguishes a keep-lane future
        # from a braking future.  Stage D linearizes it on the executable
        # reference, so it no longer inherits the old lane-change-preview
        # tangent mismatch.
        if tag.tag in (FOLLOW, LEAD_BRAKE):
            if constrain_follow and not agent_starts_behind:
                for k in range(n + 1):
                    if k < len(station):
                        _cap(k, station[k] - gap, tag.agent_id)
            continue

        if tag.tag in (CUT_IN, MERGE):
            if tag.tag == MERGE:
                # Stage A may classify on a proposed lane-change curve. A
                # stationary peer then appears to "converge" merely because
                # ego's preview reference is moving underneath it. Stage C
                # owns physical occupancy on the executed reference: require
                # the *peer's* broadcast path to converge there before
                # creating a merge half-space. An already same-lane lead is
                # delegated to SpeedPlanner/IDM; a faster rear peer overtaking
                # ego's nominal progress still gets a bound.
                lateral = [
                    project_to_extended_polyline(float(x), float(y), poly)[0]
                    for x, y in _obstacle_track_xy(agent)
                ]
                if (
                    len(lateral) < 2
                    or (
                        lateral[-1] >= lateral[0] - 0.5
                        and not (
                            agent_starts_behind
                            and any(
                                station[k] > ego_nominal_station[k]
                                for k in range(min(len(station), n + 1))
                            )
                        )
                    )
                ):
                    continue
            # A cooperative cav that lost the arbitration (role proceed) is
            # expected to yield to ego, so ego takes no bound from it.
            if proceed and not imminent:
                continue
            start = 0
            if tag.conflict_t_s is not None:
                start = max(0, int(tag.conflict_t_s / dt))
            if agent_starts_behind:
                # A rear vehicle owns its approach to the lead vehicle:
                # capping ego behind a point that is itself still behind ego
                # reverses longitudinal ordering and lets the QP evade the
                # cap laterally instead of admitting it. But "starts behind"
                # is only true at k=0 -- a fast cut-in can still overtake
                # ego's own unconstrained path within the horizon, and from
                # the stage it does, it is exactly the lead vehicle this
                # branch exists to cap. Test each stage on its own station,
                # not the single snapshot at k=0.
                for k in range(start, n + 1):
                    if k < len(station) and station[k] > ego_nominal_station[k]:
                        _cap(k, station[k] - gap, tag.agent_id)
            else:
                for k in range(start, n + 1):
                    if k < len(station):
                        _cap(k, station[k] - gap, tag.agent_id)

        elif tag.tag in (CROSSING, ONCOMING):
            if ((proceed and not imminent)
                    or tag.conflict_s_m is None or tag.conflict_t_s is None):
                continue
            lo_k = max(0, int((tag.conflict_t_s - p.crossing_clearance_time_s) / dt))
            hi_k = min(n, int((tag.conflict_t_s + p.crossing_clearance_time_s) / dt))
            centre_stop_station = (
                float(tag.conflict_s_m)
                - max(0.0, float(p.conflict_stop_buffer_m))
                - max(0.0, float(p.ego_half_length_m))
                - _projected_agent_half_extent_m(
                    agent, poly, float(tag.conflict_s_m)
                )
            )
            for k in range(lo_k, hi_k + 1):
                _cap(k, centre_stop_station, tag.agent_id)

    cor.clamp_and_check()
    return cor
