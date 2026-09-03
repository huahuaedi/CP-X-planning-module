"""Cross-CAV cooperative arbitration for spatially-conflicting maneuvers.

Pure and CARLA-free: callers translate their own maneuver state and peer
broadcast messages into ``ResourceClaim``s plus flat (x, y) positions, and
``should_yield`` decides who proceeds. This is the shared primitive behind
every "two CPX-controlled CAVs both want to do X near each other" gate
(lane change, static-obstacle avoidance lane selection, junction entry --
see call sites in ``cpx_mpc_planner.py``), so the priority rule only needs
to be got right once.

Priority is "earliest commitment wins" (``committed_at_s``), which is the
only rule that generalizes across all of the above -- lane changes have a
natural "physically ahead" ordering, but a 90-degree road junction does
not, so ordering purely by commitment time is what both cases can agree
on. Exact simultaneous commits break the tie on ``actor_id`` so every
observer resolves the same winner independently, without needing a
side channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ResourceClaim:
    """One CAV's claim on a shared resource.

    ``resource_id`` is caller-defined and only compared for equality, so
    callers control how coarse or specific conflicts are: a constant
    string (e.g. "lane_change") makes every claim of that kind conflict
    with every other regardless of which lane is involved; a specific
    lane/junction id restricts conflicts to CAVs contending for the exact
    same resource.
    """

    kind: str
    resource_id: str
    committed_at_s: float
    active: bool
    require_ahead: bool = True


def should_yield(
    *,
    my_claim: ResourceClaim,
    my_actor_id: int,
    my_position_xy: Tuple[float, float],
    my_heading_rad: float,
    peers: Sequence[Tuple[int, ResourceClaim, Tuple[float, float]]],
    range_m: float = 40.0,
) -> Optional[str]:
    """Return a yield reason, or ``None`` if ``my_claim`` may proceed.

    ``peers`` is ``(peer_actor_id, peer_claim, peer_position_xy)`` for every
    other CAV whose latest broadcast intent is available this tick.
    """

    forward_x = math.cos(float(my_heading_rad))
    forward_y = math.sin(float(my_heading_rad))
    for peer_actor_id, peer_claim, peer_position_xy in peers:
        if not bool(peer_claim.active):
            continue
        if str(peer_claim.kind) != str(my_claim.kind):
            continue
        if str(peer_claim.resource_id) != str(my_claim.resource_id):
            continue
        dx = float(peer_position_xy[0]) - float(my_position_xy[0])
        dy = float(peer_position_xy[1]) - float(my_position_xy[1])
        distance_m = math.hypot(dx, dy)
        if distance_m > float(range_m):
            continue
        if bool(my_claim.require_ahead):
            # A peer only has an ordering claim on ego's own maneuver if
            # it's ahead along ego's heading -- a peer already behind
            # cannot be the reason ego holds back.
            forward_distance_m = dx * forward_x + dy * forward_y
            if forward_distance_m <= 0.0:
                continue
        peer_wins = float(peer_claim.committed_at_s) < float(
            my_claim.committed_at_s
        ) or (
            float(peer_claim.committed_at_s) == float(my_claim.committed_at_s)
            and int(peer_actor_id) < int(my_actor_id)
        )
        if not peer_wins:
            continue
        return (
            "blocked_by_peer_claim:"
            f"kind={my_claim.kind}:resource={my_claim.resource_id}:"
            f"peer={peer_actor_id}:distance_m={distance_m:.1f}"
        )
    return None


# --------------------------------------------------------------------------- #
# Stage B: role / homotopy assignment (superset of ``should_yield``)
#
# ``should_yield`` answers one binary question ("may I proceed?"). The
# interaction-aware MPC plan needs more per conflict: a role
# (proceed / yield / make-gap), a pass side (homotopy), the conflict
# location, and tick-to-tick stability so the downstream corridor and MPC
# constraints don't flip. This block adds that without touching
# ``should_yield`` -- the two existing bridge call sites and their tests are
# unaffected. Still pure and CARLA-free: the hysteresis latch is passed in
# and returned; the caller owns persistence.
# --------------------------------------------------------------------------- #

_ROLE_PROCEED = "proceed"
_ROLE_YIELD = "yield"
_ROLE_MAKE_GAP = "make_gap"

# Merge-type conflicts get the bilateral "winner proceeds, loser opens a
# gap" treatment; other kinds (junction crossing, avoidance-lane pick) are
# plain proceed/yield.
_MERGE_KINDS = frozenset({"lane_change", "merge", "on_ramp_merge"})


@dataclass(frozen=True)
class CavIntent:
    """One connected CAV's broadcast this tick (C1 message schema).

    ``planned_path`` is ``(t_rel_s, x, y, v)`` samples of the cav's own
    plan, empty when the cav does not share a trajectory. ``cooperative``
    is False for a connected vehicle that reports state/claim but does not
    participate in role assignment -- such a cav is left to the obstacle /
    keep-out path, not assigned here.
    """

    actor_id: int
    position_xy: Tuple[float, float]
    claim: ResourceClaim
    heading_rad: float = 0.0
    speed_mps: float = 0.0
    planned_path: Tuple[Tuple[float, float, float, float], ...] = ()
    cooperative: bool = True


@dataclass(frozen=True)
class ConflictAssignment:
    """Resolved role for one (ego, cav) conflict."""

    cav_actor_id: int
    role: str                       # proceed | yield | make_gap
    homotopy_side: str              # "" | left | right | behind
    conflict_xy: Optional[Tuple[float, float]]
    cav_wins: bool
    reason: str


@dataclass
class ArbitrationLatchEntry:
    role: str = _ROLE_PROCEED
    side: str = ""
    candidate_role: str = ""
    candidate_side: str = ""
    candidate_count: int = 0


def lateral_side(
    origin_xy: Tuple[float, float],
    heading_rad: float,
    target_xy: Tuple[float, float],
) -> str:
    """"left" or "right": which side of ``origin``'s heading ``target`` is on."""

    dx = float(target_xy[0]) - float(origin_xy[0])
    dy = float(target_xy[1]) - float(origin_xy[1])
    cross = math.cos(float(heading_rad)) * dy - math.sin(float(heading_rad)) * dx
    return "left" if cross >= 0.0 else "right"


def _raw_cav_wins(
    my_claim: ResourceClaim, my_actor_id: int,
    cav_claim: ResourceClaim, cav_actor_id: int,
) -> bool:
    return float(cav_claim.committed_at_s) < float(my_claim.committed_at_s) or (
        float(cav_claim.committed_at_s) == float(my_claim.committed_at_s)
        and int(cav_actor_id) < int(my_actor_id)
    )


def assign_conflict_roles(
    *,
    my_claim: ResourceClaim,
    my_actor_id: int,
    my_position_xy: Tuple[float, float],
    my_heading_rad: float,
    cavs: Sequence[CavIntent],
    latch_state: Optional[Mapping[str, ArbitrationLatchEntry]] = None,
    range_m: float = 40.0,
    hysteresis_ticks: int = 3,
    decisive_margin_s: float = 1.0,
) -> Tuple[Sequence[ConflictAssignment], Dict[str, ArbitrationLatchEntry]]:
    """Assign a role + pass side per conflicting cooperative cav.

    Role decision uses the exact ``should_yield`` rule (earliest
    ``committed_at_s`` wins, ``actor_id`` breaks ties) so every CAV resolves
    the same winner independently. On a merge-kind conflict the loser's role
    is ``make_gap`` rather than a bare ``yield``.

    Hysteresis: a per-cav latch holds the previous role/side unless the new
    decision persists for ``hysteresis_ticks`` calls, or the commitment-time
    margin exceeds ``decisive_margin_s`` (then it switches immediately).
    Returns ``(assignments, new_latch_state)``; pass ``new_latch_state`` back
    in next call.
    """

    old = dict(latch_state or {})
    new: Dict[str, ArbitrationLatchEntry] = {}
    assignments = []

    fwd_x = math.cos(float(my_heading_rad))
    fwd_y = math.sin(float(my_heading_rad))
    is_merge = str(my_claim.kind) in _MERGE_KINDS

    for cav in cavs:
        if not bool(getattr(cav, "cooperative", True)):
            continue
        pclaim = cav.claim
        if not bool(pclaim.active):
            continue
        if str(pclaim.kind) != str(my_claim.kind):
            continue
        if str(pclaim.resource_id) != str(my_claim.resource_id):
            continue
        px, py = float(cav.position_xy[0]), float(cav.position_xy[1])
        dx = px - float(my_position_xy[0])
        dy = py - float(my_position_xy[1])
        distance_m = math.hypot(dx, dy)
        if distance_m > float(range_m):
            continue
        if bool(my_claim.require_ahead):
            if dx * fwd_x + dy * fwd_y <= 0.0:
                continue

        key = str(cav.actor_id)
        cav_wins = _raw_cav_wins(my_claim, my_actor_id, pclaim, cav.actor_id)
        raw_role = (
            (_ROLE_MAKE_GAP if is_merge else _ROLE_YIELD) if cav_wins
            else _ROLE_PROCEED
        )
        raw_side = lateral_side(my_position_xy, my_heading_rad, (px, py))

        margin_s = abs(
            float(pclaim.committed_at_s) - float(my_claim.committed_at_s)
        )
        prev = old.get(key, ArbitrationLatchEntry())
        entry = ArbitrationLatchEntry(
            role=prev.role or _ROLE_PROCEED,
            side=prev.side,
            candidate_role=prev.candidate_role,
            candidate_side=prev.candidate_side,
            candidate_count=int(prev.candidate_count),
        )

        decisive = margin_s >= float(decisive_margin_s)
        first_seen = key not in old
        if raw_role == entry.role and raw_side == entry.side:
            entry.candidate_role, entry.candidate_side, entry.candidate_count = "", "", 0
        elif decisive or first_seen:
            entry.role, entry.side = raw_role, raw_side
            entry.candidate_role, entry.candidate_side, entry.candidate_count = "", "", 0
        else:
            if raw_role == entry.candidate_role and raw_side == entry.candidate_side:
                entry.candidate_count += 1
            else:
                entry.candidate_role, entry.candidate_side = raw_role, raw_side
                entry.candidate_count = 1
            if entry.candidate_count >= int(hysteresis_ticks):
                entry.role, entry.side = raw_role, raw_side
                entry.candidate_role, entry.candidate_side, entry.candidate_count = "", "", 0

        new[key] = entry
        assignments.append(
            ConflictAssignment(
                cav_actor_id=int(cav.actor_id),
                role=entry.role,
                homotopy_side=entry.side,
                conflict_xy=(px, py),
                cav_wins=bool(cav_wins),
                reason=(
                    f"kind={my_claim.kind}:resource={my_claim.resource_id}:"
                    f"cav={cav.actor_id}:margin_s={margin_s:.2f}:"
                    f"raw={raw_role}:latched={entry.role}"
                ),
            )
        )

    return assignments, new
