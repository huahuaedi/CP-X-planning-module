"""Cross-CAV cooperative arbitration for spatially-conflicting maneuvers.

Pure and CARLA-free: callers translate maneuver state and peer broadcasts
into ``ResourceClaim`` and ``CavIntent`` values. ``assign_conflict_roles``
is the sole priority/latching implementation consumed by the conflict
pipeline.

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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


CLAIM_PROPOSED = "proposed"
CLAIM_COMMITTED = "committed"
CLAIM_RELEASED = "released"
_ACTIVE_CLAIM_PHASES = frozenset({CLAIM_PROPOSED, CLAIM_COMMITTED})


@dataclass(frozen=True)
class ResourceClaim:
    """One CAV's claim on a shared resource.

    ``resource_id`` is caller-defined and only compared for equality, so
    callers control how coarse or specific conflicts are: a constant
    string (e.g. "lane_change") makes every claim of that kind conflict
    with every other regardless of which lane is involved; a specific
    lane/junction id restricts conflicts to CAVs contending for the exact
    same resource.

    ``s_begin_m`` and ``s_end_m``, when present, must use the shared
    longitudinal coordinate of the claimed AD-map corridor.  A vehicle's
    private route-progress coordinate is not comparable with another
    vehicle's route and must not be stored here.  Missing intervals are
    deliberately treated as overlapping (conservative compatibility).
    """

    kind: str
    resource_id: str
    committed_at_s: float
    active: bool
    require_ahead: bool = True
    phase: str = CLAIM_COMMITTED
    source_corridor_id: int = 0
    target_corridor_id: int = 0
    station_corridor_id: int = 0
    s_begin_m: Optional[float] = None
    s_end_m: Optional[float] = None

    @property
    def participates(self) -> bool:
        """Whether this claim participates in distributed arbitration."""

        return bool(
            self.active and str(self.phase).strip().lower() in _ACTIVE_CLAIM_PHASES
        )

    @property
    def committed(self) -> bool:
        return bool(self.participates and str(self.phase).lower() == CLAIM_COMMITTED)

    def conflicts_with(self, other: "ResourceClaim") -> bool:
        """Return whether two active claims occupy the same spatial resource."""

        if not self.participates or not other.participates:
            return False
        if str(self.kind) != str(other.kind):
            return False
        self_spatial = bool(self.source_corridor_id or self.target_corridor_id)
        other_spatial = bool(other.source_corridor_id or other.target_corridor_id)
        if not self_spatial or not other_spatial:
            return str(self.resource_id) == str(other.resource_id)
        comparable_station = bool(
            self.station_corridor_id
            and self.station_corridor_id == other.station_corridor_id
        )
        if comparable_station and not _intervals_overlap(
            self.s_begin_m, self.s_end_m, other.s_begin_m, other.s_end_m
        ):
            return False
        same_target = bool(
            self.target_corridor_id
            and self.target_corridor_id == other.target_corridor_id
        )
        opposing_transition = bool(
            self.source_corridor_id == other.target_corridor_id
            and self.target_corridor_id == other.source_corridor_id
        )
        return same_target or opposing_transition


def _intervals_overlap(
    a_begin: Optional[float], a_end: Optional[float],
    b_begin: Optional[float], b_end: Optional[float],
) -> bool:
    if None in (a_begin, a_end, b_begin, b_end):
        return True
    a_lo, a_hi = sorted((float(a_begin), float(a_end)))
    b_lo, b_hi = sorted((float(b_begin), float(b_end)))
    return max(a_lo, b_lo) <= min(a_hi, b_hi)


# --------------------------------------------------------------------------- #
# Stage B: cooperative role assignment
#
# The interaction-aware plan needs a stable role (proceed / yield / make-gap)
# and pass side for each conflict. The hysteresis latch is passed in and
# returned; the caller owns persistence.
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
    generated_at_s: float = 0.0
    valid_until_s: float = float("inf")
    sequence: int = 0
    probability: float = 1.0


@dataclass(frozen=True)
class ConflictAssignment:
    """Resolved role for one (ego, cav) conflict."""

    cav_actor_id: int
    role: str                       # proceed | yield | make_gap
    homotopy_side: str              # left | right in ego-heading frame
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
    """Return which side of ego's heading contains ``target_xy``."""

    dx = float(target_xy[0]) - float(origin_xy[0])
    dy = float(target_xy[1]) - float(origin_xy[1])
    cross = math.cos(float(heading_rad)) * dy - math.sin(float(heading_rad)) * dx
    return "left" if cross >= 0.0 else "right"


def _raw_cav_wins(
    my_claim: ResourceClaim, my_actor_id: int,
    cav_claim: ResourceClaim, cav_actor_id: int,
) -> bool:
    # A physically committed maneuver cannot be pre-empted by a proposal.
    # This ordering is evaluated identically by both peers, then timestamp/id
    # resolves contenders in the same lifecycle phase.
    if bool(cav_claim.committed) != bool(my_claim.committed):
        return bool(cav_claim.committed)
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
    hysteresis_ticks: int = 3,
    decisive_margin_s: float = 1.0,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[Sequence[ConflictAssignment], Dict[str, ArbitrationLatchEntry]]:
    """Assign a stable role and pass-side homotopy per cooperative CAV.

    The earliest ``committed_at_s`` wins and ``actor_id`` breaks ties, so
    every CAV resolves
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

    eligibility: Dict[str, Dict[str, Any]] = {}
    for cav in cavs:
        key = str(cav.actor_id)
        if not bool(getattr(cav, "cooperative", True)):
            eligibility[key] = {"eligible": False, "reason": "not_cooperative"}
            continue
        pclaim = cav.claim
        if not bool(pclaim.participates):
            eligibility[key] = {"eligible": False, "reason": "peer_claim_inactive"}
            continue
        if not my_claim.conflicts_with(pclaim):
            eligibility[key] = {"eligible": False, "reason": "claim_resources_disjoint"}
            continue
        px, py = float(cav.position_xy[0]), float(cav.position_xy[1])
        dx = px - float(my_position_xy[0])
        dy = py - float(my_position_xy[1])
        distance_m = math.hypot(dx, dy)
        if bool(my_claim.require_ahead):
            if dx * fwd_x + dy * fwd_y <= 0.0:
                eligibility[key] = {
                    "eligible": False,
                    "reason": "peer_not_ahead",
                    "distance_m": float(distance_m),
                }
                continue

        eligibility[key] = {
            "eligible": True,
            "reason": "assigned",
            "distance_m": float(distance_m),
        }
        cav_wins = _raw_cav_wins(my_claim, my_actor_id, pclaim, cav.actor_id)
        raw_role = (
            (_ROLE_MAKE_GAP if is_merge else _ROLE_YIELD) if cav_wins
            else _ROLE_PROCEED
        )
        peer_side = lateral_side(my_position_xy, my_heading_rad, (px, py))
        raw_side = "right" if peer_side == "left" else "left"
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
                cav_wins=bool(cav_wins),
                reason=(
                    f"kind={my_claim.kind}:resource={my_claim.resource_id}:"
                    f"cav={cav.actor_id}:margin_s={margin_s:.2f}:"
                    f"raw={raw_role}:latched={entry.role}"
                ),
            )
        )

    if diagnostics is not None:
        diagnostics["eligibility"] = eligibility
        diagnostics["assignment_count"] = len(assignments)
    return assignments, new
