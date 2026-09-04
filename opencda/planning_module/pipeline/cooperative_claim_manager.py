"""Lifecycle owner for one ego CAV's cooperative maneuver claim."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .cooperative_arbitration import ResourceClaim


_LANE_CHANGES = frozenset({"lane_change_left", "lane_change_right"})
_DEFER_ROLES = frozenset({"yield", "make_gap"})


class CooperativeClaimManager:
    """Own proposal timing and the proposed-to-committed transition."""

    def __init__(self, *, enabled: bool, proposal_dwell_s: float = 0.25) -> None:
        self._enabled = bool(enabled)
        self._proposal_dwell_s = max(0.0, float(proposal_dwell_s))
        self._proposal_key: Optional[Tuple[str, int]] = None
        self._proposed_at_s = 0.0
        self._last_claim: Optional[ResourceClaim] = None

    @property
    def proposal_active(self) -> bool:
        return self._proposal_key is not None

    @property
    def current_claim(self) -> Optional[ResourceClaim]:
        return self._last_claim

    def claim(
        self, *, decision: str, target_lane_id: int, sim_time_s: float,
        maneuver_active: bool, committed_at_s: float,
    ) -> Optional[ResourceClaim]:
        if not self._enabled:
            return None
        if bool(maneuver_active):
            self._proposal_key = None
            timestamp = float(committed_at_s)
            if timestamp <= 0.0:
                timestamp = float(sim_time_s)
            self._last_claim = ResourceClaim(
                kind="lane_change", resource_id="lane_change",
                committed_at_s=timestamp, active=True,
                require_ahead=False, phase="committed",
            )
            return self._last_claim
        normalized = str(decision).strip().lower()
        if normalized not in _LANE_CHANGES or int(target_lane_id) == 0:
            self._proposal_key = None
            self._last_claim = ResourceClaim(
                kind="lane_change", resource_id="lane_change",
                committed_at_s=float(sim_time_s), active=False,
                require_ahead=False, phase="released",
            )
            return self._last_claim
        key = (normalized, int(target_lane_id))
        if key != self._proposal_key:
            self._proposal_key = key
            self._proposed_at_s = float(sim_time_s)
        self._last_claim = ResourceClaim(
            kind="lane_change", resource_id="lane_change",
            committed_at_s=float(self._proposed_at_s), active=True,
            require_ahead=False, phase="proposed",
        )
        return self._last_claim

    def defer_candidate(self, *, sim_time_s: float, assignments: Sequence[object]) -> bool:
        if not self.proposal_active:
            return False
        if float(sim_time_s) - float(self._proposed_at_s) < self._proposal_dwell_s:
            return True
        return any(
            str(getattr(assignment, "role", "")).strip().lower() in _DEFER_ROLES
            for assignment in list(assignments or ())
        )
