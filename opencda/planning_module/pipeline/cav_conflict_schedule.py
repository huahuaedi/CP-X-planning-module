"""Single owner for multi-rate cooperative-conflict scheduling state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class CoordinationScheduleDecision:
    refresh_roles: bool
    reason: str
    revision: str


@dataclass
class CAVConflictSchedule:
    """Own role cadence and cached Stage-A/B state across planning ticks."""

    coordination_period_s: float = 0.2
    latch_state: dict = field(default_factory=dict)
    tag_state: dict = field(default_factory=dict)
    assignments: tuple = ()
    _last_refresh_s: float = -float("inf")
    _last_input_revision: str = ""
    _has_observation: bool = False
    revision: int = 0

    @staticmethod
    def _claim_signature(claim: Any) -> Tuple[Any, ...]:
        if claim is None:
            return ()
        return (
            str(getattr(claim, "resource_id", "")),
            str(getattr(claim, "phase", "")),
            bool(getattr(claim, "participates", False)),
            round(float(getattr(claim, "committed_at_s", 0.0)), 3),
        )

    @staticmethod
    def _peer_signature(peers: Sequence[Any]) -> Tuple[Any, ...]:
        return tuple(sorted(
            (
                int(getattr(peer, "actor_id", -1)),
                str(getattr(getattr(peer, "claim", None), "resource_id", "")),
                str(getattr(getattr(peer, "claim", None), "phase", "")),
                int(getattr(peer, "sequence", 0)),
                len(tuple(getattr(peer, "planned_path", ()) or ())),
            )
            for peer in peers or ()
        ))

    def decide(
        self, *, sim_time_s: float, prediction_revision: str,
        claim: Any, peers: Sequence[Any], proposal: Any,
    ) -> CoordinationScheduleDecision:
        input_revision = repr((
            str(prediction_revision), self._claim_signature(claim),
            self._peer_signature(peers),
            str(getattr(proposal, "maneuver", "")),
            int(getattr(proposal, "target_corridor_id", 0)),
            bool(getattr(proposal, "committed", False)),
        ))
        if not self._has_observation:
            reason = "coordination_cache_empty"
        elif input_revision != self._last_input_revision:
            reason = "coordination_input_revision_changed"
        elif float(sim_time_s) - float(self._last_refresh_s) + 1.0e-9 >= max(
            0.01, float(self.coordination_period_s)
        ):
            reason = "coordination_period_elapsed"
        else:
            return CoordinationScheduleDecision(
                False, "coordination_cache_reused", str(self.revision)
            )
        self._last_input_revision = input_revision
        return CoordinationScheduleDecision(True, reason, str(self.revision + 1))

    def observe(self, *, sim_time_s: float, result: Any) -> None:
        self.latch_state = dict(getattr(result, "latch_state", {}) or {})
        self.tag_state = dict(getattr(result, "tag_state", {}) or {})
        self.assignments = tuple(getattr(result, "assignments", ()) or ())
        self._has_observation = True
        if bool(getattr(result, "diagnostics", {}).get(
            "coordination_roles_refreshed", False
        )):
            self._last_refresh_s = float(sim_time_s)
            self.revision += 1
