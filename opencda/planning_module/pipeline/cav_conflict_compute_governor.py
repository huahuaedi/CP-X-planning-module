"""Runtime feedback on top of resolve_conflicts' static agent/mode budget.

cav_conflict_pipeline.resolve_conflicts already bounds Stage B/C to a
fixed max_relevant_agents/max_modes_per_agent per tick, ranked by Stage A's
own classified severity so a real conflict is never dropped in favor of a
merely-nearby one (see _cap_agents_by_severity). That static cap controls
the worst case but cannot know, ahead of time, how much slower this
specific machine or this specific tick's assign/corridor work actually is.
This module owns
the one further decision: given how long the conflict pipeline has actually
been taking, should the budget step down (or back up) from its static
ceiling for the *next* tick.

Deliberately separate from resolve_conflicts, which stays a pure function of
its arguments (see its own docstring) -- runtime state belongs to a caller
that can afford to own it across ticks, not to the per-tick classify/assign/
corridor computation itself.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeBudgetDecision:
    max_relevant_agents: int
    max_modes_per_agent: int
    degraded: bool
    reason: str


class CAVConflictComputeGovernor:
    """Own the agent/mode budget's runtime step-down/recovery state.

    Hysteresis on both directions: a single slow tick (GC pause, one-off
    contention) must not degrade a scene that otherwise fits comfortably,
    and recovery must not immediately re-offer capacity a machine running
    consistently hot can't actually sustain -- both would just flap the
    budget tick to tick, which is worse than picking one level and holding
    it.
    """

    def __init__(
        self,
        *,
        max_agents: int = 6,
        max_modes: int = 6,
        min_agents: int = 2,
        min_modes: int = 1,
        budget_ms: float = 100.0,
        window: int = 5,
        degrade_streak: int = 3,
        recover_streak: int = 5,
        recover_headroom_ratio: float = 0.5,
    ) -> None:
        self._max_agents = max(int(min_agents), int(max_agents))
        self._max_modes = max(int(min_modes), int(max_modes))
        self._min_agents = max(1, int(min_agents))
        self._min_modes = max(1, int(min_modes))
        self._budget_ms = max(1.0, float(budget_ms))
        self._recover_headroom_ms = self._budget_ms * max(
            0.0, min(1.0, float(recover_headroom_ratio))
        )
        self._degrade_streak_required = max(1, int(degrade_streak))
        self._recover_streak_required = max(1, int(recover_streak))
        self._recent_ms: deque = deque(maxlen=max(1, int(window)))
        self._agents = self._max_agents
        self._modes = self._max_modes
        self._over_budget_streak = 0
        self._under_headroom_streak = 0

    @property
    def current_max_relevant_agents(self) -> int:
        return int(self._agents)

    @property
    def current_max_modes_per_agent(self) -> int:
        return int(self._modes)

    def reset(self) -> None:
        """Return to the static ceiling, e.g. on route change / new scenario."""

        self._recent_ms.clear()
        self._agents = self._max_agents
        self._modes = self._max_modes
        self._over_budget_streak = 0
        self._under_headroom_streak = 0

    def observe_stage_ms(self, stage_ms: float) -> ComputeBudgetDecision:
        """Record one tick's conflict-pipeline duration; return this tick's budget."""

        self._recent_ms.append(max(0.0, float(stage_ms)))
        if len(self._recent_ms) < self._recent_ms.maxlen:
            # Insufficient history to trust a trend either way -- hold.
            return self._decision("insufficient_history")
        average_ms = sum(self._recent_ms) / len(self._recent_ms)

        if average_ms > self._budget_ms:
            self._over_budget_streak += 1
            self._under_headroom_streak = 0
        elif average_ms <= self._recover_headroom_ms:
            self._under_headroom_streak += 1
            self._over_budget_streak = 0
        else:
            # Between the recovery floor and the budget ceiling: neither
            # trend gets to claim this tick, so both streaks lapse rather
            # than let one stale streak fire once conditions later swing
            # back its way.
            self._over_budget_streak = 0
            self._under_headroom_streak = 0

        if (
            self._over_budget_streak >= self._degrade_streak_required
            and (self._agents > self._min_agents or self._modes > self._min_modes)
        ):
            self._step_down()
            self._over_budget_streak = 0
            return self._decision(
                f"degraded_after_{self._degrade_streak_required}"
                f"_over_budget_ticks_avg_{average_ms:.0f}ms"
            )
        if (
            self._under_headroom_streak >= self._recover_streak_required
            and (self._agents < self._max_agents or self._modes < self._max_modes)
        ):
            self._step_up()
            self._under_headroom_streak = 0
            return self._decision(
                f"recovered_after_{self._recover_streak_required}"
                f"_under_headroom_ticks_avg_{average_ms:.0f}ms"
            )
        return self._decision("holding")

    def _step_down(self) -> None:
        # Modes first: removing one hypothesis per agent reduces corridor
        # work across every retained agent before sacrificing an entire
        # physical participant from negotiation.
        if self._modes > self._min_modes:
            self._modes -= 1
        elif self._agents > self._min_agents:
            self._agents -= 1

    def _step_up(self) -> None:
        if self._agents < self._max_agents:
            self._agents += 1
        elif self._modes < self._max_modes:
            self._modes += 1

    def _decision(self, reason: str) -> ComputeBudgetDecision:
        return ComputeBudgetDecision(
            max_relevant_agents=int(self._agents),
            max_modes_per_agent=int(self._modes),
            degraded=bool(
                self._agents < self._max_agents or self._modes < self._max_modes
            ),
            reason=str(reason),
        )
