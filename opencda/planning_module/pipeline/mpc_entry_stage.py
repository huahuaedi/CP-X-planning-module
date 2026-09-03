"""Structured authorization boundary immediately before MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .stage_contracts import authorize_mpc_entry


@dataclass(frozen=True)
class MPCEntryStageResult:
    authorization: Any
    hard_gate_reason: str
    stationary_stop_hold: bool

    def trace_fields(self) -> dict[str, object]:
        return dict(self.authorization.as_debug_fields())


class MPCEntryStage:
    """Own candidate/reference authorization and deterministic stop hold."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self._suspend_speed_mps = max(
            0.0, float(config.get("normal_stop_mpc_suspend_speed_mps", 0.30))
        )

    def evaluate(
        self,
        *,
        candidate_status: object,
        candidate_name: object,
        candidate_reason: object,
        final_reference_accepted: bool,
        final_reference_reason: object,
        behavior_decision: object,
        stop_goal_active: bool,
        ego_speed_mps: float,
    ) -> MPCEntryStageResult:
        authorization = authorize_mpc_entry(
            candidate_status=candidate_status,
            candidate_name=candidate_name,
            candidate_reason=candidate_reason,
            final_reference_accepted=bool(final_reference_accepted),
            final_reference_reason=final_reference_reason,
            behavior_decision=behavior_decision,
        )
        hard_gate_reason = (
            "" if authorization.allowed
            else "candidate_hard_gate:" + str(authorization.reason)
        )
        stationary_stop_hold = self.normal_stop_hold_required(
            hard_gate_active=bool(hard_gate_reason),
            stop_goal_active=bool(stop_goal_active),
            behavior_decision=str(behavior_decision),
            ego_speed_mps=float(ego_speed_mps),
        )
        return MPCEntryStageResult(
            authorization=authorization,
            hard_gate_reason=hard_gate_reason,
            stationary_stop_hold=bool(stationary_stop_hold),
        )

    def normal_stop_hold_required(
        self,
        *,
        hard_gate_active: bool,
        stop_goal_active: bool,
        behavior_decision: str,
        ego_speed_mps: float,
    ) -> bool:
        decision = str(behavior_decision or "").strip().lower()
        return bool(
            not hard_gate_active
            and stop_goal_active
            and decision in {"stop_at_intersection", "stop_sign"}
            and float(ego_speed_mps) <= self._suspend_speed_mps
        )
