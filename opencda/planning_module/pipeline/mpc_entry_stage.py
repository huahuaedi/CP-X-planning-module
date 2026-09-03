"""Structured authorization boundary immediately before MPC."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .stage_contracts import authorize_mpc_entry


@dataclass(frozen=True)
class MPCEntryStageResult:
    authorization: Any
    hard_gate_reason: str
    stationary_stop_hold: bool

    def trace_fields(self) -> dict[str, object]:
        return dict(self.authorization.as_debug_fields())


@dataclass(frozen=True)
class MPCControlContext:
    key: str
    reference_anchor_relative_m: Any
    force_replan: bool


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

    @staticmethod
    def prepare_control_context(
        *, behavior: Any, reference_source: str, stop_goal_active: bool,
        front_gap_actor_id: str, reference_samples: Any,
        ego_x_m: float, ego_y_m: float, ego_yaw_rad: float,
        mode_transition_reason: str,
    ) -> MPCControlContext:
        """Build the immutable identity used by MPC control-buffer reuse."""

        key = "|".join((
            str(behavior.maneuver), str(behavior.phase),
            str(behavior.target_lane_id), str(reference_source),
            str(bool(stop_goal_active)), str(behavior.traffic_signal_state),
            str(front_gap_actor_id),
        ))
        anchor = None
        rows = list(reference_samples or [])
        if rows:
            first = rows[0]
            dx = float(first.get("x_ref_m", first.get("x", ego_x_m))) - float(ego_x_m)
            dy = float(first.get("y_ref_m", first.get("y", ego_y_m))) - float(ego_y_m)
            anchor = (
                math.cos(ego_yaw_rad) * dx + math.sin(ego_yaw_rad) * dy,
                -math.sin(ego_yaw_rad) * dx + math.cos(ego_yaw_rad) * dy,
            )
        force = bool(
            stop_goal_active
            or str(behavior.maneuver) in {
                "stop_at_intersection", "stop_sign", "emergency_brake",
                "intersection_turn_left", "intersection_turn_right",
            }
            or mode_transition_reason
        )
        return MPCControlContext(key, anchor, force)

    @staticmethod
    def low_speed_replan_required(
        *, ego_speed_mps: float, behavior_decision: str,
        behavior_fsm_state: str, stop_goal_active: bool,
        minimum_speed_mps: float,
    ) -> bool:
        return bool(
            not stop_goal_active
            and str(behavior_decision or "").strip().lower() == "lane_follow"
            and str(behavior_fsm_state or "").strip().upper()
            in {"", "IDLE", "LANE_KEEP"}
            and float(ego_speed_mps) < float(minimum_speed_mps)
        )
