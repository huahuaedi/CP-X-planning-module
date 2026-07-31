"""Scenario-aware speed planning for the CP-X pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional


@dataclass(frozen=True)
class SpeedPlan:
    target_speed_mps: float
    speed_cap_mps: float
    stop_goal_active: bool
    reason: str = ""
    front_gap_m: Optional[float] = None
    desired_follow_gap_m: Optional[float] = None
    continuous_following_active: bool = False

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "speed_plan_target_mps": float(self.target_speed_mps),
            "speed_plan_cap_mps": float(self.speed_cap_mps),
            "speed_plan_stop_goal_active": bool(self.stop_goal_active),
            "speed_plan_reason": str(self.reason),
            "speed_plan_front_gap_m": (
                "" if self.front_gap_m is None else float(self.front_gap_m)
            ),
            "speed_plan_desired_follow_gap_m": (
                ""
                if self.desired_follow_gap_m is None
                else float(self.desired_follow_gap_m)
            ),
            "speed_plan_continuous_following_active": bool(
                self.continuous_following_active
            ),
        }


def build_speed_plan(
    *,
    scenario_decision: object,
    behavior_decision: object,
    requested_speed_mps: float,
    ego_speed_mps: float,
    config: Mapping[str, object],
    front_gap_m: Optional[float] = None,
) -> SpeedPlan:
    """Return the speed target owned by the scenario/behavior layer."""

    requested = max(0.0, float(requested_speed_mps))
    scenario_cap = getattr(scenario_decision, "speed_cap_mps", None)
    cap = requested if scenario_cap is None else min(requested, max(0.0, float(scenario_cap)))
    stop_goal = bool(getattr(scenario_decision, "stop_goal_active", False))
    reason = str(getattr(scenario_decision, "reason", "") or "")
    decision = str(behavior_decision or "").strip().lower()
    if decision in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
        stop_goal = True
    if decision in {"intersection_turn_left", "intersection_turn_right"}:
        cap = min(
            float(cap),
            max(0.1, float(config.get("full_intersection_turn_speed_cap_mps", 2.2))),
        )
    following_active = False
    following_gap_m = None
    desired_gap_m = None
    if front_gap_m is not None and math.isfinite(float(front_gap_m)) and not stop_goal:
        gap_m = max(0.0, float(front_gap_m))
        following_gap_m = float(gap_m)
        emergency_gap_m = max(
            0.5,
            float(config.get("following_emergency_gap_m", 3.0)),
        )
        standstill_gap_m = max(
            emergency_gap_m,
            float(config.get("following_standstill_gap_m", 5.0)),
        )
        time_headway_s = max(
            0.1,
            float(config.get("following_time_headway_s", 1.5)),
        )
        free_gap_margin_m = max(
            1.0,
            float(config.get("following_free_gap_margin_m", 6.0)),
        )
        minimum_follow_speed_mps = max(
            0.0,
            float(config.get("following_minimum_speed_mps", 0.35)),
        )
        desired_gap_m = standstill_gap_m + time_headway_s * max(0.0, float(ego_speed_mps))
        free_gap_m = desired_gap_m + free_gap_margin_m
        if gap_m <= emergency_gap_m:
            cap = 0.0
            stop_goal = True
            reason = _join_reason(reason, "speed_plan_obstacle_emergency_stop")
        elif gap_m < free_gap_m:
            ratio = (gap_m - emergency_gap_m) / max(
                1.0e-6,
                free_gap_m - emergency_gap_m,
            )
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            follow_cap = minimum_follow_speed_mps + smooth_ratio * max(
                0.0,
                float(cap) - minimum_follow_speed_mps,
            )
            cap = min(float(cap), float(follow_cap))
            following_active = True
    if stop_goal:
        cap = 0.0
    if not math.isfinite(cap):
        cap = 0.0
    if decision in {"intersection_turn_left", "intersection_turn_right"}:
        reason = _join_reason(reason, "speed_plan_turn_cap")
    if stop_goal:
        reason = _join_reason(reason, "speed_plan_stop_zero")
    elif following_active:
        reason = _join_reason(reason, "speed_plan_continuous_following")
    elif scenario_cap is not None and float(cap) < float(requested):
        reason = _join_reason(reason, "speed_plan_scenario_cap")
    return SpeedPlan(
        target_speed_mps=float(cap),
        speed_cap_mps=float(cap),
        stop_goal_active=bool(stop_goal),
        reason=str(reason),
        front_gap_m=following_gap_m,
        desired_follow_gap_m=desired_gap_m,
        continuous_following_active=bool(following_active),
    )


def _join_reason(first: str, second: str) -> str:
    if not first:
        return str(second)
    if str(second) in str(first).split(";"):
        return str(first)
    return f"{first};{second}"
