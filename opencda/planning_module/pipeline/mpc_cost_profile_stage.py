"""Own MPC objective-profile and adaptive-horizon lifecycle state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .candidate_evaluation import mpc_cost_profile_for_behavior


DEFAULT_ADAPTIVE_HORIZON_PROFILE_S: dict[str, float] = {
    "lane_follow": 3.0,
    "prepare_lane_change": 4.5,
    "execute_lane_change": 4.5,
    "intersection_turn": 2.2,
    "stop": 2.0,
    "recovery": 1.5,
}


@dataclass(frozen=True)
class MPCCostProfileState:
    active: str = "lane_follow"
    requested: str = "lane_follow"
    active_since_s: float = 0.0
    switch_reason: str = "initial"

    def trace_fields(self) -> dict[str, object]:
        return {
            "mpc_cost_profile": str(self.active),
            "requested_mpc_cost_profile": str(self.requested),
            "mpc_cost_profile_switch_reason": str(self.switch_reason),
        }


class MPCCostProfileStage:
    """Select and apply one stable MPC objective profile per tick."""

    def __init__(
        self, *, mpc: Any, config: Mapping[str, object],
        behavior_runtime_config: Mapping[str, object],
    ) -> None:
        self._mpc = mpc
        self._config = config
        self._behavior_runtime_config = behavior_runtime_config
        self._state = MPCCostProfileState()

    @property
    def state(self) -> MPCCostProfileState:
        return self._state

    def apply(
        self, *, behavior: str, planner_lc_state: str, planner_mode: str,
        next_macro_maneuver: str, reference_tracking_mode: str,
        sim_time_s: float, nearest_obstacle_distance_m: Optional[float],
        ego_speed_mps: float,
    ) -> MPCCostProfileState:
        previous_active = str(self._state.active)
        requested = mpc_cost_profile_for_behavior(
            behavior=behavior,
            planner_lc_state=planner_lc_state,
            planner_mode=planner_mode,
            next_macro_maneuver=next_macro_maneuver,
            reference_tracking_mode=reference_tracking_mode,
        )
        active, active_since_s, reason = select_profile_with_hysteresis(
            requested_profile=str(requested),
            active_profile=str(self._state.active),
            sim_time_s=float(sim_time_s),
            active_since_s=float(self._state.active_since_s),
            min_hold_s=float(self._behavior_runtime_config.get(
                "mpc_cost_profile_min_hold_s", 1.5
            )),
        )
        if hasattr(self._mpc, "apply_mode_cost_profile"):
            active = str(self._mpc.apply_mode_cost_profile(str(active)))
        if (
            str(active) != previous_active
            and hasattr(self._mpc, "clear_previous_solution_seed")
        ):
            self._mpc.clear_previous_solution_seed()
        if (
            bool(getattr(self._mpc, "adaptive_horizon_enabled", False))
            and hasattr(self._mpc, "blend_toward_horizon_s")
        ):
            profile_horizon_s = (
                dict(self._config.get("adaptive_horizon_profile_s", {}))
                or DEFAULT_ADAPTIVE_HORIZON_PROFILE_S
            )
            self._mpc.blend_toward_horizon_s(adaptive_target_horizon_s(
                mpc_cost_profile=str(active),
                nearest_obstacle_distance_m=nearest_obstacle_distance_m,
                ego_speed_mps=float(ego_speed_mps),
                profile_horizon_s=profile_horizon_s,
                obstacle_reference_speed_mps=float(self._config.get(
                    "adaptive_horizon_obstacle_reference_speed_mps", 2.0
                )),
                obstacle_comfortable_decel_mps2=float(self._config.get(
                    "adaptive_horizon_obstacle_comfortable_decel_mps2", 2.0
                )),
            ))
        self._state = MPCCostProfileState(
            active=str(active), requested=str(requested),
            active_since_s=float(active_since_s), switch_reason=str(reason),
        )
        return self._state


def adaptive_target_horizon_s(
    *, mpc_cost_profile: str, nearest_obstacle_distance_m: Optional[float],
    ego_speed_mps: float, profile_horizon_s: Mapping[str, float],
    obstacle_reference_speed_mps: float = 2.0,
    obstacle_comfortable_decel_mps2: float = 2.0,
) -> float:
    base = float(profile_horizon_s.get(
        str(mpc_cost_profile), profile_horizon_s.get("lane_follow", 3.0)
    ))
    if nearest_obstacle_distance_m is not None and math.isfinite(
        float(nearest_obstacle_distance_m)
    ):
        distance_reaction_s = float(nearest_obstacle_distance_m) / max(
            1.0e-3, float(obstacle_reference_speed_mps)
        )
        stopping_time_s = float(ego_speed_mps) / max(
            1.0e-3, float(obstacle_comfortable_decel_mps2)
        )
        base = min(base, max(1.0, max(distance_reaction_s, stopping_time_s)))
    return float(base)


def select_profile_with_hysteresis(
    *, requested_profile: str, active_profile: str, sim_time_s: float,
    active_since_s: float, min_hold_s: float,
) -> tuple[str, float, str]:
    requested = str(requested_profile or "lane_follow").strip() or "lane_follow"
    active = str(active_profile or "lane_follow").strip() or "lane_follow"
    elapsed_s = max(0.0, float(sim_time_s) - float(active_since_s))
    min_hold_s = max(0.0, float(min_hold_s))
    if requested == active:
        return active, float(active_since_s), "same_profile"
    if requested in {"stop", "recovery"}:
        return requested, float(sim_time_s), "safety_preempt"
    if active in {"stop", "recovery"} and elapsed_s < min_hold_s:
        return active, float(active_since_s), "hold_safety_profile"
    if elapsed_s < min_hold_s:
        return active, float(active_since_s), "min_hold"
    return requested, float(sim_time_s), "switch"
