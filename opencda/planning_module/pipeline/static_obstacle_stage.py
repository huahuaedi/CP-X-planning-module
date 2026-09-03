"""Stateful static-obstacle behavior arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence


def select_local_avoidance_lane(*, current_lane_id: int,
                                available_lane_ids: Sequence[int],
                                lane_safety_scores: Mapping[int, float],
                                lane_prediction_risks: Mapping[int, Mapping[str, object]],
                                minimum_safety_score: float) -> Optional[int]:
    current = int(current_lane_id)
    alternatives = sorted(
        {int(x) for x in available_lane_ids if int(x) not in {0, current}},
        key=lambda lane_id: abs(lane_id - current),
    )
    if not alternatives:
        return None
    nearest_delta = abs(alternatives[0] - current)
    safe = []
    for lane_id in alternatives:
        if abs(lane_id - current) != nearest_delta:
            continue
        score = float(lane_safety_scores.get(lane_id, 0.0))
        risk = dict(lane_prediction_risks.get(lane_id, {}) or {})
        if score > float(minimum_safety_score) and not risk.get("risk", False):
            safe.append((score, lane_id))
    return max(safe)[1] if safe else None


def cooldown_policy(*, failed_latched: bool,
                    route_transition_pending: bool) -> tuple[str, bool]:
    if failed_latched:
        return "cooldown_stop", True
    if route_transition_pending:
        return "cooldown_route_transition", False
    return "cooldown_stop", True


@dataclass(frozen=True)
class StaticObstacleResult:
    local_avoidance_active: bool
    target_lane_id: Optional[int]
    stop_active: bool
    status: str
    reason: str
    candidate_id: str
    candidate_since_s: float
    route_transition_pending: bool


class StaticObstacleStage:
    """Sole owner of obstacle confirmation, lane borrow and replan cooldown."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)
        self.candidate_id = ""
        self.candidate_since_s = -float("inf")
        self.failed_latched = False
        self.status = "idle"
        self.reason = "not_requested"
        self.target_lane_id: Optional[int] = None
        self.route_transition_pending = False
        self.last_replan_attempt_s = -float("inf")
        self.stop_active = False

    def evaluate(self, *, requested: bool, traffic_control_stop_active: bool,
                 obstacle_id: str, current_lane_id: int,
                 lane_change_reference_active: bool, sim_time_s: float,
                 normal_mode: bool, available_lane_ids: Sequence[int],
                 lane_safety_scores: Mapping[int, float],
                 lane_prediction_risks: Mapping[int, Mapping[str, object]],
                 cooperative_yield: Callable[[int], str],
                 attempt_replan: Callable[[], tuple[bool, bool, str]]) -> StaticObstacleResult:
        if (self.target_lane_id is not None
                and int(current_lane_id) == int(self.target_lane_id)
                and not lane_change_reference_active):
            self.target_lane_id = None
        active = self.target_lane_id is not None
        transition_hold = cooldown_hold = False
        if not requested:
            self.candidate_id = ""
            self.candidate_since_s = -float("inf")
            self.route_transition_pending = False
            self.failed_latched = False
            self.status = (
                "local_avoidance_executing" if active
                else "traffic_control_excluded" if traffic_control_stop_active
                else "idle"
            )
        else:
            if str(obstacle_id) != self.candidate_id:
                self.candidate_id = str(obstacle_id)
                self.candidate_since_s = float(sim_time_s)
            confirm_s = max(0.0, float(self.config.get(
                "static_obstacle_blocked_confirm_s", 1.0)))
            if float(sim_time_s) - self.candidate_since_s < confirm_s:
                self.status = "confirming"
            else:
                target = select_local_avoidance_lane(
                    current_lane_id=current_lane_id,
                    available_lane_ids=available_lane_ids,
                    lane_safety_scores=lane_safety_scores,
                    lane_prediction_risks=lane_prediction_risks,
                    minimum_safety_score=float(self.config.get(
                        "static_obstacle_local_lane_min_safety_score", 0.55)),
                ) if normal_mode and bool(self.config.get(
                    "static_obstacle_local_avoidance_enabled", True)) else None
                yield_reason = cooperative_yield(int(target)) if target is not None else ""
                if yield_reason:
                    target = None
                if target is not None:
                    self.target_lane_id = int(target)
                    active = True
                    self.failed_latched = False
                    self.route_transition_pending = False
                    self.status = "local_avoidance_ready"
                    self.reason = f"static_obstacle_local_lane_borrow:target_lane={target}"
                elif bool(self.config.get("static_obstacle_global_replan_enabled", False)):
                    cooldown_s = max(0.1, float(self.config.get(
                        "static_obstacle_replan_cooldown_s", 2.0)))
                    if float(sim_time_s) - self.last_replan_attempt_s < cooldown_s:
                        self.status = (
                            "cooldown_stop" if self.failed_latched
                            else "cooldown_route_transition" if self.route_transition_pending
                            else "cooldown_stop"
                        )
                        cooldown_hold = self.status != "cooldown_route_transition"
                    else:
                        self.last_replan_attempt_s = float(sim_time_s)
                        attempted, succeeded, reason = attempt_replan()
                        self.reason = str(reason)
                        if succeeded:
                            self.failed_latched = False
                            self.status = "succeeded"
                            self.route_transition_pending = True
                            transition_hold = True
                        elif attempted:
                            self.failed_latched = True
                            self.status = "failed_stop"
                else:
                    self.failed_latched = True
                    self.route_transition_pending = False
                    self.status = (
                        "local_avoidance_yield_to_peer_cav" if yield_reason
                        else "local_avoidance_unavailable_stop"
                    )
                    self.reason = yield_reason or "static_obstacle_local_avoidance_unavailable"
        self.stop_active = bool(
            self.failed_latched or transition_hold or cooldown_hold
        )
        return StaticObstacleResult(
            local_avoidance_active=bool(active),
            target_lane_id=self.target_lane_id,
            stop_active=bool(self.stop_active),
            status=str(self.status), reason=str(self.reason),
            candidate_id=str(self.candidate_id),
            candidate_since_s=float(self.candidate_since_s),
            route_transition_pending=bool(self.route_transition_pending),
        )

    def observe_behavior(self, decision: str, traffic_control_stop_active: bool) -> None:
        if self.target_lane_id is None:
            return
        if str(decision) in {"lane_change_left", "lane_change_right"}:
            self.status = "local_avoidance_executing"
        elif traffic_control_stop_active and str(decision) in {
            "stop_at_intersection", "stop_sign",
        }:
            self.status = "local_avoidance_preempted_by_traffic_control"
