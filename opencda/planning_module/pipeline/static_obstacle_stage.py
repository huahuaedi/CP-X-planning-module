"""Stateful static-obstacle behavior arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence


def select_local_avoidance_lane(*, current_lane_id: int,
                                available_lane_ids: Sequence[int],
                                lane_safety_scores: Mapping[int, float],
                                lane_prediction_risks: Mapping[int, Mapping[str, object]],
                                minimum_safety_score: float,
                                lane_to_offset: Mapping[int, int] = MappingProxyType({}),
                                excluded_lane_ids: Sequence[int] = (),
                                ) -> Optional[int]:
    current = int(current_lane_id)
    # AD-map lane ids are opaque -- |id_a - id_b| carries no lateral meaning
    # (see OpenCDAPlanningAdapter.build). Rank candidates by their real
    # corridor offset (0 = current lane, +/-1 = one lane over, ...) instead;
    # a lane missing from the map (no offset on file) is dropped rather than
    # silently treated as "very far" or "very near" by a meaningless id
    # subtraction. Ties in offset (opposite-side neighbors, both |1|) still
    # fall through to the safety-score comparison below, same as before.
    excluded = {int(x) for x in excluded_lane_ids}
    current_offset = int(lane_to_offset.get(current, 0))
    alternatives = sorted(
        (
            int(x) for x in available_lane_ids
            if int(x) not in {0, current}
            and int(x) not in excluded
            and int(x) in lane_to_offset
        ),
        key=lambda lane_id: abs(int(lane_to_offset[lane_id]) - current_offset),
    )
    if not alternatives:
        return None
    nearest_delta = abs(int(lane_to_offset[alternatives[0]]) - current_offset)
    safe = []
    for lane_id in alternatives:
        if abs(int(lane_to_offset[lane_id]) - current_offset) != nearest_delta:
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
        self.target_lane_match_frames = 0
        self.route_transition_pending = False
        self.last_replan_attempt_s = -float("inf")
        self.stop_active = False
        # Lanes a prior local-avoidance commitment stalled on (MPC never
        # found a feasible trajectory toward them) -- excluded from
        # re-selection so an abandoned target isn't immediately re-picked.
        # Cleared once the vehicle physically leaves the lane it was stuck
        # in, since that's the map context the failure was tied to.
        self._failed_target_lane_ids: set[int] = set()
        self._failed_target_lane_origin: Optional[int] = None

    def evaluate(self, *, requested: bool, traffic_control_stop_active: bool,
                 obstacle_id: str, current_lane_id: int,
                 lane_change_reference_active: bool, sim_time_s: float,
                 normal_mode: bool, available_lane_ids: Sequence[int],
                 lane_safety_scores: Mapping[int, float],
                 lane_prediction_risks: Mapping[int, Mapping[str, object]],
                 attempt_replan: Callable[[], tuple[bool, bool, str]],
                 lane_to_offset: Mapping[int, int] = MappingProxyType({}),
                 mpc_stall_failure_count: int = 0,
                 ) -> StaticObstacleResult:
        if (
            self._failed_target_lane_ids
            and int(current_lane_id) != int(self._failed_target_lane_origin or 0)
        ):
            self._failed_target_lane_ids = set()
            self._failed_target_lane_origin = None
        if (
            self.target_lane_id is not None
            and int(current_lane_id) == int(self.target_lane_id)
        ):
            self.target_lane_match_frames += 1
        else:
            self.target_lane_match_frames = 0
        release_frames = max(1, int(self.config.get(
            "static_obstacle_target_lane_release_frames", 3
        )))
        if (
            self.target_lane_id is not None
            and self.target_lane_match_frames >= release_frames
        ):
            self.target_lane_id = None
            self.target_lane_match_frames = 0
        active = self.target_lane_id is not None
        transition_hold = cooldown_hold = False
        # 100 (borrowed from the general lane_change stall watchdog) left a
        # vehicle stopped in/near a travel lane for 3+ seconds before this
        # stage would abandon a hopeless target: measured on a real replay,
        # failures accumulate at roughly 20/s once MPC is genuinely
        # infeasible (not transient solver jitter -- "primal infeasible"
        # held every single tick), so 100 is ~5s of exposure. A vehicle
        # stopped that long near a blocked lane is a standing collision
        # hazard for approaching traffic, not just a stalled maneuver in a
        # safe spot (confirmed: a car closing at ~10 m/s hit the stopped
        # ego at the 3.4s mark, well before the count reached 100). 30 is
        # ~1.5s -- long enough to ride out a brief hiccup, short enough to
        # matter for this hazard.
        stall_timeout_failures = int(self.config.get(
            "static_obstacle_mpc_stall_timeout_failures", 30
        ))
        stalled = bool(
            active
            and stall_timeout_failures > 0
            and int(mpc_stall_failure_count) >= stall_timeout_failures
        )
        if stalled:
            # MPC has reported "no feasible trajectory toward target_lane_id"
            # for stall_timeout_failures ticks in a row. Unlike a committed
            # lane_change maneuver (which LaneChangeLifecycleStage's own
            # stall watchdog can abandon), this stage previously only ever
            # released target_lane_id by *reaching* it -- a target picked
            # here that turns out to be geometrically infeasible held
            # forever, coasting the vehicle to a stop with no recovery.
            # Blacklist it (scoped to the lane the vehicle was stuck in, see
            # __init__) and fall through to the normal-mode selection below
            # so a different lane -- or a clean "no safe alternative"
            # outcome -- gets a chance the same tick.
            self._failed_target_lane_ids.add(int(self.target_lane_id))
            self._failed_target_lane_origin = int(current_lane_id)
            self.target_lane_id = None
            self.target_lane_match_frames = 0
            active = False
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
        elif active:
            # Target selection is a commitment, not a per-tick preference.
            # Re-running lane selection while the maneuver owns its reference
            # can momentarily reject the already selected lane and inject a
            # false stop into an otherwise valid lane change.
            self.failed_latched = False
            self.route_transition_pending = False
            self.status = (
                "local_avoidance_executing"
                if lane_change_reference_active
                else "local_avoidance_ready"
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
                    lane_to_offset=lane_to_offset,
                    excluded_lane_ids=self._failed_target_lane_ids,
                ) if normal_mode and bool(self.config.get(
                    "static_obstacle_local_avoidance_enabled", True)) else None
                if target is not None:
                    self.target_lane_id = int(target)
                    self.target_lane_match_frames = 0
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
                    self.status = "local_avoidance_unavailable_stop"
                    self.reason = "static_obstacle_local_avoidance_unavailable"
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
