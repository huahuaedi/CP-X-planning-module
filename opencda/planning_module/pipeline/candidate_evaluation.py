"""Candidate behavior evaluation for the planning pipeline.

This is the strategy layer between prediction and the rule-based FSM.  It
generates lane-level behavior candidates and scores them with safety, future
prediction risk, route alignment, and maneuver cost.  The first version is
intentionally lightweight: it evaluates candidate lanes before the final MPC
solve rather than solving a full MPC problem for every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


_DECISION_FOLLOW = "lane_follow"
_DECISION_CHANGE_LEFT = "lane_change_left"
_DECISION_CHANGE_RIGHT = "lane_change_right"


@dataclass
class BehaviorCandidate:
    name: str
    decision: str
    target_lane_id: int
    feasible: bool
    total_cost: float
    cost_terms: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class CandidateEvaluationFrame:
    candidates: List[BehaviorCandidate]
    selected: BehaviorCandidate

    def summary(self) -> str:
        selected = self.selected
        feasibility = "ok" if selected.feasible else "blocked"
        return (
            f"{selected.name}->L{int(selected.target_lane_id)} "
            f"cost={float(selected.total_cost):.2f} {feasibility}"
        )


def _risk_for_lane(
    lane_prediction_risks: Optional[Mapping[int, Mapping[str, object]]],
    lane_id: int,
) -> dict:
    if lane_prediction_risks is None:
        return {}
    return dict(lane_prediction_risks.get(int(lane_id), {}))


def _lane_change_decision(
    *,
    source_lane_id: int,
    target_lane_id: int,
    available_lane_ids: Sequence[int],
) -> str:
    ordered = [int(lane_id) for lane_id in list(available_lane_ids or []) if int(lane_id) != 0]
    if int(source_lane_id) not in ordered or int(target_lane_id) not in ordered:
        return _DECISION_FOLLOW
    if int(source_lane_id) == int(target_lane_id):
        return _DECISION_FOLLOW
    source_idx = ordered.index(int(source_lane_id))
    target_idx = ordered.index(int(target_lane_id))
    return _DECISION_CHANGE_LEFT if int(target_idx) > int(source_idx) else _DECISION_CHANGE_RIGHT


def _candidate_cost(
    *,
    target_lane_id: int,
    source_lane_id: int,
    route_optimal_lane_id: Optional[int],
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Optional[Mapping[int, Mapping[str, object]]],
    mpc_feedback_blocked_lane_ids: Optional[Sequence[int]],
    safety_weight: float,
    prediction_risk_weight: float,
    route_deviation_weight: float,
    lane_change_weight: float,
    mpc_feedback_weight: float,
    lane_progress_costs: Optional[Mapping[int, float]],
) -> Tuple[bool, Dict[str, float], str]:
    lane_score = max(0.0, min(1.0, float(lane_safety_scores.get(int(target_lane_id), 0.0))))
    prediction_risk = _risk_for_lane(lane_prediction_risks, int(target_lane_id))
    prediction_blocked = bool(prediction_risk.get("risk", False))
    mpc_feedback_blocked = (
        int(target_lane_id) != int(source_lane_id)
        and int(target_lane_id)
        in {int(lane_id) for lane_id in list(mpc_feedback_blocked_lane_ids or [])}
    )
    route_lane_id = int(route_optimal_lane_id or 0)
    route_distance = 0.0 if route_lane_id == 0 else abs(int(target_lane_id) - int(route_lane_id))
    lane_change_distance = 0.0 if int(target_lane_id) == int(source_lane_id) else abs(int(target_lane_id) - int(source_lane_id))
    collision_probability = max(
        0.0, min(1.0, float(prediction_risk.get("collision_probability", 1.0 if prediction_blocked else 0.0) or 0.0))
    )
    cost_terms = {
        "safety_cost": float(safety_weight) * (1.0 - float(lane_score)),
        "prediction_risk_cost": float(prediction_risk_weight) * float(collision_probability),
        "route_deviation_cost": float(route_deviation_weight) * float(route_distance),
        "lane_change_cost": float(lane_change_weight) * float(lane_change_distance),
        "mpc_feedback_cost": float(mpc_feedback_weight) if bool(mpc_feedback_blocked) else 0.0,
        "progress_cost": max(
            0.0,
            float((lane_progress_costs or {}).get(int(target_lane_id), 0.0)),
        ),
    }
    feasible = not bool(prediction_blocked or mpc_feedback_blocked)
    if prediction_blocked:
        reason = str(prediction_risk.get("reason", "prediction_risk") or "prediction_risk")
    elif mpc_feedback_blocked:
        reason = "recent_mpc_infeasible"
    else:
        reason = ""
    return bool(feasible), cost_terms, reason


def evaluate_behavior_candidates(
    *,
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Optional[Mapping[int, Mapping[str, object]]],
    ego_lane_id: int,
    selected_lane_id: int,
    available_lane_ids: Sequence[int],
    route_optimal_lane_id: Optional[int] = None,
    mode: str = "NORMAL",
    safety_weight: float = 10.0,
    prediction_risk_weight: float = 100.0,
    route_deviation_weight: float = 2.0,
    lane_change_weight: float = 1.0,
    mpc_feedback_blocked_lane_ids: Optional[Sequence[int]] = None,
    mpc_feedback_weight: float = 80.0,
    nearest_front_obstacles_by_lane: Optional[
        Mapping[int, Mapping[str, object]]
    ] = None,
    desired_speed_mps: float = 0.0,
    progress_cost_weight: float = 4.0,
    current_lane_unsafe_threshold: float = 0.5,
) -> CandidateEvaluationFrame:
    """Evaluate lane-level behavior candidates for one planning tick."""

    del mode
    lane_progress_costs: Dict[int, float] = {}
    normalized_desired_speed_mps = max(1.0, float(desired_speed_mps))
    normalized_progress_weight = max(0.0, float(progress_cost_weight))
    for lane_id, lead_obstacle in dict(
        nearest_front_obstacles_by_lane or {}
    ).items():
        lead_speed_mps = max(
            0.0, float(dict(lead_obstacle or {}).get("v", 0.0))
        )
        speed_deficit_ratio = max(
            0.0,
            min(
                1.0,
                (
                    float(normalized_desired_speed_mps)
                    - float(lead_speed_mps)
                )
                / float(normalized_desired_speed_mps),
            ),
        )
        lane_progress_costs[int(lane_id)] = float(
            normalized_progress_weight * speed_deficit_ratio
        )
    lanes = [int(lane_id) for lane_id in list(available_lane_ids or []) if int(lane_id) != 0]
    if len(lanes) == 0:
        lanes = [int(ego_lane_id)] if int(ego_lane_id) != 0 else [int(selected_lane_id)]

    source_lane_id = int(selected_lane_id or ego_lane_id or lanes[0])
    if int(source_lane_id) not in lanes:
        source_lane_id = int(ego_lane_id) if int(ego_lane_id) in lanes else int(lanes[0])

    # Record every visible lane so diagnostics can explain why a seemingly
    # attractive non-adjacent lane was rejected.  Only the source and its
    # immediate neighbours are executable in one maneuver.
    candidate_lane_ids = set(lanes)
    source_index = lanes.index(int(source_lane_id)) if int(source_lane_id) in lanes else 0

    candidates: List[BehaviorCandidate] = []
    for target_lane_id in sorted(candidate_lane_ids, key=lambda lane_id: (abs(int(lane_id) - int(source_lane_id)), int(lane_id))):
        decision = _lane_change_decision(
            source_lane_id=int(source_lane_id),
            target_lane_id=int(target_lane_id),
            available_lane_ids=lanes,
        )
        name = "keep_lane" if decision == _DECISION_FOLLOW else decision
        feasible, cost_terms, reason = _candidate_cost(
            target_lane_id=int(target_lane_id),
            source_lane_id=int(source_lane_id),
            route_optimal_lane_id=route_optimal_lane_id,
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
            mpc_feedback_blocked_lane_ids=mpc_feedback_blocked_lane_ids,
            safety_weight=float(safety_weight),
            prediction_risk_weight=float(prediction_risk_weight),
            route_deviation_weight=float(route_deviation_weight),
            lane_change_weight=float(lane_change_weight),
            mpc_feedback_weight=float(mpc_feedback_weight),
            lane_progress_costs=lane_progress_costs,
        )
        target_index = (
            lanes.index(int(target_lane_id))
            if int(target_lane_id) in lanes
            else int(source_index)
        )
        if (
            int(target_lane_id) != int(source_lane_id)
            and abs(int(target_index) - int(source_index)) != 1
        ):
            feasible = False
            reason = "not_adjacent"
        total_cost = float(sum(float(value) for value in cost_terms.values()))
        candidates.append(
            BehaviorCandidate(
                name=str(name),
                decision=str(decision),
                target_lane_id=int(target_lane_id),
                feasible=bool(feasible),
                total_cost=float(total_cost),
                cost_terms=dict(cost_terms),
                reason=str(reason),
            )
        )

    feasible_candidates = [candidate for candidate in candidates if candidate.feasible]
    selection_pool = feasible_candidates if len(feasible_candidates) > 0 else candidates
    source_candidate = next(
        (
            candidate
            for candidate in candidates
            if int(candidate.target_lane_id) == int(source_lane_id)
        ),
        None,
    )
    source_safety = max(
        0.0,
        min(1.0, float(lane_safety_scores.get(int(source_lane_id), 0.0))),
    )
    source_progress_cost = max(
        0.0, float(lane_progress_costs.get(int(source_lane_id), 0.0))
    )
    ranked_candidate = min(
        selection_pool,
        key=lambda candidate: (
            float(candidate.total_cost),
            0 if int(candidate.target_lane_id) == int(source_lane_id) else 1,
            abs(int(candidate.target_lane_id) - int(source_lane_id)),
        ),
    )
    # A safe current lane is the stable default.  Route-required changes are
    # supplied explicitly by RouteAuthorization; this generic evaluator must
    # not turn a small route-deviation cost into an unsolicited lane change.
    # It may leave the source lane only for an actual safety deficit or when
    # a slow lead creates enough progress loss for another feasible lane to
    # have a lower total cost.
    may_leave_source = bool(
        float(source_safety) < float(current_lane_unsafe_threshold)
        or (
            float(source_progress_cost) > 0.0
            and int(ranked_candidate.target_lane_id) != int(source_lane_id)
            and float(ranked_candidate.total_cost)
            < float(source_candidate.total_cost if source_candidate is not None else float("inf"))
        )
    )
    selected = (
        ranked_candidate
        if bool(may_leave_source) or source_candidate is None
        else source_candidate
    )
    return CandidateEvaluationFrame(candidates=list(candidates), selected=selected)
