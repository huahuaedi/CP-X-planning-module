"""Candidate behavior evaluation for the planning pipeline.

This is the strategy layer between prediction and the rule-based FSM.  It
generates lane-level behavior candidates and scores them with safety, future
prediction risk, route alignment, and maneuver cost.  The first version is
intentionally lightweight: it evaluates candidate lanes before the final MPC
solve rather than solving a full MPC problem for every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
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


@dataclass(frozen=True)
class CandidateSelectionResult:
    """Immutable candidate-stage output consumed by the execution pipeline."""

    decision: str
    target_lane_id: int
    target_speed_mps: float
    reference: tuple
    destination_state: tuple
    diagnostics: Mapping

    def mutable_reference(self):
        return [dict(sample) for sample in self.reference]

    def mutable_destination_state(self):
        return list(self.destination_state)

    def mutable_diagnostics(self):
        return dict(self.diagnostics)


@dataclass(frozen=True)
class CandidateGeometryPlan:
    """Single source for candidate sampling and lane-change geometry limits."""

    step_m: float
    geometry_speed_mps: float = 0.0
    geometry_length_m: float = 0.0
    operational_curvature_limit_1pm: float = 0.0


class CandidateTrajectoryEvaluator:
    """Own cross-tick risk hysteresis for geometric trajectory candidates."""

    def __init__(
        self,
        *,
        mpc_probe_enabled: bool = True,
        mpc_probe_top_k: int = 2,
        mpc_probe_interval_s: float = 0.2,
    ) -> None:
        self._risk_bucket_by_name: Dict[str, str] = {}
        self._mpc_probe_enabled = bool(mpc_probe_enabled)
        self._mpc_probe_top_k = max(2, int(mpc_probe_top_k))
        self._mpc_probe_interval_s = max(0.05, float(mpc_probe_interval_s))
        self._mpc_probe_last_time_s = -float("inf")
        self._mpc_probe_cache = {}

    def reset(self) -> None:
        self._risk_bucket_by_name.clear()
        self._mpc_probe_last_time_s = -float("inf")
        self._mpc_probe_cache.clear()

    @staticmethod
    def geometry_plan(
        *, decision, ego_speed_mps, target_speed_mps,
        lane_change_duration_s, dt_s, lane_width_m, config,
    ) -> CandidateGeometryPlan:
        """Resolve sampling once; generation and commitment consume it."""

        from .candidate_pipeline import (
            lane_change_geometry_requirements,
            lane_change_operational_curvature_limit_1pm,
        )

        step_m = max(
            float(config.get("route_tracking_min_step_m", 0.10)),
            float(dt_s) * max(
                0.5, float(ego_speed_mps), abs(float(target_speed_mps))
            ),
        )
        if str(decision) not in {
            _DECISION_CHANGE_LEFT,
            _DECISION_CHANGE_RIGHT,
        }:
            return CandidateGeometryPlan(step_m=float(step_m))
        curvature_limit = lane_change_operational_curvature_limit_1pm(
            planning_speed_mps=float(target_speed_mps),
            lateral_accel_limit_mps2=float(
                config.get(
                    "route_tracking_lane_change_lateral_accel_limit_mps2",
                    1.3,
                )
            ),
            vehicle_max_curvature_1pm=float(
                config.get("reference_vehicle_max_curvature_1pm", 0.35)
            ),
            minimum_speed_mps=float(
                config.get("lane_change_min_geometry_speed_mps", 2.0)
            ),
        )
        geometry_speed_mps, geometry_length_m, geometry_step_m = (
            lane_change_geometry_requirements(
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(target_speed_mps),
                duration_s=float(lane_change_duration_s or 4.0),
                dt_s=float(dt_s),
                lane_width_m=float(lane_width_m),
                max_curvature_1pm=float(curvature_limit),
                minimum_geometry_speed_mps=float(
                    config.get("lane_change_min_geometry_speed_mps", 2.0)
                ),
                minimum_length_m=float(
                    config.get("lane_change_min_length_m", 10.0)
                ),
                acceleration_limit_mps2=float(
                    config.get(
                        "lane_change_planning_acceleration_limit_mps2", 2.0
                    )
                ),
            )
        )
        return CandidateGeometryPlan(
            step_m=max(float(step_m), float(geometry_step_m)),
            geometry_speed_mps=float(geometry_speed_mps),
            geometry_length_m=float(geometry_length_m),
            operational_curvature_limit_1pm=float(curvature_limit),
        )

    def evaluate(
        self,
        *,
        candidate,
        ego_state,
        object_snapshots,
        prediction_trajectories,
        current_lane_id: int,
        min_object_distance_m: float,
        risk_hysteresis_margin_m: float,
    ):
        """Evaluate and remember exactly one candidate's risk class."""

        from .candidate_pipeline import evaluate_candidate_reference

        name = str(candidate.intent.name)
        result = evaluate_candidate_reference(
            candidate=candidate,
            ego_state=ego_state,
            object_snapshots=object_snapshots,
            prediction_trajectories=prediction_trajectories,
            current_lane_id=int(current_lane_id),
            min_object_distance_m=float(min_object_distance_m),
            previous_risk_bucket=str(self._risk_bucket_by_name.get(name, "")),
            risk_hysteresis_margin_m=float(risk_hysteresis_margin_m),
        )
        self._risk_bucket_by_name[name] = str(result.risk_bucket)
        return result

    def condition_and_evaluate(
        self,
        *,
        intent,
        destination_state,
        reference_samples,
        reference_debug,
        reference_pipeline,
        current_state,
        ego_location,
        ego_yaw_rad,
        ego_speed_mps,
        target_speed_mps,
        behavior_decision,
        behavior_fsm_state,
        current_lane_id,
        target_lane_id,
        stop_goal_active,
        stop_target,
        route_points,
        object_snapshots,
        prediction_trajectories,
        min_object_distance_m,
        risk_hysteresis_margin_m,
    ):
        """Own the condition -> contract -> risk boundary for one candidate."""

        from .candidate_pipeline import CandidateReferenceResult
        from .reference_pipeline import ReferencePipelineRequest

        conditioned = reference_pipeline.condition(ReferencePipelineRequest(
            destination_state=destination_state,
            reference_samples=reference_samples,
            current_state=current_state,
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            target_speed_mps=float(target_speed_mps),
            behavior_decision=str(behavior_decision),
            behavior_fsm_state=str(behavior_fsm_state),
            current_lane_id=int(current_lane_id),
            target_lane_id=int(target_lane_id),
            stop_goal_active=bool(stop_goal_active),
            stop_target=stop_target if isinstance(stop_target, Mapping) else None,
            route_points=route_points,
        ))
        destination = list(conditioned.destination_state)
        reference = [
            dict(sample) for sample in conditioned.reference_samples
        ]
        diagnostics = dict(reference_debug or {})
        diagnostics["mpc_reference_stabilizer_reason"] = str(
            conditioned.reason
        )
        candidate = CandidateReferenceResult(
            intent=intent,
            destination_state=destination,
            lane_center_reference=reference,
            reference_debug=diagnostics,
            contract_result=conditioned.validation,
        )
        evaluated = self.evaluate(
            candidate=candidate,
            ego_state=current_state,
            object_snapshots=object_snapshots,
            prediction_trajectories=prediction_trajectories,
            current_lane_id=int(current_lane_id),
            min_object_distance_m=float(min_object_distance_m),
            risk_hysteresis_margin_m=float(risk_hysteresis_margin_m),
        )
        return evaluated, destination, reference

    @staticmethod
    def lane_change_state(
        *, decision: str, baseline_decision: str, baseline_lane_change_state: str
    ) -> str:
        if str(decision) == str(baseline_decision):
            return str(baseline_lane_change_state or "LANE_KEEP")
        return {
            "intersection_turn_left": "INTERSECTION_TURN_LEFT",
            "intersection_turn_right": "INTERSECTION_TURN_RIGHT",
            "lane_change_left": "EXECUTE_LANE_CHANGE_LEFT",
            "lane_change_right": "EXECUTE_LANE_CHANGE_RIGHT",
        }.get(str(decision), "LANE_KEEP")

    def probe_mpc(
        self,
        *,
        candidate_results,
        mpc,
        sim_time_s: float,
        current_state,
        object_snapshots,
        current_acceleration_mps2: float,
        current_steering_rad: float,
        road_envelope_payload_world=None,
        required_decision: str = "",
        required_target_lane_id: int = 0,
    ) -> str:
        """Probe the bounded top-k and own its refresh cache across ticks."""

        from .candidate_pipeline import apply_mpc_probe_result, mark_mpc_probe_skipped

        rows = list(candidate_results or [])
        lane_changes = [
            row for row in rows
            if str(getattr(getattr(row, "intent", None), "decision", "")).startswith("lane_change")
            and bool(getattr(row, "feasible", False))
        ]
        if not self._mpc_probe_enabled or not lane_changes:
            return "mpc_probe_not_applicable"
        feasible = [row for row in rows if bool(getattr(row, "feasible", False))]
        feasible.sort(key=lambda row: float(getattr(row, "total_cost", float("inf"))))
        keep = [
            row for row in feasible
            if str(getattr(getattr(row, "intent", None), "decision", "")) == "lane_follow"
        ]
        required = [
            row for row in lane_changes
            if str(getattr(getattr(row, "intent", None), "decision", "")).strip().lower()
            == str(required_decision).strip().lower()
            and int(getattr(getattr(row, "intent", None), "target_lane_id", 0))
            == int(required_target_lane_id or 0)
        ]
        selected = []
        if required:
            priority = {"normal": 0, "assertive": 1, "conservative": 2}
            required.sort(key=lambda row: (
                priority.get(str(getattr(row.intent, "trajectory_variant", "")).strip().lower(), 3),
                float(getattr(row, "total_cost", float("inf"))),
            ))
            selected.extend(required)
        else:
            if keep:
                selected.append(keep[0])
            lane_changes.sort(key=lambda row: float(getattr(row, "total_cost", float("inf"))))
            selected.append(lane_changes[0])
        for row in feasible:
            if row not in selected and len(selected) < self._mpc_probe_top_k:
                selected.append(row)
        selected = selected[:self._mpc_probe_top_k]
        order = {id(row): index for index, row in enumerate(selected)}
        feasible.sort(key=lambda row: (
            0 if id(row) in order else 1,
            order.get(id(row), len(order)),
            float(getattr(row, "total_cost", float("inf"))),
        ))
        if float(sim_time_s) - self._mpc_probe_last_time_s >= self._mpc_probe_interval_s:
            self._mpc_probe_cache.clear()
            self._mpc_probe_last_time_s = float(sim_time_s)

        selected_ids = {id(row) for row in selected}
        previous_profile = str(getattr(mpc, "active_cost_profile_name", "lane_follow"))
        summaries = []
        for row in feasible:
            if id(row) not in selected_ids:
                mark_mpc_probe_skipped(row)
                continue
            intent = row.intent
            cache_key = (
                str(intent.name), str(intent.decision), int(intent.target_lane_id),
                str(intent.trajectory_variant), round(float(intent.lane_change_duration_s or 0.0), 2),
            )
            probe = self._mpc_probe_cache.get(cache_key)
            if probe is None:
                profile = mpc_cost_profile_for_behavior(
                    behavior=str(intent.decision),
                    planner_lc_state=(
                        "EXECUTE_LANE_CHANGE" if str(intent.decision).startswith("lane_change")
                        else "LANE_KEEP"
                    ),
                    planner_mode="NORMAL",
                    next_macro_maneuver="straight",
                )
                if hasattr(mpc, "apply_mode_cost_profile"):
                    mpc.apply_mode_cost_profile(profile, blend_alpha=1.0)
                probe = mpc.probe_trajectory_feasibility(
                    current_state=current_state,
                    destination_state=list(row.destination_state or []),
                    object_snapshots=object_snapshots,
                    current_acceleration_mps2=float(current_acceleration_mps2),
                    current_steering_rad=float(current_steering_rad),
                    lane_center_reference_samples=[dict(v) for v in list(row.lane_center_reference or [])],
                    stop_goal_active=bool(intent.stop_goal_active),
                    road_envelope_payload_world=(
                        road_envelope_payload_world
                        if str(intent.name) == "committed_lane_change_continuation"
                        else None
                    ),
                )
                self._mpc_probe_cache[cache_key] = dict(probe)
            apply_mpc_probe_result(
                candidate=row,
                solved=bool(probe.get("solved", False)),
                status=str(probe.get("status", "")),
                solve_time_ms=float(probe.get("solve_time_ms", 0.0) or 0.0),
                dynamic_cost=float(probe.get("dynamic_cost", 0.0) or 0.0),
            )
            summaries.append("%s:%s" % (str(intent.name), str(probe.get("status", ""))))
        if hasattr(mpc, "apply_mode_cost_profile"):
            mpc.apply_mode_cost_profile(previous_profile, blend_alpha=1.0)
        return "|".join(summaries) if summaries else "mpc_probe_no_feasible_top_k"

    @staticmethod
    def select_with_commitment(
        *,
        candidate_results,
        lane_change_phase: str,
        lane_change_option: str,
        source_lane_id: int,
        target_lane_id: int,
        progress: float,
        reference_locked: bool,
        required_decision: str = "",
        required_target_lane_id: int = 0,
    ):
        """Resolve the final candidate against one immutable commitment view."""

        from .candidate_pipeline import (
            select_best_candidate,
            select_candidate_with_commitment,
        )
        from .stage_contracts import ManeuverCommitment

        committed_decision = {
            "CHANGELANELEFT": "lane_change_left",
            "CHANGELANERIGHT": "lane_change_right",
        }.get(str(lane_change_option), "")
        commitment = ManeuverCommitment(
            state=(
                "STABILIZING"
                if str(lane_change_phase) == "target_lane_stabilization"
                else "COMMITTED" if bool(reference_locked) else "IDLE"
            ),
            decision=str(committed_decision),
            source_lane_id=int(source_lane_id),
            target_lane_id=int(target_lane_id),
            progress=float(progress),
            reference_locked=bool(reference_locked),
        )
        outcome = select_candidate_with_commitment(
            candidate_results,
            commitment=commitment,
            required_decision=str(required_decision),
            required_target_lane_id=int(required_target_lane_id),
        )
        selected = (
            outcome.selected
            if outcome.selected is not None
            else select_best_candidate(candidate_results)
        )
        return commitment, outcome, selected

    @staticmethod
    def finalize_selection(
        *, selected, candidate_results, selected_reference,
        selected_destination, selected_debug, prediction_trajectory_count,
        probe_summary, selection_outcome, commitment,
        commitment_release_reason, lane_change_phase,
        lane_change_stabilization_frames, completion_debug,
    ) -> CandidateSelectionResult:
        """Publish one selected candidate with a stable diagnostic schema."""

        from .candidate_pipeline import summarize_candidate_results

        diagnostics = dict(selected_debug or {})
        diagnostics.update({
            "stage": diagnostics.get("reference_pipeline_stage", ""),
            "intent_mode": diagnostics.get(
                "reference_pipeline_intent_mode", ""
            ),
            "fallback_reason": diagnostics.get("fallback_reason", ""),
            "reference_source": str(
                diagnostics.get(
                    "reference_source", "candidate_reference_pipeline"
                )
            ),
            "candidate_pipeline_selected": str(selected.intent.name),
            "candidate_pipeline_selected_status": str(
                selected.feasibility_status
            ),
            "candidate_pipeline_selected_reason": str(
                selected.feasibility_reason
            ),
            "candidate_selected_stop_goal_active": bool(
                selected.intent.stop_goal_active
            ),
            "candidate_pipeline_count": int(len(candidate_results)),
            "candidate_prediction_trajectory_count": int(
                prediction_trajectory_count
            ),
            "candidate_pipeline_summary": summarize_candidate_results(
                candidate_results
            ),
            "candidate_mpc_probe_summary": str(probe_summary),
            "candidate_selected_decision": str(selected.intent.decision),
            "candidate_selected_lane_id": int(selected.intent.target_lane_id),
            "candidate_selected_cost": float(selected.total_cost),
            "candidate_evaluation_summary": (
                "%s->%s:L%d cost=%.2f"
                % (
                    str(selected.intent.name),
                    str(selected.intent.decision),
                    int(selected.intent.target_lane_id),
                    float(selected.total_cost),
                )
            ),
            "candidate_selection_status": str(selection_outcome.status),
            "candidate_selection_reason": str(selection_outcome.reason),
            "lane_change_commitment_release_reason": str(
                commitment_release_reason
            ),
            "lane_change_phase": str(lane_change_phase),
            "lane_change_stabilization_frames": int(
                lane_change_stabilization_frames
            ),
        })
        diagnostics.update(commitment.as_debug_fields())
        diagnostics.update(dict(completion_debug or {}))
        return CandidateSelectionResult(
            decision=str(selected.intent.decision),
            target_lane_id=int(selected.intent.target_lane_id),
            target_speed_mps=float(selected.intent.target_speed_mps),
            reference=tuple(
                MappingProxyType(dict(sample))
                for sample in list(selected_reference or [])
            ),
            destination_state=tuple(selected_destination or ()),
            diagnostics=MappingProxyType(diagnostics),
        )


def mpc_cost_profile_for_behavior(
    *, behavior: str, planner_lc_state: str, planner_mode: str,
    next_macro_maneuver: str
) -> str:
    from opencda.planning_module.behavior_planner import (
        is_emergency_brake_decision,
        is_fixed_stop_decision,
        normalize_behavior_decision,
    )

    raw = str(behavior or "").strip().lower()
    normalized = str(normalize_behavior_decision(behavior))
    state = str(planner_lc_state or "").strip().upper()
    if is_fixed_stop_decision(normalized):
        return "stop"
    if is_emergency_brake_decision(normalized):
        return "recovery"
    if state.startswith("PREPARE_LANE_CHANGE"):
        return "prepare_lane_change"
    if raw in {"intersection_turn_left", "intersection_turn_right"}:
        return "intersection_turn"
    if state.startswith("EXECUTE_LANE_CHANGE") or normalized in {"lane_change_left", "lane_change_right"}:
        return "execute_lane_change"
    if str(planner_mode or "").strip().upper() == "INTERSECTION" and str(next_macro_maneuver or "straight").strip().lower() in {"left", "right"}:
        return "intersection_turn"
    return "lane_follow"


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
