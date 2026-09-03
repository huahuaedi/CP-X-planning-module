"""Single owner for candidate construction, probing and commitment selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from .candidate_evaluation import CandidateSelectionResult
from .candidate_pipeline import build_candidate_intents, summarize_candidate_results
from .reference_line_provider import LANE_CHANGE
from .speed_planner import SpeedConstraint
from opencda.planning_module.utility.speed_profile import curvature_speed_cap_mps


@dataclass(frozen=True)
class CandidateSelectionRequest:
    intents: Sequence[Any]
    reference_context: Any
    baseline_lane_change_state: str
    baseline_decision: str
    baseline_target_lane_id: int
    baseline_speed_mps: float
    baseline_reference: Sequence[Mapping[str, Any]]
    baseline_destination_state: Sequence[float]
    current_state: Sequence[float]
    current_lane_id: int
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    object_snapshots: Sequence[Mapping[str, Any]]
    prediction_trajectories: Mapping[str, Any]
    current_acceleration_mps2: float = 0.0
    current_steering_rad: float = 0.0
    required_decision: str = ""
    required_target_lane_id: int = 0


@dataclass(frozen=True)
class CandidatePostSelectionResult:
    decision: str
    lane_change_state: str
    stop_goal_active: bool
    speed_plan: Any
    speed_constraints: tuple
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateArbitrationRequest:
    """Complete immutable input for one candidate-arbitration cycle."""

    reference_context: Any
    selected_decision: str
    selected_target_lane_id: int
    current_lane_id: int
    target_speed_mps: float
    candidate_lane_ids: Sequence[int]
    lane_safety_scores: Mapping[int, float]
    lane_prediction_risks: Mapping[int, Mapping[str, object]]
    stop_goal_active: bool
    traffic_stop_active: bool
    lane_change_authorization: Any
    opportunistic_lane_change_allowed: bool
    stop_target: Any
    local_obstacle_avoidance_active: bool
    ego_speed_mps: float
    lane_width_m: float
    baseline_lane_change_state: str
    current_state: Sequence[float]
    ego_location: Any
    ego_yaw_rad: float
    object_snapshots: Sequence[Mapping[str, Any]]
    prediction_trajectories: Mapping[str, Any]
    current_acceleration_mps2: float
    current_steering_rad: float
    route_required: bool
    scenario_stop_required: bool
    speed_plan: Any
    turn_prepare_speed_suppressed: bool


@dataclass(frozen=True)
class CandidateArbitrationResult:
    decision: str
    target_lane_id: int
    target_speed_mps: float
    reference: tuple
    destination_state: tuple
    lane_change_state: str
    stop_goal_active: bool
    speed_plan: Any
    speed_constraints: tuple
    diagnostics: Mapping[str, Any]

    def mutable_reference(self):
        return [dict(sample) for sample in self.reference]

    def mutable_destination_state(self):
        return list(self.destination_state)


class CandidateSelectionStage:
    """Own the complete candidate lifecycle for one planning tick."""

    def __init__(
        self, *, evaluator: Any, provider: Any, maneuver_manager: Any,
        reference_pipeline: Any, fallback_manager: Any, static_obstacle_stage: Any,
        mpc: Any, config: Mapping[str, Any], map_epoch: str,
        normal_clearance_m: float, static_clearance_m: float,
        risk_hysteresis_margin_m: float, strict_ownership: bool,
        target_speed_mps: float,
    ) -> None:
        self._evaluator = evaluator
        self._provider = provider
        self._maneuver = maneuver_manager
        self._reference_pipeline = reference_pipeline
        self._fallback = fallback_manager
        self._static_obstacle = static_obstacle_stage
        self._mpc = mpc
        self._config = config
        self._map_epoch = str(map_epoch)
        self._normal_clearance_m = float(normal_clearance_m)
        self._static_clearance_m = float(static_clearance_m)
        self._risk_hysteresis_margin_m = float(risk_hysteresis_margin_m)
        self._strict_ownership = bool(strict_ownership)
        self._target_speed_mps = float(target_speed_mps)
        self._lane_change_lifecycle = None

    def build_intents(
        self, *, selected_decision: str, selected_target_lane_id: int,
        current_lane_id: int, target_speed_mps: float,
        candidate_lane_ids: Sequence[int], lane_safety_scores: Mapping[int, float],
        lane_prediction_risks: Mapping[int, Mapping[str, object]],
        stop_goal_active: bool, traffic_stop_active: bool,
        lane_change_authorized: bool, lane_change_target_lane_id: int,
        lane_change_authorization_source: str,
        lane_change_authorization_direction: str,
        local_obstacle_avoidance_active: bool, stop_target: Any,
        ego_speed_mps: float, lane_width_m: float,
        lane_change_available_distance_m: Any,
    ) -> list[Any]:
        """Construct the complete candidate set from one frozen behavior frame."""

        cfg = self._config
        return build_candidate_intents(
            selected_decision=str(selected_decision),
            selected_target_lane_id=int(selected_target_lane_id),
            current_lane_id=int(current_lane_id),
            target_speed_mps=float(target_speed_mps),
            candidate_lane_ids=list(candidate_lane_ids),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
            stop_goal_active=bool(stop_goal_active),
            traffic_stop_active=bool(traffic_stop_active),
            lane_change_authorized=bool(lane_change_authorized),
            lane_change_authorized_target_lane_id=int(lane_change_target_lane_id),
            allow_lane_change_candidates=bool(lane_change_authorized),
            stop_target=(dict(stop_target) if isinstance(stop_target, Mapping) else None),
            lane_change_assertive_duration_s=max(
                0.1, float(cfg.get("candidate_lane_change_assertive_duration_s", 3.2))
            ),
            lane_change_normal_duration_s=max(
                0.1, float(cfg.get("candidate_lane_change_normal_duration_s", 4.0))
            ),
            lane_change_conservative_duration_s=max(
                0.1, float(cfg.get("candidate_lane_change_conservative_duration_s", 5.5))
            ),
            lane_change_assertive_speed_scale=max(
                0.1, float(cfg.get("candidate_lane_change_assertive_speed_scale", 1.0))
            ),
            lane_change_normal_speed_scale=max(
                0.1, float(cfg.get("candidate_lane_change_normal_speed_scale", 0.9))
            ),
            lane_change_conservative_speed_scale=max(
                0.1, float(cfg.get("candidate_lane_change_conservative_speed_scale", 0.7))
            ),
            lane_change_authorization_source=str(lane_change_authorization_source),
            lane_change_authorization_direction=str(lane_change_authorization_direction),
            lane_change_defer_cost=float(cfg.get("candidate_lane_change_defer_cost", 10.0)),
            turn_obstacle_stop_defer_cost=float(
                cfg.get("candidate_turn_obstacle_stop_defer_cost", 90.0)
            ),
            local_obstacle_avoidance_active=bool(local_obstacle_avoidance_active),
            local_obstacle_stop_defer_cost=float(
                cfg.get("candidate_local_obstacle_stop_defer_cost", 25.0)
            ),
            human_like_lane_change_enabled=bool(
                cfg.get("human_like_lane_change_enabled", True)
            ),
            ego_speed_mps=float(ego_speed_mps), lane_width_m=float(lane_width_m),
            lane_change_available_distance_m=lane_change_available_distance_m,
            human_lane_change_min_duration_s=float(
                cfg.get("human_lane_change_min_duration_s", 3.0)
            ),
            human_lane_change_max_duration_s=float(
                cfg.get("human_lane_change_max_duration_s", 6.5)
            ),
        )

    def arbitrate(
        self,
        request: CandidateArbitrationRequest,
        *,
        sim_time_s: float,
        route_revision: str,
        road_envelope: Callable[[], Any],
        validate_contract: Callable[..., Any],
        validate_locked_reference: Callable[..., Any],
    ) -> CandidateArbitrationResult:
        """Build, select and finalize candidates through one stage boundary."""

        authorization = request.lane_change_authorization
        behavior_lane_change = str(request.selected_decision) in {
            "lane_change_left", "lane_change_right",
        }
        opportunistic_authorized = bool(
            behavior_lane_change
            and request.opportunistic_lane_change_allowed
            and not bool(authorization.allowed)
        )
        lane_change_authorized = bool(
            authorization.allowed or opportunistic_authorized
        )
        target_lane_id = int(
            authorization.target_lane_id
            if bool(authorization.allowed)
            else request.selected_target_lane_id
        )
        direction = (
            str(authorization.direction or "")
            if bool(authorization.allowed)
            else "left" if str(request.selected_decision) == "lane_change_left"
            else "right" if str(request.selected_decision) == "lane_change_right"
            else ""
        )
        intents = self.build_intents(
            selected_decision=str(request.selected_decision),
            selected_target_lane_id=int(request.selected_target_lane_id),
            current_lane_id=int(request.current_lane_id),
            target_speed_mps=float(request.target_speed_mps),
            candidate_lane_ids=request.candidate_lane_ids,
            lane_safety_scores=request.lane_safety_scores,
            lane_prediction_risks=request.lane_prediction_risks,
            stop_goal_active=bool(
                request.stop_goal_active or request.scenario_stop_required
            ),
            traffic_stop_active=bool(request.traffic_stop_active),
            lane_change_authorized=bool(lane_change_authorized),
            lane_change_target_lane_id=int(target_lane_id),
            stop_target=request.stop_target,
            lane_change_authorization_source=(
                "route" if bool(authorization.allowed) else "opportunistic"
            ),
            lane_change_authorization_direction=str(direction),
            local_obstacle_avoidance_active=bool(
                request.local_obstacle_avoidance_active
            ),
            ego_speed_mps=float(request.ego_speed_mps),
            lane_width_m=float(request.lane_width_m),
            lane_change_available_distance_m=(
                authorization.distance_to_maneuver_m
                if bool(lane_change_authorized) else None
            ),
        )
        required_decision = (
            "lane_change_left"
            if request.route_required and bool(authorization.allowed)
            and str(authorization.direction).strip().lower() == "left"
            else "lane_change_right"
            if request.route_required and bool(authorization.allowed)
            and str(authorization.direction).strip().lower() == "right"
            else ""
        )
        selected = self.run(
            CandidateSelectionRequest(
                intents=intents,
                reference_context=request.reference_context,
                baseline_lane_change_state=str(request.baseline_lane_change_state),
                baseline_decision=str(request.selected_decision),
                baseline_target_lane_id=int(request.selected_target_lane_id),
                baseline_speed_mps=float(request.target_speed_mps),
                baseline_reference=request.reference_context.baseline_reference,
                baseline_destination_state=list(
                    request.reference_context.baseline_destination_state or []
                ),
                current_state=request.current_state,
                current_lane_id=int(request.current_lane_id),
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                ego_speed_mps=float(request.ego_speed_mps),
                object_snapshots=request.object_snapshots,
                prediction_trajectories=request.prediction_trajectories,
                current_acceleration_mps2=float(
                    request.current_acceleration_mps2
                ),
                current_steering_rad=float(request.current_steering_rad),
                required_decision=str(required_decision),
                required_target_lane_id=(
                    int(authorization.target_lane_id)
                    if required_decision else 0
                ),
            ),
            sim_time_s=float(sim_time_s),
            route_revision=str(route_revision),
            road_envelope=road_envelope,
            validate_contract=validate_contract,
            validate_locked_reference=validate_locked_reference,
        )
        final = self.finalize_selected_frame(
            decision=str(selected.decision),
            lane_change_state=str(request.baseline_lane_change_state),
            reference=selected.reference,
            selected_diagnostics=selected.diagnostics,
            reference_diagnostics=request.reference_context.baseline_debug,
            ego_speed_mps=float(request.ego_speed_mps),
            scenario_stop_required=bool(request.scenario_stop_required),
            speed_plan=request.speed_plan,
            turn_prepare_speed_suppressed=bool(
                request.turn_prepare_speed_suppressed
            ),
        )
        return CandidateArbitrationResult(
            decision=str(selected.decision),
            target_lane_id=int(selected.target_lane_id),
            target_speed_mps=float(selected.target_speed_mps),
            reference=tuple(dict(x) for x in selected.reference),
            destination_state=tuple(selected.destination_state),
            lane_change_state=str(final.lane_change_state),
            stop_goal_active=bool(final.stop_goal_active),
            speed_plan=final.speed_plan,
            speed_constraints=tuple(final.speed_constraints),
            diagnostics=dict(final.diagnostics),
        )

    def finalize_selected_frame(
        self, *, decision: str, lane_change_state: str,
        reference: Sequence[Mapping[str, Any]], selected_diagnostics: Mapping[str, Any],
        reference_diagnostics: Mapping[str, Any], ego_speed_mps: float,
        scenario_stop_required: bool, speed_plan: Any,
        turn_prepare_speed_suppressed: bool,
    ) -> CandidatePostSelectionResult:
        """Interpret the winning candidate exactly once for downstream stages."""

        cfg = self._config
        debug = dict(reference_diagnostics or {})
        selected_debug = dict(selected_diagnostics or {})
        constraints = []
        if str(decision) in {"lane_change_left", "lane_change_right"}:
            curvature = float(self._provider.builder.discrete_curvature_1pm(reference))
            advisory = curvature_speed_cap_mps(
                curve_curvature_abs=curvature,
                curve_min_curvature=max(0.0, float(cfg.get(
                    "full_lane_change_curvature_min_curvature_1pm", 0.002
                ))),
                current_speed_mps=float(ego_speed_mps),
                curve_lateral_accel_limit_mps2=max(0.1, float(cfg.get(
                    "route_tracking_lane_change_lateral_accel_limit_mps2", 1.3
                ))),
                speed_enable_threshold_mps=0.0,
            )
            debug.update({
                "lane_change_reference_curvature_1pm": curvature,
                "lane_change_curvature_speed_advisory_mps": (
                    "" if advisory is None else float(advisory)
                ),
                "lane_change_longitudinal_authority": "SpeedPlanner",
            })
        elif str(decision) in {"intersection_turn_left", "intersection_turn_right"}:
            curvature = float(self._provider.builder.discrete_curvature_1pm(reference))
            advisory = curvature_speed_cap_mps(
                curve_curvature_abs=curvature,
                curve_min_curvature=max(0.0, float(cfg.get(
                    "full_intersection_turn_curvature_min_curvature_1pm", 0.01
                ))),
                current_speed_mps=float(ego_speed_mps),
                curve_lateral_accel_limit_mps2=max(0.1, float(cfg.get(
                    "full_intersection_turn_lateral_accel_comfort_mps2", 2.5
                ))),
                speed_enable_threshold_mps=0.0,
            )
            turn_cap_mps = max(0.1, float(cfg.get(
                "full_intersection_turn_speed_cap_mps", 2.2
            )))
            debug.update({
                "turn_reference_curvature_1pm": curvature,
                "turn_curvature_speed_advisory_mps": (
                    "" if advisory is None else float(advisory)
                ),
                "turn_longitudinal_authority": "SpeedPlanner",
            })
            constraint = SpeedConstraint(
                owner="selected_turn_cap", maximum_mps=turn_cap_mps,
                reason="selected_candidate_turn_speed_cap",
            )
            constraints.append(constraint)
            if turn_cap_mps < float(speed_plan.target_speed_mps):
                speed_plan = replace(
                    speed_plan, target_speed_mps=turn_cap_mps,
                    speed_cap_mps=turn_cap_mps, turn_cap_mps=turn_cap_mps,
                    limiting_owner="selected_turn_cap",
                    active_constraints=tuple(speed_plan.active_constraints)
                    + ("selected_turn_cap",),
                    external_constraints=tuple(speed_plan.external_constraints)
                    + (constraint,),
                )

        stop_goal = bool(
            scenario_stop_required
            or selected_debug.get("candidate_selected_stop_goal_active", False)
            or str(decision) in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        )
        phase = str(lane_change_state or "LANE_KEEP")
        if str(decision) in {"lane_follow", "stop_at_intersection", "stop_sign", "emergency_brake"}:
            phase = "LANE_KEEP"
        elif str(decision) in {"lane_change_left", "lane_change_right"}:
            phase = self.normalized_lane_change_state(
                decision=str(decision), lane_change_phase=str(
                    selected_debug.get("lane_change_phase", "")
                ),
            )
        debug.update(selected_debug)
        debug["candidate_pipeline_enabled"] = True
        debug["turn_prepare_speed_suppressed_by_lane_change"] = bool(
            turn_prepare_speed_suppressed
        )
        return CandidatePostSelectionResult(
            decision=str(decision), lane_change_state=phase,
            stop_goal_active=stop_goal, speed_plan=speed_plan,
            speed_constraints=tuple(constraints), diagnostics=debug,
        )

    @staticmethod
    def normalized_lane_change_state(*, decision: str, lane_change_phase: str) -> str:
        if str(lane_change_phase).strip().lower() == "target_lane_stabilization":
            return "TARGET_LANE_STABILIZATION"
        return (
            "EXECUTE_LANE_CHANGE_LEFT"
            if str(decision).strip().lower() == "lane_change_left"
            else "EXECUTE_LANE_CHANGE_RIGHT"
        )

    def set_lane_change_lifecycle(self, lifecycle: Any) -> None:
        if self._lane_change_lifecycle is not None:
            raise RuntimeError("lane-change lifecycle is already configured")
        self._lane_change_lifecycle = lifecycle

    @staticmethod
    def _baseline(request, prediction_count: int) -> CandidateSelectionResult:
        return CandidateSelectionResult(
            decision=str(request.baseline_decision),
            target_lane_id=int(request.baseline_target_lane_id),
            target_speed_mps=float(request.baseline_speed_mps),
            reference=tuple(dict(x) for x in request.baseline_reference),
            destination_state=tuple(request.baseline_destination_state),
            diagnostics={
                "candidate_pipeline_selected": "baseline_no_candidates",
                "candidate_pipeline_selected_status": "feasible",
                "candidate_pipeline_selected_reason": "",
                "candidate_pipeline_count": 0,
                "candidate_prediction_trajectory_count": int(prediction_count),
                "candidate_pipeline_summary": "[]",
            },
        )

    def run(
        self,
        request: CandidateSelectionRequest,
        *,
        sim_time_s: float,
        route_revision: str,
        road_envelope: Callable[[], Any],
        validate_contract: Callable[..., Any],
        validate_locked_reference: Callable[..., Any],
    ) -> CandidateSelectionResult:
        if self._lane_change_lifecycle is None:
            raise RuntimeError("lane-change lifecycle is not configured")
        release_reason = str(self._lane_change_lifecycle.release_completed(
            current_lane_id=int(request.current_lane_id),
            ego_location=request.ego_location,
            ego_yaw_rad=float(request.ego_yaw_rad),
        ) or "")
        intents = list(request.intents or ())
        if self._maneuver.route_lane_change_edge_completed:
            intents = [
                intent for intent in intents
                if str(getattr(intent, "decision", ""))
                not in {"lane_change_left", "lane_change_right"}
            ]
        predictions = dict(request.prediction_trajectories or {})
        if not intents:
            return self._baseline(request, len(predictions))

        candidates = self._evaluator.build_candidate_set(
            intents=intents,
            provider=self._provider,
            reference_context=request.reference_context,
            baseline_lane_change_state=str(request.baseline_lane_change_state),
            reference_pipeline=self._reference_pipeline,
            object_snapshots=request.object_snapshots,
            prediction_trajectories=predictions,
            required_lane_change_decision=str(request.required_decision),
            required_lane_change_target_lane_id=int(request.required_target_lane_id),
            static_obstacle_target_lane_id=self._static_obstacle.target_lane_id,
            normal_min_object_distance_m=self._normal_clearance_m,
            static_min_object_distance_m=self._static_clearance_m,
            risk_hysteresis_margin_m=self._risk_hysteresis_margin_m,
        )
        if self._provider.snapshot(LANE_CHANGE).active:
            static_committed = bool(
                self._static_obstacle.target_lane_id is not None
                and int(self._maneuver.lane_change.target_lane_id)
                == int(self._static_obstacle.target_lane_id)
                and int(self._maneuver.lane_change.target_lane_id)
                != int(request.current_lane_id)
            )
            committed = self._evaluator.build_committed_continuation(
                provider=self._provider,
                maneuver_manager=self._maneuver,
                config=self._config,
                mpc=self._mpc,
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                current_state=request.current_state,
                current_lane_id=int(request.current_lane_id),
                baseline_speed_mps=float(request.baseline_speed_mps),
                planner_target_speed_mps=self._target_speed_mps,
                validate_contract=validate_contract,
                object_snapshots=request.object_snapshots,
                prediction_trajectories=predictions,
                min_object_distance_m=(
                    self._static_clearance_m if static_committed
                    else self._normal_clearance_m
                ),
                risk_hysteresis_margin_m=self._risk_hysteresis_margin_m,
            )
            if committed is not None:
                candidates.append(committed)

        probe = self._evaluator.probe_mpc(
            candidate_results=candidates,
            mpc=self._mpc,
            sim_time_s=float(sim_time_s),
            current_state=request.current_state,
            object_snapshots=request.object_snapshots,
            current_acceleration_mps2=float(request.current_acceleration_mps2),
            current_steering_rad=float(request.current_steering_rad),
            road_envelope_payload_world=road_envelope(),
            required_decision=str(request.required_decision),
            required_target_lane_id=int(request.required_target_lane_id),
        )
        commitment, outcome, selected = self._evaluator.select_with_commitment(
            candidate_results=candidates,
            lane_change_phase=str(self._maneuver.lane_change.phase),
            lane_change_option=str(self._maneuver.lane_change.option),
            source_lane_id=int(self._maneuver.lane_change.source_lane_id),
            target_lane_id=int(self._maneuver.lane_change.target_lane_id),
            progress=float(self._maneuver.lane_change.progress),
            reference_locked=bool(
                self._provider.snapshot(LANE_CHANGE).mutable_samples()
            ),
            required_decision=str(request.required_decision),
            required_target_lane_id=int(request.required_target_lane_id),
        )
        if self._strict_ownership and candidates and outcome.selected is None:
            fallback = self._fallback.resolve_candidate_failure(
                candidate_results=candidates,
                baseline_decision=str(request.baseline_decision),
                baseline_target_lane_id=int(request.baseline_target_lane_id),
                current_lane_id=int(request.current_lane_id),
                current_state=request.current_state,
                ego_x_m=float(request.ego_location.x),
                ego_y_m=float(request.ego_location.y),
                reference_provider=self._provider,
                route_revision=str(route_revision),
                sim_time_s=float(sim_time_s),
                mpc_dt_s=float(self._mpc.dt_s),
                horizon_steps=int(self._mpc.horizon_steps),
                lane_change_min_first_forward_m=float(self._config.get(
                    "reference_contract_lane_change_min_first_forward_m", 0.2
                )),
                lane_follow_min_first_forward_m=float(self._config.get(
                    "reference_contract_lane_follow_min_first_forward_m", 0.2
                )),
                summarize_candidates=summarize_candidate_results,
                maneuver_commitment=commitment,
                selection_reason=str(outcome.reason),
            )
            return CandidateSelectionResult(
                decision=str(fallback.decision),
                target_lane_id=int(fallback.target_lane_id),
                target_speed_mps=float(fallback.target_speed_mps),
                reference=tuple(
                    dict(x) for x in fallback.mutable_trajectory()
                ),
                destination_state=tuple(fallback.mutable_destination_state()),
                diagnostics=fallback.diagnostics,
            )

        debug = dict(selected.reference_debug or {})
        reference = [dict(x) for x in selected.lane_center_reference or ()]
        destination = list(selected.destination_state or ())
        decision = str(selected.intent.decision)
        if decision in {"lane_change_left", "lane_change_right"}:
            reference, destination, debug = self._evaluator.activate_selected_lane_change(
                selected=selected,
                selected_reference=reference,
                selected_destination=destination,
                selected_debug=debug,
                provider=self._provider,
                maneuver_manager=self._maneuver,
                route_revision=str(route_revision),
                map_epoch=self._map_epoch,
                local_map=request.reference_context.local_map,
                current_lane_id=int(request.current_lane_id),
                current_state=request.current_state,
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                ego_speed_mps=float(request.ego_speed_mps),
                sim_time_s=float(sim_time_s),
                config=self._config,
                mpc=self._mpc,
                validate_locked_reference=validate_locked_reference,
            )
        result = self._evaluator.finalize_selection(
            selected=selected,
            candidate_results=candidates,
            selected_reference=reference,
            selected_destination=destination,
            selected_debug=debug,
            prediction_trajectory_count=len(predictions),
            probe_summary=probe,
            selection_outcome=outcome,
            commitment=commitment,
            commitment_release_reason=release_reason,
            lane_change_phase=self._maneuver.lane_change.phase,
            lane_change_stabilization_frames=(
                self._maneuver.lane_change.stabilization_frames
            ),
            completion_debug=self._maneuver.lane_change.completion_debug,
        )
        self._fallback.record_valid(
            result.reference,
            sim_time_s=float(sim_time_s),
            route_revision=str(route_revision),
        )
        return result
