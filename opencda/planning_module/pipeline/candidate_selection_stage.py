"""Single owner for candidate construction, probing and commitment selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .candidate_evaluation import CandidateSelectionResult
from .candidate_pipeline import summarize_candidate_results
from .reference_line_provider import LANE_CHANGE


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
        release_completed: Callable[[], str],
        road_envelope: Callable[[], Any],
        validate_contract: Callable[..., Any],
        validate_locked_reference: Callable[..., Any],
    ) -> CandidateSelectionResult:
        release_reason = str(release_completed() or "")
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
