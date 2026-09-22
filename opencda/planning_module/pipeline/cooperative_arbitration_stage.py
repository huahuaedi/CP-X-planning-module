"""Cooperative (multi-CAV) arbitration for one planning tick.

Owns the claim lifecycle, the coordination schedule and the compute governor,
and turns one tick's proposal, the peers' intents and the ego reference into
``(cav_resolution, defer_lane_change)``.  Reading peers off the V2X adapter is
input conversion and stays with the caller: intents arrive on the request.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .cav_conflict_compute_governor import CAVConflictComputeGovernor
from .cav_conflict_schedule import CAVConflictSchedule
from .cooperative_claim_geometry import project_claim_interval
from .cooperative_claim_manager import CooperativeClaimManager


@dataclass(frozen=True)
class CooperativeArbitrationRequest:
    proposal: Any
    current_state: Sequence[float]
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    ego_actor_id: int
    local_lane_center_reference: Sequence[Mapping[str, Any]]
    local_map_snapshot: Any
    object_snapshots: Sequence[Mapping[str, Any]]
    planned_speed_mps: float
    prediction_revision: str
    predicted_objects: Mapping[str, Any]
    sim_time_s: float
    cav_intents: Sequence[Any]
    transport_diagnostics: Mapping[str, Any]
    last_accel_mps2: float


class CooperativeArbitrationStage:
    def __init__(
        self, *, config: Mapping[str, Any], enabled: bool,
        build_conflict_reference: Callable[..., Any],
        resolve_interaction: Callable[..., Any],
        mpc: Any, route_manager: Any, maneuver_manager: Any,
        record_stage_ms: Callable[[str, float], None],
    ) -> None:
        self._config = config
        self._enabled = bool(enabled)
        self._build_conflict_reference = build_conflict_reference
        self._resolve_interaction = resolve_interaction
        self._mpc = mpc
        self._route_manager = route_manager
        self._maneuver_manager = maneuver_manager
        self._record_stage_ms = record_stage_ms
        self.schedule = CAVConflictSchedule(
            coordination_period_s=float(
                config.get("cav_coordination_period_s", 0.2)
            ),
            infeasible_emergency_streak=max(1, int(
                config.get("corridor_infeasible_emergency_streak", 2)
            )),
        )
        self.governor = CAVConflictComputeGovernor(
            max_agents=int(config.get("cav_conflict_max_relevant_agents", 6)),
            max_modes=int(config.get("cav_conflict_max_modes_per_agent", 3)),
            min_agents=int(config.get("cav_conflict_min_relevant_agents", 2)),
            min_modes=int(config.get("cav_conflict_min_modes_per_agent", 1)),
            budget_ms=float(config.get("cav_conflict_budget_ms", 100.0)),
            window=int(config.get("cav_conflict_budget_window_ticks", 5)),
            degrade_streak=int(
                config.get("cav_conflict_budget_degrade_streak", 3)
            ),
            recover_streak=int(
                config.get("cav_conflict_budget_recover_streak", 5)
            ),
        )
        # Never a real route_manager.route_revision value, so the first tick
        # that reaches the governor always takes the reset branch below --
        # harmless (it is already fresh) and avoids a separate first-tick case.
        self.governor_route_revision = "\x00uninitialized"
        self.claims = CooperativeClaimManager(
            enabled=bool(enabled),
            proposal_dwell_s=float(config.get("cav_proposal_dwell_s", 0.25)),
        )

    def run(self, request: CooperativeArbitrationRequest):
        """Returns ``(cav_result, cooperative_lane_change_deferred)``."""

        cooperative_proposal = request.proposal
        current_state = request.current_state
        ego_location = request.ego_location
        ego_yaw_rad = request.ego_yaw_rad
        ego_speed_mps = request.ego_speed_mps
        local_lane_center_reference = request.local_lane_center_reference
        local_map_snapshot = request.local_map_snapshot
        object_snapshots = request.object_snapshots
        planned_speed_mps = request.planned_speed_mps
        sim_time_s = request.sim_time_s
        cav_result = None
        cooperative_lane_change_deferred = False
        if not self._enabled:
            return cav_result, cooperative_lane_change_deferred
        # A compute budget degraded by a complex intersection must not
        # keep constraining an unrelated later maneuver or route -- the
        # pressure that earned the degradation is gone once the route
        # itself has changed, so the ceiling should be too.
        current_route_revision = str(self._route_manager.route_revision)
        if current_route_revision != self.governor_route_revision:
            self.governor_route_revision = current_route_revision
            self.governor.reset()
            self.schedule.reset(
                reason="route_revision_changed:" + current_route_revision
            )
            self.claims.reset()
        lane_change = self._maneuver_manager.lane_change
        if bool(lane_change.active):
            cooperative_proposal = cooperative_proposal.with_commitment(
                maneuver=(
                    "lane_change_left"
                    if str(lane_change.option) == "CHANGELANELEFT"
                    else "lane_change_right"
                ),
                target_corridor_id=int(lane_change.target_lane_id),
                committed_at_s=float(lane_change.committed_at_s),
            )
        claim_interval = project_claim_interval(
            local_map=local_map_snapshot,
            corridor_id=int(cooperative_proposal.target_corridor_id),
            x_m=float(ego_location.x),
            y_m=float(ego_location.y),
            lookbehind_m=float(
                self._config.get("cav_claim_lookbehind_m", 10.0)
            ),
            lookahead_m=float(
                self._config.get("cav_claim_lookahead_m", 50.0)
            ),
        )
        cooperative_proposal = cooperative_proposal.with_station_interval(
            corridor_id=int(claim_interval.corridor_id),
            s_begin_m=claim_interval.s_begin_m,
            s_end_m=claim_interval.s_end_m,
        )
        cav_claim = self.claims.claim(
            proposal=cooperative_proposal,
            sim_time_s=float(sim_time_s),
        )
        cav_intents = request.cav_intents
        schedule = self.schedule.decide(
            sim_time_s=float(sim_time_s),
            prediction_revision=str(
                request.prediction_revision
            ),
            claim=cav_claim, peers=cav_intents,
            proposal=cooperative_proposal,
        )
        _ts_sub = time.monotonic()
        conflict_reference = self.schedule.reference_for_tick(
            refresh=bool(schedule.refresh_roles),
            build=lambda: self._build_conflict_reference(
                proposal=cooperative_proposal,
                local_map=local_map_snapshot,
                current_state=current_state,
                baseline_reference=local_lane_center_reference,
                target_speed_mps=float(planned_speed_mps),
                horizon_steps=int(self._mpc.horizon_steps),
                dt_s=float(self._mpc.dt_s),
                lane_width_m=float(getattr(self._mpc, "lane_width_m", 3.5)),
            ),
        )
        self._record_stage_ms(
            "sub_cooperative_conflict_reference", time.monotonic() - _ts_sub
        )
        cached_corridor = self.schedule.cached_corridor_for_tick(
            sim_time_s=float(sim_time_s),
            reference_samples=local_lane_center_reference,
            ego_x_m=float(ego_location.x), ego_y_m=float(ego_location.y),
            dt_s=float(self._mpc.dt_s),
        )
        _ts_sub = time.monotonic()
        cav_result = self._resolve_interaction(
            reference_samples=conflict_reference.mutable_samples(),
            # Stage D's QP rows must linearize against the reference the
            # vehicle is actually driving, not the lane-change preview
            # curve used only to classify a proposed maneuver -- see
            # PlanningPipeline.resolve_cav_interaction's docstring.
            constraint_reference_samples=local_lane_center_reference,
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            actor_id=request.ego_actor_id,
            claim=cav_claim,
            obstacle_snapshots=interaction_obstacle_snapshots(
                object_snapshots,
                predicted_objects=(
                    request.predicted_objects
                ),
            ),
            cav_intents=cav_intents,
            latch_state=self.schedule.latch_state,
            tag_state=self.schedule.tag_state,
            veto_state=self.schedule.veto_state,
            horizon_steps=int(self._mpc.horizon_steps),
            dt_s=float(self._mpc.dt_s),
            mode_probability_floor=float(self._config.get(
                "prediction_mode_min_probability", 0.05
            )),
            credible_mode_probability_min=float(self._config.get(
                "prediction_credible_probability_min", 0.15
            )),
            credible_mode_ttc_s=float(self._config.get(
                "prediction_credible_ttc_s", 2.0
            )),
            credible_mode_veto_release_ticks=int(self._config.get(
                "prediction_credible_veto_release_ticks", 12
            )),
            nominal_progress_limit_m=(
                float(self._route_manager.last_status.remaining_distance_m)
                if (
                    math.isfinite(float(
                        self._route_manager.last_status.remaining_distance_m
                    ))
                    and (
                        float(self._route_manager.last_status.remaining_distance_m) > 0.0
                        or bool(self._route_manager.last_status.reached_destination)
                    )
                )
                else None
            ),
            refresh_assignments=bool(schedule.refresh_roles),
            cached_assignments=self.schedule.assignments,
            rebuild_corridor=bool(schedule.refresh_roles),
            cached_corridor=cached_corridor,
            max_braking_mps2=abs(float(self._mpc.constraints.min_acceleration_mps2)),
            current_acceleration_mps2=float(request.last_accel_mps2),
            max_jerk_mps3=float(self._mpc.constraints.max_jerk_mps3),
            comfortable_deceleration_mps2=float(self._config.get(
                "cav_conflict_comfort_deceleration_mps2", 1.5
            )),
            cooperative_preparation_time_s=float(self._config.get(
                "candidate_lane_change_normal_duration_s", 4.0
            )),
            max_relevant_agents=self.governor.current_max_relevant_agents,
            max_modes_per_agent=self.governor.current_max_modes_per_agent,
        )
        _cav_interaction_elapsed_s = time.monotonic() - _ts_sub
        self._record_stage_ms(
            "sub_resolve_cav_interaction", _cav_interaction_elapsed_s
        )
        # Feeds next tick's budget, not this one: this tick already ran
        # at whatever budget was decided last tick, so retroactively
        # shrinking its own inputs here would just make the measurement
        # describe a call that never happened.
        budget_decision = self.governor.observe_stage_ms(
            _cav_interaction_elapsed_s * 1000.0
        )
        if budget_decision.degraded:
            cav_result.diagnostics["cav_conflict_budget_decision"] = (
                budget_decision.reason
            )
        self.schedule.observe(
            sim_time_s=float(sim_time_s), result=cav_result,
            reference_samples=local_lane_center_reference,
        )
        cav_result.diagnostics["coordination_schedule_reason"] = str(
            schedule.reason
        )
        cav_result.diagnostics["coordination_revision"] = int(
            self.schedule.revision
        )
        cav_result.diagnostics["transport"] = dict(
            request.transport_diagnostics
        )
        cav_result.diagnostics["conflict_reference_source"] = str(
            conflict_reference.source
        )
        cav_result.diagnostics["conflict_reference_reason"] = str(
            conflict_reference.reason
        )
        cooperative_lane_change_deferred = (
            self.claims.defer_candidate(
                sim_time_s=float(sim_time_s),
                assignments=cav_result.assignments,
            )
        )
        return cav_result, cooperative_lane_change_deferred


def interaction_obstacle_snapshots(
    object_snapshots: Sequence[Mapping[str, Any]],
    *,
    predicted_objects: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Attach the prediction module's future to each non-connected road
    user before Stage A/C.

    Fusion priority: a connected vehicle with a fresh broadcast plan is
    handled by ``collect_cav_intents`` (its shared trajectory wins);
    every other agent gets every retained prediction hypothesis here.
    """

    from opencda.planning_module.pipeline.prediction import obstacle_track_id
    preds = dict(predicted_objects or {})
    out: list[dict[str, Any]] = []
    for snapshot in list(object_snapshots or []):
        if not isinstance(snapshot, Mapping):
            continue
        updated = dict(snapshot)
        predicted = preds.get(obstacle_track_id(snapshot))
        hypotheses = tuple(getattr(predicted, "hypotheses", ()) or ())
        if hypotheses and "predicted_modes" not in updated:
            updated["predicted_modes"] = [
                {
                    "path": hypothesis.mutable_points(),
                    "probability": float(hypothesis.probability),
                }
                for hypothesis in hypotheses
            ]
            updated.setdefault("trajectory_source", "prediction")
        out.append(updated)
    return out
