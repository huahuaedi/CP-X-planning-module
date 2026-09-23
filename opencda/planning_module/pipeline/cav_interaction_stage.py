"""Single owner for Stage A-D cooperative interaction resolution."""

from __future__ import annotations

import math

from .cav_conflict_pipeline import resolve_conflicts as resolve_cav_conflicts
from .cav_intent_codec import sample_cav_path_at
from .conflict_classifier import ClassifierParams, _agent_id
from .mpc_corridor_constraints import corridor_rows, homotopy_keepout_rows
from .mpc_obstacle_relevance import _polyline_xy
from .rss import RSSParams, longitudinal_safe_distance
from .spatiotemporal_corridor import CorridorParams, make_gap_gate_margin_m
from .speed_planner import (
    conflict_corridor_speed_constraint,
    cooperative_gap_speed_constraint,
)


class CAVInteractionStage:
    """Classify agents, build a corridor, and emit MPC constraints once."""

    @staticmethod
    def resolve(
        *, reference_samples, constraint_reference_samples=None,
        ego_location, ego_yaw_rad, ego_speed_mps,
        actor_id, claim, obstacle_snapshots, cav_intents, latch_state,
        tag_state=None, veto_state=None,
        horizon_steps, dt_s, mode_probability_floor=0.05,
        credible_mode_probability_min=0.15, credible_mode_ttc_s=2.0,
        credible_mode_veto_release_s=0.6, sim_time_s=0.0,
        nominal_progress_limit_m=None,
        refresh_assignments=True, cached_assignments=(),
        rebuild_corridor=True, cached_corridor=None,
        max_braking_mps2=3.0,
        current_acceleration_mps2=0.0, max_jerk_mps3=10.0,
        comfortable_deceleration_mps2=1.5,
        cooperative_preparation_time_s=4.0,
        max_relevant_agents=6, max_modes_per_agent=3,
    ):
        """Classify on proposal geometry and constrain executable geometry."""

        corridor_reference = (
            reference_samples
            if constraint_reference_samples is None
            else constraint_reference_samples
        )
        corridor_params = CorridorParams(
            horizon_steps=max(1, int(horizon_steps)), dt_s=float(dt_s),
            max_braking_mps2=max(1.0e-3, abs(float(max_braking_mps2))),
            max_jerk_mps3=max(0.0, abs(float(max_jerk_mps3))),
        )
        rss_params = RSSParams()
        result = resolve_cav_conflicts(
            reference_samples=reference_samples,
            corridor_reference_samples=corridor_reference,
            ego_snapshot={
                "x": float(ego_location.x), "y": float(ego_location.y),
                "v": float(ego_speed_mps), "psi": float(ego_yaw_rad),
                "a": float(current_acceleration_mps2),
            },
            my_actor_id=int(actor_id), my_claim=claim,
            obstacle_snapshots=obstacle_snapshots, cav_intents=cav_intents,
            latch_state=latch_state, tag_state=tag_state, veto_state=veto_state,
            classifier_params=ClassifierParams(
                horizon_steps=max(1, int(horizon_steps)), dt_s=float(dt_s)
            ),
            corridor_params=corridor_params, rss_params=rss_params,
            mode_probability_floor=float(mode_probability_floor),
            credible_mode_probability_min=float(credible_mode_probability_min),
            credible_mode_ttc_s=float(credible_mode_ttc_s),
            credible_mode_veto_release_s=float(credible_mode_veto_release_s),
            sim_time_s=float(sim_time_s),
            nominal_progress_limit_m=nominal_progress_limit_m,
            refresh_assignments=bool(refresh_assignments),
            cached_assignments=cached_assignments,
            rebuild_corridor=bool(rebuild_corridor),
            cached_corridor=cached_corridor,
            max_relevant_agents=int(max_relevant_agents),
            max_modes_per_agent=int(max_modes_per_agent),
        )
        steps = max(1, int(horizon_steps))
        step_s = max(1.0e-3, float(dt_s))
        tracks = {}
        for intent in list(cav_intents or []):
            points = []
            for stage in range(steps + 1):
                sampled = sample_cav_path_at(intent, float(stage) * step_s)
                if sampled is None:
                    distance_m = float(intent.speed_mps) * float(stage) * step_s
                    sampled = (
                        float(intent.position_xy[0])
                        + distance_m * math.cos(float(intent.heading_rad)),
                        float(intent.position_xy[1])
                        + distance_m * math.sin(float(intent.heading_rad)),
                        float(intent.speed_mps),
                    )
                points.append((float(sampled[0]), float(sampled[1])))
            tracks[int(intent.actor_id)] = points
        origin = (float(ego_location.x), float(ego_location.y))
        result.constraint_corridor = result.corridor
        longitudinal_rows = corridor_rows(
            result.constraint_corridor, corridor_reference, ego_origin_xy=origin
        )
        lateral_rows = homotopy_keepout_rows(
            result.assignments,
            tracks,
            ego_heading_rad=float(ego_yaw_rad),
            ego_origin_xy=origin,
            lateral_conflict_actor_ids=tuple(
                int(tag.agent_id)
                for tag in result.tags
                if str(tag.tag) in {"CROSSING", "ONCOMING"}
                and str(tag.agent_id).lstrip("-").isdigit()
            ),
        )
        result.mpc_rows = list(longitudinal_rows) + list(lateral_rows)

        snapshots_by_id = {
            _agent_id(snapshot): snapshot for snapshot in (obstacle_snapshots or ())
        }
        corridor_poly = _polyline_xy(corridor_reference)
        make_gap_gate_margins_m = {}
        for assignment in result.assignments:
            if str(assignment.role) != "make_gap":
                continue
            peer_track = tracks.get(int(assignment.cav_actor_id))
            if not peer_track:
                continue
            margin_m = make_gap_gate_margin_m(
                agent=snapshots_by_id.get(str(assignment.cav_actor_id), {}),
                track=peer_track, poly=corridor_poly,
                p=corridor_params, rss=rss_params,
            )
            if margin_m is not None:
                make_gap_gate_margins_m[str(assignment.cav_actor_id)] = margin_m
        result.diagnostics["make_gap_gate_margin_m"] = make_gap_gate_margins_m
        result.speed_constraint = conflict_corridor_speed_constraint(
            corridor=result.corridor,
            reference_samples=corridor_reference,
            ego_x_m=float(ego_location.x), ego_y_m=float(ego_location.y),
            comfortable_deceleration_mps2=float(comfortable_deceleration_mps2),
            corridor_dt_s=step_s,
        )
        peers_by_id = {
            int(intent.actor_id): intent for intent in list(cav_intents or [])
        }
        for assignment in result.assignments:
            if str(assignment.role) != "make_gap":
                continue
            peer = peers_by_id.get(int(assignment.cav_actor_id))
            if peer is None:
                continue
            gap_constraint = cooperative_gap_speed_constraint(
                reference_samples=corridor_reference,
                ego_x_m=float(ego_location.x), ego_y_m=float(ego_location.y),
                ego_speed_mps=float(ego_speed_mps),
                peer_x_m=float(peer.position_xy[0]),
                peer_y_m=float(peer.position_xy[1]),
                peer_speed_mps=float(peer.speed_mps), peer_id=str(peer.actor_id),
                peer_length_m=float(peer.length_m),
                ego_half_length_m=float(corridor_params.ego_half_length_m),
                desired_bumper_gap_m=(
                    longitudinal_safe_distance(
                        float(ego_speed_mps), float(peer.speed_mps), rss_params
                    ) + float(corridor_params.follow_extra_buffer_m)
                ),
                preparation_time_s=float(cooperative_preparation_time_s),
                comfortable_deceleration_mps2=float(
                    comfortable_deceleration_mps2
                ),
                planning_dt_s=step_s,
            )
            if gap_constraint is not None and (
                result.speed_constraint is None
                or gap_constraint.maximum_mps < result.speed_constraint.maximum_mps
            ):
                result.speed_constraint = gap_constraint
        result.diagnostics.update({
            "longitudinal_qp_row_count": len(longitudinal_rows),
            "homotopy_qp_row_count": len(lateral_rows),
            "total_qp_row_count": len(result.mpc_rows),
            "corridor_coordinate_owner": "mpc_executed_reference",
            "anticipatory_speed_cap_mps": (
                "" if result.speed_constraint is None
                else float(result.speed_constraint.maximum_mps)
            ),
            "anticipatory_speed_constraint_owner": (
                "" if result.speed_constraint is None
                else str(result.speed_constraint.owner)
            ),
            "anticipatory_speed_constraint_reason": (
                "" if result.speed_constraint is None
                else str(result.speed_constraint.reason)
            ),
            "shared_planned_paths": {
                str(intent.actor_id): [
                    (float(sample[1]), float(sample[2]))
                    for sample in intent.planned_path
                ]
                for intent in list(cav_intents or []) if intent.planned_path
            },
        })
        if cav_intents:
            validation_intent = sorted(
                cav_intents, key=lambda value: value.actor_id
            )[0]
            validation_horizon_s = min(1.0, float(steps) * step_s)
            validation_stage = min(
                steps, max(0, int(round(validation_horizon_s / step_s)))
            )
            validation_xy = tracks[int(validation_intent.actor_id)][validation_stage]
            result.diagnostics.update({
                "prediction_validation_actor_id": int(validation_intent.actor_id),
                "prediction_validation_horizon_s": float(validation_horizon_s),
                "prediction_validation_x_m": float(validation_xy[0]),
                "prediction_validation_y_m": float(validation_xy[1]),
            })
        return result
