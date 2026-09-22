"""Read-only assembly of the stable planner diagnostic schema."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .reference_geometry import pose_at_arc, project_to_polyline
from .reference_line_provider import LANE_FOLLOW


def _recent_admap_query_failures() -> list[dict[str, object]]:
    """Read admap_backend's recent lane-geometry query failure log.

    Best-effort: only the AD-map global-planner path imports admap_backend
    at all (it needs the compiled ad-map-access bindings), so this returns
    an empty list rather than raising on any other configuration.
    """

    try:
        from opencda.planning_module.Global_Planner.global_planner import (
            admap_backend,
        )

        return admap_backend.get_recent_query_failures()
    except Exception:
        return []


def _cav_conflict_summary(diag: Mapping[str, Any]) -> str:
    """One-line CSV field: per-agent tag + per-cav role + corridor status."""

    diag = dict(diag or {})
    if not diag:
        return ""
    tags = dict(diag.get("tags", {}) or {})
    roles = dict(diag.get("roles", {}) or {})
    parts = [f"{k}:{v}" for k, v in tags.items() if v != "IGNORE"]
    parts += [f"cav{k}={v}" for k, v in roles.items()]
    feasible = diag.get("corridor_feasible")
    if feasible is not None:
        parts.append("corridor_feasible" if feasible else "corridor_infeasible")
    return ";".join(parts) if parts else "no_conflict"


def _wrap_angle_rad(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _executed_reference_tracking(
    *,
    reference: object,
    ego_x_m: float,
    ego_y_m: float,
    ego_yaw_rad: float,
) -> dict[str, object]:
    """Measure ego pose against the reference actually submitted to MPC.

    Map-matcher alignment and executed-reference tracking are deliberately
    separate diagnostics.  Near a connector they may use different local
    tangents; treating the former as MPC tracking error led control tuning to
    chase a map-association signal rather than the trajectory being executed.
    """

    samples = list(reference or []) if isinstance(reference, (list, tuple)) else []
    if len(samples) < 2:
        return {
            "executed_reference_tracking_valid": False,
            "executed_reference_progress_s_m": "",
            "executed_reference_lateral_error_m": "",
            "executed_reference_heading_error_deg": "",
        }
    progress_s_m, lateral_error_m = project_to_polyline(
        samples,
        float(ego_x_m),
        float(ego_y_m),
    )
    _, _, reference_heading_rad = pose_at_arc(samples, progress_s_m)
    return {
        "executed_reference_tracking_valid": True,
        "executed_reference_progress_s_m": float(progress_s_m),
        "executed_reference_lateral_error_m": float(lateral_error_m),
        "executed_reference_heading_error_deg": math.degrees(
            _wrap_angle_rad(float(ego_yaw_rad) - float(reference_heading_rad))
        ),
    }


@dataclass(frozen=True)
class ReferenceDiagnosticsRequest:
    """Typed outputs needed for the behavior/reference diagnostic snapshot."""

    built_reference: Any
    planning_context: Any
    executable_behavior: Any
    speed_frame: Any
    route_update: Any
    mpc_feedback: Mapping[str, Any]


class PlannerDiagnosticsStage:
    """Build diagnostics without participating in planning or control."""

    @staticmethod
    def fieldnames(payload: Mapping[str, Any]) -> tuple[str, ...]:
        """Return the stable serialization order for a built trace payload."""

        return tuple(str(key) for key in payload.keys())

    @staticmethod
    def build_reference_debug_from_stages(
        owner: Any, request: ReferenceDiagnosticsRequest
    ) -> dict[str, Any]:
        """Derive the trace context from authoritative typed stage outputs."""

        planning = request.planning_context
        adapter = planning.adapter_output
        frame = planning.planner_input_frame
        behavior = planning.behavior_context
        route_behavior = behavior.route_behavior
        lane_context = route_behavior.context
        observation = behavior.scenario_observation
        scenario = observation.scenario
        scenario_decision = scenario.decision
        conflict = behavior.conflict_resolution
        executable = request.executable_behavior
        speed = request.speed_frame
        route_attempted = bool(request.route_update.route_replan_attempted)
        route_succeeded = bool(request.route_update.route_replan_succeeded)
        route_reason = str(request.route_update.route_replan_reason)
        if bool(behavior.route_replan_attempted):
            route_attempted = bool(behavior.route_replan_attempted)
            route_succeeded = bool(behavior.route_replan_succeeded)
            route_reason = str(behavior.route_replan_reason)
        source_quality = dict(adapter.source_quality)
        source_quality.update(request.route_update.trace_fields())
        return PlannerDiagnosticsStage.build_reference_debug(owner, {
            "built_reference": request.built_reference,
            "planner_input_frame": frame,
            "front_gap_actor_id": speed.front_actor_id,
            "front_gap_obstacle_speed_mps": speed.front_obstacle_speed_mps,
            "front_obstacle_lane_id": speed.front_obstacle_lane_id,
            "front_obstacle_is_source_lane": speed.front_obstacle_is_source_lane,
            "route_reference_allowed": adapter.route_reference_allowed,
            "route_reference_gate_reason": adapter.route_reference_gate_reason,
            "route_lane_change_allowed": route_behavior.route_lane_change_allowed,
            "opportunistic_lane_change_allowed": (
                executable.opportunistic_lane_change_allowed
            ),
            "lane_change_gate_reason": conflict.lane_change_gate_reason,
            "static_obstacle_local_avoidance_active": (
                executable.static_obstacle_result.local_avoidance_active
            ),
            "static_obstacle_local_target_lane_id": (
                executable.static_obstacle_result.target_lane_id
            ),
            "static_obstacle_result": executable.static_obstacle_result,
            "semantic_response": executable.semantic_response,
            "route_lane_change_required": conflict.authorization.required_by_route,
            "route_geometry_lane_change_direction": lane_context.geometry_direction,
            "route_geometry_lane_change_distance_m": lane_context.geometry_distance_m,
            "route_geometry_lane_change_reason": lane_context.geometry_reason,
            "physical_route_target_lane_id": lane_context.physical_target_lane_id,
            "topology_route_target_lane_id": lane_context.topology_target_lane_id,
            "behavior_lane_lateral_error_m": executable.lane_lateral_error_m,
            "behavior_lane_heading_error_rad": executable.lane_heading_error_rad,
            "behavior_lane_alignment_valid": executable.lane_alignment_valid,
            "lane_change_authorization": conflict.authorization,
            "behavior_override_reason": executable.override_reason,
            "scenario_decision": scenario_decision,
            "route_context": frame.planning.route,
            "full_traffic_memory_reason": scenario.traffic_memory_reason,
            "resolved_traffic_state": observation.resolved_traffic_state,
            "filtered_traffic_state": observation.filtered_traffic_state,
            "behavior_traffic_state": scenario.behavior_traffic_state,
            "traffic_stop_forward_m": observation.stop_forward_m,
            "traffic_stop_commit_distance_m": (
                scenario_decision.traffic_stop_commit_distance_m
            ),
            "traffic_stop_approach_reason": scenario_decision.reason,
            "speed_plan": speed.speed_plan,
            "candidate_frame": executable.candidate_frame,
            "mpc_feedback": request.mpc_feedback,
            "upcoming_turn_direction": observation.turn_context.direction,
            "upcoming_turn_distance_m": observation.turn_context.distance_m,
            "upcoming_turn_reason": observation.turn_context.reason,
            "source_quality": source_quality,
            "route_replan_attempted": route_attempted,
            "route_replan_succeeded": route_succeeded,
            "route_replan_reason": route_reason,
        })

    @staticmethod
    def build_reference_debug(owner: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        """Assemble the read-only behavior/reference trace for one frame."""

        self = owner
        c = context
        debug = dict(c["built_reference"].diagnostics)
        debug.update(c["planner_input_frame"].trace_fields())
        topology = self.route_manager.route_topology_validation
        debug.update({
            "stage": debug.get("reference_pipeline_stage", ""),
            "intent_mode": debug.get("reference_pipeline_intent_mode", ""),
            "fallback_reason": str(c["built_reference"].fallback_reason),
            "reference_source": str(debug.get(
                "reference_source", "behavior_reference_pipeline"
            )),
            "front_gap_actor_id": str(c["front_gap_actor_id"] or ""),
            "front_gap_obstacle_speed_mps": (
                "" if c["front_gap_obstacle_speed_mps"] is None
                else float(c["front_gap_obstacle_speed_mps"])
            ),
            "front_gap_obstacle_lane_id": int(c["front_obstacle_lane_id"]),
            "front_gap_obstacle_is_source_lane": bool(c["front_obstacle_is_source_lane"]),
            "route_reference_allowed": bool(c["route_reference_allowed"]),
            "route_reference_gate_reason": str(c["route_reference_gate_reason"]),
            "route_lane_change_allowed": bool(c["route_lane_change_allowed"]),
            "opportunistic_lane_change_allowed": bool(c["opportunistic_lane_change_allowed"]),
            "lane_change_gate_reason": str(c["lane_change_gate_reason"]),
            "static_obstacle_local_avoidance_active": bool(
                c["static_obstacle_local_avoidance_active"]
            ),
            "static_obstacle_local_target_lane_id": (
                "" if c["static_obstacle_local_target_lane_id"] is None
                else int(c["static_obstacle_local_target_lane_id"])
            ),
            "static_obstacle_candidate_since_s": float(
                c["static_obstacle_result"].candidate_since_s
            ),
            "semantic_risk_kind": str(c["semantic_response"].risk_kind),
            "semantic_behavior_action": str(c["semantic_response"].action),
            "semantic_object_type": str(c["semantic_response"].object_type),
            "semantic_observation_source": str(
                c["semantic_response"].observation_source
            ),
            "semantic_obstacle_id": str(c["semantic_response"].obstacle_id),
            "semantic_obstacle_distance_m": float(
                c["semantic_response"].distance_m
            ),
            "semantic_behavior_reason": str(c["semantic_response"].reason),
            "static_obstacle_global_replan_enabled": bool(self.config.get(
                "static_obstacle_global_replan_enabled",
                self.behavior_runtime_cfg.get("static_obstacle_global_replan_enabled", False),
            )),
            "route_lane_change_required": bool(c["route_lane_change_required"]),
            "route_progress_s_m": float(self.route_manager.route_progress_s_m),
            "route_progress_lane_index": int(self.route_manager.route_progress_lane_index),
            "route_topology_valid": bool(topology.valid),
            "route_topology_signature": " -> ".join(topology.signature),
            "route_topology_errors": ";".join(topology.errors),
            "route_topology_warnings": ";".join(topology.warnings),
            "route_lane_change_edge_id": str(self.maneuver_manager.route_lane_change_edge_id),
            "completed_route_lane_change_edge_id": str(
                self.maneuver_manager.completed_route_lane_change_edge_id
            ),
            "route_lane_change_edge_completed": bool(
                self.maneuver_manager.route_lane_change_edge_completed
            ),
            "route_recovery_pending": bool(
                self.maneuver_manager.route_recovery_pending
            ),
            "lane_change_authorization_source": str(
                self.maneuver_manager.lane_change.authorization_source
            ),
            "lane_change_returns_to_route": bool(
                self.maneuver_manager.lane_change.returns_to_route
            ),
            "route_geometry_lane_change_direction": str(
                c["route_geometry_lane_change_direction"] or ""
            ),
            "route_geometry_lane_change_distance_m": float(
                c["route_geometry_lane_change_distance_m"]
            ),
            "route_geometry_lane_change_reason": str(c["route_geometry_lane_change_reason"]),
            "route_physical_target_lane_id": int(c["physical_route_target_lane_id"]),
            "route_topology_target_lane_id": int(c["topology_route_target_lane_id"]),
            "behavior_lane_lateral_error_m": float(c["behavior_lane_lateral_error_m"]),
            "behavior_lane_heading_error_deg": math.degrees(
                float(c["behavior_lane_heading_error_rad"])
            ),
            "behavior_lane_alignment_valid": bool(c["behavior_lane_alignment_valid"]),
            "behavior_lane_change_completion_allowed": not bool(
                self._stable_reference_line_provider.snapshot("lane_change").mutable_samples()
            ),
            **dict(c["lane_change_authorization"].as_debug_fields()),
            "behavior_override_reason": str(c["behavior_override_reason"]),
            "turn_latch_reason": "scenario_manager:" + str(c["scenario_decision"].reason),
            "route_current_road_option": str(c["route_context"].current_road_option),
            "route_next_macro_maneuver": str(c["route_context"].next_macro_maneuver),
            "traffic_memory_reason": str(c["full_traffic_memory_reason"]),
            "traffic_signal_raw_state": str(
                c["planner_input_frame"].planning.traffic_control.signal_state
            ),
            "traffic_signal_resolved_state": str(c["resolved_traffic_state"]),
            "traffic_signal_filtered_state": str(c["filtered_traffic_state"]),
            "traffic_signal_behavior_state": str(c["behavior_traffic_state"]),
            "traffic_stop_forward_m": float(c["traffic_stop_forward_m"]),
            "traffic_stop_commit_distance_m": float(c["traffic_stop_commit_distance_m"]),
            "traffic_stop_approach_reason": str(c["traffic_stop_approach_reason"]),
            **dict(c["speed_plan"].as_debug_fields()),
            "candidate_evaluation_summary": str(c["candidate_frame"].summary()),
            "candidate_selected_decision": str(c["candidate_frame"].selected.decision),
            "candidate_selected_lane_id": int(c["candidate_frame"].selected.target_lane_id),
            "candidate_selected_cost": float(c["candidate_frame"].selected.total_cost),
            "mpc_feedback_summary": str(c["mpc_feedback"].get("summary", "")),
            "mpc_feedback_blocked_lane_ids": json.dumps(
                list(c["mpc_feedback"].get("blocked_lane_ids", []) or []), default=str,
            ),
            "prediction_trajectories": dict(
                c["planner_input_frame"].prediction.obstacle_future_trajectories
            ),
        })
        debug.update(c["scenario_decision"].as_debug_fields())
        distance_m = float(c["upcoming_turn_distance_m"])
        debug.update({
            "route_upcoming_turn_direction": str(c["upcoming_turn_direction"]),
            "route_upcoming_turn_distance_m": (
                "" if not math.isfinite(distance_m) else distance_m
            ),
            "route_upcoming_turn_reason": str(c["upcoming_turn_reason"]),
        })
        debug.update(c["source_quality"])
        # These three were computed by the caller (from the CP lane-closure
        # /route-replan attempt this tick) and passed in as part of the same
        # context dict, but nothing above ever read them back out of `c` --
        # they were silently dropped here, so PlannerDiagnosticsStage.build's
        # own reference_debug.get("route_replan_attempted", False) always
        # fell through to its default, regardless of what actually happened.
        debug.update({
            "route_replan_attempted": bool(c["route_replan_attempted"]),
            "route_replan_succeeded": bool(c["route_replan_succeeded"]),
            "route_replan_reason": str(c["route_replan_reason"]),
        })
        return debug

    @staticmethod
    def build(
        adapters: Any, pipeline: Any, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Assemble the full per-tick diagnostic row.

        ``adapters`` (PlanningTickAdapters) supplies every bridge-owned
        collaborator/value this needs; ``pipeline`` is the PlanningPipeline
        instance itself, for the few fields that live on its stages
        (static_obstacle, mpc_cost_profile, destination_speed). Neither is
        held past this call.
        """

        accel_mps2 = context["accel_mps2"]
        behavior_debug = context["behavior_debug"]
        behavior_decision = context["behavior_decision"]
        boundary_snapshot = context["boundary_snapshot"]
        control = context["control"]
        control_guard_reason = context["control_guard_reason"]
        destination_forward_m = context["destination_forward_m"]
        destination_lateral_m = context["destination_lateral_m"]
        destination_state = context["destination_state"]
        ego_location = context["ego_location"]
        ego_speed_mps = context["ego_speed_mps"]
        ego_transform = context["ego_transform"]
        ego_yaw_rad = context["ego_yaw_rad"]
        emergency_brake_requested = context["emergency_brake_requested"]
        fallback_reason = context["fallback_reason"]
        front_gap_m = context["front_gap_m"]
        hard_gate_active = context["hard_gate_active"]
        lane_center_reference = context["lane_center_reference"]
        local_object_snapshots = context["local_object_snapshots"]
        measured_accel_mps2 = context["measured_accel_mps2"]
        mode_transition_guard_reason = context["mode_transition_guard_reason"]
        mpc_feedback_record_reason = context["mpc_feedback_record_reason"]
        mpc_jerk_seed_accel_mps2 = context["mpc_jerk_seed_accel_mps2"]
        mpc_object_snapshots = context["mpc_object_snapshots"]
        mpc_replan_executed = context["mpc_replan_executed"]
        mpc_status = context["mpc_status"]
        mpc_stop_goal_active = context["mpc_stop_goal_active"]
        normal_stop_requested = context["normal_stop_requested"]
        object_snapshots = context["object_snapshots"]
        platform_adapter_debug = context["platform_adapter_debug"]
        post_supervisor_accel_mps2 = context["post_supervisor_accel_mps2"]
        post_supervisor_steer_rad = context["post_supervisor_steer_rad"]
        pre_supervisor_accel_mps2 = context["pre_supervisor_accel_mps2"]
        pre_supervisor_steer_rad = context["pre_supervisor_steer_rad"]
        reference_debug = context["reference_debug"]
        reference_first_forward_m = context["reference_first_forward_m"]
        reference_first_lateral_m = context["reference_first_lateral_m"]
        safety_supervisor_reason = context["safety_supervisor_reason"]
        speed_ref_mps = context["speed_ref_mps"]
        speed_target = context["speed_target"]
        stationary_traffic_stop_hold = context["stationary_traffic_stop_hold"]
        steer_rad = context["steer_rad"]
        stop_target_forward_m_debug = context["stop_target_forward_m_debug"]
        cp_summary = dict(getattr(adapters.cp_provider, "last_publish_summary", {}) or {})
        cav_resolution = context.get("cav_resolution")
        cav_diag = dict(getattr(cav_resolution, "diagnostics", {}) or {})
        executed_reference_tracking = _executed_reference_tracking(
            reference=lane_center_reference,
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_yaw_rad=float(ego_yaw_rad),
        )
        diagnostics = {
            "sim_time_s": float(context["sim_time_s"]),
            "vehicle_id": int(getattr(adapters.ego_vehicle, "id", -1)),
            "x_m": float(ego_location.x),
            "y_m": float(ego_location.y),
            "yaw_deg": float(ego_transform.rotation.yaw),
            **executed_reference_tracking,
            "speed_mps": float(ego_speed_mps),
            "measured_accel_mps2": float(measured_accel_mps2),
            "mpc_jerk_seed_accel_mps2": float(mpc_jerk_seed_accel_mps2),
            "planner": "cpx_mpc",
            **platform_adapter_debug,
            "object_count": len(object_snapshots),
            "mpc_object_count": len(mpc_object_snapshots),
            "local_object_count": len(local_object_snapshots),
            "prediction_mode": str(adapters.prediction_mode),
            "cav_conflict_summary": _cav_conflict_summary(cav_diag),
            # Structured JSONL evidence.  The compact summary remains only
            # for legacy CSV readers; new evaluators must not parse prose.
            "cav_conflict_tags": dict(cav_diag.get("tags", {}) or {}),
            "cav_conflict_tag_reasons": dict(
                cav_diag.get("tag_reasons", {}) or {}
            ),
            "cav_conflict_agent_states": dict(
                cav_diag.get("agent_states", {}) or {}
            ),
            "cav_conflict_roles": dict(cav_diag.get("roles", {}) or {}),
            "cav_make_gap_gate_margin_m": dict(
                cav_diag.get("make_gap_gate_margin_m", {}) or {}
            ),
            "cav_ego_claim": dict(cav_diag.get("ego_claim", {}) or {}),
            "cav_peer_claims": dict(cav_diag.get("peer_claims", {}) or {}),
            "cav_arbitration": dict(cav_diag.get("arbitration", {}) or {}),
            "cav_corridor_binding": ",".join(
                str(b) for b in cav_diag.get("corridor_binding", [])
            ),
            "cav_intent_count": int(cav_diag.get("cav_count", 0) or 0),
            "cav_transport": dict(cav_diag.get("transport", {}) or {}),
            "cav_shared_plan_count": int(
                cav_diag.get("shared_plan_cav_count", 0) or 0
            ),
            "cav_shared_plan_sample_count": int(
                cav_diag.get("shared_plan_sample_count", 0) or 0
            ),
            "cav_conflict_agent_count": int(
                cav_diag.get("conflict_agent_count", 0) or 0
            ),
            "cav_non_ignore_count": int(cav_diag.get("non_ignore_count", 0) or 0),
            "cav_deduplicated_agent_count": int(
                cav_diag.get("deduplicated_agent_count", 0) or 0
            ),
            "cav_longitudinal_qp_row_count": int(
                cav_diag.get("longitudinal_qp_row_count", 0) or 0
            ),
            "cav_homotopy_qp_row_count": int(
                cav_diag.get("homotopy_qp_row_count", 0) or 0
            ),
            "cav_total_qp_row_count": int(
                cav_diag.get("total_qp_row_count", 0) or 0
            ),
            "cav_anticipatory_speed_cap_mps": cav_diag.get(
                "anticipatory_speed_cap_mps", ""
            ),
            "cav_anticipatory_speed_constraint_owner": str(
                cav_diag.get("anticipatory_speed_constraint_owner", "") or ""
            ),
            "cav_coordination_revision": int(
                cav_diag.get("coordination_revision", 0) or 0
            ),
            "cav_coordination_roles_refreshed": bool(
                cav_diag.get("coordination_roles_refreshed", False)
            ),
            "cav_coordination_schedule_reason": str(
                cav_diag.get("coordination_schedule_reason", "") or ""
            ),
            "cav_corridor_rebuilt": bool(
                cav_diag.get("corridor_rebuilt", False)
            ),
            "cav_coordination_refresh_reason": str(
                cav_diag.get("coordination_refresh_reason", "") or ""
            ),
            "cav_multimodal_agent_count": int(
                cav_diag.get("multimodal_agent_count", 0) or 0
            ),
            "cav_credible_mode_veto_count": int(
                cav_diag.get("credible_mode_veto_count", 0) or 0
            ),
            "cav_credible_mode_veto_held_count": int(
                cav_diag.get("credible_mode_veto_held_count", 0) or 0
            ),
            "cav_trajectory_sources": ";".join(
                f"{k}={v}"
                for k, v in sorted(
                    dict(cav_diag.get("trajectory_source_counts", {}) or {}).items()
                )
            ),
            "cav_prediction_validation_actor_id": cav_diag.get(
                "prediction_validation_actor_id", ""
            ),
            "cav_prediction_validation_horizon_s": cav_diag.get(
                "prediction_validation_horizon_s", ""
            ),
            "cav_prediction_validation_x_m": cav_diag.get(
                "prediction_validation_x_m", ""
            ),
            "cav_prediction_validation_y_m": cav_diag.get(
                "prediction_validation_y_m", ""
            ),
            # Viewer-only structured payload; CSV serialization deliberately
            # ignores fields outside the stable scalar schema.
            "cav_shared_planned_paths": dict(
                cav_diag.get("shared_planned_paths", {}) or {}
            ),
            **adapters.perception_diagnostics(),
            "v2x_nearby_count": len(getattr(adapters.v2x_manager, "cav_nearby", {}) or {}),
            "cp_provider_summary": dict(cp_summary),
            "cp_provider_source": str(cp_summary.get("provider_source", "")),
            "native_opencda_required": bool(cp_summary.get("native_opencda_required", False)),
            "native_opencda_available": bool(cp_summary.get("native_opencda_available", False)),
            "cp_obstacle_count": int(cp_summary.get("obstacle_count", 0) or 0),
            "cp_control_count": int(cp_summary.get("control_count", 0) or 0),
            "cp_observer_cav_count": int(
                cp_summary.get("observer_cav_count", 0) or 0
            ),
            "cp_observer_cav_ids": ",".join(
                str(item)
                for item in list(cp_summary.get("observer_cav_ids", []) or [])
            ),
            "cp_multi_observer_obstacle_count": int(
                cp_summary.get("multi_observer_obstacle_count", 0) or 0
            ),
            "cp_blind_spot_shared_count": int(
                cp_summary.get("blind_spot_shared_count", 0) or 0
            ),
            "cp_blind_spot_shared_actor_ids": ",".join(
                str(item)
                for item in list(
                    cp_summary.get("blind_spot_shared_actor_ids", []) or []
                )
            ),
            **adapters.cooperative_actor_evidence(
                cp_summary=cp_summary,
                prediction_trajectories=dict(
                    reference_debug.get("prediction_trajectories", {}) or {}
                ),
                selected_reference=lane_center_reference,
            ),
            "cp_visibility_filter_enabled": bool(
                cp_summary.get("visibility_filter_enabled", False)
            ),
            "cp_visibility_backend": str(
                cp_summary.get("visibility_backend", "")
            ),
            "front_gap_m": "" if front_gap_m is None else float(front_gap_m),
            "stop_goal_active": bool(mpc_stop_goal_active),
            "normal_stop_requested": bool(normal_stop_requested),
            "emergency_brake_requested": bool(emergency_brake_requested),
            "emergency_brake_control_active": bool(
                emergency_brake_requested
                or hard_gate_active
                or str(safety_supervisor_reason).startswith(
                    "safety_supervisor_emergency_stop:"
                )
            ),
            "normal_stop_mpc_suspended": bool(stationary_traffic_stop_hold),
            "normal_stop_mpc_suspend_speed_mps": float(
                adapters.config.get("normal_stop_mpc_suspend_speed_mps", 0.30)
            ),
            "behavior_decision": str(behavior_decision.maneuver),
            "static_obstacle_stop_active_input": bool(
                pipeline.static_obstacle.stop_active
            ),
            "static_obstacle_replan_status": str(
                pipeline.static_obstacle.status
            ),
            "static_obstacle_replan_reason": str(
                pipeline.static_obstacle.reason
            ),
            "static_obstacle_candidate_id": str(
                pipeline.static_obstacle.candidate_id
            ),
            "static_obstacle_blocked_lane_id": adapters.static_obstacle_blocked_lane_id(),
            "static_obstacle_route_transition_pending": bool(
                pipeline.static_obstacle.route_transition_pending
            ),
            "behavior_fsm_state": str(behavior_decision.phase),
            "current_lane_id": int(behavior_decision.source_lane_id),
            "behavior_target_lane_id": int(behavior_decision.target_lane_id),
            "traffic_signal_state": str(behavior_decision.traffic_signal_state),
            "traffic_signal_raw_state": behavior_debug.get(
                "traffic_signal_raw_state", ""
            ),
            "traffic_signal_resolved_state": behavior_debug.get(
                "traffic_signal_resolved_state", ""
            ),
            "traffic_signal_filtered_state": behavior_debug.get(
                "traffic_signal_filtered_state", ""
            ),
            "traffic_signal_behavior_state": behavior_debug.get(
                "traffic_signal_behavior_state",
                behavior_debug.get("traffic_signal_state", ""),
            ),
            "traffic_control_from_cp": behavior_debug.get("traffic_control_from_cp", ""),
            "lane_safety_scores": json.dumps(behavior_debug.get("lane_safety_scores", {}), default=str),
            "reference_pipeline_stage": str(reference_debug.get("stage", "")),
            "reference_pipeline_intent": str(reference_debug.get("intent_mode", "")),
            "reference_pipeline_fallback": str(reference_debug.get("fallback_reason", "")),
            "planner_input_cp_traffic_control_count": reference_debug.get("planner_input_cp_traffic_control_count", ""),
            "planner_input_prediction_risky_lane_count": reference_debug.get("planner_input_prediction_risky_lane_count", ""),
            "planner_input_perception_planning_count": reference_debug.get("planner_input_perception_planning_count", ""),
            "planner_input_cp_obstacle_count": reference_debug.get("planner_input_cp_obstacle_count", ""),
            "planner_input_frame_timestamp_s": reference_debug.get("planner_input_frame_timestamp_s", ""),
            "cp_message_timestamp_s": reference_debug.get("cp_message_timestamp_s", ""),
            "cp_message_age_s": reference_debug.get("cp_message_age_s", ""),
            "cp_message_valid": reference_debug.get("cp_message_valid", ""),
            "destination_x": float(destination_state[0]),
            "destination_y": float(destination_state[1]),
            "destination_forward_m": float(destination_forward_m),
            "destination_lateral_m": float(destination_lateral_m),
            "destination_lane_id": (
                int(destination_state[4]) if len(destination_state) >= 5 else ""
            ),
            "reference_first_forward_m": reference_first_forward_m,
            "reference_first_lateral_m": reference_first_lateral_m,
            "preturn_lane_reference_reason": reference_debug.get(
                "preturn_lane_reference_reason", ""
            ),
            "preturn_raw_first_lateral_m": reference_debug.get(
                "preturn_raw_first_lateral_m", ""
            ),
            "mpc_trajectory_point_count": len(adapters.last_mpc_trajectory_points()),
            "global_route_point_count": len(adapters.display_global_route_points()),
            "global_route_topology_point_count": len(adapters.active_global_route_points()),
            "map_match_valid": bool(
                adapters.route_context.map_matching.get("valid", False)
            ),
            "map_match_ad_lane_id": adapters.route_context.map_matching.get(
                "ad_lane_id", ""
            ),
            "map_match_road_id": adapters.route_context.map_matching.get("road_id", ""),
            "map_match_section_id": adapters.route_context.map_matching.get(
                "section_id", ""
            ),
            "map_match_raw_lane_id": adapters.route_context.map_matching.get(
                "raw_lane_id", ""
            ),
            "map_match_center_x_m": adapters.route_context.map_matching.get(
                "center_x_m", ""
            ),
            "map_match_center_y_m": adapters.route_context.map_matching.get(
                "center_y_m", ""
            ),
            "map_match_lane_width_m": adapters.route_context.map_matching.get(
                "lane_width_m", ""
            ),
            "map_match_lateral_offset_m": adapters.route_context.map_matching.get(
                "lateral_offset_m", ""
            ),
            "map_match_heading_error_rad": adapters.route_context.map_matching.get(
                "heading_error_rad", ""
            ),
            "map_match_score": adapters.route_context.map_matching.get("score", ""),
            "map_match_confidence": adapters.route_context.map_matching.get(
                "confidence", ""
            ),
            "map_match_reason": adapters.route_context.map_matching.get(
                "match_reason", ""
            ),
            "map_match_candidate_count": adapters.route_context.map_matching.get(
                "candidate_count", ""
            ),
            # Read the typed snapshot directly -- local_lane_frame is a
            # write-side compatibility mirror (see RouteContextStage.build's
            # comment), not a diagnostics source of truth.
            "local_lane_frame_cache_reused": bool(
                adapters.route_context.local_map_snapshot.cache_reused
            ),
            "local_lane_frame_generation_reason": str(
                adapters.route_context.local_map_snapshot.generation_reason
            ),
            "local_lane_frame_ego_ad_lane_id": int(
                adapters.route_context.local_map_snapshot.ego_lane_id
            ),
            "local_lane_frame_forward_distance_m": float(
                adapters.route_context.local_map_snapshot.forward_distance_m
            ),
            "local_lane_frame_backward_distance_m": float(
                adapters.route_context.local_map_snapshot.backward_distance_m
            ),
            "local_lane_frame_corridors": json.dumps(
                {
                    int(corridor.offset): list(corridor.lane_ids)
                    for corridor in adapters.route_context.local_map_snapshot.corridors
                },
                sort_keys=True,
            ),
            "local_lane_frame_lane_to_offset": json.dumps(
                dict(adapters.route_context.local_map_snapshot.lane_to_offset),
                sort_keys=True,
            ),
            "local_lane_frame_route_target_ad_lane_id": int(
                adapters.route_context.local_map_snapshot.route_target_lane_id
            ),
            "local_lane_frame_target_in_frame": bool(
                adapters.route_context.local_map_snapshot.route_target_in_frame
            ),
            "local_lane_frame_target_offset": int(
                adapters.route_context.local_map_snapshot.route_target_offset
            ),
            "local_lane_frame_invariant_violations": ";".join(
                str(value)
                for value in list(
                    adapters.route_context.local_map_snapshot.invariant_violations
                    or []
                )
            ),
            "route_reference_allowed": reference_debug.get("route_reference_allowed", ""),
            "route_reference_gate_reason": reference_debug.get("route_reference_gate_reason", ""),
            "route_lane_change_allowed": reference_debug.get("route_lane_change_allowed", ""),
            "opportunistic_lane_change_allowed": reference_debug.get("opportunistic_lane_change_allowed", ""),
            "lane_change_gate_reason": reference_debug.get("lane_change_gate_reason", ""),
            "route_lane_change_required": reference_debug.get("route_lane_change_required", ""),
            "lane_change_authorized": reference_debug.get("lane_change_authorized", ""),
            "lane_change_authorization_direction": reference_debug.get("lane_change_authorization_direction", ""),
            "lane_change_authorization_reason": reference_debug.get("lane_change_authorization_reason", ""),
            "route_lane_change_edge_id": reference_debug.get("route_lane_change_edge_id", ""),
            "completed_route_lane_change_edge_id": reference_debug.get("completed_route_lane_change_edge_id", ""),
            "route_lane_change_edge_completed": reference_debug.get("route_lane_change_edge_completed", ""),
            "lane_change_required_by_route": reference_debug.get("lane_change_required_by_route", ""),
            "lane_change_distance_to_maneuver_m": reference_debug.get("lane_change_distance_to_maneuver_m", ""),
            "lane_change_authorized_target_lane_id": reference_debug.get("lane_change_authorized_target_lane_id", ""),
            "route_maneuver_normalized": reference_debug.get("route_maneuver_normalized", ""),
            "behavior_override_reason": reference_debug.get("behavior_override_reason", ""),
            "reference_follow_global_route_lane": reference_debug.get("reference_pipeline_follow_global_route_lane", ""),
            "route_current_road_option": reference_debug.get("route_current_road_option", ""),
            "route_next_macro_maneuver": reference_debug.get("route_next_macro_maneuver", ""),
            "candidate_evaluation_summary": reference_debug.get("candidate_evaluation_summary", ""),
            "candidate_selected_decision": reference_debug.get("candidate_selected_decision", ""),
            "candidate_selected_lane_id": reference_debug.get("candidate_selected_lane_id", ""),
            "candidate_selected_cost": reference_debug.get("candidate_selected_cost", ""),
            "candidate_pipeline_enabled": reference_debug.get("candidate_pipeline_enabled", ""),
            "candidate_pipeline_selected": reference_debug.get("candidate_pipeline_selected", ""),
            "candidate_pipeline_selected_status": reference_debug.get("candidate_pipeline_selected_status", ""),
            "candidate_pipeline_selected_reason": reference_debug.get("candidate_pipeline_selected_reason", ""),
            "candidate_selected_stop_goal_active": reference_debug.get(
                "candidate_selected_stop_goal_active",
                "",
            ),
            "candidate_pipeline_count": reference_debug.get("candidate_pipeline_count", ""),
            "candidate_prediction_trajectory_count": reference_debug.get("candidate_prediction_trajectory_count", ""),
            "candidate_pipeline_summary": reference_debug.get("candidate_pipeline_summary", ""),
            "candidate_mpc_probe_summary": reference_debug.get("candidate_mpc_probe_summary", ""),
            "candidate_selected_trajectory_variant": reference_debug.get("lane_change_trajectory_variant", ""),
            "candidate_selected_lane_change_duration_s": reference_debug.get("lane_change_duration_s", ""),
            "candidate_selected_lane_change_duration_comfort_reason": reference_debug.get(
                "lane_change_duration_comfort_reason", ""
            ),
            "candidate_selected_lane_change_planning_average_speed_mps": reference_debug.get(
                "lane_change_planning_average_speed_mps", ""
            ),
            "candidate_selected_lane_change_authorization_source": reference_debug.get(
                "lane_change_authorization_source", ""
            ),
            "candidate_selected_lane_change_initial_progress": reference_debug.get(
                "lane_change_initial_progress", ""
            ),
            "candidate_selected_lane_change_terminal_progress": reference_debug.get(
                "lane_change_terminal_progress", ""
            ),
            "route_tracking_lane_change_locked": reference_debug.get(
                "route_tracking_lane_change_locked", ""
            ),
            "route_tracking_lane_change_progress_index": reference_debug.get(
                "route_tracking_lane_change_progress_index", ""
            ),
            "route_tracking_lane_change_source_lane_id": reference_debug.get(
                "route_tracking_lane_change_source_lane_id", ""
            ),
            "route_tracking_lane_change_target_lane_id": reference_debug.get(
                "route_tracking_lane_change_target_lane_id", ""
            ),
            "lane_change_commitment_release_reason": reference_debug.get(
                "lane_change_commitment_release_reason", ""
            ),
            "route_recovery_pending": bool(
                reference_debug.get("route_recovery_pending", False)
            ),
            "lane_change_returns_to_route": bool(
                reference_debug.get("lane_change_returns_to_route", False)
            ),
            "lane_change_completion_reason": reference_debug.get(
                "lane_change_completion_reason", ""
            ),
            "lane_change_completion_stable_frames": reference_debug.get(
                "lane_change_completion_stable_frames", ""
            ),
            "lane_change_completion_lateral_error_m": reference_debug.get(
                "lane_change_completion_lateral_error_m", ""
            ),
            "lane_change_completion_heading_error_deg": reference_debug.get(
                "lane_change_completion_heading_error_deg", ""
            ),
            "lane_change_completion_predicted_lateral_error_m": (
                reference_debug.get(
                    "lane_change_completion_predicted_lateral_error_m", ""
                )
            ),
            "lane_change_contract_min_progress": reference_debug.get(
                "lane_change_contract_min_progress", ""
            ),
            "lane_change_contract_max_lateral_error_m": reference_debug.get(
                "lane_change_contract_max_lateral_error_m", ""
            ),
            "lane_change_contract_max_heading_error_deg": reference_debug.get(
                "lane_change_contract_max_heading_error_deg", ""
            ),
            "lane_change_contract_required_stable_frames": reference_debug.get(
                "lane_change_contract_required_stable_frames", ""
            ),
            "lane_change_contract_handoff_preview_time_s": reference_debug.get(
                "lane_change_contract_handoff_preview_time_s", ""
            ),
            "lane_change_stabilization_entry_lateral_error_m": reference_debug.get(
                "lane_change_stabilization_entry_lateral_error_m", ""
            ),
            "lane_change_stabilization_geometry_ready": reference_debug.get(
                "lane_change_stabilization_geometry_ready", ""
            ),
            "behavior_lane_lateral_error_m": reference_debug.get(
                "behavior_lane_lateral_error_m", ""
            ),
            "behavior_lane_heading_error_deg": reference_debug.get(
                "behavior_lane_heading_error_deg", ""
            ),
            "behavior_lane_alignment_valid": reference_debug.get(
                "behavior_lane_alignment_valid", ""
            ),
            "behavior_lane_change_completion_allowed": reference_debug.get(
                "behavior_lane_change_completion_allowed", ""
            ),
            "lane_change_completion_target_lane_matches": reference_debug.get(
                "lane_change_completion_target_lane_matches", ""
            ),
            "lane_change_completion_footprint_clearance_m": reference_debug.get(
                "lane_change_completion_footprint_clearance_m", ""
            ),
            "lane_change_phase": reference_debug.get("lane_change_phase", ""),
            "lane_change_stabilization_frames": reference_debug.get(
                "lane_change_stabilization_frames", ""
            ),
            "maneuver_commitment_state": reference_debug.get(
                "maneuver_commitment_state", ""
            ),
            "maneuver_commitment_decision": reference_debug.get(
                "maneuver_commitment_decision", ""
            ),
            "maneuver_commitment_source_lane_id": reference_debug.get(
                "maneuver_commitment_source_lane_id", ""
            ),
            "maneuver_commitment_target_lane_id": reference_debug.get(
                "maneuver_commitment_target_lane_id", ""
            ),
            "maneuver_commitment_progress": reference_debug.get(
                "maneuver_commitment_progress", ""
            ),
            "maneuver_commitment_reference_locked": reference_debug.get(
                "maneuver_commitment_reference_locked", ""
            ),
            "maneuver_commitment_active": reference_debug.get(
                "maneuver_commitment_active", ""
            ),
            "maneuver_commitment_committed_at_s": float(
                adapters.maneuver_manager.lane_change.committed_at_s
            ),
            "route_tracking_recovery_active": reference_debug.get(
                "route_tracking_recovery_active", ""
            ),
            "route_tracking_recovery_reason": reference_debug.get(
                "route_tracking_recovery_reason", ""
            ),
            "mpc_feedback_summary": reference_debug.get("mpc_feedback_summary", ""),
            "mpc_feedback_record_reason": str(mpc_feedback_record_reason),
            "mpc_feedback_blocked_lane_ids": reference_debug.get("mpc_feedback_blocked_lane_ids", ""),
            "mode_transition_guard_reason": str(mode_transition_guard_reason),
            "control_buffer_reason": str(adapters.control_buffer.last_reason),
            "control_buffered_step_count": int(adapters.control_buffer.buffered_step_count),
            "mpc_replan_executed": bool(mpc_replan_executed),
            "route_manager_status": json.dumps(
                adapters.route_manager.last_status.as_dict(),
                default=str,
            ),
            "route_replan_attempted": reference_debug.get(
                "route_replan_attempted", False
            ),
            "route_replan_succeeded": reference_debug.get(
                "route_replan_succeeded", False
            ),
            "route_replan_attempt_count": int(
                adapters.route_replan_attempt_count()
            ),
            "route_replan_reason": str(reference_debug.get(
                "route_replan_reason", "route_replan_not_requested"
            )),
            "route_remaining_distance_m": float(
                adapters.route_manager.last_status.remaining_distance_m
            ),
            "route_reached_destination": bool(adapters.route_manager.last_status.reached_destination),
            "mission_complete": bool(
                pipeline.destination_mission_complete
            ),
            "destination_stop_latched": bool(
                pipeline.destination_stop_latched
            ),
            "destination_stop_reason": reference_debug.get(
                "destination_stop_reason", ""
            ),
            "destination_stop_remaining_distance_m": reference_debug.get(
                "destination_stop_remaining_distance_m", ""
            ),
            "destination_stop_required_distance_m": reference_debug.get(
                "destination_stop_required_distance_m", ""
            ),
            "global_planner_backend": str(adapters.global_planner_backend),
            "global_planner_backend_warning": str(adapters.global_planner_backend_warning),
            "tracker_active_count": reference_debug.get("tracker_active_count", ""),
            "tracker_stale_count": reference_debug.get("tracker_stale_count", ""),
            "prediction_validity_reason": reference_debug.get("prediction_validity_reason", ""),
            "scenario_fsm_state": reference_debug.get("scenario_fsm_state", ""),
            "scenario_fsm_reason": reference_debug.get("scenario_fsm_reason", ""),
            "scenario_behavior_signal_state": reference_debug.get("scenario_behavior_signal_state", ""),
            "scenario_behavior_override_decision": reference_debug.get("scenario_behavior_override_decision", ""),
            "scenario_speed_cap_mps": reference_debug.get("scenario_speed_cap_mps", ""),
            "scenario_stop_goal_active": reference_debug.get("scenario_stop_goal_active", ""),
            "scenario_turn_direction": reference_debug.get("scenario_turn_direction", ""),
            "scenario_turn_latched": reference_debug.get("scenario_turn_latched", ""),
            "scenario_boundary_recovery_active": reference_debug.get(
                "scenario_boundary_recovery_active", ""
            ),
            "scenario_boundary_clearance_m": reference_debug.get(
                "scenario_boundary_clearance_m", ""
            ),
            "scenario_boundary_lateral_offset_m": reference_debug.get(
                "scenario_boundary_lateral_offset_m", ""
            ),
            "scenario_boundary_heading_error_rad": reference_debug.get(
                "scenario_boundary_heading_error_rad", ""
            ),
            "boundary_recovery_generation_reason": reference_debug.get(
                "boundary_recovery_generation_reason", ""
            ),
            "boundary_recovery_conditioning_reason": reference_debug.get(
                "boundary_recovery_conditioning_reason", ""
            ),
            "speed_plan_target_mps": reference_debug.get("speed_plan_target_mps", ""),
            "speed_plan_cap_mps": reference_debug.get("speed_plan_cap_mps", ""),
            "speed_plan_stop_goal_active": reference_debug.get("speed_plan_stop_goal_active", ""),
            "speed_plan_reason": reference_debug.get("speed_plan_reason", ""),
            "speed_plan_front_gap_m": reference_debug.get(
                "speed_plan_front_gap_m", ""
            ),
            "speed_plan_desired_follow_gap_m": reference_debug.get(
                "speed_plan_desired_follow_gap_m", ""
            ),
            "speed_plan_continuous_following_active": reference_debug.get(
                "speed_plan_continuous_following_active", ""
            ),
            "speed_plan_idm_acceleration_mps2": reference_debug.get(
                "speed_plan_idm_acceleration_mps2", ""
            ),
            "front_gap_actor_id": reference_debug.get("front_gap_actor_id", ""),
            "front_gap_obstacle_speed_mps": reference_debug.get(
                "front_gap_obstacle_speed_mps", ""
            ),
            "front_gap_obstacle_lane_id": reference_debug.get(
                "front_gap_obstacle_lane_id", ""
            ),
            "front_gap_obstacle_is_source_lane": reference_debug.get(
                "front_gap_obstacle_is_source_lane", ""
            ),
            "semantic_risk_kind": reference_debug.get(
                "semantic_risk_kind", "NONE"
            ),
            "semantic_behavior_action": reference_debug.get(
                "semantic_behavior_action", "NONE"
            ),
            "semantic_object_type": reference_debug.get(
                "semantic_object_type", "unknown"
            ),
            "semantic_observation_source": reference_debug.get(
                "semantic_observation_source", "unknown"
            ),
            "semantic_obstacle_id": reference_debug.get(
                "semantic_obstacle_id", ""
            ),
            "semantic_obstacle_distance_m": reference_debug.get(
                "semantic_obstacle_distance_m", ""
            ),
            "semantic_behavior_reason": reference_debug.get(
                "semantic_behavior_reason", ""
            ),
            "speed_owner_requested_mps": reference_debug.get(
                "speed_owner_requested_mps", ""
            ),
            "speed_owner_scenario_cap_mps": reference_debug.get(
                "speed_owner_scenario_cap_mps", ""
            ),
            "speed_owner_turn_cap_mps": reference_debug.get(
                "speed_owner_turn_cap_mps", ""
            ),
            "speed_owner_lane_change_cap_mps": reference_debug.get(
                "speed_owner_lane_change_cap_mps", ""
            ),
            "speed_owner_following_cap_mps": reference_debug.get(
                "speed_owner_following_cap_mps", ""
            ),
            "speed_owner_turn_approach_cap_mps": reference_debug.get(
                "speed_owner_turn_approach_cap_mps", ""
            ),
            "speed_owner_upcoming_turn_distance_m": reference_debug.get(
                "speed_owner_upcoming_turn_distance_m", ""
            ),
            "speed_owner_selected_target_mps": reference_debug.get(
                "speed_owner_selected_target_mps", ""
            ),
            "speed_owner_limiting_owner": reference_debug.get(
                "speed_owner_limiting_owner", ""
            ),
            "speed_owner_active_constraints": reference_debug.get(
                "speed_owner_active_constraints", ""
            ),
            "speed_owner_mpc_entry_target_mps": float(speed_ref_mps),
            "speed_owner_post_plan_delta_mps": (
                float(speed_ref_mps)
                - float(speed_target.target_mps)
            ),
            "speed_owner_target_overridden_after_plan": abs(
                float(speed_ref_mps)
                - float(speed_target.target_mps)
            ) > 1.0e-6,
            "speed_owner_proposed_post_plan_target_mps": reference_debug.get(
                "speed_owner_proposed_post_plan_target_mps", ""
            ),
            "speed_owner_ceiling_applied": reference_debug.get(
                "speed_owner_ceiling_applied", ""
            ),
            "speed_owner_ceiling_reduction_mps": reference_debug.get(
                "speed_owner_ceiling_reduction_mps", ""
            ),
            "route_turn_reference_reason": str(
                reference_debug.get("route_turn_reference_reason", "")
            ),
            "route_turn_raw_first_forward_m": reference_debug.get(
                "route_turn_raw_first_forward_m", ""
            ),
            "route_turn_raw_first_lateral_m": reference_debug.get(
                "route_turn_raw_first_lateral_m", ""
            ),
            "route_debug_reason": str(adapters.route_manager.route_debug_reason),
            "route_sync_reason": str(adapters.route_manager.route_sync_reason),
            "route_progress_index": int(adapters.route_manager.route_progress_index),
            "route_progress_s_m": float(adapters.route_manager.route_progress_s_m),
            "route_cursor_stalled_motion_m": float(
                adapters.route_manager.route_cursor.stalled_motion_m
            ),
            "route_cursor_missed_maneuver": bool(
                adapters.route_manager.route_cursor.missed_maneuver
            ),
            "route_topology_valid": reference_debug.get("route_topology_valid", ""),
            "route_topology_signature": reference_debug.get(
                "route_topology_signature", ""
            ),
            "route_topology_errors": reference_debug.get("route_topology_errors", ""),
            "route_topology_warnings": reference_debug.get(
                "route_topology_warnings", ""
            ),
            "route_geometry_lane_change_direction": reference_debug.get(
                "route_geometry_lane_change_direction", ""
            ),
            "route_geometry_lane_change_distance_m": reference_debug.get(
                "route_geometry_lane_change_distance_m", ""
            ),
            "route_geometry_lane_change_reason": reference_debug.get(
                "route_geometry_lane_change_reason", ""
            ),
            "route_physical_target_lane_id": reference_debug.get(
                "route_physical_target_lane_id", ""
            ),
            "route_topology_target_lane_id": reference_debug.get(
                "route_topology_target_lane_id", ""
            ),
            "waypoint_backend": str(adapters.waypoint_backend),
            "route_upcoming_turn_direction": str(
                reference_debug.get("route_upcoming_turn_direction", "")
            ),
            "route_upcoming_turn_distance_m": reference_debug.get(
                "route_upcoming_turn_distance_m", ""
            ),
            "route_upcoming_turn_reason": str(
                reference_debug.get("route_upcoming_turn_reason", "")
            ),
            "reference_lateral_guard_reason": str(reference_debug.get("reference_lateral_guard_reason", "")),
            "mpc_reference_stabilizer_reason": str(reference_debug.get("mpc_reference_stabilizer_reason", "")),
            "final_reference_gate_valid": reference_debug.get(
                "final_reference_gate_valid", ""
            ),
            "final_reference_gate_mode": reference_debug.get(
                "final_reference_gate_mode", ""
            ),
            "final_reference_gate_reason": reference_debug.get(
                "final_reference_gate_reason", ""
            ),
            "reference_max_curvature_1pm": reference_debug.get(
                "reference_max_curvature_1pm", ""
            ),
            "reference_first_curvature_1pm": (
                float(lane_center_reference[0].get("curvature_1pm", 0.0))
                if lane_center_reference
                else ""
            ),
            "reference_second_curvature_1pm": (
                float(lane_center_reference[1].get("curvature_1pm", 0.0))
                if lane_center_reference and len(lane_center_reference) > 1
                else ""
            ),
            "reference_contract_max_curvature_1pm": reference_debug.get(
                "reference_contract_max_curvature_1pm", ""
            ),
            "reference_curvature_margin_1pm": reference_debug.get(
                "reference_curvature_margin_1pm", ""
            ),
            "reference_pipeline_conditioning_reason": reference_debug.get(
                "reference_pipeline_conditioning_reason", ""
            ),
            "post_turn_exit_reference_source": reference_debug.get(
                "post_turn_exit_reference_source", ""
            ),
            "reference_pipeline_mode": reference_debug.get(
                "reference_pipeline_mode", ""
            ),
            "mpc_entry_allowed": reference_debug.get("mpc_entry_allowed", ""),
            "mpc_entry_status": reference_debug.get("mpc_entry_status", ""),
            "mpc_entry_reason": reference_debug.get("mpc_entry_reason", ""),
            "pipeline_error": str(reference_debug.get("pipeline_error", behavior_debug.get("pipeline_error", ""))),
            "pipeline_error_traceback": str(
                reference_debug.get(
                    "pipeline_error_traceback",
                    behavior_debug.get("pipeline_error_traceback", ""),
                )
            ),
            "stop_target_forward_m": stop_target_forward_m_debug,
            "mpc_trajectory_points": adapters.last_mpc_trajectory_points(),
            "global_route_points": adapters.route_points_for_display(
                str(getattr(adapters.route_manager, "route_revision", ""))
            ),
            "lane_reference_points": [
                [
                    float(sample.get("x_ref_m", sample.get("x", 0.0))),
                    float(sample.get("y_ref_m", sample.get("y", 0.0))),
                ]
                for sample in list(lane_center_reference or [])
            ],
            "target_speed_mps": float(speed_ref_mps),
            "mpc_status": str(mpc_status),
            "mpc_feasibility_checked": bool(mpc_replan_executed),
            "mpc_feasibility_status": str(mpc_status),
            "mpc_feasibility_reason": str(fallback_reason),
            # Which AD-map lane-geometry queries actually raised this tick
            # (lane_id, parametric_offset, exception) -- a prior guess that
            # validate_turn_swept_footprint's "no_corridor_geometry" came
            # from a parametric_offset landing right on a lane-boundary
            # seam was tried (a clamp-and-retry in admap_backend.py) and
            # made no measurable difference on a live run, so this replaces
            # that guess with the actual query failures instead of another
            # guess. See Global_Planner.global_planner.admap_backend
            # .get_recent_query_failures / _record_query_failure.
            "admap_query_failures": _recent_admap_query_failures(),
            # Section-level snapshot of which constraint groups were switched
            # on for the QP that just failed (road envelope/obstacle count,
            # corridor rows, terminal-stop constraint, etc.) -- captured by
            # MPC._build_qp/_solve_qp only when the solve is infeasible, so
            # this stays empty on every normally-solved tick. See mpc.py's
            # _last_qp_diagnostic / _last_infeasibility_diagnostic.
            "mpc_infeasibility_diagnostic": dict(
                getattr(adapters.mpc, "_last_infeasibility_diagnostic", {}) or {}
            ),
            "mpc_solve_time_ms": float(getattr(adapters.mpc, "_last_solve_time_ms", 0.0)),
            "mpc_nominal_steering_first_rad": (
                float(adapters.mpc._last_nominal_steering_profile[1])
                if len(getattr(adapters.mpc, "_last_nominal_steering_profile", ())) > 1
                else ""
            ),
            "mpc_nominal_steering_max_abs_rad": (
                max(abs(float(value)) for value in adapters.mpc._last_nominal_steering_profile)
                if getattr(adapters.mpc, "_last_nominal_steering_profile", ())
                else ""
            ),
            "mpc_steering_reference_weight": float(
                getattr(adapters.mpc, "_last_steering_reference_weight", 0.0)
            ),
            **pipeline.mpc_cost_profile.state.trace_fields(),
            "mpc_minimum_progress_enabled": bool(
                getattr(adapters.mpc, "minimum_progress_enabled", False)
            ),
            "mpc_minimum_progress_target_mps": float(
                getattr(adapters.mpc, "turn_minimum_progress_speed_mps", 0.0)
            ),
            "mpc_minimum_progress_active": bool(
                str(getattr(adapters.mpc, "active_cost_profile_name", ""))
                == "intersection_turn"
                and bool(getattr(adapters.mpc, "minimum_progress_enabled", False))
            ),
            "reference_source": str(reference_debug.get(
                "reference_source",
                "map_lane_center" if lane_center_reference else "straight_fallback",
            )),
            "final_reference_geometry_source": str(
                reference_debug.get(
                    "final_reference_geometry_source",
                    reference_debug.get("reference_source", "unknown"),
                )
            ),
            "fallback_reason": fallback_reason,
            "mpc_fallback_reason": fallback_reason,
            "control_guard_reason": str(control_guard_reason),
            "accel_cmd_mps2": float(accel_mps2),
            "steer_cmd_rad": float(steer_rad),
            "pre_supervisor_accel_cmd_mps2": float(pre_supervisor_accel_mps2),
            "pre_supervisor_steer_cmd_rad": float(pre_supervisor_steer_rad),
            "post_supervisor_accel_cmd_mps2": float(post_supervisor_accel_mps2),
            "post_supervisor_steer_cmd_rad": float(post_supervisor_steer_rad),
            "applied_throttle": float(getattr(control, "throttle", 0.0)),
            "applied_brake": float(getattr(control, "brake", 0.0)),
            "applied_steer": float(getattr(control, "steer", 0.0)),
            "platform_applied_steer_rad": float(
                getattr(control, "steer", 0.0)
            ) * float(getattr(
                adapters.vehicle_dynamics,
                "actuator_max_steer_rad",
                adapters.mpc.constraints.max_steer_rad,
            )),
            "planner_requested": True,
            "planner_executed": True,
            "fallback_active": bool(fallback_reason),
            "fallback_policy": str(adapters.fallback_policy),
            "fallback_policy_warning": str(adapters.fallback_policy_warning),
            "safety_supervisor_reason": str(safety_supervisor_reason),
            "turn_boundary_recovery_active": bool(
                adapters.safety_supervisor.turn_boundary_recovery_active
            ),
            "turn_boundary_recovery_phase": str(
                adapters.safety_supervisor.turn_boundary_recovery_phase
            ),
        }
        diagnostics.update(adapters.architecture_profile.as_debug_fields())
        diagnostics.update(
            adapters.route_context.local_map_snapshot.trace_fields()
        )
        diagnostics.update(
            adapters.reference_line_provider.snapshot(
                LANE_FOLLOW
            ).trace_fields()
        )
        diagnostics.update(
            adapters.update_evaluation_metrics(
                ego_location=ego_location,
                ego_speed_mps=float(ego_speed_mps),
                ego_yaw_rad=float(ego_yaw_rad),
                object_snapshots=object_snapshots,
                behavior_decision=str(behavior_decision.maneuver),
                behavior_fsm_state=str(behavior_decision.phase),
                mpc_replan_executed=bool(mpc_replan_executed),
                cp_summary=cp_summary,
                reference_samples=lane_center_reference,
                boundary_snapshot=boundary_snapshot,
            )
        )
        return diagnostics
    
