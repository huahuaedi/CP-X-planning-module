"""Read-only assembly of the stable planner diagnostic schema."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .local_map_snapshot import LocalMapSnapshot
from .reference_line_provider import LANE_FOLLOW


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


class PlannerDiagnosticsStage:
    """Build diagnostics without participating in planning or control."""

    @staticmethod
    def fieldnames(payload: Mapping[str, Any]) -> tuple[str, ...]:
        """Return the stable serialization order for a built trace payload."""

        return tuple(str(key) for key in payload.keys())

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
            "reference_source": "behavior_reference_pipeline",
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
        return debug

    @staticmethod
    def build(owner: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        self = owner
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
        cp_summary = dict(getattr(self.cp_provider, "last_publish_summary", {}) or {})
        cav_resolution = context.get("cav_resolution")
        cav_diag = dict(getattr(cav_resolution, "diagnostics", {}) or {})
        diagnostics = {
            "sim_time_s": float(self._sim_time_s()),
            "vehicle_id": int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            "x_m": float(ego_location.x),
            "y_m": float(ego_location.y),
            "yaw_deg": float(ego_transform.rotation.yaw),
            "speed_mps": float(ego_speed_mps),
            "measured_accel_mps2": float(measured_accel_mps2),
            "mpc_jerk_seed_accel_mps2": float(mpc_jerk_seed_accel_mps2),
            "planner": "cpx_mpc",
            **platform_adapter_debug,
            "object_count": len(object_snapshots),
            "mpc_object_count": len(mpc_object_snapshots),
            "local_object_count": len(local_object_snapshots),
            "prediction_mode": str(self._prediction_mode),
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
            "cav_multimodal_agent_count": int(
                cav_diag.get("multimodal_agent_count", 0) or 0
            ),
            "cav_credible_mode_veto_count": int(
                cav_diag.get("credible_mode_veto_count", 0) or 0
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
            **self._perception_diagnostics(),
            "v2x_nearby_count": len(getattr(self.vehicle_manager.v2x_manager, "cav_nearby", {}) or {}),
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
            **self._cooperative_actor_evidence(
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
                self.config.get("normal_stop_mpc_suspend_speed_mps", 0.30)
            ),
            "normal_stop_mpc_suspend_brake": float(
                self.config.get("normal_stop_mpc_suspend_brake", 0.08)
            ),
            "behavior_decision": str(behavior_decision.maneuver),
            "static_obstacle_stop_active_input": bool(
                self.pipeline.static_obstacle.stop_active
            ),
            "static_obstacle_replan_status": str(
                self.pipeline.static_obstacle.status
            ),
            "static_obstacle_replan_reason": str(
                self.pipeline.static_obstacle.reason
            ),
            "static_obstacle_candidate_id": str(
                self.pipeline.static_obstacle.candidate_id
            ),
            "static_obstacle_blocked_lane_id": getattr(
                self, "_static_obstacle_blocked_lane_id", ""
            ),
            "static_obstacle_route_transition_pending": bool(
                self.pipeline.static_obstacle.route_transition_pending
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
            "mpc_trajectory_point_count": len(self._last_mpc_trajectory_points()),
            "global_route_point_count": len(self._display_global_route_points()),
            "global_route_topology_point_count": len(self._active_global_route_points()),
            "map_match_valid": bool(
                self._diagnostic_map_matching.get("valid", False)
            ),
            "map_match_ad_lane_id": self._diagnostic_map_matching.get(
                "ad_lane_id", ""
            ),
            "map_match_road_id": self._diagnostic_map_matching.get("road_id", ""),
            "map_match_section_id": self._diagnostic_map_matching.get(
                "section_id", ""
            ),
            "map_match_raw_lane_id": self._diagnostic_map_matching.get(
                "raw_lane_id", ""
            ),
            "map_match_center_x_m": self._diagnostic_map_matching.get(
                "center_x_m", ""
            ),
            "map_match_center_y_m": self._diagnostic_map_matching.get(
                "center_y_m", ""
            ),
            "map_match_lane_width_m": self._diagnostic_map_matching.get(
                "lane_width_m", ""
            ),
            "map_match_lateral_offset_m": self._diagnostic_map_matching.get(
                "lateral_offset_m", ""
            ),
            "map_match_heading_error_rad": self._diagnostic_map_matching.get(
                "heading_error_rad", ""
            ),
            "map_match_score": self._diagnostic_map_matching.get("score", ""),
            "map_match_confidence": self._diagnostic_map_matching.get(
                "confidence", ""
            ),
            "map_match_reason": self._diagnostic_map_matching.get(
                "match_reason", ""
            ),
            "map_match_candidate_count": self._diagnostic_map_matching.get(
                "candidate_count", ""
            ),
            "local_lane_frame_cache_reused": bool(
                self._diagnostic_local_lane_frame.get("cache_reused", False)
            ),
            "local_lane_frame_generation_reason": self._diagnostic_local_lane_frame.get(
                "generation_reason", ""
            ),
            "local_lane_frame_ego_ad_lane_id": self._diagnostic_local_lane_frame.get(
                "ego_ad_lane_id", ""
            ),
            "local_lane_frame_forward_distance_m": self._diagnostic_local_lane_frame.get(
                "forward_distance_m", ""
            ),
            "local_lane_frame_backward_distance_m": self._diagnostic_local_lane_frame.get(
                "backward_distance_m", ""
            ),
            "local_lane_frame_corridors": json.dumps(
                self._diagnostic_local_lane_frame.get("corridors", {}),
                sort_keys=True,
            ),
            "local_lane_frame_lane_to_offset": json.dumps(
                self._diagnostic_local_lane_frame.get("lane_to_offset", {}),
                sort_keys=True,
            ),
            "local_lane_frame_route_target_ad_lane_id": self._diagnostic_local_lane_frame.get(
                "route_target_ad_lane_id", ""
            ),
            "local_lane_frame_target_in_frame": self._diagnostic_local_lane_frame.get(
                "route_target_in_frame", ""
            ),
            "local_lane_frame_target_offset": self._diagnostic_local_lane_frame.get(
                "route_target_offset", ""
            ),
            "local_lane_frame_invariant_violations": ";".join(
                str(value)
                for value in list(
                    self._diagnostic_local_lane_frame.get(
                        "invariant_violations", []
                    )
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
                self.maneuver_manager.lane_change.committed_at_s
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
            "control_buffer_reason": str(self.control_buffer.last_reason),
            "control_buffered_step_count": int(self.control_buffer.buffered_step_count),
            "mpc_replan_executed": bool(mpc_replan_executed),
            "route_manager_status": json.dumps(
                self.route_manager.last_status.as_dict(),
                default=str,
            ),
            "route_replan_attempted": reference_debug.get(
                "route_replan_attempted", False
            ),
            "route_replan_succeeded": reference_debug.get(
                "route_replan_succeeded", False
            ),
            "route_replan_attempt_count": int(
                self._route_replan_attempt_count
            ),
            "route_replan_reason": str(reference_debug.get(
                "route_replan_reason", "route_replan_not_requested"
            )),
            "route_remaining_distance_m": float(
                self.route_manager.last_status.remaining_distance_m
            ),
            "route_reached_destination": bool(self.route_manager.last_status.reached_destination),
            "destination_stop_latched": bool(
                self.pipeline.destination_stop_latched
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
            "global_planner_backend": str(self.global_planner_backend),
            "global_planner_backend_warning": str(self.global_planner_backend_warning),
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
            "route_debug_reason": str(self.route_manager.route_debug_reason),
            "route_sync_reason": str(self.route_manager.route_sync_reason),
            "route_progress_index": int(self.route_manager.route_progress_index),
            "route_progress_s_m": float(self.route_manager.route_progress_s_m),
            "route_cursor_stalled_motion_m": float(
                self.route_manager.route_cursor.stalled_motion_m
            ),
            "route_cursor_missed_maneuver": bool(
                self.route_manager.route_cursor.missed_maneuver
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
            "waypoint_backend": str(self.waypoint_backend),
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
            "mpc_trajectory_points": self._last_mpc_trajectory_points(),
            "global_route_points": self._display_global_route_points(),
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
            "mpc_solve_time_ms": float(getattr(self.mpc, "_last_solve_time_ms", 0.0)),
            "mpc_cost_profile": str(self.active_mpc_cost_profile),
            "requested_mpc_cost_profile": str(self.requested_mpc_cost_profile),
            "mpc_cost_profile_switch_reason": str(self.mpc_cost_profile_switch_reason),
            "mpc_minimum_progress_enabled": bool(
                getattr(self.mpc, "minimum_progress_enabled", False)
            ),
            "mpc_minimum_progress_target_mps": float(
                getattr(self.mpc, "turn_minimum_progress_speed_mps", 0.0)
            ),
            "mpc_minimum_progress_active": bool(
                str(getattr(self.mpc, "active_cost_profile_name", ""))
                == "intersection_turn"
                and bool(getattr(self.mpc, "minimum_progress_enabled", False))
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
            ) * float(self.mpc.constraints.max_steer_rad),
            "planner_requested": True,
            "planner_executed": True,
            "fallback_active": bool(fallback_reason),
            "fallback_policy": str(self.fallback_policy),
            "fallback_policy_warning": str(self.fallback_policy_warning),
            "safety_supervisor_reason": str(safety_supervisor_reason),
            "turn_boundary_recovery_active": bool(
                self.safety_supervisor.turn_boundary_recovery_active
            ),
            "turn_boundary_recovery_phase": str(
                self.safety_supervisor.turn_boundary_recovery_phase
            ),
        }
        diagnostics.update(self.architecture_profile.as_debug_fields())
        diagnostics.update(
            getattr(self, "_local_map_snapshot", LocalMapSnapshot()).trace_fields()
        )
        diagnostics.update(
            self._stable_reference_line_provider.snapshot(
                LANE_FOLLOW
            ).trace_fields()
        )
        diagnostics.update(
            self._update_evaluation_metrics(
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
    
