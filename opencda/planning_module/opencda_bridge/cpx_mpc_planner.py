"""Bridge from native OpenCDA vehicle managers to the CP-X MPC planner.

The bridge is intentionally small: OpenCDA still owns simulation, localization,
perception, and V2X discovery. This class consumes a custom map planner and
returns a CARLA ``VehicleControl`` directly, replacing both
OpenCDA's behavior agent and PID controller when enabled.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import yaml

from opencda.planning_module.utility.carla_compat import carla

from opencda.planning_module.pipeline.traffic_light_memory import (
    TrafficLightMemory,
)
from opencda.planning_module.pipeline.behavior_decision import BehaviorDecision
from opencda.planning_module.pipeline.map_matching import (
    DiagnosticHDMapMatcher,
    LaneProjectionCandidate,
    local_lane_frame_invariants,
    topology_relation,
)
from opencda.planning_module.pipeline.local_map_snapshot import (
    LocalMapSnapshot,
    build_local_map_snapshot,
)
from opencda.planning_module.pipeline.reference_line_provider import (
    LANE_CHANGE,
    LANE_FOLLOW,
    POST_TURN,
    TURN,
    CandidateReferenceBuildContext,
    ReferenceLineProvider,
    ReferenceLineRequest,
    TurnReferenceRequest,
)
from opencda.planning_module.pipeline.maneuver_manager import ManeuverManager
from opencda.planning_module.pipeline.nominal_trajectory import (
    NominalTrajectoryGenerator,
)
from opencda.planning_module.pipeline.fallback_manager import (
    TrajectoryFallbackManager,
)
from opencda.planning_module.pipeline.speed_planner import (
    SpeedConstraint,
    SpeedTargetPlanner,
)
from opencda.planning_module.pipeline.destination_speed_stage import (
    DestinationSpeedStage,
)
from opencda.planning_module.pipeline.reference_publication_stage import (
    ReferencePublicationStage,
)
from opencda.planning_module.pipeline.mpc_entry_stage import MPCEntryStage
from opencda.planning_module.pipeline.mpc_execution_stage import (
    MPCExecutionRequest,
    MPCExecutionStage,
)
from opencda.planning_module.pipeline.perception_stage import PerceptionStage
from opencda.planning_module.pipeline.execution_pipeline import (
    PlanningPipeline,
    ScenarioPlanningFrameRequest,
)
from opencda.planning_module.pipeline.static_obstacle_stage import StaticObstacleStage
from opencda.planning_module.pipeline.control_safety_stage import ControlSafetyStage
from opencda.planning_module.pipeline.candidate_evaluation import (
    CandidateTrajectoryEvaluator,
    mpc_cost_profile_for_behavior,
)
from opencda.planning_module.pipeline.candidate_selection_stage import (
    CandidateArbitrationRequest,
    CandidateSelectionStage,
)
from opencda.planning_module.pipeline.behavior_stage import (
    BehaviorCommandFrameRequest,
    BehaviorOverrideRequest,
    BehaviorStage,
    ConflictResolutionRequest,
    OpportunisticLaneChangeRequest,
)


class _WaypointMapAdapter:
    """Normalize waypoint queries across AD-map and CARLA map providers."""

    def __init__(self, map_planner: Any):
        self._map_planner = map_planner

    def get_waypoint(self, point: Any):
        get_waypoint = getattr(self._map_planner, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        if isinstance(point, Mapping):
            # The AD-map adapter consumes plain point mappings. Try that
            # platform-neutral form before constructing a CARLA Location.
            try:
                return get_waypoint(point)
            except Exception:
                pass
            location = carla.Location(
                x=float(point.get("x", 0.0)),
                y=float(point.get("y", 0.0)),
                z=float(point.get("z", 0.0)),
            )
            try:
                return get_waypoint(location)
            except Exception:
                return None
        try:
            return get_waypoint(point)
        except Exception:
            return None


class CPXMPCPlannerBridge:
    """Direct-control planner used inside ``VehicleManager.run_step``."""

    @property
    def reference_generator(self):
        """Compatibility view; ReferenceLineProvider owns the builder."""
        return self._stable_reference_line_provider.builder

    @reference_generator.setter
    def reference_generator(self, builder):
        provider = getattr(self, "_stable_reference_line_provider", None)
        if provider is None:
            provider = ReferenceLineProvider()
            self._stable_reference_line_provider = provider
        provider.attach_builder(builder)

    def __init__(
        self,
        vehicle_manager: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        map_planner: Any = None,
    ):
        self._ensure_planning_module_import_path()
        self.vehicle_manager = vehicle_manager
        from opencda.planning_module.pipeline.architecture_profile import (
            normalize_architecture_config,
        )

        self.config, self.architecture_profile = normalize_architecture_config(config)
        self.carla = carla
        self.map_planner = map_planner or getattr(vehicle_manager, "carla_map", None)
        self.waypoint_map_planner = self.map_planner
        # A real `carla.Map` exposes `get_topology`; the CARLA-free path passes a
        # CustomGlobalPlannerAdapter here instead, which the CARLA-GRP route
        # builder must not be handed.
        self._map_planner_is_carla_map = (
            self.map_planner is not None
            and callable(getattr(self.map_planner, "get_topology", None))
        )
        self.enabled = bool(self.config.get("enabled", True))
        self.mode = str(self.config.get("mode", "full_cpx_mpc")).strip().lower()
        self.fallback_policy = str(
            self.config.get("fallback_policy", "emergency_stop")
        ).strip().lower()
        self.fallback_policy_warning = ""
        if self.fallback_policy == "opencda":
            self.fallback_policy = "emergency_stop"
            self.fallback_policy_warning = "opencda_fallback_disabled_in_full_cpx_mpc"
        self.use_opencda_global_route = bool(
            self.config.get("use_opencda_global_route", True)
        )
        self.opencda_global_route_reference_allowed = bool(
            self.config.get("opencda_global_route_reference_allowed", True)
        )
        self.target_speed_mps = float(self.config.get("target_speed_mps", 8.0))
        self.lookahead_m = float(self.config.get("lookahead_m", 18.0))
        self.min_front_gap_m = float(self.config.get("min_front_gap_m", 8.0))
        # min_front_gap_m alone is a flat distance that doesn't scale with
        # cruise speed: at 11.18 m/s the default 8.0m gave several
        # seconds of reaction margin before target_lane_prediction_risk
        # would trip, but at 20 m/s the same 8.0m is covered in half the
        # time -- confirmed via telemetry (Interactive_Lane_Change's
        # queued lane change missed again at 20 m/s cruise, tripping this
        # exact check, after being fixed at 11.18 m/s). min_front_gap_time_s
        # defaults to 8.0/11.18 so today's calibrated distance is
        # reproduced exactly at 11.18 m/s, while the effective floor grows
        # proportionally with whatever cruise speed is configured.
        self.min_front_gap_time_s = float(
            self.config.get("min_front_gap_time_s", 8.0 / 11.18)
        )
        self.max_mpc_obstacles = max(0, int(self.config.get("max_mpc_obstacles", 4)))
        # Scenario-only isolation switch.  Map topology, route/reference
        # generation, road boundaries and MPC remain active; dynamic actor
        # observations are withheld from behavior, prediction and MPC so the
        # geometric planning chain can be tested deterministically.
        self.functional_test_ignore_dynamic_objects = bool(
            self.config.get("functional_test_ignore_dynamic_objects", False)
        )
        self._functional_test_world_actors_cleaned = False
        self.debug = bool(self.config.get("debug", True))
        self.last_debug: dict[str, Any] = {}
        self._last_accel_mps2 = 0.0
        self._last_steer_rad = 0.0
        # Multi-CAV interaction pipeline (classify -> assign -> corridor).
        # Off unless cav_conflict_enabled; also needs cost.corridor.enabled in
        # mpc.yaml for the corridor to bind in the QP.
        self._cav_conflict_enabled = bool(
            self.config.get("cav_conflict_enabled", False)
        )
        self._cav_latch: dict[str, Any] = {}
        self._cav_tag_state: dict[str, str] = {}
        self._last_cav_diagnostics: dict[str, Any] = {}
        # This CAV's own broadcast for nearby CP-X CAVs to read (its planned
        # trajectory + ResourceClaim + pose). Read peer-to-peer through
        # v2x_manager.cav_nearby; see _publish_cav_intent / _collect_cav_intents.
        self.last_cav_intent_payload = None
        self._cav_intent_sequence = 0
        self._cav_intent_broadcast_enabled = bool(
            self.config.get("cav_intent_broadcast_enabled", True)
        )
        from opencda.planning_module.pipeline.cooperative_claim_manager import (
            CooperativeClaimManager,
        )
        self._cooperative_claim_manager = CooperativeClaimManager(
            enabled=bool(self._cav_conflict_enabled),
            proposal_dwell_s=float(self.config.get("cav_proposal_dwell_s", 0.25)),
        )
        self._warned = False
        self._diagnostic_hd_map_matcher = DiagnosticHDMapMatcher()
        self._diagnostic_local_lane_frame: dict[str, object] = {}
        self._diagnostic_map_matching: dict[str, object] = {}
        self._local_map_frame_id = 0
        self._local_map_snapshot = LocalMapSnapshot()
        self._stable_reference_line_provider = ReferenceLineProvider()
        self.maneuver_manager = ManeuverManager(self.config)
        self.nominal_trajectory_generator = NominalTrajectoryGenerator()
        speed_target_planner = SpeedTargetPlanner()
        behavior_stage = BehaviorStage()
        self._authoritative_ego_waypoint: Any = None
        self._stop_release_temp_smooth_until_sim_time_s = 0.0
        self._full_signal_actor_id = ""
        self._full_traffic_memory = TrafficLightMemory(
            hold_unknown_s=float(self.config.get("full_traffic_unknown_hold_s", 1.5)),
            hold_green_unknown_s=float(
                self.config.get("full_traffic_green_unknown_hold_s", 0.25)
            ),
            green_confirm_s=float(self.config.get("full_traffic_green_confirm_s", 0.15)),
            hold_stop_unknown_until_green=bool(
                self.config.get(
                    "full_traffic_hold_stop_unknown_until_green",
                    False,
                )
            ),
        )
        from opencda.planning_module.pipeline.scenario_manager import (
            BoundaryRecoveryRequest,
            CPXScenarioManager,
        )

        scenario_manager = CPXScenarioManager(self.config)
        self._boundary_recovery_request = BoundaryRecoveryRequest()
        self._boundary_recovery_trigger_frames = 0
        self._boundary_recovery_infeasible_frames = 0
        self._boundary_recovery_cooldown_until_s = -float("inf")
        self._full_last_behavior_mode_key = ""
        fallback_manager = TrajectoryFallbackManager(
            max_hold_age_s=float(self.config.get("fallback_hold_last_valid_s", 0.35)),
            min_hold_arc_m=float(self.config.get("fallback_min_valid_arc_m", 2.0)),
            safe_stop_deceleration_mps2=float(
                self.config.get("fallback_safe_stop_deceleration_mps2", 2.0)
            ),
        )
        destination_speed_stage = DestinationSpeedStage(
            config=self.config,
            speed_planner=speed_target_planner,
            fallback_manager=fallback_manager,
            behavior_stage=behavior_stage,
        )
        from opencda.planning_module.pipeline.behavior_reference_execution_stage import (
            BehaviorReferenceExecutionStage,
        )
        behavior_reference_execution_stage = BehaviorReferenceExecutionStage(
            reference_provider=self._stable_reference_line_provider,
            fallback_manager=fallback_manager,
            behavior_stage=behavior_stage,
            lane_id_at_location=self._lane_id_at_location,
        )
        self._route_replan_last_attempt_s = -float("inf")
        self._route_replan_attempt_count = 0
        self._route_replan_last_reason = "route_replan_not_requested"
        self._static_obstacle_blocked_lane_id: object = ""
        self.draw_world_debug = bool(self.config.get("draw_world_debug", False))
        self.draw_world_debug_destination = bool(
            self.config.get("draw_world_debug_destination", False)
        )
        self.world_debug_life_time_s = float(self.config.get("world_debug_life_time_s", 0.15))
        self.full_control_buffer_min_speed_mps = max(
            0.0,
            float(self.config.get("full_control_buffer_min_speed_mps", 1.5)),
        )
        self.full_lane_change_start_lock_s = max(
            0.0,
            float(self.config.get("full_lane_change_start_lock_s", 8.0)),
        )
        self.full_dense_traffic_lane_change_lock_enabled = bool(
            self.config.get("full_dense_traffic_lane_change_lock_enabled", True)
        )
        self.full_dense_traffic_object_count = max(
            0,
            int(self.config.get("full_dense_traffic_object_count", 8)),
        )
        self.full_dense_traffic_risky_lane_count = max(
            0,
            int(self.config.get("full_dense_traffic_risky_lane_count", 2)),
        )
        self.full_prepare_lane_change_reference_lock = bool(
            self.config.get("full_prepare_lane_change_reference_lock", True)
        )
        self.full_allow_opportunistic_lane_change = bool(
            self.config.get("full_allow_opportunistic_lane_change", False)
        )
        self.full_mpc_reference_stabilizer_enabled = bool(
            self.config.get("full_mpc_reference_stabilizer_enabled", True)
        )
        self.full_candidate_pipeline_enabled = bool(
            self.config.get("full_candidate_pipeline_enabled", True)
        )
        self.full_candidate_reference_min_object_distance_m = max(
            0.0,
            float(self.config.get("full_candidate_reference_min_object_distance_m", 2.0)),
        )
        # The generic clearance above (default 2.0m, configured to 3.5m here)
        # sizes lateral gaps for negotiating with *moving* traffic. Applied
        # unmodified to a static-obstacle local-avoidance candidate it is
        # self-defeating: the whole point of that candidate is to pass close
        # to the very obstacle it is routing around, in a lane only ~3.5m
        # wide, so it always scores infeasible and the vehicle never moves
        # (confirmed via decision_veto_chain: all three lane-change variants
        # rejected on candidate_prediction_collision_risk ~1.1-1.3m, the
        # ego's own predicted clearance from the blocking obstacle, static
        # across assertive/normal/conservative timing since the obstacle
        # isn't moving). Use a tighter, still-conservative clearance just for
        # the candidate whose target lane matches the selected local-
        # avoidance lane; every other candidate keeps the full margin above.
        self.static_obstacle_local_avoidance_min_object_distance_m = max(
            0.0,
            float(
                self.config.get(
                    "static_obstacle_local_avoidance_min_object_distance_m",
                    0.8,
                )
            ),
        )
        # Two same-lane candidates (e.g. full-speed "keep_lane" vs slowed
        # "yield_slow_down") build references at different speeds/extents, so
        # their own predicted-obstacle-distance estimates can differ by more
        # than this margin purely from that shape difference, not real
        # obstacle motion -- flipping which discrete risk bucket (and thus
        # which candidate) wins every other tick and reading to MPC as a
        # discontinuous reference. Keyed by candidate name so each logical
        # candidate slot keeps its own hysteresis state across ticks.
        self.candidate_risk_hysteresis_margin_m = max(
            0.0,
            float(self.config.get("candidate_risk_hysteresis_margin_m", 1.5)),
        )
        self.candidate_mpc_probe_enabled = bool(
            self.config.get("candidate_mpc_probe_enabled", True)
        )
        self.candidate_mpc_probe_top_k = max(
            2,
            int(self.config.get("candidate_mpc_probe_top_k", 2)),
        )
        self.candidate_mpc_probe_interval_s = max(
            0.05,
            float(self.config.get("candidate_mpc_probe_interval_s", 0.2)),
        )
        self._candidate_trajectory_evaluator = CandidateTrajectoryEvaluator(
            mpc_probe_enabled=bool(self.candidate_mpc_probe_enabled),
            mpc_probe_top_k=int(self.candidate_mpc_probe_top_k),
            mpc_probe_interval_s=float(self.candidate_mpc_probe_interval_s),
        )
        self.strict_decision_ownership_enabled = bool(
            self.config.get("strict_decision_ownership_enabled", True)
        )
        self.strict_reference_validator_veto_enabled = bool(
            self.config.get("strict_reference_validator_veto_enabled", True)
        )
        self.strict_explicit_fallback_speed_mps = max(
            0.0,
            float(self.config.get("strict_explicit_fallback_speed_mps", 0.8)),
        )
        self.full_reference_stabilizer_min_forward_m = float(
            self.config.get("full_reference_stabilizer_min_forward_m", -0.25)
        )
        self.full_reference_stabilizer_min_spacing_m = max(
            0.0,
            float(self.config.get("full_reference_stabilizer_min_spacing_m", 0.35)),
        )
        self.full_reference_stabilizer_max_heading_step_rad = max(
            0.0,
            float(self.config.get("full_reference_stabilizer_max_heading_step_rad", 0.75)),
        )
        self._debug_writer = None
        self._debug_csv_file = None
        self._debug_jsonl_file = None
        self._debug_fieldnames = [
            "sim_time_s",
            "vehicle_id",
            "x_m",
            "y_m",
            "yaw_deg",
            "speed_mps",
            "measured_accel_mps2",
            "mpc_jerk_seed_accel_mps2",
            "target_speed_mps",
            "speed_plan_target_mps",
            "speed_plan_front_gap_m",
            "speed_plan_desired_follow_gap_m",
            "speed_plan_continuous_following_active",
            "speed_plan_idm_acceleration_mps2",
            "speed_plan_reason",
            "front_gap_actor_id",
            "front_gap_obstacle_speed_mps",
            "front_gap_obstacle_lane_id",
            "front_gap_obstacle_is_source_lane",
            "speed_owner_requested_mps",
            "speed_owner_scenario_cap_mps",
            "speed_owner_turn_cap_mps",
            "speed_owner_lane_change_cap_mps",
            "speed_owner_following_cap_mps",
            "speed_owner_turn_approach_cap_mps",
            "speed_owner_upcoming_turn_distance_m",
            "speed_owner_selected_target_mps",
            "speed_owner_limiting_owner",
            "speed_owner_active_constraints",
            "speed_owner_mpc_entry_target_mps",
            "speed_owner_post_plan_delta_mps",
            "speed_owner_target_overridden_after_plan",
            "speed_owner_proposed_post_plan_target_mps",
            "speed_owner_ceiling_applied",
            "speed_owner_ceiling_reduction_mps",
            "behavior_decision",
            "static_obstacle_stop_active_input",
            "static_obstacle_replan_status",
            "static_obstacle_replan_reason",
            "static_obstacle_candidate_id",
            "static_obstacle_blocked_lane_id",
            "static_obstacle_route_transition_pending",
            "behavior_fsm_state",
            "current_lane_id",
            "behavior_target_lane_id",
            "map_match_valid",
            "map_match_ad_lane_id",
            "map_match_road_id",
            "map_match_section_id",
            "map_match_raw_lane_id",
            "map_match_center_x_m",
            "map_match_center_y_m",
            "map_match_lane_width_m",
            "map_match_lateral_offset_m",
            "map_match_heading_error_rad",
            "map_match_score",
            "map_match_confidence",
            "map_match_reason",
            "map_match_candidate_count",
            "local_lane_frame_cache_reused",
            "local_lane_frame_generation_reason",
            "local_lane_frame_ego_ad_lane_id",
            "local_lane_frame_forward_distance_m",
            "local_lane_frame_backward_distance_m",
            "local_lane_frame_corridors",
            "local_lane_frame_lane_to_offset",
            "local_lane_frame_route_target_ad_lane_id",
            "local_lane_frame_target_in_frame",
            "local_lane_frame_target_offset",
            "local_lane_frame_invariant_violations",
            "architecture_map_owner",
            "local_map_frame_id",
            "local_map_timestamp_s",
            "local_map_valid",
            "local_map_ego_lane_id",
            "local_map_route_target_lane_id",
            "local_map_route_target_offset",
            "local_map_route_target_in_frame",
            "local_map_cache_reused",
            "local_map_invariant_violations",
            "local_map_centerline_lane_count",
            "local_map_admap_border_lane_count",
            "local_map_route_lane_sequence",
            "persistent_reference_active",
            "persistent_reference_state",
            "persistent_reference_route_revision",
            "persistent_reference_map_epoch",
            "persistent_reference_geometry_revision",
            "persistent_reference_progress_s_m",
            "persistent_reference_build_reason",
            "persistent_reference_last_rejected_trigger",
            "stop_goal_active",
            "normal_stop_requested",
            "emergency_brake_requested",
            "emergency_brake_control_active",
            "normal_stop_mpc_suspended",
            "normal_stop_mpc_suspend_speed_mps",
            "normal_stop_mpc_suspend_brake",
            "front_gap_m",
            "object_count",
            "mpc_object_count",
            "cp_provider_source",
            "native_opencda_available",
            "cp_obstacle_count",
            "cp_control_count",
            "v2x_nearby_count",
            "cp_observer_cav_count",
            "cp_observer_cav_ids",
            "cp_multi_observer_obstacle_count",
            "cp_blind_spot_shared_count",
            "cp_blind_spot_shared_actor_ids",
            "cp_actor_provenance",
            "cp_pedestrian_count",
            "cp_blind_spot_pedestrian_count",
            "cp_prediction_used_actor_ids",
            "cp_prediction_used_pedestrian_ids",
            "cp_candidate_relevant_actor_ids",
            "cp_candidate_relevant_pedestrian_ids",
            "cp_actor_evidence",
            "cp_visibility_filter_enabled",
            "cp_visibility_backend",
            "reference_source",
            "final_reference_geometry_source",
            "reference_pipeline_stage",
            "reference_pipeline_intent",
            "reference_pipeline_fallback",
            "lane_follow_turn_geometry_hold_reason",
            "destination_x",
            "destination_y",
            "destination_forward_m",
            "destination_lateral_m",
            "destination_lane_id",
            "reference_first_forward_m",
            "reference_first_lateral_m",
            "preturn_lane_reference_reason",
            "preturn_raw_first_lateral_m",
            "mpc_trajectory_point_count",
            "global_route_point_count",
            "route_reference_allowed",
            "route_reference_gate_reason",
            "route_lane_change_allowed",
            "opportunistic_lane_change_allowed",
            "lane_change_gate_reason",
            "route_lane_change_required",
            "lane_change_authorized",
            "lane_change_authorization_direction",
            "lane_change_authorization_reason",
            "route_lane_change_edge_id",
            "completed_route_lane_change_edge_id",
            "route_lane_change_edge_completed",
            "lane_change_required_by_route",
            "lane_change_distance_to_maneuver_m",
            "lane_change_authorized_target_lane_id",
            "route_maneuver_normalized",
            "behavior_override_reason",
            "reference_follow_global_route_lane",
            "route_current_road_option",
            "route_next_macro_maneuver",
            "mpc_status",
            "mpc_feasibility_checked",
            "mpc_feasibility_status",
            "mpc_feasibility_reason",
            "mpc_solve_time_ms",
            "mpc_cost_profile",
            "requested_mpc_cost_profile",
            "mpc_cost_profile_switch_reason",
            "mpc_fallback_reason",
            "control_guard_reason",
            "safety_supervisor_reason",
            "accel_cmd_mps2",
            "steer_cmd_rad",
            "pre_supervisor_accel_cmd_mps2",
            "pre_supervisor_steer_cmd_rad",
            "post_supervisor_accel_cmd_mps2",
            "post_supervisor_steer_cmd_rad",
            "applied_throttle",
            "applied_brake",
            "applied_steer",
            "control_interface",
            "platform_target_speed_mps",
            "platform_target_steer_rad",
            "platform_adapter_steer_rad",
            "platform_adapter_throttle",
            "platform_adapter_brake",
            "platform_adapter_steer",
            "platform_applied_steer_rad",
            "platform_actual_speed_mps",
            "platform_adapter_reason",
            "mpc_velocity_preview_time_s",
            "mpc_velocity_source_index",
            "mpc_optimized_velocity_mps",
            "nominal_speed_ref_mps",
            "pid_target_velocity_mps",
            "velocity_command_source",
            "velocity_command_valid",
            "planner_input_cp_traffic_control_count",
            "planner_input_prediction_risky_lane_count",
            "planner_input_perception_planning_count",
            "perception_mode",
            "perception_ml_active",
            "perception_camera_count",
            "planner_input_cp_obstacle_count",
            "planner_input_frame_timestamp_s",
            "cp_message_timestamp_s",
            "cp_message_age_s",
            "cp_message_valid",
            "planner_requested",
            "planner_executed",
            "fallback_active",
            "prediction_mode",
            "cav_conflict_summary",
            "cav_corridor_binding",
            "cav_intent_count",
            "cav_shared_plan_count",
            "cav_shared_plan_sample_count",
            "cav_conflict_agent_count",
            "cav_non_ignore_count",
            "cav_deduplicated_agent_count",
            "cav_longitudinal_qp_row_count",
            "cav_homotopy_qp_row_count",
            "cav_total_qp_row_count",
            "cav_multimodal_agent_count",
            "cav_credible_mode_veto_count",
            "cav_trajectory_sources",
            "cav_prediction_validation_actor_id",
            "cav_prediction_validation_horizon_s",
            "cav_prediction_validation_x_m",
            "cav_prediction_validation_y_m",
            "local_object_count",
            "traffic_signal_state",
            "traffic_signal_raw_state",
            "traffic_signal_resolved_state",
            "traffic_signal_filtered_state",
            "traffic_signal_behavior_state",
            "traffic_control_from_cp",
            "candidate_evaluation_summary",
            "candidate_selected_decision",
            "candidate_selected_lane_id",
            "candidate_selected_cost",
            "candidate_pipeline_enabled",
            "candidate_pipeline_selected",
            "candidate_pipeline_selected_status",
            "candidate_pipeline_selected_reason",
            "candidate_selected_stop_goal_active",
            "candidate_pipeline_count",
            "candidate_prediction_trajectory_count",
            "candidate_pipeline_summary",
            "candidate_mpc_probe_summary",
            "candidate_selected_trajectory_variant",
            "candidate_selected_lane_change_duration_s",
            "candidate_selected_lane_change_duration_comfort_reason",
            "candidate_selected_lane_change_planning_average_speed_mps",
            "candidate_selected_lane_change_authorization_source",
            "candidate_selected_lane_change_initial_progress",
            "candidate_selected_lane_change_terminal_progress",
            "candidate_selection_status",
            "candidate_selection_reason",
            "maneuver_commitment_state",
            "maneuver_commitment_decision",
            "maneuver_commitment_source_lane_id",
            "maneuver_commitment_target_lane_id",
            "maneuver_commitment_progress",
            "maneuver_commitment_reference_locked",
            "maneuver_commitment_active",
            "lane_change_commitment_release_reason",
            "lane_change_phase",
            "lane_change_stabilization_frames",
            "lane_change_completion_reason",
            "lane_change_completion_stable_frames",
            "lane_change_completion_lateral_error_m",
            "lane_change_completion_heading_error_deg",
            "lane_change_contract_min_progress",
            "lane_change_contract_max_lateral_error_m",
            "lane_change_contract_max_heading_error_deg",
            "lane_change_contract_required_stable_frames",
            "lane_change_stabilization_entry_lateral_error_m",
            "lane_change_stabilization_geometry_ready",
            "behavior_lane_lateral_error_m",
            "behavior_lane_heading_error_deg",
            "behavior_lane_alignment_valid",
            "behavior_lane_change_completion_allowed",
            "lane_change_completion_target_lane_matches",
            "lane_change_completion_footprint_clearance_m",
            "route_tracking_lane_change_locked",
            "route_tracking_lane_change_progress_index",
            "route_tracking_lane_change_source_lane_id",
            "route_tracking_lane_change_target_lane_id",
            "route_tracking_recovery_active",
            "route_tracking_recovery_reason",
            "mpc_feedback_summary",
            "mpc_feedback_record_reason",
            "mpc_feedback_blocked_lane_ids",
            "mode_transition_guard_reason",
            "control_buffer_reason",
            "control_buffered_step_count",
            "mpc_replan_executed",
            "route_manager_status",
            "route_replan_attempted",
            "route_replan_succeeded",
            "route_replan_attempt_count",
            "route_replan_reason",
            "route_remaining_distance_m",
            "route_reached_destination",
            "destination_stop_latched",
            "destination_stop_reason",
            "destination_stop_remaining_distance_m",
            "destination_stop_required_distance_m",
            "global_planner_backend",
            "global_planner_backend_warning",
            "tracker_active_count",
            "tracker_stale_count",
            "prediction_validity_reason",
            "object_memory_reason",
            "traffic_memory_reason",
            "decision_scenario_state",
            "decision_behavior",
            "decision_behavior_fsm",
            "decision_candidate",
            "decision_reference_source",
            "decision_reference_stage",
            "decision_mpc_status",
            "decision_final_action",
            "decision_control_source",
            "decision_veto_count",
            "decision_veto_chain",
            "decision_veto_chain_text",
            "decision_owner_summary",
            "architecture_profile",
            "architecture_behavior_owner",
            "architecture_speed_owner",
            "architecture_reference_owner",
            "architecture_control_memory_owner",
            "architecture_safety_owner",
            "architecture_normalized_overrides",
            "scenario_fsm_state",
            "scenario_fsm_reason",
            "scenario_behavior_signal_state",
            "scenario_behavior_override_decision",
            "scenario_speed_cap_mps",
            "scenario_stop_goal_active",
            "scenario_turn_direction",
            "scenario_turn_latched",
            "scenario_boundary_recovery_active",
            "scenario_boundary_clearance_m",
            "scenario_boundary_lateral_offset_m",
            "scenario_boundary_heading_error_rad",
            "boundary_recovery_generation_reason",
            "boundary_recovery_conditioning_reason",
            "traffic_stop_forward_m",
            "traffic_stop_commit_distance_m",
            "traffic_stop_approach_reason",
            "speed_plan_target_mps",
            "speed_plan_cap_mps",
            "speed_plan_stop_goal_active",
            "speed_plan_reason",
            "route_turn_reference_reason",
            "route_turn_raw_first_forward_m",
            "route_turn_raw_first_lateral_m",
            "route_debug_reason",
            "route_sync_reason",
            "route_progress_index",
            "route_progress_s_m",
            "route_cursor_stalled_motion_m",
            "route_cursor_missed_maneuver",
            "route_topology_valid",
            "route_topology_signature",
            "route_topology_errors",
            "route_topology_warnings",
            "route_geometry_lane_change_direction",
            "route_geometry_lane_change_distance_m",
            "route_geometry_lane_change_reason",
            "waypoint_backend",
            "route_upcoming_turn_direction",
            "route_upcoming_turn_distance_m",
            "route_upcoming_turn_reason",
            "turn_latch_reason",
            "opencda_style_reference_conditioning_reason",
            "reference_lateral_guard_reason",
            "mpc_reference_stabilizer_reason",
            "final_reference_gate_valid",
            "final_reference_gate_mode",
            "final_reference_gate_reason",
            "reference_max_curvature_1pm",
            "reference_contract_max_curvature_1pm",
            "reference_curvature_margin_1pm",
            "reference_pipeline_conditioning_reason",
            "post_turn_exit_reference_source",
            "reference_pipeline_mode",
            "mpc_entry_allowed",
            "mpc_entry_status",
            "mpc_entry_reason",
            "pipeline_error",
            "pipeline_error_traceback",
            "stop_target_forward_m",
            "stop_approach_speed_mps",
            "green_release_reference_active",
            "lane_safety_scores",
            "evaluation_metrics_available",
            "collision_sensor_available",
            "collision_sensor_error",
            "collision_event_this_frame",
            "collision_count",
            "collision_rate_per_km",
            "last_collision_actor_type",
            "last_collision_impulse",
            "nearest_ttc_s",
            "min_ttc_s",
            "nearest_ttc_obstacle_id",
            "nearest_ttc_reason",
            "nearest_ttc_longitudinal_gap_m",
            "nearest_ttc_lateral_gap_m",
            "nearest_ttc_bumper_gap_m",
            "nearest_ttc_closing_speed_mps",
            "tick_max_drac_mps2",
            "max_drac_mps2",
            "min_pet_s",
            "distance_traveled_m",
            "road_boundary_sample_valid",
            "road_boundary_lateral_offset_m",
            "road_boundary_lane_width_m",
            "road_boundary_ego_half_width_m",
            "road_boundary_heading_error_rad",
            "road_boundary_clearance_m",
            "road_boundary_breach",
            "road_boundary_projection_segment_index",
            "road_boundary_projection_segment_ratio",
            "road_boundary_projection_raw_heading_rad",
            "road_boundary_projection_conditioned_heading_rad",
            "road_boundary_projection_continuity_limited",
            "road_boundary_projection_reason",
            "road_boundary_geometry_source",
            "road_boundary_drivable_inside",
            "road_boundary_breach_count",
            "road_boundary_sample_count",
            "road_boundary_breach_rate",
            "Cost_RoadBoundary",
            "Cost_Repulsive",
            "Cost_Repulsive_Safe",
            "Cost_Repulsive_Collision",
            "Cost_Repulsive_LogBarrier",
            "Cost_ref",
            "Cost_LaneCenter",
            "Cost_Control",
            "Cost_VelocitySlack",
            "prediction_lane_step_resolved_count",
            "prediction_lane_step_none_count",
            "turn_boundary_recovery_active",
            "turn_boundary_recovery_phase",
        ]

        self._ensure_planning_module_import_path()
        from opencda.planning_module.MPC.mpc import MPC
        from opencda.planning_module.behavior_planner import LaneSafetyScorer, RuleBasedBehaviorPlanner
        from opencda.planning_module.opencda_bridge.cp_provider import OpenCDACPProvider
        from opencda.planning_module.opencda_bridge.planner_input_adapter import OpenCDAPlanningAdapter
        from opencda.planning_module.pipeline.control_buffer import MPCControlBuffer
        from opencda.planning_module.pipeline.mpc_feedback import BehaviorMPCFeedback
        from opencda.planning_module.pipeline.mpc_command_extractor import (
            MPCCommandExtractor,
        )
        from opencda.planning_module.pipeline.route_manager import CPXRouteManager
        from opencda.planning_module.pipeline.reference_gate import FinalReferenceGate
        from opencda.planning_module.pipeline.reference_generator import ReferenceGenerator
        from opencda.planning_module.pipeline.reference_pipeline import (
            ReferencePipeline,
        )
        from opencda.planning_module.pipeline.safety_supervisor import SafetySupervisor
        from opencda.planning_module.pipeline.runtime_input_stage import RuntimeInputStage
        from opencda.planning_module.pipeline.velocity_steering_adapter import (
            OpenCDAVelocitySteeringAdapter,
        )
        from opencda.planning_module.pipeline.tracker import CPXObstacleTracker
        from opencda.planning_module.utility.evaluation_metrics import (
            EvaluationMetricsRecorder,
            write_planning_metrics_artifacts,
        )
        from opencda.planning_module.utility.global_planner import CustomGlobalPlannerAdapter

        mpc_cfg, road_cfg = self._load_mpc_config()
        self.mpc = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)
        if self._cav_conflict_enabled:
            # One switch: cav_conflict_enabled also binds the Stage-C corridor
            # in the QP (else it is computed but ignored). mpc.yaml's
            # cost.corridor block still tunes w_slack / max_slack_m.
            self.mpc.corridor_constraint_enabled = True
        from opencda.planning_module.pipeline.actuator_mapper import CarlaActuatorMapper
        from opencda.planning_module.opencda_bridge.platform_ports import (
            ActuatorPort,
            MapLookupPort,
        )
        self.actuator_mapper = CarlaActuatorMapper(self.config)
        self.actuator_port = ActuatorPort(
            actuator_mapper=self.actuator_mapper,
            constraints=self.mpc.constraints,
            carla_module=carla,
            clock=self._sim_time_s,
        )
        runtime_input_stage = RuntimeInputStage(self.actuator_mapper)
        vehicle_curvature_margin = min(
            1.0,
            max(
                0.1,
                float(
                    self.config.get(
                        "reference_vehicle_curvature_safety_factor",
                        0.90,
                    )
                ),
            ),
        )
        vehicle_max_curvature_1pm = (
            math.tan(float(self.mpc.constraints.max_steer_rad))
            / max(1.0e-6, float(self.mpc.wheelbase_m))
        )
        self.config["reference_vehicle_max_curvature_1pm"] = (
            float(vehicle_curvature_margin)
            * float(vehicle_max_curvature_1pm)
        )
        self.reference_generator = ReferenceGenerator(
            config=self.config,
            mpc=self.mpc,
            map_planner=self.map_planner,
            map_waypoint_from_location=self._map_waypoint_from_location,
            lane_id_at_location=self._lane_id_at_location,
            body_frame_xy=self._body_frame_xy,
            target_speed_mps=float(self.target_speed_mps),
            lookahead_m=float(self.lookahead_m),
            drivable_waypoint_from_location=(
                self._drivable_waypoint_from_location
            ),
        )
        self.behavior_runtime_cfg = dict(mpc_cfg.get("behavior_planner_runtime", {}))
        self.lane_safety_scorer = LaneSafetyScorer()
        self.reference_map = _WaypointMapAdapter(self.map_planner)
        self.map_lookup_port = MapLookupPort(
            reference_map=self.reference_map,
            waypoint_map=self.waypoint_map_planner,
            carla_module=carla,
        )
        self.input_adapter = OpenCDAPlanningAdapter(self)
        self.tracker = CPXObstacleTracker(
            max_stale_s=float(self.config.get("tracker_max_stale_s", 0.5)),
            max_speed_mps=float(self.config.get("tracker_max_speed_mps", 45.0)),
            max_acceleration_mps2=float(
                self.config.get("tracker_max_acceleration_mps2", 12.0)
            ),
            max_position_jump_m=float(self.config.get("tracker_max_position_jump_m", 12.0)),
        )
        # Prediction-knowledge ablation (cv | blind | oracle). Built lazily on
        # first use so mpc.horizon_s / dt_s are settled; see
        # ``prediction_snapshot_transform``.
        self._prediction_mode = str(
            self.config.get("prediction_mode", "cv")
        ).strip().lower()
        self._prediction_snapshot_transform_cached = False
        self._prediction_snapshot_transform_fn = None
        self._oracle_trace_store = None
        if self._prediction_mode == "oracle":
            from opencda.planning_module.pipeline.prediction_ablation import (
                OracleTraceStore,
            )
            oracle_path = str(self.config.get("oracle_trace_path", "") or "")
            if not oracle_path:
                raise ValueError(
                    "prediction_mode='oracle' requires config 'oracle_trace_path'"
                )
            if not os.path.isabs(oracle_path):
                # OpenCDA runs from the repo root; resolve against it so the
                # path matches what the scenario TraceRecorder wrote.
                repo_root = Path(__file__).resolve().parents[3]
                cand = repo_root / oracle_path
                oracle_path = str(cand if cand.is_file() else oracle_path)
            if not os.path.isfile(oracle_path):
                raise FileNotFoundError(
                    "oracle_trace_path not found: %s -- run the recording "
                    "scenario first (prediction_mode: cv, "
                    "prediction_ablation.record: true)" % oracle_path
                )
            self._oracle_trace_store = OracleTraceStore.from_file(oracle_path)
        ego_extent = getattr(
            getattr(self.vehicle_manager.vehicle, "bounding_box", None),
            "extent", None,
        )
        perception_stage = PerceptionStage(
            max_mpc_obstacles=int(self.max_mpc_obstacles),
            ego_length_m=2.0 * float(getattr(ego_extent, "x", 2.25)),
            ego_width_m=2.0 * float(getattr(ego_extent, "y", 1.0)),
            lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
            lane_change_boundary_overlap_m=float(
                self.config.get("lane_change_boundary_overlap_m", 0.75)
            ),
        )
        self.final_reference_gate = FinalReferenceGate(self.config)
        self.reference_pipeline = ReferencePipeline(
            config=self.config,
            generator=self.reference_generator,
            final_gate=self.final_reference_gate,
            horizon_steps=int(self.mpc.horizon_steps),
            dt_s=float(self.mpc.dt_s),
            default_speed_mps=float(self.target_speed_mps),
        )
        reference_publication_stage = ReferencePublicationStage(
            reference_pipeline=self.reference_pipeline,
            reference_provider=self._stable_reference_line_provider,
            config=self.config,
        )
        mpc_entry_stage = MPCEntryStage(self.config)
        static_obstacle_stage = StaticObstacleStage({
            **self.behavior_runtime_cfg,
            **self.config,
        })
        self.safety_supervisor = SafetySupervisor(
            enabled=bool(self.config.get("safety_supervisor_enabled", True)),
            max_steer_delta=float(self.config.get("safety_max_steer_delta", 0.25)),
            max_throttle_delta=float(self.config.get("safety_max_throttle_delta", 0.45)),
            max_brake_delta=float(self.config.get("safety_max_brake_delta", 0.60)),
            stuck_release_min_accel_mps2=float(
                self.config.get("safety_stuck_release_min_accel_mps2", 0.01)
            ),
        )
        control_safety_stage = ControlSafetyStage(
            supervisor=self.safety_supervisor, config=self.config
        )
        candidate_selection_stage = CandidateSelectionStage(
            evaluator=self._candidate_trajectory_evaluator,
            provider=self._stable_reference_line_provider,
            maneuver_manager=self.maneuver_manager,
            reference_pipeline=self.reference_pipeline,
            fallback_manager=fallback_manager,
            static_obstacle_stage=static_obstacle_stage,
            mpc=self.mpc,
            config=self.config,
            map_epoch="admap",
            normal_clearance_m=self.full_candidate_reference_min_object_distance_m,
            static_clearance_m=self.static_obstacle_local_avoidance_min_object_distance_m,
            risk_hysteresis_margin_m=self.candidate_risk_hysteresis_margin_m,
            strict_ownership=self.strict_decision_ownership_enabled,
            target_speed_mps=self.target_speed_mps,
        )
        self.pipeline = PlanningPipeline(
            runtime_input=runtime_input_stage,
            perception=perception_stage,
            behavior=behavior_stage,
            scenario=scenario_manager,
            static_obstacle=static_obstacle_stage,
            control_safety=control_safety_stage,
            speed=speed_target_planner,
            destination_speed=destination_speed_stage,
            reference_publication=reference_publication_stage,
            mpc_entry=mpc_entry_stage,
            fallback=fallback_manager,
            behavior_reference_execution=behavior_reference_execution_stage,
            candidate_selection=candidate_selection_stage,
        )
        self.velocity_steering_adapter = OpenCDAVelocitySteeringAdapter(
            self.vehicle_manager.controller
        )
        self.mpc_command_extractor = MPCCommandExtractor(
            preview_time_s=float(
                self.config.get("mpc_velocity_command_preview_time_s", 0.6)
            ),
            velocity_source=str(
                self.config.get("mpc_velocity_command_source", "preview")
            ),
            min_acceleration_mps2=float(
                self.mpc.constraints.min_acceleration_mps2
            ),
            max_acceleration_mps2=float(
                self.mpc.constraints.max_acceleration_mps2
            ),
        )
        route_sample_distance_m = float(self.config.get("route_sample_distance_m", 1.0))
        self.global_planner_backend = "custom_admap_dijkstra"
        self.global_planner_backend_warning = ""
        road_cfg_from_map = {"lane_count": 1, "lane_width_m": 3.5}
        global_planner_mode = str(
            self.config.get("global_planner_mode", "dij")
        ).strip().lower()
        if global_planner_mode not in {
            "custom", "custom_admap", "admap", "opendrive", "dijkstra", "dij"
        }:
            raise ValueError("global_planner_mode must select the AD-map backend")
        xodr_path = self._resolve_global_planner_xodr_path()
        self.global_planner = CustomGlobalPlannerAdapter(
            xodr_path=str(xodr_path),
            cache_root=str(
                self.config.get(
                    "global_planner_cache_root",
                    Path(__file__).resolve().parents[1] / "Global_Planner" / "cache",
                )
            ),
            route_sample_distance_m=float(route_sample_distance_m),
            lane_change_penalty_m=(
                float(self.config["global_planner_lane_change_penalty_m"])
                if self.config.get("global_planner_lane_change_penalty_m") is not None
                else None
            ),
            ad_map_install_root=self.config.get("ad_map_install_root"),
        )
        self.global_planner.load(
            force_rebuild=bool(self.config.get("global_planner_force_rebuild", False))
        )
        print(
            "[CP-X OpenCDA Bridge] Using AD-map Dijkstra global planner: "
            f"{xodr_path}"
        )
        self.topology_map = self.global_planner
        # CARLA remains available to the simulation adapter, but every route,
        # lane and reference waypoint query is owned by AD-map.
        self.waypoint_map_planner = self.topology_map
        self.reference_map = _WaypointMapAdapter(self.waypoint_map_planner)
        self.map_lookup_port = MapLookupPort(
            reference_map=self.reference_map,
            waypoint_map=self.waypoint_map_planner,
            carla_module=carla,
        )
        self.reference_generator.map_planner = self.waypoint_map_planner
        self.waypoint_backend = "admap"
        self.route_manager = CPXRouteManager(
            global_planner=self.global_planner,
            route_sampling_resolution_m=float(
                self.config.get("route_sampling_resolution_m", 1.0)
            ),
            route_reference_smoothing_passes=int(
                self.config.get("route_reference_smoothing_passes", 3)
            ),
            route_turn_connector_smoothing_passes=int(
                self.config.get("route_turn_connector_smoothing_passes", 16)
            ),
            route_reference_boundary_aware=bool(
                self.config.get(
                    "route_reference_boundary_aware",
                    True,
                )
            ),
            route_reference_vehicle_half_width_m=float(
                self.config.get("reference_vehicle_half_width_m", 1.0)
            ),
            route_reference_boundary_margin_m=float(
                self.config.get(
                    "reference_contract_turn_boundary_margin_m",
                    0.15,
                )
            ),
            route_reference_tracking_reserve_m=float(
                self.config.get(
                    "route_reference_tracking_reserve_m",
                    0.20,
                )
            ),
            route_rejoin_min_lateral_m=float(
                self.config.get("route_rejoin_min_lateral_m", 0.35)
            ),
            route_rejoin_max_lateral_m=float(
                self.config.get("route_rejoin_max_lateral_m", 3.0)
            ),
            route_rejoin_distance_m=float(
                self.config.get("route_rejoin_distance_m", 8.0)
            ),
            reached_distance_m=float(self.config.get("route_reached_distance_m", 3.0)),
            stale_route_lateral_m=float(self.config.get("route_stale_lateral_m", 12.0)),
            turn_replan_max_length_ratio=float(
                self.config.get("turn_replan_max_length_ratio", 2.5)
            ),
            turn_replan_max_added_length_m=float(
                self.config.get("turn_replan_max_added_length_m", 100.0)
            ),
        )
        self.road_cfg_from_map = dict(road_cfg_from_map or {})
        self._active_route_summary = None
        self.mpc_feedback = BehaviorMPCFeedback(
            enabled=bool(self.config.get("mpc_feedback_enabled", True)),
            hold_s=float(self.config.get("mpc_feedback_hold_s", 1.5)),
            min_failures=int(self.config.get("mpc_feedback_min_failures", 1)),
        )
        from opencda.planning_module.pipeline.control_finalization_stage import (
            ControlFinalizationStage,
        )
        self.pipeline.control_finalization = ControlFinalizationStage(
            mpc=self.mpc,
            command_extractor=self.mpc_command_extractor,
            feedback=self.mpc_feedback,
            control_safety=control_safety_stage,
            config=self.config,
            hard_gate_requires_emergency_stop=_hard_gate_requires_emergency_stop,
        )
        self.control_buffer = MPCControlBuffer(
            enabled=bool(self.config.get("control_buffer_enabled", True)),
            replan_period_s=float(
                self.config.get(
                    "mpc_replan_period_s",
                    getattr(self.mpc, "trajectory_generation_period_s", 0.25),
                )
            ),
            max_reuse_s=float(self.config.get("control_buffer_max_reuse_s", 0.35)),
            max_reference_anchor_jump_m=float(
                self.config.get(
                    "control_buffer_max_reference_anchor_jump_m",
                    0.75,
                )
            ),
            max_predicted_speed_error_mps=float(
                self.config.get(
                    "control_buffer_max_predicted_speed_error_mps",
                    0.75,
                )
            ),
            max_target_speed_jump_mps=float(
                self.config.get(
                    "control_buffer_max_target_speed_jump_mps",
                    1.0,
                )
            ),
        )
        from opencda.planning_module.pipeline.lane_change_lifecycle_stage import (
            LaneChangeLifecycleStage,
        )
        self.lane_change_lifecycle_stage = LaneChangeLifecycleStage(
            provider=self._stable_reference_line_provider,
            maneuver_manager=self.maneuver_manager,
            reference_generator=self.reference_generator,
            route_manager=self.route_manager,
            mpc=self.mpc,
            control_buffer=self.control_buffer,
            config=self.config,
            target_speed_mps=self.target_speed_mps,
            map_epoch="admap",
            vehicle_extent=lambda: getattr(
                getattr(self.vehicle_manager.vehicle, "bounding_box", None),
                "extent",
                None,
            ),
        )
        candidate_selection_stage.set_lane_change_lifecycle(
            self.lane_change_lifecycle_stage
        )
        self.pipeline.mpc_execution = MPCExecutionStage(
            mpc=self.mpc,
            control_buffer=self.control_buffer,
            minimum_replan_speed_mps=float(self.full_control_buffer_min_speed_mps),
        )
        self.cp_message_path = str(
            self.config.get(
                "cp_message_path",
                Path(__file__).resolve().parents[1] / "behavior_planner" / "cp_message.json",
            )
        )
        self.behavior_planner = RuleBasedBehaviorPlanner(
            cp_message_path=str(self.cp_message_path),
            cooperative_message_check_frequency_hz=float(
                self.config.get("cooperative_message_check_frequency_hz", 5.0)
            ),
        )
        self.cp_provider = None
        cp_message_path = self.config.get("cp_message_path")
        if not cp_message_path:
            cp_message_path = self.cp_message_path
        if bool(self.config.get("publish_cp_message", True)):
            self.cp_provider = OpenCDACPProvider(
                message_path=str(cp_message_path),
                schema_version=1,
                communication_range_m=float(self.config.get("communication_range_m", 80.0)),
                prediction_horizon_s=float(self.mpc.horizon_s),
                prediction_dt_s=float(self.mpc.dt_s),
                source="native_opencda",
                require_native_opencda=bool(
                    self.config.get("require_native_opencda_cp", True)
                ),
                visibility_filter_enabled=bool(
                    self.config.get("cp_visibility_filter_enabled", False)
                ),
                visibility_backend=str(
                    self.config.get(
                        "cp_visibility_backend",
                        "actor_geometry",
                    )
                ),
                visibility_sensor_height_m=float(
                    self.config.get("cp_visibility_sensor_height_m", 1.6)
                ),
                visibility_target_tolerance_m=float(
                    self.config.get("cp_visibility_target_tolerance_m", 0.75)
                ),
            )
        self.active_mpc_cost_profile = "lane_follow"
        self.requested_mpc_cost_profile = "lane_follow"
        self.mpc_cost_profile_active_since_s = 0.0
        self.mpc_cost_profile_switch_reason = "initial"
        self._latest_opencda_update: dict[str, Any] = {}
        self.last_output = None
        self._write_planning_metrics_artifacts = write_planning_metrics_artifacts
        self.evaluation_metrics = EvaluationMetricsRecorder(
            ego_length_m=float(self.config.get("metrics_ego_length_m", 4.5)),
            lateral_conflict_width_m=float(
                self.config.get("metrics_lateral_conflict_width_m", 2.5)
            ),
            min_ego_speed_for_ttc_mps=float(
                self.config.get("metrics_min_ego_speed_for_ttc_mps", 0.5)
            ),
            pet_conflict_radius_m=float(
                self.config.get("metrics_pet_conflict_radius_m", 3.0)
            ),
            pet_bin_size_m=float(self.config.get("metrics_pet_bin_size_m", 3.0)),
        )
        self._metrics_collision_sensor = None
        self._metrics_collision_sensor_available = False
        self._metrics_collision_sensor_error = ""
        self._metrics_last_emitted_collision_count = 0
        self._metrics_boundary_breach_count = 0
        self._metrics_boundary_sample_count = 0
        self._metrics_last_collision_actor_type = ""
        self._metrics_last_collision_impulse = ""
        self._prediction_lane_step_resolved_count = 0
        self._prediction_lane_step_none_count = 0
        self._spawn_metrics_collision_sensor()

    @staticmethod
    def _ensure_planning_module_import_path() -> None:
        """Expose planning_module-local imports used by legacy MPC modules.

        The standalone planning runner is usually launched from
        ``opencda/planning_module``, so imports like ``from utility...`` work.
        Native OpenCDA scenarios are launched from the repository root, where
        that directory is not on ``sys.path``.  Add it only when the bridge is
        constructed so the default OpenCDA path stays untouched.
        """

        planning_module_root = str(Path(__file__).resolve().parents[1])
        if planning_module_root not in sys.path:
            sys.path.insert(0, planning_module_root)

    def _resolve_global_planner_xodr_path(self) -> Path:
        raw_path = str(
            self.config.get(
                "global_planner_xodr_path",
                self.config.get("xodr_path", ""),
            )
            or ""
        ).strip()
        planning_module_root = Path(__file__).resolve().parents[1]
        if raw_path:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = planning_module_root / path
            if path.exists():
                return path
            raise FileNotFoundError(f"Global planner xodr_path not found: {path}")

        map_name = str(self.config.get("global_planner_map_name", "") or "").strip()
        if not map_name:
            try:
                map_name = str(self.map_planner.name).split("/")[-1]
            except Exception:
                map_name = ""
        if not map_name:
            map_name = "Town06"
        candidates = []
        if map_name.endswith(".xodr"):
            candidates.append(planning_module_root / "Global_Planner" / "maps" / map_name)
        else:
            candidates.extend([
                planning_module_root / "Global_Planner" / "maps" / f"{map_name}.xodr",
                planning_module_root / "Global_Planner" / "maps" / f"{map_name}_Opt.xodr",
            ])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not resolve custom global planner .xodr path; checked: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def set_destination(
        self,
        *,
        start_location: Any,
        end_location: Any,
        clean: bool = False,
        end_reset: bool = True,
    ) -> None:
        """Set the CP-X global route without using OpenCDA BehaviorAgent."""

        del clean, end_reset
        start_point = self._location_to_point(start_location)
        goal_point = self._location_to_point(end_location)
        self._active_route_summary = self.route_manager.set_destination(
            start_point=start_point,
            goal_point=goal_point,
        )
        self.nominal_trajectory_generator.reset(source="destination_updated")
        self.control_buffer.reset(reason="destination_updated")
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="destination_updated")
        self.maneuver_manager.clear_turn(reason="destination_updated")
        self._clear_turn_master_reference()
        self._stable_reference_line_provider.release(
            POST_TURN, event="phase_transition"
        )

    def set_external_global_plan(self, world_plan: Sequence[Any]) -> None:
        """Accept only mission endpoints; AD-map owns the route between them."""

        locations = []
        for entry in list(world_plan or []):
            node = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
            transform = getattr(node, "transform", node)
            location = getattr(transform, "location", None)
            if location is not None:
                locations.append(location)
        if len(locations) < 2:
            raise ValueError("Global plan must provide at least start and goal")
        self._active_route_summary = self.route_manager.set_destination(
            start_point=self._location_to_point(locations[0]),
            goal_point=self._location_to_point(locations[-1]),
        )
        self._route_replan_last_reason = "admap_route_from_mission_endpoints"
        self._active_route_summary = None
        self.nominal_trajectory_generator.reset(
            source="external_global_plan_installed"
        )
        self.control_buffer.reset(reason="external_global_plan_installed")
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="external_global_plan_installed")

    def update_information(
        self,
        *,
        ego_transform: Any,
        ego_speed_kmh: float,
        detected_objects: Any = None,
        v2x_manager: Any = None,
        safety_manager: Any = None,
        map_manager: Any = None,
    ) -> None:
        """Receive the current OpenCDA tick snapshot from VehicleManager.update_info."""

        self._latest_opencda_update = {
            "ego_transform": ego_transform,
            "ego_speed_kmh": float(ego_speed_kmh),
            "detected_objects": detected_objects,
            "v2x_manager": v2x_manager,
            "safety_manager": safety_manager,
            "map_manager": map_manager,
            "sim_time_s": float(self._sim_time_s()),
        }

    def run_step(self) -> carla.VehicleControl:
        """Plan and return a low-level CARLA control command."""

        try:
            planner_output = self.execute_planning_pipeline()
        except Exception as exc:
            if self.fallback_policy == "raise":
                raise
            if self.fallback_policy == "opencda":
                raise
            control = self._emergency_stop_control()
            self.last_debug = {
                "sim_time_s": float(self._sim_time_s()),
                "vehicle_id": int(getattr(self.vehicle_manager.vehicle, "id", -1)),
                "planner": "cpx_mpc",
                "planner_requested": True,
                "planner_executed": False,
                "fallback_active": True,
                "fallback_reason": str(exc),
                "mpc_fallback_reason": str(exc),
                "control_guard_reason": "fallback_policy_emergency_stop",
                "accel_cmd_mps2": float(getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0)),
                "steer_cmd_rad": 0.0,
            }
            self._record_debug(self.last_debug)
            return control
        self.last_output = planner_output
        self.last_debug = planner_output.diagnostics_dict()
        self._record_debug(self.last_debug)
        return planner_output.control

    def _apply_velocity_steering_interface(
        self,
        *,
        target_speed_mps: float,
        target_steering_rad: float,
        actual_speed_mps: float,
        stop_goal_active: bool,
        emergency_stop: bool,
        sim_time_s: float,
    ):
        """Map planner-owned speed/steering before final safety supervision."""

        from opencda.planning_module.pipeline.velocity_steering_adapter import (
            VelocitySteeringCommand,
        )

        control, adapter_reason = self.velocity_steering_adapter.run_step(
            command=VelocitySteeringCommand(
                target_speed_mps=float(target_speed_mps),
                target_steering_rad=float(target_steering_rad),
                emergency_stop=bool(emergency_stop),
                stop_goal_active=bool(stop_goal_active),
            ),
            actual_speed_mps=float(actual_speed_mps),
            sim_time_s=float(sim_time_s),
            max_steering_rad=float(self.mpc.constraints.max_steer_rad),
            carla_module=carla,
        )
        applied_steer_rad = (
            float(getattr(control, "steer", 0.0))
            * float(self.mpc.constraints.max_steer_rad)
        )
        debug = {
            "control_interface": "planner_velocity_steering",
            "platform_target_speed_mps": float(target_speed_mps),
            "platform_target_steer_rad": float(target_steering_rad),
            "platform_adapter_steer_rad": float(applied_steer_rad),
            "platform_actual_speed_mps": float(actual_speed_mps),
            "platform_adapter_reason": str(adapter_reason),
            "platform_adapter_throttle": float(
                getattr(control, "throttle", 0.0)
            ),
            "platform_adapter_brake": float(getattr(control, "brake", 0.0)),
            "platform_adapter_steer": float(getattr(control, "steer", 0.0)),
        }
        return (
            control,
            float(self._accel_from_control(control)),
            float(applied_steer_rad),
            debug,
        )

    def execute_planning_pipeline(self):
        """Public OpenCDA bridge port for one full CP-X planning tick."""

        return self._run_full_cpx_pipeline_step()

    def _run_full_cpx_pipeline_step(self):
        """Run OpenCDAPlanningAdapter -> PlanningPipeline -> PlannerOutput."""

        from opencda.planning_module.pipeline.output import (
            BehaviorCommand,
            PlannerDiagnostics,
            PlannerOutput,
        )
        latest_update = dict(getattr(self, "_latest_opencda_update", {}) or {})
        self._clean_functional_test_dynamic_actors_once()
        if self.cp_provider is not None:
            try:
                self.cp_provider.publish(
                    world=self.vehicle_manager.vehicle.get_world(),
                    map_planner=self.map_planner,
                    ego_vehicle=self.vehicle_manager.vehicle,
                    sim_time_s=self._sim_time_s(),
                    vehicle_manager=self.vehicle_manager,
                )
            except Exception as exc:
                if self.debug:
                    print(f"[CP-X OpenCDA Bridge] native CP publish failed: {exc}")
        cycle = self.pipeline.begin_cycle(
            timestamp_s=float(self._sim_time_s()),
            ego_transform=(
                latest_update.get("ego_transform")
                or self.vehicle_manager.localizer.get_ego_pos()
            ),
            ego_speed_kmh=float(latest_update.get(
                "ego_speed_kmh", self.vehicle_manager.localizer.get_ego_spd()
            )),
            detected_objects=(
                latest_update.get("detected_objects")
                if latest_update.get("detected_objects") is not None
                else getattr(
                    self.vehicle_manager.perception_manager, "objects", {}
                ) or {}
            ),
            cp_payload=self._load_cp_message_payload(),
            ignore_dynamic_objects=bool(
                self.functional_test_ignore_dynamic_objects
            ),
            cruise_speed_mps=float(self.target_speed_mps),
            base_emergency_gap_m=float(
                self.config.get("following_emergency_gap_m", 3.0)
            ),
            emergency_standstill_buffer_m=float(
                self.config.get("following_emergency_standstill_buffer_m", 1.0)
            ),
            following_time_headway_s=float(
                self.config.get("following_time_headway_s", 1.5)
            ),
        )
        tick = cycle.tick
        perception = cycle.perception
        sim_time_s = float(tick.timestamp_s)
        ego_transform = tick.ego_transform
        ego_location = tick.ego_location
        ego_speed_mps = float(tick.ego_speed_mps)
        ego_yaw_rad = float(tick.ego_yaw_rad)
        measured_accel_mps2 = float(tick.measured_accel_mps2)
        cp_payload = dict(perception.cp_payload)
        object_snapshots = [dict(item) for item in cycle.object_snapshots]
        mpc_object_snapshots = [dict(item) for item in cycle.mpc_object_snapshots]
        local_object_snapshots = [dict(item) for item in cycle.local_object_snapshots]
        front_gap_m = perception.front_gap_m
        emergency_front_gap_m = float(cycle.emergency_gap_m)
        stop_goal_active = bool(cycle.emergency_stop_required)
        requested_speed_mps = float(cycle.requested_speed_mps)
        current_state = list(tick.current_state)

        from opencda.planning_module.pipeline.behavior_reference_execution_stage import (
            BehaviorReferenceRequest,
        )
        behavior_reference = self.pipeline.execute_behavior_reference(
            BehaviorReferenceRequest(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                requested_speed_mps=float(requested_speed_mps),
                object_snapshots=object_snapshots,
                stop_goal_active=bool(stop_goal_active),
                cp_payload=cp_payload,
                current_state=current_state,
                sim_time_s=float(tick.timestamp_s),
                route_revision=str(self.route_manager.route_revision),
            ),
            planner=self._plan_behavior_and_reference,
        )
        destination_state = list(behavior_reference.destination_state)
        lane_center_reference = [
            dict(item) for item in behavior_reference.reference_samples
        ]
        behavior_stage_result = behavior_reference.behavior_stage_result
        behavior_decision = behavior_stage_result.decision
        behavior_debug = behavior_stage_result.mutable_diagnostics()
        reference_debug = dict(behavior_reference.reference_debug)
        typed_speed_plan = behavior_reference.speed_plan
        if behavior_reference.failure_reason and self.debug:
            print(
                "[CP-X OpenCDA Bridge] behavior/reference pipeline failed: "
                + str(behavior_reference.failure_reason)
            )

        destination_application = self.pipeline.apply_destination(
            route_status=getattr(self.route_manager, "last_status", None),
            route_revision=str(self.route_manager.route_revision),
            ego_speed_mps=float(ego_speed_mps),
            current_state=current_state,
            destination_state=destination_state,
            reference_samples=lane_center_reference,
            behavior_stage_result=behavior_stage_result,
            reference_debug=reference_debug,
            fallback_lane_id=int(getattr(self._local_map_snapshot, "ego_lane_id", 0)),
        )
        destination_stage = destination_application.stage
        destination_state = destination_application.mutable_destination_state()
        lane_center_reference = destination_application.mutable_reference()
        behavior_stage_result = destination_application.behavior_stage_result
        behavior_decision = behavior_stage_result.decision
        behavior_debug = behavior_stage_result.mutable_diagnostics()
        reference_debug = destination_application.mutable_reference_debug()
        route_reached_destination = bool(destination_stage.reached_destination)
        route_remaining_distance_m = float(destination_stage.remaining_distance_m)
        route_destination_approach = bool(destination_stage.approach_active)
        destination_stopping_distance_m = float(destination_stage.required_distance_m)
        destination_stop_buffer_m = float(destination_stage.stop_buffer_m)
        destination_speed_constraint = destination_stage.constraint
        if destination_application.finished:
            setattr(self.vehicle_manager, "_opencda_agent_finished", True)
        behavior_debug.update(behavior_decision.as_debug_fields())

        # The typed behavior decision owns the final stop state. The raw
        # front-gap threshold is only an input proposal and must not re-latch
        # stop after candidate evaluation has selected a safe route maneuver.
        selected_stop_goal_active = bool(behavior_decision.stop_required)
        speed_frame = self.pipeline.resolve_speed(
            behavior=behavior_decision,
            speed_plan=typed_speed_plan,
            additional_constraints=(
                ()
                if destination_speed_constraint is None
                else (destination_speed_constraint,)
            ),
            destination_state=destination_state,
            reference_samples=lane_center_reference,
        )
        speed_target = speed_frame.target
        ceiling_result = speed_frame.ceiling
        speed_ref_mps = float(speed_target.target_mps)
        destination_state = list(ceiling_result.destination_state)
        lane_center_reference = list(ceiling_result.reference_samples)
        reference_debug.update(speed_frame.trace_fields())
        mpc_stop_goal_active = bool(selected_stop_goal_active)
        behavior_decision_normalized = str(behavior_decision.maneuver).strip().lower()
        normal_stop_requested = behavior_decision_normalized in {
            "stop_at_intersection",
            "stop_sign",
        }
        emergency_brake_requested = (
            behavior_decision_normalized == "emergency_brake"
        )
        if bool(mpc_stop_goal_active) and len(destination_state) >= 3:
            destination_state = list(destination_state)
            destination_state[2] = 0.0
        stop_target_forward_m_debug = ""
        stop_target_debug = behavior_decision.stop_target
        if bool(mpc_stop_goal_active) and isinstance(stop_target_debug, Mapping):
            try:
                stop_target_forward_m_debug, _ = self._body_frame_xy(
                    origin_x_m=float(ego_location.x),
                    origin_y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    target_x_m=float(stop_target_debug.get("x_m", stop_target_debug.get("x", ego_location.x))),
                    target_y_m=float(stop_target_debug.get("y_m", stop_target_debug.get("y", ego_location.y))),
                )
            except Exception:
                stop_target_forward_m_debug = ""

        mode_transition_guard_reason = self._apply_behavior_mode_transition_guard(
            decision=str(behavior_decision.maneuver),
            lc_state=str(behavior_decision.phase),
            target_lane_id=int(behavior_decision.target_lane_id),
            stop_goal_active=bool(mpc_stop_goal_active),
        )
        candidate_status = str(reference_debug.get(
            "candidate_pipeline_selected_status", ""
        ))
        candidate_name = str(reference_debug.get(
            "candidate_pipeline_selected", ""
        ))
        candidate_reason = str(reference_debug.get(
            "candidate_pipeline_selected_reason", ""
        ))
        admission = self.pipeline.prepare_trajectory_execution(
            publication_kwargs={
                "destination_state": destination_state,
                "reference_samples": lane_center_reference,
                "current_state": current_state,
                "ego_location": ego_location,
                "ego_yaw_rad": float(ego_yaw_rad),
                "ego_speed_mps": float(ego_speed_mps),
                "target_speed_mps": float(speed_ref_mps),
                "behavior": behavior_decision,
                "stop_goal_active": bool(mpc_stop_goal_active),
                "route_points": self._active_global_route_points(),
                "local_map": self._local_map_snapshot,
                "route_cursor": self.route_manager.route_cursor,
                "route_revision": str(self.route_manager.route_revision),
                "map_epoch": str(getattr(self, "waypoint_backend", "admap") or "admap"),
                "reference_source": str(reference_debug.get(
                    "reference_source", "planning_reference"
                )),
                "candidate_status": candidate_status,
                "candidate_reason": candidate_reason,
                "heading_error_rad": (
                    math.radians(float(reference_debug["behavior_lane_heading_error_deg"]))
                    if reference_debug.get("behavior_lane_heading_error_deg", "") != ""
                    else float("nan")
                ),
            },
            behavior=behavior_decision,
            stop_goal_active=bool(mpc_stop_goal_active),
            ego_speed_mps=float(ego_speed_mps),
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_yaw_rad=float(ego_yaw_rad),
            mode_transition_reason=str(mode_transition_guard_reason),
            front_gap_actor_id=str(reference_debug.get("front_gap_actor_id", "")),
            candidate_status=candidate_status,
            candidate_name=candidate_name,
            candidate_reason=candidate_reason,
        )
        publication_result = admission.publication
        destination_state = publication_result.mutable_destination()
        lane_center_reference = publication_result.mutable_samples()
        final_reference_gate = publication_result.gate
        mpc_reference_stabilizer_reason = str(publication_result.stabilizer_reason)
        reference_debug.update(admission.trace_fields())
        mpc_entry = admission.entry
        candidate_hard_gate_reason = str(mpc_entry.hard_gate_reason)
        stationary_traffic_stop_hold = bool(mpc_entry.stationary_stop_hold)
        mpc_control_context = admission.control_context
        destination_forward_m, destination_lateral_m = self._body_frame_xy(
            origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad), target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        reference_first_forward_m = ""
        reference_first_lateral_m = ""
        if lane_center_reference:
            first_reference = lane_center_reference[0]
            reference_first_forward_m, reference_first_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(first_reference.get("x_ref_m", first_reference.get("x", ego_location.x))),
                target_y_m=float(first_reference.get("y_ref_m", first_reference.get("y", ego_location.y))),
            )
        mpc_status = str(getattr(self.mpc, "_last_status", ""))
        # MPC constrains jerk between the previous control input and the new
        # acceleration sequence. Seed that constraint with the acceleration
        # command actually sent last tick, not the measured vehicle response.
        # The latter contains actuator lag and can stay strongly negative
        # after the speed target has recovered, otherwise forcing every new
        # solve to continue braking until the vehicle is almost stationary.
        mpc_jerk_seed_accel_mps2 = float(self._last_accel_mps2)
        road_envelope_payload_world = (
            self._current_route_tracking_lane_change_envelope_payload_world()
        )
        if road_envelope_payload_world is None:
            road_envelope_payload_world = self._rolling_turn_envelope_payload_world(
                behavior_decision=str(behavior_decision.maneuver),
                reference_samples=lane_center_reference,
            )
        cav_result = reference_debug.pop("_cav_resolution", None)
        cav_constraint_rows = ()
        if cav_result is not None:
            cav_constraint_rows = tuple(cav_result.mpc_rows or ())
            self._last_cav_diagnostics = dict(cav_result.diagnostics or {})
            reference_debug["cav_conflict_diagnostics"] = dict(
                self._last_cav_diagnostics
            )
        execution_result = self.pipeline.execute_mpc(
            MPCExecutionRequest(
                sim_time_s=float(sim_time_s),
                current_state=current_state,
                destination_state=destination_state,
                reference_samples=lane_center_reference,
                object_snapshots=self._mpc_object_snapshots_with_prediction(
                    mpc_object_snapshots,
                    prediction_trajectories=reference_debug.get(
                        "prediction_trajectories", {}
                    ),
                ),
                current_acceleration_mps2=float(mpc_jerk_seed_accel_mps2),
                current_steering_rad=float(self._last_steer_rad),
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(speed_ref_mps),
                stop_goal_active=bool(mpc_stop_goal_active),
                behavior_maneuver=str(behavior_decision.maneuver),
                behavior_phase=str(behavior_decision.phase),
                hard_gate_reason=str(candidate_hard_gate_reason),
                stationary_stop_hold=bool(stationary_traffic_stop_hold),
                control_context=mpc_control_context,
                road_envelope_payload_world=road_envelope_payload_world,
                speed_crossing_deadband_mps=float(self.config.get(
                    "control_buffer_speed_crossing_deadband_mps", 0.15,
                )),
                corridor_rows=cav_constraint_rows,
            ),
            normal_stop_control=lambda: self.carla.VehicleControl(
                throttle=0.0,
                brake=min(1.0, max(0.0, float(self.config.get(
                    "normal_stop_mpc_suspend_brake", 0.08,
                )))),
                steer=0.0,
            ),
            safe_stop_control=lambda acceleration_mps2, steering_rad: self.carla.VehicleControl(
                throttle=0.0,
                brake=min(1.0, max(
                    0.0,
                    -float(acceleration_mps2) / max(
                        1e-6,
                        abs(float(self.mpc.constraints.min_acceleration_mps2)),
                    ),
                )),
                steer=min(1.0, max(-1.0, float(steering_rad) / max(
                    1e-6, float(self.mpc.constraints.max_steer_rad),
                ))),
            ),
            emergency_stop_control=self._emergency_stop_control,
        )
        mpc_jerk_seed_accel_mps2 = float(
            execution_result.jerk_seed_acceleration_mps2
        )
        from opencda.planning_module.pipeline.control_finalization_stage import (
            ControlFinalizationRequest,
        )
        finalized_control = self.pipeline.finalize_control(
            ControlFinalizationRequest(
                execution=execution_result,
                behavior=behavior_decision,
                ego_transform=ego_transform,
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(speed_ref_mps),
                destination_state=destination_state,
                reference_samples=lane_center_reference,
                stop_goal_active=bool(mpc_stop_goal_active),
                stop_target_forward_m=stop_target_forward_m_debug,
                final_reference_accepted=bool(final_reference_gate.accepted),
                candidate_status=str(reference_debug.get(
                    "candidate_pipeline_selected_status", ""
                )),
                safety_manager=latest_update.get("safety_manager"),
                carla_module=self.carla,
                sim_time_s=float(sim_time_s),
            ),
            set_actuator_context=self._set_actuator_context,
            acceleration_from_control=self._accel_from_control,
            steering_from_control=self._steer_rad_from_control,
            apply_velocity_steering=self._apply_velocity_steering_interface,
            control_factory=self._control_from_mpc,
            boundary_metrics=self._road_boundary_metrics,
            update_boundary_recovery=self._update_boundary_recovery_request,
            reset_boundary_recovery=self._reset_boundary_recovery_request,
        )
        mpc_jerk_seed_accel_mps2 = float(
            execution_result.jerk_seed_acceleration_mps2
        )
        mpc_status = str(execution_result.status)
        fallback_reason = str(execution_result.fallback_reason)
        hard_gate_active = fallback_reason.startswith("candidate_hard_gate:")
        mpc_replan_executed = bool(execution_result.replan_executed)
        failed_replan_buffer_reused = bool(
            execution_result.failed_replan_buffer_reused
        )
        if fallback_reason and not self._warned:
            print("[CP-X OpenCDA Bridge] MPC fallback active: " + fallback_reason)
            self._warned = True
        if self._cav_conflict_enabled and self._cav_intent_broadcast_enabled:
            self._publish_cav_intent(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                sim_time_s=float(sim_time_s),
            )
        control = finalized_control.control
        accel_mps2 = float(finalized_control.acceleration_mps2)
        steer_rad = float(finalized_control.steering_rad)
        pre_supervisor_accel_mps2 = float(
            finalized_control.pre_filter_acceleration_mps2
        )
        pre_supervisor_steer_rad = float(
            finalized_control.pre_filter_steering_rad
        )
        post_supervisor_accel_mps2 = accel_mps2
        post_supervisor_steer_rad = steer_rad
        control_guard_reason = str(finalized_control.control_guard_reason)
        boundary_guard_reason = str(finalized_control.boundary_guard_reason)
        boundary_snapshot = finalized_control.boundary_snapshot
        safety_supervisor_reason = str(finalized_control.supervisor_reason)
        platform_adapter_debug = dict(finalized_control.platform_debug)
        mpc_feedback_record_reason = str(finalized_control.feedback_reason)
        self._last_accel_mps2 = post_supervisor_accel_mps2
        self._last_steer_rad = post_supervisor_steer_rad
        from opencda.planning_module.pipeline.planner_diagnostics_stage import (
            PlannerDiagnosticsStage,
        )
        diagnostics = PlannerDiagnosticsStage.build(
            self,
            {
                "accel_mps2": accel_mps2,
                "behavior_debug": behavior_debug,
                "behavior_decision": behavior_decision,
                "boundary_snapshot": boundary_snapshot,
                "control": control,
                "control_guard_reason": control_guard_reason,
                "destination_forward_m": destination_forward_m,
                "destination_lateral_m": destination_lateral_m,
                "destination_state": destination_state,
                "ego_location": ego_location,
                "ego_speed_mps": ego_speed_mps,
                "ego_transform": ego_transform,
                "ego_yaw_rad": ego_yaw_rad,
                "emergency_brake_requested": emergency_brake_requested,
                "fallback_reason": fallback_reason,
                "front_gap_m": front_gap_m,
                "hard_gate_active": hard_gate_active,
                "lane_center_reference": lane_center_reference,
                "local_object_snapshots": local_object_snapshots,
                "measured_accel_mps2": measured_accel_mps2,
                "mode_transition_guard_reason": mode_transition_guard_reason,
                "mpc_feedback_record_reason": mpc_feedback_record_reason,
                "mpc_jerk_seed_accel_mps2": mpc_jerk_seed_accel_mps2,
                "mpc_object_snapshots": mpc_object_snapshots,
                "mpc_replan_executed": mpc_replan_executed,
                "mpc_status": mpc_status,
                "mpc_stop_goal_active": mpc_stop_goal_active,
                "normal_stop_requested": normal_stop_requested,
                "object_snapshots": object_snapshots,
                "platform_adapter_debug": platform_adapter_debug,
                "post_supervisor_accel_mps2": post_supervisor_accel_mps2,
                "post_supervisor_steer_rad": post_supervisor_steer_rad,
                "pre_supervisor_accel_mps2": pre_supervisor_accel_mps2,
                "pre_supervisor_steer_rad": pre_supervisor_steer_rad,
                "reference_debug": reference_debug,
                "reference_first_forward_m": reference_first_forward_m,
                "reference_first_lateral_m": reference_first_lateral_m,
                "safety_supervisor_reason": safety_supervisor_reason,
                "speed_ref_mps": speed_ref_mps,
                "speed_target": speed_target,
                "stationary_traffic_stop_hold": stationary_traffic_stop_hold,
                "steer_rad": steer_rad,
                "stop_target_forward_m_debug": stop_target_forward_m_debug,
            },
        )
        decision_record = self.pipeline.explain_decision(diagnostics)
        diagnostics.update(decision_record.as_debug_fields())
        self._draw_world_debug_primitives(
            destination_state=destination_state,
            lane_center_reference=lane_center_reference,
        )
        return PlannerOutput(
            control=control,
            behavior_command=BehaviorCommand.from_debug(
                behavior_debug=behavior_debug,
                target_speed_mps=float(speed_ref_mps),
            ),
            reference_trajectory=[dict(sample) for sample in list(lane_center_reference or [])],
            planned_trajectory=self._last_mpc_trajectory_points(),
            predictions=dict(reference_debug.get("prediction_trajectories", {}) or {}),
            acceleration_mps2=float(post_supervisor_accel_mps2),
            steering_rad=float(post_supervisor_steer_rad),
            diagnostics=PlannerDiagnostics(diagnostics),
        )

    def _resolve_full_traffic_state_from_carla_actor(
        self,
        *,
        raw_state: str,
        signal_context: Mapping[str, object] | None,
    ) -> tuple[str, str]:
        """Resolve a temporarily missing CP signal from its latched CARLA actor."""

        state = str(raw_state or "unknown").strip().lower()
        context = dict(signal_context or {})
        actor_id = str(
            context.get(
                "signal_actor_id",
                context.get("control_id", context.get("cp_control_id", "")),
            )
            or ""
        ).strip()
        if actor_id and state in {"red", "yellow", "green"}:
            self._full_signal_actor_id = str(actor_id)

        should_query_latched_actor = (
            state == "unknown"
            and bool(self._full_signal_actor_id)
            and (
                self._full_traffic_memory.latched_stop_target is not None
                or self._full_traffic_memory.latched_stop_state
                in {"red", "yellow"}
            )
        )
        if not bool(should_query_latched_actor):
            return str(state), ""

        try:
            numeric_actor_id = int(float(self._full_signal_actor_id))
            world = self.vehicle_manager.vehicle.get_world()
            actor = world.get_actor(numeric_actor_id)
            if actor is None:
                return str(state), (
                    f"latched_carla_signal_actor_missing:{numeric_actor_id}"
                )
            actor_state = actor.get_state()
            live_state = str(
                getattr(actor_state, "name", actor_state) or "unknown"
            ).split(".")[-1].strip().lower()
            if live_state in {"red", "yellow", "green"}:
                return (
                    str(live_state),
                    f"latched_carla_signal_actor:{numeric_actor_id}:{live_state}",
                )
            return str(state), (
                f"latched_carla_signal_actor_invalid:{numeric_actor_id}:{live_state}"
            )
        except Exception as exc:
            return str(state), (
                "latched_carla_signal_actor_error:"
                f"{type(exc).__name__}"
            )

    def _resolved_debug_output_dir(self) -> Path:
        """Configured debug dir, with the prediction-ablation mode appended as
        a safety net so the ``cv`` / ``blind`` / ``oracle`` runs of one
        scenario don't clobber each other's ``opencda_planner_debug.csv`` when
        the config forgets to give them distinct ``debug_output_dir`` values.
        Idempotent: never appends a suffix that is already there."""

        base = Path(
            self.config.get(
                "debug_output_dir",
                Path(__file__).resolve().parent / "debug",
            )
        )
        mode = str(getattr(self, "_prediction_mode", "cv"))
        if mode and mode != "cv" and not base.name.endswith(f"_{mode}"):
            return base.with_name(f"{base.name}_{mode}")
        return base

    def _record_debug(self, payload: Mapping[str, Any]) -> None:
        if not bool(self.config.get("record_debug", True)):
            return
        try:
            debug_dir = self._resolved_debug_output_dir()
            debug_dir.mkdir(parents=True, exist_ok=True)
            configured_formats = self.config.get("debug_log_formats", ("jsonl",))
            if isinstance(configured_formats, str):
                configured_formats = (configured_formats,)
            formats = {
                str(item).strip().lower() for item in configured_formats or ()
            }
            unsupported = formats.difference({"csv", "jsonl"})
            if unsupported:
                raise ValueError(
                    "unsupported debug_log_formats: %s"
                    % ",".join(sorted(unsupported))
                )
            if "csv" in formats and self._debug_writer is None:
                self._debug_csv_file = open(
                    debug_dir / "opencda_planner_debug.csv",
                    "w",
                    newline="",
                    encoding="utf-8",
                )
                self._debug_writer = csv.DictWriter(
                    self._debug_csv_file,
                    fieldnames=self._debug_fieldnames,
                    extrasaction="ignore",
                )
                self._debug_writer.writeheader()
            if "jsonl" in formats and self._debug_jsonl_file is None:
                self._debug_jsonl_file = open(
                    debug_dir / "opencda_planner_debug.jsonl",
                    "w",
                    encoding="utf-8",
                )
            if self._debug_writer is not None:
                row = {name: payload.get(name, "") for name in self._debug_fieldnames}
                self._debug_writer.writerow(row)
                self._debug_csv_file.flush()
            if self._debug_jsonl_file is not None:
                self._debug_jsonl_file.write(json.dumps(dict(payload), default=str) + "\n")
                self._debug_jsonl_file.flush()
        except Exception as exc:
            # Diagnostic plumbing failing silently on the branch being debugged
            # is its own trap -- always surface it.
            print(f"[CP-X OpenCDA Bridge] debug record failed: {type(exc).__name__}: {exc}")
            if not getattr(self, "_debug_record_traceback_printed", False):
                import traceback
                traceback.print_exc()
                self._debug_record_traceback_printed = True

    def _spawn_metrics_collision_sensor(self) -> None:
        """Attach one dedicated collision sensor for formal run metrics."""

        if not bool(self.config.get("record_evaluation_metrics", True)):
            return
        if not bool(self.config.get("record_collision_sensor", True)):
            self._metrics_collision_sensor_error = "disabled_by_config"
            return
        vehicle_manager = getattr(self, "vehicle_manager", None)
        vehicle = getattr(vehicle_manager, "vehicle", None)
        try:
            world = vehicle.get_world()
            blueprint = world.get_blueprint_library().find("sensor.other.collision")
            sensor = world.spawn_actor(
                blueprint,
                self.carla.Transform(),
                attach_to=vehicle,
                attachment_type=self.carla.AttachmentType.Rigid,
            )
            sensor.listen(self._on_metrics_collision)
            self._metrics_collision_sensor = sensor
            self._metrics_collision_sensor_available = True
        except Exception as exc:
            self._metrics_collision_sensor_error = str(exc)
            if self.debug:
                print(
                    "[CP-X OpenCDA Bridge] collision metrics sensor unavailable: "
                    f"{exc}"
                )

    def _on_metrics_collision(self, event: Any) -> None:
        """Convert a CARLA collision event into a de-duplicated metric event."""

        vehicle = getattr(self.vehicle_manager, "vehicle", None)
        other_actor = getattr(event, "other_actor", None)
        other_actor_type = str(
            getattr(other_actor, "type_id", type(other_actor).__name__ or "unknown")
        )
        impulse = getattr(event, "normal_impulse", None)
        impulse_magnitude = None
        if impulse is not None:
            try:
                impulse_magnitude = math.sqrt(
                    float(impulse.x) ** 2
                    + float(impulse.y) ** 2
                    + float(impulse.z) ** 2
                )
            except Exception:
                impulse_magnitude = None
        ego_x = ego_y = ego_speed_mps = None
        try:
            location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            ego_x = float(location.x)
            ego_y = float(location.y)
            ego_speed_mps = math.sqrt(
                float(velocity.x) ** 2
                + float(velocity.y) ** 2
                + float(velocity.z) ** 2
            )
        except Exception:
            pass
        event_frame = getattr(event, "frame", "")
        other_actor_id = getattr(other_actor, "id", "")
        event_id = f"{event_frame}:{other_actor_id}"
        self.evaluation_metrics.record_collision(
            event_id,
            sim_time_s=float(self._sim_time_s()),
            ego_x=ego_x,
            ego_y=ego_y,
            ego_speed_mps=ego_speed_mps,
            other_actor_type=other_actor_type,
            impulse_magnitude=impulse_magnitude,
        )
        self._metrics_last_collision_actor_type = other_actor_type
        self._metrics_last_collision_impulse = (
            "" if impulse_magnitude is None else float(impulse_magnitude)
        )

    def _update_boundary_recovery_request(
        self,
        *,
        boundary_snapshot: Mapping[str, object],
        behavior_decision: str,
        sim_time_s: float,
        recovery_planned: bool = False,
        recovery_reference_feasible: bool = True,
    ) -> None:
        from opencda.planning_module.pipeline.scenario_manager import (
            BoundaryRecoveryRequest,
        )

        if float(sim_time_s) < float(
            getattr(
                self,
                "_boundary_recovery_cooldown_until_s",
                -float("inf"),
            )
        ):
            self._reset_boundary_recovery_request()
            return

        geometry_source = str(
            boundary_snapshot.get(
                "road_boundary_geometry_source",
                "",
            )
        )
        drivable_inside = boundary_snapshot.get(
            "road_boundary_drivable_inside",
            "",
        )
        if (
            geometry_source.startswith("drivable_footprint:")
            and drivable_inside in {True, "True", "true", "1", 1}
        ):
            # Consuming the soft boundary margin may request lower speed, but
            # it must not latch hard recovery while the complete footprint is
            # still on CARLA's driving-lane union.
            self._boundary_recovery_infeasible_frames = 0
            self._reset_boundary_recovery_request()
            return

        if bool(recovery_planned) and not bool(recovery_reference_feasible):
            self._boundary_recovery_infeasible_frames = (
                int(
                    getattr(
                        self,
                        "_boundary_recovery_infeasible_frames",
                        0,
                    )
                )
                + 1
            )
            max_failures = max(
                1,
                int(
                    self.config.get(
                        "boundary_recovery_max_infeasible_frames",
                        3,
                    )
                ),
            )
            if int(self._boundary_recovery_infeasible_frames) >= int(
                max_failures
            ):
                self._boundary_recovery_cooldown_until_s = (
                    float(sim_time_s)
                    + max(
                        0.1,
                        float(
                            self.config.get(
                                "boundary_recovery_cooldown_s",
                                2.0,
                            )
                        ),
                    )
                )
                self._reset_boundary_recovery_request()
            return
        self._boundary_recovery_infeasible_frames = 0

        try:
            valid = bool(
                boundary_snapshot.get(
                    "road_boundary_sample_valid",
                    False,
                )
            )
            clearance_m = float(
                boundary_snapshot.get("road_boundary_clearance_m", "")
            )
            lateral_offset_m = float(
                boundary_snapshot.get(
                    "road_boundary_lateral_offset_m",
                    "",
                )
            )
            heading_error_rad = float(
                boundary_snapshot.get(
                    "road_boundary_heading_error_rad",
                    "",
                )
            )
        except (TypeError, ValueError):
            valid = False
            clearance_m = float("inf")
            lateral_offset_m = 0.0
            heading_error_rad = 0.0
        if not bool(valid) or not all(
            math.isfinite(value)
            for value in (
                float(clearance_m),
                float(lateral_offset_m),
                float(heading_error_rad),
            )
        ):
            self._reset_boundary_recovery_request()
            return

        trigger_clearance_m = float(
            self.config.get(
                "boundary_recovery_trigger_clearance_m",
                -0.10,
            )
        )
        release_clearance_m = max(
            float(trigger_clearance_m),
            float(
                self.config.get(
                    "boundary_recovery_release_clearance_m",
                    0.10,
                )
            ),
        )
        if float(clearance_m) <= float(trigger_clearance_m):
            self._boundary_recovery_trigger_frames = (
                int(self._boundary_recovery_trigger_frames) + 1
            )
        else:
            self._boundary_recovery_trigger_frames = 0
        required_frames = max(
            1,
            int(
                self.config.get(
                    "boundary_recovery_trigger_frames",
                    3,
                )
            ),
        )
        previous_active = bool(
            getattr(
                getattr(self, "_boundary_recovery_request", None),
                "active",
                False,
            )
        )
        active = bool(
            (
                bool(previous_active)
                and float(clearance_m) < float(release_clearance_m)
            )
            or int(self._boundary_recovery_trigger_frames)
            >= int(required_frames)
        )
        decision = str(behavior_decision or "").strip().lower()
        turn_direction = (
            "left"
            if decision.endswith("_left")
            else "right"
            if decision.endswith("_right")
            else ""
        )
        self._boundary_recovery_request = BoundaryRecoveryRequest(
            valid=True,
            active=bool(active),
            clearance_m=float(clearance_m),
            lateral_offset_m=float(lateral_offset_m),
            heading_error_rad=float(heading_error_rad),
            turn_direction=str(turn_direction),
            timestamp_s=float(sim_time_s),
            reason=(
                "boundary_recovery_latched"
                if bool(active)
                else "boundary_recovery_monitor"
            ),
        )

    def _reset_boundary_recovery_request(self) -> None:
        from opencda.planning_module.pipeline.scenario_manager import (
            BoundaryRecoveryRequest,
        )

        self._boundary_recovery_trigger_frames = 0
        self._boundary_recovery_request = BoundaryRecoveryRequest()

    def _road_boundary_metrics(
        self,
        ego_location: Any,
        *,
        record_sample: bool = True,
        ego_yaw_rad: float | None = None,
        reference_samples: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, object]:
        """Measure ego with the same route-corridor footprint contract."""

        result: dict[str, object] = {
            "road_boundary_sample_valid": False,
            "road_boundary_lateral_offset_m": "",
            "road_boundary_lane_width_m": "",
            "road_boundary_ego_half_width_m": "",
            "road_boundary_clearance_m": "",
            "road_boundary_breach": "",
            "road_boundary_heading_error_rad": "",
            "road_boundary_projection_segment_index": "",
            "road_boundary_projection_segment_ratio": "",
            "road_boundary_projection_raw_heading_rad": "",
            "road_boundary_projection_conditioned_heading_rad": "",
            "road_boundary_projection_continuity_limited": "",
            "road_boundary_projection_reason": "",
            "road_boundary_geometry_source": "",
            "road_boundary_drivable_inside": "",
        }
        try:
            bounding_box = getattr(self.vehicle_manager.vehicle, "bounding_box", None)
            extent = getattr(bounding_box, "extent", None)
            ego_half_width_m = float(
                getattr(
                    extent,
                    "y",
                    self.config.get("metrics_ego_half_width_m", 1.0),
                )
            )
            ego_half_length_m = float(
                getattr(
                    extent,
                    "x",
                    self.config.get("reference_vehicle_half_length_m", 2.4),
                )
            )
            if ego_yaw_rad is None:
                ego_yaw_rad = math.radians(
                    float(
                        self.vehicle_manager.vehicle.get_transform().rotation.yaw
                    )
                )
            projection = None
            if len(list(reference_samples or [])) >= 2:
                projection = self.reference_generator.project_reference_corridor(
                    reference_samples=reference_samples,
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    ego_half_width_m=float(ego_half_width_m),
                    ego_half_length_m=float(ego_half_length_m),
                    safety_margin_m=float(
                        self.config.get(
                            "reference_contract_turn_boundary_margin_m",
                            0.15,
                        )
                    ),
                    max_heading_step_rad=float(
                        self.config.get(
                            "road_boundary_projection_max_heading_step_rad",
                            0.04,
                        )
                    ),
                    continuity_reset_distance_m=float(
                        self.config.get(
                            "road_boundary_projection_reset_distance_m",
                            2.5,
                        )
                    ),
                    max_position_step_m=float(
                        self.config.get(
                            "road_boundary_projection_max_position_step_m",
                            0.5,
                        )
                    ),
                )
                occupancy = projection.occupancy
            else:
                occupancy = self.reference_generator.lane_corridor_occupancy(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    ego_half_width_m=float(ego_half_width_m),
                    ego_half_length_m=float(ego_half_length_m),
                    safety_margin_m=float(
                        self.config.get(
                            "reference_contract_turn_boundary_margin_m",
                            0.15,
                        )
                    ),
                )
            if not bool(occupancy.valid):
                return result
            lateral_offset_m = float(occupancy.lateral_offset_m)
            lane_width_m = float(occupancy.lane_width_m)
            clearance_m = float(occupancy.footprint_clearance_m)
            geometry_source = "route_tangent_strip"
            drivable_inside: object = ""
            if bool(
                self.config.get(
                    "road_boundary_carla_drivable_footprint_enabled",
                    True,
                )
            ):
                drivable_occupancy = (
                    self.reference_generator.drivable_footprint_occupancy(
                        x_m=float(ego_location.x),
                        y_m=float(ego_location.y),
                        z_m=float(getattr(ego_location, "z", 0.0)),
                        heading_rad=float(ego_yaw_rad),
                        ego_half_width_m=float(ego_half_width_m),
                        ego_half_length_m=float(ego_half_length_m),
                        safety_margin_m=float(
                            self.config.get(
                                "reference_contract_turn_boundary_margin_m",
                                0.15,
                            )
                        ),
                    )
                )
                if bool(drivable_occupancy.valid):
                    clearance_m = float(
                        drivable_occupancy.min_clearance_m
                    )
                    geometry_source = str(
                        drivable_occupancy.reason
                    )
                    drivable_inside = bool(
                        drivable_occupancy.inside
                    )
            breach = (
                not bool(drivable_inside)
                if drivable_inside != ""
                else bool(clearance_m < 0.0)
            )
            if bool(record_sample):
                self._metrics_boundary_sample_count += 1
                if breach:
                    self._metrics_boundary_breach_count += 1
            result.update(
                {
                    "road_boundary_sample_valid": True,
                    "road_boundary_lateral_offset_m": float(lateral_offset_m),
                    "road_boundary_lane_width_m": float(lane_width_m),
                    "road_boundary_ego_half_width_m": float(ego_half_width_m),
                    "road_boundary_clearance_m": float(clearance_m),
                    "road_boundary_breach": bool(breach),
                    "road_boundary_heading_error_rad": float(
                        occupancy.heading_error_rad
                    ),
                    "road_boundary_projection_segment_index": (
                        int(projection.segment_index)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_segment_ratio": (
                        float(projection.segment_ratio)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_raw_heading_rad": (
                        float(projection.raw_heading_rad)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_conditioned_heading_rad": (
                        float(projection.conditioned_heading_rad)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_continuity_limited": (
                        bool(projection.continuity_limited)
                        if projection is not None
                        else False
                    ),
                    "road_boundary_projection_reason": (
                        str(projection.reason)
                        if projection is not None
                        else "lane_corridor_occupancy:map_fallback"
                    ),
                    "road_boundary_geometry_source": str(
                        geometry_source
                    ),
                    "road_boundary_drivable_inside": (
                        drivable_inside
                    ),
                }
            )
        except Exception:
            pass
        return result

    def _update_evaluation_metrics(
        self,
        *,
        ego_location: Any,
        ego_speed_mps: float,
        ego_yaw_rad: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        behavior_decision: str,
        behavior_fsm_state: str,
        mpc_replan_executed: bool,
        cp_summary: Mapping[str, Any],
        reference_samples: Sequence[Mapping[str, Any]] = (),
        boundary_snapshot: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Update run metrics and return fields for the unified debug row."""

        if not bool(self.config.get("record_evaluation_metrics", True)):
            return {
                "evaluation_metrics_available": False,
                "collision_sensor_available": bool(
                    self._metrics_collision_sensor_available
                ),
            }
        sim_time_s = float(self._sim_time_s())
        self.evaluation_metrics.update(
            ego_state={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "v": float(ego_speed_mps),
                "psi": float(ego_yaw_rad),
            },
            obstacle_snapshots=list(object_snapshots or []),
            sim_time_s=sim_time_s,
            behavior_decision=str(behavior_decision),
            fsm_state=str(behavior_fsm_state),
            collision_count=int(self.evaluation_metrics.collision_count),
            last_collision_actor_type=str(
                self._metrics_last_collision_actor_type
            ),
            cp_provider_source=str(cp_summary.get("provider_source", "")),
            native_opencda_available=bool(
                cp_summary.get("native_opencda_available", False)
            ),
            native_opencda_required=bool(
                cp_summary.get("native_opencda_required", False)
            ),
            cp_obstacle_count=int(cp_summary.get("obstacle_count", 0) or 0),
        )
        if bool(mpc_replan_executed):
            self.evaluation_metrics.record_mpc_status(self.mpc.get_runtime_status())
            self.evaluation_metrics.record_mpc_extras(
                lateral_offset_m=None,
                heading_error_rad=None,
                cost_terms=self.mpc.get_last_cost_terms(),
            )
        sample = dict(self.evaluation_metrics.samples[-1])
        boundary = (
            dict(boundary_snapshot)
            if boundary_snapshot is not None
            else self._road_boundary_metrics(
                ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                reference_samples=reference_samples,
            )
        )
        self.evaluation_metrics.record_road_boundary(
            sample_valid=bool(boundary["road_boundary_sample_valid"]),
            lateral_offset_m=(
                boundary["road_boundary_lateral_offset_m"]
                if boundary["road_boundary_lateral_offset_m"] != ""
                else None
            ),
            lane_width_m=(
                boundary["road_boundary_lane_width_m"]
                if boundary["road_boundary_lane_width_m"] != ""
                else None
            ),
            ego_half_width_m=(
                boundary["road_boundary_ego_half_width_m"]
                if boundary["road_boundary_ego_half_width_m"] != ""
                else None
            ),
            clearance_m=(
                boundary["road_boundary_clearance_m"]
                if boundary["road_boundary_clearance_m"] != ""
                else None
            ),
            breach=bool(boundary["road_boundary_breach"]),
        )
        summary = self.evaluation_metrics.summary()
        boundary_sample_count = int(self._metrics_boundary_sample_count)
        collision_count = int(self.evaluation_metrics.collision_count)
        collision_this_frame = (
            collision_count > int(self._metrics_last_emitted_collision_count)
        )
        self._metrics_last_emitted_collision_count = collision_count
        cost_terms = dict(self.mpc.get_last_cost_terms())
        return {
            "evaluation_metrics_available": True,
            "collision_sensor_available": bool(
                self._metrics_collision_sensor_available
            ),
            "collision_sensor_error": str(self._metrics_collision_sensor_error),
            "collision_event_this_frame": bool(collision_this_frame),
            "collision_count": collision_count,
            "collision_rate_per_km": summary.get("collision_rate_per_km", ""),
            "last_collision_actor_type": str(
                self._metrics_last_collision_actor_type
            ),
            "last_collision_impulse": self._metrics_last_collision_impulse,
            "nearest_ttc_s": sample.get("nearest_ttc_s", ""),
            "min_ttc_s": summary.get("min_ttc_s", ""),
            "nearest_ttc_obstacle_id": sample.get(
                "nearest_ttc_obstacle_id", ""
            ),
            "nearest_ttc_reason": sample.get("nearest_ttc_reason", ""),
            "nearest_ttc_longitudinal_gap_m": sample.get(
                "nearest_ttc_longitudinal_gap_m", ""
            ),
            "nearest_ttc_lateral_gap_m": sample.get(
                "nearest_ttc_lateral_gap_m", ""
            ),
            "nearest_ttc_bumper_gap_m": sample.get(
                "nearest_ttc_bumper_gap_m", ""
            ),
            "nearest_ttc_closing_speed_mps": sample.get(
                "nearest_ttc_closing_speed_mps", ""
            ),
            "tick_max_drac_mps2": sample.get("max_drac_mps2", ""),
            "max_drac_mps2": summary.get("max_drac_mps2", ""),
            "min_pet_s": summary.get("min_pet_s", ""),
            "distance_traveled_m": summary.get("distance_traveled_m", ""),
            **boundary,
            "road_boundary_breach_count": int(
                self._metrics_boundary_breach_count
            ),
            "road_boundary_sample_count": boundary_sample_count,
            "road_boundary_breach_rate": (
                float(self._metrics_boundary_breach_count)
                / float(boundary_sample_count)
                if boundary_sample_count > 0
                else ""
            ),
            "Cost_RoadBoundary": cost_terms.get("Cost_RoadBoundary", ""),
            "Cost_Repulsive": cost_terms.get("Cost_Repulsive", ""),
            "Cost_Repulsive_Safe": cost_terms.get("Cost_Repulsive_Safe", ""),
            "Cost_Repulsive_Collision": cost_terms.get(
                "Cost_Repulsive_Collision", ""
            ),
            "Cost_Repulsive_LogBarrier": cost_terms.get(
                "Cost_Repulsive_LogBarrier", ""
            ),
            "Cost_ref": cost_terms.get("Cost_ref", ""),
            "Cost_LaneCenter": cost_terms.get("Cost_LaneCenter", ""),
            "Cost_Control": cost_terms.get("Cost_Control", ""),
            "Cost_VelocitySlack": cost_terms.get("Cost_VelocitySlack", ""),
            "prediction_lane_step_resolved_count": int(
                self._prediction_lane_step_resolved_count
            ),
            "prediction_lane_step_none_count": int(
                self._prediction_lane_step_none_count
            ),
        }

    def destroy(self) -> None:
        sensor = getattr(self, "_metrics_collision_sensor", None)
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass
            self._metrics_collision_sensor = None
        if bool(self.config.get("record_evaluation_metrics", True)):
            try:
                debug_dir = self._resolved_debug_output_dir()
                self._write_planning_metrics_artifacts(
                    artifact_dir=str(debug_dir),
                    recorder=self.evaluation_metrics,
                    scenario_name=str(
                        self.config.get("scenario_name", "opencda_scenario")
                    ),
                )
            except Exception as exc:
                if self.debug:
                    print(
                        "[CP-X OpenCDA Bridge] metrics artifact write failed: "
                        f"{exc}"
                    )
        for handle_name in ("_debug_csv_file", "_debug_jsonl_file"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_name, None)
        self._debug_writer = None

    def _plan_behavior_and_reference(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        speed_ref_mps: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        stop_goal_active: bool,
        cp_payload: Mapping[str, Any] | None = None,
    ):
        sim_time_s = self._sim_time_s()
        route_replan_attempted = False
        route_replan_succeeded = False
        route_replan_reason = "route_replan_not_requested"
        additional_speed_constraints = []
        adapter_output = self.input_adapter.build(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            object_snapshots=object_snapshots,
            cp_payload=cp_payload,
        )
        planner_input_frame = adapter_output.frame
        local_map_snapshot = getattr(
            self, "_local_map_snapshot", LocalMapSnapshot()
        )
        ego_pose = adapter_output.ego_pose
        current_state = adapter_output.current_state
        current_lane_id = int(
            local_map_snapshot.ego_lane_id
            if local_map_snapshot.frame_id > 0 and local_map_snapshot.ego_lane_id != 0
            else adapter_output.current_lane_id
        )
        ego_waypoint = adapter_output.ego_waypoint
        lane_safety_scores = dict(adapter_output.lane_safety_scores)
        front_dist_by_lane = dict(adapter_output.front_distance_by_lane)
        route_points = list(adapter_output.route_points)
        route_context = planner_input_frame.planning.route
        route_optimal_lane_id = int(adapter_output.route_optimal_lane_id)
        route_reference_allowed = bool(adapter_output.route_reference_allowed)
        route_reference_gate_reason = str(adapter_output.route_reference_gate_reason)
        route_behavior = self.pipeline.resolve_route_context(
            adapter_output=adapter_output,
            local_map_snapshot=local_map_snapshot,
            route_manager=self.route_manager,
            maneuver_manager=self.maneuver_manager,
            reference_provider=self._stable_reference_line_provider,
            current_lane_id=int(current_lane_id),
            available_lane_ids=tuple(planner_input_frame.map_lane.allowed_lane_ids),
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            config=self.config,
        )
        lane_change_context = route_behavior.context
        route_lane_change_allowed = bool(
            route_behavior.route_lane_change_allowed
        )
        topology_route_target_lane_id = int(
            lane_change_context.topology_target_lane_id
        )
        physical_route_target_lane_id = int(
            lane_change_context.physical_target_lane_id
        )
        route_geometry_lane_change_direction = str(
            lane_change_context.geometry_direction
        )
        route_geometry_lane_change_distance_m = float(
            lane_change_context.geometry_distance_m
        )
        route_geometry_lane_change_reason = str(
            lane_change_context.geometry_reason
        )
        lane_change_authorization = route_behavior.authorization
        if str(route_behavior.replan_reason):
            (
                route_replan_attempted,
                route_replan_succeeded,
                route_replan_reason,
            ) = self._attempt_turn_route_replan(
                ego_location=ego_location,
                trigger_reason=str(route_behavior.replan_reason),
            )
        route_lane_change_required = bool(lane_change_authorization.required_by_route)
        source_quality = dict(adapter_output.source_quality)
        scenario_observation = self.pipeline.observe_planning_frame(
            ScenarioPlanningFrameRequest(
                adapter_output=adapter_output,
                traffic_memory=self._full_traffic_memory,
                route_manager=self.route_manager,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                current_lane_id=int(current_lane_id),
                cruise_speed_mps=float(self.target_speed_mps),
                sim_time_s=float(sim_time_s),
                config=self.config,
                boundary_recovery_request=(
                    getattr(self, "_boundary_recovery_request", None)
                    if bool(self.config.get("boundary_recovery_enabled", False))
                    else None
                ),
            ),
            resolve_actor_state=self._resolve_full_traffic_state_from_carla_actor,
            project_stop_target=lambda *, stop_target: self.reference_generator.stop_target_forward(
                ego_location=ego_location, ego_yaw_rad=float(ego_yaw_rad),
                stop_target=(dict(stop_target) if isinstance(stop_target, Mapping) else None),
                fallback_destination_state=[],
            ),
        )
        turn_context = scenario_observation.turn_context
        scenario_result = scenario_observation.scenario
        resolved_traffic_state = str(scenario_observation.resolved_traffic_state)
        filtered_traffic_state = str(scenario_observation.filtered_traffic_state)
        full_traffic_memory_reason = str(scenario_result.traffic_memory_reason)
        traffic_stop_forward_m = float(scenario_observation.stop_forward_m)
        upcoming_turn_direction = str(turn_context.direction)
        upcoming_turn_distance_m = float(turn_context.distance_m)
        upcoming_turn_reason = str(turn_context.reason)
        route_advanced_to_lane_change = bool(
            turn_context.route_advanced_to_lane_change
        )
        scenario_decision = scenario_result.decision
        conflict_resolution = self.pipeline.resolve_conflicts(
            ConflictResolutionRequest(
                route_authorization=lane_change_authorization,
                opportunistic_request=OpportunisticLaneChangeRequest(
                    enabled=bool(self.full_allow_opportunistic_lane_change),
                    sim_time_s=float(sim_time_s),
                    start_lock_until_s=float(self.full_lane_change_start_lock_s),
                    dense_traffic_lock_enabled=bool(
                        self.full_dense_traffic_lane_change_lock_enabled
                    ),
                    object_count=len(list(object_snapshots or [])),
                    dense_object_count=int(self.full_dense_traffic_object_count),
                    lane_prediction_risks=dict(
                        planner_input_frame.prediction.lane_prediction_risks
                    ),
                    dense_risky_lane_count=int(
                        self.full_dense_traffic_risky_lane_count
                    ),
                ),
                owner_state=str(scenario_decision.state),
                ego_speed_mps=float(ego_speed_mps),
                planning_speed_mps=float(speed_ref_mps),
                lane_change_duration_s=max(0.1, float(self.config.get(
                    "candidate_lane_change_normal_duration_s", 4.0
                ))),
                dt_s=float(self.mpc.dt_s),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                distance_to_turn_m=float(upcoming_turn_distance_m),
                config=self.config,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
            ),
            maneuver_manager=self.maneuver_manager,
        )
        lane_change_authorization = conflict_resolution.authorization
        lateral_ownership = conflict_resolution.lateral_ownership
        lateral_handoff = lateral_ownership.handoff
        opportunistic_lane_change_allowed = bool(
            conflict_resolution.opportunistic_allowed
        )
        lane_change_gate_reason = str(
            conflict_resolution.lane_change_gate_reason
        )
        if lateral_handoff.action == "release":
            released, release_result = self._stable_reference_line_provider.release(
                LANE_CHANGE,
                event=str(lateral_ownership.reference_release_event),
            )
            if (
                not released
                and self._stable_reference_line_provider.snapshot(
                    LANE_CHANGE
                ).active
            ):
                raise RuntimeError(
                    "lane-change semantic ownership was released but its "
                    "reference remained active: " + str(release_result)
                )
            reset_lane_change = getattr(
                self.behavior_planner, "_reset_lane_change_state", None
            )
            if callable(reset_lane_change):
                reset_lane_change(reason=str(lateral_handoff.reason))
        lane_change_authorized = bool(lane_change_authorization.allowed)
        behavior_traffic_state = str(scenario_result.behavior_traffic_state)
        behavior_stop_target = scenario_result.behavior_stop_target
        traffic_stop_commit_distance_m = float(
            scenario_decision.traffic_stop_commit_distance_m
        )
        traffic_stop_approach_speed_cap_mps = float(
            scenario_decision.speed_cap_mps
            if scenario_decision.speed_cap_mps is not None
            else self.target_speed_mps
        )
        traffic_stop_approach_reason = str(scenario_decision.reason)
        filtered_signal_context = dict(scenario_result.signal_context)
        mpc_feedback = self.mpc_feedback.candidate_feedback(
            current_time_s=float(sim_time_s)
        )
        command_frame = self.pipeline.produce_behavior_command(
            BehaviorCommandFrameRequest(
                adapter_output=adapter_output, ego_pose=ego_pose,
                ego_location=ego_location, ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                current_lane_id=int(current_lane_id),
                target_speed_mps=float(self.target_speed_mps),
                sim_time_s=float(sim_time_s),
                route_optimal_lane_id=int(route_optimal_lane_id),
                route_next_macro_maneuver=str(route_context.next_macro_maneuver),
                route_points=route_points,
                front_distance_by_lane=front_dist_by_lane,
                lane_safety_scores=lane_safety_scores,
                object_snapshots=object_snapshots,
                lane_change_authorization=lane_change_authorization,
                opportunistic_lane_change_allowed=bool(
                    opportunistic_lane_change_allowed
                ),
                behavior_traffic_state=str(behavior_traffic_state),
                behavior_stop_target=behavior_stop_target,
                signal_context=filtered_signal_context,
                scenario_stop_required=bool(scenario_decision.stop_goal_active),
                lane_change_reference_active=bool(
                    self._stable_reference_line_provider.snapshot(
                        LANE_CHANGE
                    ).active
                ),
                mpc_feedback=mpc_feedback,
                max_deceleration_mps2=float(
                    self.mpc.constraints.min_acceleration_mps2
                ),
                config=self.config,
                runtime_config=self.behavior_runtime_cfg,
            ),
            behavior_planner=self.behavior_planner,
            static_obstacle_stage=self.pipeline.static_obstacle,
            reference_map=self.reference_map,
            nearest_front_obstacles=self._nearest_front_obstacle_by_lane,
            attempt_replan=lambda obstacle: self._attempt_static_obstacle_route_replan(
                ego_location=ego_location, obstacle=obstacle,
            ),
            object_track_id=self._object_track_id,
        )
        command_result = command_frame.command
        candidate_frame = command_frame.candidate_frame
        nearest_front_obstacles_by_lane = dict(
            command_frame.nearest_front_obstacles_by_lane
        )
        candidate_lane_ids = list(command_frame.candidate_lane_ids)
        behavior_lane_alignment_valid = bool(command_result.lane_alignment_valid)
        behavior_lane_lateral_error_m = float(command_result.lane_lateral_error_m)
        behavior_lane_heading_error_rad = float(command_result.lane_heading_error_rad)
        traffic_control_stop_active = bool(command_result.traffic_control_stop_active)
        static_obstacle_result = command_result.static_obstacle_result
        static_obstacle_local_avoidance_active = bool(
            static_obstacle_result.local_avoidance_active
        )
        static_obstacle_local_target_lane_id = (
            static_obstacle_result.target_lane_id
        )
        static_obstacle_stop_active = bool(static_obstacle_result.stop_active)
        opportunistic_lane_change_allowed = bool(
            command_result.opportunistic_lane_change_allowed
        )
        preferred_target_lane_id = int(command_result.preferred_target_lane_id)
        decision = str(command_result.decision)
        target_lane_id = int(command_result.target_lane_id)
        lc_state = str(command_result.phase)
        lane_change_commitment_pending_stabilization = bool(
            self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()
        )
        turn_prepare_speed_suppressed_by_lane_change = bool(
            lane_change_commitment_pending_stabilization
            and str(scenario_decision.state).strip().upper() == "PREPARE_TURN"
            and not bool(scenario_decision.stop_goal_active)
        )
        scenario_speed_cap_active = (
            scenario_decision.speed_cap_mps is not None
            and float(scenario_decision.speed_cap_mps) < float(self.target_speed_mps)
            and not bool(turn_prepare_speed_suppressed_by_lane_change)
        )
        route_turn_decision = self._route_option_turn_decision(
            current_road_option=str(route_context.current_road_option),
            next_macro_maneuver="",
        )
        if bool(route_advanced_to_lane_change):
            # Suppress a stale turn proposal once topology reports a later
            # lane change. ManeuverManager owns turn commitment and
            # ReferenceLineProvider owns TURN geometry lifetime; this routing
            # hint must not clear either owner (it can also be observed before
            # an upcoming turn while a prerequisite lane change is pending).
            route_turn_decision = ""
        route_turn_prepare_decision = ""
        scenario_behavior_override = str(scenario_decision.behavior_override_decision or "")
        override_result = self.pipeline.apply_behavior_overrides(
            BehaviorOverrideRequest(
                decision=str(decision), target_lane_id=int(target_lane_id),
                phase=str(lc_state), current_lane_id=int(current_lane_id),
                lane_change_authorized=bool(lane_change_authorized),
                opportunistic_lane_change_allowed=bool(opportunistic_lane_change_allowed),
                lane_change_gate_reason=str(lane_change_gate_reason),
                lane_change_authorization_reason=str(lane_change_authorization.reason),
                prepare_reference_lock=bool(self.full_prepare_lane_change_reference_lock),
                scenario_speed_cap_active=bool(scenario_speed_cap_active),
                scenario_reason=str(traffic_stop_approach_reason),
                scenario_override_decision=str(scenario_behavior_override),
                scenario_override_phase=str(scenario_decision.behavior_override_lc_state),
                scenario_stop_required=bool(scenario_decision.stop_goal_active),
                local_avoidance_active=bool(static_obstacle_local_avoidance_active),
                ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
                lane_change_commitment_active=bool(lane_change_commitment_pending_stabilization),
                route_turn_decision=str(route_turn_decision or route_turn_prepare_decision),
                route_current_road_option=str(route_context.current_road_option),
                stop_goal_active=bool(stop_goal_active),
            )
        )
        decision = str(override_result.decision)
        target_lane_id = int(override_result.target_lane_id)
        lc_state = str(override_result.phase)
        stop_goal_active = bool(override_result.stop_goal_active)
        behavior_override_reason = str(override_result.reason)
        if str(override_result.reset_lane_change_reason):
            reset_lane_change = getattr(
                self.behavior_planner, "_reset_lane_change_state", None
            )
            if callable(reset_lane_change):
                reset_lane_change(reason=str(override_result.reset_lane_change_reason))
        front_gap_m, front_gap_actor_id = self.pipeline.front_gap(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            object_snapshots=object_snapshots,
            current_lane_id=int(current_lane_id),
            lane_assignments=dict(adapter_output.lane_assignments),
            lane_change_direction=(
                "left" if str(decision) == "lane_change_left"
                else "right" if str(decision) == "lane_change_right"
                else ""
            ),
            lane_change_progress=float(self.maneuver_manager.lane_change.progress),
            return_actor_id=True,
        )
        front_gap_obstacle_speed_mps = None
        if front_gap_actor_id:
            for _snapshot in object_snapshots:
                if str(self._object_track_id(_snapshot)) == str(front_gap_actor_id):
                    front_gap_obstacle_speed_mps = max(
                        0.0,
                        float(
                            _snapshot.get(
                                "v", _snapshot.get("speed_mps", 0.0)
                            )
                            or 0.0
                        ),
                    )
                    break
        front_obstacle_lane_id = int(
            adapter_output.lane_assignments.get(
                str(front_gap_actor_id), 0
            ) or 0
        ) if front_gap_actor_id else 0
        lane_change_source_lane_id = int(
            self.maneuver_manager.lane_change.source_lane_id
            if lane_change_commitment_pending_stabilization
            else current_lane_id
        )
        front_obstacle_is_source_lane = bool(
            front_gap_actor_id
            and int(front_obstacle_lane_id) != 0
            and int(front_obstacle_lane_id) == int(lane_change_source_lane_id)
            and (
                int(target_lane_id) != int(lane_change_source_lane_id)
                or lane_change_commitment_pending_stabilization
            )
        )
        speed_plan = self.pipeline.propose_speed(
            scenario_decision=scenario_decision,
            behavior_decision=str(decision),
            requested_speed_mps=float(speed_ref_mps),
            ego_speed_mps=float(ego_speed_mps),
            config=dict(self.config),
            front_gap_m=front_gap_m,
            front_obstacle_speed_mps=front_gap_obstacle_speed_mps,
            upcoming_turn_direction=str(upcoming_turn_direction),
            upcoming_turn_distance_m=(
                None
                if not math.isfinite(float(upcoming_turn_distance_m))
                else float(upcoming_turn_distance_m)
            ),
            lane_change_commitment_active=bool(
                lane_change_commitment_pending_stabilization
            ),
            front_obstacle_is_source_lane=bool(
                front_obstacle_is_source_lane
            ),
            previous_idm_acceleration_mps2=getattr(
                self, "_previous_following_idm_acceleration_mps2", None
            ),
            additional_constraints=tuple(additional_speed_constraints),
        )
        self._previous_following_idm_acceleration_mps2 = (
            None
            if speed_plan.idm_acceleration_mps2 is None
            else float(speed_plan.idm_acceleration_mps2)
        )
        planned_speed_mps = float(speed_plan.target_speed_mps)
        stop_goal_active = bool(stop_goal_active or speed_plan.stop_goal_active)
        planner_mode = "INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL"

        nominal_state = self.nominal_trajectory_generator.current
        base_temporary_destination_state = nominal_state.target_state()
        built_reference = self._stable_reference_line_provider.build_behavior_reference(
            map_planner=self.reference_map,
            ego_pose=ego_pose,
            ego_state=current_state,
            route_points=route_points,
            previous_reference=nominal_state.mutable_samples(),
            previous_target_state=base_temporary_destination_state,
            behavior_runtime_config=self.behavior_runtime_cfg,
            decision=str(decision),
            lane_change_state=str(lc_state),
            target_lane_id=int(target_lane_id),
            current_lane_id=int(current_lane_id),
            route_optimal_lane_id=int(route_optimal_lane_id),
            route_reference_allowed=bool(route_reference_allowed),
            route_reference_gate_reason=str(route_reference_gate_reason),
            in_junction=bool(planner_input_frame.map_lane.in_junction),
            next_macro_maneuver=str(
                planner_input_frame.planning.route.next_macro_maneuver
            ),
            planner_mode=str(planner_mode),
            lookahead_m=float(self.lookahead_m),
            target_speed_mps=float(planned_speed_mps),
            ego_speed_mps=float(ego_speed_mps),
            horizon_steps=int(self.mpc.horizon_steps),
            dt_s=float(self.mpc.dt_s),
            reference_freeze_count=int(nominal_state.reference_freeze_count),
            sim_time_s=float(sim_time_s),
            stop_release_smooth_until_s=float(
                self._stop_release_temp_smooth_until_sim_time_s
            ),
            authoritative_ego_waypoint=self._authoritative_ego_waypoint,
        )
        local_lane_center_reference = built_reference.mutable_samples()
        nominal_destination_state = built_reference.mutable_destination_state()
        nominal_freeze_count = int(built_reference.reference_freeze_count)
        from opencda.planning_module.pipeline.planner_diagnostics_stage import (
            PlannerDiagnosticsStage,
        )
        reference_debug = PlannerDiagnosticsStage.build_reference_debug(self, {
            "built_reference": built_reference,
            "planner_input_frame": planner_input_frame,
            "front_gap_actor_id": front_gap_actor_id,
            "front_gap_obstacle_speed_mps": front_gap_obstacle_speed_mps,
            "front_obstacle_lane_id": front_obstacle_lane_id,
            "front_obstacle_is_source_lane": front_obstacle_is_source_lane,
            "object_snapshots": object_snapshots,
            "route_reference_allowed": route_reference_allowed,
            "route_reference_gate_reason": route_reference_gate_reason,
            "route_lane_change_allowed": route_lane_change_allowed,
            "opportunistic_lane_change_allowed": opportunistic_lane_change_allowed,
            "lane_change_gate_reason": lane_change_gate_reason,
            "static_obstacle_local_avoidance_active": static_obstacle_local_avoidance_active,
            "static_obstacle_local_target_lane_id": static_obstacle_local_target_lane_id,
            "static_obstacle_result": static_obstacle_result,
            "route_lane_change_required": route_lane_change_required,
            "route_geometry_lane_change_direction": route_geometry_lane_change_direction,
            "route_geometry_lane_change_distance_m": route_geometry_lane_change_distance_m,
            "route_geometry_lane_change_reason": route_geometry_lane_change_reason,
            "physical_route_target_lane_id": physical_route_target_lane_id,
            "topology_route_target_lane_id": topology_route_target_lane_id,
            "behavior_lane_lateral_error_m": behavior_lane_lateral_error_m,
            "behavior_lane_heading_error_rad": behavior_lane_heading_error_rad,
            "behavior_lane_alignment_valid": behavior_lane_alignment_valid,
            "lane_change_authorization": lane_change_authorization,
            "behavior_override_reason": behavior_override_reason,
            "scenario_decision": scenario_decision,
            "route_context": route_context,
            "full_traffic_memory_reason": full_traffic_memory_reason,
            "resolved_traffic_state": resolved_traffic_state,
            "filtered_traffic_state": filtered_traffic_state,
            "behavior_traffic_state": behavior_traffic_state,
            "traffic_stop_forward_m": traffic_stop_forward_m,
            "traffic_stop_commit_distance_m": traffic_stop_commit_distance_m,
            "traffic_stop_approach_reason": traffic_stop_approach_reason,
            "speed_plan": speed_plan,
            "candidate_frame": candidate_frame,
            "mpc_feedback": mpc_feedback,
            "upcoming_turn_direction": upcoming_turn_direction,
            "upcoming_turn_distance_m": upcoming_turn_distance_m,
            "upcoming_turn_reason": upcoming_turn_reason,
            "source_quality": source_quality,
            "route_replan_attempted": bool(route_replan_attempted),
            "route_replan_succeeded": bool(route_replan_succeeded),
            "route_replan_reason": str(route_replan_reason),
        })
        # Cooperative arbitration precedes physical maneuver commitment.
        # The first proposal tick is deliberately deferred so both peers can
        # exchange the same proposed claims before either installs a locked
        # lane-change reference.  This is also the sole CAV resolution call
        # for the tick; its corridor rows are reused by MPC below.
        cav_result = None
        cooperative_lane_change_deferred = False
        if self._cav_conflict_enabled:
            lane_change = self.maneuver_manager.lane_change
            cav_claim = self._cooperative_claim_manager.claim(
                decision=str(decision), target_lane_id=int(target_lane_id),
                sim_time_s=float(sim_time_s),
                maneuver_active=bool(lane_change.active),
                committed_at_s=float(lane_change.committed_at_s),
            )
            cav_result = self.pipeline.resolve_cav_interaction(
                reference_samples=local_lane_center_reference,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                actor_id=int(getattr(self.vehicle_manager.vehicle, "id", -1)),
                claim=cav_claim,
                obstacle_snapshots=self._interaction_obstacle_snapshots(
                    object_snapshots,
                    predicted_objects=(
                        planner_input_frame.prediction.predicted_objects
                    ),
                ),
                cav_intents=self._collect_cav_intents(),
                latch_state=self._cav_latch,
                tag_state=self._cav_tag_state,
                horizon_steps=int(self.mpc.horizon_steps),
                dt_s=float(self.mpc.dt_s),
                mode_probability_floor=float(self.config.get(
                    "prediction_mode_min_probability", 0.05
                )),
                credible_mode_probability_min=float(self.config.get(
                    "prediction_credible_probability_min", 0.15
                )),
                credible_mode_ttc_s=float(self.config.get(
                    "prediction_credible_ttc_s", 2.0
                )),
            )
            self._cav_latch = dict(cav_result.latch_state or {})
            self._cav_tag_state = dict(cav_result.tag_state or {})
            cooperative_lane_change_deferred = (
                self._cooperative_claim_manager.defer_candidate(
                    sim_time_s=float(sim_time_s),
                    assignments=cav_result.assignments,
                )
            )
            reference_debug["cav_conflict_diagnostics"] = dict(
                cav_result.diagnostics or {}
            )
            reference_debug["cav_candidate_deferred"] = bool(
                cooperative_lane_change_deferred
            )
            reference_debug["_cav_resolution"] = cav_result
        if bool(self.full_candidate_pipeline_enabled):
            candidate_reference_context = CandidateReferenceBuildContext(
                map_planner=self.reference_map,
                local_map=getattr(self, "_local_map_snapshot", None),
                planner_config=self.config,
                ego_pose=ego_pose,
                current_state=current_state,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                route_points=route_points,
                previous_reference=nominal_state.mutable_samples(),
                previous_target_state=list(base_temporary_destination_state or []),
                behavior_runtime_config=self.behavior_runtime_cfg,
                baseline_decision=str(decision),
                baseline_target_lane_id=int(target_lane_id),
                baseline_speed_mps=float(planned_speed_mps),
                baseline_destination_state=nominal_destination_state,
                baseline_reference=local_lane_center_reference,
                baseline_debug=reference_debug,
                current_lane_id=int(current_lane_id),
                route_optimal_lane_id=int(route_optimal_lane_id),
                route_reference_allowed=bool(route_reference_allowed),
                route_reference_gate_reason=str(route_reference_gate_reason),
                in_junction=bool(planner_input_frame.map_lane.in_junction),
                next_macro_maneuver=str(
                    planner_input_frame.planning.route.next_macro_maneuver
                ),
                planner_mode=str(planner_mode),
                lookahead_m=float(self.lookahead_m),
                horizon_steps=int(self.mpc.horizon_steps),
                dt_s=float(self.mpc.dt_s),
                reference_freeze_count=int(nominal_freeze_count),
                sim_time_s=float(sim_time_s),
                stop_release_smooth_until_s=float(
                    self._stop_release_temp_smooth_until_sim_time_s
                ),
                authoritative_ego_waypoint=self._authoritative_ego_waypoint,
                route_revision=str(self.route_manager.route_revision),
                map_epoch=str(self.waypoint_backend or "admap"),
                upcoming_turn_direction=str(upcoming_turn_direction),
                upcoming_turn_distance_m=float(upcoming_turn_distance_m),
                lane_change_duration_s=float(
                    self.maneuver_manager.lane_change.resolved_duration_s
                ),
                lane_change_duration_reason=str(
                    self.maneuver_manager.lane_change.duration_comfort_reason
                ),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
            )
            candidate_result = self.pipeline.arbitrate_candidates(
                CandidateArbitrationRequest(
                    reference_context=candidate_reference_context,
                    selected_decision=str(decision),
                    selected_target_lane_id=int(target_lane_id),
                    current_lane_id=int(current_lane_id),
                    target_speed_mps=float(planned_speed_mps),
                    candidate_lane_ids=list(candidate_lane_ids),
                    lane_safety_scores=lane_safety_scores,
                    lane_prediction_risks=dict(
                        planner_input_frame.prediction.lane_prediction_risks
                    ),
                    stop_goal_active=bool(stop_goal_active),
                    traffic_stop_active=bool(scenario_decision.stop_goal_active),
                    lane_change_authorization=lane_change_authorization,
                    opportunistic_lane_change_allowed=bool(
                        opportunistic_lane_change_allowed
                    ),
                    stop_target=behavior_stop_target,
                    local_obstacle_avoidance_active=bool(
                        static_obstacle_local_avoidance_active
                    ),
                    ego_speed_mps=float(ego_speed_mps),
                    lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                    baseline_lane_change_state=str(lc_state),
                    current_state=current_state,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    object_snapshots=object_snapshots,
                    prediction_trajectories=(
                        planner_input_frame.prediction.hypothesis_trajectories(
                            minimum_probability=float(self.config.get(
                                "prediction_mode_min_probability", 0.05
                            ))
                        )
                    ),
                    current_acceleration_mps2=float(self._last_accel_mps2),
                    current_steering_rad=float(self._last_steer_rad),
                    route_required=bool(route_lane_change_required),
                    scenario_stop_required=bool(
                        scenario_decision.stop_goal_active
                    ),
                    speed_plan=speed_plan,
                    turn_prepare_speed_suppressed=bool(
                        turn_prepare_speed_suppressed_by_lane_change
                    ),
                    cooperative_lane_change_deferred=bool(
                        cooperative_lane_change_deferred
                    ),
                ),
                sim_time_s=float(sim_time_s),
                route_revision=str(self.route_manager.route_revision),
                road_envelope=(
                    self._current_route_tracking_lane_change_envelope_payload_world
                ),
                validate_contract=self._validate_candidate_reference_contract,
                validate_locked_reference=(
                    self._validate_route_tracking_lane_change_reference
                ),
            )
            decision = str(candidate_result.decision)
            target_lane_id = int(candidate_result.target_lane_id)
            planned_speed_mps = float(candidate_result.target_speed_mps)
            local_lane_center_reference = candidate_result.mutable_reference()
            nominal_destination_state = candidate_result.mutable_destination_state()
            lc_state = str(candidate_result.lane_change_state)
            stop_goal_active = bool(candidate_result.stop_goal_active)
            speed_plan = candidate_result.speed_plan
            additional_speed_constraints.extend(candidate_result.speed_constraints)
            reference_debug = dict(candidate_result.diagnostics)
        else:
            reference_debug["candidate_pipeline_enabled"] = False

        boundary_recovery_active = bool(
            self.config.get("boundary_recovery_enabled", False)
            and scenario_decision.boundary_recovery_active
        )
        # MPC is downstream of candidate selection. Its objective profile must
        # describe the maneuver that will actually execute, not the behavior
        # proposal that existed before candidate arbitration.
        self._apply_mpc_cost_profile(
            behavior=str(decision),
            planner_lc_state=str(lc_state),
            planner_mode=str(planner_mode),
            next_macro_maneuver=str(
                planner_input_frame.planning.route.next_macro_maneuver
            ),
            sim_time_s=float(sim_time_s),
            nearest_obstacle_distance_m=(
                float(front_gap_m)
                if front_gap_m is not None and math.isfinite(float(front_gap_m))
                else None
            ),
            ego_speed_mps=float(ego_speed_mps),
        )
        post_turn = self._stable_reference_line_provider.resolve_post_turn_reference(
            maneuver_manager=self.maneuver_manager,
            decision=str(decision),
            scenario_state=str(getattr(scenario_decision, "state", "")),
            exit_alignment_valid=bool(behavior_lane_alignment_valid),
            exit_lateral_error_m=float(behavior_lane_lateral_error_m),
            exit_heading_error_rad=float(behavior_lane_heading_error_rad),
            local_map=self._local_map_snapshot,
            ego_x_m=float(ego_location.x), ego_y_m=float(ego_location.y),
            current_state=current_state, current_lane_id=int(current_lane_id),
            target_speed_mps=float(planned_speed_mps),
            horizon_steps=int(self.mpc.horizon_steps), dt_s=float(self.mpc.dt_s),
            route_revision=str(self.route_manager.route_revision),
            map_epoch=str(self.waypoint_backend or "admap"), config=self.config,
            destination_state=nominal_destination_state,
            reference_samples=local_lane_center_reference,
            debug_fields=reference_debug,
        )
        if bool(post_turn.clear_turn_reference):
            self._clear_turn_master_reference()
        nominal_destination_state = post_turn.mutable_destination_state()
        local_lane_center_reference = post_turn.mutable_samples()
        reference_debug = dict(post_turn.debug_fields)

        reference_debug["opencda_style_reference_conditioning_reason"] = ""
        # ReferenceLineProvider is the sole geometry owner. ManeuverManager
        # owns lifecycle only and cannot replace a contract-approved path.
        self.nominal_trajectory_generator.update(
            target_state=nominal_destination_state,
            samples=local_lane_center_reference,
            reference_freeze_count=int(nominal_freeze_count),
            source=str(reference_debug.get("reference_source", "planning_tick")),
        )
        return (
            list(nominal_destination_state),
            list(local_lane_center_reference),
            self.pipeline.finalize_behavior_frame(
                maneuver=str(decision),
                phase=str(lc_state),
                source_lane_id=int(current_lane_id),
                target_lane_id=int(target_lane_id),
                requested_speed_mps=float(planned_speed_mps),
                stop_required=bool(stop_goal_active),
                route_required=bool(route_lane_change_required),
                traffic_signal_state=str(behavior_traffic_state),
                boundary_recovery_active=bool(boundary_recovery_active),
                stop_target=(
                    dict(behavior_stop_target)
                    if isinstance(behavior_stop_target, Mapping)
                    else None
                ),
                reason=str(behavior_override_reason),
                lane_safety_scores=lane_safety_scores,
                raw_signal_state=str(
                    planner_input_frame.planning.traffic_control.signal_state
                ),
                resolved_signal_state=str(resolved_traffic_state),
                filtered_signal_state=str(filtered_traffic_state),
                traffic_control_from_cp=bool(
                    planner_input_frame.planning.traffic_control.from_cp
                ),
                scenario_state=str(scenario_decision.state),
            ),
            reference_debug,
            speed_plan,
        )

    def _publish_cav_intent(
        self, *, ego_location: Any, ego_yaw_rad: float,
        ego_speed_mps: float, sim_time_s: float,
    ) -> None:
        """Store this CAV's broadcast (planned trajectory + claim + pose) on
        ``self.last_cav_intent_payload`` for the V2X adapter to transport."""

        from opencda.planning_module.pipeline.cav_intent_codec import (
            build_ego_cav_intent,
            cav_intent_to_payload,
        )

        states = getattr(self.mpc, "_last_x_solution", None)
        planned = (
            [] if states is None
            else [[float(s[0]), float(s[1]), float(s[2]), float(s[3])] for s in states]
        )
        self._cav_intent_sequence += 1
        intent = build_ego_cav_intent(
            actor_id=int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            position_xy=(float(ego_location.x), float(ego_location.y)),
            heading_rad=float(ego_yaw_rad),
            speed_mps=float(ego_speed_mps),
            claim=self._ego_cav_claim(sim_time_s=float(sim_time_s)),
            planned_states=planned,
            dt_s=float(self.mpc.dt_s),
            generated_at_s=float(sim_time_s),
            valid_for_s=float(self.config.get("cav_intent_valid_for_s", 0.5)),
            sequence=int(self._cav_intent_sequence),
        )
        self.last_cav_intent_payload = cav_intent_to_payload(intent)

    def _collect_cav_intents(self) -> list:
        """CavIntent list for every nearby CP-X CAV.

        The simulator adapter currently exposes nearby peers through
        ``v2x_manager.cav_nearby``.  Only their serialized intent payload is
        consumed here; peer CARLA pose and planner internals are deliberately
        not read.  The same payload boundary can therefore be replaced by a
        real V2X transport without changing planning logic.
        """

        if not self._cav_conflict_enabled:
            return []
        v2x_manager = getattr(self.vehicle_manager, "v2x_manager", None)
        cav_nearby = dict(getattr(v2x_manager, "cav_nearby", {}) or {})
        if not cav_nearby:
            return []
        from opencda.planning_module.pipeline.cav_intent_codec import collect_cav_intents

        records = []
        for cav_id, cav_manager in cav_nearby.items():
            cav_bridge = getattr(cav_manager, "cpx_planner", None)
            if cav_bridge is None:
                continue
            payload = getattr(cav_bridge, "last_cav_intent_payload", None)
            if payload is not None:
                records.append(payload)
        return collect_cav_intents(
            records,
            self_actor_id=int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            now_s=float(self._sim_time_s()),
            minimum_probability=float(
                self.config.get("cav_intent_minimum_probability", 0.05)
            ),
        )

    def _ego_cav_claim(self, *, sim_time_s: float):
        """Ego's own cooperative ResourceClaim, active while it is committed
        to a lateral maneuver -- else None (Stage B is skipped, Stage A/C
        still handle plain obstacle conflicts)."""

        if not self._cav_conflict_enabled:
            return None
        lane_change = getattr(self.maneuver_manager, "lane_change", None)
        active = bool(getattr(lane_change, "active", False)) or str(
            getattr(lane_change, "phase", "")
        ) in ("executing", "target_lane_stabilization")
        if active:
            return self._cooperative_claim_manager.claim(
                decision="lane_change_left",
                target_lane_id=int(getattr(lane_change, "target_lane_id", 0) or 0),
                sim_time_s=float(sim_time_s), maneuver_active=True,
                committed_at_s=float(
                    getattr(lane_change, "committed_at_s", 0.0)
                ),
            )
        claim = self._cooperative_claim_manager.current_claim
        if claim is not None:
            return claim
        return self._cooperative_claim_manager.claim(
            decision="lane_follow", target_lane_id=0,
            sim_time_s=float(sim_time_s), maneuver_active=False,
            committed_at_s=0.0,
        )

    def _validate_route_tracking_lane_change_reference(
        self,
        *,
        reference: Sequence[Mapping[str, object]],
        ego_location: carla.Location,
        ego_yaw_rad: float,
    ) -> tuple[bool, str]:
        """Validate heading, curvature, spacing, and road-corridor reachability."""

        samples = [dict(sample) for sample in list(reference or [])]
        if len(samples) < 2:
            return False, "lane_change_validation:too_few_points"
        points: list[tuple[float, float]] = []
        headings: list[float] = []
        distances: list[float] = []
        boundary_failures = 0
        max_lane_fraction = float(
            self.config.get(
                "route_tracking_lane_change_max_lane_center_fraction",
                0.70,
            )
        )
        for sample in samples:
            try:
                x_m = float(sample.get("x_ref_m", sample.get("x", "")))
                y_m = float(sample.get("y_ref_m", sample.get("y", "")))
            except Exception:
                return False, "lane_change_validation:non_finite_point"
            if not math.isfinite(x_m) or not math.isfinite(y_m):
                return False, "lane_change_validation:non_finite_point"
            points.append((x_m, y_m))
            try:
                waypoint = self._map_waypoint_from_location(
                    self.carla.Location(x=float(x_m), y=float(y_m), z=0.0)
                )
            except Exception:
                waypoint = None
            waypoint_xyh = self.reference_generator.waypoint_geometry(waypoint)
            if waypoint is None or waypoint_xyh is None:
                boundary_failures += 1
            else:
                lane_width_m = self.reference_generator.waypoint_lane_width(waypoint)
                lane_distance_m = math.hypot(
                    float(x_m) - float(waypoint_xyh[0]),
                    float(y_m) - float(waypoint_xyh[1]),
                )
                if float(lane_distance_m) > (
                    float(max_lane_fraction) * float(lane_width_m)
                ):
                    boundary_failures += 1
        for first, second in zip(points[:-1], points[1:]):
            dx_m = float(second[0]) - float(first[0])
            dy_m = float(second[1]) - float(first[1])
            distance_m = math.hypot(dx_m, dy_m)
            if distance_m <= 1.0e-4:
                return False, "lane_change_validation:duplicate_point"
            distances.append(float(distance_m))
            headings.append(math.atan2(dy_m, dx_m))
        first_heading_error_rad = abs(
            self._wrap_angle(float(headings[0]) - float(ego_yaw_rad))
        )
        max_heading_error_rad = math.radians(
            float(
                self.config.get(
                    "route_tracking_recovery_heading_error_deg",
                    25.0,
                )
            )
        )
        if float(first_heading_error_rad) > float(max_heading_error_rad):
            return (
                False,
                "lane_change_validation:heading_error:"
                f"{math.degrees(first_heading_error_rad):.1f}deg",
            )
        max_curvature_1pm = 0.0
        max_heading_jump_rad = 0.0
        for index, (first, second) in enumerate(
            zip(headings[:-1], headings[1:])
        ):
            heading_jump_rad = abs(self._wrap_angle(second - first))
            max_heading_jump_rad = max(
                float(max_heading_jump_rad),
                float(heading_jump_rad),
            )
            max_curvature_1pm = max(
                float(max_curvature_1pm),
                float(heading_jump_rad)
                / max(1.0e-3, float(distances[index + 1])),
            )
        curvature_limit = float(
            self.config.get(
                "route_tracking_lane_change_max_curvature_1pm",
                0.35,
            )
        )
        if float(max_curvature_1pm) > float(curvature_limit):
            return (
                False,
                "lane_change_validation:curvature:"
                f"{float(max_curvature_1pm):.3f}",
            )
        heading_jump_limit = float(
            self.config.get(
                "route_tracking_lane_change_max_heading_jump_rad",
                0.35,
            )
        )
        if float(max_heading_jump_rad) > float(heading_jump_limit):
            return (
                False,
                "lane_change_validation:heading_jump:"
                f"{float(max_heading_jump_rad):.3f}",
            )
        max_boundary_failures = int(
            self.config.get(
                "route_tracking_lane_change_max_boundary_failures",
                1,
            )
        )
        if int(boundary_failures) > int(max_boundary_failures):
            return (
                False,
                "lane_change_validation:boundary:"
                f"{int(boundary_failures)}",
            )
        return (
            True,
            "lane_change_validation:valid:"
            f"heading={math.degrees(first_heading_error_rad):.1f}deg:"
            f"curvature={float(max_curvature_1pm):.3f}:"
            f"boundary_failures={int(boundary_failures)}",
        )

    def _attempt_turn_route_replan(
        self,
        *,
        ego_location: Any,
        trigger_reason: str = "turn_reference_unavailable",
    ) -> tuple[bool, bool, str]:
        """Request a bounded route rebuild while preserving stop-on-failure."""

        # Master isolation switch: while validating the route / authorization /
        # reference layers, replan must be off so a downstream failure cannot
        # feed back and reset route progress under the test.
        if not bool(self.config.get("route_replan_enabled", True)):
            self._route_replan_last_reason = "route_replan_disabled_for_isolation"
            return False, False, "route_replan_disabled_for_isolation"

        now_s = float(self._sim_time_s())
        cooldown_s = max(
            0.1,
            float(self.config.get("turn_route_replan_cooldown_s", 2.0)),
        )
        elapsed_s = float(now_s) - float(self._route_replan_last_attempt_s)
        if elapsed_s < cooldown_s:
            reason = (
                "route_replan_cooldown:"
                f"remaining={float(cooldown_s - elapsed_s):.2f}"
            )
            self._route_replan_last_reason = str(reason)
            return False, False, str(reason)

        self._route_replan_last_attempt_s = float(now_s)
        self._route_replan_attempt_count += 1
        result = self.route_manager.replan_from(
            start_point={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "z": float(getattr(ego_location, "z", 0.0)),
            },
            trigger_reason=str(trigger_reason),
        )
        self._route_replan_last_reason = str(result.reason)
        if not bool(result.success):
            return True, False, str(result.reason)

        self._active_route_summary = self.route_manager.active_route_summary
        self.nominal_trajectory_generator.reset(source="turn_route_replanned")
        self._reset_route_tracking_lane_change_reference()
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="turn_route_replanned")
        self.control_buffer.reset(reason="turn_route_replanned")
        return True, True, str(result.reason)

    def _current_route_tracking_lane_change_envelope_payload_world(
        self,
    ) -> Optional[Mapping[str, object]]:
        if (
            not self.maneuver_manager.lane_change.envelope_blocks
            or str(self.maneuver_manager.lane_change.phase)
            != "executing"
        ):
            return None
        return {
            "blocks": self.maneuver_manager.lane_change.envelope_blocks,
            "epsilon0": self.maneuver_manager.lane_change.envelope_epsilon0,
            "rho": float(getattr(self.mpc, "road_envelope_rho", -8.0)),
        }

    def _rolling_turn_envelope_payload_world(
        self,
        *,
        behavior_decision: str,
        reference_samples: Sequence[Mapping[str, object]],
    ) -> Optional[Mapping[str, object]]:
        """Build an MPC road envelope for only the current turn horizon."""

        if str(behavior_decision or "").strip().lower() not in {
            "intersection_turn_left",
            "intersection_turn_right",
        }:
            return None
        if not bool(self.config.get("turn_mpc_road_envelope_enabled", True)):
            return None
        from opencda.planning_module.pipeline.candidate_pipeline import (
            build_turn_reference_envelope_blocks,
        )
        from opencda.planning_module.MPC.lane_keep import (
            road_envelope_conservativeness_correction,
        )

        vehicle = getattr(getattr(self, "vehicle_manager", None), "vehicle", None)
        extent = getattr(getattr(vehicle, "bounding_box", None), "extent", None)
        ego_half_width_m = max(
            0.1,
            float(
                getattr(
                    extent,
                    "y",
                    self.config.get("metrics_ego_half_width_m", 1.0),
                )
            ),
        )
        blocks = build_turn_reference_envelope_blocks(
            reference_samples=reference_samples,
            ego_half_width_m=float(ego_half_width_m),
            safety_margin_m=max(
                0.0,
                float(
                    self.config.get(
                        "turn_mpc_road_envelope_safety_margin_m",
                        self.config.get(
                            "reference_contract_turn_boundary_margin_m",
                            0.15,
                        ),
                    )
                ),
            ),
            default_lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
            longitudinal_overlap_m=max(
                0.0,
                float(self.config.get("turn_mpc_road_envelope_overlap_m", 0.75)),
            ),
        )
        if not blocks:
            return None
        rho = float(getattr(self.mpc, "road_envelope_rho", -8.0))
        return {
            "blocks": blocks,
            "epsilon0": road_envelope_conservativeness_correction(
                blocks,
                rho=float(rho),
            ),
            "rho": float(rho),
            # This is recovery slack, not extra drivable width.  Keeping the
            # 10k envelope penalty means MPC still prefers the body-safe tube,
            # while the larger ceiling prevents a small tracking error at the
            # turn apex from making the entire QP mathematically infeasible.
            "max_slack_m": max(
                0.10,
                float(
                    self.config.get(
                        "turn_mpc_road_envelope_recovery_slack_m",
                        1.5,
                    )
                ),
            ),
        }

    def _lane_change_lifecycle(self):
        stage = getattr(self, "lane_change_lifecycle_stage", None)
        if stage is not None:
            return stage
        from opencda.planning_module.pipeline.lane_change_lifecycle_stage import (
            LaneChangeLifecycleStage,
        )
        stage = LaneChangeLifecycleStage(
            provider=self._stable_reference_line_provider,
            maneuver_manager=self.maneuver_manager,
            reference_generator=self.reference_generator,
            route_manager=self.route_manager,
            mpc=self.mpc,
            control_buffer=getattr(self, "control_buffer", None),
            config=self.config,
            target_speed_mps=float(getattr(self, "target_speed_mps", 3.0)),
            map_epoch=str(getattr(self, "waypoint_backend", "admap") or "admap"),
            vehicle_extent=lambda: getattr(
                getattr(
                    getattr(self, "vehicle_manager", None), "vehicle", None
                ),
                "bounding_box",
                None,
            ) and getattr(self.vehicle_manager.vehicle.bounding_box, "extent", None),
        )
        self.lane_change_lifecycle_stage = stage
        return stage

    # Transitional compatibility ports for focused tests and external tools.
    # The lifecycle implementation and state ownership live in the stage.
    def _reset_route_tracking_lane_change_reference(self) -> None:
        self._lane_change_lifecycle().reset_reference()

    def _release_completed_lane_change_commitment(
        self, *, current_lane_id: int, ego_location: Any, ego_yaw_rad: float,
    ) -> str:
        return self._lane_change_lifecycle().release_completed(
            current_lane_id=current_lane_id,
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
        )

    def _start_target_lane_stabilization(
        self, *, ego_location: Any, ego_yaw_rad: float,
    ) -> str:
        return self._lane_change_lifecycle()._start_stabilization(
            ego_location=ego_location, ego_yaw_rad=ego_yaw_rad
        )
    def _clear_turn_master_reference(self) -> None:
        provider = getattr(self, "_stable_reference_line_provider", None)
        if provider is not None:
            provider.release(TURN, event="reset")

    @staticmethod
    def _normalized_final_lc_state(
        *, decision: str, lc_state: str, lane_change_phase: str = ""
    ) -> str:
        """Compatibility wrapper; CandidateSelectionStage owns normalization."""

        if str(decision or "").strip().lower() not in {
            "lane_change_left", "lane_change_right",
        }:
            return str(lc_state or "LANE_KEEP")
        from opencda.planning_module.pipeline.candidate_selection_stage import (
            CandidateSelectionStage,
        )
        return CandidateSelectionStage.normalized_lane_change_state(
            decision=str(decision), lane_change_phase=str(lane_change_phase),
        )

    @staticmethod
    def _route_option_turn_decision(*, current_road_option: str, next_macro_maneuver: str) -> str:
        route_option = str(current_road_option or "").strip().upper()
        macro = (
            str(next_macro_maneuver or "")
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        macro_tokens = set(macro.split())
        if route_option == "LEFT" or {"left", "turn"}.issubset(macro_tokens):
            return "intersection_turn_left"
        if route_option == "RIGHT" or {"right", "turn"}.issubset(macro_tokens):
            return "intersection_turn_right"
        return ""

    def _apply_behavior_mode_transition_guard(
        self,
        *,
        decision: str,
        lc_state: str,
        target_lane_id: int,
        stop_goal_active: bool,
    ) -> str:
        mode_key = self._behavior_mode_key(
            decision=str(decision),
            lc_state=str(lc_state),
            target_lane_id=int(target_lane_id),
            stop_goal_active=bool(stop_goal_active),
        )
        previous_key = str(getattr(self, "_full_last_behavior_mode_key", "") or "")
        self._full_last_behavior_mode_key = str(mode_key)
        if not previous_key or previous_key == str(mode_key):
            return ""
        reset_control_buffer = getattr(self.control_buffer, "reset", None)
        if callable(reset_control_buffer):
            reset_control_buffer(reason="control_buffer_reset_mode_transition")
        return f"mode_transition:{previous_key}->{mode_key}:reset_control_buffer"

    @staticmethod
    def _behavior_mode_key(
        *,
        decision: str,
        lc_state: str,
        target_lane_id: int,
        stop_goal_active: bool,
    ) -> str:
        normalized_decision = str(decision or "").strip().lower()
        normalized_fsm = str(lc_state or "").strip().upper()
        if bool(stop_goal_active) or normalized_decision in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }:
            return "stop"
        if normalized_decision in {"intersection_turn_left", "intersection_turn_right"}:
            return str(normalized_decision)
        if normalized_decision == "route_recovery":
            return "route_recovery"
        if normalized_decision in {"lane_change_left", "lane_change_right"}:
            return f"{normalized_decision}:{int(target_lane_id)}"
        if normalized_fsm.startswith("EXECUTE_LANE_CHANGE"):
            return f"{normalized_fsm.lower()}:{int(target_lane_id)}"
        return f"lane_follow:{int(target_lane_id)}"

    def _validate_candidate_reference_contract(
        self,
        *,
        decision: str,
        lc_state: str,
        current_lane_id: int,
        speed_ref_mps: float,
        stop_goal_active: bool,
        current_state: Sequence[float],
        destination_state: Sequence[float],
        lane_center_reference: Sequence[Mapping[str, object]],
    ):
        from opencda.planning_module.pipeline.reference_contract import (
            contract_from_config,
            validate_reference_contract,
        )

        normalized_decision = str(decision or "").strip().lower()
        normalized_fsm = str(lc_state or "").strip().upper()
        lane_change_active = (
            normalized_decision in {"lane_change_left", "lane_change_right"}
            or normalized_fsm.startswith("EXECUTE_LANE_CHANGE")
        )
        turn_active = (
            normalized_decision in {"intersection_turn_left", "intersection_turn_right"}
            or normalized_fsm.startswith("INTERSECTION_TURN")
        )
        stop_like = bool(stop_goal_active) or normalized_decision in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        direct_target_tracking_enabled = bool(
            self.config.get(
                "route_tracking_lane_change_direct_target_tracking_enabled",
                False,
            )
        )
        contract_mode = (
            "emergency_stop"
            if normalized_decision == "emergency_brake"
            else "stop"
            if bool(stop_like)
            else "lane_change_direct"
            if bool(lane_change_active) and bool(direct_target_tracking_enabled)
            else "lane_change"
            if bool(lane_change_active)
            else "intersection_turn"
            if bool(turn_active)
            else "lane_follow"
        )
        expected_lane_id = int(current_lane_id)
        if bool(lane_change_active) and len(destination_state or []) >= 5:
            try:
                expected_lane_id = int(float(destination_state[4]))
            except (TypeError, ValueError):
                expected_lane_id = int(current_lane_id)
        contract = contract_from_config(
            mode=str(contract_mode),
            expected_lane_id=int(expected_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            config=dict(self.config),
            default_speed_mps=max(float(self.target_speed_mps), float(speed_ref_mps), 0.1),
        )
        recovery_reference_active = any(
            str(sample.get("lane_transition_kind", ""))
            == "ego_anchored_lane_recovery"
            for sample in list(lane_center_reference or [])[:2]
        )
        validation = validate_reference_contract(
            reference_samples=lane_center_reference,
            destination_state=destination_state,
            ego_state=current_state,
            contract=contract,
            check_destination_body_lateral=not bool(
                lane_change_active or turn_active or recovery_reference_active
            ),
        )
        if (
            bool(validation.valid)
            and bool(turn_active)
            and bool(
                self.config.get(
                    "reference_contract_turn_vehicle_footprint_enabled",
                    True,
                )
            )
        ):
            boundary_valid, boundary_reason = (
                self._reference_vehicle_footprint_boundary_valid(
                    reference_samples=lane_center_reference,
                )
            )
            if not bool(boundary_valid):
                validation.valid = False
                validation.violations.append(str(boundary_reason))
        return validation

    def _reference_vehicle_footprint_boundary_valid(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
    ) -> tuple[bool, str]:
        """Delegate candidate turn-corridor validation to ReferenceGenerator."""

        vehicle_manager = getattr(self, "vehicle_manager", None)
        vehicle = getattr(vehicle_manager, "vehicle", None)
        bounding_box = getattr(vehicle, "bounding_box", None)
        extent = getattr(bounding_box, "extent", None)
        ego_half_width_m = float(
            getattr(
                extent,
                "y",
                self.config.get("metrics_ego_half_width_m", 1.0),
            )
        )
        ego_half_length_m = float(
            getattr(
                extent,
                "x",
                self.config.get("reference_vehicle_half_length_m", 2.4),
            )
        )
        safety_margin_m = max(
            0.0,
            float(
                self.config.get(
                    "reference_contract_turn_boundary_margin_m",
                    0.15,
                )
            ),
        )
        max_failures = max(
            0,
            int(
                self.config.get(
                    "reference_contract_turn_max_boundary_failures",
                    1,
                )
            ),
        )
        validation = self.reference_generator.validate_turn_swept_footprint(
            reference_samples=reference_samples,
            ego_half_width_m=max(0.1, float(ego_half_width_m)),
            ego_half_length_m=max(0.1, float(ego_half_length_m)),
            safety_margin_m=float(safety_margin_m),
            max_violations=int(max_failures),
        )
        return bool(validation.valid), (
            "" if bool(validation.valid) else str(validation.reason)
        )

    def _sim_time_s(self) -> float:
        try:
            snapshot = self.vehicle_manager.vehicle.get_world().get_snapshot()
            return float(snapshot.timestamp.elapsed_seconds)
        except Exception:
            return 0.0

    def prediction_snapshot_transform(self):
        """Return the obstacle-snapshot transform for the active
        prediction-knowledge ablation mode, or ``None`` for the default
        ``cv`` pipeline behaviour.

        ``cv``     -> None (constant-velocity/acceleration kinematic rollout).
        ``blind``  -> every obstacle frozen at its current pose ("no predicted
                      trajectory": ego knows position, assumes no motion).
        ``oracle`` -> each obstacle's recorded ground-truth future + a
                      maneuver-intent label, replayed from ``oracle_trace_path``.
        """

        if self._prediction_snapshot_transform_cached:
            return self._prediction_snapshot_transform_fn
        from opencda.planning_module.pipeline.prediction_ablation import (
            build_snapshot_transform,
        )
        self._prediction_snapshot_transform_fn = build_snapshot_transform(
            prediction_mode=self._prediction_mode,
            horizon_s=float(self.mpc.horizon_s),
            dt_s=float(self.mpc.dt_s),
            oracle_store=self._oracle_trace_store,
            freeze_unmatched_oracle=bool(
                self.config.get("oracle_freeze_unmatched", False)
            ),
            synthetic_actor_ids=tuple(
                int(value)
                for value in list(
                    self.config.get("synthetic_prediction_actor_ids", []) or []
                )
            ),
        )
        self._prediction_snapshot_transform_cached = True
        return self._prediction_snapshot_transform_fn

    def _obstacle_lane_step_fn(self):
        """Return a ``(x, y, distance_m) -> (x, y, heading_rad) | None``
        closure for lane-curve-aware obstacle prediction, or None to keep the
        old straight-line-only fallback.

        ``obstacle_future_trajectory`` (behavior_planner/trajectory_risk.py)
        only follows the lane centerline when given this closure; without
        it, every obstacle without a CP-supplied ``predicted_trajectory``
        keeps being extrapolated as a straight line at its current heading,
        which is wrong for a vehicle following a curved lane (e.g. mid-turn
        at an intersection).
        """

        if not bool(self.config.get("prediction_lane_following_enabled", True)):
            return None
        from utility.global_planner import lane_step_xy_heading

        get_waypoint_fn = self.reference_map.get_waypoint

        def _step(x_m: float, y_m: float, distance_m: float):
            result = lane_step_xy_heading(
                float(x_m),
                float(y_m),
                float(distance_m),
                get_waypoint_fn=get_waypoint_fn,
            )
            if result is None:
                self._prediction_lane_step_none_count += 1
            else:
                self._prediction_lane_step_resolved_count += 1
            return result

        return _step

    def _assign_obstacles_to_lanes(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        ego_waypoint: Any = None,
        ego_lane_id: int = 0,
    ) -> dict[str, int]:
        from utility.global_planner import canonical_lane_id_for_waypoint

        assignments: dict[str, int] = {}
        for snapshot in list(object_snapshots or []):
            obstacle_id = self._object_track_id(snapshot)
            if not obstacle_id:
                continue
            waypoint = self.reference_map.get_waypoint({
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "z": float(snapshot.get("z", 0.0)),
            })
            # Ego, obstacles, route and prediction all use opaque AD-map ids.
            # Cross-road continuity is handled by topology, not by rewriting
            # an obstacle into a small canonical lane index.
            lane_id = int(canonical_lane_id_for_waypoint(waypoint) or 0)
            if int(lane_id) != 0:
                assignments[obstacle_id] = int(lane_id)
        return assignments

    @staticmethod
    def _object_track_id(snapshot: Mapping[str, Any]) -> str:
        for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id"):
            value = snapshot.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        try:
            return "xy:{:.1f}:{:.1f}".format(
                float(snapshot.get("x", snapshot.get("x_m", 0.0))),
                float(snapshot.get("y", snapshot.get("y_m", 0.0))),
            )
        except Exception:
            return ""

    @staticmethod
    def _nearest_front_distance_by_lane(
        *,
        ego_snapshot: Mapping[str, object],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        lane_assignments: Mapping[str, int],
        available_lane_ids: Sequence[int],
    ) -> dict[int, float]:
        ego_x = float(ego_snapshot.get("x", 0.0))
        ego_y = float(ego_snapshot.get("y", 0.0))
        ego_psi = float(ego_snapshot.get("psi", 0.0))
        cos_h = math.cos(ego_psi)
        sin_h = math.sin(ego_psi)
        nearest: dict[int, float] = {}
        allowed = {int(lane_id) for lane_id in list(available_lane_ids or [])}
        for snapshot in list(obstacle_snapshots or []):
            obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
            lane_id = int(lane_assignments.get(obstacle_id, 0))
            if lane_id not in allowed:
                continue
            dx = float(snapshot.get("x", 0.0)) - ego_x
            dy = float(snapshot.get("y", 0.0)) - ego_y
            longitudinal = dx * cos_h + dy * sin_h
            if longitudinal <= 0.0:
                continue
            nearest[lane_id] = min(float(nearest.get(lane_id, float("inf"))), float(longitudinal))
        return {
            int(lane_id): float(distance)
            for lane_id, distance in nearest.items()
            if math.isfinite(float(distance))
        }

    @classmethod
    def _nearest_front_obstacle_by_lane(
        cls,
        *,
        ego_snapshot: Mapping[str, object],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        lane_assignments: Mapping[str, int],
        available_lane_ids: Sequence[int],
    ) -> dict[int, dict[str, Any]]:
        """Return the nearest complete front-obstacle record per lane."""

        ego_x = float(ego_snapshot.get("x", 0.0))
        ego_y = float(ego_snapshot.get("y", 0.0))
        ego_psi = float(ego_snapshot.get("psi", 0.0))
        cos_h = math.cos(ego_psi)
        sin_h = math.sin(ego_psi)
        allowed = {int(lane_id) for lane_id in list(available_lane_ids or [])}
        nearest: dict[int, dict[str, Any]] = {}
        for raw_snapshot in list(obstacle_snapshots or []):
            snapshot = dict(raw_snapshot)
            obstacle_id = cls._object_track_id(snapshot)
            lane_id = int(lane_assignments.get(str(obstacle_id), 0))
            if lane_id not in allowed:
                continue
            obstacle_x = float(snapshot.get("x", snapshot.get("x_m", 0.0)))
            obstacle_y = float(snapshot.get("y", snapshot.get("y_m", 0.0)))
            longitudinal_m = (
                (obstacle_x - ego_x) * cos_h
                + (obstacle_y - ego_y) * sin_h
            )
            if longitudinal_m <= 0.0:
                continue
            previous = nearest.get(int(lane_id))
            if previous is not None and float(
                previous.get("front_distance_m", float("inf"))
            ) <= float(longitudinal_m):
                continue
            snapshot["vehicle_id"] = str(obstacle_id)
            snapshot["x"] = float(obstacle_x)
            snapshot["y"] = float(obstacle_y)
            snapshot["v"] = max(
                0.0,
                float(snapshot.get("v", snapshot.get("speed_mps", 0.0))),
            )
            snapshot["front_distance_m"] = float(longitudinal_m)
            nearest[int(lane_id)] = snapshot
        return nearest

    def _attempt_static_obstacle_route_replan(
        self,
        *,
        ego_location: Any,
        obstacle: Mapping[str, object],
    ) -> tuple[bool, bool, str]:
        """Block the obstacle lane and atomically rebuild the active route."""

        block_fn = getattr(self.global_planner, "block_lane_at_position", None)
        if not callable(block_fn):
            reason = "static_obstacle_block_lane_unsupported"
            return True, False, str(reason)
        blocked_lane_id = block_fn({
            "x": float(obstacle.get("x", obstacle.get("x_m", 0.0))),
            "y": float(obstacle.get("y", obstacle.get("y_m", 0.0))),
            "z": float(obstacle.get("z", obstacle.get("z_m", 0.0))),
        })
        if blocked_lane_id is None:
            reason = "static_obstacle_lane_mapping_failed"
            return True, False, str(reason)
        self._static_obstacle_blocked_lane_id = blocked_lane_id

        result = self.route_manager.replan_from(
            start_point={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "z": float(getattr(ego_location, "z", 0.0)),
            },
            trigger_reason="static_obstacle",
        )
        if not bool(result.success):
            return True, False, str(result.reason)

        self._active_route_summary = self.route_manager.active_route_summary
        self.nominal_trajectory_generator.reset(
            source="static_obstacle_route_replanned"
        )
        self._reset_route_tracking_lane_change_reference()
        self.maneuver_manager.clear_required_lane_change()
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="static_obstacle_route_replanned")
        self.control_buffer.reset(reason="static_obstacle_route_replanned")
        if hasattr(self.mpc, "clear_previous_solution_seed"):
            self.mpc.clear_previous_solution_seed()
        return True, True, str(result.reason)

    def _load_cp_message_payload(self) -> dict[str, Any]:
        try:
            with open(self.cp_message_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return dict(payload or {})
        except Exception:
            return {}

    @staticmethod
    def _traffic_context_from_cp_control(
        *,
        selected_control: Mapping[str, object] | None,
        ego_location: carla.Location,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        if not isinstance(selected_control, Mapping):
            return {"signal_state": "unknown", "from_cp": False}, None
        state = str(
            selected_control.get(
                "signal_state",
                selected_control.get("state", "unknown"),
            )
            or "unknown"
        ).strip().lower()
        stop_line = selected_control.get("stop_line_position", selected_control.get("stop_line", None))
        stop_target = None
        if isinstance(stop_line, Mapping):
            x_value = stop_line.get("x", stop_line.get("x_m", None))
            y_value = stop_line.get("y", stop_line.get("y_m", None))
            if x_value is not None and y_value is not None:
                distance_m = math.hypot(float(x_value) - float(ego_location.x), float(y_value) - float(ego_location.y))
                stop_target = {
                    "x_m": float(x_value),
                    "y_m": float(y_value),
                    "lane_id": int(float(
                        selected_control.get("lane_id", stop_line.get("lane_id", 0)) or 0
                    )),
                    "road_id": int(float(
                        selected_control.get("road_id", stop_line.get("road_id", 0)) or 0
                    )),
                    "distance_m": float(distance_m),
                    "source": "opencda_cp_control",
                }
        context = {
            "signal_state": str(state),
            "signal_source": str(selected_control.get("source", "opencda_cp")),
            "source": str(selected_control.get("source", "opencda_cp")),
            "cp_control_id": str(selected_control.get("control_id", selected_control.get("id", ""))),
            "control_id": str(selected_control.get("control_id", selected_control.get("id", ""))),
            "signal_actor_id": str(
                selected_control.get(
                    "signal_actor_id",
                    selected_control.get(
                        "control_id",
                        selected_control.get("id", ""),
                    ),
                )
            ),
            "cp_provider_source": str(selected_control.get("provider_source", "")),
            "provider_source": str(selected_control.get("provider_source", "")),
            "from_cp": True,
            "traffic_control_from_cp": True,
            "confidence": float(selected_control.get("confidence", 1.0) or 0.0),
            "ego_passed_stop_line": bool(selected_control.get("ego_passed_stop_line", False)),
        }
        return context, stop_target

    def _select_relevant_traffic_control(
        self,
        *,
        traffic_controls: Sequence[Mapping[str, object]],
        ego_location: carla.Location,
        ego_heading_rad: float,
        current_lane_id: int,
        current_road_id: int,
        sim_time_s: float,
    ) -> Mapping[str, object] | None:
        best_control: Mapping[str, object] | None = None
        best_score: tuple[float, float, float] | None = None
        cos_h = math.cos(float(ego_heading_rad))
        sin_h = math.sin(float(ego_heading_rad))
        for control in list(traffic_controls or []):
            if not isinstance(control, Mapping):
                continue
            if not self._cp_message_is_fresh(control, sim_time_s=float(sim_time_s)):
                continue
            stop_line = control.get("stop_line_position", control.get("stop_line", None))
            if not isinstance(stop_line, Mapping):
                continue
            x_value = stop_line.get("x", stop_line.get("x_m", None))
            y_value = stop_line.get("y", stop_line.get("y_m", None))
            if x_value is None or y_value is None:
                continue
            dx_m = float(x_value) - float(ego_location.x)
            dy_m = float(y_value) - float(ego_location.y)
            forward_m = cos_h * dx_m + sin_h * dy_m
            lateral_m = -sin_h * dx_m + cos_h * dy_m
            if bool(control.get("ego_passed_stop_line", False)) or float(forward_m) < -1.0:
                continue
            lane_id = int(float(
                control.get(
                    "lane_id",
                    stop_line.get("lane_id", 0),
                )
                or 0
            ))
            road_id = int(float(
                control.get(
                    "road_id",
                    stop_line.get("road_id", 0),
                )
                or 0
            ))
            road_mismatch = 1.0 if road_id and current_road_id and road_id != current_road_id else 0.0
            lane_mismatch = 1.0 if lane_id and current_lane_id and lane_id != current_lane_id else 0.0
            score = (road_mismatch, lane_mismatch, abs(float(lateral_m)) + 0.01 * float(forward_m))
            if best_score is None or score < best_score:
                best_control = control
                best_score = score
        return best_control

    @staticmethod
    def _cp_message_is_fresh(message: Mapping[str, object], *, sim_time_s: float) -> bool:
        try:
            valid_until_s = float(message.get("valid_until_s", "nan"))
            if math.isfinite(valid_until_s):
                return float(sim_time_s) <= valid_until_s
        except Exception:
            pass
        try:
            timestamp_s = float(message.get("timestamp_s", sim_time_s))
            ttl_s = float(message.get("ttl_s", 0.0))
        except Exception:
            return True
        if float(ttl_s) <= 0.0:
            return True
        return float(sim_time_s) <= float(timestamp_s) + float(ttl_s)

    def _planning_module_global_route_summary(
        self,
        *,
        ego_location: carla.Location,
        ego_heading_rad: float,
        fallback_lane_id: int,
        ego_waypoint: Any = None,
    ) -> dict[str, object]:
        if getattr(self, "global_planner_backend", "") == "custom_admap_dijkstra":
            self._authoritative_ego_waypoint = None
            matched_waypoint = None
            try:
                raw_candidates = list(
                    self.global_planner.get_waypoint_candidates(
                        {
                            "x": float(ego_location.x),
                            "y": float(ego_location.y),
                            "z": float(getattr(ego_location, "z", 0.0)),
                        }
                    )
                    or []
                )
                previous_frame = dict(self._diagnostic_local_lane_frame or {})
                previous_corridors = {
                    int(key): list(value or [])
                    for key, value in dict(
                        previous_frame.get("corridors", {}) or {}
                    ).items()
                }
                previous_lane_id = int(
                    getattr(self._diagnostic_hd_map_matcher.previous, "ad_lane_id", 0)
                    or 0
                )
                candidates = []
                waypoint_by_lane: dict[int, object] = {}
                from utility.global_planner import world_heading_rad

                for item in raw_candidates:
                    waypoint = item.get("waypoint")
                    if waypoint is None:
                        continue
                    position = dict(getattr(waypoint, "position", {}) or {})
                    ad_lane_id = int(item.get("ad_lane_id", 0) or 0)
                    waypoint_by_lane.setdefault(ad_lane_id, waypoint)
                    candidates.append(LaneProjectionCandidate(
                        ad_lane_id=ad_lane_id,
                        road_id=int(getattr(waypoint, "road_id", 0) or 0),
                        section_id=int(getattr(waypoint, "section_id", 0) or 0),
                        raw_lane_id=int(getattr(waypoint, "lane_id", 0) or 0),
                        center_x_m=float(position.get("x", ego_location.x)),
                        center_y_m=float(position.get("y", ego_location.y)),
                        heading_rad=float(world_heading_rad(waypoint) or 0.0),
                        lane_width_m=max(
                            0.1, float(getattr(waypoint, "lane_width_m", 3.5) or 3.5)
                        ),
                        snap_distance_m=float(item.get("snap_distance_m", 0.0)),
                        is_in_lane=bool(item.get("is_in_lane", False)),
                        probability=float(item.get("probability", 0.0)),
                        topology_relation=topology_relation(
                            candidate_lane_id=ad_lane_id,
                            previous_lane_id=previous_lane_id,
                            previous_corridors=previous_corridors,
                        ),
                    ))
                matched = self._diagnostic_hd_map_matcher.update(
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    ego_heading_rad=float(ego_heading_rad),
                    candidates=candidates,
                )
                matched_waypoint = waypoint_by_lane.get(int(matched.ad_lane_id))
                self._diagnostic_map_matching = {
                    **matched.as_dict(),
                    "candidate_count": len(candidates),
                }
            except Exception as exc:
                self._diagnostic_map_matching = {
                    "valid": False,
                    "match_reason": f"diagnostic_map_match_failed:{exc}",
                    "candidate_count": 0,
                }
            try:
                authoritative_lane_id = int(
                    self._diagnostic_map_matching.get("ad_lane_id", 0) or 0
                )
                self.route_manager.sync_route_progress(
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    ego_heading_rad=float(ego_heading_rad),
                    current_lane_id=int(authoritative_lane_id),
                )
                # RouteManager owns the only runtime cursor.  The global
                # planner stores immutable topology and never advances a
                # second per-query nearest-node index for behavior.
                route_values = self.route_manager.get_route_info(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    query_key="authoritative_route_cursor",
                    fallback_lane_id=int(authoritative_lane_id or fallback_lane_id),
                    ego_waypoint=matched_waypoint,
                )
                summary = SimpleNamespace(**dict(route_values))
                summary.distance_to_destination_m = float(
                    route_values.get("remaining_distance_m", 0.0) or 0.0
                )
            except Exception as exc:
                return {
                    "route_found": False,
                    "optimal_lane_id": int(fallback_lane_id),
                    "current_road_option": "",
                    "next_macro_maneuver": "Continue Straight",
                    "next_macro_distance_m": float("inf"),
                    "remaining_distance_m": 0.0,
                    "debug_reason": f"admap_route_query_failed:{exc}",
                }
            ad_target_lane_id = int(
                getattr(summary, "optimal_lane_id", 0) or 0
            )
            local_target_lane_id = int(fallback_lane_id)
            # The continuity-aware matcher is the sole current-lane owner.
            # The local graph consumes that match to describe adjacency; it
            # must not independently replace the ego lane at an intersection.
            ad_current_lane_id = int(
                self._diagnostic_map_matching.get("ad_lane_id", 0) or 0
            )
            self._authoritative_ego_waypoint = matched_waypoint
            local_direction = ""
            local_offset = 0
            target_in_local_frame = False
            try:
                local_graph = self.global_planner.get_local_lane_graph(
                    float(ego_location.x),
                    float(ego_location.y),
                    z_m=float(getattr(ego_location, "z", 0.0)),
                    forward_distance_m=100.0,
                    backward_distance_m=100.0,
                    ego_waypoint=matched_waypoint,
                )
                # Freeze the actual AD-map centre geometry into this frame.
                # Downstream planning must not call the map again and derive a
                # different centreline from the same lane identity.
                lane_centerlines: dict[int, list[dict[str, float]]] = {}
                local_lane_ids = {
                    int(lane_id)
                    for lane_ids in dict(local_graph.get("corridors", {}) or {}).values()
                    for lane_id in list(lane_ids or [])
                }
                try:
                    raw_route_lane_sequence = (
                        self.global_planner.get_local_route_lane_sequence(
                            float(ego_location.x),
                            float(ego_location.y),
                            forward_distance_m=100.0,
                            backward_distance_m=100.0,
                        )
                    )
                except Exception:
                    raw_route_lane_sequence = []
                # Keep stored-route order. The corridor is a set-like lookup;
                # it must never be used to infer successor topology.
                local_graph["route_lane_sequence"] = [
                    int(lane_id)
                    for lane_id in list(raw_route_lane_sequence or [])
                    if int(lane_id) in local_lane_ids
                ]
                for lane_id in sorted(local_lane_ids):
                    try:
                        centerline_waypoints = self.global_planner.get_lane_centerline(
                            int(lane_id)
                        )
                    except Exception:
                        centerline_waypoints = []
                    samples: list[dict[str, float]] = []
                    for waypoint in list(centerline_waypoints or []):
                        position = dict(getattr(waypoint, "position", {}) or {})
                        try:
                            sample = {
                                "x_m": float(position["x"]),
                                "y_m": float(position["y"]),
                                "lane_width_m": max(
                                    0.1,
                                    float(
                                        getattr(waypoint, "lane_width_m", 3.5)
                                        or 3.5
                                    ),
                                ),
                            }
                            left_boundary = getattr(
                                waypoint, "left_boundary_position", None
                            )
                            right_boundary = getattr(
                                waypoint, "right_boundary_position", None
                            )
                            if isinstance(left_boundary, Mapping) and isinstance(
                                right_boundary, Mapping
                            ):
                                sample.update({
                                    "left_boundary_x_m": float(left_boundary["x"]),
                                    "left_boundary_y_m": float(left_boundary["y"]),
                                    "right_boundary_x_m": float(right_boundary["x"]),
                                    "right_boundary_y_m": float(right_boundary["y"]),
                                })
                            samples.append(sample)
                        except (KeyError, TypeError, ValueError):
                            continue
                    if len(samples) >= 2:
                        lane_centerlines[int(lane_id)] = samples
                local_graph["lane_centerlines"] = lane_centerlines
                self._diagnostic_local_lane_frame = dict(local_graph)
                if int(ad_current_lane_id) == 0:
                    # Compatibility/degraded-mode fallback only. In normal
                    # AD-map operation the continuity matcher above is valid
                    # and remains authoritative.
                    ad_current_lane_id = int(
                        local_graph.get("ego_ad_lane_id", 0) or 0
                    )
                lane_to_offset = dict(local_graph.get("lane_to_offset", {}) or {})
                target_in_local_frame = int(ad_target_lane_id) in {
                    int(lane_id) for lane_id in lane_to_offset
                }
                offset = int(lane_to_offset.get(int(ad_target_lane_id), 0))
                local_offset = int(offset)
                local_direction = "left" if offset > 0 else "right" if offset < 0 else ""
                if bool(target_in_local_frame) and int(ad_target_lane_id) != 0:
                    local_target_lane_id = int(ad_target_lane_id)
            except Exception:
                # Route information remains usable even if this tick's
                # topology-to-local-lane projection cannot be resolved.
                local_target_lane_id = int(fallback_lane_id)
            violations = local_lane_frame_invariants(
                matched_lane_id=int(
                    self._diagnostic_map_matching.get("ad_lane_id", 0) or 0
                ),
                corridors={
                    int(key): list(value or [])
                    for key, value in dict(
                        self._diagnostic_local_lane_frame.get("corridors", {}) or {}
                    ).items()
                },
                target_lane_id=int(ad_target_lane_id),
                reported_target_offset=int(local_offset),
            )
            self._diagnostic_local_lane_frame["invariant_violations"] = list(
                violations
            )
            self._diagnostic_local_lane_frame["route_target_offset"] = int(
                local_offset
            )
            self._diagnostic_local_lane_frame["route_target_ad_lane_id"] = int(
                ad_target_lane_id
            )
            self._diagnostic_local_lane_frame["route_target_in_frame"] = bool(
                target_in_local_frame
            )
            self._local_map_frame_id = int(
                getattr(self, "_local_map_frame_id", 0)
            ) + 1
            self._local_map_snapshot = build_local_map_snapshot(
                frame_id=int(self._local_map_frame_id),
                timestamp_s=float(self._sim_time_s()),
                route_revision=str(
                    getattr(self.route_manager, "route_revision", "") or ""
                ),
                match=self._diagnostic_map_matching,
                local_graph=self._diagnostic_local_lane_frame,
                route_target_lane_id=int(ad_target_lane_id),
                invariant_violations=violations,
            )
            # Compatibility mirror only. New planning consumers must read the
            # immutable snapshot, not mutate this dictionary.
            self._diagnostic_local_lane_frame = (
                self._local_map_snapshot.as_legacy_dict()
            )
            ad_current_lane_id = int(self._local_map_snapshot.ego_lane_id)
            target_in_local_frame = bool(
                self._local_map_snapshot.route_target_in_frame
            )
            local_offset = int(self._local_map_snapshot.route_target_offset)
            local_direction = (
                "left" if local_offset > 0 else "right" if local_offset < 0 else ""
            )
            if target_in_local_frame and int(ad_target_lane_id) != 0:
                local_target_lane_id = int(ad_target_lane_id)
            next_macro_maneuver = str(
                getattr(summary, "next_macro_maneuver", "Continue Straight")
            )
            normalized_macro = (
                next_macro_maneuver.strip().lower().replace("-", "_").replace(" ", "_")
            )
            if (
                normalized_macro in {"lane_change_left", "lane_change_right"}
                and int(ad_current_lane_id) != 0
                and int(ad_current_lane_id) == int(ad_target_lane_id)
                and int(local_offset) == 0
            ):
                # The route backend can keep reporting the consumed edge for
                # a few progress samples. Expose completion immediately so
                # behavior and ManeuverManager do not restart/retain it.
                next_macro_maneuver = "Lane Follow"
            route_result = {
                "route_found": bool(getattr(summary, "route_found", False)),
                "optimal_lane_id": int(local_target_lane_id),
                "authoritative_current_lane_id": int(ad_current_lane_id),
                "ad_current_lane_id": int(ad_current_lane_id),
                "ad_target_lane_id": int(ad_target_lane_id),
                "lane_change_direction": str(local_direction),
                "lane_change_offset": int(local_offset),
                "target_in_local_frame": bool(target_in_local_frame),
                "diagnostic_map_matching": dict(self._diagnostic_map_matching),
                "diagnostic_local_lane_frame": dict(
                    self._diagnostic_local_lane_frame
                ),
                "current_road_option": str(getattr(summary, "current_road_option", "")),
                "next_macro_maneuver": str(next_macro_maneuver),
                "next_macro_distance_m": float(
                    getattr(summary, "next_macro_distance_m", float("inf"))
                ),
                "remaining_distance_m": float(
                    getattr(summary, "distance_to_destination_m", 0.0) or 0.0
                ),
                "debug_reason": "admap_topology_geometry_active",
            }
            # AD-map owns route identity, maneuver semantics, progress, and
            # destination completion in this backend.  Publish the exact
            # per-tick query consumed by behavior so RouteManagerStatus/CSV
            # cannot remain frozen at the last AD-map route rebuild.  CARLA's
            # independently synchronized index is geometry-only.
            route_manager = getattr(self, "route_manager", None)
            publish = getattr(
                route_manager,
                "accept_authoritative_route_summary",
                None,
            )
            if callable(publish):
                publish(
                    summary,
                    debug_reason="admap_authoritative_route_active",
                )
            return route_result
        if not hasattr(self, "route_manager"):
            return {
                "route_found": False,
                "optimal_lane_id": int(fallback_lane_id),
                "current_road_option": "",
                "next_macro_maneuver": "Continue Straight",
                "debug_reason": "route_manager_unavailable",
            }
        return self.route_manager.get_route_info(
            x_m=float(ego_location.x),
            y_m=float(ego_location.y),
            query_key=f"vehicle_{int(getattr(self.vehicle_manager.vehicle, 'id', 0))}",
            fallback_lane_id=int(fallback_lane_id),
            ego_waypoint=ego_waypoint,
        )

    def _apply_mpc_cost_profile(
        self,
        *,
        behavior: str,
        planner_lc_state: str,
        planner_mode: str,
        next_macro_maneuver: str,
        sim_time_s: float,
        nearest_obstacle_distance_m: Optional[float] = None,
        ego_speed_mps: float = 0.0,
    ) -> None:
        requested = mpc_cost_profile_for_behavior(
            behavior=behavior,
            planner_lc_state=planner_lc_state,
            planner_mode=planner_mode,
            next_macro_maneuver=next_macro_maneuver,
        )
        (
            self.active_mpc_cost_profile,
            self.mpc_cost_profile_active_since_s,
            self.mpc_cost_profile_switch_reason,
        ) = _select_mpc_cost_profile_with_hysteresis(
            requested_profile=str(requested),
            active_profile=str(self.active_mpc_cost_profile),
            sim_time_s=float(sim_time_s),
            active_since_s=float(self.mpc_cost_profile_active_since_s),
            min_hold_s=float(self.behavior_runtime_cfg.get("mpc_cost_profile_min_hold_s", 1.5)),
        )
        self.requested_mpc_cost_profile = str(requested)
        if hasattr(self.mpc, "apply_mode_cost_profile"):
            self.active_mpc_cost_profile = str(
                self.mpc.apply_mode_cost_profile(str(self.active_mpc_cost_profile))
            )
        if bool(
            getattr(self.mpc, "adaptive_horizon_enabled", False)
        ) and hasattr(self.mpc, "blend_toward_horizon_s"):
            # adaptive_horizon_enabled/min_s/max_s live on the MPC instance
            # (parsed from MPC/mpc.yaml, the same file horizon_s itself comes
            # from) -- self.config here is the bridge/scenario config, a
            # separate namespace that was never going to have that key.
            profile_horizon_s = (
                dict(self.config.get("adaptive_horizon_profile_s", {}))
                or _DEFAULT_ADAPTIVE_HORIZON_PROFILE_S
            )
            self.mpc.blend_toward_horizon_s(
                _adaptive_target_horizon_s(
                    mpc_cost_profile=str(self.active_mpc_cost_profile),
                    nearest_obstacle_distance_m=nearest_obstacle_distance_m,
                    ego_speed_mps=float(ego_speed_mps),
                    profile_horizon_s=profile_horizon_s,
                    obstacle_reference_speed_mps=float(
                        self.config.get(
                            "adaptive_horizon_obstacle_reference_speed_mps", 2.0
                        )
                    ),
                    obstacle_comfortable_decel_mps2=float(
                        self.config.get(
                            "adaptive_horizon_obstacle_comfortable_decel_mps2", 2.0
                        )
                    ),
                )
            )

    def _load_mpc_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg_path = self.config.get("mpc_config_path")
        if not cfg_path:
            cfg_path = Path(__file__).resolve().parents[1] / "MPC" / "mpc.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        mpc_cfg = dict(payload.get("mpc", payload))
        road_cfg = dict(payload.get("road", {}))
        road_cfg.setdefault("lane_count", int(self.config.get("lane_count", 3)))
        road_cfg.setdefault("lane_width_m", float(self.config.get("lane_width_m", 3.5)))
        return mpc_cfg, road_cfg

    def _clean_functional_test_dynamic_actors_once(self) -> None:
        """Remove non-ego vehicles for an explicitly isolated functional run.

        An interrupted CARLA scenario can leave actors in the shared world.
        An empty traffic-manager list prevents new spawns but cannot remove
        those stale actors, so planner-side filtering alone still permits a
        physical collision.  This destructive cleanup is gated by the
        scenario-only ``functional_test_ignore_dynamic_objects`` switch and
        never runs in normal planning or traffic tests.
        """
        if (
            not bool(self.functional_test_ignore_dynamic_objects)
            or bool(self._functional_test_world_actors_cleaned)
        ):
            return
        self._functional_test_world_actors_cleaned = True
        ego_vehicle = getattr(self.vehicle_manager, "vehicle", None)
        ego_id = getattr(ego_vehicle, "id", None)
        try:
            world = ego_vehicle.get_world()
            actors = list(world.get_actors().filter("vehicle.*"))
        except Exception:
            return
        removed = 0
        for actor in actors:
            if ego_id is not None and getattr(actor, "id", None) == ego_id:
                continue
            try:
                actor.destroy()
                removed += 1
            except Exception:
                continue
        if self.debug:
            print(
                "[CP-X OpenCDA Bridge] Functional isolation removed "
                f"{int(removed)} non-ego vehicle actor(s)."
            )

    def _perception_diagnostics(self) -> dict[str, object]:
        manager = getattr(self.vehicle_manager, "perception_manager", None)
        activated = bool(getattr(manager, "activate", False))
        ml_active = bool(activated and getattr(manager, "ml_manager", None) is not None)
        return {
            "perception_mode": (
                "opencda_ml_yolov5_lidar_fusion"
                if ml_active
                else "carla_ground_truth"
            ),
            "perception_ml_active": bool(ml_active),
            "perception_camera_count": int(
                getattr(manager, "camera_num", 0) or 0
            ),
        }

    def _interaction_obstacle_snapshots(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        predicted_objects: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Attach the prediction module's future to each non-connected road
        user before Stage A/C.

        Fusion priority: a connected vehicle with a fresh broadcast plan is
        handled by ``_collect_cav_intents`` (its shared trajectory wins);
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

    def _mpc_object_snapshots_with_prediction(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        prediction_trajectories: Mapping[str, Sequence[Mapping[str, object]]],
    ) -> list[dict[str, Any]]:
        """Attach this tick's predicted trajectory to each MPC obstacle.

        Without this, ``MPC._get_object_state_at_stage`` (MPC/mpc.py) never
        sees ``planner_input_frame.prediction.obstacle_future_trajectories``
        at all -- it only recognizes a ``predicted_trajectory`` already
        shaped as one ``[x, y, v, psi]`` entry per stage, so it silently
        falls back to its own constant-velocity extrapolation for every
        obstacle, independent of (and less accurate than) the
        constant-acceleration/CP-supplied prediction the rest of the
        pipeline already computed. ``bridge.tracker.predict()`` is already
        called with ``horizon_s=self.mpc.horizon_s, dt_s=self.mpc.dt_s`` (see
        planner_input_adapter.py), so each trajectory here already has one
        point per MPC stage -- this only needs to convert the shape and
        attach it, not resample it.
        """

        from opencda.planning_module.pipeline.prediction import (
            mpc_stage_trajectory,
            obstacle_track_id,
        )

        if not prediction_trajectories:
            return [dict(snapshot) for snapshot in list(object_snapshots or [])]
        horizon_steps = int(self.mpc.horizon_steps)
        dt_s = float(self.mpc.dt_s)
        annotated: list[dict[str, Any]] = []
        for snapshot in list(object_snapshots or []):
            if not isinstance(snapshot, Mapping):
                continue
            updated = dict(snapshot)
            points = prediction_trajectories.get(obstacle_track_id(snapshot))
            if points:
                updated["predicted_trajectory"] = mpc_stage_trajectory(
                    list(points),
                    fallback_heading_rad=float(snapshot.get("psi", snapshot.get("heading_rad", 0.0))),
                    horizon_steps=horizon_steps,
                    dt_s=dt_s,
                )
            annotated.append(updated)
        return annotated


    @staticmethod
    def _cooperative_actor_evidence(
        *,
        cp_summary: Mapping[str, Any],
        prediction_trajectories: Mapping[
            str, Sequence[Mapping[str, Any]]
        ],
        selected_reference: Sequence[Mapping[str, Any]],
        candidate_proximity_m: float = 3.0,
    ) -> dict[str, Any]:
        """Build an auditable CP-to-prediction-to-candidate evidence chain.

        ``candidate_relevant`` means that a predicted actor position enters
        the selected reference's spatial safety envelope at a corresponding
        horizon step. It deliberately does not claim that the actor changed
        the selected decision; proving that stronger counterfactual requires
        evaluating the same candidate set with that actor removed.
        """

        provenance = [
            dict(item)
            for item in list(cp_summary.get("actor_provenance", []) or [])
            if isinstance(item, Mapping)
        ]
        prediction_by_actor = {
            str(key).rsplit(":", 1)[-1]: list(points or [])
            for key, points in dict(prediction_trajectories or {}).items()
        }
        reference = list(selected_reference or [])
        prediction_used_ids: list[str] = []
        prediction_used_pedestrian_ids: list[str] = []
        candidate_relevant_ids: list[str] = []
        candidate_relevant_pedestrian_ids: list[str] = []
        evidence: list[dict[str, Any]] = []

        for actor in provenance:
            message_id = str(actor.get("actor_id", ""))
            actor_id = message_id.rsplit(":", 1)[-1]
            actor_type = str(actor.get("actor_type", "unknown"))
            predicted_points = prediction_by_actor.get(actor_id, [])
            used_by_prediction = bool(predicted_points)
            min_distance_m: float | None = None
            if used_by_prediction and reference:
                for index in range(min(len(reference), len(predicted_points))):
                    try:
                        ref = reference[index]
                        point = predicted_points[index]
                        rx = float(ref.get("x_ref_m", ref.get("x", 0.0)))
                        ry = float(ref.get("y_ref_m", ref.get("y", 0.0)))
                        px = float(point.get("x", point.get("x_m", 0.0)))
                        py = float(point.get("y", point.get("y_m", 0.0)))
                    except (TypeError, ValueError):
                        continue
                    distance_m = math.hypot(rx - px, ry - py)
                    min_distance_m = (
                        distance_m
                        if min_distance_m is None
                        else min(min_distance_m, distance_m)
                    )
            candidate_relevant = bool(
                min_distance_m is not None
                and min_distance_m <= float(candidate_proximity_m)
            )
            if used_by_prediction:
                prediction_used_ids.append(message_id)
                if actor_type == "pedestrian":
                    prediction_used_pedestrian_ids.append(message_id)
            if candidate_relevant:
                candidate_relevant_ids.append(message_id)
                if actor_type == "pedestrian":
                    candidate_relevant_pedestrian_ids.append(message_id)
            evidence.append({
                **actor,
                "used_by_prediction": bool(used_by_prediction),
                "candidate_relevant": bool(candidate_relevant),
                "candidate_min_predicted_distance_m": (
                    None if min_distance_m is None else float(min_distance_m)
                ),
            })

        pedestrian_count = sum(
            str(item.get("actor_type", "")) == "pedestrian"
            for item in provenance
        )
        blind_pedestrian_count = sum(
            str(item.get("actor_type", "")) == "pedestrian"
            and bool(item.get("blind_spot_shared", False))
            for item in provenance
        )
        return {
            "cp_actor_provenance": json.dumps(provenance, default=str),
            "cp_pedestrian_count": int(pedestrian_count),
            "cp_blind_spot_pedestrian_count": int(blind_pedestrian_count),
            "cp_prediction_used_actor_ids": ",".join(prediction_used_ids),
            "cp_prediction_used_pedestrian_ids": ",".join(
                prediction_used_pedestrian_ids
            ),
            "cp_candidate_relevant_actor_ids": ",".join(
                candidate_relevant_ids
            ),
            "cp_candidate_relevant_pedestrian_ids": ",".join(
                candidate_relevant_pedestrian_ids
            ),
            "cp_actor_evidence": json.dumps(evidence, default=str),
        }

    def _world_debug(self):
        port = getattr(self, "world_debug_port", None)
        if port is not None:
            return port
        from opencda.planning_module.opencda_bridge.platform_ports import WorldDebugPort
        port = WorldDebugPort(
            vehicle_manager=getattr(self, "vehicle_manager", None),
            carla_module=getattr(self, "carla", None),
            mpc=getattr(self, "mpc", None),
            route_points=self._active_global_route_points,
            enabled=bool(getattr(self, "draw_world_debug", False)),
            draw_destination=bool(getattr(self, "draw_world_debug_destination", False)),
            life_time_s=float(getattr(self, "world_debug_life_time_s", 0.15)),
            report_error=lambda reason: print(
                "[CP-X OpenCDA Bridge] world debug draw failed: " + reason
            ) if bool(getattr(self, "debug", False)) else None,
        )
        self.world_debug_port = port
        return port

    def _draw_world_debug_primitives(
        self, *, destination_state: Sequence[float],
        lane_center_reference: Sequence[Mapping[str, Any]],
    ) -> None:
        self._world_debug().draw(
            destination_state=destination_state,
            reference_samples=lane_center_reference,
        )

    def _last_mpc_trajectory_points(self) -> list[tuple[float, float]]:
        return self._world_debug().mpc_trajectory_points()
    def _active_global_route_points(self) -> list[list[float]]:
        """Return planner-owned route topology geometry; never display-filter it."""

        latest_update = dict(getattr(self, "_latest_opencda_update", {}) or {})
        ego_transform = latest_update.get("ego_transform")
        if ego_transform is not None:
            loc = ego_transform.location
            return self.route_manager.geometry_route_points(
                x_m=float(loc.x),
                y_m=float(loc.y),
                query_key=f"vehicle_{int(getattr(self.vehicle_manager.vehicle, 'id', 0))}_polyline",
            )
        route_points = self.route_manager.geometry_route_points()
        return [list(point) for point in route_points]

    def _display_global_route_points(self) -> list[list[float]]:
        """Return the diagnostics-only smoothed route polyline."""

        return self._world_debug().display_route_points()
    def _map_waypoint_from_location(self, location: carla.Location):
        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        port = getattr(self, "map_lookup_port", None) or MapLookupPort.from_owner(self)
        return port.waypoint(location)

    def _drivable_waypoint_from_location(
        self,
        location: carla.Location,
    ):
        """Return a waypoint only when the point is on a driving lane."""

        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        port = getattr(self, "map_lookup_port", None) or MapLookupPort.from_owner(self)
        return port.drivable_waypoint(location) or port.waypoint(location)

    def _lane_id_at_location(self, location: carla.Location) -> int:
        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        port = getattr(self, "map_lookup_port", None) or MapLookupPort.from_owner(self)
        return port.lane_id(location)

    @staticmethod
    def _location_to_point(location: Any) -> dict[str, float]:
        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        return MapLookupPort.location_to_point(location)

    @staticmethod
    def _body_frame_xy(
        *,
        origin_x_m: float,
        origin_y_m: float,
        heading_rad: float,
        target_x_m: float,
        target_y_m: float,
    ) -> tuple[float, float]:
        from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
        return MapLookupPort.body_frame_xy(
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
            heading_rad=heading_rad,
            target_x_m=target_x_m,
            target_y_m=target_y_m,
        )

    def _set_actuator_context(
        self,
        *,
        ego_speed_mps: float,
        target_speed_mps: float,
        stop_goal_active: bool,
    ) -> None:
        self.actuator_port.set_context(
            ego_speed_mps=ego_speed_mps,
            target_speed_mps=target_speed_mps,
            stop_goal_active=stop_goal_active,
        )

    def _control_from_mpc(self, acceleration_mps2: float, steering_angle_rad: float) -> carla.VehicleControl:
        return self.actuator_port.control(acceleration_mps2, steering_angle_rad)

    def _accel_from_control(self, control: carla.VehicleControl) -> float:
        return self.actuator_port.acceleration(control)

    def _steer_rad_from_control(self, control: carla.VehicleControl) -> float:
        return self.actuator_port.steering(control)

    def _emergency_stop_control(self) -> carla.VehicleControl:
        self._last_accel_mps2 = float(getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0))
        self._last_steer_rad = 0.0
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

def _route_destination_stop_gate(
    *,
    route_found: bool,
    remaining_distance_m: float,
    ego_speed_mps: float,
    deceleration_mps2: float,
    buffer_m: float,
) -> tuple[bool, float]:
    """Return a physics-based destination approach stop decision."""

    deceleration = max(0.5, float(deceleration_mps2))
    required_distance_m = (
        max(0.0, float(ego_speed_mps)) ** 2 / (2.0 * deceleration)
        + max(0.0, float(buffer_m))
    )
    active = bool(
        route_found
        and math.isfinite(float(remaining_distance_m))
        and float(remaining_distance_m) >= 0.0
        and float(remaining_distance_m) <= float(required_distance_m)
    )
    return active, float(required_distance_m)


def _destination_approach_speed_cap(
    *,
    remaining_distance_m: float,
    deceleration_mps2: float,
    buffer_m: float,
) -> float:
    """Return the continuous speed cap that stops at the route buffer.

    This is the inverse of the constant-deceleration stopping-distance
    equation.  It deliberately owns only longitudinal planning; reference
    geometry remains owned by the persistent reference-line provider.
    """

    if not math.isfinite(float(remaining_distance_m)):
        return float("inf")
    usable_distance_m = max(
        0.0,
        float(remaining_distance_m) - max(0.0, float(buffer_m)),
    )
    deceleration = max(0.5, float(deceleration_mps2))
    return math.sqrt(2.0 * deceleration * usable_distance_m)


def _hard_gate_requires_emergency_stop(
    *,
    fallback_reason: str,
    behavior_decision: str,
    stop_goal_active: bool,
) -> bool:
    """Reserve full braking for hard gates that represent a stop hazard.

    A geometry/continuity contract veto means MPC must not consume that
    reference, but it is not evidence of an imminent collision.  Those
    failures use the bounded tracking fallback and remain subject to the
    downstream safety supervisor.  Collision, explicit stop, and emergency
    behavior retain deterministic full braking.
    """

    reason = str(fallback_reason or "").strip().lower()
    decision = str(behavior_decision or "").strip().lower()
    if not reason.startswith("candidate_hard_gate:"):
        return False
    if bool(stop_goal_active) or decision in {
        "emergency_brake",
        "stop_at_intersection",
        "stop_sign",
    }:
        return True
    hazard_tokens = (
        "collision_risk",
        "emergency_brake_direct_control",
        "stop_missing_target_hard_lock",
    )
    return any(token in reason for token in hazard_tokens)


_DEFAULT_ADAPTIVE_HORIZON_PROFILE_S: dict[str, float] = {
    "lane_follow": 3.0,
    "prepare_lane_change": 4.5,
    "execute_lane_change": 4.5,
    "intersection_turn": 2.2,
    "stop": 2.0,
    "recovery": 1.5,
}


def _adaptive_target_horizon_s(
    *,
    mpc_cost_profile: str,
    nearest_obstacle_distance_m: Optional[float],
    ego_speed_mps: float,
    profile_horizon_s: Mapping[str, float],
    obstacle_reference_speed_mps: float = 2.0,
    obstacle_comfortable_decel_mps2: float = 2.0,
) -> float:
    """Pick a prediction-horizon target from the active behavior mode, then
    shorten it further if a nearby obstacle needs quicker reaction -- a
    human driver looks less far ahead through a tight turn than down an open
    lane, and less still when something close needs immediate attention.

    The obstacle term is the MAX of two independent estimates, not a single
    distance/current_speed ratio: that ratio blows up as ego comfortably
    decelerates toward a stop behind a closing lead vehicle (the exact
    "shouldn't horizon keep shrinking here?" case this was built for) --
    dividing by ego's own shrinking speed makes the estimate grow right when
    it should keep shrinking. distance_reaction_s (distance over a fixed
    reference speed, not ego's live one) shrinks monotonically as the gap
    closes regardless of ego's speed; stopping_time_s (ego's own speed over a
    comfortable deceleration) shrinks to 0 as ego actually comes to a stop.
    Taking the max avoids either term alone causing a premature shrink (e.g.
    ego already slow with an unrelated, still-distant obstacle ahead).
    """

    base = float(
        profile_horizon_s.get(
            str(mpc_cost_profile),
            profile_horizon_s.get("lane_follow", 3.0),
        )
    )
    if nearest_obstacle_distance_m is not None and math.isfinite(
        float(nearest_obstacle_distance_m)
    ):
        distance_reaction_s = float(nearest_obstacle_distance_m) / max(
            1.0e-3, float(obstacle_reference_speed_mps)
        )
        stopping_time_s = float(ego_speed_mps) / max(
            1.0e-3, float(obstacle_comfortable_decel_mps2)
        )
        reaction_s = max(distance_reaction_s, stopping_time_s)
        base = min(base, max(1.0, reaction_s))
    return float(base)


def _select_mpc_cost_profile_with_hysteresis(
    *,
    requested_profile: str,
    active_profile: str,
    sim_time_s: float,
    active_since_s: float,
    min_hold_s: float,
) -> tuple[str, float, str]:
    requested = str(requested_profile or "lane_follow").strip() or "lane_follow"
    active = str(active_profile or "lane_follow").strip() or "lane_follow"
    elapsed_s = max(0.0, float(sim_time_s) - float(active_since_s))
    min_hold_s = max(0.0, float(min_hold_s))
    if requested == active:
        return active, float(active_since_s), "same_profile"
    if requested in {"stop", "recovery"}:
        return requested, float(sim_time_s), "safety_preempt"
    if active in {"stop", "recovery"} and elapsed_s < min_hold_s:
        return active, float(active_since_s), "hold_safety_profile"
    if elapsed_s < min_hold_s:
        return active, float(active_since_s), "min_hold"
    return requested, float(sim_time_s), "switch"


def cpx_planner_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether a vehicle config requests the CP-X planner bridge."""

    planner_cfg = dict(config.get("planner", {}) or {})
    if planner_cfg and not bool(planner_cfg.get("enabled", True)):
        return False
    planner_type = str(planner_cfg.get("type", "")).strip().lower()
    env_type = str(os.environ.get("OPENCDA_PLANNER", "")).strip().lower()
    if planner_type:
        return planner_type in {"cpx_mpc", "cp_x_mpc"}
    return env_type in {"cpx_mpc", "cp_x_mpc"}
