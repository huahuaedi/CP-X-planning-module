"""Bridge from native OpenCDA vehicle managers to the CP-X MPC planner.

The bridge is intentionally small: OpenCDA still owns simulation, localization,
perception, and V2X discovery. This class consumes a custom map planner and
returns a CARLA ``VehicleControl`` directly, replacing both
OpenCDA's behavior agent and PID controller when enabled.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import re
import sys
import traceback
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
    ReferenceLineProvider,
    ReferenceLineRequest,
)
from opencda.planning_module.pipeline.maneuver_manager import ManeuverManager
from opencda.planning_module.pipeline.nominal_trajectory import (
    NominalTrajectoryGenerator,
)
from opencda.planning_module.pipeline.fallback_manager import (
    FailureReason,
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
from opencda.planning_module.pipeline.behavior_stage import (
    BehaviorCandidateRequest,
    BehaviorOverrideRequest,
    BehaviorStage,
    OpportunisticLaneChangeRequest,
    RouteLaneChangeRequest,
)
from opencda.planning_module.utility.speed_profile import (
    curvature_speed_cap_mps,
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


def _static_obstacle_cooldown_policy(
    *,
    failed_latched: bool,
    route_transition_pending: bool,
) -> tuple[str, bool]:
    """Return debug status and stop ownership during a replan cooldown."""

    if bool(failed_latched):
        return "cooldown_stop", True
    if bool(route_transition_pending):
        return "cooldown_route_transition", False
    return "cooldown_stop", True


def _lane_change_execution_active(
        *, reference_locked: bool, phase: object) -> bool:
    """Return whether a committed lane change still owns route execution.

    The route progress tracker is allowed to observe a temporarily lapsed lane
    change requirement while ego follows the locked lateral trajectory.  That
    lapse must not be interpreted as a missed maneuver until the trajectory is
    released.  The phase check also protects the stabilization hand-off, where
    the semantic route instruction may already have advanced.
    """
    normalized_phase = str(phase or "").strip().lower()
    return bool(reference_locked) or normalized_phase in {
        "executing",
        "target_lane_stabilization",
    }


def _select_static_obstacle_local_avoidance_lane(
    *,
    current_lane_id: int,
    available_lane_ids: Sequence[int],
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Mapping[int, Mapping[str, object]],
    minimum_safety_score: float,
) -> int | None:
    """Select one adjacent, prediction-safe lane for local obstacle bypass.

    This helper deliberately does not alter the global route or the map.  It
    only authorizes the existing behavior/reference candidate pipeline to
    evaluate a local lane-borrow trajectory.  The downstream FSM, reference
    contract, MPC probe and safety supervisor retain veto authority.
    """

    current = int(current_lane_id)
    alternatives = sorted(
        {
            int(lane_id)
            for lane_id in list(available_lane_ids or [])
            if int(lane_id) != 0 and int(lane_id) != current
        },
        key=lambda lane_id: abs(int(lane_id) - current),
    )
    if not alternatives:
        return None

    nearest_delta = abs(int(alternatives[0]) - current)
    adjacent = [
        int(lane_id)
        for lane_id in alternatives
        if abs(int(lane_id) - current) == int(nearest_delta)
    ]
    safe = []
    for lane_id in adjacent:
        score = float(lane_safety_scores.get(int(lane_id), 0.0))
        risk = dict(lane_prediction_risks.get(int(lane_id), {}) or {})
        if score <= float(minimum_safety_score) or bool(risk.get("risk", False)):
            continue
        safe.append((float(score), int(lane_id)))
    if not safe:
        return None
    safe.sort(key=lambda row: (-float(row[0]), abs(int(row[1]) - current), -int(row[1])))
    return int(safe[0][1])


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
        self._actuator_ego_speed_mps = 0.0
        self._actuator_target_speed_mps = 0.0
        self._actuator_stop_goal_active = False
        self._warned = False
        self._diagnostic_hd_map_matcher = DiagnosticHDMapMatcher()
        self._diagnostic_local_lane_frame: dict[str, object] = {}
        self._diagnostic_map_matching: dict[str, object] = {}
        self._local_map_frame_id = 0
        self._local_map_snapshot = LocalMapSnapshot()
        self._stable_reference_line_provider = ReferenceLineProvider()
        self.maneuver_manager = ManeuverManager(self.config)
        self.nominal_trajectory_generator = NominalTrajectoryGenerator()
        self.speed_target_planner = SpeedTargetPlanner()
        self.behavior_stage = BehaviorStage()
        self.destination_speed_stage = DestinationSpeedStage(
            config=self.config,
            speed_planner=self.speed_target_planner,
        )
        self._authoritative_ego_waypoint: Any = None
        self._stop_release_temp_smooth_until_sim_time_s = 0.0
        self._full_latched_stop_target: dict[str, object] | None = None
        self._full_latched_stop_state = "unknown"
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

        self._scenario_manager = CPXScenarioManager(self.config)
        self._boundary_recovery_request = BoundaryRecoveryRequest()
        self._boundary_recovery_trigger_frames = 0
        self._boundary_recovery_infeasible_frames = 0
        self._boundary_recovery_cooldown_until_s = -float("inf")
        self._full_last_behavior_mode_key = ""
        self._trajectory_fallback_manager = TrajectoryFallbackManager(
            max_hold_age_s=float(self.config.get("fallback_hold_last_valid_s", 0.35)),
            min_hold_arc_m=float(self.config.get("fallback_min_valid_arc_m", 2.0)),
            safe_stop_deceleration_mps2=float(
                self.config.get("fallback_safe_stop_deceleration_mps2", 2.0)
            ),
        )
        self._route_replan_last_attempt_s = -float("inf")
        self._route_replan_attempt_count = 0
        self._route_replan_last_reason = "route_replan_not_requested"
        self._static_obstacle_replan_last_attempt_s = -float("inf")
        self._static_obstacle_replan_failed_latched = False
        self._static_obstacle_replan_status = "idle"
        self._static_obstacle_candidate_id = ""
        self._static_obstacle_candidate_since_s = -float("inf")
        self._static_obstacle_route_transition_pending = False
        self._static_obstacle_blocked_lane_id: object = ""
        self._static_obstacle_replan_reason = "not_requested"
        self._static_obstacle_local_target_lane_id: Optional[int] = None
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
        self.full_lane_follow_max_destination_lateral_m = max(
            0.0,
            float(self.config.get("full_lane_follow_max_destination_lateral_m", 1.2)),
        )
        self.full_lane_follow_max_reference_first_lateral_m = max(
            0.0,
            float(self.config.get("full_lane_follow_max_reference_first_lateral_m", 0.65)),
        )
        self.full_stop_max_destination_lateral_m = max(
            0.0,
            float(self.config.get("full_stop_max_destination_lateral_m", 1.0)),
        )
        self.full_stop_max_reference_first_lateral_m = max(
            0.0,
            float(self.config.get("full_stop_max_reference_first_lateral_m", 0.55)),
        )
        # The lateral-only guard below lets a persistent physical heading
        # bias go uncorrected for many ticks: each tick's lateral offset is
        # individually small enough to stay under the lateral thresholds
        # above, but a several-degree heading error against the true lane
        # tangent (from compute_ego_lane_offset) integrates into lateral
        # drift at low speed (v*sin(heading_error)) over a few seconds,
        # eventually crossing the lateral threshold anyway -- just late,
        # after the vehicle has drifted toward an adjacent lane and often
        # after the maneuver window (e.g. an approaching intersection stop)
        # has already closed. Checking heading directly forces the same
        # already-working true-waypoint rebuild before that drift compounds.
        self.full_lane_follow_max_heading_error_deg = max(
            0.0,
            float(self.config.get("full_lane_follow_max_heading_error_deg", 4.0)),
        )
        self.full_stop_max_heading_error_deg = max(
            0.0,
            float(self.config.get("full_stop_max_heading_error_deg", 4.0)),
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
        self._candidate_risk_bucket_state: dict[str, str] = {}
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
        self._candidate_mpc_probe_last_time_s = -float("inf")
        self._candidate_mpc_probe_cache: dict[tuple[object, ...], dict[str, object]] = {}
        self.candidate_lane_change_assertive_duration_s = max(
            0.1,
            float(self.config.get("candidate_lane_change_assertive_duration_s", 3.2)),
        )
        self.candidate_lane_change_normal_duration_s = max(
            0.1,
            float(self.config.get("candidate_lane_change_normal_duration_s", 4.0)),
        )
        self.candidate_lane_change_conservative_duration_s = max(
            0.1,
            float(self.config.get("candidate_lane_change_conservative_duration_s", 5.5)),
        )
        self.candidate_lane_change_assertive_speed_scale = max(
            0.1,
            float(self.config.get("candidate_lane_change_assertive_speed_scale", 1.0)),
        )
        self.candidate_lane_change_normal_speed_scale = max(
            0.1,
            float(self.config.get("candidate_lane_change_normal_speed_scale", 0.9)),
        )
        self.candidate_lane_change_conservative_speed_scale = max(
            0.1,
            float(self.config.get("candidate_lane_change_conservative_speed_scale", 0.7)),
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
        from opencda.planning_module.pipeline.decision_record import build_decision_record
        from opencda.planning_module.pipeline.mpc_feedback import BehaviorMPCFeedback
        from opencda.planning_module.pipeline.mpc_command_extractor import (
            MPCCommandExtractor,
        )
        from opencda.planning_module.pipeline.planner_pipeline import CPXPlanningPipeline
        from opencda.planning_module.pipeline.route_manager import CPXRouteManager
        from opencda.planning_module.pipeline.reference_gate import FinalReferenceGate
        from opencda.planning_module.pipeline.reference_generator import ReferenceGenerator
        from opencda.planning_module.pipeline.reference_pipeline import (
            ReferencePipeline,
        )
        from opencda.planning_module.pipeline.safety_supervisor import SafetySupervisor
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
        from opencda.planning_module.pipeline.actuator_mapper import CarlaActuatorMapper
        self.actuator_mapper = CarlaActuatorMapper(self.config)
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
        self.input_adapter = OpenCDAPlanningAdapter(self)
        self.tracker = CPXObstacleTracker(
            max_stale_s=float(self.config.get("tracker_max_stale_s", 0.5)),
            max_speed_mps=float(self.config.get("tracker_max_speed_mps", 45.0)),
            max_acceleration_mps2=float(
                self.config.get("tracker_max_acceleration_mps2", 12.0)
            ),
            max_position_jump_m=float(self.config.get("tracker_max_position_jump_m", 12.0)),
        )
        self.planning_pipeline = CPXPlanningPipeline(self)
        self.final_reference_gate = FinalReferenceGate(self.config)
        self.reference_pipeline = ReferencePipeline(
            config=self.config,
            generator=self.reference_generator,
            final_gate=self.final_reference_gate,
            horizon_steps=int(self.mpc.horizon_steps),
            dt_s=float(self.mpc.dt_s),
            default_speed_mps=float(self.target_speed_mps),
        )
        self.reference_publication_stage = ReferencePublicationStage(
            reference_pipeline=self.reference_pipeline,
            reference_provider=self._stable_reference_line_provider,
        )
        self.mpc_entry_stage = MPCEntryStage(self.config)
        self._build_decision_record = build_decision_record
        self.safety_supervisor = SafetySupervisor(
            enabled=bool(self.config.get("safety_supervisor_enabled", True)),
            max_steer_delta=float(self.config.get("safety_max_steer_delta", 0.25)),
            max_throttle_delta=float(self.config.get("safety_max_throttle_delta", 0.45)),
            max_brake_delta=float(self.config.get("safety_max_brake_delta", 0.60)),
            stuck_release_min_accel_mps2=float(
                self.config.get("safety_stuck_release_min_accel_mps2", 0.01)
            ),
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
        self._clear_post_turn_exit_reference()

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
            planner_output = self.planning_pipeline.run_step()
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
        from opencda.planning_module.pipeline.reference_pipeline import (
            ReferencePipelineRequest,
        )

        latest_update = dict(getattr(self, "_latest_opencda_update", {}) or {})
        sim_time_s = float(self._sim_time_s())
        ego_transform = latest_update.get("ego_transform") or self.vehicle_manager.localizer.get_ego_pos()
        ego_speed_kmh = float(
            latest_update.get("ego_speed_kmh", self.vehicle_manager.localizer.get_ego_spd())
        )
        ego_speed_mps = ego_speed_kmh / 3.6
        measured_accel_mps2 = self.actuator_mapper.update_measurement(
            speed_mps=float(ego_speed_mps),
            timestamp_s=float(sim_time_s),
        )
        ego_location = ego_transform.location
        ego_yaw_rad = math.radians(float(ego_transform.rotation.yaw))

        self._clean_functional_test_dynamic_actors_once()

        local_object_snapshots = self._collect_object_snapshots(
            detected_objects=latest_update.get("detected_objects")
        )
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
        cp_payload = self._load_cp_message_payload()
        object_snapshots = self._fused_planning_object_snapshots(
            local_object_snapshots=local_object_snapshots,
            cp_obstacles=list(cp_payload.get("obstacles", []) or []),
            ego_location=ego_location,
            sim_time_s=float(self._sim_time_s()),
        )
        if bool(self.functional_test_ignore_dynamic_objects):
            object_snapshots = []
        mpc_object_snapshots = self._limit_obstacles_for_mpc(
            object_snapshots=object_snapshots,
            ego_location=ego_location,
        )
        front_gap_m, front_gap_actor_id_early = self._front_gap_m(
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            object_snapshots=object_snapshots,
            return_actor_id=True,
        )
        front_gap_obstacle_speed_mps_early = None
        if front_gap_actor_id_early:
            for _snapshot in object_snapshots:
                if str(self._object_track_id(_snapshot)) == str(front_gap_actor_id_early):
                    front_gap_obstacle_speed_mps_early = max(
                        0.0,
                        float(
                            _snapshot.get("v", _snapshot.get("speed_mps", 0.0))
                            or 0.0
                        ),
                    )
                    break
        from opencda.planning_module.pipeline.speed_planner import (
            effective_emergency_gap_m as _effective_emergency_gap_m,
        )

        emergency_front_gap_m = _effective_emergency_gap_m(
            base_emergency_gap_m=max(
                0.5,
                float(self.config.get("following_emergency_gap_m", 3.0)),
            ),
            ego_speed_mps=float(ego_speed_mps),
            front_obstacle_speed_mps=front_gap_obstacle_speed_mps_early,
            standstill_buffer_m=max(
                0.0,
                float(
                    self.config.get(
                        "following_emergency_standstill_buffer_m", 1.0
                    )
                ),
            ),
            time_headway_s=max(
                0.1, float(self.config.get("following_time_headway_s", 1.5))
            ),
        )
        front_gap_at_emergency_threshold = (
            front_gap_m is not None
            and float(front_gap_m) <= float(emergency_front_gap_m)
        )
        # A maneuver commitment owns lateral intent, never collision safety.
        # The former 0.5 m/s "committed lane-change crawl" bypassed this stop
        # and drove into the blocked vehicle.  Every maneuver now observes the
        # same emergency-gap contract; recovery is handled by the trajectory
        # fallback owner after the vehicle is safe, not by overriding speed.
        stop_goal_active = bool(front_gap_at_emergency_threshold)
        requested_speed_mps = 0.0 if stop_goal_active else self.target_speed_mps
        current_state = [
            float(ego_location.x),
            float(ego_location.y),
            float(ego_speed_mps),
            float(ego_yaw_rad),
        ]

        behavior_debug: dict[str, Any] = {}
        reference_debug: dict[str, Any] = {}
        try:
            (
                destination_state,
                lane_center_reference,
                behavior_stage_result,
                reference_debug,
                typed_speed_plan,
            ) = (
                self._plan_behavior_and_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=ego_yaw_rad,
                    ego_speed_mps=ego_speed_mps,
                    speed_ref_mps=requested_speed_mps,
                    object_snapshots=object_snapshots,
                    stop_goal_active=stop_goal_active,
                    cp_payload=cp_payload,
                )
            )
            behavior_decision = behavior_stage_result.decision
            behavior_debug = behavior_stage_result.mutable_diagnostics()
        except Exception as exc:
            # Persist the failing stage and source line.  ``str(exc)`` alone
            # made a route/reference failure appear as the same opaque tuple
            # error for hundreds of frames in the CSV.
            pipeline_traceback = traceback.format_exc(limit=8).strip()
            if self.debug:
                print(
                    "[CP-X OpenCDA Bridge] behavior/reference pipeline failed: "
                    f"{exc}\n{pipeline_traceback}"
                )
            generated_fallback = (
                self._stable_reference_line_provider.lane_fallback_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    current_state=current_state,
                    speed_ref_mps=float(requested_speed_mps),
                )
            )
            destination_state = generated_fallback.destination_state
            lane_center_reference = generated_fallback.samples
            behavior_stage_result = self.behavior_stage.finalize(
                maneuver="lane_follow",
                phase="FALLBACK",
                source_lane_id=self._lane_id_at_location(ego_location),
                target_lane_id=0,
                requested_speed_mps=float(requested_speed_mps),
                stop_required=False,
                route_required=False,
                traffic_signal_state="unknown",
                boundary_recovery_active=False,
                stop_target=None,
                reason="behavior_reference_pipeline_exception",
                diagnostics={
                    "pipeline_error": str(exc),
                    "pipeline_error_traceback": pipeline_traceback,
                },
            )
            behavior_decision = behavior_stage_result.decision
            behavior_debug = behavior_stage_result.mutable_diagnostics()
            reference_debug = {
                "reference_source": "current_lane_center_exception_fallback",
                "pipeline_error": str(exc),
                "pipeline_error_traceback": pipeline_traceback,
            }
            typed_speed_plan = None

        route_status = getattr(
            getattr(self, "route_manager", None), "last_status", None
        )
        destination_stage = self.destination_speed_stage.evaluate(
            route_status=route_status,
            route_revision=str(getattr(self.route_manager, "route_revision", "")),
            ego_speed_mps=float(ego_speed_mps),
        )
        route_reached_destination = bool(destination_stage.reached_destination)
        route_remaining_distance_m = float(destination_stage.remaining_distance_m)
        route_destination_approach = bool(destination_stage.approach_active)
        destination_stopping_distance_m = float(destination_stage.required_distance_m)
        destination_stop_buffer_m = float(destination_stage.stop_buffer_m)
        destination_speed_constraint = destination_stage.constraint
        # Entering the physical braking envelope is a speed-planning event,
        # not an emergency-stop event.  Keep the accepted route/reference and
        # lower its longitudinal target continuously.  Latching a zero-speed
        # fallback here used to make OpenCDA's PID apply near-full braking and
        # stop roughly ten metres before the destination.
        if bool(route_destination_approach) and not bool(route_reached_destination):
            reference_debug.update({
                "destination_stop_latched": False,
                "destination_stop_reason": "route_destination_approach_speed_profile",
                "destination_stop_remaining_distance_m": float(
                    route_remaining_distance_m
                ),
                "destination_stop_required_distance_m": float(
                    destination_stopping_distance_m
                ),
            })
        if bool(destination_stage.stop_latched):
            fallback_manager = getattr(
                self, "_trajectory_fallback_manager", TrajectoryFallbackManager()
            )
            self._trajectory_fallback_manager = fallback_manager
            stop_result = fallback_manager.bounded_safe_stop(
                current_speed_mps=float(ego_speed_mps),
                current_reference=lane_center_reference,
                reason=(
                    "route_destination_reached"
                    if route_reached_destination
                    else "route_destination_approach"
                ),
            )
            stop_reference = stop_result.mutable_trajectory()
            if len(stop_reference) >= 2:
                lane_center_reference = stop_reference
                terminal = dict(stop_reference[-1])
                destination_state = [
                    float(terminal.get("x_ref_m", terminal.get("x", current_state[0]))),
                    float(terminal.get("y_ref_m", terminal.get("y", current_state[1]))),
                    0.0,
                    float(terminal.get("heading_rad", current_state[3])),
                    int(
                        behavior_debug.get(
                            "target_lane_id", getattr(
                                getattr(self, "_local_map_snapshot", None),
                                "ego_lane_id",
                                0,
                            ),
                        )
                        or 0
                    ),
                ]
            behavior_debug.update({
                "decision": "destination_stop",
                "lc_state": "DESTINATION_STOP",
                "stop_goal_active": True,
                "target_speed_mps": 0.0,
            })
            behavior_stage_result = self.behavior_stage.destination_stop(
                behavior_stage_result
            )
            behavior_decision = behavior_stage_result.decision
            behavior_debug = behavior_stage_result.mutable_diagnostics()
            reference_debug.update({
                "reference_source": "persistent_bounded_safe_stop",
                "final_reference_geometry_source": (
                    "persistent_bounded_safe_stop"
                ),
                "destination_stop_latched": True,
                "destination_stop_reason": str(stop_result.reason),
                "destination_stop_remaining_distance_m": float(
                    route_remaining_distance_m
                ),
                "destination_stop_required_distance_m": float(
                    destination_stopping_distance_m
                ),
            })
            if float(ego_speed_mps) <= float(
                self.config.get("destination_stop_complete_speed_mps", 0.15)
            ):
                setattr(self.vehicle_manager, "_opencda_agent_finished", True)

        behavior_debug.update(behavior_decision.as_debug_fields())

        # The typed behavior decision owns the final stop state. The raw
        # front-gap threshold is only an input proposal and must not re-latch
        # stop after candidate evaluation has selected a safe route maneuver.
        selected_stop_goal_active = bool(behavior_decision.stop_required)
        speed_target = self.speed_target_planner.resolve(
            behavior=behavior_decision,
            speed_plan=typed_speed_plan,
            additional_constraints=(
                ()
                if destination_speed_constraint is None
                else (destination_speed_constraint,)
            ),
        )
        ceiling_result = self.speed_target_planner.apply(
            speed_target,
            destination_state=destination_state,
            reference_samples=lane_center_reference,
        )
        speed_ref_mps = float(speed_target.target_mps)
        destination_state = list(ceiling_result.destination_state)
        lane_center_reference = list(ceiling_result.reference_samples)
        reference_debug.update(speed_target.as_debug_fields())
        reference_debug.update({
            # Compatibility diagnostics now mirror the immutable resolver
            # result instead of the earlier SpeedPlan proposal.
            "speed_owner_selected_target_mps": float(
                speed_target.target_mps
            ),
            "speed_owner_limiting_owner": str(
                speed_target.limiting_owner
            ),
            "speed_owner_active_constraints": ";".join(
                constraint.owner for constraint in speed_target.constraints
            ),
            "speed_owner_proposed_post_plan_target_mps": float(
                speed_target.target_mps
            ),
            "speed_owner_ceiling_applied": bool(ceiling_result.applied),
            "speed_owner_ceiling_reduction_mps": float(
                ceiling_result.reduction_mps
            ),
        })
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

        publication_result = self.reference_publication_stage.run(
            destination_state=destination_state,
            reference_samples=lane_center_reference,
            current_state=current_state,
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            target_speed_mps=float(speed_ref_mps),
            behavior=behavior_decision,
            stop_goal_active=bool(mpc_stop_goal_active),
            route_points=self._active_global_route_points(),
            local_map=self._local_map_snapshot,
            route_cursor=self.route_manager.route_cursor,
            route_revision=str(self.route_manager.route_revision),
            map_epoch=str(getattr(self, "waypoint_backend", "admap") or "admap"),
            reference_source=str(
                reference_debug.get("reference_source", "planning_reference")
            ),
            candidate_status=str(reference_debug.get(
                "candidate_pipeline_selected_status", ""
            )),
            candidate_reason=str(reference_debug.get(
                "candidate_pipeline_selected_reason", ""
            )),
        )
        destination_state = publication_result.mutable_destination()
        lane_center_reference = publication_result.mutable_samples()
        final_reference_gate = publication_result.gate
        mpc_reference_stabilizer_reason = str(
            publication_result.stabilizer_reason
        )
        reference_debug.update(publication_result.debug_fields)
        destination_forward_m, destination_lateral_m = self._body_frame_xy(
            origin_x_m=float(ego_location.x),
            origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        reference_first_forward_m = ""
        reference_first_lateral_m = ""
        if lane_center_reference:
            first_reference = lane_center_reference[0]
            reference_first_forward_m, reference_first_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(first_reference.get("x_ref_m", first_reference.get("x", ego_location.x))),
                target_y_m=float(first_reference.get("y_ref_m", first_reference.get("y", ego_location.y))),
            )

        mpc_status = str(getattr(self.mpc, "_last_status", ""))
        mode_transition_guard_reason = self._apply_behavior_mode_transition_guard(
            decision=str(behavior_decision.maneuver),
            lc_state=str(behavior_decision.phase),
            target_lane_id=int(behavior_decision.target_lane_id),
            stop_goal_active=bool(mpc_stop_goal_active),
        )
        mpc_entry = self.mpc_entry_stage.evaluate(
            candidate_status=reference_debug.get(
                "candidate_pipeline_selected_status", ""
            ),
            candidate_name=reference_debug.get(
                "candidate_pipeline_selected", ""
            ),
            candidate_reason=reference_debug.get(
                "candidate_pipeline_selected_reason", ""
            ),
            final_reference_accepted=bool(final_reference_gate.accepted),
            final_reference_reason=str(final_reference_gate.reason),
            behavior_decision=str(behavior_decision.maneuver),
            stop_goal_active=bool(mpc_stop_goal_active),
            ego_speed_mps=float(ego_speed_mps),
        )
        reference_debug.update(mpc_entry.trace_fields())
        candidate_hard_gate_reason = str(mpc_entry.hard_gate_reason)
        stationary_traffic_stop_hold = bool(mpc_entry.stationary_stop_hold)
        control_context_key = "|".join((
            str(behavior_decision.maneuver),
            str(behavior_decision.phase),
            str(behavior_decision.target_lane_id),
            str(reference_debug.get("reference_source", "")),
            str(bool(mpc_stop_goal_active)),
            str(behavior_decision.traffic_signal_state),
            # A buffered control sequence was optimized against whichever
            # vehicle _front_gap_m() picked as "ahead of me" -- if that
            # identity changes (e.g. the source-lane vehicle a lane change
            # was following drops out of the gate and a different, target-
            # lane vehicle takes over), the old sequence's braking/following
            # intent no longer means what it did when it was solved, even
            # though decision/lc_state/target_lane haven't changed yet.
            str(reference_debug.get("front_gap_actor_id", "")),
        ))
        reference_anchor_relative_m = None
        if lane_center_reference:
            anchor_x_m = float(
                lane_center_reference[0].get(
                    "x_ref_m",
                    lane_center_reference[0].get("x", ego_location.x),
                )
            )
            anchor_y_m = float(
                lane_center_reference[0].get(
                    "y_ref_m",
                    lane_center_reference[0].get("y", ego_location.y),
                )
            )
            anchor_dx_m = float(anchor_x_m) - float(ego_location.x)
            anchor_dy_m = float(anchor_y_m) - float(ego_location.y)
            ego_yaw_rad = math.radians(float(ego_transform.rotation.yaw))
            reference_anchor_relative_m = (
                math.cos(ego_yaw_rad) * anchor_dx_m
                + math.sin(ego_yaw_rad) * anchor_dy_m,
                -math.sin(ego_yaw_rad) * anchor_dx_m
                + math.cos(ego_yaw_rad) * anchor_dy_m,
            )
        # MPC constrains jerk between the previous control input and the new
        # acceleration sequence. Seed that constraint with the acceleration
        # command actually sent last tick, not the measured vehicle response.
        # The latter contains actuator lag and can stay strongly negative
        # after the speed target has recovered, otherwise forcing every new
        # solve to continue braking until the vehicle is almost stationary.
        mpc_jerk_seed_accel_mps2 = float(self._last_accel_mps2)
        if str(candidate_hard_gate_reason):
            self.control_buffer.reset(reason="control_buffer_reference_hard_veto")
        elif bool(stationary_traffic_stop_hold):
            self.control_buffer.update_from_solution(
                u_solution=[[0.0, 0.0]],
                plan_time_s=float(sim_time_s),
                dt_s=float(self.mpc.dt_s),
                context_key=str(control_context_key),
                reference_anchor_relative_m=reference_anchor_relative_m,
            )
        failed_replan_buffer_reused = False
        failed_replan_maneuver_steer_held = False
        try:
            if str(candidate_hard_gate_reason):
                raise RuntimeError(str(candidate_hard_gate_reason))
            force_replan = (
                not bool(stationary_traffic_stop_hold)
                and (
                    bool(mpc_stop_goal_active)
                    or str(behavior_decision.maneuver)
                    in {
                        "stop_at_intersection",
                        "stop_sign",
                        "emergency_brake",
                        "intersection_turn_left",
                        "intersection_turn_right",
                    }
                    or bool(mode_transition_guard_reason)
                )
            )
            low_speed_buffer_replan = self._low_speed_control_buffer_force_replan(
                ego_speed_mps=float(ego_speed_mps),
                behavior_decision=str(behavior_decision.maneuver),
                behavior_fsm_state=str(behavior_decision.phase),
                stop_goal_active=bool(mpc_stop_goal_active),
            )
            force_replan = bool(force_replan) or bool(low_speed_buffer_replan)
            mpc_replan_executed = bool(
                self.control_buffer.should_replan(
                    sim_time_s=float(sim_time_s),
                    force_replan=bool(force_replan),
                    context_key=str(control_context_key),
                    reference_anchor_relative_m=reference_anchor_relative_m,
                    ego_speed_mps=float(ego_speed_mps),
                    target_speed_mps=float(speed_ref_mps),
                    speed_error_crossing_deadband_mps=float(
                        self.config.get(
                            "control_buffer_speed_crossing_deadband_mps",
                            0.15,
                        )
                    ),
                )
            )
            if bool(mpc_replan_executed):
                road_envelope_payload_world = (
                    self._current_route_tracking_lane_change_envelope_payload_world()
                )
                if road_envelope_payload_world is None:
                    road_envelope_payload_world = (
                        self._rolling_turn_envelope_payload_world(
                            behavior_decision=str(
                                behavior_decision.maneuver
                            ),
                            reference_samples=lane_center_reference,
                        )
                    )
                self.mpc.plan_trajectory(
                    current_state=current_state,
                    destination_state=destination_state,
                    object_snapshots=self._mpc_object_snapshots_with_prediction(
                        mpc_object_snapshots,
                        prediction_trajectories=reference_debug.get(
                            "prediction_trajectories", {}
                        ),
                    ),
                    current_acceleration_mps2=float(mpc_jerk_seed_accel_mps2),
                    current_steering_rad=float(self._last_steer_rad),
                    lane_center_reference_samples=lane_center_reference,
                    stop_goal_active=bool(mpc_stop_goal_active),
                    road_envelope_payload_world=road_envelope_payload_world,
                )
                mpc_status = str(getattr(self.mpc, "_last_status", "")).strip().lower()
                if mpc_status and mpc_status not in {"solved", "solved inaccurate"}:
                    raise RuntimeError(f"MPC status={mpc_status}")
                u_solution = getattr(self.mpc, "_last_u_solution", None)
                if u_solution is None or len(u_solution) == 0:
                    raise RuntimeError("MPC did not expose a control solution")
                x_solution = getattr(self.mpc, "_last_x_solution", None)
                predicted_speed_sequence_mps = (
                    None
                    if x_solution is None or len(x_solution) == 0
                    # Column 2 is speed; the (x, y) world-origin offset
                    # baked into _last_x_solution doesn't touch it.
                    else [float(state[2]) for state in x_solution]
                )
                self.control_buffer.update_from_solution(
                    u_solution=u_solution,
                    plan_time_s=float(sim_time_s),
                    dt_s=float(self.mpc.dt_s),
                    context_key=str(control_context_key),
                    reference_anchor_relative_m=reference_anchor_relative_m,
                    predicted_speed_sequence_mps=predicted_speed_sequence_mps,
                    target_speed_mps=float(speed_ref_mps),
                )
                accel_mps2 = float(u_solution[0, 0])
                steer_rad = float(u_solution[0, 1])
            else:
                buffered = self.control_buffer.sample(
                    sim_time_s=float(sim_time_s),
                    context_key=str(control_context_key),
                    reference_anchor_relative_m=reference_anchor_relative_m,
                )
                if buffered is None:
                    raise RuntimeError("MPC control buffer empty")
                accel_mps2, steer_rad, _buffer_reason = buffered
                mpc_status = (
                    "stop_hold_direct"
                    if bool(stationary_traffic_stop_hold)
                    else "buffer_reuse"
                )
            self._set_actuator_context(
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(speed_ref_mps),
                stop_goal_active=bool(mpc_stop_goal_active),
            )
            # Nominal actuation is produced once, below, from the optimized
            # velocity/steering command.  Do not also execute the legacy
            # acceleration-to-pedals mapper on the same successful solution.
            control = None
            if bool(stationary_traffic_stop_hold):
                hold_brake = min(
                    1.0,
                    max(
                        0.0,
                        float(
                            self.config.get(
                                "normal_stop_mpc_suspend_brake",
                                0.08,
                            )
                        ),
                    ),
                )
                control = self.carla.VehicleControl(
                    throttle=0.0,
                    brake=float(hold_brake),
                    steer=float(getattr(control, "steer", 0.0)),
                )
                accel_mps2 = float(self._accel_from_control(control))
                steer_rad = float(self._steer_rad_from_control(control))
            fallback_reason = ""
        except Exception as exc:
            mpc_replan_executed = True
            hard_gate_active = str(exc).startswith("candidate_hard_gate:")
            hard_gate_emergency_stop = _hard_gate_requires_emergency_stop(
                fallback_reason=str(exc),
                behavior_decision=str(behavior_decision.maneuver),
                stop_goal_active=bool(mpc_stop_goal_active),
            )
            if bool(hard_gate_active):
                mpc_replan_executed = False
            fallback_reason = str(exc)
            if bool(hard_gate_emergency_stop):
                control = self._emergency_stop_control()
                accel_mps2 = float(self._last_accel_mps2)
                steer_rad = 0.0
            else:
                normalized_behavior = str(
                    behavior_decision.maneuver
                ).strip().lower()
                maneuver_tracking_active = normalized_behavior in {
                    "intersection_turn_left",
                    "intersection_turn_right",
                    "lane_change_left",
                    "lane_change_right",
                }
                buffered_after_failure = (
                    self.control_buffer.sample(
                        sim_time_s=float(sim_time_s),
                        context_key=str(control_context_key),
                        reference_anchor_relative_m=reference_anchor_relative_m,
                    )
                    if (
                        not bool(hard_gate_active)
                        and not bool(mpc_stop_goal_active)
                    )
                    else None
                )
                if buffered_after_failure is not None:
                    (
                        accel_mps2,
                        steer_rad,
                        _failed_replan_buffer_reason,
                    ) = buffered_after_failure
                    self._set_actuator_context(
                        ego_speed_mps=float(ego_speed_mps),
                        target_speed_mps=float(speed_ref_mps),
                        stop_goal_active=False,
                    )
                    control = self._control_from_mpc(
                        float(accel_mps2), float(steer_rad)
                    )
                    self._last_accel_mps2 = float(accel_mps2)
                    self._last_steer_rad = float(steer_rad)
                    failed_replan_buffer_reused = True
                else:
                    previous_valid_steer_rad = float(self._last_steer_rad)
                    control = self._fallback_control(
                        ego_transform=ego_transform,
                        ego_speed_mps=ego_speed_mps,
                        destination_state=destination_state,
                        stop_goal_active=mpc_stop_goal_active,
                    )
                    accel_mps2 = self._last_accel_mps2
                    steer_rad = self._last_steer_rad
                    if bool(maneuver_tracking_active):
                        # A failed maneuver solve must not transfer lateral
                        # ownership to the destination-point fallback.  Hold
                        # the last accepted steering direction only briefly.
                        # Once the optimized buffer has already expired,
                        # repeatedly holding the full turn command can drive
                        # the vehicle off-road forever (the diagnosed Town06
                        # vegetation collision). Decay it toward neutral so a
                        # prolonged solver outage is fail-passive laterally.
                        failed_steer_decay = (
                            max(
                                0.0,
                                min(
                                    1.0,
                                    float(
                                        self.config.get(
                                            "turn_failed_replan_steer_decay",
                                            0.65,
                                        )
                                    ),
                                ),
                            )
                            if normalized_behavior in {
                                "intersection_turn_left",
                                "intersection_turn_right",
                            }
                            else 1.0
                        )
                        steer_rad = (
                            float(previous_valid_steer_rad)
                            * float(failed_steer_decay)
                        )
                        self._set_actuator_context(
                            ego_speed_mps=float(ego_speed_mps),
                            target_speed_mps=float(speed_ref_mps),
                            stop_goal_active=False,
                        )
                        control = self._control_from_mpc(
                            float(accel_mps2), float(steer_rad)
                        )
                        self._last_steer_rad = float(steer_rad)
                        failed_replan_maneuver_steer_held = True
            if bool(hard_gate_active) and str(
                behavior_decision.maneuver
            ) == "emergency_brake":
                mpc_status = "emergency_brake_direct"
            else:
                mpc_status = (
                    "candidate_hard_gate"
                    if bool(hard_gate_active)
                    else "buffer_reuse_after_failed_replan"
                    if bool(failed_replan_buffer_reused)
                    else "maneuver_steer_hold_after_failed_replan"
                    if bool(failed_replan_maneuver_steer_held)
                    else str(getattr(self.mpc, "_last_status", str(exc)))
                )
            if not self._warned:
                print(f"[CP-X OpenCDA Bridge] MPC fallback active: {fallback_reason}")
                self._warned = True
        mpc_feedback_record_reason = self.mpc_feedback.record_result(
            decision=str(behavior_decision.maneuver),
            target_lane_id=int(behavior_decision.target_lane_id),
            status=str(mpc_status),
            reason=str(fallback_reason),
            timestamp_s=float(sim_time_s),
            success=not bool(fallback_reason),
        )

        platform_adapter_debug: dict[str, object] = {
            "control_interface": "mpc_acceleration_steering",
        }
        hard_gate_active = str(fallback_reason).startswith(
            "candidate_hard_gate:"
        )
        # The platform adapter is part of actuation, not a post-processing
        # owner. Run it before every safety guard so signal, boundary and
        # collision decisions remain authoritative at apply_control(). Keep
        # non-gate MPC fallback controls intact instead of converting them
        # back into a routine target-speed command.
        if (
            not str(fallback_reason)
            or bool(hard_gate_active)
            or bool(failed_replan_buffer_reused)
        ):
            normalized_behavior_maneuver = str(
                behavior_decision.maneuver
            ).strip().lower()
            optimized_state_solution = getattr(self.mpc, "_last_x_solution", None)
            if (
                str(mpc_status).strip().lower() in {"solved", "solved inaccurate"}
                and optimized_state_solution is not None
                and len(optimized_state_solution) > 0
            ):
                tracking_command = self.mpc_command_extractor.extract(
                    state_solution=optimized_state_solution,
                    steering_rad=float(steer_rad),
                    solution_dt_s=float(self.mpc.dt_s),
                    timestamp_s=float(sim_time_s),
                    max_velocity_mps=float(self.mpc.constraints.max_velocity_mps),
                )
            else:
                tracking_command = self.mpc_command_extractor.hold(
                    steering_rad=float(steer_rad),
                    reason=(
                        "mpc_velocity_command_hold_after_failed_replan"
                        if bool(failed_replan_buffer_reused)
                        else "mpc_velocity_command_hold_control_buffer"
                    ),
                )
            emergency_stop_required = bool(
                _hard_gate_requires_emergency_stop(
                    fallback_reason=str(fallback_reason),
                    behavior_decision=str(normalized_behavior_maneuver),
                    stop_goal_active=bool(mpc_stop_goal_active),
                )
                or normalized_behavior_maneuver == "emergency_brake"
            )
            platform_target_velocity_mps = (
                self.mpc_command_extractor.platform_target_velocity(
                    nominal_velocity_mps=float(speed_ref_mps),
                    stop_goal_active=bool(mpc_stop_goal_active),
                    emergency_stop=bool(emergency_stop_required),
                )
            )
            (
                control,
                accel_mps2,
                steer_rad,
                platform_adapter_debug,
            ) = self._apply_velocity_steering_interface(
                target_speed_mps=float(platform_target_velocity_mps),
                target_steering_rad=float(tracking_command.target_steering_rad),
                actual_speed_mps=float(ego_speed_mps),
                stop_goal_active=bool(mpc_stop_goal_active),
                emergency_stop=bool(emergency_stop_required),
                sim_time_s=float(sim_time_s),
            )
            platform_adapter_debug.update({
                "mpc_velocity_preview_time_s": float(
                    tracking_command.velocity_preview_time_s
                ),
                "mpc_velocity_source_index": float(
                    tracking_command.velocity_source_index
                ),
                "mpc_optimized_velocity_mps": float(
                    tracking_command.target_velocity_mps
                ),
                "nominal_speed_ref_mps": float(speed_ref_mps),
                "pid_target_velocity_mps": float(
                    platform_target_velocity_mps
                ),
                "velocity_command_source": "speed_target_planner",
                "velocity_command_valid": bool(tracking_command.valid),
            })

        control, accel_mps2, steer_rad, control_guard_reason = (
            self.safety_supervisor.enforce_signal_stop(
                control=control,
                carla_module=self.carla,
                accel_mps2=float(accel_mps2),
                steer_rad=float(steer_rad),
                ego_transform=ego_transform,
                ego_speed_mps=float(ego_speed_mps),
                destination_state=destination_state,
                stop_goal_active=bool(mpc_stop_goal_active),
                traffic_signal_state=str(
                    behavior_decision.traffic_signal_state
                ),
                min_acceleration_mps2=float(
                    self.mpc.constraints.min_acceleration_mps2
                ),
                control_factory=self._control_from_mpc,
                config=self.config,
                stop_target_forward_m=stop_target_forward_m_debug,
            )
        )
        boundary_guard_reason = ""
        boundary_snapshot = None
        turn_behavior_active = str(behavior_decision.maneuver) in {
            "intersection_turn_left",
            "intersection_turn_right",
        }
        if bool(turn_behavior_active):
            boundary_snapshot = self._road_boundary_metrics(
                ego_location,
                record_sample=True,
                ego_yaw_rad=float(ego_yaw_rad),
                reference_samples=lane_center_reference,
            )
            if bool(self.config.get("boundary_recovery_enabled", False)):
                self._update_boundary_recovery_request(
                    boundary_snapshot=boundary_snapshot,
                    behavior_decision=str(
                        behavior_decision.maneuver
                    ),
                    sim_time_s=float(sim_time_s),
                    recovery_planned=bool(
                        behavior_decision.boundary_recovery_active
                    ),
                    recovery_reference_feasible=bool(
                        str(
                            reference_debug.get(
                                "candidate_pipeline_selected_status",
                                "",
                            )
                        ).strip().lower()
                        != "infeasible"
                        and bool(final_reference_gate.accepted)
                    ),
                )
            else:
                self._reset_boundary_recovery_request()
            (
                control,
                accel_mps2,
                steer_rad,
                boundary_guard_reason,
            ) = self.safety_supervisor.enforce_turn_boundary(
                control=control,
                carla_module=self.carla,
                accel_mps2=float(accel_mps2),
                steer_rad=float(steer_rad),
                ego_speed_mps=float(ego_speed_mps),
                behavior_decision=str(behavior_decision.maneuver),
                boundary_clearance_m=boundary_snapshot.get(
                    "road_boundary_clearance_m", ""
                ),
                min_acceleration_mps2=float(
                    self.mpc.constraints.min_acceleration_mps2
                ),
                config=self.config,
                control_factory=self._control_from_mpc,
                boundary_recovery_planned=bool(
                    behavior_decision.boundary_recovery_active
                ),
            )
            control_guard_reason = ";".join(
                reason
                for reason in (
                    str(control_guard_reason),
                    str(boundary_guard_reason),
                )
                if reason
            )
        else:
            self._reset_boundary_recovery_request()
        pre_supervisor_accel_mps2 = float(accel_mps2)
        pre_supervisor_steer_rad = float(steer_rad)
        control, safety_supervisor_reason = self.safety_supervisor.filter_control(
            control=control,
            carla_module=self.carla,
            safety_manager=latest_update.get("safety_manager"),
            behavior_decision=str(behavior_decision.maneuver),
            traffic_signal_state=str(behavior_decision.traffic_signal_state),
            stop_goal_active=bool(mpc_stop_goal_active),
            planner_accel_mps2=float(pre_supervisor_accel_mps2),
            sim_time_s=float(sim_time_s),
        )
        post_supervisor_accel_mps2 = self._accel_from_control(control)
        post_supervisor_steer_rad = self._steer_rad_from_control(control)
        self._last_accel_mps2 = float(post_supervisor_accel_mps2)
        self._last_steer_rad = float(post_supervisor_steer_rad)
        cp_summary = dict(getattr(self.cp_provider, "last_publish_summary", {}) or {})
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
                getattr(self, "_last_static_obstacle_stop_active_input", False)
            ),
            "static_obstacle_replan_status": str(
                getattr(self, "_static_obstacle_replan_status", "idle")
            ),
            "static_obstacle_replan_reason": str(
                getattr(self, "_static_obstacle_replan_reason", "not_requested")
            ),
            "static_obstacle_candidate_id": str(
                getattr(self, "_static_obstacle_candidate_id", "")
            ),
            "static_obstacle_blocked_lane_id": getattr(
                self, "_static_obstacle_blocked_lane_id", ""
            ),
            "static_obstacle_route_transition_pending": bool(
                getattr(
                    self,
                    "_static_obstacle_route_transition_pending",
                    False,
                )
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
            "route_replan_reason": str(self._route_replan_last_reason),
            "route_remaining_distance_m": float(
                self.route_manager.last_status.remaining_distance_m
            ),
            "route_reached_destination": bool(self.route_manager.last_status.reached_destination),
            "destination_stop_latched": bool(
                self.destination_speed_stage.stop_latched
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
            "snapshot_repr_diag": reference_debug.get("snapshot_repr_diag", ""),
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
        decision_record = self._build_decision_record(
            scenario_state=diagnostics.get("scenario_fsm_state", ""),
            behavior_decision=diagnostics.get("behavior_decision", ""),
            behavior_fsm_state=diagnostics.get("behavior_fsm_state", ""),
            candidate_selected_name=diagnostics.get("candidate_pipeline_selected", ""),
            candidate_selected_decision=diagnostics.get("candidate_selected_decision", ""),
            candidate_selected_status=diagnostics.get("candidate_pipeline_selected_status", ""),
            candidate_selected_reason=diagnostics.get("candidate_pipeline_selected_reason", ""),
            candidate_pipeline_summary=diagnostics.get("candidate_pipeline_summary", ""),
            candidate_mpc_probe_summary=diagnostics.get("candidate_mpc_probe_summary", ""),
            reference_source=diagnostics.get("reference_source", ""),
            reference_stage=diagnostics.get("reference_pipeline_stage", ""),
            reference_fallback_reason=diagnostics.get("reference_pipeline_fallback", ""),
            reference_lateral_guard_reason=diagnostics.get("reference_lateral_guard_reason", ""),
            reference_stabilizer_reason=diagnostics.get("mpc_reference_stabilizer_reason", ""),
            final_reference_gate_reason=diagnostics.get("final_reference_gate_reason", ""),
            lane_change_authorized=diagnostics.get("lane_change_authorized", ""),
            lane_change_gate_reason=diagnostics.get("lane_change_gate_reason", ""),
            route_lane_change_required=diagnostics.get("route_lane_change_required", ""),
            behavior_override_reason=diagnostics.get("behavior_override_reason", ""),
            mode_transition_guard_reason=diagnostics.get("mode_transition_guard_reason", ""),
            mpc_status=diagnostics.get("mpc_status", ""),
            mpc_fallback_reason=diagnostics.get("mpc_fallback_reason", ""),
            control_guard_reason=diagnostics.get("control_guard_reason", ""),
            control_buffer_reason=diagnostics.get("control_buffer_reason", ""),
            safety_supervisor_reason=diagnostics.get("safety_supervisor_reason", ""),
            applied_throttle=diagnostics.get("applied_throttle", 0.0),
            applied_brake=diagnostics.get("applied_brake", 0.0),
            applied_steer=diagnostics.get("applied_steer", 0.0),
        )
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

    def _full_latched_stop_target_for_signal(
        self,
        *,
        traffic_state: str,
        stop_target: Mapping[str, object] | None,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        current_lane_id: int,
    ) -> tuple[dict[str, object] | None, str]:
        state = str(traffic_state or "unknown").strip().lower()
        if state not in {"red", "yellow"}:
            if self._full_latched_stop_target is not None:
                self._full_latched_stop_target = None
                self._full_latched_stop_state = str(state)
                return None, "stop_target_latch_release"
            self._full_latched_stop_state = str(state)
            return None, ""

        if self._full_latched_stop_target is not None and self._full_latched_stop_state in {"red", "yellow"}:
            return dict(self._full_latched_stop_target), "stop_target_latch_reuse"

        latched: dict[str, object] | None = None
        if isinstance(stop_target, Mapping):
            try:
                x_value = stop_target.get("x_m", stop_target.get("x", None))
                y_value = stop_target.get("y_m", stop_target.get("y", None))
                if x_value is not None and y_value is not None:
                    latched = dict(stop_target)
                    latched["x_m"] = float(x_value)
                    latched["y_m"] = float(y_value)
                    latched["x"] = float(x_value)
                    latched["y"] = float(y_value)
                    latched["source"] = str(latched.get("source", "")) + ":latched_world_stop_target"
            except Exception:
                latched = None
        if latched is None:
            distance_m = max(
                2.0,
                float(self.config.get("full_latched_virtual_stop_distance_m", 12.0)),
            )
            x_m = float(ego_location.x) + float(distance_m) * math.cos(float(ego_yaw_rad))
            y_m = float(ego_location.y) + float(distance_m) * math.sin(float(ego_yaw_rad))
            latched = {
                "x_m": float(x_m),
                "y_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "distance_m": float(distance_m),
                "source": "latched_virtual_stop_target",
            }

        self._full_latched_stop_target = dict(latched)
        self._full_latched_stop_state = str(state)
        return dict(latched), "stop_target_latch_create"

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
                self._full_latched_stop_target is not None
                or str(self._full_latched_stop_state) in {"red", "yellow"}
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

    def _record_debug(self, payload: Mapping[str, Any]) -> None:
        if not bool(self.config.get("record_debug", True)):
            return
        try:
            debug_dir = Path(
                self.config.get(
                    "debug_output_dir",
                    Path(__file__).resolve().parent / "debug",
                )
            )
            debug_dir.mkdir(parents=True, exist_ok=True)
            if self._debug_writer is None:
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
                self._debug_jsonl_file = open(
                    debug_dir / "opencda_planner_debug.jsonl",
                    "w",
                    encoding="utf-8",
                )
            # CSV keeps the curated stable schema; JSONL keeps the COMPLETE
            # payload (every key), so no diagnostic field is ever lost.
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
                debug_dir = Path(
                    self.config.get(
                        "debug_output_dir",
                        Path(__file__).resolve().parent / "debug",
                    )
                )
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
        from opencda.planning_module.behavior_planner import (
            MpcReferenceGenerationContext,
            compute_ego_lane_offset,
            compute_temp_destination,
            evaluate_intersection_obstacle_response,
            generate_mpc_reference,
            select_reference_intent,
        )
        from opencda.planning_module.pipeline.candidate_pipeline import (
            build_candidate_intents,
            lane_change_geometry_requirements,
            lane_change_operational_curvature_limit_1pm,
        )
        from opencda.planning_module.pipeline.speed_planner import (
            turn_approach_lookahead_m,
        )

        sim_time_s = self._sim_time_s()
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
        # Route maneuver authorization depends on topology availability, not
        # on permission to use the global route as XY reference geometry.
        # AD-map owns geometry even when route_reference_allowed is false.
        route_lane_change_allowed = bool(route_context.route_found)
        from opencda.planning_module.pipeline.route_authorization import (
            lane_change_target_reached,
            suppress_lane_change_for_lateral_owner,
        )

        adjacent_lane_directions: dict[int, str] = {}
        # Direction ownership is topology-only. ``ego_waypoint`` comes from
        # the active waypoint provider; route intent remains authoritative
        # even when that provider cannot expose a stable AD lane id here.
        route_topology_direction = str(
            adapter_output.route_summary.get("lane_change_direction", "") or ""
        ).strip().lower()
        if route_topology_direction in {"left", "right"}:
            adjacent_lane_directions[int(route_optimal_lane_id)] = (
                route_topology_direction
            )
        topology_current_lane_id = int(
            local_map_snapshot.ego_lane_id
            if local_map_snapshot.frame_id > 0
            else adapter_output.route_summary.get("ad_current_lane_id", 0) or 0
        )
        topology_route_target_lane_id = int(
            local_map_snapshot.route_target_lane_id
            if local_map_snapshot.frame_id > 0
            else adapter_output.route_summary.get("ad_target_lane_id", 0) or 0
        )
        topology_lane_offset = int(
            local_map_snapshot.route_target_offset
            if local_map_snapshot.frame_id > 0
            else adapter_output.route_summary.get("lane_change_offset", 0) or 0
        )
        topology_target_in_local_frame = bool(
            local_map_snapshot.route_target_in_frame
            if local_map_snapshot.frame_id > 0
            else adapter_output.route_summary.get("target_in_local_frame", False)
        )
        physical_route_target_lane_id = int(
            local_map_snapshot.lane_at_ego_station(topology_lane_offset) or 0
            if (
                local_map_snapshot.frame_id > 0
                and int(topology_lane_offset) != 0
            )
            else topology_route_target_lane_id
        )
        if int(physical_route_target_lane_id) == 0:
            physical_route_target_lane_id = int(topology_route_target_lane_id)
        # The rolling HD-map frame owns physical left/right whenever it has a
        # target. CARLA canonical ids are point-local semantic labels and may
        # renumber across a junction; their numeric ordering must not override
        # the signed corridor offset.
        if bool(topology_target_in_local_frame) and int(topology_lane_offset) != 0:
            route_topology_direction = (
                "left" if int(topology_lane_offset) > 0 else "right"
            )
            adjacent_lane_directions[int(route_optimal_lane_id)] = str(
                route_topology_direction
            )
            adjacent_lane_directions[int(physical_route_target_lane_id)] = str(
                route_topology_direction
            )
        physical_direction_reason = (
            "admap_topology_direction_missing"
            if route_topology_direction not in {"left", "right"}
            else "admap_topology_direction"
        )

        # These gates are meters-from-maneuver, but the time available to
        # recover from a transient block (e.g. a background vehicle briefly
        # dropping the target lane's safety score right when a route-required
        # change is authorized) is distance/speed -- at a fixed distance, a
        # faster ego has strictly less time to retry before crossing the
        # "too close" line, so the same transient block that resolves fine
        # at low speed can burn through the whole margin and permanently
        # abandon the lane change at higher speed (confirmed via debug CSV on
        # cpx_single_left_lane_turn: a ~7.6s block consumed a few meters at
        # near-zero speed post-emergency-brake, but the same block duration
        # at cruise speed would consume tens of meters instead). Scaling both
        # bounds by a minimum retry-time margin keeps that recovery window
        # roughly constant in TIME regardless of speed, instead of shrinking
        # as speed increases. prep's margin is kept larger than latest's so
        # the valid window (prep > latest) never inverts and permanently
        # denies the maneuver.
        route_lane_change_preparation_start_distance_m = max(
            float(
                self.config.get(
                    "route_lane_change_preparation_start_distance_m", 45.0
                )
            ),
            float(ego_speed_mps)
            * float(
                self.config.get(
                    "route_lane_change_preparation_time_margin_s", 15.0
                )
            ),
        )
        route_lane_change_latest_start_distance_m = max(
            float(
                self.config.get("route_lane_change_latest_start_distance_m", 12.0)
            ),
            float(ego_speed_mps)
            * float(
                self.config.get(
                    "route_lane_change_latest_retry_time_margin_s", 9.0
                )
            ),
        )
        explicit_lane_change_start_distance_m = max(
            float(self.config.get("route_lane_change_min_trigger_distance_m", 8.0)),
            float(ego_speed_mps)
            * float(self.config.get("route_tracking_lane_change_duration_s", 4.0))
            + float(self.config.get("route_lane_change_trigger_buffer_m", 3.0)),
        )
        authorization_maneuver = str(route_context.next_macro_maneuver)
        normalized_authorization_maneuver = (
            authorization_maneuver.strip().lower().replace("-", "_").replace(" ", "_")
        )
        if (
            route_topology_direction in {"left", "right"}
            and normalized_authorization_maneuver
            in {"lane_change_left", "lane_change_right", "change_lane_left", "change_lane_right"}
        ):
            authorization_maneuver = f"lane_change_{route_topology_direction}"
        # W5: the arc-length route model sees a route-required lane change one
        # full preparation window ahead and names its physical direction --
        # before route_optimal_lane_id / the local HD frame diverge from the
        # current lane. Feed that straight into authorization so the maneuver
        # is prepared on the approach instead of only after ego is level with
        # (or past) the connector.
        (
            route_geometry_lane_change_direction,
            route_geometry_lane_change_distance_m,
            route_geometry_lane_change_reason,
        ) = self.route_manager.upcoming_lane_change(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            lookahead_m=float(route_lane_change_preparation_start_distance_m),
        )
        route_lane_change_edge_id = (
            self.route_manager.upcoming_lane_change_edge_id(
                lookahead_m=float(route_lane_change_preparation_start_distance_m)
            )
        )
        self.maneuver_manager.observe_route_lane_change_edge(
            route_lane_change_edge_id
        )
        # RouteGeometry names the long-lived topology edge; its target AD lane
        # id may be a downstream segment rather than the lane physically next
        # to ego.  Lateral geometry must start with the adjacent corridor in
        # this LocalMapSnapshot.  In the blocked run, commanding topology id
        # 340154 directly from 340156 skipped adjacent 340155 and produced a
        # 33 m completion error.  Resolve direction -> local offset -> physical
        # lane exactly once here, without reinterpreting numeric lane ids.
        authorization_topology_lane_offset = int(topology_lane_offset)
        geometry_direction = str(
            route_geometry_lane_change_direction or ""
        ).strip().lower()
        if (
            geometry_direction in {"left", "right"}
            and local_map_snapshot.frame_id > 0
        ):
            adjacent_offset = 1 if geometry_direction == "left" else -1
            adjacent_lane_id = int(
                local_map_snapshot.lane_at_ego_station(adjacent_offset) or 0
            )
            if adjacent_lane_id != 0:
                physical_route_target_lane_id = int(adjacent_lane_id)
                authorization_topology_lane_offset = int(adjacent_offset)
                route_topology_direction = str(geometry_direction)
                adjacent_lane_directions[int(adjacent_lane_id)] = str(
                    geometry_direction
                )
        lane_change_authorization = self.behavior_stage.authorize_route_lane_change(
            RouteLaneChangeRequest(
                route_lane_change_allowed=bool(route_lane_change_allowed),
                current_lane_id=int(current_lane_id),
                route_required_lane_id=int(physical_route_target_lane_id),
                next_macro_maneuver=str(authorization_maneuver),
                current_road_option=str(route_context.current_road_option),
                remaining_distance_m=float(route_context.next_macro_distance_m),
                available_lane_ids=tuple(planner_input_frame.map_lane.allowed_lane_ids),
                lane_safety_scores=dict(lane_safety_scores),
                lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
                preparation_start_distance_m=float(route_lane_change_preparation_start_distance_m),
                latest_start_distance_m=float(route_lane_change_latest_start_distance_m),
                target_safety_threshold=float(self.config.get("route_lane_change_target_safety_threshold", 0.65)),
                require_adjacent=bool(self.config.get("route_lane_change_require_adjacent", True)),
                explicit_lane_change_start_distance_m=float(explicit_lane_change_start_distance_m),
                adjacent_lane_directions=dict(adjacent_lane_directions),
                topology_current_lane_id=int(topology_current_lane_id),
                topology_target_lane_id=int(physical_route_target_lane_id),
                topology_lane_offset=int(authorization_topology_lane_offset),
                topology_target_in_local_frame=bool(topology_target_in_local_frame),
                route_geometry_direction=str(route_geometry_lane_change_direction or ""),
                route_geometry_distance_m=(
                    float(route_geometry_lane_change_distance_m)
                    if math.isfinite(float(route_geometry_lane_change_distance_m))
                    else None
                ),
            ),
            maneuver_manager=self.maneuver_manager,
        )
        # RouteCursor alone owns progress.  If odometry has carried ego past
        # the remaining arc of the active lane-change topology while route s
        # is stationary, the latched authorization describes a connector
        # behind the vehicle.  Cancel it and rebuild topology from ego pose.
        route_cursor = self.route_manager.route_cursor
        missed_route_maneuver = bool(route_cursor.missed_maneuver) and not (
            _lane_change_execution_active(
                reference_locked=bool(
                    self._stable_reference_line_provider.snapshot(
                        LANE_CHANGE
                    ).mutable_samples()
                ),
                phase=self.maneuver_manager.lane_change.phase,
            )
        )
        if missed_route_maneuver:
            missed_reason = (
                "route_cursor_missed_lane_change:"
                f"s={float(route_cursor.route_s_m):.2f}:"
                f"untracked_motion={float(route_cursor.stalled_motion_m):.2f}"
            )
            self.behavior_stage.reset_route_lane_change_authorization()
            self.maneuver_manager.clear_required_lane_change()
            self._attempt_turn_route_replan(
                ego_location=ego_location,
                trigger_reason="turn_missed_lane_change_route_unreachable",
            )
            lane_change_authorization = dataclasses.replace(
                lane_change_authorization,
                allowed=False,
                required_by_route=False,
                reason=str(missed_reason),
            )
        cooperative_lane_change_yield_reason = (
            self._cooperative_lane_change_yield_reason(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
            )
            if bool(lane_change_authorization.allowed)
            else ""
        )
        if cooperative_lane_change_yield_reason:
            lane_change_authorization = dataclasses.replace(
                lane_change_authorization,
                allowed=False,
                reason=cooperative_lane_change_yield_reason,
            )
            cooperative_wait_speed_cap_mps = self._cooperative_wait_speed_cap_mps(
                ego_location=ego_location,
                ego_speed_mps=float(ego_speed_mps),
                cooperative_lane_change_yield_reason=(
                    cooperative_lane_change_yield_reason
                ),
            )
            if cooperative_wait_speed_cap_mps is not None:
                additional_speed_constraints.append(
                    SpeedConstraint(
                        owner="cooperative_wait",
                        maximum_mps=float(cooperative_wait_speed_cap_mps),
                        reason=str(cooperative_lane_change_yield_reason),
                    )
                )
        route_lane_change_required = bool(lane_change_authorization.required_by_route)
        # If the route ever genuinely required a specific lane, remember it.
        # The route's own next-macro-maneuver progression advances on
        # arc-length along the original polyline regardless of which lane
        # ego actually occupies, so once the requirement lapses (denied as
        # "too close", or the route just quietly stops asking for it --
        # "already_in_required_lane"/"route_maneuver_does_not_require_lane_
        # change" can appear even though ego's own lane never changed,
        # because route_optimal_lane_id drifted to match current_lane_id
        # instead of the other way around) while ego is STILL in the lane it
        # started in, the lane change was missed, not completed. Left alone,
        # ego just keeps lane_follow-ing straight through and past the
        # junction the plan needed it to turn at, off the swept/tested route
        # corridor entirely (confirmed via debug CSV on a deterministic
        # lane-blocking-vehicle test: ego sailed ~85m past its own
        # destination with no lane change ever attempted and no replan, and
        # the actor was later destroyed off-route). Treat a lapsed
        # requirement the same as turn_reference_unavailable: request a
        # fresh route from wherever ego actually is instead of continuing to
        # chase a plan that assumed a lane change that never happened.
        if bool(lane_change_authorization.required_by_route):
            self.maneuver_manager.remember_required_lane_change(
                int(lane_change_authorization.target_lane_id),
                (
                    int(topology_route_target_lane_id)
                    if int(topology_route_target_lane_id or 0) != 0
                    else None
                ),
            )
        elif self.maneuver_manager.lane_change.required_target_lane_id is not None:
            if lane_change_target_reached(
                current_lane_id=int(current_lane_id),
                remembered_target_lane_id=int(
                    self.maneuver_manager.lane_change.required_target_lane_id
                ),
                current_ad_lane_id=int(topology_current_lane_id or 0),
                remembered_target_ad_lane_id=int(
                    self.maneuver_manager.lane_change.required_target_ad_lane_id or 0
                ),
                target_in_local_frame=bool(topology_target_in_local_frame),
                target_lane_offset=int(topology_lane_offset),
            ):
                self.maneuver_manager.clear_required_lane_change()
            elif _lane_change_execution_active(
                reference_locked=bool(
                    self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()
                ),
                phase=self.maneuver_manager.lane_change.phase,
            ):
                # The route instruction can advance before the locked
                # trajectory has physically reached its target lane.  Keep
                # the remembered requirement and let the committed geometry
                # finish; otherwise a mid-maneuver route replan resets the
                # reference and produces a one-frame lane-follow interruption.
                pass
            elif bool(self.config.get("missed_lane_change_route_replan_enabled", True)):
                self._attempt_turn_route_replan(
                    ego_location=ego_location,
                    trigger_reason="lane_change_missed_route_unreachable",
                )
                self.maneuver_manager.clear_required_lane_change()
        lane_change_authorized = bool(lane_change_authorization.allowed)
        opportunistic_authorization = (
            self.behavior_stage.authorize_opportunistic_lane_change(
                OpportunisticLaneChangeRequest(
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
                )
            )
        )
        opportunistic_lane_change_allowed = bool(
            opportunistic_authorization.allowed
        )
        lane_change_gate_reason = (
            ""
            if bool(opportunistic_lane_change_allowed)
            else str(opportunistic_authorization.reason)
        )
        signal_context = dict(adapter_output.signal_context)
        source_quality = dict(adapter_output.source_quality)
        raw_traffic_state = str(
            planner_input_frame.planning.traffic_control.signal_state
        )
        resolved_traffic_state, signal_actor_resolution_reason = (
            self._resolve_full_traffic_state_from_carla_actor(
                raw_state=str(raw_traffic_state),
                signal_context=signal_context,
            )
        )
        raw_stop_target = (
            planner_input_frame.planning.traffic_control.stop_target.as_dict()
            if planner_input_frame.planning.traffic_control.stop_target.active
            else None
        )
        filtered_traffic_state, filtered_stop_target, full_traffic_memory_reason = (
            self._full_traffic_memory.update(
                state=str(resolved_traffic_state),
                stop_target=raw_stop_target,
                sim_time_s=float(sim_time_s),
            )
        )
        if str(signal_actor_resolution_reason):
            full_traffic_memory_reason = (
                f"{signal_actor_resolution_reason};{full_traffic_memory_reason}"
                if str(full_traffic_memory_reason)
                else str(signal_actor_resolution_reason)
            )
        filtered_stop_target, stop_latch_reason = self._full_latched_stop_target_for_signal(
            traffic_state=str(filtered_traffic_state),
            stop_target=(
                dict(filtered_stop_target)
                if isinstance(filtered_stop_target, Mapping)
                else None
            ),
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            current_lane_id=int(current_lane_id),
        )
        if str(stop_latch_reason):
            full_traffic_memory_reason = (
                f"{full_traffic_memory_reason};{stop_latch_reason}"
                if str(full_traffic_memory_reason)
                else str(stop_latch_reason)
            )
        traffic_stop_forward_m, traffic_stop_target_reliable = self.reference_generator.stop_target_forward(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            stop_target=(
                dict(filtered_stop_target)
                if isinstance(filtered_stop_target, Mapping)
                else None
            ),
            fallback_destination_state=[],
        )
        (
            upcoming_turn_direction,
            upcoming_turn_distance_m,
            upcoming_turn_reason,
        ) = self.route_manager.upcoming_turn(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            lookahead_m=float(
                turn_approach_lookahead_m(
                    cruise_speed_mps=float(self.target_speed_mps),
                    config=dict(self.config),
                )
            ),
        )
        route_macro_text = str(route_context.next_macro_maneuver or "").strip().lower()
        route_macro_normalized = route_macro_text.replace("-", "_").replace(" ", "_")
        route_advanced_to_lane_change = route_macro_normalized in {
            "lane_change_left",
            "lane_change_right",
            "change_lane_left",
            "change_lane_right",
        }
        route_macro_direction = (
            "left" if "turn left" in route_macro_text
            else "right" if "turn right" in route_macro_text
            else ""
        )
        # L1: RouteGeometry.next_turn() is the authoritative turn source. The
        # next-macro text is only a *fallback* for when the arc-length model
        # does not (yet) see a junction turn in the lookahead -- it must not
        # overwrite a geometry hit, which previously masked next_turn()
        # entirely and made a route-geometry turn regression invisible.
        if route_macro_direction and str(upcoming_turn_reason) != "route_geometry_turn_ahead":
            upcoming_turn_direction = str(route_macro_direction)
            upcoming_turn_distance_m = float(route_context.next_macro_distance_m)
            upcoming_turn_reason = "admap_route_macro_direction_fallback"
        (
            turn_exit_heading_error_rad,
            turn_exit_lateral_m,
            turn_exit_alignment_reason,
        ) = self.route_manager.route_alignment(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            heading_lookahead_m=float(
                self.config.get("scenario_turn_exit_heading_lookahead_m", 5.0)
            ),
        )
        turn_exit_alignment_valid = bool(
            math.isfinite(float(turn_exit_heading_error_rad))
            and math.isfinite(float(turn_exit_lateral_m))
        )
        turn_exit_aligned = bool(
            turn_exit_alignment_valid
            and abs(float(turn_exit_heading_error_rad))
            <= float(
                self.config.get(
                    "scenario_turn_exit_max_heading_error_rad",
                    0.15,
                )
            )
            and float(turn_exit_lateral_m)
            <= float(
                self.config.get(
                    "scenario_turn_exit_max_lateral_m",
                    0.75,
                )
            )
        )
        scenario_decision = self._scenario_manager.update(
            traffic_state=str(filtered_traffic_state),
            stop_target=(
                dict(filtered_stop_target)
                if isinstance(filtered_stop_target, Mapping)
                else None
            ),
            stop_forward_m=float(traffic_stop_forward_m),
            stop_target_reliable=bool(traffic_stop_target_reliable),
            ego_speed_mps=float(ego_speed_mps),
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            current_road_option=str(route_context.current_road_option),
            next_macro_maneuver=str(route_context.next_macro_maneuver),
            sim_time_s=float(sim_time_s),
            upcoming_turn_direction=str(upcoming_turn_direction),
            upcoming_turn_distance_m=float(upcoming_turn_distance_m),
            turn_exit_alignment_valid=bool(turn_exit_alignment_valid),
            turn_exit_aligned=bool(turn_exit_aligned),
            turn_exit_heading_error_rad=float(turn_exit_heading_error_rad),
            turn_exit_lateral_m=float(turn_exit_lateral_m),
            boundary_recovery_request=(
                getattr(self, "_boundary_recovery_request", None)
                if bool(
                    self.config.get(
                        "boundary_recovery_enabled",
                        False,
                    )
                )
                else None
            ),
        )
        # Do not start a Frenet lane change that cannot finish, including its
        # fixed target-lane handoff, before the next turn connector.
        lane_change_operational_curvature_1pm = (
            lane_change_operational_curvature_limit_1pm(
                planning_speed_mps=float(speed_ref_mps),
                lateral_accel_limit_mps2=float(
                    self.config.get(
                        "route_tracking_lane_change_lateral_accel_limit_mps2",
                        1.3,
                    )
                ),
                vehicle_max_curvature_1pm=float(
                    self.config.get("reference_vehicle_max_curvature_1pm", 0.35)
                ),
                minimum_speed_mps=float(
                    self.config.get("lane_change_min_geometry_speed_mps", 2.0)
                ),
            )
        )
        _, lane_change_geometry_arc_m, _ = lane_change_geometry_requirements(
            ego_speed_mps=float(ego_speed_mps),
            target_speed_mps=float(speed_ref_mps),
            duration_s=float(self.candidate_lane_change_normal_duration_s),
            dt_s=float(self.mpc.dt_s),
            lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
            max_curvature_1pm=float(lane_change_operational_curvature_1pm),
            minimum_geometry_speed_mps=float(
                self.config.get("lane_change_min_geometry_speed_mps", 2.0)
            ),
            minimum_length_m=float(
                self.config.get("lane_change_min_length_m", 10.0)
            ),
            acceleration_limit_mps2=float(
                self.config.get(
                    "lane_change_planning_acceleration_limit_mps2", 2.0
                )
            ),
        )
        lane_change_handoff_arc_m = max(
            0.0,
            float(
                self.config.get(
                    "lane_change_to_turn_reference_transition_arc_m", 10.0
                )
            ),
        )
        lane_change_start_transition = (
            self.maneuver_manager.lane_change_start_feasibility(
                authorization_allowed=bool(lane_change_authorization.allowed),
                distance_to_turn_m=float(upcoming_turn_distance_m),
                geometry_arc_m=float(lane_change_geometry_arc_m),
                handoff_arc_m=float(lane_change_handoff_arc_m),
            )
        )
        if lane_change_start_transition.action == "deny":
            self.behavior_stage.reset_route_lane_change_authorization()
            lane_change_authorization = dataclasses.replace(
                lane_change_authorization,
                allowed=False,
                reason=str(lane_change_start_transition.reason),
            )

        # Turn and lane-change references cannot both own lateral geometry.
        # Release the semantic commitment and persistent reference together,
        # before candidate commitment selection can revive the old maneuver.
        lateral_handoff = self.maneuver_manager.transfer_lateral_ownership_to_turn(
            owner_state=scenario_decision.state,
        )
        if lateral_handoff.action == "release":
            self.behavior_stage.reset_route_lane_change_authorization()
            self.maneuver_manager.clear_required_lane_change()
            self._stable_reference_line_provider.release(
                LANE_CHANGE,
                event=str(lateral_handoff.reason),
            )
            reset_lane_change = getattr(
                self.behavior_planner, "_reset_lane_change_state", None
            )
            if callable(reset_lane_change):
                reset_lane_change(reason=str(lateral_handoff.reason))
        lane_change_authorization = suppress_lane_change_for_lateral_owner(
            lane_change_authorization,
            owner_state=scenario_decision.state,
        )
        lane_change_authorized = bool(lane_change_authorization.allowed)
        if str(lane_change_authorization.reason).startswith(
            "scenario_lateral_owner:"
        ):
            opportunistic_lane_change_allowed = False
            lane_change_gate_reason = (
                "opportunistic_lane_change_suppressed:"
                + str(lane_change_authorization.reason)
            )
        behavior_traffic_state = str(scenario_decision.behavior_signal_state)
        behavior_stop_target = (
            dict(scenario_decision.behavior_stop_target)
            if isinstance(scenario_decision.behavior_stop_target, Mapping)
            else None
        )
        traffic_stop_commit_distance_m = float(
            scenario_decision.traffic_stop_commit_distance_m
        )
        traffic_stop_approach_speed_cap_mps = float(
            scenario_decision.speed_cap_mps
            if scenario_decision.speed_cap_mps is not None
            else self.target_speed_mps
        )
        traffic_stop_approach_reason = str(scenario_decision.reason)
        filtered_signal_context = dict(signal_context or {})
        filtered_signal_context["raw_signal_state"] = str(raw_traffic_state)
        filtered_signal_context["resolved_signal_state"] = str(
            resolved_traffic_state
        )
        filtered_signal_context["signal_state"] = str(filtered_traffic_state)
        filtered_signal_context["behavior_signal_state"] = str(behavior_traffic_state)
        filtered_signal_context["scenario_owns_traffic_control"] = True
        filtered_signal_context["scenario_fsm_state"] = str(
            scenario_decision.state
        )
        filtered_signal_context["traffic_stop_forward_m"] = float(traffic_stop_forward_m)
        filtered_signal_context["traffic_stop_commit_distance_m"] = float(traffic_stop_commit_distance_m)
        filtered_signal_context["route_upcoming_turn_direction"] = str(
            upcoming_turn_direction
        )
        filtered_signal_context["route_upcoming_turn_distance_m"] = (
            ""
            if not math.isfinite(float(upcoming_turn_distance_m))
            else float(upcoming_turn_distance_m)
        )
        filtered_signal_context["route_upcoming_turn_reason"] = str(
            upcoming_turn_reason
        )
        filtered_signal_context["turn_exit_heading_error_rad"] = (
            ""
            if not bool(turn_exit_alignment_valid)
            else float(turn_exit_heading_error_rad)
        )
        filtered_signal_context["turn_exit_lateral_m"] = (
            ""
            if not bool(turn_exit_alignment_valid)
            else float(turn_exit_lateral_m)
        )
        filtered_signal_context["turn_exit_aligned"] = bool(turn_exit_aligned)
        filtered_signal_context["turn_exit_alignment_reason"] = str(
            turn_exit_alignment_reason
        )
        if str(traffic_stop_approach_reason):
            filtered_signal_context["traffic_stop_approach_reason"] = str(traffic_stop_approach_reason)
        if str(full_traffic_memory_reason):
            filtered_signal_context["traffic_memory_reason"] = str(full_traffic_memory_reason)
        mpc_feedback = self.mpc_feedback.candidate_feedback(
            current_time_s=float(sim_time_s)
        )
        nearest_front_obstacles_by_lane = self._nearest_front_obstacle_by_lane(
            ego_snapshot={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "psi": float(ego_yaw_rad),
            },
            obstacle_snapshots=object_snapshots,
            lane_assignments=dict(
                planner_input_frame.prediction.lane_assignments or {}
            ),
            available_lane_ids=list(
                planner_input_frame.map_lane.allowed_lane_ids
            ),
        )
        if bool(lane_change_authorized):
            candidate_lane_ids = [
                int(current_lane_id),
                int(lane_change_authorization.target_lane_id),
            ]
        elif bool(opportunistic_lane_change_allowed):
            candidate_lane_ids = list(planner_input_frame.map_lane.allowed_lane_ids)
        else:
            candidate_lane_ids = [int(current_lane_id)]
        candidate_frame = self.behavior_stage.evaluate_lane_candidates(
            BehaviorCandidateRequest(
                lane_safety_scores=dict(lane_safety_scores),
                lane_prediction_risks=dict(
                    planner_input_frame.prediction.lane_prediction_risks
                ),
                ego_lane_id=int(current_lane_id),
                available_lane_ids=tuple(candidate_lane_ids),
                route_optimal_lane_id=int(route_optimal_lane_id),
                in_junction=bool(planner_input_frame.map_lane.in_junction),
                mpc_feedback_blocked_lane_ids=tuple(
                    mpc_feedback.get("blocked_lane_ids", []) or []
                ),
                mpc_feedback_weight=float(
                    self.config.get("mpc_feedback_candidate_weight", 80.0)
                ),
                nearest_front_obstacles_by_lane=dict(
                    nearest_front_obstacles_by_lane
                ),
                desired_speed_mps=float(self.target_speed_mps),
                progress_cost_weight=float(
                    self.config.get("candidate_progress_cost_weight", 4.0)
                ),
            )
        )
        preferred_target_lane_id = (
            int(lane_change_authorization.target_lane_id)
            if bool(lane_change_authorized)
            else int(candidate_frame.selected.target_lane_id)
            if bool(opportunistic_lane_change_allowed)
            else int(current_lane_id)
        )

        try:
            behavior_lane_alignment = compute_ego_lane_offset(
                self.reference_map,
                ego_pose,
            )
        except Exception:
            behavior_lane_alignment = {
                "lane_id": 0,
                "lateral_offset_m": float("inf"),
                "heading_error_rad": float("inf"),
            }
        behavior_lane_alignment_valid = bool(
            int(behavior_lane_alignment.get("lane_id", 0) or 0) != 0
            and math.isfinite(
                float(behavior_lane_alignment.get("lateral_offset_m", float("nan")))
            )
            and math.isfinite(
                float(behavior_lane_alignment.get("heading_error_rad", float("nan")))
            )
        )
        behavior_lane_lateral_error_m = float(
            behavior_lane_alignment.get("lateral_offset_m", 0.0)
        )
        behavior_lane_heading_error_rad = float(
            behavior_lane_alignment.get("heading_error_rad", 0.0)
        )
        if not bool(behavior_lane_alignment_valid):
            behavior_lane_lateral_error_m = float("inf")
            behavior_lane_heading_error_rad = float("inf")

        static_front_obstacle = nearest_front_obstacles_by_lane.get(
            int(current_lane_id)
        )
        actual_obstacle_mode = (
            "INTERSECTION"
            if bool(planner_input_frame.map_lane.in_junction)
            else "NORMAL"
        )
        obstacle_evaluation_mode = str(actual_obstacle_mode)
        if (
            obstacle_evaluation_mode == "NORMAL"
            and bool(
                self.config.get(
                    "static_obstacle_replan_normal_mode_enabled",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_replan_normal_mode_enabled",
                        True,
                    ),
                )
            )
        ):
            # Reuse the same conservative two-condition classifier on normal
            # roads when explicitly enabled. The classifier itself remains
            # intersection-scoped for backward compatibility.
            obstacle_evaluation_mode = "INTERSECTION"
        static_obstacle_response = evaluate_intersection_obstacle_response(
            mode=str(obstacle_evaluation_mode),
            front_obstacle_speed_mps=(
                None
                if static_front_obstacle is None
                else float(static_front_obstacle.get("v", 0.0))
            ),
            original_max_velocity_mps=float(self.target_speed_mps),
            moving_obstacle_speed_threshold_mps=float(
                self.config.get(
                    "static_obstacle_speed_threshold_mps",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_speed_threshold_mps",
                        self.behavior_runtime_cfg.get(
                            "intersection_obstacle_moving_speed_threshold_mps",
                            0.5,
                        ),
                    ),
                )
            ),
            route_lane_safety_score=float(
                lane_safety_scores.get(int(current_lane_id), 1.0)
            ),
            static_obstacle_replan_lane_safety_threshold=float(
                self.config.get(
                    "static_obstacle_replan_lane_safety_threshold",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_replan_lane_safety_threshold",
                        self.behavior_runtime_cfg.get(
                            "intersection_static_obstacle_replan_lane_safety_threshold",
                            0.5,
                        ),
                    ),
                )
            ),
        )
        traffic_control_stop_active = bool(
            scenario_decision.stop_goal_active
            or str(behavior_traffic_state).strip().lower()
            in {"red", "yellow", "stop"}
        )
        static_replan_requested = bool(
            self.config.get(
                "static_obstacle_replan_enabled",
                self.behavior_runtime_cfg.get(
                    "static_obstacle_replan_enabled",
                    True,
                ),
            )
            and static_obstacle_response.get(
                "request_static_obstacle_replan", False
            )
            and not bool(traffic_control_stop_active)
        )
        static_obstacle_transition_hold = False
        static_obstacle_cooldown_hold = False
        latched_local_target_lane_id = getattr(
            self, "_static_obstacle_local_target_lane_id", None
        )
        if (
            latched_local_target_lane_id is not None
            and int(current_lane_id) == int(latched_local_target_lane_id)
            and not bool(self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples())
        ):
            # The local lane-borrow maneuver has geometrically converged.
            self._static_obstacle_local_target_lane_id = None
            latched_local_target_lane_id = None
        static_obstacle_local_avoidance_active = bool(
            latched_local_target_lane_id is not None
        )
        static_obstacle_local_target_lane_id: int | None = (
            None
            if latched_local_target_lane_id is None
            else int(latched_local_target_lane_id)
        )
        static_obstacle_id = (
            ""
            if static_front_obstacle is None
            else self._object_track_id(static_front_obstacle)
        )
        if not bool(static_replan_requested):
            self._static_obstacle_candidate_id = ""
            self._static_obstacle_candidate_since_s = -float("inf")
            # Seeing a clear frame closes the previous encounter. If the same
            # object blocks the route again during cooldown, it is a new
            # encounter and must hold stop until a retry is allowed.
            self._static_obstacle_route_transition_pending = False
            self._static_obstacle_replan_failed_latched = False
            self._static_obstacle_replan_status = (
                "local_avoidance_executing"
                if bool(static_obstacle_local_avoidance_active)
                else "traffic_control_excluded"
                if bool(traffic_control_stop_active)
                and bool(
                    static_obstacle_response.get(
                        "request_static_obstacle_replan", False
                    )
                )
                else "idle"
            )
        else:
            if str(static_obstacle_id) != str(self._static_obstacle_candidate_id):
                self._static_obstacle_candidate_id = str(static_obstacle_id)
                self._static_obstacle_candidate_since_s = float(sim_time_s)
            blocked_confirm_s = max(
                0.0,
                float(
                    self.config.get(
                        "static_obstacle_blocked_confirm_s",
                        self.behavior_runtime_cfg.get(
                            "static_obstacle_blocked_confirm_s",
                            1.0,
                        ),
                    )
                ),
            )
            blocked_elapsed_s = max(
                0.0,
                float(sim_time_s)
                - float(self._static_obstacle_candidate_since_s),
            )
            if blocked_elapsed_s < blocked_confirm_s:
                self._static_obstacle_replan_status = "confirming"
            else:
                local_avoidance_enabled = bool(
                    self.config.get(
                        "static_obstacle_local_avoidance_enabled",
                        self.behavior_runtime_cfg.get(
                            "static_obstacle_local_avoidance_enabled", True
                        ),
                    )
                )
                local_target_lane_id = (
                    _select_static_obstacle_local_avoidance_lane(
                        current_lane_id=int(current_lane_id),
                        available_lane_ids=list(
                            planner_input_frame.map_lane.allowed_lane_ids
                        ),
                        lane_safety_scores=lane_safety_scores,
                        lane_prediction_risks=dict(
                            planner_input_frame.prediction.lane_prediction_risks
                        ),
                        minimum_safety_score=float(
                            self.config.get(
                                "static_obstacle_local_lane_min_safety_score",
                                self.behavior_runtime_cfg.get(
                                    "static_obstacle_local_lane_min_safety_score",
                                    0.55,
                                ),
                            )
                        ),
                    )
                    if bool(local_avoidance_enabled)
                    and str(actual_obstacle_mode) == "NORMAL"
                    else None
                )
                avoidance_lane_yield_reason = ""
                if local_target_lane_id is not None:
                    avoidance_lane_yield_reason = (
                        self._cooperative_avoidance_lane_yield_reason(
                            target_lane_id=int(local_target_lane_id),
                            ego_location=ego_location,
                            ego_yaw_rad=float(ego_yaw_rad),
                        )
                    )
                    if avoidance_lane_yield_reason:
                        local_target_lane_id = None
                if local_target_lane_id is not None:
                    static_obstacle_local_avoidance_active = True
                    static_obstacle_local_target_lane_id = int(local_target_lane_id)
                    self._static_obstacle_local_target_lane_id = int(
                        local_target_lane_id
                    )
                    self._static_obstacle_replan_failed_latched = False
                    self._static_obstacle_route_transition_pending = False
                    self._static_obstacle_replan_status = "local_avoidance_ready"
                    self._static_obstacle_replan_reason = (
                        "static_obstacle_local_lane_borrow:"
                        f"target_lane={int(local_target_lane_id)}"
                    )
                elif bool(
                    self.config.get(
                        "static_obstacle_global_replan_enabled",
                        self.behavior_runtime_cfg.get(
                            "static_obstacle_global_replan_enabled", False
                        ),
                    )
                ):
                    attempted, succeeded, replan_reason = (
                        self._attempt_static_obstacle_route_replan(
                            ego_location=ego_location,
                            obstacle=dict(static_front_obstacle or {}),
                        )
                    )
                    self._static_obstacle_replan_reason = str(replan_reason)
                    if bool(succeeded):
                        self._static_obstacle_replan_failed_latched = False
                        self._static_obstacle_replan_status = "succeeded"
                        self._static_obstacle_route_transition_pending = True
                        static_obstacle_transition_hold = True
                    elif bool(attempted):
                        self._static_obstacle_replan_failed_latched = True
                        self._static_obstacle_replan_status = "failed_stop"
                    else:
                        (
                            self._static_obstacle_replan_status,
                            static_obstacle_cooldown_hold,
                        ) = _static_obstacle_cooldown_policy(
                            failed_latched=bool(
                                self._static_obstacle_replan_failed_latched
                            ),
                            route_transition_pending=bool(
                                self._static_obstacle_route_transition_pending
                            ),
                        )
                else:
                    # Local avoidance is unavailable or unsafe.  Preserve the
                    # active global route and stop behind the obstacle; a
                    # cooperative road-closure event may request rerouting via
                    # the separate BehaviorPlanner reroute-message path. Retry
                    # continues every tick (this whole branch re-runs
                    # unconditionally next step), so a cooperative-yield hold
                    # self-clears as soon as the peer's claim does.
                    self._static_obstacle_replan_failed_latched = True
                    self._static_obstacle_route_transition_pending = False
                    self._static_obstacle_replan_status = (
                        "local_avoidance_yield_to_peer_cav"
                        if avoidance_lane_yield_reason
                        else "local_avoidance_unavailable_stop"
                    )
                    self._static_obstacle_replan_reason = (
                        avoidance_lane_yield_reason
                        or "static_obstacle_local_avoidance_unavailable"
                    )

        if bool(static_obstacle_local_avoidance_active):
            # A confirmed blocker is an explicit behavior-level reason to
            # consider an adjacent lane.  It bypasses only the route-demand
            # gate; prediction, lane-safety, reference and MPC safety gates
            # remain unchanged.
            opportunistic_lane_change_allowed = True
            preferred_target_lane_id = int(static_obstacle_local_target_lane_id)
        static_obstacle_stop_active = bool(
            self._static_obstacle_replan_failed_latched
            or static_obstacle_transition_hold
            or static_obstacle_cooldown_hold
        )
        self._last_static_obstacle_stop_active_input = bool(static_obstacle_stop_active)

        command = self.behavior_planner.update(
            static_obstacle_stop_active=bool(static_obstacle_stop_active),
            lane_safety_scores=lane_safety_scores,
            ego_lane_id=int(current_lane_id),
            selected_lane_id=int(current_lane_id),
            ego_lateral_offset_m=float(behavior_lane_lateral_error_m),
            ego_heading_error_rad=float(behavior_lane_heading_error_rad),
            mode="INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL",
            route_optimal_lane_id=int(route_optimal_lane_id),
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            front_obstacle_distance_by_lane=front_dist_by_lane,
            current_time_s=float(sim_time_s),
            wall_time_s=float(sim_time_s),
            traffic_signal_state=str(behavior_traffic_state),
            traffic_stop_target=(
                dict(behavior_stop_target)
                if isinstance(behavior_stop_target, Mapping)
                else None
            ),
            traffic_signal_context=dict(filtered_signal_context or {}),
            ego_speed_mps=float(ego_speed_mps),
            ego_max_deceleration_mps2=abs(float(self.mpc.constraints.min_acceleration_mps2)),
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            ego_position_xy=(float(ego_location.x), float(ego_location.y)),
            global_route_points=route_points,
            nearest_front_obstacles_by_lane=nearest_front_obstacles_by_lane,
            lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
            preferred_target_lane_id=int(preferred_target_lane_id),
            local_avoidance_target_lane_id=(
                int(static_obstacle_local_target_lane_id)
                if bool(static_obstacle_local_avoidance_active)
                and static_obstacle_local_target_lane_id is not None
                else None
            ),
            lane_change_completion_allowed=not bool(
                self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()
            ),
        )
        decision = str(command.get("decision", "lane_follow"))
        target_lane_id = int(command.get("target_lane_id", current_lane_id) or current_lane_id)
        lc_state = str(command.get("lc_state", "LANE_KEEP"))
        if bool(static_obstacle_local_avoidance_active):
            if str(decision) in {"lane_change_left", "lane_change_right"}:
                self._static_obstacle_replan_status = "local_avoidance_executing"
            elif bool(traffic_control_stop_active) and str(decision) in {
                "stop_at_intersection",
                "stop_sign",
            }:
                self._static_obstacle_replan_status = (
                    "local_avoidance_preempted_by_traffic_control"
                )
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
            # The AD route has consumed the connector. A stale CARLA road
            # option must not recreate the turn after ScenarioManager released
            # it, even while CARLA still reports ego inside the junction.
            route_turn_decision = ""
            self.maneuver_manager.clear_turn(
                reason="route_advanced_to_lane_change"
            )
            self._clear_turn_master_reference()
        route_turn_prepare_decision = ""
        scenario_behavior_override = str(scenario_decision.behavior_override_decision or "")
        override_result = self.behavior_stage.apply_overrides(
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
        front_gap_m, front_gap_actor_id = self._front_gap_m(
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
        speed_plan = self.speed_target_planner.propose(
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
        nominal_destination_state = compute_temp_destination(
            map_planner=self.reference_map,
            ego_pose=ego_pose,
            target_lane_id=int(target_lane_id),
            decision=str(decision),
            lookahead_m=float(self.lookahead_m),
            target_v_mps=float(planned_speed_mps),
            global_route_points=route_points,
            mode_reference_xy=(
                None
                if nominal_state.target is None
                else (
                    float(nominal_state.target.x_m),
                    float(nominal_state.target.y_m),
                )
            ),
            prev_mode=(
                None
                if nominal_state.target is None
                else float(nominal_state.target.mode_value)
            ),
            prev_road_id=(
                None
                if nominal_state.target is None
                else nominal_state.target.road_id
            ),
            prev_entered_intersection=(
                bool(nominal_state.target.entered_intersection)
                if nominal_state.target is not None else False
            ),
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            mode_override=str(planner_mode),
            follow_global_route_lane=bool(
                route_reference_allowed and planner_input_frame.map_lane.in_junction
            ),
        )

        reference_intent = select_reference_intent(
            behavior_decision=str(decision),
            planner_fsm_state=str(lc_state),
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            reference_target_lane_id=int(target_lane_id),
            current_lane_id=int(current_lane_id),
            route_optimal_lane_id=int(route_optimal_lane_id),
            global_route_reference_allowed=bool(route_reference_allowed),
            traffic_control_lane_lock_active=False,
        )
        ref_context = MpcReferenceGenerationContext(
            map_planner=self.reference_map,
            ego_pose=ego_pose,
            ego_state=current_state,
            active_global_route_points=route_points,
            previous_lane_center_reference=nominal_state.mutable_samples(),
            behavior_runtime_cfg=self.behavior_runtime_cfg,
            reference_intent=reference_intent,
            current_applied_behavior=str(decision),
            cached_planner_lc_state=str(lc_state),
            reference_target_lane_id=int(target_lane_id),
            current_lane_id=int(current_lane_id),
            global_route_reference_allowed=bool(route_reference_allowed),
            global_route_reference_gate_reason=str(route_reference_gate_reason),
            should_follow_global_route_lane_for_reference=bool(
                reference_intent.follow_global_route_lane
            ),
            traffic_control_lane_lock_active=False,
            final_goal_stop_active=False,
            stop_target_state=None,
            follow_target_state=None,
            current_temp_reference_xy=(
                float(nominal_destination_state[0]),
                float(nominal_destination_state[1]),
            ),
            current_temp_mode_value=(
                float(nominal_destination_state[5])
                if len(nominal_destination_state) >= 6 else 0.0
            ),
            current_temp_road_id=(
                int(nominal_destination_state[6])
                if len(nominal_destination_state) >= 7 else None
            ),
            current_temp_entered_intersection=(
                bool(float(nominal_destination_state[7]) > 0.5)
                if len(nominal_destination_state) >= 8 else False
            ),
            active_reference_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            current_temp_mode_str=str(planner_mode),
            lane_reference_speed_mps=max(
                1.0,
                float(ego_speed_mps),
                abs(float(planned_speed_mps)),
            ),
            lane_reference_step_distance_m=max(
                0.5,
                float(self.mpc.dt_s)
                * max(1.0, float(ego_speed_mps), abs(float(planned_speed_mps))),
            ),
            mpc_horizon_steps=int(self.mpc.horizon_steps),
            mpc_dt_s=float(self.mpc.dt_s),
            temporary_destination_state=nominal_destination_state,
            lane_reference_freeze_count=int(nominal_state.reference_freeze_count),
            sim_time_s=float(sim_time_s),
            stop_release_temp_smooth_until_sim_time_s=float(self._stop_release_temp_smooth_until_sim_time_s),
            authoritative_ego_waypoint=self._authoritative_ego_waypoint,
        )
        ref_output = generate_mpc_reference(ref_context)
        local_lane_center_reference = [
            dict(sample) for sample in list(ref_output.local_lane_center_reference or [])
        ]
        nominal_destination_state = list(
            ref_output.temporary_destination_state or nominal_destination_state
        )
        nominal_freeze_count = int(ref_output.lane_reference_freeze_count)
        reference_debug = dict(ref_output.mpc_reference_result.trace.as_trace_fields())
        reference_debug.update(planner_input_frame.trace_fields())
        route_topology_validation = self.route_manager.route_topology_validation
        reference_debug.update({
            "stage": reference_debug.get("reference_pipeline_stage", ""),
            "intent_mode": reference_debug.get("reference_pipeline_intent_mode", ""),
            "fallback_reason": str(ref_output.last_reference_fallback_reason),
            "reference_source": "behavior_reference_pipeline",
            "front_gap_actor_id": str(front_gap_actor_id or ""),
            "front_gap_obstacle_speed_mps": (
                ""
                if front_gap_obstacle_speed_mps is None
                else float(front_gap_obstacle_speed_mps)
            ),
            "front_gap_obstacle_lane_id": int(front_obstacle_lane_id),
            "front_gap_obstacle_is_source_lane": bool(
                front_obstacle_is_source_lane
            ),
            "snapshot_repr_diag": str(
                [
                    {
                        k: v
                        for k, v in dict(snap).items()
                        if k in (
                            "track_id", "object_id", "vehicle_id",
                            "actor_id", "id", "v", "speed_mps", "x", "y",
                        )
                    }
                    for snap in list(object_snapshots or [])
                ]
            ),
            "route_reference_allowed": bool(route_reference_allowed),
            "route_reference_gate_reason": str(route_reference_gate_reason),
            "route_lane_change_allowed": bool(route_lane_change_allowed),
            "opportunistic_lane_change_allowed": bool(opportunistic_lane_change_allowed),
            "lane_change_gate_reason": str(lane_change_gate_reason),
            "static_obstacle_local_avoidance_active": bool(
                static_obstacle_local_avoidance_active
            ),
            "static_obstacle_local_target_lane_id": (
                ""
                if static_obstacle_local_target_lane_id is None
                else int(static_obstacle_local_target_lane_id)
            ),
            "static_obstacle_candidate_since_s": float(
                self._static_obstacle_candidate_since_s
            ),
            "static_obstacle_global_replan_enabled": bool(
                self.config.get(
                    "static_obstacle_global_replan_enabled",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_global_replan_enabled", False
                    ),
                )
            ),
            "route_lane_change_required": bool(route_lane_change_required),
            "route_progress_s_m": float(self.route_manager.route_progress_s_m),
            "route_progress_lane_index": int(
                self.route_manager.route_progress_lane_index
            ),
            "route_topology_valid": bool(route_topology_validation.valid),
            "route_topology_signature": " -> ".join(
                route_topology_validation.signature
            ),
            "route_topology_errors": ";".join(route_topology_validation.errors),
            "route_topology_warnings": ";".join(route_topology_validation.warnings),
            "route_lane_change_edge_id": str(
                self.maneuver_manager.route_lane_change_edge_id
            ),
            "completed_route_lane_change_edge_id": str(
                self.maneuver_manager.completed_route_lane_change_edge_id
            ),
            "route_lane_change_edge_completed": bool(
                self.maneuver_manager.route_lane_change_edge_completed
            ),
            "route_geometry_lane_change_direction": str(
                route_geometry_lane_change_direction or ""
            ),
            "route_geometry_lane_change_distance_m": float(
                route_geometry_lane_change_distance_m
            ),
            "route_geometry_lane_change_reason": str(
                route_geometry_lane_change_reason
            ),
            "route_physical_target_lane_id": int(
                physical_route_target_lane_id
            ),
            "route_topology_target_lane_id": int(
                topology_route_target_lane_id
            ),
            "behavior_lane_lateral_error_m": float(
                behavior_lane_lateral_error_m
            ),
            "behavior_lane_heading_error_deg": math.degrees(
                float(behavior_lane_heading_error_rad)
            ),
            "behavior_lane_alignment_valid": bool(
                behavior_lane_alignment_valid
            ),
            "behavior_lane_change_completion_allowed": not bool(
                self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()
            ),
            **dict(lane_change_authorization.as_debug_fields()),
            "behavior_override_reason": str(behavior_override_reason),
            # ScenarioManager is the sole turn-intent owner.  The former
            # bridge-local turn_latch_reason variable was removed with that
            # migration; expose the authoritative FSM reason instead.
            "turn_latch_reason": "scenario_manager:" + str(
                scenario_decision.reason
            ),
            "route_current_road_option": str(route_context.current_road_option),
            "route_next_macro_maneuver": str(route_context.next_macro_maneuver),
            "traffic_memory_reason": str(full_traffic_memory_reason),
            "traffic_signal_raw_state": str(
                planner_input_frame.planning.traffic_control.signal_state
            ),
            "traffic_signal_resolved_state": str(resolved_traffic_state),
            "traffic_signal_filtered_state": str(filtered_traffic_state),
            "traffic_signal_behavior_state": str(behavior_traffic_state),
            "traffic_stop_forward_m": float(traffic_stop_forward_m),
            "traffic_stop_commit_distance_m": float(traffic_stop_commit_distance_m),
            "traffic_stop_approach_reason": str(traffic_stop_approach_reason),
            **dict(speed_plan.as_debug_fields()),
            "traffic_signal_state_raw": str(planner_input_frame.planning.traffic_control.signal_state),
            "traffic_signal_state_filtered": str(filtered_traffic_state),
            "candidate_evaluation_summary": str(candidate_frame.summary()),
            "candidate_selected_decision": str(candidate_frame.selected.decision),
            "candidate_selected_lane_id": int(candidate_frame.selected.target_lane_id),
            "candidate_selected_cost": float(candidate_frame.selected.total_cost),
            "mpc_feedback_summary": str(mpc_feedback.get("summary", "")),
            "mpc_feedback_blocked_lane_ids": json.dumps(
                list(mpc_feedback.get("blocked_lane_ids", []) or []),
                default=str,
            ),
            "prediction_trajectories": dict(
                planner_input_frame.prediction.obstacle_future_trajectories
            ),
        })
        reference_debug.update(scenario_decision.as_debug_fields())
        reference_debug.update({
            "route_upcoming_turn_direction": str(upcoming_turn_direction),
            "route_upcoming_turn_distance_m": (
                ""
                if not math.isfinite(float(upcoming_turn_distance_m))
                else float(upcoming_turn_distance_m)
            ),
            "route_upcoming_turn_reason": str(upcoming_turn_reason),
        })
        reference_debug.update(source_quality)
        if bool(self.full_candidate_pipeline_enabled):
            traffic_stop_active = bool(scenario_decision.stop_goal_active)
            behavior_lane_change_proposed = str(decision) in {
                "lane_change_left",
                "lane_change_right",
            }
            opportunistic_lane_change_authorized = bool(
                behavior_lane_change_proposed
                and opportunistic_lane_change_allowed
                and not lane_change_authorized
            )
            candidate_lane_change_authorized = bool(
                lane_change_authorized or opportunistic_lane_change_authorized
            )
            candidate_lane_change_target_lane_id = int(
                lane_change_authorization.target_lane_id
                if lane_change_authorized
                else target_lane_id
            )
            candidate_lane_change_authorization_source = (
                "route" if lane_change_authorized else "opportunistic"
            )
            candidate_intents = build_candidate_intents(
                selected_decision=str(decision),
                selected_target_lane_id=int(target_lane_id),
                current_lane_id=int(current_lane_id),
                target_speed_mps=float(planned_speed_mps),
                candidate_lane_ids=list(candidate_lane_ids),
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
                stop_goal_active=bool(stop_goal_active or scenario_decision.stop_goal_active),
                traffic_stop_active=bool(traffic_stop_active),
                lane_change_authorized=bool(candidate_lane_change_authorized),
                lane_change_authorized_target_lane_id=int(
                    candidate_lane_change_target_lane_id
                ),
                # Route-required and explicitly proposed opportunistic changes
                # share one downstream candidate/reference/MPC gate.
                allow_lane_change_candidates=bool(
                    candidate_lane_change_authorized
                ),
                stop_target=(
                    dict(behavior_stop_target)
                    if isinstance(behavior_stop_target, Mapping)
                    else None
                ),
                lane_change_assertive_duration_s=float(
                    self.candidate_lane_change_assertive_duration_s
                ),
                lane_change_normal_duration_s=float(
                    self.candidate_lane_change_normal_duration_s
                ),
                lane_change_conservative_duration_s=float(
                    self.candidate_lane_change_conservative_duration_s
                ),
                lane_change_assertive_speed_scale=float(
                    self.candidate_lane_change_assertive_speed_scale
                ),
                lane_change_normal_speed_scale=float(
                    self.candidate_lane_change_normal_speed_scale
                ),
                lane_change_conservative_speed_scale=float(
                    self.candidate_lane_change_conservative_speed_scale
                ),
                lane_change_authorization_source=str(
                    candidate_lane_change_authorization_source
                ),
                lane_change_authorization_direction=(
                    str(lane_change_authorization.direction or "")
                    if lane_change_authorized
                    else "left" if str(decision) == "lane_change_left"
                    else "right" if str(decision) == "lane_change_right"
                    else ""
                ),
                lane_change_defer_cost=float(
                    self.config.get("candidate_lane_change_defer_cost", 10.0)
                ),
                turn_obstacle_stop_defer_cost=float(
                    self.config.get("candidate_turn_obstacle_stop_defer_cost", 90.0)
                ),
                local_obstacle_avoidance_active=bool(
                    static_obstacle_local_avoidance_active
                ),
                local_obstacle_stop_defer_cost=float(
                    self.config.get(
                        "candidate_local_obstacle_stop_defer_cost", 25.0
                    )
                ),
                human_like_lane_change_enabled=bool(
                    self.config.get("human_like_lane_change_enabled", True)
                ),
                ego_speed_mps=float(ego_speed_mps),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                lane_change_available_distance_m=(
                    lane_change_authorization.distance_to_maneuver_m
                    if bool(candidate_lane_change_authorized)
                    else None
                ),
                human_lane_change_min_duration_s=float(
                    self.config.get("human_lane_change_min_duration_s", 3.0)
                ),
                human_lane_change_max_duration_s=float(
                    self.config.get("human_lane_change_max_duration_s", 6.5)
                ),
            )
            (
                decision,
                target_lane_id,
                planned_speed_mps,
                local_lane_center_reference,
                nominal_destination_state,
                selected_candidate_debug,
            ) = self._select_candidate_reference_for_mpc(
                candidate_intents=candidate_intents,
                baseline_decision=str(decision),
                baseline_lc_state=str(lc_state),
                baseline_target_lane_id=int(target_lane_id),
                baseline_speed_ref_mps=float(planned_speed_mps),
                baseline_destination_state=nominal_destination_state,
                baseline_reference=local_lane_center_reference,
                baseline_reference_debug=reference_debug,
                base_temporary_destination_state=base_temporary_destination_state,
                previous_nominal_reference=nominal_state.mutable_samples(),
                nominal_reference_freeze_count=int(nominal_freeze_count),
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                ego_pose=ego_pose,
                current_state=current_state,
                current_lane_id=int(current_lane_id),
                route_optimal_lane_id=int(route_optimal_lane_id),
                route_points=route_points,
                route_reference_allowed=bool(route_reference_allowed),
                route_reference_gate_reason=str(route_reference_gate_reason),
                planner_input_frame=planner_input_frame,
                planner_mode=str(planner_mode),
                object_snapshots=object_snapshots,
                upcoming_turn_direction=str(upcoming_turn_direction),
                upcoming_turn_distance_m=float(upcoming_turn_distance_m),
                required_lane_change_decision=(
                    "lane_change_left"
                    if bool(route_lane_change_required)
                    and bool(lane_change_authorized)
                    and str(lane_change_authorization.direction).strip().lower() == "left"
                    else "lane_change_right"
                    if bool(route_lane_change_required)
                    and bool(lane_change_authorized)
                    and str(lane_change_authorization.direction).strip().lower() == "right"
                    else ""
                ),
                required_lane_change_target_lane_id=(
                    int(lane_change_authorization.target_lane_id)
                    if bool(route_lane_change_required) and bool(lane_change_authorized)
                    else 0
                ),
            )
            # Candidate selection owns maneuver geometry, never longitudinal
            # authority.  In particular, do not re-cap a lane-change speed
            # here after SpeedPlanner has selected it.  The curvature-derived
            # value remains diagnostic so a future unified SpeedPlanner can
            # consume it explicitly, but it must not silently rewrite the MPC
            # entry target.  Turns retain their safety cap below because the
            # final turn decision is not known when the earlier speed plan is
            # built; moving that input upstream is a separate change.
            #
            # For turns this is the only place the configured
            # full_intersection_turn_speed_cap_mps actually reaches the
            # winning candidate at all -- confirmed via debug CSV at 35mph:
            # speed climbed past 4 m/s through an entire intersection_turn_left
            # with the cap doing nothing, because only build_speed_plan's
            # (bypassed) turn_cap_mps was ever computed against it.
            if str(decision) in {"lane_change_left", "lane_change_right"}:
                lane_change_curvature_1pm = float(
                    self.reference_generator.discrete_curvature_1pm(
                        local_lane_center_reference
                    )
                )
                lane_change_curvature_cap_mps = curvature_speed_cap_mps(
                    curve_curvature_abs=float(lane_change_curvature_1pm),
                    curve_min_curvature=max(
                        0.0,
                        float(
                            self.config.get(
                                "full_lane_change_curvature_min_curvature_1pm",
                                0.002,
                            )
                        ),
                    ),
                    current_speed_mps=float(ego_speed_mps),
                    curve_lateral_accel_limit_mps2=max(
                        0.1,
                        float(
                            self.config.get(
                                "route_tracking_lane_change_lateral_accel_limit_mps2",
                                1.3,
                            )
                        ),
                    ),
                    speed_enable_threshold_mps=0.0,
                )
                reference_debug.update({
                    "lane_change_reference_curvature_1pm": float(
                        lane_change_curvature_1pm
                    ),
                    "lane_change_curvature_speed_advisory_mps": (
                        ""
                        if lane_change_curvature_cap_mps is None
                        else float(lane_change_curvature_cap_mps)
                    ),
                    "lane_change_longitudinal_authority": "SpeedPlanner",
                })
            elif str(decision) in {"intersection_turn_left", "intersection_turn_right"}:
                # full_intersection_turn_speed_cap_mps is a per-fleet ceiling,
                # not a per-turn comfort speed: two turns at different
                # intersections can have very different connector curvature
                # (confirmed via debug CSV -- this route's right turn measured
                # ~0.145 1/m vs. the left turn's ~0.091 1/m), so a single
                # static cap that is comfortable for a gentle turn can still
                # be too fast for a tighter one, causing the turn's swept
                # vehicle envelope to exceed the drivable corridor and the
                # candidate to be permanently rejected with no fallback.
                # Derive this turn's own cap from its actual winning-candidate
                # curvature and take the tighter of that and the static
                # ceiling.
                turn_ceiling_mps = max(
                    0.1,
                    float(
                        self.config.get("full_intersection_turn_speed_cap_mps", 2.2)
                    ),
                )
                turn_curvature_1pm = float(
                    self.reference_generator.discrete_curvature_1pm(
                        local_lane_center_reference
                    )
                )
                turn_curvature_cap_mps = curvature_speed_cap_mps(
                    curve_curvature_abs=float(turn_curvature_1pm),
                    curve_min_curvature=max(
                        0.0,
                        float(
                            self.config.get(
                                "full_intersection_turn_curvature_min_curvature_1pm",
                                0.01,
                            )
                        ),
                    ),
                    current_speed_mps=float(ego_speed_mps),
                    curve_lateral_accel_limit_mps2=max(
                        0.1,
                        float(
                            self.config.get(
                                "full_intersection_turn_lateral_accel_comfort_mps2",
                                2.5,
                            )
                        ),
                    ),
                    speed_enable_threshold_mps=0.0,
                )
                turn_cap_mps = float(turn_ceiling_mps)
                reference_debug.update({
                    "turn_reference_curvature_1pm": float(turn_curvature_1pm),
                    "turn_curvature_speed_advisory_mps": (
                        "" if turn_curvature_cap_mps is None
                        else float(turn_curvature_cap_mps)
                    ),
                    "turn_longitudinal_authority": "SpeedPlanner",
                })
                selected_turn_constraint = SpeedConstraint(
                    owner="selected_turn_cap",
                    maximum_mps=float(turn_cap_mps),
                    reason="selected_candidate_turn_speed_cap",
                )
                additional_speed_constraints.append(selected_turn_constraint)
                if float(turn_cap_mps) < float(speed_plan.target_speed_mps):
                    speed_plan = dataclasses.replace(
                        speed_plan,
                        target_speed_mps=float(turn_cap_mps),
                        speed_cap_mps=float(turn_cap_mps),
                        turn_cap_mps=float(turn_cap_mps),
                        limiting_owner="selected_turn_cap",
                        active_constraints=tuple(speed_plan.active_constraints)
                        + ("selected_turn_cap",),
                        external_constraints=tuple(speed_plan.external_constraints)
                        + (selected_turn_constraint,),
                    )
            # The upstream front-gap flag proposes an obstacle-stop candidate;
            # it must not remain a global stop latch after a safe lane-change
            # candidate wins. Traffic-control stops remain hard and exclusive.
            stop_goal_active = bool(
                scenario_decision.stop_goal_active
                or selected_candidate_debug.get(
                    "candidate_selected_stop_goal_active",
                    False,
                )
                or str(decision)
                in {"stop_at_intersection", "stop_sign", "emergency_brake"}
            )
            if str(decision) == "lane_follow":
                lc_state = "LANE_KEEP"
            if str(decision) in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
                lc_state = "LANE_KEEP"
            reference_debug.update(selected_candidate_debug)
            # Candidate commitment can restore a lane-change decision after
            # an upstream authorization gate temporarily set the baseline
            # back to lane-follow.  Normalize the FSM from the FINAL decision
            # and locked maneuver phase; otherwise diagnostics and downstream
            # control context can become lane_change_right + LANE_KEEP.
            if str(decision) in {"lane_change_left", "lane_change_right"}:
                selected_phase = str(
                    selected_candidate_debug.get("lane_change_phase", "")
                ).strip().lower()
                lc_state = self._normalized_final_lc_state(
                    decision=str(decision),
                    lc_state=str(lc_state),
                    lane_change_phase=str(selected_phase),
                )
            reference_debug["candidate_pipeline_enabled"] = True
            reference_debug["turn_prepare_speed_suppressed_by_lane_change"] = bool(
                turn_prepare_speed_suppressed_by_lane_change
            )
        else:
            reference_debug["candidate_pipeline_enabled"] = False

        boundary_recovery_active = bool(
            self.config.get("boundary_recovery_enabled", False)
        ) and bool(
            getattr(
                scenario_decision,
                "boundary_recovery_active",
                False,
            )
        )
        if bool(boundary_recovery_active):
            (
                generated_recovery,
                recovery_reference,
                recovery_conditioning_reason,
            ) = self._stable_reference_line_provider.boundary_recovery_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                base_reference_samples=local_lane_center_reference,
                target_speed_mps=float(speed_plan.target_speed_mps),
                horizon_steps=int(self.mpc.horizon_steps),
                dt_s=float(self.mpc.dt_s),
                max_curvature_1pm=float(
                    self.config.get(
                        "boundary_recovery_max_curvature_1pm",
                        self.config.get(
                            "reference_vehicle_max_curvature_1pm",
                            0.22,
                        ),
                    )
                ),
            )
            if recovery_reference:
                local_lane_center_reference = list(recovery_reference)
                terminal = local_lane_center_reference[-1]
                nominal_destination_state = [
                    float(terminal.get("x_ref_m", terminal.get("x", ego_location.x))),
                    float(terminal.get("y_ref_m", terminal.get("y", ego_location.y))),
                    float(speed_plan.target_speed_mps),
                    float(terminal.get("heading_rad", ego_yaw_rad)),
                    int(terminal.get("lane_id", current_lane_id) or current_lane_id),
                ]
                reference_debug.update({
                    "reference_source": "ego_anchored_boundary_recovery",
                    "final_reference_geometry_source": (
                        "ego_anchored_boundary_recovery"
                    ),
                    "stage": "boundary_recovery_reference",
                    "intent_mode": "boundary_recovery",
                    "boundary_recovery_active": True,
                    "boundary_recovery_generation_reason": str(
                        generated_recovery.reason
                    ),
                    "boundary_recovery_conditioning_reason": str(
                        recovery_conditioning_reason
                    ),
                })
            else:
                reference_debug.update({
                    "boundary_recovery_active": True,
                    "boundary_recovery_generation_reason": str(
                        generated_recovery.reason
                    ),
                    "candidate_pipeline_selected_status": "infeasible",
                    "candidate_pipeline_selected_reason": (
                        "boundary_recovery_reference_generation_failed:"
                        + str(generated_recovery.reason)
                    ),
                })
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
        # A completed turn must not hand geometry directly to the generic
        # behavior reference.  On Town06 that reference selected alternating
        # points on opposite sides of the outgoing lane (destination lateral
        # +/-1.95 m), which made MPC steer from one line to the other.  Consume
        # the still-locked turn as the activation token for one immutable
        # AD-map exit-centerline handoff.
        scenario_state = str(getattr(scenario_decision, "state", "")).strip().upper()
        post_turn_snapshot = self._stable_reference_line_provider.snapshot(POST_TURN)
        hold_arc_m = max(
            0.0,
            float(self.config.get("post_turn_exit_reference_arc_m", 12.0)),
        )
        exit_aligned = bool(
            behavior_lane_alignment_valid
            and abs(float(behavior_lane_lateral_error_m))
            <= float(self.config.get("post_turn_exit_max_lateral_m", 0.35))
            and abs(float(behavior_lane_heading_error_rad))
            <= math.radians(
                float(self.config.get("post_turn_exit_max_heading_error_deg", 5.0))
            )
        )
        post_turn_action = self.maneuver_manager.resolve_post_turn_phase(
            decision=str(decision),
            scenario_state=str(scenario_state),
            turn_reference_active=bool(
                self._stable_reference_line_provider.snapshot(TURN).active
            ),
            post_turn_reference_active=bool(post_turn_snapshot.active),
            travelled_s_m=float(post_turn_snapshot.travelled_s_m),
            required_s_m=float(hold_arc_m),
            exit_aligned=bool(exit_aligned),
        )
        if str(post_turn_action) == "activate":
            activated = self._start_post_turn_exit_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                target_speed_mps=float(planned_speed_mps),
            )
            if bool(activated):
                self._clear_turn_master_reference()

        post_turn_snapshot = self._stable_reference_line_provider.snapshot(
            POST_TURN
        )
        post_turn_exit_active = bool(post_turn_snapshot.active)
        if bool(post_turn_exit_active):
            if str(post_turn_action) == "complete":
                self._clear_post_turn_exit_reference()
                post_turn_exit_active = False
            else:
                exit_reference, exit_reason = self._post_turn_exit_reference_window(
                    ego_location=ego_location,
                    target_speed_mps=float(planned_speed_mps),
                )
                if exit_reference:
                    from opencda.planning_module.behavior_planner.reference_pipeline import (
                        lane_center_destination_from_reference_arc_length,
                    )

                    local_lane_center_reference = [
                        dict(sample) for sample in exit_reference
                    ]
                    seed_destination = list(nominal_destination_state or [])
                    if len(seed_destination) < 5:
                        seed_destination = [
                            float(current_state[0]),
                            float(current_state[1]),
                            float(planned_speed_mps),
                            float(current_state[3]),
                            int(current_lane_id),
                        ]
                    seed_destination[2] = float(planned_speed_mps)
                    seed_destination[4] = int(
                        post_turn_snapshot.target_lane_id or current_lane_id
                    )
                    nominal_destination_state = (
                        lane_center_destination_from_reference_arc_length(
                            destination_state=seed_destination,
                            lane_center_reference=local_lane_center_reference,
                            target_arc_length_m=float(
                                self.config.get(
                                    "post_turn_exit_destination_arc_m", 6.0
                                )
                            ),
                        )
                        or seed_destination
                    )
                    reference_debug["reference_source"] = (
                        "admap_post_turn_exit_centerline"
                    )
                    reference_debug["final_reference_geometry_source"] = (
                        "admap_post_turn_exit_centerline"
                    )
                    reference_debug["post_turn_exit_reason"] = str(exit_reason)
                    reference_debug["post_turn_exit_reference_source"] = str(
                        post_turn_snapshot.build_reason
                    )

        committed_lane_change_reference_active = bool(
            str(decision) in {"lane_change_left", "lane_change_right"}
            and self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()
        )
        lateral_guard_reason = ""
        if (
            not bool(committed_lane_change_reference_active)
            and not bool(boundary_recovery_active)
        ):
            lateral_guard_reason = self._full_reference_lateral_guard_reason(
                decision=str(decision),
                lc_state=str(lc_state),
                stop_goal_active=bool(stop_goal_active),
                destination_state=nominal_destination_state,
                lane_center_reference=local_lane_center_reference,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                heading_error_rad=(
                    float(behavior_lane_heading_error_rad)
                    if bool(behavior_lane_alignment_valid)
                    else float("nan")
                ),
            )
        if str(lateral_guard_reason):
            # This guard is diagnostic only. Geometry ownership remains with
            # the already conditioned behavior/AD-map reference. Replacing it
            # for a few frames produced a 0.9 m destination jump and then
            # switched straight back, which made an otherwise valid MPC QP
            # infeasible at lane-change completion.
            reference_debug["lateral_guard_validation"] = "warning"
        reference_debug["reference_lateral_guard_reason"] = str(lateral_guard_reason)
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
            self.behavior_stage.finalize(
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
                diagnostics={
                "lane_safety_scores": dict(lane_safety_scores),
                "traffic_signal_raw_state": str(
                    planner_input_frame.planning.traffic_control.signal_state
                ),
                "traffic_signal_resolved_state": str(resolved_traffic_state),
                "traffic_signal_filtered_state": str(filtered_traffic_state),
                "traffic_signal_behavior_state": str(behavior_traffic_state),
                "traffic_control_from_cp": bool(planner_input_frame.planning.traffic_control.from_cp),
                "boundary_recovery_scenario_state": str(
                    scenario_decision.state
                ),
                },
            ),
            reference_debug,
            speed_plan,
        )

    def _cooperative_lane_change_yield_reason(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
    ) -> str:
        """Hold in lane if a nearby CPX-controlled peer is already mid-lane-change.

        Both CAVs independently deciding to change lanes at the same moment
        near each other is exactly the situation that produced the
        multi-CAV mutual-interference gridlock diagnosed in Construction_Zone
        testing. Serialize on physical order instead: whichever CAV is
        already committed to a lane change goes first; a trailing peer holds
        lane_follow until that commitment clears (state resets to IDLE, which
        stops being broadcast as active -- see ManeuverCommitment.active).

        Thin wrapper around the shared cooperative_arbitration module: any
        active peer lane-change conflicts with ego's own (resource_id is a
        constant, not the specific lane, since two CAVs changing lanes near
        each other at the same time is the thing being serialized,
        regardless of which lanes are involved).
        """
        if not bool(
            self.config.get("cooperative_lane_change_yield_enabled", True)
        ):
            return ""
        v2x_manager = getattr(self.vehicle_manager, "v2x_manager", None)
        cav_intents = dict(getattr(v2x_manager, "cav_intents", {}) or {})
        cav_nearby = dict(getattr(v2x_manager, "cav_nearby", {}) or {})
        if not cav_intents or not cav_nearby:
            return ""
        from opencda.planning_module.pipeline.cooperative_arbitration import (
            ResourceClaim,
            should_yield,
        )

        peers: list[tuple[int, ResourceClaim, tuple[float, float]]] = []
        for peer_id, message in cav_intents.items():
            if not isinstance(message, Mapping):
                continue
            decision = str(message.get("maneuver_commitment_decision", ""))
            if decision not in ("lane_change_left", "lane_change_right"):
                continue
            peer_manager = cav_nearby.get(str(peer_id))
            peer_vehicle = getattr(peer_manager, "vehicle", None)
            if peer_vehicle is None:
                continue
            try:
                peer_location = peer_vehicle.get_location()
            except Exception:
                continue
            try:
                peer_actor_id = int(peer_id)
            except (TypeError, ValueError):
                continue
            peers.append((
                peer_actor_id,
                ResourceClaim(
                    kind="lane_change",
                    resource_id="lane_change",
                    committed_at_s=float(
                        message.get("maneuver_commitment_committed_at_s", 0.0) or 0.0
                    ),
                    active=bool(message.get("maneuver_commitment_active", False)),
                ),
                (float(peer_location.x), float(peer_location.y)),
            ))
        if not peers:
            return ""
        my_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=float(self._sim_time_s()),
            active=True,
        )
        my_actor_id = int(getattr(self.vehicle_manager.vehicle, "id", -1))
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=my_actor_id,
            my_position_xy=(float(ego_location.x), float(ego_location.y)),
            my_heading_rad=float(ego_yaw_rad),
            peers=peers,
            range_m=float(
                self.config.get("cooperative_lane_change_yield_range_m", 40.0)
            ),
        )
        return str(reason) if reason else ""

    def _cooperative_wait_speed_cap_mps(
        self,
        *,
        ego_location: carla.Location,
        ego_speed_mps: float,
        cooperative_lane_change_yield_reason: str,
    ) -> Optional[float]:
        """Cap speed while queued behind a peer's lane change.

        ``_cooperative_lane_change_yield_reason`` already holds ego's own
        lane change back until the peer clears -- necessary but not
        sufficient. That peer is normally in an ADJACENT lane, outside
        ego's own-lane ``_front_gap_m`` search cone, so the ordinary
        following-cap in speed_planner.py never sees it and has no reason
        to slow down for it. Left unconstrained, ego keeps accelerating
        toward its full cruise target while waiting, closes the real gap
        to the peer it intends to merge behind, and by the time its own
        turn opens up the gap has fallen under trajectory_risk.py's
        min_front_gap_m -- so the now-authorized lane change gets denied
        by target_lane_prediction_risk and is missed once the route's own
        lane-change requirement lapses (diagnosed via Interactive_Lane_
        Change telemetry: gap fell from ~8.3m to ~6.5m across the wait
        window). This does not touch that prediction-risk check at all;
        it just stops ego from closing the gap in the first place while
        it has nowhere to go yet.

        The trigger distance is deliberately larger than trajectory_risk.
        py's own min_front_gap_m (8.0m default): reusing that exact value
        here gave this cap zero lead time -- telemetry showed the yield
        reason (and therefore this function) only ever starts firing once
        the gap has *already* dropped to ~7.9m, one tick past the hard
        floor, so there was never a tick left where capping ego's speed
        could still have prevented the gap sliding on down to ~6.5-6.9m
        and tripping target_lane_prediction_risk. A separate, wider
        trigger (cooperative_wait_trigger_gap_m, default 15.0m) gives the
        cap several seconds of runway to hold ego at the peer's speed
        before the hard threshold is anywhere close.
        """
        if not cooperative_lane_change_yield_reason:
            return None
        match = re.search(r"peer=(-?\d+)", cooperative_lane_change_yield_reason)
        if match is None:
            return None
        peer_id = match.group(1)
        v2x_manager = getattr(self.vehicle_manager, "v2x_manager", None)
        cav_nearby = dict(getattr(v2x_manager, "cav_nearby", {}) or {})
        peer_manager = cav_nearby.get(str(peer_id))
        peer_vehicle = getattr(peer_manager, "vehicle", None)
        if peer_vehicle is None:
            return None
        try:
            peer_location = peer_vehicle.get_location()
            peer_velocity = peer_vehicle.get_velocity()
        except Exception:
            return None
        peer_speed_mps = math.sqrt(
            float(peer_velocity.x) ** 2
            + float(peer_velocity.y) ** 2
            + float(peer_velocity.z) ** 2
        )
        distance_m = math.hypot(
            float(peer_location.x) - float(ego_location.x),
            float(peer_location.y) - float(ego_location.y),
        )
        # Both floors below are flat distances that don't scale with
        # cruise speed -- also give them the same reaction-time margin
        # regardless of configured cruise speed, matching min_front_gap_m's
        # own speed scaling in planner_input_adapter.py. Scaled off the
        # *configured* cruise target (self.target_speed_mps), not ego's
        # live instantaneous speed -- this wait window happens while ego
        # is still mid-acceleration toward that target, so scaling off
        # the live speed barely moved either floor at the moment it
        # mattered (confirmed via telemetry: identical denial, identical
        # distances down to the decimal, before and after that version).
        min_gap_m = max(
            0.5,
            float(self.target_speed_mps) * float(self.min_front_gap_time_s),
            float(self.config.get("cooperative_wait_min_gap_m", 8.0)),
        )
        trigger_gap_m = max(
            float(min_gap_m),
            float(self.target_speed_mps)
            * float(self.config.get("cooperative_wait_trigger_time_s", 15.0 / 11.18)),
            float(self.config.get("cooperative_wait_trigger_gap_m", 15.0)),
        )
        if float(distance_m) >= float(trigger_gap_m):
            return None
        # min(ego_speed, peer_speed) was the original cap here, but it does
        # nothing when both CAVs are ramping up toward the same cruise
        # target in near lockstep from a similar start (confirmed via
        # telemetry at 20 m/s cruise: peer's speed tracked ego's own climb
        # tick-for-tick, ~7->11 m/s over the same 2s window, so "cap at
        # peer's speed" never actually differed from where ego was already
        # headed -- three separate threshold-tuning attempts on the
        # trigger/min-gap distances above produced bit-identical
        # trajectories because the actual constraining value never
        # changed). Reuse the same IDM model used for ordinary front-
        # vehicle following instead: it reacts to the actual gap being
        # smaller than the desired safe spacing even when closing speed is
        # ~0, which a plain speed-match can't express.
        from opencda.planning_module.behavior_planner.car_follow import (
            idm_acceleration as _cooperative_wait_idm_acceleration,
        )

        idm_accel = _cooperative_wait_idm_acceleration(
            v=float(ego_speed_mps),
            v_lead=max(0.0, float(peer_speed_mps)),
            gap_m=max(0.1, float(distance_m)),
            v_desired=max(0.1, float(self.target_speed_mps)),
            a_max=max(
                0.05,
                float(self.config.get("following_idm_max_acceleration_mps2", 2.0)),
            ),
            b_comfort=max(
                0.05,
                float(
                    self.config.get(
                        "following_idm_comfort_deceleration_mps2", 3.0
                    )
                ),
            ),
            time_headway_s=max(
                0.05, float(self.config.get("following_time_headway_s", 1.5))
            ),
            min_gap_m=float(min_gap_m),
            delta=max(
                1.0, float(self.config.get("following_idm_acceleration_exponent", 4.0))
            ),
        )
        cap_horizon_s = max(
            0.05, float(self.config.get("cooperative_wait_cap_horizon_s", 1.0))
        )
        return max(0.0, float(ego_speed_mps) + float(idm_accel) * float(cap_horizon_s))

    def _cooperative_avoidance_lane_yield_reason(
        self,
        *,
        target_lane_id: int,
        ego_location: carla.Location,
        ego_yaw_rad: float,
    ) -> str:
        """Hold back if a peer CAV already claimed this exact avoidance lane.

        Construction_Zone testing with every CAV controlled surfaced a
        multi-CAV gridlock: several CAVs converge on the same one or two
        usable bypass lanes at once, so each one's lane_safety_scores for
        that lane stays low (correctly -- a peer really is right there) and
        nobody ever moves, forever, since nothing breaks the symmetry.
        Rather than blind the safety scorer to peer CAVs (a peer stopped in
        your target lane is a real hazard, CAV or not), arbitrate who is
        even allowed to attempt this specific lane: whichever CAV has been
        blocked by its obstacle the longest goes first (a reasonable stand-in
        for "committed first", since local-avoidance commitment itself is
        decided in the same step this reads); the rest hold in place and
        re-check every tick, so as soon as the leader clears the lane (moves
        through, or its own commitment resets) the next one in line takes
        its turn instead of everyone staying wedged forever.
        """
        if not bool(
            self.config.get("cooperative_avoidance_lane_yield_enabled", True)
        ):
            return ""
        v2x_manager = getattr(self.vehicle_manager, "v2x_manager", None)
        cav_intents = dict(getattr(v2x_manager, "cav_intents", {}) or {})
        cav_nearby = dict(getattr(v2x_manager, "cav_nearby", {}) or {})
        if not cav_intents or not cav_nearby:
            return ""
        from opencda.planning_module.pipeline.cooperative_arbitration import (
            ResourceClaim,
            should_yield,
        )

        resource_id = str(int(target_lane_id))
        peers: list[tuple[int, ResourceClaim, tuple[float, float]]] = []
        for peer_id, message in cav_intents.items():
            if not isinstance(message, Mapping):
                continue
            if not bool(message.get("static_obstacle_local_avoidance_active", False)):
                continue
            peer_target_lane_id = message.get("static_obstacle_local_target_lane_id", "")
            if str(peer_target_lane_id) != resource_id:
                continue
            peer_manager = cav_nearby.get(str(peer_id))
            peer_vehicle = getattr(peer_manager, "vehicle", None)
            if peer_vehicle is None:
                continue
            try:
                peer_location = peer_vehicle.get_location()
            except Exception:
                continue
            try:
                peer_actor_id = int(peer_id)
            except (TypeError, ValueError):
                continue
            peers.append((
                peer_actor_id,
                ResourceClaim(
                    kind="avoidance_lane",
                    resource_id=resource_id,
                    committed_at_s=float(
                        message.get("static_obstacle_candidate_since_s", 0.0) or 0.0
                    ),
                    active=True,
                    require_ahead=False,
                ),
                (float(peer_location.x), float(peer_location.y)),
            ))
        if not peers:
            return ""
        my_claim = ResourceClaim(
            kind="avoidance_lane",
            resource_id=resource_id,
            committed_at_s=float(self._static_obstacle_candidate_since_s),
            active=True,
            require_ahead=False,
        )
        my_actor_id = int(getattr(self.vehicle_manager.vehicle, "id", -1))
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=my_actor_id,
            my_position_xy=(float(ego_location.x), float(ego_location.y)),
            my_heading_rad=float(ego_yaw_rad),
            peers=peers,
            range_m=float(
                self.config.get("cooperative_avoidance_lane_yield_range_m", 40.0)
            ),
        )
        return str(reason) if reason else ""

    def _route_tracking_lane_change_window(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        target_speed_mps: float,
        step_distance_m: float,
    ) -> tuple[list[dict[str, object]], str]:
        """Advance monotonically over the locked lane-change trajectory."""

        min_first_forward_m = float(
            self.config.get(
                "reference_contract_lane_change_min_first_forward_m",
                0.2,
            )
        )
        stable_provider = getattr(
            self, "_stable_reference_line_provider", ReferenceLineProvider()
        )
        stable_window = stable_provider.locked_lane_change_window(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            target_lane_id=int(self.maneuver_manager.lane_change.target_lane_id),
            target_speed_mps=float(target_speed_mps),
            spacing_m=max(0.05, float(step_distance_m)),
            count=int(self.mpc.horizon_steps),
            min_first_forward_m=float(min_first_forward_m),
            anchor_margin_m=float(
                self.config.get(
                    "lane_change_reference_anchor_margin_m", 0.05
                )
            ),
            max_projection_advance_m=max(2.0, 2.0 * float(step_distance_m)),
        )
        self._stable_reference_line_provider = stable_provider
        self.maneuver_manager.advance_lane_change(
            progress_s_m=float(stable_window.projection_s_m)
        )
        best_index = int(stable_window.master_index)
        self.maneuver_manager.advance_lane_change(progress_index=int(best_index))
        window = [dict(sample) for sample in stable_window.samples]
        progress_pairs = self.maneuver_manager.lane_change.progress_pairs
        if window and not progress_pairs:
            sample = window[0]
            required = (
                "lane_change_source_x_m", "lane_change_source_y_m",
                "lane_change_target_x_m", "lane_change_target_y_m",
            )
            if all(key in sample for key in required):
                from opencda.planning_module.pipeline.candidate_pipeline import (
                    _lane_change_initial_progress,
                )
                self.maneuver_manager.advance_lane_change(
                    progress=_lane_change_initial_progress(
                        source_sample={
                            "x_ref_m": sample["lane_change_source_x_m"],
                            "y_ref_m": sample["lane_change_source_y_m"],
                        },
                        target_sample={
                            "x_ref_m": sample["lane_change_target_x_m"],
                            "y_ref_m": sample["lane_change_target_y_m"],
                        },
                        ego_x_m=float(ego_location.x),
                        ego_y_m=float(ego_location.y),
                    )
                )
        if progress_pairs:
            # Under direct target-lane tracking (progress_pairs populated),
            # the per-sample "lane_change_progress" tag is a stale
            # time-schedule, not a genuine crossing measurement -- taking
            # its max against a live geometric read would let one bad
            # (over-reported) schedule value permanently inflate progress,
            # since this accumulator is monotonic. Measure live progress
            # instead, monotonic only against its own prior value.
            from opencda.planning_module.pipeline.candidate_pipeline import (
                _lane_change_initial_progress,
            )

            pair_index = min(int(best_index), len(progress_pairs) - 1)
            source_sample, target_sample = progress_pairs[pair_index]
            live_progress = _lane_change_initial_progress(
                source_sample=source_sample,
                target_sample=target_sample,
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
            )
            self.maneuver_manager.advance_lane_change(progress=float(live_progress))
        return (
            window,
            "lane_change_locked_window:"
            f"phase={str(self.maneuver_manager.lane_change.phase)}:"
            f"index={int(best_index)}:s={float(stable_window.start_s_m):.2f}:"
            f"provider={stable_window.reason}:"
            f"progress={float(self.maneuver_manager.lane_change.progress):.3f}",
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

    def _reset_route_tracking_lane_change_reference(self) -> None:
        self.maneuver_manager.lane_change.reset()
        self._stable_reference_line_provider.release(
            LANE_CHANGE, event="reset"
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

    def _select_candidate_reference_for_mpc(
        self,
        *,
        candidate_intents: Sequence[object],
        baseline_decision: str,
        baseline_lc_state: str,
        baseline_target_lane_id: int,
        baseline_speed_ref_mps: float,
        baseline_destination_state: Sequence[float] | None,
        baseline_reference: Sequence[Mapping[str, object]],
        baseline_reference_debug: Mapping[str, object],
        base_temporary_destination_state: Sequence[float] | None,
        previous_nominal_reference: Sequence[Mapping[str, object]],
        nominal_reference_freeze_count: int,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        ego_pose: Mapping[str, object],
        current_state: Sequence[float],
        current_lane_id: int,
        route_optimal_lane_id: int,
        route_points: Sequence[Sequence[float]],
        route_reference_allowed: bool,
        route_reference_gate_reason: str,
        planner_input_frame: Any,
        planner_mode: str,
        object_snapshots: Sequence[Mapping[str, object]],
        upcoming_turn_direction: str,
        upcoming_turn_distance_m: float,
        required_lane_change_decision: str = "",
        required_lane_change_target_lane_id: int = 0,
    ) -> tuple[str, int, float, list[dict[str, object]], list[float], dict[str, object]]:
        from opencda.planning_module.behavior_planner import (
            MpcReferenceGenerationContext,
            compute_temp_destination,
            generate_mpc_reference,
            select_reference_intent,
        )
        from opencda.planning_module.pipeline.candidate_pipeline import (
            CandidateBehaviorIntent,
            CandidateReferenceResult,
            apply_mpc_probe_result,
            evaluate_candidate_reference,
            mark_mpc_probe_skipped,
            predicted_lane_change_average_speed_mps,
            lane_change_geometry_requirements,
            lane_change_operational_curvature_limit_1pm,
            select_best_candidate,
            select_candidate_with_commitment,
            summarize_candidate_results,
        )
        from opencda.planning_module.pipeline.stage_contracts import (
            ManeuverCommitment,
        )
        from opencda.planning_module.pipeline.reference_pipeline import (
            ReferencePipelineRequest,
        )

        lane_change_commitment_release_reason = (
            self._release_completed_lane_change_commitment(
                current_lane_id=int(current_lane_id),
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
            )
        )
        intents = list(candidate_intents or [])
        if self.maneuver_manager.route_lane_change_edge_completed:
            # Completion can be detected at the start of this very candidate
            # tick.  Remove the just-completed edge immediately; waiting for
            # next-tick authorization lets commitment selection install the
            # same maneuver a second time with a downstream lane as target.
            intents = [
                intent
                for intent in intents
                if str(getattr(intent, "decision", ""))
                not in {"lane_change_left", "lane_change_right"}
            ]
        prediction_trajectories = dict(
            planner_input_frame.prediction.obstacle_future_trajectories
        )
        if not intents:
            return (
                str(baseline_decision),
                int(baseline_target_lane_id),
                float(baseline_speed_ref_mps),
                [dict(sample) for sample in list(baseline_reference or [])],
                list(baseline_destination_state or []),
                {
                    "candidate_pipeline_selected": "baseline_no_candidates",
                    "candidate_pipeline_selected_status": "feasible",
                    "candidate_pipeline_selected_reason": "",
                    "candidate_pipeline_count": 0,
                    "candidate_prediction_trajectory_count": int(len(prediction_trajectories)),
                    "candidate_pipeline_summary": "[]",
                },
            )

        candidate_results = []
        keep_lane_reference: list[dict[str, object]] = []
        for intent in intents:
            candidate_decision = str(getattr(intent, "decision", baseline_decision))
            candidate_target_lane_id = int(getattr(intent, "target_lane_id", current_lane_id) or current_lane_id)
            candidate_speed_ref_mps = float(getattr(intent, "target_speed_mps", baseline_speed_ref_mps))
            route_required_candidate = bool(
                str(required_lane_change_decision)
                and str(candidate_decision) == str(required_lane_change_decision)
                and int(candidate_target_lane_id)
                == int(required_lane_change_target_lane_id)
            )
            candidate_stop_goal_active = bool(getattr(intent, "stop_goal_active", False)) or candidate_decision in {
                "stop_at_intersection",
                "stop_sign",
                "emergency_brake",
            }
            candidate_lc_state = self._candidate_lc_state(
                decision=str(candidate_decision),
                baseline_decision=str(baseline_decision),
                baseline_lc_state=str(baseline_lc_state),
            )
            candidate_lane_reference_step_m = max(
                float(self.config.get("route_tracking_min_step_m", 0.10)),
                float(self.mpc.dt_s)
                * max(
                    0.5,
                    float(ego_speed_mps),
                    abs(float(candidate_speed_ref_mps)),
                ),
            )
            candidate_geometry_length_m = 0.0
            candidate_geometry_speed_mps = 0.0
            if str(candidate_decision) in {"lane_change_left", "lane_change_right"}:
                operational_curvature_limit_1pm = (
                    lane_change_operational_curvature_limit_1pm(
                        planning_speed_mps=float(candidate_speed_ref_mps),
                        lateral_accel_limit_mps2=float(
                            self.config.get(
                                "route_tracking_lane_change_lateral_accel_limit_mps2",
                                1.3,
                            )
                        ),
                        vehicle_max_curvature_1pm=float(
                            self.config.get(
                                "reference_vehicle_max_curvature_1pm", 0.35
                            )
                        ),
                        minimum_speed_mps=float(
                            self.config.get("lane_change_min_geometry_speed_mps", 2.0)
                        ),
                    )
                )
                (
                    candidate_geometry_speed_mps,
                    candidate_geometry_length_m,
                    candidate_geometry_step_m,
                ) = lane_change_geometry_requirements(
                    ego_speed_mps=float(ego_speed_mps),
                    target_speed_mps=float(candidate_speed_ref_mps),
                    duration_s=float(
                        getattr(intent, "lane_change_duration_s", 4.0) or 4.0
                    ),
                    dt_s=float(self.mpc.dt_s),
                    lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                    max_curvature_1pm=float(operational_curvature_limit_1pm),
                    minimum_geometry_speed_mps=float(
                        self.config.get("lane_change_min_geometry_speed_mps", 2.0)
                    ),
                    minimum_length_m=float(
                        self.config.get("lane_change_min_length_m", 10.0)
                    ),
                    acceleration_limit_mps2=float(
                        self.config.get(
                            "lane_change_planning_acceleration_limit_mps2", 2.0
                        )
                    ),
                )
                candidate_lane_reference_step_m = max(
                    float(candidate_lane_reference_step_m),
                    float(candidate_geometry_step_m),
                )
            same_as_baseline = (
                str(candidate_decision) == str(baseline_decision)
                and int(candidate_target_lane_id) == int(baseline_target_lane_id)
                and abs(float(candidate_speed_ref_mps) - float(baseline_speed_ref_mps)) < 1.0e-3
                # Lane-change candidates require the geometry floor computed
                # below. The generic baseline may have been sampled from the
                # near-zero controller speed and is therefore not an
                # equivalent reference even when decision/target/speed match.
                and str(candidate_decision)
                not in {"lane_change_left", "lane_change_right"}
            )
            if bool(same_as_baseline) and baseline_destination_state is not None:
                destination_state = list(baseline_destination_state)
                reference = [dict(sample) for sample in list(baseline_reference or [])]
                candidate_reference_debug = dict(baseline_reference_debug or {})
            else:
                previous_temp = list(base_temporary_destination_state or [])
                candidate_temp_destination = compute_temp_destination(
                    map_planner=self.reference_map,
                    ego_pose=ego_pose,
                    target_lane_id=int(candidate_target_lane_id),
                    decision=str(candidate_decision),
                    lookahead_m=float(self.lookahead_m),
                    target_v_mps=float(candidate_speed_ref_mps),
                    global_route_points=route_points,
                    mode_reference_xy=(
                        None
                        if not previous_temp
                        else (float(previous_temp[0]), float(previous_temp[1]))
                    ),
                    prev_mode=(
                        None
                        if len(previous_temp) < 6
                        else float(previous_temp[5])
                    ),
                    prev_road_id=(
                        None
                        if len(previous_temp) < 7
                        else int(previous_temp[6])
                    ),
                    prev_entered_intersection=(
                        False
                        if len(previous_temp) < 8
                        else bool(float(previous_temp[7]) > 0.5)
                    ),
                    next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
                    mode_override=str(planner_mode),
                    follow_global_route_lane=bool(
                        route_reference_allowed and planner_input_frame.map_lane.in_junction
                    ),
                )
                reference_intent = select_reference_intent(
                    behavior_decision=str(candidate_decision),
                    planner_fsm_state=str(candidate_lc_state),
                    ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
                    reference_target_lane_id=int(candidate_target_lane_id),
                    current_lane_id=int(current_lane_id),
                    route_optimal_lane_id=int(route_optimal_lane_id),
                    global_route_reference_allowed=bool(route_reference_allowed),
                    traffic_control_lane_lock_active=False,
                )
                ref_context = MpcReferenceGenerationContext(
                    map_planner=self.reference_map,
                    ego_pose=ego_pose,
                    ego_state=current_state,
                    active_global_route_points=route_points,
                    previous_lane_center_reference=previous_nominal_reference,
                    behavior_runtime_cfg=self.behavior_runtime_cfg,
                    reference_intent=reference_intent,
                    current_applied_behavior=str(candidate_decision),
                    cached_planner_lc_state=str(candidate_lc_state),
                    reference_target_lane_id=int(candidate_target_lane_id),
                    current_lane_id=int(current_lane_id),
                    global_route_reference_allowed=bool(route_reference_allowed),
                    global_route_reference_gate_reason=str(route_reference_gate_reason),
                    should_follow_global_route_lane_for_reference=bool(reference_intent.follow_global_route_lane),
                    traffic_control_lane_lock_active=False,
                    final_goal_stop_active=False,
                    stop_target_state=None,
                    follow_target_state=None,
                    current_temp_reference_xy=(
                        float(candidate_temp_destination[0]),
                        float(candidate_temp_destination[1]),
                    ),
                    current_temp_mode_value=(
                        float(candidate_temp_destination[5])
                        if len(candidate_temp_destination) >= 6 else 0.0
                    ),
                    current_temp_road_id=(
                        int(candidate_temp_destination[6])
                        if len(candidate_temp_destination) >= 7 else None
                    ),
                    current_temp_entered_intersection=(
                        bool(float(candidate_temp_destination[7]) > 0.5)
                        if len(candidate_temp_destination) >= 8 else False
                    ),
                    active_reference_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
                    current_temp_mode_str=str(planner_mode),
                    lane_reference_speed_mps=max(
                        1.0,
                        float(ego_speed_mps),
                        abs(float(candidate_speed_ref_mps)),
                    ),
                    lane_reference_step_distance_m=max(
                        0.05,
                        float(candidate_lane_reference_step_m),
                    ),
                    mpc_horizon_steps=int(self.mpc.horizon_steps),
                    mpc_dt_s=float(self.mpc.dt_s),
                    temporary_destination_state=candidate_temp_destination,
                    lane_reference_freeze_count=int(nominal_reference_freeze_count),
                    sim_time_s=float(self._sim_time_s()),
                    stop_release_temp_smooth_until_sim_time_s=float(self._stop_release_temp_smooth_until_sim_time_s),
                    authoritative_ego_waypoint=self._authoritative_ego_waypoint,
                )
                ref_output = generate_mpc_reference(ref_context)
                destination_state = list(ref_output.temporary_destination_state or candidate_temp_destination)
                reference = [dict(sample) for sample in list(ref_output.local_lane_center_reference or [])]
                candidate_reference_debug = dict(ref_output.mpc_reference_result.trace.as_trace_fields())
                candidate_reference_debug["fallback_reason"] = str(ref_output.last_reference_fallback_reason)

            if bool(route_required_candidate) and len(route_points) >= 2:
                # ReferenceLineProvider owns the geometry for exactly one
                # physical lateral transition.  A later topology target must
                # not be baked into this maneuver's persistent reference.
                lane_reference_step_m = max(
                    0.05,
                    float(candidate_lane_reference_step_m),
                )
                provider = self._stable_reference_line_provider
                lane_change_master_count = max(
                    int(self.mpc.horizon_steps),
                    int(math.ceil(
                        float(candidate_geometry_length_m)
                        / max(0.05, float(lane_reference_step_m))
                    )) + int(self.mpc.horizon_steps),
                )
                target_master, target_anchor_reason = (
                    provider.lane_change_target_reference(
                        getattr(self, "_local_map_snapshot", None),
                        target_lane_id=int(candidate_target_lane_id),
                        target_speed_mps=float(candidate_speed_ref_mps),
                    )
                )
                target_window = provider.window_from_reference(
                    target_master,
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    lower_s_m=0.0,
                    first_forward_m=float(lane_reference_step_m),
                    spacing_m=float(lane_reference_step_m),
                    count=int(lane_change_master_count),
                ) if target_master else None
                continuous_target_reference = (
                    [dict(sample) for sample in target_window.samples]
                    if target_window is not None else []
                )
                if continuous_target_reference:
                    reference = [dict(sample) for sample in continuous_target_reference]
                    candidate_reference_debug.update({
                        "reference_source": "local_map_target_corridor_center",
                        "admap_target_lane_resolved": True,
                        "admap_target_lane_reason": str(target_anchor_reason),
                    })
                else:
                    candidate_reference_debug.update({
                        "admap_target_lane_resolved": False,
                        "admap_target_lane_reason": str(target_anchor_reason),
                    })

            if candidate_decision in {"intersection_turn_left", "intersection_turn_right"}:
                turn_reference, turn_destination, turn_reference_reason = (
                    self._waypoint_turn_reference(
                        ego_location=ego_location,
                        ego_yaw_rad=float(ego_yaw_rad),
                        current_state=current_state,
                        current_lane_id=int(current_lane_id),
                        target_lane_id=int(candidate_target_lane_id),
                        target_speed_mps=float(candidate_speed_ref_mps),
                        destination_state=destination_state,
                        turn_direction=(
                            "left" if candidate_decision.endswith("_left") else "right"
                        ),
                    )
                )
                if turn_reference:
                    raw_turn_forward_m, raw_turn_lateral_m = self._body_frame_xy(
                        origin_x_m=float(ego_location.x),
                        origin_y_m=float(ego_location.y),
                        heading_rad=float(ego_yaw_rad),
                        target_x_m=float(turn_reference[0].get(
                            "x_ref_m", turn_reference[0].get("x", ego_location.x)
                        )),
                        target_y_m=float(turn_reference[0].get(
                            "y_ref_m", turn_reference[0].get("y", ego_location.y)
                        )),
                    )
                    reference = [dict(sample) for sample in turn_reference]
                    destination_state = list(turn_destination)
                    candidate_reference_debug.update({
                        "reference_pipeline_stage": "waypoint_turn",
                        "reference_pipeline_intent": str(candidate_decision),
                        "reference_pipeline_intent_mode": "intersection_turn",
                        "reference_pipeline_follow_global_route_lane": 1,
                        "reference_source": "admap_waypoint_turn",
                        "fallback_reason": "",
                        "route_turn_reference_reason": str(turn_reference_reason),
                        "route_turn_raw_first_forward_m": float(raw_turn_forward_m),
                        "route_turn_raw_first_lateral_m": float(raw_turn_lateral_m),
                    })
                else:
                    candidate_reference_debug["route_turn_reference_reason"] = str(
                        turn_reference_reason
                    )

            if (
                candidate_decision == "lane_follow"
                and str(upcoming_turn_direction) in {"left", "right"}
                and math.isfinite(float(upcoming_turn_distance_m))
            ):
                # PREPARE_TURN geometry is owned by the continuously matched
                # current AD-map lane, not by the global route polyline.  The
                # latter is topology and may contain connector smoothing that
                # extends tens of metres backward into the incoming lane.
                # Feeding it to lane-follow pulled the ego from 500144 back to
                # 500145 long before the right-turn connector.
                stable_provider = getattr(
                    self,
                    "_stable_reference_line_provider",
                    ReferenceLineProvider(),
                )
                self._stable_reference_line_provider = stable_provider
                incoming_lane_reference, incoming_snapshot_reason = (
                    stable_provider.preturn_lane_reference(
                        getattr(self, "_local_map_snapshot", LocalMapSnapshot()),
                        lane_id=int(current_lane_id),
                        ego_x_m=float(ego_location.x),
                        ego_y_m=float(ego_location.y),
                        target_speed_mps=float(candidate_speed_ref_mps),
                        first_forward_m=float(candidate_lane_reference_step_m),
                        spacing_m=float(candidate_lane_reference_step_m),
                        horizon_steps=int(self.mpc.horizon_steps),
                    )
                )
                candidate_reference_debug[
                    "preturn_lane_reference_reason"
                ] = str(incoming_snapshot_reason)
                if incoming_lane_reference:
                    _, preturn_raw_first_lateral_m = self._body_frame_xy(
                        origin_x_m=float(ego_location.x),
                        origin_y_m=float(ego_location.y),
                        heading_rad=float(ego_yaw_rad),
                        target_x_m=float(
                            incoming_lane_reference[0].get(
                                "x_ref_m", incoming_lane_reference[0].get("x")
                            )
                        ),
                        target_y_m=float(
                            incoming_lane_reference[0].get(
                                "y_ref_m", incoming_lane_reference[0].get("y")
                            )
                        ),
                    )
                    candidate_reference_debug[
                        "preturn_raw_first_lateral_m"
                    ] = float(preturn_raw_first_lateral_m)
                    reference = [
                        dict(sample) for sample in incoming_lane_reference
                    ]
                    candidate_reference_debug["reference_source"] = (
                        "admap_current_lane_center_preturn"
                    )
                transition_reference: list[dict[str, object]] = []
                transition_arc_m = max(
                    0.0,
                    float(
                        self.config.get(
                            "lane_follow_to_turn_reference_transition_arc_m",
                            12.0,
                        )
                    ),
                )
                if float(upcoming_turn_distance_m) <= float(transition_arc_m):
                    normalized_turn_direction = str(
                        upcoming_turn_direction
                    ).strip().lower()
                    turn_snapshot = self._stable_reference_line_provider.snapshot(TURN)
                    if (
                        turn_snapshot.active
                        and str(turn_snapshot.maneuver_direction)
                        != str(normalized_turn_direction)
                    ):
                        self._clear_turn_master_reference()
                    (
                        transition_reference,
                        _transition_destination,
                        transition_reference_reason,
                    ) = self._waypoint_turn_reference(
                        ego_location=ego_location,
                        ego_yaw_rad=float(ego_yaw_rad),
                        current_state=current_state,
                        current_lane_id=int(current_lane_id),
                        target_lane_id=int(candidate_target_lane_id),
                        target_speed_mps=float(candidate_speed_ref_mps),
                        destination_state=destination_state,
                        lock_master=True,
                        turn_direction=str(normalized_turn_direction),
                    )
                    # PREPARE_TURN is still longitudinally owned by the
                    # SpeedPlanner.  Borrow only connector geometry here;
                    # _waypoint_turn_reference's low intersection speed is
                    # reserved for the actual intersection_turn mode.
                    for sample in transition_reference:
                        sample["v_ref_mps"] = float(candidate_speed_ref_mps)
                        sample["speed_ref_mps"] = float(candidate_speed_ref_mps)
                        sample["speed_mps"] = float(candidate_speed_ref_mps)
                    if transition_reference:
                        candidate_reference_debug.update({
                            "reference_source": "admap_preturn_connector_transition",
                            "route_turn_reference_reason": str(
                                transition_reference_reason
                            ),
                        })
                if transition_reference:
                    # The immutable AD-map turn master already contains the
                    # incoming lane, connector and outgoing lane in topology
                    # order.  It is the sole geometry owner from transition
                    # entry through the turn; a second bridge-side blend made
                    # an artificial opposite-curvature S bend.
                    reference = [
                        dict(sample) for sample in transition_reference
                    ]
                    turn_hold_reason = "lane_follow_to_turn_locked_master_window"
                else:
                    turn_hold_reason = ""
                if turn_hold_reason:
                    candidate_reference_debug["lane_follow_turn_geometry_hold_reason"] = str(
                        turn_hold_reason
                    )
            if (
                candidate_decision in {"lane_change_left", "lane_change_right"}
                and keep_lane_reference
            ):
                lane_change_duration_s = max(
                    float(self.mpc.dt_s),
                    float(getattr(intent, "lane_change_duration_s", 4.0) or 4.0),
                )
                provider = self._stable_reference_line_provider
                reference, lane_change_geometry_debug = (
                    provider.lane_change_nominal(
                    getattr(self, "_local_map_snapshot", None),
                    ego_x_m=float(current_state[0]),
                    ego_y_m=float(current_state[1]),
                    ego_heading_rad=float(current_state[3]),
                    current_lane_id=int(current_lane_id),
                    target_lane_id=int(candidate_target_lane_id),
                    target_reference=reference,
                    fallback_source_reference=keep_lane_reference,
                    target_speed_mps=float(candidate_speed_ref_mps),
                    geometry_speed_mps=float(candidate_geometry_speed_mps),
                    geometry_length_m=float(candidate_geometry_length_m),
                    transition_duration_s=float(lane_change_duration_s),
                    spacing_m=float(candidate_lane_reference_step_m),
                    horizon_steps=int(self.mpc.horizon_steps),
                    lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                    )
                )
                candidate_reference_debug["lane_change_source_corridor_reason"] = (
                    str(lane_change_geometry_debug.get(
                        "source_corridor_reason", ""
                    ))
                )
                if lane_change_geometry_debug.get("rejection"):
                    candidate_reference_debug["lane_change_geometry_rejection"] = (
                        str(lane_change_geometry_debug["rejection"])
                    )
                if reference and len(destination_state) >= 4:
                    terminal = reference[-1]
                    destination_state = list(destination_state)
                    destination_state[0] = float(
                        terminal.get("x_ref_m", terminal.get("x", destination_state[0]))
                    )
                    destination_state[1] = float(
                        terminal.get("y_ref_m", terminal.get("y", destination_state[1]))
                    )
                    destination_state[2] = float(candidate_speed_ref_mps)
                    destination_state[3] = float(
                        terminal.get("heading_rad", destination_state[3])
                    )
                    if len(destination_state) >= 5:
                        destination_state[4] = int(candidate_target_lane_id)
                candidate_reference_debug.update({
                    "lane_change_trajectory_variant": str(
                        getattr(intent, "trajectory_variant", "normal")
                    ),
                    "lane_change_duration_s": float(
                        self.maneuver_manager.lane_change.resolved_duration_s
                        or lane_change_duration_s
                    ),
                    "lane_change_duration_comfort_reason": str(
                        self.maneuver_manager.lane_change.duration_comfort_reason
                    ),
                    "lane_change_reference_profile": "frenet_quintic_d_of_s",
                    "lane_change_geometry_speed_mps": float(
                        candidate_geometry_speed_mps
                    ),
                    "lane_change_geometry_length_m": float(
                        candidate_geometry_length_m
                    ),
                    "lane_change_geometry_step_m": float(
                        candidate_lane_reference_step_m
                    ),
                    "lane_change_operational_curvature_limit_1pm": float(
                        operational_curvature_limit_1pm
                    ),
                    "lane_change_authorization_source": (
                        "opportunistic"
                        if str(getattr(intent, "reason", "")).startswith("opportunistic_")
                        else "route"
                    ),
                    "lane_change_initial_progress": (
                        float(reference[0].get("lane_change_initial_progress", 0.0))
                        if reference else 0.0
                    ),
                    "lane_change_terminal_progress": (
                        float(reference[-1].get("lane_change_progress", 0.0))
                        if reference else 0.0
                    ),
                })

            conditioned = self.reference_pipeline.condition(
                ReferencePipelineRequest(
                    destination_state=destination_state,
                    reference_samples=reference,
                    current_state=current_state,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    ego_speed_mps=float(ego_speed_mps),
                    target_speed_mps=float(candidate_speed_ref_mps),
                    behavior_decision=str(candidate_decision),
                    behavior_fsm_state=str(candidate_lc_state),
                    current_lane_id=int(current_lane_id),
                    target_lane_id=int(candidate_target_lane_id),
                    stop_goal_active=bool(candidate_stop_goal_active),
                    stop_target=(
                        getattr(intent, "stop_target", None)
                        if isinstance(
                            getattr(intent, "stop_target", None), Mapping
                        )
                        else None
                    ),
                    route_points=route_points,
                )
            )
            destination_state = list(conditioned.destination_state)
            reference = [
                dict(sample) for sample in conditioned.reference_samples
            ]
            stabilizer_reason = str(conditioned.reason)
            contract_result = conditioned.validation
            candidate_reference_debug["mpc_reference_stabilizer_reason"] = str(stabilizer_reason)
            candidate_result = CandidateReferenceResult(
                intent=intent,
                destination_state=list(destination_state),
                lane_center_reference=[dict(sample) for sample in list(reference or [])],
                reference_debug=dict(candidate_reference_debug),
                contract_result=contract_result,
            )
            candidate_is_static_obstacle_local_avoidance = bool(
                self._static_obstacle_local_target_lane_id is not None
                and int(candidate_target_lane_id)
                == int(self._static_obstacle_local_target_lane_id)
                and int(candidate_target_lane_id) != int(current_lane_id)
            )
            evaluated_candidate_result = evaluate_candidate_reference(
                candidate=candidate_result,
                ego_state=current_state,
                object_snapshots=object_snapshots,
                prediction_trajectories=prediction_trajectories,
                current_lane_id=int(current_lane_id),
                min_object_distance_m=float(
                    self.static_obstacle_local_avoidance_min_object_distance_m
                    if candidate_is_static_obstacle_local_avoidance
                    else self.full_candidate_reference_min_object_distance_m
                ),
                previous_risk_bucket=str(
                    self._candidate_risk_bucket_state.get(str(intent.name), "")
                ),
                risk_hysteresis_margin_m=float(
                    self.candidate_risk_hysteresis_margin_m
                ),
            )
            self._candidate_risk_bucket_state[str(intent.name)] = str(
                evaluated_candidate_result.risk_bucket
            )
            candidate_results.append(evaluated_candidate_result)
            if (
                str(candidate_decision) == "lane_follow"
                and int(candidate_target_lane_id) == int(current_lane_id)
                and not keep_lane_reference
                and reference
            ):
                keep_lane_reference = [
                    dict(sample) for sample in list(reference or [])
                ]

        # A committed maneuver owns one immutable master trajectory. Replanned
        # lane-change variants remain useful before commitment, but they must
        # not replace the executing trajectory after commitment has started.
        if bool(self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()):
            commitment_phase = str(self.maneuver_manager.lane_change.phase)
            stabilization_active = bool(
                commitment_phase == "target_lane_stabilization"
            )
            committed_decision = (
                "lane_change_left"
                if str(self.maneuver_manager.lane_change.option)
                == "CHANGELANELEFT"
                else "lane_change_right"
            )
            committed_speed_mps = max(
                0.5,
                float(
                    self.maneuver_manager.lane_change.target_speed_mps
                    or baseline_speed_ref_mps
                ),
            )
            committed_step_m = max(
                0.1,
                float(self.mpc.dt_s) * float(committed_speed_mps),
            )
            committed_reference, committed_window_reason = (
                self._route_tracking_lane_change_window(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    target_speed_mps=float(committed_speed_mps),
                    step_distance_m=float(committed_step_m),
                )
            )
            # Windowing the locked master path (nearest-point search plus
            # low-speed tail padding) can transiently read a higher raw
            # curvature than the master path was built for, even though the
            # locked path itself was shaped to satisfy the contract at lock
            # time. reference_pipeline.py already repairs exactly this case
            # for mode="lane_change" via curvature_feasible_samples before
            # validating; _validate_candidate_reference_contract below had no
            # equivalent repair, so a one-tick windowing spike hard-rejected
            # the committed candidate outright and forced an emergency-brake
            # fallback mid-maneuver. Apply the same repair here so the
            # candidate is judged on the same shaped geometry the final
            # reference pipeline would have produced anyway.
            if committed_reference:
                from opencda.planning_module.pipeline.reference_contract import (
                    contract_from_config,
                )

                committed_lane_change_contract = contract_from_config(
                    mode="lane_change",
                    expected_lane_id=int(
                        self.maneuver_manager.lane_change.target_lane_id
                    ),
                    horizon_steps=int(self.mpc.horizon_steps),
                    config=dict(self.config),
                    default_speed_mps=max(
                        float(self.target_speed_mps),
                        float(committed_speed_mps),
                        0.1,
                    ),
                )
                committed_reference, committed_curvature_reason = (
                    self._stable_reference_line_provider.condition_lane_change_reference(
                        committed_reference,
                        ego_location=ego_location,
                        ego_heading_rad=float(ego_yaw_rad),
                        max_curvature_1pm=float(
                            committed_lane_change_contract.max_curvature_1pm
                        ),
                        mode="committed_lane_change",
                    )
                )
                if committed_curvature_reason:
                    committed_window_reason = (
                        str(committed_window_reason)
                        + ";"
                        + str(committed_curvature_reason)
                    )
            committed_destination: list[float] = []
            if committed_reference:
                terminal = dict(committed_reference[-1])
                committed_destination = [
                    float(
                        terminal.get(
                            "x_ref_m",
                            terminal.get("x", current_state[0]),
                        )
                    ),
                    float(
                        terminal.get(
                            "y_ref_m",
                            terminal.get("y", current_state[1]),
                        )
                    ),
                    float(committed_speed_mps),
                    float(terminal.get("heading_rad", current_state[3])),
                    int(self.maneuver_manager.lane_change.target_lane_id),
                ]
            committed_contract = self._validate_candidate_reference_contract(
                decision=str(committed_decision),
                lc_state=(
                    "TARGET_LANE_STABILIZATION"
                    if bool(stabilization_active)
                    else "EXECUTE_LANE_CHANGE_LEFT"
                    if committed_decision == "lane_change_left"
                    else "EXECUTE_LANE_CHANGE_RIGHT"
                ),
                current_lane_id=int(current_lane_id),
                speed_ref_mps=float(committed_speed_mps),
                stop_goal_active=False,
                current_state=current_state,
                destination_state=committed_destination,
                lane_center_reference=committed_reference,
            )
            committed_intent = CandidateBehaviorIntent(
                name="committed_lane_change_continuation",
                decision=str(committed_decision),
                target_lane_id=int(
                    self.maneuver_manager.lane_change.target_lane_id
                ),
                target_speed_mps=float(committed_speed_mps),
                base_cost=-100.0,
                reason=(
                    "target_lane_stabilization"
                    if bool(stabilization_active)
                    else "locked_maneuver_execution"
                ),
                trajectory_variant=(
                    "target_lane_stabilization"
                    if bool(stabilization_active)
                    else "locked"
                ),
            )
            committed_result = CandidateReferenceResult(
                intent=committed_intent,
                destination_state=list(committed_destination),
                lane_center_reference=[
                    dict(sample) for sample in committed_reference
                ],
                reference_debug={
                    "reference_source": (
                        "target_lane_stabilization_reference"
                        if bool(stabilization_active)
                        else "locked_quintic_lane_change_reference"
                    ),
                    "candidate_lane_change_window_reason": str(
                        committed_window_reason
                    ),
                    "lane_change_phase": str(commitment_phase),
                    "lane_change_stabilization_frames": int(
                        self.maneuver_manager.lane_change.stabilization_frames
                    ),
                    "route_tracking_lane_change_locked": True,
                    "route_tracking_lane_change_progress_index": int(
                        self.maneuver_manager.lane_change.progress_index
                    ),
                    "route_tracking_lane_change_source_lane_id": int(
                        self.maneuver_manager.lane_change.source_lane_id
                    ),
                    "route_tracking_lane_change_target_lane_id": int(
                        self.maneuver_manager.lane_change.target_lane_id
                    ),
                    "lane_change_duration_s": float(
                        self.maneuver_manager.lane_change.resolved_duration_s
                    ),
                    "lane_change_duration_comfort_reason": str(
                        self.maneuver_manager.lane_change.duration_comfort_reason
                    ),
                },
                contract_result=committed_contract,
            )
            committed_is_static_obstacle_local_avoidance = bool(
                self._static_obstacle_local_target_lane_id is not None
                and int(self.maneuver_manager.lane_change.target_lane_id)
                == int(self._static_obstacle_local_target_lane_id)
                and int(self.maneuver_manager.lane_change.target_lane_id)
                != int(current_lane_id)
            )
            evaluated_committed_result = evaluate_candidate_reference(
                candidate=committed_result,
                ego_state=current_state,
                object_snapshots=object_snapshots,
                prediction_trajectories=prediction_trajectories,
                current_lane_id=int(current_lane_id),
                min_object_distance_m=float(
                    self.static_obstacle_local_avoidance_min_object_distance_m
                    if committed_is_static_obstacle_local_avoidance
                    else self.full_candidate_reference_min_object_distance_m
                ),
                previous_risk_bucket=str(
                    self._candidate_risk_bucket_state.get(
                        str(committed_intent.name), ""
                    )
                ),
                risk_hysteresis_margin_m=float(
                    self.candidate_risk_hysteresis_margin_m
                ),
            )
            self._candidate_risk_bucket_state[str(committed_intent.name)] = str(
                evaluated_committed_result.risk_bucket
            )
            candidate_results.append(
                evaluated_committed_result
            )

        probe_summary = self._probe_candidate_results_for_mpc(
            candidate_results=candidate_results,
            current_state=current_state,
            object_snapshots=object_snapshots,
            apply_mpc_probe_result=apply_mpc_probe_result,
            mark_mpc_probe_skipped=mark_mpc_probe_skipped,
            road_envelope_payload_world=(
                self._current_route_tracking_lane_change_envelope_payload_world()
            ),
            required_decision=str(required_lane_change_decision),
            required_target_lane_id=int(required_lane_change_target_lane_id),
        )
        committed_decision = (
            "lane_change_left"
            if str(self.maneuver_manager.lane_change.option) == "CHANGELANELEFT"
            else "lane_change_right"
            if str(self.maneuver_manager.lane_change.option) == "CHANGELANERIGHT"
            else ""
        )
        maneuver_commitment = ManeuverCommitment(
            state=(
                "STABILIZING"
                if str(self.maneuver_manager.lane_change.phase)
                == "target_lane_stabilization"
                else "COMMITTED"
                if bool(self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples())
                else "IDLE"
            ),
            decision=str(committed_decision),
            source_lane_id=int(self.maneuver_manager.lane_change.source_lane_id),
            target_lane_id=int(self.maneuver_manager.lane_change.target_lane_id),
            progress=float(self.maneuver_manager.lane_change.progress),
            reference_locked=bool(self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()),
        )
        selection_outcome = select_candidate_with_commitment(
            candidate_results,
            commitment=maneuver_commitment,
            required_decision=str(required_lane_change_decision),
            required_target_lane_id=int(required_lane_change_target_lane_id),
        )

        if (
            bool(self.strict_decision_ownership_enabled)
            and candidate_results
            and selection_outcome.selected is None
        ):
            return self._explicit_fallback_candidate_for_mpc(
                candidate_results=candidate_results,
                baseline_decision=str(baseline_decision),
                baseline_target_lane_id=int(baseline_target_lane_id),
                current_lane_id=int(current_lane_id),
                current_state=current_state,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                baseline_speed_ref_mps=float(baseline_speed_ref_mps),
                summarize_candidate_results=summarize_candidate_results,
                maneuver_commitment=maneuver_commitment,
                selection_reason=str(selection_outcome.reason),
            )

        selected = (
            selection_outcome.selected
            if selection_outcome.selected is not None
            else select_best_candidate(candidate_results)
        )
        selected_debug = dict(selected.reference_debug or {})
        selected_reference = [
            dict(sample)
            for sample in list(selected.lane_center_reference or [])
        ]
        selected_destination = list(selected.destination_state or [])
        selected_decision = str(selected.intent.decision)
        if selected_decision in {"lane_change_left", "lane_change_right"}:
            selected_target_lane_id = int(selected.intent.target_lane_id)
            route_option = (
                "CHANGELANELEFT"
                if selected_decision == "lane_change_left"
                else "CHANGELANERIGHT"
            )
            planned_lane_change_speed_mps = predicted_lane_change_average_speed_mps(
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(selected.intent.target_speed_mps),
                duration_s=float(selected.intent.lane_change_duration_s or 4.0),
                acceleration_limit_mps2=float(
                    self.config.get(
                        "lane_change_planning_acceleration_limit_mps2",
                        2.0,
                    )
                ),
            )
            selected_operational_curvature_limit_1pm = (
                lane_change_operational_curvature_limit_1pm(
                    planning_speed_mps=float(selected.intent.target_speed_mps),
                    lateral_accel_limit_mps2=float(
                        self.config.get(
                            "route_tracking_lane_change_lateral_accel_limit_mps2",
                            1.3,
                        )
                    ),
                    vehicle_max_curvature_1pm=float(
                        self.config.get("reference_vehicle_max_curvature_1pm", 0.35)
                    ),
                    minimum_speed_mps=float(
                        self.config.get("lane_change_min_geometry_speed_mps", 2.0)
                    ),
                )
            )
            (
                lane_change_geometry_speed_mps,
                lane_change_geometry_length_m,
                geometry_step_distance_m,
            ) = lane_change_geometry_requirements(
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(selected.intent.target_speed_mps),
                duration_s=float(selected.intent.lane_change_duration_s or 4.0),
                dt_s=float(self.mpc.dt_s),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                max_curvature_1pm=float(
                    selected_operational_curvature_limit_1pm
                ),
                minimum_geometry_speed_mps=float(
                    self.config.get("lane_change_min_geometry_speed_mps", 2.0)
                ),
                minimum_length_m=float(
                    self.config.get("lane_change_min_length_m", 10.0)
                ),
                acceleration_limit_mps2=float(
                    self.config.get(
                        "lane_change_planning_acceleration_limit_mps2", 2.0
                    )
                ),
            )
            step_distance_m = max(0.1, float(geometry_step_distance_m))
            selected_is_committed_continuation = bool(
                str(selected.intent.name)
                == "committed_lane_change_continuation"
            )
            lock_matches = bool(
                self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()
                and int(self.maneuver_manager.lane_change.target_lane_id)
                == int(selected_target_lane_id)
                and str(self.maneuver_manager.lane_change.option)
                == str(route_option)
                and (
                    bool(selected_is_committed_continuation)
                    or int(self.maneuver_manager.lane_change.source_lane_id)
                    == int(current_lane_id)
                )
            )
            lock_reason = "candidate_lane_change_lock_reused"
            if not bool(lock_matches):
                # The selected candidate has already passed the reference
                # contract and MPC probe.  It is the nominal trajectory; make
                # that exact geometry persistent instead of rebuilding a
                # second quintic from waypoint queries at the hand-off.
                completion_snapshot = getattr(self, "_local_map_snapshot", None)
                completion_reference, completion_reason = (
                    self._stable_reference_line_provider.lane_change_completion_reference(
                        completion_snapshot,
                        target_lane_id=int(selected_target_lane_id),
                        target_speed_mps=float(selected.intent.target_speed_mps),
                    )
                )
                installed, install_reason = (
                    self._stable_reference_line_provider.install(
                        LANE_CHANGE,
                        selected_reference,
                        route_revision=str(self.route_manager.route_revision),
                        map_epoch=str(getattr(self, "waypoint_backend", "admap") or "admap"),
                        event="maneuver_started",
                        source_lane_id=int(current_lane_id),
                        target_lane_id=int(selected_target_lane_id),
                        maneuver_direction=(
                            "left" if route_option == "CHANGELANELEFT" else "right"
                        ),
                        build_reason="accepted_candidate_nominal_trajectory",
                        ego_x_m=float(current_state[0]),
                        ego_y_m=float(current_state[1]),
                    )
                )
                if installed:
                    self.maneuver_manager.begin_lane_change(
                        option=str(route_option),
                        phase="executing",
                        source_lane_id=int(current_lane_id),
                        target_lane_id=int(selected_target_lane_id),
                        target_speed_mps=float(selected.intent.target_speed_mps),
                        completion_reference=completion_reference,
                        committed_at_s=float(self._sim_time_s()),
                    )
                lock_reason = (
                    "accepted_candidate_committed:"
                    + str(install_reason)
                    + ":"
                    + str(completion_reason)
                )
            locked_window, window_reason = (
                self._route_tracking_lane_change_window(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    target_speed_mps=float(selected.intent.target_speed_mps),
                    step_distance_m=float(step_distance_m),
                )
            )
            locked_valid, locked_validation_reason = (
                self._validate_route_tracking_lane_change_reference(
                    reference=locked_window,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                )
            )
            if locked_window and bool(locked_valid):
                selected_reference = [
                    dict(sample) for sample in locked_window
                ]
                terminal = selected_reference[-1]
                if len(selected_destination) >= 4:
                    selected_destination[0] = float(
                        terminal.get("x_ref_m", terminal.get("x", selected_destination[0]))
                    )
                    selected_destination[1] = float(
                        terminal.get("y_ref_m", terminal.get("y", selected_destination[1]))
                    )
                    selected_destination[2] = float(
                        selected.intent.target_speed_mps
                    )
                    selected_destination[3] = float(
                        terminal.get("heading_rad", selected_destination[3])
                    )
                    if len(selected_destination) >= 5:
                        selected_destination[4] = int(selected_target_lane_id)
            selected_debug.update({
                "lane_change_planning_average_speed_mps": float(
                    planned_lane_change_speed_mps
                ),
                "lane_change_geometry_speed_mps": float(
                    lane_change_geometry_speed_mps
                ),
                "lane_change_geometry_length_m": float(
                    lane_change_geometry_length_m
                ),
                "lane_change_geometry_step_m": float(step_distance_m),
                "route_tracking_lane_change_locked": bool(
                    self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()
                ),
                "route_tracking_lane_change_progress_index": int(
                    self.maneuver_manager.lane_change.progress_index
                ),
                "route_tracking_lane_change_source_lane_id": int(
                    self.maneuver_manager.lane_change.source_lane_id
                ),
                "route_tracking_lane_change_target_lane_id": int(
                    self.maneuver_manager.lane_change.target_lane_id
                ),
                "candidate_lane_change_lock_reason": str(lock_reason),
                "candidate_lane_change_window_reason": str(window_reason),
                "candidate_lane_change_lock_validation_reason": str(
                    locked_validation_reason
                ),
            })
        selected_debug.update({
            "stage": selected_debug.get("reference_pipeline_stage", ""),
            "intent_mode": selected_debug.get("reference_pipeline_intent_mode", ""),
            "fallback_reason": selected_debug.get("fallback_reason", ""),
            "reference_source": str(
                selected_debug.get("reference_source", "candidate_reference_pipeline")
            ),
            "candidate_pipeline_selected": str(selected.intent.name),
            "candidate_pipeline_selected_status": str(selected.feasibility_status),
            "candidate_pipeline_selected_reason": str(selected.feasibility_reason),
            "candidate_selected_stop_goal_active": bool(
                selected.intent.stop_goal_active
            ),
            "candidate_pipeline_count": int(len(candidate_results)),
            "candidate_prediction_trajectory_count": int(len(prediction_trajectories)),
            "candidate_pipeline_summary": summarize_candidate_results(candidate_results),
            "candidate_mpc_probe_summary": str(probe_summary),
            "candidate_selected_decision": str(selected.intent.decision),
            "candidate_selected_lane_id": int(selected.intent.target_lane_id),
            "candidate_selected_cost": float(selected.total_cost),
            "candidate_evaluation_summary": (
                f"{selected.intent.name}->{selected.intent.decision}"
                f":L{int(selected.intent.target_lane_id)}"
                f" cost={float(selected.total_cost):.2f}"
            ),
            "candidate_selection_status": str(selection_outcome.status),
            "candidate_selection_reason": str(selection_outcome.reason),
            "lane_change_commitment_release_reason": str(
                lane_change_commitment_release_reason
            ),
            "lane_change_phase": str(self.maneuver_manager.lane_change.phase),
            "lane_change_stabilization_frames": int(
                self.maneuver_manager.lane_change.stabilization_frames
            ),
        })
        selected_debug.update(maneuver_commitment.as_debug_fields())
        selected_debug.update(
            dict(self.maneuver_manager.lane_change.completion_debug)
        )
        fallback_manager = getattr(
            self, "_trajectory_fallback_manager", TrajectoryFallbackManager()
        )
        self._trajectory_fallback_manager = fallback_manager
        fallback_manager.record_valid(
            selected_reference,
            sim_time_s=float(self._sim_time_s()),
            route_revision=str(
                getattr(self.route_manager, "route_revision", "")
            ),
        )
        return (
            str(selected_decision),
            int(selected.intent.target_lane_id),
            float(selected.intent.target_speed_mps),
            list(selected_reference),
            list(selected_destination),
            selected_debug,
        )

    def _release_completed_lane_change_commitment(
        self,
        *,
        current_lane_id: int,
        ego_location: carla.Location,
        ego_yaw_rad: float,
    ) -> str:
        """Release only after the ego converges to the locked target path."""

        if not self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples():
            return ""
        phase = str(self.maneuver_manager.lane_change.phase)
        target_lane_id = int(self.maneuver_manager.lane_change.target_lane_id)
        from opencda.planning_module.pipeline.stage_contracts import (
            LaneChangeContract,
        )

        lane_change_contract = LaneChangeContract.from_config(self.config)
        entry_min_progress = float(lane_change_contract.min_progress)
        completion_reference = [
            dict(sample)
            for sample in self.maneuver_manager.lane_change.completion_reference
        ]
        if not completion_reference:
            # Compatibility for commitments restored from older state.
            completion_reference = [
                dict(sample) for sample in self._stable_reference_line_provider.snapshot(LANE_CHANGE).mutable_samples()
            ]
        terminal_samples = [
            dict(sample)
            for sample in completion_reference
            if float(sample.get("lane_change_progress", 0.0) or 0.0) >= 0.9
        ]
        if not terminal_samples and completion_reference:
            # A target-corridor centerline is itself the completion geometry;
            # unlike a Frenet transition it intentionally has no scheduled
            # lane_change_progress tag.
            terminal_samples = [dict(sample) for sample in completion_reference]
        target_corridor_sample = (
            min(
                terminal_samples,
                key=lambda sample: (
                    float(sample.get("x_ref_m", sample.get("x", ego_location.x)))
                    - float(ego_location.x)
                ) ** 2
                + (
                    float(sample.get("y_ref_m", sample.get("y", ego_location.y)))
                    - float(ego_location.y)
                ) ** 2,
            )
            if terminal_samples
            else None
        )
        stabilization_entry_lateral_error_m = float("inf")
        stabilization_entry_heading_error_rad = float("inf")
        if target_corridor_sample is not None:
            target_x_m = float(
                target_corridor_sample.get(
                    "x_ref_m", target_corridor_sample.get("x", ego_location.x)
                )
            )
            target_y_m = float(
                target_corridor_sample.get(
                    "y_ref_m", target_corridor_sample.get("y", ego_location.y)
                )
            )
            target_heading_rad = float(
                target_corridor_sample.get("heading_rad", ego_yaw_rad)
            )
            stabilization_entry_lateral_error_m = (
                -math.sin(float(target_heading_rad))
                * (float(ego_location.x) - float(target_x_m))
                + math.cos(float(target_heading_rad))
                * (float(ego_location.y) - float(target_y_m))
            )
            stabilization_entry_heading_error_rad = math.atan2(
                math.sin(float(ego_yaw_rad) - float(target_heading_rad)),
                math.cos(float(ego_yaw_rad) - float(target_heading_rad)),
            )
        target_lane_width_m = max(
            0.1, float(getattr(self.mpc, "lane_width_m", 3.5))
        )
        stabilization_geometry_ready = lane_change_contract.stabilization_handoff_ready(
            progress=float(self.maneuver_manager.lane_change.progress),
            lateral_error_m=float(stabilization_entry_lateral_error_m),
            heading_error_rad=float(stabilization_entry_heading_error_rad),
            lane_width_m=float(target_lane_width_m),
        )
        # Lane IDs identify the source/target topology but do not own motion
        # phase transitions.  Enter stabilization only from continuous
        # progress and convergence to the locked target corridor.  This is
        # robust both when the map ID flips early and when a road-boundary
        # re-anchor changes the ID namespace during the maneuver.
        handoff_transition = self.maneuver_manager.lane_change_handoff_transition(
            geometry_ready=bool(
                float(self.maneuver_manager.lane_change.progress)
                >= float(entry_min_progress)
                and bool(stabilization_geometry_ready)
            ),
        )
        if handoff_transition.action == "start_stabilization":
            start_reason = self._start_target_lane_stabilization(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
            )
            if str(start_reason).startswith(
                "target_lane_stabilization_started"
            ):
                return str(start_reason)
            self.maneuver_manager.complete_lane_change(
                "stabilization_handoff_unavailable"
            )
            self._reset_route_tracking_lane_change_reference()
            return (
                "lane_change_stabilization_unavailable_to_lane_follow_recovery:"
                f"target_lane={int(target_lane_id)}:"
                f"{str(start_reason)}"
            )

        phase = str(self.maneuver_manager.lane_change.phase)
        if str(phase) == "target_lane_stabilization":
            timeout_frames = max(
                1,
                int(
                    self.config.get(
                        "lane_change_stabilization_timeout_frames",
                        100,
                    )
                ),
            )
            stabilization_transition = (
                self.maneuver_manager.tick_lane_change_stabilization(
                    timeout_frames=int(timeout_frames)
                )
            )
            if stabilization_transition.action == "abandon":
                self._reset_route_tracking_lane_change_reference()
                return (
                    "lane_change_stabilization_timeout_to_lane_follow_recovery:"
                    f"target_lane={int(target_lane_id)}:"
                    f"map_lane={int(current_lane_id)}"
                )
        completion_progress = float(self.maneuver_manager.lane_change.progress)
        if (
            str(phase) == "target_lane_stabilization"
            and math.isfinite(float(stabilization_entry_lateral_error_m))
        ):
            target_lane_width_m = max(
                0.1,
                float(getattr(self.mpc, "lane_width_m", 3.5)),
            )
            completion_progress = min(
                1.0,
                max(
                    0.0,
                    1.0
                    - abs(float(stabilization_entry_lateral_error_m))
                    / float(target_lane_width_m),
                ),
            )
        from opencda.planning_module.pipeline.stage_contracts import (
            evaluate_lane_change_completion,
        )

        vehicle_manager = getattr(self, "vehicle_manager", None)
        vehicle = getattr(vehicle_manager, "vehicle", None)
        extent = getattr(getattr(vehicle, "bounding_box", None), "extent", None)
        occupancy = self.reference_generator.lane_corridor_occupancy(
            x_m=float(ego_location.x),
            y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            ego_half_width_m=float(
                getattr(
                    extent,
                    "y",
                    self.config.get("reference_vehicle_half_width_m", 1.0),
                )
            ),
            ego_half_length_m=float(
                getattr(
                    extent,
                    "x",
                    self.config.get("reference_vehicle_half_length_m", 2.4),
                )
            ),
            safety_margin_m=float(
                self.config.get(
                    "lane_change_completion_footprint_margin_m",
                    0.05,
                )
            ),
            corridor_sample=target_corridor_sample,
            prefer_tracking_point=True,
        )
        completion = evaluate_lane_change_completion(
            reference_samples=completion_reference,
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            progress=float(completion_progress),
            previous_stable_frames=int(
                self.maneuver_manager.lane_change.completion_stable_frames
            ),
            target_lane_matches=bool(
                int(current_lane_id) == int(target_lane_id)
            ),
            footprint_clearance_m=(
                float(occupancy.footprint_clearance_m)
                if bool(occupancy.valid)
                else float("-inf")
            ),
            min_footprint_clearance_m=0.0,
            contract=lane_change_contract,
        )
        completion_debug = {
            **lane_change_contract.as_debug_fields(),
            "lane_change_stabilization_entry_lateral_error_m": float(
                stabilization_entry_lateral_error_m
            ),
            "lane_change_stabilization_entry_heading_error_deg": math.degrees(
                float(stabilization_entry_heading_error_rad)
            ),
            "lane_change_stabilization_geometry_ready": bool(
                stabilization_geometry_ready
            ),
            "lane_change_completion_reason": str(completion.reason),
            "lane_change_completion_stable_frames": int(
                completion.stable_frames
            ),
            "lane_change_completion_lateral_error_m": float(
                completion.target_lateral_error_m
            ),
            "lane_change_completion_heading_error_deg": math.degrees(
                float(completion.target_heading_error_rad)
            ),
            "lane_change_completion_target_lane_matches": bool(
                completion.target_lane_matches
            ),
            "lane_change_completion_footprint_clearance_m": float(
                completion.footprint_clearance_m
            ),
        }
        transition_arc_m = max(
            0.0, float(self.maneuver_manager.lane_change.transition_to_turn_arc_m)
        )
        transition_progress_m = float(
            self.maneuver_manager.lane_change.progress_s_m
        )
        completion_transition = self.maneuver_manager.accept_lane_change_completion(
            stable_frames=int(completion.stable_frames),
            debug=completion_debug,
            geometrically_complete=bool(completion.complete),
            completion_reason=str(completion.reason),
            transition_progress_m=float(transition_progress_m),
            transition_arc_m=float(transition_arc_m),
        )
        completion_latched = bool(
            self.maneuver_manager.lane_change.geometry_completion_latched
        )
        self.maneuver_manager.lane_change.completion_debug[
            "lane_change_geometry_completion_latched"
        ] = bool(completion_latched)
        self.maneuver_manager.lane_change.completion_debug.update({
            "lane_change_to_turn_transition_arc_m": float(transition_arc_m),
            "lane_change_to_turn_transition_progress_m": float(
                transition_progress_m
            ),
        })
        if completion_transition.action != "complete":
            return ""
        # The stable provider projects the ego onto the immutable handoff
        # master and exposes real arc length.  The old index * nominal-step
        # approximation could jump several samples when the target-lane
        # reference was resampled, releasing this phase after only one or two
        # frames and handing an incompatible geometry to MPC.
        self.maneuver_manager.complete_lane_change(
            str(completion_transition.reason)
        )
        self._reset_route_tracking_lane_change_reference()
        # The next tick changes geometry ownership from the locked Frenet
        # maneuver to AD-map lane-follow / turn approach.  Reusing the former
        # QP rollout or buffered control across that discontinuity caused a
        # short burst of primal-infeasible solves immediately after otherwise
        # successful lane changes.
        clear_seed = getattr(self.mpc, "clear_solution_memory", None)
        if not callable(clear_seed):
            clear_seed = getattr(self.mpc, "clear_previous_solution_seed", None)
        if callable(clear_seed):
            clear_seed()
        control_buffer = getattr(self, "control_buffer", None)
        reset_buffer = getattr(control_buffer, "reset", None)
        if callable(reset_buffer):
            reset_buffer(reason="lane_change_geometrically_complete")
        return (
            "lane_change_commitment_released:"
            f"target_lane={int(target_lane_id)}:"
            f"map_lane={int(current_lane_id)}:"
            f"progress={float(completion.progress):.3f}:"
            f"lateral_error={float(completion.target_lateral_error_m):.3f}:"
            f"heading_error_deg="
            f"{math.degrees(float(completion.target_heading_error_rad)):.2f}"
        )

    def _start_target_lane_stabilization(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
    ) -> str:
        """Replace the completed lateral crossing with a target-lane handoff."""

        target_lane_id = int(self.maneuver_manager.lane_change.target_lane_id)
        # Stabilization is a geometric phase, not a separate low-speed
        # behavior. Preserve the committed maneuver speed; curvature and
        # traffic-control constraints are applied by the unified speed path.
        committed_speed_mps = float(
            self.maneuver_manager.lane_change.target_speed_mps
            or self.target_speed_mps
        )
        speed_mps = max(
            0.5,
            min(
                float(getattr(self, "target_speed_mps", committed_speed_mps)),
                float(committed_speed_mps),
            ),
        )
        step_distance_m = max(
            0.10,
            float(self.mpc.dt_s) * float(speed_mps),
        )
        transition_arc_m = max(
            float(step_distance_m),
            float(
                self.config.get(
                    "lane_change_to_turn_reference_transition_arc_m",
                    10.0,
                )
            ),
        )
        master_steps = max(
            int(self.mpc.horizon_steps),
            int(math.ceil(float(transition_arc_m) / float(step_distance_m)))
            + 1,
        )
        route_points = []
        route_points_fn = getattr(
            getattr(self, "route_manager", None),
            "geometry_route_points",
            None,
        )
        if callable(route_points_fn):
            route_points = route_points_fn(
                x_m=float(ego_location.x),
                y_m=float(ego_location.y),
                query_key="lane_change_to_turn_transition",
            )
        reference, curvature_reason = (
            self._stable_reference_line_provider.target_lane_stabilization_master(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                target_lane_id=int(target_lane_id),
                horizon_steps=int(master_steps),
                step_distance_m=float(step_distance_m),
                # Preserve the selected AD-map successor/turn branch instead
                # of extending a local target lane independently of topology.
                route_points=route_points,
                target_speed_mps=float(speed_mps),
                max_curvature_1pm=float(
                    self.config.get(
                        "reference_vehicle_max_curvature_1pm",
                        0.20,
                    )
                )
            )
        )
        if len(reference) < int(self.mpc.horizon_steps):
            return (
                "target_lane_stabilization_failed:"
                f"short_reference={len(reference)}"
            )
        installed, install_reason = self._stable_reference_line_provider.install(
            LANE_CHANGE,
            reference,
            route_revision=str(
                getattr(getattr(self, "route_manager", None), "route_revision", "")
            ),
            map_epoch=str(getattr(self, "waypoint_backend", "admap") or "admap"),
            event="phase_transition",
            source_lane_id=int(self.maneuver_manager.lane_change.source_lane_id),
            target_lane_id=int(target_lane_id),
            build_reason="target_lane_stabilization_reference",
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
        )
        if not installed:
            return "target_lane_stabilization_failed:" + str(install_reason)
        self.maneuver_manager.begin_lane_change_stabilization()
        # Do not manufacture completion here. The release contract measures
        # geometric progress against the immutable target-lane centerline.
        self.maneuver_manager.update_lane_change_target_speed(float(speed_mps))
        self.maneuver_manager.set_lane_change_transition_arc(
            arc_m=float(transition_arc_m), step_m=float(step_distance_m)
        )
        return (
            "target_lane_stabilization_started:"
            f"target_lane={int(target_lane_id)}:"
            f"N={len(reference)}:"
            f"transition_arc_m={float(transition_arc_m):.2f}:"
            f"speed={float(speed_mps):.2f}:"
            f"curvature_conditioning={str(curvature_reason or 'not_required')}"
        )

    def _probe_candidate_results_for_mpc(
        self,
        *,
        candidate_results: Sequence[object],
        current_state: Sequence[float],
        object_snapshots: Sequence[Mapping[str, object]],
        apply_mpc_probe_result: Any,
        mark_mpc_probe_skipped: Any,
        road_envelope_payload_world: Optional[Mapping[str, object]] = None,
        required_decision: str = "",
        required_target_lane_id: int = 0,
    ) -> str:
        """Run side-effect-free MPC probes for the best keep/lane-change paths."""

        rows = list(candidate_results or [])
        lane_change_rows = [
            row
            for row in rows
            if str(getattr(getattr(row, "intent", None), "decision", "")).startswith(
                "lane_change"
            )
            and bool(getattr(row, "feasible", False))
        ]
        if not bool(self.candidate_mpc_probe_enabled) or not lane_change_rows:
            return "mpc_probe_not_applicable"

        feasible_rows = [
            row for row in rows if bool(getattr(row, "feasible", False))
        ]
        feasible_rows.sort(key=lambda row: float(getattr(row, "total_cost", float("inf"))))
        keep_rows = [
            row
            for row in feasible_rows
            if str(getattr(getattr(row, "intent", None), "decision", ""))
            == "lane_follow"
        ]
        selected_for_probe = []
        normalized_required_decision = str(required_decision).strip().lower()
        required_rows = [
            row
            for row in lane_change_rows
            if str(getattr(getattr(row, "intent", None), "decision", ""))
            .strip().lower() == normalized_required_decision
            and int(getattr(getattr(row, "intent", None), "target_lane_id", 0))
            == int(required_target_lane_id or 0)
        ]
        if required_rows:
            variant_priority = {"normal": 0, "assertive": 1, "conservative": 2}
            required_rows.sort(key=lambda row: (
                variant_priority.get(
                    str(getattr(getattr(row, "intent", None), "trajectory_variant", ""))
                    .strip().lower(),
                    3,
                ),
                float(getattr(row, "total_cost", float("inf"))),
            ))
            selected_for_probe.extend(required_rows)
        else:
            if keep_rows:
                selected_for_probe.append(keep_rows[0])
            lane_change_rows.sort(
                key=lambda row: float(getattr(row, "total_cost", float("inf")))
            )
            selected_for_probe.append(lane_change_rows[0])
        for row in feasible_rows:
            if row in selected_for_probe:
                continue
            if len(selected_for_probe) >= int(self.candidate_mpc_probe_top_k):
                break
            selected_for_probe.append(row)
        selected_for_probe = selected_for_probe[: int(self.candidate_mpc_probe_top_k)]
        selected_probe_order = {
            id(row): index for index, row in enumerate(selected_for_probe)
        }
        feasible_rows.sort(key=lambda row: (
            0 if id(row) in selected_probe_order else 1,
            selected_probe_order.get(id(row), len(selected_probe_order)),
            float(getattr(row, "total_cost", float("inf"))),
        ))

        sim_time_s = float(self._sim_time_s())
        refresh_cache = (
            float(sim_time_s) - float(self._candidate_mpc_probe_last_time_s)
            >= float(self.candidate_mpc_probe_interval_s)
        )
        if bool(refresh_cache):
            self._candidate_mpc_probe_cache = {}
            self._candidate_mpc_probe_last_time_s = float(sim_time_s)

        probe_rows = []
        selected_ids = {id(row) for row in selected_for_probe}
        previous_profile = str(
            getattr(self.mpc, "active_cost_profile_name", "lane_follow")
        )
        for row in feasible_rows:
            if id(row) not in selected_ids:
                mark_mpc_probe_skipped(row)
                continue
            intent = getattr(row, "intent", None)
            destination = list(getattr(row, "destination_state", []) or [])
            reference = [
                dict(sample)
                for sample in list(getattr(row, "lane_center_reference", []) or [])
            ]
            cache_key = (
                str(getattr(intent, "name", "")),
                str(getattr(intent, "decision", "")),
                int(getattr(intent, "target_lane_id", 0) or 0),
                str(getattr(intent, "trajectory_variant", "")),
                round(float(getattr(intent, "lane_change_duration_s", 0.0) or 0.0), 2),
            )
            probe = self._candidate_mpc_probe_cache.get(cache_key)
            if probe is None:
                probe_profile = _mpc_cost_profile_for_behavior(
                    behavior=str(getattr(intent, "decision", "")),
                    planner_lc_state=(
                        "EXECUTE_LANE_CHANGE"
                        if str(getattr(intent, "decision", "")).startswith(
                            "lane_change"
                        )
                        else "LANE_KEEP"
                    ),
                    planner_mode="NORMAL",
                    next_macro_maneuver="straight",
                )
                if hasattr(self.mpc, "apply_mode_cost_profile"):
                    self.mpc.apply_mode_cost_profile(
                        str(probe_profile),
                        blend_alpha=1.0,
                    )
                probe = self.mpc.probe_trajectory_feasibility(
                    current_state=current_state,
                    destination_state=destination,
                    object_snapshots=object_snapshots,
                    current_acceleration_mps2=float(self._last_accel_mps2),
                    current_steering_rad=float(self._last_steer_rad),
                    lane_center_reference_samples=reference,
                    stop_goal_active=bool(
                        getattr(intent, "stop_goal_active", False)
                    ),
                    road_envelope_payload_world=(
                        road_envelope_payload_world
                        if str(getattr(intent, "name", ""))
                        == "committed_lane_change_continuation"
                        else None
                    ),
                )
                self._candidate_mpc_probe_cache[cache_key] = dict(probe)
            apply_mpc_probe_result(
                candidate=row,
                solved=bool(probe.get("solved", False)),
                status=str(probe.get("status", "")),
                solve_time_ms=float(probe.get("solve_time_ms", 0.0) or 0.0),
                dynamic_cost=float(probe.get("dynamic_cost", 0.0) or 0.0),
            )
            probe_rows.append(
                "%s:%s"
                % (
                    str(getattr(intent, "name", "")),
                    str(probe.get("status", "")),
                )
            )
        if hasattr(self.mpc, "apply_mode_cost_profile"):
            self.mpc.apply_mode_cost_profile(
                str(previous_profile),
                blend_alpha=1.0,
            )
        return "|".join(probe_rows) if probe_rows else "mpc_probe_no_feasible_top_k"

    def _waypoint_turn_reference(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        current_state: Sequence[float],
        current_lane_id: int,
        target_lane_id: int,
        target_speed_mps: float,
        destination_state: Sequence[float] | None,
        lock_master: bool = False,
        turn_direction: str = "",
    ) -> tuple[list[dict[str, object]], list[float], str]:
        """Delegate turn geometry to the single ReferenceLineProvider owner."""

        return self._stable_reference_line_provider.turn_reference(
            local_map=getattr(self, "_local_map_snapshot", None),
            config=self.config,
            horizon_steps=int(self.mpc.horizon_steps),
            dt_s=float(self.mpc.dt_s),
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            current_state=current_state,
            current_lane_id=int(current_lane_id),
            target_lane_id=int(target_lane_id),
            target_speed_mps=float(target_speed_mps),
            destination_state=destination_state,
            lock_master=bool(lock_master),
            turn_direction=str(turn_direction),
            route_revision=str(
                getattr(getattr(self, "route_manager", None), "route_revision", "")
            ),
            map_epoch=str(getattr(self, "waypoint_backend", "admap") or "admap"),
        )


    def _clear_turn_master_reference(self) -> None:
        provider = getattr(self, "_stable_reference_line_provider", None)
        if provider is not None:
            provider.release(TURN, event="reset")

    def _start_post_turn_exit_reference(
        self,
        *,
        ego_location: Any,
        ego_yaw_rad: float,
        current_lane_id: int,
        target_speed_mps: float,
    ) -> bool:
        """Lock the immutable AD-map route centreline after a completed turn."""

        step_distance_m = max(
            0.25,
            float(self.mpc.dt_s) * max(1.0, float(target_speed_mps)),
        )
        hold_arc_m = max(
            step_distance_m,
            float(self.config.get("post_turn_exit_reference_arc_m", 12.0)),
        )
        stable_provider = getattr(
            self, "_stable_reference_line_provider", ReferenceLineProvider()
        )
        self._stable_reference_line_provider = stable_provider
        required_arc_m = float(hold_arc_m) + max(
            2.0,
            0.5
            * float(self.mpc.horizon_steps)
            * float(self.mpc.dt_s)
            * max(1.0, float(target_speed_mps)),
        )
        reference, reference_source = stable_provider.post_turn_master(
            getattr(self, "_local_map_snapshot", LocalMapSnapshot()),
            start_lane_id=int(current_lane_id),
            target_speed_mps=float(target_speed_mps),
            horizon_steps=int(self.mpc.horizon_steps),
            required_arc_m=float(required_arc_m),
        )
        if not reference:
            return False
        terminal_lane_ids = [
            int(sample.get("lane_id", 0) or 0)
            for sample in reference
            if int(sample.get("lane_id", 0) or 0) != 0
        ]
        target_lane_id = int(
            terminal_lane_ids[-1] if terminal_lane_ids else current_lane_id
        )
        installed, _ = stable_provider.install(
            POST_TURN,
            reference,
            route_revision=str(
                getattr(getattr(self, "route_manager", None), "route_revision", "")
            ),
            map_epoch=str(getattr(self, "waypoint_backend", "admap") or "admap"),
            event="phase_transition",
            source_lane_id=int(current_lane_id),
            target_lane_id=int(target_lane_id),
            build_reason=str(reference_source),
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
        )
        return bool(installed)

    def _post_turn_exit_reference_window(
        self,
        *,
        ego_location: Any,
        target_speed_mps: float,
    ) -> tuple[list[dict[str, object]], str]:
        """Return a monotonic rolling window over the locked exit centerline."""

        stable_provider = getattr(
            self, "_stable_reference_line_provider", ReferenceLineProvider()
        )
        self._stable_reference_line_provider = stable_provider
        if not stable_provider.snapshot(POST_TURN).active:
            return [], "post_turn_exit_reference_missing"
        spacing_m = max(
            0.25,
            float(self.mpc.dt_s) * max(1.0, float(target_speed_mps)),
        )
        stable_window = stable_provider.window(
            POST_TURN,
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            first_forward_m=float(
                self.config.get("reference_contract_lane_follow_min_first_forward_m", 0.0)
            ),
            spacing_m=float(spacing_m),
            count=int(self.mpc.horizon_steps),
            max_projection_advance_m=max(2.0, 2.0 * float(spacing_m)),
        )
        post_turn_snapshot = stable_provider.snapshot(POST_TURN)
        window = [dict(sample) for sample in stable_window.samples]
        remaining_arc_m = 0.0
        for first, second in zip(window[:-1], window[1:]):
            first_x = float(first.get("x_ref_m", first.get("x", 0.0)))
            first_y = float(first.get("y_ref_m", first.get("y", 0.0)))
            second_x = float(second.get("x_ref_m", second.get("x", first_x)))
            second_y = float(second.get("y_ref_m", second.get("y", first_y)))
            remaining_arc_m += math.hypot(
                float(second_x) - float(first_x),
                float(second_y) - float(first_y),
            )
        minimum_remaining_arc_m = max(
            1.5,
            0.5
            * float(self.mpc.horizon_steps)
            * float(self.mpc.dt_s)
            * max(1.0, float(target_speed_mps)),
        )
        if float(remaining_arc_m) + 1.0e-3 < float(minimum_remaining_arc_m):
            self._clear_post_turn_exit_reference()
            return (
                [],
                "post_turn_exit_reference_exhausted:"
                f"remaining_arc_m={float(remaining_arc_m):.2f}:"
                f"required_m={float(minimum_remaining_arc_m):.2f}",
            )
        for sample in window:
            sample["speed_ref_mps"] = float(target_speed_mps)
            sample["v_ref_mps"] = float(target_speed_mps)
            sample["speed_mps"] = float(target_speed_mps)
        return (
            window,
            "post_turn_exit_locked_window:"
            f"s={float(post_turn_snapshot.progress_s_m):.2f}:"
            f"travel={float(post_turn_snapshot.travelled_s_m):.2f}:"
            f"provider={str(stable_window.reason)}",
        )

    def _clear_post_turn_exit_reference(self) -> None:
        provider = getattr(self, "_stable_reference_line_provider", None)
        if provider is not None:
            provider.release(POST_TURN, event="phase_transition")

    def _explicit_fallback_candidate_for_mpc(
        self,
        *,
        candidate_results: Sequence[object],
        baseline_decision: str,
        baseline_target_lane_id: int,
        current_lane_id: int,
        current_state: Sequence[float],
        ego_location: carla.Location,
        ego_yaw_rad: float,
        summarize_candidate_results: Any,
        baseline_speed_ref_mps: Optional[float] = None,
        maneuver_commitment: Any = None,
        selection_reason: str = "",
    ) -> tuple[str, int, float, list[dict[str, object]], list[float], dict[str, object]]:
        """Submit one typed failure to the sole fallback policy owner."""

        del ego_yaw_rad, baseline_speed_ref_mps
        committed = bool(
            maneuver_commitment is not None
            and bool(getattr(maneuver_commitment, "active", False))
        )
        baseline_normalized = str(baseline_decision or "").strip().lower()
        reference_mode = (
            LANE_CHANGE
            if committed
            else TURN
            if baseline_normalized in {
                "intersection_turn_left",
                "intersection_turn_right",
            }
            else LANE_FOLLOW
        )
        provider = self._stable_reference_line_provider
        snapshot = provider.snapshot(reference_mode)
        current_reference = []
        if snapshot.active:
            window = provider.window(
                reference_mode,
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
                first_forward_m=max(
                    0.0,
                    float(
                        self.config.get(
                            "reference_contract_lane_change_min_first_forward_m"
                            if reference_mode == LANE_CHANGE
                            else "reference_contract_lane_follow_min_first_forward_m",
                            0.2,
                        )
                    ),
                ),
                spacing_m=max(
                    0.1,
                    float(self.mpc.dt_s)
                    * max(0.5, float(current_state[2])),
                ),
                count=int(self.mpc.horizon_steps),
                max_projection_advance_m=max(
                    2.0, 2.0 * max(0.5, float(current_state[2]))
                ),
            )
            current_reference = [dict(sample) for sample in window.samples]

        # A rejected candidate cannot own policy, but its geometry can provide
        # the road-aligned braking corridor when no persistent master exists.
        if len(current_reference) < 2:
            geometric_rows = [
                candidate
                for candidate in list(candidate_results or [])
                if len(list(getattr(candidate, "lane_center_reference", []) or []))
                >= 2
            ]
            contract_valid_rows = [
                candidate
                for candidate in geometric_rows
                if getattr(candidate, "contract_result", None) is not None
                and bool(getattr(candidate.contract_result, "valid", False))
            ]
            if contract_valid_rows or geometric_rows:
                geometric_source = (contract_valid_rows or geometric_rows)[0]
                current_reference = [
                    dict(sample)
                    for sample in list(
                        getattr(geometric_source, "lane_center_reference", []) or []
                    )
                ]

        collision_veto = any(
            "collision_risk" in str(getattr(candidate, "feasibility_reason", ""))
            for candidate in list(candidate_results or [])
        )
        failure = FailureReason(
            stage="candidate_selection",
            code="all_candidates_infeasible",
            severity="unsafe" if collision_veto else "degraded",
            recoverable=not collision_veto,
            details=str(selection_reason or "no_feasible_candidate"),
        )
        fallback = self._trajectory_fallback_manager.resolve(
            sim_time_s=float(self._sim_time_s()),
            route_revision=str(self.route_manager.route_revision),
            current_speed_mps=float(current_state[2]),
            current_reference=current_reference,
            failure_reason=failure,
        )
        reference = fallback.mutable_trajectory()
        target_lane_id = int(
            getattr(maneuver_commitment, "target_lane_id", 0)
            if committed
            else baseline_target_lane_id
        ) or int(current_lane_id)
        # A recoverable trajectory failure is not an emergency behavior
        # transition.  Turning it into ``emergency_brake`` activates the
        # direct-control hard gate on every later tick, so no repaired
        # candidate is ever generated or probed.  Preserve the committed
        # maneuver while following the bounded stop profile; reserve the
        # emergency state for an actual unsafe collision veto.
        retained_decision = (
            str(getattr(maneuver_commitment, "decision", baseline_decision))
            if committed
            else str(baseline_decision)
        )
        decision = "emergency_brake" if collision_veto else retained_decision
        destination = []
        if reference:
            terminal = dict(reference[-1])
            destination = [
                float(terminal.get("x_ref_m", terminal.get("x", current_state[0]))),
                float(terminal.get("y_ref_m", terminal.get("y", current_state[1]))),
                float(fallback.target_speed_mps),
                float(terminal.get("heading_rad", current_state[3])),
                int(target_lane_id),
            ]
        debug = {
            "stage": "fallback_manager",
            "intent_mode": str(decision),
            "fallback_reason": str(fallback.reason),
            "reference_source": "reference_line_provider:" + str(reference_mode),
            "candidate_pipeline_selected": str(fallback.mode),
            "candidate_pipeline_selected_status": "explicit_fallback",
            "candidate_pipeline_selected_reason": str(fallback.reason),
            "candidate_pipeline_count": int(len(candidate_results)),
            "candidate_pipeline_summary": str(
                summarize_candidate_results(candidate_results)
            ),
            "candidate_selected_decision": str(decision),
            "candidate_selected_lane_id": int(target_lane_id),
            "candidate_selection_failure_reason": str(failure.label()),
            "route_replan_attempted": False,
            "route_replan_succeeded": False,
        }
        if maneuver_commitment is not None:
            debug.update(maneuver_commitment.as_debug_fields())
        return (
            str(decision),
            int(target_lane_id),
            float(fallback.target_speed_mps),
            reference,
            destination,
            debug,
        )

    @staticmethod
    def _candidate_lc_state(
        *,
        decision: str,
        baseline_decision: str,
        baseline_lc_state: str,
    ) -> str:
        if str(decision) == str(baseline_decision):
            return str(baseline_lc_state or "LANE_KEEP")
        if str(decision) == "intersection_turn_left":
            return "INTERSECTION_TURN_LEFT"
        if str(decision) == "intersection_turn_right":
            return "INTERSECTION_TURN_RIGHT"
        if str(decision) == "lane_change_left":
            return "EXECUTE_LANE_CHANGE_LEFT"
        if str(decision) == "lane_change_right":
            return "EXECUTE_LANE_CHANGE_RIGHT"
        return "LANE_KEEP"

    @staticmethod
    def _normalized_final_lc_state(
        *, decision: str, lc_state: str, lane_change_phase: str = ""
    ) -> str:
        """Keep the public FSM consistent with the final selected action."""

        normalized_decision = str(decision or "").strip().lower()
        normalized_phase = str(lane_change_phase or "").strip().lower()
        if normalized_decision in {"lane_change_left", "lane_change_right"}:
            if normalized_phase == "target_lane_stabilization":
                return "TARGET_LANE_STABILIZATION"
            return (
                "EXECUTE_LANE_CHANGE_LEFT"
                if normalized_decision == "lane_change_left"
                else "EXECUTE_LANE_CHANGE_RIGHT"
            )
        return str(lc_state or "LANE_KEEP")

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

        now_s = float(self._sim_time_s())
        cooldown_s = max(
            0.1,
            float(
                self.config.get(
                    "static_obstacle_replan_cooldown_s",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_replan_cooldown_s",
                        2.0,
                    ),
                )
            ),
        )
        elapsed_s = now_s - float(self._static_obstacle_replan_last_attempt_s)
        if elapsed_s < cooldown_s:
            reason = "static_obstacle_replan_cooldown:remaining={:.2f}".format(
                cooldown_s - elapsed_s
            )
            self._static_obstacle_replan_reason = str(reason)
            return False, False, str(reason)

        self._static_obstacle_replan_last_attempt_s = float(now_s)
        block_fn = getattr(self.global_planner, "block_lane_at_position", None)
        if not callable(block_fn):
            reason = "static_obstacle_block_lane_unsupported"
            self._static_obstacle_replan_reason = str(reason)
            return True, False, str(reason)
        blocked_lane_id = block_fn({
            "x": float(obstacle.get("x", obstacle.get("x_m", 0.0))),
            "y": float(obstacle.get("y", obstacle.get("y_m", 0.0))),
            "z": float(obstacle.get("z", obstacle.get("z_m", 0.0))),
        })
        if blocked_lane_id is None:
            reason = "static_obstacle_lane_mapping_failed"
            self._static_obstacle_replan_reason = str(reason)
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
        self._static_obstacle_replan_reason = str(result.reason)
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
        requested = _mpc_cost_profile_for_behavior(
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

    def _collect_object_snapshots(self, detected_objects: Any = None) -> list[dict[str, Any]]:
        objects = detected_objects
        if objects is None:
            objects = getattr(self.vehicle_manager.perception_manager, "objects", {}) or {}
        if not isinstance(objects, Mapping):
            objects = getattr(objects, "objects", {}) or {}
        vehicles = list(objects.get("vehicles", []) or [])
        snapshots: list[dict[str, Any]] = []
        for index, obj in enumerate(vehicles):
            if isinstance(obj, Mapping):
                normalized = self._normalize_local_object_snapshot(obj)
                if normalized is not None:
                    snapshots.append(dict(normalized))
                continue
            actor = getattr(obj, "carla_actor", None) or getattr(obj, "vehicle", None) or obj
            if actor is None:
                continue
            try:
                get_transform = getattr(actor, "get_transform", None)
                transform = get_transform() if callable(get_transform) else None
                location = getattr(transform, "location", None)
                if location is None:
                    get_location = getattr(actor, "get_location", None)
                    location = (
                        get_location()
                        if callable(get_location)
                        else getattr(actor, "location", None)
                    )
                if location is None:
                    continue

                get_velocity = getattr(actor, "get_velocity", None)
                velocity = (
                    get_velocity()
                    if callable(get_velocity)
                    else getattr(actor, "velocity", None)
                )
                velocity_x = float(getattr(velocity, "x", 0.0))
                velocity_y = float(getattr(velocity, "y", 0.0))
                velocity_z = float(getattr(velocity, "z", 0.0))
                bbox = getattr(actor, "bounding_box", None)
                extent = getattr(bbox, "extent", None)
                speed_mps = math.sqrt(
                    velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2
                )
                rotation = getattr(transform, "rotation", None)
                if rotation is not None:
                    heading_rad = math.radians(float(getattr(rotation, "yaw", 0.0)))
                elif speed_mps > 0.05:
                    heading_rad = math.atan2(velocity_y, velocity_x)
                else:
                    heading_rad = 0.0

                raw_actor_id = getattr(
                    actor, "id", getattr(actor, "carla_id", None))
                try:
                    has_stable_actor_id = int(raw_actor_id) >= 0
                except (TypeError, ValueError):
                    has_stable_actor_id = bool(str(raw_actor_id or "").strip())
                actor_id = (
                    str(raw_actor_id)
                    if has_stable_actor_id
                    else "opencda_detection:%d" % int(index)
                )
                length_m = 2.0 * float(getattr(extent, "x", 2.2))
                width_m = 2.0 * float(getattr(extent, "y", 0.9))
                if not math.isfinite(length_m) or length_m <= 0.1:
                    length_m = 4.5
                if not math.isfinite(width_m) or width_m <= 0.1:
                    width_m = 2.0
                snapshots.append({
                    "vehicle_id": actor_id,
                    "id": actor_id,
                    "x": float(location.x),
                    "y": float(location.y),
                    "v": float(speed_mps),
                    "psi": float(heading_rad),
                    "length_m": float(length_m),
                    "width_m": float(width_m),
                    "source": (
                        "opencda_perception"
                        if transform is not None
                        else "opencda_ml_lidar_fusion"
                    ),
                    "provider_source": "native_opencda_perception",
                    "confidence": float(getattr(actor, "confidence", 1.0)),
                })
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return snapshots

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

    def _fused_planning_object_snapshots(
        self,
        *,
        local_object_snapshots: Sequence[Mapping[str, Any]],
        cp_obstacles: Sequence[Mapping[str, Any]],
        ego_location: carla.Location,
        sim_time_s: float,
    ) -> list[dict[str, Any]]:
        fused_by_key: dict[str, dict[str, Any]] = {}
        priorities_by_key: dict[str, int] = {}

        for snapshot in list(local_object_snapshots or []):
            normalized = self._normalize_local_object_snapshot(snapshot)
            if normalized is not None:
                self._upsert_fused_obstacle(
                    fused_by_key=fused_by_key,
                    priorities_by_key=priorities_by_key,
                    snapshot=normalized,
                    priority=self._obstacle_source_priority(normalized),
                )

        for obstacle in list(cp_obstacles or []):
            if not isinstance(obstacle, Mapping):
                continue
            if not self._cp_message_is_fresh(obstacle, sim_time_s=float(sim_time_s)):
                continue
            normalized = self._normalize_cp_obstacle_snapshot(obstacle)
            if normalized is not None:
                if self._is_duplicate_native_perception_cp_obstacle(
                    cp_snapshot=normalized,
                    fused_snapshots=fused_by_key.values(),
                ):
                    continue
                self._upsert_fused_obstacle(
                    fused_by_key=fused_by_key,
                    priorities_by_key=priorities_by_key,
                    snapshot=normalized,
                    priority=self._obstacle_source_priority(normalized),
                )

        return list(fused_by_key.values())

    @staticmethod
    def _is_duplicate_native_perception_cp_obstacle(
        *,
        cp_snapshot: Mapping[str, Any],
        fused_snapshots: Sequence[Mapping[str, Any]],
        max_position_delta_m: float = 1.0,
    ) -> bool:
        provider_source = str(cp_snapshot.get("provider_source", "")).strip().lower()
        source = str(cp_snapshot.get("source", "")).strip().lower()
        if "perception" not in provider_source and "perception" not in source:
            return False
        try:
            cp_x = float(cp_snapshot.get("x", 0.0))
            cp_y = float(cp_snapshot.get("y", 0.0))
        except Exception:
            return False
        for existing in list(fused_snapshots or []):
            existing_provider = str(existing.get("provider_source", "")).strip().lower()
            existing_source = str(existing.get("source", "")).strip().lower()
            if "perception" not in existing_provider and "perception" not in existing_source:
                continue
            try:
                dx = cp_x - float(existing.get("x", 0.0))
                dy = cp_y - float(existing.get("y", 0.0))
            except Exception:
                continue
            if math.hypot(dx, dy) <= float(max_position_delta_m):
                return True
        return False

    def _limit_obstacles_for_mpc(
        self,
        *,
        object_snapshots: Sequence[Mapping[str, Any]],
        ego_location: carla.Location,
    ) -> list[dict[str, Any]]:
        fused = [dict(item) for item in list(object_snapshots or []) if isinstance(item, Mapping)]
        if self.max_mpc_obstacles > 0 and len(fused) > self.max_mpc_obstacles:
            fused.sort(
                key=lambda item: (
                    float(item.get("x", 0.0)) - float(ego_location.x)
                ) ** 2
                + (
                    float(item.get("y", 0.0)) - float(ego_location.y)
                ) ** 2
            )
            fused = fused[: self.max_mpc_obstacles]
        return fused

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
    def _normalize_local_object_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
        try:
            obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
            if not obstacle_id:
                return None
            return {
                "vehicle_id": obstacle_id,
                "id": obstacle_id,
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "v": float(snapshot.get("v", 0.0)),
                "psi": float(snapshot.get("psi", 0.0)),
                "length_m": float(snapshot.get("length_m", 4.5)),
                "width_m": float(snapshot.get("width_m", 2.0)),
                "source": str(snapshot.get("source", "opencda_perception")),
                "provider_source": str(snapshot.get("provider_source", "native_opencda_perception")),
                "confidence": float(snapshot.get("confidence", 1.0)),
            }
        except Exception:
            return None

    @staticmethod
    def _normalize_cp_obstacle_snapshot(obstacle: Mapping[str, Any]) -> dict[str, Any] | None:
        try:
            raw_id = str(obstacle.get("id", obstacle.get("vehicle_id", ""))).strip()
            if not raw_id:
                return None
            state = obstacle.get("state", [])
            if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray)):
                state_values = list(state)
            else:
                state_values = []
            x_m = obstacle.get("x", obstacle.get("x_m", state_values[0] if len(state_values) >= 1 else None))
            y_m = obstacle.get("y", obstacle.get("y_m", state_values[1] if len(state_values) >= 2 else None))
            speed_mps = obstacle.get("v", obstacle.get("speed_mps", state_values[2] if len(state_values) >= 3 else 0.0))
            heading_rad = obstacle.get("psi", obstacle.get("heading_rad", state_values[3] if len(state_values) >= 4 else 0.0))
            if x_m is None or y_m is None:
                return None
            shape = obstacle.get("shape", {})
            shape = dict(shape) if isinstance(shape, Mapping) else {}
            obstacle_id = raw_id.rsplit(":", 1)[-1] if ":" in raw_id else raw_id
            provider_source = str(obstacle.get("provider_source", "opencda_cp"))
            source = str(obstacle.get("source", "opencda_cp"))
            return {
                "vehicle_id": obstacle_id,
                "id": obstacle_id,
                "cp_message_id": raw_id,
                "x": float(x_m),
                "y": float(y_m),
                "v": float(speed_mps),
                "psi": float(heading_rad),
                "length_m": float(shape.get("length_m", obstacle.get("length_m", 4.5))),
                "width_m": float(shape.get("width_m", obstacle.get("width_m", 2.0))),
                "source": source,
                "provider_source": provider_source,
                "confidence": float(obstacle.get("confidence", 0.5)),
                "lane_id": int(float(obstacle.get("lane_id", 0) or 0)),
                "road_id": int(float(obstacle.get("road_id", 0) or 0)),
                "object_type": str(obstacle.get("type", "unknown")),
                "observed_by_cav_ids": list(
                    obstacle.get("observed_by_cav_ids", []) or []
                ),
                "not_observed_by_cav_ids": list(
                    obstacle.get("not_observed_by_cav_ids", []) or []
                ),
                "blind_spot_shared": bool(
                    obstacle.get("blind_spot_shared", False)
                ),
            }
        except Exception:
            return None

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

    @staticmethod
    def _obstacle_source_priority(snapshot: Mapping[str, Any]) -> int:
        provider_source = str(snapshot.get("provider_source", "")).lower()
        source = str(snapshot.get("source", "")).lower()
        if "perception" in provider_source or "perception" in source:
            return 100
        if "v2x" in provider_source or "v2x" in source:
            return 80
        if "fallback" in provider_source or "fallback" in source or "carla" in source:
            return 40
        return 60

    @staticmethod
    def _fused_obstacle_key(snapshot: Mapping[str, Any]) -> str:
        obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
        return obstacle_id.rsplit(":", 1)[-1] if ":" in obstacle_id else obstacle_id

    @classmethod
    def _upsert_fused_obstacle(
        cls,
        *,
        fused_by_key: dict[str, dict[str, Any]],
        priorities_by_key: dict[str, int],
        snapshot: Mapping[str, Any],
        priority: int,
    ) -> None:
        key = cls._fused_obstacle_key(snapshot)
        if not key:
            return
        previous_priority = int(priorities_by_key.get(key, -1))
        previous = fused_by_key.get(key)
        previous_confidence = float(previous.get("confidence", 0.0)) if isinstance(previous, Mapping) else -1.0
        confidence = float(snapshot.get("confidence", 0.0))
        if int(priority) > previous_priority or (
            int(priority) == previous_priority and float(confidence) >= previous_confidence
        ):
            fused_by_key[key] = dict(snapshot)
            priorities_by_key[key] = int(priority)

    def _draw_world_debug_primitives(
        self,
        *,
        destination_state: Sequence[float],
        lane_center_reference: Sequence[Mapping[str, Any]],
    ) -> None:
        """Draw planner primitives into CARLA's debug layer for OpenCDA runs."""

        if not bool(self.draw_world_debug):
            return
        try:
            world = self.vehicle_manager.vehicle.get_world()
            debug = getattr(world, "debug", None)
            if debug is None:
                return
            z_m = float(getattr(self.vehicle_manager.vehicle.get_location(), "z", 0.0)) + 0.35
            life_time = max(0.05, float(self.world_debug_life_time_s))

            route_points = self._display_global_route_points()
            self._draw_debug_polyline(
                debug=debug,
                points_xy=[(float(p[0]), float(p[1])) for p in route_points],
                z_m=z_m + 0.05,
                color=self.carla.Color(255, 210, 20),
                thickness=0.08,
                life_time_s=life_time,
                max_segments=120,
            )

            reference_points = [
                (
                    float(sample.get("x_ref_m", sample.get("x", 0.0))),
                    float(sample.get("y_ref_m", sample.get("y", 0.0))),
                )
                for sample in list(lane_center_reference or [])
            ]
            self._draw_debug_polyline(
                debug=debug,
                points_xy=reference_points,
                z_m=z_m + 0.15,
                color=self.carla.Color(245, 245, 245),
                thickness=0.06,
                life_time_s=life_time,
                max_segments=80,
            )

            mpc_points = self._last_mpc_trajectory_points()
            self._draw_debug_polyline(
                debug=debug,
                points_xy=mpc_points,
                z_m=z_m + 0.25,
                color=self.carla.Color(30, 230, 70),
                thickness=0.10,
                life_time_s=life_time,
                max_segments=80,
            )

            if (
                bool(self.draw_world_debug_destination)
                and destination_state is not None
                and len(destination_state) >= 2
            ):
                debug.draw_point(
                    self.carla.Location(
                        x=float(destination_state[0]),
                        y=float(destination_state[1]),
                        z=z_m + 0.55,
                    ),
                    size=0.18,
                    color=self.carla.Color(30, 145, 255),
                    life_time=life_time,
                    persistent_lines=False,
                )
        except Exception as exc:
            if self.debug:
                print(f"[CP-X OpenCDA Bridge] world debug draw failed: {exc}")

    def _last_mpc_trajectory_points(self) -> list[tuple[float, float]]:
        x_solution = getattr(self.mpc, "_last_x_solution", None)
        if x_solution is None:
            return []
        points: list[tuple[float, float]] = []
        try:
            for state in list(x_solution):
                if len(state) < 2:
                    continue
                points.append((float(state[0]), float(state[1])))
        except Exception:
            return []
        return points

    def _draw_debug_polyline(
        self,
        *,
        debug: Any,
        points_xy: Sequence[Sequence[float]],
        z_m: float,
        color: Any,
        thickness: float,
        life_time_s: float,
        max_segments: int,
    ) -> None:
        points = [
            (float(point[0]), float(point[1]))
            for point in list(points_xy or [])
            if len(point) >= 2
        ]
        if len(points) < 2:
            return
        stride = max(1, int(len(points) / max(1, int(max_segments))))
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        for first, second in zip(sampled[:-1], sampled[1:]):
            if math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1])) < 1.0e-3:
                continue
            debug.draw_line(
                self.carla.Location(x=float(first[0]), y=float(first[1]), z=float(z_m)),
                self.carla.Location(x=float(second[0]), y=float(second[1]), z=float(z_m)),
                thickness=float(thickness),
                color=color,
                life_time=float(life_time_s),
                persistent_lines=False,
            )

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
        """Return a display-only smoothing of the active topology polyline.

        AD-map lane-change edges are topological cross-lane markers and may be
        nearly lateral in XY.  The minimap must not render that marker as a
        physical 90-degree road corner.  This filtered copy is diagnostics
        only: route progress, authorization, references and MPC continue to
        consume ``_active_global_route_points`` unchanged.
        """
        raw = [list(point) for point in self._active_global_route_points()]
        if len(raw) < 5:
            return raw
        signature_indices = sorted({0, len(raw) // 4, len(raw) // 2, 3 * len(raw) // 4, len(raw) - 1})
        signature = tuple(
            (len(raw), index, round(float(raw[index][0]), 3), round(float(raw[index][1]), 3))
            for index in signature_indices
        )
        if signature == getattr(self, "_display_route_cache_signature", None):
            return [list(point) for point in getattr(self, "_display_route_cache", ())]

        xy = [(float(point[0]), float(point[1])) for point in raw]
        # A symmetric triangular metric-like filter spreads a short lateral
        # topology marker over neighboring longitudinal samples. Straight
        # sections remain exactly straight and endpoints remain unchanged.
        radius = 6
        smoothed: list[tuple[float, float]] = []
        for index in range(len(xy)):
            if index == 0 or index == len(xy) - 1:
                smoothed.append(xy[index])
                continue
            first = max(0, index - radius)
            last = min(len(xy) - 1, index + radius)
            weighted_x = 0.0
            weighted_y = 0.0
            total_weight = 0.0
            for neighbor in range(first, last + 1):
                weight = float(radius + 1 - abs(neighbor - index))
                weighted_x += weight * xy[neighbor][0]
                weighted_y += weight * xy[neighbor][1]
                total_weight += weight
            smoothed.append((weighted_x / total_weight, weighted_y / total_weight))

        display: list[list[float]] = []
        for index, (x_m, y_m) in enumerate(smoothed):
            other = smoothed[index + 1] if index + 1 < len(smoothed) else smoothed[index - 1]
            base = smoothed[index] if index + 1 < len(smoothed) else smoothed[index - 1]
            heading = math.atan2(other[1] - base[1], other[0] - base[0])
            z_m = float(raw[index][2]) if len(raw[index]) >= 3 else 0.0
            display.append([float(x_m), float(y_m), float(z_m), float(heading)])
        self._display_route_cache_signature = signature
        self._display_route_cache = tuple(tuple(point) for point in display)
        return display

    def _map_waypoint_from_location(self, location: carla.Location):
        waypoint_map_planner = getattr(
            self,
            "waypoint_map_planner",
            self.map_planner,
        )
        if waypoint_map_planner is None:
            return None
        point = {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }
        get_waypoint = getattr(waypoint_map_planner, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        try:
            return get_waypoint(point)
        except Exception:
            pass
        try:
            return get_waypoint(
                carla.Location(
                    x=float(location.x),
                    y=float(location.y),
                    z=float(location.z),
                )
            )
        except Exception:
            return None

    def _drivable_waypoint_from_location(
        self,
        location: carla.Location,
    ):
        """Return a waypoint only when the point is on a driving lane."""

        waypoint_map_planner = getattr(
            self,
            "waypoint_map_planner",
            self.map_planner,
        )
        get_drivable_waypoint = getattr(
            waypoint_map_planner,
            "get_drivable_waypoint",
            None,
        )
        if callable(get_drivable_waypoint):
            try:
                return get_drivable_waypoint({
                    "x": float(location.x),
                    "y": float(location.y),
                    "z": float(location.z),
                })
            except Exception:
                return None

        try:
            world = self.vehicle_manager.vehicle.get_world()
            carla_map = world.get_map()
            return carla_map.get_waypoint(
                carla.Location(
                    x=float(location.x),
                    y=float(location.y),
                    z=float(location.z),
                ),
                project_to_road=False,
                lane_type=self.carla.LaneType.Driving,
            )
        except TypeError:
            try:
                return carla_map.get_waypoint(
                    carla.Location(
                        x=float(location.x),
                        y=float(location.y),
                        z=float(location.z),
                    ),
                    project_to_road=False,
                )
            except Exception:
                return None
        except Exception:
            return None

    def _lane_id_at_location(self, location: carla.Location) -> int:
        waypoint = self.reference_map.get_waypoint(
            {
                "x": float(location.x),
                "y": float(location.y),
                "z": float(getattr(location, "z", 0.0)),
            }
        )
        if waypoint is None:
            return 0
        try:
            from utility.global_planner import canonical_lane_id_for_waypoint

            lane_id = int(canonical_lane_id_for_waypoint(waypoint) or 0)
            return lane_id
        except Exception:
            return int(getattr(waypoint, "ad_lane_id", 0) or 0)

    @staticmethod
    def _location_to_point(location: Any) -> dict[str, float]:
        return {
            "x": float(getattr(location, "x", 0.0)),
            "y": float(getattr(location, "y", 0.0)),
            "z": float(getattr(location, "z", 0.0)),
        }

    def _front_gap_m(
        self,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        lane_change_direction: str = "",
        lane_change_progress: float = 0.0,
        current_lane_id: Optional[int] = None,
        lane_assignments: Optional[Mapping[str, int]] = None,
        return_actor_id: bool = False,
    ):
        """Nearest-ahead gap in ego's body frame.

        Outside an active lane change (``lane_change_direction == ""``),
        this is a plain nearest-ahead search within a +/-2.5 m lateral
        gate -- unchanged from before.

        During an active lane change ("left"/"right"), the source lane's
        front vehicle must not be dropped the instant the maneuver starts
        (ego hasn't moved yet -- it's still physically in the source lane),
        but also must not keep braking ego once ego's body has actually
        cleared it. This computes the source-lane gap and target-lane gap
        *separately* (split at ego's current heading, not by lane_id) and
        blends between them as a smooth function of ``lane_change_progress``
        (alpha in [0, 1], 0 = still at the source lane center, 1 = at the
        target lane center -- pass ManeuverManager lane-change progress):

          - alpha <= alpha_clear: fully the source-lane gap. alpha_clear is
            the progress at which ego's own body -- not just its center --
            has crossed the source/target lane boundary, derived from
            vehicle width and lane width, not a fixed distance or a
            lane_id switch: alpha_clear = 0.5 + vehicle_width_m / (2 *
            lane_width_m).
          - alpha_clear < alpha < 1: smoothstep blend toward the
            target-lane gap.
          - alpha >= 1: fully the target-lane gap.

        Sign convention (lateral = -dx*sin_h + dy*cos_h): validated against
        the cpx_lane_change_speed_* scenarios -- positive lateral is the
        left-hand side of ego's current heading.

        ``return_actor_id=True`` also returns the id of whichever object
        dominates the blended gap (None if neither side has one), so a
        caller can detect "the object being used as my front-vehicle
        reference just changed" even when the gap distance itself moves
        smoothly -- see control_context_key in _run_full_cpx_pipeline_step.
        """

        cos_h = math.cos(ego_yaw_rad)
        sin_h = math.sin(ego_yaw_rad)
        ego_half_length_m = 2.25
        try:
            ego_half_length_m = max(
                0.0,
                float(self.vehicle_manager.vehicle.bounding_box.extent.x),
            )
        except Exception:
            pass

        def _nearest_gap(
            *, min_lateral_m: float, max_lateral_m: float
        ) -> tuple[Optional[float], Optional[str]]:
            best_gap = None
            best_actor_id = None
            for snapshot in object_snapshots:
                if current_lane_id is not None and lane_assignments is not None:
                    obstacle_id = self._object_track_id(snapshot)
                    assigned_lane_id = int(
                        lane_assignments.get(str(obstacle_id), 0) or 0
                    )
                    if assigned_lane_id != int(current_lane_id):
                        continue
                dx = float(snapshot.get("x", 0.0)) - float(ego_location.x)
                dy = float(snapshot.get("y", 0.0)) - float(ego_location.y)
                longitudinal = dx * cos_h + dy * sin_h
                lateral = -dx * sin_h + dy * cos_h
                if (
                    longitudinal <= 0.0
                    or lateral < float(min_lateral_m)
                    or lateral > float(max_lateral_m)
                ):
                    continue
                object_half_length_m = max(
                    0.0,
                    0.5 * float(snapshot.get("length_m", 4.5) or 4.5),
                )
                clearance_m = max(
                    0.0,
                    float(longitudinal)
                    - float(ego_half_length_m)
                    - float(object_half_length_m),
                )
                if best_gap is None or float(clearance_m) < float(best_gap):
                    best_gap = float(clearance_m)
                    best_actor_id = self._object_track_id(snapshot)
            return best_gap, best_actor_id

        # When stable map assignments are available, "own lane" is the
        # currently map-matched ego lane. Adjacent-lane actors never enter
        # longitudinal following, including during a lane change; once ego's
        # map match moves to the target lane, that lane naturally becomes its
        # own lane on the next tick.
        strict_current_lane = bool(
            current_lane_id is not None and lane_assignments is not None
        )
        direction = (
            ""
            if strict_current_lane
            else str(lane_change_direction or "").strip().lower()
        )
        if direction not in {"left", "right"}:
            best_gap, best_actor_id = _nearest_gap(
                min_lateral_m=-2.5, max_lateral_m=2.5
            )
            if bool(return_actor_id):
                return best_gap, (
                    None if best_gap is None else str(best_actor_id)
                )
            return best_gap

        # A small overlap around the ego-heading split line keeps an object
        # sitting right at the boundary visible to both searches, instead
        # of a strict 0.0 cutoff creating a blind seam between them.
        boundary_overlap_m = max(
            0.0, float(self.config.get("lane_change_boundary_overlap_m", 0.75))
        )
        if direction == "left":
            source_gap, source_actor_id = _nearest_gap(
                min_lateral_m=-2.5, max_lateral_m=boundary_overlap_m
            )
            target_gap, target_actor_id = _nearest_gap(
                min_lateral_m=-boundary_overlap_m, max_lateral_m=2.5
            )
        else:
            source_gap, source_actor_id = _nearest_gap(
                min_lateral_m=-boundary_overlap_m, max_lateral_m=2.5
            )
            target_gap, target_actor_id = _nearest_gap(
                min_lateral_m=-2.5, max_lateral_m=boundary_overlap_m
            )

        try:
            vehicle_width_m = max(
                0.5,
                float(self.vehicle_manager.vehicle.bounding_box.extent.y) * 2.0,
            )
        except Exception:
            vehicle_width_m = 2.0
        lane_width_m = max(1.0, float(getattr(self.mpc, "lane_width_m", 3.5)))
        alpha_clear = min(
            0.95, 0.5 + float(vehicle_width_m) / (2.0 * float(lane_width_m))
        )
        alpha = max(0.0, min(1.0, float(lane_change_progress)))
        if alpha <= alpha_clear:
            blend_weight = 0.0
        else:
            span = max(1.0e-6, 1.0 - float(alpha_clear))
            ramp = min(1.0, (float(alpha) - float(alpha_clear)) / float(span))
            blend_weight = float(ramp) * float(ramp) * (3.0 - 2.0 * float(ramp))

        _no_constraint_gap_m = 1.0e6
        source_value = (
            _no_constraint_gap_m if source_gap is None else float(source_gap)
        )
        target_value = (
            _no_constraint_gap_m if target_gap is None else float(target_gap)
        )
        blended_gap = (
            (1.0 - blend_weight) * source_value + blend_weight * target_value
        )
        best_gap = (
            None if blended_gap >= 0.5 * _no_constraint_gap_m else float(blended_gap)
        )
        best_actor_id = (
            target_actor_id if blend_weight >= 0.5 else source_actor_id
        )
        if bool(return_actor_id):
            return best_gap, (None if best_gap is None else str(best_actor_id))
        return best_gap

    @staticmethod
    def _body_frame_xy(
        *,
        origin_x_m: float,
        origin_y_m: float,
        heading_rad: float,
        target_x_m: float,
        target_y_m: float,
    ) -> tuple[float, float]:
        dx_m = float(target_x_m) - float(origin_x_m)
        dy_m = float(target_y_m) - float(origin_y_m)
        cos_h = math.cos(float(heading_rad))
        sin_h = math.sin(float(heading_rad))
        forward_m = dx_m * cos_h + dy_m * sin_h
        lateral_m = -dx_m * sin_h + dy_m * cos_h
        return float(forward_m), float(lateral_m)

    def _set_actuator_context(
        self,
        *,
        ego_speed_mps: float,
        target_speed_mps: float,
        stop_goal_active: bool,
    ) -> None:
        self._actuator_ego_speed_mps = float(ego_speed_mps)
        self._actuator_target_speed_mps = float(target_speed_mps)
        self._actuator_stop_goal_active = bool(stop_goal_active)

    def _control_from_mpc(self, acceleration_mps2: float, steering_angle_rad: float) -> carla.VehicleControl:
        max_accel = max(1e-6, float(self.mpc.constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        pedals = self.actuator_mapper.map_acceleration(
            acceleration_mps2=float(acceleration_mps2),
            max_acceleration_mps2=float(max_accel),
            min_acceleration_mps2=-float(max_brake),
            ego_speed_mps=float(self._actuator_ego_speed_mps),
            target_speed_mps=float(self._actuator_target_speed_mps),
            stop_goal_active=bool(self._actuator_stop_goal_active),
            timestamp_s=float(self._sim_time_s()),
        )
        steer = min(1.0, max(-1.0, float(steering_angle_rad) / max_steer))
        return carla.VehicleControl(
            throttle=float(pedals.throttle),
            brake=float(pedals.brake),
            steer=steer,
        )

    def _accel_from_control(self, control: carla.VehicleControl) -> float:
        max_accel = max(1e-6, float(self.mpc.constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        return self.actuator_mapper.acceleration_from_command(
            throttle=float(getattr(control, "throttle", 0.0)),
            brake=float(getattr(control, "brake", 0.0)),
            max_acceleration_mps2=float(max_accel),
            min_acceleration_mps2=-float(max_brake),
            ego_speed_mps=float(self._actuator_ego_speed_mps),
            target_speed_mps=float(self._actuator_target_speed_mps),
            stop_goal_active=bool(self._actuator_stop_goal_active),
        )

    def _steer_rad_from_control(self, control: carla.VehicleControl) -> float:
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        return float(getattr(control, "steer", 0.0)) * float(max_steer)

    def _emergency_stop_control(self) -> carla.VehicleControl:
        self._last_accel_mps2 = float(getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0))
        self._last_steer_rad = 0.0
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)

    def _full_reference_lateral_guard_reason(
        self,
        *,
        decision: str,
        lc_state: str,
        stop_goal_active: bool,
        destination_state: Sequence[float] | None,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        heading_error_rad: float = float("nan"),
    ) -> str:
        normalized_decision = str(decision or "").strip().lower()
        normalized_lc_state = str(lc_state or "").strip().upper()
        lane_follow_like = (
            normalized_decision == "lane_follow"
            and normalized_lc_state in {"", "IDLE", "LANE_KEEP"}
        )
        stop_like = bool(stop_goal_active) or normalized_decision in {
            "stop_at_intersection",
            "stop_sign",
        }
        if not bool(lane_follow_like or stop_like):
            return ""

        max_destination_lateral_m = (
            float(self.full_stop_max_destination_lateral_m)
            if bool(stop_like)
            else float(self.full_lane_follow_max_destination_lateral_m)
        )
        max_reference_first_lateral_m = (
            float(self.full_stop_max_reference_first_lateral_m)
            if bool(stop_like)
            else float(self.full_lane_follow_max_reference_first_lateral_m)
        )
        reasons: list[str] = []
        if destination_state is not None and len(destination_state) >= 2:
            _, destination_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(destination_state[0]),
                target_y_m=float(destination_state[1]),
            )
            if abs(float(destination_lateral_m)) > float(max_destination_lateral_m):
                reasons.append(f"dest_lat={destination_lateral_m:.2f}")

        if lane_center_reference:
            first = dict(list(lane_center_reference)[0])
            _, reference_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(first.get("x_ref_m", first.get("x", ego_location.x))),
                target_y_m=float(first.get("y_ref_m", first.get("y", ego_location.y))),
            )
            if abs(float(reference_lateral_m)) > float(max_reference_first_lateral_m):
                reasons.append(f"ref_lat={reference_lateral_m:.2f}")

        if math.isfinite(float(heading_error_rad)):
            max_heading_error_deg = (
                float(self.full_stop_max_heading_error_deg)
                if bool(stop_like)
                else float(self.full_lane_follow_max_heading_error_deg)
            )
            heading_error_deg = math.degrees(float(heading_error_rad))
            if abs(float(heading_error_deg)) > float(max_heading_error_deg):
                reasons.append(f"heading={heading_error_deg:.2f}")

        if not reasons:
            return ""
        mode = "stop" if bool(stop_like) else "lane_follow"
        return f"{mode}_lateral_guard:" + ":".join(reasons)


    def _low_speed_control_buffer_force_replan(
        self,
        *,
        ego_speed_mps: float,
        behavior_decision: str,
        behavior_fsm_state: str,
        stop_goal_active: bool,
    ) -> bool:
        """Keep low-speed control closed-loop until the vehicle is moving."""

        normalized_behavior = str(behavior_decision or "").strip().lower()
        normalized_fsm = str(behavior_fsm_state or "").strip().upper()
        return bool(
            not bool(stop_goal_active)
            and normalized_behavior == "lane_follow"
            and normalized_fsm in {"", "IDLE", "LANE_KEEP"}
            and float(ego_speed_mps)
            < float(self.full_control_buffer_min_speed_mps)
        )

    def _fallback_control(
        self,
        ego_transform: carla.Transform,
        ego_speed_mps: float,
        destination_state: Sequence[float],
        stop_goal_active: bool,
    ) -> carla.VehicleControl:
        if stop_goal_active:
            self._last_accel_mps2 = float(self.mpc.constraints.min_acceleration_mps2)
            self._last_steer_rad = 0.0
            return carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0)

        dx = float(destination_state[0]) - float(ego_transform.location.x)
        dy = float(destination_state[1]) - float(ego_transform.location.y)
        target_yaw = math.atan2(dy, dx)
        yaw_error = self._wrap_angle(target_yaw - math.radians(float(ego_transform.rotation.yaw)))
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        steer_rad = min(max_steer, max(-max_steer, 0.7 * yaw_error))
        speed_error = float(self.target_speed_mps) - float(ego_speed_mps)
        accel = min(
            float(self.mpc.constraints.max_acceleration_mps2),
            max(float(self.mpc.constraints.min_acceleration_mps2), 0.6 * speed_error),
        )
        self._last_accel_mps2 = float(accel)
        self._last_steer_rad = float(steer_rad)
        return self._control_from_mpc(accel, steer_rad)

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


def _mpc_cost_profile_for_behavior(
    *,
    behavior: str,
    planner_lc_state: str,
    planner_mode: str,
    next_macro_maneuver: str,
) -> str:
    from opencda.planning_module.behavior_planner import (
        is_emergency_brake_decision,
        is_fixed_stop_decision,
        normalize_behavior_decision,
    )

    raw_behavior = str(behavior or "").strip().lower()
    normalized_behavior = str(normalize_behavior_decision(behavior))
    normalized_lc_state = str(planner_lc_state or "").strip().upper()
    normalized_mode = str(planner_mode or "").strip().upper()
    normalized_maneuver = str(next_macro_maneuver or "straight").strip().lower()
    if bool(is_fixed_stop_decision(normalized_behavior)):
        return "stop"
    if bool(is_emergency_brake_decision(normalized_behavior)):
        return "recovery"
    if normalized_lc_state.startswith("PREPARE_LANE_CHANGE"):
        return "prepare_lane_change"
    if raw_behavior in {"intersection_turn_left", "intersection_turn_right"}:
        return "intersection_turn"
    if normalized_lc_state.startswith("EXECUTE_LANE_CHANGE") or normalized_behavior in {
        "lane_change_left",
        "lane_change_right",
    }:
        return "execute_lane_change"
    if normalized_mode == "INTERSECTION" and normalized_maneuver in {"left", "right"}:
        return "intersection_turn"
    return "lane_follow"


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
