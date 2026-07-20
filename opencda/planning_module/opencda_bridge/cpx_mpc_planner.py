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
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import carla
import yaml


class _CarlaMapPlannerAdapter:
    """Small adapter so planning-module helpers can query a CARLA map with dict poses."""

    def __init__(self, map_planner: Any):
        self._map_planner = map_planner

    def get_waypoint(self, point: Any):
        get_waypoint = getattr(self._map_planner, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        if isinstance(point, Mapping):
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


class _Mode2TrafficLightMemory:
    """Temporal debounce for traffic controls in OpenCDA-reference MPC mode."""

    def __init__(self, *, hold_unknown_s: float = 0.8) -> None:
        self.hold_unknown_s = max(0.0, float(hold_unknown_s))
        self._last_stop_state = "unknown"
        self._last_stop_target: dict[str, object] | None = None
        self._hold_until_s = -float("inf")

    def update(
        self,
        *,
        state: str,
        stop_target: Mapping[str, object] | None,
        sim_time_s: float,
    ) -> tuple[str, dict[str, object] | None, str]:
        normalized_state = str(state or "unknown").strip().lower()
        reason = ""
        if normalized_state in {"red", "yellow"}:
            self._last_stop_state = str(normalized_state)
            self._last_stop_target = dict(stop_target or {}) if isinstance(stop_target, Mapping) else None
            self._hold_until_s = float(sim_time_s) + float(self.hold_unknown_s)
            return str(normalized_state), self._last_stop_target, "raw_stop"
        if (
            normalized_state == "unknown"
            and float(sim_time_s) <= float(self._hold_until_s)
            and self._last_stop_state in {"red", "yellow"}
        ):
            reason = f"traffic_memory_hold_{self._last_stop_state}"
            return str(self._last_stop_state), self._last_stop_target, reason
        if normalized_state == "green":
            self._last_stop_state = "green"
            self._last_stop_target = None
            self._hold_until_s = -float("inf")
        return str(normalized_state), None, reason


class _Mode2ObjectTrackMemory:
    """Sliding-window object state smoothing for MPC inputs."""

    def __init__(
        self,
        *,
        alpha: float = 0.55,
        max_stale_s: float = 0.4,
    ) -> None:
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self.max_stale_s = max(0.0, float(max_stale_s))
        self._tracks: dict[str, dict[str, object]] = {}

    def update(
        self,
        *,
        object_snapshots: Sequence[Mapping[str, object]],
        sim_time_s: float,
    ) -> tuple[list[dict[str, object]], str]:
        seen_ids: set[str] = set()
        for index, snapshot in enumerate(list(object_snapshots or [])):
            track_id = self._track_id(snapshot, index)
            seen_ids.add(track_id)
            current = dict(snapshot)
            previous = self._tracks.get(track_id)
            if previous is not None:
                current = self._smooth_snapshot(previous, current)
                current["track_age_frames"] = int(previous.get("track_age_frames", 0) or 0) + 1
            else:
                current["track_age_frames"] = 1
            current["last_seen_s"] = float(sim_time_s)
            current["memory_track_id"] = str(track_id)
            current["object_memory_fresh"] = True
            self._tracks[track_id] = dict(current)

        output: list[dict[str, object]] = []
        stale_count = 0
        for track_id, track in list(self._tracks.items()):
            age_s = float(sim_time_s) - float(track.get("last_seen_s", -float("inf")) or -float("inf"))
            if age_s > float(self.max_stale_s):
                self._tracks.pop(track_id, None)
                continue
            snapshot = dict(track)
            if track_id not in seen_ids:
                stale_count += 1
                snapshot["object_memory_fresh"] = False
                snapshot["source"] = str(snapshot.get("source", "")) + ":memory_hold"
            output.append(snapshot)
        return output, f"object_memory_tracks={len(output)}:stale={stale_count}"

    @staticmethod
    def _track_id(snapshot: Mapping[str, object], index: int) -> str:
        for key in ("vehicle_id", "id", "track_id", "memory_track_id"):
            value = snapshot.get(key)
            if value not in {None, ""}:
                return str(value)
        return f"anonymous:{index}"

    def _smooth_snapshot(
        self,
        previous: Mapping[str, object],
        current: Mapping[str, object],
    ) -> dict[str, object]:
        smoothed = dict(current)
        for key in ("x", "y", "v", "speed_mps", "length_m", "width_m"):
            if key in current and key in previous:
                try:
                    smoothed[key] = (
                        float(self.alpha) * float(current[key])
                        + (1.0 - float(self.alpha)) * float(previous[key])
                    )
                except Exception:
                    pass
        if "psi" in current and "psi" in previous:
            try:
                previous_psi = float(previous["psi"])
                current_psi = float(current["psi"])
                delta = math.atan2(
                    math.sin(current_psi - previous_psi),
                    math.cos(current_psi - previous_psi),
                )
                smoothed["psi"] = previous_psi + float(self.alpha) * delta
            except Exception:
                pass
        return smoothed


class _Mode2ReferenceMemory:
    """Keep the accepted OpenCDA reference temporally consistent."""

    def __init__(
        self,
        *,
        max_first_point_jump_m: float = 2.0,
        max_destination_jump_m: float = 4.0,
        max_reuse_age_s: float = 1.0,
    ) -> None:
        self.max_first_point_jump_m = max(0.0, float(max_first_point_jump_m))
        self.max_destination_jump_m = max(0.0, float(max_destination_jump_m))
        self.max_reuse_age_s = max(0.0, float(max_reuse_age_s))
        self._last_reference: list[dict[str, object]] = []
        self._last_destination: list[float] | None = None
        self._last_time_s = -float("inf")

    def stabilize(
        self,
        *,
        reference: Sequence[Mapping[str, object]],
        destination_state: Sequence[float] | None,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        stop_goal_active: bool,
        sim_time_s: float,
    ) -> tuple[list[dict[str, object]], list[float] | None, str]:
        current_reference = [dict(sample) for sample in list(reference or [])]
        current_destination = list(destination_state) if destination_state is not None else None
        if bool(stop_goal_active) or not self._last_reference or self._last_destination is None:
            self._accept(current_reference, current_destination, sim_time_s)
            return current_reference, current_destination, "reference_memory_accept"
        if not current_reference or current_destination is None:
            reused_reference, reused_destination = self._reusable_previous(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                sim_time_s=float(sim_time_s),
            )
            if reused_reference and reused_destination is not None:
                return reused_reference, reused_destination, "reference_memory_reuse_missing"
            return current_reference, current_destination, "reference_memory_missing"

        first_jump_m = self._point_jump_m(current_reference[0], self._last_reference[0])
        destination_jump_m = math.hypot(
            float(current_destination[0]) - float(self._last_destination[0]),
            float(current_destination[1]) - float(self._last_destination[1]),
        )
        if (
            first_jump_m > float(self.max_first_point_jump_m)
            or destination_jump_m > float(self.max_destination_jump_m)
        ):
            reused_reference, reused_destination = self._reusable_previous(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                sim_time_s=float(sim_time_s),
            )
            if reused_reference and reused_destination is not None:
                return reused_reference, reused_destination, (
                    "reference_memory_reuse_jump"
                    f":first={first_jump_m:.2f}:dest={destination_jump_m:.2f}"
                )
        self._accept(current_reference, current_destination, sim_time_s)
        return current_reference, current_destination, "reference_memory_accept"

    def _accept(
        self,
        reference: Sequence[Mapping[str, object]],
        destination_state: Sequence[float] | None,
        sim_time_s: float,
    ) -> None:
        self._last_reference = [dict(sample) for sample in list(reference or [])]
        self._last_destination = list(destination_state) if destination_state is not None else None
        self._last_time_s = float(sim_time_s)

    def reset(self) -> None:
        self._last_reference = []
        self._last_destination = None
        self._last_time_s = -float("inf")

    def _reusable_previous(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        sim_time_s: float,
    ) -> tuple[list[dict[str, object]], list[float] | None]:
        if float(sim_time_s) - float(self._last_time_s) > float(self.max_reuse_age_s):
            return [], None
        cos_h = math.cos(float(ego_yaw_rad))
        sin_h = math.sin(float(ego_yaw_rad))
        kept: list[dict[str, object]] = []
        for sample in list(self._last_reference or []):
            x_m = float(sample.get("x_ref_m", sample.get("x", ego_location.x)))
            y_m = float(sample.get("y_ref_m", sample.get("y", ego_location.y)))
            dx_m = x_m - float(ego_location.x)
            dy_m = y_m - float(ego_location.y)
            forward_m = dx_m * cos_h + dy_m * sin_h
            if forward_m >= -0.25:
                kept.append(dict(sample))
        if len(kept) < 2 or self._last_destination is None:
            return [], None
        return kept, list(self._last_destination)

    @staticmethod
    def _point_jump_m(first: Mapping[str, object], second: Mapping[str, object]) -> float:
        return math.hypot(
            float(first.get("x_ref_m", first.get("x", 0.0))) - float(second.get("x_ref_m", second.get("x", 0.0))),
            float(first.get("y_ref_m", first.get("y", 0.0))) - float(second.get("y_ref_m", second.get("y", 0.0))),
        )


class _Mode2TrajectoryMemory:
    """Continuity gate for MPC outputs in OpenCDA-reference tracking mode."""

    def __init__(
        self,
        *,
        max_accel_jump_mps2: float = 1.2,
        max_steer_jump_rad: float = 0.12,
        blend_alpha: float = 0.45,
        max_reuse_age_s: float = 0.5,
    ) -> None:
        self.max_accel_jump_mps2 = max(0.0, float(max_accel_jump_mps2))
        self.max_steer_jump_rad = max(0.0, float(max_steer_jump_rad))
        self.blend_alpha = min(1.0, max(0.0, float(blend_alpha)))
        self.max_reuse_age_s = max(0.0, float(max_reuse_age_s))
        self._last_control: carla.VehicleControl | None = None
        self._last_accel_mps2 = 0.0
        self._last_steer_rad = 0.0
        self._last_time_s = -float("inf")

    def accept_or_blend(
        self,
        *,
        control: carla.VehicleControl,
        accel_mps2: float,
        steer_rad: float,
        control_factory: Any,
        sim_time_s: float,
    ) -> tuple[carla.VehicleControl, float, float, str]:
        if self._last_control is None:
            self._accept(control, accel_mps2, steer_rad, sim_time_s)
            return control, float(accel_mps2), float(steer_rad), "trajectory_memory_accept"
        accel_jump = abs(float(accel_mps2) - float(self._last_accel_mps2))
        steer_jump = abs(float(steer_rad) - float(self._last_steer_rad))
        if accel_jump > float(self.max_accel_jump_mps2) or steer_jump > float(self.max_steer_jump_rad):
            alpha = float(self.blend_alpha)
            blended_accel = (1.0 - alpha) * float(self._last_accel_mps2) + alpha * float(accel_mps2)
            blended_steer = (1.0 - alpha) * float(self._last_steer_rad) + alpha * float(steer_rad)
            blended_control = control_factory(float(blended_accel), float(blended_steer))
            self._accept(blended_control, blended_accel, blended_steer, sim_time_s)
            return (
                blended_control,
                float(blended_accel),
                float(blended_steer),
                f"trajectory_memory_blend:accel={accel_jump:.2f}:steer={steer_jump:.2f}",
            )
        self._accept(control, accel_mps2, steer_rad, sim_time_s)
        return control, float(accel_mps2), float(steer_rad), "trajectory_memory_accept"

    def reuse_if_fresh(
        self,
        *,
        sim_time_s: float,
        stop_goal_active: bool,
        control_factory: Any,
    ) -> tuple[carla.VehicleControl | None, float, float, str]:
        if self._last_control is None:
            return None, 0.0, 0.0, ""
        if float(sim_time_s) - float(self._last_time_s) > float(self.max_reuse_age_s):
            return None, 0.0, 0.0, ""
        if bool(stop_goal_active):
            accel_mps2 = min(float(self._last_accel_mps2), -0.5)
            control = control_factory(float(accel_mps2), float(self._last_steer_rad))
            self._accept(control, accel_mps2, self._last_steer_rad, sim_time_s)
            return control, float(accel_mps2), float(self._last_steer_rad), "trajectory_memory_reuse_stop"
        return (
            self._last_control,
            float(self._last_accel_mps2),
            float(self._last_steer_rad),
            "trajectory_memory_reuse",
        )

    def _accept(
        self,
        control: carla.VehicleControl,
        accel_mps2: float,
        steer_rad: float,
        sim_time_s: float,
    ) -> None:
        self._last_control = control
        self._last_accel_mps2 = float(accel_mps2)
        self._last_steer_rad = float(steer_rad)
        self._last_time_s = float(sim_time_s)


class CPXMPCPlannerBridge:
    """Direct-control planner used inside ``VehicleManager.run_step``."""

    def __init__(
        self,
        vehicle_manager: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        map_planner: Any = None,
    ):
        self.vehicle_manager = vehicle_manager
        self.config = dict(config or {})
        self.carla = carla
        self.map_planner = map_planner or getattr(vehicle_manager, "carla_map", None)
        self.enabled = bool(self.config.get("enabled", True))
        self.mode = str(self.config.get("mode", "full_cpx_mpc")).strip().lower()
        self.fallback_policy = str(
            self.config.get("fallback_policy", "emergency_stop")
        ).strip().lower()
        self.fallback_policy_warning = ""
        if (
            self.mode not in {"opencda_reference_mpc", "opencda_ref_mpc", "mode2"}
            and self.fallback_policy == "opencda"
        ):
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
        self.max_mpc_obstacles = max(0, int(self.config.get("max_mpc_obstacles", 4)))
        self.debug = bool(self.config.get("debug", True))
        self.last_debug: dict[str, Any] = {}
        self._last_accel_mps2 = 0.0
        self._last_steer_rad = 0.0
        self._warned = False
        self._previous_lane_center_reference: list[dict[str, object]] = []
        self._temporary_destination_state: list[float] | None = None
        self._lane_reference_freeze_count = 0
        self._stop_release_temp_smooth_until_sim_time_s = 0.0
        self._mode2_stuck_stop_ticks = 0
        self._mode2_pid_hold_until_sim_time_s = -float("inf")
        self._mode2_consecutive_mpc_success = 0
        self._mode2_last_control_source = "pid"
        self._mode2_last_pid_control = None
        self._mode2_last_stop_goal_active = False
        self._mode2_release_until_sim_time_s = -float("inf")
        self._mode2_traffic_memory = _Mode2TrafficLightMemory(
            hold_unknown_s=float(self.config.get("mode2_traffic_unknown_hold_s", 0.8))
        )
        self._mode2_object_memory = _Mode2ObjectTrackMemory(
            alpha=float(self.config.get("mode2_object_memory_alpha", 0.55)),
            max_stale_s=float(self.config.get("mode2_object_memory_max_stale_s", 0.4)),
        )
        self._mode2_reference_memory = _Mode2ReferenceMemory(
            max_first_point_jump_m=float(self.config.get("mode2_reference_memory_max_first_jump_m", 2.0)),
            max_destination_jump_m=float(self.config.get("mode2_reference_memory_max_destination_jump_m", 4.0)),
            max_reuse_age_s=float(self.config.get("mode2_reference_memory_max_reuse_age_s", 1.0)),
        )
        self._mode2_trajectory_memory = _Mode2TrajectoryMemory(
            max_accel_jump_mps2=float(self.config.get("mode2_trajectory_memory_max_accel_jump_mps2", 1.2)),
            max_steer_jump_rad=float(self.config.get("mode2_trajectory_memory_max_steer_jump_rad", 0.12)),
            blend_alpha=float(self.config.get("mode2_trajectory_memory_blend_alpha", 0.45)),
            max_reuse_age_s=float(self.config.get("mode2_trajectory_memory_max_reuse_age_s", 0.5)),
        )
        self._full_reference_memory = _Mode2ReferenceMemory(
            max_first_point_jump_m=float(self.config.get("full_reference_memory_max_first_jump_m", 0.85)),
            max_destination_jump_m=float(self.config.get("full_reference_memory_max_destination_jump_m", 2.0)),
            max_reuse_age_s=float(self.config.get("full_reference_memory_max_reuse_age_s", 0.8)),
        )
        self._full_trajectory_memory = _Mode2TrajectoryMemory(
            max_accel_jump_mps2=float(self.config.get("full_trajectory_memory_max_accel_jump_mps2", 0.9)),
            max_steer_jump_rad=float(self.config.get("full_trajectory_memory_max_steer_jump_rad", 0.08)),
            blend_alpha=float(self.config.get("full_trajectory_memory_blend_alpha", 0.35)),
            max_reuse_age_s=float(self.config.get("full_trajectory_memory_max_reuse_age_s", 0.5)),
        )
        self.strict_lane_follow_reference = bool(
            self.config.get("strict_lane_follow_reference", False)
        )
        self.draw_world_debug = bool(self.config.get("draw_world_debug", False))
        self.world_debug_life_time_s = float(self.config.get("world_debug_life_time_s", 0.15))
        self.overspeed_guard_enabled = bool(self.config.get("overspeed_guard_enabled", False))
        self.overspeed_margin_mps = float(self.config.get("overspeed_margin_mps", 0.75))
        self.overspeed_brake_gain = float(self.config.get("overspeed_brake_gain", 0.10))
        self.overspeed_min_brake = float(self.config.get("overspeed_min_brake", 0.15))
        self.overspeed_max_brake = float(self.config.get("overspeed_max_brake", 0.55))
        self.low_speed_lateral_recovery_enabled = bool(
            self.config.get("low_speed_lateral_recovery_enabled", False)
        )
        self.low_speed_lateral_recovery_speed_mps = float(
            self.config.get("low_speed_lateral_recovery_speed_mps", 0.6)
        )
        self.low_speed_lateral_recovery_threshold_m = float(
            self.config.get("low_speed_lateral_recovery_threshold_m", 1.5)
        )
        self.low_speed_lateral_recovery_target_speed_mps = float(
            self.config.get("low_speed_lateral_recovery_target_speed_mps", 1.2)
        )
        self.low_speed_lateral_recovery_max_steer_rad = float(
            self.config.get("low_speed_lateral_recovery_max_steer_rad", 0.14)
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
            "target_speed_mps",
            "behavior_decision",
            "behavior_fsm_state",
            "current_lane_id",
            "behavior_target_lane_id",
            "stop_goal_active",
            "front_gap_m",
            "object_count",
            "mpc_object_count",
            "cp_provider_source",
            "native_opencda_available",
            "cp_obstacle_count",
            "cp_control_count",
            "v2x_nearby_count",
            "reference_source",
            "reference_pipeline_stage",
            "reference_pipeline_intent",
            "reference_pipeline_fallback",
            "destination_x",
            "destination_y",
            "destination_forward_m",
            "destination_lateral_m",
            "destination_lane_id",
            "reference_first_forward_m",
            "reference_first_lateral_m",
            "mpc_trajectory_point_count",
            "global_route_point_count",
            "route_reference_allowed",
            "route_reference_gate_reason",
            "route_lane_change_allowed",
            "opportunistic_lane_change_allowed",
            "lane_change_gate_reason",
            "route_lane_change_required",
            "behavior_override_reason",
            "reference_follow_global_route_lane",
            "route_current_road_option",
            "route_next_macro_maneuver",
            "mpc_status",
            "mpc_solve_time_ms",
            "mpc_cost_profile",
            "requested_mpc_cost_profile",
            "mpc_cost_profile_switch_reason",
            "mpc_fallback_reason",
            "control_guard_reason",
            "safety_supervisor_reason",
            "accel_cmd_mps2",
            "steer_cmd_rad",
            "planner_input_cp_traffic_control_count",
            "planner_input_prediction_risky_lane_count",
            "planner_input_perception_planning_count",
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
            "traffic_control_from_cp",
            "candidate_evaluation_summary",
            "candidate_selected_decision",
            "candidate_selected_lane_id",
            "candidate_selected_cost",
            "mpc_feedback_summary",
            "mpc_feedback_record_reason",
            "mpc_feedback_blocked_lane_ids",
            "control_buffer_reason",
            "control_buffered_step_count",
            "mpc_replan_executed",
            "route_manager_status",
            "route_remaining_distance_m",
            "route_reached_destination",
            "global_planner_backend",
            "global_planner_backend_warning",
            "tracker_active_count",
            "tracker_stale_count",
            "prediction_validity_reason",
            "object_memory_reason",
            "traffic_memory_reason",
            "reference_memory_reason",
            "trajectory_memory_reason",
            "stop_target_forward_m",
            "stop_approach_speed_mps",
            "green_release_reference_active",
            "lane_safety_scores",
        ]

        self._ensure_planning_module_import_path()
        from opencda.planning_module.MPC.mpc import MPC
        from opencda.planning_module.behavior_planner import LaneSafetyScorer, RuleBasedBehaviorPlanner
        from opencda.planning_module.opencda_bridge.cp_provider import OpenCDACPProvider
        from opencda.planning_module.opencda_bridge.planner_input_adapter import OpenCDAPlanningAdapter
        from opencda.planning_module.pipeline.control_buffer import MPCControlBuffer
        from opencda.planning_module.pipeline.mpc_feedback import BehaviorMPCFeedback
        from opencda.planning_module.pipeline.planner_pipeline import CPXPlanningPipeline
        from opencda.planning_module.pipeline.route_manager import CPXRouteManager
        from opencda.planning_module.pipeline.safety_supervisor import SafetySupervisor
        from opencda.planning_module.pipeline.tracker import CPXObstacleTracker
        from opencda.planning_module.utility.carla_lane_graph import build_lane_center_waypoints
        from opencda.planning_module.utility.global_planner import CustomGlobalPlannerAdapter
        from opencda.planning_module.utility.legacy_global_planner import AStarGlobalPlanner

        mpc_cfg, road_cfg = self._load_mpc_config()
        self.mpc = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)
        self.behavior_runtime_cfg = dict(mpc_cfg.get("behavior_planner_runtime", {}))
        self.lane_safety_scorer = LaneSafetyScorer()
        self.reference_map = _CarlaMapPlannerAdapter(self.map_planner)
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
        self.safety_supervisor = SafetySupervisor(
            enabled=bool(self.config.get("safety_supervisor_enabled", True)),
            max_steer_delta=float(self.config.get("safety_max_steer_delta", 0.25)),
            max_throttle_delta=float(self.config.get("safety_max_throttle_delta", 0.45)),
            max_brake_delta=float(self.config.get("safety_max_brake_delta", 0.60)),
        )
        route_sample_distance_m = float(self.config.get("route_sample_distance_m", 2.0))
        self.global_planner_backend = "legacy_astar"
        self.global_planner_backend_warning = ""
        road_cfg_from_map = {}
        global_planner_mode = str(
            self.config.get("global_planner_mode", "legacy")
        ).strip().lower()
        if global_planner_mode in {"custom", "custom_admap", "admap", "opendrive", "dijkstra", "dij"}:
            try:
                xodr_path = self._resolve_global_planner_xodr_path()
                self.global_planner = CustomGlobalPlannerAdapter(
                    xodr_path=str(xodr_path),
                    cache_root=str(
                        self.config.get(
                            "global_planner_cache_root",
                            Path(__file__).resolve().parents[1]
                            / "Global_Planner"
                            / "cache",
                        )
                    ),
                    route_sample_distance_m=float(route_sample_distance_m),
                    ad_map_install_root=self.config.get("ad_map_install_root"),
                )
                self.global_planner.load(
                    force_rebuild=bool(self.config.get("global_planner_force_rebuild", False))
                )
                self.global_planner_backend = "custom_admap_dijkstra"
                road_cfg_from_map = {"lane_count": 1, "lane_width_m": 3.5}
                print(
                    "[CP-X OpenCDA Bridge] Using custom AD-map Dijkstra global planner: "
                    f"{xodr_path}"
                )
            except Exception as exc:
                if bool(self.config.get("global_planner_strict", False)):
                    raise
                self.global_planner_backend_warning = (
                    f"custom_global_planner_failed_fallback_legacy:{exc}"
                )
                print(
                    "[CP-X OpenCDA Bridge] Custom AD-map Dijkstra global planner failed; "
                    f"falling back to legacy route planner: {exc}"
                )
                lane_center_waypoints, road_cfg_from_map = build_lane_center_waypoints(
                    map_obj=self.map_planner,
                    carla=carla,
                    sample_distance_m=float(route_sample_distance_m),
                )
                self.global_planner = AStarGlobalPlanner(
                    lane_center_waypoints=lane_center_waypoints,
                    world_map=self.map_planner,
                    route_sample_distance_m=float(route_sample_distance_m),
                )
        else:
            lane_center_waypoints, road_cfg_from_map = build_lane_center_waypoints(
                map_obj=self.map_planner,
                carla=carla,
                sample_distance_m=float(route_sample_distance_m),
            )
            self.global_planner = AStarGlobalPlanner(
                lane_center_waypoints=lane_center_waypoints,
                world_map=self.map_planner,
                route_sample_distance_m=float(route_sample_distance_m),
            )
        self.route_manager = CPXRouteManager(
            global_planner=self.global_planner,
            reached_distance_m=float(self.config.get("route_reached_distance_m", 3.0)),
            stale_route_lateral_m=float(self.config.get("route_stale_lateral_m", 12.0)),
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
            )
        self.active_mpc_cost_profile = "lane_follow"
        self.requested_mpc_cost_profile = "lane_follow"
        self.mpc_cost_profile_active_since_s = 0.0
        self.mpc_cost_profile_switch_reason = "initial"
        self._latest_opencda_update: dict[str, Any] = {}
        self.last_output = None

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
        self._temporary_destination_state = None
        self._previous_lane_center_reference = []
        self._lane_reference_freeze_count = 0

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

        if self.mode in {"opencda_reference_mpc", "opencda_ref_mpc", "mode2"}:
            return self._run_opencda_reference_mpc()

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

    def _run_full_cpx_pipeline_step(self):
        """Run OpenCDAPlanningAdapter -> PlanningPipeline -> PlannerOutput."""

        from opencda.planning_module.pipeline.output import (
            BehaviorCommand,
            PlannerDiagnostics,
            PlannerOutput,
        )

        latest_update = dict(getattr(self, "_latest_opencda_update", {}) or {})
        sim_time_s = float(self._sim_time_s())
        ego_transform = latest_update.get("ego_transform") or self.vehicle_manager.localizer.get_ego_pos()
        ego_speed_kmh = float(
            latest_update.get("ego_speed_kmh", self.vehicle_manager.localizer.get_ego_spd())
        )
        ego_speed_mps = ego_speed_kmh / 3.6
        ego_location = ego_transform.location
        ego_yaw_rad = math.radians(float(ego_transform.rotation.yaw))

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
        mpc_object_snapshots = self._limit_obstacles_for_mpc(
            object_snapshots=object_snapshots,
            ego_location=ego_location,
        )
        front_gap_m = self._front_gap_m(
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            object_snapshots=object_snapshots,
        )
        stop_goal_active = front_gap_m is not None and front_gap_m < self.min_front_gap_m
        speed_ref_mps = 0.0 if stop_goal_active else self.target_speed_mps

        behavior_debug: dict[str, Any] = {}
        reference_debug: dict[str, Any] = {}
        try:
            destination_state, lane_center_reference, behavior_debug, reference_debug = (
                self._plan_behavior_and_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=ego_yaw_rad,
                    ego_speed_mps=ego_speed_mps,
                    speed_ref_mps=speed_ref_mps,
                    object_snapshots=object_snapshots,
                    stop_goal_active=stop_goal_active,
                    cp_payload=cp_payload,
                )
            )
        except Exception as exc:
            if self.debug:
                print(f"[CP-X OpenCDA Bridge] behavior/reference pipeline failed: {exc}")
            destination_state, lane_center_reference = self._build_route_reference(
                ego_location=ego_location,
                ego_yaw_rad=ego_yaw_rad,
                speed_ref_mps=speed_ref_mps,
            )
            behavior_debug = {
                "decision": "lane_follow",
                "lc_state": "FALLBACK",
                "target_lane_id": "",
                "pipeline_error": str(exc),
            }
            reference_debug = {"reference_source": "simple_lane_center_fallback"}

        mpc_stop_goal_active = bool(stop_goal_active) or str(
            behavior_debug.get("decision", "")
        ) in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        if bool(mpc_stop_goal_active) and len(destination_state) >= 3:
            destination_state = list(destination_state)
            destination_state[2] = 0.0

        current_state = [
            float(ego_location.x),
            float(ego_location.y),
            float(ego_speed_mps),
            float(ego_yaw_rad),
        ]
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
        trajectory_memory_reason = ""
        try:
            force_replan = bool(mpc_stop_goal_active) or str(
                behavior_debug.get("decision", "")
            ) in {"stop_at_intersection", "stop_sign", "emergency_brake"}
            mpc_replan_executed = bool(
                self.control_buffer.should_replan(
                    sim_time_s=float(sim_time_s),
                    force_replan=bool(force_replan),
                )
            )
            if bool(mpc_replan_executed):
                self.mpc.plan_trajectory(
                    current_state=current_state,
                    destination_state=destination_state,
                    object_snapshots=mpc_object_snapshots,
                    current_acceleration_mps2=float(self._last_accel_mps2),
                    current_steering_rad=float(self._last_steer_rad),
                    lane_center_reference_samples=lane_center_reference,
                    stop_goal_active=bool(mpc_stop_goal_active),
                )
                mpc_status = str(getattr(self.mpc, "_last_status", "")).strip().lower()
                if mpc_status and mpc_status not in {"solved", "solved inaccurate"}:
                    raise RuntimeError(f"MPC status={mpc_status}")
                u_solution = getattr(self.mpc, "_last_u_solution", None)
                if u_solution is None or len(u_solution) == 0:
                    raise RuntimeError("MPC did not expose a control solution")
                self.control_buffer.update_from_solution(
                    u_solution=u_solution,
                    plan_time_s=float(sim_time_s),
                    dt_s=float(self.mpc.dt_s),
                )
                accel_mps2 = float(u_solution[0, 0])
                steer_rad = float(u_solution[0, 1])
            else:
                buffered = self.control_buffer.sample(sim_time_s=float(sim_time_s))
                if buffered is None:
                    raise RuntimeError("MPC control buffer empty")
                accel_mps2, steer_rad, _buffer_reason = buffered
                mpc_status = "buffer_reuse"
            control = self._control_from_mpc(accel_mps2, steer_rad)
            if bool(self.config.get("full_trajectory_memory_enabled", True)):
                (
                    control,
                    accel_mps2,
                    steer_rad,
                    trajectory_memory_reason,
                ) = self._full_trajectory_memory.accept_or_blend(
                    control=control,
                    accel_mps2=float(accel_mps2),
                    steer_rad=float(steer_rad),
                    control_factory=self._control_from_mpc,
                    sim_time_s=float(sim_time_s),
                )
            fallback_reason = ""
        except Exception as exc:
            mpc_replan_executed = True
            if bool(self.config.get("full_trajectory_memory_enabled", True)):
                memory_control, memory_accel, memory_steer, memory_reason = (
                    self._full_trajectory_memory.reuse_if_fresh(
                        sim_time_s=float(sim_time_s),
                        stop_goal_active=bool(mpc_stop_goal_active),
                        control_factory=self._control_from_mpc,
                    )
                )
                if memory_control is not None:
                    control = memory_control
                    accel_mps2 = float(memory_accel)
                    steer_rad = float(memory_steer)
                    fallback_reason = f"fallback_to_full_trajectory_memory:{exc}"
                    trajectory_memory_reason = str(memory_reason)
                else:
                    control = None
            else:
                control = None
            if control is None:
                fallback_reason = str(exc)
                trajectory_memory_reason = ""
                control = self._fallback_control(
                    ego_transform=ego_transform,
                    ego_speed_mps=ego_speed_mps,
                    destination_state=destination_state,
                    stop_goal_active=mpc_stop_goal_active,
                )
                accel_mps2 = self._last_accel_mps2
                steer_rad = self._last_steer_rad
            mpc_status = str(getattr(self.mpc, "_last_status", str(exc)))
            if not self._warned:
                print(f"[CP-X OpenCDA Bridge] MPC fallback active: {fallback_reason}")
                self._warned = True
        mpc_feedback_record_reason = self.mpc_feedback.record_result(
            decision=str(behavior_debug.get("decision", "")),
            target_lane_id=int(behavior_debug.get("target_lane_id", 0) or 0),
            status=str(mpc_status),
            reason=str(fallback_reason),
            timestamp_s=float(sim_time_s),
            success=not bool(fallback_reason),
        )

        control_guard_reason = ""
        control, accel_mps2, steer_rad, control_guard_reason = self._apply_control_safety_guards(
            control=control,
            accel_mps2=float(accel_mps2),
            steer_rad=float(steer_rad),
            ego_transform=ego_transform,
            ego_speed_mps=float(ego_speed_mps),
            speed_ref_mps=float(speed_ref_mps),
            destination_state=destination_state,
            destination_lateral_m=float(destination_lateral_m),
            stop_goal_active=bool(mpc_stop_goal_active),
            behavior_decision=str(behavior_debug.get("decision", "")),
            behavior_fsm_state=str(behavior_debug.get("lc_state", "")),
        )
        control, safety_supervisor_reason = self.safety_supervisor.filter_control(
            control=control,
            carla_module=self.carla,
            safety_manager=latest_update.get("safety_manager"),
        )
        self._last_accel_mps2 = float(accel_mps2)
        self._last_steer_rad = float(steer_rad)
        cp_summary = dict(getattr(self.cp_provider, "last_publish_summary", {}) or {})
        diagnostics = {
            "sim_time_s": float(self._sim_time_s()),
            "vehicle_id": int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            "x_m": float(ego_location.x),
            "y_m": float(ego_location.y),
            "yaw_deg": float(ego_transform.rotation.yaw),
            "speed_mps": float(ego_speed_mps),
            "planner": "cpx_mpc",
            "object_count": len(object_snapshots),
            "mpc_object_count": len(mpc_object_snapshots),
            "local_object_count": len(local_object_snapshots),
            "v2x_nearby_count": len(getattr(self.vehicle_manager.v2x_manager, "cav_nearby", {}) or {}),
            "cp_provider_summary": dict(cp_summary),
            "cp_provider_source": str(cp_summary.get("provider_source", "")),
            "native_opencda_required": bool(cp_summary.get("native_opencda_required", False)),
            "native_opencda_available": bool(cp_summary.get("native_opencda_available", False)),
            "cp_obstacle_count": int(cp_summary.get("obstacle_count", 0) or 0),
            "cp_control_count": int(cp_summary.get("control_count", 0) or 0),
            "front_gap_m": "" if front_gap_m is None else float(front_gap_m),
            "stop_goal_active": bool(mpc_stop_goal_active),
            "behavior_decision": str(behavior_debug.get("decision", "")),
            "behavior_fsm_state": str(behavior_debug.get("lc_state", "")),
            "current_lane_id": behavior_debug.get("current_lane_id", ""),
            "behavior_target_lane_id": behavior_debug.get("target_lane_id", ""),
            "traffic_signal_state": behavior_debug.get("traffic_signal_state", ""),
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
            "mpc_trajectory_point_count": len(self._last_mpc_trajectory_points()),
            "global_route_point_count": len(self._active_global_route_points()),
            "route_reference_allowed": reference_debug.get("route_reference_allowed", ""),
            "route_reference_gate_reason": reference_debug.get("route_reference_gate_reason", ""),
            "route_lane_change_allowed": reference_debug.get("route_lane_change_allowed", ""),
            "opportunistic_lane_change_allowed": reference_debug.get("opportunistic_lane_change_allowed", ""),
            "lane_change_gate_reason": reference_debug.get("lane_change_gate_reason", ""),
            "route_lane_change_required": reference_debug.get("route_lane_change_required", ""),
            "behavior_override_reason": reference_debug.get("behavior_override_reason", ""),
            "reference_follow_global_route_lane": reference_debug.get("reference_pipeline_follow_global_route_lane", ""),
            "route_current_road_option": reference_debug.get("route_current_road_option", ""),
            "route_next_macro_maneuver": reference_debug.get("route_next_macro_maneuver", ""),
            "candidate_evaluation_summary": reference_debug.get("candidate_evaluation_summary", ""),
            "candidate_selected_decision": reference_debug.get("candidate_selected_decision", ""),
            "candidate_selected_lane_id": reference_debug.get("candidate_selected_lane_id", ""),
            "candidate_selected_cost": reference_debug.get("candidate_selected_cost", ""),
            "mpc_feedback_summary": reference_debug.get("mpc_feedback_summary", ""),
            "mpc_feedback_record_reason": str(mpc_feedback_record_reason),
            "mpc_feedback_blocked_lane_ids": reference_debug.get("mpc_feedback_blocked_lane_ids", ""),
            "control_buffer_reason": str(self.control_buffer.last_reason),
            "control_buffered_step_count": int(self.control_buffer.buffered_step_count),
            "mpc_replan_executed": bool(mpc_replan_executed),
            "route_manager_status": json.dumps(
                self.route_manager.last_status.as_dict(),
                default=str,
            ),
            "route_remaining_distance_m": float(
                self.route_manager.last_status.remaining_distance_m
            ),
            "route_reached_destination": bool(self.route_manager.last_status.reached_destination),
            "global_planner_backend": str(self.global_planner_backend),
            "global_planner_backend_warning": str(self.global_planner_backend_warning),
            "tracker_active_count": reference_debug.get("tracker_active_count", ""),
            "tracker_stale_count": reference_debug.get("tracker_stale_count", ""),
            "prediction_validity_reason": reference_debug.get("prediction_validity_reason", ""),
            "reference_memory_reason": str(reference_debug.get("reference_memory_reason", "")),
            "trajectory_memory_reason": str(trajectory_memory_reason),
            "mpc_trajectory_points": self._last_mpc_trajectory_points(),
            "global_route_points": self._active_global_route_points(),
            "lane_reference_points": [
                [
                    float(sample.get("x_ref_m", sample.get("x", 0.0))),
                    float(sample.get("y_ref_m", sample.get("y", 0.0))),
                ]
                for sample in list(lane_center_reference or [])
            ],
            "target_speed_mps": float(speed_ref_mps),
            "mpc_status": str(mpc_status),
            "mpc_solve_time_ms": float(getattr(self.mpc, "_last_solve_time_ms", 0.0)),
            "mpc_cost_profile": str(self.active_mpc_cost_profile),
            "requested_mpc_cost_profile": str(self.requested_mpc_cost_profile),
            "mpc_cost_profile_switch_reason": str(self.mpc_cost_profile_switch_reason),
            "reference_source": str(reference_debug.get(
                "reference_source",
                "map_lane_center" if lane_center_reference else "straight_fallback",
            )),
            "fallback_reason": fallback_reason,
            "mpc_fallback_reason": fallback_reason,
            "control_guard_reason": str(control_guard_reason),
            "accel_cmd_mps2": float(accel_mps2),
            "steer_cmd_rad": float(steer_rad),
            "planner_requested": True,
            "planner_executed": True,
            "fallback_active": bool(fallback_reason),
            "fallback_policy": str(self.fallback_policy),
            "fallback_policy_warning": str(self.fallback_policy_warning),
            "safety_supervisor_reason": str(safety_supervisor_reason),
        }
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
            acceleration_mps2=float(accel_mps2),
            steering_rad=float(steer_rad),
            diagnostics=PlannerDiagnostics(diagnostics),
        )

    def _run_opencda_reference_mpc(self) -> carla.VehicleControl:
        """Use OpenCDA's BehaviorAgent/LocalPlanner reference, then track it with MPC."""

        ego_transform = self.vehicle_manager.localizer.get_ego_pos()
        ego_speed_kmh = float(self.vehicle_manager.localizer.get_ego_spd())
        ego_speed_mps = ego_speed_kmh / 3.6
        ego_location = ego_transform.location
        ego_yaw_rad = math.radians(float(ego_transform.rotation.yaw))
        sim_time_s = float(self._sim_time_s())

        local_object_snapshots = self._collect_object_snapshots()
        if self.cp_provider is not None:
            try:
                self.cp_provider.publish(
                    world=self.vehicle_manager.vehicle.get_world(),
                    map_planner=self.map_planner,
                    ego_vehicle=self.vehicle_manager.vehicle,
                    sim_time_s=sim_time_s,
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
            sim_time_s=sim_time_s,
        )
        object_snapshots, object_memory_reason = self._mode2_object_memory.update(
            object_snapshots=object_snapshots,
            sim_time_s=float(sim_time_s),
        )
        mpc_object_snapshots = self._limit_obstacles_for_mpc(
            object_snapshots=object_snapshots,
            ego_location=ego_location,
        )

        target_speed_kmh, target_loc, opencda_error = self._opencda_behavior_target()
        target_speed_kmh = float(target_speed_kmh or 0.0)
        speed_ref_mps = max(0.0, target_speed_kmh / 3.6)
        if speed_ref_mps > 1.0e-6 and not bool(
            self.config.get("mode2_use_opencda_trajectory_speed", False)
        ):
            min_tracking_speed_mps = float(self.config.get("mode2_min_tracking_speed_mps", 1.0))
            speed_ref_mps = max(float(min_tracking_speed_mps), float(speed_ref_mps))
        opencda_stop_active = target_loc is None or target_speed_kmh <= 1.0e-6

        selected_control = self._select_mode2_relevant_traffic_control(
            cp_payload=cp_payload,
            ego_location=ego_location,
            ego_heading_rad=ego_yaw_rad,
            sim_time_s=sim_time_s,
        )
        signal_context, stop_target = self._traffic_context_from_cp_control(
            selected_control=selected_control,
            ego_location=ego_location,
        )
        raw_traffic_state = str(signal_context.get("signal_state", "unknown")).strip().lower()
        traffic_state, stop_target, traffic_memory_reason = self._mode2_traffic_memory.update(
            state=str(raw_traffic_state),
            stop_target=stop_target,
            sim_time_s=float(sim_time_s),
        )
        signal_context = dict(signal_context)
        signal_context["raw_signal_state"] = str(raw_traffic_state)
        signal_context["signal_state"] = str(traffic_state)
        if str(traffic_memory_reason).startswith("traffic_memory_hold"):
            signal_context["from_cp"] = True
            signal_context["traffic_control_from_cp"] = True
        cp_stop_active = traffic_state in {"red", "yellow"}
        if bool(cp_stop_active):
            speed_ref_mps = 0.0
            target_speed_kmh = 0.0
        elif speed_ref_mps > 1.0e-6:
            target_speed_kmh = float(speed_ref_mps) * 3.6
        spurious_opencda_stop = bool(
            opencda_stop_active
            and not cp_stop_active
            and traffic_state not in {"red", "yellow"}
            and not self._has_close_forward_obstacle(
                object_snapshots=object_snapshots,
                ego_location=ego_location,
                ego_yaw_rad=ego_yaw_rad,
                max_forward_m=5.0,
                max_lateral_m=2.0,
            )
        )

        lane_center_reference = self._opencda_local_planner_reference_samples(
            target_speed_mps=float(speed_ref_mps),
            ego_transform=ego_transform,
        )
        if not lane_center_reference and target_loc is not None:
            lane_center_reference = self._reference_samples_from_target_location(
                ego_location=ego_location,
                ego_yaw_rad=ego_yaw_rad,
                target_loc=target_loc,
                target_speed_mps=float(speed_ref_mps),
            )
        if bool(spurious_opencda_stop):
            self._mode2_stuck_stop_ticks += 1
            recovery_speed_mps = float(
                self.config.get(
                    "opencda_reference_recovery_speed_mps",
                    min(max(float(self.target_speed_mps), 1.0), 2.0),
                )
            )
            recovered_target = self._next_opencda_forward_target(
                ego_transform=ego_transform,
            )
            if recovered_target is not None:
                target_loc = recovered_target
                target_speed_kmh = float(recovery_speed_mps) * 3.6
                speed_ref_mps = float(recovery_speed_mps)
                opencda_stop_active = False
                lane_center_reference = self._reference_samples_from_target_location(
                    ego_location=ego_location,
                    ego_yaw_rad=ego_yaw_rad,
                    target_loc=recovered_target,
                    target_speed_mps=float(recovery_speed_mps),
                )
                opencda_error = (
                    f"{opencda_error}:spurious_opencda_stop_recovery"
                    if opencda_error
                    else "spurious_opencda_stop_recovery"
                )
        else:
            self._mode2_stuck_stop_ticks = 0

        stop_goal_active = bool(opencda_stop_active or cp_stop_active)
        stop_target_forward_m = None
        if bool(stop_goal_active) and isinstance(stop_target, Mapping):
            x_value = stop_target.get("x_m", stop_target.get("x", None))
            y_value = stop_target.get("y_m", stop_target.get("y", None))
            if x_value is not None and y_value is not None:
                stop_target_forward_m, _ = self._body_frame_xy(
                    origin_x_m=float(ego_location.x),
                    origin_y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    target_x_m=float(x_value),
                    target_y_m=float(y_value),
                )
        stop_approach_speed_mps = 0.0
        if bool(stop_goal_active) and stop_target_forward_m is not None:
            stop_approach_distance_m = float(self.config.get("mode2_stop_approach_distance_m", 8.0))
            if float(stop_target_forward_m) > stop_approach_distance_m:
                stop_approach_speed_mps = min(
                    float(self.config.get("mode2_stop_approach_speed_mps", 2.0)),
                    max(float(self.config.get("mode2_stop_approach_min_speed_mps", 0.8)), float(self.target_speed_mps)),
                )
                speed_ref_mps = float(stop_approach_speed_mps)
                target_speed_kmh = float(stop_approach_speed_mps) * 3.6

        release_reference_active = False
        if bool(self._mode2_last_stop_goal_active) and not bool(stop_goal_active):
            self._mode2_release_until_sim_time_s = float(sim_time_s) + float(
                self.config.get("mode2_green_release_reference_s", 1.2)
            )
            self._mode2_reference_memory.reset()
        if not bool(stop_goal_active) and float(sim_time_s) <= float(self._mode2_release_until_sim_time_s):
            release_reference_active = True
            release_speed_mps = max(
                float(self.config.get("mode2_green_release_speed_mps", 1.8)),
                float(speed_ref_mps),
            )
            release_target = self._next_opencda_forward_target(ego_transform=ego_transform)
            if release_target is None and target_loc is not None:
                release_target = target_loc
            if release_target is not None:
                lane_center_reference = self._reference_samples_from_target_location(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    target_loc=release_target,
                    target_speed_mps=float(release_speed_mps),
                )
                target_loc = release_target
                speed_ref_mps = float(release_speed_mps)
                target_speed_kmh = float(release_speed_mps) * 3.6
                opencda_error = (
                    f"{opencda_error}:green_release_reference"
                    if opencda_error
                    else "green_release_reference"
                )
        if bool(stop_goal_active):
            stop_reference = self._mode2_stop_reference_samples(
                ego_location=ego_location,
                ego_yaw_rad=ego_yaw_rad,
                lane_center_reference=lane_center_reference,
                target_loc=target_loc,
                stop_target=stop_target if cp_stop_active else None,
                approach_speed_mps=float(stop_approach_speed_mps),
            )
            if stop_reference:
                lane_center_reference = stop_reference
        destination_state = self._destination_from_opencda_reference(
            lane_center_reference=lane_center_reference,
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            target_loc=target_loc,
            speed_ref_mps=0.0 if stop_goal_active else float(speed_ref_mps),
            stop_target=None,
        )
        reference_memory_reason = ""
        if lane_center_reference and destination_state is not None:
            lane_center_reference, destination_state, reference_memory_reason = (
                self._mode2_reference_memory.stabilize(
                    reference=lane_center_reference,
                    destination_state=destination_state,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    stop_goal_active=bool(stop_goal_active or release_reference_active),
                    sim_time_s=float(sim_time_s),
                )
            )

        fallback_reason = ""
        control_guard_reason = ""
        trajectory_memory_reason = ""
        mpc_status = ""
        mpc_solve_time_ms = 0.0
        control_source = "mpc"
        if not lane_center_reference or destination_state is None:
            fallback_reason = "missing_opencda_reference"
            control = self._opencda_pid_fallback_control(target_speed_kmh, target_loc)
            accel_mps2 = self._last_accel_mps2
            steer_rad = self._last_steer_rad
            control_source = "pid"
            self._mode2_note_pid_fallback(sim_time_s=sim_time_s)
        else:
            current_state = [
                float(ego_location.x),
                float(ego_location.y),
                float(ego_speed_mps),
                float(ego_yaw_rad),
            ]
            try:
                if hasattr(self.mpc, "apply_mode_cost_profile"):
                    profile_name = "tracking_stop" if bool(stop_goal_active) else "tracking"
                    self.mpc.apply_mode_cost_profile(profile_name, blend_alpha=1.0)
                early_pid_stop = bool(stop_goal_active) and self._mode2_should_use_pid_for_stop(
                    ego_speed_mps=float(ego_speed_mps),
                    ego_location=ego_location,
                    ego_yaw_rad=ego_yaw_rad,
                    destination_state=destination_state,
                )
                if bool(early_pid_stop):
                    raise RuntimeError("mode2_early_pid_stop")
                self.mpc.plan_trajectory(
                    current_state=current_state,
                    destination_state=destination_state,
                    object_snapshots=mpc_object_snapshots,
                    current_acceleration_mps2=float(self._last_accel_mps2),
                    current_steering_rad=float(self._last_steer_rad),
                    lane_center_reference_samples=lane_center_reference,
                    stop_goal_active=bool(stop_goal_active),
                )
                mpc_status = str(getattr(self.mpc, "_last_status", ""))
                normalized_status = mpc_status.strip().lower()
                if normalized_status and normalized_status not in {"solved", "solved inaccurate"}:
                    raise RuntimeError(f"MPC status={mpc_status}")
                u_solution = getattr(self.mpc, "_last_u_solution", None)
                if u_solution is None or len(u_solution) == 0:
                    raise RuntimeError("MPC did not expose a control solution")
                accel_mps2 = float(u_solution[0, 0])
                steer_rad = float(u_solution[0, 1])
                proposed_control = self._control_from_mpc(accel_mps2, steer_rad)
                (
                    proposed_control,
                    accel_mps2,
                    steer_rad,
                    trajectory_memory_reason,
                ) = self._mode2_trajectory_memory.accept_or_blend(
                    control=proposed_control,
                    accel_mps2=float(accel_mps2),
                    steer_rad=float(steer_rad),
                    control_factory=self._control_from_mpc,
                    sim_time_s=float(sim_time_s),
                )
                (
                    control,
                    accel_mps2,
                    steer_rad,
                    control_source,
                    arbitration_reason,
                ) = self._mode2_arbitrate_control(
                    proposed_control=proposed_control,
                    proposed_accel_mps2=float(accel_mps2),
                    proposed_steer_rad=float(steer_rad),
                    target_speed_kmh=float(target_speed_kmh),
                    target_loc=target_loc,
                    sim_time_s=sim_time_s,
                    stop_goal_active=bool(stop_goal_active),
                    destination_state=destination_state,
                    ego_location=ego_location,
                    ego_yaw_rad=ego_yaw_rad,
                )
                control_guard_reason = str(arbitration_reason)
            except Exception as exc:
                memory_control, memory_accel, memory_steer, memory_reason = (
                    self._mode2_trajectory_memory.reuse_if_fresh(
                        sim_time_s=float(sim_time_s),
                        stop_goal_active=bool(stop_goal_active),
                        control_factory=self._control_from_mpc,
                    )
                )
                if memory_control is not None:
                    fallback_reason = f"fallback_to_trajectory_memory:{exc}"
                    control = memory_control
                    accel_mps2 = float(memory_accel)
                    steer_rad = float(memory_steer)
                    control_source = "memory"
                    trajectory_memory_reason = str(memory_reason)
                else:
                    fallback_reason = f"fallback_to_opencda_pid:{exc}"
                    control = self._opencda_pid_fallback_control(target_speed_kmh, target_loc)
                    accel_mps2 = self._last_accel_mps2
                    steer_rad = self._last_steer_rad
                    control_source = "pid"
                    self._mode2_note_pid_fallback(sim_time_s=sim_time_s)

        mpc_solve_time_ms = float(getattr(self.mpc, "_last_solve_time_ms", 0.0))
        if fallback_reason == "":
            destination_forward_m, destination_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(destination_state[0]),
                target_y_m=float(destination_state[1]),
            )
        else:
            destination_forward_m, destination_lateral_m = ("", "")

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

        self._last_accel_mps2 = float(accel_mps2)
        self._last_steer_rad = float(steer_rad)
        cp_summary = dict(getattr(self.cp_provider, "last_publish_summary", {}) or {})
        self.last_debug = {
            "sim_time_s": sim_time_s,
            "vehicle_id": int(getattr(self.vehicle_manager.vehicle, "id", -1)),
            "x_m": float(ego_location.x),
            "y_m": float(ego_location.y),
            "yaw_deg": float(ego_transform.rotation.yaw),
            "speed_mps": float(ego_speed_mps),
            "target_speed_mps": float(speed_ref_mps),
            "behavior_decision": "opencda_stop" if stop_goal_active else "opencda_follow",
            "behavior_fsm_state": "OPENCDA_REFERENCE_MPC",
            "stop_goal_active": bool(stop_goal_active),
            "front_gap_m": "",
            "object_count": len(object_snapshots),
            "mpc_object_count": len(mpc_object_snapshots),
            "local_object_count": len(local_object_snapshots),
            "cp_provider_source": str(cp_summary.get("provider_source", "")),
            "native_opencda_available": bool(cp_summary.get("native_opencda_available", False)),
            "cp_obstacle_count": int(cp_summary.get("obstacle_count", 0) or 0),
            "cp_control_count": int(cp_summary.get("control_count", 0) or 0),
            "v2x_nearby_count": len(getattr(self.vehicle_manager.v2x_manager, "cav_nearby", {}) or {}),
            "reference_source": "opencda_local_planner",
            "reference_pipeline_stage": "opencda_behavior_agent>opencda_local_planner>mpc",
            "reference_pipeline_intent": "mode2_opencda_reference_mpc",
            "reference_pipeline_fallback": str(opencda_error),
            "destination_x": "" if destination_state is None else float(destination_state[0]),
            "destination_y": "" if destination_state is None else float(destination_state[1]),
            "destination_forward_m": destination_forward_m,
            "destination_lateral_m": destination_lateral_m,
            "destination_lane_id": "" if destination_state is None or len(destination_state) < 5 else int(destination_state[4]),
            "reference_first_forward_m": reference_first_forward_m,
            "reference_first_lateral_m": reference_first_lateral_m,
            "mpc_trajectory_point_count": len(self._last_mpc_trajectory_points()),
            "global_route_point_count": len(self._active_global_route_points()),
            "mpc_status": str(mpc_status or getattr(self.mpc, "_last_status", "")),
            "mpc_solve_time_ms": float(mpc_solve_time_ms),
            "mpc_cost_profile": "tracking_stop" if bool(stop_goal_active) else "tracking",
            "requested_mpc_cost_profile": "tracking_stop" if bool(stop_goal_active) else "tracking",
            "mpc_cost_profile_switch_reason": "mode2",
            "mpc_fallback_reason": str(fallback_reason),
            "control_guard_reason": (
                str(control_guard_reason)
                if str(control_guard_reason)
                else f"mode2_control_source:{control_source}"
            ),
            "accel_cmd_mps2": float(accel_mps2),
            "steer_cmd_rad": float(steer_rad),
            "planner_input_cp_traffic_control_count": len(list(cp_payload.get("control", []) or [])),
            "planner_input_prediction_risky_lane_count": "",
            "planner_input_perception_planning_count": len(object_snapshots),
            "planner_input_cp_obstacle_count": len(list(cp_payload.get("obstacles", []) or [])),
            "traffic_signal_state": str(traffic_state),
            "traffic_control_from_cp": bool(signal_context.get("from_cp", False)),
            "object_memory_reason": str(object_memory_reason),
            "traffic_memory_reason": str(traffic_memory_reason),
            "reference_memory_reason": str(reference_memory_reason),
            "trajectory_memory_reason": str(trajectory_memory_reason),
            "stop_target_forward_m": "" if stop_target_forward_m is None else float(stop_target_forward_m),
            "stop_approach_speed_mps": float(stop_approach_speed_mps),
            "green_release_reference_active": bool(release_reference_active),
            "lane_safety_scores": "",
            "mpc_trajectory_points": self._last_mpc_trajectory_points(),
            "global_route_points": self._active_global_route_points(),
            "lane_reference_points": [
                [
                    float(sample.get("x_ref_m", sample.get("x", 0.0))),
                    float(sample.get("y_ref_m", sample.get("y", 0.0))),
                ]
                for sample in list(lane_center_reference or [])
            ],
        }
        self._mode2_last_stop_goal_active = bool(stop_goal_active)
        self._draw_world_debug_primitives(
            destination_state=destination_state or [],
            lane_center_reference=lane_center_reference,
        )
        self._record_debug(self.last_debug)
        return control

    def _opencda_behavior_target(self) -> tuple[float, Any, str]:
        agent = getattr(self.vehicle_manager, "agent", None)
        if agent is None or not hasattr(agent, "run_step"):
            return 0.0, None, "missing_opencda_agent"
        try:
            target_speed_kmh, target_loc = agent.run_step(float(self.target_speed_mps) * 3.6)
            return float(target_speed_kmh or 0.0), target_loc, ""
        except SystemExit:
            raise
        except Exception as exc:
            if self.debug:
                print(f"[CP-X OpenCDA Bridge] OpenCDA agent run_step failed: {exc}")
            return 0.0, None, str(exc)

    def _opencda_pid_fallback_control(self, target_speed_kmh: float, target_loc: Any) -> carla.VehicleControl:
        controller = getattr(self.vehicle_manager, "controller", None)
        if controller is None or not hasattr(controller, "run_step"):
            return self._fallback_brake_control()
        try:
            control = controller.run_step(float(target_speed_kmh or 0.0), target_loc)
            self._mode2_last_pid_control = control
            self._last_accel_mps2 = 0.0
            self._last_steer_rad = float(getattr(control, "steer", 0.0)) * float(
                getattr(self.mpc.constraints, "max_steer_rad", 0.3)
            )
            return control
        except Exception as exc:
            if self.debug:
                print(f"[CP-X OpenCDA Bridge] OpenCDA PID fallback failed: {exc}")
            return self._fallback_brake_control()

    @staticmethod
    def _fallback_brake_control() -> carla.VehicleControl:
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)

    def _mode2_note_pid_fallback(self, *, sim_time_s: float) -> None:
        self._mode2_consecutive_mpc_success = 0
        self._mode2_last_control_source = "pid"
        hold_s = float(self.config.get("mode2_pid_hold_after_mpc_fail_s", 0.75))
        self._mode2_pid_hold_until_sim_time_s = max(
            float(self._mode2_pid_hold_until_sim_time_s),
            float(sim_time_s) + max(0.0, hold_s),
        )

    def _mode2_arbitrate_control(
        self,
        *,
        proposed_control: carla.VehicleControl,
        proposed_accel_mps2: float,
        proposed_steer_rad: float,
        target_speed_kmh: float,
        target_loc: Any,
        sim_time_s: float,
        stop_goal_active: bool,
        destination_state: Sequence[float],
        ego_location: carla.Location,
        ego_yaw_rad: float,
    ) -> tuple[carla.VehicleControl, float, float, str, str]:
        if float(sim_time_s) < float(self._mode2_pid_hold_until_sim_time_s):
            control = self._opencda_pid_fallback_control(target_speed_kmh, target_loc)
            return control, self._last_accel_mps2, self._last_steer_rad, "pid", "mode2_pid_hold"

        _, destination_lateral_m = self._body_frame_xy(
            origin_x_m=float(ego_location.x),
            origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        destination_forward_m, _ = self._body_frame_xy(
            origin_x_m=float(ego_location.x),
            origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        max_lateral_m = float(self.config.get("mode2_mpc_takeover_max_lateral_m", 1.5))
        if abs(float(destination_lateral_m)) > max_lateral_m:
            self._mode2_note_pid_fallback(sim_time_s=sim_time_s)
            control = self._opencda_pid_fallback_control(target_speed_kmh, target_loc)
            return control, self._last_accel_mps2, self._last_steer_rad, "pid", "mode2_lateral_gate"

        self._mode2_consecutive_mpc_success += 1
        min_success = int(
            self.config.get(
                "mode2_mpc_takeover_success_frames_stop" if bool(stop_goal_active)
                else "mode2_mpc_takeover_success_frames",
                6 if bool(stop_goal_active) else 3,
            )
        )
        if bool(stop_goal_active) and float(destination_forward_m) > float(
            self.config.get("mode2_stop_approach_distance_m", 8.0)
        ):
            min_success = int(self.config.get("mode2_mpc_takeover_success_frames", 3))
        if self._mode2_last_control_source != "mpc" and self._mode2_consecutive_mpc_success < max(1, min_success):
            control = self._opencda_pid_fallback_control(target_speed_kmh, target_loc)
            return control, self._last_accel_mps2, self._last_steer_rad, "pid", "mode2_wait_mpc_stability"

        last_pid = self._mode2_last_pid_control
        if last_pid is not None and self._mode2_last_control_source != "mpc":
            max_steer_jump = float(self.config.get("mode2_mpc_takeover_max_steer_jump", 0.25))
            max_throttle_jump = float(self.config.get("mode2_mpc_takeover_max_throttle_jump", 0.45))
            max_brake_jump = float(self.config.get("mode2_mpc_takeover_max_brake_jump", 0.45))
            if (
                abs(float(getattr(proposed_control, "steer", 0.0)) - float(getattr(last_pid, "steer", 0.0))) > max_steer_jump
                or abs(float(getattr(proposed_control, "throttle", 0.0)) - float(getattr(last_pid, "throttle", 0.0))) > max_throttle_jump
                or abs(float(getattr(proposed_control, "brake", 0.0)) - float(getattr(last_pid, "brake", 0.0))) > max_brake_jump
            ):
                self._mode2_note_pid_fallback(sim_time_s=sim_time_s)
                control = self._opencda_pid_fallback_control(target_speed_kmh, target_loc)
                return control, self._last_accel_mps2, self._last_steer_rad, "pid", "mode2_takeover_jump_gate"

        self._mode2_last_control_source = "mpc"
        return (
            proposed_control,
            float(proposed_accel_mps2),
            float(proposed_steer_rad),
            "mpc",
            "mode2_mpc_active",
        )

    @staticmethod
    def _has_close_forward_obstacle(
        *,
        object_snapshots: Sequence[Mapping[str, Any]],
        ego_location: carla.Location,
        ego_yaw_rad: float,
        max_forward_m: float,
        max_lateral_m: float,
    ) -> bool:
        cos_h = math.cos(float(ego_yaw_rad))
        sin_h = math.sin(float(ego_yaw_rad))
        for snapshot in list(object_snapshots or []):
            try:
                dx_m = float(snapshot.get("x", 0.0)) - float(ego_location.x)
                dy_m = float(snapshot.get("y", 0.0)) - float(ego_location.y)
            except Exception:
                continue
            forward_m = dx_m * cos_h + dy_m * sin_h
            lateral_m = -dx_m * sin_h + dy_m * cos_h
            if 0.5 <= float(forward_m) <= float(max_forward_m) and abs(float(lateral_m)) <= float(max_lateral_m):
                return True
        return False

    def _next_opencda_forward_target(self, *, ego_transform: carla.Transform):
        local_planner = getattr(getattr(self.vehicle_manager, "agent", None), "get_local_planner", lambda: None)()
        if local_planner is None:
            return None
        try:
            entries = list(local_planner.get_waypoint_buffer() or [])
        except Exception:
            entries = []
        if not entries:
            try:
                entries = list(local_planner.get_waypoints_queue() or [])[:20]
            except Exception:
                entries = []
        ex = float(ego_transform.location.x)
        ey = float(ego_transform.location.y)
        yaw_rad = math.radians(float(ego_transform.rotation.yaw))
        cos_h = math.cos(yaw_rad)
        sin_h = math.sin(yaw_rad)
        for entry in entries:
            waypoint = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
            transform = getattr(waypoint, "transform", None)
            location = getattr(transform, "location", None)
            if location is None:
                continue
            dx_m = float(location.x) - ex
            dy_m = float(location.y) - ey
            distance_m = math.hypot(dx_m, dy_m)
            forward_m = dx_m * cos_h + dy_m * sin_h
            if distance_m > 0.5 and forward_m > 0.5 * distance_m:
                return location
        return None

    def _select_mode2_relevant_traffic_control(
        self,
        *,
        cp_payload: Mapping[str, Any],
        ego_location: carla.Location,
        ego_heading_rad: float,
        sim_time_s: float,
    ) -> Mapping[str, object] | None:
        traffic_controls = list(cp_payload.get("control", []) or [])
        waypoint = self._map_waypoint_from_location(ego_location)
        return self._select_relevant_traffic_control(
            traffic_controls=traffic_controls,
            ego_location=ego_location,
            ego_heading_rad=float(ego_heading_rad),
            current_lane_id=int(getattr(waypoint, "lane_id", 0) or 0),
            current_road_id=int(getattr(waypoint, "road_id", 0) or 0),
            sim_time_s=float(sim_time_s),
        )

    def _opencda_local_planner_reference_samples(
        self,
        *,
        target_speed_mps: float,
        ego_transform: carla.Transform,
    ) -> list[dict[str, float]]:
        local_planner = getattr(getattr(self.vehicle_manager, "agent", None), "get_local_planner", lambda: None)()
        if local_planner is None:
            return []

        raw_points: list[tuple[float, float, float, int, float]] = []
        try:
            trajectory = list(local_planner.get_trajectory() or [])
        except Exception:
            trajectory = []
        for entry in trajectory:
            point = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
            location = getattr(point, "location", None)
            transform = getattr(point, "transform", None)
            if location is None and transform is not None:
                location = getattr(transform, "location", None)
            if location is None:
                continue
            speed_kmh = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else None
            if bool(self.config.get("mode2_use_opencda_trajectory_speed", False)):
                speed_mps = float(target_speed_mps if speed_kmh is None else float(speed_kmh) / 3.6)
            else:
                speed_mps = float(target_speed_mps)
            raw_points.append((
                float(location.x),
                float(location.y),
                float(speed_mps),
                0,
                3.5,
            ))

        if len(raw_points) < 2:
            try:
                waypoint_buffer = list(local_planner.get_waypoint_buffer() or [])
            except Exception:
                waypoint_buffer = []
            for entry in waypoint_buffer:
                waypoint = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
                transform = getattr(waypoint, "transform", None)
                location = getattr(transform, "location", None)
                if location is None:
                    continue
                raw_points.append((
                    float(location.x),
                    float(location.y),
                    float(target_speed_mps),
                    int(getattr(waypoint, "lane_id", 0) or 0),
                    float(getattr(waypoint, "lane_width", 3.5) or 3.5),
                ))

        return self._reference_samples_from_xy_speed(
            raw_points,
            ego_transform=ego_transform,
        )

    def _reference_samples_from_target_location(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        target_loc: Any,
        target_speed_mps: float,
    ) -> list[dict[str, float]]:
        if target_loc is None:
            return []
        try:
            target_x = float(target_loc.x)
            target_y = float(target_loc.y)
        except Exception:
            return []
        distance_m = max(1.0, math.hypot(target_x - float(ego_location.x), target_y - float(ego_location.y)))
        sample_count = max(2, min(int(self.mpc.horizon_steps) + 1, int(math.ceil(distance_m / 0.75)) + 1))
        raw_points = []
        for idx in range(sample_count):
            ratio = float(idx) / float(max(1, sample_count - 1))
            raw_points.append((
                float(ego_location.x) + ratio * (target_x - float(ego_location.x)),
                float(ego_location.y) + ratio * (target_y - float(ego_location.y)),
                float(target_speed_mps),
                0,
                3.5,
            ))
        return self._reference_samples_from_xy_speed(
            raw_points,
            fallback_heading_rad=float(ego_yaw_rad),
            ego_transform=carla.Transform(
                ego_location,
                carla.Rotation(yaw=math.degrees(float(ego_yaw_rad))),
            ),
        )

    def _reference_samples_from_xy_speed(
        self,
        raw_points: Sequence[tuple[float, float, float, int, float]],
        *,
        fallback_heading_rad: float = 0.0,
        ego_transform: carla.Transform | None = None,
    ) -> list[dict[str, float]]:
        raw_points = self._clean_mode2_reference_raw_points(
            raw_points=raw_points,
            ego_transform=ego_transform,
        )
        if len(raw_points) < 2:
            return []
        samples: list[dict[str, float]] = []
        max_points = max(2, int(getattr(self.mpc, "horizon_steps", 20)) + 1)
        selected = list(raw_points)[:max_points]
        previous_heading_rad = None
        for idx, (x_m, y_m, speed_mps, lane_id, lane_width_m) in enumerate(selected):
            if idx < len(selected) - 1:
                nx, ny = selected[idx + 1][0], selected[idx + 1][1]
                heading_rad = math.atan2(float(ny) - float(y_m), float(nx) - float(x_m))
            elif samples:
                heading_rad = float(samples[-1]["heading_rad"])
            else:
                heading_rad = float(fallback_heading_rad)
            if previous_heading_rad is not None:
                while heading_rad - previous_heading_rad > math.pi:
                    heading_rad -= 2.0 * math.pi
                while heading_rad - previous_heading_rad < -math.pi:
                    heading_rad += 2.0 * math.pi
            previous_heading_rad = float(heading_rad)
            lane_width = max(0.1, float(lane_width_m or 3.5))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "v_ref_mps": float(speed_mps),
                "lane_id": int(lane_id or 0),
                "lane_width_m": float(lane_width),
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * float(lane_width),
                "road_right_width_m": 0.5 * float(lane_width),
            })
        return samples

    def _clean_mode2_reference_raw_points(
        self,
        *,
        raw_points: Sequence[tuple[float, float, float, int, float]],
        ego_transform: carla.Transform | None,
        min_spacing_m: float = 0.35,
    ) -> list[tuple[float, float, float, int, float]]:
        cleaned: list[tuple[float, float, float, int, float]] = []
        ex = ey = cos_h = sin_h = None
        candidates: list[tuple[float, float, float, float, float, int, float]] = []
        if ego_transform is not None:
            ex = float(ego_transform.location.x)
            ey = float(ego_transform.location.y)
            yaw_rad = math.radians(float(ego_transform.rotation.yaw))
            cos_h = math.cos(yaw_rad)
            sin_h = math.sin(yaw_rad)
        for point in list(raw_points or []):
            if len(point) < 5:
                continue
            x_m, y_m, speed_mps, lane_id, lane_width_m = point
            x_m = float(x_m)
            y_m = float(y_m)
            forward_m = float("nan")
            lateral_m = float("nan")
            if ex is not None and ey is not None and cos_h is not None and sin_h is not None:
                dx_m = x_m - ex
                dy_m = y_m - ey
                forward_m = dx_m * cos_h + dy_m * sin_h
                lateral_m = -dx_m * sin_h + dy_m * cos_h
                # Keep points around the nose, but remove clearly behind points.
                if float(forward_m) < -0.25:
                    continue
            candidates.append((
                float(forward_m),
                float(lateral_m),
                x_m,
                y_m,
                float(speed_mps),
                int(lane_id or 0),
                float(lane_width_m or 3.5),
            ))
        if ex is not None and bool(self.config.get("mode2_sort_reference_by_forward", True)):
            candidates.sort(key=lambda item: (float(item[0]), abs(float(item[1]))))
        previous_forward_m = None
        for forward_m, _lateral_m, x_m, y_m, speed_mps, lane_id, lane_width_m in candidates:
            if previous_forward_m is not None and math.isfinite(float(forward_m)):
                min_forward_spacing_m = float(
                    self.config.get("mode2_min_reference_forward_spacing_m", 0.25)
                )
                if float(forward_m) - float(previous_forward_m) < min_forward_spacing_m:
                    continue
            if cleaned:
                prev_x, prev_y = cleaned[-1][0], cleaned[-1][1]
                if math.hypot(x_m - float(prev_x), y_m - float(prev_y)) < float(min_spacing_m):
                    continue
            cleaned.append((x_m, y_m, float(speed_mps), int(lane_id or 0), float(lane_width_m or 3.5)))
            if math.isfinite(float(forward_m)):
                previous_forward_m = float(forward_m)
        return cleaned

    def _mode2_stop_reference_samples(
        self,
        *,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        lane_center_reference: Sequence[Mapping[str, Any]],
        target_loc: Any,
        stop_target: Mapping[str, Any] | None,
        approach_speed_mps: float = 0.0,
    ) -> list[dict[str, float]]:
        """Build a short, non-degenerate stop reference for mode 2."""

        raw_points: list[tuple[float, float, float, int, float]] = []
        reference_speed_mps = max(0.0, float(approach_speed_mps))
        target_x = target_y = None
        if isinstance(stop_target, Mapping):
            target_x = stop_target.get("x_m", stop_target.get("x", None))
            target_y = stop_target.get("y_m", stop_target.get("y", None))

        # CP traffic controls carry a fixed stop line. Prefer that fixed point
        # over OpenCDA's local-planner samples; otherwise the stop target moves
        # forward with the ego vehicle and the MPC keeps creeping through red.
        if target_x is None or target_y is None:
            target_x = target_y = None
        else:
            target_x = float(target_x)
            target_y = float(target_y)

        lane_samples = [dict(sample) for sample in list(lane_center_reference or [])]
        if lane_samples and (target_x is None or target_y is None):
            raw_points.append((
                float(ego_location.x),
                float(ego_location.y),
                float(reference_speed_mps),
                int(lane_samples[0].get("lane_id", 0) or 0),
                float(lane_samples[0].get("lane_width_m", 3.5) or 3.5),
            ))
            for sample in lane_samples:
                raw_points.append((
                    float(sample.get("x_ref_m", sample.get("x", ego_location.x))),
                    float(sample.get("y_ref_m", sample.get("y", ego_location.y))),
                    float(reference_speed_mps),
                    int(sample.get("lane_id", 0) or 0),
                    float(sample.get("lane_width_m", 3.5) or 3.5),
                ))
            return self._reference_samples_from_xy_speed(
                raw_points,
                fallback_heading_rad=float(ego_yaw_rad),
            )

        if (target_x is None or target_y is None) and target_loc is not None:
            target_x = getattr(target_loc, "x", None)
            target_y = getattr(target_loc, "y", None)
        if target_x is None or target_y is None:
            stop_distance_m = max(2.0, float(self.config.get("mode2_default_stop_reference_m", 4.0)))
            target_x = float(ego_location.x) + stop_distance_m * math.cos(float(ego_yaw_rad))
            target_y = float(ego_location.y) + stop_distance_m * math.sin(float(ego_yaw_rad))

        target_x = float(target_x)
        target_y = float(target_y)
        dx_m = target_x - float(ego_location.x)
        dy_m = target_y - float(ego_location.y)
        forward_m = dx_m * math.cos(float(ego_yaw_rad)) + dy_m * math.sin(float(ego_yaw_rad))
        lateral_m = -dx_m * math.sin(float(ego_yaw_rad)) + dy_m * math.cos(float(ego_yaw_rad))
        if forward_m < 0.5:
            target_x = float(ego_location.x) + 0.5 * math.cos(float(ego_yaw_rad))
            target_y = float(ego_location.y) + 0.5 * math.sin(float(ego_yaw_rad))
            forward_m = 0.5
            lateral_m = 0.0
        max_stop_lateral_m = float(self.config.get("mode2_stop_reference_max_lateral_m", 1.5))
        if abs(float(lateral_m)) > max_stop_lateral_m:
            lateral_m = max(-max_stop_lateral_m, min(max_stop_lateral_m, float(lateral_m)))
            target_x = float(ego_location.x) + forward_m * math.cos(float(ego_yaw_rad)) - lateral_m * math.sin(float(ego_yaw_rad))
            target_y = float(ego_location.y) + forward_m * math.sin(float(ego_yaw_rad)) + lateral_m * math.cos(float(ego_yaw_rad))

        distance_m = max(1.0, math.hypot(target_x - float(ego_location.x), target_y - float(ego_location.y)))
        sample_count = max(3, min(int(getattr(self.mpc, "horizon_steps", 20)) + 1, int(math.ceil(distance_m / 0.5)) + 1))
        for idx in range(sample_count):
            ratio = float(idx) / float(max(1, sample_count - 1))
            sample_speed_mps = float(reference_speed_mps)
            if float(reference_speed_mps) > 0.0 and ratio > 0.65:
                taper = max(0.0, 1.0 - (ratio - 0.65) / 0.35)
                sample_speed_mps = float(reference_speed_mps) * float(taper)
            raw_points.append((
                float(ego_location.x) + ratio * (target_x - float(ego_location.x)),
                float(ego_location.y) + ratio * (target_y - float(ego_location.y)),
                float(sample_speed_mps),
                0,
                3.5,
            ))
        return self._reference_samples_from_xy_speed(
            raw_points,
            fallback_heading_rad=float(ego_yaw_rad),
        )

    def _mode2_should_use_pid_for_stop(
        self,
        *,
        ego_speed_mps: float,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        destination_state: Sequence[float],
    ) -> bool:
        if len(destination_state or []) < 2:
            return True
        destination_forward_m, _ = self._body_frame_xy(
            origin_x_m=float(ego_location.x),
            origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        if float(destination_forward_m) > float(self.config.get("mode2_stop_approach_distance_m", 8.0)):
            return False
        max_stop_mpc_speed_mps = float(self.config.get("mode2_stop_mpc_max_speed_mps", 1.2))
        if float(ego_speed_mps) > max_stop_mpc_speed_mps:
            return True
        _, lateral_m = self._body_frame_xy(
            origin_x_m=float(ego_location.x),
            origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        return abs(float(lateral_m)) > float(self.config.get("mode2_stop_mpc_max_lateral_m", 2.0))

    def _destination_from_opencda_reference(
        self,
        *,
        lane_center_reference: Sequence[Mapping[str, Any]],
        ego_location: carla.Location,
        ego_yaw_rad: float,
        target_loc: Any,
        speed_ref_mps: float,
        stop_target: Mapping[str, Any] | None,
    ) -> list[float] | None:
        if isinstance(stop_target, Mapping):
            x_value = stop_target.get("x_m", stop_target.get("x", None))
            y_value = stop_target.get("y_m", stop_target.get("y", None))
            if x_value is not None and y_value is not None:
                return [
                    float(x_value),
                    float(y_value),
                    float(speed_ref_mps),
                    float(ego_yaw_rad),
                    int(stop_target.get("lane_id", 0) or 0),
                ]
        if lane_center_reference:
            index = min(len(lane_center_reference) - 1, max(1, int(len(lane_center_reference) * 0.6)))
            sample = dict(lane_center_reference[index])
            return [
                float(sample.get("x_ref_m", sample.get("x", ego_location.x))),
                float(sample.get("y_ref_m", sample.get("y", ego_location.y))),
                float(speed_ref_mps),
                float(sample.get("heading_rad", ego_yaw_rad)),
                int(sample.get("lane_id", 0) or 0),
            ]
        if target_loc is not None:
            try:
                return [
                    float(target_loc.x),
                    float(target_loc.y),
                    float(speed_ref_mps),
                    float(math.atan2(float(target_loc.y) - float(ego_location.y), float(target_loc.x) - float(ego_location.x))),
                    0,
                ]
            except Exception:
                return None
        return None

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
            row = {name: payload.get(name, "") for name in self._debug_fieldnames}
            self._debug_writer.writerow(row)
            self._debug_csv_file.flush()
            if self._debug_jsonl_file is not None:
                self._debug_jsonl_file.write(json.dumps(dict(payload), default=str) + "\n")
                self._debug_jsonl_file.flush()
        except Exception as exc:
            if self.debug:
                print(f"[CP-X OpenCDA Bridge] debug record failed: {exc}")

    def destroy(self) -> None:
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
            compute_temp_destination,
            generate_mpc_reference,
            select_reference_intent,
        )
        from opencda.planning_module.pipeline.candidate_evaluation import (
            evaluate_behavior_candidates,
        )

        sim_time_s = self._sim_time_s()
        adapter_output = self.input_adapter.build(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            object_snapshots=object_snapshots,
            cp_payload=cp_payload,
        )
        planner_input_frame = adapter_output.frame
        ego_pose = adapter_output.ego_pose
        current_state = adapter_output.current_state
        current_lane_id = int(adapter_output.current_lane_id)
        ego_waypoint = adapter_output.ego_waypoint
        lane_safety_scores = dict(adapter_output.lane_safety_scores)
        front_dist_by_lane = dict(adapter_output.front_distance_by_lane)
        route_points = list(adapter_output.route_points)
        route_context = planner_input_frame.planning.route
        route_optimal_lane_id = int(adapter_output.route_optimal_lane_id)
        route_reference_allowed = bool(adapter_output.route_reference_allowed)
        route_reference_gate_reason = str(adapter_output.route_reference_gate_reason)
        route_lane_change_allowed = bool(route_reference_allowed) and (
            "direct_fallback" not in str(route_reference_gate_reason)
        )
        route_lane_change_required = (
            bool(route_lane_change_allowed)
            and int(route_optimal_lane_id) != 0
            and int(route_optimal_lane_id) != int(current_lane_id)
        )
        prediction_risky_lane_count = 0
        for risk in dict(planner_input_frame.prediction.lane_prediction_risks).values():
            risk_mapping = dict(risk) if isinstance(risk, Mapping) else {}
            if bool(risk_mapping.get("risk", False)):
                prediction_risky_lane_count += 1
        dense_traffic_active = (
            bool(self.full_dense_traffic_lane_change_lock_enabled)
            and (
                len(list(object_snapshots or [])) >= int(self.full_dense_traffic_object_count)
                or int(prediction_risky_lane_count) >= int(self.full_dense_traffic_risky_lane_count)
            )
        )
        start_lane_change_lock_active = (
            float(sim_time_s) <= float(self.full_lane_change_start_lock_s)
        )
        opportunistic_lane_change_allowed = bool(route_lane_change_allowed) and (
            bool(route_lane_change_required)
            or (
                not bool(start_lane_change_lock_active)
                and not bool(dense_traffic_active)
            )
        )
        lane_change_gate_reason = ""
        if bool(route_lane_change_allowed) and not bool(opportunistic_lane_change_allowed):
            reasons = []
            if bool(start_lane_change_lock_active):
                reasons.append("start_lock")
            if bool(dense_traffic_active):
                reasons.append("dense_traffic")
            lane_change_gate_reason = "opportunistic_lane_change_suppressed:" + "+".join(reasons)
        signal_context = dict(adapter_output.signal_context)
        source_quality = dict(adapter_output.source_quality)
        mpc_feedback = self.mpc_feedback.candidate_feedback(
            current_time_s=float(sim_time_s)
        )
        candidate_lane_ids = (
            list(planner_input_frame.map_lane.allowed_lane_ids)
            if bool(opportunistic_lane_change_allowed)
            else [int(current_lane_id)]
        )
        candidate_frame = evaluate_behavior_candidates(
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
            ego_lane_id=int(current_lane_id),
            selected_lane_id=int(current_lane_id),
            available_lane_ids=list(candidate_lane_ids),
            route_optimal_lane_id=int(route_optimal_lane_id),
            mode="INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL",
            mpc_feedback_blocked_lane_ids=list(
                mpc_feedback.get("blocked_lane_ids", []) or []
            ),
            mpc_feedback_weight=float(self.config.get("mpc_feedback_candidate_weight", 80.0)),
        )
        preferred_target_lane_id = (
            int(candidate_frame.selected.target_lane_id)
            if bool(opportunistic_lane_change_allowed)
            else int(current_lane_id)
        )

        command = self.behavior_planner.update(
            lane_safety_scores=lane_safety_scores,
            ego_lane_id=int(current_lane_id),
            selected_lane_id=int(current_lane_id),
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL",
            route_optimal_lane_id=int(route_optimal_lane_id),
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            front_obstacle_distance_by_lane=front_dist_by_lane,
            current_time_s=float(sim_time_s),
            wall_time_s=float(sim_time_s),
            traffic_signal_state=str(planner_input_frame.planning.traffic_control.signal_state),
            traffic_stop_target=(
                planner_input_frame.planning.traffic_control.stop_target.as_dict()
                if planner_input_frame.planning.traffic_control.stop_target.active
                else None
            ),
            traffic_signal_context=dict(signal_context or {}),
            ego_speed_mps=float(ego_speed_mps),
            ego_max_deceleration_mps2=abs(float(self.mpc.constraints.min_acceleration_mps2)),
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            ego_position_xy=(float(ego_location.x), float(ego_location.y)),
            global_route_points=route_points,
            nearest_front_obstacles_by_lane={},
            lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
            preferred_target_lane_id=int(preferred_target_lane_id),
        )
        decision = str(command.get("decision", "lane_follow"))
        target_lane_id = int(command.get("target_lane_id", current_lane_id) or current_lane_id)
        lc_state = str(command.get("lc_state", "LANE_KEEP"))
        behavior_override_reason = ""
        if (
            not bool(opportunistic_lane_change_allowed)
            and str(decision) in {"lane_change_left", "lane_change_right"}
        ):
            decision = "lane_follow"
            target_lane_id = int(current_lane_id)
            lc_state = "LANE_KEEP"
            behavior_override_reason = (
                str(lane_change_gate_reason)
                if str(lane_change_gate_reason)
                else "lane_change_suppressed_without_valid_route"
            )
            reset_lane_change = getattr(self.behavior_planner, "_reset_lane_change_state", None)
            if callable(reset_lane_change):
                reset_lane_change(reason=str(behavior_override_reason))
        if (
            bool(self.full_prepare_lane_change_reference_lock)
            and str(lc_state).upper().startswith("PREPARE_LANE_CHANGE")
            and not bool(route_lane_change_required)
        ):
            decision = "lane_follow"
            target_lane_id = int(current_lane_id)
            lc_state = "LANE_KEEP"
            behavior_override_reason = (
                str(behavior_override_reason) + ";"
                if str(behavior_override_reason)
                else ""
            ) + "prepare_lane_change_reference_locked_to_current_lane"
            reset_lane_change = getattr(self.behavior_planner, "_reset_lane_change_state", None)
            if callable(reset_lane_change):
                reset_lane_change(reason="prepare_lane_change_reference_locked")
        if str(decision) in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
            speed_ref_mps = 0.0
        planner_mode = "INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL"
        self._apply_mpc_cost_profile(
            behavior=str(decision),
            planner_lc_state=str(lc_state),
            planner_mode=str(planner_mode),
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            sim_time_s=float(sim_time_s),
        )

        self._temporary_destination_state = compute_temp_destination(
            map_planner=self.reference_map,
            ego_pose=ego_pose,
            target_lane_id=int(target_lane_id),
            decision=str(decision),
            lookahead_m=float(self.lookahead_m),
            target_v_mps=float(speed_ref_mps),
            global_route_points=route_points,
            mode_reference_xy=(
                None
                if self._temporary_destination_state is None
                else (
                    float(self._temporary_destination_state[0]),
                    float(self._temporary_destination_state[1]),
                )
            ),
            prev_mode=(
                None
                if self._temporary_destination_state is None or len(self._temporary_destination_state) < 6
                else float(self._temporary_destination_state[5])
            ),
            prev_road_id=(
                None
                if self._temporary_destination_state is None or len(self._temporary_destination_state) < 7
                else int(self._temporary_destination_state[6])
            ),
            prev_entered_intersection=(
                False
                if self._temporary_destination_state is None or len(self._temporary_destination_state) < 8
                else bool(float(self._temporary_destination_state[7]) > 0.5)
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
            previous_lane_center_reference=self._previous_lane_center_reference,
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
                float(self._temporary_destination_state[0]),
                float(self._temporary_destination_state[1]),
            ),
            current_temp_mode_value=(
                float(self._temporary_destination_state[5])
                if len(self._temporary_destination_state) >= 6 else 0.0
            ),
            current_temp_road_id=(
                int(self._temporary_destination_state[6])
                if len(self._temporary_destination_state) >= 7 else None
            ),
            current_temp_entered_intersection=(
                bool(float(self._temporary_destination_state[7]) > 0.5)
                if len(self._temporary_destination_state) >= 8 else False
            ),
            active_reference_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            current_temp_mode_str=str(planner_mode),
            lane_reference_speed_mps=max(
                1.0,
                float(ego_speed_mps),
                abs(float(speed_ref_mps)),
            ),
            lane_reference_step_distance_m=max(
                0.5,
                float(self.mpc.dt_s)
                * max(1.0, float(ego_speed_mps), abs(float(speed_ref_mps))),
            ),
            mpc_horizon_steps=int(self.mpc.horizon_steps),
            mpc_dt_s=float(self.mpc.dt_s),
            temporary_destination_state=self._temporary_destination_state,
            lane_reference_freeze_count=int(self._lane_reference_freeze_count),
            sim_time_s=float(sim_time_s),
            stop_release_temp_smooth_until_sim_time_s=float(self._stop_release_temp_smooth_until_sim_time_s),
        )
        ref_output = generate_mpc_reference(ref_context)
        local_lane_center_reference = [
            dict(sample) for sample in list(ref_output.local_lane_center_reference or [])
        ]
        self._temporary_destination_state = list(ref_output.temporary_destination_state or self._temporary_destination_state)
        self._lane_reference_freeze_count = int(ref_output.lane_reference_freeze_count)
        reference_debug = dict(ref_output.mpc_reference_result.trace.as_trace_fields())
        reference_debug.update(planner_input_frame.trace_fields())
        reference_debug.update({
            "stage": reference_debug.get("reference_pipeline_stage", ""),
            "intent_mode": reference_debug.get("reference_pipeline_intent_mode", ""),
            "fallback_reason": str(ref_output.last_reference_fallback_reason),
            "reference_source": "behavior_reference_pipeline",
            "route_reference_allowed": bool(route_reference_allowed),
            "route_reference_gate_reason": str(route_reference_gate_reason),
            "route_lane_change_allowed": bool(route_lane_change_allowed),
            "opportunistic_lane_change_allowed": bool(opportunistic_lane_change_allowed),
            "lane_change_gate_reason": str(lane_change_gate_reason),
            "route_lane_change_required": bool(route_lane_change_required),
            "behavior_override_reason": str(behavior_override_reason),
            "route_current_road_option": str(route_context.current_road_option),
            "route_next_macro_maneuver": str(route_context.next_macro_maneuver),
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
        reference_debug.update(source_quality)
        # Keep OpenCDA aligned with the standalone planning runner: the
        # reference pipeline owns lane-follow trimming, blending, jump freezing,
        # and geometry guards. The optional strict lane-follow path below is an
        # experimental fallback and should stay disabled for normal testing.
        strict_lane_follow = (
            bool(self.strict_lane_follow_reference)
            and str(decision) == "lane_follow"
            and str(lc_state or "").upper() in {"IDLE", "LANE_KEEP"}
            and not bool(stop_goal_active)
        )
        if bool(strict_lane_follow):
            lane_follow_reference_source = str(
                self.config.get("lane_follow_reference_source", "auto")
            ).strip().lower()
            step_distance_m = max(
                1.0,
                float(self.mpc.dt_s) * max(float(speed_ref_mps), 3.0),
            )
            strict_reference = []
            strict_reference_source = "opencda_current_lane_center_strict"
            if lane_follow_reference_source in {"global_route", "route", "literal_route"}:
                strict_reference = self._route_aligned_reference_samples(
                    ego_location=ego_location,
                    ego_heading_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    route_points=route_points,
                )
                strict_reference_source = "global_route_aligned_lane_follow"
            if not strict_reference:
                strict_reference = self._current_lane_center_reference_samples(
                    start_waypoint=ego_waypoint,
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    route_points=route_points,
                )
            if self._reference_opposes_heading(
                reference_samples=strict_reference,
                ego_heading_rad=float(ego_yaw_rad),
                max_heading_error_rad=0.5 * math.pi,
            ) or self._reference_lateral_offset_too_large(
                reference_samples=strict_reference,
                ego_state=current_state,
                max_lateral_offset_m=float(
                    self.config.get("lane_follow_reference_max_initial_lateral_m", 1.75)
                ),
            ):
                route_reference = self._route_aligned_reference_samples(
                    ego_location=ego_location,
                    ego_heading_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=max(
                        1.0,
                        float(self.mpc.dt_s) * max(float(speed_ref_mps), 3.0),
                    ),
                    route_points=route_points,
                )
                if route_reference:
                    strict_reference = route_reference
                    strict_reference_source = "global_route_aligned_lane_follow"
            if strict_reference:
                from opencda.planning_module.behavior_planner.reference_pipeline import (
                    lane_center_destination_from_reference,
                )

                local_lane_center_reference = strict_reference
                self._temporary_destination_state = lane_center_destination_from_reference(
                    destination_state=self._temporary_destination_state,
                    lane_center_reference=local_lane_center_reference,
                    ego_state=current_state,
                    target_forward_m=float(
                        self.behavior_runtime_cfg.get(
                            "lane_follow_destination_reference_forward_m", 6.0
                        )
                    ),
                )
                self._lane_reference_freeze_count = 0
                reference_debug["reference_source"] = str(strict_reference_source)
                reference_debug["fallback_reason"] = (
                    f"{reference_debug.get('fallback_reason', '')}:"
                    if str(reference_debug.get("fallback_reason", ""))
                    else ""
                ) + str(strict_reference_source)
                reference_debug["stage"] = "strict_lane_follow"
        reference_memory_reason = ""
        if (
            bool(self.config.get("full_reference_memory_enabled", True))
            and local_lane_center_reference
            and self._temporary_destination_state is not None
        ):
            stabilized_reference, stabilized_destination, reference_memory_reason = (
                self._full_reference_memory.stabilize(
                    reference=local_lane_center_reference,
                    destination_state=self._temporary_destination_state,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    stop_goal_active=bool(stop_goal_active),
                    sim_time_s=float(sim_time_s),
                )
            )
            if stabilized_reference:
                local_lane_center_reference = [
                    dict(sample) for sample in list(stabilized_reference or [])
                ]
            if stabilized_destination is not None:
                self._temporary_destination_state = list(stabilized_destination)
        reference_debug["reference_memory_reason"] = str(reference_memory_reason)
        self._previous_lane_center_reference = [
            dict(sample) for sample in list(local_lane_center_reference or [])
        ]
        return (
            list(self._temporary_destination_state),
            list(local_lane_center_reference),
            {
                "decision": str(decision),
                "lc_state": str(lc_state),
                "target_lane_id": int(target_lane_id),
                "current_lane_id": int(current_lane_id),
                "lane_safety_scores": dict(lane_safety_scores),
                "traffic_signal_state": str(planner_input_frame.planning.traffic_control.signal_state),
                "traffic_control_from_cp": bool(planner_input_frame.planning.traffic_control.from_cp),
            },
            reference_debug,
        )

    def _sim_time_s(self) -> float:
        try:
            snapshot = self.vehicle_manager.vehicle.get_world().get_snapshot()
            return float(snapshot.timestamp.elapsed_seconds)
        except Exception:
            return 0.0

    def _assign_obstacles_to_lanes(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
    ) -> dict[str, int]:
        from utility.global_planner import canonical_lane_id_for_waypoint

        assignments: dict[str, int] = {}
        for snapshot in list(object_snapshots or []):
            obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
            if not obstacle_id:
                continue
            waypoint = self.reference_map.get_waypoint({
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "z": float(snapshot.get("z", 0.0)),
            })
            lane_id = int(canonical_lane_id_for_waypoint(waypoint))
            if int(lane_id) != 0:
                assignments[obstacle_id] = int(lane_id)
        return assignments

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
    ) -> dict[str, object]:
        del ego_heading_rad
        if not hasattr(self, "route_manager"):
            try:
                summary = self.global_planner.get_current_route_info(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    query_key=f"vehicle_{int(getattr(self.vehicle_manager.vehicle, 'id', 0))}",
                )
            except Exception as exc:
                return {
                    "route_found": False,
                    "optimal_lane_id": int(fallback_lane_id),
                    "current_road_option": "",
                    "next_macro_maneuver": "Continue Straight",
                    "debug_reason": f"planning_module_global_route_missing:{exc}",
                }
            lane_id = int(getattr(summary, "optimal_lane_id", fallback_lane_id) or fallback_lane_id)
            if int(lane_id) == 0:
                lane_id = int(fallback_lane_id)
            return {
                "route_found": bool(getattr(summary, "route_found", False)),
                "optimal_lane_id": int(lane_id),
                "current_road_option": str(getattr(summary, "current_road_option", "")),
                "next_macro_maneuver": str(
                    getattr(summary, "next_macro_maneuver", "Continue Straight")
                ),
                "debug_reason": str(
                    getattr(summary, "debug_reason", "planning_module_global_route")
                ),
            }
        return self.route_manager.get_route_info(
            x_m=float(ego_location.x),
            y_m=float(ego_location.y),
            query_key=f"vehicle_{int(getattr(self.vehicle_manager.vehicle, 'id', 0))}",
            fallback_lane_id=int(fallback_lane_id),
        )

    def _route_optimal_lane_id(self, *, ego_location: carla.Location, fallback_lane_id: int) -> int:
        return int(
            self._planning_module_global_route_summary(
                ego_location=ego_location,
                ego_heading_rad=0.0,
                fallback_lane_id=int(fallback_lane_id),
            ).get("optimal_lane_id", fallback_lane_id)
        )

    @staticmethod
    def _nearest_route_index_ahead(
        *,
        route_entries: Sequence[Any],
        ego_location: carla.Location,
        ego_heading_rad: float,
    ) -> Optional[int]:
        cos_h = math.cos(float(ego_heading_rad))
        sin_h = math.sin(float(ego_heading_rad))
        best_index = None
        best_score = None
        for index, (waypoint, _) in enumerate(list(route_entries or [])):
            transform = getattr(waypoint, "transform", None)
            location = getattr(transform, "location", None)
            if location is None:
                continue
            dx_m = float(location.x) - float(ego_location.x)
            dy_m = float(location.y) - float(ego_location.y)
            forward_m = cos_h * dx_m + sin_h * dy_m
            lateral_m = -sin_h * dx_m + cos_h * dy_m
            distance_m = math.hypot(dx_m, dy_m)
            behind_penalty = 20.0 if float(forward_m) < -2.0 else 0.0
            score = (
                float(behind_penalty),
                abs(float(lateral_m)) + 0.15 * max(0.0, -float(forward_m)),
                float(distance_m),
                int(index),
            )
            if best_score is None or score < best_score:
                best_index = int(index)
                best_score = score
        return best_index

    @staticmethod
    def _road_option_name(option: object) -> str:
        if option is None:
            return ""
        name = getattr(option, "name", None)
        if name is not None:
            return str(name).strip().upper()
        text = str(option).strip()
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text.strip().upper()

    @staticmethod
    def _next_macro_maneuver_from_road_options(options: Sequence[str]) -> str:
        normalized = [str(option).strip().upper() for option in list(options or [])]
        for option in normalized:
            if option in {"LEFT", "CHANGELANELEFT"}:
                return "Left Turn"
            if option in {"RIGHT", "CHANGELANERIGHT"}:
                return "Right Turn"
        for option in normalized:
            if option == "STRAIGHT":
                return "Continue Straight"
        return "Continue Straight"

    def _legacy_route_optimal_lane_id(self, *, ego_location: carla.Location, fallback_lane_id: int) -> int:
        from utility.global_planner import canonical_lane_id_for_waypoint

        route = getattr(getattr(self.vehicle_manager, "agent", None), "initial_global_route", None)
        best_waypoint = None
        best_distance = float("inf")
        for entry in list(route or [])[:200]:
            waypoint = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
            transform = getattr(waypoint, "transform", None)
            location = getattr(transform, "location", None)
            if location is None:
                continue
            distance = math.hypot(float(location.x) - float(ego_location.x), float(location.y) - float(ego_location.y))
            if float(distance) < float(best_distance):
                best_distance = float(distance)
                best_waypoint = waypoint
        lane_id = int(canonical_lane_id_for_waypoint(best_waypoint))
        return int(lane_id) if int(lane_id) != 0 else int(fallback_lane_id)

    def _apply_mpc_cost_profile(
        self,
        *,
        behavior: str,
        planner_lc_state: str,
        planner_mode: str,
        next_macro_maneuver: str,
        sim_time_s: float,
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

    def _collect_object_snapshots(self, detected_objects: Any = None) -> list[dict[str, Any]]:
        objects = detected_objects
        if objects is None:
            objects = getattr(self.vehicle_manager.perception_manager, "objects", {}) or {}
        if not isinstance(objects, Mapping):
            objects = getattr(objects, "objects", {}) or {}
        vehicles = list(objects.get("vehicles", []) or [])
        snapshots: list[dict[str, Any]] = []
        for index, obj in enumerate(vehicles):
            actor = getattr(obj, "carla_actor", None) or getattr(obj, "vehicle", None) or obj
            if actor is None or not hasattr(actor, "get_transform"):
                continue
            try:
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                bbox = getattr(actor, "bounding_box", None)
                extent = getattr(bbox, "extent", None)
                speed_mps = math.sqrt(
                    float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
                )
                actor_id = str(getattr(actor, "id", getattr(actor, "carla_id", index)))
                snapshots.append({
                    "vehicle_id": actor_id,
                    "id": actor_id,
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "v": float(speed_mps),
                    "psi": math.radians(float(transform.rotation.yaw)),
                    "length_m": 2.0 * float(getattr(extent, "x", 2.2)),
                    "width_m": 2.0 * float(getattr(extent, "y", 0.9)),
                    "source": "opencda_perception",
                    "provider_source": "native_opencda_perception",
                    "confidence": 1.0,
                })
            except RuntimeError:
                continue
        return snapshots

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
            }
        except Exception:
            return None

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

    def _build_route_reference(
        self,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        speed_ref_mps: float,
    ):
        samples = self._route_samples_from_custom_planner(
            ego_location=ego_location,
        )
        if not samples:
            dest_x = float(ego_location.x + self.lookahead_m * math.cos(ego_yaw_rad))
            dest_y = float(ego_location.y + self.lookahead_m * math.sin(ego_yaw_rad))
            return [dest_x, dest_y, float(speed_ref_mps), float(ego_yaw_rad)], []

        destination = samples[-1]
        return [
            float(destination["x_ref_m"]),
            float(destination["y_ref_m"]),
            float(speed_ref_mps),
            float(destination["heading_rad"]),
        ], samples

    def _route_samples_from_custom_planner(self, *, ego_location: carla.Location):
        if self.map_planner is None:
            return []
        waypoint = self._map_waypoint_from_location(ego_location)
        if waypoint is None:
            return []

        from utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad

        samples: list[dict[str, float]] = []
        traveled_m = 0.0
        current = waypoint
        step_m = max(1.0, min(3.0, float(self.lookahead_m)))
        while current is not None and traveled_m <= float(self.lookahead_m):
            xy_heading = self._waypoint_xy_heading(current)
            if xy_heading is None:
                break
            lane_width_m = self._waypoint_lane_width(current)
            samples.append({
                "x_ref_m": float(xy_heading[0]),
                "y_ref_m": float(xy_heading[1]),
                "heading_rad": float(world_heading_rad(current) or xy_heading[2]),
                "lane_id": int(canonical_lane_id_for_waypoint(current)),
                "lane_width_m": lane_width_m,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * lane_width_m,
                "road_right_width_m": 0.5 * lane_width_m,
            })
            candidates = list(current.next(step_m) or [])
            if not candidates:
                break
            current = candidates[0]
            traveled_m += step_m
        return samples

    def _current_lane_center_reference_samples(
        self,
        *,
        start_waypoint: Any,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        route_points: Sequence[Sequence[float]] | None = None,
    ) -> list[dict[str, float]]:
        """Build a strict lane-follow reference from the current CARLA lane center."""

        if start_waypoint is None:
            return []

        from utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad

        samples: list[dict[str, float]] = []
        current = start_waypoint
        step_m = max(0.5, float(step_distance_m))
        previous_heading = float(world_heading_rad(current) or 0.0)
        first_step_m = max(
            step_m,
            float(self.config.get("lane_follow_reference_first_point_m", 2.0)),
        )
        first_candidates = list(current.next(first_step_m) or [])
        if first_candidates:
            first_current = self._select_smooth_next_waypoint(
                current_waypoint=current,
                candidates=first_candidates,
                previous_heading_rad=float(previous_heading),
                route_points=route_points,
            )
            if first_current is not None:
                current = first_current
                previous_heading = float(world_heading_rad(current) or previous_heading)
        for _ in range(max(1, int(horizon_steps)) + 1):
            xy_heading = self._waypoint_xy_heading(current)
            if xy_heading is None:
                break
            lane_width_m = self._waypoint_lane_width(current)
            heading_rad = float(world_heading_rad(current) or xy_heading[2])
            lane_id = int(canonical_lane_id_for_waypoint(current) or current_lane_id)
            samples.append({
                "x_ref_m": float(xy_heading[0]),
                "y_ref_m": float(xy_heading[1]),
                "x": float(xy_heading[0]),
                "y": float(xy_heading[1]),
                "heading_rad": float(heading_rad),
                "lane_id": int(lane_id),
                "lane_width_m": float(lane_width_m),
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * float(lane_width_m),
                "road_right_width_m": 0.5 * float(lane_width_m),
            })

            candidates = list(current.next(step_m) or [])
            if not candidates:
                break
            current = self._select_smooth_next_waypoint(
                current_waypoint=current,
                candidates=candidates,
                previous_heading_rad=float(previous_heading),
                route_points=route_points,
            )
            if current is None:
                break
            previous_heading = float(world_heading_rad(current) or previous_heading)
        return samples

    @staticmethod
    def _select_smooth_next_waypoint(
        *,
        current_waypoint: Any,
        candidates: Sequence[Any],
        previous_heading_rad: float,
        route_points: Sequence[Sequence[float]] | None = None,
    ):
        if not candidates:
            return None
        route_candidate = CPXMPCPlannerBridge._select_route_aligned_candidate(
            candidates=candidates,
            route_points=route_points,
        )
        if route_candidate is not None:
            return route_candidate
        current_road_id = int(getattr(current_waypoint, "road_id", 0) or 0)
        current_lane_id = int(getattr(current_waypoint, "lane_id", 0) or 0)

        def heading_of(waypoint: Any) -> float:
            transform = getattr(waypoint, "transform", None)
            rotation = getattr(transform, "rotation", None)
            return math.radians(float(getattr(rotation, "yaw", math.degrees(previous_heading_rad))))

        def heading_cost(waypoint: Any) -> float:
            delta = heading_of(waypoint) - float(previous_heading_rad)
            return abs(math.atan2(math.sin(delta), math.cos(delta)))

        same_lane = [
            candidate
            for candidate in candidates
            if int(getattr(candidate, "road_id", 0) or 0) == current_road_id
            and int(getattr(candidate, "lane_id", 0) or 0) == current_lane_id
        ]
        if same_lane:
            return min(same_lane, key=heading_cost)
        same_raw_lane = [
            candidate
            for candidate in candidates
            if int(getattr(candidate, "lane_id", 0) or 0) == current_lane_id
        ]
        if same_raw_lane:
            return min(same_raw_lane, key=heading_cost)
        return min(candidates, key=heading_cost)

    @staticmethod
    def _reference_opposes_heading(
        *,
        reference_samples: Sequence[Mapping[str, Any]],
        ego_heading_rad: float,
        max_heading_error_rad: float,
    ) -> bool:
        points = [
            (
                float(sample.get("x_ref_m", sample.get("x", 0.0))),
                float(sample.get("y_ref_m", sample.get("y", 0.0))),
            )
            for sample in list(reference_samples or [])
        ]
        for first, second in zip(points[:-1], points[1:]):
            dx = float(second[0]) - float(first[0])
            dy = float(second[1]) - float(first[1])
            if math.hypot(dx, dy) <= 0.25:
                continue
            reference_heading = math.atan2(dy, dx)
            error = CPXMPCPlannerBridge._wrap_angle_static(
                float(ego_heading_rad) - float(reference_heading)
            )
            return abs(float(error)) > float(max_heading_error_rad)
        return False

    @staticmethod
    def _reference_lateral_offset_too_large(
        *,
        reference_samples: Sequence[Mapping[str, Any]],
        ego_state: Sequence[float],
        max_lateral_offset_m: float,
    ) -> bool:
        if not reference_samples or len(ego_state) < 4:
            return False
        sample = dict(list(reference_samples)[0])
        dx_m = float(sample.get("x_ref_m", sample.get("x", 0.0))) - float(ego_state[0])
        dy_m = float(sample.get("y_ref_m", sample.get("y", 0.0))) - float(ego_state[1])
        heading_rad = float(ego_state[3])
        lateral_m = -math.sin(heading_rad) * dx_m + math.cos(heading_rad) * dy_m
        return abs(float(lateral_m)) > max(0.0, float(max_lateral_offset_m))

    def _route_aligned_reference_samples(
        self,
        *,
        ego_location: carla.Location,
        ego_heading_rad: float,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        route_points: Sequence[Sequence[float]] | None,
    ) -> list[dict[str, float]]:
        route_xy = [
            (float(point[0]), float(point[1]))
            for point in list(route_points or [])
            if len(point) >= 2
        ]
        if len(route_xy) < 2:
            return []
        ego_xy = (float(ego_location.x), float(ego_location.y))
        route_progress = [0.0]
        for first, second in zip(route_xy[:-1], route_xy[1:]):
            route_progress.append(
                float(route_progress[-1])
                + math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
            )
        base_s = self._project_point_to_polyline_s(
            route_xy=route_xy,
            route_progress=route_progress,
            point_xy=ego_xy,
        )
        samples: list[dict[str, float]] = []
        lane_width_m = self._waypoint_lane_width(self._map_waypoint_from_location(ego_location))
        for step_index in range(max(1, int(horizon_steps)) + 1):
            target_s = float(base_s) + float(step_index) * max(0.5, float(step_distance_m))
            x_m, y_m, heading_rad = self._sample_polyline_at_s(
                route_xy=route_xy,
                route_progress=route_progress,
                target_s=target_s,
                fallback_heading_rad=float(ego_heading_rad),
            )
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": float(lane_width_m),
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * float(lane_width_m),
                "road_right_width_m": 0.5 * float(lane_width_m),
            })
        return samples

    @staticmethod
    def _project_point_to_polyline_s(
        *,
        route_xy: Sequence[tuple[float, float]],
        route_progress: Sequence[float],
        point_xy: tuple[float, float],
    ) -> float:
        best_distance_m = float("inf")
        best_s = float(route_progress[0]) if route_progress else 0.0
        px, py = float(point_xy[0]), float(point_xy[1])
        for idx, (first, second) in enumerate(zip(route_xy[:-1], route_xy[1:])):
            ax, ay = float(first[0]), float(first[1])
            bx, by = float(second[0]), float(second[1])
            dx = bx - ax
            dy = by - ay
            segment_len_sq = dx * dx + dy * dy
            if segment_len_sq <= 1.0e-9:
                continue
            ratio = ((px - ax) * dx + (py - ay) * dy) / segment_len_sq
            ratio = min(1.0, max(0.0, float(ratio)))
            proj_x = ax + ratio * dx
            proj_y = ay + ratio * dy
            distance_m = math.hypot(px - proj_x, py - proj_y)
            if distance_m < best_distance_m:
                best_distance_m = float(distance_m)
                segment_len_m = math.sqrt(segment_len_sq)
                best_s = float(route_progress[idx]) + ratio * segment_len_m
        return float(best_s)

    @staticmethod
    def _sample_polyline_at_s(
        *,
        route_xy: Sequence[tuple[float, float]],
        route_progress: Sequence[float],
        target_s: float,
        fallback_heading_rad: float,
    ) -> tuple[float, float, float]:
        if len(route_xy) == 0:
            return 0.0, 0.0, float(fallback_heading_rad)
        if len(route_xy) == 1:
            return float(route_xy[0][0]), float(route_xy[0][1]), float(fallback_heading_rad)
        if float(target_s) <= float(route_progress[0]):
            first, second = route_xy[0], route_xy[1]
            heading = math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))
            return float(first[0]), float(first[1]), float(heading)
        for idx in range(len(route_xy) - 1):
            s0 = float(route_progress[idx])
            s1 = float(route_progress[idx + 1])
            if float(target_s) > s1:
                continue
            first = route_xy[idx]
            second = route_xy[idx + 1]
            denom = max(1.0e-6, float(s1) - float(s0))
            ratio = min(1.0, max(0.0, (float(target_s) - float(s0)) / denom))
            x_m = float(first[0]) + ratio * (float(second[0]) - float(first[0]))
            y_m = float(first[1]) + ratio * (float(second[1]) - float(first[1]))
            heading = math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))
            return float(x_m), float(y_m), float(heading)
        before = route_xy[-2]
        last = route_xy[-1]
        heading = math.atan2(float(last[1]) - float(before[1]), float(last[0]) - float(before[0]))
        return float(last[0]), float(last[1]), float(heading)

    @staticmethod
    def _select_route_aligned_candidate(
        *,
        candidates: Sequence[Any],
        route_points: Sequence[Sequence[float]] | None,
    ):
        route_xy = [
            (float(point[0]), float(point[1]))
            for point in list(route_points or [])
            if len(point) >= 2
        ]
        if not candidates or len(route_xy) < 2:
            return None

        def candidate_xy(candidate: Any):
            transform = getattr(candidate, "transform", None)
            location = getattr(transform, "location", None)
            if location is None:
                return None
            return float(location.x), float(location.y)

        scored = []
        for candidate in candidates:
            xy = candidate_xy(candidate)
            if xy is None:
                continue
            dist_m = min(
                math.hypot(float(xy[0]) - float(route_x), float(xy[1]) - float(route_y))
                for route_x, route_y in route_xy
            )
            scored.append((dist_m, candidate))
        if not scored:
            return None
        best_dist_m, best_candidate = min(scored, key=lambda item: item[0])
        if float(best_dist_m) <= 5.0:
            return best_candidate
        return None

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

            route_points = self._active_global_route_points()
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

            if destination_state is not None and len(destination_state) >= 2:
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
        """Return the active Planning Module global route polyline."""

        latest_update = dict(getattr(self, "_latest_opencda_update", {}) or {})
        ego_transform = latest_update.get("ego_transform")
        if ego_transform is not None:
            loc = ego_transform.location
            return self.route_manager.route_points(
                x_m=float(loc.x),
                y_m=float(loc.y),
                query_key=f"vehicle_{int(getattr(self.vehicle_manager.vehicle, 'id', 0))}_polyline",
            )
        route_points = self.route_manager.route_points()
        if route_points:
            return route_points

        summary = None
        try:
            summary = self._active_route_summary
        except Exception:
            summary = None
        route_waypoints = list(getattr(summary, "route_waypoints", []) or [])
        points: list[list[float]] = []
        for index, raw_point in enumerate(route_waypoints):
            try:
                x_m = float(raw_point[0])
                y_m = float(raw_point[1])
                z_m = float(raw_point[2]) if len(raw_point) >= 3 else 0.0
            except Exception:
                continue
            if index < len(route_waypoints) - 1:
                try:
                    nx_m = float(route_waypoints[index + 1][0])
                    ny_m = float(route_waypoints[index + 1][1])
                    heading_rad = math.atan2(ny_m - y_m, nx_m - x_m)
                except Exception:
                    heading_rad = points[-1][3] if points else 0.0
            else:
                heading_rad = points[-1][3] if points else 0.0
            if points and math.hypot(
                float(points[-1][0]) - float(x_m),
                float(points[-1][1]) - float(y_m),
            ) < 1.0e-6:
                continue
            points.append([
                float(x_m),
                float(y_m),
                float(z_m),
                float(heading_rad),
            ])
        return points

    def _map_waypoint_from_location(self, location: carla.Location):
        if self.map_planner is None:
            return None
        point = {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }
        get_waypoint = getattr(self.map_planner, "get_waypoint", None)
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

    @staticmethod
    def _location_to_point(location: Any) -> dict[str, float]:
        return {
            "x": float(getattr(location, "x", 0.0)),
            "y": float(getattr(location, "y", 0.0)),
            "z": float(getattr(location, "z", 0.0)),
        }

    @staticmethod
    def _waypoint_xy_heading(waypoint):
        transform = getattr(waypoint, "transform", None)
        location = getattr(transform, "location", None)
        rotation = getattr(transform, "rotation", None)
        if location is not None:
            heading_rad = math.radians(float(getattr(rotation, "yaw", 0.0)))
            return float(location.x), float(location.y), float(heading_rad)
        position = getattr(waypoint, "position", None)
        if isinstance(position, Mapping):
            return (
                float(position["x"]),
                float(position["y"]),
                float(getattr(waypoint, "heading_rad", 0.0) or 0.0),
            )
        return None

    @staticmethod
    def _waypoint_lane_width(waypoint) -> float:
        for attr_name in ("lane_width_m", "lane_width"):
            value = getattr(waypoint, attr_name, None)
            if value is not None:
                try:
                    return max(0.1, float(value))
                except Exception:
                    pass
        return 3.5

    def _front_gap_m(
        self,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        object_snapshots: Sequence[Mapping[str, Any]],
    ) -> Optional[float]:
        cos_h = math.cos(ego_yaw_rad)
        sin_h = math.sin(ego_yaw_rad)
        best_gap = None
        for snapshot in object_snapshots:
            dx = float(snapshot.get("x", 0.0)) - float(ego_location.x)
            dy = float(snapshot.get("y", 0.0)) - float(ego_location.y)
            longitudinal = dx * cos_h + dy * sin_h
            lateral = -dx * sin_h + dy * cos_h
            if longitudinal <= 0.0 or abs(lateral) > 2.5:
                continue
            best_gap = longitudinal if best_gap is None else min(best_gap, longitudinal)
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

    def _control_from_mpc(self, acceleration_mps2: float, steering_angle_rad: float) -> carla.VehicleControl:
        max_accel = max(1e-6, float(self.mpc.constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        throttle = min(1.0, max(0.0, float(acceleration_mps2) / max_accel))
        brake = min(1.0, max(0.0, -float(acceleration_mps2) / max_brake))
        steer = min(1.0, max(-1.0, float(steering_angle_rad) / max_steer))
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)

    def _emergency_stop_control(self) -> carla.VehicleControl:
        self._last_accel_mps2 = float(getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0))
        self._last_steer_rad = 0.0
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)

    def _apply_control_safety_guards(
        self,
        *,
        control: carla.VehicleControl,
        accel_mps2: float,
        steer_rad: float,
        ego_transform: carla.Transform,
        ego_speed_mps: float,
        speed_ref_mps: float,
        destination_state: Sequence[float],
        destination_lateral_m: float,
        stop_goal_active: bool,
        behavior_decision: str,
        behavior_fsm_state: str,
    ) -> tuple[carla.VehicleControl, float, float, str]:
        """Apply last-mile control guards before sending commands to CARLA.

        These guards do not change the behavior decision or MPC reference. They
        only prevent two observed failure modes in the OpenCDA bridge:
        persistent speed overshoot and saturated low-speed steering while the
        planner is nominally in lane-follow.
        """

        if bool(stop_goal_active):
            return control, float(accel_mps2), float(steer_rad), ""

        max_brake_accel = max(1.0e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        max_steer = max(1.0e-6, float(self.mpc.constraints.max_steer_rad))

        if bool(self.overspeed_guard_enabled) and float(speed_ref_mps) > 0.1:
            overspeed_mps = float(ego_speed_mps) - float(speed_ref_mps)
            if overspeed_mps > float(self.overspeed_margin_mps):
                brake = min(
                    float(self.overspeed_max_brake),
                    max(
                        float(self.overspeed_min_brake),
                        float(self.overspeed_min_brake)
                        + float(self.overspeed_brake_gain)
                        * (overspeed_mps - float(self.overspeed_margin_mps)),
                    ),
                )
                guarded_steer_rad = min(
                    max_steer,
                    max(-max_steer, float(steer_rad)),
                )
                guarded_control = self._control_from_mpc(
                    -float(brake) * max_brake_accel,
                    guarded_steer_rad,
                )
                return (
                    guarded_control,
                    -float(brake) * max_brake_accel,
                    float(guarded_steer_rad),
                    "overspeed_guard",
                )

        normalized_behavior = str(behavior_decision or "").strip().lower()
        normalized_fsm = str(behavior_fsm_state or "").strip().upper()
        lane_follow_like = (
            normalized_behavior == "lane_follow"
            and normalized_fsm in {"", "IDLE", "LANE_KEEP"}
        )
        if (
            bool(self.low_speed_lateral_recovery_enabled)
            and bool(lane_follow_like)
            and float(ego_speed_mps) < float(self.low_speed_lateral_recovery_speed_mps)
            and abs(float(destination_lateral_m))
            > float(self.low_speed_lateral_recovery_threshold_m)
            and len(destination_state) >= 2
        ):
            dx = float(destination_state[0]) - float(ego_transform.location.x)
            dy = float(destination_state[1]) - float(ego_transform.location.y)
            target_yaw = math.atan2(dy, dx)
            yaw_error = self._wrap_angle(
                target_yaw - math.radians(float(ego_transform.rotation.yaw))
            )
            max_recovery_steer = min(
                max_steer,
                max(0.01, float(self.low_speed_lateral_recovery_max_steer_rad)),
            )
            guarded_steer_rad = min(
                max_recovery_steer,
                max(-max_recovery_steer, 0.45 * float(yaw_error)),
            )
            target_speed = max(0.1, float(self.low_speed_lateral_recovery_target_speed_mps))
            accel = min(
                float(self.mpc.constraints.max_acceleration_mps2),
                max(
                    0.15,
                    0.55 * (target_speed - float(ego_speed_mps)),
                ),
            )
            return (
                self._control_from_mpc(float(accel), float(guarded_steer_rad)),
                float(accel),
                float(guarded_steer_rad),
                "low_speed_lateral_recovery",
            )

        return control, float(accel_mps2), float(steer_rad), ""

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

    @staticmethod
    def _wrap_angle_static(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


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
    if normalized_lc_state.startswith("EXECUTE_LANE_CHANGE") or normalized_behavior in {
        "lane_change_left",
        "lane_change_right",
    }:
        return "execute_lane_change"
    if normalized_mode == "INTERSECTION" and normalized_maneuver in {"left", "right"}:
        return "intersection_turn"
    return "lane_follow"


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
