"""Geometric lane-change phase transition and completion owner."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

from .reference_line_provider import LANE_CHANGE
from .stage_contracts import LaneChangeContract


class LaneChangeLifecycleStage:
    """Advance crossing -> stabilization -> completion from geometry only."""

    def __init__(
        self, *, provider: Any, maneuver_manager: Any, reference_generator: Any,
        route_manager: Any, mpc: Any, control_buffer: Any,
        config: Mapping[str, Any], target_speed_mps: float, map_epoch: str,
        vehicle_extent: Callable[[], Any],
    ) -> None:
        self._provider = provider
        self._maneuver = maneuver_manager
        self._generator = reference_generator
        self._route = route_manager
        self._mpc = mpc
        self._buffer = control_buffer
        self._config = config
        self._target_speed_mps = float(target_speed_mps)
        self._map_epoch = str(map_epoch)
        self._vehicle_extent = vehicle_extent

    def reset_reference(self) -> None:
        self._maneuver.lane_change.reset()
        self._provider.release(LANE_CHANGE, event="reset")

    def release_completed(
        self, *, current_lane_id: int, ego_location: Any, ego_yaw_rad: float,
    ) -> str:
        snapshot = self._provider.snapshot(LANE_CHANGE)
        if not snapshot.mutable_samples():
            return ""
        lifecycle = self._maneuver.lane_change
        target_lane_id = int(lifecycle.target_lane_id)
        contract = LaneChangeContract.from_config(self._config)
        completion_reference = [dict(x) for x in lifecycle.completion_reference]
        if not completion_reference:
            completion_reference = snapshot.mutable_samples()
        alignment = self._provider.lane_change_completion_alignment(
            reference_samples=completion_reference,
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
        )
        target_sample = alignment.mutable_target_sample() if alignment.available else None
        lateral_error = float(alignment.lateral_error_m)
        heading_error = float(alignment.heading_error_rad)
        lane_width_m = max(0.1, float(getattr(self._mpc, "lane_width_m", 3.5)))
        geometry_ready = contract.stabilization_handoff_ready(
            progress=float(lifecycle.progress),
            lateral_error_m=lateral_error,
            heading_error_rad=heading_error,
            lane_width_m=lane_width_m,
        )
        handoff = self._maneuver.lane_change_handoff_transition(
            geometry_ready=bool(
                float(lifecycle.progress) >= float(contract.min_progress)
                and geometry_ready
            )
        )
        if handoff.action == "start_stabilization":
            reason = self._start_stabilization(
                ego_location=ego_location, ego_yaw_rad=float(ego_yaw_rad)
            )
            if reason.startswith("target_lane_stabilization_started"):
                return reason
            self._maneuver.complete_lane_change("stabilization_handoff_unavailable")
            self.reset_reference()
            return (
                "lane_change_stabilization_unavailable_to_lane_follow_recovery:"
                f"target_lane={target_lane_id}:{reason}"
            )

        phase = str(lifecycle.phase)
        if phase == "target_lane_stabilization":
            transition = self._maneuver.tick_lane_change_stabilization(
                timeout_frames=max(1, int(self._config.get(
                    "lane_change_stabilization_timeout_frames", 100
                )))
            )
            if transition.action == "abandon":
                self.reset_reference()
                return (
                    "lane_change_stabilization_timeout_to_lane_follow_recovery:"
                    f"target_lane={target_lane_id}:map_lane={int(current_lane_id)}"
                )
        progress = float(lifecycle.progress)
        if phase == "target_lane_stabilization" and math.isfinite(lateral_error):
            progress = min(1.0, max(0.0, 1.0 - abs(lateral_error) / lane_width_m))
        extent = self._vehicle_extent()
        occupancy = self._generator.lane_corridor_occupancy(
            x_m=float(ego_location.x),
            y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            ego_half_width_m=float(getattr(
                extent, "y", self._config.get("reference_vehicle_half_width_m", 1.0)
            )),
            ego_half_length_m=float(getattr(
                extent, "x", self._config.get("reference_vehicle_half_length_m", 2.4)
            )),
            safety_margin_m=float(self._config.get(
                "lane_change_completion_footprint_margin_m", 0.05
            )),
            corridor_sample=target_sample,
            prefer_tracking_point=True,
        )
        completion = self._maneuver.evaluate_lane_change_completion(
            alignment=alignment,
            progress=progress,
            target_lane_matches=int(current_lane_id) == target_lane_id,
            footprint_clearance_m=(
                float(occupancy.footprint_clearance_m)
                if bool(occupancy.valid) else float("-inf")
            ),
            contract=contract,
        )
        transition = self._maneuver.accept_evaluated_lane_change_completion(
            completion=completion,
            contract=contract,
            stabilization_geometry_ready=bool(geometry_ready),
            stabilization_lateral_error_m=lateral_error,
            stabilization_heading_error_rad=heading_error,
        )
        if transition.action != "complete":
            return ""
        self._maneuver.complete_lane_change(str(transition.reason))
        self.reset_reference()
        clear_seed = getattr(self._mpc, "clear_solution_memory", None)
        if not callable(clear_seed):
            clear_seed = getattr(self._mpc, "clear_previous_solution_seed", None)
        if callable(clear_seed):
            clear_seed()
        reset = getattr(self._buffer, "reset", None)
        if callable(reset):
            reset(reason="lane_change_geometrically_complete")
        return (
            "lane_change_commitment_released:"
            f"target_lane={target_lane_id}:map_lane={int(current_lane_id)}:"
            f"progress={float(completion.progress):.3f}:"
            f"lateral_error={float(completion.target_lateral_error_m):.3f}:"
            "heading_error_deg="
            f"{math.degrees(float(completion.target_heading_error_rad)):.2f}"
        )

    def _start_stabilization(self, *, ego_location: Any, ego_yaw_rad: float) -> str:
        lifecycle = self._maneuver.lane_change
        target_lane_id = int(lifecycle.target_lane_id)
        committed_speed = float(lifecycle.target_speed_mps or self._target_speed_mps)
        speed_mps = max(0.5, min(self._target_speed_mps, committed_speed))
        step_m = max(0.10, float(self._mpc.dt_s) * speed_mps)
        transition_arc_m = max(step_m, float(self._config.get(
            "lane_change_to_turn_reference_transition_arc_m", 10.0
        )))
        master_steps = max(
            int(self._mpc.horizon_steps),
            int(math.ceil(transition_arc_m / step_m)) + 1,
        )
        route_points_fn = getattr(self._route, "geometry_route_points", None)
        route_points = (
            route_points_fn(
                x_m=float(ego_location.x),
                y_m=float(ego_location.y),
                query_key="lane_change_to_turn_transition",
            )
            if callable(route_points_fn) else []
        )
        reference, curvature_reason = self._provider.target_lane_stabilization_master(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            target_lane_id=target_lane_id,
            horizon_steps=master_steps,
            step_distance_m=step_m,
            route_points=route_points,
            target_speed_mps=speed_mps,
            max_curvature_1pm=float(self._config.get(
                "reference_vehicle_max_curvature_1pm", 0.20
            )),
        )
        if len(reference) < int(self._mpc.horizon_steps):
            return f"target_lane_stabilization_failed:short_reference={len(reference)}"
        installed, reason = self._provider.install(
            LANE_CHANGE,
            reference,
            route_revision=str(getattr(self._route, "route_revision", "")),
            map_epoch=self._map_epoch,
            event="phase_transition",
            source_lane_id=int(lifecycle.source_lane_id),
            target_lane_id=target_lane_id,
            build_reason="target_lane_stabilization_reference",
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
        )
        if not installed:
            return "target_lane_stabilization_failed:" + str(reason)
        self._maneuver.begin_lane_change_stabilization()
        self._maneuver.update_lane_change_target_speed(speed_mps)
        self._maneuver.set_lane_change_transition_arc(
            arc_m=transition_arc_m, step_m=step_m
        )
        return (
            "target_lane_stabilization_started:"
            f"target_lane={target_lane_id}:N={len(reference)}:"
            f"transition_arc_m={transition_arc_m:.2f}:speed={speed_mps:.2f}:"
            f"curvature_conditioning={str(curvature_reason or 'not_required')}"
        )
