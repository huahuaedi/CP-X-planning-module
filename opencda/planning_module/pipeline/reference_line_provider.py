"""Single lifecycle owner for all planner reference-line modes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from .behavior_decision import BehaviorDecision

from .stable_reference_line_provider import (
    StableReferenceLineProvider,
    StableReferenceWindow,
)


LANE_FOLLOW = "lane_follow"
LANE_CHANGE = "lane_change"
CONNECTOR = "connector"
TURN = "turn"
POST_TURN = "post_turn"

_MODES = {LANE_FOLLOW, LANE_CHANGE, CONNECTOR, TURN, POST_TURN}
_INSTALL_EVENTS = {
    "initial_route",
    "route_changed",
    "map_epoch_changed",
    "maneuver_started",
    "phase_transition",
    "same_route_extension",
}
_RELEASE_EVENTS = {
    "route_changed",
    "map_epoch_changed",
    "maneuver_completed",
    "maneuver_abandoned",
    "phase_transition",
    "reset",
}


@dataclass(frozen=True)
class ReferenceLineSnapshot:
    mode: str
    active: bool = False
    route_revision: str = ""
    map_epoch: str = ""
    geometry_revision: int = 0
    samples: tuple = ()
    activation_s_m: float = 0.0
    progress_s_m: float = 0.0
    source_lane_id: int = 0
    target_lane_id: int = 0
    maneuver_direction: str = ""
    install_event: str = ""
    build_reason: str = ""
    last_validation_failure: str = ""

    @property
    def travelled_s_m(self) -> float:
        return max(0.0, float(self.progress_s_m) - float(self.activation_s_m))

    def mutable_samples(self):
        return [dict(sample) for sample in self.samples]

    def trace_fields(self):
        return {
            "persistent_reference_active": bool(self.active),
            "persistent_reference_state": (
                "DEGRADED" if self.last_validation_failure
                else "ACTIVE" if self.active else "UNINITIALIZED"
            ),
            "persistent_reference_route_revision": str(self.route_revision),
            "persistent_reference_map_epoch": str(self.map_epoch),
            "persistent_reference_geometry_revision": int(self.geometry_revision),
            "persistent_reference_progress_s_m": float(self.progress_s_m),
            "persistent_reference_build_reason": str(self.build_reason),
            "persistent_reference_last_rejected_trigger": (
                "validation_failure:" + str(self.last_validation_failure)
                if self.last_validation_failure else ""
            ),
        }


@dataclass(frozen=True)
class ReferenceLineRequest:
    """Complete typed input contract for one reference publication."""

    local_map: Any
    route_cursor: Any
    behavior: BehaviorDecision
    route_revision: str
    map_epoch: str
    ego_x_m: float
    ego_y_m: float


@dataclass(frozen=True)
class ReferenceLineResult:
    mode: str
    samples: tuple
    accepted: bool
    reason: str
    geometry_revision: int

    def mutable_samples(self):
        return [dict(sample) for sample in self.samples]


@dataclass(frozen=True)
class LaneChangeWindowResult:
    samples: tuple
    projection_s_m: float
    start_s_m: float
    master_index: int
    reason: str

    def mutable_samples(self):
        return [dict(sample) for sample in self.samples]


@dataclass(frozen=True)
class LaneChangeAlignment:
    """Ego alignment to the immutable target-corridor completion geometry."""

    available: bool
    lateral_error_m: float
    heading_error_rad: float
    target_sample: Mapping[str, object]

    def mutable_target_sample(self):
        return dict(self.target_sample)


@dataclass(frozen=True)
class BehaviorReferenceResult:
    """Provider-owned output of the legacy reference geometry builder."""

    samples: tuple
    destination_state: tuple
    reference_freeze_count: int
    diagnostics: Mapping[str, object]
    fallback_reason: str

    def mutable_samples(self):
        return [dict(sample) for sample in self.samples]

    def mutable_destination_state(self):
        return list(self.destination_state)


@dataclass(frozen=True)
class TurnReferenceRequest:
    local_map: Any
    config: Mapping[str, object]
    horizon_steps: int
    dt_s: float
    ego_location: Any
    ego_yaw_rad: float
    current_state: Sequence[float]
    current_lane_id: int
    target_lane_id: int
    target_speed_mps: float
    destination_state: Optional[Sequence[float]] = None
    lock_master: bool = False
    turn_direction: str = ""
    route_revision: str = ""
    map_epoch: str = "admap"


@dataclass(frozen=True)
class CandidateReferenceOverrideResult:
    samples: tuple
    destination_state: tuple
    diagnostics: Mapping[str, object]

    def mutable_samples(self):
        return [dict(sample) for sample in self.samples]

    def mutable_destination_state(self):
        return list(self.destination_state)


@dataclass(frozen=True)
class CandidateReferenceBuildContext:
    map_planner: Any
    local_map: Any
    planner_config: Mapping[str, object]
    ego_pose: Mapping[str, object]
    current_state: Sequence[float]
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    route_points: Sequence[Sequence[float]]
    previous_reference: Sequence[Mapping[str, object]]
    previous_target_state: Sequence[float]
    behavior_runtime_config: Mapping[str, object]
    baseline_decision: str
    baseline_target_lane_id: int
    baseline_speed_mps: float
    baseline_destination_state: Optional[Sequence[float]]
    baseline_reference: Sequence[Mapping[str, object]]
    baseline_debug: Mapping[str, object]
    current_lane_id: int
    route_optimal_lane_id: int
    route_reference_allowed: bool
    route_reference_gate_reason: str
    in_junction: bool
    next_macro_maneuver: str
    planner_mode: str
    lookahead_m: float
    horizon_steps: int
    dt_s: float
    reference_freeze_count: int
    sim_time_s: float
    stop_release_smooth_until_s: float
    authoritative_ego_waypoint: Any
    route_revision: str
    map_epoch: str
    upcoming_turn_direction: str
    upcoming_turn_distance_m: float
    lane_change_duration_s: float
    lane_change_duration_reason: str
    lane_width_m: float


class ReferenceLineProvider(StableReferenceLineProvider):
    """Own immutable masters and monotonic windows for every planning mode.

    Geometry can only be installed/released by explicit route, map, maneuver,
    or phase events. Validation/MPC failure is diagnostic and never rebuilds
    a master.
    """

    def __init__(self) -> None:
        self._builder = None
        self._snapshots = {
            mode: ReferenceLineSnapshot(mode=mode) for mode in _MODES
        }

    def attach_builder(self, builder: Any) -> None:
        self._builder = builder

    @property
    def builder(self) -> Any:
        if self._builder is None:
            raise RuntimeError("reference builder is not attached")
        return self._builder

    def snapshot(self, mode: str) -> ReferenceLineSnapshot:
        return self._snapshots[self._normalize_mode(mode)]

    def lane_change_completion_alignment(
        self, *, reference_samples, ego_x_m, ego_y_m, ego_heading_rad
    ) -> LaneChangeAlignment:
        """Measure completion alignment without changing reference ownership."""

        samples = [dict(sample) for sample in list(reference_samples or [])]
        terminal = [
            sample
            for sample in samples
            if float(sample.get("lane_change_progress", 0.0) or 0.0) >= 0.9
        ]
        # A target-corridor centerline is already completion geometry and
        # intentionally carries no scheduled Frenet progress tag.
        if not terminal:
            terminal = samples
        if not terminal:
            return LaneChangeAlignment(
                available=False,
                lateral_error_m=float("inf"),
                heading_error_rad=float("inf"),
                target_sample=MappingProxyType({}),
            )
        target = min(
            terminal,
            key=lambda sample: (
                float(sample.get("x_ref_m", sample.get("x", ego_x_m)))
                - float(ego_x_m)
            ) ** 2
            + (
                float(sample.get("y_ref_m", sample.get("y", ego_y_m)))
                - float(ego_y_m)
            ) ** 2,
        )
        target_x_m = float(target.get("x_ref_m", target.get("x", ego_x_m)))
        target_y_m = float(target.get("y_ref_m", target.get("y", ego_y_m)))
        target_heading_rad = float(target.get("heading_rad", ego_heading_rad))
        dx_m = float(ego_x_m) - target_x_m
        dy_m = float(ego_y_m) - target_y_m
        return LaneChangeAlignment(
            available=True,
            lateral_error_m=(
                -math.sin(target_heading_rad) * dx_m
                + math.cos(target_heading_rad) * dy_m
            ),
            heading_error_rad=math.atan2(
                math.sin(float(ego_heading_rad) - target_heading_rad),
                math.cos(float(ego_heading_rad) - target_heading_rad),
            ),
            target_sample=MappingProxyType(dict(target)),
        )

    def publish(
        self,
        request: ReferenceLineRequest,
        samples: Sequence[Mapping[str, object]],
        *,
        valid: bool,
        validation_reason: str = "",
        build_reason: str = "planning_reference",
    ) -> ReferenceLineResult:
        """Sole acceptance boundary for every reference delivered to MPC.

        Builders submit geometry; they cannot mutate lifecycle state or
        decide whether rejected geometry replaces a persistent master.
        """

        mode = self.mode_for(
            request.behavior,
            segment_kind=str(
                getattr(request.route_cursor, "segment_kind", "") or ""
            ),
        )
        rows = [dict(sample) for sample in list(samples or [])]
        current = self.snapshot(mode)
        if not bool(valid) or len(rows) < 2:
            self.mark_validation_failure(
                mode, str(validation_reason or "reference_invalid")
            )
            retained = current.samples if current.active else tuple(
                MappingProxyType(dict(sample)) for sample in rows
            )
            return ReferenceLineResult(
                mode=mode,
                samples=retained,
                accepted=False,
                reason=str(validation_reason or "reference_invalid"),
                geometry_revision=int(current.geometry_revision),
            )

        route_changed = bool(
            current.active
            and str(current.route_revision) != str(request.route_revision)
        )
        map_changed = bool(
            current.active and str(current.map_epoch) != str(request.map_epoch)
        )
        if not current.active:
            event = (
                "maneuver_started"
                if mode in {LANE_CHANGE, CONNECTOR, TURN, POST_TURN}
                else "initial_route"
            )
        elif route_changed:
            event = "route_changed"
        elif map_changed:
            event = "map_epoch_changed"
        else:
            # A planning tick is not a lifecycle event.  In particular, the
            # rolling local window must never increment geometry revision or
            # silently replace the persistent master.  The submitted window
            # is still returned to MPC; only route/map/maneuver transitions
            # are allowed to install a new master.
            return ReferenceLineResult(
                mode=mode,
                samples=tuple(
                    MappingProxyType(dict(sample)) for sample in rows
                ),
                accepted=True,
                reason="persistent_reference_retained",
                geometry_revision=int(current.geometry_revision),
            )
        installed, reason = self.install(
            mode,
            rows,
            route_revision=str(request.route_revision),
            map_epoch=str(request.map_epoch),
            event=event,
            source_lane_id=int(request.behavior.source_lane_id),
            target_lane_id=int(request.behavior.target_lane_id),
            maneuver_direction=str(request.behavior.direction),
            build_reason=str(build_reason),
            ego_x_m=float(request.ego_x_m),
            ego_y_m=float(request.ego_y_m),
        )
        snapshot = self.snapshot(mode)
        return ReferenceLineResult(
            mode=mode,
            samples=tuple(
                MappingProxyType(dict(sample)) for sample in rows
            ),
            accepted=bool(installed or snapshot.active),
            reason=str(reason),
            geometry_revision=int(snapshot.geometry_revision),
        )

    def build_behavior_reference(
        self,
        *,
        map_planner,
        ego_pose,
        ego_state,
        route_points,
        previous_reference,
        previous_target_state,
        behavior_runtime_config,
        decision,
        lane_change_state,
        target_lane_id,
        current_lane_id,
        route_optimal_lane_id,
        route_reference_allowed,
        route_reference_gate_reason,
        in_junction,
        next_macro_maneuver,
        planner_mode,
        lookahead_m,
        target_speed_mps,
        ego_speed_mps,
        horizon_steps,
        dt_s,
        reference_freeze_count,
        sim_time_s,
        stop_release_smooth_until_s,
        authoritative_ego_waypoint,
        lane_reference_step_distance_m=None,
    ) -> BehaviorReferenceResult:
        """Build one behavior reference through the provider-owned entry.

        This is the compatibility boundary around the existing geometric
        builder.  Callers no longer construct its context or invoke it
        directly, which leaves one place to replace when the AD-map-native
        implementation fully supersedes it.
        """

        from opencda.planning_module.behavior_planner import (
            MpcReferenceGenerationContext,
            compute_temp_destination,
            generate_mpc_reference,
            select_reference_intent,
        )

        prior = list(previous_target_state or [])
        destination = compute_temp_destination(
            map_planner=map_planner,
            ego_pose=ego_pose,
            target_lane_id=int(target_lane_id),
            decision=str(decision),
            lookahead_m=float(lookahead_m),
            target_v_mps=float(target_speed_mps),
            global_route_points=list(route_points or []),
            mode_reference_xy=(
                (float(prior[0]), float(prior[1])) if len(prior) >= 2 else None
            ),
            prev_mode=(float(prior[5]) if len(prior) >= 6 else None),
            prev_road_id=(int(prior[6]) if len(prior) >= 7 else None),
            prev_entered_intersection=(
                bool(float(prior[7]) > 0.5) if len(prior) >= 8 else False
            ),
            next_macro_maneuver=str(next_macro_maneuver),
            mode_override=str(planner_mode),
            follow_global_route_lane=bool(route_reference_allowed and in_junction),
        )
        intent = select_reference_intent(
            behavior_decision=str(decision),
            planner_fsm_state=str(lane_change_state),
            ego_in_junction=bool(in_junction),
            reference_target_lane_id=int(target_lane_id),
            current_lane_id=int(current_lane_id),
            route_optimal_lane_id=int(route_optimal_lane_id),
            global_route_reference_allowed=bool(route_reference_allowed),
            traffic_control_lane_lock_active=False,
        )
        context = MpcReferenceGenerationContext(
            map_planner=map_planner,
            ego_pose=ego_pose,
            ego_state=ego_state,
            active_global_route_points=list(route_points or []),
            previous_lane_center_reference=[
                dict(sample) for sample in list(previous_reference or [])
            ],
            behavior_runtime_cfg=behavior_runtime_config,
            reference_intent=intent,
            current_applied_behavior=str(decision),
            cached_planner_lc_state=str(lane_change_state),
            reference_target_lane_id=int(target_lane_id),
            current_lane_id=int(current_lane_id),
            global_route_reference_allowed=bool(route_reference_allowed),
            global_route_reference_gate_reason=str(route_reference_gate_reason),
            should_follow_global_route_lane_for_reference=bool(
                intent.follow_global_route_lane
            ),
            traffic_control_lane_lock_active=False,
            final_goal_stop_active=False,
            stop_target_state=None,
            follow_target_state=None,
            current_temp_reference_xy=(float(destination[0]), float(destination[1])),
            current_temp_mode_value=(
                float(destination[5]) if len(destination) >= 6 else 0.0
            ),
            current_temp_road_id=(
                int(destination[6]) if len(destination) >= 7 else None
            ),
            current_temp_entered_intersection=(
                bool(float(destination[7]) > 0.5) if len(destination) >= 8 else False
            ),
            active_reference_maneuver=str(next_macro_maneuver),
            current_temp_mode_str=str(planner_mode),
            lane_reference_speed_mps=max(
                1.0, float(ego_speed_mps), abs(float(target_speed_mps))
            ),
            lane_reference_step_distance_m=(
                max(0.05, float(lane_reference_step_distance_m))
                if lane_reference_step_distance_m is not None
                else max(
                    0.5,
                    float(dt_s)
                    * max(
                        1.0, float(ego_speed_mps), abs(float(target_speed_mps))
                    ),
                )
            ),
            mpc_horizon_steps=int(horizon_steps),
            mpc_dt_s=float(dt_s),
            temporary_destination_state=destination,
            lane_reference_freeze_count=int(reference_freeze_count),
            sim_time_s=float(sim_time_s),
            stop_release_temp_smooth_until_sim_time_s=float(
                stop_release_smooth_until_s
            ),
            authoritative_ego_waypoint=authoritative_ego_waypoint,
        )
        output = generate_mpc_reference(context)
        samples = tuple(
            MappingProxyType(dict(sample))
            for sample in list(output.local_lane_center_reference or [])
        )
        resolved_destination = tuple(
            output.temporary_destination_state or destination
        )
        diagnostics = MappingProxyType(
            dict(output.mpc_reference_result.trace.as_trace_fields())
        )
        return BehaviorReferenceResult(
            samples=samples,
            destination_state=resolved_destination,
            reference_freeze_count=int(output.lane_reference_freeze_count),
            diagnostics=diagnostics,
            fallback_reason=str(output.last_reference_fallback_reason),
        )

    def turn_reference(self, request: TurnReferenceRequest):
        """Produce and window the sole immutable AD-map turn reference."""
        local_map = request.local_map
        config = request.config
        horizon_steps = int(request.horizon_steps)
        dt_s = float(request.dt_s)
        ego_location = request.ego_location
        ego_yaw_rad = float(request.ego_yaw_rad)
        current_state = request.current_state
        current_lane_id = int(request.current_lane_id)
        target_lane_id = int(request.target_lane_id)
        target_speed_mps = float(request.target_speed_mps)
        destination_state = request.destination_state
        lock_master = bool(request.lock_master)
        turn_direction = str(request.turn_direction)
        route_revision = str(request.route_revision)
        map_epoch = str(request.map_epoch)
        turn_speed_mps = min(
            max(0.4, float(target_speed_mps)),
            float(config.get("waypoint_turn_speed_cap_mps", 2.2)),
        )
        step_distance_m = max(
            float(config.get("waypoint_turn_min_step_m", 0.35)),
            float(dt_s) * max(0.8, float(turn_speed_mps)),
        )
        reason = ""
        master = self.snapshot(TURN).mutable_samples()
        if not master:
            # GlobalRoute owns topology only. TURN geometry comes from the
            # topology-ordered AD-map lane and connector centre lines in the
            # local snapshot. The former route-polyline path was a second
            # geometry owner and exposed junction node discontinuities as a
            # lateral jump at PREPARE->TURN.
            master, reason = self.reference_from_local_map(
                local_map,
                start_lane_id=int(current_lane_id),
                target_speed_mps=float(turn_speed_mps),
                maximum_join_distance_m=float(
                    config.get("local_map_maximum_join_distance_m", 5.0)
                ),
            )
            if bool(lock_master) and master:
                installed, install_reason = self.install(
                    TURN, master,
                    route_revision=str(route_revision),
                    map_epoch=str(map_epoch or "admap"),
                    event="maneuver_started",
                    source_lane_id=int(current_lane_id),
                    target_lane_id=int(target_lane_id),
                    maneuver_direction=str(turn_direction or "").strip().lower(),
                    build_reason=str(reason),
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                )
                suffix = "turn_master_locked" if installed else str(install_reason)
                reason = ";".join(
                    item for item in (str(reason), suffix) if item
                )
        if not master:
            return [], list(destination_state or []), str(reason)
        if self.snapshot(TURN).active:
            window = self.window(
                TURN,
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
                first_forward_m=float(config.get(
                    "reference_contract_intersection_turn_min_first_forward_m",
                    0.2,
                )),
                spacing_m=float(step_distance_m),
                count=int(horizon_steps),
                max_projection_advance_m=max(2.0, 2.0 * float(step_distance_m)),
            )
            reference = [dict(sample) for sample in window.samples]
            reason = ";".join(
                item for item in (
                    str(reason), "turn_master_window:" + str(window.reason)
                ) if item
            )
        else:
            reference = [dict(sample) for sample in master]
        for sample in reference:
            sample["v_ref_mps"] = float(turn_speed_mps)
            sample["speed_ref_mps"] = float(turn_speed_mps)
            sample["speed_mps"] = float(turn_speed_mps)
        reference, curvature_reason = self.builder.curvature_feasible_turn_samples(
            reference_samples=reference,
            ego_location=ego_location,
            ego_heading_rad=float(ego_yaw_rad),
            max_curvature_1pm=float(
                config.get("reference_vehicle_max_curvature_1pm", 0.20)
            ),
        )
        if curvature_reason:
            reason = ";".join(
                item for item in (str(reason), str(curvature_reason)) if item
            )
        from opencda.planning_module.behavior_planner.reference_pipeline import (
            lane_center_destination_from_reference_arc_length,
        )
        seed = list(destination_state or [])
        if len(seed) < 5:
            seed = [
                float(current_state[0]), float(current_state[1]),
                float(turn_speed_mps), float(current_state[3]),
                int(target_lane_id or current_lane_id),
            ]
        seed[2] = float(turn_speed_mps)
        destination = lane_center_destination_from_reference_arc_length(
            destination_state=seed,
            lane_center_reference=reference,
            target_arc_length_m=float(
                config.get("waypoint_turn_destination_arc_m", 4.5)
            ),
        ) or seed
        return [dict(sample) for sample in reference], list(destination), str(reason)

    def intersection_turn_candidate(
        self, request: TurnReferenceRequest, *, decision: str
    ) -> CandidateReferenceOverrideResult:
        """Build the complete candidate-stage view of an AD-map turn."""

        reference, destination, reason = self.turn_reference(request)
        diagnostics = {"route_turn_reference_reason": str(reason)}
        if reference:
            first = reference[0]
            dx_m = float(first.get("x_ref_m", first.get("x", 0.0))) - float(
                request.ego_location.x
            )
            dy_m = float(first.get("y_ref_m", first.get("y", 0.0))) - float(
                request.ego_location.y
            )
            cos_h = math.cos(float(request.ego_yaw_rad))
            sin_h = math.sin(float(request.ego_yaw_rad))
            diagnostics.update({
                "reference_pipeline_stage": "waypoint_turn",
                "reference_pipeline_intent": str(decision),
                "reference_pipeline_intent_mode": "intersection_turn",
                "reference_pipeline_follow_global_route_lane": 1,
                "reference_source": "admap_waypoint_turn",
                "fallback_reason": "",
                "route_turn_raw_first_forward_m": float(
                    cos_h * dx_m + sin_h * dy_m
                ),
                "route_turn_raw_first_lateral_m": float(
                    -sin_h * dx_m + cos_h * dy_m
                ),
            })
        return CandidateReferenceOverrideResult(
            samples=tuple(
                MappingProxyType(dict(sample)) for sample in reference
            ),
            destination_state=tuple(destination),
            diagnostics=MappingProxyType(diagnostics),
        )

    def preturn_candidate(
        self,
        request: TurnReferenceRequest,
        *,
        upcoming_turn_distance_m: float,
        first_forward_m: float,
    ) -> CandidateReferenceOverrideResult:
        """Select current-lane or locked connector geometry before a turn."""

        reference, lane_reason = self.preturn_lane_reference(
            request.local_map,
            lane_id=int(request.current_lane_id),
            ego_x_m=float(request.ego_location.x),
            ego_y_m=float(request.ego_location.y),
            target_speed_mps=float(request.target_speed_mps),
            first_forward_m=float(first_forward_m),
            spacing_m=float(first_forward_m),
            horizon_steps=int(request.horizon_steps),
        )
        diagnostics = {"preturn_lane_reference_reason": str(lane_reason)}
        if reference:
            first = reference[0]
            dx_m = float(first.get("x_ref_m", first.get("x", 0.0))) - float(
                request.ego_location.x
            )
            dy_m = float(first.get("y_ref_m", first.get("y", 0.0))) - float(
                request.ego_location.y
            )
            diagnostics.update({
                "preturn_raw_first_lateral_m": float(
                    -math.sin(float(request.ego_yaw_rad)) * dx_m
                    + math.cos(float(request.ego_yaw_rad)) * dy_m
                ),
                "reference_source": "admap_current_lane_center_preturn",
            })

        transition_arc_m = max(
            0.0,
            float(
                request.config.get(
                    "lane_follow_to_turn_reference_transition_arc_m", 12.0
                )
            ),
        )
        direction = str(request.turn_direction or "").strip().lower()
        transition_reference = []
        transition_reason = ""
        if float(upcoming_turn_distance_m) <= float(transition_arc_m):
            snapshot = self.snapshot(TURN)
            if snapshot.active and str(snapshot.maneuver_direction) != direction:
                self.release(TURN, event="reset")
            transition_reference, _unused_destination, transition_reason = (
                self.turn_reference(replace(
                    request,
                    lock_master=True,
                    turn_direction=direction,
                ))
            )
            # PREPARE_TURN remains longitudinally owned by SpeedPlanner.
            for sample in transition_reference:
                sample["v_ref_mps"] = float(request.target_speed_mps)
                sample["speed_ref_mps"] = float(request.target_speed_mps)
                sample["speed_mps"] = float(request.target_speed_mps)
        if transition_reference:
            reference = [dict(sample) for sample in transition_reference]
            diagnostics.update({
                "reference_source": "admap_preturn_connector_transition",
                "route_turn_reference_reason": str(transition_reason),
                "lane_follow_turn_geometry_hold_reason": (
                    "lane_follow_to_turn_locked_master_window"
                ),
            })
        return CandidateReferenceOverrideResult(
            samples=tuple(
                MappingProxyType(dict(sample)) for sample in reference
            ),
            destination_state=tuple(request.destination_state or ()),
            diagnostics=MappingProxyType(diagnostics),
        )

    def route_lane_change_target_candidate(
        self,
        *,
        local_map,
        target_lane_id,
        target_speed_mps,
        ego_x_m,
        ego_y_m,
        spacing_m,
        geometry_length_m,
        horizon_steps,
        destination_state,
    ) -> CandidateReferenceOverrideResult:
        """Resolve one topology-required target corridor from LocalMapSnapshot."""

        spacing_m = max(0.05, float(spacing_m))
        count = max(
            int(horizon_steps),
            int(math.ceil(float(geometry_length_m) / spacing_m))
            + int(horizon_steps),
        )
        master, reason = self.lane_change_target_reference(
            local_map,
            target_lane_id=int(target_lane_id),
            target_speed_mps=float(target_speed_mps),
        )
        window = (
            self.window_from_reference(
                master,
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                lower_s_m=0.0,
                first_forward_m=spacing_m,
                spacing_m=spacing_m,
                count=int(count),
            )
            if master
            else None
        )
        reference = (
            [dict(sample) for sample in window.samples]
            if window is not None
            else []
        )
        diagnostics = {
            "admap_target_lane_resolved": bool(reference),
            "admap_target_lane_reason": str(reason),
        }
        if reference:
            diagnostics["reference_source"] = (
                "local_map_target_corridor_center"
            )
        return CandidateReferenceOverrideResult(
            samples=tuple(
                MappingProxyType(dict(sample)) for sample in reference
            ),
            destination_state=tuple(destination_state or ()),
            diagnostics=MappingProxyType(diagnostics),
        )

    def lane_change_candidate(
        self,
        *,
        local_map,
        current_state,
        current_lane_id,
        target_lane_id,
        target_reference,
        source_reference,
        target_speed_mps,
        geometry_speed_mps,
        geometry_length_m,
        transition_duration_s,
        spacing_m,
        horizon_steps,
        lane_width_m,
        destination_state,
        trajectory_variant,
        duration_s,
        duration_reason,
        authorization_source,
        operational_curvature_limit_1pm,
    ) -> CandidateReferenceOverrideResult:
        """Generate and describe one Frenet lane-change candidate."""

        reference, geometry_debug = self.lane_change_nominal(
            local_map,
            ego_x_m=float(current_state[0]),
            ego_y_m=float(current_state[1]),
            ego_heading_rad=float(current_state[3]),
            current_lane_id=int(current_lane_id),
            target_lane_id=int(target_lane_id),
            target_reference=target_reference,
            fallback_source_reference=source_reference,
            target_speed_mps=float(target_speed_mps),
            geometry_speed_mps=float(geometry_speed_mps),
            geometry_length_m=float(geometry_length_m),
            transition_duration_s=float(transition_duration_s),
            spacing_m=float(spacing_m),
            horizon_steps=int(horizon_steps),
            lane_width_m=float(lane_width_m),
        )
        destination = list(destination_state or [])
        if reference and len(destination) >= 4:
            terminal = reference[-1]
            destination[0] = float(
                terminal.get("x_ref_m", terminal.get("x", destination[0]))
            )
            destination[1] = float(
                terminal.get("y_ref_m", terminal.get("y", destination[1]))
            )
            destination[2] = float(target_speed_mps)
            destination[3] = float(
                terminal.get("heading_rad", destination[3])
            )
            if len(destination) >= 5:
                destination[4] = int(target_lane_id)
        diagnostics = {
            "lane_change_source_corridor_reason": str(
                geometry_debug.get("source_corridor_reason", "")
            ),
            "lane_change_trajectory_variant": str(trajectory_variant),
            "lane_change_duration_s": float(duration_s),
            "lane_change_duration_comfort_reason": str(duration_reason),
            "lane_change_reference_profile": "frenet_quintic_d_of_s",
            "lane_change_geometry_speed_mps": float(geometry_speed_mps),
            "lane_change_geometry_length_m": float(geometry_length_m),
            "lane_change_geometry_step_m": float(spacing_m),
            "lane_change_operational_curvature_limit_1pm": float(
                operational_curvature_limit_1pm
            ),
            "lane_change_authorization_source": str(authorization_source),
            "lane_change_initial_progress": (
                float(reference[0].get("lane_change_initial_progress", 0.0))
                if reference else 0.0
            ),
            "lane_change_terminal_progress": (
                float(reference[-1].get("lane_change_progress", 0.0))
                if reference else 0.0
            ),
        }
        if geometry_debug.get("rejection"):
            diagnostics["lane_change_geometry_rejection"] = str(
                geometry_debug["rejection"]
            )
        return CandidateReferenceOverrideResult(
            samples=tuple(
                MappingProxyType(dict(sample)) for sample in reference
            ),
            destination_state=tuple(destination),
            diagnostics=MappingProxyType(diagnostics),
        )

    def candidate_intent_reference(
        self,
        *,
        intent,
        context: CandidateReferenceBuildContext,
        lane_change_state: str,
        geometry_plan,
        keep_lane_reference,
        required_lane_change_decision: str = "",
        required_lane_change_target_lane_id: int = 0,
    ) -> CandidateReferenceOverrideResult:
        """Resolve every geometric override for one behavior intent."""

        decision = str(getattr(intent, "decision", context.baseline_decision))
        target_lane_id = int(
            getattr(intent, "target_lane_id", context.current_lane_id)
            or context.current_lane_id
        )
        target_speed_mps = float(
            getattr(intent, "target_speed_mps", context.baseline_speed_mps)
        )
        same_as_baseline = bool(
            decision == str(context.baseline_decision)
            and target_lane_id == int(context.baseline_target_lane_id)
            and abs(target_speed_mps - float(context.baseline_speed_mps)) < 1e-3
            and decision not in {"lane_change_left", "lane_change_right"}
            and context.baseline_destination_state is not None
        )
        if same_as_baseline:
            destination = list(context.baseline_destination_state or [])
            reference = [dict(x) for x in context.baseline_reference]
            diagnostics = dict(context.baseline_debug)
        else:
            built = self.build_behavior_reference(
                map_planner=context.map_planner,
                ego_pose=context.ego_pose,
                ego_state=context.current_state,
                route_points=context.route_points,
                previous_reference=context.previous_reference,
                previous_target_state=context.previous_target_state,
                behavior_runtime_config=context.behavior_runtime_config,
                decision=decision,
                lane_change_state=str(lane_change_state),
                target_lane_id=target_lane_id,
                current_lane_id=int(context.current_lane_id),
                route_optimal_lane_id=int(context.route_optimal_lane_id),
                route_reference_allowed=bool(context.route_reference_allowed),
                route_reference_gate_reason=str(
                    context.route_reference_gate_reason
                ),
                in_junction=bool(context.in_junction),
                next_macro_maneuver=str(context.next_macro_maneuver),
                planner_mode=str(context.planner_mode),
                lookahead_m=float(context.lookahead_m),
                target_speed_mps=target_speed_mps,
                ego_speed_mps=float(context.ego_speed_mps),
                horizon_steps=int(context.horizon_steps),
                dt_s=float(context.dt_s),
                reference_freeze_count=int(context.reference_freeze_count),
                sim_time_s=float(context.sim_time_s),
                stop_release_smooth_until_s=float(
                    context.stop_release_smooth_until_s
                ),
                authoritative_ego_waypoint=context.authoritative_ego_waypoint,
                lane_reference_step_distance_m=max(
                    0.05, float(geometry_plan.step_m)
                ),
            )
            destination = built.mutable_destination_state()
            reference = built.mutable_samples()
            diagnostics = dict(built.diagnostics)
            diagnostics["fallback_reason"] = str(built.fallback_reason)

        route_required = bool(
            str(required_lane_change_decision)
            and decision == str(required_lane_change_decision)
            and target_lane_id == int(required_lane_change_target_lane_id)
            and len(context.route_points) >= 2
        )
        if route_required:
            override = self.route_lane_change_target_candidate(
                local_map=context.local_map,
                target_lane_id=target_lane_id,
                target_speed_mps=target_speed_mps,
                ego_x_m=float(context.ego_location.x),
                ego_y_m=float(context.ego_location.y),
                spacing_m=float(geometry_plan.step_m),
                geometry_length_m=float(geometry_plan.geometry_length_m),
                horizon_steps=int(context.horizon_steps),
                destination_state=destination,
            )
            if override.samples:
                reference = override.mutable_samples()
            diagnostics.update(dict(override.diagnostics))

        turn_request = TurnReferenceRequest(
            local_map=context.local_map,
            config=context.planner_config,
            horizon_steps=int(context.horizon_steps),
            dt_s=float(context.dt_s),
            ego_location=context.ego_location,
            ego_yaw_rad=float(context.ego_yaw_rad),
            current_state=context.current_state,
            current_lane_id=int(context.current_lane_id),
            target_lane_id=target_lane_id,
            target_speed_mps=target_speed_mps,
            destination_state=destination,
            turn_direction=(
                "left" if decision.endswith("_left")
                else "right" if decision.endswith("_right")
                else str(context.upcoming_turn_direction)
            ),
            route_revision=str(context.route_revision),
            map_epoch=str(context.map_epoch),
        )
        if decision in {"intersection_turn_left", "intersection_turn_right"}:
            override = self.intersection_turn_candidate(
                turn_request, decision=decision
            )
            if override.samples:
                reference = override.mutable_samples()
                destination = override.mutable_destination_state()
            diagnostics.update(dict(override.diagnostics))
        elif (
            decision == "lane_follow"
            and str(context.upcoming_turn_direction) in {"left", "right"}
            and math.isfinite(float(context.upcoming_turn_distance_m))
        ):
            override = self.preturn_candidate(
                turn_request,
                upcoming_turn_distance_m=float(
                    context.upcoming_turn_distance_m
                ),
                first_forward_m=float(geometry_plan.step_m),
            )
            if override.samples:
                reference = override.mutable_samples()
            diagnostics.update(dict(override.diagnostics))

        if decision in {"lane_change_left", "lane_change_right"} and keep_lane_reference:
            duration_s = max(
                float(context.dt_s),
                float(getattr(intent, "lane_change_duration_s", 4.0) or 4.0),
            )
            override = self.lane_change_candidate(
                local_map=context.local_map,
                current_state=context.current_state,
                current_lane_id=int(context.current_lane_id),
                target_lane_id=target_lane_id,
                target_reference=reference,
                source_reference=keep_lane_reference,
                target_speed_mps=target_speed_mps,
                geometry_speed_mps=float(geometry_plan.geometry_speed_mps),
                geometry_length_m=float(geometry_plan.geometry_length_m),
                transition_duration_s=duration_s,
                spacing_m=float(geometry_plan.step_m),
                horizon_steps=int(context.horizon_steps),
                lane_width_m=float(context.lane_width_m),
                destination_state=destination,
                trajectory_variant=str(
                    getattr(intent, "trajectory_variant", "normal")
                ),
                duration_s=float(context.lane_change_duration_s or duration_s),
                duration_reason=str(context.lane_change_duration_reason),
                authorization_source=(
                    "opportunistic"
                    if str(getattr(intent, "reason", "")).startswith(
                        "opportunistic_"
                    )
                    else "route"
                ),
                operational_curvature_limit_1pm=float(
                    geometry_plan.operational_curvature_limit_1pm
                ),
            )
            reference = override.mutable_samples()
            destination = override.mutable_destination_state()
            diagnostics.update(dict(override.diagnostics))
        return CandidateReferenceOverrideResult(
            samples=tuple(MappingProxyType(dict(x)) for x in reference),
            destination_state=tuple(destination),
            diagnostics=MappingProxyType(diagnostics),
        )

    def lane_fallback_reference(
        self,
        *,
        ego_location,
        ego_yaw_rad,
        current_state,
        speed_ref_mps,
    ):
        """Build the provider-owned lane fallback used after pipeline errors.

        This geometry deliberately remains a transient fallback: an exception
        is not a lifecycle event and therefore must not replace a persistent
        lane-follow master.
        """

        return self.builder.build_lane_fallback(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            current_state=current_state,
            speed_ref_mps=float(speed_ref_mps),
        )

    def boundary_recovery_reference(
        self,
        *,
        ego_location,
        ego_yaw_rad,
        current_lane_id,
        base_reference_samples,
        target_speed_mps,
        horizon_steps,
        dt_s,
        max_curvature_1pm,
    ):
        """Build and condition the sole transient boundary-recovery line."""

        generated = self.builder.build_boundary_recovery(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            current_lane_id=int(current_lane_id),
            base_reference_samples=base_reference_samples,
            target_speed_mps=float(target_speed_mps),
            horizon_steps=int(horizon_steps),
            dt_s=float(dt_s),
        )
        samples = [dict(sample) for sample in list(generated.samples or [])]
        conditioning_reason = ""
        if samples:
            samples, conditioning_reason = self.builder.curvature_feasible_samples(
                reference_samples=samples,
                ego_location=ego_location,
                ego_heading_rad=float(ego_yaw_rad),
                max_curvature_1pm=float(max_curvature_1pm),
                mode="boundary_recovery",
            )
        for sample in samples:
            sample["speed_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["v_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["speed_mps"] = max(0.0, float(target_speed_mps))
            sample["reference_mode"] = "boundary_recovery"
        return generated, samples, str(conditioning_reason)

    def start_post_turn_exit(
        self, *, local_map: Any, ego_x_m: float, ego_y_m: float,
        current_lane_id: int, target_speed_mps: float, horizon_steps: int,
        dt_s: float, hold_arc_m: float, route_revision: str, map_epoch: str,
    ) -> bool:
        """Build and install the one immutable outgoing-lane centerline."""

        step_m = max(0.25, float(dt_s) * max(1.0, float(target_speed_mps)))
        required_arc_m = max(step_m, float(hold_arc_m)) + max(
            2.0, 0.5 * float(horizon_steps) * float(dt_s)
            * max(1.0, float(target_speed_mps))
        )
        reference, source = self.post_turn_master(
            local_map, start_lane_id=int(current_lane_id),
            target_speed_mps=float(target_speed_mps),
            horizon_steps=int(horizon_steps), required_arc_m=required_arc_m,
        )
        if not reference:
            return False
        lane_ids = [int(x.get("lane_id", 0) or 0) for x in reference
                    if int(x.get("lane_id", 0) or 0) != 0]
        installed, _ = self.install(
            POST_TURN, reference, route_revision=str(route_revision),
            map_epoch=str(map_epoch), event="phase_transition",
            source_lane_id=int(current_lane_id),
            target_lane_id=int(lane_ids[-1] if lane_ids else current_lane_id),
            build_reason=str(source), ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
        )
        return bool(installed)

    def post_turn_exit_window(
        self, *, ego_x_m: float, ego_y_m: float, target_speed_mps: float,
        horizon_steps: int, dt_s: float, first_forward_m: float,
    ) -> tuple[list[dict[str, object]], str]:
        """Advance and return the sole persistent post-turn reference."""

        if not self.snapshot(POST_TURN).active:
            return [], "post_turn_exit_reference_missing"
        spacing_m = max(0.25, float(dt_s) * max(1.0, float(target_speed_mps)))
        stable_window = self.window(
            POST_TURN, ego_x_m=float(ego_x_m), ego_y_m=float(ego_y_m),
            first_forward_m=float(first_forward_m), spacing_m=spacing_m,
            count=int(horizon_steps),
            max_projection_advance_m=max(2.0, 2.0 * spacing_m),
        )
        snapshot = self.snapshot(POST_TURN)
        window = [dict(sample) for sample in stable_window.samples]
        remaining = sum(math.hypot(
            float(b.get("x_ref_m", b.get("x", 0.0)))
            - float(a.get("x_ref_m", a.get("x", 0.0))),
            float(b.get("y_ref_m", b.get("y", 0.0)))
            - float(a.get("y_ref_m", a.get("y", 0.0))),
        ) for a, b in zip(window[:-1], window[1:]))
        required = max(
            1.5, 0.5 * float(horizon_steps) * float(dt_s)
            * max(1.0, float(target_speed_mps)),
        )
        if remaining + 1.0e-3 < required:
            self.release(POST_TURN, event="phase_transition")
            return [], (
                f"post_turn_exit_reference_exhausted:remaining_arc_m={remaining:.2f}:"
                f"required_m={required:.2f}"
            )
        for sample in window:
            sample["speed_ref_mps"] = sample["v_ref_mps"] = (
                sample["speed_mps"]
            ) = float(target_speed_mps)
        return window, (
            f"post_turn_exit_locked_window:s={snapshot.progress_s_m:.2f}:"
            f"travel={snapshot.travelled_s_m:.2f}:provider={stable_window.reason}"
        )

    @staticmethod
    def mode_for(behavior: BehaviorDecision, *, segment_kind: str = "") -> str:
        maneuver = str(behavior.maneuver or "").strip().lower()
        phase = str(behavior.phase or "").strip().upper()
        if maneuver in {"lane_change_left", "lane_change_right"} or (
            "LANE_CHANGE" in phase
        ):
            return LANE_CHANGE
        if maneuver in {"intersection_turn_left", "intersection_turn_right"}:
            return TURN
        if "POST_TURN" in phase or "TURN_EXIT" in phase:
            return POST_TURN
        if str(segment_kind) == "junction_connector":
            return CONNECTOR
        return LANE_FOLLOW

    def install(
        self,
        mode: str,
        samples: Sequence[Mapping[str, object]],
        *,
        route_revision: str,
        map_epoch: str,
        event: str,
        source_lane_id: int = 0,
        target_lane_id: int = 0,
        maneuver_direction: str = "",
        build_reason: str = "",
        ego_x_m: Optional[float] = None,
        ego_y_m: Optional[float] = None,
    ) -> tuple[bool, str]:
        normalized_mode = self._normalize_mode(mode)
        normalized_event = str(event or "")
        if normalized_event not in _INSTALL_EVENTS:
            return False, "reference_install_event_forbidden:" + normalized_event
        rows = [dict(sample) for sample in list(samples or [])]
        if len(rows) < 2:
            return False, "reference_too_short"
        current = self.snapshot(normalized_mode)
        if current.active and normalized_event == "initial_route":
            return False, "active_reference_cannot_reinitialize"
        if current.active and normalized_event == "route_changed" and (
            str(route_revision) == str(current.route_revision)
        ):
            return False, "route_revision_unchanged"
        if current.active and normalized_event == "map_epoch_changed" and (
            str(map_epoch) == str(current.map_epoch)
        ):
            return False, "map_epoch_unchanged"
        if current.active and normalized_event == "same_route_extension" and (
            str(route_revision) != str(current.route_revision)
            or str(map_epoch) != str(current.map_epoch)
        ):
            return False, "extension_owner_mismatch"
        activation_s_m = 0.0
        if ego_x_m is not None and ego_y_m is not None:
            activation = self.window_from_reference(
                rows,
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                lower_s_m=0.0,
                first_forward_m=0.0,
                spacing_m=0.5,
                count=1,
            )
            activation_s_m = float(activation.projection_s_m)
        immutable = tuple(MappingProxyType(dict(sample)) for sample in rows)
        self._snapshots[normalized_mode] = ReferenceLineSnapshot(
            mode=normalized_mode,
            active=True,
            route_revision=str(route_revision or ""),
            map_epoch=str(map_epoch or ""),
            geometry_revision=int(current.geometry_revision) + 1,
            samples=immutable,
            activation_s_m=float(activation_s_m),
            progress_s_m=float(activation_s_m),
            source_lane_id=int(source_lane_id),
            target_lane_id=int(target_lane_id),
            maneuver_direction=str(maneuver_direction or ""),
            install_event=normalized_event,
            build_reason=str(build_reason or normalized_event),
        )
        return True, "reference_installed:%s:%s" % (
            normalized_mode, normalized_event
        )

    def window(
        self,
        mode: str,
        *,
        ego_x_m: float,
        ego_y_m: float,
        first_forward_m: float,
        spacing_m: float,
        count: int,
        max_projection_advance_m: float = float("inf"),
        ego_heading_rad: Optional[float] = None,
        min_body_forward_m: Optional[float] = None,
    ) -> StableReferenceWindow:
        normalized_mode = self._normalize_mode(mode)
        current = self.snapshot(normalized_mode)
        if not current.active:
            return StableReferenceWindow((), 0.0, 0.0, "reference_inactive")
        result = self.window_from_reference(
            current.mutable_samples(),
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            lower_s_m=float(current.progress_s_m),
            first_forward_m=float(first_forward_m),
            spacing_m=float(spacing_m),
            count=int(count),
            max_projection_advance_m=float(max_projection_advance_m),
        )
        # Arc-length lookahead is not identical to longitudinal lookahead in
        # the ego body frame.  This matters near the end of a lane change,
        # where the vehicle heading and the immutable master can differ by a
        # few degrees: a nominal 0.25 m arc offset was observed as only
        # 0.15 m forward and intermittently violated the reference contract.
        # Drop only leading samples that are behind the requested body-frame
        # anchor; do not rebuild or otherwise mutate the persistent master.
        if ego_heading_rad is not None and min_body_forward_m is not None:
            cosine = math.cos(float(ego_heading_rad))
            sine = math.sin(float(ego_heading_rad))
            rows = [dict(sample) for sample in result.samples]
            first_kept = 0
            for index, sample in enumerate(rows):
                dx_m = float(sample.get("x_ref_m", sample.get("x", 0.0))) - float(ego_x_m)
                dy_m = float(sample.get("y_ref_m", sample.get("y", 0.0))) - float(ego_y_m)
                if dx_m * cosine + dy_m * sine >= float(min_body_forward_m):
                    first_kept = index
                    break
            else:
                first_kept = max(0, len(rows) - 1)
            if first_kept:
                rows = rows[first_kept:]
                result = StableReferenceWindow(
                    tuple(rows),
                    float(result.projection_s_m),
                    float(rows[0].get("s_ref_m", result.start_s_m)),
                    str(result.reason) + ":body_forward_anchored",
                )
        self._snapshots[normalized_mode] = ReferenceLineSnapshot(
            mode=current.mode,
            active=current.active,
            route_revision=current.route_revision,
            map_epoch=current.map_epoch,
            geometry_revision=current.geometry_revision,
            samples=current.samples,
            activation_s_m=current.activation_s_m,
            progress_s_m=max(
                float(current.progress_s_m), float(result.projection_s_m)
            ),
            source_lane_id=current.source_lane_id,
            target_lane_id=current.target_lane_id,
            maneuver_direction=current.maneuver_direction,
            install_event=current.install_event,
            build_reason=current.build_reason,
            last_validation_failure=current.last_validation_failure,
        )
        return result

    def locked_lane_change_window(
        self, *, ego_x_m, ego_y_m, ego_heading_rad, target_lane_id,
        target_speed_mps, spacing_m, count, min_first_forward_m,
        anchor_margin_m=0.05, max_projection_advance_m=float("inf")
    ) -> LaneChangeWindowResult:
        """Produce a complete MPC window from the immutable LC master."""

        master = self.snapshot(LANE_CHANGE).mutable_samples()
        if not master:
            return LaneChangeWindowResult(
                (), 0.0, 0.0, 0,
                "lane_change_window_missing_locked_reference",
            )
        spacing_m = max(0.05, float(spacing_m))
        window = self.window(
            LANE_CHANGE,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            first_forward_m=float(min_first_forward_m) + max(
                0.02, float(anchor_margin_m)
            ),
            spacing_m=float(spacing_m),
            count=int(count),
            max_projection_advance_m=float(max_projection_advance_m),
            ego_heading_rad=float(ego_heading_rad),
            min_body_forward_m=float(min_first_forward_m),
        )
        master_index = min(
            range(len(master)),
            key=lambda index: abs(
                float(master[index].get("s_ref_m", index * spacing_m))
                - float(window.start_s_m)
            ),
        )
        samples = window.mutable_samples() if hasattr(window, "mutable_samples") else [
            dict(sample) for sample in window.samples
        ]
        while samples and len(samples) < int(count):
            previous = dict(samples[-1])
            heading_rad = float(previous.get("heading_rad", ego_heading_rad))
            x_m = float(previous.get("x_ref_m", previous.get("x", ego_x_m)))
            y_m = float(previous.get("y_ref_m", previous.get("y", ego_y_m)))
            padded = dict(previous)
            padded["x_ref_m"] = x_m + spacing_m * math.cos(heading_rad)
            padded["y_ref_m"] = y_m + spacing_m * math.sin(heading_rad)
            padded["x"] = float(padded["x_ref_m"])
            padded["y"] = float(padded["y_ref_m"])
            padded["lane_id"] = int(target_lane_id)
            padded["lane_change_progress"] = 1.0
            samples.append(padded)
        for sample in samples:
            sample["speed_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["v_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["speed_mps"] = max(0.0, float(target_speed_mps))
        return LaneChangeWindowResult(
            tuple(samples), float(window.projection_s_m),
            float(window.start_s_m), int(master_index), str(window.reason)
        )

    def release(self, mode: str, *, event: str) -> tuple[bool, str]:
        normalized_mode = self._normalize_mode(mode)
        normalized_event = str(event or "")
        if normalized_event not in _RELEASE_EVENTS:
            return False, "reference_release_event_forbidden:" + normalized_event
        current = self.snapshot(normalized_mode)
        if not current.active:
            return False, "reference_already_inactive"
        self._snapshots[normalized_mode] = ReferenceLineSnapshot(
            mode=normalized_mode,
            geometry_revision=current.geometry_revision,
            build_reason="released:" + normalized_event,
        )
        return True, "reference_released:%s:%s" % (
            normalized_mode, normalized_event
        )

    def mark_validation_failure(self, mode: str, reason: str) -> None:
        normalized_mode = self._normalize_mode(mode)
        current = self.snapshot(normalized_mode)
        if not current.active:
            return
        self._snapshots[normalized_mode] = ReferenceLineSnapshot(
            mode=current.mode,
            active=current.active,
            route_revision=current.route_revision,
            map_epoch=current.map_epoch,
            geometry_revision=current.geometry_revision,
            samples=current.samples,
            activation_s_m=current.activation_s_m,
            progress_s_m=current.progress_s_m,
            source_lane_id=current.source_lane_id,
            target_lane_id=current.target_lane_id,
            maneuver_direction=current.maneuver_direction,
            install_event=current.install_event,
            build_reason=current.build_reason,
            last_validation_failure=str(reason),
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized not in _MODES:
            raise ValueError("unsupported reference mode: " + normalized)
        return normalized
