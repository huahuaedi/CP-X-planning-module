"""Executable, bridge-independent CP-X planning stage owner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .perception_stage import PerceptionStage, PerceptionStageResult
from .runtime_input_stage import RuntimeInputStage, RuntimeTickSnapshot
from .speed_planner import effective_emergency_gap_m
from .cav_conflict_pipeline import resolve_conflicts as resolve_cav_conflicts
from .conflict_classifier import ClassifierParams
from .spatiotemporal_corridor import CorridorParams


@dataclass(frozen=True)
class PlanningCycle:
    """Immutable normalized inputs and the initial longitudinal safety intent."""

    tick: RuntimeTickSnapshot
    perception: PerceptionStageResult
    emergency_gap_m: float
    emergency_stop_required: bool
    requested_speed_mps: float

    @property
    def object_snapshots(self):
        return self.perception.fused_objects

    @property
    def mpc_object_snapshots(self):
        return self.perception.mpc_objects

    @property
    def local_object_snapshots(self):
        return self.perception.local_objects


@dataclass(frozen=True)
class TrajectoryAdmission:
    """Published reference together with its sole MPC admission decision."""

    publication: Any
    entry: Any
    control_context: Any

    def trace_fields(self) -> dict[str, object]:
        fields = dict(self.publication.debug_fields)
        fields.update(self.entry.trace_fields())
        return fields


@dataclass(frozen=True)
class ScenarioPlanningFrameRequest:
    """Immutable adapter-frame view consumed by scenario observation."""

    adapter_output: Any
    traffic_memory: Any
    route_manager: Any
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    current_lane_id: int
    cruise_speed_mps: float
    sim_time_s: float
    config: Mapping[str, object]
    boundary_recovery_request: Any = None


class PlanningPipeline:
    """Own and sequence planning stages without owning OpenCDA runtime I/O."""

    def __init__(
        self,
        *,
        runtime_input: RuntimeInputStage,
        perception: PerceptionStage,
        behavior: Any,
        scenario: Any,
        static_obstacle: Any,
        control_safety: Any,
        speed: Any,
        destination_speed: Any,
        reference_publication: Any,
        mpc_entry: Any,
        mpc_execution: Any = None,
        fallback: Any = None,
        behavior_reference_execution: Any = None,
        candidate_selection: Any = None,
        control_finalization: Any = None,
    ) -> None:
        self._runtime_input = runtime_input
        self._perception = perception
        self.behavior = behavior
        self.scenario = scenario
        self.static_obstacle = static_obstacle
        self.control_safety = control_safety
        self.speed = speed
        self.destination_speed = destination_speed
        self.reference_publication = reference_publication
        self.mpc_entry = mpc_entry
        self.mpc_execution = mpc_execution
        self.fallback = fallback
        self.behavior_reference_execution = behavior_reference_execution
        self.candidate_selection = candidate_selection
        self.control_finalization = control_finalization

    def begin_tick(
        self, *, timestamp_s: float, ego_transform: Any, ego_speed_kmh: float
    ) -> RuntimeTickSnapshot:
        return self._runtime_input.build(
            timestamp_s=float(timestamp_s),
            ego_transform=ego_transform,
            ego_speed_kmh=float(ego_speed_kmh),
        )

    def perceive(
        self,
        tick: RuntimeTickSnapshot,
        *,
        detected_objects: Any,
        cp_payload: Mapping[str, Any],
        ignore_dynamic_objects: bool,
    ) -> PerceptionStageResult:
        return self._perception.build(
            detected_objects=detected_objects,
            cp_payload=cp_payload,
            ego_location=tick.ego_location,
            ego_yaw_rad=float(tick.ego_yaw_rad),
            timestamp_s=float(tick.timestamp_s),
            ignore_dynamic_objects=bool(ignore_dynamic_objects),
        )

    def begin_cycle(
        self,
        *,
        timestamp_s: float,
        ego_transform: Any,
        ego_speed_kmh: float,
        detected_objects: Any,
        cp_payload: Mapping[str, Any],
        ignore_dynamic_objects: bool,
        cruise_speed_mps: float,
        base_emergency_gap_m: float,
        emergency_standstill_buffer_m: float,
        following_time_headway_s: float,
    ) -> PlanningCycle:
        """Create the single authoritative input snapshot for one planner tick."""

        tick = self.begin_tick(
            timestamp_s=timestamp_s,
            ego_transform=ego_transform,
            ego_speed_kmh=ego_speed_kmh,
        )
        perception = self.perceive(
            tick,
            detected_objects=detected_objects,
            cp_payload=cp_payload,
            ignore_dynamic_objects=ignore_dynamic_objects,
        )
        emergency_gap_m = effective_emergency_gap_m(
            base_emergency_gap_m=max(0.5, float(base_emergency_gap_m)),
            ego_speed_mps=float(tick.ego_speed_mps),
            front_obstacle_speed_mps=perception.front_actor_speed_mps,
            standstill_buffer_m=max(0.0, float(emergency_standstill_buffer_m)),
            time_headway_s=max(0.1, float(following_time_headway_s)),
        )
        stop_required = bool(
            perception.front_gap_m is not None
            and float(perception.front_gap_m) <= float(emergency_gap_m)
        )
        return PlanningCycle(
            tick=tick,
            perception=perception,
            emergency_gap_m=float(emergency_gap_m),
            emergency_stop_required=stop_required,
            requested_speed_mps=(
                0.0 if stop_required else max(0.0, float(cruise_speed_mps))
            ),
        )

    def front_gap(self, **kwargs):
        return self._perception.front_gap(**kwargs)

    def evaluate_destination(self, **kwargs):
        return self.destination_speed.evaluate(**kwargs)

    def apply_destination(self, **kwargs):
        return self.destination_speed.apply(**kwargs)

    def finalize_behavior(self, **kwargs):
        return self.behavior.finalize(**kwargs)

    def finalize_behavior_frame(self, **kwargs):
        return self.behavior.finalize_planning_frame(**kwargs)

    def destination_stop_behavior(self, result):
        return self.behavior.destination_stop(result)

    def authorize_route_lane_change(self, request, *, maneuver_manager):
        return self.behavior.authorize_route_lane_change(
            request, maneuver_manager=maneuver_manager
        )

    def resolve_route_lane_change(self, context, **kwargs):
        return self.behavior.resolve_route_lane_change(context, **kwargs)

    def resolve_conflicts(self, request, **kwargs):
        return self.behavior.resolve_conflicts(request, **kwargs)

    @staticmethod
    def resolve_cav_interaction(
        *, reference_samples, constraint_reference_samples=None,
        ego_location, ego_yaw_rad, ego_speed_mps,
        actor_id, claim, obstacle_snapshots, cav_intents, latch_state,
        tag_state=None,
        horizon_steps, dt_s, mode_probability_floor=0.05,
        credible_mode_probability_min=0.15, credible_mode_ttc_s=2.0,
        refresh_assignments=True, cached_assignments=(),
    ):
        """Classify on proposal geometry and constrain the executable geometry."""

        result = resolve_cav_conflicts(
            reference_samples=reference_samples,
            ego_snapshot={
                "x": float(ego_location.x), "y": float(ego_location.y),
                "v": float(ego_speed_mps), "psi": float(ego_yaw_rad),
            },
            my_actor_id=int(actor_id), my_claim=claim,
            obstacle_snapshots=obstacle_snapshots, cav_intents=cav_intents,
            latch_state=latch_state, tag_state=tag_state,
            classifier_params=ClassifierParams(
                horizon_steps=max(1, int(horizon_steps)), dt_s=float(dt_s)
            ),
            corridor_params=CorridorParams(
                horizon_steps=max(1, int(horizon_steps)), dt_s=float(dt_s)
            ),
            mode_probability_floor=float(mode_probability_floor),
            credible_mode_probability_min=float(credible_mode_probability_min),
            credible_mode_ttc_s=float(credible_mode_ttc_s),
            refresh_assignments=bool(refresh_assignments),
            cached_assignments=cached_assignments,
        )
        from .cav_intent_codec import sample_cav_path_at
        from .mpc_corridor_constraints import corridor_rows, homotopy_keepout_rows

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
        qp_reference = (
            reference_samples
            if constraint_reference_samples is None
            else constraint_reference_samples
        )
        longitudinal_rows = corridor_rows(
            result.corridor, qp_reference, ego_origin_xy=origin
        )
        lateral_rows = homotopy_keepout_rows(
            result.assignments,
            tracks,
            ego_heading_rad=float(ego_yaw_rad),
            ego_origin_xy=origin,
        )
        result.mpc_rows = list(longitudinal_rows) + list(lateral_rows)
        result.diagnostics.update({
            "longitudinal_qp_row_count": len(longitudinal_rows),
            "homotopy_qp_row_count": len(lateral_rows),
            "total_qp_row_count": len(result.mpc_rows),
            "shared_planned_paths": {
                str(intent.actor_id): [
                    (float(sample[1]), float(sample[2]))
                    for sample in intent.planned_path
                ]
                for intent in list(cav_intents or [])
                if intent.planned_path
            },
        })
        if cav_intents:
            validation_intent = sorted(cav_intents, key=lambda value: value.actor_id)[0]
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

    def prepare_route_lane_change(self, **kwargs):
        return self.behavior.prepare_route_lane_change(**kwargs)

    def resolve_route_context(self, **kwargs):
        return self.behavior.resolve_route_context(**kwargs)

    def observe_planning_frame(
        self, request: ScenarioPlanningFrameRequest, *,
        resolve_actor_state: Callable[..., Any],
        project_stop_target: Callable[..., Any],
    ):
        """Resolve traffic memory and turn context from one frozen frame."""

        frame = request.adapter_output.frame
        route = frame.planning.route
        traffic = frame.planning.traffic_control
        raw_stop_target = (
            traffic.stop_target.as_dict() if traffic.stop_target.active else None
        )
        return self.scenario.observe_planning_context(
            raw_traffic_state=str(traffic.signal_state),
            raw_stop_target=raw_stop_target,
            signal_context=dict(request.adapter_output.signal_context),
            traffic_memory=request.traffic_memory,
            resolve_actor_state=resolve_actor_state,
            project_stop_target=project_stop_target,
            prepare_turn_context=lambda: self.behavior.prepare_turn_scenario_context(
                route_manager=request.route_manager,
                ego_x_m=float(request.ego_location.x),
                ego_y_m=float(request.ego_location.y),
                ego_heading_rad=float(request.ego_yaw_rad),
                cruise_speed_mps=float(request.cruise_speed_mps),
                next_macro_maneuver=str(route.next_macro_maneuver),
                next_macro_distance_m=float(route.next_macro_distance_m),
                config=request.config,
            ),
            sim_time_s=float(request.sim_time_s),
            ego_x_m=float(request.ego_location.x),
            ego_y_m=float(request.ego_location.y),
            ego_yaw_rad=float(request.ego_yaw_rad),
            ego_speed_mps=float(request.ego_speed_mps),
            current_lane_id=int(request.current_lane_id),
            ego_in_junction=bool(frame.map_lane.in_junction),
            current_road_option=str(route.current_road_option),
            next_macro_maneuver=str(route.next_macro_maneuver),
            virtual_stop_distance_m=float(request.config.get(
                "full_latched_virtual_stop_distance_m", 12.0
            )),
            boundary_recovery_request=request.boundary_recovery_request,
        )

    def resolve_static_obstacle(self, **kwargs):
        return self.static_obstacle.evaluate(**kwargs)

    def reset_route_lane_change_authorization(self):
        self.behavior.reset_route_lane_change_authorization()

    def produce_behavior_command(self, request, **kwargs):
        return self.behavior.produce_command_from_frame(request, **kwargs)

    def apply_behavior_overrides(self, request):
        return self.behavior.apply_overrides(request)

    def resolve_speed(
        self, *, behavior, speed_plan, additional_constraints,
        destination_state, reference_samples,
    ):
        return self.speed.resolve_frame(
            behavior=behavior,
            speed_plan=speed_plan,
            additional_constraints=additional_constraints,
            destination_state=destination_state,
            reference_samples=reference_samples,
        )

    def propose_speed(self, **kwargs):
        return self.speed.propose(**kwargs)

    @property
    def destination_stop_latched(self) -> bool:
        return bool(self.destination_speed.stop_latched)

    def publish_reference(self, **kwargs):
        return self.reference_publication.run(**kwargs)

    def prepare_trajectory_execution(
        self,
        *,
        publication_kwargs,
        behavior,
        stop_goal_active,
        ego_speed_mps,
        ego_x_m,
        ego_y_m,
        ego_yaw_rad,
        mode_transition_reason,
        front_gap_actor_id,
        candidate_status,
        candidate_name,
        candidate_reason,
    ) -> TrajectoryAdmission:
        publication = self.reference_publication.run(**publication_kwargs)
        entry = self.mpc_entry.evaluate(
            candidate_status=candidate_status,
            candidate_name=candidate_name,
            candidate_reason=candidate_reason,
            final_reference_accepted=bool(publication.gate.accepted),
            final_reference_reason=str(publication.gate.reason),
            behavior_decision=str(behavior.maneuver),
            stop_goal_active=bool(stop_goal_active),
            ego_speed_mps=float(ego_speed_mps),
        )
        control_context = self.mpc_entry.prepare_control_context(
            behavior=behavior,
            reference_source=str(
                publication.debug_fields.get("reference_source", "")
            ),
            stop_goal_active=bool(stop_goal_active),
            front_gap_actor_id=str(front_gap_actor_id),
            reference_samples=publication.mutable_samples(),
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_yaw_rad=float(ego_yaw_rad),
            mode_transition_reason=str(mode_transition_reason),
        )
        return TrajectoryAdmission(
            publication=publication,
            entry=entry,
            control_context=control_context,
        )

    def evaluate_mpc_entry(self, **kwargs):
        return self.mpc_entry.evaluate(**kwargs)

    def prepare_mpc_control_context(self, **kwargs):
        return self.mpc_entry.prepare_control_context(**kwargs)

    def execute_mpc(self, request, **kwargs):
        if self.mpc_execution is None:
            raise RuntimeError("MPC execution stage is not configured")
        return self.mpc_execution.run(request, **kwargs)

    def apply_control_safety(self, **kwargs):
        return self.control_safety.run(**kwargs)

    def finalize_control(self, request, **kwargs):
        if self.control_finalization is None:
            raise RuntimeError("control finalization stage is not configured")
        return self.control_finalization.run(request, **kwargs)

    @staticmethod
    def explain_decision(diagnostics):
        from .decision_record import decision_record_from_diagnostics

        return decision_record_from_diagnostics(diagnostics)

    def record_valid_trajectory(self, trajectory, **kwargs):
        if self.fallback is None:
            return False
        return self.fallback.record_valid(trajectory, **kwargs)

    def resolve_fallback(self, **kwargs):
        if self.fallback is None:
            raise RuntimeError("fallback stage is not configured")
        return self.fallback.resolve(**kwargs)

    def bounded_safe_stop(self, **kwargs):
        if self.fallback is None:
            raise RuntimeError("fallback stage is not configured")
        return self.fallback.bounded_safe_stop(**kwargs)

    def resolve_candidate_failure(self, **kwargs):
        if self.fallback is None:
            raise RuntimeError("fallback stage is not configured")
        return self.fallback.resolve_candidate_failure(**kwargs)

    def execute_behavior_reference(self, request, *, planner):
        if self.behavior_reference_execution is None:
            raise RuntimeError("behavior/reference execution stage is not configured")
        return self.behavior_reference_execution.run(request, planner=planner)

    def arbitrate_candidates(self, request, **kwargs):
        if self.candidate_selection is None:
            raise RuntimeError("candidate selection stage is not configured")
        return self.candidate_selection.arbitrate(request, **kwargs)

    def cooperative_conflict_reference(self, **kwargs):
        if self.candidate_selection is None:
            raise RuntimeError("candidate selection stage is not configured")
        return self.candidate_selection.cooperative_conflict_reference(**kwargs)
