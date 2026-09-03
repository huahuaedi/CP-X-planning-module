"""Executable, bridge-independent CP-X planning stage owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .perception_stage import PerceptionStage, PerceptionStageResult
from .runtime_input_stage import RuntimeInputStage, RuntimeTickSnapshot
from .speed_planner import effective_emergency_gap_m


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

    def destination_stop_behavior(self, result):
        return self.behavior.destination_stop(result)

    def authorize_route_lane_change(self, request, *, maneuver_manager):
        return self.behavior.authorize_route_lane_change(
            request, maneuver_manager=maneuver_manager
        )

    def resolve_route_lane_change(self, context, **kwargs):
        return self.behavior.resolve_route_lane_change(context, **kwargs)

    def resolve_lateral_ownership(self, **kwargs):
        return self.behavior.resolve_lateral_ownership(**kwargs)

    def prepare_route_lane_change(self, **kwargs):
        return self.behavior.prepare_route_lane_change(**kwargs)

    def prepare_turn_scenario_context(self, **kwargs):
        return self.behavior.prepare_turn_scenario_context(**kwargs)

    def resolve_scenario(self, **kwargs):
        return self.scenario.update_planning_context(**kwargs)

    def resolve_static_obstacle(self, **kwargs):
        return self.static_obstacle.evaluate(**kwargs)

    def reset_route_lane_change_authorization(self):
        self.behavior.reset_route_lane_change_authorization()

    def authorize_opportunistic_lane_change(self, request):
        return self.behavior.authorize_opportunistic_lane_change(request)

    def evaluate_behavior_candidates(self, request):
        return self.behavior.evaluate_lane_candidates(request)

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

    def select_candidate(self, request, **kwargs):
        if self.candidate_selection is None:
            raise RuntimeError("candidate selection stage is not configured")
        return self.candidate_selection.run(request, **kwargs)
