"""Executable, bridge-independent CP-X planning stage owner."""

from __future__ import annotations

from typing import Any, Mapping

from .perception_stage import PerceptionStage, PerceptionStageResult
from .runtime_input_stage import RuntimeInputStage, RuntimeTickSnapshot


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
        speed: Any,
        destination_speed: Any,
        reference_publication: Any,
        mpc_entry: Any,
        fallback: Any = None,
    ) -> None:
        self._runtime_input = runtime_input
        self._perception = perception
        self.behavior = behavior
        self.scenario = scenario
        self.static_obstacle = static_obstacle
        self.speed = speed
        self.destination_speed = destination_speed
        self.reference_publication = reference_publication
        self.mpc_entry = mpc_entry
        self.fallback = fallback

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

    def front_gap(self, **kwargs):
        return self._perception.front_gap(**kwargs)

    def evaluate_destination(self, **kwargs):
        return self.destination_speed.evaluate(**kwargs)

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
        target = self.speed.resolve(
            behavior=behavior,
            speed_plan=speed_plan,
            additional_constraints=additional_constraints,
        )
        ceiling = self.speed.apply(
            target,
            destination_state=destination_state,
            reference_samples=reference_samples,
        )
        return target, ceiling

    def propose_speed(self, **kwargs):
        return self.speed.propose(**kwargs)

    @property
    def destination_stop_latched(self) -> bool:
        return bool(self.destination_speed.stop_latched)

    def publish_reference(self, **kwargs):
        return self.reference_publication.run(**kwargs)

    def evaluate_mpc_entry(self, **kwargs):
        return self.mpc_entry.evaluate(**kwargs)

    def prepare_mpc_control_context(self, **kwargs):
        return self.mpc_entry.prepare_control_context(**kwargs)

    def low_speed_mpc_replan_required(self, **kwargs):
        return self.mpc_entry.low_speed_replan_required(**kwargs)

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
