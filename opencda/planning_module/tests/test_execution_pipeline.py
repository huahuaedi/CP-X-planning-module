from types import SimpleNamespace

import pytest

from pipeline.execution_pipeline import PlanningPipeline
from pipeline.perception_stage import PerceptionStage
from pipeline.runtime_input_stage import RuntimeInputStage


class _Mapper:
    def update_measurement(self, **_kwargs):
        return 0.25


def test_pipeline_sequences_runtime_and_perception_without_bridge():
    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(max_mpc_obstacles=4),
        behavior=object(),
        scenario=object(),
        static_obstacle=object(),
        speed=object(),
        destination_speed=object(),
        reference_publication=object(),
        mpc_entry=object(),
    )
    transform = SimpleNamespace(
        location=SimpleNamespace(x=1.0, y=2.0, z=0.0),
        rotation=SimpleNamespace(yaw=0.0),
    )

    tick = pipeline.begin_tick(
        timestamp_s=3.0, ego_transform=transform, ego_speed_kmh=18.0
    )
    perception = pipeline.perceive(
        tick,
        detected_objects={"vehicles": [{
            "vehicle_id": "front", "x": 10.0, "y": 2.0, "v": 2.0,
        }]},
        cp_payload={},
        ignore_dynamic_objects=False,
    )

    assert tick.ego_speed_mps == pytest.approx(5.0)
    assert perception.front_actor_id == "front"
    assert len(perception.fused_objects) == 1
    assert not hasattr(pipeline, "bridge")


def test_pipeline_owns_speed_resolution_sequence():
    calls = []

    class Speed:
        def resolve(self, **kwargs):
            calls.append(("resolve", kwargs))
            return "target"

        def apply(self, target, **kwargs):
            calls.append(("apply", target, kwargs))
            return "ceiling"

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(),
        behavior=object(), scenario=object(), static_obstacle=object(),
        speed=Speed(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
    )

    result = pipeline.resolve_speed(
        behavior="behavior", speed_plan="plan", additional_constraints=(),
        destination_state=[1.0], reference_samples=[{"x": 1.0}],
    )

    assert result == ("target", "ceiling")
    assert [call[0] for call in calls] == ["resolve", "apply"]


def test_pipeline_is_the_only_behavior_stage_caller():
    calls = []

    class Behavior:
        def finalize(self, **kwargs):
            calls.append(("finalize", kwargs))
            return "decision"

        def reset_route_lane_change_authorization(self):
            calls.append(("reset", {}))

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(),
        behavior=Behavior(), scenario=object(), static_obstacle=object(),
        speed=object(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
    )

    assert pipeline.finalize_behavior(maneuver="lane_follow") == "decision"
    pipeline.reset_route_lane_change_authorization()
    assert [call[0] for call in calls] == ["finalize", "reset"]


def test_pipeline_is_the_only_fallback_stage_caller():
    calls = []

    class Fallback:
        def record_valid(self, trajectory, **kwargs):
            calls.append(("record", trajectory, kwargs))
            return True

        def resolve(self, **kwargs):
            calls.append(("resolve", kwargs))
            return "fallback"

        def bounded_safe_stop(self, **kwargs):
            calls.append(("stop", kwargs))
            return "safe-stop"

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(),
        behavior=object(), scenario=object(), static_obstacle=object(),
        speed=object(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
        fallback=Fallback(),
    )

    assert pipeline.record_valid_trajectory([{"x": 1.0}], sim_time_s=1.0)
    assert pipeline.resolve_fallback(reason="failure") == "fallback"
    assert pipeline.bounded_safe_stop(reason="destination") == "safe-stop"
    assert [call[0] for call in calls] == ["record", "resolve", "stop"]
