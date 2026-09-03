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
        control_safety=object(),
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


def test_pipeline_cycle_owns_initial_emergency_speed_intent():
    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(max_mpc_obstacles=4),
        behavior=object(), scenario=object(), static_obstacle=object(),
        control_safety=object(), speed=object(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
    )
    transform = SimpleNamespace(
        location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        rotation=SimpleNamespace(yaw=0.0),
    )
    cycle = pipeline.begin_cycle(
        timestamp_s=4.0,
        ego_transform=transform,
        ego_speed_kmh=18.0,
        detected_objects={"vehicles": [{
            "vehicle_id": "front", "x": 2.0, "y": 0.0, "v": 0.0,
        }]},
        cp_payload={},
        ignore_dynamic_objects=False,
        cruise_speed_mps=8.0,
        base_emergency_gap_m=3.0,
        emergency_standstill_buffer_m=1.0,
        following_time_headway_s=1.5,
    )

    assert cycle.tick.ego_speed_mps == pytest.approx(5.0)
    assert cycle.emergency_stop_required
    assert cycle.requested_speed_mps == 0.0
    assert cycle.perception.front_actor_id == "front"


def test_pipeline_owns_speed_resolution_sequence():
    calls = []

    class Speed:
        def resolve_frame(self, **kwargs):
            calls.append(("resolve_frame", kwargs))
            return "speed-frame"

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(),
        behavior=object(), scenario=object(), static_obstacle=object(), control_safety=object(),
        speed=Speed(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
    )

    result = pipeline.resolve_speed(
        behavior="behavior", speed_plan="plan", additional_constraints=(),
        destination_state=[1.0], reference_samples=[{"x": 1.0}],
    )

    assert result == "speed-frame"
    assert [call[0] for call in calls] == ["resolve_frame"]


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
        behavior=Behavior(), scenario=object(), static_obstacle=object(), control_safety=object(),
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
        behavior=object(), scenario=object(), static_obstacle=object(), control_safety=object(),
        speed=object(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
        fallback=Fallback(),
    )

    assert pipeline.record_valid_trajectory([{"x": 1.0}], sim_time_s=1.0)
    assert pipeline.resolve_fallback(reason="failure") == "fallback"
    assert pipeline.bounded_safe_stop(reason="destination") == "safe-stop"
    assert [call[0] for call in calls] == ["record", "resolve", "stop"]


def test_pipeline_owns_reference_publication_and_mpc_admission_sequence():
    calls = []
    publication = SimpleNamespace(
        gate=SimpleNamespace(accepted=True, reason="accepted"),
        debug_fields={"reference_source": "lane_follow"},
        mutable_samples=lambda: [{"x_ref_m": 1.0, "y_ref_m": 0.0}],
    )

    class Publication:
        def run(self, **kwargs):
            calls.append(("publish", kwargs))
            return publication

    class Entry:
        def evaluate(self, **kwargs):
            calls.append(("entry", kwargs))
            return SimpleNamespace(trace_fields=lambda: {"mpc_entry_allowed": True})

        def prepare_control_context(self, **kwargs):
            calls.append(("context", kwargs))
            return "control-context"

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(), behavior=object(), scenario=object(),
        static_obstacle=object(), control_safety=object(), speed=object(),
        destination_speed=object(), reference_publication=Publication(),
        mpc_entry=Entry(),
    )
    behavior = SimpleNamespace(maneuver="lane_follow")
    result = pipeline.prepare_trajectory_execution(
        publication_kwargs={"reference_samples": []},
        behavior=behavior,
        stop_goal_active=False,
        ego_speed_mps=2.0,
        ego_x_m=0.0,
        ego_y_m=0.0,
        ego_yaw_rad=0.0,
        mode_transition_reason="",
        front_gap_actor_id="",
        candidate_status="feasible",
        candidate_name="lane_follow",
        candidate_reason="",
    )
    assert result.publication is publication
    assert result.control_context == "control-context"
    assert result.trace_fields()["mpc_entry_allowed"] is True
    assert [call[0] for call in calls] == ["publish", "entry", "context"]
