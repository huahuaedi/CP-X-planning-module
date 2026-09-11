from types import SimpleNamespace

import pytest

from pipeline.execution_pipeline import (
    PlanningPipeline,
    ScenarioPlanningFrameRequest,
)
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


def test_pipeline_observes_scenario_from_frozen_adapter_frame():
    captured = {}

    class Behavior:
        def prepare_turn_scenario_context(self, **kwargs):
            captured["turn"] = kwargs
            return "turn-context"

    class Scenario:
        def observe_planning_context(self, **kwargs):
            captured["scenario"] = kwargs
            assert kwargs["prepare_turn_context"]() == "turn-context"
            return "scenario-observation"

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()), perception=PerceptionStage(),
        behavior=Behavior(), scenario=Scenario(), static_obstacle=object(),
        control_safety=object(), speed=object(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
    )
    stop_target = SimpleNamespace(active=False)
    route = SimpleNamespace(
        current_road_option="LANEFOLLOW", next_macro_maneuver="turn_right",
        next_macro_distance_m=20.0,
    )
    adapter = SimpleNamespace(
        signal_context={"source": "test"},
        frame=SimpleNamespace(
            planning=SimpleNamespace(route=route, traffic_control=SimpleNamespace(
                signal_state="green", stop_target=stop_target,
            )),
            map_lane=SimpleNamespace(in_junction=False),
        ),
    )
    ego = SimpleNamespace(x=1.0, y=2.0)
    result = pipeline.observe_planning_frame(
        ScenarioPlanningFrameRequest(
            adapter_output=adapter, traffic_memory=object(),
            route_manager=object(), ego_location=ego, ego_yaw_rad=0.0,
            ego_speed_mps=5.0, current_lane_id=10, cruise_speed_mps=8.0,
            sim_time_s=3.0, config={},
        ),
        resolve_actor_state=lambda **_kwargs: ("green", ""),
        project_stop_target=lambda **_kwargs: (float("inf"), False),
    )

    assert result == "scenario-observation"
    assert captured["scenario"]["raw_traffic_state"] == "green"
    assert captured["turn"]["next_macro_maneuver"] == "turn_right"


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


def test_resolve_cav_interaction_builds_corridor_rows_on_the_constraint_reference():
    # Classification runs on a straight +y reference (so the crosser still
    # reads as CROSSING exactly as in the default-behavior test below).
    # constraint_reference_samples stands in for the actually-executed
    # geometry and is rotated 45 degrees from it -- e.g. a proposed lane
    # change under evaluation while the vehicle still drives the original
    # lane-follow line. The QP row's tangent must come from the *executed*
    # reference, not the classification one, or Stage D points the
    # longitudinal band the wrong way relative to the path MPC tracks (the
    # reference-divergence bug fixed by restoring this parameter).
    classification_reference = [
        {"x_ref_m": 0.0, "y_ref_m": float(y)} for y in range(0, 61, 2)
    ]
    executed_reference = [
        {"x_ref_m": float(k), "y_ref_m": float(k)} for k in range(0, 61, 2)
    ]
    crosser = {
        "id": "x", "x": -8.0, "y": 18.0, "v": 7.0, "psi": 0.0,
        "predicted_trajectory": [
            {"x": -8.0 + 0.7 * k, "y": 18.0} for k in range(21)
        ],
    }
    ego_location = SimpleNamespace(x=0.0, y=0.0)

    result = PlanningPipeline.resolve_cav_interaction(
        reference_samples=classification_reference,
        constraint_reference_samples=executed_reference,
        ego_location=ego_location, ego_yaw_rad=1.5707963267948966,
        ego_speed_mps=9.0, actor_id=1, claim=None,
        obstacle_snapshots=[crosser], cav_intents=[], latch_state={},
        horizon_steps=20, dt_s=0.1,
    )

    longitudinal_rows = [
        row for row in result.mpc_rows if row.slack_group == "corridor"
    ]
    assert longitudinal_rows
    half_sqrt2 = 0.7071067811865476
    for row in longitudinal_rows:
        # Executed (diagonal) reference tangent, not the classification
        # reference's straight-+y (0, 1).
        assert row.a_x == pytest.approx(half_sqrt2, abs=1.0e-6)
        assert row.a_y == pytest.approx(half_sqrt2, abs=1.0e-6)


def test_resolve_cav_interaction_defaults_constraint_reference_to_reference_samples():
    reference = [{"x_ref_m": 0.0, "y_ref_m": float(y)} for y in range(0, 61, 2)]
    crosser = {
        "id": "x", "x": -8.0, "y": 18.0, "v": 7.0, "psi": 0.0,
        "predicted_trajectory": [
            {"x": -8.0 + 0.7 * k, "y": 18.0} for k in range(21)
        ],
    }
    ego_location = SimpleNamespace(x=0.0, y=0.0)

    result = PlanningPipeline.resolve_cav_interaction(
        reference_samples=reference,
        ego_location=ego_location, ego_yaw_rad=1.5707963267948966,
        ego_speed_mps=9.0, actor_id=1, claim=None,
        obstacle_snapshots=[crosser], cav_intents=[], latch_state={},
        horizon_steps=20, dt_s=0.1,
    )

    longitudinal_rows = [
        row for row in result.mpc_rows if row.slack_group == "corridor"
    ]
    assert longitudinal_rows
    for row in longitudinal_rows:
        assert row.a_x == pytest.approx(0.0, abs=1.0e-6)
        assert row.a_y == pytest.approx(1.0, abs=1.0e-6)
