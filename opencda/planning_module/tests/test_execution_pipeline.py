from types import SimpleNamespace

import pytest

from pipeline.cooperative_arbitration import CavIntent, ResourceClaim
from pipeline.behavior_reference_finalization_stage import (
    BehaviorFrameFinalizationRequest,
    BehaviorReferenceFinalizationRequest,
    MPCCostProfileRequest,
)
from pipeline.execution_pipeline import (
    BehaviorContextRequest,
    CooperativePlanningFrame,
    ExecutableBehaviorRequest,
    NominalPlanningRequest,
    PlanningPipeline,
    ScenarioPlanningFrameRequest,
)
from pipeline.cav_interaction_stage import CAVInteractionStage
from pipeline.perception_stage import PerceptionStage
from pipeline.runtime_input_stage import RuntimeInputStage
from pipeline.reference_planning_stage import PostTurnReferenceRequest


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


def test_pipeline_owns_cooperative_resolution_and_speed_handoff():
    resolution = SimpleNamespace(speed_constraint="cav-cap")

    class Cooperative:
        def run(self, request):
            assert request == "cooperative-request"
            return resolution, True

    class Speed:
        def constrain_plan(self, speed_plan, constraint):
            assert speed_plan == "candidate-speed-plan"
            assert constraint == "cav-cap"
            return "constrained-speed-plan"

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()), perception=PerceptionStage(),
        behavior=object(), scenario=object(), static_obstacle=object(),
        control_safety=object(), speed=Speed(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
        cooperative=Cooperative(),
    )

    frame = pipeline.resolve_cooperative("cooperative-request")
    speed_plan = pipeline.constrain_speed_from_cooperative(
        "candidate-speed-plan", frame
    )

    assert frame.cav_resolution is resolution
    assert frame.lane_change_deferred
    assert speed_plan == "constrained-speed-plan"


def test_pipeline_owns_selected_candidate_finalization_order():
    calls = []

    class Speed:
        def constrain_plan(self, plan, constraint):
            calls.append(("speed", constraint.owner))
            return SimpleNamespace(
                target_speed_mps=min(
                    float(plan.target_speed_mps), float(constraint.maximum_mps)
                ),
                upcoming_turn_distance_m=plan.upcoming_turn_distance_m,
            )

        def turn_curvature_constraint(self, curvature, _config, **kwargs):
            calls.append(("turn_constraint", curvature, kwargs))
            return SimpleNamespace(owner="turn", maximum_mps=4.0)

    class CostProfile:
        def apply(self, **kwargs):
            calls.append(("profile", kwargs["behavior"]))

    class ReferencePlanning:
        def finalize_post_turn(self, request):
            calls.append(("post_turn", request.target_speed_mps))
            assert request.debug_fields["turn_longitudinal_authority"] == (
                "SpeedPlanner"
            )
            return SimpleNamespace(
                destination_state=(1.0, 2.0, request.target_speed_mps, 0.0, 7),
                reference_samples=({"x_ref_m": 1.0, "y_ref_m": 2.0},),
                debug_fields=request.debug_fields,
                mutable_destination_state=lambda: [
                    1.0, 2.0, request.target_speed_mps, 0.0, 7
                ],
                mutable_samples=lambda: [
                    {"x_ref_m": 1.0, "y_ref_m": 2.0}
                ],
            )

    class Behavior:
        def finalize_planning_frame(self, **kwargs):
            calls.append(("behavior", kwargs["requested_speed_mps"]))
            return SimpleNamespace(decision="final-behavior")

    provider = SimpleNamespace(turn_master_curvature_1pm=lambda: 0.2)
    cooperative = CooperativePlanningFrame(
        cav_resolution=SimpleNamespace(
            speed_constraint=SimpleNamespace(owner="cav", maximum_mps=7.0)
        ),
        lane_change_deferred=False,
    )
    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()), perception=PerceptionStage(),
        behavior=Behavior(), scenario=object(), static_obstacle=object(),
        control_safety=object(), speed=Speed(), destination_speed=object(),
        reference_publication=object(), mpc_entry=object(),
        reference_planning=ReferencePlanning(),
        mpc_cost_profile=CostProfile(),
    )
    post_turn = PostTurnReferenceRequest(
        maneuver_manager=object(), decision="turn_left", scenario_state="TURN",
        exit_alignment_valid=False, exit_lateral_error_m=0.0,
        exit_heading_error_rad=0.0, local_map=object(), ego_x_m=0.0,
        ego_y_m=0.0, current_state=(0.0,) * 5, current_lane_id=3,
        target_speed_mps=9.0, horizon_steps=10, dt_s=0.2,
        route_revision="r1", map_epoch="admap", config={},
        destination_state=(1.0, 2.0, 9.0, 0.0, 7),
        reference_samples=({"x_ref_m": 1.0, "y_ref_m": 2.0},),
        debug_fields={"reference_tracking_mode": "turn"},
        reference_freeze_count=0,
    )
    result = pipeline.finalize_behavior_reference(
        BehaviorReferenceFinalizationRequest(
            speed_plan=SimpleNamespace(
                target_speed_mps=9.0, upcoming_turn_distance_m=12.0
            ),
            cooperative_frame=cooperative,
            reference_provider=provider,
            config={},
            mpc_profile=MPCCostProfileRequest(
                behavior="turn_left", planner_lc_state="IDLE",
                planner_mode="intersection_turn", reference_tracking_mode="turn",
                next_macro_maneuver="turn_left", sim_time_s=1.0,
                nearest_obstacle_distance_m=None, ego_speed_mps=3.0,
            ),
            post_turn=post_turn,
            behavior=BehaviorFrameFinalizationRequest(
                maneuver="turn_left", phase="IDLE", source_lane_id=3,
                target_lane_id=7, stop_required=False, route_required=False,
                traffic_signal_state="green", boundary_recovery_active=False,
                stop_target=None, reason="route", lane_safety_scores={},
                raw_signal_state="green", resolved_signal_state="green",
                filtered_signal_state="green", traffic_control_from_cp=False,
                scenario_state="TURN",
            ),
        )
    )

    assert result.speed_plan.target_speed_mps == pytest.approx(4.0)
    assert result.destination_state[2] == pytest.approx(4.0)
    assert result.cav_resolution is cooperative.cav_resolution
    assert calls[0] == ("speed", "cav")
    assert calls[1][0] == "turn_constraint"
    assert calls[2] == ("speed", "turn")
    assert calls[3:] == [
        ("profile", "turn_left"),
        ("post_turn", 4.0),
        ("behavior", 4.0),
    ]


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


def test_pipeline_owns_route_scenario_conflict_context_sequence():
    calls = []
    authorization = SimpleNamespace(required_by_route=True)
    route_behavior = SimpleNamespace(
        authorization=authorization,
        replan_reason="route_discontinuity",
    )
    handoff = SimpleNamespace(action="release", reason="maneuver_complete")
    conflict = SimpleNamespace(
        authorization=authorization,
        lateral_ownership=SimpleNamespace(
            handoff=handoff,
            reference_release_event="phase_transition",
        ),
    )

    class Behavior:
        def resolve_route_context(self, **kwargs):
            calls.append(("route", kwargs))
            return route_behavior

        def prepare_turn_scenario_context(self, **kwargs):
            calls.append(("turn", kwargs))
            return SimpleNamespace(distance_m=18.0)

        def resolve_conflicts(self, request, **kwargs):
            calls.append(("conflict", request, kwargs))
            assert request.route_authorization is authorization
            assert request.distance_to_turn_m == pytest.approx(18.0)
            return conflict

    scenario_observation = SimpleNamespace(
        turn_context=SimpleNamespace(distance_m=18.0),
        scenario=SimpleNamespace(decision=SimpleNamespace(state="LANE_KEEP")),
    )

    class Scenario:
        def observe_planning_context(self, **kwargs):
            calls.append(("scenario", kwargs))
            assert kwargs["prepare_turn_context"]().distance_m == 18.0
            return scenario_observation

    class ReferenceProvider:
        def release(self, mode, *, event):
            calls.append(("release", mode, event))
            return True, "released"

        def snapshot(self, _mode):
            return SimpleNamespace(active=False)

    stop_target = SimpleNamespace(active=False)
    adapter = SimpleNamespace(
        signal_context={},
        frame=SimpleNamespace(
            map_lane=SimpleNamespace(
                allowed_lane_ids=(7, 8), in_junction=False
            ),
            planning=SimpleNamespace(
                route=SimpleNamespace(
                    current_road_option="LANEFOLLOW",
                    next_macro_maneuver="turn_right",
                    next_macro_distance_m=18.0,
                ),
                traffic_control=SimpleNamespace(
                    signal_state="green", stop_target=stop_target,
                ),
            ),
        ),
    )
    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(), behavior=Behavior(), scenario=Scenario(),
        static_obstacle=object(), control_safety=object(), speed=object(),
        destination_speed=object(), reference_publication=object(),
        mpc_entry=object(),
    )
    reset_reasons = []
    stage_names = []
    result = pipeline.resolve_behavior_context(
        BehaviorContextRequest(
            adapter_output=adapter,
            local_map_snapshot="local-map",
            route_manager="route-manager",
            maneuver_manager="maneuver-manager",
            reference_provider=ReferenceProvider(),
            traffic_memory="traffic-memory",
            opportunistic_request="opportunistic-request",
            ego_location=SimpleNamespace(x=1.0, y=2.0),
            ego_yaw_rad=0.0,
            ego_speed_mps=4.0,
            planning_speed_mps=6.0,
            current_lane_id=7,
            sim_time_s=5.0,
            cruise_speed_mps=8.0,
            mpc_dt_s=0.1,
            lane_width_m=3.5,
            config={},
        ),
        resolve_actor_state=lambda **_kwargs: ("green", ""),
        project_stop_target=lambda **_kwargs: (float("inf"), False),
        attempt_turn_replan=lambda reason: (
            True, True, "replanned:" + reason
        ),
        reset_lane_change=lambda **kwargs: reset_reasons.append(kwargs),
        observe_stage_duration=lambda name, _seconds: stage_names.append(name),
    )

    assert result.route_behavior is route_behavior
    assert result.scenario_observation is scenario_observation
    assert result.conflict_resolution is conflict
    assert result.route_replan_attempted
    assert result.route_replan_succeeded
    assert result.route_replan_reason == "replanned:route_discontinuity"
    assert stage_names == [
        "sub_resolve_route_context",
        "sub_observe_planning_frame",
        "sub_resolve_conflicts",
    ]
    assert [entry[0] for entry in calls] == [
        "route", "scenario", "turn", "conflict", "release"
    ]
    assert reset_reasons == [{"reason": "maneuver_complete"}]


def test_pipeline_freezes_command_and_override_into_executable_behavior():
    calls = []
    command = SimpleNamespace(
        decision="lane_change_left",
        target_lane_id=8,
        phase="PREPARE_LANE_CHANGE_LEFT",
        opportunistic_lane_change_allowed=False,
        static_obstacle_result=SimpleNamespace(local_avoidance_active=False),
    )
    command_frame = SimpleNamespace(command=command)
    override = SimpleNamespace(
        decision="lane_follow",
        target_lane_id=7,
        phase="LANE_KEEP",
        stop_goal_active=False,
        reason="route turn suppressed by later lane change",
        reset_lane_change_reason="authorization_lost",
    )

    class Behavior:
        def produce_command_from_frame(self, request, **kwargs):
            calls.append(("command", request, kwargs))
            return command_frame

        def apply_overrides(self, request):
            calls.append(("override", request))
            assert request.route_turn_decision == ""
            assert request.scenario_speed_cap_active is False
            return override

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(), behavior=Behavior(), scenario=object(),
        static_obstacle="static-stage", control_safety=object(), speed=object(),
        destination_speed=object(), reference_publication=object(),
        mpc_entry=object(),
    )
    command_request = SimpleNamespace(current_lane_id=7)
    reset_reasons = []
    stage_names = []
    result = pipeline.resolve_executable_behavior(
        ExecutableBehaviorRequest(
            command_request=command_request,
            scenario_decision=SimpleNamespace(
                state="PREPARE_TURN",
                stop_goal_active=False,
                speed_cap_mps=3.0,
                behavior_override_decision="",
                behavior_override_lc_state="",
            ),
            lane_change_authorized=False,
            lane_change_gate_reason="not_authorized",
            lane_change_authorization_reason="route_not_required",
            prepare_reference_lock=True,
            lane_change_commitment_active=True,
            route_advanced_to_lane_change=True,
            route_current_road_option="LEFT",
            ego_in_junction=False,
            stop_goal_active=False,
            cruise_speed_mps=8.0,
            scenario_reason="prepare_turn",
        ),
        behavior_planner="behavior-planner",
        reference_map="reference-map",
        nearest_front_obstacles=lambda **_kwargs: {},
        attempt_replan=lambda _obstacle: None,
        object_track_id=lambda item: item.get("id"),
        reset_lane_change=lambda **kwargs: reset_reasons.append(kwargs),
        observe_stage_duration=lambda name, _seconds: stage_names.append(name),
    )

    assert result.command_frame is command_frame
    assert result.decision == "lane_follow"
    assert result.target_lane_id == 7
    assert result.phase == "LANE_KEEP"
    assert result.turn_prepare_speed_suppressed
    assert not result.scenario_speed_cap_active
    assert [entry[0] for entry in calls] == ["command", "override"]
    assert calls[0][2]["static_obstacle_stage"] == "static-stage"
    assert stage_names == ["sub_produce_behavior_command"]
    assert reset_reasons == [{"reason": "authorization_lost"}]


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


def test_pipeline_owns_nominal_behavior_destination_speed_sequence():
    calls = []
    decision = SimpleNamespace(
        maneuver="lane_follow",
        as_debug_fields=lambda: {"typed_behavior": True},
    )
    behavior_result = SimpleNamespace(
        decision=decision,
        mutable_diagnostics=lambda: {"behavior_stage": "ready"},
    )
    behavior_reference = SimpleNamespace(
        destination_state=(1.0, 2.0, 4.0, 0.0, 7),
        reference_samples=({"x_ref_m": 1.0, "y_ref_m": 2.0},),
        behavior_stage_result=behavior_result,
        reference_debug={"reference_source": "test"},
        speed_plan="speed-plan",
        cav_resolution="cav-result",
        failure_reason="",
    )

    class BehaviorExecution:
        def run(self, request, *, planner):
            calls.append(("behavior", request))
            assert planner is planner_callback
            return behavior_reference

    destination_application = SimpleNamespace(
        stage=SimpleNamespace(constraint="destination-cap"),
        behavior_stage_result=behavior_result,
        finished=True,
        mutable_destination_state=lambda: [3.0, 4.0, 2.0, 0.0, 7],
        mutable_reference=lambda: [{"x_ref_m": 3.0, "y_ref_m": 4.0}],
        mutable_reference_debug=lambda: {"destination_applied": True},
    )

    class Destination:
        def apply(self, **kwargs):
            calls.append(("destination", kwargs))
            return destination_application

    speed_frame = SimpleNamespace(
        target=SimpleNamespace(target_mps=2.0),
        ceiling=SimpleNamespace(
            destination_state=(3.0, 4.0, 2.0, 0.0, 7),
            reference_samples=({"x_ref_m": 3.0, "y_ref_m": 4.0},),
        ),
        trace_fields=lambda: {"speed_owner": "destination"},
    )

    class Speed:
        def resolve_frame(self, **kwargs):
            calls.append(("speed", kwargs))
            return speed_frame

    def planner_callback(**_kwargs):
        raise AssertionError("the execution boundary owns this callback")

    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(),
        behavior=object(), scenario=object(), static_obstacle=object(),
        control_safety=object(), speed=Speed(),
        destination_speed=Destination(), reference_publication=object(),
        mpc_entry=object(), behavior_reference_execution=BehaviorExecution(),
    )
    durations = []
    result = pipeline.resolve_nominal_plan(
        NominalPlanningRequest(
            behavior_request="behavior-request",
            route_status="route-status",
            route_revision="revision-1",
            ego_speed_mps=3.0,
            current_state=(0.0, 0.0, 3.0, 0.0),
            fallback_lane_id=7,
        ),
        planner=planner_callback,
        observe_stage_duration=lambda name, seconds: durations.append(
            (name, seconds)
        ),
    )

    assert [entry[0] for entry in calls] == [
        "behavior", "destination", "speed"
    ]
    assert [entry[0] for entry in durations] == [
        "execute_behavior_reference", "apply_destination", "resolve_speed"
    ]
    assert result.mission_finished
    assert result.behavior is decision
    assert result.speed_target.target_mps == pytest.approx(2.0)
    assert result.cav_resolution == "cav-result"
    assert result.mutable_behavior_debug()["typed_behavior"] is True
    assert result.mutable_reference_debug()["speed_owner"] == "destination"


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
        def apply_behavior_mode_transition(self, **kwargs):
            calls.append(("mode", kwargs))
            return "mode-transition"

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
        front_gap_actor_id="",
        candidate_status="feasible",
        candidate_name="lane_follow",
        candidate_reason="",
        reset_control_buffer=lambda **_kwargs: None,
    )
    assert result.publication is publication
    assert result.control_context == "control-context"
    assert result.mode_transition_reason == "mode-transition"
    assert result.trace_fields()["mpc_entry_allowed"] is True
    assert [call[0] for call in calls] == [
        "publish", "mode", "entry", "context"
    ]


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
    # Classification begins 10 m behind ego while execution begins at ego.
    # Station bounds must be rebased, not copied between these polylines.
    classification_reference = [
        {"x_ref_m": 0.0, "y_ref_m": float(y)} for y in range(-10, 61, 2)
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

    result = CAVInteractionStage.resolve(
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
    assert result.constraint_corridor is not None
    # Stage C and Stage D now share the executed-reference coordinate owner.
    # The old classification-station leak produced a cap above 20 m here.
    assert result.constraint_corridor is result.corridor
    assert result.corridor.s_hi[1] < 10.0
    assert longitudinal_rows[0].upper == pytest.approx(
        result.corridor.s_hi[1]
    )


def test_resolve_cav_interaction_defaults_constraint_reference_to_reference_samples():
    reference = [{"x_ref_m": 0.0, "y_ref_m": float(y)} for y in range(0, 61, 2)]
    crosser = {
        "id": "x", "x": -8.0, "y": 18.0, "v": 7.0, "psi": 0.0,
        "predicted_trajectory": [
            {"x": -8.0 + 0.7 * k, "y": 18.0} for k in range(21)
        ],
    }
    ego_location = SimpleNamespace(x=0.0, y=0.0)

    result = CAVInteractionStage.resolve(
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


def test_make_gap_gate_margin_is_non_positive_once_peer_converges_into_corridor():
    reference = [{"x_ref_m": 0.0, "y_ref_m": float(y)} for y in range(0, 61, 2)]
    ego_location = SimpleNamespace(x=0.0, y=0.0)
    # Same converging-peer geometry as
    # test_ego_opens_a_gap_for_a_cooperative_merging_cav in
    # test_cav_conflict_integration.py: cav one lane over, merging toward
    # ego's line within the horizon, committed earlier than ego.
    cav_path = tuple(
        (0.1 * k, 3.4 - 0.3 * k, 12.0 + 9.0 * 0.1 * k, 9.0)
        for k in range(21)
    )
    cav = CavIntent(
        actor_id=2, position_xy=(3.4, 12.0),
        claim=ResourceClaim(kind="lane_change", resource_id="lane_change",
                             committed_at_s=3.0, active=True),
        heading_rad=1.5707963267948966, speed_mps=9.0, planned_path=cav_path,
    )
    my_claim = ResourceClaim(kind="lane_change", resource_id="lane_change",
                              committed_at_s=10.0, active=True)

    result = CAVInteractionStage.resolve(
        reference_samples=reference,
        ego_location=ego_location, ego_yaw_rad=1.5707963267948966,
        ego_speed_mps=9.0, actor_id=1, claim=my_claim,
        obstacle_snapshots=[], cav_intents=[cav], latch_state={},
        horizon_steps=20, dt_s=0.1,
    )

    assert result.diagnostics["roles"].get("2") == "make_gap"
    assert any(h < 1.0e8 for h in result.corridor.s_hi)
    margins = result.diagnostics["make_gap_gate_margin_m"]
    assert margins["2"] <= 0.0


def test_make_gap_gate_margin_stays_positive_for_a_gentle_real_world_merge():
    # Same 0.12 m/s lateral rate as
    # test_gradual_real_world_merge_convergence_stays_below_corridor_gate
    # in test_spatiotemporal_corridor.py, derived from a real
    # cpx_two_cav_merge_conflict run where cav_total_qp_row_count was 0 for
    # all 686 frames.  The margin should read positive (gate not open) so a
    # future real run can be told apart from one where it is stuck at ~0.
    reference = [{"x_ref_m": 0.0, "y_ref_m": float(y)} for y in range(0, 61, 2)]
    ego_location = SimpleNamespace(x=0.0, y=0.0)
    lateral_rate_mps = 4.2 / 34.25
    cav_path = tuple(
        (0.1 * k, 3.4 - lateral_rate_mps * (0.1 * k), 12.0 + 9.0 * 0.1 * k, 9.0)
        for k in range(21)
    )
    cav = CavIntent(
        actor_id=2, position_xy=(3.4, 12.0),
        claim=ResourceClaim(kind="lane_change", resource_id="lane_change",
                             committed_at_s=3.0, active=True),
        heading_rad=1.5707963267948966, speed_mps=9.0, planned_path=cav_path,
    )
    my_claim = ResourceClaim(kind="lane_change", resource_id="lane_change",
                              committed_at_s=10.0, active=True)

    result = CAVInteractionStage.resolve(
        reference_samples=reference,
        ego_location=ego_location, ego_yaw_rad=1.5707963267948966,
        ego_speed_mps=9.0, actor_id=1, claim=my_claim,
        obstacle_snapshots=[], cav_intents=[cav], latch_state={},
        horizon_steps=20, dt_s=0.1,
    )

    assert result.diagnostics["roles"].get("2") == "make_gap"
    assert all(h >= 1.0e8 for h in result.corridor.s_hi)
    margins = result.diagnostics["make_gap_gate_margin_m"]
    assert margins["2"] > 0.0
