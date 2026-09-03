from types import SimpleNamespace

from pipeline.candidate_selection_stage import (
    CandidateArbitrationRequest,
    CandidateSelectionRequest,
    CandidateSelectionStage,
)
from pipeline.candidate_evaluation import CandidateSelectionResult
from pipeline.speed_planner import SpeedPlan


def _post_selection_stage(config=None):
    provider = SimpleNamespace(
        builder=SimpleNamespace(discrete_curvature_1pm=lambda _rows: 0.1)
    )
    return CandidateSelectionStage(
        evaluator=object(), provider=provider,
        maneuver_manager=object(), reference_pipeline=object(),
        fallback_manager=object(),
        static_obstacle_stage=SimpleNamespace(target_lane_id=None),
        mpc=object(), config=dict(config or {}), map_epoch="admap",
        normal_clearance_m=2.0, static_clearance_m=1.0,
        risk_hysteresis_margin_m=0.2, strict_ownership=True,
        target_speed_mps=8.0,
    )


def test_post_selection_normalizes_lane_change_without_speed_override():
    stage = _post_selection_stage()
    speed = SpeedPlan(target_speed_mps=6.0, speed_cap_mps=6.0,
                      stop_goal_active=False)
    result = stage.finalize_selected_frame(
        decision="lane_change_left", lane_change_state="LANE_KEEP",
        reference=({}, {}),
        selected_diagnostics={"lane_change_phase": "executing"},
        reference_diagnostics={}, ego_speed_mps=5.0,
        scenario_stop_required=False, speed_plan=speed,
        turn_prepare_speed_suppressed=False,
    )
    assert result.lane_change_state == "EXECUTE_LANE_CHANGE_LEFT"
    assert result.speed_plan is speed
    assert result.speed_constraints == ()
    assert result.diagnostics["lane_change_longitudinal_authority"] == "SpeedPlanner"


def test_post_selection_turn_submits_named_speed_constraint():
    stage = _post_selection_stage({"full_intersection_turn_speed_cap_mps": 2.2})
    speed = SpeedPlan(target_speed_mps=6.0, speed_cap_mps=6.0,
                      stop_goal_active=False)
    result = stage.finalize_selected_frame(
        decision="intersection_turn_right", lane_change_state="LANE_KEEP",
        reference=({}, {}), selected_diagnostics={}, reference_diagnostics={},
        ego_speed_mps=5.0, scenario_stop_required=False, speed_plan=speed,
        turn_prepare_speed_suppressed=False,
    )
    assert result.speed_plan.target_speed_mps == 2.2
    assert result.speed_plan.limiting_owner == "selected_turn_cap"
    assert result.speed_constraints[0].owner == "selected_turn_cap"


def test_arbitrate_owns_intent_selection_and_finalization():
    stage = _post_selection_stage()
    captured = {}
    stage.build_intents = lambda **kwargs: captured.setdefault("intent", kwargs) or []

    def select(request, **kwargs):
        captured["selection"] = request
        return CandidateSelectionResult(
            decision="lane_follow", target_lane_id=12, target_speed_mps=6.0,
            reference=({"x_ref_m": 2.0, "y_ref_m": 0.0},),
            destination_state=(2.0, 0.0, 6.0, 0.0, 12), diagnostics={},
        )

    stage.run = select
    speed = SpeedPlan(target_speed_mps=6.0, speed_cap_mps=6.0,
                      stop_goal_active=False)
    context = SimpleNamespace(
        baseline_reference=({"x_ref_m": 1.0, "y_ref_m": 0.0},),
        baseline_destination_state=(1.0, 0.0, 6.0, 0.0, 12),
        baseline_debug={},
    )
    authorization = SimpleNamespace(
        allowed=False, target_lane_id=0, direction="",
        distance_to_maneuver_m=float("inf"),
    )
    result = stage.arbitrate(
        CandidateArbitrationRequest(
            reference_context=context, selected_decision="lane_follow",
            selected_target_lane_id=12, current_lane_id=12,
            target_speed_mps=6.0, candidate_lane_ids=(12,),
            lane_safety_scores={12: 1.0}, lane_prediction_risks={},
            stop_goal_active=False, traffic_stop_active=False,
            lane_change_authorization=authorization,
            opportunistic_lane_change_allowed=False, stop_target=None,
            local_obstacle_avoidance_active=False, ego_speed_mps=5.0,
            lane_width_m=3.5, baseline_lane_change_state="LANE_KEEP",
            current_state=(0.0, 0.0, 5.0, 0.0),
            ego_location=SimpleNamespace(x=0.0, y=0.0), ego_yaw_rad=0.0,
            object_snapshots=(), prediction_trajectories={},
            current_acceleration_mps2=0.0, current_steering_rad=0.0,
            route_required=False, scenario_stop_required=False,
            speed_plan=speed, turn_prepare_speed_suppressed=False,
        ),
        sim_time_s=1.0, route_revision="route:1",
        road_envelope=lambda: None,
        validate_contract=lambda **_kwargs: None,
        validate_locked_reference=lambda **_kwargs: None,
    )

    assert captured["intent"]["selected_decision"] == "lane_follow"
    assert captured["selection"].reference_context is context
    assert result.decision == "lane_follow"
    assert result.mutable_destination_state()[0] == 2.0


def test_no_candidates_returns_typed_baseline_and_runs_completion_release():
    maneuver = SimpleNamespace(
        route_lane_change_edge_completed=False,
        lane_change=SimpleNamespace(
            phase="idle", option="", source_lane_id=1, target_lane_id=1,
            progress=0.0, stabilization_frames=0, completion_debug={},
        ),
    )
    stage = CandidateSelectionStage(
        evaluator=object(),
        provider=object(),
        maneuver_manager=maneuver,
        reference_pipeline=object(),
        fallback_manager=object(),
        static_obstacle_stage=SimpleNamespace(target_lane_id=None),
        mpc=object(),
        config={},
        map_epoch="admap",
        normal_clearance_m=2.0,
        static_clearance_m=1.0,
        risk_hysteresis_margin_m=0.2,
        strict_ownership=True,
        target_speed_mps=8.0,
    )
    releases = []
    stage.set_lane_change_lifecycle(SimpleNamespace(
        release_completed=lambda **_kwargs: releases.append(True) or ""
    ))
    request = CandidateSelectionRequest(
        intents=(),
        reference_context=SimpleNamespace(local_map=None),
        baseline_lane_change_state="LANE_FOLLOW",
        baseline_decision="lane_follow",
        baseline_target_lane_id=12,
        baseline_speed_mps=6.0,
        baseline_reference=({"x_ref_m": 1.0, "y_ref_m": 2.0},),
        baseline_destination_state=(1.0, 2.0, 6.0, 0.0, 12),
        current_state=(0.0, 0.0, 2.0, 0.0),
        current_lane_id=12,
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.0,
        ego_speed_mps=2.0,
        object_snapshots=(),
        prediction_trajectories={},
    )
    result = stage.run(
        request,
        sim_time_s=1.0,
        route_revision="route:1",
        road_envelope=lambda: None,
        validate_contract=lambda **_kwargs: None,
        validate_locked_reference=lambda **_kwargs: None,
    )

    assert releases == [True]
    assert result.decision == "lane_follow"
    assert result.target_lane_id == 12
    assert result.target_speed_mps == 6.0
    assert result.mutable_diagnostics()["candidate_pipeline_count"] == 0
