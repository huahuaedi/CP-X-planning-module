from types import SimpleNamespace

from pipeline.candidate_selection_stage import (
    CandidateSelectionRequest,
    CandidateSelectionStage,
)


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
    releases = []
    result = stage.run(
        request,
        sim_time_s=1.0,
        route_revision="route:1",
        release_completed=lambda: releases.append(True) or "",
        road_envelope=lambda: None,
        validate_contract=lambda **_kwargs: None,
        validate_locked_reference=lambda **_kwargs: None,
    )

    assert releases == [True]
    assert result.decision == "lane_follow"
    assert result.target_lane_id == 12
    assert result.target_speed_mps == 6.0
    assert result.mutable_diagnostics()["candidate_pipeline_count"] == 0
