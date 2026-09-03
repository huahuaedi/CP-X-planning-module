from types import SimpleNamespace

from pipeline.behavior_reference_execution_stage import (
    BehaviorReferenceExecutionStage,
    BehaviorReferenceRequest,
)


def _request():
    return BehaviorReferenceRequest(
        ego_location=SimpleNamespace(x=1.0, y=2.0),
        ego_yaw_rad=0.0,
        ego_speed_mps=3.0,
        requested_speed_mps=5.0,
        object_snapshots=(),
        stop_goal_active=False,
        cp_payload={},
        current_state=(1.0, 2.0, 3.0, 0.0),
        sim_time_s=4.0,
        route_revision="route:1",
    )


def test_success_returns_one_typed_result():
    stage = BehaviorReferenceExecutionStage(
        reference_provider=object(), fallback_manager=object(),
        behavior_stage=object(), lane_id_at_location=lambda _location: 12,
    )
    behavior = SimpleNamespace(decision="lane_follow")
    result = stage.run(
        _request(),
        planner=lambda **_kwargs: (
            [3.0, 4.0, 5.0, 0.0, 12],
            [{"x_ref_m": 1.0, "y_ref_m": 2.0}],
            behavior,
            {"reference_source": "lane"},
            "speed-plan",
        ),
    )
    assert result.behavior_stage_result is behavior
    assert result.speed_plan == "speed-plan"
    assert result.failure_reason == ""


def test_failure_is_resolved_inside_stage():
    generated = SimpleNamespace(
        samples=({"x_ref_m": 1.0, "y_ref_m": 2.0, "heading_rad": 0.0},),
        destination_state=(1.0, 2.0, 0.0, 0.0, 12),
    )
    fallback = SimpleNamespace(
        mode="bounded_safe_stop",
        target_speed_mps=0.0,
        reason="behavior_reference:pipeline_exception",
        mutable_trajectory=lambda: [
            {"x_ref_m": 1.0, "y_ref_m": 2.0, "heading_rad": 0.0}
        ],
    )
    finalized = SimpleNamespace(decision="fallback")

    class Provider:
        def lane_fallback_reference(self, **_kwargs):
            return generated

    class Fallback:
        def resolve(self, **kwargs):
            assert kwargs["route_revision"] == "route:1"
            return fallback

    class Behavior:
        def finalize(self, **kwargs):
            assert kwargs["phase"] == "FALLBACK"
            return finalized

    stage = BehaviorReferenceExecutionStage(
        reference_provider=Provider(), fallback_manager=Fallback(),
        behavior_stage=Behavior(), lane_id_at_location=lambda _location: 12,
    )

    def fail(**_kwargs):
        raise RuntimeError("broken reference")

    result = stage.run(_request(), planner=fail)
    assert result.behavior_stage_result is finalized
    assert result.speed_plan is None
    assert result.failure_reason == "broken reference"
    assert result.reference_debug["pipeline_error"] == "broken reference"
