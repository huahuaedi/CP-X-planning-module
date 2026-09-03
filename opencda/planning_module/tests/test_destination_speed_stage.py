from types import SimpleNamespace

from pipeline.destination_speed_stage import DestinationSpeedStage
from pipeline.speed_planner import SpeedTargetPlanner


def _status(*, remaining_m, reached=False, found=True):
    return SimpleNamespace(
        route_found=found,
        reached_destination=reached,
        remaining_distance_m=remaining_m,
    )


def test_destination_stage_latches_approach_and_terminal_stop():
    stage = DestinationSpeedStage(
        config={
            "fallback_safe_stop_deceleration_mps2": 2.5,
            "destination_stop_buffer_m": 1.5,
        },
        speed_planner=SpeedTargetPlanner(),
    )
    approaching = stage.evaluate(
        route_status=_status(remaining_m=20.0),
        route_revision="route-1",
        ego_speed_mps=10.0,
    )
    assert approaching.approach_active
    assert not approaching.stop_latched
    assert approaching.constraint is not None

    terminal = stage.evaluate(
        route_status=_status(remaining_m=1.0),
        route_revision="route-1",
        ego_speed_mps=1.0,
    )
    assert terminal.stop_latched
    assert stage.stop_latched


def test_destination_stage_route_change_releases_old_stop_latch():
    stage = DestinationSpeedStage(
        config={"destination_stop_buffer_m": 1.5},
        speed_planner=SpeedTargetPlanner(),
    )
    stage.evaluate(
        route_status=_status(remaining_m=0.0, reached=True),
        route_revision="route-1",
        ego_speed_mps=0.0,
    )
    result = stage.evaluate(
        route_status=_status(remaining_m=100.0),
        route_revision="route-2",
        ego_speed_mps=1.0,
    )
    assert not result.stop_latched
    assert not result.approach_active


def test_destination_apply_owns_terminal_reference_and_behavior():
    behavior_result = SimpleNamespace(
        decision=SimpleNamespace(),
        mutable_diagnostics=lambda: {"target_lane_id": 7},
    )
    stopped_behavior = SimpleNamespace(decision=SimpleNamespace())

    class Fallback:
        def bounded_safe_stop(self, **_kwargs):
            return SimpleNamespace(
                reason="route_destination_reached",
                mutable_trajectory=lambda: [
                    {"x_ref_m": 1.0, "y_ref_m": 2.0, "heading_rad": 0.0},
                    {"x_ref_m": 2.0, "y_ref_m": 2.0, "heading_rad": 0.0},
                ],
            )

    class Behavior:
        def destination_stop(self, result):
            assert result is behavior_result
            return stopped_behavior

    stage = DestinationSpeedStage(
        config={"destination_stop_complete_speed_mps": 0.15},
        speed_planner=SpeedTargetPlanner(),
        fallback_manager=Fallback(),
        behavior_stage=Behavior(),
    )
    result = stage.apply(
        route_status=_status(remaining_m=0.0, reached=True),
        route_revision="route-1",
        ego_speed_mps=0.1,
        current_state=(0.0, 0.0, 0.1, 0.0),
        destination_state=(),
        reference_samples=(),
        behavior_stage_result=behavior_result,
        reference_debug={},
        fallback_lane_id=3,
    )
    assert result.finished
    assert result.behavior_stage_result is stopped_behavior
    assert result.destination_state[-1] == 7
    assert result.reference_debug["reference_source"] == "persistent_bounded_safe_stop"
