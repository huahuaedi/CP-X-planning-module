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
