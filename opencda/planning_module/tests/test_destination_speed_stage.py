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


def test_mission_completes_from_buffered_stop_even_when_route_never_reports_arrival():
    # Regression for the single_intersection_town06_carla right-turn case:
    # route_reached_distance_m (1.0m) is tighter than destination_stop_buffer_m
    # (1.5m), so a vehicle that latches its terminal stop at the buffer can
    # sit at ~1.5m forever with route_status.reached_destination staying
    # False -- remaining distance can never close the last 0.5m once speed
    # is latched to zero. mission_complete must not depend on
    # reached_destination; the buffered-stop + low-speed path is sufficient.
    stage = DestinationSpeedStage(
        config={"destination_stop_buffer_m": 1.5},
        speed_planner=SpeedTargetPlanner(),
    )
    approaching = stage.evaluate(
        route_status=_status(remaining_m=20.0, reached=False),
        route_revision="route-1",
        ego_speed_mps=10.0,
    )
    assert not approaching.mission_complete

    result = stage.evaluate(
        route_status=_status(remaining_m=1.497, reached=False),
        route_revision="route-1",
        ego_speed_mps=0.136,
    )
    assert result.stop_latched
    assert not result.reached_destination
    assert result.mission_complete

    # Latched, not recomputed: a later noisy uptick in measured speed must
    # not un-complete a mission already achieved.
    later = stage.evaluate(
        route_status=_status(remaining_m=1.497, reached=False),
        route_revision="route-1",
        ego_speed_mps=0.3,
    )
    assert later.mission_complete


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
    assert stage.mission_complete
    result = stage.evaluate(
        route_status=_status(remaining_m=100.0),
        route_revision="route-2",
        ego_speed_mps=1.0,
    )
    assert not result.stop_latched
    assert not result.approach_active
    assert not result.mission_complete


def test_mission_completion_latches_only_after_vehicle_stops():
    stage = DestinationSpeedStage(
        config={
            "destination_stop_buffer_m": 1.5,
            "destination_stop_complete_speed_mps": 0.15,
        },
        speed_planner=SpeedTargetPlanner(),
    )

    moving = stage.evaluate(
        route_status=_status(remaining_m=1.0),
        route_revision="route-1",
        ego_speed_mps=0.5,
    )
    stopped = stage.evaluate(
        route_status=_status(remaining_m=1.0),
        route_revision="route-1",
        ego_speed_mps=0.1,
    )
    noisy = stage.evaluate(
        route_status=_status(remaining_m=1.0),
        route_revision="route-1",
        ego_speed_mps=0.2,
    )

    assert moving.stop_latched
    assert not moving.mission_complete
    assert stopped.mission_complete
    assert noisy.mission_complete


def test_missing_route_cannot_complete_mission_from_zero_default_distance():
    stage = DestinationSpeedStage(
        config={"destination_stop_buffer_m": 1.5},
        speed_planner=SpeedTargetPlanner(),
    )

    result = stage.evaluate(
        route_status=_status(remaining_m=0.0, found=False),
        route_revision="route-1",
        ego_speed_mps=0.0,
    )

    assert not result.stop_latched
    assert not result.mission_complete


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
    assert result.stage.mission_complete
    assert result.behavior_stage_result is stopped_behavior
    assert result.destination_state[-1] == 7
    assert result.reference_debug["reference_source"] == "persistent_bounded_safe_stop"
    assert result.reference_debug["destination_mission_complete"]
