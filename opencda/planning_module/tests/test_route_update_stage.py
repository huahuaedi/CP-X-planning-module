from types import SimpleNamespace
from unittest.mock import Mock

from pipeline.route_manager import LaneClosureRouteResult
from pipeline.route_update_stage import RouteUpdateRequest, RouteUpdateStage


def _request(route_manager, reset, **overrides):
    values = dict(
        route_manager=route_manager,
        ego_location=SimpleNamespace(x=3.0, y=4.0, z=0.5),
        cp_payload={"lane_events": [{
            "id": "closure-1", "type": "lane_closure",
            "position": {"x": 8.0, "y": 4.0},
        }]},
        stop_goal_active=False,
        lane_closure_reroute_enabled=True,
        reset_for_route_revision=reset,
    )
    values.update(overrides)
    return RouteUpdateRequest(**values)


def test_route_change_resets_downstream_lifecycle_once():
    route_manager = Mock()
    route_manager.apply_lane_closures.return_value = LaneClosureRouteResult(
        attempted=True,
        success=True,
        route_changed=True,
        reason="admap_route_replanned:cp_lane_closure:closure-1",
        handled_message_ids=("closure-1",),
        blocked_lane_ids=(17,),
    )
    reset = Mock()

    frame = RouteUpdateStage().run(_request(route_manager, reset))

    assert frame.route_replan_attempted
    assert frame.route_replan_succeeded
    assert not frame.stop_goal_active
    assert frame.trace_fields()["cp_lane_closure_blocked_lane_ids"] == [17]
    route_manager.apply_lane_closures.assert_called_once_with(
        messages=[{
            "id": "closure-1", "type": "lane_closure",
            "position": {"x": 8.0, "y": 4.0},
        }],
        start_point={"x": 3.0, "y": 4.0, "z": 0.5},
    )
    reset.assert_called_once_with(reason="cp_lane_closure_route_replanned")


def test_failed_route_update_becomes_stop_required_data():
    route_manager = Mock()
    route_manager.apply_lane_closures.return_value = LaneClosureRouteResult(
        attempted=True,
        success=False,
        route_changed=False,
        reason="no_alternate_route",
    )
    reset = Mock()

    frame = RouteUpdateStage().run(_request(route_manager, reset))

    assert frame.stop_goal_active
    assert not frame.route_replan_succeeded
    assert frame.route_replan_reason == "no_alternate_route"
    reset.assert_not_called()


def test_route_manager_exception_is_a_typed_failure_not_control_flow():
    route_manager = Mock()
    route_manager.apply_lane_closures.side_effect = RuntimeError("boom")

    frame = RouteUpdateStage().run(_request(route_manager, Mock()))

    assert frame.lane_closure.attempted
    assert not frame.lane_closure.success
    assert frame.stop_goal_active
    assert "cp_lane_closure_apply_exception" in frame.route_replan_reason


def test_disabled_route_updates_are_explicit_no_ops():
    route_manager = Mock()
    reset = Mock()

    frame = RouteUpdateStage().run(_request(
        route_manager,
        reset,
        lane_closure_reroute_enabled=False,
    ))

    assert not frame.route_replan_attempted
    assert frame.route_replan_reason == "route_replan_not_requested"
    assert frame.trace_fields()["cp_lane_closure_reason"] == (
        "cp_lane_closure_disabled"
    )
    route_manager.apply_lane_closures.assert_not_called()
    reset.assert_not_called()
