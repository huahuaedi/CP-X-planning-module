import math
from types import SimpleNamespace

from pipeline.behavior_stage import (
    BehaviorCandidateRequest,
    BehaviorOverrideRequest,
    BehaviorStage,
    ConflictResolutionRequest,
    OpportunisticLaneChangeRequest,
    RouteLaneChangeRequest,
)
from pipeline.maneuver_manager import ManeuverManager


def _route_lane_change_request(**overrides):
    values = dict(
        route_lane_change_allowed=True,
        current_lane_id=10,
        route_required_lane_id=11,
        next_macro_maneuver="lane_change_right",
        current_road_option="LANEFOLLOW",
        remaining_distance_m=20.0,
        available_lane_ids=(10, 11),
        lane_safety_scores={10: 1.0, 11: 1.0},
        lane_prediction_risks={},
        preparation_start_distance_m=45.0,
        latest_start_distance_m=12.0,
        target_safety_threshold=0.65,
        require_adjacent=True,
        explicit_lane_change_start_distance_m=30.0,
        adjacent_lane_directions={11: "right"},
        topology_current_lane_id=10,
        topology_target_lane_id=11,
        topology_lane_offset=-1,
        topology_target_in_local_frame=True,
        route_geometry_direction="right",
        route_geometry_distance_m=20.0,
    )
    values.update(overrides)
    return RouteLaneChangeRequest(**values)


class _FakeSnapshot:
    def __init__(self, *, frame_id, ego_lane_id, target_lane_id, offset,
                 in_frame, lane_by_offset):
        self.frame_id = frame_id
        self.ego_lane_id = ego_lane_id
        self.route_target_lane_id = target_lane_id
        self.route_target_offset = offset
        self.route_target_in_frame = in_frame
        self._lane_by_offset = dict(lane_by_offset)

    def lane_at_ego_station(self, offset):
        return self._lane_by_offset.get(int(offset), 0)


class _FakeRouteManager:
    def __init__(self, *, direction, distance_m, reason, edge_id):
        self._result = (direction, distance_m, reason)
        self._edge_id = edge_id
        self.calls = []

    def upcoming_lane_change(self, *, ego_x_m, ego_y_m, ego_heading_rad, lookahead_m):
        self.calls.append(float(lookahead_m))
        return self._result

    def upcoming_lane_change_edge_id(self, *, lookahead_m):
        return self._edge_id


class _FakeTurnRouteManager:
    def __init__(self, turn, alignment):
        self.turn = turn
        self.alignment = alignment
        self.lookahead_m = None

    def upcoming_turn(self, **kwargs):
        self.lookahead_m = float(kwargs["lookahead_m"])
        return self.turn

    def route_alignment(self, **kwargs):
        return self.alignment


def _prepare(**overrides):
    kwargs = dict(
        route_manager=_FakeRouteManager(
            direction="right", distance_m=18.0,
            reason="route_geometry_lane_change_ahead", edge_id="edge-7",
        ),
        local_map_snapshot=_FakeSnapshot(
            frame_id=5, ego_lane_id=100, target_lane_id=340154, offset=-1,
            in_frame=True, lane_by_offset={-1: 340155},
        ),
        route_summary={"lane_change_direction": "right"},
        current_lane_id=100,
        route_optimal_lane_id=101,
        route_found=True,
        next_macro_maneuver="LANE-CHANGE-RIGHT",
        current_road_option="LANEFOLLOW",
        next_macro_distance_m=30.0,
        available_lane_ids=(100, 340155),
        lane_safety_scores={100: 1.0, 340155: 1.0},
        lane_prediction_risks={},
        ego_x_m=0.0,
        ego_y_m=0.0,
        ego_heading_rad=0.0,
        ego_speed_mps=5.0,
        config={},
    )
    kwargs.update(overrides)
    return BehaviorStage.prepare_route_lane_change(**kwargs)


def test_prepare_route_lane_change_resolves_physical_adjacent_target():
    context = _prepare()

    # Topology named a downstream AD segment; the physically adjacent corridor
    # in the local frame wins.
    assert context.topology_target_lane_id == 340154
    assert context.physical_target_lane_id == 340155
    assert context.request.route_required_lane_id == 340155
    assert context.request.topology_target_lane_id == 340155
    assert context.request.next_macro_maneuver == "lane_change_right"
    assert context.request.route_geometry_distance_m == 18.0
    assert context.geometry_direction == "right"
    assert context.edge_id == "edge-7"


def test_prepare_route_lane_change_drops_non_finite_geometry_distance():
    context = _prepare(
        route_manager=_FakeRouteManager(
            direction="", distance_m=float("inf"),
            reason="route_geometry_no_lane_change_in_lookahead", edge_id="",
        ),
    )

    assert context.request.route_geometry_distance_m is None
    assert math.isinf(context.geometry_distance_m)
    assert context.request.next_macro_maneuver == "lane_change_right"


def test_prepare_route_lane_change_scales_trigger_windows_with_speed():
    slow = _prepare(ego_speed_mps=0.0)
    fast = _prepare(ego_speed_mps=10.0)

    assert slow.request.preparation_start_distance_m == 45.0
    assert fast.request.preparation_start_distance_m == 150.0
    assert fast.preparation_start_distance_m == fast.request.preparation_start_distance_m


def test_turn_context_keeps_route_geometry_as_authoritative_source():
    route = _FakeTurnRouteManager(
        ("right", 18.0, "route_geometry_turn_ahead"),
        (0.04, 0.2, "route_alignment"),
    )

    context = BehaviorStage.prepare_turn_scenario_context(
        route_manager=route,
        ego_x_m=1.0,
        ego_y_m=2.0,
        ego_heading_rad=0.1,
        cruise_speed_mps=8.0,
        next_macro_maneuver="Turn Left",
        next_macro_distance_m=40.0,
        config={},
    )

    assert context.direction == "right"
    assert context.distance_m == 18.0
    assert context.reason == "route_geometry_turn_ahead"
    assert context.exit_alignment_valid
    assert context.exit_aligned
    assert route.lookahead_m is not None


def test_turn_context_uses_route_macro_only_when_geometry_has_no_turn():
    route = _FakeTurnRouteManager(
        ("", float("inf"), "route_geometry_no_turn_in_lookahead"),
        (float("nan"), float("nan"), "route_alignment_unavailable"),
    )

    context = BehaviorStage.prepare_turn_scenario_context(
        route_manager=route,
        ego_x_m=0.0,
        ego_y_m=0.0,
        ego_heading_rad=0.0,
        cruise_speed_mps=8.0,
        next_macro_maneuver="Turn Right",
        next_macro_distance_m=27.0,
        config={},
    )

    assert context.direction == "right"
    assert context.distance_m == 27.0
    assert context.reason == "admap_route_macro_direction_fallback"
    assert not context.exit_alignment_valid
    assert not context.exit_aligned


def test_turn_context_marks_route_advance_to_lane_change():
    route = _FakeTurnRouteManager(
        ("", float("inf"), "route_geometry_no_turn_in_lookahead"),
        (0.0, 0.0, "route_alignment"),
    )

    context = BehaviorStage.prepare_turn_scenario_context(
        route_manager=route,
        ego_x_m=0.0,
        ego_y_m=0.0,
        ego_heading_rad=0.0,
        cruise_speed_mps=8.0,
        next_macro_maneuver="lane-change-left",
        next_macro_distance_m=12.0,
        config={},
    )

    assert context.route_advanced_to_lane_change


def test_behavior_stage_produces_typed_decision_and_separate_diagnostics():
    result = BehaviorStage().finalize(
        maneuver="lane_change_right",
        phase="EXECUTING",
        source_lane_id=10,
        target_lane_id=11,
        requested_speed_mps=8.0,
        stop_required=False,
        route_required=True,
        traffic_signal_state="green",
        boundary_recovery_active=False,
        stop_target=None,
        diagnostics={"lane_safety_scores": {11: 1.0}},
    )
    assert result.decision.direction == "right"
    assert result.decision.route_required
    assert result.diagnostics["decision"] == "lane_change_right"
    assert "lane_safety_scores" not in result.decision.as_debug_fields()


def test_destination_stop_is_a_typed_behavior_transition():
    stage = BehaviorStage()
    moving = stage.finalize(
        maneuver="lane_follow",
        phase="LANE_KEEP",
        source_lane_id=10,
        target_lane_id=10,
        requested_speed_mps=12.0,
        stop_required=False,
        route_required=False,
        traffic_signal_state="unknown",
        boundary_recovery_active=False,
        stop_target=None,
    )
    stopped = stage.destination_stop(moving)
    assert stopped.decision.maneuver == "destination_stop"
    assert stopped.decision.stop_required
    assert stopped.decision.requested_speed_mps == 0.0


def test_behavior_stage_owns_route_authorization_latch():
    stage = BehaviorStage()
    manager = ManeuverManager()
    authorized = stage.authorize_route_lane_change(
        _route_lane_change_request(), maneuver_manager=manager
    )
    latched = stage.authorize_route_lane_change(
        _route_lane_change_request(
            next_macro_maneuver="lane_follow",
            route_geometry_direction="",
            route_geometry_distance_m=None,
        ),
        maneuver_manager=manager,
    )

    assert authorized.allowed
    assert latched.allowed
    assert latched.reason == "route_lane_change_authorization_latched"


def test_completed_route_edge_is_rejected_by_behavior_stage():
    stage = BehaviorStage()
    manager = ManeuverManager()
    manager.observe_route_lane_change_edge("route-1:lane-change-a")
    manager.begin_lane_change("lane_change_right", "executing", 10, 11, 8.0, [])
    manager.complete_lane_change("complete")

    result = stage.authorize_route_lane_change(
        _route_lane_change_request(), maneuver_manager=manager
    )

    assert not result.allowed
    assert "edge_already_completed" in result.reason


def test_route_lane_change_stage_owns_required_target_memory():
    stage = BehaviorStage()
    manager = ManeuverManager()

    result = stage.resolve_route_lane_change(
        _prepare(),
        maneuver_manager=manager,
        route_cursor=SimpleNamespace(
            missed_maneuver=False, route_s_m=4.0, stalled_motion_m=0.0
        ),
        current_lane_id=100,
        execution_active=False,
        replan_missed_lane_change=True,
    )

    assert result.authorization.allowed
    assert manager.lane_change.required_target_lane_id == 340155
    assert manager.lane_change.required_target_ad_lane_id == 340154
    assert result.replan_reason == ""


def test_route_lane_change_stage_invalidates_missed_cursor_once():
    stage = BehaviorStage()
    manager = ManeuverManager()

    result = stage.resolve_route_lane_change(
        _prepare(),
        maneuver_manager=manager,
        route_cursor=SimpleNamespace(
            missed_maneuver=True, route_s_m=31.5, stalled_motion_m=9.0
        ),
        current_lane_id=100,
        execution_active=False,
        replan_missed_lane_change=True,
    )

    assert not result.authorization.allowed
    assert not result.authorization.required_by_route
    assert result.authorization.reason.startswith(
        "route_cursor_missed_lane_change:"
    )
    assert result.replan_reason == "turn_missed_lane_change_route_unreachable"
    assert manager.lane_change.required_target_lane_id is None


def test_active_lane_change_is_not_invalidated_by_missed_cursor():
    stage = BehaviorStage()
    manager = ManeuverManager()

    result = stage.resolve_route_lane_change(
        _prepare(),
        maneuver_manager=manager,
        route_cursor=SimpleNamespace(
            missed_maneuver=True, route_s_m=31.5, stalled_motion_m=9.0
        ),
        current_lane_id=100,
        execution_active=True,
        replan_missed_lane_change=True,
    )

    assert result.authorization.allowed
    assert result.replan_reason == ""


def test_lateral_ownership_denies_lane_change_that_cannot_finish_before_turn():
    stage = BehaviorStage()
    manager = ManeuverManager()
    authorization = stage.authorize_route_lane_change(
        _route_lane_change_request(), maneuver_manager=manager
    )

    result = stage.resolve_lateral_ownership(
        authorization=authorization,
        maneuver_manager=manager,
        owner_state="LANE_FOLLOW",
        ego_speed_mps=8.0,
        planning_speed_mps=8.0,
        lane_change_duration_s=4.0,
        dt_s=0.1,
        lane_width_m=3.5,
        distance_to_turn_m=2.0,
        config={},
    )

    assert result.start_transition.action == "deny"
    assert not result.authorization.allowed
    assert "no_longer_feasible" in result.authorization.reason
    assert result.geometry_arc_m > 2.0


def test_turn_lateral_owner_releases_lane_change_commitment_once():
    stage = BehaviorStage()
    manager = ManeuverManager()
    manager.remember_required_lane_change(11, 1011)
    manager.begin_lane_change(
        "lane_change_right", "executing", 10, 11, 8.0, []
    )
    authorization = stage.authorize_route_lane_change(
        _route_lane_change_request(), maneuver_manager=manager
    )

    result = stage.resolve_lateral_ownership(
        authorization=authorization,
        maneuver_manager=manager,
        owner_state="PREPARE_TURN",
        ego_speed_mps=8.0,
        planning_speed_mps=8.0,
        lane_change_duration_s=4.0,
        dt_s=0.1,
        lane_width_m=3.5,
        distance_to_turn_m=40.0,
        config={},
    )

    assert result.handoff.action == "release"
    assert not manager.lane_change.active
    assert manager.lane_change.required_target_lane_id is None
    assert not result.authorization.allowed
    assert result.authorization.reason.startswith("scenario_lateral_owner:")


def test_opportunistic_authorization_is_owned_by_behavior_stage():
    stage = BehaviorStage()
    allowed = stage.authorize_opportunistic_lane_change(
        OpportunisticLaneChangeRequest(
            enabled=True, sim_time_s=10.0, start_lock_until_s=2.0,
            dense_traffic_lock_enabled=True, object_count=1,
            dense_object_count=5, lane_prediction_risks={},
            dense_risky_lane_count=2,
        )
    )
    blocked = stage.authorize_opportunistic_lane_change(
        OpportunisticLaneChangeRequest(
            enabled=True, sim_time_s=10.0, start_lock_until_s=2.0,
            dense_traffic_lock_enabled=True, object_count=1,
            dense_object_count=5,
            lane_prediction_risks={10: {"risk": True}, 11: {"risk": True}},
            dense_risky_lane_count=2,
        )
    )

    assert allowed.allowed
    assert not blocked.allowed
    assert blocked.reason == "opportunistic_lane_change_dense_traffic_lock"


def test_conflict_resolution_exposes_cooperative_prediction_seam():
    stage = BehaviorStage()
    manager = ManeuverManager()
    authorization = stage.authorize_route_lane_change(
        _route_lane_change_request(), maneuver_manager=manager
    )
    opportunistic = OpportunisticLaneChangeRequest(
        enabled=True, sim_time_s=10.0, start_lock_until_s=2.0,
        dense_traffic_lock_enabled=False, object_count=1,
        dense_object_count=5, lane_prediction_risks={},
        dense_risky_lane_count=2,
    )

    result = stage.resolve_conflicts(
        ConflictResolutionRequest(
            route_authorization=authorization,
            opportunistic_request=opportunistic,
            owner_state="LANE_FOLLOW",
            ego_speed_mps=8.0,
            planning_speed_mps=8.0,
            lane_change_duration_s=4.0,
            dt_s=0.1,
            lane_width_m=3.5,
            distance_to_turn_m=100.0,
            config={},
            ego_location=SimpleNamespace(x=0.0, y=0.0),
            ego_yaw_rad=0.0,
        ),
        maneuver_manager=manager,
        cooperative_yield_reason=lambda _location, _yaw: "yield:peer=2",
        cooperative_wait_speed_cap=lambda _location, _speed, _reason: 3.0,
    )

    assert not result.authorization.allowed
    assert result.authorization.reason == "yield:peer=2"
    assert result.cooperative_wait_speed_cap_mps == 3.0
    assert result.opportunistic_allowed


def test_behavior_stage_consumes_prediction_when_selecting_lane_candidate():
    frame = BehaviorStage.evaluate_lane_candidates(BehaviorCandidateRequest(
        lane_safety_scores={10: 1.0, 11: 1.0},
        lane_prediction_risks={11: {
            "risk": True, "collision_probability": 0.4,
            "reason": "probabilistic_conflict",
        }},
        ego_lane_id=10, available_lane_ids=(10, 11),
        route_optimal_lane_id=11, in_junction=False,
        mpc_feedback_blocked_lane_ids=(), mpc_feedback_weight=80.0,
        nearest_front_obstacles_by_lane={}, desired_speed_mps=8.0,
        progress_cost_weight=4.0,
    ))

    assert frame.selected.target_lane_id == 10
    assert frame.selected.feasible


def test_behavior_override_priority_is_single_and_deterministic():
    base = dict(
        decision="lane_change_right", target_lane_id=11, phase="EXECUTING",
        current_lane_id=10, lane_change_authorized=True,
        opportunistic_lane_change_allowed=False, lane_change_gate_reason="",
        lane_change_authorization_reason="authorized",
        prepare_reference_lock=True, scenario_speed_cap_active=False,
        scenario_reason="turn_started",
        scenario_override_decision="intersection_turn_right",
        scenario_override_phase="INTERSECTION_TURN_RIGHT",
        scenario_stop_required=False, local_avoidance_active=False,
        ego_in_junction=True, lane_change_commitment_active=False,
        route_turn_decision="intersection_turn_right",
        route_current_road_option="RIGHT", stop_goal_active=False,
    )
    result = BehaviorStage.apply_overrides(BehaviorOverrideRequest(**base))

    assert result.decision == "intersection_turn_right"
    assert result.target_lane_id == 10
    assert result.phase == "INTERSECTION_TURN_RIGHT"


def test_unauthorized_lane_change_is_normalized_once():
    result = BehaviorStage.apply_overrides(BehaviorOverrideRequest(
        decision="lane_change_left", target_lane_id=11, phase="EXECUTING",
        current_lane_id=10, lane_change_authorized=False,
        opportunistic_lane_change_allowed=False,
        lane_change_gate_reason="prediction_conflict",
        lane_change_authorization_reason="prediction_conflict",
        prepare_reference_lock=True, scenario_speed_cap_active=False,
        scenario_reason="", scenario_override_decision="",
        scenario_override_phase="", scenario_stop_required=False,
        local_avoidance_active=False, ego_in_junction=False,
        lane_change_commitment_active=False, route_turn_decision="",
        route_current_road_option="LANEFOLLOW", stop_goal_active=False,
    ))

    assert result.decision == "lane_follow"
    assert result.target_lane_id == 10
    assert result.phase == "LANE_KEEP"
    assert result.reason == "prediction_conflict"
    assert result.reset_lane_change_reason == "prediction_conflict"
