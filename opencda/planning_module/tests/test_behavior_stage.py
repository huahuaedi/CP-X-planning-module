import math

from pipeline.behavior_stage import (
    BehaviorCandidateRequest,
    BehaviorOverrideRequest,
    BehaviorStage,
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
