from types import MappingProxyType, SimpleNamespace

from pipeline.nominal_trajectory import NominalTrajectoryGenerator
from pipeline.reference_planning_stage import (
    BehaviorReferencePreparationRequest,
    BehaviorReferenceRequest,
    CandidatePlanningRequest,
    PostTurnReferenceRequest,
    ReferencePlanningStage,
)


class _Provider:
    def __init__(self):
        self.behavior_kwargs = None
        self.post_turn_kwargs = None
        self.releases = []

    def build_behavior_reference(self, **kwargs):
        self.behavior_kwargs = kwargs
        return "built-reference"

    def resolve_post_turn_reference(self, **kwargs):
        self.post_turn_kwargs = kwargs
        return SimpleNamespace(
            clear_turn_reference=True,
            mutable_destination_state=lambda: [9.0, 1.0, 3.0, 0.0],
            mutable_samples=lambda: [{"x_ref_m": 9.0, "y_ref_m": 1.0}],
            debug_fields=MappingProxyType({"reference_source": "post_turn"}),
        )

    def release(self, mode, *, event):
        self.releases.append((mode, event))


class _CandidateSelection:
    def __init__(self):
        self.request = None
        self.kwargs = None

    def arbitrate(self, request, **kwargs):
        self.request = request
        self.kwargs = kwargs
        return "candidate-result"

    def cooperative_conflict_reference(self, **kwargs):
        return kwargs


def _behavior_request():
    return BehaviorReferenceRequest(
        map_planner="map", local_map="local-map", ego_pose={"x": 1.0},
        ego_state=[1.0, 2.0, 3.0, 0.0], route_points=[(1.0, 2.0)],
        behavior_runtime_config={}, decision="lane_follow",
        lane_change_state="LANE_KEEP", target_lane_id=4, current_lane_id=4,
        route_optimal_lane_id=4, route_reference_allowed=True,
        route_reference_gate_reason="", in_junction=False,
        next_macro_maneuver="lane_follow", planner_mode="NORMAL",
        lookahead_m=20.0, target_speed_mps=8.0, ego_speed_mps=7.0,
        horizon_steps=10, dt_s=0.2, sim_time_s=2.0,
        stop_release_smooth_until_s=0.0, authoritative_ego_waypoint="wp",
    )


def test_stage_builds_from_last_accepted_nominal_trajectory():
    provider = _Provider()
    nominal = NominalTrajectoryGenerator()
    nominal.update(
        target_state=[3.0, 4.0, 5.0, 0.1],
        samples=[{"x_ref_m": 3.0, "y_ref_m": 4.0}],
        reference_freeze_count=7,
        source="test",
    )
    stage = ReferencePlanningStage(
        provider=provider, nominal_trajectory_generator=nominal,
        candidate_selection=_CandidateSelection(),
    )

    frame = stage.build_behavior_reference(_behavior_request())

    assert frame.built_reference == "built-reference"
    assert provider.behavior_kwargs["previous_target_state"][:2] == [3.0, 4.0]
    assert provider.behavior_kwargs["previous_reference"][0]["x_ref_m"] == 3.0
    assert provider.behavior_kwargs["reference_freeze_count"] == 7


def test_stage_prepares_baseline_from_typed_upstream_frames():
    provider = _Provider()
    stage = ReferencePlanningStage(
        provider=provider,
        nominal_trajectory_generator=NominalTrajectoryGenerator(),
        candidate_selection=_CandidateSelection(),
    )
    adapter = SimpleNamespace(
        ego_pose={"x": 1.0}, current_state=[1.0, 2.0, 3.0, 0.0],
        route_points=[(1.0, 2.0)], route_optimal_lane_id=4,
        route_reference_allowed=True, route_reference_gate_reason="",
    )
    planning = SimpleNamespace(
        adapter_output=adapter,
        current_lane_id=4,
        planner_input_frame=SimpleNamespace(
            map_lane=SimpleNamespace(in_junction=False),
            planning=SimpleNamespace(route=SimpleNamespace(
                next_macro_maneuver="lane_follow"
            )),
        ),
    )
    executable = SimpleNamespace(
        decision="lane_follow", phase="LANE_KEEP", target_lane_id=4,
    )

    prepared = stage.prepare_behavior_reference(
        BehaviorReferencePreparationRequest(
            map_planner="map", local_map="local-map",
            planning_context=planning, executable_behavior=executable,
            speed_frame=SimpleNamespace(target_speed_mps=8.0),
            behavior_runtime_config={}, planner_mode="NORMAL",
            lookahead_m=20.0, ego_speed_mps=7.0, horizon_steps=10,
            dt_s=0.2, sim_time_s=2.0, stop_release_smooth_until_s=0.0,
            authoritative_ego_waypoint="wp",
        )
    )

    assert prepared.built_reference == "built-reference"
    assert prepared.request.current_lane_id == 4
    assert prepared.request.target_speed_mps == 8.0
    assert provider.behavior_kwargs["route_points"] == ((1.0, 2.0),)


def test_stage_owns_post_turn_release_and_nominal_publication():
    provider = _Provider()
    nominal = NominalTrajectoryGenerator()
    stage = ReferencePlanningStage(
        provider=provider, nominal_trajectory_generator=nominal,
        candidate_selection=_CandidateSelection(),
    )
    request = PostTurnReferenceRequest(
        maneuver_manager="maneuver", decision="lane_follow",
        scenario_state="CLEAR", exit_alignment_valid=True,
        exit_lateral_error_m=0.0, exit_heading_error_rad=0.0,
        local_map="local-map", ego_x_m=1.0, ego_y_m=2.0,
        current_state=[1.0, 2.0, 3.0, 0.0], current_lane_id=4,
        target_speed_mps=6.0, horizon_steps=10, dt_s=0.2,
        route_revision="route-1", map_epoch="admap", config={},
        destination_state=[8.0, 1.0, 3.0, 0.0],
        reference_samples=[{"x_ref_m": 8.0, "y_ref_m": 1.0}],
        debug_fields={}, reference_freeze_count=5,
    )

    stage.finalize_post_turn(request)

    assert provider.releases == [("turn", "reset")]
    assert nominal.current.source == "post_turn"
    assert nominal.current.reference_freeze_count == 5
    assert nominal.current.target.x_m == 9.0


def test_stage_builds_candidate_context_and_delegates_arbitration():
    provider = _Provider()
    candidate_selection = _CandidateSelection()
    stage = ReferencePlanningStage(
        provider=provider,
        nominal_trajectory_generator=NominalTrajectoryGenerator(),
        candidate_selection=candidate_selection,
    )
    baseline_request = _behavior_request()
    baseline_frame = SimpleNamespace(
        built_reference=SimpleNamespace(reference_freeze_count=3),
        mutable_previous_reference=lambda: [{"x_ref_m": 0.0}],
        mutable_previous_target_state=lambda: [0.0, 0.0, 4.0, 0.0],
    )
    authorization = SimpleNamespace(
        allowed=False, target_lane_id=0, direction="",
        distance_to_maneuver_m=None,
    )
    request = CandidatePlanningRequest(
        baseline_request=baseline_request, baseline_frame=baseline_frame,
        baseline_destination_state=[8.0, 0.0, 4.0, 0.0],
        baseline_reference=[{"x_ref_m": 1.0}], baseline_debug={"base": True},
        planner_config={}, selected_decision="lane_follow",
        selected_target_lane_id=4, current_lane_id=4, target_speed_mps=8.0,
        candidate_lane_ids=[4], lane_safety_scores={4: 1.0},
        lane_prediction_risks={}, stop_goal_active=False,
        traffic_stop_active=False, lane_change_authorization=authorization,
        opportunistic_lane_change_allowed=False, stop_target=None,
        local_obstacle_avoidance_active=False,
        current_state=[1.0, 2.0, 3.0, 0.0],
        ego_location=SimpleNamespace(x=1.0, y=2.0), ego_yaw_rad=0.0,
        object_snapshots=[], prediction_trajectories={},
        current_acceleration_mps2=0.1, current_steering_rad=0.0,
        route_required=False, scenario_stop_required=False,
        speed_plan="speed-plan", turn_prepare_speed_suppressed=False,
        cooperative_lane_change_deferred=False,
        lane_change_mpc_stall_failure_count=0,
        route_revision="route-1", map_epoch="admap",
        upcoming_turn_direction="", upcoming_turn_distance_m=100.0,
        lane_change_duration_s=4.0, lane_change_duration_reason="comfort",
        lane_width_m=3.5, validate_contract=lambda **_kwargs: True,
    )

    result = stage.arbitrate_candidates(request)

    assert result == "candidate-result"
    assert candidate_selection.request.reference_context.baseline_debug == {
        "base": True
    }
    assert candidate_selection.request.baseline_lane_change_state == "LANE_KEEP"
    assert candidate_selection.kwargs["route_revision"] == "route-1"
