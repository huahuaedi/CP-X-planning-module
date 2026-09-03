import importlib.util
import math
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "pipeline"
for package_name in ("opencda", "opencda.planning_module", "opencda.planning_module.pipeline"):
    if package_name not in sys.modules:
        module = types.ModuleType(package_name)
        module.__path__ = [str(PIPELINE_ROOT)]
        sys.modules[package_name] = module

REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "opencda.planning_module.pipeline.reference_contract",
    PIPELINE_ROOT / "reference_contract.py",
)
reference_contract = importlib.util.module_from_spec(REFERENCE_SPEC)
sys.modules[REFERENCE_SPEC.name] = reference_contract
REFERENCE_SPEC.loader.exec_module(reference_contract)

CANDIDATE_SPEC = importlib.util.spec_from_file_location(
    "opencda.planning_module.pipeline.candidate_pipeline",
    PIPELINE_ROOT / "candidate_pipeline.py",
)
candidate_pipeline = importlib.util.module_from_spec(CANDIDATE_SPEC)
sys.modules[CANDIDATE_SPEC.name] = candidate_pipeline
CANDIDATE_SPEC.loader.exec_module(candidate_pipeline)


class _Location:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Transform:
    def __init__(self, x, y):
        self.location = _Location(x, y)


class _Waypoint:
    def __init__(self, x, y, previous=None, left=None, right=None):
        self.transform = _Transform(x, y)
        self._previous = previous
        self._left = left
        self._right = right
        self.lane_width = 3.5

    def previous(self, _distance):
        return [] if self._previous is None else [self._previous]

    def get_left_lane(self):
        return self._left

    def get_right_lane(self):
        return self._right


class _MapPlanner:
    def __init__(self, projected_waypoint):
        self.projected_waypoint = projected_waypoint
        self.request = None

    def get_waypoint(self, request):
        self.request = request
        return self.projected_waypoint


class _ContractResult:
    def __init__(self, valid=True, reason=""):
        self.valid = bool(valid)
        self._reason = str(reason)

    def reason(self):
        return self._reason


class CandidatePipelineTest(unittest.TestCase):
    def test_lane_follow_has_no_generic_fixed_ratio_yield_candidate(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=15.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=False,
            lane_change_authorized_target_lane_id=1,
            allow_lane_change_candidates=False,
        )

        self.assertNotIn("yield_slow_down", {intent.name for intent in intents})
        self.assertEqual(
            {float(intent.target_speed_mps) for intent in intents},
            {15.0},
        )

    def test_confirmed_local_avoidance_keeps_stop_as_deferred_fallback(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_change_left",
            selected_target_lane_id=2,
            current_lane_id=1,
            target_speed_mps=8.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 0.1, 2: 0.95},
            lane_prediction_risks={2: {"risk": False}},
            stop_goal_active=True,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
            lane_change_authorization_source="opportunistic",
            lane_change_authorization_direction="left",
            local_obstacle_avoidance_active=True,
            local_obstacle_stop_defer_cost=25.0,
        )

        stop_intent = next(
            intent for intent in intents if intent.name == "obstacle_stop"
        )
        lane_change_intents = [
            intent for intent in intents if intent.decision == "lane_change_left"
        ]
        self.assertEqual(stop_intent.base_cost, 25.0)
        self.assertTrue(lane_change_intents)
        self.assertLess(
            min(intent.base_cost for intent in lane_change_intents), 25.0
        )

    def test_static_obstacle_stop_is_a_mandatory_zero_speed_candidate(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="static_obstacle_stop",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=12.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 0.0, 2: 1.0},
            lane_prediction_risks={1: {}, 2: {}},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].decision, "static_obstacle_stop")
        self.assertEqual(float(intents[0].target_speed_mps), 0.0)
        self.assertTrue(bool(intents[0].stop_goal_active))

    def test_physical_direction_overrides_opaque_topology_sign(self):
        physical_left = _Waypoint(1.0, 3.5)
        physical_right = _Waypoint(1.0, -3.5)
        ego = _Waypoint(1.0, 0.0, left=physical_left, right=physical_right)

        direction, reason = candidate_pipeline.physical_adjacent_direction(
            ego_waypoint=ego,
            target_waypoint=_Waypoint(1.1, 3.45),
        )

        self.assertEqual(direction, "left")
        self.assertEqual(reason, "physical_direction_from_carla_adjacency")

    def test_route_lane_change_anchor_uses_post_jump_physical_lane(self):
        target_ego_station = _Waypoint(1.0, 3.5)
        target_mid = _Waypoint(2.0, 3.5, previous=target_ego_station)
        target_after_jump = _Waypoint(3.0, 3.5, previous=target_mid)
        map_planner = _MapPlanner(target_after_jump)

        anchor, reason = candidate_pipeline.route_lane_change_target_anchor(
            map_planner=map_planner,
            route_points=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 3.5, 0.0],
                [4.0, 3.5, 0.0],
            ],
            ego_x_m=1.0,
            ego_y_m=0.0,
            nominal_step_m=1.0,
        )

        self.assertIs(anchor, target_ego_station)
        self.assertEqual(reason, "target_anchor_from_post_change_lane")
        self.assertAlmostEqual(float(map_planner.request["x"]), 3.0)
        self.assertAlmostEqual(float(map_planner.request["y"]), 3.5)

    def test_route_lane_change_anchor_requires_lateral_route_edge(self):
        anchor, reason = candidate_pipeline.route_lane_change_target_anchor(
            map_planner=_MapPlanner(_Waypoint(2.0, 0.0)),
            route_points=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            ego_x_m=0.0,
            ego_y_m=0.0,
            nominal_step_m=1.0,
        )

        self.assertIsNone(anchor)
        self.assertEqual(reason, "target_anchor_no_lateral_route_edge")

    def test_topology_direction_builds_lane_change_when_local_ids_collide(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=12.0,
            candidate_lane_ids=[1],
            lane_safety_scores={1: 1.0},
            lane_prediction_risks={1: {}},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=1,
            allow_lane_change_candidates=True,
            lane_change_authorization_direction="right",
        )

        lane_changes = [
            intent for intent in intents
            if intent.decision == "lane_change_right"
        ]
        self.assertTrue(lane_changes)
        self.assertTrue(all(intent.target_lane_id == 1 for intent in lane_changes))

    def test_authorized_direction_does_not_depend_on_opaque_lane_id_order(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=900,
            current_lane_id=900,
            target_speed_mps=12.0,
            candidate_lane_ids=[900, 120],
            lane_safety_scores={900: 1.0, 120: 1.0},
            lane_prediction_risks={900: {}, 120: {}},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=120,
            allow_lane_change_candidates=True,
            lane_change_authorization_direction="left",
        )
        lane_changes = [intent for intent in intents if intent.target_lane_id == 120]
        self.assertTrue(lane_changes)
        self.assertTrue(all(intent.decision == "lane_change_left" for intent in lane_changes))
    def test_traffic_stop_candidate_has_hard_priority(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=True,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].decision, "stop_at_intersection")
        self.assertEqual(intents[0].target_speed_mps, 0.0)

    def test_obstacle_stop_keeps_authorized_lane_change_candidates(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=2,
            current_lane_id=2,
            target_speed_mps=3.0,
            candidate_lane_ids=[2, 1],
            lane_safety_scores={2: 0.4, 1: 1.0},
            lane_prediction_risks={},
            stop_goal_active=True,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=1,
            lane_change_authorization_direction="right",
            allow_lane_change_candidates=True,
            lane_change_authorization_source="route",
        )

        self.assertTrue(any(intent.name == "obstacle_stop" for intent in intents))
        self.assertTrue(any(
            intent.decision == "lane_change_right"
            for intent in intents
        ))

    def test_obstacle_stop_is_deferred_while_a_turn_is_in_progress(self):
        turn_intents = candidate_pipeline.build_candidate_intents(
            selected_decision="intersection_turn_left",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1],
            lane_safety_scores={1: 1.0},
            lane_prediction_risks={},
            stop_goal_active=True,
            traffic_stop_active=False,
            lane_change_authorized=False,
            lane_change_authorized_target_lane_id=0,
            allow_lane_change_candidates=False,
        )
        obstacle_stop = next(
            intent for intent in turn_intents if intent.name == "obstacle_stop"
        )
        self.assertGreater(obstacle_stop.base_cost, 0.0)
        self.assertEqual(
            obstacle_stop.reason,
            "front_obstacle_stop_candidate_defer_turn_in_progress",
        )
        self.assertTrue(any(
            intent.decision == "intersection_turn_left" for intent in turn_intents
        ))

        # The same trigger without an in-progress turn keeps the candidate
        # undeferred -- only a committed turn gets this handicap.
        lane_follow_intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1],
            lane_safety_scores={1: 1.0},
            lane_prediction_risks={},
            stop_goal_active=True,
            traffic_stop_active=False,
            lane_change_authorized=False,
            lane_change_authorized_target_lane_id=0,
            allow_lane_change_candidates=False,
        )
        undeferred_obstacle_stop = next(
            intent for intent in lane_follow_intents if intent.name == "obstacle_stop"
        )
        self.assertEqual(undeferred_obstacle_stop.base_cost, 0.0)
        self.assertEqual(
            undeferred_obstacle_stop.reason,
            "front_obstacle_stop_candidate",
        )

    def test_lane_change_candidate_requires_authorization(self):
        denied = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=False,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )
        self.assertFalse(any("lane_change" in intent.decision for intent in denied))

        allowed = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            lane_change_authorization_direction="left",
            allow_lane_change_candidates=True,
        )
        self.assertTrue(any(intent.decision == "lane_change_left" for intent in allowed))
        lane_change_intents = [
            intent for intent in allowed if intent.decision == "lane_change_left"
        ]
        self.assertEqual(
            {intent.trajectory_variant for intent in lane_change_intents},
            {"assertive", "normal", "conservative"},
        )
        self.assertEqual(
            {intent.lane_change_duration_s for intent in lane_change_intents},
            {3.2, 4.0, 5.5},
        )

    def test_human_lane_change_clear_gap_preserves_cruise_speed(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_change_left",
            selected_target_lane_id=2,
            current_lane_id=1,
            target_speed_mps=15.6464,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={2: {
                "risk": False,
                "min_front_gap_m": 80.0,
                "min_rear_gap_m": 60.0,
            }},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
            human_like_lane_change_enabled=True,
            ego_speed_mps=15.0,
            lane_width_m=3.5,
            lane_change_available_distance_m=100.0,
        )
        lane_changes = [
            intent for intent in intents
            if intent.decision == "lane_change_left"
        ]
        self.assertEqual(len(lane_changes), 3)
        self.assertTrue(all(
            abs(intent.target_speed_mps - 15.6464) < 1.0e-6
            for intent in lane_changes
        ))
        self.assertTrue(all(
            3.0 <= intent.lane_change_duration_s <= 6.5
            for intent in lane_changes
        ))
        self.assertLess(
            next(x for x in lane_changes if x.trajectory_variant == "assertive").lane_change_duration_s,
            next(x for x in lane_changes if x.trajectory_variant == "conservative").lane_change_duration_s,
        )

    def test_human_lane_change_only_slows_for_constraining_front_gap(self):
        profiles = candidate_pipeline.build_human_lane_change_profiles(
            ego_speed_mps=10.0,
            lane_width_m=3.5,
            target_lane_prediction_risk={
                "risk": False,
                "min_front_gap_m": 14.0,
                "min_rear_gap_m": 50.0,
            },
        )
        self.assertEqual(profiles[0].speed_scale, 1.0)
        self.assertLess(profiles[1].speed_scale, 1.0)
        self.assertLess(profiles[2].speed_scale, profiles[1].speed_scale)
        self.assertIn("front_gap_mild_slowdown", profiles[1].reason)

    def test_human_lane_change_does_not_slow_with_close_rear_vehicle(self):
        profiles = candidate_pipeline.build_human_lane_change_profiles(
            ego_speed_mps=10.0,
            lane_width_m=3.5,
            target_lane_prediction_risk={
                "risk": False,
                "min_front_gap_m": 14.0,
                "min_rear_gap_m": 9.0,
            },
        )
        self.assertTrue(all(profile.speed_scale == 1.0 for profile in profiles))
        self.assertIn("rear_gap_maintain_speed", profiles[1].reason)

    def test_route_required_lane_change_penalizes_keep_lane_defer(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_change_right",
            selected_target_lane_id=1,
            current_lane_id=2,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=1,
            allow_lane_change_candidates=True,
            lane_change_authorization_source="route",
            lane_change_defer_cost=10.0,
        )

        keep = next(intent for intent in intents if intent.name == "keep_lane")
        normal = next(
            intent for intent in intents
            if intent.trajectory_variant == "normal"
        )
        self.assertEqual(keep.reason, "defer_route_lane_change")
        self.assertGreater(keep.base_cost, normal.base_cost)
        self.assertNotIn("yield_slow_down", {intent.name for intent in intents})

    def test_lane_change_reference_uses_quintic_time_blend(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0, "heading_rad": 0.0}
            for index in range(6)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 3.5, "heading_rad": 0.0}
            for index in range(6)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=4.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
        )

        self.assertEqual(len(shaped), 6)
        self.assertGreater(shaped[0]["y_ref_m"], 0.0)
        self.assertLess(shaped[0]["y_ref_m"], shaped[-1]["y_ref_m"])
        # duration_s=4.0 calls for more blend time than the 6 samples at
        # dt_s=0.5 (3.0s) actually span. The blend must still finish by the
        # last available sample -- if it didn't, the caller splices the
        # unblended target tail directly onto an only-partially-blended
        # last point, a lateral-offset jump that reads as a curvature spike
        # independent of how generous duration_s is. So the last sample
        # lands at the target lane, not short of it.
        self.assertAlmostEqual(shaped[-1]["y_ref_m"], 3.5, places=6)
        self.assertTrue(all(
            row["lane_transition_kind"] == "lateral_lane_change"
            for row in shaped
        ))

    def test_lane_change_reference_preserves_current_lateral_progress(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0}
            for index in range(6)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 4.0}
            for index in range(6)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=4.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
            ego_x_m=1.0,
            ego_y_m=2.0,
        )

        self.assertAlmostEqual(shaped[0]["lane_change_initial_progress"], 0.5)
        self.assertGreater(shaped[0]["y_ref_m"], 2.0)
        self.assertTrue(all(
            second["lane_change_progress"] >= first["lane_change_progress"]
            for first, second in zip(shaped[:-1], shaped[1:])
        ))

    def test_lane_change_reference_progress_floor_prevents_regression(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0}
            for index in range(6)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 4.0}
            for index in range(6)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=4.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
            ego_x_m=1.0,
            ego_y_m=0.5,
            initial_progress_floor=0.6,
        )

        self.assertAlmostEqual(
            shaped[0]["lane_change_initial_progress"],
            0.6,
        )

    def test_lane_change_reference_blend_geometry_false_uses_raw_target_xy(self):
        # blend_geometry=False lets MPC's own QP determine the transient
        # lateral path (subject to its hard steer-rate/accel constraints)
        # instead of being forced to track a pre-shaped geometric blend that
        # can demand more steering rate than those constraints allow. The
        # returned samples should carry the target lane's own x/y verbatim
        # -- not interpolated toward source -- while progress bookkeeping
        # (still needed by commitment/completion gating) stays populated.
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0, "heading_rad": 0.0}
            for index in range(6)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 3.5, "heading_rad": 0.0}
            for index in range(6)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=4.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
            blend_geometry=False,
        )

        self.assertEqual(len(shaped), 6)
        self.assertTrue(all(
            abs(float(row["y_ref_m"]) - 3.5) < 1.0e-9 for row in shaped
        ))
        self.assertTrue(all(
            abs(float(row["x_ref_m"]) - float(index + 1)) < 1.0e-9
            for index, row in enumerate(shaped)
        ))
        self.assertIn("lane_change_progress", shaped[0])
        self.assertIn("lane_change_initial_progress", shaped[0])

    @staticmethod
    def _discrete_curvature_1pm(samples):
        max_curvature = 0.0
        for first, second in zip(samples[:-1], samples[1:]):
            dx = float(second["x_ref_m"]) - float(first["x_ref_m"])
            dy = float(second["y_ref_m"]) - float(first["y_ref_m"])
            step_m = max(1.0e-6, math.hypot(dx, dy))
            dtheta = math.atan2(
                math.sin(float(second["heading_rad"]) - float(first["heading_rad"])),
                math.cos(float(second["heading_rad"]) - float(first["heading_rad"])),
            )
            max_curvature = max(max_curvature, abs(dtheta) / step_m)
        return max_curvature

    def test_lane_change_duration_comfort_check_accepts_already_comfortable_duration(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0, "heading_rad": 0.0}
            for index in range(20)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 3.5, "heading_rad": 0.0}
            for index in range(20)
        ]
        duration_s, shaped, reason = candidate_pipeline.select_comfortable_lane_change_duration_s(
            target_reference=target,
            source_reference=source,
            initial_duration_s=6.0,
            duration_max_s=8.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
            lateral_accel_limit_mps2=1.3,
            curvature_fn=self._discrete_curvature_1pm,
        )

        self.assertEqual(duration_s, 6.0)
        self.assertEqual(reason, "lane_change_duration_within_comfort_limit")
        self.assertEqual(len(shaped), 20)

    def test_lane_change_duration_comfort_check_widens_too_short_duration(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0, "heading_rad": 0.0}
            for index in range(20)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 3.5, "heading_rad": 0.0}
            for index in range(20)
        ]
        duration_s, shaped, reason = candidate_pipeline.select_comfortable_lane_change_duration_s(
            target_reference=target,
            source_reference=source,
            initial_duration_s=1.5,
            duration_max_s=8.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
            lateral_accel_limit_mps2=1.3,
            curvature_fn=self._discrete_curvature_1pm,
        )

        self.assertGreater(duration_s, 1.5)
        self.assertEqual(reason, "lane_change_duration_within_comfort_limit")
        implied_lateral_accel = 3.0 ** 2 * self._discrete_curvature_1pm(shaped)
        self.assertLessEqual(implied_lateral_accel, 1.3 + 1.0e-6)

    def test_lane_change_duration_comfort_check_caps_at_max_when_never_satisfied(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0, "heading_rad": 0.0}
            for index in range(20)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 3.5, "heading_rad": 0.0}
            for index in range(20)
        ]
        duration_s, shaped, reason = candidate_pipeline.select_comfortable_lane_change_duration_s(
            target_reference=target,
            source_reference=source,
            initial_duration_s=1.5,
            duration_max_s=1.5,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
            lateral_accel_limit_mps2=1.3,
            curvature_fn=self._discrete_curvature_1pm,
        )

        self.assertEqual(duration_s, 1.5)
        self.assertEqual(reason, "lane_change_duration_capped_at_max")
        self.assertEqual(len(shaped), 20)

    def test_road_curvature_does_not_force_lane_change_to_max_duration(self):
        source = [
            {"x_ref_m": float(index), "y_ref_m": 0.0, "heading_rad": 0.02 * index}
            for index in range(20)
        ]
        target = [
            {"x_ref_m": float(index), "y_ref_m": 3.5, "heading_rad": 0.02 * index}
            for index in range(20)
        ]

        duration_s, shaped, reason = (
            candidate_pipeline.select_comfortable_lane_change_duration_s(
                target_reference=target,
                source_reference=source,
                initial_duration_s=4.0,
                duration_max_s=8.0,
                dt_s=0.1,
                current_lane_id=1,
                target_lane_id=2,
                target_speed_mps=12.0,
                lateral_accel_limit_mps2=1.3,
                # Model a curved road whose baseline curvature is unchanged
                # by the lateral maneuver.
                curvature_fn=lambda _samples: 0.037,
            )
        )

        self.assertEqual(duration_s, 4.0)
        self.assertEqual(reason, "lane_change_duration_within_comfort_limit")
        self.assertTrue(shaped)

    def test_lane_change_distance_uses_reachable_average_speed(self):
        average_speed = candidate_pipeline.predicted_lane_change_average_speed_mps(
            ego_speed_mps=4.5,
            target_speed_mps=12.0,
            duration_s=3.75,
            acceleration_limit_mps2=2.0,
        )

        self.assertAlmostEqual(average_speed, 8.25)
        self.assertLess(average_speed, 12.0)

    def test_lane_change_operational_curvature_uses_lateral_acceleration(self):
        limit = candidate_pipeline.lane_change_operational_curvature_limit_1pm(
            planning_speed_mps=12.0,
            lateral_accel_limit_mps2=1.3,
            vehicle_max_curvature_1pm=0.35,
        )

        self.assertAlmostEqual(limit, 1.3 / (12.0 * 12.0))
        _, length_m, _ = candidate_pipeline.lane_change_geometry_requirements(
            ego_speed_mps=5.0,
            target_speed_mps=12.0,
            duration_s=4.38,
            dt_s=0.1,
            lane_width_m=3.5,
            max_curvature_1pm=limit,
        )
        self.assertGreater(length_m, 45.0)

    def test_stopped_lane_change_geometry_keeps_minimum_spatial_length(self):
        geometry_speed, length_m, step_m = (
            candidate_pipeline.lane_change_geometry_requirements(
                ego_speed_mps=0.0,
                target_speed_mps=0.8,
                duration_s=4.38,
                dt_s=0.05,
                lane_width_m=3.5,
                max_curvature_1pm=0.35,
                minimum_geometry_speed_mps=2.0,
                minimum_length_m=10.0,
            )
        )

        self.assertEqual(geometry_speed, 2.0)
        self.assertGreaterEqual(length_m, 10.0)
        self.assertGreater(step_m, 0.11)
        self.assertLess(step_m, 0.12)

        sample_count = int(math.ceil(4.38 / 0.05)) + 20
        source = [
            {
                "x_ref_m": float(index + 1) * step_m,
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
            }
            for index in range(sample_count)
        ]
        target = [
            {
                "x_ref_m": float(index + 1) * step_m,
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
            }
            for index in range(sample_count)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=4.38,
            dt_s=0.05,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=0.8,
        )
        self.assertLessEqual(self._discrete_curvature_1pm(shaped), 0.35)

    def test_envelope_blocks_anchor_at_lock_position_and_span_lane_width(self):
        source = [
            {
                "x_ref_m": float(index), "y_ref_m": 0.0, "heading_rad": 0.0,
                "lane_width_m": 3.5, "road_left_width_m": 1.75, "road_right_width_m": 1.75,
            }
            for index in range(20)
        ]
        target = [
            {
                "x_ref_m": float(index), "y_ref_m": 3.5, "heading_rad": 0.0,
                "lane_width_m": 3.5, "road_left_width_m": 1.75, "road_right_width_m": 1.75,
            }
            for index in range(20)
        ]

        blocks = candidate_pipeline.build_route_tracking_lane_change_envelope_blocks(
            source_reference=source,
            target_reference=target,
            master_step_count=20,
            step_distance_m=0.3,
            road_boundary_margin_m=0.5,
        )

        self.assertEqual(len(blocks), 2)
        source_block, target_block = blocks
        # Back edge anchored at the lock position (x=0), extending forward.
        self.assertAlmostEqual(
            source_block.x_center_m - source_block.half_length_m, 0.0, places=6
        )
        # Target block offset from source by exactly the lane width, same x.
        self.assertAlmostEqual(target_block.y_center_m - source_block.y_center_m, 3.5, places=6)
        self.assertAlmostEqual(target_block.x_center_m, source_block.x_center_m, places=6)
        # half_length covers the full locked master-array span.
        full_length_m = (20 - 1) * 0.3
        self.assertGreaterEqual(source_block.half_length_m, 0.5 * full_length_m)
        # half_width reflects the lane half-width minus the boundary margin.
        self.assertAlmostEqual(source_block.half_width_m, 1.75 - 0.5, places=6)

    def test_lane_change_reference_uses_continuous_two_lane_corridor(self):
        source = [
            {
                "x_ref_m": float(index + 1), "y_ref_m": 0.0,
                "heading_rad": 0.0, "lane_width_m": 3.5,
                "road_left_width_m": 1.75, "road_right_width_m": 1.75,
            }
            for index in range(20)
        ]
        target = [
            {
                "x_ref_m": float(index + 1), "y_ref_m": 3.5,
                "heading_rad": 0.0, "lane_width_m": 3.5,
                "road_left_width_m": 1.75, "road_right_width_m": 1.75,
            }
            for index in range(20)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=2.0,
            dt_s=0.1,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=10.0,
        )
        for sample in shaped:
            half_width = float(sample["road_left_width_m"])
            self.assertGreater(half_width, 3.4)
            self.assertAlmostEqual(
                half_width, float(sample["road_right_width_m"]), places=7
            )

    def test_envelope_blocks_empty_when_references_missing(self):
        self.assertEqual(
            candidate_pipeline.build_route_tracking_lane_change_envelope_blocks(
                source_reference=[],
                target_reference=[{"x_ref_m": 0.0, "y_ref_m": 0.0, "heading_rad": 0.0}],
                master_step_count=20,
                step_distance_m=0.3,
            ),
            [],
        )

    def test_turn_envelope_is_rolling_and_reserves_vehicle_width(self):
        reference = [
            {
                "x_ref_m": 0.0,
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_width_m": 3.5,
                "road_left_width_m": 1.75,
                "road_right_width_m": 1.75,
            },
            {
                "x_ref_m": 1.0,
                "y_ref_m": 0.2,
                "heading_rad": 0.2,
                "lane_width_m": 3.5,
                "road_left_width_m": 1.75,
                "road_right_width_m": 1.75,
            },
            {
                "x_ref_m": 1.8,
                "y_ref_m": 0.8,
                "heading_rad": 0.6,
                "lane_width_m": 3.5,
                "road_left_width_m": 1.75,
                "road_right_width_m": 1.75,
            },
        ]

        blocks = candidate_pipeline.build_turn_reference_envelope_blocks(
            reference_samples=reference,
            ego_half_width_m=1.05,
            safety_margin_m=0.15,
            longitudinal_overlap_m=0.75,
        )

        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(blocks[0].half_width_m, 0.55, places=6)
        self.assertAlmostEqual(blocks[1].half_width_m, 0.55, places=6)
        self.assertNotAlmostEqual(
            blocks[0].heading_rad,
            blocks[1].heading_rad,
            places=3,
        )
        # Rebuilding from a later rolling window must not retain the first
        # segment from the old horizon.
        later = candidate_pipeline.build_turn_reference_envelope_blocks(
            reference_samples=reference[1:],
            ego_half_width_m=1.05,
            safety_margin_m=0.15,
            longitudinal_overlap_m=0.75,
        )
        self.assertEqual(len(later), 1)
        self.assertAlmostEqual(later[0].x_center_m, blocks[1].x_center_m)
        self.assertAlmostEqual(later[0].y_center_m, blocks[1].y_center_m)

    def test_selected_lane_change_has_only_configured_variants(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_change_left",
            selected_target_lane_id=2,
            current_lane_id=1,
            target_speed_mps=4.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
            lane_change_assertive_duration_s=3.0,
            lane_change_normal_duration_s=4.5,
            lane_change_conservative_duration_s=6.0,
            lane_change_authorization_source="opportunistic",
        )

        lane_changes = [
            intent for intent in intents if intent.decision == "lane_change_left"
        ]
        self.assertEqual(len(lane_changes), 3)
        self.assertEqual(
            [intent.lane_change_duration_s for intent in lane_changes],
            [3.0, 4.5, 6.0],
        )
        self.assertEqual(
            [intent.target_speed_mps for intent in lane_changes],
            [4.0, 4.0, 4.0],
        )
        self.assertTrue(all(
            intent.name.startswith("opportunistic_lane_change_left_")
            for intent in lane_changes
        ))
        self.assertTrue(all(
            intent.reason == "opportunistic_lane_change_authorized"
            for intent in lane_changes
        ))

    def test_intersection_turn_candidate_does_not_include_plain_keep_lane(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="intersection_turn_right",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].decision, "intersection_turn_right")
        self.assertEqual(intents[0].reason, "route_option_turn_required")

    def test_selects_feasible_candidate_over_invalid_contract(self):
        bad_intent = candidate_pipeline.CandidateBehaviorIntent(
            name="bad",
            decision="lane_follow",
            target_lane_id=1,
            target_speed_mps=3.0,
            base_cost=0.0,
        )
        good_intent = candidate_pipeline.CandidateBehaviorIntent(
            name="good",
            decision="lane_follow",
            target_lane_id=1,
            target_speed_mps=2.0,
            base_cost=10.0,
        )
        bad = candidate_pipeline.CandidateReferenceResult(
            intent=bad_intent,
            destination_state=[5.0, 0.0, 3.0, 0.0],
            lane_center_reference=[{"x_ref_m": 1.0, "y_ref_m": 0.0}],
            contract_result=_ContractResult(valid=False, reason="bad_reference"),
        )
        good = candidate_pipeline.CandidateReferenceResult(
            intent=good_intent,
            destination_state=[5.0, 0.0, 2.0, 0.0],
            lane_center_reference=[{"x_ref_m": 1.0, "y_ref_m": 0.0}],
            contract_result=_ContractResult(valid=True),
        )
        evaluated = [
            candidate_pipeline.evaluate_candidate_reference(
                candidate=bad,
                ego_state=[0.0, 0.0, 0.0, 0.0],
                object_snapshots=[],
                current_lane_id=1,
            ),
            candidate_pipeline.evaluate_candidate_reference(
                candidate=good,
                ego_state=[0.0, 0.0, 0.0, 0.0],
                object_snapshots=[],
                current_lane_id=1,
            ),
        ]
        selected = candidate_pipeline.select_best_candidate(evaluated)
        self.assertEqual(selected.intent.name, "good")

    def test_prediction_trajectory_on_current_lane_becomes_follow_cost(self):
        intent = candidate_pipeline.CandidateBehaviorIntent(
            name="keep",
            decision="lane_follow",
            target_lane_id=1,
            target_speed_mps=3.0,
            base_cost=0.0,
        )
        candidate = candidate_pipeline.CandidateReferenceResult(
            intent=intent,
            destination_state=[5.0, 0.0, 3.0, 0.0],
            lane_center_reference=[
                {"x_ref_m": 1.0, "y_ref_m": 0.0},
                {"x_ref_m": 2.0, "y_ref_m": 0.0},
            ],
            contract_result=_ContractResult(valid=True),
        )
        evaluated = candidate_pipeline.evaluate_candidate_reference(
            candidate=candidate,
            ego_state=[0.0, 0.0, 0.0, 0.0],
            object_snapshots=[],
            prediction_trajectories={
                "obstacle": [
                    {"x": 1.2, "y": 0.0},
                    {"x": 2.1, "y": 0.0},
                ]
            },
            current_lane_id=1,
            min_object_distance_m=1.0,
        )
        self.assertTrue(evaluated.feasible)
        self.assertIn("candidate_prediction_lead_follow", evaluated.feasibility_reason)

    def test_prediction_trajectory_blocks_lane_change_candidate(self):
        intent = candidate_pipeline.CandidateBehaviorIntent(
            name="change",
            decision="lane_change_left",
            target_lane_id=2,
            target_speed_mps=3.0,
            base_cost=0.0,
        )
        candidate = candidate_pipeline.CandidateReferenceResult(
            intent=intent,
            destination_state=[5.0, 0.0, 3.0, 0.0],
            lane_center_reference=[
                {"x_ref_m": 1.0, "y_ref_m": 0.0},
                {"x_ref_m": 2.0, "y_ref_m": 0.0},
            ],
            contract_result=_ContractResult(valid=True),
        )
        evaluated = candidate_pipeline.evaluate_candidate_reference(
            candidate=candidate,
            ego_state=[0.0, 0.0, 0.0, 0.0],
            object_snapshots=[],
            prediction_trajectories={
                "obstacle": [
                    {"x": 1.2, "y": 0.0},
                    {"x": 2.1, "y": 0.0},
                ]
            },
            current_lane_id=1,
            min_object_distance_m=1.0,
        )
        self.assertFalse(evaluated.feasible)
        self.assertIn("candidate_prediction_collision_risk", evaluated.feasibility_reason)

    def test_route_lane_change_does_not_bypass_unified_cost_ranking(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
                base_cost=0.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=1.0,
        )
        change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.7,
                base_cost=10.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="mpc_probe_solved",
            total_cost=20.0,
        )
        selected = candidate_pipeline.select_best_candidate([keep, change])
        self.assertEqual(selected.intent.name, "keep_lane")

    def test_committed_lane_change_cannot_be_replaced_by_keep_lane(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=1.0,
        )
        change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="committed_lane_change_continuation",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=20.0,
        )
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=0.45,
            reference_locked=True,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [keep, change],
            commitment=commitment,
        )

        self.assertEqual(outcome.status, "selected_committed")
        self.assertIs(outcome.selected, change)

    def test_route_required_lane_change_outranks_soft_keep_lane_cost(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="mpc_probe_solved",
            total_cost=13.0,
        )
        change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right_assertive",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="mpc_probe_solved",
            total_cost=49.0,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [keep, change],
            commitment=candidate_pipeline.ManeuverCommitment(),
            required_decision="lane_change_right",
            required_target_lane_id=1,
        )

        self.assertEqual(outcome.status, "selected_route_required")
        self.assertEqual(outcome.reason, "feasible_route_required_candidate")
        self.assertIs(outcome.selected, change)

    def test_route_required_prefers_feasible_normal_over_cheaper_assertive(self):
        def lane_change(variant, cost):
            return candidate_pipeline.CandidateReferenceResult(
                intent=candidate_pipeline.CandidateBehaviorIntent(
                    name=f"route_lane_change_right_{variant}",
                    decision="lane_change_right",
                    target_lane_id=1,
                    target_speed_mps=3.0,
                    trajectory_variant=variant,
                ),
                destination_state=[],
                lane_center_reference=[],
                feasibility_status="mpc_probe_solved",
                total_cost=cost,
            )

        assertive = lane_change("assertive", 1.0)
        normal = lane_change("normal", 10.0)
        conservative = lane_change("conservative", 5.0)
        outcome = candidate_pipeline.select_candidate_with_commitment(
            [assertive, normal, conservative],
            commitment=candidate_pipeline.ManeuverCommitment(),
            required_decision="lane_change_right",
            required_target_lane_id=1,
        )

        self.assertIs(outcome.selected, normal)
        self.assertEqual(
            outcome.reason,
            "feasible_route_required_normal_candidate",
        )

    def test_route_required_lane_change_defers_when_candidate_is_infeasible(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="mpc_probe_solved",
            total_cost=13.0,
        )
        change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right_normal",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.7,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="infeasible",
            total_cost=1000.0,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [keep, change],
            commitment=candidate_pipeline.ManeuverCommitment(),
            required_decision="lane_change_right",
            required_target_lane_id=1,
        )

        self.assertEqual(outcome.status, "selected")
        self.assertEqual(outcome.reason, "route_required_candidate_infeasible_defer")
        self.assertIs(outcome.selected, keep)

    def test_committed_lane_change_requests_locked_reference_when_rerank_fails(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=1.0,
        )
        invalid_change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="infeasible",
            feasibility_reason="curvature_out_of_contract",
            total_cost=10020.0,
        )
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=0.45,
            reference_locked=True,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [keep, invalid_change],
            commitment=commitment,
        )

        self.assertEqual(outcome.status, "committed_reference_required")
        self.assertIsNone(outcome.selected)
        self.assertEqual(
            outcome.reason,
            "committed_maneuver_missing_locked_continuation",
        )

    def test_committed_lane_change_rejects_new_variant_when_locked_is_invalid(self):
        locked = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="committed_lane_change_continuation",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="infeasible",
            total_cost=1000.0,
        )
        regenerated = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right_normal",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=1.0,
        )
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=0.45,
            reference_locked=True,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [locked, regenerated],
            commitment=commitment,
        )

        self.assertIsNone(outcome.selected)
        self.assertEqual(outcome.reason, "locked_maneuver_reference_infeasible")

    def test_committed_contract_valid_reference_survives_one_probe_failure(self):
        locked = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="committed_lane_change_continuation",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            contract_result=reference_contract.ReferenceValidationResult(valid=True),
            feasibility_status="mpc_probe_infeasible",
            total_cost=10000.0,
        )
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=0.45,
            reference_locked=True,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [locked], commitment=commitment
        )

        self.assertIs(outcome.selected, locked)
        self.assertEqual(outcome.status, "selected_committed")
        self.assertEqual(
            outcome.reason,
            "locked_maneuver_reference_preserved_after_probe_failure",
        )

    def test_progress_alone_does_not_end_locked_commitment(self):
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=1.0,
            reference_locked=True,
        )

        self.assertTrue(commitment.active)

    def test_lane_change_alignment_removes_longitudinal_sample_offset(self):
        source = [
            {"x_ref_m": float(i), "y_ref_m": 0.0}
            for i in range(1, 21)
        ]
        target = [
            {"x_ref_m": float(i) + 2.0, "y_ref_m": -3.5}
            for i in range(1, 21)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            source_reference=source,
            target_reference=target,
            duration_s=4.0,
            dt_s=0.1,
            current_lane_id=2,
            target_lane_id=1,
            target_speed_mps=3.0,
        )
        self.assertEqual(len(shaped), 20)
        self.assertLess(abs(float(shaped[0]["x_ref_m"]) - 1.0), 0.05)
        self.assertLess(float(shaped[-1]["y_ref_m"]), -1.5)

    def test_turn_curvature_contract_violation_is_softened(self):
        intent = candidate_pipeline.CandidateBehaviorIntent(
            name="turn",
            decision="intersection_turn_left",
            target_lane_id=1,
            target_speed_mps=1.0,
            base_cost=0.0,
        )
        candidate = candidate_pipeline.CandidateReferenceResult(
            intent=intent,
            destination_state=[4.0, 2.0, 1.0, 0.0],
            lane_center_reference=[
                {"x_ref_m": 1.0, "y_ref_m": 0.0},
                {"x_ref_m": 2.0, "y_ref_m": 0.4},
            ],
            contract_result=_ContractResult(
                valid=False,
                reason="curvature_out_of_contract",
            ),
        )
        evaluated = candidate_pipeline.evaluate_candidate_reference(
            candidate=candidate,
            ego_state=[0.0, 0.0, 0.0, 0.0],
            object_snapshots=[],
            current_lane_id=1,
        )
        self.assertTrue(evaluated.feasible)
        self.assertIn("turn_contract_softened", evaluated.feasibility_reason)

    def test_mpc_probe_result_controls_candidate_feasibility(self):
        intent = candidate_pipeline.CandidateBehaviorIntent(
            name="lane_change",
            decision="lane_change_left",
            target_lane_id=2,
            target_speed_mps=2.0,
        )
        candidate = candidate_pipeline.CandidateReferenceResult(
            intent=intent,
            destination_state=[5.0, 1.0, 2.0, 0.0],
            lane_center_reference=[],
            contract_result=_ContractResult(valid=True),
            feasibility_status="feasible",
        )
        candidate_pipeline.apply_mpc_probe_result(
            candidate=candidate,
            solved=False,
            status="primal infeasible",
            solve_time_ms=2.0,
        )
        self.assertFalse(candidate.feasible)
        self.assertIn("mpc_probe:primal infeasible", candidate.feasibility_reason)


class LaneCostTopologyAliasTests(unittest.TestCase):
    """`lane_safety_scores`/`lane_prediction_risks` are keyed by this tick's
    canonical recount at ego's own cross-section, which only assigns unique
    numbers to lanes visible from there. When AD-map topology proves a
    route target is a real, different, farther lane that the recount
    happens to number the same as current_lane_id (topology_alias_target),
    looking it up by that shared key would silently read ego's own lane's
    entry instead -- and zero out the lane-change cost for what is,
    physically, a real lane change."""

    def test_alias_target_does_not_borrow_egos_own_lane_safety_score(self):
        # Lane 1 (ego's own key) reads as perfectly safe; if the alias
        # target silently reused that key it would look equally safe.
        cost_ordinary_same_key = candidate_pipeline._lane_cost(
            lane_id=1,
            current_lane_id=1,
            lane_safety_scores={1: 1.0},
            lane_prediction_risks={},
        )
        cost_alias_target = candidate_pipeline._lane_cost(
            lane_id=1,
            current_lane_id=1,
            lane_safety_scores={1: 1.0},
            lane_prediction_risks={},
            is_topology_alias_target=True,
        )

        self.assertGreater(cost_alias_target, cost_ordinary_same_key)

    def test_alias_target_still_charges_lane_change_cost(self):
        # Same numeric id as current_lane_id, so the ordinary branch's
        # "int(lane_id) != int(current_lane_id)" check would zero out the
        # lane-change cost even though this is a real physical lane change.
        cost_ordinary_same_key = candidate_pipeline._lane_cost(
            lane_id=1,
            current_lane_id=1,
            lane_safety_scores={1: 0.0},
            lane_prediction_risks={},
        )
        cost_alias_target = candidate_pipeline._lane_cost(
            lane_id=1,
            current_lane_id=1,
            lane_safety_scores={1: 0.0},
            lane_prediction_risks={},
            is_topology_alias_target=True,
        )

        self.assertAlmostEqual(cost_alias_target - cost_ordinary_same_key, 5.0)

    def test_non_alias_lookup_is_unaffected(self):
        cost_default = candidate_pipeline._lane_cost(
            lane_id=2,
            current_lane_id=1,
            lane_safety_scores={2: 0.75},
            lane_prediction_risks={},
        )
        cost_explicit_false = candidate_pipeline._lane_cost(
            lane_id=2,
            current_lane_id=1,
            lane_safety_scores={2: 0.75},
            lane_prediction_risks={},
            is_topology_alias_target=False,
        )

        self.assertAlmostEqual(cost_default, cost_explicit_false)

    def test_opaque_ad_lane_ids_do_not_determine_direction(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=500200,
            current_lane_id=490100,
            target_speed_mps=8.0,
            candidate_lane_ids=[490100, 500200],
            lane_safety_scores={490100: 1.0, 500200: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=500200,
            lane_change_authorization_direction="right",
            allow_lane_change_candidates=True,
        )

        lane_changes = [intent for intent in intents if "lane_change" in intent.decision]
        self.assertTrue(lane_changes)
        self.assertTrue(all(intent.decision == "lane_change_right" for intent in lane_changes))

    def test_route_authorization_direction_overrides_stale_behavior_direction(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_change_left",
            selected_target_lane_id=500144,
            current_lane_id=11640145,
            target_speed_mps=12.0,
            candidate_lane_ids=[11640145, 500144],
            lane_safety_scores={11640145: 1.0, 500144: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=500144,
            lane_change_authorization_direction="right",
            lane_change_authorization_source="route",
            allow_lane_change_candidates=True,
        )

        route_changes = [
            intent for intent in intents
            if intent.target_lane_id == 500144 and "lane_change" in intent.decision
        ]
        self.assertTrue(route_changes)
        self.assertTrue(all(
            intent.decision == "lane_change_right" for intent in route_changes
        ))


if __name__ == "__main__":
    unittest.main()
