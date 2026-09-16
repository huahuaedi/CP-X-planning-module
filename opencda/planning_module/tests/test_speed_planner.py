import importlib.util
import math
import pathlib
import sys
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "speed_planner_under_test",
    ROOT / "pipeline" / "speed_planner.py",
)
speed_planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = speed_planner
SPEC.loader.exec_module(speed_planner)
build_speed_plan = speed_planner.build_speed_plan
enforce_speed_ceiling = speed_planner.enforce_speed_ceiling
turn_approach_lookahead_m = speed_planner.turn_approach_lookahead_m
SpeedTargetPlanner = speed_planner.SpeedTargetPlanner
SpeedConstraint = speed_planner.SpeedConstraint
SpeedPlan = speed_planner.SpeedPlan
conflict_corridor_speed_constraint = (
    speed_planner.conflict_corridor_speed_constraint
)
cooperative_gap_speed_constraint = speed_planner.cooperative_gap_speed_constraint


class SpeedPlannerTest(unittest.TestCase):
    def test_turn_curvature_constraint_uses_lateral_acceleration_relation(self):
        constraint = SpeedTargetPlanner.turn_curvature_constraint(
            0.1,
            {
                "full_intersection_turn_lateral_accel_comfort_mps2": 2.5,
                "full_intersection_turn_curvature_min_curvature_1pm": 0.01,
            },
        )

        self.assertIsNotNone(constraint)
        self.assertEqual(constraint.owner, "turn_master_curvature")
        self.assertEqual(constraint.maximum_mps, 5.0)

    def test_turn_curvature_constraint_ignores_straight_master(self):
        constraint = SpeedTargetPlanner.turn_curvature_constraint(
            0.005,
            {"full_intersection_turn_curvature_min_curvature_1pm": 0.01},
        )

        self.assertIsNone(constraint)

    def test_turn_curvature_constraint_uses_approach_distance_envelope(self):
        constraint = SpeedTargetPlanner.turn_curvature_constraint(
            0.1,
            {
                "full_intersection_turn_lateral_accel_comfort_mps2": 2.5,
                "turn_approach_comfort_decel_mps2": 1.5,
            },
            distance_to_turn_m=12.0,
        )

        self.assertIsNotNone(constraint)
        self.assertAlmostEqual(constraint.maximum_mps, math.sqrt(61.0))
        self.assertIn("curve_speed_mps=5.000", constraint.reason)
        self.assertIn("distance_m=12.000", constraint.reason)

    def test_turn_curvature_constraint_reaches_curve_speed_at_turn_entry(self):
        constraint = SpeedTargetPlanner.turn_curvature_constraint(
            0.1,
            {"full_intersection_turn_lateral_accel_comfort_mps2": 2.5},
            distance_to_turn_m=0.0,
        )

        self.assertIsNotNone(constraint)
        self.assertEqual(constraint.maximum_mps, 5.0)

    def test_open_conflict_corridor_does_not_change_speed_path(self):
        constraint = conflict_corridor_speed_constraint(
            corridor=SimpleNamespace(
                s_hi=[1.0e9] * 4,
                binding=[""] * 4,
            ),
            reference_samples=[
                {"x_ref_m": 0.0, "y_ref_m": 0.0},
                {"x_ref_m": 30.0, "y_ref_m": 0.0},
            ],
            ego_x_m=5.0,
            ego_y_m=0.0,
            comfortable_deceleration_mps2=1.5,
        )

        self.assertIsNone(constraint)

    def test_conflict_corridor_creates_comfortable_approach_constraint(self):
        constraint = conflict_corridor_speed_constraint(
            corridor=SimpleNamespace(
                s_hi=[1.0e9, 20.0, 20.0, 1.0e9],
                binding=["", "peer-7", "peer-7", ""],
            ),
            reference_samples=[
                {"x_ref_m": 0.0, "y_ref_m": 0.0},
                {"x_ref_m": 30.0, "y_ref_m": 0.0},
            ],
            ego_x_m=5.0,
            ego_y_m=0.0,
            comfortable_deceleration_mps2=2.0,
        )

        self.assertIsNotNone(constraint)
        self.assertEqual(constraint.owner, "cav_conflict")
        self.assertAlmostEqual(constraint.maximum_mps, math.sqrt(60.0))
        self.assertIn("binding=peer-7", constraint.reason)

    def test_moving_peer_bound_is_not_a_stationary_stop_line(self):
        constraint = conflict_corridor_speed_constraint(
            corridor=SimpleNamespace(
                s_hi=[5.0, 5.8, 6.6, 7.4],
                binding=["peer"] * 4,
            ),
            reference_samples=[
                {"x_ref_m": 0.0, "y_ref_m": 0.0},
                {"x_ref_m": 30.0, "y_ref_m": 0.0},
            ],
            ego_x_m=5.0,
            ego_y_m=0.0,
            comfortable_deceleration_mps2=2.0,
            corridor_dt_s=0.1,
        )
        self.assertIsNotNone(constraint)
        self.assertAlmostEqual(constraint.maximum_mps, 8.0)
        self.assertIn("bound_velocity_mps=8.000", constraint.reason)

    def test_proposed_make_gap_prepares_room_without_zero_speed_step(self):
        constraint = cooperative_gap_speed_constraint(
            reference_samples=[
                {"x_ref_m": 0.0, "y_ref_m": 0.0},
                {"x_ref_m": 30.0, "y_ref_m": 0.0},
            ],
            ego_x_m=5.0, ego_y_m=0.0, ego_speed_mps=8.0,
            peer_x_m=11.0, peer_y_m=3.6, peer_speed_mps=8.0,
            peer_id="peer", peer_length_m=4.8,
            ego_half_length_m=2.45, desired_bumper_gap_m=15.0,
            preparation_time_s=4.0,
            comfortable_deceleration_mps2=1.5, planning_dt_s=0.1,
        )
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint.owner, "cooperative_gap")
        self.assertAlmostEqual(constraint.maximum_mps, 7.85)

    def test_cooperative_gap_releases_once_required_gap_is_open(self):
        constraint = cooperative_gap_speed_constraint(
            reference_samples=[
                {"x_ref_m": 0.0, "y_ref_m": 0.0},
                {"x_ref_m": 60.0, "y_ref_m": 0.0},
            ],
            ego_x_m=5.0, ego_y_m=0.0, ego_speed_mps=8.0,
            peer_x_m=40.0, peer_y_m=3.6, peer_speed_mps=8.0,
            peer_id="peer", peer_length_m=4.8,
            ego_half_length_m=2.45, desired_bumper_gap_m=15.0,
            preparation_time_s=4.0,
            comfortable_deceleration_mps2=1.5, planning_dt_s=0.1,
        )
        self.assertIsNone(constraint)

    def test_late_conflict_constraint_flows_through_single_speed_owner(self):
        planner = SpeedTargetPlanner()
        original = SpeedPlan(
            target_speed_mps=12.0,
            speed_cap_mps=12.0,
            stop_goal_active=False,
            requested_speed_mps=12.0,
        )
        constraint = SpeedConstraint(
            owner="cav_conflict",
            maximum_mps=6.0,
            reason="cooperative_corridor_approach",
        )

        constrained = planner.constrain_plan(original, constraint)
        target = planner.resolve(
            behavior=SimpleNamespace(
                requested_speed_mps=12.0, stop_required=False
            ),
            speed_plan=constrained,
        )

        self.assertEqual(original.target_speed_mps, 12.0)
        self.assertEqual(constrained.target_speed_mps, 6.0)
        self.assertEqual(constrained.external_constraints, (constraint,))
        self.assertEqual(target.target_mps, 6.0)
        self.assertEqual(target.limiting_owner, "cav_conflict")

    def test_speed_target_planner_is_single_final_ceiling_owner(self):
        behavior = SimpleNamespace(requested_speed_mps=12.0, stop_required=False)
        target = SpeedTargetPlanner().resolve(
            behavior=behavior,
            speed_plan=SpeedPlan(
                target_speed_mps=5.0,
                speed_cap_mps=5.0,
                stop_goal_active=False,
                requested_speed_mps=12.0,
                scenario_cap_mps=9.0,
                turn_cap_mps=5.0,
                limiting_owner="turn_cap",
            ),
        )
        self.assertEqual(target.target_mps, 5.0)
        self.assertEqual(target.limiting_owner, "turn_cap")

    def test_behavior_stop_is_final_zero_constraint(self):
        behavior = SimpleNamespace(requested_speed_mps=12.0, stop_required=True)
        target = SpeedTargetPlanner().resolve(
            behavior=behavior,
            speed_plan=SpeedPlan(
                target_speed_mps=8.0,
                speed_cap_mps=8.0,
                stop_goal_active=False,
            ),
        )
        self.assertEqual(target.target_mps, 0.0)
        self.assertEqual(target.limiting_owner, "behavior_stop")

    def test_destination_approach_is_a_named_speed_constraint(self):
        behavior = SimpleNamespace(requested_speed_mps=12.0, stop_required=False)
        target = SpeedTargetPlanner().resolve(
            behavior=behavior,
            speed_plan=SpeedPlan(
                target_speed_mps=12.0,
                speed_cap_mps=12.0,
                stop_goal_active=False,
            ),
            additional_constraints=(
                SpeedConstraint(
                    owner="destination_approach",
                    maximum_mps=6.3,
                    reason="route_destination_approach_speed_profile",
                ),
            ),
        )

        self.assertEqual(target.target_mps, 6.3)
        self.assertEqual(target.limiting_owner, "destination_approach")

    def test_candidate_behavior_speed_can_only_lower_typed_speed_plan(self):
        behavior = SimpleNamespace(requested_speed_mps=4.0, stop_required=False)
        target = SpeedTargetPlanner().resolve(
            behavior=behavior,
            speed_plan=SpeedPlan(
                target_speed_mps=8.0,
                speed_cap_mps=8.0,
                stop_goal_active=False,
                requested_speed_mps=12.0,
            ),
        )

        self.assertEqual(target.target_mps, 4.0)
        self.assertEqual(target.limiting_owner, "behavior_request")

    def test_destination_approach_does_not_release_after_vehicle_slows(self):
        planner = SpeedTargetPlanner()
        first, active, _ = planner.destination_approach_constraint(
            route_revision="route-4",
            route_found=True,
            route_reached_destination=False,
            remaining_distance_m=20.0,
            ego_speed_mps=10.0,
            deceleration_mps2=2.5,
            buffer_m=1.5,
        )
        self.assertTrue(active)
        self.assertIsNotNone(first)

        slowed, active, _ = planner.destination_approach_constraint(
            route_revision="route-4",
            route_found=True,
            route_reached_destination=False,
            remaining_distance_m=18.0,
            ego_speed_mps=2.0,
            deceleration_mps2=2.5,
            buffer_m=1.5,
        )
        self.assertTrue(active)
        self.assertIsNotNone(slowed)
        self.assertLessEqual(slowed.maximum_mps, first.maximum_mps)

    def test_destination_approach_resets_on_route_revision(self):
        planner = SpeedTargetPlanner()
        planner.destination_approach_constraint(
            route_revision="route-4",
            route_found=True,
            route_reached_destination=False,
            remaining_distance_m=20.0,
            ego_speed_mps=10.0,
            deceleration_mps2=2.5,
            buffer_m=1.5,
        )
        constraint, active, _ = planner.destination_approach_constraint(
            route_revision="route-5",
            route_found=True,
            route_reached_destination=False,
            remaining_distance_m=100.0,
            ego_speed_mps=2.0,
            deceleration_mps2=2.5,
            buffer_m=1.5,
        )
        self.assertFalse(active)
        self.assertIsNone(constraint)

    def test_turn_preview_scales_with_cruise_speed(self):
        config = {
            "full_intersection_turn_speed_cap_mps": 5.0,
            "turn_approach_comfort_decel_mps2": 2.5,
            "turn_approach_entry_buffer_m": 5.0,
            "turn_approach_preview_margin_m": 10.0,
        }
        preview_35_mph = turn_approach_lookahead_m(
            cruise_speed_mps=15.6464,
            config=config,
        )
        preview_60_mph = turn_approach_lookahead_m(
            cruise_speed_mps=26.8224,
            config=config,
        )
        self.assertGreater(preview_35_mph, 50.0)
        self.assertGreater(preview_60_mph, 140.0)
        self.assertGreater(preview_60_mph, preview_35_mph)

    @staticmethod
    def _legacy_result(
        *, scenario_cap, scenario_stop, decision, requested, ego_speed, front_gap
    ):
        """Frozen pre-instrumentation behavior used as a regression oracle."""
        cap = requested if scenario_cap is None else min(requested, max(0.0, scenario_cap))
        stop_goal = scenario_stop or decision in {
            "stop_at_intersection", "stop_sign", "emergency_brake"
        }
        if decision in {"intersection_turn_left", "intersection_turn_right"}:
            cap = min(cap, 2.2)
        following_active = False
        if front_gap is not None and math.isfinite(front_gap) and not stop_goal:
            desired_gap = 5.0 + 1.5 * max(0.0, ego_speed)
            free_gap = desired_gap + 6.0
            if front_gap <= 3.0:
                cap = 0.0
                stop_goal = True
            elif front_gap < free_gap:
                ratio = (front_gap - 3.0) / max(1.0e-6, free_gap - 3.0)
                smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
                follow_cap = 0.35 + smooth_ratio * max(0.0, cap - 0.35)
                cap = min(cap, follow_cap)
                following_active = True
        if stop_goal:
            cap = 0.0
        if not math.isfinite(cap):
            cap = 0.0
        return cap, stop_goal, following_active

    @staticmethod
    def _plan(front_gap_m):
        return build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=3.0,
            ego_speed_mps=2.0,
            config={},
            front_gap_m=front_gap_m,
        )

    def test_clear_road_keeps_requested_speed(self):
        plan = self._plan(30.0)
        self.assertAlmostEqual(plan.target_speed_mps, 3.0)
        self.assertFalse(plan.stop_goal_active)
        self.assertEqual(plan.limiting_owner, "behavior_request")

    def test_following_gap_reduces_speed_without_stop_latch(self):
        plan = self._plan(7.9)
        self.assertGreater(plan.target_speed_mps, 0.0)
        self.assertLess(plan.target_speed_mps, 3.0)
        self.assertFalse(plan.stop_goal_active)
        self.assertIn("speed_plan_continuous_following", plan.reason)
        self.assertEqual(plan.limiting_owner, "following_cap")
        self.assertIn("following_cap", plan.active_constraints)

    def test_idm_accelerates_rear_vehicle_toward_cruise_after_lane_change(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=12.0,
            ego_speed_mps=8.0,
            config={},
            front_gap_m=60.0,
            front_obstacle_speed_mps=11.0,
        )
        self.assertGreater(plan.target_speed_mps, 8.0)
        self.assertLessEqual(plan.target_speed_mps, 12.0)
        self.assertGreater(plan.idm_acceleration_mps2, 0.0)
        self.assertTrue(plan.continuous_following_active)
        self.assertIn("idm_following", plan.active_constraints)

    def test_idm_matches_a_slower_lead_without_emergency_stop(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=12.0,
            ego_speed_mps=12.0,
            config={},
            front_gap_m=18.0,
            front_obstacle_speed_mps=7.0,
        )
        self.assertLess(plan.target_speed_mps, 12.0)
        self.assertLess(plan.idm_acceleration_mps2, 0.0)
        self.assertFalse(plan.stop_goal_active)
        self.assertEqual(plan.limiting_owner, "idm_following")

    def test_pure_idm_applies_its_equilibrium_correction_at_dynamic_gap(self):
        ego_speed = 8.0
        lead_speed = 8.0
        desired_gap = 5.0 + 1.5 * ego_speed
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=12.0,
            ego_speed_mps=ego_speed,
            config={},
            front_gap_m=desired_gap,
            front_obstacle_speed_mps=lead_speed,
        )
        self.assertLess(plan.target_speed_mps, lead_speed)
        self.assertGreater(plan.target_speed_mps, lead_speed - 0.5)
        self.assertAlmostEqual(plan.desired_follow_gap_m, desired_gap)

    def test_idm_uses_small_lead_relative_catchup_when_gap_is_large(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=15.0,
            ego_speed_mps=10.0,
            config={},
            front_gap_m=50.0,
            front_obstacle_speed_mps=10.0,
        )
        self.assertGreater(plan.target_speed_mps, 10.0)
        self.assertLessEqual(plan.target_speed_mps, 12.0)

    def test_idm_may_temporarily_exceed_cruise_target_to_close_large_gap(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=12.0,
            ego_speed_mps=12.0,
            config={"following_max_catchup_delta_mps": 2.0},
            front_gap_m=60.0,
            front_obstacle_speed_mps=12.0,
        )
        self.assertGreater(plan.target_speed_mps, 12.0)
        self.assertLessEqual(plan.target_speed_mps, 14.0)
        self.assertEqual(plan.limiting_owner, "idm_following")

    def test_idm_catchup_never_exceeds_explicit_scenario_cap(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=12.5,
                stop_goal_active=False,
                reason="road_speed_limit",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=12.0,
            ego_speed_mps=12.0,
            config={"following_max_catchup_delta_mps": 2.0},
            front_gap_m=60.0,
            front_obstacle_speed_mps=12.0,
        )
        self.assertLessEqual(plan.target_speed_mps, 12.5)

    def test_idm_never_raises_scenario_speed_ceiling(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=9.0,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=12.0,
            ego_speed_mps=8.5,
            config={},
            front_gap_m=80.0,
            front_obstacle_speed_mps=12.0,
        )
        self.assertLessEqual(plan.target_speed_mps, 9.0)
        self.assertGreater(plan.idm_acceleration_mps2, 0.0)

    def test_emergency_gap_requests_stop(self):
        plan = self._plan(2.5)
        self.assertEqual(plan.target_speed_mps, 0.0)
        self.assertTrue(plan.stop_goal_active)
        self.assertIn("speed_plan_obstacle_emergency_stop", plan.reason)
        self.assertEqual(plan.limiting_owner, "emergency_stop")

    def test_scenario_cap_records_longitudinal_authority(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=1.5,
                stop_goal_active=False,
                reason="traffic_light_approach",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=3.0,
            ego_speed_mps=2.0,
            config={},
            front_gap_m=None,
        )
        self.assertEqual(plan.target_speed_mps, 1.5)
        self.assertEqual(plan.limiting_owner, "scenario_cap")
        self.assertEqual(plan.as_debug_fields()["speed_owner_scenario_cap_mps"], 1.5)

    def test_turn_does_not_use_a_fixed_speed_without_reference_geometry(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="intersection_turn_right",
            requested_speed_mps=4.0,
            ego_speed_mps=2.0,
            config={"full_intersection_turn_speed_cap_mps": 2.2},
            front_gap_m=None,
        )
        self.assertEqual(plan.target_speed_mps, 4.0)
        self.assertNotEqual(plan.limiting_owner, "turn_cap")
        self.assertNotIn("turn_cap", plan.active_constraints)

    def test_distant_upcoming_turn_preserves_cruise_speed(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="prepare_turn_right",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=15.6464,
            ego_speed_mps=15.0,
            config={
                "full_intersection_turn_speed_cap_mps": 5.0,
                "turn_approach_comfort_decel_mps2": 2.5,
                "turn_approach_entry_buffer_m": 5.0,
            },
            upcoming_turn_direction="right",
            upcoming_turn_distance_m=100.0,
        )
        self.assertAlmostEqual(plan.target_speed_mps, 15.6464)
        self.assertNotEqual(plan.limiting_owner, "turn_approach_cap")

    def test_upcoming_turn_waits_for_persistent_reference_curvature(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="prepare_turn_right",
            ),
            behavior_decision="lane_follow",
            requested_speed_mps=15.6464,
            ego_speed_mps=15.0,
            config={
                "full_intersection_turn_speed_cap_mps": 5.0,
                "turn_approach_comfort_decel_mps2": 2.5,
                "turn_approach_entry_buffer_m": 5.0,
            },
            upcoming_turn_direction="right",
            upcoming_turn_distance_m=35.0,
        )
        self.assertAlmostEqual(plan.target_speed_mps, 15.6464)
        self.assertNotEqual(plan.limiting_owner, "turn_approach_cap")
        self.assertEqual(plan.upcoming_turn_distance_m, 35.0)

    def test_stop_overrides_turn_approach_cap(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=0.0,
                stop_goal_active=True,
                reason="traffic_light_stop",
            ),
            behavior_decision="stop_at_intersection",
            requested_speed_mps=15.6464,
            ego_speed_mps=10.0,
            config={"full_intersection_turn_speed_cap_mps": 5.0},
            upcoming_turn_direction="right",
            upcoming_turn_distance_m=35.0,
        )
        self.assertEqual(plan.target_speed_mps, 0.0)
        self.assertTrue(plan.stop_goal_active)
        self.assertEqual(plan.limiting_owner, "normal_stop")

    def test_lane_change_cap_records_longitudinal_authority(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_change_left",
            requested_speed_mps=5.0,
            ego_speed_mps=3.68,
            config={
                "full_lane_change_dynamic_speed_cap_enabled": False,
                "full_lane_change_speed_cap_mps": 3.0,
            },
            front_gap_m=None,
        )
        self.assertEqual(plan.target_speed_mps, 3.0)
        self.assertEqual(plan.limiting_owner, "lane_change_cap")
        self.assertIn("lane_change_cap", plan.active_constraints)
        self.assertEqual(plan.lane_change_cap_mps, 3.0)
        self.assertIn("speed_plan_lane_change_cap", plan.reason)

    def test_dynamic_lane_change_cap_waits_for_winning_reference_geometry(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_change_left",
            requested_speed_mps=15.0,
            ego_speed_mps=12.0,
            config={
                "full_lane_change_dynamic_speed_cap_enabled": True,
                "full_lane_change_speed_cap_mps": 3.0,
            },
            front_gap_m=None,
        )
        self.assertEqual(plan.target_speed_mps, 15.0)
        self.assertIsNone(plan.lane_change_cap_mps)
        self.assertNotIn("lane_change_cap", plan.active_constraints)
        self.assertNotIn("speed_plan_lane_change_cap", plan.reason)

    def test_lane_change_cap_does_not_raise_an_already_slower_request(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_change_right",
            requested_speed_mps=2.0,
            ego_speed_mps=1.8,
            config={
                "full_lane_change_dynamic_speed_cap_enabled": False,
                "full_lane_change_speed_cap_mps": 3.0,
            },
            front_gap_m=None,
        )
        self.assertEqual(plan.target_speed_mps, 2.0)
        self.assertNotEqual(plan.limiting_owner, "lane_change_cap")

    def test_instrumentation_is_behaviorally_equivalent_to_legacy_speed_logic(self):
        decisions = ["lane_follow", "stop_at_intersection", "emergency_brake"]
        for decision in decisions:
            for scenario_cap in (None, 1.5, 4.0):
                for scenario_stop in (False, True):
                    for front_gap in (None, 2.5, 7.9, 30.0):
                        expected = self._legacy_result(
                            scenario_cap=scenario_cap,
                            scenario_stop=scenario_stop,
                            decision=decision,
                            requested=3.0,
                            ego_speed=2.0,
                            front_gap=front_gap,
                        )
                        plan = build_speed_plan(
                            scenario_decision=SimpleNamespace(
                                speed_cap_mps=scenario_cap,
                                stop_goal_active=scenario_stop,
                                reason="",
                            ),
                            behavior_decision=decision,
                            requested_speed_mps=3.0,
                            ego_speed_mps=2.0,
                            config={},
                            front_gap_m=front_gap,
                        )
                        actual = (
                            plan.target_speed_mps,
                            plan.stop_goal_active,
                            plan.continuous_following_active,
                        )
                        self.assertEqual(actual, expected)

    def test_speed_ceiling_prevents_candidate_from_raising_speed(self):
        result = enforce_speed_ceiling(
            proposed_target_mps=3.0,
            ceiling_mps=2.5,
            destination_state=[1.0, 2.0, 3.0, 0.0, 1],
            reference_samples=[
                {"x_ref_m": 1.0, "y_ref_m": 2.0, "v_ref_mps": 3.0},
                {"x_ref_m": 2.0, "y_ref_m": 2.0, "speed_ref_mps": 2.8},
            ],
        )
        self.assertEqual(result.target_speed_mps, 2.5)
        self.assertEqual(result.destination_state[2], 2.5)
        self.assertEqual(result.reference_samples[0]["v_ref_mps"], 2.5)
        self.assertEqual(result.reference_samples[1]["speed_ref_mps"], 2.5)
        self.assertTrue(result.applied)

    def test_committed_lane_change_ignores_prepare_turn_speed_caps(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                state="PREPARE_TURN",
                speed_cap_mps=2.0,
                stop_goal_active=False,
                reason="turn_prepare",
            ),
            behavior_decision="lane_change_right",
            requested_speed_mps=12.0,
            ego_speed_mps=11.0,
            config={
                "full_intersection_turn_speed_cap_mps": 2.2,
                "turn_approach_comfort_decel_mps2": 2.5,
                "turn_approach_entry_buffer_m": 5.0,
            },
            upcoming_turn_direction="right",
            upcoming_turn_distance_m=20.0,
            lane_change_commitment_active=True,
        )

        self.assertEqual(plan.target_speed_mps, 12.0)
        self.assertIsNone(plan.scenario_cap_mps)
        self.assertIsNone(plan.turn_approach_cap_mps)

    def test_lane_change_drops_normal_source_lane_idm_constraint(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                state="LANE_FOLLOW",
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_change_left",
            requested_speed_mps=12.0,
            ego_speed_mps=10.0,
            config={},
            front_gap_m=35.0,
            front_obstacle_speed_mps=4.0,
            lane_change_commitment_active=True,
            front_obstacle_is_source_lane=True,
        )

        self.assertEqual(plan.target_speed_mps, 12.0)
        self.assertFalse(plan.continuous_following_active)
        self.assertIn(
            "source_lane_following_suppressed", plan.active_constraints
        )

    def test_lane_change_keeps_target_lane_idm_constraint(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                state="LANE_FOLLOW",
                speed_cap_mps=None,
                stop_goal_active=False,
                reason="",
            ),
            behavior_decision="lane_change_left",
            requested_speed_mps=12.0,
            ego_speed_mps=10.0,
            config={},
            front_gap_m=18.0,
            front_obstacle_speed_mps=4.0,
            lane_change_commitment_active=True,
            front_obstacle_is_source_lane=False,
        )

        self.assertLess(plan.target_speed_mps, 12.0)
        self.assertTrue(plan.continuous_following_active)

    def test_committed_lane_change_does_not_suppress_real_stop(self):
        plan = build_speed_plan(
            scenario_decision=SimpleNamespace(
                state="TRAFFIC_LIGHT_STOP",
                speed_cap_mps=0.0,
                stop_goal_active=True,
                reason="red_light",
            ),
            behavior_decision="lane_change_right",
            requested_speed_mps=12.0,
            ego_speed_mps=11.0,
            config={},
            lane_change_commitment_active=True,
        )

        self.assertEqual(plan.target_speed_mps, 0.0)
        self.assertTrue(plan.stop_goal_active)

    def test_speed_ceiling_preserves_candidate_slowdown_and_geometry(self):
        result = enforce_speed_ceiling(
            proposed_target_mps=1.5,
            ceiling_mps=3.0,
            destination_state=[4.0, 5.0, 1.5, 0.2, 1],
            reference_samples=[
                {"x_ref_m": 4.0, "y_ref_m": 5.0, "v_ref_mps": 1.5},
            ],
        )
        self.assertEqual(result.target_speed_mps, 1.5)
        self.assertEqual(result.destination_state[:2], [4.0, 5.0])
        self.assertEqual(result.reference_samples[0]["x_ref_m"], 4.0)
        self.assertFalse(result.applied)

    def test_zero_speed_ceiling_remains_zero_through_reference(self):
        result = enforce_speed_ceiling(
            proposed_target_mps=2.0,
            ceiling_mps=0.0,
            destination_state=[1.0, 2.0, 2.0],
            reference_samples=[{"v_ref_mps": 2.0, "speed_mps": 2.0}],
        )
        self.assertEqual(result.target_speed_mps, 0.0)
        self.assertEqual(result.destination_state[2], 0.0)
        self.assertEqual(result.reference_samples[0]["v_ref_mps"], 0.0)
        self.assertEqual(result.reference_samples[0]["speed_mps"], 0.0)


if __name__ == "__main__":
    unittest.main()
