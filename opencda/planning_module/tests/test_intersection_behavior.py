import unittest

from behavior_planner.planner import (
    RuleBasedBehaviorPlanner,
    evaluate_intersection_obstacle_response,
    intersection_route_follow_maneuver,
)


class RuleBasedBehaviorPlannerIntersectionTests(unittest.TestCase):
    def test_confirmed_static_obstacle_local_avoidance_enters_execute_immediately(self):
        planner = RuleBasedBehaviorPlanner(
            lane_keep_min_hold_s=10.0,
            prepare_lane_change_min_hold_s=10.0,
        )

        result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            current_time_s=1.0,
            lane_prediction_risks={2: {"risk": False}},
            preferred_target_lane_id=2,
            local_avoidance_target_lane_id=2,
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(result["target_lane_id"], 2)
        self.assertEqual(result["lc_state"], "EXECUTE_LANE_CHANGE_LEFT")

    def test_static_obstacle_replan_failure_has_independent_stop_state(self):
        planner = RuleBasedBehaviorPlanner()

        blocked = planner.update(
            lane_safety_scores={1: 0.0},
            ego_lane_id=1,
            mode="INTERSECTION",
            static_obstacle_stop_active=True,
        )
        released = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            mode="INTERSECTION",
            traffic_signal_state="green",
            static_obstacle_stop_active=False,
        )

        self.assertEqual(blocked["decision"], "static_obstacle_stop")
        self.assertEqual(blocked["lc_state"], "STATIC_OBSTACLE_STOP")
        self.assertEqual(int(blocked["target_lane_id"]), 1)
        self.assertEqual(released["decision"], "lane_follow")
        self.assertNotEqual(released["lc_state"], "STATIC_OBSTACLE_STOP")

    def test_intersection_lane_follow_maneuver_after_reaching_leftmost_turn_lane(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="left",
            decision="lane_follow",
            target_lane_id=4,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "left")

    def test_intersection_lane_follow_maneuver_after_reaching_rightmost_turn_lane(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="right",
            decision="lane_follow",
            target_lane_id=1,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "right")

    def test_intersection_lane_follow_maneuver_does_not_activate_during_lane_change(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="left",
            decision="lane_change_left",
            target_lane_id=3,
            available_lane_ids=[1, 2, 3, 4],
            current_road_option="LaneFollow",
        )

        self.assertEqual(maneuver, "left")

    def test_intersection_lane_follow_maneuver_activates_when_blue_dot_is_on_turn_block(self):
        maneuver = intersection_route_follow_maneuver(
            mode="INTERSECTION",
            next_macro_maneuver="left",
            decision="lane_change_left",
            target_lane_id=2,
            available_lane_ids=[1, 2],
            current_road_option="LEFT",
        )

        self.assertEqual(maneuver, "lane_follow")

    def test_normal_mode_keeps_route_lane_when_it_is_still_safe(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.6, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_normal_mode_leaves_route_lane_when_front_obstacle_blocks_it(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_normal_mode_leaves_route_lane_when_optimal_lane_score_is_unsafe_even_without_front_obstacle(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.49, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={},
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_normal_mode_waits_for_safe_target_lane_before_detour(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.10},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_candidate_evaluation_reports_selected_detour_and_rejections(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9, 3: 0.05},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(result["selected_candidate"], "lane_change_left_to_2")
        self.assertIn("candidate_scores", result)
        self.assertTrue(
            any(
                candidate["name"] == "lane_change_left_to_2"
                and candidate["status"] == "selected"
                for candidate in result["candidate_scores"]
            )
        )
        self.assertTrue(
            any(
                candidate["name"] == "lane_change_left_to_3"
                and candidate["reason"] in {"not_adjacent", "target_lane_safety"}
                for candidate in result["rejected_candidates"]
            )
        )

    def test_candidate_evaluation_rejects_prediction_risky_lane_change(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": True, "min_ttc_s": 1.1}},
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(result["selected_candidate"], "lane_keep")
        self.assertTrue(
            any(
                candidate["name"] == "lane_change_left_to_2"
                and candidate["reason"] == "prediction_risk"
                for candidate in result["rejected_candidates"]
            )
        )

    def test_normal_mode_does_not_leave_optimal_lane_until_it_drops_below_unsafe_threshold(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
            optimal_lane_unsafe_threshold=0.50,
        )

        result = planner.update(
            lane_safety_scores={1: 0.51, 2: 0.95},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_normal_mode_keeps_current_safe_lane_even_if_route_lane_is_safe_too(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 1.0},
            ego_lane_id=1,
            selected_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=2,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={},
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_normal_mode_returns_to_route_lane_when_current_selected_lane_is_no_longer_safe(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.2, 2: 1.0},
            ego_lane_id=1,
            selected_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=2,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={},
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_normal_mode_does_not_request_left_lane_change_when_no_left_lane_exists(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=2,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={},
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_intersection_mode_moves_toward_route_lane_when_it_differs(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=4,
            next_macro_maneuver="left",
        )
        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_intersection_mode_moves_toward_route_lane_for_right_turn_too(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.2, 2: 0.3, 3: 0.9, 4: 1.0},
            ego_lane_id=4,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )
        self.assertEqual(result["decision"], "lane_change_right")
        self.assertEqual(int(result["target_lane_id"]), 3)

    def test_intersection_mode_keeps_lane_after_reaching_route_lane(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        left_result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.5, 3: 0.6},
            ego_lane_id=3,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
        )
        self.assertEqual(left_result["decision"], "lane_follow")
        self.assertEqual(int(left_result["target_lane_id"]), 3)

        planner.reset()
        right_result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.5, 3: 0.6},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )
        self.assertEqual(right_result["decision"], "lane_follow")
        self.assertEqual(int(right_result["target_lane_id"]), 1)

    def test_intersection_mode_respects_planner_selected_lane_while_idle(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 1.0},
            ego_lane_id=2,
            selected_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)
        self.assertEqual(int(result["selected_lane_id"]), 1)

    def test_intersection_mode_releases_only_after_ego_match_reaches_route_lane(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first_result = planner.update(
            lane_safety_scores={1: 1.0, 2: 1.0},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )

        self.assertEqual(first_result["decision"], "lane_change_right")
        self.assertEqual(int(first_result["target_lane_id"]), 1)

        second_result = planner.update(
            lane_safety_scores={1: 1.0, 2: 1.0},
            ego_lane_id=2,
            selected_lane_id=int(first_result["selected_lane_id"]),
            ego_lateral_offset_m=1.5,
            ego_heading_error_rad=0.15,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )

        self.assertEqual(second_result["decision"], "lane_change_right")
        self.assertEqual(int(second_result["target_lane_id"]), 1)

        completed_result = planner.update(
            lane_safety_scores={1: 1.0, 2: 1.0},
            ego_lane_id=1,
            selected_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )

        self.assertEqual(completed_result["decision"], "lane_follow")
        self.assertEqual(int(completed_result["target_lane_id"]), 1)
        self.assertEqual(str(completed_result["lc_state"]), "IDLE")

    def test_intersection_mode_moves_toward_route_lane_for_straight_maneuver(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 0.1, 2: 0.9, 3: 0.2},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="straight",
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_intersection_mode_does_not_request_left_lane_change_when_no_left_lane_exists(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        result = planner.update(
            lane_safety_scores={1: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=2,
            next_macro_maneuver="left",
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_intersection_mode_waits_for_safe_left_target_lane(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.10, 3: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_intersection_mode_reroutes_near_stop_when_optimal_lane_is_blocked(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
            optimal_lane_unsafe_threshold=0.50,
            intersection_reroute_stop_distance_threshold_m=20.0,
        )

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.9, 3: 0.09},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
            traffic_stop_target={
                "distance_m": 19.0,
                "road_id": 10,
                "section_id": 0,
                "lane_id": 1,
                "x_m": 0.0,
                "y_m": 12.0,
            },
        )

        self.assertEqual(result["decision"], "reroute")
        self.assertEqual(int(result["target_lane_id"]), 1)
        self.assertEqual(len(result["reroute_messages"]), 1)
        self.assertEqual(result["reroute_messages"][0]["type"], "lane_closure")
        self.assertEqual(int(result["reroute_messages"][0]["road_id"]), 10)
        self.assertEqual(result["reroute_messages"][0]["lane_ids"], [3])

    def test_intersection_mode_holds_lane_when_optimal_lane_is_blocked_but_stop_is_far(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
            optimal_lane_unsafe_threshold=0.50,
            intersection_reroute_stop_distance_threshold_m=20.0,
        )

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.9, 3: 0.09},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
            traffic_stop_target={
                "distance_m": 21.0,
                "road_id": 10,
                "section_id": 0,
                "lane_id": 1,
                "x_m": 0.0,
                "y_m": 12.0,
            },
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(int(result["target_lane_id"]), 1)

    def test_intersection_mode_keeps_lane_change_when_optimal_lane_is_still_feasible(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
            optimal_lane_unsafe_threshold=0.50,
            intersection_reroute_stop_distance_threshold_m=20.0,
        )

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.9, 3: 0.49},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
            traffic_stop_target={
                "distance_m": 19.0,
                "road_id": 10,
                "section_id": 0,
                "lane_id": 1,
                "x_m": 0.0,
                "y_m": 12.0,
            },
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_intersection_mode_does_not_repeat_acknowledged_blocked_optimal_lane_reroute(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
            optimal_lane_unsafe_threshold=0.50,
            intersection_reroute_stop_distance_threshold_m=20.0,
        )

        first_result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.9, 3: 0.09},
            ego_lane_id=1,
            selected_lane_id=3,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
            traffic_stop_target={
                "distance_m": 19.0,
                "road_id": 10,
                "section_id": 0,
                "lane_id": 1,
                "x_m": 0.0,
                "y_m": 12.0,
            },
        )

        self.assertEqual(first_result["decision"], "reroute")
        planner.acknowledge_reroute_success(
            [message["id"] for message in first_result["reroute_messages"]]
        )

        second_result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.9, 3: 0.09},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
            traffic_stop_target={
                "distance_m": 19.0,
                "road_id": 10,
                "section_id": 0,
                "lane_id": 1,
                "x_m": 0.0,
                "y_m": 12.0,
            },
        )

        self.assertEqual(second_result["decision"], "lane_follow")
        self.assertEqual(int(second_result["target_lane_id"]), 1)

    def test_intersection_mode_lane_changes_when_target_lane_is_safe(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 1.0, 2: 0.11, 3: 1.0},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=3,
            next_macro_maneuver="left",
        )

        self.assertEqual(result["decision"], "lane_change_left")
        self.assertEqual(int(result["target_lane_id"]), 2)

    def test_lane_change_completes_to_lane_follow_on_detour_lane(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(first["decision"], "lane_change_left")

        second = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(second["decision"], "lane_follow")

    def test_lane_id_match_does_not_complete_lane_change_before_geometry_converges(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lateral_complete_m=0.35,
            heading_complete_rad=0.10,
        )

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(first["decision"], "lane_change_left")

        not_converged = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=2,
            ego_lateral_offset_m=0.60,
            ego_heading_error_rad=0.15,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
            lane_change_completion_allowed=False,
        )
        self.assertEqual(not_converged["decision"], "lane_change_left")
        self.assertEqual(
            not_converged["lc_state"],
            "EXECUTE_LANE_CHANGE_LEFT",
        )

        converged = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=2,
            ego_lateral_offset_m=0.10,
            ego_heading_error_rad=0.03,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
            lane_change_completion_allowed=True,
        )
        self.assertEqual(converged["decision"], "lane_follow")

    def test_normal_mode_holds_current_safe_lane_when_route_lane_becomes_safe_again(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(first["decision"], "lane_change_left")

        second = planner.update(
            lane_safety_scores={1: 0.9, 2: 0.8},
            ego_lane_id=2,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={},
        )
        self.assertEqual(second["decision"], "lane_follow")
        self.assertEqual(int(second["target_lane_id"]), 2)

    def test_ongoing_lane_change_reverses_when_target_lane_becomes_worse_than_previous_lane(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_abort_safety_threshold=0.50,
        )

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(first["decision"], "lane_change_left")
        self.assertEqual(int(first["target_lane_id"]), 2)

        second = planner.update(
            lane_safety_scores={1: 0.8, 2: 0.2},
            ego_lane_id=1,
            ego_lateral_offset_m=0.4,
            ego_heading_error_rad=0.05,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={},
        )
        self.assertEqual(second["decision"], "lane_change_right")
        self.assertEqual(int(second["target_lane_id"]), 1)

    def test_mode_change_resets_lane_change_loop(self):
        planner = RuleBasedBehaviorPlanner(hysteresis_delta=0.05)

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.9},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="NORMAL",
            route_optimal_lane_id=1,
            next_macro_maneuver="straight",
            front_obstacle_distance_by_lane={1: 4.0},
        )
        self.assertEqual(first["decision"], "lane_change_left")

        second = planner.update(
            lane_safety_scores={1: 0.9, 2: 0.1},
            ego_lane_id=1,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION",
            route_optimal_lane_id=1,
            next_macro_maneuver="right",
        )
        self.assertEqual(second["decision"], "lane_follow")

    def test_intersection_obstacle_response_follows_moving_obstacle_speed(self):
        response = evaluate_intersection_obstacle_response(
            mode="INTERSECTION",
            front_obstacle_speed_mps=3.5,
            original_max_velocity_mps=8.0,
            moving_obstacle_speed_threshold_mps=0.5,
            route_lane_safety_score=0.2,
            static_obstacle_replan_lane_safety_threshold=0.5,
        )

        self.assertTrue(bool(response["follow_moving_obstacle"]))
        self.assertFalse(bool(response["request_static_obstacle_replan"]))
        self.assertAlmostEqual(float(response["speed_cap_mps"]), 3.5)

    def test_intersection_obstacle_response_requests_replan_for_static_obstacle(self):
        response = evaluate_intersection_obstacle_response(
            mode="INTERSECTION",
            front_obstacle_speed_mps=0.0,
            original_max_velocity_mps=8.0,
            moving_obstacle_speed_threshold_mps=0.5,
            route_lane_safety_score=0.4,
            static_obstacle_replan_lane_safety_threshold=0.5,
        )

        self.assertFalse(bool(response["follow_moving_obstacle"]))
        self.assertTrue(bool(response["request_static_obstacle_replan"]))
        self.assertAlmostEqual(float(response["speed_cap_mps"]), 8.0)

    def test_intersection_obstacle_response_does_not_request_replan_when_lane_is_still_safe(self):
        response = evaluate_intersection_obstacle_response(
            mode="INTERSECTION",
            front_obstacle_speed_mps=0.0,
            original_max_velocity_mps=8.0,
            moving_obstacle_speed_threshold_mps=0.5,
            route_lane_safety_score=0.5,
            static_obstacle_replan_lane_safety_threshold=0.5,
        )

        self.assertFalse(bool(response["follow_moving_obstacle"]))
        self.assertFalse(bool(response["request_static_obstacle_replan"]))
        self.assertAlmostEqual(float(response["speed_cap_mps"]), 8.0)


class DisplayLaneChangeDirectionTests(unittest.TestCase):
    """`_candidate_record` debug/rejected-candidate labels must agree with
    what `_start_one_step_lane_change`/`_adjacent_lane_id` would actually
    drive -- position within this tick's available_lane_ids, not raw id
    magnitude, since opaque AD-map lane IDs have no numeric lateral order."""

    def test_uses_list_position_when_both_ids_are_available(self):
        # Raw magnitude would say "right" (5 < 9), but position in this
        # tick's available list says "left" -- position must win.
        decision = RuleBasedBehaviorPlanner._display_lane_change_direction(
            desired_lane_id=9,
            ego_lane_id=5,
            available_lane_ids=[5, 9],
        )

        self.assertEqual(decision, "lane_change_left")

    def test_falls_back_to_raw_magnitude_when_ego_id_is_stale(self):
        # ego_lane_id is not in the freshly-built available list (e.g. a
        # stale AD-map lane ID from outside the local frame) -- there is no
        # position to compare, so this falls back
        # to the old magnitude comparison rather than raising.
        decision = RuleBasedBehaviorPlanner._display_lane_change_direction(
            desired_lane_id=2,
            ego_lane_id=99,
            available_lane_ids=[1, 2, 3],
        )

        self.assertEqual(decision, "lane_change_right")

    def test_agrees_with_adjacent_lane_id_direction(self):
        available = [1, 2, 3]
        target_from_left = RuleBasedBehaviorPlanner._adjacent_lane_id(
            reference_lane_id=2,
            available_lane_ids=available,
            direction="left",
        )
        decision = RuleBasedBehaviorPlanner._display_lane_change_direction(
            desired_lane_id=target_from_left,
            ego_lane_id=2,
            available_lane_ids=available,
        )

        self.assertEqual(decision, "lane_change_left")


if __name__ == "__main__":
    unittest.main()
