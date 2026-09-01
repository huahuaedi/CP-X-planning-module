import unittest

from opencda.planning_module.pipeline.scenario_manager import (
    BOUNDARY_RECOVERY,
    BoundaryRecoveryRequest,
    CPXScenarioManager,
    INTERSECTION_TURN,
    LANE_FOLLOW,
    PREPARE_TURN,
    TRAFFIC_LIGHT_APPROACH,
    TRAFFIC_LIGHT_STOP,
    TURN_EXIT_STABILIZATION,
)


class CPXScenarioManagerTests(unittest.TestCase):
    def test_default_pipeline_keeps_boundary_recovery_diagnostic_only(self):
        manager = CPXScenarioManager()

        decision = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=0.4,
            ego_in_junction=False,
            current_road_option="RIGHT",
            next_macro_maneuver="right",
            sim_time_s=10.0,
            boundary_recovery_request=BoundaryRecoveryRequest(
                valid=True,
                active=True,
                clearance_m=-0.3,
                lateral_offset_m=0.3,
                heading_error_rad=-0.2,
                turn_direction="right",
            ),
        )

        self.assertEqual(decision.state, INTERSECTION_TURN)
        self.assertFalse(decision.boundary_recovery_active)

    def test_boundary_recovery_preempts_normal_turn_and_owns_speed(self):
        manager = CPXScenarioManager({
            "boundary_recovery_enabled": True,
            "boundary_recovery_speed_mps": 0.55,
        })

        decision = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=0.4,
            ego_in_junction=True,
            current_road_option="RIGHT",
            next_macro_maneuver="right",
            sim_time_s=10.0,
            boundary_recovery_request=BoundaryRecoveryRequest(
                valid=True,
                active=True,
                clearance_m=-0.3,
                lateral_offset_m=0.3,
                heading_error_rad=-0.2,
                turn_direction="right",
            ),
        )

        self.assertEqual(decision.state, BOUNDARY_RECOVERY)
        self.assertTrue(decision.boundary_recovery_active)
        self.assertEqual(
            decision.behavior_override_decision,
            "intersection_turn_right",
        )
        self.assertEqual(
            decision.behavior_override_lc_state,
            "BOUNDARY_RECOVERY_RIGHT",
        )
        self.assertAlmostEqual(decision.speed_cap_mps, 0.55)

    def test_boundary_recovery_releases_after_geometric_stability(self):
        manager = CPXScenarioManager({
            "boundary_recovery_enabled": True,
            "boundary_recovery_release_frames": 3,
        })
        manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=0.4,
            ego_in_junction=True,
            current_road_option="RIGHT",
            next_macro_maneuver="right",
            sim_time_s=1.0,
            boundary_recovery_request=BoundaryRecoveryRequest(
                valid=True,
                active=True,
                clearance_m=-0.2,
                lateral_offset_m=0.3,
                heading_error_rad=-0.2,
                turn_direction="right",
            ),
        )
        released = None
        for index in range(3):
            released = manager.update(
                traffic_state="unknown",
                stop_target=None,
                stop_forward_m=0.0,
                stop_target_reliable=False,
                ego_speed_mps=0.5,
                ego_in_junction=True,
                current_road_option="RIGHT",
                next_macro_maneuver="right",
                sim_time_s=1.1 + 0.1 * index,
                boundary_recovery_request=BoundaryRecoveryRequest(
                    valid=True,
                    active=False,
                    clearance_m=0.2,
                    lateral_offset_m=0.1,
                    heading_error_rad=0.04,
                    turn_direction="right",
                ),
            )

        self.assertEqual(released.state, INTERSECTION_TURN)
        self.assertFalse(released.boundary_recovery_active)

    def test_red_stop_has_priority_over_boundary_recovery(self):
        manager = CPXScenarioManager({
            "boundary_recovery_enabled": True,
        })

        decision = manager.update(
            traffic_state="red",
            stop_target={"x_m": 5.0, "y_m": 0.0},
            stop_forward_m=5.0,
            stop_target_reliable=True,
            ego_speed_mps=1.0,
            ego_in_junction=True,
            current_road_option="RIGHT",
            next_macro_maneuver="right",
            sim_time_s=1.0,
            boundary_recovery_request=BoundaryRecoveryRequest(
                valid=True,
                active=True,
                clearance_m=-0.3,
                turn_direction="right",
            ),
        )

        self.assertEqual(decision.state, TRAFFIC_LIGHT_STOP)
        self.assertTrue(decision.stop_goal_active)

    def test_route_turn_lookahead_enters_prepare_turn(self):
        manager = CPXScenarioManager({
            "full_intersection_turn_speed_cap_mps": 2.2,
            "scenario_turn_prepare_speed_cap_mps": 2.8,
        })

        decision = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=3.0,
            ego_in_junction=False,
            current_road_option="LANEFOLLOW",
            next_macro_maneuver="straight",
            sim_time_s=1.0,
            upcoming_turn_direction="right",
            upcoming_turn_distance_m=12.0,
        )

        self.assertEqual(decision.state, PREPARE_TURN)
        self.assertEqual(
            decision.behavior_override_decision,
            "lane_follow",
        )
        self.assertEqual(decision.behavior_override_lc_state, "LANE_KEEP")
        self.assertEqual(decision.turn_direction, "right")
        self.assertFalse(decision.turn_latched)
        # Scenario selection reserves the turn direction but does not impose
        # an early fixed low-speed cap. SpeedPlanner owns distance-based
        # deceleration toward the turn-entry speed.
        self.assertEqual(decision.speed_cap_mps, 8.0)

    def test_far_macro_turn_does_not_enter_prepare_turn(self):
        manager = CPXScenarioManager({
            "target_speed_mps": 5.0,
            "scenario_turn_prepare_lookahead_m": 15.0,
        })

        decision = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=5.0,
            ego_in_junction=False,
            current_road_option="LANEFOLLOW",
            next_macro_maneuver="Turn Left",
            sim_time_s=1.0,
            upcoming_turn_direction="left",
            upcoming_turn_distance_m=121.98,
        )

        self.assertEqual(decision.state, LANE_FOLLOW)
        self.assertEqual(decision.behavior_override_decision, "")

    def test_red_far_is_approach_not_stop(self):
        manager = CPXScenarioManager({
            "target_speed_mps": 8.0,
            "traffic_stop_min_commit_distance_m": 10.0,
            "traffic_stop_approach_slow_distance_m": 22.0,
        })

        decision = manager.update(
            traffic_state="red",
            stop_target={"x_m": 50.0, "y_m": 0.0},
            stop_forward_m=50.0,
            stop_target_reliable=True,
            ego_speed_mps=4.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=1.0,
        )

        self.assertEqual(decision.state, TRAFFIC_LIGHT_APPROACH)
        self.assertEqual(decision.behavior_signal_state, "unknown")
        self.assertFalse(decision.stop_goal_active)
        self.assertIsNone(decision.behavior_stop_target)

    def test_red_near_commits_to_stop(self):
        manager = CPXScenarioManager({
            "target_speed_mps": 8.0,
            "traffic_stop_min_commit_distance_m": 10.0,
        })

        decision = manager.update(
            traffic_state="red",
            stop_target={"x_m": 8.0, "y_m": 0.0},
            stop_forward_m=8.0,
            stop_target_reliable=True,
            ego_speed_mps=2.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=1.0,
        )

        self.assertEqual(decision.state, TRAFFIC_LIGHT_STOP)
        self.assertEqual(decision.behavior_signal_state, "red")
        self.assertTrue(decision.stop_goal_active)
        self.assertEqual(decision.behavior_override_decision, "stop_at_intersection")

    def test_route_option_latches_intersection_turn(self):
        manager = CPXScenarioManager({
            "full_intersection_turn_speed_cap_mps": 2.2,
            "scenario_turn_exit_hold_s": 1.0,
            "scenario_turn_exit_stable_frames": 2,
        })

        first = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=3.0,
            ego_in_junction=True,
            current_road_option="LEFT",
            next_macro_maneuver="left",
            sim_time_s=10.0,
        )
        second = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=3.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=10.5,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=True,
        )
        third = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=3.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=12.0,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=True,
        )

        self.assertEqual(first.state, INTERSECTION_TURN)
        self.assertEqual(first.behavior_override_decision, "intersection_turn_left")
        self.assertEqual(second.state, TURN_EXIT_STABILIZATION)
        self.assertEqual(second.behavior_override_decision, "intersection_turn_left")
        self.assertEqual(third.state, LANE_FOLLOW)

    def test_active_turn_is_not_preempted_by_downstream_red_light(self):
        manager = CPXScenarioManager({
            "scenario_turn_exit_stable_frames": 3,
        })
        manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=2.0,
            ego_in_junction=True,
            current_road_option="LEFT",
            next_macro_maneuver="Turn Left",
            sim_time_s=1.0,
        )

        decision = manager.update(
            traffic_state="red",
            stop_target={"x_m": 53.0, "y_m": 0.0},
            stop_forward_m=53.0,
            stop_target_reliable=True,
            ego_speed_mps=2.0,
            ego_in_junction=False,
            current_road_option="STRAIGHT",
            next_macro_maneuver="Continue Straight",
            sim_time_s=1.1,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=False,
            turn_exit_heading_error_rad=0.2,
            turn_exit_lateral_m=0.8,
        )

        self.assertEqual(decision.state, TURN_EXIT_STABILIZATION)
        self.assertEqual(
            decision.behavior_override_decision,
            "intersection_turn_left",
        )

    def test_large_junction_polygon_activates_lead_in_without_early_exit(self):
        manager = CPXScenarioManager({
            "scenario_turn_exit_stable_frames": 1,
        })
        lead_in = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=5.0,
            # Town06 marks the entry road as junction well before the route
            # actually reaches its LEFT connector.
            ego_in_junction=True,
            current_road_option="LaneFollow",
            next_macro_maneuver="left",
            sim_time_s=1.0,
            upcoming_turn_direction="left",
            upcoming_turn_distance_m=12.95,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=True,
        )
        held = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=5.0,
            ego_in_junction=True,
            current_road_option="LaneFollow",
            next_macro_maneuver="left",
            sim_time_s=1.05,
            upcoming_turn_direction="left",
            upcoming_turn_distance_m=12.5,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=True,
        )

        self.assertEqual(lead_in.state, INTERSECTION_TURN)
        self.assertEqual(held.state, INTERSECTION_TURN)
        self.assertFalse(manager._turn_connector_seen)
        self.assertNotEqual(held.state, TURN_EXIT_STABILIZATION)

    def test_turn_exit_waits_for_outgoing_route_alignment(self):
        manager = CPXScenarioManager({
            "scenario_turn_exit_hold_s": 0.5,
            "scenario_turn_exit_stable_frames": 1,
        })
        manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=2.0,
            ego_in_junction=True,
            current_road_option="RIGHT",
            next_macro_maneuver="right",
            sim_time_s=1.0,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=False,
            turn_exit_heading_error_rad=0.5,
            turn_exit_lateral_m=0.4,
        )

        held = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=1.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=3.0,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=False,
            turn_exit_heading_error_rad=0.47,
            turn_exit_lateral_m=0.6,
        )
        released = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=1.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=3.1,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=True,
            turn_exit_heading_error_rad=0.05,
            turn_exit_lateral_m=0.2,
        )

        self.assertEqual(held.state, TURN_EXIT_STABILIZATION)
        self.assertIn("turn_exit_stabilization", held.reason)
        self.assertEqual(released.state, LANE_FOLLOW)

    def test_aligned_turn_exit_releases_inside_long_junction_lane(self):
        manager = CPXScenarioManager({
            "scenario_turn_exit_stable_frames": 1,
        })
        manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=2.0,
            ego_in_junction=True,
            current_road_option="RIGHT",
            next_macro_maneuver="right",
            sim_time_s=1.0,
        )

        released = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=1.0,
            # Town06 connector 5960149 remains marked as a junction after
            # the vehicle is already aligned to the outgoing straight.
            ego_in_junction=True,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=2.0,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=True,
            turn_exit_heading_error_rad=0.04,
            turn_exit_lateral_m=0.05,
        )

        self.assertEqual(released.state, LANE_FOLLOW)

    def test_post_turn_lane_change_macro_waits_for_stable_exit_alignment(self):
        manager = CPXScenarioManager({
            "scenario_turn_exit_hold_s": 3.0,
            "scenario_turn_exit_stable_frames": 3,
        })
        manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=3.0,
            ego_in_junction=True,
            current_road_option="LEFT",
            next_macro_maneuver="Turn Left",
            sim_time_s=10.0,
        )

        held_in_junction = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=3.0,
            ego_in_junction=True,
            # Both the AD-map route option and junction flag can lag behind
            # the AD-map macro at a segment boundary.
            current_road_option="LEFT",
            next_macro_maneuver="Lane Change Right",
            sim_time_s=10.1,
            upcoming_turn_direction="right",
            turn_exit_alignment_valid=True,
            turn_exit_aligned=False,
        )

        self.assertEqual(held_in_junction.state, INTERSECTION_TURN)
        self.assertEqual(
            held_in_junction.behavior_override_decision,
            "intersection_turn_left",
        )

        decisions = []
        for frame in range(3):
            decisions.append(manager.update(
                traffic_state="unknown",
                stop_target=None,
                stop_forward_m=0.0,
                stop_target_reliable=False,
                ego_speed_mps=3.0,
                ego_in_junction=False,
                current_road_option="LaneFollow",
                next_macro_maneuver="Lane Change Right",
                sim_time_s=10.2 + 0.1 * frame,
                turn_exit_alignment_valid=True,
                turn_exit_aligned=True,
                turn_exit_heading_error_rad=0.02,
                turn_exit_lateral_m=0.1,
            ))

        self.assertEqual(decisions[0].state, "TURN_EXIT_STABILIZATION")
        self.assertEqual(decisions[1].state, "TURN_EXIT_STABILIZATION")
        self.assertEqual(decisions[2].state, LANE_FOLLOW)
        self.assertEqual(manager._turn_direction, "")

    def test_turn_exit_hold_prevents_early_lane_follow_contract_switch(self):
        manager = CPXScenarioManager({
            "scenario_turn_exit_hold_s": 3.0,
            "scenario_turn_exit_stable_frames": 2,
        })
        manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=2.0,
            ego_in_junction=True,
            current_road_option="RIGHT",
            next_macro_maneuver="right",
            sim_time_s=10.0,
        )

        held = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=2.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=12.5,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=True,
        )
        released = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=2.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="straight",
            sim_time_s=13.1,
            turn_exit_alignment_valid=True,
            turn_exit_aligned=True,
        )

        self.assertEqual(held.state, TURN_EXIT_STABILIZATION)
        self.assertEqual(
            held.behavior_override_decision,
            "intersection_turn_right",
        )
        self.assertEqual(released.state, LANE_FOLLOW)

    def test_next_macro_turn_does_not_trigger_far_before_junction(self):
        manager = CPXScenarioManager({})

        decision = manager.update(
            traffic_state="unknown",
            stop_target=None,
            stop_forward_m=0.0,
            stop_target_reliable=False,
            ego_speed_mps=4.0,
            ego_in_junction=False,
            current_road_option="LaneFollow",
            next_macro_maneuver="left",
            sim_time_s=1.0,
        )

        self.assertEqual(decision.state, LANE_FOLLOW)
        self.assertEqual(decision.behavior_override_decision, "")


if __name__ == "__main__":
    unittest.main()
