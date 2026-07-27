import unittest

from opencda.planning_module.pipeline.scenario_manager import (
    CPXScenarioManager,
    INTERSECTION_TURN,
    LANE_FOLLOW,
    PREPARE_TURN,
    TRAFFIC_LIGHT_APPROACH,
    TRAFFIC_LIGHT_STOP,
)


class CPXScenarioManagerTests(unittest.TestCase):
    def test_carla_turn_lookahead_enters_prepare_turn(self):
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
            "intersection_turn_right",
        )
        self.assertEqual(decision.speed_cap_mps, 2.8)

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
        )

        self.assertEqual(first.state, INTERSECTION_TURN)
        self.assertEqual(first.behavior_override_decision, "intersection_turn_left")
        self.assertEqual(second.state, INTERSECTION_TURN)
        self.assertEqual(second.behavior_override_decision, "intersection_turn_left")
        self.assertEqual(third.state, LANE_FOLLOW)

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
