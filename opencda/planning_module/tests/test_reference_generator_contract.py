import unittest

from behavior_planner.reference_generator import select_reference_intent


class ReferenceGeneratorContractTests(unittest.TestCase):
    def test_lane_follow_tracks_lane_center_and_uses_route_as_hint(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=2,
            global_route_reference_allowed=True,
            traffic_control_lane_lock_active=False,
        )

        self.assertEqual(intent.mode, "lane_follow")
        self.assertEqual(intent.target_lane_id, 1)
        self.assertFalse(intent.follow_global_route_lane)
        self.assertEqual(intent.lateral_reference_source, "lane_center")
        self.assertEqual(intent.longitudinal_target_kind, "speed_profile")
        self.assertEqual(intent.route_role, "lane_choice_hint_only")

    def test_stop_keeps_lateral_lane_reference_and_uses_stop_target_longitudinally(self):
        intent = select_reference_intent(
            behavior_decision="stop_at_intersection",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=1,
            global_route_reference_allowed=False,
            traffic_control_lane_lock_active=True,
        )

        self.assertEqual(intent.mode, "stop")
        self.assertFalse(intent.follow_global_route_lane)
        self.assertEqual(intent.lateral_reference_source, "lane_center")
        self.assertEqual(intent.longitudinal_target_kind, "stop_target")
        self.assertEqual(intent.stop_target_role, "longitudinal_speed_target")

    def test_lane_change_uses_blended_lateral_reference(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            ego_in_junction=False,
            reference_target_lane_id=2,
            current_lane_id=1,
            route_optimal_lane_id=1,
            global_route_reference_allowed=False,
            traffic_control_lane_lock_active=False,
        )

        self.assertEqual(intent.mode, "lane_change")
        self.assertEqual(intent.target_lane_id, 2)
        self.assertEqual(intent.lateral_reference_source, "lane_change_blend")
        self.assertEqual(intent.longitudinal_target_kind, "speed_profile")


if __name__ == "__main__":
    unittest.main()
