import unittest

from pipeline.maneuver_manager import ManeuverManager


class ManeuverManagerTests(unittest.TestCase):
    def test_lane_change_identity_is_installed_atomically(self):
        manager = ManeuverManager()
        state = manager.begin_lane_change(
            option="lane_change_right", phase="executing",
            source_lane_id=1, target_lane_id=2, target_speed_mps=8.0,
            completion_reference=[{"x_ref_m": 1.0, "y_ref_m": 2.0}],
        )
        self.assertTrue(state.active)
        self.assertEqual((state.source_lane_id, state.target_lane_id), (1, 2))

    def test_progress_is_monotonic(self):
        manager = ManeuverManager()
        manager.begin_lane_change("lane_change_right", "executing", 1, 2, 8.0, [])
        manager.advance_lane_change(progress=0.7, progress_index=12, progress_s_m=18.0)
        state = manager.advance_lane_change(progress=0.4, progress_index=8, progress_s_m=10.0)
        self.assertEqual((state.progress, state.progress_index, state.progress_s_m),
                         (0.7, 12, 18.0))

    def test_completion_and_abandonment_are_explicit(self):
        manager = ManeuverManager()
        manager.begin_lane_change("lane_change_left", "executing", 2, 1, 7.0, [])
        self.assertTrue(manager.complete_lane_change("contract_satisfied"))
        self.assertFalse(manager.lane_change.active)
        self.assertEqual(manager.lane_change.completed_option, "lane_change_left")

        manager.begin_lane_change("lane_change_right", "executing", 1, 2, 7.0, [])
        self.assertTrue(manager.abandon_lane_change("route_changed"))
        self.assertFalse(manager.lane_change.active)
        self.assertNotEqual(manager.lane_change.completed_option, "lane_change_right")

    def test_geometric_completion_is_latched_during_handoff(self):
        manager = ManeuverManager()
        manager.begin_lane_change(
            "lane_change_right", "executing", 1, 2, 7.0, []
        )
        manager.begin_lane_change_stabilization()
        manager.record_lane_change_completion_evidence(
            5, {"reason": "converged"}, geometrically_complete=True
        )
        manager.record_lane_change_completion_evidence(
            0, {"reason": "temporary_error"}, geometrically_complete=False
        )
        self.assertTrue(manager.lane_change.geometry_completion_latched)

    def test_manager_is_lane_change_transition_owner(self):
        manager = ManeuverManager()
        manager.begin_lane_change(
            "lane_change_left", "executing", 1, 2, 7.0, []
        )
        self.assertEqual(
            manager.lane_change_handoff_transition(
                geometry_ready=False
            ).action,
            "hold",
        )
        self.assertEqual(
            manager.lane_change_handoff_transition(
                geometry_ready=True
            ).action,
            "start_stabilization",
        )
        manager.begin_lane_change_stabilization()
        self.assertEqual(
            manager.tick_lane_change_stabilization(timeout_frames=2).action,
            "hold",
        )

    def test_completion_waits_for_transition_arc(self):
        manager = ManeuverManager()
        manager.begin_lane_change(
            "lane_change_left", "executing", 1, 2, 7.0, []
        )
        manager.begin_lane_change_stabilization()
        waiting = manager.accept_lane_change_completion(
            stable_frames=5,
            debug={},
            geometrically_complete=True,
            completion_reason="converged",
            transition_progress_m=4.0,
            transition_arc_m=10.0,
        )
        complete = manager.accept_lane_change_completion(
            stable_frames=6,
            debug={},
            geometrically_complete=True,
            completion_reason="converged",
            transition_progress_m=10.0,
            transition_arc_m=10.0,
        )

        self.assertEqual(waiting.action, "hold")
        self.assertEqual(waiting.reason, "transition_arc_incomplete")
        self.assertEqual(complete.action, "complete")
        self.assertTrue(manager.lane_change.active)

    def test_post_turn_phase_has_no_geometry_interface(self):
        manager = ManeuverManager()
        manager.turn.decision = "intersection_turn_right"
        manager.turn.phase = "turn"
        action = manager.resolve_post_turn_phase(
            "lane_follow", "LANE_FOLLOW", True, False, 0.0, 12.0, False)
        self.assertEqual(action, "activate")
        self.assertEqual(manager.turn.phase, "post_turn")
        self.assertFalse(hasattr(manager, "update"))
        self.assertFalse(hasattr(manager, "active_plan"))

    def test_post_turn_releases_only_after_distance_and_alignment(self):
        manager = ManeuverManager()
        manager.turn.decision = "intersection_turn_left"
        manager.turn.phase = "post_turn"
        self.assertEqual(manager.resolve_post_turn_phase(
            "lane_follow", "LANE_FOLLOW", False, True, 6.0, 12.0, True), "hold")
        self.assertEqual(manager.resolve_post_turn_phase(
            "lane_follow", "LANE_FOLLOW", False, True, 12.0, 12.0, True), "complete")
        self.assertFalse(manager.turn.active)

    def test_new_lane_change_requires_geometry_and_handoff_before_turn(self):
        manager = ManeuverManager()

        denied = manager.lane_change_start_feasibility(
            authorization_allowed=True,
            distance_to_turn_m=7.76,
            geometry_arc_m=10.0,
            handoff_arc_m=10.0,
        )
        allowed = manager.lane_change_start_feasibility(
            authorization_allowed=True,
            distance_to_turn_m=24.0,
            geometry_arc_m=10.0,
            handoff_arc_m=10.0,
        )

        self.assertEqual(denied.action, "deny")
        self.assertIn("required_lane_change_no_longer_feasible", denied.reason)
        self.assertEqual(allowed.action, "allow")

    def test_turn_ownership_atomically_releases_lane_change(self):
        manager = ManeuverManager()
        manager.begin_lane_change(
            "lane_change_right", "target_lane_stabilization",
            340155, 340154, 2.2, [],
        )

        transition = manager.transfer_lateral_ownership_to_turn(
            owner_state="INTERSECTION_TURN"
        )

        self.assertEqual(transition.action, "release")
        self.assertFalse(manager.lane_change.active)
        self.assertEqual(manager.last_release["outcome"], "abandoned")
        self.assertIn("intersection_turn", manager.last_release["reason"])

    def test_lane_follow_does_not_release_active_lane_change(self):
        manager = ManeuverManager()
        manager.begin_lane_change(
            "lane_change_right", "executing", 1, 2, 3.0, []
        )

        transition = manager.transfer_lateral_ownership_to_turn(
            owner_state="LANE_FOLLOW"
        )

        self.assertEqual(transition.action, "hold")
        self.assertTrue(manager.lane_change.active)

    def test_completed_route_edge_cannot_be_recommitted_until_cursor_advances(self):
        manager = ManeuverManager()
        first_edge = "7:lane_change:10-11:right:500144-540156"
        next_edge = "7:lane_change:30-31:right:540156-540155"
        manager.observe_route_lane_change_edge(first_edge)
        manager.begin_lane_change(
            "lane_change_right", "executing", 500144, 540156, 7.0, []
        )

        self.assertTrue(manager.complete_lane_change("contract_satisfied"))
        self.assertTrue(manager.route_lane_change_edge_completed)

        # Re-observing the same topology edge must not make it executable
        # again merely because its direction is also "right".
        manager.observe_route_lane_change_edge(first_edge)
        self.assertTrue(manager.route_lane_change_edge_completed)

        # A later edge has a distinct identity and is therefore eligible.
        manager.observe_route_lane_change_edge(next_edge)
        self.assertFalse(manager.route_lane_change_edge_completed)
        self.assertEqual(manager.lane_change.completed_option, "")


if __name__ == "__main__":
    unittest.main()
