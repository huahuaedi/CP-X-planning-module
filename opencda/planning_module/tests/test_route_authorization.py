import importlib.util
from pathlib import Path
import sys
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "route_authorization.py"
_SPEC = importlib.util.spec_from_file_location("route_authorization", str(_MODULE_PATH))
route_authorization = importlib.util.module_from_spec(_SPEC)
sys.modules["route_authorization"] = route_authorization
_SPEC.loader.exec_module(route_authorization)

authorize_route_lane_change = route_authorization.authorize_route_lane_change
authorize_opportunistic_lane_change = (
    route_authorization.authorize_opportunistic_lane_change
)
suppress_lane_change_for_lateral_owner = route_authorization.suppress_lane_change_for_lateral_owner
lane_change_target_reached = route_authorization.lane_change_target_reached
normalize_route_maneuver = route_authorization.normalize_route_maneuver
RouteManeuver = route_authorization.RouteManeuver
LaneChangeAuthorization = route_authorization.LaneChangeAuthorization
RouteLaneChangeAuthorizationLatch = (
    route_authorization.RouteLaneChangeAuthorizationLatch
)


class RouteAuthorizationTest(unittest.TestCase):
    def test_opportunistic_authorization_does_not_require_route_geometry(self):
        authorization = authorize_opportunistic_lane_change(
            enabled=True,
            start_lock_active=False,
            dense_traffic_lock_active=False,
        )

        self.assertTrue(authorization.allowed)
        self.assertEqual(
            authorization.reason, "opportunistic_lane_change_authorized"
        )

    def test_opportunistic_authorization_respects_start_lock(self):
        authorization = authorize_opportunistic_lane_change(
            enabled=True,
            start_lock_active=True,
            dense_traffic_lock_active=False,
        )

        self.assertFalse(authorization.allowed)
        self.assertEqual(
            authorization.reason, "opportunistic_lane_change_start_lock"
        )

    def test_opportunistic_authorization_respects_dense_traffic_lock(self):
        authorization = authorize_opportunistic_lane_change(
            enabled=True,
            start_lock_active=False,
            dense_traffic_lock_active=True,
        )

        self.assertFalse(authorization.allowed)
        self.assertEqual(
            authorization.reason,
            "opportunistic_lane_change_dense_traffic_lock",
        )

    def test_route_authorization_stays_latched_after_dynamic_gate_lapses(self):
        latch = RouteLaneChangeAuthorizationLatch()
        ready = LaneChangeAuthorization(
            allowed=True,
            direction="right",
            reason="route_lane_change_authorized_by_geometry",
            required_by_route=True,
            distance_to_maneuver_m=46.0,
            target_lane_id=500144,
            maneuver="lane_change_right",
        )
        waiting_again = LaneChangeAuthorization(
            allowed=False,
            direction=None,
            reason="explicit_lane_change_trigger_too_far",
            required_by_route=False,
            distance_to_maneuver_m=43.0,
            target_lane_id=500145,
            maneuver="lane_change_right",
        )

        self.assertTrue(
            latch.update(ready, target_reached=False, in_turn_connector=False).allowed
        )
        stabilized = latch.update(
            waiting_again,
            target_reached=False,
            in_turn_connector=False,
        )

        self.assertTrue(stabilized.allowed)
        self.assertTrue(stabilized.required_by_route)
        self.assertEqual(stabilized.target_lane_id, 500144)
        self.assertEqual(stabilized.reason, "route_lane_change_authorization_latched")
        self.assertEqual(stabilized.distance_to_maneuver_m, 43.0)

    def test_route_authorization_latch_releases_at_target(self):
        latch = RouteLaneChangeAuthorizationLatch()
        ready = LaneChangeAuthorization(
            allowed=True,
            direction="right",
            reason="route_lane_change_authorized_by_geometry",
            required_by_route=True,
            distance_to_maneuver_m=20.0,
            target_lane_id=500144,
            maneuver="lane_change_right",
        )
        latch.update(ready, target_reached=False, in_turn_connector=False)
        released = latch.update(
            ready,
            target_reached=True,
            in_turn_connector=False,
        )

        self.assertIsNone(latch.active)
        self.assertIs(released, ready)

    def test_completion_uses_zero_offset_across_ad_road_segments(self):
        self.assertTrue(lane_change_target_reached(
            current_lane_id=1,
            remembered_target_lane_id=2,
            current_ad_lane_id=11640144,
            remembered_target_ad_lane_id=500144,
            target_in_local_frame=True,
            target_lane_offset=0,
        ))

    def test_nonzero_offset_is_not_completed_when_target_is_visible(self):
        self.assertFalse(lane_change_target_reached(
            current_lane_id=1,
            remembered_target_lane_id=2,
            current_ad_lane_id=11640145,
            remembered_target_ad_lane_id=500144,
            target_in_local_frame=True,
            target_lane_offset=-1,
        ))

    def test_completion_prefers_ad_identity_over_colliding_local_ids(self):
        self.assertTrue(lane_change_target_reached(
            current_lane_id=2,
            remembered_target_lane_id=1,
            current_ad_lane_id=540155,
            remembered_target_ad_lane_id=540155,
        ))

    def test_completion_does_not_mix_ad_and_local_identity_domains(self):
        self.assertFalse(lane_change_target_reached(
            current_lane_id=1,
            remembered_target_lane_id=1,
            current_ad_lane_id=540156,
            remembered_target_ad_lane_id=540155,
        ))

    def test_completion_falls_back_to_local_ids_without_ad_identity(self):
        self.assertTrue(lane_change_target_reached(
            current_lane_id=2,
            remembered_target_lane_id=2,
        ))

    def test_turn_exit_stabilization_has_exclusive_lateral_authority(self):
        authorization = route_authorization.LaneChangeAuthorization(
            allowed=True,
            direction="right",
            reason="route_lane_change_authorized_by_topology",
            required_by_route=True,
            distance_to_maneuver_m=8.0,
            target_lane_id=2,
            maneuver="lane_change_right",
        )
        suppressed = suppress_lane_change_for_lateral_owner(
            authorization,
            owner_state="TURN_EXIT_STABILIZATION",
        )
        self.assertFalse(suppressed.allowed)
        self.assertTrue(suppressed.required_by_route)
        self.assertEqual(suppressed.direction, "right")
        self.assertEqual(suppressed.target_lane_id, 2)
        self.assertEqual(
            suppressed.reason,
            "scenario_lateral_owner:turn_exit_stabilization",
        )

    def test_lane_follow_keeps_route_authorization(self):
        authorization = route_authorization.LaneChangeAuthorization(
            allowed=True,
            direction="right",
            reason="route_lane_change_authorized_by_topology",
            required_by_route=True,
            distance_to_maneuver_m=8.0,
            target_lane_id=2,
            maneuver="lane_change_right",
        )
        self.assertIs(
            suppress_lane_change_for_lateral_owner(
                authorization,
                owner_state="LANE_FOLLOW",
            ),
            authorization,
        )

    def test_ad_topology_nonzero_offset_overrides_colliding_local_lane_ids(self):
        auth = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=1,
            route_required_lane_id=1,
            next_macro_maneuver="Lane Change Right",
            current_road_option="LANEFOLLOW",
            remaining_distance_m=5.0,
            available_lane_ids=[1],
            lane_safety_scores={1: 1.0},
            lane_prediction_risks={1: {}},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=12.0,
            target_safety_threshold=0.65,
            explicit_lane_change_start_distance_m=20.0,
            topology_current_lane_id=540156,
            topology_target_lane_id=540155,
            topology_lane_offset=-1,
        )

        self.assertTrue(auth.allowed)
        self.assertTrue(auth.required_by_route)
        self.assertEqual(auth.direction, "right")
        self.assertEqual(
            auth.reason,
            "route_lane_change_authorized_by_topology",
        )

    def test_opaque_ad_lane_ids_use_explicit_topology_direction(self):
        auth = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=900,
            route_required_lane_id=120,
            next_macro_maneuver="lane_change_left",
            current_road_option="lane_follow",
            remaining_distance_m=10.0,
            available_lane_ids=[900, 120],
            lane_safety_scores={120: 1.0},
            lane_prediction_risks={120: {}},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=5.0,
            target_safety_threshold=0.65,
            require_adjacent=True,
            explicit_lane_change_start_distance_m=20.0,
            adjacent_lane_directions={120: "left"},
        )
        self.assertTrue(auth.allowed)
        self.assertEqual(auth.direction, "left")

    def test_opaque_ad_lane_id_not_in_contact_graph_is_not_adjacent(self):
        auth = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=900,
            route_required_lane_id=901,
            next_macro_maneuver="lane_change_left",
            current_road_option="lane_follow",
            remaining_distance_m=10.0,
            available_lane_ids=[900, 901],
            lane_safety_scores={901: 1.0},
            lane_prediction_risks={901: {}},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=5.0,
            target_safety_threshold=0.65,
            require_adjacent=True,
            explicit_lane_change_start_distance_m=20.0,
            adjacent_lane_directions={120: "left"},
        )
        self.assertFalse(auth.allowed)
        self.assertEqual(auth.reason, "required_lane_not_adjacent")
    def test_continue_straight_never_requires_lane_change(self):
        auth = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=1,
            route_required_lane_id=2,
            next_macro_maneuver="Continue Straight",
            current_road_option="LANEFOLLOW",
            remaining_distance_m=30.0,
            available_lane_ids=[1, 2],
            lane_safety_scores={2: 1.0},
            lane_prediction_risks={},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=12.0,
            target_safety_threshold=0.65,
        )
        self.assertFalse(auth.allowed)
        self.assertFalse(auth.required_by_route)
        self.assertEqual(auth.reason, "route_maneuver_does_not_require_lane_change")

    def test_left_turn_authorizes_left_adjacent_lane_in_window(self):
        auth = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=1,
            route_required_lane_id=2,
            next_macro_maneuver="left",
            current_road_option="LANEFOLLOW",
            remaining_distance_m=30.0,
            available_lane_ids=[1, 2],
            lane_safety_scores={2: 0.95},
            lane_prediction_risks={},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=12.0,
            target_safety_threshold=0.65,
            adjacent_lane_directions={2: "left"},
        )
        self.assertTrue(auth.allowed)
        self.assertTrue(auth.required_by_route)
        self.assertEqual(auth.direction, "left")
        self.assertEqual(auth.target_lane_id, 2)

    def test_maneuver_too_far_denies_lane_change(self):
        auth = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=1,
            route_required_lane_id=2,
            next_macro_maneuver="left",
            current_road_option="LANEFOLLOW",
            remaining_distance_m=80.0,
            available_lane_ids=[1, 2],
            lane_safety_scores={2: 0.95},
            lane_prediction_risks={},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=12.0,
            target_safety_threshold=0.65,
            adjacent_lane_directions={2: "left"},
        )
        self.assertFalse(auth.allowed)
        self.assertEqual(auth.reason, "maneuver_too_far_for_lane_change")

    def test_prediction_risk_denies_target_lane(self):
        auth = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=1,
            route_required_lane_id=2,
            next_macro_maneuver="left",
            current_road_option="LANEFOLLOW",
            remaining_distance_m=30.0,
            available_lane_ids=[1, 2],
            lane_safety_scores={2: 0.95},
            lane_prediction_risks={2: {"risk": True}},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=12.0,
            target_safety_threshold=0.65,
            adjacent_lane_directions={2: "left"},
        )
        self.assertFalse(auth.allowed)
        self.assertEqual(auth.reason, "target_lane_prediction_risk")

    def test_explicit_route_lane_change_uses_grp_trigger_not_destination_distance(self):
        auth = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=2,
            route_required_lane_id=1,
            next_macro_maneuver="Lane Change Right",
            current_road_option="LANEFOLLOW",
            remaining_distance_m=120.0,
            available_lane_ids=[1, 2],
            lane_safety_scores={1: 0.95},
            lane_prediction_risks={},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=12.0,
            target_safety_threshold=0.65,
            adjacent_lane_directions={1: "right"},
        )

        self.assertTrue(auth.allowed)
        self.assertTrue(auth.required_by_route)
        self.assertEqual(auth.direction, "right")
        self.assertEqual(auth.target_lane_id, 1)

    def test_explicit_route_lane_change_waits_until_dynamic_start_distance(self):
        common = dict(
            route_lane_change_allowed=True,
            current_lane_id=2,
            route_required_lane_id=1,
            next_macro_maneuver="Lane Change Right",
            current_road_option="LANEFOLLOW",
            available_lane_ids=[1, 2],
            lane_safety_scores={1: 0.95},
            lane_prediction_risks={},
            preparation_start_distance_m=45.0,
            latest_start_distance_m=12.0,
            target_safety_threshold=0.65,
            adjacent_lane_directions={1: "right"},
            explicit_lane_change_start_distance_m=15.0,
        )

        waiting = authorize_route_lane_change(
            remaining_distance_m=21.0,
            **common,
        )
        ready = authorize_route_lane_change(
            remaining_distance_m=14.0,
            **common,
        )

        self.assertFalse(waiting.allowed)
        self.assertEqual(waiting.reason, "explicit_lane_change_trigger_too_far")
        self.assertTrue(ready.allowed)
        self.assertEqual(ready.direction, "right")

    def test_normalize_straight(self):
        self.assertEqual(normalize_route_maneuver("Continue Straight"), RouteManeuver.GO_STRAIGHT)


if __name__ == "__main__":
    unittest.main()
