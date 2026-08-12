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
normalize_route_maneuver = route_authorization.normalize_route_maneuver
RouteManeuver = route_authorization.RouteManeuver


class RouteAuthorizationTest(unittest.TestCase):
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
