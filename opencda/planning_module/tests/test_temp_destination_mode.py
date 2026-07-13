import math
import unittest
from types import SimpleNamespace

from behavior_planner.temp_destination import (
    build_reference_samples,
    compute_ego_lane_offset,
    compute_temp_destination,
    compute_temp_destination_mode,
)


class _Waypoint:
    def __init__(self, ad_lane_id, lane_id, x, y, *, width=3.5, intersection=False):
        self.ad_lane_id = int(ad_lane_id)
        self.lane_id = int(lane_id)
        self.road_id = 1
        self.section_id = 0
        self.parametric_offset = float(x)
        self.position = {"x": float(x), "y": float(y), "z": 0.0}
        self.heading = 0.0
        self.lane_width_m = float(width)
        self.is_intersection = bool(intersection)
        self._left = None
        self._right = None

    def left(self):
        return self._left

    def right(self):
        return self._right


class _Planner:
    def __init__(self, waypoints, route_points):
        self.waypoints = list(waypoints)
        self.active_route = SimpleNamespace(sampled_waypoints=list(route_points))

    def get_waypoint(self, point):
        return min(
            self.waypoints,
            key=lambda waypoint: math.hypot(
                waypoint.position["x"] - float(point["x"]),
                waypoint.position["y"] - float(point["y"]),
            ),
        )


class TempDestinationModeTests(unittest.TestCase):
    def setUp(self):
        self.right0 = _Waypoint(10, -1, 0.0, 0.0, width=3.6)
        self.right10 = _Waypoint(10, -1, 10.0, 0.0, width=3.6)
        self.right20 = _Waypoint(10, -1, 20.0, 0.0, width=3.6, intersection=True)
        self.left0 = _Waypoint(20, -2, 0.0, 3.6, width=3.6)
        self.left10 = _Waypoint(20, -2, 10.0, 3.6, width=3.6)
        self.left20 = _Waypoint(20, -2, 20.0, 3.6, width=3.6, intersection=True)
        for right, left in (
            (self.right0, self.left0),
            (self.right10, self.left10),
            (self.right20, self.left20),
        ):
            right._left = left
            left._right = right
        self.planner = _Planner(
            [self.right0, self.right10, self.right20, self.left0, self.left10, self.left20],
            [self.right0, self.right10, self.right20],
        )

    def test_mode_uses_custom_route_to_detect_upcoming_intersection(self):
        mode, road_id, entered = compute_temp_destination_mode(
            global_planner=self.planner,
            ego_position=[0.0, 0.0],
            mode_reference_xy=[0.0, 0.0],
            prev_mode=0.0,
            prev_road_id=1,
            prev_entered_intersection=False,
            next_macro_maneuver="straight",
            intersection_threshold_m=20.0,
        )

        self.assertEqual(mode, 1.0)
        self.assertEqual(road_id, 1)
        self.assertFalse(entered)

    def test_blue_dot_rolls_on_active_custom_route(self):
        destination = compute_temp_destination(
            global_planner=self.planner,
            ego_position=[0.0, 0.0],
            target_lane_id=1,
            decision="lane_follow",
            lookahead_m=10.0,
            target_v_mps=6.0,
            global_route_points=[[0.0, 100.0], [20.0, 100.0]],
            follow_global_route_lane=True,
        )

        self.assertAlmostEqual(destination[0], 10.0)
        self.assertAlmostEqual(destination[1], 0.0)

    def test_lane_change_reference_uses_custom_neighbor_and_width(self):
        samples = build_reference_samples(
            global_planner=self.planner,
            ego_position=[0.0, 0.0],
            target_lane_id=2,
            decision="lane_change_left",
            horizon_steps=2,
            step_distance_m=10.0,
            global_route_points=[],
        )

        self.assertEqual(len(samples), 3)
        self.assertTrue(all(sample["lane_id"] == 2 for sample in samples[:2]))
        self.assertTrue(all(abs(sample["y_ref_m"] - 3.6) < 1.0e-6 for sample in samples[:2]))
        self.assertTrue(all(abs(sample["lane_width_m"] - 3.6) < 1.0e-6 for sample in samples))
        self.assertAlmostEqual(samples[0]["road_left_width_m"], 1.8)
        self.assertAlmostEqual(samples[0]["road_right_width_m"], 5.4)

    def test_stop_decision_uses_explicit_stop_state(self):
        destination = compute_temp_destination(
            global_planner=self.planner,
            ego_position=[0.0, 0.0],
            target_lane_id=1,
            decision="traffic_light_stop",
            lookahead_m=10.0,
            target_v_mps=0.0,
            global_route_points=[],
            stop_target_state=[7.0, 0.0, 0.0, 0.0, 1.0],
        )

        self.assertEqual(destination, [7.0, 0.0, 0.0, 0.0, 1.0])

    def test_ego_lane_offset_uses_custom_lane_center(self):
        result = compute_ego_lane_offset(
            global_planner=self.planner,
            ego_position=[0.0, 1.0],
            ego_heading_rad=0.0,
        )

        self.assertAlmostEqual(result["offset_m"], 1.0)
        self.assertEqual(result["lane_id"], 1)
        self.assertAlmostEqual(result["lane_width_m"], 3.6)


if __name__ == "__main__":
    unittest.main()
