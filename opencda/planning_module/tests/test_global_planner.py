import math
import unittest
from types import SimpleNamespace

from behavior_planner.reroute import reroute_from_lane_closure_messages
from behavior_planner.temp_destination import compute_temp_destination
from utility.custom_planner import behavior_lane_id, behavior_lane_waypoints


class _Waypoint:
    def __init__(self, ad_lane_id, lane_id, x, y, *, width=3.6, road_id=1):
        self.ad_lane_id = int(ad_lane_id)
        self.lane_id = int(lane_id)
        self.road_id = int(road_id)
        self.section_id = 0
        self.parametric_offset = float(x)
        self.position = {"x": float(x), "y": float(y), "z": 0.0}
        self.heading = 0.0
        self.lane_width_m = float(width)
        self.is_intersection = False
        self._left = None
        self._right = None

    def left(self):
        return self._left

    def right(self):
        return self._right


def _route(*waypoints):
    return SimpleNamespace(
        sampled_waypoints=list(waypoints),
        resolved_start=waypoints[0],
        resolved_goal=waypoints[-1],
        length_m=abs(float(waypoints[-1].position["x"] - waypoints[0].position["x"])),
        sampling_resolution_m=1.0,
    )


class _Planner:
    def __init__(self, lanes, route):
        self.lanes = list(lanes)
        self.active_route = route
        self.blocked_lanes = []
        self.fail = False

    def get_waypoint(self, position):
        return min(
            self.lanes,
            key=lambda wp: math.hypot(
                float(wp.position["x"]) - float(position["x"]),
                float(wp.position["y"]) - float(position["y"]),
            ),
        )

    def add_blocked_lane(self, ad_lane_id):
        lane_id = int(ad_lane_id)
        if lane_id not in self.blocked_lanes:
            self.blocked_lanes.append(lane_id)

    def trace_route(self, start, goal, activate_route=True):
        if self.fail:
            raise RuntimeError("no route")
        usable = [wp for wp in self.lanes if wp.ad_lane_id not in self.blocked_lanes]
        if not usable:
            raise RuntimeError("no route")
        planned = _route(usable[0], _Waypoint(usable[0].ad_lane_id, usable[0].lane_id, goal["x"], goal["y"]))
        if activate_route:
            self.active_route = planned
        return planned

    def get_lane_context(self, position):
        waypoint = self.get_waypoint(position)
        return {
            "road_id": waypoint.road_id,
            "section_id": waypoint.section_id,
            "lane_id": behavior_lane_id(waypoint),
            "lane_ids": [behavior_lane_id(item) for item in behavior_lane_waypoints(waypoint)],
            "opendrive_lane_id": waypoint.lane_id,
            "lane_width_m": waypoint.lane_width_m,
            "behavior_lane_by_opendrive_lane": {
                item.lane_id: behavior_lane_id(item) for item in behavior_lane_waypoints(waypoint)
            },
        }

    def find_opt_lane(self, position):
        return self.get_waypoint(position).lane_id


class CustomGlobalPlannerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.right = _Waypoint(101, -1, 0.0, 0.0)
        self.left = _Waypoint(202, -2, 0.0, 3.6)
        self.right._left = self.left
        self.left._right = self.right
        self.old_route = _route(self.right, _Waypoint(101, -1, 20.0, 0.0))
        self.planner = _Planner([self.right, self.left], self.old_route)

    def test_waypoint_wrapper_supplies_lane_width_and_neighbors(self):
        self.assertAlmostEqual(self.right.lane_width_m, 3.6)
        self.assertEqual([wp.ad_lane_id for wp in behavior_lane_waypoints(self.left)], [101, 202])
        self.assertEqual(behavior_lane_id(self.left), 2)

    def test_reroute_blocks_custom_ad_lane_id_and_replaces_route(self):
        result = reroute_from_lane_closure_messages(
            messages=[{"id": "closure", "type": "lane_closure", "position": [0.0, 0.0]}],
            global_planner=self.planner,
            ego_position=[0.0, 0.0],
            goal_position=[20.0, 3.6],
            current_route_points=[[0.0, 0.0], [20.0, 0.0]],
        )

        self.assertFalse(result["reroute_failed"])
        self.assertEqual(self.planner.blocked_lanes, [101])
        self.assertEqual(result["handled_message_ids"], ["closure"])
        self.assertIs(self.planner.active_route, result["route"])
        self.assertEqual(result["route"].sampled_waypoints[0].ad_lane_id, 202)

    def test_whole_road_closure_blocks_all_same_direction_ad_lane_ids(self):
        result = reroute_from_lane_closure_messages(
            messages=[{"id": "road", "type": "lane_closure", "position": [0.0, 0.0], "block_entire_road": True}],
            global_planner=self.planner,
            ego_position=[0.0, 0.0],
            goal_position=[20.0, 0.0],
        )

        self.assertTrue(result["reroute_failed"])
        self.assertEqual(self.planner.blocked_lanes, [101, 202])
        self.assertEqual(result["handled_message_ids"], [])

    def test_failed_reroute_preserves_old_route_and_message(self):
        self.planner.fail = True
        result = reroute_from_lane_closure_messages(
            messages=[{"id": "closure", "type": "lane_closure", "position": [0.0, 0.0]}],
            global_planner=self.planner,
            ego_position=[0.0, 0.0],
            goal_position=[20.0, 0.0],
        )

        self.assertTrue(result["reroute_failed"])
        self.assertIs(self.planner.active_route, self.old_route)
        self.assertEqual(result["handled_message_ids"], [])

    def test_blue_dot_uses_replacement_route_samples(self):
        self.planner.active_route = _route(self.left, _Waypoint(202, -2, 20.0, 3.6))
        destination = compute_temp_destination(
            global_planner=self.planner,
            ego_position=[0.0, 3.6],
            target_lane_id=2,
            decision="lane_follow",
            lookahead_m=10.0,
            target_v_mps=5.0,
            global_route_points=[[0.0, 0.0], [20.0, 0.0]],
            follow_global_route_lane=True,
        )

        self.assertAlmostEqual(destination[0], 10.0)
        self.assertAlmostEqual(destination[1], 3.6)


if __name__ == "__main__":
    unittest.main()
