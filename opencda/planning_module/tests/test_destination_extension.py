import math
import unittest

from opencda.scenario_testing.utils.destination_extension import \
    follow_lane_successor


class _Object:
    pass


class _Waypoint:
    def __init__(self, x, y, yaw=0.0, lane_id=1):
        self.lane_id = lane_id
        self.transform = _Object()
        self.transform.location = _Object()
        self.transform.location.x = float(x)
        self.transform.location.y = float(y)
        self.transform.location.z = 0.0
        self.transform.rotation = _Object()
        self.transform.rotation.yaw = float(yaw)
        self.successors = []

    def next(self, _distance_m):
        return list(self.successors)


class DestinationExtensionTests(unittest.TestCase):
    def test_follows_longitudinal_chain_across_road_lane_id_change(self):
        first = _Waypoint(0, 0, lane_id=1)
        second = _Waypoint(2, 0, lane_id=1)
        third = _Waypoint(4, 0, lane_id=2)
        first.successors = [second]
        second.successors = [third]

        result, remaining = follow_lane_successor(first, 4.0, step_m=2.0)

        self.assertIs(result, third)
        self.assertAlmostEqual(remaining, 0.0)

    def test_split_prefers_same_lane_continuation(self):
        first = _Waypoint(0, 0, lane_id=1)
        lane_change_branch = _Waypoint(2, 0, yaw=0.0, lane_id=2)
        same_lane_branch = _Waypoint(
            2 * math.cos(math.radians(5)),
            2 * math.sin(math.radians(5)),
            yaw=5.0,
            lane_id=1,
        )
        first.successors = [lane_change_branch, same_lane_branch]

        result, _ = follow_lane_successor(first, 2.0, step_m=2.0)

        self.assertIs(result, same_lane_branch)

    def test_stops_cleanly_at_lane_end(self):
        first = _Waypoint(0, 0)
        second = _Waypoint(2, 0)
        first.successors = [second]

        result, remaining = follow_lane_successor(first, 6.0, step_m=2.0)

        self.assertIs(result, second)
        self.assertAlmostEqual(remaining, 4.0)


if __name__ == "__main__":
    unittest.main()
