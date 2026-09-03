import unittest
import math

from utility.global_planner import CustomGlobalPlannerAdapter


class _Waypoint:
    def __init__(self, lane_id, *, intersection=False):
        self.ad_lane_id = int(lane_id)
        self.is_intersection = bool(intersection)
        self._left = None
        self._right = None

    def left(self):
        return self._left

    def right(self):
        return self._right


class RouteOptionClassificationTests(unittest.TestCase):
    @staticmethod
    def _turn_points():
        return (
            [[0.0, float(y)] for y in range(6)]
            + [[float(x), 5.0] for x in range(1, 9)]
        )

    def test_intersection_successor_cannot_overwrite_turn_as_lane_change(self):
        points = self._turn_points()
        waypoints = [
            _Waypoint(10 if index < 6 else 20, intersection=3 <= index <= 8)
            for index in range(len(points))
        ]
        # Reproduce the ambiguous AD-map seam from Town06: the first
        # connector waypoint is also returned as a lateral neighbor.
        waypoints[5]._right = waypoints[6]

        options = CustomGlobalPlannerAdapter._lane_aware_road_options(
            points, waypoints
        )

        self.assertIn("LEFT", options)
        self.assertNotIn("CHANGELANELEFT", options)
        self.assertNotIn("CHANGELANERIGHT", options)

    def test_non_intersection_parallel_transition_remains_lane_change(self):
        points = [[float(index), 0.0] for index in range(14)]
        waypoints = [_Waypoint(10) for _ in points]
        waypoints[6] = _Waypoint(20)
        waypoints[5]._right = waypoints[6]

        options = CustomGlobalPlannerAdapter._lane_aware_road_options(
            points, waypoints
        )

        self.assertIn("CHANGELANERIGHT", options)

    def test_distributed_intersection_curvature_uses_complete_connector_arc(self):
        points = [[0.0, float(y)] for y in range(5)]
        # A gradual clockwise quarter circle: every small local window turns
        # only a little, while the connector as a whole turns about 90 deg.
        for step in range(1, 17):
            angle = math.pi - (math.pi / 2.0) * (float(step) / 16.0)
            points.append([
                5.0 + 5.0 * math.cos(angle),
                4.0 + 5.0 * math.sin(angle),
            ])
        points.extend([[float(x), 9.0] for x in range(6, 11)])
        waypoints = [
            _Waypoint(20, intersection=5 <= index <= 20)
            for index in range(len(points))
        ]

        options = CustomGlobalPlannerAdapter._lane_aware_road_options(
            points, waypoints
        )

        connector_options = options[5:21]
        self.assertTrue(connector_options)
        self.assertEqual(set(connector_options), {"LEFT"})


if __name__ == "__main__":
    unittest.main()
