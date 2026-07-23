import unittest
import threading
import math
from types import SimpleNamespace

import numpy as np

from utility.global_planner import AStarGlobalPlanner, RoutePlanSummary, WaypointNode
from utility.carla_lane_graph import (
    canonical_lane_id_for_waypoint,
    canonical_lane_ids_for_waypoint,
    canonical_lane_waypoint_for_lane_id,
)


class _DummyLaneType:
    name = "Driving"


class _DummyLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _DummyRotation:
    def __init__(self, yaw=0.0):
        self.yaw = float(yaw)


class _DummyTransform:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.location = _DummyLocation(x=x, y=y, z=z)
        self.rotation = _DummyRotation(yaw=yaw)


class _DummyWaypoint:
    def __init__(self, road_id, section_id, lane_id, x_m, y_m, yaw_deg=0.0):
        self.road_id = int(road_id)
        self.section_id = int(section_id)
        self.lane_id = int(lane_id)
        self.lane_type = _DummyLaneType()
        self.transform = _DummyTransform(x=x_m, y=y_m, yaw=yaw_deg)
        self._left_lane = None
        self._right_lane = None

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane

    def set_neighbors(self, *, left=None, right=None):
        self._left_lane = left
        self._right_lane = right


class _DummyWorldMap:
    def __init__(self, waypoint):
        self._waypoint = waypoint

    def get_waypoint(self, location, project_to_road=True):
        del location
        del project_to_road
        return self._waypoint


class _DummyStackedWorldMap:
    def __init__(self, lower_waypoint, upper_waypoint, split_z_m=3.0):
        self._lower_waypoint = lower_waypoint
        self._upper_waypoint = upper_waypoint
        self._split_z_m = float(split_z_m)

    def get_waypoint(self, location, project_to_road=True):
        del project_to_road
        if float(getattr(location, "z", 0.0)) >= float(self._split_z_m):
            return self._upper_waypoint
        return self._lower_waypoint


class ActiveRouteLaneTests(unittest.TestCase):
    def test_returns_current_route_lane_before_turn(self):
        route_opt_lane = AStarGlobalPlanner._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=[2, 2, 1, 1],
            fallback_lane_id=1,
        )

        self.assertEqual(int(route_opt_lane), 2)

    def test_updates_when_remaining_route_lane_changes(self):
        route_opt_lane = AStarGlobalPlanner._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=[1, 1, 1],
            fallback_lane_id=2,
        )

        self.assertEqual(int(route_opt_lane), 1)

    def test_skips_invalid_entries_and_uses_first_valid_remaining_lane(self):
        route_opt_lane = AStarGlobalPlanner._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=[0, 0, 2, 2],
            fallback_lane_id=1,
        )

        self.assertEqual(int(route_opt_lane), 2)

    def test_negative_carla_lane_ids_are_treated_as_valid(self):
        route_opt_lane = AStarGlobalPlanner._active_lane_id_from_remaining_sequence(
            remaining_lane_ids=[-1, -1, -2],
            fallback_lane_id=-2,
        )

        self.assertEqual(int(route_opt_lane), -1)


class BlueDotRouteProgressTests(unittest.TestCase):
    def _make_planner(self) -> AStarGlobalPlanner:
        planner = object.__new__(AStarGlobalPlanner)
        planner._route_sample_distance_m = 2.0
        planner._stored_route_xy = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 4.0],
                [0.0, 8.0],
                [0.0, 10.0],
                [2.0, 10.0],
                [4.0, 10.0],
                [6.0, 10.0],
                [8.0, 10.0],
            ],
            dtype=float,
        )
        planner._stored_route_cum_dists = AStarGlobalPlanner._route_cumulative_distances(
            planner._stored_route_xy,
        )
        planner._stored_route_options = [
            "LANEFOLLOW",
            "LANEFOLLOW",
            "LANEFOLLOW",
            "LEFT",
            "LEFT",
            "LEFT",
            "LANEFOLLOW",
            "LANEFOLLOW",
        ]
        planner._stored_route_lane_ids = [2, 2, 2, 2, 1, 1, 1, 1]
        planner._stored_route_summary = RoutePlanSummary(
            route_found=True,
            start_road_id="start",
            start_lane_id=2,
            goal_road_id="goal",
            goal_lane_id=1,
            optimal_lane_id=2,
            distance_to_destination_m=18.0,
            next_macro_maneuver="Left Turn",
            route_waypoints=planner._stored_route_xy.tolist(),
            road_options=["LANEFOLLOW", "LEFT"],
        )
        planner._route_info_query_state = {}
        return planner

    def test_blue_dot_lookup_stays_on_current_lane_until_route_progress_reaches_change(self):
        planner = self._make_planner()

        first_summary = planner.get_current_route_info(
            x_m=0.1,
            y_m=8.6,
            query_key="blue_dot",
        )
        second_summary = planner.get_current_route_info(
            x_m=1.7,
            y_m=9.4,
            query_key="blue_dot",
        )

        self.assertEqual(int(first_summary.optimal_lane_id), 2)
        self.assertEqual(int(second_summary.optimal_lane_id), 2)

    def test_blue_dot_lookup_switches_to_active_route_lane_through_intersection(self):
        planner = self._make_planner()

        planner.get_current_route_info(
            x_m=0.1,
            y_m=8.6,
            query_key="blue_dot",
        )
        planner.get_current_route_info(
            x_m=1.7,
            y_m=9.4,
            query_key="blue_dot",
        )
        planner.get_current_route_info(
            x_m=2.4,
            y_m=10.0,
            query_key="blue_dot",
        )
        final_summary = planner.get_current_route_info(
            x_m=3.2,
            y_m=10.0,
            query_key="blue_dot",
        )

        self.assertEqual(int(final_summary.optimal_lane_id), 1)
        self.assertEqual(str(final_summary.current_road_option).upper(), "LEFT")
        self.assertEqual(str(final_summary.next_macro_maneuver), "Continue Straight")

    def test_blue_dot_lookup_updates_after_turn_block_is_passed(self):
        planner = self._make_planner()

        planner.get_current_route_info(
            x_m=0.1,
            y_m=8.6,
            query_key="blue_dot",
        )
        planner.get_current_route_info(
            x_m=1.7,
            y_m=9.4,
            query_key="blue_dot",
        )
        planner.get_current_route_info(
            x_m=3.2,
            y_m=10.0,
            query_key="blue_dot",
        )
        final_summary = planner.get_current_route_info(
            x_m=7.6,
            y_m=10.0,
            query_key="blue_dot",
        )

        self.assertEqual(int(final_summary.optimal_lane_id), 1)
        self.assertEqual(str(final_summary.current_road_option).upper(), "LANEFOLLOW")


class StaticObstacleAvoidanceRouteTests(unittest.TestCase):
    def _make_planner(self) -> AStarGlobalPlanner:
        planner = object.__new__(AStarGlobalPlanner)
        planner._nodes = [
            WaypointNode(0, 0.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, 1, (1, 3), None),
            WaypointNode(1, 10.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 10.0, "straight", False, 2, (2, 4), None),
            WaypointNode(2, 20.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 20.0, "straight", False, None, (), None),
            WaypointNode(3, 0.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, 4, (4,), None),
            WaypointNode(4, 10.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 10.0, "straight", False, 5, (5,), None),
            WaypointNode(5, 20.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 20.0, "straight", False, None, (2,), None),
        ]
        planner._node_x_m = np.asarray([node.x_m for node in planner._nodes], dtype=float)
        planner._node_y_m = np.asarray([node.y_m for node in planner._nodes], dtype=float)
        planner._adjacency = {
            0: [(1, 10.0), (3, 3.5)],
            1: [(2, 10.0), (4, 3.5)],
            2: [],
            3: [(4, 10.0)],
            4: [(5, 10.0)],
            5: [(2, 3.5)],
        }
        planner._stored_route_xy = None
        planner._stored_route_cum_dists = None
        planner._stored_route_options = None
        planner._stored_route_lane_ids = None
        planner._stored_route_summary = None
        planner._route_info_query_state = {}
        planner._last_trace_per_waypoint_options = []
        planner._last_trace_per_waypoint_lane_ids = []
        return planner

    def test_internal_astar_can_avoid_blocked_point_on_lane(self):
        planner = self._make_planner()

        summary = planner.plan_route_astar_avoiding_points(
            start_xy=[0.0, 0.0],
            goal_xy=[20.0, 0.0],
            blocked_points_xy=[[10.0, 0.0]],
            blocked_lane_ids=[1],
            block_radius_m=1.0,
        )

        self.assertTrue(bool(summary.route_found))
        self.assertTrue(any(abs(float(point[1]) - 3.5) <= 1e-6 for point in summary.route_waypoints))
        self.assertFalse(
            any(
                abs(float(point[0]) - 10.0) <= 1e-6 and abs(float(point[1])) <= 1e-6
                for point in summary.route_waypoints
            )
        )

    def test_plan_route_astar_finds_direct_route_with_no_obstacles(self):
        planner = self._make_planner()

        summary = planner.plan_route_astar(
            start_xy=[0.0, 0.0],
            goal_xy=[20.0, 0.0],
        )

        self.assertTrue(bool(summary.route_found))
        self.assertTrue(all(abs(float(point[1])) <= 1e-6 for point in summary.route_waypoints))
        self.assertAlmostEqual(float(summary.route_waypoints[0][0]), 0.0, places=6)
        self.assertAlmostEqual(float(summary.route_waypoints[-1][0]), 20.0, places=6)

    def test_internal_astar_with_segment_penalty_prefers_unblocked_lane(self):
        planner = self._make_planner()

        summary = planner.plan_route_astar_with_segment_penalties(
            start_xy=[0.0, 0.0],
            goal_xy=[20.0, 0.0],
            segment_penalties={("1:0", 1): 1.0e6},
        )

        self.assertTrue(bool(summary.route_found))
        self.assertTrue(any(abs(float(point[1]) - 3.5) <= 1e-6 for point in summary.route_waypoints))

    def test_internal_astar_with_blocked_segment_uses_adjacent_lane(self):
        planner = self._make_planner()

        summary = planner.plan_route_astar_blocking_segments(
            start_xy=[0.0, 0.0],
            goal_xy=[20.0, 0.0],
            blocked_segments=[("1:0", 1)],
        )

        self.assertTrue(bool(summary.route_found))
        self.assertTrue(any(abs(float(point[1]) - 3.5) <= 1e-6 for point in summary.route_waypoints))
        self.assertFalse(any(abs(float(point[1])) <= 1e-6 and abs(float(point[0]) - 10.0) <= 1e-6 for point in summary.route_waypoints))

    def test_route_penalty_from_waypoints_detects_blocked_lane_usage(self):
        planner = self._make_planner()

        blocked_lane_penalty = planner.route_penalty_from_waypoints(
            [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
            {("1:0", 1): 1.0e6},
        )
        adjacent_lane_penalty = planner.route_penalty_from_waypoints(
            [[0.0, 3.5], [10.0, 3.5], [20.0, 3.5]],
            {("1:0", 1): 1.0e6},
        )

        self.assertGreater(float(blocked_lane_penalty), 0.0)
        self.assertEqual(float(adjacent_lane_penalty), 0.0)

    def test_internal_astar_can_use_nearby_same_direction_goal_when_nearest_goal_is_opposite_direction(self):
        planner = self._make_planner()
        planner._nodes.append(
            WaypointNode(6, 20.0, -0.5, 1, 3.5, "2:0", "negative", math.pi, 0.0, "straight", False, None, (), None)
        )
        planner._node_x_m = np.asarray([node.x_m for node in planner._nodes], dtype=float)
        planner._node_y_m = np.asarray([node.y_m for node in planner._nodes], dtype=float)
        planner._adjacency[6] = []

        summary = planner.plan_route_astar_with_segment_penalties(
            start_xy=[0.0, 0.0],
            goal_xy=[20.0, -0.5],
            segment_penalties={},
        )

        self.assertTrue(bool(summary.route_found))
        self.assertEqual(int(summary.goal_graph_index), 2)

    def test_internal_astar_can_use_adjacent_start_lane_when_nearest_start_lane_is_disconnected(self):
        planner = object.__new__(AStarGlobalPlanner)
        planner._nodes = [
            WaypointNode(0, 0.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, 1, (1,), None),
            WaypointNode(1, 4.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 4.0, "straight", False, None, (), None),
            WaypointNode(2, 0.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, 3, (3,), None),
            WaypointNode(3, 4.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 4.0, "straight", False, 4, (4,), None),
            WaypointNode(4, 8.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 8.0, "straight", False, None, (), None),
        ]
        planner._node_x_m = np.asarray([node.x_m for node in planner._nodes], dtype=float)
        planner._node_y_m = np.asarray([node.y_m for node in planner._nodes], dtype=float)
        planner._adjacency = {
            0: [(1, 4.0)],
            1: [],
            2: [(3, 4.0)],
            3: [(4, 4.0)],
            4: [],
        }
        planner._stored_route_xy = None
        planner._stored_route_cum_dists = None
        planner._stored_route_options = None
        planner._stored_route_lane_ids = None
        planner._stored_route_summary = None
        planner._route_info_query_state = {}
        planner._last_trace_per_waypoint_options = []
        planner._last_trace_per_waypoint_lane_ids = []

        summary = planner.plan_route_astar_with_segment_penalties(
            start_xy=[0.1, 0.1],
            goal_xy=[8.0, 3.5],
            segment_penalties={},
        )

        self.assertTrue(bool(summary.route_found))
        self.assertEqual(int(summary.start_graph_index), 2)


class IntersectionLaneChangeAdjacencyTests(unittest.TestCase):
    """`_build_adjacency` must never create a lane-change edge touching a
    junction node, even though forward/successor edges must still cross it."""

    @staticmethod
    def _new_planner(nodes) -> AStarGlobalPlanner:
        planner = object.__new__(AStarGlobalPlanner)
        planner._nodes = nodes
        planner._lane_change_penalty_m = 5.0
        planner._lane_change_progress_tolerance_m = 5.0
        planner._lane_change_distance_factor = 1.8
        planner._max_heading_diff_rad = 0.8
        return planner

    def _lane_change_targets(self, adjacency, from_index, to_indices):
        neighbor_indices = {int(edge_target) for edge_target, _cost in adjacency[from_index]}
        return neighbor_indices.intersection({int(i) for i in to_indices})

    def test_no_lane_change_edge_between_intersection_nodes(self):
        # Two lanes, three progress steps; the middle step sits inside a junction.
        nodes = [
            WaypointNode(0, 0.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, 1, (1,), None),
            WaypointNode(1, 10.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 10.0, "straight", True, 2, (2,), None),
            WaypointNode(2, 20.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 20.0, "straight", False, None, (), None),
            WaypointNode(3, 0.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, 4, (4,), None),
            WaypointNode(4, 10.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 10.0, "straight", True, 5, (5,), None),
            WaypointNode(5, 20.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 20.0, "straight", False, None, (), None),
        ]
        planner = self._new_planner(nodes)

        adjacency = planner._build_adjacency()

        # No lane-change edge between the two junction nodes, in either direction.
        self.assertEqual(self._lane_change_targets(adjacency, 1, [4]), set())
        self.assertEqual(self._lane_change_targets(adjacency, 4, [1]), set())
        # Forward/successor edges through the junction node must still exist.
        self.assertIn(2, {int(edge_target) for edge_target, _cost in adjacency[1]})
        self.assertIn(5, {int(edge_target) for edge_target, _cost in adjacency[4]})

    def test_lane_change_edge_still_created_outside_intersection(self):
        nodes = [
            WaypointNode(0, 0.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, 1, (1,), None),
            WaypointNode(1, 10.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 10.0, "straight", False, None, (), None),
            WaypointNode(2, 0.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, 3, (3,), None),
            WaypointNode(3, 10.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 10.0, "straight", False, None, (), None),
        ]
        planner = self._new_planner(nodes)

        adjacency = planner._build_adjacency()

        self.assertEqual(self._lane_change_targets(adjacency, 0, [2]), {2})
        self.assertEqual(self._lane_change_targets(adjacency, 1, [3]), {3})

    def test_no_lane_change_edge_into_intersection_node_from_outside(self):
        # `node` itself is not in the junction, but the only same-progress
        # neighbor candidate is -- must not jump laterally into it.
        nodes = [
            WaypointNode(0, 0.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, None, (), None),
            WaypointNode(1, 0.0, 3.5, 2, 3.5, "1:0", "positive", 0.0, 0.0, "straight", True, None, (), None),
        ]
        planner = self._new_planner(nodes)

        adjacency = planner._build_adjacency()

        self.assertEqual(self._lane_change_targets(adjacency, 0, [1]), set())
        self.assertEqual(self._lane_change_targets(adjacency, 1, [0]), set())

    def test_no_lane_change_edge_across_opposing_directions(self):
        nodes = [
            WaypointNode(0, 0.0, 0.0, 1, 3.5, "1:0", "positive", 0.0, 0.0, "straight", False, None, (), None),
            WaypointNode(1, 0.0, -3.5, -1, 3.5, "1:0", "negative", math.pi, 0.0, "straight", False, None, (), None),
        ]
        planner = self._new_planner(nodes)

        adjacency = planner._build_adjacency()

        self.assertEqual(adjacency[0], [])
        self.assertEqual(adjacency[1], [])


class SuccessorLaneSnappingTests(unittest.TestCase):
    """`_normalize_waypoints` must recover a successor link even when
    waypoint.next(d) lands between two sampled grid points on the target
    lane -- the common case at junction connectors, whose lengths rarely
    divide evenly by the sample distance."""

    @staticmethod
    def _waypoint(
        carla_waypoint_key,
        position,
        road_id,
        *,
        is_intersection=False,
        next_position=None,
        next_key=None,
    ):
        entry = {
            "carla_waypoint": None,
            "carla_waypoint_key": carla_waypoint_key,
            "position": list(position),
            "heading_rad": 0.0,
            "lane_id": 1,
            "carla_lane_id": 1,
            "road_id": road_id,
            "direction": "positive",
            "lane_width_m": 3.5,
            "is_intersection": bool(is_intersection),
            "maneuver": "straight",
        }
        if next_position is not None:
            entry["next"] = list(next_position)
        if next_key is not None:
            entry["next_key"] = next_key
        return entry

    def test_next_key_snaps_to_nearest_sampled_point_in_target_lane(self):
        lane_center_waypoints = [
            self._waypoint(
                (1, 0, 1, 0.0), [0.0, 0.0], "1:0",
                next_position=[10.76, 0.0], next_key=(2, 0, 1, 0.76),
            ),
            self._waypoint((2, 0, 1, 0.0), [10.0, 0.0], "2:0", is_intersection=True),
            self._waypoint((2, 0, 1, 3.0), [13.0, 0.0], "2:0", is_intersection=True),
        ]

        planner = AStarGlobalPlanner(
            lane_center_waypoints=lane_center_waypoints,
            world_map=None,
            route_sample_distance_m=3.0,
        )

        node0 = planner._nodes[0]
        self.assertIsNotNone(node0.next_index)
        self.assertEqual(int(node0.next_index), 1)

    def test_next_key_does_not_snap_beyond_tolerance(self):
        lane_center_waypoints = [
            self._waypoint(
                (1, 0, 1, 0.0), [0.0, 0.0], "1:0",
                next_position=[50.0, 0.0], next_key=(2, 0, 1, 50.0),
            ),
            self._waypoint((2, 0, 1, 0.0), [10.0, 0.0], "2:0", is_intersection=True),
        ]

        planner = AStarGlobalPlanner(
            lane_center_waypoints=lane_center_waypoints,
            world_map=None,
            route_sample_distance_m=3.0,
        )

        node0 = planner._nodes[0]
        self.assertIsNone(node0.next_index)


class LaneContextConsistencyTests(unittest.TestCase):
    def test_canonical_lane_helpers_use_rightmost_as_lane_one(self):
        right_wp = _DummyWaypoint(road_id=1, section_id=0, lane_id=1, x_m=0.0, y_m=0.0)
        left_wp = _DummyWaypoint(road_id=1, section_id=0, lane_id=2, x_m=3.5, y_m=0.0)
        right_wp.set_neighbors(left=left_wp)
        left_wp.set_neighbors(right=right_wp)

        self.assertEqual(int(canonical_lane_id_for_waypoint(right_wp)), 1)
        self.assertEqual(int(canonical_lane_id_for_waypoint(left_wp)), 2)
        self.assertEqual(list(canonical_lane_ids_for_waypoint(right_wp)), [1, 2])
        self.assertIs(canonical_lane_waypoint_for_lane_id(left_wp, 1), right_wp)
        self.assertIs(canonical_lane_waypoint_for_lane_id(right_wp, 2), left_wp)

    def test_canonical_lane_helpers_preserve_negative_carla_lane_ids(self):
        right_wp = _DummyWaypoint(road_id=1, section_id=0, lane_id=-2, x_m=0.0, y_m=0.0)
        left_wp = _DummyWaypoint(road_id=1, section_id=0, lane_id=-1, x_m=3.5, y_m=0.0)
        right_wp.set_neighbors(left=left_wp)
        left_wp.set_neighbors(right=right_wp)

        self.assertEqual(int(canonical_lane_id_for_waypoint(right_wp)), 1)
        self.assertEqual(int(canonical_lane_id_for_waypoint(left_wp)), 2)
        self.assertEqual(list(canonical_lane_ids_for_waypoint(right_wp)), [1, 2])
        self.assertIs(canonical_lane_waypoint_for_lane_id(left_wp, 1), right_wp)
        self.assertIs(canonical_lane_waypoint_for_lane_id(right_wp, 2), left_wp)

    def test_local_lane_context_uses_heading_to_pick_same_direction_lane(self):
        planner = object.__new__(AStarGlobalPlanner)
        planner._world_map = None
        planner._nodes = [
            WaypointNode(
                index=0,
                x_m=0.5,
                y_m=0.0,
                lane_id=1,
                lane_width_m=3.5,
                road_id="1:0",
                direction="positive",
                heading_rad=0.0,
                progress_m=0.5,
                maneuver="straight",
                is_intersection=False,
                next_index=None,
                successor_indices=(),
                carla_waypoint=None,
            ),
            WaypointNode(
                index=1,
                x_m=3.8,
                y_m=0.0,
                lane_id=2,
                lane_width_m=3.5,
                road_id="1:0",
                direction="positive",
                heading_rad=0.0,
                progress_m=0.5,
                maneuver="straight",
                is_intersection=False,
                next_index=None,
                successor_indices=(),
                carla_waypoint=None,
            ),
            WaypointNode(
                index=2,
                x_m=0.1,
                y_m=0.0,
                lane_id=1,
                lane_width_m=3.5,
                road_id="1:0",
                direction="negative",
                heading_rad=np.pi,
                progress_m=0.1,
                maneuver="straight",
                is_intersection=False,
                next_index=None,
                successor_indices=(),
                carla_waypoint=None,
            ),
        ]
        planner._node_x_m = np.asarray([node.x_m for node in planner._nodes], dtype=float)
        planner._node_y_m = np.asarray([node.y_m for node in planner._nodes], dtype=float)
        planner._lane_change_progress_tolerance_m = 5.0
        planner._lane_context_lock = threading.Lock()
        planner._lane_context_cache_state = None
        planner._lane_context_cache_result = None
        planner._lane_context_cache_threshold_m = 1.0
        planner._lane_context_cache_heading_threshold_rad = np.pi / 9.0

        context = planner.get_local_lane_context(
            x_m=0.0,
            y_m=0.0,
            heading_rad=0.0,
        )

        self.assertEqual(int(context["lane_id"]), 1)
        self.assertEqual(list(context["lane_ids"]), [1, 2])
        self.assertEqual(int(context["lane_count"]), 2)

    def test_local_lane_context_uses_only_locally_available_same_direction_lanes(self):
        planner = object.__new__(AStarGlobalPlanner)
        planner._world_map = None
        planner._nodes = [
            WaypointNode(
                index=0,
                x_m=0.0,
                y_m=0.0,
                lane_id=1,
                lane_width_m=3.5,
                road_id="1:0",
                direction="positive",
                heading_rad=0.0,
                progress_m=0.0,
                maneuver="straight",
                is_intersection=False,
                next_index=None,
                successor_indices=(),
                carla_waypoint=None,
            ),
            WaypointNode(
                index=1,
                x_m=3.7,
                y_m=0.0,
                lane_id=2,
                lane_width_m=3.5,
                road_id="1:0",
                direction="positive",
                heading_rad=0.0,
                progress_m=0.0,
                maneuver="straight",
                is_intersection=False,
                next_index=None,
                successor_indices=(),
                carla_waypoint=None,
            ),
            WaypointNode(
                index=2,
                x_m=7.4,
                y_m=40.0,
                lane_id=3,
                lane_width_m=3.5,
                road_id="1:0",
                direction="positive",
                heading_rad=0.0,
                progress_m=40.0,
                maneuver="straight",
                is_intersection=False,
                next_index=None,
                successor_indices=(),
                carla_waypoint=None,
            ),
        ]
        planner._node_x_m = np.asarray([node.x_m for node in planner._nodes], dtype=float)
        planner._node_y_m = np.asarray([node.y_m for node in planner._nodes], dtype=float)
        planner._lane_change_progress_tolerance_m = 5.0
        planner._lane_context_lock = threading.Lock()
        planner._lane_context_cache_state = None
        planner._lane_context_cache_result = None
        planner._lane_context_cache_threshold_m = 1.0
        planner._lane_context_cache_heading_threshold_rad = np.pi / 9.0

        context = planner.get_local_lane_context(
            x_m=0.0,
            y_m=0.0,
            heading_rad=0.0,
        )

        self.assertEqual(list(context["lane_ids"]), [1, 2])
        self.assertEqual(int(context["lane_count"]), 2)

    def test_local_lane_context_prefers_runtime_projected_waypoint_lane_id(self):
        right_wp = _DummyWaypoint(road_id=7, section_id=0, lane_id=1, x_m=0.0, y_m=0.0)
        left_wp = _DummyWaypoint(road_id=7, section_id=0, lane_id=2, x_m=3.5, y_m=0.0)
        right_wp.set_neighbors(left=left_wp)
        left_wp.set_neighbors(right=right_wp)

        planner = object.__new__(AStarGlobalPlanner)
        planner._world_map = _DummyWorldMap(right_wp)
        planner._nodes = [
            WaypointNode(
                index=0,
                x_m=0.0,
                y_m=0.0,
                lane_id=2,
                lane_width_m=3.5,
                road_id="7:0",
                direction="positive",
                heading_rad=0.0,
                progress_m=0.0,
                maneuver="straight",
                is_intersection=False,
                next_index=None,
                successor_indices=(),
                carla_waypoint=left_wp,
            ),
        ]
        planner._node_x_m = np.asarray([0.0], dtype=float)
        planner._node_y_m = np.asarray([0.0], dtype=float)
        planner._lane_change_progress_tolerance_m = 5.0
        planner._lane_context_lock = threading.Lock()
        planner._lane_context_cache_state = None
        planner._lane_context_cache_result = None
        planner._lane_context_cache_threshold_m = 1.0
        planner._lane_context_cache_heading_threshold_rad = np.pi / 9.0

        context = planner.get_local_lane_context(
            x_m=0.0,
            y_m=0.0,
            heading_rad=0.0,
        )

        self.assertEqual(int(context["lane_id"]), 1)
        self.assertEqual(list(context["lane_ids"]), [1, 2])

    def test_local_lane_context_uses_query_height_for_stacked_roads(self):
        lower_wp = _DummyWaypoint(road_id=8, section_id=0, lane_id=1, x_m=0.0, y_m=0.0)
        upper_wp = _DummyWaypoint(road_id=9, section_id=0, lane_id=2, x_m=0.0, y_m=0.0)

        planner = object.__new__(AStarGlobalPlanner)
        planner._world_map = _DummyStackedWorldMap(lower_wp, upper_wp, split_z_m=3.0)
        planner._nodes = [
            WaypointNode(
                index=0,
                x_m=0.0,
                y_m=0.0,
                lane_id=1,
                lane_width_m=3.5,
                road_id="8:0",
                direction="positive",
                heading_rad=0.0,
                progress_m=0.0,
                maneuver="straight",
                is_intersection=False,
                next_index=None,
                successor_indices=(),
                carla_waypoint=lower_wp,
            ),
        ]
        planner._node_x_m = np.asarray([0.0], dtype=float)
        planner._node_y_m = np.asarray([0.0], dtype=float)
        planner._lane_change_progress_tolerance_m = 5.0
        planner._lane_context_lock = threading.Lock()
        planner._lane_context_cache_state = None
        planner._lane_context_cache_result = None
        planner._lane_context_cache_threshold_m = 1.0
        planner._lane_context_cache_heading_threshold_rad = np.pi / 9.0
        planner._lane_context_cache_z_threshold_m = 2.5

        lower_context = planner.get_local_lane_context(
            x_m=0.0,
            y_m=0.0,
            heading_rad=0.0,
            z_m=0.0,
        )
        upper_context = planner.get_local_lane_context(
            x_m=0.0,
            y_m=0.0,
            heading_rad=0.0,
            z_m=6.0,
        )

        self.assertEqual(int(lower_context["lane_id"]), 1)
        self.assertEqual(str(lower_context["road_id"]), "8:0")
        self.assertEqual(int(upper_context["lane_id"]), 1)
        self.assertEqual(str(upper_context["road_id"]), "9:0")


class CarlaBlockedLaneRerouteTests(unittest.TestCase):
    def test_plan_route_carla_with_blocked_lanes_falls_back_when_section_id_does_not_match(self):
        class _DummyGraph(dict):
            def has_edge(self, edge_start, edge_end):
                return edge_start in self and edge_end in self[edge_start]

        planner = object.__new__(AStarGlobalPlanner)
        planner._carla_route_planner = SimpleNamespace(
            _graph=_DummyGraph({
                0: {
                    1: {
                        "length": 42.0,
                        "weight": 42.0,
                    }
                }
            }),
            _road_id_to_edge={
                12: {
                    1: {
                        -2: (0, 1),
                    }
                }
            },
        )

        def _plan_route_from_locations(**kwargs):
            del kwargs
            return RoutePlanSummary(
                route_found=True,
                start_road_id="1:0",
                start_lane_id=1,
                goal_road_id="2:0",
                goal_lane_id=1,
                optimal_lane_id=1,
                distance_to_destination_m=10.0,
                next_macro_maneuver="Continue Straight",
                route_waypoints=[[0.0, 0.0], [10.0, 0.0]],
            )

        planner.plan_route_from_locations = _plan_route_from_locations

        summary = planner.plan_route_carla_with_blocked_lanes(
            start_location=_DummyLocation(0.0, 0.0, 0.0),
            goal_location=_DummyLocation(10.0, 0.0, 0.0),
            blocked_raw_lanes=[(12, 0, -2)],
        )

        self.assertTrue(bool(summary.route_found))
        self.assertTrue(math.isinf(planner._carla_route_planner._graph[0][1]["length"]))
        self.assertTrue(math.isinf(planner._carla_route_planner._graph[0][1]["weight"]))

    def test_penalized_carla_reroute_trusts_trace_penalty_not_xy_resnap_penalty(self):
        planner = object.__new__(AStarGlobalPlanner)
        planner._carla_route_planner = SimpleNamespace(
            trace_route=lambda start_location, goal_location: [("trace", "LANEFOLLOW")],
        )
        planner._location_xy = lambda location: (
            float(getattr(location, "x", 0.0)),
            float(getattr(location, "y", 0.0)),
        )
        planner._candidate_start_carla_waypoints = lambda **kwargs: [kwargs.get("start_waypoint", None)]
        planner._build_route_summary_from_carla_trace = lambda **kwargs: RoutePlanSummary(
            route_found=True,
            start_road_id="1:0",
            start_lane_id=1,
            goal_road_id="2:0",
            goal_lane_id=1,
            optimal_lane_id=1,
            distance_to_destination_m=10.0,
            next_macro_maneuver="Continue Straight",
            route_waypoints=[[0.0, 0.0], [10.0, 3.5], [20.0, 3.5]],
        )
        planner._carla_trace_segment_penalty = lambda **kwargs: 0.0
        planner.route_penalty_from_waypoints = lambda *args, **kwargs: 1.0e9
        planner._finalize_route_plan = lambda summary, replace_stored_route=False: summary

        start_waypoint = SimpleNamespace(
            transform=SimpleNamespace(location=_DummyLocation(0.0, 0.0, 0.0)),
        )
        summary = planner.plan_route_from_locations_with_segment_penalties(
            start_location=_DummyLocation(0.0, 0.0, 0.0),
            goal_location=_DummyLocation(20.0, 0.0, 0.0),
            segment_penalties={("12:0", 1): 1.0e9},
            start_waypoint=start_waypoint,
        )

        self.assertTrue(bool(summary.route_found))
        self.assertEqual(
            summary.route_waypoints,
            [[0.0, 0.0], [10.0, 3.5], [20.0, 3.5]],
        )

    def test_waypoint_blocked_carla_reroute_temporarily_marks_only_localized_edge_inf(self):
        class _DummyGraph(dict):
            def has_edge(self, edge_start, edge_end):
                return edge_start in self and edge_end in self[edge_start]

            def copy(self):
                cloned_graph = _DummyGraph()
                for edge_start, edge_targets in self.items():
                    cloned_graph[edge_start] = {
                        edge_end: dict(edge_attrs)
                        for edge_end, edge_attrs in dict(edge_targets).items()
                    }
                return cloned_graph

        planner = object.__new__(AStarGlobalPlanner)
        original_graph = _DummyGraph(
            {
                0: {
                    1: {"length": 14.0, "weight": 14.0},
                    2: {"length": 6.0, "weight": 6.0},
                }
            }
        )
        planner._carla_route_planner = SimpleNamespace(
            _graph=original_graph,
            _road_id_to_edge={
                20: {
                    0: {
                        -2: (0, 1),
                    }
                }
            },
        )

        observed_blocked_length = []

        def _plan_route_from_locations(**kwargs):
            del kwargs
            observed_blocked_length.append(planner._carla_route_planner._graph[0][1]["length"])
            return RoutePlanSummary(
                route_found=True,
                start_road_id="20:0",
                start_lane_id=1,
                goal_road_id="21:0",
                goal_lane_id=1,
                optimal_lane_id=1,
                distance_to_destination_m=12.0,
                next_macro_maneuver="Continue Straight",
                route_waypoints=[[0.0, 0.0], [3.5, 3.5], [10.0, 0.0]],
            )

        planner.plan_route_from_locations = _plan_route_from_locations

        summary = planner.plan_route_from_locations_with_blocked_carla_waypoints(
            start_location=_DummyLocation(0.0, 0.0, 0.0),
            goal_location=_DummyLocation(10.0, 0.0, 0.0),
            blocked_waypoints=[_DummyWaypoint(20, 0, -2, 5.0, 0.0)],
        )

        self.assertTrue(bool(summary.route_found))
        self.assertEqual(len(observed_blocked_length), 1)
        self.assertTrue(math.isinf(observed_blocked_length[0]))
        self.assertEqual(original_graph[0][1]["length"], 14.0)
        self.assertEqual(original_graph[0][2]["length"], 6.0)


if __name__ == "__main__":
    unittest.main()
