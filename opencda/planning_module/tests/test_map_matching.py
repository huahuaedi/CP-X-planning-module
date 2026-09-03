import math
import sys
import types
import unittest

import numpy as np


if "carla" not in sys.modules:
    fake_carla = types.ModuleType("carla")

    class _Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class _VehicleControl:
        def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
            self.throttle = throttle
            self.brake = brake
            self.steer = steer

    fake_carla.Location = _Location
    fake_carla.VehicleControl = _VehicleControl
    sys.modules["carla"] = fake_carla

from pipeline.map_matching import (
    DiagnosticHDMapMatcher,
    LaneProjectionCandidate,
    local_lane_frame_invariants,
    topology_relation,
)
from pipeline.route_authorization import authorize_route_lane_change
from utility.global_planner import CustomGlobalPlannerAdapter


def candidate(
    lane_id,
    *,
    y=0.0,
    heading=0.0,
    relation="unknown",
    inside=True,
):
    return LaneProjectionCandidate(
        ad_lane_id=lane_id,
        road_id=1,
        section_id=0,
        raw_lane_id=-lane_id,
        center_x_m=0.0,
        center_y_m=y,
        heading_rad=heading,
        lane_width_m=3.5,
        snap_distance_m=abs(y),
        is_in_lane=inside,
        probability=1.0,
        topology_relation=relation,
    )


class DiagnosticHDMapMatcherTests(unittest.TestCase):
    def test_low_confidence_short_lane_excursion_is_held_by_hysteresis(self):
        matcher = DiagnosticHDMapMatcher()
        matcher.update(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            candidates=[candidate(10, y=0.0, relation="same")],
        )

        for _ in range(6):
            result = matcher.update(
                ego_x_m=0.0,
                ego_y_m=0.0,
                ego_heading_rad=0.0,
                candidates=[
                    candidate(20, y=0.0, relation="longitudinal"),
                    candidate(10, y=0.2, relation="same"),
                ],
            )
            self.assertEqual(result.ad_lane_id, 10)
            self.assertEqual(
                result.match_reason,
                "pose_geometry_transition_hysteresis",
            )

    def test_high_confidence_transition_is_not_delayed(self):
        matcher = DiagnosticHDMapMatcher()
        matcher.update(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            candidates=[candidate(10, relation="same")],
        )

        result = matcher.update(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            candidates=[
                candidate(20, y=0.0, relation="longitudinal"),
                candidate(10, y=2.0, relation="same", inside=False),
            ],
        )

        self.assertEqual(result.ad_lane_id, 20)

    def test_pose_geometry_selects_nearest_aligned_lane(self):
        matcher = DiagnosticHDMapMatcher()
        result = matcher.update(
            ego_x_m=0.0,
            ego_y_m=0.2,
            ego_heading_rad=0.0,
            candidates=[candidate(10, y=0.0), candidate(20, y=3.5)],
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.ad_lane_id, 10)
        self.assertAlmostEqual(result.lateral_offset_m, 0.2)

    def test_heading_rejects_close_opposite_direction_lane(self):
        matcher = DiagnosticHDMapMatcher()
        result = matcher.update(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            candidates=[
                candidate(10, y=0.1, heading=math.pi),
                candidate(20, y=0.4, heading=0.0),
            ],
        )

        self.assertEqual(result.ad_lane_id, 20)

    def test_history_prefers_longitudinal_successor_over_unrelated_lane(self):
        matcher = DiagnosticHDMapMatcher()
        matcher.update(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            candidates=[candidate(10, relation="same")],
        )
        result = matcher.update(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            candidates=[
                candidate(11, y=0.15, relation="longitudinal"),
                candidate(99, y=0.0, relation="disconnected"),
            ],
        )

        self.assertEqual(result.ad_lane_id, 11)

    def test_topology_relation_uses_previous_rolling_corridors(self):
        corridors = {0: [10, 11], 1: [20], -1: [30]}

        self.assertEqual(topology_relation(
            candidate_lane_id=11,
            previous_lane_id=10,
            previous_corridors=corridors,
        ), "longitudinal")
        self.assertEqual(topology_relation(
            candidate_lane_id=20,
            previous_lane_id=10,
            previous_corridors=corridors,
        ), "left")
        self.assertEqual(topology_relation(
            candidate_lane_id=30,
            previous_lane_id=10,
            previous_corridors=corridors,
        ), "right")

    def test_local_frame_invariants_report_wrong_target_offset(self):
        violations = local_lane_frame_invariants(
            matched_lane_id=10,
            corridors={0: [10, 11], 1: [20], -1: [30]},
            target_lane_id=30,
            reported_target_offset=1,
        )

        self.assertEqual(violations, ["route_target_offset_mismatch"])

    def test_optimal_lane_does_not_skip_turn_for_remote_lane_change(self):
        planner = CustomGlobalPlannerAdapter.__new__(CustomGlobalPlannerAdapter)
        planner._stored_route_lane_ids = [10, 10, 10, 30, 30]
        planner._stored_route_options = [
            "LANEFOLLOW", "LEFT", "LANEFOLLOW", "CHANGELANERIGHT", "LANEFOLLOW"
        ]
        planner._stored_route_waypoints = [None] * 5

        self.assertEqual(planner._optimal_lane_from_index(0), 10)

    def test_authorization_rejects_topology_target_outside_local_frame(self):
        result = authorize_route_lane_change(
            route_lane_change_allowed=True,
            current_lane_id=1,
            route_required_lane_id=1,
            next_macro_maneuver="lane_change_right",
            current_road_option="lane_follow",
            remaining_distance_m=20.0,
            available_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            preparation_start_distance_m=50.0,
            latest_start_distance_m=5.0,
            target_safety_threshold=0.5,
            topology_current_lane_id=100,
            topology_target_lane_id=200,
            topology_lane_offset=0,
            topology_target_in_local_frame=False,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "route_target_outside_local_frame")

    def test_stored_route_extends_local_frame_across_longitudinal_connector(self):
        connector = _RouteWaypoint(5960149)
        outgoing = _RouteWaypoint(540156)
        outgoing_right = _RouteWaypoint(540155)
        outgoing._right = outgoing_right
        planner = _route_adapter(
            waypoints=[connector, outgoing],
            route_xy=[[0.0, 0.0], [20.0, 0.0]],
        )
        corridors = {0: [connector.ad_lane_id]}
        lane_to_offset = {connector.ad_lane_id: 0}

        planner._merge_stored_route_into_local_lane_graph(
            x_m=0.0,
            y_m=0.0,
            forward_distance_m=100.0,
            backward_distance_m=100.0,
            corridors=corridors,
            lane_to_offset=lane_to_offset,
        )

        self.assertEqual(lane_to_offset[outgoing.ad_lane_id], 0)
        self.assertEqual(lane_to_offset[outgoing_right.ad_lane_id], -1)
        self.assertIn(outgoing.ad_lane_id, corridors[0])
        self.assertIn(outgoing_right.ad_lane_id, corridors[-1])

    def test_stored_route_changes_offset_only_for_real_lateral_adjacency(self):
        current = _RouteWaypoint(540156)
        right = _RouteWaypoint(540155)
        current._right = right
        planner = _route_adapter(
            waypoints=[current, right],
            route_xy=[[0.0, 0.0], [20.0, 0.0]],
        )
        corridors = {0: [current.ad_lane_id]}
        lane_to_offset = {current.ad_lane_id: 0}

        planner._merge_stored_route_into_local_lane_graph(
            x_m=0.0,
            y_m=0.0,
            forward_distance_m=100.0,
            backward_distance_m=100.0,
            corridors=corridors,
            lane_to_offset=lane_to_offset,
        )

        self.assertEqual(lane_to_offset[right.ad_lane_id], -1)
        self.assertIn(right.ad_lane_id, corridors[-1])


class _RouteWaypoint:
    def __init__(self, ad_lane_id):
        self.ad_lane_id = int(ad_lane_id)
        self._left = None
        self._right = None

    def left(self):
        return self._left

    def right(self):
        return self._right


def _route_adapter(*, waypoints, route_xy):
    planner = CustomGlobalPlannerAdapter.__new__(CustomGlobalPlannerAdapter)
    planner._stored_route_waypoints = list(waypoints)
    planner._stored_route_xy = np.asarray(route_xy, dtype=float)
    deltas = np.diff(planner._stored_route_xy, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    planner._stored_route_cum_dists = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return planner


class ADMapRouteProgressTests(unittest.TestCase):
    def test_global_planner_nearest_query_is_stateless_geometry_only(self):
        planner = object.__new__(CustomGlobalPlannerAdapter)
        first_leg = [[float(index), 0.0] for index in range(50)]
        distant_leg = [[10.1 + float(index), 0.0] for index in range(50)]
        planner._stored_route_xy = np.asarray(first_leg + distant_leg, dtype=float)
        deltas = np.diff(planner._stored_route_xy, axis=0)
        planner._stored_route_cum_dists = np.concatenate((
            np.asarray([0.0]),
            np.cumsum(np.linalg.norm(deltas, axis=1)),
        ))
        index = planner._nearest_stored_route_index(10.1, 0.0, "ego")

        self.assertEqual(index, 50)
        self.assertFalse(hasattr(planner, "_query_indices"))


if __name__ == "__main__":
    unittest.main()
