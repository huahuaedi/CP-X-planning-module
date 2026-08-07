import math
import unittest

from opencda.planning_module.pipeline.route_manager import (
    CPXRouteManager,
    _route_required_carla_lane_id,
    _smooth_carla_reference_samples,
    select_route_aligned_waypoint_candidate,
)
from opencda.planning_module.pipeline.reference_contract import (
    contract_from_config,
    validate_reference_contract,
)


class _Location:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Transform:
    def __init__(self, x, y, z=0.0):
        self.location = _Location(x, y, z)

    def transform(self, location):
        """Match carla.Transform's point-transform method."""
        return location


class _Waypoint:
    def __init__(self, x, y, lane_id=1, road_id=1):
        self.transform = _Transform(x, y)
        self.lane_id = int(lane_id)
        self.road_id = int(road_id)
        self.lane_width = 3.5


class _SteppableWaypoint(_Waypoint):
    """Waypoint stub whose ``.next()`` walks a straight line, like a real
    CARLA waypoint chain following a lane -- used to test gap-bridging."""

    def __init__(self, x, y, heading_rad=0.0, lane_id=1, road_id=1):
        super().__init__(x, y, lane_id=lane_id, road_id=road_id)
        self._heading_rad = float(heading_rad)

    def next(self, distance):
        x = self.transform.location.x + float(distance) * math.cos(self._heading_rad)
        y = self.transform.location.y + float(distance) * math.sin(self._heading_rad)
        return [_SteppableWaypoint(x, y, self._heading_rad, self.lane_id, self.road_id)]


class _ConnectableWaypoint(_Waypoint):
    """Waypoint stub with settable get_left_lane()/get_right_lane()
    neighbors -- used to test lane-adjacency-based (as opposed to
    local-recount-based) lane id resolution."""

    def __init__(self, x, y, lane_id=1, road_id=1):
        super().__init__(x, y, lane_id=lane_id, road_id=road_id)
        self._left_lane = None
        self._right_lane = None

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane

    def set_neighbors(self, *, left=None, right=None):
        self._left_lane = left
        self._right_lane = right


class RouteManagerCarlaReferenceTest(unittest.TestCase):
    def test_boundary_aware_smoothing_never_leaves_shrunk_corridor(self):
        raw = [
            {
                "x_ref_m": x_m,
                "y_ref_m": y_m,
                "heading_rad": heading_rad,
                "lane_width_m": 2.8,
                "corridor_center_x_m": x_m,
                "corridor_center_y_m": y_m,
                "corridor_heading_rad": heading_rad,
            }
            for x_m, y_m, heading_rad in (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, math.pi / 2.0),
                (1.0, 2.0, math.pi / 2.0),
            )
        ]

        smoothed = _smooth_carla_reference_samples(
            raw,
            passes=8,
            boundary_aware=True,
            vehicle_half_width_m=1.0,
            boundary_margin_m=0.1,
            tracking_reserve_m=0.1,
        )

        for sample in smoothed[1:-1]:
            heading = float(sample["corridor_heading_rad"])
            dx_m = (
                float(sample["x_ref_m"])
                - float(sample["corridor_center_x_m"])
            )
            dy_m = (
                float(sample["y_ref_m"])
                - float(sample["corridor_center_y_m"])
            )
            lateral_m = (
                -math.sin(heading) * dx_m + math.cos(heading) * dy_m
            )
            self.assertLessEqual(abs(lateral_m), 0.201)

    def test_weighted_selector_uses_route_tangent_at_close_fork(self):
        class _Candidate:
            def __init__(self, x, y, yaw):
                self.transform = type(
                    "Transform",
                    (),
                    {
                        "location": _Location(x, y),
                        "rotation": type("Rotation", (), {"yaw": float(yaw)})(),
                    },
                )()

        wrong = _Candidate(2.0, 0.05, 80.0)
        aligned = _Candidate(2.0, 0.20, 0.0)
        selected = select_route_aligned_waypoint_candidate(
            candidates=[wrong, aligned],
            route_points=[[0.0, 0.0], [10.0, 0.0]],
            previous_heading_rad=0.0,
        )

        self.assertIs(selected, aligned)

    def test_internal_carla_route_exposes_lane_change_before_later_turn(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0, lane_id=1), "LANEFOLLOW"),
            (_Waypoint(4.0, 0.0, lane_id=1), "LANEFOLLOW"),
            (_Waypoint(8.0, -1.0, lane_id=1), "CHANGELANERIGHT"),
            (_Waypoint(12.0, -3.5, lane_id=2), "CHANGELANERIGHT"),
            (_Waypoint(18.0, -3.5, lane_id=2), "LANEFOLLOW"),
            (_Waypoint(24.0, -3.5, lane_id=2), "LEFT"),
        ]

        route_info = manager.get_route_info(
            x_m=0.0,
            y_m=0.0,
            query_key="lane_change_test",
            fallback_lane_id=1,
        )

        self.assertEqual(route_info["next_macro_maneuver"], "Lane Change Right")
        self.assertEqual(route_info["optimal_lane_id"], 2)
        self.assertEqual(route_info["debug_reason"], "carla_grp_route_active")
        self.assertGreater(route_info["remaining_distance_m"], 20.0)

    def test_external_leaderboard_route_preserves_turn_semantics(self):
        class _Map:
            @staticmethod
            def get_waypoint(location):
                return _Waypoint(location.x, location.y, lane_id=2, road_id=7)

        manager = CPXRouteManager(global_planner=object(), carla_map=_Map())
        manager.set_external_carla_route([
            (_Transform(0.0, 0.0), "LANEFOLLOW"),
            (_Transform(5.0, 0.0), "LANEFOLLOW"),
            (_Transform(10.0, 0.0), "RIGHT"),
            (_Transform(12.0, -2.0), "RIGHT"),
        ])

        route_info = manager.get_route_info(
            x_m=0.0,
            y_m=0.0,
            query_key="external_test",
            fallback_lane_id=1,
        )
        direction, distance_m, reason = manager.upcoming_turn(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            lookahead_m=15.0,
        )

        self.assertTrue(route_info["route_found"])
        self.assertEqual(route_info["optimal_lane_id"], 1)
        self.assertEqual(route_info["next_macro_maneuver"], "Right Turn")
        self.assertEqual(direction, "right")
        self.assertAlmostEqual(distance_m, 10.0)
        self.assertEqual(reason, "carla_route_turn_ahead")
        self.assertEqual(
            manager.carla_route_debug_reason,
            "external_leaderboard_route_ready",
        )

    def test_reports_upcoming_carla_turn_before_current_segment(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(5.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(10.0, 0.0), "RIGHT"),
            (_Waypoint(12.0, -2.0), "RIGHT"),
        ]

        direction, distance_m, reason = manager.upcoming_turn(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            lookahead_m=15.0,
        )

        self.assertEqual(direction, "right")
        self.assertAlmostEqual(distance_m, 10.0)
        self.assertEqual(reason, "carla_route_turn_ahead")

    def test_lane_change_geometry_is_not_reported_as_intersection_turn(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0, lane_id=1), "LANEFOLLOW"),
            (_Waypoint(4.0, 0.0, lane_id=1), "LANEFOLLOW"),
            (_Waypoint(8.0, 1.5, lane_id=1), "CHANGELANELEFT"),
            (_Waypoint(11.0, 3.5, lane_id=2), "CHANGELANELEFT"),
            (_Waypoint(16.0, 3.5, lane_id=2), "LANEFOLLOW"),
        ]

        direction, distance_m, reason = manager.upcoming_turn(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            lookahead_m=20.0,
        )

        self.assertEqual(direction, "")
        self.assertTrue(math.isinf(distance_m))
        self.assertEqual(reason, "carla_route_lane_change_precedes_turn")

    def test_infers_turn_from_geometry_when_route_option_is_wrong(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(0.0, 5.0), "LANEFOLLOW"),
            (_Waypoint(0.0, 10.0), "STRAIGHT"),
            (_Waypoint(3.0, 13.0), "LANEFOLLOW"),
            (_Waypoint(7.0, 13.0), "LANEFOLLOW"),
        ]

        direction, distance_m, reason = manager.upcoming_turn(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=math.pi / 2.0,
            lookahead_m=20.0,
        )

        self.assertEqual(direction, "right")
        self.assertAlmostEqual(distance_m, 10.0)
        self.assertEqual(reason, "carla_route_geometry_turn_ahead")

    def test_reference_smoothly_rejoins_route_from_small_lateral_offset(self):
        manager = CPXRouteManager(
            global_planner=object(),
            carla_rejoin_min_lateral_m=0.35,
            carla_rejoin_max_lateral_m=3.0,
            carla_rejoin_distance_m=8.0,
        )
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW")
            for index in range(20)
        ]

        reference, reason = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=1.3,
            ego_heading_rad=0.0,
            horizon_steps=16,
            step_distance_m=0.5,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )

        first_forward_m = float(reference[0]["x_ref_m"])
        first_lateral_m = float(reference[0]["y_ref_m"]) - 1.3
        self.assertIn("route_rejoin", reason)
        self.assertGreater(first_forward_m, 0.2)
        self.assertLess(abs(first_lateral_m), 0.2)
        self.assertLess(abs(float(reference[-1]["y_ref_m"])), 0.05)
        self.assertTrue(all(
            sample["lane_transition_kind"] == "longitudinal_successor"
            for sample in reference
        ))

    def test_explicit_carla_lane_change_is_not_marked_as_longitudinal_successor(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0, lane_id=1), "LANEFOLLOW"),
            (_Waypoint(2.0, 0.0, lane_id=1), "LANEFOLLOW"),
            (_Waypoint(4.0, 1.0, lane_id=2), "CHANGELANELEFT"),
            (_Waypoint(6.0, 2.0, lane_id=2), "CHANGELANELEFT"),
        ]

        reference, _ = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=10,
            step_distance_m=0.5,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )

        self.assertTrue(any(
            sample["lane_transition_kind"] == "lateral_lane_change"
            for sample in reference
        ))

    def test_straight_junction_lane_id_transition_keeps_route_geometry(self):
        """Lane renumbering at a junction must not redirect a straight route."""

        manager = CPXRouteManager(
            global_planner=object(),
            carla_reference_smoothing_passes=3,
        )
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0, lane_id=1, road_id=10), "LANEFOLLOW"),
            (_Waypoint(2.0, 0.0, lane_id=1, road_id=10), "STRAIGHT"),
            (_Waypoint(4.0, 0.0, lane_id=2, road_id=20), "STRAIGHT"),
            (_Waypoint(6.0, 0.0, lane_id=2, road_id=20), "LANEFOLLOW"),
            (_Waypoint(8.0, 0.0, lane_id=2, road_id=20), "LANEFOLLOW"),
            (_Waypoint(10.0, 0.0, lane_id=2, road_id=20), "LANEFOLLOW"),
        ]
        manager._carla_route_debug_reason = "carla_grp_route_ready"

        reference, reason = manager.carla_waypoint_reference(
            ego_x_m=1.9,
            ego_y_m=0.15,
            ego_heading_rad=0.0,
            horizon_steps=14,
            step_distance_m=0.5,
            target_speed_mps=2.0,
            fallback_lane_id=2,
            anchor_to_ego_heading=False,
        )

        self.assertIn("carla_grp_waypoint_chain_smoothed", reason)
        self.assertEqual(len(reference), 14)
        self.assertTrue(all(
            float(current["x_ref_m"]) > float(previous["x_ref_m"])
            for previous, current in zip(reference[:-1], reference[1:])
        ))
        self.assertLess(
            max(abs(float(row["y_ref_m"])) for row in reference),
            0.16,
        )
        self.assertTrue(all(
            row["lane_transition_kind"] == "longitudinal_successor"
            for row in reference
        ))
        self.assertIn(2, {int(row["lane_id"]) for row in reference})

    def test_samples_selected_turn_connector_without_duplicate_points(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(2.0, 0.0), "LEFT"),
            (_Waypoint(4.0, 1.0), "LEFT"),
            (_Waypoint(5.0, 3.0), "LEFT"),
            (_Waypoint(5.0, 5.0), "LANEFOLLOW"),
        ]
        manager._carla_route_debug_reason = "carla_grp_route_ready"

        reference, reason = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=12,
            step_distance_m=0.5,
            target_speed_mps=1.2,
            fallback_lane_id=1,
        )

        self.assertEqual(reason, "carla_grp_waypoint_chain_smoothed")
        self.assertEqual(len(reference), 12)
        self.assertGreater(reference[0]["x_ref_m"], 0.0)
        self.assertTrue(any(sample["y_ref_m"] > 0.5 for sample in reference))
        for previous, current in zip(reference[:-1], reference[1:]):
            distance_m = math.hypot(
                current["x_ref_m"] - previous["x_ref_m"],
                current["y_ref_m"] - previous["y_ref_m"],
            )
            self.assertGreater(distance_m, 1.0e-3)

    def test_route_progress_never_moves_backward(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW")
            for index in range(12)
        ]
        manager._carla_route_debug_reason = "carla_grp_route_ready"

        manager.carla_waypoint_reference(
            ego_x_m=6.2,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=4,
            step_distance_m=0.5,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )
        progressed_index = manager._carla_route_progress_index
        manager.carla_waypoint_reference(
            ego_x_m=3.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=4,
            step_distance_m=0.5,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )

        self.assertGreaterEqual(manager._carla_route_progress_index, progressed_index)

    def test_first_sync_searches_entire_long_route(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW")
            for index in range(300)
        ]
        manager._carla_route_debug_reason = "carla_grp_route_ready"

        reason = manager.sync_carla_route_progress(
            ego_x_m=240.2,
            ego_y_m=0.1,
            ego_heading_rad=0.0,
        )
        reference, reference_reason = manager.carla_waypoint_reference(
            ego_x_m=240.2,
            ego_y_m=0.1,
            ego_heading_rad=0.0,
            horizon_steps=6,
            step_distance_m=0.5,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )

        self.assertIn("carla_route_progress_global_init", reason)
        self.assertGreaterEqual(manager.carla_route_progress_index, 239)
        self.assertEqual(reference_reason, "carla_grp_waypoint_chain_smoothed")
        self.assertEqual(len(reference), 6)
        self.assertLess(abs(reference[0]["x_ref_m"] - 240.7), 0.2)

    def test_geometry_route_uses_carla_waypoints_instead_of_custom_polyline(self):
        class _GlobalPlanner:
            def get_current_route_info(self, **_kwargs):
                return None

        manager = CPXRouteManager(global_planner=_GlobalPlanner())
        manager._fallback_route_points = [[100.0, 100.0], [110.0, 100.0]]
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(2.0, 0.0), "LEFT"),
            (_Waypoint(3.0, 2.0), "LEFT"),
        ]

        points = manager.geometry_route_points()

        self.assertEqual([(row[0], row[1]) for row in points], [
            (0.0, 0.0),
            (2.0, 0.0),
            (3.0, 2.0),
        ])

    def test_smoothed_connector_has_unique_points_and_continuous_headings(self):
        manager = CPXRouteManager(
            global_planner=object(),
            carla_reference_smoothing_passes=4,
        )
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(2.0, 0.0), "LEFT"),
            (_Waypoint(3.0, 0.2), "LEFT"),
            (_Waypoint(3.8, 1.0), "LEFT"),
            (_Waypoint(4.0, 2.0), "LEFT"),
            (_Waypoint(4.0, 4.0), "LANEFOLLOW"),
        ]

        reference, _ = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=16,
            step_distance_m=0.35,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )

        distances = [
            math.hypot(
                float(current["x_ref_m"]) - float(previous["x_ref_m"]),
                float(current["y_ref_m"]) - float(previous["y_ref_m"]),
            )
            for previous, current in zip(reference[:-1], reference[1:])
        ]
        heading_steps = [
            abs(math.atan2(
                math.sin(float(current["heading_rad"]) - float(previous["heading_rad"])),
                math.cos(float(current["heading_rad"]) - float(previous["heading_rad"])),
            ))
            for previous, current in zip(reference[:-1], reference[1:])
        ]
        self.assertEqual(len(reference), 16)
        self.assertGreater(min(distances), 1.0e-3)
        self.assertLess(max(heading_steps), 0.75)
        contract = contract_from_config(
            mode="intersection_turn",
            expected_lane_id=1,
            horizon_steps=16,
            config={},
            default_speed_mps=2.0,
        )
        validation = validate_reference_contract(
            reference_samples=reference,
            destination_state=[
                float(reference[-1]["x_ref_m"]),
                float(reference[-1]["y_ref_m"]),
                1.0,
                0.0,
                1,
            ],
            ego_state=[0.0, 0.0, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=False,
        )
        self.assertTrue(validation.valid, validation.reason())

    def test_turn_connector_is_anchored_to_ego_pose_and_heading(self):
        manager = CPXRouteManager(
            global_planner=object(),
            carla_reference_smoothing_passes=3,
        )
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(2.0, 0.0), "LEFT"),
            (_Waypoint(3.5, 0.5), "LEFT"),
            (_Waypoint(4.5, 2.0), "LEFT"),
            (_Waypoint(4.5, 4.0), "LANEFOLLOW"),
            (_Waypoint(4.5, 7.0), "LANEFOLLOW"),
        ]

        reference, reason = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.7,
            ego_heading_rad=0.0,
            horizon_steps=20,
            step_distance_m=0.35,
            target_speed_mps=1.5,
            fallback_lane_id=1,
            anchor_to_ego_heading=True,
            ego_anchor_distance_m=6.0,
        )

        self.assertIn("ego_heading_connector", reason)
        self.assertEqual(len(reference), 20)
        first = reference[0]
        self.assertGreater(float(first["x_ref_m"]), 0.2)
        self.assertLess(abs(float(first["y_ref_m"]) - 0.7), 0.12)
        first_heading = math.atan2(
            float(reference[1]["y_ref_m"]) - float(first["y_ref_m"]),
            float(reference[1]["x_ref_m"]) - float(first["x_ref_m"]),
        )
        self.assertLess(abs(first_heading), 0.15)
        contract = contract_from_config(
            mode="intersection_turn",
            expected_lane_id=1,
            horizon_steps=20,
            config={},
            default_speed_mps=2.0,
        )
        validation = validate_reference_contract(
            reference_samples=reference,
            destination_state=[
                float(reference[-1]["x_ref_m"]),
                float(reference[-1]["y_ref_m"]),
                1.0,
                float(reference[-1]["heading_rad"]),
                1,
            ],
            ego_state=[0.0, 0.7, 0.0, 0.0],
            contract=contract,
            check_destination_body_lateral=False,
        )
        self.assertTrue(validation.valid, validation.reason())

    def test_route_alignment_uses_outgoing_heading_lookahead(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(2.0, 0.0), "RIGHT"),
            (_Waypoint(3.0, 1.0), "RIGHT"),
            (_Waypoint(3.0, 4.0), "LANEFOLLOW"),
            (_Waypoint(3.0, 8.0), "LANEFOLLOW"),
        ]

        heading_error, lateral_m, _ = manager.carla_route_alignment(
            ego_x_m=2.8,
            ego_y_m=0.8,
            ego_heading_rad=0.0,
            heading_lookahead_m=4.0,
        )

        self.assertGreater(heading_error, 1.2)
        self.assertLess(lateral_m, 0.5)

    def test_bridges_large_gap_between_consecutive_carla_route_waypoints(self):
        """GlobalRoutePlanner.trace_route() can emit two consecutive
        waypoints several meters apart at some junction connectors (a real
        discontinuity in its topology graph, not a step the ego can take in
        one tick). This starves the fine-grained reference geometry and can
        strand the route-progress index on the near side of the gap forever
        -- see route_manager.py::_bridge_carla_route_gaps for the full story.
        """

        manager = CPXRouteManager(global_planner=object(), carla_map=object())
        start = _SteppableWaypoint(0.0, 0.0, heading_rad=0.0)
        target = _Waypoint(10.0, 0.0)
        entries = [(start, "LANEFOLLOW"), (target, "CHANGELANERIGHT")]

        bridged = manager._bridge_carla_route_gaps(entries)

        self.assertGreater(len(bridged), len(entries))
        points = [
            (waypoint.transform.location.x, waypoint.transform.location.y)
            for waypoint, _option in bridged
        ]
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertEqual(points[-1], (10.0, 0.0))
        gap_threshold_m = max(3.0, 2.5 * manager.carla_route_sampling_resolution_m)
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            self.assertLessEqual(math.hypot(x2 - x1, y2 - y1), gap_threshold_m)

    def test_bridges_lane_change_gap_with_lateral_interpolation_not_forward_stepping(self):
        """A CHANGELANE-tagged edge is a deliberate lateral jump --
        GlobalRoutePlanner._lane_change_link() links it with an explicit
        empty ``path=[]`` (global_route_planner.py), so the two waypoints
        sit on different, roughly-parallel lanes rather than being far apart
        along the SAME lane. ``waypoint.next()`` only walks forward along
        the lane it's already on, so it can never close a purely lateral
        gap -- using it here would walk straight past the lane change while
        barely approaching the target lane, producing a spurious trail of
        points along the wrong lane (this showed up as a wrong, zigzagging
        "corner" instead of a smooth lane change in the debug minimap).
        """

        class _LaneChangeMap:
            @staticmethod
            def get_waypoint(location):
                return _Waypoint(location.x, location.y, lane_id=2, road_id=1)

        manager = CPXRouteManager(global_planner=object(), carla_map=_LaneChangeMap())
        start = _Waypoint(0.0, 0.0, lane_id=1, road_id=1)
        # Real trace_route() output tags BOTH sides of a lane-change edge
        # with the same CHANGELANE option (see global_route_planner.py's
        # trace_route(), which appends the pair back-to-back).
        target = _Waypoint(0.0, 3.5, lane_id=2, road_id=1)
        entries = [(start, "CHANGELANERIGHT"), (target, "CHANGELANERIGHT")]

        bridged = manager._bridge_carla_route_gaps(entries)

        points = [
            (waypoint.transform.location.x, waypoint.transform.location.y)
            for waypoint, _option in bridged
        ]
        self.assertGreater(len(bridged), len(entries))
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertEqual(points[-1], (0.0, 3.5))
        # Every inserted point must lie on the straight lateral line between
        # the two lanes (x stays 0), not walk forward along the old lane.
        for x, _y in points:
            self.assertAlmostEqual(x, 0.0)
        ys = [y for _x, y in points]
        self.assertEqual(ys, sorted(ys))

    def test_route_required_lane_id_uses_hop_offset_when_ego_waypoint_is_known(self):
        """When ego_waypoint is provided, the target lane id is derived as
        ego's own current_lane_id plus the real lane-adjacency hop count to
        the maneuver point's waypoint, instead of an independent local
        recount at that (possibly differently-laned) point -- see
        route_manager.py::_route_required_carla_lane_id and
        carla_lane_graph.lane_hop_offset for why an independent recount is
        unsound whenever the lane count differs between the two points.
        """

        ego_lane1 = _ConnectableWaypoint(0.0, 0.0, lane_id=1, road_id=1)
        ego_lane2 = _ConnectableWaypoint(0.0, 3.5, lane_id=2, road_id=1)
        ego_lane3 = _ConnectableWaypoint(0.0, 7.0, lane_id=3, road_id=1)
        ego_lane1.set_neighbors(left=ego_lane2)
        ego_lane2.set_neighbors(left=ego_lane3, right=ego_lane1)
        ego_lane3.set_neighbors(right=ego_lane2)

        # The CHANGELANERIGHT node's waypoint is ego's real right-hand
        # physical neighbor (lane_hop_offset(ego_lane2, ego_lane1) == -1).
        nodes = [
            (0.0, 0.0, 0.0, ego_lane2, "LANEFOLLOW"),
            (10.0, 0.0, 0.0, ego_lane1, "CHANGELANERIGHT"),
        ]

        target_lane_id = _route_required_carla_lane_id(
            nodes=nodes,
            start_index=0,
            fallback_lane_id=2,
            ego_waypoint=ego_lane2,
        )

        self.assertEqual(target_lane_id, 1)

    def test_does_not_bridge_gaps_below_threshold(self):
        manager = CPXRouteManager(global_planner=object(), carla_map=object())
        entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(1.0, 0.0), "LANEFOLLOW"),
        ]

        bridged = manager._bridge_carla_route_gaps(entries)

        self.assertEqual(len(bridged), len(entries))

    def test_unbounded_extrapolation_drifts_far_past_a_short_route(self):
        # Default behavior (today's, unchanged): once the requested
        # lookahead exceeds the real route length, samples keep extrapolating
        # straight ahead without bound.
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW") for index in range(5)
        ]

        reference, _ = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=20,
            step_distance_m=1.0,
            target_speed_mps=1.0,
            fallback_lane_id=1,
        )

        self.assertGreater(float(reference[-1]["x_ref_m"]), 15.0)

    def test_max_extrapolation_m_caps_drift_past_a_short_route(self):
        # A short connector (like an intersection turn) combined with a long
        # requested horizon must not hand MPC a reference tail that runs
        # arbitrarily far past the real geometry in a straight line.
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW") for index in range(5)
        ]

        reference, _ = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=20,
            step_distance_m=1.0,
            target_speed_mps=1.0,
            fallback_lane_id=1,
            max_extrapolation_m=2.0,
        )

        last_real_x_m = 4.0
        for sample in reference:
            self.assertLessEqual(
                float(sample["x_ref_m"]),
                last_real_x_m + 2.0 + 1.0e-6,
            )
            self.assertTrue(math.isfinite(float(sample["heading_rad"])))

    def test_extrapolated_samples_inherit_real_lane_width_by_default(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW") for index in range(5)
        ]

        reference, _ = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=20,
            step_distance_m=1.0,
            target_speed_mps=1.0,
            fallback_lane_id=1,
            max_extrapolation_m=2.0,
        )

        # Real (first) and extrapolated (last) samples both keep the real
        # waypoint's own lane width when no override is given -- today's
        # unchanged behavior.
        self.assertAlmostEqual(float(reference[0]["lane_width_m"]), 3.5)
        self.assertAlmostEqual(float(reference[-1]["lane_width_m"]), 3.5)

    def test_extrapolated_lane_width_m_overrides_only_the_parked_tail(self):
        manager = CPXRouteManager(global_planner=object())
        manager._carla_route_entries = [
            (_Waypoint(float(index), 0.0), "LANEFOLLOW") for index in range(5)
        ]

        reference, _ = manager.carla_waypoint_reference(
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            horizon_steps=20,
            step_distance_m=1.0,
            target_speed_mps=1.0,
            fallback_lane_id=1,
            max_extrapolation_m=2.0,
            extrapolated_lane_width_m=8.0,
        )

        # The first sample is still real geometry -- unaffected by the
        # override. The last sample is parked past the real route's end --
        # it should carry the generous override width instead of the real
        # waypoint's normal 3.5m lane width.
        self.assertAlmostEqual(float(reference[0]["lane_width_m"]), 3.5)
        self.assertAlmostEqual(float(reference[-1]["lane_width_m"]), 8.0)

    def test_replan_from_atomically_replaces_route_on_success(self):
        summary = type(
            "Summary",
            (),
            {"route_found": True, "route_waypoints": [(0.0, 0.0), (2.0, 0.0)]},
        )()
        planner = type(
            "Planner",
            (),
            {"plan_route_from_locations": lambda self, **kwargs: summary},
        )()
        manager = CPXRouteManager(global_planner=planner)
        manager._goal_point = {"x": 10.0, "y": 0.0, "z": 0.0}
        manager._build_carla_route = lambda **kwargs: setattr(
            manager,
            "_carla_route_entries",
            [(_Waypoint(2.0, 0.0), "LANEFOLLOW"), (_Waypoint(10.0, 0.0), "LANEFOLLOW")],
        )

        result = manager.replan_from(
            start_point={"x": 2.0, "y": 0.0, "z": 0.0},
            trigger_reason="turn_reference_unavailable",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.route_point_count, 2)
        self.assertEqual(manager._start_point["x"], 2.0)
        self.assertIn("carla_grp_route_replanned", manager.carla_route_debug_reason)

    def test_replan_from_restores_previous_route_on_failure(self):
        planner = type(
            "Planner",
            (),
            {"plan_route_from_locations": lambda self, **kwargs: object()},
        )()
        manager = CPXRouteManager(global_planner=planner)
        old_entries = [
            (_Waypoint(0.0, 0.0), "LANEFOLLOW"),
            (_Waypoint(5.0, 0.0), "LANEFOLLOW"),
        ]
        manager._goal_point = {"x": 10.0, "y": 0.0, "z": 0.0}
        manager._start_point = {"x": 0.0, "y": 0.0, "z": 0.0}
        manager._carla_route_entries = old_entries
        manager._carla_route_debug_reason = "old_route_ready"
        manager._build_carla_route = lambda **kwargs: setattr(
            manager, "_carla_route_entries", []
        )

        result = manager.replan_from(
            start_point={"x": 2.0, "y": 1.0, "z": 0.0},
            trigger_reason="turn_reference_unavailable",
        )

        self.assertFalse(result.success)
        self.assertIs(manager._carla_route_entries, old_entries)
        self.assertEqual(manager._start_point["x"], 0.0)
        self.assertEqual(manager.carla_route_debug_reason, "old_route_ready")


if __name__ == "__main__":
    unittest.main()
