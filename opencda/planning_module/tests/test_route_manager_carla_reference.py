import math
import unittest

from opencda.planning_module.pipeline.route_manager import CPXRouteManager
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


class _Waypoint:
    def __init__(self, x, y, lane_id=1, road_id=1):
        self.transform = _Transform(x, y)
        self.lane_id = int(lane_id)
        self.road_id = int(road_id)
        self.lane_width = 3.5


class RouteManagerCarlaReferenceTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
