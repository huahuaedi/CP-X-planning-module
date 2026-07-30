import math
import sys
import types
import unittest


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


from opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge
from pipeline.reference_generator import ReferenceGenerator


class RouteTrackingLaneChangeTests(unittest.TestCase):
    @staticmethod
    def _bridge():
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "reference_contract_lane_change_min_first_forward_m": 0.2,
            "route_tracking_recovery_heading_error_deg": 25.0,
            "route_tracking_lane_change_max_curvature_1pm": 0.35,
            "route_tracking_lane_change_max_heading_jump_rad": 0.35,
            "route_tracking_lane_change_max_lane_center_fraction": 0.70,
            "route_tracking_lane_change_max_boundary_failures": 1,
        }
        bridge.mpc = types.SimpleNamespace(horizon_steps=20, dt_s=0.1)
        bridge._route_tracking_lane_change_progress_index = 0
        bridge._route_tracking_lane_change_progress = 0.0
        bridge._route_tracking_lane_change_target_lane_id = 2
        bridge._route_tracking_lane_change_source_lane_id = 1
        bridge._route_tracking_lane_change_target_speed_mps = 2.0
        bridge._route_tracking_lane_change_phase = "executing"
        bridge._route_tracking_lane_change_stabilization_frames = 0
        bridge._route_tracking_lane_change_option = "CHANGELANERIGHT"
        bridge._route_tracking_lane_change_completed_option = ""
        bridge._route_tracking_lane_change_completion_stable_frames = 0
        bridge._route_tracking_lane_change_completion_debug = {}
        bridge.carla = sys.modules["carla"]
        bridge.reference_generator = ReferenceGenerator(
            config=bridge.config,
            mpc=bridge.mpc,
            map_planner=None,
            map_waypoint_from_location=lambda _location: None,
            lane_id_at_location=lambda _location: 1,
            body_frame_xy=bridge._body_frame_xy,
            target_speed_mps=3.0,
            lookahead_m=18.0,
        )
        bridge.reference_generator.lane_corridor_occupancy = (
            lambda **_kwargs: types.SimpleNamespace(
                valid=True,
                footprint_clearance_m=0.5,
            )
        )
        return bridge

    def test_target_lane_entry_replaces_quintic_with_stabilization_reference(self):
        bridge = self._bridge()
        bridge._route_tracking_lane_change_progress = 0.60
        bridge._route_tracking_lane_change_reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 0.60,
            }
            for index in range(30)
        ]
        bridge.reference_generator.target_lane_stabilization_samples = (
            lambda **_kwargs: [
                {
                    "x_ref_m": 5.8 + 0.2 * float(index),
                    "y_ref_m": 0.0,
                    "heading_rad": 0.0,
                    "lane_id": 2,
                }
                for index in range(25)
            ]
        )

        reason = bridge._release_completed_lane_change_commitment(
            current_lane_id=2,
            ego_location=bridge.carla.Location(x=5.0, y=0.4),
            ego_yaw_rad=0.02,
        )

        self.assertIn("target_lane_stabilization_started", reason)
        self.assertEqual(
            bridge._route_tracking_lane_change_phase,
            "target_lane_stabilization",
        )
        self.assertTrue(
            all(
                row["lane_transition_kind"] == "target_lane_stabilization"
                for row in bridge._route_tracking_lane_change_reference
            )
        )
        self.assertTrue(
            all(
                float(row["lane_change_progress"]) == 1.0
                for row in bridge._route_tracking_lane_change_reference
            )
        )

    def test_target_lane_entry_never_continues_old_quintic_when_handoff_fails(self):
        bridge = self._bridge()
        bridge._route_tracking_lane_change_progress = 0.60
        bridge._route_tracking_lane_change_reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 0.60,
            }
            for index in range(30)
        ]
        bridge.reference_generator.target_lane_stabilization_samples = (
            lambda **_kwargs: []
        )

        reason = bridge._release_completed_lane_change_commitment(
            current_lane_id=2,
            ego_location=bridge.carla.Location(x=5.0, y=0.4),
            ego_yaw_rad=0.02,
        )

        self.assertIn(
            "lane_change_stabilization_unavailable_to_lane_follow_recovery",
            reason,
        )
        self.assertEqual(bridge._route_tracking_lane_change_reference, [])
        self.assertEqual(
            bridge._route_tracking_lane_change_completed_option,
            "CHANGELANERIGHT",
        )

    def test_commitment_releases_after_geometric_convergence(self):
        bridge = self._bridge()
        bridge._route_tracking_lane_change_progress = 0.96
        bridge._route_tracking_lane_change_phase = "target_lane_stabilization"
        bridge._route_tracking_lane_change_reference = [
            {
                "x_ref_m": float(index),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(1, 11)
        ]

        reason = ""
        for _ in range(5):
            reason = bridge._release_completed_lane_change_commitment(
                current_lane_id=2,
                ego_location=bridge.carla.Location(x=5.0, y=0.1),
                ego_yaw_rad=0.02,
            )

        self.assertIn("lane_change_commitment_released", reason)
        self.assertEqual(bridge._route_tracking_lane_change_reference, [])
        self.assertEqual(
            bridge._route_tracking_lane_change_completed_option,
            "CHANGELANERIGHT",
        )

    def test_lane_id_change_alone_does_not_release_commitment(self):
        bridge = self._bridge()
        bridge._route_tracking_lane_change_progress = 0.96
        bridge._route_tracking_lane_change_phase = "target_lane_stabilization"
        bridge._route_tracking_lane_change_reference = [
            {
                "x_ref_m": float(index),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(1, 11)
        ]

        reason = bridge._release_completed_lane_change_commitment(
            current_lane_id=2,
            ego_location=bridge.carla.Location(x=5.0, y=1.2),
            ego_yaw_rad=0.25,
        )

        self.assertEqual(reason, "")
        self.assertTrue(bridge._route_tracking_lane_change_reference)

    def test_locked_window_advances_without_regenerating_master(self):
        bridge = self._bridge()
        master = []
        for index in range(60):
            progress = min(1.0, float(index + 1) / 40.0)
            master.append(
                {
                    "x_ref_m": 0.3 * float(index + 1),
                    "y_ref_m": 3.5 * progress,
                    "heading_rad": 0.0,
                    "lane_id": 1 if progress < 0.5 else 2,
                    "lane_change_progress": progress,
                }
            )
        bridge._route_tracking_lane_change_reference = master

        first, _ = bridge._route_tracking_lane_change_window(
            ego_location=bridge.carla.Location(x=0.0, y=0.0),
            ego_yaw_rad=0.0,
            target_speed_mps=3.0,
            step_distance_m=0.3,
        )
        second, _ = bridge._route_tracking_lane_change_window(
            ego_location=bridge.carla.Location(x=2.1, y=0.5),
            ego_yaw_rad=0.0,
            target_speed_mps=3.0,
            step_distance_m=0.3,
        )

        self.assertEqual(len(first), 20)
        self.assertEqual(len(second), 20)
        self.assertGreaterEqual(
            bridge._route_tracking_lane_change_progress_index,
            6,
        )
        self.assertGreater(
            float(second[0]["lane_change_progress"]),
            float(first[0]["lane_change_progress"]),
        )
        self.assertEqual(bridge._route_tracking_lane_change_reference, master)

    def test_validation_rejects_reference_over_25_degree_heading_error(self):
        bridge = self._bridge()
        bridge.reference_generator._map_waypoint_callback = lambda location: types.SimpleNamespace(
            transform=types.SimpleNamespace(
                location=types.SimpleNamespace(x=location.x, y=location.y),
                rotation=types.SimpleNamespace(yaw=30.0),
            ),
            lane_width=3.5,
        )
        reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": float(index + 1),
                "heading_rad": math.pi / 4.0,
            }
            for index in range(20)
        ]

        valid, reason = bridge._validate_route_tracking_lane_change_reference(
            reference=reference,
            ego_location=bridge.carla.Location(),
            ego_yaw_rad=0.0,
        )

        self.assertFalse(valid)
        self.assertIn("heading_error", reason)

    def test_quintic_lane_recovery_removes_turn_exit_curvature_spike(self):
        bridge = self._bridge()
        bridge.config["lane_recovery_anchor_forward_m"] = 0.8
        bridge.reference_generator.config["lane_recovery_anchor_forward_m"] = 0.8
        ego = bridge.carla.Location(x=21.88, y=52.97)
        waypoint = object()
        lane_samples = []
        for index in range(24):
            lane_samples.append({
                "x_ref_m": 22.5 + 0.3 * float(index),
                "y_ref_m": 52.37,
                "heading_rad": 0.0,
                "lane_width_m": 3.5,
            })
        bridge.reference_generator._current_lane_center_reference_samples = (
            lambda **kwargs: [dict(sample) for sample in lane_samples]
        )

        reference = bridge.reference_generator._ego_anchored_lane_recovery_reference_samples(
            ego_location=ego,
            ego_yaw_rad=math.radians(-8.36),
            start_waypoint=waypoint,
            current_lane_id=1,
            horizon_steps=20,
            step_distance_m=0.3,
            route_points=[],
        )

        self.assertEqual(len(reference), 20)
        forward_m, _ = bridge._body_frame_xy(
            origin_x_m=ego.x,
            origin_y_m=ego.y,
            heading_rad=math.radians(-8.36),
            target_x_m=float(reference[0]["x_ref_m"]),
            target_y_m=float(reference[0]["y_ref_m"]),
        )
        self.assertGreaterEqual(forward_m, 0.79)
        headings = []
        distances = []
        for first, second in zip(reference[:-1], reference[1:]):
            dx_m = float(second["x_ref_m"]) - float(first["x_ref_m"])
            dy_m = float(second["y_ref_m"]) - float(first["y_ref_m"])
            distances.append(math.hypot(dx_m, dy_m))
            headings.append(math.atan2(dy_m, dx_m))
        curvatures = [
            abs(
                math.atan2(
                    math.sin(second - first),
                    math.cos(second - first),
                )
            )
            / max(1.0e-6, distances[index + 1])
            for index, (first, second) in enumerate(
                zip(headings[:-1], headings[1:])
            )
        ]
        self.assertLess(max(curvatures), 0.35)


if __name__ == "__main__":
    unittest.main()
