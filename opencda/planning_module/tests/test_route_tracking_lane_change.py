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


from opencda_bridge.cpx_mpc_planner import (
    CPXMPCPlannerBridge,
    _adaptive_target_horizon_s,
)
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
        bridge._lane_id_discontinuity_since_lock = False
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
                "lane_change_progress": 1.0,
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
            ego_location=bridge.carla.Location(x=5.0, y=3.2),
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

    def test_committed_right_change_cannot_publish_lane_keep_fsm(self):
        state = CPXMPCPlannerBridge._normalized_final_lc_state(
            decision="lane_change_right",
            lc_state="LANE_KEEP",
            lane_change_phase="executing",
        )

        self.assertEqual(state, "EXECUTE_LANE_CHANGE_RIGHT")

    def test_lane_id_change_does_not_stabilize_between_lane_centers(self):
        bridge = self._bridge()
        bridge._route_tracking_lane_change_progress = 0.60
        bridge._route_tracking_lane_change_reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ]
        stabilization_calls = []
        bridge.reference_generator.target_lane_stabilization_samples = (
            lambda **kwargs: stabilization_calls.append(kwargs) or []
        )

        reason = bridge._release_completed_lane_change_commitment(
            current_lane_id=2,
            ego_location=bridge.carla.Location(x=5.0, y=0.4),
            ego_yaw_rad=0.02,
        )

        self.assertEqual(reason, "")
        self.assertEqual(stabilization_calls, [])
        self.assertEqual(bridge._route_tracking_lane_change_phase, "executing")
        self.assertFalse(
            bridge._route_tracking_lane_change_completion_debug[
                "lane_change_stabilization_geometry_ready"
            ]
        )

    def test_committed_change_publishes_stabilization_phase(self):
        state = CPXMPCPlannerBridge._normalized_final_lc_state(
            decision="lane_change_right",
            lc_state="LANE_KEEP",
            lane_change_phase="target_lane_stabilization",
        )

        self.assertEqual(state, "TARGET_LANE_STABILIZATION")

    def test_heading_misalignment_delays_stabilization_handoff(self):
        # current_lane_id/progress alone only capture that ego has crossed
        # into the target lane's lateral extent -- heading can still be
        # mid-turn. Starting the one-shot stabilization quintic while
        # heading is far from the target lane's own direction forces it to
        # reconcile a large heading gap over a short distance, producing a
        # sharp overshoot-then-correct steering profile (this is what
        # produced the observed post-lane-change lane-line touch). Confirm
        # the handoff is withheld -- not failed, just delayed -- when
        # heading is 30 degrees off a target lane reporting 0 degrees.
        bridge = self._bridge()
        bridge._route_tracking_lane_change_progress = 0.60
        bridge._route_tracking_lane_change_reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ]
        bridge.reference_generator._map_waypoint_callback = (
            lambda location: types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(
                        x=location.x, y=location.y
                    ),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=3.5,
            )
        )
        stabilization_calls = []
        bridge.reference_generator.target_lane_stabilization_samples = (
            lambda **kwargs: stabilization_calls.append(kwargs) or []
        )

        reason = bridge._release_completed_lane_change_commitment(
            current_lane_id=2,
            ego_location=bridge.carla.Location(x=5.0, y=3.2),
            ego_yaw_rad=math.radians(30.0),
        )

        self.assertEqual(reason, "")
        self.assertEqual(stabilization_calls, [])
        self.assertEqual(bridge._route_tracking_lane_change_phase, "executing")
        self.assertTrue(bridge._route_tracking_lane_change_reference)

    def test_heading_alignment_within_threshold_allows_stabilization_handoff(self):
        bridge = self._bridge()
        bridge._route_tracking_lane_change_progress = 0.60
        bridge._route_tracking_lane_change_reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ]
        bridge.reference_generator._map_waypoint_callback = (
            lambda location: types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(
                        x=location.x, y=location.y
                    ),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=3.5,
            )
        )
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
            ego_location=bridge.carla.Location(x=5.0, y=3.2),
            ego_yaw_rad=math.radians(5.0),
        )

        self.assertIn("target_lane_stabilization_started", reason)
        self.assertEqual(
            bridge._route_tracking_lane_change_phase,
            "target_lane_stabilization",
        )

    def test_target_lane_entry_never_continues_old_quintic_when_handoff_fails(self):
        bridge = self._bridge()
        bridge._route_tracking_lane_change_progress = 0.60
        bridge._route_tracking_lane_change_reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ]
        bridge.reference_generator.target_lane_stabilization_samples = (
            lambda **_kwargs: []
        )

        reason = bridge._release_completed_lane_change_commitment(
            current_lane_id=2,
            ego_location=bridge.carla.Location(x=5.0, y=3.2),
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

    def test_direct_tracking_progress_reflects_live_lag_not_stale_schedule(self):
        # Under direct target-lane tracking, the per-sample "lane_change_progress"
        # tag is a stale elapsed-time schedule (MPC's own QP, not the schedule,
        # now determines the real transient path). If ego lags that schedule
        # (e.g. steer-rate-limited), progress must reflect the real geometric
        # lag -- not the inflated scheduled value -- otherwise commitment/
        # completion gating (entry_min_progress/min_progress) could fire before
        # the vehicle has actually crossed, or (if the schedule under-reports)
        # never fire at all.
        bridge = self._bridge()
        master = []
        pairs = []
        for index in range(60):
            scheduled_progress = min(1.0, float(index + 1) / 20.0)
            master.append(
                {
                    "x_ref_m": 0.3 * float(index + 1),
                    "y_ref_m": 3.5,
                    "heading_rad": 0.0,
                    "lane_id": 2,
                    # Deliberately inflated vs. where ego will actually be,
                    # simulating a schedule that has outrun real progress.
                    "lane_change_progress": scheduled_progress,
                }
            )
            pairs.append((
                {"x_ref_m": 0.3 * float(index + 1), "y_ref_m": 0.0},
                {"x_ref_m": 0.3 * float(index + 1), "y_ref_m": 3.5},
            ))
        bridge._route_tracking_lane_change_reference = master
        bridge._route_tracking_lane_change_progress_pairs = pairs

        # Ego is laterally only 30% of the way across (y=1.05 of a 3.5m gap),
        # even though the nearest station's *scheduled* tag already claims
        # 100% (index 19+ -> scheduled_progress=1.0).
        window, _ = bridge._route_tracking_lane_change_window(
            ego_location=bridge.carla.Location(x=6.0, y=1.05),
            ego_yaw_rad=0.0,
            target_speed_mps=3.0,
            step_distance_m=0.3,
        )

        self.assertTrue(window)
        self.assertAlmostEqual(
            bridge._route_tracking_lane_change_progress, 0.30, places=2
        )

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


class AdaptiveTargetHorizonTests(unittest.TestCase):
    _PROFILE_HORIZON_S = {
        "lane_follow": 3.0,
        "prepare_lane_change": 4.5,
        "execute_lane_change": 4.5,
        "intersection_turn": 1.5,
        "stop": 2.0,
        "recovery": 1.5,
    }

    def test_uses_the_active_mode_profile_when_no_obstacle_info(self):
        for mode, expected in self._PROFILE_HORIZON_S.items():
            with self.subTest(mode=mode):
                self.assertEqual(
                    _adaptive_target_horizon_s(
                        mpc_cost_profile=mode,
                        nearest_obstacle_distance_m=None,
                        ego_speed_mps=3.0,
                        profile_horizon_s=self._PROFILE_HORIZON_S,
                    ),
                    expected,
                )

    def test_unlisted_mode_falls_back_to_lane_follow(self):
        self.assertEqual(
            _adaptive_target_horizon_s(
                mpc_cost_profile="some_unlisted_mode",
                nearest_obstacle_distance_m=None,
                ego_speed_mps=3.0,
                profile_horizon_s=self._PROFILE_HORIZON_S,
            ),
            self._PROFILE_HORIZON_S["lane_follow"],
        )

    def test_nearby_obstacle_shortens_horizon_below_the_mode_base(self):
        # distance_reaction_s = 3.0/2.0 = 1.5, stopping_time_s = 3.0/2.0 = 1.5
        # -> max is 1.5, well under execute_lane_change's 4.5s mode base.
        target = _adaptive_target_horizon_s(
            mpc_cost_profile="execute_lane_change",
            nearest_obstacle_distance_m=3.0,
            ego_speed_mps=3.0,
            profile_horizon_s=self._PROFILE_HORIZON_S,
        )
        self.assertAlmostEqual(target, 1.5)

    def test_distant_obstacle_does_not_shorten_horizon_below_the_mode_base(self):
        target = _adaptive_target_horizon_s(
            mpc_cost_profile="lane_follow",
            nearest_obstacle_distance_m=200.0,
            ego_speed_mps=3.0,
            profile_horizon_s=self._PROFILE_HORIZON_S,
        )
        self.assertAlmostEqual(target, self._PROFILE_HORIZON_S["lane_follow"])

    def test_slowing_to_a_stop_behind_a_close_obstacle_keeps_shrinking(self):
        # The bug this fixes: as ego comfortably decelerates toward a near-
        # stopped lead vehicle, a naive distance/ego_speed ratio blows up
        # (dividing by ego's own shrinking speed) instead of continuing to
        # shrink. distance_reaction_s (3.0/2.0=1.5) doesn't depend on ego's
        # speed at all, so it keeps the target short even as ego crawls to
        # a near-stop (0.1 m/s) close behind the obstacle.
        target = _adaptive_target_horizon_s(
            mpc_cost_profile="lane_follow",
            nearest_obstacle_distance_m=3.0,
            ego_speed_mps=0.1,
            profile_horizon_s=self._PROFILE_HORIZON_S,
        )
        self.assertAlmostEqual(target, 1.5)

    def test_fast_approach_to_a_close_obstacle_uses_the_larger_of_the_two_estimates(self):
        # Ego hasn't started decelerating yet (still at speed) right next to
        # a close obstacle: stopping_time_s (5.0/2.0=2.5) exceeds
        # distance_reaction_s (5.0/2.0=2.5 here too, but check the max logic
        # with a faster speed) -- use a speed high enough that
        # stopping_time_s dominates.
        target = _adaptive_target_horizon_s(
            mpc_cost_profile="lane_follow",
            nearest_obstacle_distance_m=5.0,
            ego_speed_mps=8.0,
            profile_horizon_s=self._PROFILE_HORIZON_S,
        )
        # distance_reaction_s = 5.0/2.0 = 2.5, stopping_time_s = 8.0/2.0 = 4.0
        # -> max is 4.0, capped at lane_follow's own 3.0 base.
        self.assertAlmostEqual(target, 3.0)


if __name__ == "__main__":
    unittest.main()
