import math
import inspect
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
from pipeline.local_map_snapshot import LocalMapSnapshot, build_local_map_snapshot
from pipeline.maneuver_manager import ManeuverManager
from pipeline.fallback_manager import TrajectoryFallbackManager
from pipeline.candidate_evaluation import CandidateTrajectoryEvaluator
from pipeline.reference_line_provider import (
    CandidateReferenceBuildContext,
    LANE_CHANGE,
    LANE_FOLLOW,
    POST_TURN,
    TURN,
    ReferenceLineProvider,
    TurnReferenceRequest,
)


class RouteTrackingLaneChangeTests(unittest.TestCase):
    @staticmethod
    def _install_lane_change_reference(bridge, samples):
        installed, reason = bridge._stable_reference_line_provider.install(
            LANE_CHANGE,
            samples,
            route_revision="route-test",
            map_epoch="town06-test",
            event="maneuver_started",
            source_lane_id=1,
            target_lane_id=2,
        )
        if not installed:
            raise AssertionError(reason)

    @staticmethod
    def _lane_change_reference(bridge):
        return bridge._stable_reference_line_provider.snapshot(
            LANE_CHANGE
        ).mutable_samples()

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
        bridge.maneuver_manager = ManeuverManager(bridge.config)
        bridge.maneuver_manager.lane_change.target_lane_id = 2
        bridge.maneuver_manager.lane_change.source_lane_id = 1
        bridge.maneuver_manager.lane_change.target_speed_mps = 2.0
        bridge.maneuver_manager.lane_change.phase = "executing"
        bridge.maneuver_manager.lane_change.stabilization_frames = 0
        bridge.maneuver_manager.lane_change.option = "CHANGELANERIGHT"
        bridge.maneuver_manager.lane_change.completed_option = ""
        bridge.maneuver_manager.lane_change.completion_stable_frames = 0
        bridge.maneuver_manager.lane_change.transition_to_turn_arc_m = 0.0
        bridge.maneuver_manager.lane_change.transition_to_turn_step_m = 0.0
        bridge._stable_reference_line_provider = ReferenceLineProvider()
        fallback = TrajectoryFallbackManager()
        bridge.pipeline = types.SimpleNamespace(resolve_fallback=fallback.resolve)
        bridge.route_manager = types.SimpleNamespace(route_revision="route-test")
        bridge._sim_time_s = lambda: 1.0
        bridge.strict_explicit_fallback_speed_mps = 0.8
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

    def test_explicit_failure_is_owned_by_fallback_manager(self):
        bridge = self._bridge()
        bridge._stable_reference_line_provider.install(
            LANE_FOLLOW,
            [
                {"x_ref_m": 0.5, "y_ref_m": 0.0, "speed_ref_mps": 1.0},
                {"x_ref_m": 1.5, "y_ref_m": 0.0, "speed_ref_mps": 1.0},
                {"x_ref_m": 2.5, "y_ref_m": 0.0, "speed_ref_mps": 1.0},
            ],
            route_revision="route-test",
            map_epoch="town06-test",
            event="initial_route",
        )

        result = TrajectoryFallbackManager().resolve_candidate_failure(
                candidate_results=[],
                baseline_decision="lane_follow",
                baseline_target_lane_id=500144,
                current_lane_id=500145,
                current_state=[0.0, 0.0, 1.0, 0.0],
                ego_x_m=0.0,
                ego_y_m=0.0,
                reference_provider=bridge._stable_reference_line_provider,
                route_revision="route-test",
                sim_time_s=1.0,
                mpc_dt_s=0.1,
                horizon_steps=20,
                lane_change_min_first_forward_m=0.2,
                lane_follow_min_first_forward_m=0.2,
                summarize_candidates=lambda _rows: "[]",
                selection_reason="all_candidates_infeasible",
            )
        decision = result.decision
        lane_id = result.target_lane_id
        speed_mps = result.target_speed_mps
        reference = result.mutable_trajectory()
        destination = result.mutable_destination_state()
        debug = result.mutable_diagnostics()

        # A recoverable geometry failure owns a bounded stop trajectory, not
        # the emergency behavior state.  Keeping lane_follow here lets the
        # candidate pipeline retry on the next tick instead of entering the
        # direct-control emergency hard gate forever.
        self.assertEqual(decision, "lane_follow")
        self.assertEqual(lane_id, 500144)
        self.assertEqual(speed_mps, 0.0)
        self.assertEqual(destination[2], 0.0)
        self.assertTrue(reference)
        self.assertEqual(
            debug["candidate_pipeline_selected"],
            "bounded_safe_stop",
        )

    def test_display_route_smoothing_does_not_mutate_topology_route(self):
        bridge = self._bridge()
        raw = (
            [[float(x), 0.0, 0.0, math.pi] for x in range(20, 9, -1)]
            + [[10.0, -float(y), 0.0, -math.pi / 2.0] for y in range(1, 4)]
            + [[float(x), -3.0, 0.0, math.pi] for x in range(9, -1, -1)]
        )
        bridge._active_global_route_points = lambda: [list(point) for point in raw]

        display = bridge._display_global_route_points()

        self.assertEqual(bridge._active_global_route_points(), raw)
        self.assertEqual(len(display), len(raw))
        raw_headings = [
            math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(raw, raw[1:])
        ]
        display_headings = [
            math.atan2(b[1] - a[1], b[0] - a[0])
            for a, b in zip(display, display[1:])
        ]
        max_raw_jump = max(
            abs(math.atan2(math.sin(b - a), math.cos(b - a)))
            for a, b in zip(raw_headings, raw_headings[1:])
        )
        max_display_jump = max(
            abs(math.atan2(math.sin(b - a), math.cos(b - a)))
            for a, b in zip(display_headings, display_headings[1:])
        )
        self.assertLess(max_display_jump, max_raw_jump)

    def test_stabilization_uses_fixed_arc_and_admap_route_points(self):
        bridge = self._bridge()
        bridge.config["lane_change_to_turn_reference_transition_arc_m"] = 10.0
        bridge.target_speed_mps = 5.0
        route_points = [[0.0, 0.0], [8.0, 0.0], [12.0, -4.0]]
        bridge.route_manager = types.SimpleNamespace(
            geometry_route_points=lambda **_kwargs: route_points
        )
        calls = []
        bridge.reference_generator.target_lane_stabilization_samples = (
            lambda **kwargs: calls.append(kwargs) or [
                {
                    "x_ref_m": 0.5 * float(index + 1),
                    "y_ref_m": 0.0,
                    "heading_rad": 0.0,
                    "lane_id": 2,
                }
                for index in range(kwargs["horizon_steps"])
            ]
        )

        reason = bridge._start_target_lane_stabilization(
            ego_location=bridge.carla.Location(x=0.0, y=0.0),
            ego_yaw_rad=0.0,
        )

        self.assertIn("transition_arc_m=10.00", reason)
        self.assertEqual(calls[0]["route_points"], route_points)
        self.assertGreaterEqual(calls[0]["horizon_steps"], 21)
        self.assertEqual(bridge.maneuver_manager.lane_change.transition_to_turn_arc_m, 10.0)





    def test_turn_master_reference_is_reused_across_handoff(self):
        bridge = self._bridge()
        local_map_calls = []

        def reference_from_local_map(snapshot, **kwargs):
            local_map_calls.append((snapshot, kwargs))
            return ([
                {
                    "x_ref_m": 0.35 * float(index + 1),
                    "y_ref_m": -0.01 * float(index) ** 2,
                    "heading_rad": 0.0,
                    "lane_id": 5960149,
                    "lane_width_m": 3.5,
                }
                for index in range(80)
            ], "master_source")

        bridge._local_map_snapshot = types.SimpleNamespace(valid=True)
        bridge._stable_reference_line_provider.reference_from_local_map = (
            reference_from_local_map
        )
        bridge.reference_generator.curvature_feasible_turn_samples = (
            lambda reference_samples, **_kwargs: (
                [dict(sample) for sample in reference_samples],
                "",
            )
        )

        first, _destination, first_reason = bridge._stable_reference_line_provider.turn_reference(TurnReferenceRequest(
            local_map=bridge._local_map_snapshot,
            config=bridge.config,
            horizon_steps=bridge.mpc.horizon_steps,
            dt_s=bridge.mpc.dt_s,
            ego_location=bridge.carla.Location(x=0.0, y=0.0),
            ego_yaw_rad=0.0,
            current_state=[0.0, 0.0, 3.0, 0.0],
            current_lane_id=500144,
            target_lane_id=5960149,
            target_speed_mps=5.0,
            destination_state=None,
            lock_master=True,
            turn_direction="right",
            route_revision="route-test",
        ))
        second, _destination, second_reason = bridge._stable_reference_line_provider.turn_reference(TurnReferenceRequest(
            local_map=bridge._local_map_snapshot,
            config=bridge.config,
            horizon_steps=bridge.mpc.horizon_steps,
            dt_s=bridge.mpc.dt_s,
            ego_location=bridge.carla.Location(x=1.0, y=0.0),
            ego_yaw_rad=0.0,
            current_state=[1.0, 0.0, 3.0, 0.0],
            current_lane_id=5960149,
            target_lane_id=5960149,
            target_speed_mps=5.0,
            destination_state=None,
            route_revision="route-test",
        ))

        self.assertEqual(len(local_map_calls), 1)
        self.assertIs(local_map_calls[0][0], bridge._local_map_snapshot)
        self.assertEqual(local_map_calls[0][1]["start_lane_id"], 500144)
        turn_snapshot = bridge._stable_reference_line_provider.snapshot(TURN)
        self.assertEqual(len(turn_snapshot.samples), 80)
        self.assertEqual(turn_snapshot.maneuver_direction, "right")
        self.assertEqual(len(first), bridge.mpc.horizon_steps)
        self.assertEqual(len(second), bridge.mpc.horizon_steps)
        self.assertIn("turn_master_locked", first_reason)
        self.assertIn("turn_master_window", second_reason)
        self.assertGreaterEqual(
            turn_snapshot.progress_s_m,
            0.0,
        )

    def test_candidate_reference_selection_requires_upcoming_turn_context(self):
        parameters = inspect.signature(
            CandidateReferenceBuildContext
        ).parameters

        self.assertIn("upcoming_turn_direction", parameters)
        self.assertIn("upcoming_turn_distance_m", parameters)

    def test_route_required_probe_budget_prioritizes_normal_then_assertive(self):
        bridge = self._bridge()
        evaluator = CandidateTrajectoryEvaluator(
            mpc_probe_enabled=True,
            mpc_probe_top_k=2,
            mpc_probe_interval_s=0.05,
        )
        probed = []
        bridge.mpc.active_cost_profile_name = "lane_follow"
        bridge.mpc.apply_mode_cost_profile = lambda *_args, **_kwargs: None
        bridge.mpc.probe_trajectory_feasibility = lambda **kwargs: (
            probed.append(kwargs["destination_state"][0])
            or {"solved": True, "status": "solved"}
        )

        def row(name, decision, variant, marker, cost):
            return types.SimpleNamespace(
                feasible=True,
                total_cost=float(cost),
                intent=types.SimpleNamespace(
                    name=name,
                    decision=decision,
                    target_lane_id=2 if decision.startswith("lane_change") else 1,
                    trajectory_variant=variant,
                    lane_change_duration_s=4.0,
                    stop_goal_active=False,
                ),
                destination_state=[marker],
                lane_center_reference=[],
                reference_debug={},
                feasibility_status="feasible",
                feasibility_reason="",
            )

        rows = [
            row("keep", "lane_follow", "", "keep", 0.0),
            row("assertive", "lane_change_right", "assertive", "assertive", 1.0),
            row("normal", "lane_change_right", "normal", "normal", 10.0),
            row("conservative", "lane_change_right", "conservative", "conservative", 5.0),
        ]
        evaluator.probe_mpc(
            candidate_results=rows,
            mpc=bridge.mpc,
            sim_time_s=1.0,
            current_state=[0.0, 0.0, 0.0, 0.0],
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            current_steering_rad=0.0,
            required_decision="lane_change_right",
            required_target_lane_id=2,
        )

        self.assertEqual(probed, ["normal", "assertive"])
        self.assertEqual(rows[0].feasibility_status, "mpc_probe_skipped")

    def test_target_lane_entry_replaces_quintic_with_stabilization_reference(self):
        bridge = self._bridge()
        bridge.maneuver_manager.lane_change.progress = 0.95
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ])
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
            bridge.maneuver_manager.lane_change.phase,
            "target_lane_stabilization",
        )
        self.assertTrue(
            all(
                row["lane_transition_kind"] == "target_lane_stabilization"
                for row in self._lane_change_reference(bridge)
            )
        )
        self.assertTrue(
            all(
                float(row["lane_change_progress"]) == 1.0
                for row in self._lane_change_reference(bridge)
            )
        )

    def test_longitudinal_lane_id_change_does_not_block_stabilization_entry(self):
        # AD-map may split one physical corridor into several longitudinal
        # lane IDs.  The continuous geometry contract owns the motion phase;
        # lane identity remains topology/diagnostic data only.
        bridge = self._bridge()
        bridge.maneuver_manager.lane_change.progress = 0.95
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ])
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
            current_lane_id=3,
            ego_location=bridge.carla.Location(x=5.0, y=3.2),
            ego_yaw_rad=0.02,
        )

        self.assertTrue(reason.startswith("target_lane_stabilization_started"))
        self.assertEqual(
            bridge.maneuver_manager.lane_change.phase,
            "target_lane_stabilization",
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
        bridge.maneuver_manager.lane_change.progress = 0.95
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ])
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
        self.assertEqual(
            bridge.maneuver_manager.lane_change.phase, "executing"
        )
        self.assertFalse(
            bridge.maneuver_manager.lane_change.completion_debug[
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
        bridge.maneuver_manager.lane_change.progress = 0.95
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ])
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
        self.assertEqual(
            bridge.maneuver_manager.lane_change.phase, "executing"
        )
        self.assertTrue(self._lane_change_reference(bridge))

    def test_heading_alignment_within_threshold_allows_stabilization_handoff(self):
        bridge = self._bridge()
        bridge.maneuver_manager.lane_change.progress = 0.95
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ])
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
            bridge.maneuver_manager.lane_change.phase,
            "target_lane_stabilization",
        )

    def test_target_lane_entry_never_continues_old_quintic_when_handoff_fails(self):
        bridge = self._bridge()
        bridge.maneuver_manager.lane_change.progress = 0.95
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.5,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(30)
        ])
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
        self.assertEqual(self._lane_change_reference(bridge), [])
        self.assertEqual(
            bridge.maneuver_manager.lane_change.completed_option,
            "CHANGELANERIGHT",
        )

    def test_commitment_releases_after_geometric_convergence(self):
        bridge = self._bridge()
        bridge.maneuver_manager.lane_change.progress = 0.96
        bridge.maneuver_manager.lane_change.phase = "target_lane_stabilization"
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(1, 11)
        ])

        reason = ""
        for _ in range(5):
            reason = bridge._release_completed_lane_change_commitment(
                current_lane_id=2,
                ego_location=bridge.carla.Location(x=5.0, y=0.1),
                ego_yaw_rad=0.02,
            )

        self.assertIn("lane_change_commitment_released", reason)
        self.assertEqual(self._lane_change_reference(bridge), [])
        self.assertEqual(
            bridge.maneuver_manager.lane_change.completed_option,
            "CHANGELANERIGHT",
        )

    def test_stabilization_release_uses_projected_arc_not_sample_index(self):
        bridge = self._bridge()
        bridge.maneuver_manager.lane_change.progress = 0.96
        bridge.maneuver_manager.lane_change.phase = "target_lane_stabilization"
        bridge.maneuver_manager.lane_change.transition_to_turn_arc_m = 10.0
        bridge.maneuver_manager.lane_change.transition_to_turn_step_m = 1.0
        # Resampling may legitimately produce a large point index even though
        # the ego has moved only two metres over the immutable reference.
        bridge.maneuver_manager.lane_change.progress_index = 50
        bridge.maneuver_manager.lane_change.progress_s_m = 2.0
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(1, 21)
        ])

        reason = ""
        for _ in range(8):
            reason = bridge._release_completed_lane_change_commitment(
                current_lane_id=2,
                ego_location=bridge.carla.Location(x=5.0, y=0.1),
                ego_yaw_rad=0.02,
            )

        self.assertEqual(reason, "")
        self.assertTrue(self._lane_change_reference(bridge))

    def test_post_turn_exit_locks_admap_topology_centerline(self):
        bridge = self._bridge()
        bridge.config["post_turn_exit_reference_arc_m"] = 12.0
        bridge._local_map_snapshot = build_local_map_snapshot(
            frame_id=1,
            timestamp_s=1.0,
            match={"valid": True, "ad_lane_id": 5960149},
            local_graph={
                "corridors": {0: [5960149, 540156]},
                "lane_to_offset": {5960149: 0, 540156: 0},
                "route_lane_sequence": [5960149, 540156],
                "lane_centerlines": {
                    5960149: [
                        {"x_m": float(index), "y_m": 0.0}
                        for index in range(16)
                    ],
                    540156: [
                        {"x_m": 15.0 + float(index), "y_m": 0.0}
                        for index in range(32)
                    ],
                },
            },
            route_target_lane_id=540156,
        )

        activated = bridge._stable_reference_line_provider.start_post_turn_exit(
            local_map=bridge._local_map_snapshot,
            ego_x_m=10.0, ego_y_m=0.0,
            current_lane_id=5960149,
            target_speed_mps=2.2,
            horizon_steps=bridge.mpc.horizon_steps, dt_s=bridge.mpc.dt_s,
            hold_arc_m=12.0, route_revision="route-1", map_epoch="town06",
        )
        window, reason = bridge._stable_reference_line_provider.post_turn_exit_window(
            ego_x_m=11.0, ego_y_m=0.0,
            target_speed_mps=2.2,
            horizon_steps=bridge.mpc.horizon_steps, dt_s=bridge.mpc.dt_s,
            first_forward_m=0.0,
        )

        self.assertTrue(activated)
        snapshot = bridge._stable_reference_line_provider.snapshot(POST_TURN)
        self.assertGreater(len(snapshot.samples), 20)
        self.assertGreater(snapshot.activation_s_m, 5.0)
        self.assertEqual(snapshot.activation_s_m, snapshot.progress_s_m - 1.0)
        self.assertEqual(len(window), bridge.mpc.horizon_steps)
        self.assertIn("post_turn_exit_locked_window", reason)
        self.assertIn("travel=1.00", reason)
        self.assertEqual(
            {
                row.get("reference_geometry_owner")
                for row in snapshot.samples
            },
            {"local_map_snapshot"},
        )
        successor_kinds = [
            row.get("lane_transition_kind")
            for row in snapshot.samples
            if int(row.get("lane_id", 0)) == 540156
        ]
        self.assertGreater(len(successor_kinds), 20)
        self.assertEqual(set(successor_kinds), {"longitudinal_successor"})

    def test_post_turn_exit_does_not_reshape_admap_master(self):
        bridge = self._bridge()
        bridge.config["post_turn_exit_reference_arc_m"] = 12.0
        connector = [
            {"x_m": 25.0 - float(index), "y_m": -23.0}
            for index in range(16)
        ] + [
            {"x_m": 10.0, "y_m": -24.0 - float(index)}
            for index in range(12)
        ]
        exit_lane = [
            {"x_m": 10.0, "y_m": -35.0 - float(index)}
            for index in range(64)
        ]
        bridge._local_map_snapshot = build_local_map_snapshot(
            frame_id=1,
            timestamp_s=1.0,
            match={"valid": True, "ad_lane_id": 5960149},
            local_graph={
                "corridors": {0: [5960149, 540156]},
                "lane_to_offset": {5960149: 0, 540156: 0},
                "route_lane_sequence": [5960149, 540156],
                "lane_centerlines": {
                    5960149: connector,
                    540156: exit_lane,
                },
            },
            route_target_lane_id=540156,
        )

        activated = bridge._stable_reference_line_provider.start_post_turn_exit(
            local_map=bridge._local_map_snapshot,
            ego_x_m=10.0, ego_y_m=-34.0,
            current_lane_id=5960149,
            target_speed_mps=2.2,
            horizon_steps=bridge.mpc.horizon_steps, dt_s=bridge.mpc.dt_s,
            hold_arc_m=12.0, route_revision="route-1", map_epoch="town06",
        )

        self.assertTrue(activated)
        outgoing = [
            row for row in bridge._stable_reference_line_provider.snapshot(
                POST_TURN
            ).samples
            if int(row.get("lane_id", 0)) == 540156
        ]
        self.assertGreater(len(outgoing), 20)
        self.assertEqual(
            bridge._stable_reference_line_provider.snapshot(
                POST_TURN
            ).target_lane_id,
            540156,
        )
        self.assertTrue(
            all(abs(float(row["x_ref_m"]) - 10.0) < 1.0e-9 for row in outgoing)
        )

    def test_post_turn_exit_rejects_point_count_without_real_arc_length(self):
        bridge = self._bridge()
        bridge.config["post_turn_exit_reference_arc_m"] = 12.0
        bridge.route_manager = types.SimpleNamespace(
            geometry_route_points=lambda **_kwargs: [[0.0, 0.0], [2.0, 0.0]]
        )
        bridge.reference_generator.lane_center_samples = (
            lambda **kwargs: [
                {
                    "x_ref_m": 0.02 * float(index + 1),
                    "y_ref_m": 0.0,
                    "heading_rad": 0.0,
                    "lane_id": 5960149,
                }
                for index in range(kwargs["horizon_steps"])
            ]
        )

        activated = bridge._stable_reference_line_provider.start_post_turn_exit(
            local_map=LocalMapSnapshot(),
            ego_x_m=0.0, ego_y_m=0.0,
            current_lane_id=5960149,
            target_speed_mps=2.2,
            horizon_steps=bridge.mpc.horizon_steps, dt_s=bridge.mpc.dt_s,
            hold_arc_m=12.0, route_revision="route-1", map_epoch="town06",
        )

        self.assertFalse(activated)
        self.assertFalse(
            bridge._stable_reference_line_provider.snapshot(POST_TURN).active
        )

    def test_post_turn_exit_exhaustion_clears_lock(self):
        bridge = self._bridge()
        bridge._stable_reference_line_provider.install(
            POST_TURN,
            [
            {
                "x_ref_m": 0.02 * float(index + 1),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
            }
            for index in range(bridge.mpc.horizon_steps)
            ],
            route_revision="route-1",
            map_epoch="town06",
            event="phase_transition",
        )

        window, reason = bridge._stable_reference_line_provider.post_turn_exit_window(
            ego_x_m=0.0, ego_y_m=0.0,
            target_speed_mps=2.2,
            horizon_steps=bridge.mpc.horizon_steps, dt_s=bridge.mpc.dt_s,
            first_forward_m=0.0,
        )

        self.assertEqual(window, [])
        self.assertIn("post_turn_exit_reference_exhausted", reason)
        self.assertFalse(
            bridge._stable_reference_line_provider.snapshot(POST_TURN).active
        )

    def test_lane_id_change_alone_does_not_release_commitment(self):
        bridge = self._bridge()
        bridge.maneuver_manager.lane_change.progress = 0.96
        bridge.maneuver_manager.lane_change.phase = "target_lane_stabilization"
        self._install_lane_change_reference(bridge, [
            {
                "x_ref_m": float(index),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_change_progress": 1.0,
            }
            for index in range(1, 11)
        ])

        reason = bridge._release_completed_lane_change_commitment(
            current_lane_id=2,
            ego_location=bridge.carla.Location(x=5.0, y=1.2),
            ego_yaw_rad=0.25,
        )

        self.assertEqual(reason, "")
        self.assertTrue(self._lane_change_reference(bridge))

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
        self._install_lane_change_reference(bridge, master)

        first, _ = bridge.maneuver_manager.locked_lane_change_window(
            provider=bridge._stable_reference_line_provider,
            ego_x_m=0.0,
            ego_y_m=0.0,
            ego_heading_rad=0.0,
            target_speed_mps=3.0,
            spacing_m=0.3,
            horizon_steps=bridge.mpc.horizon_steps,
        )
        second, _ = bridge.maneuver_manager.locked_lane_change_window(
            provider=bridge._stable_reference_line_provider,
            ego_x_m=2.1,
            ego_y_m=0.5,
            ego_heading_rad=0.0,
            target_speed_mps=3.0,
            spacing_m=0.3,
            horizon_steps=bridge.mpc.horizon_steps,
        )

        self.assertEqual(len(first), 20)
        self.assertEqual(len(second), 20)
        self.assertGreaterEqual(
            bridge.maneuver_manager.lane_change.progress_index,
            6,
        )
        self.assertGreater(
            float(second[0]["lane_change_progress"]),
            float(first[0]["lane_change_progress"]),
        )
        self.assertEqual(self._lane_change_reference(bridge), master)

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
        self._install_lane_change_reference(bridge, master)
        bridge.maneuver_manager.lane_change.progress_pairs = pairs

        # Ego is laterally only 30% of the way across (y=1.05 of a 3.5m gap),
        # even though the nearest station's *scheduled* tag already claims
        # 100% (index 19+ -> scheduled_progress=1.0).
        window, _ = bridge.maneuver_manager.locked_lane_change_window(
            provider=bridge._stable_reference_line_provider,
            ego_x_m=6.0,
            ego_y_m=1.05,
            ego_heading_rad=0.0,
            target_speed_mps=3.0,
            spacing_m=0.3,
            horizon_steps=bridge.mpc.horizon_steps,
        )

        self.assertTrue(window)
        self.assertAlmostEqual(
            bridge.maneuver_manager.lane_change.progress, 0.30, places=2
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
