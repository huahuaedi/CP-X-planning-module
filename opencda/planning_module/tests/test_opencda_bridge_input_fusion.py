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
    _Mode2ObjectTrackMemory,
    _Mode2ReferenceMemory,
    _Mode2TrafficLightMemory,
    _Mode2TrajectoryMemory,
)


class OpenCDABridgeInputFusionTests(unittest.TestCase):
    @staticmethod
    def _longitudinal_guard_bridge():
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "full_low_speed_launch_ramp_enabled": False,
            "lane_follow_negative_accel_release_min_front_gap_m": 12.0,
        }
        bridge.mpc = types.SimpleNamespace(
            constraints=types.SimpleNamespace(
                min_acceleration_mps2=-3.0,
                max_acceleration_mps2=2.0,
                max_steer_rad=0.5,
            )
        )
        bridge.min_front_gap_m = 8.0
        bridge.overspeed_guard_enabled = True
        bridge.overspeed_margin_mps = 0.5
        bridge.overspeed_release_margin_mps = 0.1
        bridge.overspeed_decel_gain = 0.8
        bridge.overspeed_max_decel_mps2 = 0.8
        bridge._overspeed_guard_active = False
        bridge.lane_follow_negative_accel_release_enabled = True
        bridge.lane_follow_negative_accel_release_error_mps = 0.05
        bridge.lane_follow_negative_accel_release_max_lateral_m = 0.5
        bridge.lane_follow_speed_recovery_enabled = True
        bridge.lane_follow_speed_recovery_enter_error_mps = 0.20
        bridge.lane_follow_speed_recovery_release_error_mps = 0.05
        bridge.lane_follow_speed_recovery_min_accel_mps2 = 0.80
        bridge.lane_follow_speed_recovery_max_lateral_m = 0.50
        bridge._lane_follow_speed_recovery_active = False
        bridge.full_low_speed_launch_enabled = False
        bridge.full_low_speed_launch_speed_mps = 0.35
        bridge.full_low_speed_launch_min_accel_mps2 = 0.8
        bridge.low_speed_lateral_recovery_enabled = False
        bridge._full_launch_start_s = None
        bridge._full_launch_start_xy = None
        bridge._control_from_mpc = lambda accel, steer: types.SimpleNamespace(
            throttle=max(0.0, float(accel)),
            brake=max(0.0, -float(accel)),
            steer=float(steer),
        )
        return bridge

    def test_overspeed_guard_uses_hysteresis_until_release_margin(self):
        bridge = self._longitudinal_guard_bridge()
        transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        _, accel, _, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(1.0, 0.0),
            accel_mps2=1.0,
            steer_rad=0.0,
            ego_transform=transform,
            ego_speed_mps=3.6,
            speed_ref_mps=3.0,
            destination_state=[8.0, 0.0, 3.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            traffic_signal_state="unknown",
            front_gap_m=None,
            sim_time_s=1.0,
        )
        self.assertEqual(reason, "overspeed_guard_hysteresis")
        self.assertLess(accel, 0.0)

        _, _, _, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(1.0, 0.0),
            accel_mps2=1.0,
            steer_rad=0.0,
            ego_transform=transform,
            ego_speed_mps=3.3,
            speed_ref_mps=3.0,
            destination_state=[8.0, 0.0, 3.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            traffic_signal_state="unknown",
            front_gap_m=None,
            sim_time_s=1.1,
        )
        self.assertEqual(reason, "overspeed_guard_hysteresis")

        _, accel, _, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(0.4, 0.0),
            accel_mps2=0.4,
            steer_rad=0.0,
            ego_transform=transform,
            ego_speed_mps=3.05,
            speed_ref_mps=3.0,
            destination_state=[8.0, 0.0, 3.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            traffic_signal_state="unknown",
            front_gap_m=None,
            sim_time_s=1.2,
        )
        self.assertEqual(reason, "")
        self.assertAlmostEqual(accel, 0.4)

    def test_red_light_stationary_stop_uses_direct_brake_hold(self):
        bridge = self._longitudinal_guard_bridge()
        bridge.config.update(
            {
                "full_stop_hold_speed_mps": 0.10,
                "full_stop_hold_brake": 0.60,
            }
        )
        transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        control, accel, steer, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(0.5, 0.2),
            accel_mps2=0.5,
            steer_rad=0.2,
            ego_transform=transform,
            ego_speed_mps=0.05,
            speed_ref_mps=0.0,
            destination_state=[1.5, 0.0, 0.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=True,
            behavior_decision="stop_at_intersection",
            behavior_fsm_state="IDLE",
            traffic_signal_state="red",
            front_gap_m=None,
            sim_time_s=1.0,
        )

        self.assertEqual(reason, "red_yellow_stop_stationary_hold")
        self.assertAlmostEqual(control.throttle, 0.0)
        self.assertAlmostEqual(control.brake, 0.60)
        self.assertAlmostEqual(control.steer, 0.0)
        self.assertAlmostEqual(accel, -1.8)
        self.assertAlmostEqual(steer, 0.0)

    def test_lane_follow_releases_unexplained_brake_below_target(self):
        bridge = self._longitudinal_guard_bridge()
        transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        _, accel, _, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(-0.35, 0.0),
            accel_mps2=-0.35,
            steer_rad=0.0,
            ego_transform=transform,
            ego_speed_mps=2.9,
            speed_ref_mps=3.0,
            destination_state=[8.0, 0.0, 3.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            traffic_signal_state="unknown",
            front_gap_m=18.0,
            sim_time_s=1.0,
        )

        self.assertEqual(reason, "lane_follow_negative_accel_release")
        self.assertAlmostEqual(accel, 0.0)

    def test_lane_follow_keeps_braking_when_front_gap_is_close(self):
        bridge = self._longitudinal_guard_bridge()
        transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        _, accel, _, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(-0.35, 0.0),
            accel_mps2=-0.35,
            steer_rad=0.0,
            ego_transform=transform,
            ego_speed_mps=2.9,
            speed_ref_mps=3.0,
            destination_state=[8.0, 0.0, 3.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            traffic_signal_state="unknown",
            front_gap_m=8.0,
            sim_time_s=1.0,
        )

        self.assertEqual(reason, "")
        self.assertAlmostEqual(accel, -0.35)

    def test_lane_follow_speed_recovery_holds_minimum_acceleration(self):
        bridge = self._longitudinal_guard_bridge()
        transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        _, accel, _, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(0.1, 0.0),
            accel_mps2=0.1,
            steer_rad=0.0,
            ego_transform=transform,
            ego_speed_mps=2.7,
            speed_ref_mps=3.0,
            destination_state=[8.0, 0.0, 3.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            traffic_signal_state="unknown",
            front_gap_m=18.0,
            sim_time_s=1.0,
        )

        self.assertEqual(reason, "lane_follow_speed_recovery")
        self.assertAlmostEqual(accel, 0.8)

        _, accel, _, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(0.2, 0.0),
            accel_mps2=0.2,
            steer_rad=0.0,
            ego_transform=transform,
            ego_speed_mps=2.96,
            speed_ref_mps=3.0,
            destination_state=[8.0, 0.0, 3.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            traffic_signal_state="unknown",
            front_gap_m=18.0,
            sim_time_s=1.1,
        )

        self.assertEqual(reason, "")
        self.assertAlmostEqual(accel, 0.2)

    def test_lane_follow_speed_recovery_does_not_override_close_gap(self):
        bridge = self._longitudinal_guard_bridge()
        transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        _, accel, _, reason = bridge._apply_control_safety_guards(
            control=bridge._control_from_mpc(0.1, 0.0),
            accel_mps2=0.1,
            steer_rad=0.0,
            ego_transform=transform,
            ego_speed_mps=2.5,
            speed_ref_mps=3.0,
            destination_state=[8.0, 0.0, 3.0, 0.0, 1],
            destination_lateral_m=0.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="IDLE",
            traffic_signal_state="green",
            front_gap_m=8.0,
            sim_time_s=1.0,
        )

        self.assertEqual(reason, "")
        self.assertAlmostEqual(accel, 0.1)

    def test_road_boundary_metrics_uses_vehicle_footprint(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {}
        bridge._metrics_boundary_sample_count = 0
        bridge._metrics_boundary_breach_count = 0
        waypoint = types.SimpleNamespace(
            transform=types.SimpleNamespace(
                location=types.SimpleNamespace(x=0.0, y=0.0),
                rotation=types.SimpleNamespace(yaw=0.0),
            ),
            lane_width=3.5,
        )
        bridge.map_planner = types.SimpleNamespace(
            get_waypoint=lambda _location: waypoint
        )
        vehicle = types.SimpleNamespace(
            bounding_box=types.SimpleNamespace(
                extent=types.SimpleNamespace(y=1.0)
            )
        )
        bridge.vehicle_manager = types.SimpleNamespace(vehicle=vehicle)

        inside = bridge._road_boundary_metrics(
            types.SimpleNamespace(x=0.0, y=0.5)
        )
        outside = bridge._road_boundary_metrics(
            types.SimpleNamespace(x=0.0, y=1.0)
        )

        self.assertTrue(inside["road_boundary_sample_valid"])
        self.assertFalse(inside["road_boundary_breach"])
        self.assertAlmostEqual(inside["road_boundary_clearance_m"], 0.25)
        self.assertTrue(outside["road_boundary_breach"])
        self.assertEqual(bridge._metrics_boundary_sample_count, 2)
        self.assertEqual(bridge._metrics_boundary_breach_count, 1)

    def test_strict_reference_veto_hard_gates_explicit_fallback(self):
        reason = CPXMPCPlannerBridge._candidate_hard_gate_reason(
            reference_debug={
                "candidate_pipeline_selected_status": "explicit_fallback",
                "candidate_pipeline_selected": "explicit_fallback_keep_lane",
                "candidate_pipeline_selected_reason": "all_candidates_infeasible",
                "mpc_reference_stabilizer_reason": (
                    "contract_violation:first_lateral_out_of_contract;"
                    "strict_reference_veto"
                ),
            },
            behavior_decision="lane_follow",
            stop_goal_active=False,
        )

        self.assertIn("candidate_hard_gate", reason)
        self.assertIn("strict_reference_veto", reason)

    def test_emergency_brake_always_uses_direct_control_hard_gate(self):
        reason = CPXMPCPlannerBridge._candidate_hard_gate_reason(
            reference_debug={
                "candidate_pipeline_selected_status": "explicit_fallback",
                "candidate_pipeline_selected": "explicit_fallback_emergency_stop",
                "candidate_pipeline_selected_reason": "all_candidates_infeasible",
            },
            behavior_decision="emergency_brake",
            stop_goal_active=True,
        )

        self.assertIn("emergency_brake_direct_control", reason)

    def test_turn_explicit_fallback_hard_stops_on_prediction_collision_veto(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {}
        bridge.mpc = types.SimpleNamespace(dt_s=0.1, horizon_steps=3)
        bridge._turn_latch_decision = ""
        bridge._turn_latch_until_sim_time_s = 0.0
        bridge._sim_time_s = lambda: 1.0
        route_reference = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.8},
            {"x_ref_m": 2.0, "y_ref_m": 0.1, "lane_id": 1, "speed_ref_mps": 0.8},
            {"x_ref_m": 3.0, "y_ref_m": 0.3, "lane_id": 1, "speed_ref_mps": 0.8},
        ]
        bridge._carla_waypoint_turn_reference = lambda **_kwargs: (
            route_reference,
            [3.0, 0.3, 0.8, 0.0, 1],
            "carla_grp_waypoint_chain_smoothed",
        )
        bridge._validate_candidate_reference_contract = lambda **_kwargs: types.SimpleNamespace(
            valid=True,
            reason=lambda: "",
        )
        bridge._build_ego_heading_emergency_stop_reference = lambda **_kwargs: (
            [
                {"x_ref_m": 0.5, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.0},
                {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.0},
                {"x_ref_m": 1.5, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.0},
            ],
            [0.5, 0.0, 0.0, 0.0, 1],
        )

        decision, _, speed_mps, _, _, debug = bridge._explicit_fallback_candidate_for_mpc(
            candidate_results=[
                types.SimpleNamespace(
                    feasibility_reason="candidate_prediction_collision_risk:0.80"
                )
            ],
            baseline_decision="intersection_turn_left",
            baseline_target_lane_id=1,
            current_lane_id=1,
            current_state=[0.0, 0.0, 1.0, 0.0],
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_yaw_rad=0.0,
            summarize_candidate_results=lambda _rows: "collision",
        )

        self.assertEqual(decision, "emergency_brake")
        self.assertEqual(speed_mps, 0.0)
        self.assertEqual(debug["reference_source"], "explicit_fallback_ego_heading_stop")
        self.assertIn("candidate_collision_risk_veto", debug["carla_turn_reference_reason"])

    def test_turn_stabilizer_does_not_treat_normal_curve_spacing_as_duplicate(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {}
        bridge.target_speed_mps = 3.0
        bridge.mpc = types.SimpleNamespace(dt_s=0.1, horizon_steps=4)
        bridge.full_reference_stabilizer_min_forward_m = -0.25
        bridge.full_reference_stabilizer_min_spacing_m = 0.35
        bridge.full_reference_stabilizer_max_heading_step_rad = 0.75
        bridge.full_lane_follow_max_destination_lateral_m = 1.2
        bridge.full_lane_follow_max_reference_first_lateral_m = 0.65
        bridge.full_stop_max_destination_lateral_m = 1.0
        bridge.full_stop_max_reference_first_lateral_m = 0.55
        bridge.strict_reference_validator_veto_enabled = True
        reference = [
            {"x_ref_m": 0.34, "y_ref_m": 0.00, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 0.68, "y_ref_m": 0.01, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 1.02, "y_ref_m": 0.03, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 1.36, "y_ref_m": 0.06, "lane_id": 1, "speed_ref_mps": 1.0},
        ]

        destination, stabilized, reason = bridge._stabilize_mpc_reference_input(
            destination_state=[1.36, 0.06, 1.0, 0.0, 1],
            lane_center_reference=reference,
            current_state=[0.0, 0.0, 1.0, 0.0],
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_yaw_rad=0.0,
            ego_speed_mps=1.0,
            speed_ref_mps=1.0,
            stop_goal_active=False,
            behavior_decision="intersection_turn_left",
            behavior_fsm_state="INTERSECTION_TURN_LEFT",
            current_lane_id=1,
            stop_target=None,
        )

        self.assertEqual(len(stabilized), 4)
        self.assertNotIn("drop_duplicate_sample", reason)
        self.assertNotIn("creep_turn_reference", reason)
        self.assertAlmostEqual(float(destination[0]), 1.36)

    def test_strict_lane_follow_rebuilds_curved_reference_before_veto(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {}
        bridge.target_speed_mps = 3.0
        bridge.mpc = types.SimpleNamespace(dt_s=0.1, horizon_steps=3)
        bridge.full_reference_stabilizer_min_forward_m = -0.25
        bridge.full_reference_stabilizer_min_spacing_m = 0.35
        bridge.full_reference_stabilizer_max_heading_step_rad = 0.75
        bridge.full_lane_follow_max_destination_lateral_m = 1.2
        bridge.full_lane_follow_max_reference_first_lateral_m = 0.65
        bridge.full_stop_max_destination_lateral_m = 1.0
        bridge.full_stop_max_reference_first_lateral_m = 0.55
        bridge.strict_reference_validator_veto_enabled = True
        bridge._map_waypoint_from_location = lambda _location: object()
        bridge._active_global_route_points = lambda: []
        bridge._current_lane_center_reference_samples = lambda **_kwargs: [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            {"x_ref_m": 3.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
        ]

        destination, reference, reason = bridge._stabilize_mpc_reference_input(
            destination_state=[2.0, 2.0, 2.0, 0.0, 1],
            lane_center_reference=[
                {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
                {"x_ref_m": 1.1, "y_ref_m": 1.0, "lane_id": 1, "speed_ref_mps": 2.0},
                {"x_ref_m": 2.0, "y_ref_m": 1.1, "lane_id": 1, "speed_ref_mps": 2.0},
            ],
            current_state=[0.0, 0.0, 1.0, 0.0],
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_yaw_rad=0.0,
            ego_speed_mps=1.0,
            speed_ref_mps=3.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="LANE_KEEP",
            current_lane_id=1,
            stop_target=None,
        )

        self.assertIn("curvature_out_of_contract", reason)
        self.assertIn("rebuilt_current_lane_reference_after_contract_veto", reason)
        self.assertNotIn("strict_reference_veto", reason)
        self.assertEqual([float(sample["y_ref_m"]) for sample in reference], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(float(destination[1]), 0.0)

    def test_fuses_local_and_cp_obstacles_with_source_priority_and_ttl(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.max_mpc_obstacles = 10

        fused = bridge._fused_planning_object_snapshots(
            local_object_snapshots=[
                {
                    "vehicle_id": "42",
                    "x": 10.0,
                    "y": 0.0,
                    "v": 3.0,
                    "psi": 0.0,
                    "length_m": 4.5,
                    "width_m": 2.0,
                    "source": "opencda_perception",
                    "provider_source": "native_opencda_perception",
                }
            ],
            cp_obstacles=[
                {
                    "id": "native_opencda_v2x:42",
                    "state": [12.0, 0.0, 7.0, 0.0],
                    "provider_source": "native_opencda_v2x",
                    "source": "opencda_v2x",
                    "timestamp_s": 10.0,
                    "ttl_s": 1.0,
                    "shape": {"length_m": 4.8, "width_m": 2.1},
                },
                {
                    "id": "native_opencda_v2x:99",
                    "state": [20.0, 1.0, 5.0, 0.1],
                    "provider_source": "native_opencda_v2x",
                    "source": "opencda_v2x",
                    "timestamp_s": 10.0,
                    "ttl_s": 1.0,
                    "shape": {"length_m": 4.8, "width_m": 2.1},
                },
                {
                    "id": "native_opencda_v2x:100",
                    "state": [30.0, 1.0, 5.0, 0.1],
                    "provider_source": "native_opencda_v2x",
                    "source": "opencda_v2x",
                    "timestamp_s": 1.0,
                    "ttl_s": 1.0,
                },
            ],
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            sim_time_s=10.5,
        )

        by_id = {item["vehicle_id"]: item for item in fused}

        self.assertEqual(set(by_id), {"42", "99"})
        self.assertEqual(by_id["42"]["source"], "opencda_perception")
        self.assertAlmostEqual(by_id["42"]["x"], 10.0)
        self.assertEqual(by_id["99"]["source"], "opencda_v2x")

        bridge.max_mpc_obstacles = 1
        limited = bridge._limit_obstacles_for_mpc(
            object_snapshots=fused,
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(len(limited), 1)

    def test_selects_nearest_fresh_forward_same_lane_control(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        selected = bridge._select_relevant_traffic_control(
            traffic_controls=[
                {
                    "id": "behind",
                    "state": "red",
                    "timestamp_s": 1.0,
                    "ttl_s": 5.0,
                    "stop_line": {"x_m": -5.0, "y_m": 0.0, "lane_id": 1, "road_id": 7},
                },
                {
                    "id": "side_lane",
                    "state": "red",
                    "timestamp_s": 1.0,
                    "ttl_s": 5.0,
                    "stop_line": {"x_m": 8.0, "y_m": 3.5, "lane_id": 2, "road_id": 7},
                },
                {
                    "id": "same_lane",
                    "state": "red",
                    "timestamp_s": 1.0,
                    "ttl_s": 5.0,
                    "stop_line": {"x_m": 12.0, "y_m": 0.1, "lane_id": 1, "road_id": 7},
                },
            ],
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_heading_rad=0.0,
            current_lane_id=1,
            current_road_id=7,
            sim_time_s=2.0,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], "same_lane")

    def test_skips_duplicate_native_perception_cp_obstacles_by_position(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.max_mpc_obstacles = 10

        fused = bridge._fused_planning_object_snapshots(
            local_object_snapshots=[
                {
                    "vehicle_id": "101",
                    "x": 15.0,
                    "y": -2.0,
                    "v": 2.5,
                    "psi": 0.0,
                    "source": "opencda_perception",
                    "provider_source": "native_opencda_perception",
                }
            ],
            cp_obstacles=[
                {
                    "id": "native_opencda_perception:perception:0",
                    "state": [15.2, -2.1, 2.5, 0.0],
                    "source": "opencda_perception",
                    "provider_source": "native_opencda_perception",
                    "timestamp_s": 3.0,
                    "ttl_s": 1.0,
                },
                {
                    "id": "native_opencda_v2x:202",
                    "state": [30.0, -2.0, 4.0, 0.0],
                    "source": "opencda_v2x",
                    "provider_source": "native_opencda_v2x",
                    "timestamp_s": 3.0,
                    "ttl_s": 1.0,
                },
            ],
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            sim_time_s=3.5,
        )

        by_source = [(item["vehicle_id"], item["provider_source"]) for item in fused]

        self.assertEqual(len(fused), 2)
        self.assertIn(("101", "native_opencda_perception"), by_source)
        self.assertIn(("202", "native_opencda_v2x"), by_source)

    def test_mode2_reference_cleanup_sorts_by_forward_distance(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "mode2_sort_reference_by_forward": True,
            "mode2_min_reference_forward_spacing_m": 0.25,
        }
        ego_transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        cleaned = bridge._clean_mode2_reference_raw_points(
            raw_points=[
                (5.0, 0.0, 3.0, 1, 3.5),
                (2.0, 0.0, 3.0, 1, 3.5),
                (3.0, 0.0, 3.0, 1, 3.5),
                (-1.0, 0.0, 3.0, 1, 3.5),
                (2.05, 0.0, 3.0, 1, 3.5),
            ],
            ego_transform=ego_transform,
        )

        self.assertEqual([round(item[0], 2) for item in cleaned], [2.0, 3.0, 5.0])

    def test_global_route_summary_uses_planning_module_global_planner(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.vehicle_manager = types.SimpleNamespace(vehicle=types.SimpleNamespace(id=7))
        bridge.global_planner = types.SimpleNamespace(
            get_current_route_info=lambda **_: types.SimpleNamespace(
                route_found=True,
                optimal_lane_id=2,
                current_road_option="LEFT",
                next_macro_maneuver="Left Turn",
                debug_reason="planning_module_global_route",
            )
        )

        summary = bridge._planning_module_global_route_summary(
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_heading_rad=0.0,
            fallback_lane_id=1,
        )

        self.assertTrue(summary["route_found"])
        self.assertEqual(summary["optimal_lane_id"], 2)
        self.assertEqual(summary["current_road_option"], "LEFT")
        self.assertEqual(summary["next_macro_maneuver"], "Left Turn")

    def test_mode2_stop_reference_prefers_cp_stop_line_over_moving_lane_samples(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "mode2_stop_reference_max_lateral_m": 1.5,
            "mode2_default_stop_reference_m": 4.0,
        }
        bridge.mpc = types.SimpleNamespace(horizon_steps=20)

        samples = bridge._mode2_stop_reference_samples(
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_yaw_rad=0.0,
            lane_center_reference=[
                {"x_ref_m": 5.0, "y_ref_m": 0.0, "lane_id": 1, "lane_width_m": 3.5},
                {"x_ref_m": 6.0, "y_ref_m": 0.0, "lane_id": 1, "lane_width_m": 3.5},
            ],
            target_loc=None,
            stop_target={"x_m": 2.0, "y_m": 0.0, "lane_id": 1},
        )

        self.assertGreaterEqual(len(samples), 2)
        self.assertAlmostEqual(samples[-1]["x_ref_m"], 2.0)
        self.assertAlmostEqual(samples[-1]["y_ref_m"], 0.0)

    def test_mode2_stop_reference_uses_approach_speed_before_final_taper(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "mode2_stop_reference_max_lateral_m": 1.5,
            "mode2_default_stop_reference_m": 4.0,
        }
        bridge.mpc = types.SimpleNamespace(horizon_steps=20)

        samples = bridge._mode2_stop_reference_samples(
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_yaw_rad=0.0,
            lane_center_reference=[],
            target_loc=None,
            stop_target={"x_m": 10.0, "y_m": 0.0, "lane_id": 1},
            approach_speed_mps=2.0,
        )

        self.assertGreater(samples[1]["v_ref_mps"], 1.5)
        self.assertAlmostEqual(samples[-1]["v_ref_mps"], 0.0)

    def test_mode2_far_stop_does_not_force_early_pid(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "mode2_stop_approach_distance_m": 8.0,
            "mode2_stop_mpc_max_speed_mps": 1.2,
            "mode2_stop_mpc_max_lateral_m": 2.0,
        }

        should_use_pid = bridge._mode2_should_use_pid_for_stop(
            ego_speed_mps=3.0,
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_yaw_rad=0.0,
            destination_state=[20.0, 0.0, 0.0, 0.0, 0],
        )

        self.assertFalse(should_use_pid)

    def test_mode2_traffic_memory_holds_red_through_unknown(self):
        memory = _Mode2TrafficLightMemory(hold_unknown_s=0.8)

        state, stop_target, reason = memory.update(
            state="red",
            stop_target={"x_m": 10.0, "y_m": 0.0},
            sim_time_s=1.0,
        )
        self.assertEqual(state, "red")
        self.assertEqual(reason, "raw_stop")

        state, stop_target, reason = memory.update(
            state="unknown",
            stop_target=None,
            sim_time_s=1.4,
        )

        self.assertEqual(state, "red")
        self.assertEqual(stop_target["x_m"], 10.0)
        self.assertEqual(reason, "traffic_memory_hold_red")

    def test_traffic_memory_uses_asymmetric_unknown_hold_and_green_confirm(self):
        memory = _Mode2TrafficLightMemory(
            hold_unknown_s=1.5,
            hold_green_unknown_s=0.25,
            green_confirm_s=0.5,
        )
        memory.update(
            state="red",
            stop_target={"x_m": 10.0, "y_m": 0.0},
            sim_time_s=1.0,
        )

        state, stop_target, reason = memory.update(
            state="unknown",
            stop_target=None,
            sim_time_s=2.2,
        )
        self.assertEqual(state, "red")
        self.assertEqual(stop_target["x_m"], 10.0)
        self.assertEqual(reason, "traffic_memory_hold_red")

        state, _, reason = memory.update(
            state="green",
            stop_target=None,
            sim_time_s=2.3,
        )
        self.assertEqual(state, "red")
        self.assertEqual(reason, "traffic_memory_wait_green_confirm")

        state, _, reason = memory.update(
            state="green",
            stop_target=None,
            sim_time_s=2.9,
        )
        self.assertEqual(state, "green")
        self.assertEqual(reason, "traffic_memory_green_release")

        state, _, reason = memory.update(
            state="unknown",
            stop_target=None,
            sim_time_s=3.0,
        )
        self.assertEqual(state, "green")
        self.assertEqual(reason, "traffic_memory_hold_green")

        state, _, reason = memory.update(
            state="unknown",
            stop_target=None,
            sim_time_s=3.2,
        )
        self.assertEqual(state, "unknown")
        self.assertEqual(reason, "")

    def test_full_mode_green_confirmation_releases_after_short_debounce(self):
        memory = _Mode2TrafficLightMemory(
            hold_unknown_s=1.5,
            hold_green_unknown_s=0.25,
            green_confirm_s=0.15,
        )
        stop_target = {"x_m": 10.0, "y_m": 0.0}
        memory.update(state="red", stop_target=stop_target, sim_time_s=1.0)

        state, target, reason = memory.update(
            state="green",
            stop_target=None,
            sim_time_s=2.0,
        )
        self.assertEqual(state, "red")
        self.assertEqual(target, stop_target)
        self.assertEqual(reason, "traffic_memory_wait_green_confirm")

        state, target, reason = memory.update(
            state="green",
            stop_target=None,
            sim_time_s=2.16,
        )
        self.assertEqual(state, "green")
        self.assertIsNone(target)
        self.assertEqual(reason, "traffic_memory_green_release")

    def test_full_mode_holds_red_through_long_unknown_until_green(self):
        memory = _Mode2TrafficLightMemory(
            hold_unknown_s=1.5,
            hold_green_unknown_s=0.25,
            green_confirm_s=0.15,
            hold_stop_unknown_until_green=True,
        )
        stop_target = {"x_m": 10.0, "y_m": 0.0}
        memory.update(state="red", stop_target=stop_target, sim_time_s=1.0)

        state, target, reason = memory.update(
            state="unknown",
            stop_target=None,
            sim_time_s=5.0,
        )

        self.assertEqual(state, "red")
        self.assertEqual(target, stop_target)
        self.assertEqual(
            reason,
            "traffic_memory_fail_safe_hold_red_until_green",
        )

    def test_full_mode_resolves_unknown_from_latched_carla_signal_actor(self):
        signal_actor = types.SimpleNamespace(
            get_state=lambda: types.SimpleNamespace(name="Green")
        )
        world = types.SimpleNamespace(
            get_actor=lambda actor_id: signal_actor if actor_id == 42 else None
        )
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.vehicle_manager = types.SimpleNamespace(
            vehicle=types.SimpleNamespace(get_world=lambda: world)
        )
        bridge._full_signal_actor_id = ""
        bridge._full_latched_stop_target = {"x_m": 10.0, "y_m": 0.0}
        bridge._full_latched_stop_state = "red"

        state, reason = bridge._resolve_full_traffic_state_from_carla_actor(
            raw_state="red",
            signal_context={"signal_actor_id": "42"},
        )
        self.assertEqual(state, "red")
        self.assertEqual(reason, "")
        self.assertEqual(bridge._full_signal_actor_id, "42")

        state, reason = bridge._resolve_full_traffic_state_from_carla_actor(
            raw_state="unknown",
            signal_context={},
        )
        self.assertEqual(state, "green")
        self.assertEqual(reason, "latched_carla_signal_actor:42:green")

    def test_full_mode_keeps_unknown_when_latched_signal_actor_is_missing(self):
        world = types.SimpleNamespace(get_actor=lambda _actor_id: None)
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.vehicle_manager = types.SimpleNamespace(
            vehicle=types.SimpleNamespace(get_world=lambda: world)
        )
        bridge._full_signal_actor_id = "42"
        bridge._full_latched_stop_target = {"x_m": 10.0, "y_m": 0.0}
        bridge._full_latched_stop_state = "red"

        state, reason = bridge._resolve_full_traffic_state_from_carla_actor(
            raw_state="unknown",
            signal_context={},
        )
        self.assertEqual(state, "unknown")
        self.assertEqual(reason, "latched_carla_signal_actor_missing:42")

    def test_mode2_object_memory_smooths_and_temporarily_holds_tracks(self):
        memory = _Mode2ObjectTrackMemory(alpha=0.5, max_stale_s=0.4)

        tracks, reason = memory.update(
            object_snapshots=[{"vehicle_id": "7", "x": 0.0, "y": 0.0, "v": 2.0, "psi": 0.0}],
            sim_time_s=1.0,
        )
        self.assertEqual(len(tracks), 1)
        self.assertIn("object_memory_tracks=1", reason)

        tracks, reason = memory.update(
            object_snapshots=[{"vehicle_id": "7", "x": 2.0, "y": 0.0, "v": 4.0, "psi": 0.0}],
            sim_time_s=1.1,
        )
        self.assertAlmostEqual(tracks[0]["x"], 1.0)
        self.assertAlmostEqual(tracks[0]["v"], 3.0)
        self.assertTrue(tracks[0]["object_memory_fresh"])

        tracks, reason = memory.update(object_snapshots=[], sim_time_s=1.3)
        self.assertEqual(len(tracks), 1)
        self.assertFalse(tracks[0]["object_memory_fresh"])

    def test_mode2_reference_memory_reuses_previous_on_large_jump(self):
        memory = _Mode2ReferenceMemory(
            max_first_point_jump_m=1.0,
            max_destination_jump_m=2.0,
            max_reuse_age_s=1.0,
        )
        ego = sys.modules["carla"].Location(0.0, 0.0, 0.0)
        first_reference = [
            {"x_ref_m": 2.0, "y_ref_m": 0.0},
            {"x_ref_m": 3.0, "y_ref_m": 0.0},
        ]
        accepted, destination, reason = memory.stabilize(
            reference=first_reference,
            destination_state=[3.0, 0.0, 3.0, 0.0, 0],
            ego_location=ego,
            ego_yaw_rad=0.0,
            stop_goal_active=False,
            sim_time_s=1.0,
        )
        self.assertEqual(reason, "reference_memory_accept")

        accepted, destination, reason = memory.stabilize(
            reference=[
                {"x_ref_m": 20.0, "y_ref_m": 0.0},
                {"x_ref_m": 21.0, "y_ref_m": 0.0},
            ],
            destination_state=[21.0, 0.0, 3.0, 0.0, 0],
            ego_location=ego,
            ego_yaw_rad=0.0,
            stop_goal_active=False,
            sim_time_s=1.2,
        )

        self.assertTrue(reason.startswith("reference_memory_reuse_jump"))
        self.assertEqual([sample["x_ref_m"] for sample in accepted], [2.0, 3.0])
        self.assertEqual(destination[0], 3.0)

    def test_mode2_trajectory_memory_blends_large_control_jump(self):
        memory = _Mode2TrajectoryMemory(
            max_accel_jump_mps2=1.0,
            max_steer_jump_rad=0.1,
            blend_alpha=0.5,
            max_reuse_age_s=0.5,
        )

        def control_factory(accel, steer):
            return types.SimpleNamespace(accel=accel, steer=steer)

        control, accel, steer, reason = memory.accept_or_blend(
            control=control_factory(0.0, 0.0),
            accel_mps2=0.0,
            steer_rad=0.0,
            control_factory=control_factory,
            sim_time_s=1.0,
        )
        self.assertEqual(reason, "trajectory_memory_accept")

        control, accel, steer, reason = memory.accept_or_blend(
            control=control_factory(3.0, 0.4),
            accel_mps2=3.0,
            steer_rad=0.4,
            control_factory=control_factory,
            sim_time_s=1.1,
        )

        self.assertTrue(reason.startswith("trajectory_memory_blend"))
        self.assertAlmostEqual(accel, 1.5)
        self.assertAlmostEqual(steer, 0.2)


if __name__ == "__main__":
    unittest.main()
