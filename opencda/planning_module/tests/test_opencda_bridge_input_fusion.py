import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


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
    _lane_change_execution_active,
    _select_static_obstacle_local_avoidance_lane,
    _should_suspend_mpc_for_normal_stop,
    _static_obstacle_cooldown_policy,
)
from opencda_bridge.cp_provider import OpenCDACPProvider
from pipeline.traffic_light_memory import TrafficLightMemory
from pipeline.reference_gate import FinalReferenceGate
from pipeline.reference_generator import GeneratedReference, ReferenceGenerator
from pipeline.reference_pipeline import ReferencePipeline, ReferencePipelineRequest
from pipeline.route_manager import RouteReplanResult


class OpenCDABridgeInputFusionTests(unittest.TestCase):
    def test_functional_isolation_removes_only_non_ego_vehicle_actors_once(self):
        ego = SimpleNamespace(id=10)
        stale_a = Mock(id=20)
        stale_b = Mock(id=30)

        class _Actors(list):
            def filter(self, pattern):
                self.pattern = pattern
                return self

        actors = _Actors([ego, stale_a, stale_b])
        ego.get_world = Mock(return_value=SimpleNamespace(get_actors=Mock(return_value=actors)))
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.functional_test_ignore_dynamic_objects = True
        bridge._functional_test_world_actors_cleaned = False
        bridge.vehicle_manager = SimpleNamespace(vehicle=ego)
        bridge.debug = False

        bridge._clean_functional_test_dynamic_actors_once()
        bridge._clean_functional_test_dynamic_actors_once()

        stale_a.destroy.assert_called_once_with()
        stale_b.destroy.assert_called_once_with()
        self.assertFalse(hasattr(ego, "destroy"))
        self.assertTrue(bridge._functional_test_world_actors_cleaned)

    def test_missed_lane_change_replan_is_suppressed_during_locked_execution(self):
        self.assertTrue(_lane_change_execution_active(
            reference_locked=True,
            phase="executing",
        ))

    def test_missed_lane_change_replan_is_suppressed_during_stabilization(self):
        self.assertTrue(_lane_change_execution_active(
            reference_locked=False,
            phase="target_lane_stabilization",
        ))

    def test_missed_lane_change_replan_resumes_after_release(self):
        self.assertFalse(_lane_change_execution_active(
            reference_locked=False,
            phase="idle",
        ))

    def test_static_obstacle_local_avoidance_selects_safest_adjacent_lane(self):
        selected = _select_static_obstacle_local_avoidance_lane(
            current_lane_id=2,
            available_lane_ids=[1, 2, 3],
            lane_safety_scores={1: 0.72, 2: 0.1, 3: 0.91},
            lane_prediction_risks={1: {"risk": False}, 3: {"risk": False}},
            minimum_safety_score=0.55,
        )

        self.assertEqual(selected, 3)

    def test_static_obstacle_local_avoidance_rejects_prediction_risk(self):
        selected = _select_static_obstacle_local_avoidance_lane(
            current_lane_id=1,
            available_lane_ids=[1, 2],
            lane_safety_scores={1: 0.1, 2: 0.95},
            lane_prediction_risks={2: {"risk": True}},
            minimum_safety_score=0.55,
        )

        self.assertIsNone(selected)

    def test_new_static_obstacle_encounter_holds_stop_during_cooldown(self):
        status, stop_active = _static_obstacle_cooldown_policy(
            failed_latched=False,
            route_transition_pending=False,
        )

        self.assertEqual(status, "cooldown_stop")
        self.assertTrue(stop_active)

    def test_continuous_observation_after_success_can_enter_replanned_route(self):
        status, stop_active = _static_obstacle_cooldown_policy(
            failed_latched=False,
            route_transition_pending=True,
        )

        self.assertEqual(status, "cooldown_route_transition")
        self.assertFalse(stop_active)

    def test_nearest_front_obstacle_by_lane_keeps_speed_and_identity(self):
        nearest = CPXMPCPlannerBridge._nearest_front_obstacle_by_lane(
            ego_snapshot={"x": 0.0, "y": 0.0, "psi": 0.0},
            obstacle_snapshots=[
                {"track_id": "far", "x": 12.0, "y": 0.0, "v": 0.0},
                {"track_id": "near", "x": 5.0, "y": 0.0, "v": 0.2},
                {"track_id": "rear", "x": -2.0, "y": 0.0, "v": 0.0},
            ],
            lane_assignments={"far": 1, "near": 1, "rear": 1},
            available_lane_ids=[1],
        )

        self.assertEqual(nearest[1]["vehicle_id"], "near")
        self.assertAlmostEqual(float(nearest[1]["front_distance_m"]), 5.0)
        self.assertAlmostEqual(float(nearest[1]["v"]), 0.2)

    def test_static_obstacle_route_replan_blocks_lane_and_resets_route_state(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {"static_obstacle_replan_cooldown_s": 2.0}
        bridge.behavior_runtime_cfg = {}
        bridge._static_obstacle_replan_last_attempt_s = -float("inf")
        bridge._static_obstacle_replan_reason = "not_requested"
        bridge._static_obstacle_blocked_lane_id = ""
        bridge._sim_time_s = lambda: 10.0
        bridge.global_planner = Mock()
        bridge.global_planner.block_lane_at_position.return_value = 17
        bridge.route_manager = Mock()
        bridge.route_manager.replan_from.return_value = RouteReplanResult(
            True, "carla_grp_route_replanned:static_obstacle", 12
        )
        bridge.route_manager.active_route_summary = {"route": "new"}
        bridge._temporary_destination_state = [1.0, 2.0]
        bridge._previous_lane_center_reference = [{"x": 1.0}]
        bridge._lane_reference_freeze_count = 4
        bridge._last_required_lane_change_target_lane_id = 2
        bridge._reset_route_tracking_lane_change_reference = Mock()
        bridge._lane_id_tracker = Mock()
        bridge.maneuver_manager = Mock()
        bridge.control_buffer = Mock()
        bridge.mpc = Mock()

        attempted, succeeded, reason = bridge._attempt_static_obstacle_route_replan(
            ego_location=SimpleNamespace(x=3.0, y=4.0, z=0.0),
            obstacle={"x": 8.0, "y": 4.0, "z": 0.0},
        )

        self.assertTrue(attempted)
        self.assertTrue(succeeded)
        self.assertIn("replanned", reason)
        bridge.global_planner.block_lane_at_position.assert_called_once()
        bridge.route_manager.replan_from.assert_called_once()
        self.assertEqual(bridge._active_route_summary, {"route": "new"})
        self.assertEqual(bridge._static_obstacle_blocked_lane_id, 17)
        bridge.maneuver_manager.reset.assert_called_once_with(
            reason="static_obstacle_route_replanned"
        )
        bridge.control_buffer.reset.assert_called_once_with(
            reason="static_obstacle_route_replanned"
        )

    def test_turn_route_replan_refreshes_route_owned_state(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {"turn_route_replan_cooldown_s": 2.0}
        bridge._route_replan_last_attempt_s = -float("inf")
        bridge._route_replan_attempt_count = 0
        bridge._route_replan_last_reason = ""
        bridge._sim_time_s = lambda: 10.0
        bridge.route_manager = Mock()
        bridge.route_manager.replan_from.return_value = RouteReplanResult(
            True, "carla_grp_route_replanned:test", 12
        )
        bridge.route_manager.active_route_summary = {"route": "new"}
        bridge._temporary_destination_state = [1.0, 2.0]
        bridge._previous_lane_center_reference = [{"x": 1.0}]
        bridge._lane_reference_freeze_count = 4
        bridge._reset_route_tracking_lane_change_reference = Mock()
        bridge.maneuver_manager = Mock()
        bridge.control_buffer = Mock()

        attempted, succeeded, reason = bridge._attempt_turn_route_replan(
            ego_location=SimpleNamespace(x=3.0, y=4.0, z=0.0)
        )

        self.assertTrue(attempted)
        self.assertTrue(succeeded)
        self.assertIn("replanned", reason)
        self.assertEqual(bridge._active_route_summary, {"route": "new"})
        self.assertIsNone(bridge._temporary_destination_state)
        self.assertEqual(bridge._previous_lane_center_reference, [])
        bridge.maneuver_manager.reset.assert_called_once_with(
            reason="turn_route_replanned"
        )
        bridge.control_buffer.reset.assert_called_once_with(
            reason="turn_route_replanned"
        )

    def test_turn_route_replan_respects_cooldown(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {"turn_route_replan_cooldown_s": 2.0}
        bridge._route_replan_last_attempt_s = 9.0
        bridge._route_replan_attempt_count = 1
        bridge._route_replan_last_reason = ""
        bridge._sim_time_s = lambda: 10.0
        bridge.route_manager = Mock()

        attempted, succeeded, reason = bridge._attempt_turn_route_replan(
            ego_location=SimpleNamespace(x=3.0, y=4.0, z=0.0)
        )

        self.assertFalse(attempted)
        self.assertFalse(succeeded)
        self.assertIn("route_replan_cooldown", reason)
        bridge.route_manager.replan_from.assert_not_called()

    def test_normal_stop_suspends_mpc_inside_low_speed_capture_region(self):
        self.assertTrue(_should_suspend_mpc_for_normal_stop(
            candidate_hard_gate_active=False,
            stop_goal_active=True,
            behavior_decision="stop_at_intersection",
            ego_speed_mps=0.25,
            suspend_speed_mps=0.30,
        ))

    def test_normal_stop_keeps_mpc_above_capture_speed(self):
        self.assertFalse(_should_suspend_mpc_for_normal_stop(
            candidate_hard_gate_active=False,
            stop_goal_active=True,
            behavior_decision="stop_sign",
            ego_speed_mps=0.31,
            suspend_speed_mps=0.30,
        ))

    def test_emergency_brake_never_uses_normal_stop_hold(self):
        self.assertFalse(_should_suspend_mpc_for_normal_stop(
            candidate_hard_gate_active=False,
            stop_goal_active=True,
            behavior_decision="emergency_brake",
            ego_speed_mps=0.0,
            suspend_speed_mps=0.30,
        ))

    def test_uncommitted_stop_does_not_suspend_mpc(self):
        self.assertFalse(_should_suspend_mpc_for_normal_stop(
            candidate_hard_gate_active=False,
            stop_goal_active=False,
            behavior_decision="stop_at_intersection",
            ego_speed_mps=0.0,
            suspend_speed_mps=0.30,
        ))

    def test_cp_normalization_preserves_cooperative_provenance(self):
        snapshot = CPXMPCPlannerBridge._normalize_cp_obstacle_snapshot({
            "id": "native_opencda_multi_vantage:42",
            "type": "pedestrian",
            "state": [5.0, 1.0, 1.2, 0.0],
            "observed_by_cav_ids": ["10", "11"],
            "not_observed_by_cav_ids": ["9"],
            "blind_spot_shared": True,
        })

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["vehicle_id"], "42")
        self.assertEqual(snapshot["object_type"], "pedestrian")
        self.assertEqual(snapshot["observed_by_cav_ids"], ["10", "11"])
        self.assertEqual(snapshot["not_observed_by_cav_ids"], ["9"])
        self.assertTrue(snapshot["blind_spot_shared"])

    def test_cooperative_actor_evidence_tracks_prediction_and_candidate_relevance(self):
        evidence = CPXMPCPlannerBridge._cooperative_actor_evidence(
            cp_summary={
                "actor_provenance": [{
                    "actor_id": "native_opencda_multi_vantage:42",
                    "actor_type": "pedestrian",
                    "observer_cav_ids": ["11"],
                    "not_observed_by_cav_ids": ["9"],
                    "visible_to_ego": False,
                    "visible_to_auxiliary": True,
                    "blind_spot_shared": True,
                    "distance_to_ego_m": 12.0,
                }]
            },
            prediction_trajectories={
                "42": [
                    {"x": 5.0, "y": 0.5},
                    {"x": 6.0, "y": 0.5},
                ]
            },
            selected_reference=[
                {"x_ref_m": 5.0, "y_ref_m": 0.0},
                {"x_ref_m": 6.0, "y_ref_m": 0.0},
            ],
        )

        self.assertEqual(evidence["cp_pedestrian_count"], 1)
        self.assertEqual(evidence["cp_blind_spot_pedestrian_count"], 1)
        self.assertIn(":42", evidence["cp_prediction_used_pedestrian_ids"])
        self.assertIn(":42", evidence["cp_candidate_relevant_pedestrian_ids"])
        self.assertIn('"used_by_prediction": true', evidence["cp_actor_evidence"])
        self.assertIn('"candidate_relevant": true', evidence["cp_actor_evidence"])

    @staticmethod
    def _attach_reference_generator(bridge):
        bridge.reference_generator = ReferenceGenerator(
            config=bridge.config,
            mpc=bridge.mpc,
            map_planner=None,
            map_waypoint_from_location=lambda location: (
                bridge._map_waypoint_from_location(location)
                if hasattr(bridge, "_map_waypoint_from_location")
                else None
            ),
            lane_id_at_location=lambda _location: 1,
            body_frame_xy=bridge._body_frame_xy,
            target_speed_mps=float(getattr(bridge, "target_speed_mps", 3.0)),
            lookahead_m=18.0,
        )
        bridge.final_reference_gate = FinalReferenceGate(bridge.config)
        bridge.reference_pipeline = ReferencePipeline(
            config=bridge.config,
            generator=bridge.reference_generator,
            final_gate=bridge.final_reference_gate,
            horizon_steps=int(bridge.mpc.horizon_steps),
            dt_s=float(bridge.mpc.dt_s),
            default_speed_mps=float(getattr(bridge, "target_speed_mps", 3.0)),
        )
        bridge._active_global_route_points = lambda: []
        return bridge.reference_generator

    def test_set_destination_does_not_require_removed_full_mode_memories(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.route_manager = Mock()
        bridge.route_manager.set_destination.return_value = {"route": "ok"}
        bridge._lane_id_tracker = Mock()
        bridge.control_buffer = Mock()
        bridge._temporary_destination_state = [1.0, 2.0]
        bridge._previous_lane_center_reference = [{"x": 1.0}]
        bridge._lane_reference_freeze_count = 4
        bridge._turn_latch_decision = "intersection_turn_left"
        bridge._turn_latch_until_sim_time_s = 10.0

        bridge.set_destination(
            start_location={"x": 0.0, "y": 0.0, "z": 0.0},
            end_location={"x": 10.0, "y": 0.0, "z": 0.0},
        )

        bridge._lane_id_tracker.reset.assert_called_once_with()
        bridge.control_buffer.reset.assert_called_once_with(
            reason="destination_updated"
        )
        self.assertEqual(bridge._turn_latch_decision, "")

    def test_collect_object_snapshots_accepts_external_mapping_detections(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        snapshots = bridge._collect_object_snapshots(
            detected_objects={
                "vehicles": [
                    {
                        "id": "mdrive_detection:7",
                        "x": 12.0,
                        "y": -3.0,
                        "v": 2.5,
                        "psi": 0.25,
                        "length_m": 4.2,
                        "width_m": 1.9,
                        "confidence": 0.8,
                        "source": "mdrive_cooperative_perception",
                    }
                ]
            }
        )

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["vehicle_id"], "mdrive_detection:7")
        self.assertAlmostEqual(snapshots[0]["x"], 12.0)
        self.assertAlmostEqual(snapshots[0]["confidence"], 0.8)

    def test_collect_object_snapshots_accepts_ml_fusion_vehicle_without_transform(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        fused_vehicle = types.SimpleNamespace(
            carla_id=-1,
            location=types.SimpleNamespace(x=18.0, y=2.5, z=0.0),
            velocity=types.SimpleNamespace(x=3.0, y=4.0, z=0.0),
            bounding_box=types.SimpleNamespace(
                extent=types.SimpleNamespace(x=2.1, y=0.95, z=0.8)
            ),
            get_transform=lambda: None,
            get_location=lambda: types.SimpleNamespace(
                x=18.0, y=2.5, z=0.0),
            get_velocity=lambda: types.SimpleNamespace(
                x=3.0, y=4.0, z=0.0),
        )

        snapshots = bridge._collect_object_snapshots(
            detected_objects={"vehicles": [fused_vehicle]}
        )

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(
            snapshots[0]["vehicle_id"], "opencda_detection:0")
        self.assertEqual(
            snapshots[0]["source"], "opencda_ml_lidar_fusion")
        self.assertAlmostEqual(snapshots[0]["x"], 18.0)
        self.assertAlmostEqual(snapshots[0]["y"], 2.5)
        self.assertAlmostEqual(snapshots[0]["v"], 5.0)
        self.assertAlmostEqual(snapshots[0]["psi"], 0.927295218, places=6)
        self.assertAlmostEqual(snapshots[0]["length_m"], 4.2)
        self.assertAlmostEqual(snapshots[0]["width_m"], 1.9)

    def test_low_speed_lane_follow_forces_closed_loop_replan(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.full_control_buffer_min_speed_mps = 1.5

        self.assertTrue(
            bridge._low_speed_control_buffer_force_replan(
                ego_speed_mps=0.2,
                behavior_decision="lane_follow",
                behavior_fsm_state="IDLE",
                stop_goal_active=False,
            )
        )
        self.assertFalse(
            bridge._low_speed_control_buffer_force_replan(
                ego_speed_mps=2.0,
                behavior_decision="lane_follow",
                behavior_fsm_state="IDLE",
                stop_goal_active=False,
            )
        )
        self.assertFalse(
            bridge._low_speed_control_buffer_force_replan(
                ego_speed_mps=0.2,
                behavior_decision="intersection_turn_left",
                behavior_fsm_state="IDLE",
                stop_goal_active=False,
            )
        )

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
        bridge.mpc = types.SimpleNamespace(dt_s=0.1, horizon_steps=20)
        bridge.target_speed_mps = 3.0
        self._attach_reference_generator(bridge)
        vehicle = types.SimpleNamespace(
            bounding_box=types.SimpleNamespace(
                extent=types.SimpleNamespace(x=2.4, y=1.0)
            )
        )
        bridge.vehicle_manager = types.SimpleNamespace(vehicle=vehicle)

        inside = bridge._road_boundary_metrics(
            types.SimpleNamespace(x=0.0, y=0.5),
            ego_yaw_rad=0.0,
        )
        outside = bridge._road_boundary_metrics(
            types.SimpleNamespace(x=0.0, y=1.0),
            ego_yaw_rad=0.0,
        )

        self.assertTrue(inside["road_boundary_sample_valid"])
        self.assertFalse(inside["road_boundary_breach"])
        self.assertAlmostEqual(inside["road_boundary_clearance_m"], 0.10)
        self.assertTrue(outside["road_boundary_breach"])
        self.assertEqual(bridge._metrics_boundary_sample_count, 2)
        self.assertEqual(bridge._metrics_boundary_breach_count, 1)

    def test_road_boundary_projection_limits_rolling_reference_heading_jump(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "road_boundary_projection_max_heading_step_rad": 0.04,
        }
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
        bridge.mpc = types.SimpleNamespace(dt_s=0.1, horizon_steps=20)
        bridge.target_speed_mps = 3.0
        self._attach_reference_generator(bridge)
        bridge.vehicle_manager = types.SimpleNamespace(
            vehicle=types.SimpleNamespace(
                bounding_box=types.SimpleNamespace(
                    extent=types.SimpleNamespace(x=2.4, y=1.0)
                )
            )
        )

        def reference(start_x, heading):
            return [
                {
                    "x_ref_m": start_x + index,
                    "y_ref_m": 0.0,
                    "corridor_center_x_m": start_x + index,
                    "corridor_center_y_m": 0.0,
                    "corridor_heading_rad": heading,
                    "lane_width_m": 3.5,
                }
                for index in range(3)
            ]

        bridge._road_boundary_metrics(
            types.SimpleNamespace(x=0.9, y=0.0),
            ego_yaw_rad=0.0,
            reference_samples=reference(0.0, 0.0),
        )
        jumped = bridge._road_boundary_metrics(
            types.SimpleNamespace(x=1.01, y=0.0),
            ego_yaw_rad=0.0,
            reference_samples=reference(1.0, 0.11),
        )

        self.assertAlmostEqual(
            jumped["road_boundary_projection_raw_heading_rad"],
            0.11,
        )
        self.assertAlmostEqual(
            jumped["road_boundary_projection_conditioned_heading_rad"],
            0.04,
        )
        self.assertTrue(
            jumped["road_boundary_projection_continuity_limited"]
        )

    def test_boundary_feedback_requires_persistence_then_latches(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "boundary_recovery_trigger_clearance_m": -0.10,
            "boundary_recovery_trigger_frames": 3,
            "boundary_recovery_release_clearance_m": 0.10,
        }
        bridge._boundary_recovery_trigger_frames = 0
        bridge._reset_boundary_recovery_request()
        snapshot = {
            "road_boundary_sample_valid": True,
            "road_boundary_clearance_m": -0.2,
            "road_boundary_lateral_offset_m": 0.3,
            "road_boundary_heading_error_rad": -0.2,
        }

        for index in range(2):
            bridge._update_boundary_recovery_request(
                boundary_snapshot=snapshot,
                behavior_decision="intersection_turn_right",
                sim_time_s=float(index),
            )
            self.assertFalse(bridge._boundary_recovery_request.active)
        bridge._update_boundary_recovery_request(
            boundary_snapshot=snapshot,
            behavior_decision="intersection_turn_right",
            sim_time_s=2.0,
        )

        self.assertTrue(bridge._boundary_recovery_request.active)
        self.assertEqual(
            bridge._boundary_recovery_request.turn_direction,
            "right",
        )

        bridge._update_boundary_recovery_request(
            boundary_snapshot={
                **snapshot,
                "road_boundary_clearance_m": 0.2,
                "road_boundary_lateral_offset_m": 0.1,
                "road_boundary_heading_error_rad": 0.04,
            },
            behavior_decision="intersection_turn_right",
            sim_time_s=3.0,
        )
        self.assertFalse(bridge._boundary_recovery_request.active)

    def test_soft_margin_does_not_latch_recovery_inside_drivable_union(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "boundary_recovery_trigger_clearance_m": -0.10,
            "boundary_recovery_trigger_frames": 3,
        }
        bridge._boundary_recovery_trigger_frames = 0
        bridge._boundary_recovery_infeasible_frames = 0
        bridge._boundary_recovery_cooldown_until_s = -float("inf")
        bridge._reset_boundary_recovery_request()
        snapshot = {
            "road_boundary_sample_valid": True,
            "road_boundary_clearance_m": -0.11,
            "road_boundary_lateral_offset_m": 0.4,
            "road_boundary_heading_error_rad": -0.2,
            "road_boundary_geometry_source": (
                "drivable_footprint:carla_driving_lane_union"
            ),
            "road_boundary_drivable_inside": True,
        }

        for index in range(10):
            bridge._update_boundary_recovery_request(
                boundary_snapshot=snapshot,
                behavior_decision="intersection_turn_right",
                sim_time_s=0.05 * float(index),
            )

        self.assertFalse(bridge._boundary_recovery_request.active)
        self.assertEqual(bridge._boundary_recovery_trigger_frames, 0)

    def test_infeasible_boundary_recovery_enters_cooldown(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "boundary_recovery_max_infeasible_frames": 3,
            "boundary_recovery_cooldown_s": 2.0,
        }
        bridge._boundary_recovery_trigger_frames = 0
        bridge._boundary_recovery_infeasible_frames = 0
        bridge._boundary_recovery_cooldown_until_s = -float("inf")
        bridge._reset_boundary_recovery_request()
        snapshot = {
            "road_boundary_sample_valid": True,
            "road_boundary_clearance_m": -0.3,
            "road_boundary_lateral_offset_m": 0.4,
            "road_boundary_heading_error_rad": -0.2,
        }

        for index in range(3):
            bridge._update_boundary_recovery_request(
                boundary_snapshot=snapshot,
                behavior_decision="intersection_turn_right",
                sim_time_s=10.0 + 0.05 * float(index),
                recovery_planned=True,
                recovery_reference_feasible=False,
            )

        self.assertFalse(bridge._boundary_recovery_request.active)
        self.assertGreater(
            bridge._boundary_recovery_cooldown_until_s,
            12.0,
        )
        bridge._update_boundary_recovery_request(
            boundary_snapshot=snapshot,
            behavior_decision="intersection_turn_right",
            sim_time_s=11.0,
        )
        self.assertFalse(bridge._boundary_recovery_request.active)

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

    def test_geometry_hard_gate_does_not_request_emergency_stop(self):
        from opencda.planning_module.opencda_bridge.cpx_mpc_planner import (
            _hard_gate_requires_emergency_stop,
        )

        self.assertFalse(_hard_gate_requires_emergency_stop(
            fallback_reason=(
                "candidate_hard_gate:lane_change_left:"
                "final_reference_gate:destination_lane_error_out_of_contract"
            ),
            behavior_decision="lane_change_left",
            stop_goal_active=False,
        ))

    def test_collision_hard_gate_still_requests_emergency_stop(self):
        from opencda.planning_module.opencda_bridge.cpx_mpc_planner import (
            _hard_gate_requires_emergency_stop,
        )

        self.assertTrue(_hard_gate_requires_emergency_stop(
            fallback_reason=(
                "candidate_hard_gate:lane_change_left:"
                "candidate_prediction_collision_risk"
            ),
            behavior_decision="lane_change_left",
            stop_goal_active=False,
        ))

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
        bridge._waypoint_turn_reference = lambda **_kwargs: (
            route_reference,
            [3.0, 0.3, 0.8, 0.0, 1],
            "carla_grp_waypoint_chain_smoothed",
        )
        bridge._validate_candidate_reference_contract = lambda **_kwargs: types.SimpleNamespace(
            valid=True,
            reason=lambda: "",
        )
        bridge.reference_generator = types.SimpleNamespace(
            _build_ego_heading_emergency_stop_reference=lambda **_kwargs: (
                [
                    {"x_ref_m": 0.5, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.0},
                    {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.0},
                    {"x_ref_m": 1.5, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 0.0},
                ],
                [0.5, 0.0, 0.0, 0.0, 1],
            ),
            emergency_stop_reference=lambda **_kwargs: GeneratedReference(
                samples=[
                    {
                        "x_ref_m": 0.5,
                        "y_ref_m": 0.0,
                        "lane_id": 1,
                        "speed_ref_mps": 0.0,
                    }
                ],
                destination_state=[0.5, 0.0, 0.0, 0.0, 1],
                source="ego_heading_emergency_stop",
            ),
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
        self.assertIn("candidate_collision_risk_veto", debug["route_turn_reference_reason"])

    def test_turn_exit_contract_miss_keeps_retained_turn_instead_of_stopping(self):
        from pipeline.maneuver_manager import ManeuverManager

        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {}
        bridge.mpc = types.SimpleNamespace(dt_s=0.1, horizon_steps=3)
        bridge._scenario_manager = types.SimpleNamespace(
            state="TURN_EXIT_STABILIZATION",
            turn_speed_cap_mps=5.0,
        )
        bridge.maneuver_manager = ManeuverManager()
        bridge.maneuver_manager.update(
            reference_samples=[
                {"x_ref_m": 1.0, "y_ref_m": 0.0, "heading_rad": 0.0, "speed_ref_mps": 5.0},
                {"x_ref_m": 2.0, "y_ref_m": 0.1, "heading_rad": 0.1, "speed_ref_mps": 5.0},
                {"x_ref_m": 3.0, "y_ref_m": 0.3, "heading_rad": 0.2, "speed_ref_mps": 5.0},
            ],
            destination_state=[3.0, 0.3, 5.0, 0.2, 1],
            decision="intersection_turn_left",
            behavior_fsm_state="INTERSECTION_TURN_LEFT",
            current_lane_id=1,
            target_lane_id=1,
            ego_x_m=0.0,
            ego_y_m=0.0,
            reference_source="accepted_turn",
            route_current_option="LEFT",
        )

        decision, _, speed_mps, reference, _, debug = (
            bridge._explicit_fallback_candidate_for_mpc(
                candidate_results=[
                    types.SimpleNamespace(
                        feasibility_reason=(
                            "turn_swept_footprint:outside_drivable_union:"
                            "min_clearance=-0.150"
                        )
                    )
                ],
                baseline_decision="intersection_turn_left",
                baseline_target_lane_id=1,
                current_lane_id=1,
                current_state=[0.0, 0.0, 4.0, 0.0],
                ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
                ego_yaw_rad=0.0,
                summarize_candidate_results=lambda _rows: "contract miss",
                selection_reason="no_active_maneuver_commitment",
            )
        )

        self.assertEqual(decision, "intersection_turn_left")
        self.assertEqual(speed_mps, 5.0)
        self.assertTrue(reference)
        self.assertEqual(
            debug["candidate_pipeline_selected"],
            "retained_turn_exit_continuation",
        )
        self.assertEqual(
            debug["reference_source"],
            "unified_maneuver_turn_exit_continuation",
        )

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
        self._attach_reference_generator(bridge)
        reference = [
            {"x_ref_m": 0.34, "y_ref_m": 0.00, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 0.68, "y_ref_m": 0.01, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 1.02, "y_ref_m": 0.03, "lane_id": 1, "speed_ref_mps": 1.0},
            {"x_ref_m": 1.36, "y_ref_m": 0.06, "lane_id": 1, "speed_ref_mps": 1.0},
        ]

        conditioned = bridge.reference_pipeline.condition(
            ReferencePipelineRequest(
                destination_state=[1.36, 0.06, 1.0, 0.0, 1],
                reference_samples=reference,
                current_state=[0.0, 0.0, 1.0, 0.0],
                ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
                ego_yaw_rad=0.0,
                ego_speed_mps=1.0,
                target_speed_mps=1.0,
                stop_goal_active=False,
                behavior_decision="intersection_turn_left",
                behavior_fsm_state="INTERSECTION_TURN_LEFT",
                current_lane_id=1,
                target_lane_id=1,
                stop_target=None,
            )
        )
        destination = conditioned.destination_state
        stabilized = conditioned.reference_samples
        reason = conditioned.reason

        self.assertEqual(len(stabilized), 4)
        self.assertNotIn("drop_duplicate_sample", reason)
        self.assertNotIn("creep_turn_reference", reason)
        self.assertAlmostEqual(float(destination[0]), 1.36)

    def test_lane_follow_curvature_is_conditioned_before_contract_veto(self):
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
        self._attach_reference_generator(bridge)
        bridge.reference_generator._ego_anchored_lane_recovery_reference_samples = (
            lambda **_kwargs: [
                {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
                {"x_ref_m": 2.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
                {"x_ref_m": 3.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
            ]
        )

        conditioned = bridge.reference_pipeline.condition(
            ReferencePipelineRequest(
                destination_state=[2.0, 2.0, 2.0, 0.0, 1],
                reference_samples=[
                    {"x_ref_m": 1.0, "y_ref_m": 0.0, "lane_id": 1, "speed_ref_mps": 2.0},
                    {"x_ref_m": 1.1, "y_ref_m": 1.0, "lane_id": 1, "speed_ref_mps": 2.0},
                    {"x_ref_m": 2.0, "y_ref_m": 1.1, "lane_id": 1, "speed_ref_mps": 2.0},
                ],
                current_state=[0.0, 0.0, 1.0, 0.0],
                ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
                ego_yaw_rad=0.0,
                ego_speed_mps=1.0,
                target_speed_mps=3.0,
                stop_goal_active=False,
                behavior_decision="lane_follow",
                behavior_fsm_state="LANE_KEEP",
                current_lane_id=1,
                target_lane_id=1,
                stop_target=None,
            )
        )
        destination = conditioned.destination_state
        reference = conditioned.reference_samples
        reason = conditioned.reason

        self.assertIn("curvature_feasible_lane_follow", reason)
        self.assertTrue(conditioned.validation.valid, conditioned.validation.reason())
        self.assertLessEqual(conditioned.validation.max_curvature_1pm, 0.351)
        self.assertNotIn("strict_reference_veto", reason)
        self.assertEqual(len(reference), 3)
        self.assertAlmostEqual(
            float(destination[1]),
            float(reference[-1]["y_ref_m"]),
        )

    def test_final_reference_pipeline_holds_lane_recovery_across_short_valid_gap(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {
            "reference_pipeline_lane_follow_recovery_release_frames": 3,
        }
        bridge.target_speed_mps = 3.0
        bridge.mpc = types.SimpleNamespace(dt_s=0.1, horizon_steps=3)
        bridge._map_waypoint_from_location = lambda _location: object()
        self._attach_reference_generator(bridge)
        recovery = [
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1},
            {"x_ref_m": 2.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1},
            {"x_ref_m": 3.0, "y_ref_m": 0.0, "heading_rad": 0.0, "lane_id": 1},
        ]
        bridge.reference_generator._ego_anchored_lane_recovery_reference_samples = (
            lambda **_kwargs: [dict(row) for row in recovery]
        )
        base = dict(
            current_state=[0.0, 0.0, 1.0, 0.0],
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_yaw_rad=0.0,
            ego_speed_mps=1.0,
            target_speed_mps=2.0,
            stop_goal_active=False,
            behavior_decision="lane_follow",
            behavior_fsm_state="LANE_KEEP",
            current_lane_id=1,
            target_lane_id=1,
        )

        first = bridge.reference_pipeline.finalize(ReferencePipelineRequest(
            destination_state=[3.0, 2.0, 2.0, 0.0, 1],
            reference_samples=recovery,
            **base,
        ))
        second = bridge.reference_pipeline.finalize(ReferencePipelineRequest(
            destination_state=[3.0, 0.0, 2.0, 0.0, 1],
            reference_samples=recovery,
            **base,
        ))

        self.assertIn("lane_recovery_reference", first.conditioning_reason)
        self.assertIn(
            "lane_recovery_reference_hysteresis",
            second.conditioning_reason,
        )

    def test_candidate_contract_allows_lane_center_recovery_body_lateral(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {}
        bridge.target_speed_mps = 3.0
        bridge.mpc = types.SimpleNamespace(horizon_steps=4)
        reference = [
            {
                "x_ref_m": 0.8,
                "y_ref_m": 0.0,
                "lane_id": 1,
                "lane_transition_kind": "ego_anchored_lane_recovery",
            },
            {
                "x_ref_m": 2.0,
                "y_ref_m": 0.5,
                "lane_id": 1,
                "lane_transition_kind": "ego_anchored_lane_recovery",
            },
            {
                "x_ref_m": 3.5,
                "y_ref_m": 1.2,
                "lane_id": 1,
                "lane_transition_kind": "ego_anchored_lane_recovery",
            },
            {
                "x_ref_m": 5.0,
                "y_ref_m": 2.0,
                "lane_id": 1,
                "lane_transition_kind": "ego_anchored_lane_recovery",
            },
        ]

        result = bridge._validate_candidate_reference_contract(
            decision="lane_follow",
            lc_state="LANE_KEEP",
            current_lane_id=1,
            speed_ref_mps=3.0,
            stop_goal_active=False,
            current_state=[0.0, 0.0, 1.0, 0.0],
            destination_state=[5.0, 2.0, 3.0, 0.0, 1],
            lane_center_reference=reference,
        )

        self.assertTrue(result.valid, result.reason())
        self.assertGreater(result.destination_body_lateral_m, 1.5)

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

    def test_admap_target_preserves_opaque_lane_identity(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.vehicle_manager = types.SimpleNamespace(vehicle=types.SimpleNamespace(id=7))
        bridge.global_planner_backend = "custom_admap_dijkstra"
        carla_right = types.SimpleNamespace(
            road_id=1,
            section_id=0,
            lane_id=-2,
            get_left_lane=lambda: None,
            get_right_lane=lambda: None,
        )
        carla_current = types.SimpleNamespace(
            road_id=1,
            section_id=0,
            lane_id=-1,
            get_left_lane=lambda: None,
            get_right_lane=lambda: carla_right,
            next=lambda _distance: [],
        )
        carla_right.get_left_lane = lambda: carla_current
        bridge.global_planner = types.SimpleNamespace(
            get_local_lane_graph=lambda *_args, **_kwargs: {
                "ego_ad_lane_id": 500145,
                "lane_to_offset": {500145: 0, 500144: -1},
            },
            get_current_route_info=lambda **_: types.SimpleNamespace(
                route_found=True,
                optimal_lane_id=500144,
                current_road_option="LANEFOLLOW",
                next_macro_maneuver="Lane Change Right",
                next_macro_distance_m=30.0,
                distance_to_destination_m=100.0,
            ),
        )
        bridge.route_manager = types.SimpleNamespace(
            accept_authoritative_route_summary=Mock()
        )

        summary = bridge._planning_module_global_route_summary(
            ego_location=sys.modules["carla"].Location(0.0, 0.0, 0.0),
            ego_heading_rad=0.0,
            fallback_lane_id=2,
            ego_waypoint=carla_current,
        )

        self.assertEqual(summary["optimal_lane_id"], 500144)
        self.assertEqual(summary["ad_current_lane_id"], 500145)
        self.assertEqual(summary["ad_target_lane_id"], 500144)
        self.assertEqual(summary["lane_change_offset"], -1)
        bridge.route_manager.accept_authoritative_route_summary.assert_called_once()
        _, kwargs = bridge.route_manager.accept_authoritative_route_summary.call_args
        self.assertEqual(kwargs["debug_reason"], "admap_authoritative_route_active")

    def test_traffic_memory_holds_red_through_unknown(self):
        memory = TrafficLightMemory(hold_unknown_s=0.8)

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
        memory = TrafficLightMemory(
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
        memory = TrafficLightMemory(
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
        memory = TrafficLightMemory(
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

    def test_cp_visibility_filter_rejects_hit_before_target(self):
        provider = OpenCDACPProvider.__new__(OpenCDACPProvider)
        provider.visibility_filter_enabled = True
        provider.visibility_sensor_height_m = 1.6
        provider.visibility_target_tolerance_m = 0.5
        location = sys.modules["carla"].Location
        target = types.SimpleNamespace(
            bounding_box=types.SimpleNamespace(
                extent=types.SimpleNamespace(x=0.4, y=0.4, z=0.9)
            )
        )
        world = types.SimpleNamespace(
            cast_ray=lambda _start, _end: [
                types.SimpleNamespace(location=location(3.0, 0.0, 1.6))
            ]
        )

        visible, reason = provider._line_of_sight_visible(
            world=world,
            observer_location=location(0.0, 0.0, 0.0),
            target_actor=target,
            target_location=location(10.0, 0.0, 0.0),
        )

        self.assertFalse(visible)
        self.assertTrue(reason.startswith("occluded:"))

    def test_cp_visibility_filter_accepts_clear_ray(self):
        provider = OpenCDACPProvider.__new__(OpenCDACPProvider)
        provider.visibility_filter_enabled = True
        provider.visibility_sensor_height_m = 1.6
        provider.visibility_target_tolerance_m = 0.5
        location = sys.modules["carla"].Location
        target = types.SimpleNamespace(
            bounding_box=types.SimpleNamespace(
                extent=types.SimpleNamespace(x=0.4, y=0.4, z=0.9)
            )
        )
        world = types.SimpleNamespace(cast_ray=lambda _start, _end: [])

        visible, reason = provider._line_of_sight_visible(
            world=world,
            observer_location=location(0.0, 0.0, 0.0),
            target_actor=target,
            target_location=location(10.0, 0.0, 0.0),
        )

        self.assertTrue(visible)
        self.assertEqual(reason, "line_of_sight_clear")

    def test_cp_actor_geometry_visibility_detects_vehicle_occluder(self):
        provider = OpenCDACPProvider.__new__(OpenCDACPProvider)
        provider.visibility_filter_enabled = True
        provider.visibility_backend = "actor_geometry"
        location = sys.modules["carla"].Location
        target = types.SimpleNamespace(id=30)
        blocker = types.SimpleNamespace(
            id=20,
            type_id="vehicle.truck",
            get_location=lambda: location(5.0, 0.0, 0.0),
            bounding_box=types.SimpleNamespace(
                extent=types.SimpleNamespace(x=2.5, y=1.2)
            ),
        )

        visible, reason = provider._line_of_sight_visible(
            world=types.SimpleNamespace(),
            observer_location=location(0.0, 0.0, 0.0),
            target_actor=target,
            target_location=location(10.0, 0.0, 0.0),
            occluder_actors=[blocker, target],
        )

        self.assertFalse(visible)
        self.assertEqual(reason, "occluded_by_actor:20")

    def test_obstacle_assignment_preserves_admap_lane_id(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        ego_waypoint = types.SimpleNamespace(
            road_id=1,
            section_id=0,
            lane_id=2,
            ad_lane_id=1002,
            left=lambda: None,
            right=lambda: None,
        )
        obstacle_waypoint = types.SimpleNamespace(
            road_id=1,
            section_id=1,
            lane_id=2,
            ad_lane_id=1002,
            left=lambda: None,
            right=lambda: None,
        )
        bridge.reference_map = types.SimpleNamespace(
            get_waypoint=lambda _pos: obstacle_waypoint
        )

        assignments = bridge._assign_obstacles_to_lanes(
            [{"vehicle_id": "front_car", "x": 10.0, "y": 0.0, "z": 0.0}],
            ego_waypoint=ego_waypoint,
            ego_lane_id=1002,
        )

        self.assertEqual(assignments["front_car"], 1002)

    def test_obstacle_in_different_lane_still_uses_fresh_recount(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        ego_waypoint = types.SimpleNamespace(
            road_id=1,
            section_id=0,
            lane_id=2,
            get_left_lane=lambda: None,
            get_right_lane=lambda: None,
        )
        obstacle_waypoint = types.SimpleNamespace(
            road_id=9,
            section_id=0,
            lane_id=1,
            get_left_lane=lambda: None,
            get_right_lane=lambda: None,
        )
        bridge.reference_map = types.SimpleNamespace(
            get_waypoint=lambda _pos: obstacle_waypoint
        )

        assignments = bridge._assign_obstacles_to_lanes(
            [{"vehicle_id": "other_lane_car", "x": 10.0, "y": 3.5, "z": 0.0}],
            ego_waypoint=ego_waypoint,
            ego_lane_id=5,
        )

        self.assertEqual(assignments["other_lane_car"], 1)

    def test_missing_ego_context_falls_back_to_fresh_recount(self):
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        obstacle_waypoint = types.SimpleNamespace(
            road_id=1,
            section_id=0,
            lane_id=2,
            get_left_lane=lambda: None,
            get_right_lane=lambda: None,
        )
        bridge.reference_map = types.SimpleNamespace(
            get_waypoint=lambda _pos: obstacle_waypoint
        )

        assignments = bridge._assign_obstacles_to_lanes(
            [{"vehicle_id": "car", "x": 10.0, "y": 0.0, "z": 0.0}],
        )

        self.assertEqual(assignments["car"], 1)


if __name__ == "__main__":
    unittest.main()
