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
        pass

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
