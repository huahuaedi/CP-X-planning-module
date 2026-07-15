import math
import numpy as np
import types
import unittest

from MPC.lane_keep import (
    LaneKeepingStageReference,
    evaluate_lane_keeping_profile,
    evaluate_lane_keeping_stage,
    signed_lateral_offset,
)
from MPC.mpc import MPC, MPCRepulsivePotentialSpec
from behavior_planner.temp_destination import _build_route_reference_samples_from_anchor


class _DummyWaypoint:
    def __init__(
        self,
        *,
        x_m: float,
        y_m: float,
        yaw_deg: float,
        lane_id: int = 1,
        lane_width_m: float = 3.5,
        road_id: int = 1,
        section_id: int = 0,
    ) -> None:
        self.transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=float(x_m), y=float(y_m), z=0.0),
            rotation=types.SimpleNamespace(yaw=float(yaw_deg)),
        )
        self.lane_id = int(lane_id)
        self.lane_width = float(lane_width_m)
        self.road_id = int(road_id)
        self.section_id = int(section_id)
        self.is_junction = False
        self.lane_type = "Driving"
        self._left_lane = None
        self._right_lane = None

    def set_neighbors(self, *, left=None, right=None):
        self._left_lane = left
        self._right_lane = right
        return self

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane


class _DummyMap:
    def __init__(self, *waypoints):
        self._waypoints = list(waypoints)

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        del project_to_road
        del lane_type
        if len(self._waypoints) == 0:
            return None
        return min(
            self._waypoints,
            key=lambda waypoint: (
                (float(waypoint.transform.location.x) - float(location.x)) ** 2
                + (float(waypoint.transform.location.y) - float(location.y)) ** 2
            ),
        )


class _DummyCarla:
    class Location:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class LaneType:
        Driving = "Driving"


class LaneKeepingMathTests(unittest.TestCase):
    def test_signed_lateral_offset_matches_lane_frame_formula(self):
        reference = LaneKeepingStageReference(
            x_center_m=10.0,
            y_center_m=5.0,
            heading_rad=math.pi / 2.0,
            lane_width_m=4.0,
            lane_id=1,
        )

        d_perp_m = signed_lateral_offset(
            x_m=11.0,
            y_m=5.0,
            reference=reference,
        )

        self.assertAlmostEqual(float(d_perp_m), -1.0)

    def test_road_boundary_cost_activates_near_road_edge(self):
        reference = LaneKeepingStageReference(
            x_center_m=0.0,
            y_center_m=0.0,
            heading_rad=0.0,
            lane_width_m=4.0,
            lane_id=1,
            road_center_offset_m=0.0,
            road_left_width_m=6.0,
            road_right_width_m=6.0,
        )

        inside = evaluate_lane_keeping_stage(
            stage_index=1,
            x_m=0.0,
            y_m=0.0,
            reference=reference,
            centering_weight=2.0,
            boundary_weight=20.0,
            safe_region_alpha=0.7,
            road_boundary_margin_m=0.5,
        )
        near_boundary = evaluate_lane_keeping_stage(
            stage_index=2,
            x_m=0.0,
            y_m=5.8,
            reference=reference,
            centering_weight=2.0,
            boundary_weight=20.0,
            safe_region_alpha=0.7,
            road_boundary_margin_m=0.5,
        )
        outside = evaluate_lane_keeping_stage(
            stage_index=3,
            x_m=0.0,
            y_m=6.2,
            reference=reference,
            centering_weight=2.0,
            boundary_weight=20.0,
            safe_region_alpha=0.7,
            road_boundary_margin_m=0.5,
        )

        self.assertAlmostEqual(float(inside.boundary_cost), 0.0)
        self.assertGreater(float(near_boundary.boundary_cost), 0.0)
        self.assertFalse(bool(near_boundary.outside_road))
        self.assertTrue(bool(outside.outside_road))
        self.assertGreater(float(outside.boundary_cost), float(near_boundary.boundary_cost))

    def test_lane_keeping_profile_returns_stage_costs_and_total(self):
        reference_samples = [
            {
                "x_ref_m": 0.0,
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
                "lane_width_m": 4.0,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 6.0,
                "road_right_width_m": 6.0,
            }
            for _ in range(3)
        ]

        profile = evaluate_lane_keeping_profile(
            state_xy=[(0.0, 0.0), (0.0, 1.0), (0.0, 2.1)],
            lane_references=reference_samples,
            centering_weight=1.0,
            boundary_weight=10.0,
            safe_region_alpha=0.75,
            road_boundary_margin_m=0.5,
            default_lane_width_m=4.0,
        )

        diagnostics = profile.as_dict()
        self.assertEqual(len(diagnostics["d_perp_m"]), 3)
        self.assertEqual(len(diagnostics["U_lane"]), 3)
        self.assertAlmostEqual(
            float(diagnostics["J_lane"]),
            sum(float(value) for value in diagnostics["U_lane"]),
        )
        self.assertFalse(bool(diagnostics["outside_road"][-1]))


class MPCLaneKeepingIntegrationTests(unittest.TestCase):
    def test_stage_sample_preserves_lane_width(self):
        mpc = object.__new__(MPC)
        mpc.lane_center_reference_local_window = 0
        mpc.lane_width_m = 3.5

        sample = mpc._get_lane_center_stage_sample(
            lane_center_reference=[
                {
                    "x_ref_m": 0.0,
                    "y_ref_m": 0.0,
                    "heading_rad": 0.0,
                    "lane_id": 2,
                    "lane_width_m": 4.2,
                }
            ],
            stage_index=0,
        )

        self.assertIsNotNone(sample)
        self.assertAlmostEqual(float(sample["lane_width_m"]), 4.2)
        self.assertEqual(int(sample["lane_id"]), 2)

    def test_route_reference_samples_include_lane_width(self):
        anchor_wp = _DummyWaypoint(x_m=0.0, y_m=0.0, yaw_deg=0.0, lane_width_m=3.7)
        forward_wp = _DummyWaypoint(x_m=2.0, y_m=0.0, yaw_deg=0.0, lane_width_m=3.9)
        world_map = _DummyMap(anchor_wp, forward_wp)

        samples = _build_route_reference_samples_from_anchor(
            world_map=world_map,
            carla=_DummyCarla,
            anchor_wp=anchor_wp,
            route_points=[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            horizon_steps=2,
            step_distance_m=2.0,
            fallback_lane_id=1,
            target_lane_id=1,
            follow_route_lane=True,
        )

        self.assertEqual(len(samples), 2)
        self.assertTrue(all("lane_width_m" in sample for sample in samples))
        self.assertAlmostEqual(float(samples[0]["lane_width_m"]), 3.7, places=3)
        self.assertAlmostEqual(float(samples[1]["lane_width_m"]), 3.9, places=3)

    def test_route_reference_samples_include_three_lane_road_boundary_width(self):
        right_wp = _DummyWaypoint(x_m=0.0, y_m=0.0, yaw_deg=0.0, lane_id=1, lane_width_m=3.5)
        middle_wp = _DummyWaypoint(x_m=0.0, y_m=3.5, yaw_deg=0.0, lane_id=2, lane_width_m=3.5)
        left_wp = _DummyWaypoint(x_m=0.0, y_m=7.0, yaw_deg=0.0, lane_id=3, lane_width_m=3.5)
        right_wp.set_neighbors(left=middle_wp)
        middle_wp.set_neighbors(left=left_wp, right=right_wp)
        left_wp.set_neighbors(right=middle_wp)
        world_map = _DummyMap(middle_wp)

        samples = _build_route_reference_samples_from_anchor(
            world_map=world_map,
            carla=_DummyCarla,
            anchor_wp=middle_wp,
            route_points=[[0.0, 3.5], [2.0, 3.5]],
            horizon_steps=1,
            step_distance_m=2.0,
            fallback_lane_id=2,
            target_lane_id=2,
            follow_route_lane=True,
        )

        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(float(samples[0]["road_center_offset_m"]), 0.0, places=3)
        self.assertAlmostEqual(float(samples[0]["road_left_width_m"]), 5.25, places=3)
        self.assertAlmostEqual(float(samples[0]["road_right_width_m"]), 5.25, places=3)

    def test_lane_reference_normalization_falls_back_to_single_lane_road(self):
        mpc = object.__new__(MPC)
        mpc.lane_width_m = 4.0

        reference = mpc._normalized_lane_reference_sample_dict(
            {
                "x_ref_m": 0.0,
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
                "lane_width_m": 4.0,
            }
        )

        self.assertIsNotNone(reference)
        self.assertAlmostEqual(float(reference["road_center_offset_m"]), 0.0)
        self.assertAlmostEqual(float(reference["road_left_width_m"]), 2.0)
        self.assertAlmostEqual(float(reference["road_right_width_m"]), 2.0)

    def test_mpc_qp_uses_two_road_boundary_slacks_per_stage(self):
        mpc = MPC(
            {
                "horizon_s": 0.2,
                "plan_dt_s": 0.1,
                "wheelbase_m": 2.7,
                "cost": {
                    "attractive": {"w_attractive": 0.0},
                    "lane_center_follow": {"enabled": False, "w0": 0.0},
                    "road_boundary": {
                        "enabled": True,
                        "w_boundary": 10000.0,
                        "margin_m": 0.5,
                    },
                    "control": {"w_control": 0.0, "q_a": 0.0, "q_delta": 0.0},
                    "repulsive_potential": {"enabled": False},
                },
            },
            {"lane_width_m": 4.0, "lane_count": 3},
        )
        x_ref_rollout = np.zeros((mpc.horizon_steps + 1, 4), dtype=float)
        u_ref_rollout = np.zeros((mpc.horizon_steps, 2), dtype=float)
        lane_reference = [
            {
                "x_ref_m": 0.0,
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
                "lane_width_m": 4.0,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 6.0,
                "road_right_width_m": 6.0,
            }
            for _ in range(mpc.horizon_steps + 1)
        ]

        _, _, _, _, _, index = mpc._build_qp(
            x0=np.zeros(4, dtype=float),
            x_ref_target=np.zeros(4, dtype=float),
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout,
            u_ref_rollout=u_ref_rollout,
            lane_center_reference=lane_reference,
            speed_upper_bound_mps=None,
            reachable_speed_floor_profile_mps=None,
        )

        self.assertEqual(int(index.road_boundary_slack_pair_count), int(mpc.horizon_steps))
        self.assertEqual(int(index.road_boundary_slack_count), 2 * int(mpc.horizon_steps))
        self.assertFalse(hasattr(index, "lane_boundary_slack_index"))

    def test_obstacle_potential_uses_collision_field_only(self):
        mpc = object.__new__(MPC)
        mpc.lane_width_m = 3.5
        mpc.ego_length_m = 4.5
        mpc.ego_width_m = 2.0
        mpc.repulsive_cost = MPCRepulsivePotentialSpec(
            enabled=True,
            w_safe_zone=100.0,
            w_collision_zone=100.0,
            safe_exponential_gain=3.0,
            safe_distance_shift=1.5,
            collision_exponential_gain=10.0,
            collision_distance_shift=1.5,
            max_braking_deceleration_mps2=8.0,
            comfort_deceleration_mps2=2.0,
            reaction_time_s=1.0,
            static_longitudinal_buffer_m=2.0,
            static_lateral_buffer_m=1.0,
            shape_exponent=4.0,
            min_lateral_approach_speed_mps=0.1,
            max_longitudinal_zone_length_m=25.0,
            limit_lateral_zone_to_lane_width=True,
            max_lateral_zone_lane_fraction=1.0,
            project_hessian_psd=True,
            min_hessian_eig=1.0e-9,
        )

        geometry = mpc._superellipsoid_zone_geometry(
            ego_state=[0.0, 0.0, 0.0, 0.0],
            obstacle_state=[0.0, 0.0, 0.0, 0.0],
            obstacle_length_m=4.5,
            obstacle_width_m=2.0,
        )
        cost_safe, cost_collision = mpc._superellipsoid_obstacle_cost_components(
            ego_state=[0.0, 0.0, 0.0, 0.0],
            obstacle_state=[0.0, 0.0, 0.0, 0.0],
            obstacle_length_m=4.5,
            obstacle_width_m=2.0,
        )

        self.assertAlmostEqual(float(geometry["xc_m"]), 3.25)
        self.assertAlmostEqual(float(geometry["yc_m"]), 1.5)
        self.assertAlmostEqual(float(geometry["xs_m"]), float(geometry["xc_m"]))
        self.assertAlmostEqual(float(geometry["ys_m"]), float(geometry["yc_m"]))
        self.assertAlmostEqual(float(cost_safe), 0.0)
        self.assertGreater(float(cost_collision), 0.0)

    def test_consecutive_solver_failures_trigger_seed_reset_threshold(self):
        mpc = object.__new__(MPC)
        mpc.reference_consecutive_solver_failure_reset_threshold = 4
        mpc._consecutive_solver_failure_count = 0
        mpc._last_failure_reset_triggered = False
        mpc._last_x_solution = np.ones((2, 4))
        mpc._last_u_solution = np.ones((1, 2))
        mpc._previous_x_solution = np.ones((2, 4))
        mpc._previous_u_solution = np.ones((1, 2))

        self.assertFalse(mpc._record_solver_failure_state(solved=False))
        self.assertFalse(mpc._record_solver_failure_state(solved=False))
        self.assertFalse(mpc._record_solver_failure_state(solved=False))
        self.assertTrue(mpc._record_solver_failure_state(solved=False))

        mpc._clear_all_solution_memory()
        self.assertIsNone(mpc._last_x_solution)
        self.assertIsNone(mpc._last_u_solution)
        self.assertIsNone(mpc._previous_x_solution)
        self.assertIsNone(mpc._previous_u_solution)

        mpc._record_clean_restart_result(solved=True)
        self.assertEqual(int(mpc._consecutive_solver_failure_count), 0)

    def test_mpc_uses_predicted_obstacle_state_at_stage(self):
        mpc = object.__new__(MPC)

        state = mpc._get_object_state_at_stage(
            object_snapshot={
                "x": 0.0,
                "y": 0.0,
                "v": 1.0,
                "psi": 0.0,
                "predicted_trajectory": [
                    [10.0, 1.0, 2.0, 0.1],
                    [20.0, 2.0, 3.0, 0.2],
                ],
            },
            stage_index=1,
            dt_s=0.1,
        )

        self.assertEqual(state, [20.0, 2.0, 3.0, 0.2])


if __name__ == "__main__":
    unittest.main()
