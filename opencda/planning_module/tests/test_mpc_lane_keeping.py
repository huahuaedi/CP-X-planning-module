import math
import numpy as np
import types
import unittest

from MPC.lane_keep import (
    LaneKeepingStageReference,
    RoadEnvelopeBlock,
    evaluate_lane_keeping_profile,
    evaluate_lane_keeping_stage,
    road_envelope_block_signed_distance,
    road_envelope_conservativeness_correction,
    road_envelope_union_logsumexp,
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


class RoadEnvelopeBlockMathTests(unittest.TestCase):
    @staticmethod
    def _block(x=0.0, y=0.0, heading=0.0, half_length=5.0, half_width=1.75):
        return RoadEnvelopeBlock(
            x_center_m=x, y_center_m=y, heading_rad=heading,
            half_length_m=half_length, half_width_m=half_width,
        )

    def test_zero_at_boundary_along_axis(self):
        block = self._block()
        g_b, _, _ = road_envelope_block_signed_distance(block, 5.0, 0.0)
        self.assertAlmostEqual(g_b, 0.0, places=6)

    def test_negative_inside_positive_outside(self):
        block = self._block()
        g_center, _, _ = road_envelope_block_signed_distance(block, 0.0, 0.0)
        g_outside, _, _ = road_envelope_block_signed_distance(block, 20.0, 20.0)
        self.assertLess(g_center, 0.0)
        self.assertGreater(g_outside, 0.0)

    def test_analytic_gradient_matches_central_difference(self):
        block = self._block()
        h = 1.0e-4
        for x, y in [(3.0, 0.5), (4.9, 1.0), (-2.0, -1.5), (0.5, 1.7)]:
            _, dgdx, dgdy = road_envelope_block_signed_distance(block, x, y)
            gx1, _, _ = road_envelope_block_signed_distance(block, x + h, y)
            gx2, _, _ = road_envelope_block_signed_distance(block, x - h, y)
            gy1, _, _ = road_envelope_block_signed_distance(block, x, y + h)
            gy2, _, _ = road_envelope_block_signed_distance(block, x, y - h)
            self.assertAlmostEqual(dgdx, (gx1 - gx2) / (2 * h), places=4)
            self.assertAlmostEqual(dgdy, (gy1 - gy2) / (2 * h), places=4)

    def test_rejects_odd_shape_exponent(self):
        with self.assertRaises(ValueError):
            RoadEnvelopeBlock(
                x_center_m=0.0, y_center_m=0.0, heading_rad=0.0,
                half_length_m=5.0, half_width_m=1.75, shape_exponent=3.0,
            )

    def test_union_bounds_hold(self):
        source = self._block(y=0.0)
        target = self._block(y=3.5)
        for x, y in [(2.0, 0.0), (2.0, 1.75), (2.0, 3.5), (2.0, 0.9)]:
            g_lse, _, _, weights = road_envelope_union_logsumexp(
                [source, target], rho=-8.0, x_m=x, y_m=y
            )
            g1, _, _ = road_envelope_block_signed_distance(source, x, y)
            g2, _, _ = road_envelope_block_signed_distance(target, x, y)
            g_min = min(g1, g2)
            self.assertLessEqual(g_lse, g_min + 1.0e-9)
            self.assertGreaterEqual(g_lse, g_min + math.log(2.0) / -8.0 - 1.0e-9)
            self.assertAlmostEqual(sum(weights), 1.0, places=9)

    def test_union_weights_stable_across_rho_and_g_range(self):
        source = self._block(y=0.0, half_length=50.0, half_width=50.0)
        target = self._block(y=200.0, half_length=50.0, half_width=50.0)
        for rho in (-1.0, -8.0, -50.0):
            for x, y in [(0.0, 0.0), (0.0, 100.0), (0.0, 200.0), (500.0, 500.0)]:
                g_lse, dgdx, dgdy, weights = road_envelope_union_logsumexp(
                    [source, target], rho=rho, x_m=x, y_m=y
                )
                self.assertTrue(math.isfinite(g_lse))
                self.assertTrue(math.isfinite(dgdx))
                self.assertTrue(math.isfinite(dgdy))
                self.assertAlmostEqual(sum(weights), 1.0, places=6)
                for w in weights:
                    self.assertGreaterEqual(w, -1.0e-9)
                    self.assertLessEqual(w, 1.0 + 1.0e-9)

    def test_conservativeness_correction_is_nonpositive(self):
        source = self._block(y=0.0)
        target = self._block(y=3.5)
        epsilon0 = road_envelope_conservativeness_correction([source, target], rho=-8.0)
        self.assertLessEqual(epsilon0, 0.0)

    def test_conservativeness_correction_is_the_true_minimum_over_probes(self):
        # epsilon0 = min over boundary probe points of g_lse (Theorem 1,
        # Eq 58) -- so at least one probe point must hit it exactly, and
        # every other probe point's g_lse must be >= epsilon0.
        source = RoadEnvelopeBlockMathTests._block(y=0.0)
        target = RoadEnvelopeBlockMathTests._block(y=3.5)
        blocks = [source, target]
        epsilon0 = road_envelope_conservativeness_correction(blocks, rho=-8.0)
        from MPC.lane_keep import road_envelope_block_boundary_probe_points_xy

        probe_lse_values = [
            road_envelope_union_logsumexp(blocks, rho=-8.0, x_m=x, y_m=y)[0]
            for block in blocks
            for x, y in road_envelope_block_boundary_probe_points_xy(block)
        ]
        self.assertAlmostEqual(min(probe_lse_values), epsilon0, places=9)
        self.assertTrue(all(value >= epsilon0 - 1.0e-9 for value in probe_lse_values))

    def test_corrected_constraint_accepts_deep_interior_rejects_far_exterior(self):
        # The runtime constraint g_lse(x,y) - epsilon0 <= 0 must accept a
        # point deep inside either block and reject a point far outside
        # both -- the basic sanity check on which side of the corrected
        # boundary each region falls.
        source = RoadEnvelopeBlockMathTests._block(y=0.0)
        target = RoadEnvelopeBlockMathTests._block(y=3.5)
        blocks = [source, target]
        epsilon0 = road_envelope_conservativeness_correction(blocks, rho=-8.0)

        g_lse_inside, _, _, _ = road_envelope_union_logsumexp(
            blocks, rho=-8.0, x_m=1.0, y_m=0.0
        )
        self.assertLess(g_lse_inside - epsilon0, 0.0)

        g_lse_outside, _, _, _ = road_envelope_union_logsumexp(
            blocks, rho=-8.0, x_m=1.0, y_m=50.0
        )
        self.assertGreater(g_lse_outside - epsilon0, 0.0)


class MPCLaneKeepingIntegrationTests(unittest.TestCase):

    def test_speed_tracking_reference_is_independent_of_warm_start_gain(self):
        mpc = MPC(*self._minimal_mpc_config(speed_soft_enabled=False))
        x0 = np.array([0.0, 0.0, 2.0, 0.0], dtype=float)
        target = np.array([20.0, 0.0, 8.0, 0.0], dtype=float)

        mpc.reference_speed_gain = 0.2
        slow_rollout, _ = mpc._reference_rollout(
            x0=x0,
            x_ref_target=target,
            lane_center_reference=None,
            object_snapshots=[],
            speed_upper_bound_mps=8.0,
        )
        slow_speed_reference = mpc._speed_tracking_reference(
            x0=x0,
            x_ref_target=target,
            linearization_rollout=slow_rollout,
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            speed_upper_bound_mps=8.0,
        )

        mpc.reference_speed_gain = 3.0
        fast_rollout, _ = mpc._reference_rollout(
            x0=x0,
            x_ref_target=target,
            lane_center_reference=None,
            object_snapshots=[],
            speed_upper_bound_mps=8.0,
        )
        fast_speed_reference = mpc._speed_tracking_reference(
            x0=x0,
            x_ref_target=target,
            linearization_rollout=fast_rollout,
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            speed_upper_bound_mps=8.0,
        )

        self.assertFalse(np.allclose(slow_rollout[:, 2], fast_rollout[:, 2]))
        np.testing.assert_allclose(
            slow_speed_reference,
            fast_speed_reference,
            atol=1.0e-9,
        )
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

    @staticmethod
    def _lane_change_mpc_config(*, road_envelope_enabled: bool):
        return (
            {
                "horizon_s": 1.0,
                "plan_dt_s": 0.1,
                "wheelbase_m": 2.7,
                "constraints": {
                    "min_steer_rad": -0.5,
                    "max_steer_rad": 0.5,
                    "min_steer_rate_rps": -0.35,
                    "max_steer_rate_rps": 0.35,
                    "min_acceleration_mps2": -3.0,
                    "max_acceleration_mps2": 3.0,
                },
                "cost": {
                    "attractive": {"w_attractive": 0.5, "q_x": 1.0, "q_y": 1.0, "q_v": 1.0, "q_psi": 1.0},
                    "lane_center_follow": {"enabled": True, "w0": 28.0, "xy_w0": 2.0, "q_psi": 2.5},
                    # Mirrors mpc.yaml's execute_lane_change profile's tight
                    # slack cap -- the exact setting under which the
                    # reference-jump bug was diagnosed as genuine primal
                    # infeasibility, not merely high cost.
                    "road_boundary": {
                        "enabled": True,
                        "w_boundary": 14000.0,
                        "margin_m": 0.6,
                        "max_slack_m": 0.07,
                    },
                    "control": {"w_control": 8.0, "q_a": 8.0, "q_delta": 160.0},
                    "repulsive_potential": {"enabled": False},
                    "road_envelope": {
                        "enabled": road_envelope_enabled,
                        "w_envelope": 10000.0,
                        "max_slack_m": 0.10,
                        "rho": -8.0,
                    },
                },
            },
            {"lane_width_m": 3.5, "lane_count": 3},
        )

    def test_road_envelope_slack_count_matches_flag_and_is_mutually_exclusive(self):
        x0 = np.array([0.0, 0.0, 2.6, 0.0], dtype=float)
        horizon_steps = 10
        x_ref_rollout = np.tile(x0, (horizon_steps + 1, 1))
        u_ref_rollout = np.zeros((horizon_steps, 2), dtype=float)
        blocks = [
            RoadEnvelopeBlock(x_center_m=1.5, y_center_m=0.0, heading_rad=0.0, half_length_m=1.5, half_width_m=1.25),
            RoadEnvelopeBlock(x_center_m=1.5, y_center_m=3.5, heading_rad=0.0, half_length_m=1.5, half_width_m=1.25),
        ]
        epsilon0 = road_envelope_conservativeness_correction(blocks, rho=-8.0)
        road_envelope_blocks = {"blocks": blocks, "epsilon0": epsilon0, "rho": -8.0}

        mpc_on = MPC(*self._lane_change_mpc_config(road_envelope_enabled=True))
        mpc_on.horizon_steps = horizon_steps
        _, _, _, _, _, index_on = mpc_on._build_qp(
            x0=x0, x_ref_target=x0, object_snapshots=[],
            current_acceleration_mps2=0.0, current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout, u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=None, reachable_speed_floor_profile_mps=None,
            road_envelope_blocks=road_envelope_blocks,
        )
        self.assertEqual(int(index_on.road_envelope_slack_count), horizon_steps)
        self.assertEqual(int(index_on.road_boundary_slack_pair_count), 0)

        mpc_off = MPC(*self._lane_change_mpc_config(road_envelope_enabled=False))
        mpc_off.horizon_steps = horizon_steps
        _, _, _, _, _, index_off = mpc_off._build_qp(
            x0=x0, x_ref_target=x0, object_snapshots=[],
            current_acceleration_mps2=0.0, current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout, u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=None, reachable_speed_floor_profile_mps=None,
            road_envelope_blocks=road_envelope_blocks,
        )
        self.assertEqual(int(index_off.road_envelope_slack_count), 0)
        # Toggle off (even with blocks supplied) falls back to today's
        # road_boundary behavior unchanged, not to no boundary at all.
        self.assertEqual(int(index_off.road_boundary_slack_pair_count), horizon_steps)

        # No blocks supplied even though the flag is on -- must fall back
        # to today's road_boundary behavior exactly (fail-open, not
        # fail-locked-out).
        _, _, _, _, _, index_no_blocks = mpc_on._build_qp(
            x0=x0, x_ref_target=x0, object_snapshots=[],
            current_acceleration_mps2=0.0, current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout, u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=None, reachable_speed_floor_profile_mps=None,
            road_envelope_blocks=None,
        )
        self.assertEqual(int(index_no_blocks.road_envelope_slack_count), 0)
        self.assertEqual(int(index_no_blocks.road_boundary_slack_pair_count), horizon_steps)

    def test_road_envelope_fixes_the_diagnosed_reference_jump_infeasibility(self):
        # THE critical regression test. Reproduces the exact diagnosed bug:
        # ego sits at the source lane's centerline (unmoved), but
        # lane_center_reference has already switched to the target lane's
        # samples (a ~3.5m jump) -- exactly what happens the instant a
        # committed lane change's tracked reference source flips. Under
        # today's single-reference-line road_boundary constraint (with the
        # tight execute_lane_change slack cap), this must be genuinely
        # primal infeasible, not just high-cost. With road_envelope blocks
        # covering both the source and target lane, it must solve cleanly.
        x0 = np.array([0.0, 0.0, 2.6, 0.0], dtype=float)
        horizon_steps = 10
        x_ref_rollout = np.tile(x0, (horizon_steps + 1, 1))
        u_ref_rollout = np.zeros((horizon_steps, 2), dtype=float)
        target_lane_reference = [
            {
                "x_ref_m": 0.3 * float(k), "y_ref_m": 3.5, "heading_rad": 0.0,
                "lane_id": 2, "lane_width_m": 3.5, "road_center_offset_m": 0.0,
                "road_left_width_m": 1.75, "road_right_width_m": 1.75,
            }
            for k in range(horizon_steps + 1)
        ]

        mpc_off = MPC(*self._lane_change_mpc_config(road_envelope_enabled=False))
        mpc_off.horizon_steps = horizon_steps
        P, q, A, l, u, _ = mpc_off._build_qp(
            x0=x0, x_ref_target=x0, object_snapshots=[],
            current_acceleration_mps2=0.0, current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout, u_ref_rollout=u_ref_rollout,
            lane_center_reference=target_lane_reference,
            speed_upper_bound_mps=None, reachable_speed_floor_profile_mps=None,
        )
        solution_off, status_off, _ = mpc_off._solve_qp(P=P, q=q, A=A, l=l, u=u)
        self.assertIsNone(solution_off)
        self.assertIn("infeasible", status_off)

        source_block = RoadEnvelopeBlock(x_center_m=1.5, y_center_m=0.0, heading_rad=0.0, half_length_m=1.5, half_width_m=1.25)
        target_block = RoadEnvelopeBlock(x_center_m=1.5, y_center_m=3.5, heading_rad=0.0, half_length_m=1.5, half_width_m=1.25)
        blocks = [source_block, target_block]
        epsilon0 = road_envelope_conservativeness_correction(blocks, rho=-8.0)
        road_envelope_blocks = {"blocks": blocks, "epsilon0": epsilon0, "rho": -8.0}

        mpc_on = MPC(*self._lane_change_mpc_config(road_envelope_enabled=True))
        mpc_on.horizon_steps = horizon_steps
        P2, q2, A2, l2, u2, _ = mpc_on._build_qp(
            x0=x0, x_ref_target=x0, object_snapshots=[],
            current_acceleration_mps2=0.0, current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout, u_ref_rollout=u_ref_rollout,
            lane_center_reference=target_lane_reference,
            speed_upper_bound_mps=None, reachable_speed_floor_profile_mps=None,
            road_envelope_blocks=road_envelope_blocks,
        )
        solution_on, status_on, _ = mpc_on._solve_qp(P=P2, q=q2, A=A2, l=l2, u=u2)
        self.assertIsNotNone(solution_on)
        self.assertIn("solved", status_on)

    def test_road_envelope_gradient_points_away_from_nearer_block(self):
        # Sign-convention sanity check before this ever reaches CARLA: the
        # gradient of g_lse must point in the direction that *increases*
        # g_lse (away from the corridor), so the linearized constraint
        # correctly pulls the QP's solution back toward whichever block is
        # nearer, not away from it.
        source_block = RoadEnvelopeBlock(x_center_m=1.5, y_center_m=0.0, heading_rad=0.0, half_length_m=1.5, half_width_m=1.25)
        target_block = RoadEnvelopeBlock(x_center_m=1.5, y_center_m=3.5, heading_rad=0.0, half_length_m=1.5, half_width_m=1.25)
        blocks = [source_block, target_block]

        # x=1.5 (within both blocks' along-axis span), y=6.0 -- above the
        # target block's far edge (y=3.5+1.25=4.75), outside both blocks,
        # closer to the target block. Moving further in +y increases
        # distance from the nearer (target) block, so dg_dy must be > 0.
        g_lse0, dg_dx, dg_dy, _weights = road_envelope_union_logsumexp(
            blocks=blocks, rho=-8.0, x_m=1.5, y_m=6.0,
        )
        self.assertGreater(g_lse0, 0.0)  # genuinely outside the union
        self.assertGreater(dg_dy, 0.0)
        self.assertAlmostEqual(dg_dx, 0.0, places=6)  # centered in x -> no x-pull

    @staticmethod
    def _minimal_mpc_config(*, speed_soft_enabled: bool, min_acceleration_mps2: float = -3.0):
        return (
            {
                "horizon_s": 0.3,
                "plan_dt_s": 0.1,
                "wheelbase_m": 2.7,
                "constraints": {
                    "min_acceleration_mps2": min_acceleration_mps2,
                    "max_acceleration_mps2": 3.0,
                },
                "cost": {
                    "attractive": {"w_attractive": 0.0},
                    "lane_center_follow": {"enabled": False, "w0": 0.0},
                    "road_boundary": {"enabled": False},
                    "control": {"w_control": 0.0, "q_a": 0.0, "q_delta": 0.0},
                    "repulsive_potential": {"enabled": False},
                    "speed_soft_constraint": {
                        "enabled": speed_soft_enabled,
                        "weight": 200.0,
                        "max_slack_mps": 3.0,
                    },
                },
            },
            {"lane_width_m": 4.0, "lane_count": 3},
        )

    def test_speed_slack_variable_count_matches_flag(self):
        mpc_enabled = MPC(*self._minimal_mpc_config(speed_soft_enabled=True))
        mpc_disabled = MPC(*self._minimal_mpc_config(speed_soft_enabled=False))
        x_ref_rollout = np.zeros((mpc_enabled.horizon_steps + 1, 4), dtype=float)
        u_ref_rollout = np.zeros((mpc_enabled.horizon_steps, 2), dtype=float)

        _, _, _, _, _, index_enabled = mpc_enabled._build_qp(
            x0=np.zeros(4, dtype=float),
            x_ref_target=np.zeros(4, dtype=float),
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout,
            u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=None,
            reachable_speed_floor_profile_mps=None,
        )
        _, _, _, _, _, index_disabled = mpc_disabled._build_qp(
            x0=np.zeros(4, dtype=float),
            x_ref_target=np.zeros(4, dtype=float),
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout,
            u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=None,
            reachable_speed_floor_profile_mps=None,
        )

        self.assertEqual(int(index_enabled.speed_slack_count), int(mpc_enabled.horizon_steps))
        self.assertEqual(int(index_disabled.speed_slack_count), 0)
        self.assertEqual(
            int(index_enabled.total_variables),
            int(index_disabled.total_variables) + int(mpc_enabled.horizon_steps),
        )

    def test_soft_speed_cap_makes_an_otherwise_infeasible_speed_drop_solvable(self):
        # v0 = 5.0, min_acceleration = -3.0, dt = 0.1 -> the fastest the QP can
        # bring v down by stage 1 is 5.0 - 3.0*0.1 = 4.7 m/s. A hard cap of
        # 3.0 m/s is therefore infeasible at stage 1 (dynamics+accel bound
        # cannot reach it). With the soft constraint enabled (max_slack_mps
        # = 3.0 default), v=4.7 is within the allowed 3.0+3.0=6.0 ceiling, so
        # the same problem becomes solvable.
        x0 = np.array([0.0, 0.0, 5.0, 0.0], dtype=float)

        mpc_enabled = MPC(*self._minimal_mpc_config(speed_soft_enabled=True))
        x_ref_rollout = np.tile(x0, (mpc_enabled.horizon_steps + 1, 1))
        u_ref_rollout = np.zeros((mpc_enabled.horizon_steps, 2), dtype=float)
        P, q, A, l, u, index = mpc_enabled._build_qp(
            x0=x0,
            x_ref_target=x0,
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout,
            u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=3.0,
            reachable_speed_floor_profile_mps=None,
        )
        solution_enabled, status_enabled, _ = mpc_enabled._solve_qp(P=P, q=q, A=A, l=l, u=u)
        self.assertIsNotNone(solution_enabled)
        self.assertIn("solved", status_enabled)

        mpc_disabled = MPC(*self._minimal_mpc_config(speed_soft_enabled=False))
        P2, q2, A2, l2, u2, _ = mpc_disabled._build_qp(
            x0=x0,
            x_ref_target=x0,
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout,
            u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=3.0,
            reachable_speed_floor_profile_mps=None,
        )
        solution_disabled, status_disabled, _ = mpc_disabled._solve_qp(P=P2, q=q2, A=A2, l=l2, u=u2)
        self.assertTrue(solution_disabled is None or "solved" not in status_disabled)

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
            log_barrier_enabled=False,
            log_barrier_replace_exponential=False,
            w_log_barrier=0.0,
            log_barrier_gain=4.0,
            cross_track_suppression_enabled=False,
            cross_track_full_suppression_m=1.0,
            cross_track_full_response_m=2.0,
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

    @staticmethod
    def _bare_mpc_with_repulsive_cost(**overrides) -> MPC:
        mpc = object.__new__(MPC)
        mpc.lane_width_m = 3.5
        mpc.ego_length_m = 4.5
        mpc.ego_width_m = 2.0
        base = dict(
            enabled=True,
            w_safe_zone=0.0,
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
            log_barrier_enabled=False,
            log_barrier_replace_exponential=False,
            w_log_barrier=0.0,
            log_barrier_gain=4.0,
            cross_track_suppression_enabled=False,
            cross_track_full_suppression_m=1.0,
            cross_track_full_response_m=2.0,
        )
        base.update(overrides)
        mpc.repulsive_cost = MPCRepulsivePotentialSpec(**base)
        return mpc

    def _obstacle_state_at_rc(self, mpc: MPC, target_rc: float):
        # Along the obstacle's local x-axis, x_local_m == target_rc * xc_m
        # (y_local_m == 0), since rc = |x_local/xc|^n + |y_local/yc|^n)^(1/n).
        geometry = mpc._superellipsoid_zone_geometry(
            ego_state=[0.0, 0.0, 0.0, 0.0],
            obstacle_state=[0.0, 0.0, 0.0, 0.0],
            obstacle_length_m=4.5,
            obstacle_width_m=2.0,
        )
        xc_m = float(geometry["xc_m"])
        ego_x_m = float(target_rc) * xc_m
        return [ego_x_m, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]

    def test_log_barrier_disabled_by_default_matches_exponential_only_baseline(self):
        mpc = self._bare_mpc_with_repulsive_cost()
        ego_state, obstacle_state = self._obstacle_state_at_rc(mpc, target_rc=1.0)

        baseline = mpc._superellipsoid_obstacle_cost(
            ego_state=ego_state, obstacle_state=obstacle_state,
            obstacle_length_m=4.5, obstacle_width_m=2.0,
        )
        cost_safe, cost_collision = mpc._superellipsoid_obstacle_cost_components(
            ego_state=ego_state, obstacle_state=obstacle_state,
            obstacle_length_m=4.5, obstacle_width_m=2.0,
        )
        self.assertAlmostEqual(float(baseline), float(cost_safe + cost_collision))

    def test_log_barrier_enabled_increases_cost_near_collision_zone(self):
        mpc_off = self._bare_mpc_with_repulsive_cost(log_barrier_enabled=False)
        mpc_on = self._bare_mpc_with_repulsive_cost(log_barrier_enabled=True, w_log_barrier=50.0)
        # rc == collision_distance_shift (margin == 0): well inside the region
        # softplus is actively penalizing.
        ego_state, obstacle_state = self._obstacle_state_at_rc(mpc_on, target_rc=1.5)

        cost_off = mpc_off._superellipsoid_obstacle_cost(
            ego_state=ego_state, obstacle_state=obstacle_state,
            obstacle_length_m=4.5, obstacle_width_m=2.0,
        )
        cost_on = mpc_on._superellipsoid_obstacle_cost(
            ego_state=ego_state, obstacle_state=obstacle_state,
            obstacle_length_m=4.5, obstacle_width_m=2.0,
        )
        self.assertGreater(float(cost_on), float(cost_off))

    def test_log_barrier_far_field_approaches_exponential_only_baseline(self):
        mpc_off = self._bare_mpc_with_repulsive_cost(log_barrier_enabled=False)
        mpc_on = self._bare_mpc_with_repulsive_cost(log_barrier_enabled=True, w_log_barrier=50.0, log_barrier_gain=4.0)
        # theta=4.0, margin=5.0 far beyond the zone: softplus(-theta*margin)
        # = log(1+exp(-20)) ~= 2e-9, i.e. effectively (but not exactly) zero.
        ego_state, obstacle_state = self._obstacle_state_at_rc(mpc_on, target_rc=1.5 + 5.0)

        cost_off = mpc_off._superellipsoid_obstacle_cost(
            ego_state=ego_state, obstacle_state=obstacle_state,
            obstacle_length_m=4.5, obstacle_width_m=2.0,
        )
        cost_on = mpc_on._superellipsoid_obstacle_cost(
            ego_state=ego_state, obstacle_state=obstacle_state,
            obstacle_length_m=4.5, obstacle_width_m=2.0,
        )
        self.assertAlmostEqual(float(cost_on), float(cost_off), places=6)

    def test_log_barrier_finite_and_monotonic_across_rc_sweep(self):
        mpc = self._bare_mpc_with_repulsive_cost(log_barrier_enabled=True, w_log_barrier=50.0)
        # rc from _superellipsoid_zone_geometry is a norm-like quantity and
        # therefore always >= 0 -- a negative target_rc here would alias to
        # its absolute value (V-shaped, not monotonic), so the sweep must
        # stay non-negative to validly test monotonicity in rc.
        rc_values = [0.25 * i for i in range(28)]  # 0.0 .. 6.75 step 0.25
        costs = []
        for rc in rc_values:
            ego_state, obstacle_state = self._obstacle_state_at_rc(mpc, target_rc=rc)
            cost = mpc._superellipsoid_obstacle_cost(
                ego_state=ego_state, obstacle_state=obstacle_state,
                obstacle_length_m=4.5, obstacle_width_m=2.0,
            )
            self.assertTrue(math.isfinite(float(cost)), f"non-finite cost at rc={rc}")
            costs.append(float(cost))
        for earlier, later in zip(costs, costs[1:]):
            self.assertGreaterEqual(earlier + 1e-9, later)

    def test_log_barrier_replace_exponential_zeroes_collision_term(self):
        mpc = self._bare_mpc_with_repulsive_cost(
            log_barrier_enabled=True,
            log_barrier_replace_exponential=True,
            w_log_barrier=50.0,
        )
        ego_state, obstacle_state = self._obstacle_state_at_rc(mpc, target_rc=1.5)

        total = mpc._superellipsoid_obstacle_cost(
            ego_state=ego_state, obstacle_state=obstacle_state,
            obstacle_length_m=4.5, obstacle_width_m=2.0,
        )
        geometry = mpc._superellipsoid_zone_geometry(
            ego_state=ego_state, obstacle_state=obstacle_state,
            obstacle_length_m=4.5, obstacle_width_m=2.0,
        )
        expected_log_term = mpc._log_barrier_obstacle_cost_component(float(geometry["rc"]))
        self.assertAlmostEqual(float(total), float(expected_log_term))

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


class AdaptiveHorizonTests(unittest.TestCase):
    @staticmethod
    def _adaptive_mpc_config():
        return (
            {
                "horizon_s": 2.0,
                "plan_dt_s": 0.1,
                "wheelbase_m": 2.7,
                "adaptive_horizon_min_s": 1.0,
                "adaptive_horizon_max_s": 5.0,
                "constraints": {
                    "min_acceleration_mps2": -3.0,
                    "max_acceleration_mps2": 3.0,
                },
                "cost": {
                    "attractive": {"w_attractive": 0.0},
                    "lane_center_follow": {"enabled": False, "w0": 0.0},
                    "road_boundary": {"enabled": False},
                    "control": {"w_control": 0.0, "q_a": 0.0, "q_delta": 0.0},
                    "repulsive_potential": {"enabled": False},
                    "speed_soft_constraint": {"enabled": False},
                },
            },
            {"lane_width_m": 4.0, "lane_count": 3},
        )

    def test_blend_toward_horizon_s_moves_partway_to_target(self):
        mpc = MPC(*self._adaptive_mpc_config())
        self.assertAlmostEqual(mpc.horizon_s, 2.0)

        mpc.blend_toward_horizon_s(4.5, blend_alpha=0.5)

        # dt_s=0.1 rounds the blended 3.25s target to the nearest step (3.2s
        # -> 32 steps), not a bit-exact 3.25.
        self.assertAlmostEqual(mpc.horizon_s, 3.2, places=6)
        self.assertEqual(mpc.horizon_steps, 32)

    def test_blend_toward_horizon_s_clamps_to_configured_bounds(self):
        mpc = MPC(*self._adaptive_mpc_config())

        mpc.blend_toward_horizon_s(50.0, blend_alpha=1.0)
        self.assertAlmostEqual(mpc.horizon_s, mpc.adaptive_horizon_max_s, places=6)

        mpc.blend_toward_horizon_s(0.01, blend_alpha=1.0)
        self.assertAlmostEqual(mpc.horizon_s, mpc.adaptive_horizon_min_s, places=6)

    def test_build_qp_matches_horizon_immediately_after_a_change(self):
        mpc = MPC(*self._adaptive_mpc_config())
        mpc.blend_toward_horizon_s(4.0, blend_alpha=1.0)
        self.assertEqual(mpc.horizon_steps, 40)

        x_ref_rollout = np.zeros((mpc.horizon_steps + 1, 4), dtype=float)
        u_ref_rollout = np.zeros((mpc.horizon_steps, 2), dtype=float)
        _, _, _, _, _, index = mpc._build_qp(
            x0=np.zeros(4, dtype=float),
            x_ref_target=np.zeros(4, dtype=float),
            object_snapshots=[],
            current_acceleration_mps2=0.0,
            current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout,
            u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=None,
            reachable_speed_floor_profile_mps=None,
        )

        self.assertEqual(index.horizon_steps, 40)

    def test_small_repeated_blends_hold_steady_until_drift_accumulates(self):
        mpc = MPC(*self._adaptive_mpc_config())
        self.assertEqual(mpc.horizon_steps, 20)
        self.assertEqual(mpc.adaptive_horizon_min_step_change, 3)

        # Small alpha => the continuous blend target drifts by well under
        # one step per call. horizon_steps must hold steady (preserving
        # warm start) on the first two calls, only actually committing a
        # new value once the accumulated drift reaches 3 steps.
        mpc.blend_toward_horizon_s(3.0, blend_alpha=0.1)
        self.assertEqual(mpc.horizon_steps, 20)

        mpc.blend_toward_horizon_s(3.0, blend_alpha=0.1)
        self.assertEqual(mpc.horizon_steps, 20)

        mpc.blend_toward_horizon_s(3.0, blend_alpha=0.1)
        self.assertEqual(mpc.horizon_steps, 23)

        mpc.blend_toward_horizon_s(3.0, blend_alpha=0.1)
        self.assertEqual(mpc.horizon_steps, 23)

    def test_min_step_change_is_configurable(self):
        mpc_cfg, road_cfg = self._adaptive_mpc_config()
        mpc_cfg = dict(mpc_cfg)
        mpc_cfg["adaptive_horizon_min_step_change"] = 1
        mpc = MPC(mpc_cfg, road_cfg)
        self.assertEqual(mpc.adaptive_horizon_min_step_change, 1)

        # With a min_step_change of 1, even a single step of drift commits.
        mpc.blend_toward_horizon_s(3.0, blend_alpha=0.1)
        self.assertEqual(mpc.horizon_steps, 21)


class CrossTrackLateralScaleTests(unittest.TestCase):
    def test_zero_at_and_below_the_suppression_bound(self):
        for offset_m in (0.0, 0.5, 1.0):
            self.assertEqual(
                MPC._cross_track_lateral_scale(
                    cross_track_abs_m=offset_m,
                    full_suppression_m=1.0,
                    full_response_m=2.0,
                ),
                0.0,
            )

    def test_one_at_and_above_the_response_bound(self):
        for offset_m in (2.0, 3.0, 10.0):
            self.assertEqual(
                MPC._cross_track_lateral_scale(
                    cross_track_abs_m=offset_m,
                    full_suppression_m=1.0,
                    full_response_m=2.0,
                ),
                1.0,
            )

    def test_linear_ramp_between_the_bounds(self):
        self.assertAlmostEqual(
            MPC._cross_track_lateral_scale(
                cross_track_abs_m=1.5,
                full_suppression_m=1.0,
                full_response_m=2.0,
            ),
            0.5,
        )

    def test_only_the_absolute_offset_matters(self):
        self.assertEqual(
            MPC._cross_track_lateral_scale(
                cross_track_abs_m=-1.5,
                full_suppression_m=1.0,
                full_response_m=2.0,
            ),
            MPC._cross_track_lateral_scale(
                cross_track_abs_m=1.5,
                full_suppression_m=1.0,
                full_response_m=2.0,
            ),
        )


class CrossTrackSuppressionIntegrationTests(unittest.TestCase):
    @staticmethod
    def _obstacle_mpc_config(*, cross_track_suppression_enabled: bool):
        return (
            {
                "horizon_s": 0.3,
                "plan_dt_s": 0.1,
                "wheelbase_m": 2.7,
                "constraints": {
                    "min_acceleration_mps2": -3.0,
                    "max_acceleration_mps2": 3.0,
                },
                "cost": {
                    "attractive": {"w_attractive": 0.0},
                    "lane_center_follow": {"enabled": False, "w0": 0.0},
                    "road_boundary": {"enabled": False},
                    "control": {"w_control": 0.0, "q_a": 0.0, "q_delta": 0.0},
                    "speed_soft_constraint": {"enabled": False},
                    "repulsive_potential": {
                        "enabled": True,
                        "w_collision_zone": 100.0,
                        "collision_exponential_gain": 6.0,
                        "collision_distance_shift": 1.5,
                        "static_longitudinal_buffer_m": 2.0,
                        "static_lateral_buffer_m": 0.5,
                        "project_hessian_psd": False,
                        "cross_track_suppression_enabled": cross_track_suppression_enabled,
                        "cross_track_full_suppression_m": 1.0,
                        "cross_track_full_response_m": 2.0,
                    },
                },
            },
            {"lane_width_m": 4.0, "lane_count": 3},
        )

    @staticmethod
    def _build_qp_with_obstacle_directly_ahead(*, cross_track_suppression_enabled: bool):
        mpc = MPC(
            *CrossTrackSuppressionIntegrationTests._obstacle_mpc_config(
                cross_track_suppression_enabled=cross_track_suppression_enabled
            )
        )
        # Ego rolls straight ahead along +x (heading 0); the obstacle sits
        # stationary a small (0.3m, below the 1.0m suppression bound), but
        # nonzero, cross-track offset away -- a directly-ahead, closing-
        # distance lead vehicle with a slight residual lateral offset, same
        # as the CSV scenario this fix targets (a real, measurable but
        # small y-pull, not an already-exactly-zero-by-symmetry case).
        x_ref_rollout = np.array(
            [[1.0 * k, 0.0, 2.0, 0.0] for k in range(mpc.horizon_steps + 1)],
            dtype=float,
        )
        u_ref_rollout = np.zeros((mpc.horizon_steps, 2), dtype=float)
        obstacle = {
            "x": 5.0,
            "y": 0.3,
            "v": 0.0,
            "psi": 0.0,
            "length_m": 4.5,
            "width_m": 2.0,
        }
        P, q, _, _, _, index = mpc._build_qp(
            x0=np.array([0.0, 0.0, 2.0, 0.0], dtype=float),
            x_ref_target=np.array([3.0, 0.0, 2.0, 0.0], dtype=float),
            object_snapshots=[obstacle],
            current_acceleration_mps2=0.0,
            current_steering_rad=0.0,
            x_ref_rollout=x_ref_rollout,
            u_ref_rollout=u_ref_rollout,
            lane_center_reference=None,
            speed_upper_bound_mps=None,
            reachable_speed_floor_profile_mps=None,
        )
        x_idx = index.state_index(1, 0)
        y_idx = index.state_index(1, 1)
        return q[x_idx], q[y_idx], P[x_idx, y_idx]

    def test_suppresses_only_the_cross_track_component(self):
        x_q_off, y_q_off, xy_p_off = self._build_qp_with_obstacle_directly_ahead(
            cross_track_suppression_enabled=False
        )
        x_q_on, y_q_on, xy_p_on = self._build_qp_with_obstacle_directly_ahead(
            cross_track_suppression_enabled=True
        )

        # Disabled: the small residual lateral offset produces a clearly
        # nonzero y-pull and x/y coupling (proving this isn't a hollow test).
        self.assertGreater(abs(y_q_off), 1.0e-3)
        self.assertGreater(abs(xy_p_off), 1.0e-3)
        # Enabled: both vanish (offset is within the suppression bound)...
        self.assertAlmostEqual(y_q_on, 0.0, places=9)
        self.assertAlmostEqual(xy_p_on, 0.0, places=9)
        # ...while the along-track (braking) term is completely unaffected.
        self.assertAlmostEqual(x_q_off, x_q_on, places=9)


class MinimumProgressContractTests(unittest.TestCase):
    @staticmethod
    def _mpc(profile="intersection_turn"):
        mpc = object.__new__(MPC)
        mpc.constraints = types.SimpleNamespace(
            min_velocity_mps=0.0,
            max_velocity_mps=15.0,
        )
        mpc.minimum_progress_enabled = True
        mpc.active_cost_profile_name = str(profile)
        mpc.turn_minimum_progress_speed_mps = 1.5
        mpc.turn_minimum_progress_ramp_accel_mps2 = 0.6
        mpc.dt_s = 0.1
        return mpc

    def test_turn_floor_holds_progress_when_already_moving(self):
        floor = self._mpc()._minimum_progress_lower_bound_mps(
            current_speed_mps=5.0,
            future_state_index=4,
        )
        self.assertAlmostEqual(floor, 1.5)

    def test_turn_floor_ramps_reachably_from_rest(self):
        mpc = self._mpc()
        first = mpc._minimum_progress_lower_bound_mps(
            current_speed_mps=0.0,
            future_state_index=1,
        )
        tenth = mpc._minimum_progress_lower_bound_mps(
            current_speed_mps=0.0,
            future_state_index=10,
        )
        self.assertAlmostEqual(first, 0.06)
        self.assertAlmostEqual(tenth, 0.6)

    def test_floor_is_inactive_outside_turn(self):
        floor = self._mpc(profile="lane_follow")._minimum_progress_lower_bound_mps(
            current_speed_mps=5.0,
            future_state_index=4,
        )
        self.assertAlmostEqual(floor, 0.0)


if __name__ == "__main__":
    unittest.main()
