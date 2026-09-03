"""Tests for pipeline/reference_geometry.py and pipeline/route_geometry.py."""
import math
import unittest

from opencda.planning_module.pipeline import reference_geometry as rg
from opencda.planning_module.pipeline.route_geometry import (
    JUNCTION_TURN,
    LANE_CHANGE,
    LANE_FOLLOW,
    RouteGeometry,
)


def _straight(n, spacing, x0=0.0, y0=0.0, heading=0.0):
    return [
        (x0 + i * spacing * math.cos(heading), y0 + i * spacing * math.sin(heading))
        for i in range(n)
    ]


class CurvatureSpacingInvarianceTests(unittest.TestCase):
    def _kink(self, spacing, deg=60.0):
        arm = _straight(int(10 / spacing) + 1, spacing)
        hx, hy = arm[-1]
        ang = math.radians(deg)
        turn = [
            (hx + i * spacing * math.cos(ang), hy + i * spacing * math.sin(ang))
            for i in range(1, int(10 / spacing) + 1)
        ]
        return arm + turn

    def test_curvature_is_spacing_invariant_within_the_window(self):
        k_dense = rg.max_curvature_1pm(self._kink(0.1))
        k_mid = rg.max_curvature_1pm(self._kink(1.0))
        k_coarse = rg.max_curvature_1pm(self._kink(3.0))
        # Old adjacent-sample d(theta)/ds gave ~10 /m at 0.1 m spacing.
        self.assertLess(k_dense, 1.0)
        # All three saturate at ~ (kink angle) / eval_arc, independent of spacing.
        expected = math.radians(60.0) / rg.CURVATURE_EVAL_ARC_M
        for k in (k_dense, k_mid, k_coarse):
            self.assertAlmostEqual(k, expected, delta=0.08)

    def test_straight_line_has_zero_curvature(self):
        self.assertEqual(rg.max_curvature_1pm(_straight(30, 1.0)), 0.0)


class ResamplePolylineTests(unittest.TestCase):
    def test_resample_is_uniform_and_spacing_independent(self):
        # both polylines span exactly 16 m
        coarse = rg.resample_polyline(_straight(5, 4.0), 1.0)
        fine = rg.resample_polyline(_straight(65, 0.25), 1.0)
        self.assertEqual(len(coarse), len(fine))
        for i in range(len(coarse) - 1):
            self.assertAlmostEqual(
                math.hypot(coarse[i + 1][0] - coarse[i][0],
                           coarse[i + 1][1] - coarse[i][1]),
                1.0, delta=1e-6,
            )

    def test_endpoints_preserved(self):
        pts = _straight(10, 3.3)
        out = rg.resample_polyline(pts, 1.0)
        self.assertAlmostEqual(out[0][0], pts[0][0])
        self.assertAlmostEqual(out[-1][0], pts[-1][0], delta=1e-6)


class StableReferenceLineTests(unittest.TestCase):
    def test_heading_and_curvature_use_real_arc_length(self):
        radius = 20.0
        angles = [0.0, 0.01, 0.05, 0.12, 0.25, 0.42, 0.65]
        points = [(radius * math.sin(a), radius * (1.0 - math.cos(a))) for a in angles]
        line = rg.build_reference_line(points, spacing_m=0.5)
        self.assertTrue(line.valid)
        self.assertTrue(all(b.s_m > a.s_m for a, b in zip(line.points, line.points[1:])))
        interior = [abs(p.curvature_1pm) for p in line.points[3:-3]]
        self.assertAlmostEqual(sum(interior) / len(interior), 1.0 / radius, delta=0.015)

    def test_frenet_lane_change_returns_xy_for_mpc(self):
        source = [
            {"x_ref_m": float(i), "y_ref_m": 0.0, "lane_id": 101, "lane_width_m": 3.5}
            for i in range(31)
        ]
        line = rg.build_reference_line(source, spacing_m=1.0)
        path = rg.frenet_lane_change_path(
            line,
            lateral_offset_m=3.5,
            transition_length_m=20.0,
            target_lane_id=202,
            target_speed_mps=8.0,
        )
        self.assertGreaterEqual(len(path), 30)
        self.assertAlmostEqual(path[0]["y_ref_m"], 0.0, delta=1.0e-6)
        self.assertAlmostEqual(path[-1]["y_ref_m"], 3.5, delta=1.0e-6)
        self.assertAlmostEqual(path[-1]["heading_rad"], 0.0, delta=0.02)
        self.assertEqual(path[-1]["lane_id"], 202)
        self.assertTrue(all(
            sample["lane_transition_kind"] == "lateral_lane_change"
            for sample in path
        ))

    def test_frenet_prefix_is_valid_during_receding_horizon_lane_change(self):
        from opencda.planning_module.pipeline.reference_contract import (
            contract_from_config,
            validate_reference_contract,
        )

        source = [
            {"x_ref_m": float(i), "y_ref_m": 0.0, "lane_id": 101}
            for i in range(81)
        ]
        path = rg.frenet_lane_change_path(
            rg.build_reference_line(source, spacing_m=1.0),
            lateral_offset_m=3.5,
            transition_length_m=50.0,
            target_lane_id=202,
            target_speed_mps=10.0,
        )
        prefix = path[1:25]
        destination = [prefix[7]["x_ref_m"], prefix[7]["y_ref_m"], 10.0, 0.0, 202]
        contract = contract_from_config(
            mode="lane_change",
            expected_lane_id=202,
            horizon_steps=20,
            config={},
            default_speed_mps=12.0,
        )
        result = validate_reference_contract(
            reference_samples=prefix,
            destination_state=destination,
            ego_state=[0.0, 0.0, 10.0, 0.0],
            contract=contract,
            check_destination_body_lateral=False,
        )
        self.assertNotIn("destination_lane_error_out_of_contract", result.violations)

    def test_frenet_lane_change_converges_to_curved_target_centerline(self):
        source = [
            {"x_ref_m": float(i), "y_ref_m": 0.0, "lane_id": 101}
            for i in range(41)
        ]
        target = [
            {
                "x_ref_m": float(i),
                "y_ref_m": 3.5 + 0.004 * max(0.0, float(i) - 20.0) ** 2,
                "lane_id": 202,
            }
            for i in range(41)
        ]
        source_line = rg.build_reference_line(source, spacing_m=1.0)
        target_line = rg.build_reference_line(target, spacing_m=1.0)
        path = rg.frenet_lane_change_path(
            source_line,
            lateral_offset_m=3.5,
            transition_length_m=20.0,
            target_lane_id=202,
            target_speed_mps=8.0,
            target_reference_line=target_line,
        )
        self.assertAlmostEqual(path[0]["y_ref_m"], 0.0, delta=1.0e-6)
        # Beyond the transition, the real curved target -- not source+3.5 --
        # owns both position and tangent.
        self.assertGreater(path[-1]["y_ref_m"], 4.8)
        self.assertGreater(path[-1]["heading_rad"], 0.1)
        self.assertEqual(path[-1]["lane_id"], 202)

    def test_frenet_lane_change_keeps_target_tail_after_short_source_ends(self):
        source = [{"x_ref_m": float(i), "y_ref_m": 0.0, "lane_id": 101}
                  for i in range(21)]
        target = [{"x_ref_m": float(i), "y_ref_m": 3.5, "lane_id": 202}
                  for i in range(41)]
        path = rg.frenet_lane_change_path(
            rg.build_reference_line(source, spacing_m=1.0),
            lateral_offset_m=3.5,
            transition_length_m=20.0,
            target_lane_id=202,
            target_speed_mps=8.0,
            target_reference_line=rg.build_reference_line(target, spacing_m=1.0),
        )
        self.assertGreater(len(path), len(source))
        self.assertAlmostEqual(path[-1]["x_ref_m"], 40.0, delta=1e-6)
        self.assertAlmostEqual(path[-1]["y_ref_m"], 3.5, delta=1e-6)
        self.assertEqual(path[-1]["lane_change_progress"], 1.0)

    def test_frenet_lane_change_accepts_target_shorter_than_source(self):
        source = [{"x_ref_m": float(i), "y_ref_m": 0.0, "lane_id": 101}
                  for i in range(41)]
        target = [{"x_ref_m": float(i), "y_ref_m": 3.5, "lane_id": 202}
                  for i in range(21)]

        path = rg.frenet_lane_change_path(
            rg.build_reference_line(source, spacing_m=1.0),
            lateral_offset_m=3.5,
            transition_length_m=20.0,
            target_lane_id=202,
            target_speed_mps=8.0,
            target_reference_line=rg.build_reference_line(target, spacing_m=1.0),
        )

        self.assertTrue(path)
        self.assertEqual(path[-1]["lane_id"], 202)
        self.assertAlmostEqual(path[-1]["y_ref_m"], 3.5, delta=1e-6)


class ProjectionTests(unittest.TestCase):
    def test_project_returns_arc_length_and_signed_lateral(self):
        pts = _straight(20, 1.0)  # along +x
        s_m, lat_m = rg.project_to_polyline(pts, 5.3, 2.0)
        self.assertAlmostEqual(s_m, 5.3, delta=1e-6)
        self.assertAlmostEqual(lat_m, 2.0, delta=1e-6)   # +y is left of +x travel

    def test_monotonic_lower_bound(self):
        pts = _straight(20, 1.0)
        s_m, _ = rg.project_to_polyline(pts, 3.0, 0.0, s_lower_m=8.0)
        self.assertGreaterEqual(s_m, 8.0)

    def test_projection_upper_bound_rejects_nearby_later_branch(self):
        # The final branch returns spatially close to the query, but a
        # persistent cursor at s=2 may only search the next 3 m this tick.
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (2.0, 1.0)]
        s_m, _ = rg.project_to_polyline(
            points,
            2.0,
            0.9,
            s_lower_m=2.0,
            s_upper_m=5.0,
        )
        self.assertLessEqual(s_m, 5.0 + 1e-9)


class RouteGeometrySegmentationTests(unittest.TestCase):
    def _entries(self, spec):
        """spec: list of (dx, dy, option). Builds (wp, option) entries."""
        entries = []
        x = y = 0.0
        for dx, dy, opt in spec:
            x += dx
            y += dy
            wp = type("WP", (), {"transform": type("T", (), {
                "location": type("L", (), {"x": x, "y": y, "z": 0.0})()})(),
                "lane_width_m": 3.5, "lane_id": -1, "is_junction": False})()
            entries.append((wp, opt))
        return entries

    def test_lane_follow_then_lane_change_then_turn(self):
        spec = (
            [(2.0, 0.0, "LANEFOLLOW")] * 10
            + [(2.0, 0.5, "CHANGELANERIGHT")] * 4
            + [(2.0, 0.0, "LANEFOLLOW")] * 6
            + [(1.4, 1.4, "RIGHT")] * 8
            + [(0.0, 2.0, "LANEFOLLOW")] * 6
        )
        geom = RouteGeometry.from_entries(self._entries(spec))
        self.assertTrue(geom.valid)
        kinds = [s.kind for s in geom.segments]
        self.assertIn(LANE_CHANGE, kinds)
        self.assertIn(JUNCTION_TURN, kinds)
        self.assertEqual(kinds[0], LANE_FOLLOW)

        prog = geom.project(1.0, 0.0)          # near the start
        lc = geom.next_lane_change(prog.s_m, 200.0)
        turn = geom.next_turn(prog.s_m, 200.0)
        self.assertIsNotNone(lc)
        self.assertEqual(lc[0], "right")
        self.assertIsNotNone(turn)
        self.assertEqual(turn[0], "right")
        # the lane change comes before the turn
        self.assertLess(lc[1], turn[1])

    def test_turn_surfaces_after_progress_passes_the_lane_change(self):
        spec = (
            [(2.0, 0.0, "LANEFOLLOW")] * 4
            + [(2.0, 0.6, "CHANGELANERIGHT")] * 3
            + [(2.0, 0.0, "LANEFOLLOW")] * 3
            + [(1.4, -1.4, "RIGHT")] * 8
        )
        geom = RouteGeometry.from_entries(self._entries(spec))
        # once ego is past the lane-change span, the next turn is still found
        # (the flat upcoming_turn used to return "" whenever a lane change was
        # anywhere in the lookahead window)
        past_lc_s = geom.segments[1].s_end_m + 1.0
        turn = geom.next_turn(past_lc_s, 100.0)
        self.assertIsNotNone(turn)
        self.assertEqual(turn[0], "right")

    def test_sample_spacing_is_fixed_regardless_of_node_density(self):
        spec = [(0.7, 0.0, "LANEFOLLOW")] * 60   # dense nodes
        geom = RouteGeometry.from_entries(self._entries(spec))
        poses = geom.sample(0.0, count=10, spacing_m=2.0)
        self.assertEqual(len(poses), 10)
        for i in range(len(poses) - 1):
            d = math.hypot(poses[i + 1].x_m - poses[i].x_m,
                           poses[i + 1].y_m - poses[i].y_m)
            self.assertAlmostEqual(d, 2.0, delta=1e-3)

    def test_sample_extrapolates_past_route_end(self):
        spec = [(1.0, 0.0, "LANEFOLLOW")] * 10
        geom = RouteGeometry.from_entries(self._entries(spec))
        poses = geom.sample(geom.total_m - 1.0, count=6, spacing_m=1.0)
        self.assertEqual(len(poses), 6)
        self.assertGreater(poses[-1].x_m, geom.total_m)  # extrapolated forward

    def test_junction_geometry_marks_admap_connector_without_angle_guess(self):
        def jwp(x, y):
            return type("WP", (), {"transform": type("T", (), {
                "location": type("L", (), {"x": x, "y": y, "z": 0.0})()})(),
                "lane_width_m": 3.5, "lane_id": 1, "is_junction": True})()

        def fwp(x, y):
            w = jwp(x, y)
            w.is_junction = False
            return w

        entries = [(fwp(i * 2.0, 0.0), "STRAIGHT") for i in range(6)]         # +x
        cx, cy = 10.0, 6.0                                                    # arc centre
        for k in range(1, 13):
            ang = -math.pi / 2 + k * (math.pi / 2) / 12                       # -pi/2 -> 0
            entries.append((jwp(cx + 6.0 * math.cos(ang),
                                cy + 6.0 * math.sin(ang)), "STRAIGHT"))
        for i in range(1, 6):                                                # now +y
            entries.append((fwp(16.0, 6.0 + i * 2.0), "STRAIGHT"))
        geom = RouteGeometry.from_entries(entries)
        turn = geom.next_turn(0.0, 200.0)
        self.assertIsNone(turn)
        self.assertTrue(
            any(segment.kind == "junction_connector" for segment in geom.segments)
        )

    def test_lane_change_segment_carries_exact_ad_lane_ids(self):
        entries = self._entries(
            [(1.0, 0.0, "LANEFOLLOW")] * 3
            + [(1.0, 0.0, "CHANGELANERIGHT")] * 2
            + [(1.0, 0.0, "LANEFOLLOW")] * 3
        )
        for index, (wp, _) in enumerate(entries):
            wp.ad_lane_id = 500144 if index >= 5 else 11640145
        geom = RouteGeometry.from_entries(entries)
        segment = geom.next_lane_change_segment(0.0, 100.0)
        self.assertIsNotNone(segment)
        self.assertEqual(segment.source_lane_id, 11640145)
        self.assertEqual(segment.target_lane_id, 500144)
        self.assertEqual(geom.waypoint_for_lane_id(500144).ad_lane_id, 500144)

    def test_opposite_lane_change_directions_are_distinct_topology_segments(self):
        entries = self._entries(
            [(1.0, 0.0, "LANEFOLLOW")] * 2
            + [(1.0, 0.4, "CHANGELANELEFT")] * 2
            + [(1.0, -0.4, "CHANGELANERIGHT")] * 2
            + [(1.0, 0.0, "LANEFOLLOW")] * 2
        )
        geom = RouteGeometry.from_entries(entries)

        lane_changes = [seg for seg in geom.segments if seg.kind == LANE_CHANGE]
        self.assertEqual([seg.direction for seg in lane_changes], ["left", "right"])

    def test_topology_validation_reports_ordered_macro_signature(self):
        entries = self._entries(
            [(2.0, 0.0, "LANEFOLLOW")] * 3
            + [(2.0, -0.5, "CHANGELANERIGHT")] * 2
            + [(2.0, 0.0, "LANEFOLLOW")] * 2
            + [(1.0, -1.0, "RIGHT")] * 4
            + [(0.0, -2.0, "LANEFOLLOW")] * 2
        )
        for index, (waypoint, _) in enumerate(entries):
            waypoint.ad_lane_id = 500144 if index >= 5 else 500145
        geom = RouteGeometry.from_entries(entries)

        validation = geom.validate_topology()

        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(
            validation.signature,
            (
                "lane_follow",
                "lane_change:right",
                "lane_follow",
                "junction_turn:right",
                "lane_follow",
            ),
        )

    def test_turn_direction_from_road_option_wins_over_sweep(self):
        # When a node carries an explicit RIGHT option it is honoured even if
        # the local sweep is small.
        def jwp(x, y, opt):
            w = type("WP", (), {"transform": type("T", (), {
                "location": type("L", (), {"x": x, "y": y, "z": 0.0})()})(),
                "lane_width_m": 3.5, "lane_id": 1, "is_junction": True})()
            return (w, opt)
        entries = ([jwp(i * 2.0, 0.0, "LANEFOLLOW") for i in range(4)]
                   + [jwp(8 + i * 2.0, i * 0.3, "RIGHT") for i in range(1, 6)]
                   + [jwp(18 + i * 2.0, 1.5, "LANEFOLLOW") for i in range(1, 4)])
        geom = RouteGeometry.from_entries(entries)
        turn = geom.next_turn(0.0, 100.0)
        self.assertIsNotNone(turn)
        self.assertEqual(turn[0], "right")

    def test_invalid_route_is_safe(self):
        geom = RouteGeometry.from_entries([])
        self.assertFalse(geom.valid)
        prog = geom.project(0.0, 0.0)
        self.assertTrue(prog.off_route)
        self.assertIsNone(geom.next_turn(0.0, 50.0))


if __name__ == "__main__":
    unittest.main()
