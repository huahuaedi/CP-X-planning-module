import math
import types
import unittest

from pipeline.reference_generator import ReferenceGenerator


def _body_frame_xy(
    *,
    origin_x_m,
    origin_y_m,
    heading_rad,
    target_x_m,
    target_y_m,
):
    dx_m = float(target_x_m) - float(origin_x_m)
    dy_m = float(target_y_m) - float(origin_y_m)
    cos_h = math.cos(float(heading_rad))
    sin_h = math.sin(float(heading_rad))
    return (
        dx_m * cos_h + dy_m * sin_h,
        -dx_m * sin_h + dy_m * cos_h,
    )


class ReferenceGeneratorTests(unittest.TestCase):
    @staticmethod
    def _generator(config=None, horizon_steps=20):
        return ReferenceGenerator(
            config=dict(config or {}),
            mpc=types.SimpleNamespace(dt_s=0.1, horizon_steps=horizon_steps),
            map_planner=None,
            map_waypoint_from_location=lambda _location: None,
            lane_id_at_location=lambda _location: 1,
            body_frame_xy=_body_frame_xy,
            target_speed_mps=3.0,
            lookahead_m=18.0,
        )

    def test_straight_reference_has_stable_horizon_and_spacing(self):
        generator = self._generator(horizon_steps=5)
        ego = types.SimpleNamespace(x=2.0, y=-1.0, z=0.0)

        reference = generator._straight_reference_samples(
            ego_location=ego,
            ego_yaw_rad=0.0,
            current_lane_id=2,
            horizon_steps=5,
            step_distance_m=0.4,
        )

        self.assertEqual(len(reference), 6)
        self.assertTrue(all(int(row["lane_id"]) == 2 for row in reference))
        self.assertAlmostEqual(reference[0]["x_ref_m"], 2.5)
        self.assertAlmostEqual(reference[-1]["x_ref_m"], 5.0)
        self.assertTrue(all(abs(row["y_ref_m"] + 1.0) < 1.0e-9 for row in reference))

    def test_corridor_projection_interpolates_segment_heading(self):
        generator = self._generator()
        samples = [
            {
                "x_ref_m": 0.0,
                "y_ref_m": 0.0,
                "corridor_center_x_m": 0.0,
                "corridor_center_y_m": 0.0,
                "corridor_heading_rad": 0.0,
                "lane_width_m": 3.5,
            },
            {
                "x_ref_m": 1.0,
                "y_ref_m": 0.0,
                "corridor_center_x_m": 1.0,
                "corridor_center_y_m": 0.0,
                "corridor_heading_rad": 0.2,
                "lane_width_m": 3.5,
            },
        ]

        projection = generator.project_reference_corridor(
            reference_samples=samples,
            x_m=0.5,
            y_m=0.0,
            heading_rad=0.1,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
        )

        self.assertTrue(projection.valid)
        self.assertAlmostEqual(projection.segment_ratio, 0.5)
        self.assertAlmostEqual(projection.conditioned_heading_rad, 0.1)
        self.assertFalse(projection.continuity_limited)

    def test_corridor_projection_limits_rolling_window_heading_jump(self):
        generator = self._generator()

        def samples(start_x, heading):
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

        first = generator.project_reference_corridor(
            reference_samples=samples(0.0, 0.0),
            x_m=0.9,
            y_m=0.0,
            heading_rad=0.0,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            max_heading_step_rad=0.04,
        )
        second = generator.project_reference_corridor(
            reference_samples=samples(1.0, 0.11),
            x_m=1.01,
            y_m=0.0,
            heading_rad=0.0,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            max_heading_step_rad=0.04,
        )

        self.assertAlmostEqual(first.conditioned_heading_rad, 0.0)
        self.assertAlmostEqual(second.raw_heading_rad, 0.11)
        self.assertAlmostEqual(second.conditioned_heading_rad, 0.04)
        self.assertTrue(second.continuity_limited)

    def test_corridor_projection_uses_tracking_heading_not_stale_map_lane_during_maneuver(self):
        # Maneuver-geometry samples (lane change / turn) carry no
        # corridor_center_* tags, so project_reference_corridor used to fall
        # back to a live nearest-lane map lookup for its offset/heading
        # decomposition. Mid-lane-change this is ambiguous -- the queried
        # point can snap to a map lane whose heading has nothing to do with
        # the vehicle's actual, smoothly rotating commanded heading, which
        # produced a large phantom "offset" tracking the vehicle's own yaw
        # change rather than any real drift. Confirm the fix: ego sitting
        # exactly on the (linearly interpolated) tracking path reports ~0
        # lateral offset and ~0 heading error, even when the mocked map
        # lookup would disagree sharply (flat heading=0, far-off center).
        generator = self._generator()

        def stale_flat_map_waypoint(location):
            return types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=location.x, y=0.0),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=3.5,
            )

        generator._map_waypoint_callback = stale_flat_map_waypoint
        samples = [
            {
                "x_ref_m": 1.0,
                "y_ref_m": 0.3,
                "heading_rad": 0.2,
                "lane_width_m": 3.5,
            },
            {
                "x_ref_m": 2.0,
                "y_ref_m": 0.6,
                "heading_rad": 0.5,
                "lane_width_m": 3.5,
            },
        ]

        projection = generator.project_reference_corridor(
            reference_samples=samples,
            x_m=1.5,
            y_m=0.45,
            heading_rad=0.35,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
        )

        self.assertTrue(projection.valid)
        self.assertAlmostEqual(projection.occupancy.lateral_offset_m, 0.0, places=6)
        self.assertAlmostEqual(projection.occupancy.heading_error_rad, 0.0, places=6)

    def test_corridor_projection_limits_a_same_tick_segment_flip_jump(self):
        # Two parallel segments 2.16m apart (source lane vs target lane
        # during a lane change), both in the same candidate list. Tick 1:
        # ego sits near the first segment, so it's picked. Tick 2: ego has
        # moved close enough to the second segment that the argmin
        # legitimately flips to it -- the raw jump (1.6m) is under
        # continuity_reset_distance_m (2.5, so it isn't treated as a full
        # reset) but is far larger than one tick of realistic driving
        # motion. Before this fix, position was accepted as-is whenever the
        # jump was under continuity_reset_distance_m; only heading was ever
        # rate-limited. Confirm position is now rate-limited the same way.
        generator = self._generator()
        samples = [
            {"x_ref_m": 0.0, "y_ref_m": 0.0, "corridor_center_x_m": 0.0, "corridor_center_y_m": 0.0, "corridor_heading_rad": 0.0, "lane_width_m": 3.5},
            {"x_ref_m": 1.0, "y_ref_m": 0.0, "corridor_center_x_m": 1.0, "corridor_center_y_m": 0.0, "corridor_heading_rad": 0.0, "lane_width_m": 3.5},
            {"x_ref_m": 0.0, "y_ref_m": 2.16, "corridor_center_x_m": 0.0, "corridor_center_y_m": 2.16, "corridor_heading_rad": 0.0, "lane_width_m": 3.5},
            {"x_ref_m": 1.0, "y_ref_m": 2.16, "corridor_center_x_m": 1.0, "corridor_center_y_m": 2.16, "corridor_heading_rad": 0.0, "lane_width_m": 3.5},
        ]

        first = generator.project_reference_corridor(
            reference_samples=samples,
            x_m=0.5, y_m=0.3,
            heading_rad=0.0,
            ego_half_width_m=1.0, ego_half_length_m=2.4,
            max_position_step_m=0.5,
        )
        second = generator.project_reference_corridor(
            reference_samples=samples,
            x_m=0.5, y_m=1.9,
            heading_rad=0.0,
            ego_half_width_m=1.0, ego_half_length_m=2.4,
            max_position_step_m=0.5,
        )

        self.assertAlmostEqual(first.projected_y_m, 0.0, places=3)
        # The *conditioned* (returned) position must move at most
        # max_position_step_m (0.5) from the previous tick's position, not
        # jump straight to the new segment (y=2.16), even though the raw
        # argmin has legitimately flipped to it.
        self.assertTrue(second.continuity_limited)
        self.assertLessEqual(
            math.hypot(
                second.projected_x_m - first.projected_x_m,
                second.projected_y_m - first.projected_y_m,
            ),
            0.5 + 1e-6,
        )

    def test_boundary_recovery_reference_is_ego_anchored_and_time_spaced(self):
        generator = self._generator(
            config={"boundary_recovery_first_arc_m": 0.25},
            horizon_steps=20,
        )
        ego = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        base = []
        for index in range(16):
            x_m = 0.25 * float(index + 1)
            y_m = 0.08 * max(0.0, float(index - 3)) ** 1.25
            base.append({
                "x_ref_m": x_m,
                "y_ref_m": y_m,
                "heading_rad": math.atan2(
                    0.10 * max(0.0, float(index - 3)),
                    1.0,
                ),
                "lane_id": 1,
                "lane_width_m": 3.5,
                "corridor_center_x_m": x_m,
                "corridor_center_y_m": y_m,
                "corridor_heading_rad": 0.0,
            })

        generated = generator.build_boundary_recovery(
            ego_location=ego,
            ego_yaw_rad=0.0,
            current_lane_id=1,
            base_reference_samples=base,
            target_speed_mps=0.55,
            horizon_steps=20,
            dt_s=0.1,
        )

        self.assertEqual(len(generated.samples), 20)
        self.assertAlmostEqual(
            math.hypot(
                generated.samples[0]["x_ref_m"],
                generated.samples[0]["y_ref_m"],
            ),
            0.25,
            places=2,
        )
        self.assertLess(
            abs(float(generated.samples[0]["heading_rad"])),
            math.radians(12.0),
        )
        spacings = [
            math.hypot(
                float(second["x_ref_m"]) - float(first["x_ref_m"]),
                float(second["y_ref_m"]) - float(first["y_ref_m"]),
            )
            for first, second in zip(
                generated.samples[:-1],
                generated.samples[1:],
            )
        ]
        self.assertTrue(all(0.045 <= spacing <= 0.065 for spacing in spacings))
        self.assertTrue(
            all(
                abs(float(sample["speed_ref_mps"]) - 0.55) < 1.0e-9
                for sample in generated.samples
            )
        )

    def test_drivable_footprint_checks_points_against_lane_union(self):
        generator = self._generator()

        def waypoint(location):
            return types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(
                        x=location.x,
                        y=location.y,
                    ),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=4.0,
            )

        generator._drivable_waypoint_callback = waypoint
        occupancy = generator.drivable_footprint_occupancy(
            x_m=0.0,
            y_m=0.0,
            heading_rad=math.radians(35.0),
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
        )

        self.assertTrue(occupancy.valid)
        self.assertTrue(occupancy.inside)
        self.assertEqual(occupancy.checked_point_count, 8)
        self.assertAlmostEqual(occupancy.min_clearance_m, 1.85)

    def test_drivable_footprint_rejects_an_outside_corner(self):
        generator = self._generator()

        def waypoint(location):
            if float(location.x) > 1.0:
                return None
            return types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(
                        x=location.x,
                        y=location.y,
                    ),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=3.5,
            )

        generator._drivable_waypoint_callback = waypoint
        occupancy = generator.drivable_footprint_occupancy(
            x_m=0.0,
            y_m=0.0,
            heading_rad=0.0,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
        )

        self.assertTrue(occupancy.valid)
        self.assertFalse(occupancy.inside)
        self.assertLess(occupancy.min_clearance_m, 0.0)
        self.assertIn("outside_driving_lane", occupancy.reason)


    def test_missing_stop_target_uses_ego_heading_hard_lock(self):
        generator = self._generator(horizon_steps=6)
        ego = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

        generated = generator.stop_reference(
            ego_location=ego,
            ego_yaw_rad=0.0,
            current_state=[0.0, 0.0, 1.0, 0.0],
            current_lane_id=1,
            stop_target=None,
            fallback_destination_state=[],
            ego_speed_mps=1.0,
        )
        reference = generated.samples
        destination = generated.destination_state
        reason = generated.reason

        self.assertEqual(len(reference), 7)
        self.assertEqual(float(destination[2]), 0.0)
        self.assertIn("stop_missing_target_hard_lock", reason)
        self.assertTrue(all(float(row["speed_ref_mps"]) == 0.0 for row in reference))

    def test_polyline_smoothing_produces_forward_monotonic_samples(self):
        generator = self._generator(horizon_steps=8)
        reference = generator._smooth_reference_polyline_samples(
            raw_samples=[
                {"x_ref_m": 0.0, "y_ref_m": 0.0, "lane_id": 1},
                {"x_ref_m": 2.0, "y_ref_m": 0.0, "lane_id": 1},
                {"x_ref_m": 4.0, "y_ref_m": 1.0, "lane_id": 1},
                {"x_ref_m": 6.0, "y_ref_m": 3.0, "lane_id": 1},
            ],
            horizon_steps=8,
            step_distance_m=0.5,
            fallback_heading_rad=0.0,
        )

        self.assertEqual(len(reference), 9)
        progress = [
            math.hypot(float(row["x_ref_m"]), float(row["y_ref_m"]))
            for row in reference
        ]
        self.assertTrue(all(b > a for a, b in zip(progress[:-1], progress[1:])))

    def test_lane_recovery_quintic_starts_ahead_of_ego(self):
        generator = self._generator(
            config={"lane_recovery_anchor_forward_m": 0.8},
            horizon_steps=10,
        )
        lane_samples = [
            {
                "x_ref_m": 1.0 + 0.4 * index,
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_width_m": 3.5,
            }
            for index in range(14)
        ]
        generator._current_lane_center_reference_samples = (
            lambda **_kwargs: [dict(row) for row in lane_samples]
        )
        ego = types.SimpleNamespace(x=0.0, y=0.5, z=0.0)

        reference = generator._ego_anchored_lane_recovery_reference_samples(
            ego_location=ego,
            ego_yaw_rad=0.0,
            start_waypoint=object(),
            current_lane_id=1,
            horizon_steps=10,
            step_distance_m=0.4,
            route_points=[],
        )

        self.assertEqual(len(reference), 10)
        first_forward_m, _ = _body_frame_xy(
            origin_x_m=ego.x,
            origin_y_m=ego.y,
            heading_rad=0.0,
            target_x_m=reference[0]["x_ref_m"],
            target_y_m=reference[0]["y_ref_m"],
        )
        self.assertGreaterEqual(first_forward_m, 0.79)

    def test_turn_conditioner_limits_curvature_to_vehicle_contract(self):
        generator = self._generator(horizon_steps=8)
        ego = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        raw = [
            {
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "heading_rad": 0.0,
                "lane_id": 1,
            }
            for x_m, y_m in (
                (0.5, 0.0),
                (1.0, 0.0),
                (1.3, 0.4),
                (1.3, 0.9),
                (1.3, 1.4),
                (1.3, 1.9),
            )
        ]

        shaped, reason = generator.curvature_feasible_turn_samples(
            reference_samples=raw,
            ego_location=ego,
            ego_heading_rad=0.0,
            max_curvature_1pm=0.20,
        )

        self.assertIn("curvature_feasible_turn", reason)
        self.assertLessEqual(
            generator._max_discrete_curvature_1pm(shaped),
            0.201,
        )
        self.assertTrue(
            all(bool(sample["reference_curvature_limited"]) for sample in shaped)
        )

    def test_discrete_curvature_1pm_matches_internal_computation(self):
        generator = self._generator(horizon_steps=8)
        raw = [
            {"x_ref_m": float(x_m), "y_ref_m": float(y_m), "heading_rad": 0.0}
            for x_m, y_m in (
                (0.5, 0.0),
                (1.0, 0.0),
                (1.3, 0.4),
                (1.3, 0.9),
                (1.3, 1.4),
                (1.3, 1.9),
            )
        ]

        self.assertEqual(
            generator.discrete_curvature_1pm(raw),
            generator._max_discrete_curvature_1pm(raw),
        )
        self.assertGreater(generator.discrete_curvature_1pm(raw), 0.0)

    def test_turn_swept_footprint_corrects_centerline_offset(self):
        generator = self._generator(
            config={"turn_swept_footprint_correction_padding_m": 0.05},
            horizon_steps=6,
        )

        def waypoint(location):
            return types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=location.x, y=0.0),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=3.5,
            )

        generator._map_waypoint_callback = waypoint
        raw = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 0.9,
                "heading_rad": 0.0,
                "lane_id": 1,
                "lane_width_m": 3.5,
            }
            for index in range(8)
        ]

        corrected, validation, reason = generator.ensure_turn_swept_footprint(
            reference_samples=raw,
            horizon_steps=6,
            step_distance_m=0.5,
            fallback_heading_rad=0.0,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
            max_violations=0,
        )

        self.assertTrue(validation.valid)
        self.assertIn("turn_swept_footprint_corrected", reason)
        self.assertLess(
            max(abs(float(row["y_ref_m"])) for row in corrected),
            0.60,
        )

    def test_turn_swept_footprint_rejects_impossibly_narrow_corridor(self):
        generator = self._generator(horizon_steps=6)

        def waypoint(location):
            return types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=location.x, y=0.0),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=2.0,
            )

        generator._map_waypoint_callback = waypoint
        raw = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
                "lane_width_m": 2.0,
            }
            for index in range(8)
        ]

        _, validation, reason = generator.ensure_turn_swept_footprint(
            reference_samples=raw,
            horizon_steps=6,
            step_distance_m=0.5,
            fallback_heading_rad=0.0,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
            max_violations=0,
        )

        self.assertFalse(validation.valid)
        self.assertIn("outside_corridor", reason)

    def test_turn_swept_footprint_prefers_drivable_lane_union(self):
        generator = self._generator(horizon_steps=6)

        # The nearest single-lane strip is intentionally too narrow for a
        # rotated vehicle, while every footprint point belongs to one of the
        # junction's incoming/connector/outgoing driving lanes.
        def narrow_waypoint(location):
            return types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=location.x, y=0.0),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=2.0,
            )

        def drivable_waypoint(location):
            return types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=location.x, y=location.y),
                    rotation=types.SimpleNamespace(yaw=45.0),
                ),
                lane_width=3.5,
            )

        generator._map_waypoint_callback = narrow_waypoint
        generator._drivable_waypoint_callback = drivable_waypoint
        raw = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 0.0,
                "heading_rad": math.radians(45.0),
                "lane_id": 1,
            }
            for index in range(8)
        ]

        unchanged, validation, reason = generator.ensure_turn_swept_footprint(
            reference_samples=raw,
            horizon_steps=6,
            step_distance_m=0.5,
            fallback_heading_rad=0.0,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
            max_violations=0,
        )

        self.assertTrue(validation.valid)
        self.assertIn("drivable_union_valid", reason)
        self.assertEqual(unchanged, raw)

    def test_turn_union_seam_tolerance_is_small_and_ratio_bounded(self):
        generator = self._generator(
            config={
                "turn_drivable_union_seam_max_pose_violations": 3,
                "turn_drivable_union_seam_max_violation_ratio": 0.10,
            },
            horizon_steps=20,
        )
        generator._drivable_waypoint_callback = lambda location: (
            None
            if 9.45 <= float(location.x) <= 9.55
            else types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(
                        x=location.x,
                        y=location.y,
                    ),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=3.5,
            )
        )
        raw = [
            {
                "x_ref_m": 0.5 * float(index + 1),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
            }
            for index in range(20)
        ]
        validation = generator.validate_turn_swept_footprint(
            reference_samples=raw,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
        )
        self.assertTrue(validation.valid)
        self.assertIn("seam_tolerated", validation.reason)

        generator.config["turn_drivable_union_seam_max_pose_violations"] = 0
        rejected = generator.validate_turn_swept_footprint(
            reference_samples=raw,
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
        )
        self.assertFalse(rejected.valid)
        self.assertIn("outside_drivable_union", rejected.reason)


class BoundaryRecoveryProgressTests(unittest.TestCase):
    """validate_boundary_recovery_progress's worsening tolerance must scale
    with how far off the corridor the recovery is starting from -- a fixed
    tolerance sized for a small violation makes recovery from a large one
    (e.g. after several turn-connector smoothing passes have cut the corner)
    permanently unpassable, and reference_pipeline.py does not retry once
    already inside the boundary-recovery path, so that reads as the vehicle
    stopping forever. See reference_generator.py::validate_boundary_recovery_progress.
    """

    @staticmethod
    def _generator():
        return ReferenceGenerator(
            config={},
            mpc=types.SimpleNamespace(dt_s=0.1, horizon_steps=20),
            map_planner=None,
            map_waypoint_from_location=lambda _location: None,
            lane_id_at_location=lambda _location: 1,
            body_frame_xy=_body_frame_xy,
            target_speed_mps=3.0,
            lookahead_m=18.0,
        )

    @staticmethod
    def _stub_clearances(generator, clearances):
        def _lane_corridor_occupancy(**kwargs):
            del kwargs
            index = _lane_corridor_occupancy.calls
            _lane_corridor_occupancy.calls += 1
            return types.SimpleNamespace(
                valid=True,
                footprint_clearance_m=float(clearances[min(index, len(clearances) - 1)]),
            )

        _lane_corridor_occupancy.calls = 0
        generator.lane_corridor_occupancy = _lane_corridor_occupancy

    def _samples(self, count):
        return [
            {"x_ref_m": float(index), "y_ref_m": 0.0, "heading_rad": 0.0}
            for index in range(count)
        ]

    def test_scaled_tolerance_accepts_a_recovery_dip_proportional_to_the_initial_violation(self):
        """Starting 0.5m past the boundary, a recovery that briefly dips to
        -0.7m before improving is a perfectly reasonable path -- it needs
        room to swing back -- so it should pass even though the dip (0.2m)
        exceeds the base 0.08m tolerance on its own."""

        generator = self._generator()
        self._stub_clearances(generator, [-0.5, -0.7, -0.3, -0.1])

        result = generator.validate_boundary_recovery_progress(
            reference_samples=self._samples(4),
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
            max_worsening_m=0.08,
            min_terminal_improvement_m=0.03,
        )

        # already_off_m=0.5 -> effective tolerance = 0.08 + 0.6*0.5 = 0.38,
        # and the dip is only 0.2m (-0.5 -> -0.7), so this passes.
        self.assertTrue(result.valid, result.reason)

    def test_small_violation_still_uses_a_tight_tolerance(self):
        """A recovery that starts only slightly off the corridor should not
        get a much larger worsening allowance -- the scaling only matters
        once the starting violation itself is large."""

        generator = self._generator()
        self._stub_clearances(generator, [-0.02, -0.30, -0.10, 0.05])

        result = generator.validate_boundary_recovery_progress(
            reference_samples=self._samples(4),
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
            max_worsening_m=0.08,
        )

        # already_off_m=0.02 -> effective tolerance = 0.08 + 0.6*0.02 = 0.092,
        # the dip is 0.28m, still well outside even the scaled tolerance.
        self.assertFalse(result.valid)

    def test_does_not_worsen_relative_to_a_safe_start(self):
        """A recovery that starts already safe (>=0) must not use the
        violation-scaled slack at all -- the scaling only exists to make
        genuine off-corridor recovery possible, not to loosen the guard for
        paths that had no need to dip in the first place."""

        generator = self._generator()
        self._stub_clearances(generator, [0.05, -0.50, 0.10, 0.20])

        result = generator.validate_boundary_recovery_progress(
            reference_samples=self._samples(4),
            ego_half_width_m=1.0,
            ego_half_length_m=2.4,
            safety_margin_m=0.15,
            max_worsening_m=0.08,
        )

        self.assertFalse(result.valid)


if __name__ == "__main__":
    unittest.main()
