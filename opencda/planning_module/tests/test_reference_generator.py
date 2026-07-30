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
