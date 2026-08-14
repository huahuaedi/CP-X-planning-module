import math
import types
import unittest

import carla

from pipeline.reference_gate import FinalReferenceGate
from pipeline.reference_generator import ReferenceGenerator
from pipeline.reference_pipeline import ReferencePipeline, ReferencePipelineRequest


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
    return (
        math.cos(heading_rad) * dx_m + math.sin(heading_rad) * dy_m,
        -math.sin(heading_rad) * dx_m + math.cos(heading_rad) * dy_m,
    )


class ReferencePipelineSweptFootprintTests(unittest.TestCase):
    @staticmethod
    def _pipeline(lane_width_m, extra_config=None):
        config = {
            "reference_vehicle_half_width_m": 1.0,
            "reference_vehicle_half_length_m": 2.4,
            "reference_contract_turn_boundary_margin_m": 0.15,
            "reference_contract_turn_max_boundary_failures": 0,
            "reference_contract_intersection_turn_min_first_forward_m": 0.2,
        }
        config.update(extra_config or {})

        def waypoint(location):
            return types.SimpleNamespace(
                transform=types.SimpleNamespace(
                    location=types.SimpleNamespace(x=location.x, y=0.0),
                    rotation=types.SimpleNamespace(yaw=0.0),
                ),
                lane_width=float(lane_width_m),
            )

        mpc = types.SimpleNamespace(dt_s=0.1, horizon_steps=4)
        generator = ReferenceGenerator(
            config=config,
            mpc=mpc,
            map_planner=None,
            map_waypoint_from_location=waypoint,
            lane_id_at_location=lambda _location: 1,
            body_frame_xy=_body_frame_xy,
            target_speed_mps=2.0,
            lookahead_m=18.0,
        )
        return ReferencePipeline(
            config=config,
            generator=generator,
            final_gate=FinalReferenceGate(config),
            horizon_steps=4,
            dt_s=0.1,
            default_speed_mps=2.0,
        )

    @staticmethod
    def _request():
        reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
                "speed_ref_mps": 1.0,
            }
            for index in range(4)
        ]
        return ReferencePipelineRequest(
            destination_state=[4.0, 0.0, 1.0, 0.0, 1],
            reference_samples=reference,
            current_state=[0.0, 0.0, 1.0, 0.0],
            ego_location=carla.Location(x=0.0, y=0.0, z=0.0),
            ego_yaw_rad=0.0,
            ego_speed_mps=1.0,
            target_speed_mps=1.0,
            behavior_decision="intersection_turn_right",
            behavior_fsm_state="INTERSECTION_TURN_RIGHT",
            current_lane_id=1,
            target_lane_id=1,
            stop_goal_active=False,
            route_points=(),
        )

    def test_pipeline_accepts_turn_with_valid_swept_footprint(self):
        result = self._pipeline(lane_width_m=3.5).finalize(self._request())

        self.assertTrue(result.accepted)

    def test_pipeline_rejects_turn_when_vehicle_cannot_fit_corridor(self):
        result = self._pipeline(lane_width_m=2.0).finalize(self._request())

        self.assertFalse(result.accepted)
        self.assertIn("turn_swept_footprint", result.gate.reason)

    def test_direct_target_tracking_widens_lane_change_first_lateral_limit(self):
        # condition()'s own _validate() call builds conditioned.validation,
        # which finalize() uses to *override* self.final_gate.validate()'s
        # result whenever it fails (reference_pipeline.py, finalize()). If
        # _validate() isn't also toggle-aware, that override silently
        # reintroduces the standard "lane_change" mode's tight 1.25m
        # first-lateral limit even though the top-level gate was widened --
        # exactly the bug this test locks in the fix for.
        pipeline = self._pipeline(
            lane_width_m=3.5,
            extra_config={
                "route_tracking_lane_change_direct_target_tracking_enabled": True
            },
        )
        reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.2,
                "heading_rad": 0.0,
                "lane_id": 2,
                "speed_ref_mps": 1.0,
            }
            for index in range(4)
        ]
        request = ReferencePipelineRequest(
            destination_state=[4.0, 3.2, 1.0, 0.0, 2],
            reference_samples=reference,
            current_state=[0.0, 0.0, 1.0, 0.0],
            ego_location=carla.Location(x=0.0, y=0.0, z=0.0),
            ego_yaw_rad=0.0,
            ego_speed_mps=1.0,
            target_speed_mps=1.0,
            behavior_decision="lane_change_right",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            current_lane_id=1,
            target_lane_id=2,
            stop_goal_active=False,
            route_points=(),
        )

        result = pipeline.finalize(request)

        self.assertTrue(result.accepted, result.gate.reason)

    def test_direct_target_tracking_toggle_off_still_rejects_wide_offset(self):
        # Regression guard: with the toggle off (today's default), the
        # standard "lane_change" mode's tighter limit must still apply.
        pipeline = self._pipeline(lane_width_m=3.5)
        reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 3.2,
                "heading_rad": 0.0,
                "lane_id": 2,
                "speed_ref_mps": 1.0,
            }
            for index in range(4)
        ]
        request = ReferencePipelineRequest(
            destination_state=[4.0, 3.2, 1.0, 0.0, 2],
            reference_samples=reference,
            current_state=[0.0, 0.0, 1.0, 0.0],
            ego_location=carla.Location(x=0.0, y=0.0, z=0.0),
            ego_yaw_rad=0.0,
            ego_speed_mps=1.0,
            target_speed_mps=1.0,
            behavior_decision="lane_change_right",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            current_lane_id=1,
            target_lane_id=2,
            stop_goal_active=False,
            route_points=(),
        )

        result = pipeline.finalize(request)

        self.assertFalse(result.accepted)
        self.assertIn("first_lateral_out_of_contract", result.gate.reason)

    def test_lane_change_destination_is_aligned_after_reference_cleaning(self):
        pipeline = self._pipeline(lane_width_m=3.5)
        reference = [
            {
                "x_ref_m": float(index + 1),
                "y_ref_m": 0.5 * float(index + 1),
                "heading_rad": 0.0,
                "lane_id": 2,
                "speed_ref_mps": 1.0,
            }
            for index in range(4)
        ]
        request = ReferencePipelineRequest(
            destination_state=[4.0, 3.5, 1.0, 0.0, 2],
            reference_samples=reference,
            current_state=[0.0, 0.0, 1.0, 0.0],
            ego_location=carla.Location(x=0.0, y=0.0, z=0.0),
            ego_yaw_rad=0.0,
            ego_speed_mps=1.0,
            target_speed_mps=1.0,
            behavior_decision="lane_change_left",
            behavior_fsm_state="EXECUTE_LANE_CHANGE_LEFT",
            current_lane_id=1,
            target_lane_id=2,
            stop_goal_active=False,
            route_points=(),
        )

        result = pipeline.finalize(request)

        self.assertTrue(result.accepted, result.gate.reason)
        self.assertEqual(result.destination_state[:2], [4.0, 2.0])
        self.assertIn(
            "lane_change_destination_aligned_to_reference",
            result.conditioning_reason,
        )

    def test_pipeline_accepts_improving_boundary_recovery_from_invalid_start(self):
        pipeline = self._pipeline(lane_width_m=3.5)
        pipeline.horizon_steps = 20
        pipeline.generator.mpc.horizon_steps = 20
        ego = carla.Location(x=0.0, y=0.30, z=0.0)
        base = [
            {
                "x_ref_m": 0.25 * float(index + 1),
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
                "lane_width_m": 3.5,
                "corridor_center_x_m": 0.25 * float(index + 1),
                "corridor_center_y_m": 0.0,
                "corridor_heading_rad": 0.0,
            }
            for index in range(20)
        ]
        generated = pipeline.generator.build_boundary_recovery(
            ego_location=ego,
            ego_yaw_rad=-0.20,
            current_lane_id=1,
            base_reference_samples=base,
            target_speed_mps=0.55,
            horizon_steps=20,
            dt_s=0.1,
        )
        request = ReferencePipelineRequest(
            destination_state=generated.destination_state,
            reference_samples=generated.samples,
            current_state=[0.0, 0.30, 0.2, -0.20],
            ego_location=ego,
            ego_yaw_rad=-0.20,
            ego_speed_mps=0.2,
            target_speed_mps=0.55,
            behavior_decision="intersection_turn_right",
            behavior_fsm_state="BOUNDARY_RECOVERY_RIGHT",
            current_lane_id=1,
            target_lane_id=1,
            stop_goal_active=False,
            route_points=(),
        )

        result = pipeline.finalize(request)

        self.assertTrue(result.accepted, result.gate.reason)
        self.assertIn(
            "boundary_recovery_progress:valid",
            result.conditioning_reason,
        )


if __name__ == "__main__":
    unittest.main()
