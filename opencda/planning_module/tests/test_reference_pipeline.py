import unittest
from types import SimpleNamespace

from behavior_planner.reference_generator import select_reference_intent
from behavior_planner.reference_pipeline import (
    MpcReferenceGenerationContext,
    MpcReferenceGenerationOutput,
    build_mpc_reference_result,
    generate_mpc_reference,
    lane_center_destination_from_reference_arc_length,
    summarize_reference_pipeline_history,
    trace_reference_pipeline,
)


class ReferencePipelineTraceTests(unittest.TestCase):
    def test_turn_destination_uses_reference_arc_length(self):
        reference = [
            {
                "x_ref_m": 1.0,
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
            },
            {
                "x_ref_m": 2.0,
                "y_ref_m": 0.0,
                "heading_rad": 0.0,
                "lane_id": 1,
            },
            {
                "x_ref_m": 2.0,
                "y_ref_m": 2.0,
                "heading_rad": 1.5708,
                "lane_id": 2,
            },
            {
                "x_ref_m": 2.0,
                "y_ref_m": 4.0,
                "heading_rad": 1.5708,
                "lane_id": 2,
            },
        ]

        destination = lane_center_destination_from_reference_arc_length(
            destination_state=[20.0, 0.0, 1.5, 0.0, 1],
            lane_center_reference=reference,
            target_arc_length_m=2.5,
        )

        self.assertEqual(destination[:2], [2.0, 2.0])
        self.assertAlmostEqual(destination[3], 1.5708)
        self.assertEqual(destination[4], 2.0)

    def test_generate_mpc_reference_accepts_context_object(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=1,
            global_route_reference_allowed=False,
            traffic_control_lane_lock_active=False,
        )
        world_map = SimpleNamespace(get_waypoint=lambda *args, **kwargs: None)
        carla = SimpleNamespace(
            LaneType=SimpleNamespace(Driving=1),
            Location=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        ego_transform = SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )
        context = MpcReferenceGenerationContext(
            world_map=world_map,
            carla=carla,
            ego_transform=ego_transform,
            ego_state=[0.0, 0.0, 2.0, 0.0],
            active_global_route_points=[],
            previous_lane_center_reference=None,
            behavior_runtime_cfg={},
            reference_intent=intent,
            current_applied_behavior="lane_follow",
            cached_planner_lc_state="LANE_KEEP",
            reference_target_lane_id=1,
            current_lane_id=1,
            global_route_reference_allowed=False,
            global_route_reference_gate_reason="",
            should_follow_global_route_lane_for_reference=False,
            traffic_control_lane_lock_active=False,
            final_goal_stop_active=False,
            stop_target_state=None,
            follow_target_state=None,
            current_temp_reference_xy=None,
            current_temp_mode_value=0.0,
            current_temp_road_id=None,
            current_temp_entered_intersection=False,
            active_reference_maneuver="straight",
            current_temp_mode_str="NORMAL",
            lane_reference_speed_mps=2.0,
            lane_reference_step_distance_m=2.0,
            mpc_horizon_steps=4,
            mpc_dt_s=0.1,
            temporary_destination_state=[6.0, 0.0, 2.0, 0.0, 1],
            lane_reference_freeze_count=0,
            sim_time_s=1.0,
            stop_release_temp_smooth_until_sim_time_s=0.0,
        )

        output = generate_mpc_reference(context)

        self.assertIsInstance(output, MpcReferenceGenerationOutput)
        self.assertTrue(output.mpc_reference_result.has_samples)
        self.assertEqual(output.reference_target_lane_id, 1)
        self.assertFalse(output.should_follow_global_route_lane_for_reference)

    def test_mpc_reference_generation_output_names_runner_handoff_fields(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=1,
            global_route_reference_allowed=False,
            traffic_control_lane_lock_active=False,
        )
        result = build_mpc_reference_result(
            samples=[{"x_ref_m": 1.0, "y_ref_m": 2.0, "lane_id": 1}],
            intent=intent,
            behavior_decision="lane_follow",
            fsm_state="LANE_KEEP",
            target_lane_id=1,
            follow_global_route_lane=False,
            fallback_reason="",
            stabilized=False,
            jump_m=0.0,
            first_forward_m=2.0,
            first_lateral_m=0.1,
        )

        output = MpcReferenceGenerationOutput(
            mpc_reference_result=result,
            local_lane_center_reference=list(result.samples),
            temporary_destination_state=[3.0, 4.0, 5.0],
            reference_target_lane_id=1,
            should_follow_global_route_lane_for_reference=False,
            lane_reference_freeze_count=0,
            last_reference_fallback_reason="",
            last_reference_stabilized=False,
            last_reference_jump_m=0.0,
            first_reference_forward_m=2.0,
            first_reference_lateral_m=0.1,
            reference_geometry_guard_active=False,
            reference_geometry_guard_reason="",
        )

        self.assertIs(output.mpc_reference_result, result)
        self.assertEqual(output.local_lane_center_reference[0]["lane_id"], 1)
        self.assertEqual(output.temporary_destination_state[2], 5.0)
        self.assertFalse(output.should_follow_global_route_lane_for_reference)

    def test_mpc_reference_result_packages_samples_and_trace(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=1,
            global_route_reference_allowed=False,
            traffic_control_lane_lock_active=False,
        )
        samples = [
            {"x_ref_m": 1.0, "y_ref_m": 2.0, "lane_id": 1},
            {"x_ref_m": 2.0, "y_ref_m": 2.0, "lane_id": 1},
        ]

        result = build_mpc_reference_result(
            samples=samples,
            intent=intent,
            behavior_decision="lane_follow",
            fsm_state="LANE_KEEP",
            target_lane_id=1,
            follow_global_route_lane=False,
            fallback_reason="",
            stabilized=False,
            jump_m=0.25,
            first_forward_m=1.5,
            first_lateral_m=0.1,
        )

        self.assertTrue(result.has_samples)
        self.assertEqual(result.samples, samples)
        self.assertEqual(result.first_sample, samples[0])
        self.assertEqual(result.trace.violations, [])
        self.assertEqual(result.trace_fields()["reference_pipeline_target_lane_id"], 1)

    def test_lane_follow_trace_has_no_violation_for_lane_center_reference(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=2,
            global_route_reference_allowed=True,
            traffic_control_lane_lock_active=False,
        )

        trace = trace_reference_pipeline(
            intent=intent,
            behavior_decision="lane_follow",
            fsm_state="LANE_KEEP",
            target_lane_id=1,
            output_reference_sample={"lane_id": 1},
            follow_global_route_lane=False,
            fallback_reason="",
            stabilized=False,
            jump_m=0.0,
            first_forward_m=2.0,
            first_lateral_m=0.1,
        )

        self.assertEqual(trace.violations, [])
        self.assertEqual(trace.lateral_reference_source, "lane_center")
        self.assertEqual(trace.route_role, "lane_choice_hint_only")

    def test_lane_follow_trace_flags_direct_global_route_tracking(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=2,
            global_route_reference_allowed=True,
            traffic_control_lane_lock_active=False,
        )

        trace = trace_reference_pipeline(
            intent=intent,
            behavior_decision="lane_follow",
            fsm_state="LANE_KEEP",
            target_lane_id=1,
            output_reference_sample={"lane_id": 2},
            follow_global_route_lane=True,
            fallback_reason="",
            stabilized=False,
            jump_m=0.0,
            first_forward_m=2.0,
            first_lateral_m=0.1,
        )

        self.assertIn("lane_follow_direct_global_route_tracking", trace.violations)

    def test_stop_trace_keeps_stop_target_longitudinal(self):
        intent = select_reference_intent(
            behavior_decision="stop_at_intersection",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=1,
            global_route_reference_allowed=False,
            traffic_control_lane_lock_active=True,
        )

        trace = trace_reference_pipeline(
            intent=intent,
            behavior_decision="stop_at_intersection",
            fsm_state="LANE_KEEP",
            target_lane_id=1,
            output_reference_sample={"lane_id": 1},
            follow_global_route_lane=False,
            fallback_reason="",
            stabilized=False,
            jump_m=0.0,
            first_forward_m=2.0,
            first_lateral_m=0.0,
        )

        self.assertEqual(trace.violations, [])
        self.assertEqual(trace.longitudinal_target_kind, "stop_target")
        self.assertEqual(trace.stop_target_role, "longitudinal_speed_target")

    def test_non_lane_change_trace_flags_large_lateral_jump(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="LANE_KEEP",
            ego_in_junction=False,
            reference_target_lane_id=1,
            current_lane_id=1,
            route_optimal_lane_id=1,
            global_route_reference_allowed=False,
            traffic_control_lane_lock_active=False,
        )

        trace = trace_reference_pipeline(
            intent=intent,
            behavior_decision="lane_follow",
            fsm_state="LANE_KEEP",
            target_lane_id=1,
            output_reference_sample={"lane_id": 1},
            follow_global_route_lane=False,
            fallback_reason="",
            stabilized=False,
            jump_m=4.0,
            first_forward_m=2.0,
            first_lateral_m=4.5,
            max_non_lc_lateral_m=3.0,
        )

        self.assertIn("non_lc_reference_lateral_jump", trace.violations)

    def test_lane_change_trace_accepts_blended_reference(self):
        intent = select_reference_intent(
            behavior_decision="lane_follow",
            planner_fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            ego_in_junction=False,
            reference_target_lane_id=2,
            current_lane_id=1,
            route_optimal_lane_id=1,
            global_route_reference_allowed=False,
            traffic_control_lane_lock_active=False,
        )

        trace = trace_reference_pipeline(
            intent=intent,
            behavior_decision="lane_follow",
            fsm_state="EXECUTE_LANE_CHANGE_RIGHT",
            target_lane_id=2,
            output_reference_sample={"lane_id": 2},
            follow_global_route_lane=False,
            fallback_reason="",
            stabilized=False,
            jump_m=0.0,
            first_forward_m=2.0,
            first_lateral_m=2.8,
        )

        self.assertEqual(trace.violations, [])
        self.assertEqual(trace.lateral_reference_source, "lane_change_blend")

    def test_reference_pipeline_summary_aggregates_debug_fields(self):
        rows = [
            {
                "reference_pipeline_stage": "intent>raw>validated",
                "reference_pipeline_lateral_source": "lane_center",
                "reference_pipeline_intent_mode": "lane_follow",
                "reference_pipeline_stabilized": 0,
                "reference_pipeline_jump_m": 0.4,
                "reference_pipeline_first_forward_m": 2.0,
                "reference_pipeline_first_lateral_m": 0.2,
                "reference_pipeline_violation_count": 0,
                "reference_pipeline_violations": "",
            },
            {
                "reference_pipeline_stage": "intent>raw>validated>fallback>stabilized",
                "reference_pipeline_lateral_source": "lane_center",
                "reference_pipeline_intent_mode": "lane_follow",
                "reference_pipeline_fallback_reason": "first_sample_lateral_jump",
                "reference_pipeline_stabilized": 1,
                "reference_pipeline_jump_m": 3.5,
                "reference_pipeline_first_forward_m": 1.0,
                "reference_pipeline_first_lateral_m": 4.2,
                "reference_pipeline_violation_count": 1,
                "reference_pipeline_violations": "non_lc_reference_lateral_jump",
            },
        ]

        summary = summarize_reference_pipeline_history(rows)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["violation_samples"], 1)
        self.assertAlmostEqual(summary["violation_rate"], 0.5)
        self.assertAlmostEqual(summary["stabilized_rate"], 0.5)
        self.assertAlmostEqual(summary["max_reference_jump_m"], 3.5)
        self.assertAlmostEqual(summary["max_first_lateral_abs_m"], 4.2)
        self.assertEqual(
            summary["violations_by_type"]["non_lc_reference_lateral_jump"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
