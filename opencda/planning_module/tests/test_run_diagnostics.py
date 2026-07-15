import unittest

from utility.run_diagnostics import build_planning_debug_summary


class RunDiagnosticsTests(unittest.TestCase):
    def test_builds_ok_summary_when_layers_are_clean(self):
        summary = build_planning_debug_summary(
            scenario_name="town10_scenario_6",
            run_status={
                "finished": True,
                "reason": "destination_reached",
                "final_behavior": "lane_follow",
                "final_fsm_state": "LANE_KEEP",
            },
            metrics_summary={
                "collision_count": 0,
                "mpc_plan_success_rate": 1.0,
                "mpc_plan_attempts": 10,
            },
            reference_summary={
                "samples": 10,
                "violation_rate": 0.0,
                "stabilized_rate": 0.0,
            },
            speed_summary={
                "samples": 10,
                "rise_limited_rate": 0.0,
                "binding_caps": {"base": 10},
            },
        )

        self.assertEqual(summary["layer_health"]["mission"], "ok")
        self.assertEqual(summary["layer_health"]["safety"], "ok")
        self.assertEqual(summary["layer_health"]["reference"], "ok")
        self.assertEqual(summary["layer_health"]["speed"], "ok")
        self.assertEqual(summary["layer_health"]["mpc"], "ok")
        self.assertEqual(summary["likely_issues"], [])

    def test_surfaces_reference_speed_and_mpc_issues(self):
        summary = build_planning_debug_summary(
            scenario_name="town10_scenario_5",
            run_status={
                "finished": False,
                "reason": "not_finished",
                "mpc_consecutive_failures": 2,
            },
            metrics_summary={
                "collision_count": 1,
                "mpc_plan_success_rate": 0.85,
                "mpc_plan_attempts": 20,
            },
            reference_summary={
                "samples": 20,
                "violation_rate": 0.2,
                "stabilized_rate": 0.3,
                "violations_by_type": {"non_lc_reference_lateral_jump": 4},
            },
            speed_summary={
                "samples": 20,
                "rise_limited_rate": 0.5,
                "binding_caps": {"reference_jump": 6, "stop_profile": 3},
            },
        )

        self.assertEqual(summary["layer_health"]["mission"], "warn")
        self.assertEqual(summary["layer_health"]["safety"], "fail")
        self.assertEqual(summary["layer_health"]["reference"], "fail")
        self.assertEqual(summary["layer_health"]["speed"], "warn")
        self.assertEqual(summary["layer_health"]["mpc"], "fail")
        self.assertIn("collision_detected", summary["likely_issues"])
        self.assertIn("reference_pipeline_violation", summary["likely_issues"])
        self.assertIn("speed_limited_by_reference_jump", summary["likely_issues"])
        self.assertIn("speed_limited_by_stop_profile", summary["likely_issues"])
        self.assertIn("mpc_solver_success_rate_low", summary["likely_issues"])
        self.assertIn("scenario_not_finished:not_finished", summary["likely_issues"])


if __name__ == "__main__":
    unittest.main()
