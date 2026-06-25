import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest

PLANNING_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLANNING_MODULE_ROOT not in sys.path:
    sys.path.insert(0, PLANNING_MODULE_ROOT)

_METRICS_MODULE_PATH = os.path.join(
    PLANNING_MODULE_ROOT,
    "utility",
    "evaluation_metrics.py",
)
_METRICS_SPEC = importlib.util.spec_from_file_location(
    "evaluation_metrics_under_test",
    _METRICS_MODULE_PATH,
)
evaluation_metrics = importlib.util.module_from_spec(_METRICS_SPEC)
assert _METRICS_SPEC is not None and _METRICS_SPEC.loader is not None
sys.modules[_METRICS_SPEC.name] = evaluation_metrics
_METRICS_SPEC.loader.exec_module(evaluation_metrics)

EvaluationMetricsRecorder = evaluation_metrics.EvaluationMetricsRecorder
compute_pairwise_ttc_drac = evaluation_metrics.compute_pairwise_ttc_drac
write_planning_metrics_artifacts = evaluation_metrics.write_planning_metrics_artifacts


class EvaluationMetricsTests(unittest.TestCase):
    def test_ttc_and_drac_use_longitudinal_closing_gap(self):
        ttc_s, drac_mps2 = compute_pairwise_ttc_drac(
            ego_state={"x": 0.0, "y": 0.0, "v": 10.0, "psi": 0.0},
            obstacle_snapshot={
                "x": 25.0,
                "y": 0.2,
                "v": 5.0,
                "psi": 0.0,
                "length_m": 4.0,
            },
            ego_length_m=4.0,
            lateral_conflict_width_m=2.0,
        )

        self.assertAlmostEqual(ttc_s, 21.0 / 5.0)
        self.assertAlmostEqual(drac_mps2, 25.0 / 42.0)

    def test_ttc_is_infinite_when_obstacle_is_not_conflicting(self):
        ttc_s, drac_mps2 = compute_pairwise_ttc_drac(
            ego_state=[0.0, 0.0, 10.0, 0.0],
            obstacle_snapshot={"x": 25.0, "y": 10.0, "v": 0.0, "psi": 0.0},
            lateral_conflict_width_m=2.0,
        )

        self.assertEqual(ttc_s, float("inf"))
        self.assertEqual(drac_mps2, 0.0)

    def test_recorder_accumulates_summary_metrics(self):
        recorder = EvaluationMetricsRecorder(ego_length_m=4.0)
        recorder.update(
            ego_state={"x": 0.0, "y": 0.0, "v": 10.0, "psi": 0.0},
            obstacle_snapshots=[{"vehicle_id": "a", "x": 20.0, "y": 0.0, "v": 0.0}],
            sim_time_s=0.0,
        )
        recorder.update(
            ego_state={"x": 10.0, "y": 0.0, "v": 10.0, "psi": 0.0},
            obstacle_snapshots=[{"vehicle_id": "a", "x": 20.0, "y": 0.0, "v": 0.0}],
            sim_time_s=1.0,
        )
        recorder.record_collision(event_id="1:42")
        recorder.record_collision(event_id="1:42")
        recorder.record_mpc_status({"solver_status": "solved"})
        recorder.record_mpc_status({"solver_status": "maximum iterations reached"})

        summary = recorder.summary()
        self.assertEqual(summary["collision_count"], 1)
        self.assertAlmostEqual(summary["distance_traveled_m"], 10.0)
        self.assertGreater(summary["collision_rate_per_km"], 0.0)
        self.assertEqual(summary["mpc_plan_attempts"], 2)
        self.assertEqual(summary["mpc_plan_successes"], 1)
        self.assertAlmostEqual(summary["mpc_plan_success_rate"], 0.5)
        self.assertIsNotNone(summary["min_ttc_s"])
        self.assertIsNotNone(summary["max_drac_mps2"])

    def test_write_artifacts_outputs_json_summary_and_timeseries_csv(self):
        recorder = EvaluationMetricsRecorder()
        recorder.update(
            ego_state=[0.0, 0.0, 5.0, 0.0],
            obstacle_snapshots=[],
            sim_time_s=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = write_planning_metrics_artifacts(
                artifact_dir=tmp_dir,
                recorder=recorder,
                scenario_name="town10",
            )

            self.assertTrue(os.path.isfile(artifacts["json_path"]))
            self.assertTrue(os.path.isfile(artifacts["csv_path"]))
            with open(artifacts["json_path"], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["scenario_name"], "town10")
            self.assertIn("collision_rate_per_km", payload["summary"])
            with open(artifacts["csv_path"], "r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_index"], "0")


if __name__ == "__main__":
    unittest.main()
