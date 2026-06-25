import csv
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import planning_runner
from carla_scenario import runner
from carla_scenario.runner import _write_mpc_cost_artifacts


class RunnerCostPlotTests(unittest.TestCase):
    def test_write_mpc_cost_artifacts_creates_plot_and_csv_in_scenario_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_dir = os.path.join(tmp_dir, "town10")
            os.makedirs(scenario_dir, exist_ok=True)
            scenario_path = os.path.join(scenario_dir, "town10_sumo.yaml")
            with open(scenario_path, "w", encoding="utf-8") as handle:
                handle.write("name: town10_sumo\n")

            artifacts = _write_mpc_cost_artifacts(
                cost_history=[
                    {
                        "sample_index": 0,
                        "sim_time_s": 0.0,
                        "Cost_Total": 10.5,
                        "Cost_ref": 4.0,
                        "Cost_LaneCenter": 1.0,
                        "Cost_RoadBoundary": 0.5,
                        "Cost_LaneBoundary": 0.5,
                        "Cost_Lane": 1.5,
                        "Cost_Repulsive": 3.0,
                        "Cost_Repulsive_Safe": 1.0,
                        "Cost_Repulsive_Collision": 2.0,
                        "Cost_Control": 2.0,
                        "solver_status": "solved",
                        "solve_time_ms": 5.0,
                    },
                    {
                        "sample_index": 1,
                        "sim_time_s": 0.5,
                        "Cost_Total": 8.25,
                        "Cost_ref": 3.5,
                        "Cost_LaneCenter": 1.0,
                        "Cost_RoadBoundary": 0.25,
                        "Cost_LaneBoundary": 0.25,
                        "Cost_Lane": 1.25,
                        "Cost_Repulsive": 2.0,
                        "Cost_Repulsive_Safe": 0.5,
                        "Cost_Repulsive_Collision": 1.5,
                        "Cost_Control": 1.5,
                        "solver_status": "solved",
                        "solve_time_ms": 4.0,
                    },
                ],
                scenario_cfg={
                    "name": "town10_sumo",
                    "_scenario_path": scenario_path,
                    "_scenario_dir": scenario_dir,
                },
            )

            self.assertTrue(os.path.isfile(artifacts["csv_path"]))
            self.assertTrue(os.path.isfile(artifacts["plot_path"]))
            with open(artifacts["csv_path"], "r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["solver_status"], "solved")
            self.assertEqual(rows[0]["Cost_RoadBoundary"], "0.5")
            self.assertEqual(rows[0]["Cost_LaneBoundary"], "0.5")
            self.assertEqual(rows[1]["Cost_Lane"], "1.25")

    def test_opencda_scenario_artifact_dir_resolves_nested_yaml_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_dir = os.path.join(
                tmp_dir,
                "opencda_scenario",
                "all_usecase_scenario",
            )
            os.makedirs(scenario_dir, exist_ok=True)
            with open(
                os.path.join(scenario_dir, "all_usecase_scenario.yaml"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("name: all_usecase_scenario\n")

            with patch.object(runner, "PROJECT_ROOT", tmp_dir):
                artifact_dir = runner._scenario_artifact_dir(
                    {
                        "name": "all_usecase_scenario",
                        "runner_module": "opencda_scenario.runner",
                    }
                )

            self.assertEqual(artifact_dir, scenario_dir)

    def test_write_mpc_cost_artifacts_returns_empty_mapping_when_no_samples_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = _write_mpc_cost_artifacts(
                cost_history=[],
                scenario_cfg={
                    "name": "town10_sumo",
                    "_scenario_dir": tmp_dir,
                },
            )

            self.assertEqual(artifacts, {})
            self.assertFalse(os.path.exists(os.path.join(tmp_dir, "mpc_cost_plot.png")))
            self.assertFalse(os.path.exists(os.path.join(tmp_dir, "mpc_cost_history.csv")))

    def test_fast_cost_plot_path_still_uses_labeled_matplotlib_plot_when_available(self):
        class _FakeAxis:
            def __init__(self):
                self.labels = []
                self.legend_called = False

            def plot(self, x_values, values, label=None, color=None, linewidth=None):
                del x_values
                del values
                del color
                del linewidth
                self.labels.append(label)

            def set_ylabel(self, text):
                del text

            def grid(self, enabled, alpha=None):
                del enabled
                del alpha

            def legend(self, loc=None):
                del loc
                self.legend_called = True

            def set_title(self, text):
                del text

            def set_xlabel(self, text):
                del text

        class _FakeFigure:
            def tight_layout(self):
                return None

            def savefig(self, path, dpi=None, bbox_inches=None):
                del dpi
                del bbox_inches
                with open(path, "wb") as handle:
                    handle.write(b"fake-plot")

        fake_axes = [_FakeAxis() for _ in range(6)]
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = (
            lambda rows, cols, figsize=None, sharex=None: (_FakeFigure(), fake_axes)
        )
        fake_pyplot.close = lambda fig: None
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.use = lambda backend: None
        fake_matplotlib.pyplot = fake_pyplot

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_path = os.path.join(tmp_dir, "mpc_cost_plot.png")
            with patch.dict(
                sys.modules,
                {
                    "matplotlib": fake_matplotlib,
                    "matplotlib.pyplot": fake_pyplot,
                },
            ):
                planning_runner._write_mpc_cost_plot(
                    plot_path=plot_path,
                    cost_history=[
                        {
                            "sim_time_s": 0.0,
                            "Cost_Total": 10.0,
                            "Cost_ref": 3.0,
                            "Cost_LaneCenter": 2.0,
                            "Cost_RoadBoundary": 1.0,
                            "Cost_Repulsive_Collision": 2.5,
                            "Cost_Control": 1.5,
                        }
                    ],
                    scenario_name="town10",
                    prefer_fast_fallback=True,
                )

            self.assertTrue(os.path.isfile(plot_path))
            self.assertEqual(
                [axis.labels[0] for axis in fake_axes],
                [
                    "Cost_Total",
                    "Cost_ref",
                    "Cost_LaneCenter",
                    "Cost_RoadBoundary",
                    "Cost_Repulsive_Collision",
                    "Cost_Control",
                ],
            )
            self.assertTrue(all(axis.legend_called for axis in fake_axes))


if __name__ == "__main__":
    unittest.main()
