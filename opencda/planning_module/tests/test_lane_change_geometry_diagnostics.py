import importlib.util
import pathlib
import unittest


MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "pipeline"
    / "lane_change_geometry_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location(
    "lane_change_geometry_diagnostics_under_test", MODULE_PATH
)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)
build_lane_change_geometry_diagnostic = diagnostics.build_lane_change_geometry_diagnostic


class LaneChangeGeometryDiagnosticsTest(unittest.TestCase):
    def test_projects_all_curves_into_source_route_s(self):
        source = [{"x_ref_m": float(i), "y_ref_m": 0.0} for i in range(6)]
        target = [{"x_ref_m": float(i), "y_ref_m": 3.5} for i in range(6)]
        locked = [
            {"x_ref_m": float(i), "y_ref_m": 3.5 * float(i) / 5.0}
            for i in range(6)
        ]
        result = build_lane_change_geometry_diagnostic(
            source_reference=source,
            target_reference=target,
            locked_reference=locked,
        )
        self.assertEqual(result["coordinate_frame"], "source_lane_route_s_frenet")
        self.assertAlmostEqual(result["target_curve"][3]["route_s_m"], 3.0)
        self.assertAlmostEqual(result["target_curve"][3]["lateral_m"], 3.5)
        self.assertAlmostEqual(result["locked_curve"][-1]["lateral_m"], 3.5)
        self.assertAlmostEqual(
            result["source_target_station_mismatch"][3]["station_mismatch_m"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
