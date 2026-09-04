"""Unit checks for the CARLA experiment evidence exporter."""

import importlib.util
from pathlib import Path


_REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scenario_testing"
    / "export_cav_interaction_report.py"
)
_SPEC = importlib.util.spec_from_file_location("cav_experiment_report", _REPORT_PATH)
REPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(REPORT)


def test_prediction_error_matches_future_peer_pose_with_tick_tolerance():
    observer = [{
        "sim_time_s": "2.00",
        "cav_prediction_validation_horizon_s": "1.0",
        "cav_prediction_validation_x_m": "13.0",
        "cav_prediction_validation_y_m": "4.0",
    }]
    peer = [{"sim_time_s": "3.04", "x_m": "13.3", "y_m": "4.4"}]
    errors = REPORT._prediction_errors(observer, peer)
    assert len(errors) == 1
    assert abs(errors[0][1] - 0.5) < 1e-9


def test_counterfactual_position_delta_uses_aligned_samples():
    on_rows = [{"sim_time_s": "1.0", "x_m": "2.0", "y_m": "3.0"}]
    off_rows = [{"sim_time_s": "1.03", "x_m": "2.0", "y_m": "4.0"}]
    assert REPORT._trajectory_delta(on_rows, off_rows) == 1.0
