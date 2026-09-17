import math

import pytest

from pipeline.planner_diagnostics_stage import _executed_reference_tracking


def test_executed_reference_tracking_uses_published_reference_tangent():
    reference = [
        {"x_ref_m": 10.0, "y_ref_m": 0.0},
        {"x_ref_m": 10.0, "y_ref_m": -5.0},
        {"x_ref_m": 10.0, "y_ref_m": -10.0},
    ]

    result = _executed_reference_tracking(
        reference=reference,
        ego_x_m=9.75,
        ego_y_m=-2.0,
        ego_yaw_rad=math.radians(-110.0),
    )

    assert result["executed_reference_tracking_valid"] is True
    assert result["executed_reference_progress_s_m"] == pytest.approx(2.0)
    assert result["executed_reference_lateral_error_m"] == pytest.approx(-0.25)
    assert result["executed_reference_heading_error_deg"] == pytest.approx(-20.0)


def test_executed_reference_tracking_rejects_incomplete_reference():
    result = _executed_reference_tracking(
        reference=[{"x_ref_m": 1.0, "y_ref_m": 2.0}],
        ego_x_m=1.0,
        ego_y_m=2.0,
        ego_yaw_rad=0.0,
    )

    assert result == {
        "executed_reference_tracking_valid": False,
        "executed_reference_progress_s_m": "",
        "executed_reference_lateral_error_m": "",
        "executed_reference_heading_error_deg": "",
    }
