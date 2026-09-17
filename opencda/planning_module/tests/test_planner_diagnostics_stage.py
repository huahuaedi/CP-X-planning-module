import math

import pytest

from pipeline.planner_diagnostics_stage import (
    _executed_reference_tracking,
    _route_points_for_display,
)


class _FakeOwner:
    def __init__(self, points):
        self._points = list(points)
        self.display_calls = 0

    def _display_global_route_points(self):
        self.display_calls += 1
        return list(self._points)


def test_route_points_for_display_emits_full_route_on_first_call():
    owner = _FakeOwner([[0.0, 0.0], [10.0, 0.0]])
    result = _route_points_for_display(owner, "route-1")
    assert result == [[0.0, 0.0], [10.0, 0.0]]
    assert owner.display_calls == 1


def test_route_points_for_display_omits_unchanged_route_on_later_ticks():
    owner = _FakeOwner([[0.0, 0.0], [10.0, 0.0]])
    _route_points_for_display(owner, "route-1")
    result = _route_points_for_display(owner, "route-1")
    assert result == []
    assert owner.display_calls == 1


def test_route_points_for_display_re_emits_on_route_change():
    owner = _FakeOwner([[0.0, 0.0], [10.0, 0.0]])
    _route_points_for_display(owner, "route-1")
    owner._points = [[0.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
    result = _route_points_for_display(owner, "route-2")
    assert result == [[0.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
    assert owner.display_calls == 2


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
