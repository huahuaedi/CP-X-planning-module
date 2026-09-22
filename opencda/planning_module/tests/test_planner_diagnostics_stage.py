import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge  # noqa: E402
from pipeline.planner_diagnostics_stage import _executed_reference_tracking  # noqa: E402


class _FakeBridge:
    """Stand-in exercising the real bound method, no other bridge state."""

    def __init__(self, points):
        self._points = list(points)
        self.display_calls = 0

    def _display_global_route_points(self):
        self.display_calls += 1
        return list(self._points)

    _route_points_for_display = CPXMPCPlannerBridge._route_points_for_display


def test_route_points_for_display_emits_full_route_on_first_call():
    bridge = _FakeBridge([[0.0, 0.0], [10.0, 0.0]])
    result = bridge._route_points_for_display("route-1")
    assert result == [[0.0, 0.0], [10.0, 0.0]]
    assert bridge.display_calls == 1


def test_route_points_for_display_omits_unchanged_route_on_later_ticks():
    bridge = _FakeBridge([[0.0, 0.0], [10.0, 0.0]])
    bridge._route_points_for_display("route-1")
    result = bridge._route_points_for_display("route-1")
    assert result == []
    assert bridge.display_calls == 1


def test_route_points_for_display_re_emits_on_route_change():
    bridge = _FakeBridge([[0.0, 0.0], [10.0, 0.0]])
    bridge._route_points_for_display("route-1")
    bridge._points = [[0.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
    result = bridge._route_points_for_display("route-2")
    assert result == [[0.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
    assert bridge.display_calls == 2


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
