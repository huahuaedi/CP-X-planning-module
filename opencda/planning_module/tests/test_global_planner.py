"""AD-map-only tests for the planning-facing global-planner adapter."""

import math
from types import SimpleNamespace

from utility.global_planner import (
    INVALID_LANE_ID,
    CustomGlobalPlannerAdapter,
    lane_step_xy_heading,
)


class _Waypoint:
    def __init__(self, x_m, y_m, yaw_deg, *, successors=()):
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x_m), y=float(y_m)),
            rotation=SimpleNamespace(yaw=float(yaw_deg)),
        )
        self._successors = list(successors)

    def next(self, distance_m):
        del distance_m
        return list(self._successors)

    def previous(self, distance_m):
        del distance_m
        return []


def test_public_planner_surface_is_admap_only():
    assert INVALID_LANE_ID == 0
    assert CustomGlobalPlannerAdapter.__name__ == "CustomGlobalPlannerAdapter"


def test_lane_step_uses_the_resolved_waypoint_successor():
    successor = _Waypoint(10.0, 2.0, 15.0)
    current = _Waypoint(0.0, 0.0, 0.0, successors=[successor])

    result = lane_step_xy_heading(
        0.0, 0.0, 5.0, get_waypoint_fn=lambda pose: current,
    )

    assert result is not None
    assert result[0] == 10.0
    assert result[1] == 2.0
    assert math.isclose(result[2], math.radians(15.0))


def test_lane_step_returns_none_without_admap_successor():
    current = _Waypoint(0.0, 0.0, 0.0)
    assert lane_step_xy_heading(
        0.0, 0.0, 5.0, get_waypoint_fn=lambda pose: current,
    ) is None
    assert lane_step_xy_heading(
        0.0, 0.0, 5.0, get_waypoint_fn=lambda pose: None,
    ) is None
