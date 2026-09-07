"""End-to-end Stage A/C matrix for the documented conflict classes."""

import math

import pytest

from pipeline.cav_conflict_pipeline import resolve_conflicts


REF = [{"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(0, 101, 2)]
EGO = {"x": 0.0, "y": 0.0, "v": 10.0, "psi": 0.0}
BIG = 1.0e9


def _track(points):
    return [{"x": float(x), "y": float(y)} for x, y in points]


@pytest.mark.parametrize("expected,agent", [
    ("FOLLOW", {
        "id": "peer", "x": 24.0, "y": 0.1, "v": 7.0, "psi": 0.0,
    }),
    ("LEAD_BRAKE", {
        "id": "peer", "x": 24.0, "y": 0.1, "v": 7.0, "psi": 0.0,
        "a": -2.0,
    }),
    ("CUT_IN", {
        "id": "peer", "x": 8.0, "y": 3.6, "v": 9.0, "psi": -0.15,
        "predicted_trajectory": _track(
            [(8.0 + k, 3.6 - 0.18 * k) for k in range(20)]
        ),
    }),
    ("MERGE", {
        "id": "peer", "x": 8.0, "y": 3.6, "v": 9.0, "psi": -0.04,
        "predicted_trajectory": _track(
            [(8.0 + k, 3.6 - 0.06 * k) for k in range(20)]
        ),
    }),
    ("CROSSING", {
        "id": "peer", "x": 22.0, "y": -6.0, "v": 7.0,
        "psi": math.pi / 2.0,
        "predicted_trajectory": _track(
            [(22.0, -6.0 + k) for k in range(20)]
        ),
    }),
    ("ONCOMING", {
        "id": "peer", "x": 35.0, "y": 0.1, "v": 8.0, "psi": math.pi,
        "predicted_trajectory": _track(
            [(35.0 - k, 0.1) for k in range(20)]
        ),
    }),
])
def test_conflict_classification_matrix(expected, agent):
    result = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot=EGO,
        my_actor_id=1,
        obstacle_snapshots=[agent],
    )
    assert result.diagnostics["tags"]["peer"] == expected
    if expected not in {"FOLLOW", "LEAD_BRAKE"}:
        assert any(cap < BIG for cap in result.corridor.s_hi)


def test_nearby_but_non_conflicting_vehicle_is_a_true_no_op():
    result = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot=EGO,
        my_actor_id=1,
        obstacle_snapshots=[{
            "id": "peer", "x": 12.0, "y": 3.6, "v": 10.0, "psi": 0.0,
        }],
    )
    assert result.diagnostics["tags"]["peer"] == "IGNORE"
    assert all(cap >= BIG for cap in result.corridor.s_hi)
