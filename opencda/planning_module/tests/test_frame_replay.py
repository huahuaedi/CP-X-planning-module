"""Harness check for tools.frame_replay on a synthetic divergence frame.

This does NOT assert that the abnormal deflection happens -- baking the
suspected bug into a pass condition would be wrong. It asserts that:

* the audit reconstructs rows and *detects* the pre/post reference tangent
  disagreement that is the suspected lateral-coupling source, and
* the replay grid runs end to end and produces finite diagnostics for
  every toggle combination,

so the harness is trustworthy before it is pointed at a real captured
frame (plan steps 1-2).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pipeline.cav_conflict_pipeline import resolve_conflicts
from pipeline.conflict_classifier import ClassifierParams
from pipeline.spatiotemporal_corridor import CorridorParams

from tools.frame_replay import (
    FrameCapture,
    audit_rows,
    audit_summary,
    replay,
    replay_matrix,
)

HORIZON = 20
DT = 0.2


def _straight_reference(n=60, step=2.0):
    # Ego drives +y; straight lane centre from the origin.
    return [{"x_ref_m": 0.0, "y_ref_m": float(step * k)} for k in range(n)]


def _reanchored_reference(n=60, step=2.0, rot_deg=6.0, shift_x=0.4):
    # What the publication stage might hand MPC: same corridor, but rotated
    # a few degrees and laterally shifted (re-anchored). Geometrically a
    # different curve from the one Stage C/D built on.
    rot = math.radians(rot_deg)
    out = []
    for k in range(n):
        s = step * k
        x = shift_x + (-math.sin(rot) * 0.0 + math.cos(rot) * 0.0)  # placeholder
        x = shift_x + s * math.sin(rot)
        y = s * math.cos(rot)
        out.append({"x_ref_m": float(x), "y_ref_m": float(y)})
    return out


def _crosser(horizon):
    return {
        "id": "x",
        "x": -8.0,
        "y": 18.0,
        "v": 7.0,
        "psi": 0.0,
        "length_m": 4.5,
        "width_m": 2.0,
        "predicted_trajectory": [
            {"x": -8.0 + 0.7 * k, "y": 18.0} for k in range(horizon + 1)
        ],
    }


@pytest.fixture
def divergence_capture() -> FrameCapture:
    pre_ref = _straight_reference()
    post_ref = _reanchored_reference()
    ego = {"x": 0.0, "y": 0.0, "v": 9.0, "psi": math.pi / 2.0}
    crosser = _crosser(HORIZON)

    r = resolve_conflicts(
        reference_samples=pre_ref,
        ego_snapshot=ego,
        my_actor_id=1,
        my_claim=None,
        obstacle_snapshots=[crosser],
        cav_intents=[],
        classifier_params=ClassifierParams(horizon_steps=HORIZON, dt_s=DT),
        corridor_params=CorridorParams(
            horizon_steps=HORIZON,
            dt_s=DT,
            crossing_clearance_time_s=3.0,
            conflict_stop_buffer_m=4.0,
        ),
    )
    assert any(h < 1e8 for h in r.corridor.s_hi), "crosser should cap s_hi"

    return FrameCapture(
        tick=175,
        sim_time_s=35.0,
        current_state=[0.0, 0.0, 9.0, math.pi / 2.0],
        ego_origin_xy=(0.0, 0.0),
        ego_yaw_rad=math.pi / 2.0,
        ego_speed_mps=9.0,
        destination_state=[0.0, 40.0, 9.0, math.pi / 2.0, 1],
        target_speed_mps=9.0,
        pre_publication_reference=pre_ref,
        published_reference=post_ref,
        mpc_object_snapshots=[crosser],
        corridor_s_lo=list(r.corridor.s_lo),
        corridor_s_hi=list(r.corridor.s_hi),
        corridor_binding=list(r.corridor.binding),
    )


def test_audit_detects_pre_post_tangent_disagreement(divergence_capture):
    cap = divergence_capture
    from tools.frame_replay import _rows_for

    pre_rows, _ = _rows_for(cap, "pre")
    assert pre_rows, "expected longitudinal corridor rows"

    audits = audit_rows(
        pre_rows,
        cap.ego_origin_xy,
        pre_reference=cap.reference("pre"),
        post_reference=cap.reference("post"),
    )
    summary = audit_summary(audits)

    # Every row here is a longitudinal band; on its own (pre) reference the
    # normal is ~parallel to the tangent.
    bands = [a for a in audits if a.kind == "longitudinal_band"]
    assert bands
    assert max(a.normal_vs_pre_tangent_deg for a in bands) < 2.0

    # The published reference is rotated ~6 deg, so the audit must show a
    # tangent disagreement and flag lateral coupling against the tracked
    # reference.
    assert summary["max_pre_post_tangent_disagreement_deg"] > 3.0
    assert summary["max_lateral_coupling_deg"] > 3.0
    assert summary["flagged_row_count"] >= 1


def test_projecting_rows_onto_post_reference_removes_the_coupling(divergence_capture):
    cap = divergence_capture
    from tools.frame_replay import _rows_for

    post_rows, _ = _rows_for(cap, "post")
    audits = audit_rows(
        post_rows,
        cap.ego_origin_xy,
        pre_reference=cap.reference("pre"),
        post_reference=cap.reference("post"),
    )
    summary = audit_summary(audits)
    # Rebuilding the Stage-D projection on the tracked reference should make
    # the bands longitudinal again (this is exactly what "Fix A" does).
    assert summary["max_lateral_coupling_deg"] < 1.0
    assert summary["flagged_row_count"] == 0


def test_replay_grid_runs_and_is_finite(divergence_capture):
    results = replay_matrix(divergence_capture, warm_start_values=(False,))
    assert len(results) == (2 * 2 * 1) + (2 * 1 * 1)  # rows-on: 4 (pre+post), rows-off: 2
    for res in results:
        assert math.isfinite(res.max_abs_heading_error_deg)
        assert math.isfinite(res.max_abs_lateral_error_m)
        assert math.isfinite(res.final_progress_m)
        assert len(res.heading_error_deg_by_stage) == len(res.trajectory_world)


def test_replay_single_combo_baseline(divergence_capture):
    res = replay(
        divergence_capture,
        conflict_rows=True,
        repulsive=True,
        warm_start=False,
        corridor_ref="pre",
    )
    assert res.row_count > 0
    assert res.trajectory_world
    # No assertion on the deflection magnitude: this number is the thing
    # under investigation, recorded not gated.


def test_capture_json_round_trip(divergence_capture, tmp_path: Path):
    path = divergence_capture.to_json(tmp_path / "frame.json")
    loaded = FrameCapture.from_json(path)
    assert loaded.tick == divergence_capture.tick
    assert loaded.corridor_s_hi == divergence_capture.corridor_s_hi
    assert len(loaded.pre_publication_reference) == len(
        divergence_capture.pre_publication_reference
    )
    a = audit_rows(
        _rows(loaded, "pre"),
        loaded.ego_origin_xy,
        pre_reference=loaded.reference("pre"),
        post_reference=loaded.reference("post"),
    )
    assert a


def _rows(cap: FrameCapture, which: str):
    from tools.frame_replay import _rows_for

    rows, _ = _rows_for(cap, which)
    return rows


def test_numpy_import_present():
    # frame_replay seeds mpc._last_u_solution as an ndarray.
    assert np.array([1.0]).dtype == np.float64
