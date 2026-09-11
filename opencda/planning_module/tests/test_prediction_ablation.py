import json
import math

from pipeline.prediction import build_prediction_frame
from pipeline.prediction_ablation import (
    OracleTraceStore,
    TraceRecorder,
    build_snapshot_transform,
    frozen_snapshot_transform,
)


def _obstacle(x, y, v, psi=0.0, oid="42"):
    return {"id": oid, "x": x, "y": y, "v": v, "psi": psi}


def test_cv_mode_returns_no_transform():
    assert build_snapshot_transform(
        prediction_mode="cv", horizon_s=3.0, dt_s=0.1
    ) is None


def test_synthetic_brake_stays_at_rest_after_stopping():
    transform = build_snapshot_transform(
        prediction_mode="synthetic_multimodal", horizon_s=5.0, dt_s=0.1
    )
    for speed in (0.0, 2.0, 6.0):
        output = transform(_obstacle(10.0, 0.0, speed), speed + 10.0)
        points = next(mode["points"] for mode in output["trajectory_hypotheses"]
                      if mode["maneuver"] == "brake")
        assert all(b["x"] >= a["x"] - 1e-9
                   for a, b in zip(points, points[1:]))
        stopped = [point for point in points if point["t"] >= speed / 2.0]
        assert stopped
        assert all(point["v"] == 0.0 for point in stopped)
        assert all(abs(point["x"] - (10.0 + speed ** 2 / 4.0)) < 1e-9
                   for point in stopped)


def test_blind_freezes_obstacle_future():
    tf = frozen_snapshot_transform(horizon_s=2.0, dt_s=0.5)
    out = tf(_obstacle(10.0, 0.0, 8.0), 0.0)
    hyp = out["trajectory_hypotheses"][0]
    assert hyp["maneuver"] == "stationary_assumed"
    assert out["v"] == 0.0
    # every predicted point sits on the current position
    assert all(abs(p["x"] - 10.0) < 1e-9 and abs(p["y"]) < 1e-9 for p in hyp["points"])
    assert len(hyp["points"]) == 4


def test_blind_transform_flows_through_build_prediction_frame():
    ego = {"x": 0.0, "y": 0.0, "v": 10.0, "psi": 0.0}
    # obstacle 20 m ahead in ego lane, closing at 12 m/s -> CV would flag risk
    obs = [_obstacle(20.0, 0.0, -12.0 if False else 12.0, psi=math.pi)]
    tf = build_snapshot_transform(prediction_mode="blind", horizon_s=3.0, dt_s=0.5)
    frame = build_prediction_frame(
        ego_snapshot=ego, obstacle_snapshots=obs,
        lane_assignments={"42": 1}, available_lane_ids=[1],
        horizon_s=3.0, dt_s=0.5, min_front_gap_m=5.0, min_rear_gap_m=5.0,
        min_ttc_s=2.0, snapshot_transform=tf,
    )
    pts = frame.obstacle_future_trajectories["42"]
    assert pts and all(abs(p["x"] - 20.0) < 1e-6 for p in pts)


def test_oracle_replays_recorded_future_and_labels_turn(tmp_path):
    trace = tmp_path / "t.jsonl"
    with trace.open("w") as fh:
        # actor curves: heading swings from 0 to ~pi/2 over 3 s -> turn_left
        for k in range(0, 31):
            t = k * 0.1
            psi = (math.pi / 2.0) * min(1.0, t / 3.0)
            x = 5.0 + 4.0 * t * math.cos(psi / 2)
            y = 0.0 + 4.0 * t * math.sin(psi / 2)
            fh.write(json.dumps({"t": t, "actors": {"7": {
                "x": x, "y": y, "v": 4.0, "psi": psi, "lane_id": 1}}}) + "\n")
    store = OracleTraceStore.from_file(trace)
    tf = store.snapshot_transform(horizon_s=3.0, dt_s=0.1)
    # snapshot near the actor's position at t=0.0
    out = tf({"id": "7", "x": 5.0, "y": 0.0, "v": 4.0, "psi": 0.0}, 0.0)
    hyp = out["trajectory_hypotheses"][0]
    assert out["prediction_source"] == "ablation_oracle"
    assert hyp["maneuver"] == "turn_left"
    assert len(hyp["points"]) >= 25
    # replayed point matches the recorded trace, not a straight CV line
    last = hyp["points"][-1]
    assert last["y"] > 3.0  # actual curve went well off the x-axis


def test_oracle_unmatched_obstacle_falls_back_to_cv_by_default(tmp_path):
    trace = tmp_path / "t.jsonl"
    with trace.open("w") as fh:
        fh.write(json.dumps({"t": 0.0, "actors": {"7": {
            "x": 100.0, "y": 100.0, "v": 0.0, "psi": 0.0, "lane_id": 1}}}) + "\n")
        fh.write(json.dumps({"t": 0.1, "actors": {"7": {
            "x": 100.0, "y": 100.0, "v": 0.0, "psi": 0.0, "lane_id": 1}}}) + "\n")
    store = OracleTraceStore.from_file(trace)
    tf = store.snapshot_transform(horizon_s=3.0, dt_s=0.1)
    snap = {"id": "9", "x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0}
    out = tf(snap, 0.0)
    assert "trajectory_hypotheses" not in out  # untouched -> pipeline CV rollout


def test_recorder_roundtrip(tmp_path):
    path = tmp_path / "rec.jsonl"
    rec = TraceRecorder(path)
    for k in range(5):
        rec.record(sim_time_s=k * 0.1,
                   actors={"3": {"x": 1.0 + 5.0 * k * 0.1, "y": 2.0, "v": 5.0, "psi": 0.0}})
    rec.close()
    assert rec.rows_written == 5
    store = OracleTraceStore.from_file(path)
    tf = store.snapshot_transform(horizon_s=0.3, dt_s=0.1)
    out = tf({"id": "3", "x": 1.0, "y": 2.0}, 0.0)
    assert out["trajectory_hypotheses"][0]["points"][0]["x"] > 1.0
