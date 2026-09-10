from pipeline.prediction import build_prediction_frame
from pipeline.prediction_ablation import synthetic_multimodal_snapshot_transform
from utility.planning_context import PredictionContext


def test_synthetic_transform_targets_one_actor_and_normalizes_probabilities():
    transform = synthetic_multimodal_snapshot_transform(
        horizon_s=2.0, dt_s=0.2, actor_ids=[7]
    )
    untouched = transform({"id": 8, "x": 0, "y": 0, "v": 5}, 1.0)
    assert "trajectory_hypotheses" not in untouched
    transformed = transform({"id": 7, "x": 0, "y": 0, "v": 5, "psi": 0}, 1.0)
    modes = transformed["trajectory_hypotheses"]
    assert [mode["maneuver"] for mode in modes] == [
        "lane_keep", "brake", "lane_change_left"
    ]
    assert abs(sum(mode["probability"] for mode in modes) - 1.0) < 1.0e-9
    assert modes[1]["points"][-1]["x"] < modes[0]["points"][-1]["x"]
    assert modes[2]["points"][-1]["y"] > 3.0


def test_prediction_frame_preserves_all_synthetic_hypotheses():
    transform = synthetic_multimodal_snapshot_transform(
        horizon_s=2.0, dt_s=0.2, actor_ids=[7]
    )
    frame = build_prediction_frame(
        ego_snapshot={"x": -10, "y": 0, "v": 8, "psi": 0},
        obstacle_snapshots=[{"id": 7, "x": 0, "y": 0, "v": 5, "psi": 0}],
        lane_assignments={"7": 1}, available_lane_ids=[1],
        horizon_s=2.0, dt_s=0.2, min_front_gap_m=5.0,
        min_rear_gap_m=5.0, min_ttc_s=2.0, snapshot_transform=transform,
    )
    assert len(frame.predicted_objects["7"].hypotheses) == 3
    assert len(frame.hypothesis_trajectories(0.05)) == 3
    assert len(frame.hypothesis_trajectories(0.30)) == 1

    context = PredictionContext(predicted_objects=dict(frame.predicted_objects))
    assert len(context.hypothesis_trajectories(0.05)) == 3
    assert len(context.hypothesis_trajectories(0.30)) == 1


def test_synthetic_prediction_revision_updates_at_configured_cadence():
    transform = synthetic_multimodal_snapshot_transform(
        horizon_s=2.0, dt_s=0.2, actor_ids=[7], update_period_s=0.2
    )
    initial = transform(
        {"id": 7, "x": 0, "y": 0, "v": 5, "psi": 0}, 1.0
    )
    held = transform(
        {"id": 7, "x": 0.25, "y": 0, "v": 5, "psi": 0}, 1.05
    )
    refreshed = transform(
        {"id": 7, "x": 1.0, "y": 0, "v": 5, "psi": 0}, 1.20
    )

    assert initial["prediction_timestamp_s"] == 1.0
    assert held["prediction_timestamp_s"] == 1.0
    assert held["trajectory_hypotheses"] is initial["trajectory_hypotheses"]
    assert refreshed["prediction_timestamp_s"] == 1.20
    assert refreshed["trajectory_hypotheses"] is not initial["trajectory_hypotheses"]


def test_prediction_frame_revision_follows_prediction_not_measurement_tick():
    transform = synthetic_multimodal_snapshot_transform(
        horizon_s=2.0, dt_s=0.2, actor_ids=[7], update_period_s=0.2
    )

    def frame(timestamp_s, x_m):
        return build_prediction_frame(
            ego_snapshot={"x": -10, "y": 0, "v": 8, "psi": 0},
            obstacle_snapshots=[
                {"id": 7, "x": x_m, "y": 0, "v": 5, "psi": 0}
            ],
            lane_assignments={"7": 1}, available_lane_ids=[1],
            horizon_s=2.0, dt_s=0.2, min_front_gap_m=5.0,
            min_rear_gap_m=5.0, min_ttc_s=2.0,
            revision=f"measurement:{timestamp_s}",
            timestamp_s=timestamp_s, snapshot_transform=transform,
        )

    initial = frame(1.0, 0.0)
    held = frame(1.05, 0.25)
    refreshed = frame(1.20, 1.0)
    assert held.revision == initial.revision
    assert refreshed.revision != initial.revision
