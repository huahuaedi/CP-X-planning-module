import math
from types import SimpleNamespace

import pytest

from pipeline.perception_stage import PerceptionStage
from pipeline.tracker import CPXObstacleTracker


def test_perception_stage_builds_one_fused_obstacle_view():
    stage = PerceptionStage(
        max_mpc_obstacles=1,
        ego_length_m=4.5,
        ego_width_m=2.0,
        lane_width_m=3.5,
        lane_change_boundary_overlap_m=0.75,
    )
    location = SimpleNamespace(x=1.0, y=2.0)

    result = stage.build(
        detected_objects={"vehicles": [{
            "vehicle_id": "local", "x": -5.0, "y": 2.0, "v": 3.0,
        }]},
        cp_payload={"obstacles": [{
            "vehicle_id": "remote", "x": 8.0, "y": 2.0,
            "speed_mps": 4.5,
        }]},
        ego_location=location,
        ego_yaw_rad=0.25,
        timestamp_s=10.0,
        ignore_dynamic_objects=False,
    )

    assert len(result.local_objects) == 1
    assert len(result.fused_objects) == 2
    assert len(result.mpc_objects) == 1
    assert result.front_gap_m == pytest.approx(
        7.0 * math.cos(0.25) - 2.25 - 2.25
    )
    assert result.front_actor_id == "remote"
    assert result.front_actor_speed_mps == 4.5


def test_ignore_dynamic_objects_is_applied_before_mpc_and_gap():
    observed = {}

    stage = PerceptionStage(
        max_mpc_obstacles=8,
        ego_length_m=4.5,
        ego_width_m=2.0,
        lane_width_m=3.5,
        lane_change_boundary_overlap_m=0.75,
    )

    result = stage.build(
        detected_objects={"vehicles": [{"vehicle_id": "local"}]},
        cp_payload={},
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.0,
        timestamp_s=1.0,
        ignore_dynamic_objects=True,
    )

    assert result.fused_objects == ()
    assert result.mpc_objects == ()
    assert observed == {}


def test_perception_stage_tracks_before_front_speed_is_resolved():
    """The initial speed/safety decision must consume tracker kinematics."""

    stage = PerceptionStage(
        obstacle_tracker=CPXObstacleTracker(),
        max_mpc_obstacles=4,
    )
    ego = SimpleNamespace(x=0.0, y=0.0)

    stage.build(
        detected_objects={"vehicles": [{
            "vehicle_id": "front", "x": 10.0, "y": 0.0, "v": 0.0,
        }]},
        cp_payload={},
        ego_location=ego,
        ego_yaw_rad=0.0,
        timestamp_s=1.0,
        ignore_dynamic_objects=False,
    )
    result = stage.build(
        detected_objects={"vehicles": [{
            "vehicle_id": "front", "x": 10.4, "y": 0.0, "v": 0.0,
        }]},
        cp_payload={},
        ego_location=ego,
        ego_yaw_rad=0.0,
        timestamp_s=1.1,
        ignore_dynamic_objects=False,
    )

    assert result.front_actor_speed_mps == pytest.approx(4.0)
    assert result.fused_objects[0]["kinematics_source"] == "position_finite_difference"


def test_fusion_preserves_cp_provenance_and_prediction_with_local_kinematics():
    stage = PerceptionStage(max_mpc_obstacles=4)

    fused = stage.fuse(
        local_objects=[{
            "vehicle_id": "42", "x": 10.0, "y": 0.0, "v": 3.0,
            "psi": 0.0, "timestamp_s": 10.0,
        }],
        cp_obstacles=[{
            "id": "native_opencda_v2x:42",
            "state": [12.0, 0.0, 7.0, 0.0],
            "source": "opencda_v2x",
            "provider_source": "native_opencda_v2x",
            "timestamp_s": 9.8,
            "ttl_s": 1.0,
            "type": "car",
            "predicted_modes": [{
                "probability": 1.0,
                "path": [{"t": 0.0, "x": 12.0, "y": 0.0}],
            }],
        }],
        timestamp_s=10.0,
    )

    assert len(fused) == 1
    obstacle = fused[0]
    assert obstacle["x"] == 10.0
    assert obstacle["v"] == 3.0
    assert obstacle["object_type"] == "vehicle"
    assert obstacle["observation_sources"] == ["local", "cp"]
    assert obstacle["locally_observed"] is True
    assert obstacle["cooperatively_observed"] is True
    assert obstacle["cp_age_s"] == pytest.approx(0.2)
    assert len(obstacle["predicted_modes"]) == 1


def test_position_deduplication_keeps_cp_evidence_on_local_object():
    stage = PerceptionStage(max_mpc_obstacles=4)

    fused = stage.fuse(
        local_objects=[{
            "vehicle_id": "local-actor", "x": 15.0, "y": -2.0,
            "v": 2.5, "psi": 0.0,
            "source": "opencda_perception",
            "provider_source": "native_opencda_perception",
        }],
        cp_obstacles=[{
            "id": "native_opencda_perception:anonymous-0",
            "state": [15.2, -2.1, 2.5, 0.0],
            "source": "opencda_perception",
            "provider_source": "native_opencda_perception",
            "timestamp_s": 3.0,
            "ttl_s": 1.0,
            "observed_by_cav_ids": [7],
            "blind_spot_shared": True,
        }],
        timestamp_s=3.5,
    )

    assert len(fused) == 1
    obstacle = fused[0]
    assert obstacle["vehicle_id"] == "local-actor"
    assert obstacle["observation_sources"] == ["local", "cp"]
    assert obstacle["cooperatively_observed"] is True
    assert obstacle["observed_by_cav_ids"] == [7]
    assert obstacle["blind_spot_shared"] is True


def test_stale_cp_observation_never_enters_common_observation_contract():
    stage = PerceptionStage(max_mpc_obstacles=4)

    fused = stage.fuse(
        local_objects=[],
        cp_obstacles=[{
            "id": "remote-vru",
            "state": [5.0, 1.0, 1.0, 0.0],
            "type": "pedestrian",
            "timestamp_s": 1.0,
            "ttl_s": 0.5,
        }],
        timestamp_s=2.0,
    )

    assert fused == []
