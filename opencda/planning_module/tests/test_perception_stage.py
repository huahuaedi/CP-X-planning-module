from types import SimpleNamespace

from pipeline.perception_stage import PerceptionStage


def test_perception_stage_builds_one_fused_obstacle_view():
    calls = []

    def collect_local(**kwargs):
        calls.append(("collect", kwargs))
        return [{"vehicle_id": "local", "v": 3.0}]

    def front_gap(**kwargs):
        calls.append(("gap", kwargs))
        return 8.0, "remote"

    stage = PerceptionStage(
        collect_local=collect_local,
        front_gap=front_gap,
        object_track_id=lambda item: str(item["vehicle_id"]),
        max_mpc_obstacles=1,
    )
    location = SimpleNamespace(x=1.0, y=2.0)

    result = stage.build(
        detected_objects={"vehicles": []},
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
    assert result.front_gap_m == 8.0
    assert result.front_actor_id == "remote"
    assert result.front_actor_speed_mps == 4.5
    assert [name for name, _ in calls] == ["collect", "gap"]


def test_ignore_dynamic_objects_is_applied_before_mpc_and_gap():
    observed = {}

    def empty_gap(**kwargs):
        observed["gap"] = list(kwargs["object_snapshots"])
        return None, ""

    stage = PerceptionStage(
        collect_local=lambda **_kwargs: [{"vehicle_id": "local"}],
        front_gap=empty_gap,
        object_track_id=lambda item: str(item["vehicle_id"]),
        max_mpc_obstacles=8,
    )

    result = stage.build(
        detected_objects=None,
        cp_payload={},
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.0,
        timestamp_s=1.0,
        ignore_dynamic_objects=True,
    )

    assert result.fused_objects == ()
    assert result.mpc_objects == ()
    assert observed == {"gap": []}
