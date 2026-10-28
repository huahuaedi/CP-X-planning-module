import json
import math
from types import SimpleNamespace

from cpx_planning.prediction_ros_adapter import (
    ROSPredictionBridge,
    prediction_session_id,
)


def _stamp(value):
    seconds = int(value)
    return SimpleNamespace(
        sec=seconds,
        nanosec=int(round((float(value) - seconds) * 1.0e9)),
    )


def _pose_stamped(timestamp_s, x, y, yaw=0.0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(timestamp_s)),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=0.0),
            orientation=SimpleNamespace(
                x=0.0, y=0.0,
                z=math.sin(0.5 * yaw), w=math.cos(0.5 * yaw),
            ),
        ),
    )


def _velocity(vx, vy=0.0):
    return SimpleNamespace(
        twist=SimpleNamespace(
            linear=SimpleNamespace(x=vx, y=vy, z=0.0)
        )
    )


def _frame(timestamp_s=10.0, revision=4, mode_count=6):
    modes = []
    for index in range(mode_count):
        modes.append(SimpleNamespace(
            mode_id="mode_%d" % index,
            confidence=float(index + 1),
            points=[],
            waypoints=[
                _pose_stamped(timestamp_s + 0.1, 1.0 + index, 2.0, 0.2),
                _pose_stamped(timestamp_s + 0.2, 2.0 + index, 2.2, 0.2),
            ],
            velocities=[_velocity(3.0), _velocity(3.5)],
        ))
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="map", stamp=_stamp(timestamp_s)),
        revision=revision,
        map_polyline_count=9,
        objects=[SimpleNamespace(track_id="car-1", trajectories=modes)],
    )


def test_six_modes_are_time_rebased_and_attached_without_ranking_loss():
    bridge = ROSPredictionBridge(max_stale_s=0.5)
    assert bridge.update_frame(_frame())

    result = bridge.attach(
        [{"vehicle_id": "car-1", "x": 0.0, "y": 0.0}],
        horizon_s=8.0,
        dt_s=0.1,
        ego_snapshot={},
        timestamp_s=10.05,
    )

    assert len(result[0]["trajectory_hypotheses"]) == 6
    assert len(result[0]["predicted_modes"]) == 6
    assert result[0]["plan_revision"] == "mtr_ros:4"
    assert result[0]["prediction_source"] == "mtr_ros"
    assert math.isclose(
        sum(item["probability"] for item in result[0]["trajectory_hypotheses"]),
        1.0,
    )
    assert result[0]["predicted_trajectory"][0]["x"] == 6.0
    assert math.isclose(result[0]["predicted_trajectory"][0]["t"], 0.05)
    assert result[0]["predicted_trajectory"][0]["v"] == 3.0
    assert math.isclose(result[0]["predicted_trajectory"][0]["psi"], 0.2)


def test_stale_frame_falls_back_by_leaving_snapshot_unannotated():
    bridge = ROSPredictionBridge(max_stale_s=0.5)
    assert bridge.update_frame(_frame())
    result = bridge.attach(
        [{"vehicle_id": "car-1", "x": 0.0, "y": 0.0}],
        horizon_s=8.0,
        dt_s=0.1,
        ego_snapshot={},
        timestamp_s=10.51,
    )
    assert "predicted_trajectory" not in result[0]
    assert bridge.last_attached_count == 0


def test_compact_trajectory_points_preserve_time_speed_and_heading():
    compact_mode = SimpleNamespace(
        mode_id="mode_0",
        confidence=1.0,
        points=[SimpleNamespace(
            x=4.0,
            y=5.0,
            z=0.0,
            yaw=0.25,
            speed_mps=6.5,
            time_from_start_s=0.2,
        )],
        waypoints=[],
        velocities=[],
    )
    frame = SimpleNamespace(
        header=SimpleNamespace(frame_id="map", stamp=_stamp(10.0)),
        revision=5,
        objects=[SimpleNamespace(
            track_id="car-1", trajectories=[compact_mode]
        )],
    )
    bridge = ROSPredictionBridge(max_stale_s=0.5)

    assert bridge.update_frame(frame)
    result = bridge.attach(
        [{"vehicle_id": "car-1", "x": 0.0, "y": 0.0}],
        horizon_s=8.0,
        dt_s=0.1,
        ego_snapshot={},
        timestamp_s=10.05,
    )

    point = result[0]["predicted_trajectory"][0]
    assert point["x"] == 4.0
    assert point["y"] == 5.0
    assert math.isclose(point["t"], 0.15)
    assert point["v"] == 6.5
    assert point["psi"] == 0.25


def test_out_of_order_revision_is_rejected_and_reset_accepts_new_epoch():
    bridge = ROSPredictionBridge()
    assert bridge.update_frame(_frame(timestamp_s=10.0, revision=4))
    assert not bridge.update_frame(_frame(timestamp_s=10.0, revision=3))
    bridge.reset()
    assert bridge.update_frame(_frame(timestamp_s=1.0, revision=1))


class _Agent:
    TYPE_UNKNOWN = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_CYCLIST = 3

    def __init__(self):
        self.track_id = ""
        self.object_type = 0
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )
        self.twist = SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0, z=0.0)
        )
        self.size = SimpleNamespace(x=0.0, y=0.0, z=0.0)


class _Scene:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.session_id = ""
        self.ego = None
        self.objects = []
        self.map_polylines = []


class _Map:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.revision = 0
        self.map_polylines = []


class _Polyline:
    def __init__(self):
        self.global_type = 0
        self.points = []


class _Point:
    def __init__(self):
        self.x = self.y = self.z = 0.0


def test_scene_contains_real_map_and_deduplicates_local_cp_object():
    bridge = ROSPredictionBridge(
        scene_type=_Scene,
        agent_type=_Agent,
        map_polyline_type=_Polyline,
        point_type=_Point,
    )
    scene = bridge.build_scene(
        stamp=_stamp(3.0),
        ego_pose={"x": 1.0, "y": 2.0, "heading_rad": 0.5},
        ego_speed_mps=4.0,
        object_snapshots=[{
            "vehicle_id": "abc", "type": "car", "x": 3.0, "y": 4.0,
            "v": 2.0, "psi": 0.0, "source": "opencda_perception",
        }],
        cp_obstacles=[{
            "id": "native_opencda_multi_vantage:abc", "type": "vehicle",
            "state": [3.1, 4.0, 2.0, 0.0],
        }],
        map_polylines=[{
            "global_type": 2,
            "points": [{"x": 0.0, "y": 0.0}, {"x": 5.0, "y": 0.0}],
        }],
    )
    assert scene.header.frame_id == "map"
    assert len(scene.objects) == 1
    assert scene.objects[0].track_id == "abc"
    assert len(scene.map_polylines) == 1
    assert len(scene.map_polylines[0].points) == 2
    diagnostics = bridge.diagnostics
    assert diagnostics["prediction_ros_scene_input_count"] == 2
    assert diagnostics["prediction_ros_scene_object_count"] == 1
    assert diagnostics["prediction_ros_scene_duplicate_count"] == 1
    assert diagnostics["prediction_ros_scene_track_ids"] == "abc"


def test_cp_only_scene_uses_planner_identity_and_modes_attach_to_fused_object():
    bridge = ROSPredictionBridge(
        max_stale_s=0.5,
        scene_type=_Scene,
        agent_type=_Agent,
        map_polyline_type=_Polyline,
        point_type=_Point,
    )
    scene = bridge.build_scene(
        stamp=_stamp(10.0),
        ego_pose={"x": 0.0, "y": 0.0, "heading_rad": 0.0},
        ego_speed_mps=4.0,
        object_snapshots=[],
        cp_obstacles=[{
            "id": "native_opencda_multi_vantage:walker-17",
            "type": "pedestrian",
            "state": [20.0, 1.0, 1.2, -1.57],
            "provider_source": "native_opencda_multi_vantage",
        }],
        map_polylines=[],
    )

    assert len(scene.objects) == 1
    assert scene.objects[0].track_id == "walker-17"

    frame = _frame(timestamp_s=10.0, revision=8, mode_count=6)
    frame.objects[0].track_id = scene.objects[0].track_id
    assert bridge.update_frame(frame)
    annotated = bridge.attach(
        [{"vehicle_id": "walker-17", "x": 20.0, "y": 1.0}],
        horizon_s=8.0,
        dt_s=0.1,
        ego_snapshot={},
        timestamp_s=10.05,
    )

    assert len(annotated[0]["predicted_modes"]) == 6
    assert bridge.last_attached_count == 1
    assert bridge.last_attached_mode_count == 6


def test_static_map_message_preserves_geometry_and_revision():
    bridge = ROSPredictionBridge(
        map_type=_Map,
        map_polyline_type=_Polyline,
        point_type=_Point,
    )
    message = bridge.build_map(
        stamp=_stamp(3.0),
        revision=7,
        map_polylines=[{
            "global_type": 15,
            "points": [
                {"x": 1.0, "y": 2.0, "z": 0.0},
                {"x": 3.0, "y": 4.0, "z": 0.0},
            ],
        }],
    )

    assert message.header.frame_id == "map"
    assert message.revision == 7
    assert len(message.map_polylines) == 1
    assert message.map_polylines[0].global_type == 15
    assert message.map_polylines[0].points[1].x == 3.0


def test_diagnostics_report_received_and_attached_six_modes():
    bridge = ROSPredictionBridge(max_stale_s=0.5)
    assert bridge.update_frame(_frame(mode_count=6))
    bridge.attach(
        [{"vehicle_id": "car-1", "x": 0.0, "y": 0.0}],
        horizon_s=8.0,
        dt_s=0.1,
        ego_snapshot={},
        timestamp_s=10.05,
    )

    diagnostics = bridge.diagnostics
    assert diagnostics["prediction_ros_frame_object_count"] == 1
    assert diagnostics["prediction_ros_frame_mode_count"] == 6
    assert diagnostics["prediction_ros_cached_mode_count"] == 6
    assert diagnostics["prediction_bridge_attached_count"] == 1
    assert diagnostics["prediction_bridge_attached_mode_count"] == 6
    assert math.isclose(diagnostics["prediction_ros_frame_age_s"], 0.05)


def test_diagnostics_are_strict_json_safe_before_first_prediction_frame():
    bridge = ROSPredictionBridge(max_stale_s=0.5)

    diagnostics = bridge.diagnostics

    assert diagnostics["prediction_ros_frame_timestamp_s"] is None
    assert diagnostics["prediction_ros_frame_age_s"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_ros_round_trip_latency_is_reported_and_reset():
    bridge = ROSPredictionBridge()
    bridge.record_latency_ms(43.25)

    assert bridge.diagnostics["prediction_bridge_latency_ms"] == 43.25

    bridge.reset()
    assert bridge.diagnostics["prediction_bridge_latency_ms"] == 0.0


def test_prediction_request_and_response_preserve_session_owner():
    bridge = ROSPredictionBridge(
        scene_type=_Scene,
        agent_type=_Agent,
        map_polyline_type=_Polyline,
        point_type=_Point,
        expected_session_id="run:vehicle/17",
    )
    scene = bridge.build_scene(
        stamp=_stamp(3.0),
        ego_pose={"x": 1.0, "y": 2.0, "heading_rad": 0.0},
        ego_speed_mps=4.0,
        object_snapshots=[],
        cp_obstacles=[],
        map_polylines=[],
    )

    assert prediction_session_id(scene) == "run:vehicle/17"
    assert scene.session_id == "run:vehicle/17"
    matching = _frame()
    matching.header.frame_id = scene.header.frame_id
    matching.session_id = scene.session_id
    assert bridge.update_frame(matching)

    wrong = _frame(timestamp_s=11.0, revision=5)
    wrong.session_id = "other"
    assert not bridge.update_frame(wrong)
    assert bridge.last_error.endswith("prediction_frame_session_mismatch")
