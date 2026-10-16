import json

from opencda.data_receiver import DataReceiver


def _control_message(
    sequence,
    cycle_time_s,
    acceleration_mps2=0.0,
    session_id="vehicle-42",
):
    return json.dumps(
        {
            "message_type": "planner_control",
            "sequence": sequence,
            "data": {
                "cycle_time_s": cycle_time_s,
                "target_speed_mps": 5.0,
                "stop_goal_active": True,
                "mission_complete": True,
                "emergency_stop": True,
                "session_id": session_id,
                "emergency_stop_reason": "hard_gate_stop_hazard",
                "safety_supervisor_reason": "test_supervisor",
                "emergency_brake_control_active": True,
                "prediction_bridge_attached_count": 1,
                "prediction_bridge_attached_mode_count": 6,
                "prediction_bridge_latency_ms": 43.25,
                "prediction_ros_revision": 17,
                "prediction_ros_frame_age_s": 0.15,
                "prediction_ros_scene_input_count": 2,
                "prediction_ros_scene_object_count": 1,
                "prediction_ros_scene_duplicate_count": 1,
                "prediction_ros_scene_track_ids": "actor-30",
                "compact_schema_revision": 2,
                "mpc_obstacle_input_count": 4,
                "mpc_obstacle_relevant_count": 1,
                "mpc_obstacle_ignored_count": 3,
                "mpc_obstacle_ignored_ids": ["2", "3", "4"],
                "mpc_infeasibility_diagnostic": {"status": "test"},
                "acceleration_mps2": acceleration_mps2,
                "steering_rad": 0.1,
                "planning_cycle_time_ms": 20.0,
            },
        }
    ).encode("utf-8")


def test_pop_latest_planner_control_discards_older_results():
    receiver = DataReceiver()
    receiver._receive_message(_control_message(1, 1.0, 0.1))
    receiver._receive_message(_control_message(2, 1.2, 0.2))

    selected = receiver.pop_latest_planner_control(1.25)

    assert selected["sequence"] == 2
    assert selected["acceleration_mps2"] == 0.2
    assert selected["stop_goal_active"] is True
    assert selected["mission_complete"] is True
    assert selected["emergency_stop"] is True
    assert selected["session_id"] == "vehicle-42"
    assert selected["emergency_stop_reason"] == "hard_gate_stop_hazard"
    assert selected["safety_supervisor_reason"] == "test_supervisor"
    assert selected["emergency_brake_control_active"] is True
    assert selected["prediction_bridge_attached_count"] == 1
    assert selected["prediction_bridge_attached_mode_count"] == 6
    assert selected["prediction_bridge_latency_ms"] == 43.25
    assert selected["prediction_ros_revision"] == 17
    assert selected["prediction_ros_frame_age_s"] == 0.15
    assert selected["prediction_ros_scene_input_count"] == 2
    assert selected["prediction_ros_scene_object_count"] == 1
    assert selected["prediction_ros_scene_duplicate_count"] == 1
    assert selected["prediction_ros_scene_track_ids"] == "actor-30"
    assert selected["compact_schema_revision"] == 2
    assert selected["mpc_obstacle_input_count"] == 4
    assert selected["mpc_obstacle_relevant_count"] == 1
    assert selected["mpc_obstacle_ignored_count"] == 3
    assert selected["mpc_obstacle_ignored_ids"] == ["2", "3", "4"]
    assert selected["mpc_infeasibility_diagnostic"] == {"status": "test"}
    assert receiver.pop_latest_planner_control(1.25) is None


def test_pop_latest_planner_control_retains_future_result():
    receiver = DataReceiver()
    receiver._receive_message(_control_message(3, 2.0))

    assert receiver.pop_latest_planner_control(1.9) is None
    assert receiver.pop_latest_planner_control(2.0)["sequence"] == 3


def test_pop_latest_planner_control_prefers_sequence_for_equal_timestamp():
    receiver = DataReceiver()
    receiver._receive_message(_control_message(4, 3.0, 0.4))
    receiver._receive_message(_control_message(5, 3.0, 0.5))

    selected = receiver.pop_latest_planner_control(3.0)

    assert selected["sequence"] == 5
    assert selected["acceleration_mps2"] == 0.5


def test_pop_latest_planner_control_isolates_vehicle_sessions():
    receiver = DataReceiver()
    receiver._receive_message(
        _control_message(6, 4.0, 0.6, session_id="vehicle-a")
    )
    receiver._receive_message(
        _control_message(7, 4.0, 0.7, session_id="vehicle-b")
    )

    selected_a = receiver.pop_latest_planner_control(
        4.0, session_id="vehicle-a"
    )
    selected_b = receiver.pop_latest_planner_control(
        4.0, session_id="vehicle-b"
    )

    assert selected_a["sequence"] == 6
    assert selected_a["acceleration_mps2"] == 0.6
    assert selected_b["sequence"] == 7
    assert selected_b["acceleration_mps2"] == 0.7


def test_shared_receiver_uses_reference_counted_singleton():
    first = DataReceiver.acquire_shared(port=65060, start=False)
    second = DataReceiver.acquire_shared(port=65060, start=False)

    assert first is second
    assert first._shared_reference_count == 2

    DataReceiver.release_shared(first)
    assert second._shared_reference_count == 1
    assert DataReceiver._shared_instances[("127.0.0.1", 65060)] is second

    DataReceiver.release_shared(second)
    assert ("127.0.0.1", 65060) not in DataReceiver._shared_instances
