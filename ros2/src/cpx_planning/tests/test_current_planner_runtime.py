from pathlib import Path
from unittest import mock

from cpx_planning.current_planner_runtime import (
    CurrentPlannerRuntime,
    _current_planning_module_root,
)


def test_runtime_constructs_current_source_with_ros_safe_configuration():
    map_planner = object()
    config = {"publish_cp_message": True, "draw_world_debug": True}
    with mock.patch(
        "cpx_planning.current_planner_runtime.CPXMPCPlannerBridge"
    ) as bridge_type:
        runtime = CurrentPlannerRuntime(
            config=config,
            map_planner=map_planner,
            actor_id=42,
        )

    kwargs = bridge_type.call_args.kwargs
    assert kwargs["vehicle_manager"] is runtime.vehicle_manager
    assert kwargs["map_planner"] is map_planner
    assert kwargs["config"]["publish_cp_message"] is False
    assert kwargs["config"]["draw_world_debug"] is False
    assert config["publish_cp_message"] is True
    assert Path(kwargs["config"]["mpc_config_path"]) == (
        _current_planning_module_root() / "MPC" / "mpc.yaml"
    )


def test_step_passes_normalized_external_tick_to_current_bridge():
    with mock.patch(
        "cpx_planning.current_planner_runtime.CPXMPCPlannerBridge"
    ) as bridge_type:
        bridge = bridge_type.return_value
        bridge_output = mock.Mock()
        bridge_output.diagnostics_dict.return_value = {"mpc_status": "solved"}
        bridge.execute_planning_pipeline.return_value = bridge_output
        runtime = CurrentPlannerRuntime(config={}, map_planner=object())

    output = runtime.step(
        sim_time_s=3.5,
        ego_pose={"x": 1.0, "y": 2.0, "z": 0.2, "heading_rad": 0.25},
        ego_speed_mps=4.0,
        object_snapshots=[{"actor_id": "peer", "x": 5.0, "y": 2.0}],
        cp_payload={"obstacles": []},
    )

    assert output is bridge_output
    assert runtime.vehicle_manager.world.timestamp_s == 3.5
    kwargs = bridge.update_external_information.call_args.kwargs
    assert kwargs["ego_speed_mps"] == 4.0
    assert kwargs["sim_time_s"] == 3.5
    assert kwargs["object_snapshots"][0]["actor_id"] == "peer"
    assert kwargs["cp_payload"] == {"obstacles": []}
    bridge.execute_planning_pipeline.assert_called_once_with()
    assert bridge.last_output is bridge_output
    assert bridge.last_debug == {"mpc_status": "solved"}


def test_set_destination_uses_numeric_location_boundary():
    with mock.patch(
        "cpx_planning.current_planner_runtime.CPXMPCPlannerBridge"
    ) as bridge_type:
        bridge = bridge_type.return_value
        runtime = CurrentPlannerRuntime(config={}, map_planner=object())

    runtime.set_destination(
        start={"x": 1.0, "y": 2.0, "z": 0.0},
        goal={"x": 8.0, "y": 9.0, "z": 0.5},
    )

    kwargs = bridge.set_destination.call_args.kwargs
    assert kwargs["start_location"].x == 1.0
    assert kwargs["start_location"].y == 2.0
    assert kwargs["end_location"].x == 8.0
    assert kwargs["end_location"].y == 9.0
