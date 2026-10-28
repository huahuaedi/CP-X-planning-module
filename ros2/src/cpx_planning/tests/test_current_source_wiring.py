import importlib.util
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path):
    return (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")


def test_planner_node_uses_current_runtime_not_copied_planner_core():
    source = _source("cpx_planning/planner_node.py")

    assert "CurrentPlannerRuntime" in source
    assert "planner_core.cpx_mpc_planner" not in source
    assert "DistributedInputAdapter" not in source


def test_planner_callbacks_buffer_exact_cycles_outside_planning_worker():
    source = _source("cpx_planning/planner_node.py")
    receive_source = source.split("    def _receive(self, input_name, message):", 1)[1]
    receive_source = receive_source.split("    def _planning_worker_loop(self):", 1)[0]

    assert "SynchronizedInputBuffer" in source
    assert "wait_for_latest_complete" in source
    assert "run_planning_cycle()" not in receive_source
    # Callbacks only decode ownership and append to the session-aware buffer.
    assert "_input_message_session_id" in receive_source
    assert "session_id=incoming_session_id" in receive_source
    assert "with self._planning_lock" not in receive_source


def test_new_opencda_session_resets_planner_before_applying_the_frame():
    source = _source("cpx_planning/planner_node.py")
    worker_source = source.split(
        "    def _planning_worker_loop(self):", 1
    )[1].split("    def _receive_timing_event", 1)[0]

    assert "self._activate_frame_session(" in worker_source
    assert "session_id=frame.session_id" in worker_source
    assert worker_source.index("self._activate_frame_session(") < worker_source.index("self._apply_message(input_name, message)")
    assert "self._session_contexts" in source
    assert "def _create_session_context(" in source
    assert "context.planner_runtime.close()" in source


def test_existing_session_switch_uses_constant_time_context_restore():
    source = _source("cpx_planning/planner_node.py")
    activation_source = source.split(
        "    def _activate_frame_session(self, messages, session_id=None):", 1
    )[1].split("    def _receive_timing_event", 1)[0]

    fast_path = activation_source.index(
        "if existing_context is not None and not run_changed:"
    )
    map_resolution = activation_source.index(
        "desired_xodr_path = resolve_global_planner_xodr_path("
    )
    assert fast_path < map_resolution
    assert "self._load_session_context(existing_context)" in activation_source
    assert "self._frame_session_payload = session_payload" in activation_source


def test_simulator_safety_status_is_not_a_planner_input():
    node_source = _source("cpx_planning/planner_node.py")
    adapter_source = _source("cpx_planning/ros_input_adapter.py")

    assert '"safety_status"' not in node_source
    assert '"/cpx/safety_status"' not in node_source
    assert "PlannerSafetyManager" not in adapter_source
    assert '"safety_manager"' not in adapter_source
    assert '"session"' in node_source
    assert '"/cpx/planning/session"' in node_source


def test_ros_input_adapter_accepts_session_qualified_map_headers():
    source = _source("cpx_planning/ros_input_adapter.py")

    assert 'frame_id.split("|cpx_session=", 1)[0]' in source
    assert "coordinate_frame != \"map\"" in source


def test_ros_output_uses_finalized_platform_speed_not_behavior_request():
    source = _source("cpx_planning/ros_output_adapter.py")

    assert "platform_target_speed_mps" in source
    assert "message.longitudinal.velocity = max(0.0, execution_speed_mps)" in source


def test_all_planner_json_boundaries_sanitize_non_finite_diagnostics():
    source = _source("cpx_planning/planner_node.py")

    assert "def _strict_json_dumps(" in source
    assert 'kwargs["allow_nan"] = False' in source
    assert "json.dumps(_json_safe(value), **kwargs)" in source
    assert "tcp_control_output.data = _strict_json_dumps(" in source
    assert "message.data = _strict_json_dumps(payload" in source
    assert "output_file.write(_strict_json_dumps(record" in source


def test_transport_callbacks_are_isolated_from_control_input_callbacks():
    source = _source("cpx_planning/planner_node.py")

    assert "self._input_callback_group = ReentrantCallbackGroup()" in source
    assert "self._prediction_callback_group = ReentrantCallbackGroup()" in source
    assert "self._timing_callback_group = ReentrantCallbackGroup()" in source
    assert "callback_group=self._input_callback_group" in source
    assert "callback_group=self._prediction_callback_group" in source
    assert "callback_group=self._timing_callback_group" in source
    assert "event_session_id and not owns_session(" in source
    assert "depth=4" in source
    assert "reliability=ReliabilityPolicy.BEST_EFFORT" in source


def test_planning_system_does_not_launch_copied_stage_nodes():
    source = _source("cpx_planning/planning_system.py")

    assert "GlobalPlannerNode" in source
    assert "CPXPlannerNode" in source
    for copied_stage_node in (
        "BehaviorContextNode",
        "BehaviorDecisionNode",
        "ReferencePlannerNode",
        "MPCNode",
    ):
        assert copied_stage_node not in source


def test_planning_system_keeps_one_session_aware_admap_owner():
    system_source = _source("cpx_planning/planning_system.py")
    node_source = _source("cpx_planning/planner_node.py")
    adapter_source = _source("cpx_planning/utility/global_planner.py")
    global_planner_source = _source("cpx_planning/global_planner_node.py")
    component_source = _source("cpx_planning/component_interfaces.py")

    assert "map_owner=global_planner_node" in system_source
    assert "resolve_global_planner_xodr_path" in node_source
    assert "map_activator(str(desired_xodr_path))" in node_source
    assert "def activate_map(" in adapter_source
    assert "def reset_dynamic_state(" in adapter_source
    assert "reset_for_new_run" in node_source
    assert "def get_lane_centerline(" in adapter_source
    assert "def get_dense_route_entries(" in adapter_source
    assert 'operation == "get_dense_route_entries"' in global_planner_source
    assert "def get_dense_route_entries(" in component_source

def test_ros_map_adapter_preserves_true_drivable_membership():
    adapter_source = _source("cpx_planning/utility/global_planner.py")
    bridge_spec = importlib.util.find_spec(
        "opencda.planning_module.opencda_bridge.cpx_mpc_planner"
    )
    assert bridge_spec is not None and bridge_spec.origin is not None
    bridge_source = Path(bridge_spec.origin).read_text(encoding="utf-8")

    assert "def get_drivable_waypoint(" in adapter_source
    assert "candidate.get(\"is_in_lane\", False)" in adapter_source
    assert "return port.drivable_waypoint(location) or port.waypoint(location)" not in bridge_source
    assert "return port.drivable_waypoint(location)" in bridge_source


def test_ros_admap_waypoints_preserve_route_corridor_geometry():
    waypoint_source = _source(
        "cpx_planning/Global_Planner/global_planner/waypoint.py"
    )
    planner_source = _source(
        "cpx_planning/Global_Planner/global_planner/planner.py"
    )

    assert "def transform(" in waypoint_source
    assert "left_boundary_position" in waypoint_source
    assert "right_boundary_position" in waypoint_source
    assert "backend.get_lane_boundaries_enu" in planner_source
    assert "include_boundaries=False" in planner_source
    backend_source = _source(
        "cpx_planning/Global_Planner/global_planner/admap_backend.py"
    )
    assert "def get_lane_boundaries_enu(" in backend_source
