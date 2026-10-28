"""Static checks for the single-process OpenCDA communication boundary."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_communication_launch_starts_only_unified_bridge():
    source = (PACKAGE_ROOT / "launch" / "communication.launch.py").read_text()

    assert source.count("Node(") == 1
    assert 'executable="opencda_bridge"' in source
    assert 'executable="localization_publisher"' not in source
    assert 'executable="data_subscriber"' not in source


def test_unified_bridge_owns_only_planner_input_tcp_ports():
    source = (
        PACKAGE_ROOT / "cpx_comm_test" / "opencda_bridge_node.py"
    ).read_text()

    for port in range(5051, 5056):
        assert str(port) in source
    assert '"safety_status": 5056' not in source
    assert "AsyncLatestTcpJsonSender(" in source


def test_setup_registers_unified_bridge_entry_point():
    source = (PACKAGE_ROOT / "setup.py").read_text()

    assert (
        "opencda_bridge = cpx_comm_test.opencda_bridge_node:main" in source
    )


def test_control_forwarding_isolated_from_input_queue_drain():
    source = (
        PACKAGE_ROOT / "cpx_comm_test" / "opencda_bridge_node.py"
    ).read_text()

    assert "MultiThreadedExecutor(num_threads=2)" in source
    assert "self._input_worker_loop" in source
    assert "self._input_wake.set" in source
    assert "self._input_worker.start()" in source
    assert "self._input_guard" not in source
    assert "self._output_callback_group" in source
    assert "callback_group=self._output_callback_group" in source
    assert "max_messages=INPUT_BATCH_PER_STREAM" in source
    assert "self.output_sender.send_latest(" in source
    assert "reliability=ReliabilityPolicy.BEST_EFFORT" in source


def test_atomic_cycle_emits_one_compact_timing_event():
    source = (
        PACKAGE_ROOT / "cpx_comm_test" / "opencda_bridge_node.py"
    ).read_text()

    assert '"planning_cycle"' in source
    assert "logical_streams=published_streams" in source
    assert "for logical_stream in published_streams" not in source
    assert '"session_id": str(control.get("session_id", "") or "")' in source
    assert '"cycle_id": cycle_id_from_timestamp(cycle_time_s)' in source


def test_atomic_planning_cycle_preserves_legacy_topic_fanout():
    source = (
        PACKAGE_ROOT / "cpx_comm_test" / "opencda_bridge_node.py"
    ).read_text()

    assert 'data.get("message_type") == "planning_cycle"' in source
    assert "def _publish_planning_cycle(self, data):" in source
    assert 'payload.get("streams", {})' in source
    assert "for stream in INPUT_STREAM_ORDER:" in source


def test_bridge_forwards_opencda_session_without_safety_status():
    source = (
        PACKAGE_ROOT / "cpx_comm_test" / "opencda_bridge_node.py"
    ).read_text()

    assert '"session_id": str(data.get("session_id", "") or "")' in source
    assert '"planner_config": dict(data.get("planner_config", {}) or {})' in source
    assert '"status": dict(self._payload(data).get("status"' not in source
    assert 'String, "/cpx/planning/session"' in source


def test_bridge_embeds_session_identity_in_internal_input_headers():
    source = (
        PACKAGE_ROOT / "cpx_comm_test" / "opencda_bridge_node.py"
    ).read_text()

    assert "def _session_frame_id(data):" in source
    assert '"{}|cpx_session={}"' in source
    assert "message.header.frame_id = _session_frame_id(data)" in source


def test_integration_keeps_verbose_bridge_debug_off_control_path():
    integration_launch = (
        PACKAGE_ROOT.parent
        / "cpx_planning"
        / "launch"
        / "opencda_integration.launch.py"
    ).read_text(encoding="utf-8")

    assert (
        'bridge_debug = LaunchConfiguration("bridge_debug")'
        in integration_launch
    )
    assert (
        'DeclareLaunchArgument("bridge_debug", default_value="false")'
        in integration_launch
    )
    assert '"debug": bridge_debug' in integration_launch
    assert '"debug": debug' in integration_launch
