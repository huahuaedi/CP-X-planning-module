"""Optional transport timing for the OpenCDA-to-ROS boundary."""

import json
import time

from std_msgs.msg import String
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


debug_time = False


def configure_timing(node, stream):
    """Create one timing publisher only when the debug_time parameter is true."""
    node.declare_parameter("debug_time", debug_time)
    enabled = bool(node.get_parameter("debug_time").value)
    # Timing is observability data, not control data.  A reliable queue used to
    # feed every timing sample back through every planner shard and could make
    # instrumentation itself compete with the 20 Hz input callbacks.  Keep a
    # small best-effort queue so a slow consumer drops diagnostics rather than
    # applying back-pressure to the control path.
    timing_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=32,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    publisher = (
        node.create_publisher(String, "/cpx/timing/events", timing_qos)
        if enabled else None
    )
    return enabled, publisher, str(stream)


def cycle_id_from_timestamp(timestamp_s):
    """Match the planner's microsecond-quantized simulation cycle key."""
    quantum_ns = 1000
    raw_nanoseconds = max(0, int(round(float(timestamp_s or 0.0) * 1e9)))
    return ((raw_nanoseconds + quantum_ns // 2) // quantum_ns) * quantum_ns


# Compatibility for code and tests that imported the former private helper.
_cycle_id = cycle_id_from_timestamp


def publish_transport_timing(
    publisher,
    data,
    stream,
    publish_started_ns,
    publish_finished_ns,
    logical_streams=None,
):
    """Publish TCP receipt and ROS publication times without changing the real data."""
    if publisher is None:
        return
    timing = dict(data.get("_timing", {}) or {})
    source_send_ns = int(data.get("cycle_send_started_wall_time_ns", 0) or 0)
    tcp_received_ns = int(timing.get("tcp_received_wall_time_ns", 0) or 0)
    event = {
        "cycle_id": cycle_id_from_timestamp(data.get("timestamp_s", 0.0)),
        "stream": str(stream),
        "session_id": str(data.get("session_id", "") or ""),
        "source_send_wall_time_ns": source_send_ns,
        "tcp_received_wall_time_ns": tcp_received_ns,
        "ros_publish_started_wall_time_ns": int(publish_started_ns),
        "ros_publish_finished_wall_time_ns": int(publish_finished_ns),
    }
    if logical_streams:
        event["logical_streams"] = [str(name) for name in logical_streams]
    if source_send_ns > 0 and tcp_received_ns > 0:
        event["source_to_tcp_receive_ms"] = (tcp_received_ns - source_send_ns) / 1000000.0
    if source_send_ns > 0:
        event["input_transfer_source_to_ros_publish_ms"] = (publish_finished_ns - source_send_ns) / 1000000.0
    if tcp_received_ns > 0:
        event["tcp_receive_to_ros_publish_ms"] = (publish_started_ns - tcp_received_ns) / 1000000.0
        event["publisher_node_scheduling_ms"] = (publish_started_ns - tcp_received_ns) / 1000000.0
    event["ros_conversion_and_publish_ms"] = (publish_finished_ns - publish_started_ns) / 1000000.0
    message = String()
    message.data = json.dumps(event, allow_nan=False, separators=(",", ":"))
    publisher.publish(message)
