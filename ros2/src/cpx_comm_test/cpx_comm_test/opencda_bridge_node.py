"""Single ROS 2 boundary between OpenCDA TCP streams and CP-X topics."""

import json
import math
import threading
import time
from urllib.parse import quote

from autoware_control_msgs.msg import Control
from autoware_perception_msgs.msg import TrackedObjects
from autoware_perception_msgs.msg import TrafficLightElement, TrafficLightGroup
from cpx_interfaces.msg import CooperativeMessageArray, LaneEvent
from cpx_interfaces.msg import TrafficLightObservation, TrafficLightObservationArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .cooperative_message_publisher import (
    EVENT_TYPES, _fill_position, _set_duration, _set_time,
)
from .tcp_json_receiver import TcpJsonReceiver
from .tcp_json_sender import (
    AsyncLatestTcpJsonSender,
    ros_message_to_dict,
    TcpJsonSender,
)
from .timing import (
    configure_timing,
    cycle_id_from_timestamp,
    publish_transport_timing,
)
from .runtime_pause_monitor import RuntimePauseMonitor
from .traffic_light_publisher import _group_id
from .v2x_publisher import _tracked_object_from_item


INPUT_PORTS = {
    "localization": 5051,
    "perception": 5052,
    "traffic_light": 5053,
    "v2x": 5054,
    "cooperative": 5055,
}
INPUT_STREAM_ORDER = (
    "localization", "perception", "traffic_light", "v2x", "cooperative",
)

# A bounded batch prevents six input queues from monopolizing the Python GIL
# while a fresh planner control is waiting to be sent back to OpenCDA.
INPUT_BATCH_PER_STREAM = 2


def _set_stamp(stamp, timestamp_s):
    seconds = int(timestamp_s)
    stamp.sec = seconds
    stamp.nanosec = int((float(timestamp_s) - seconds) * 1e9)


def _session_frame_id(data):
    """Carry transport ownership without changing any message schema.

    These topics are an internal CP-X planning boundary rather than TF input.
    Encoding the session in ``frame_id`` lets equal simulation timestamps from
    multiple CAVs remain distinguishable until the atomic planner buffer.
    """
    frame_id = str(data.get("frame_id", "map") or "map")
    session_id = str(data.get("session_id", "") or "")
    if not session_id:
        return frame_id
    return "{}|cpx_session={}".format(
        frame_id,
        quote(session_id, safe=""),
    )


class OpenCDABridgeNode(Node):
    """Own all OpenCDA TCP connections and matching ROS boundary topics."""

    def __init__(self):
        super().__init__("opencda_bridge_node")
        self.declare_parameter("debug", False)
        self.debug = bool(self.get_parameter("debug").value)
        self._runtime_pause_monitor = RuntimePauseMonitor()
        _, self.timing_publisher, _ = configure_timing(self, "opencda_bridge")

        # These are high-rate, full-snapshot internal topics. Every atomic
        # source cycle republishes session/configuration and all sensor state,
        # while the Planner's exact-cycle buffer already rejects incomplete
        # frames. A small best-effort queue therefore preserves the newest
        # complete frame without allowing a slow shard/DDS reader to block the
        # TCP receiver thread through reliable-writer back-pressure.
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        compact_control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.localization_publisher = self.create_publisher(
            Odometry, "/cpx/localization", input_qos)
        self.destination_publisher = self.create_publisher(
            PoseStamped, "/cpx/final_destination", input_qos)
        self.perception_publisher = self.create_publisher(
            TrackedObjects, "/cpx/perception", input_qos)
        self.traffic_light_publisher = self.create_publisher(
            TrafficLightObservationArray, "/cpx/traffic_light", input_qos)
        self.v2x_publisher = self.create_publisher(
            TrackedObjects, "/cpx/v2x", input_qos)
        self.cp_obstacles_publisher = self.create_publisher(
            TrackedObjects, "/cpx/cp_obstacles", input_qos)
        self.cooperative_publisher = self.create_publisher(
            CooperativeMessageArray, "/cpx/cooperative_messages", input_qos)
        self.session_publisher = self.create_publisher(
            String, "/cpx/planning/session", input_qos)

        self.output_sender = AsyncLatestTcpJsonSender(
            5060,
            self.get_logger(),
            on_complete=self._output_send_completed,
        )
        self.output_sequence = 0
        # TCP input conversion runs on its own bounded worker rather than a
        # guard-condition callback.  Under sustained two-CAV traffic the
        # Python rclpy executor could leave both the guard and control
        # subscription unscheduled for more than a second even though the
        # conversion itself took only a few milliseconds.  ROS publishers are
        # thread-safe, so separating input drain from executor dispatch keeps
        # the output callback available without adding another ROS node.
        self._input_stop = threading.Event()
        self._input_wake = threading.Event()
        self._output_callback_group = MutuallyExclusiveCallbackGroup()
        self._handlers = {
            "localization": self._publish_localization,
            "perception": self._publish_perception,
            "traffic_light": self._publish_traffic_lights,
            "v2x": self._publish_v2x,
            "cooperative": self._publish_cooperative,
        }
        self.receivers = {}
        try:
            for stream, port in INPUT_PORTS.items():
                receiver = TcpJsonReceiver(
                    port,
                    self.get_logger(),
                    on_message=self._input_wake.set,
                )
                self.receivers[stream] = receiver
                if not receiver.wait_until_ready():
                    raise RuntimeError(
                        "OpenCDA TCP port {} ({}) is unavailable: {}".format(
                            port, stream, receiver.bind_error or "bind timeout"
                        )
                    )
        except Exception:
            for receiver in self.receivers.values():
                receiver.close()
            raise

        self._input_worker = threading.Thread(
            target=self._input_worker_loop,
            name="opencda-input-publisher",
            daemon=True,
        )
        self._input_worker.start()

        self.create_subscription(
            String, "/cpx/planner_control_output",
            self._forward_planner_control, compact_control_qos,
            callback_group=self._output_callback_group)
        if self.debug:
            self._create_debug_subscriptions()
        self.get_logger().info(
            "Unified OpenCDA bridge owns TCP ports 5051-5055 and 5060; "
            "simulator SafetyManager telemetry stays outside planning.")

    @staticmethod
    def _payload(data):
        return data.get("data", data)

    @staticmethod
    def _timestamp(data):
        return float(data.get("timestamp_s", 0.0) or 0.0)

    def _input_worker_loop(self):
        """Drain bounded TCP batches independently of rclpy scheduling."""
        while not self._input_stop.is_set():
            self._input_wake.wait(timeout=0.1)
            self._input_wake.clear()
            while (
                not self._input_stop.is_set()
                and self._publish_pending_inputs()
            ):
                pass

    def _publish_pending_inputs(self):
        """Publish one bounded batch and report whether work remains."""
        backlog_remaining = False
        for stream, receiver in self.receivers.items():
            for data in receiver.get_messages(
                    max_messages=INPUT_BATCH_PER_STREAM):
                started_ns = time.time_ns()
                if data.get("message_type") == "planning_cycle":
                    published_streams = self._publish_planning_cycle(data)
                    finished_ns = time.time_ns()
                    # The atomic cycle has one TCP receipt and one contiguous
                    # fan-out interval. Publish one timing event and let the
                    # recorder associate it with all logical streams. The old
                    # five-event form multiplied callback load by five without
                    # adding any timing information.
                    publish_transport_timing(
                        self.timing_publisher,
                        data,
                        "planning_cycle",
                        started_ns,
                        finished_ns,
                        logical_streams=published_streams,
                    )
                else:
                    self._handlers[stream](data)
                    publish_transport_timing(
                        self.timing_publisher, data, stream,
                        started_ns, time.time_ns())
            backlog_remaining = receiver.has_messages() or backlog_remaining
        return bool(backlog_remaining)

    def _publish_planning_cycle(self, data):
        """Fan one atomic simulator cycle out to the existing ROS topics."""
        payload = self._payload(data)
        streams = payload.get("streams", {})
        if not isinstance(streams, dict):
            self.get_logger().warning(
                "Ignoring planning_cycle without a streams mapping.")
            return ()

        missing = [name for name in INPUT_STREAM_ORDER if name not in streams]
        if missing:
            self.get_logger().warning(
                "Ignoring incomplete planning_cycle; missing: {}".format(
                    ", ".join(missing)))
            return ()

        for stream in INPUT_STREAM_ORDER:
            child = dict(data)
            child["message_type"] = stream
            child["data"] = streams[stream]
            self._handlers[stream](child)
        return INPUT_STREAM_ORDER

    def _publish_localization(self, data):
        self._publish_session(data)
        payload = self._payload(data)
        ego_state = payload.get("ego_state", {})
        message = Odometry()
        _set_stamp(message.header.stamp, self._timestamp(data))
        message.header.frame_id = _session_frame_id(data)
        message.child_frame_id = "base_link"
        message.pose.pose.position.x = float(ego_state.get("x", 0.0))
        message.pose.pose.position.y = float(ego_state.get("y", 0.0))
        message.pose.pose.position.z = float(ego_state.get("z", 0.0))
        yaw = float(ego_state.get("psi", 0.0))
        message.pose.pose.orientation.z = math.sin(yaw / 2.0)
        message.pose.pose.orientation.w = math.cos(yaw / 2.0)
        message.twist.twist.linear.x = float(ego_state.get("v", 0.0))
        self.localization_publisher.publish(message)

        raw_destination = payload.get("final_destination")
        if isinstance(raw_destination, dict):
            destination = PoseStamped()
            destination.header = message.header
            destination.pose.position.x = float(raw_destination.get("x", 0.0))
            destination.pose.position.y = float(raw_destination.get("y", 0.0))
            destination.pose.position.z = float(raw_destination.get("z", 0.0))
            destination.pose.orientation.w = 1.0
            self.destination_publisher.publish(destination)

    def _tracked_objects(self, data, items):
        message = TrackedObjects()
        _set_stamp(message.header.stamp, self._timestamp(data))
        message.header.frame_id = _session_frame_id(data)
        message.objects = [_tracked_object_from_item(item) for item in items]
        return message

    def _publish_perception(self, data):
        message = self._tracked_objects(
            data, self._payload(data).get("objects", []))
        self.perception_publisher.publish(message)

    def _publish_v2x(self, data):
        payload = self._payload(data)
        self.v2x_publisher.publish(self._tracked_objects(
            data, payload.get("nearby_cavs", [])))
        self.cp_obstacles_publisher.publish(self._tracked_objects(
            data, payload.get("cp_obstacles", [])))

    def _publish_traffic_lights(self, data):
        message = TrafficLightObservationArray()
        _set_stamp(message.header.stamp, self._timestamp(data))
        message.header.frame_id = _session_frame_id(data)
        colors = {
            "red": TrafficLightElement.RED,
            "yellow": TrafficLightElement.AMBER,
            "amber": TrafficLightElement.AMBER,
            "green": TrafficLightElement.GREEN,
            "white": TrafficLightElement.WHITE,
        }
        for item in self._payload(data).get("traffic_lights", []):
            observation = TrafficLightObservation()
            observation.source_id = str(item.get("id", ""))
            observation.type = str(item.get("type", "traffic_light"))
            observation.state = str(item.get("state", "unknown"))
            position = item.get("position", {})
            if isinstance(position, dict):
                observation.position.x = float(position.get("x", 0.0))
                observation.position.y = float(position.get("y", 0.0))
                observation.position.z = float(position.get("z", 0.0))
            observation.confidence = float(item.get("confidence", 1.0))
            element = TrafficLightElement()
            element.color = colors.get(
                observation.state.strip().lower(), TrafficLightElement.UNKNOWN)
            element.shape = TrafficLightElement.CIRCLE
            element.status = TrafficLightElement.SOLID_ON
            element.confidence = observation.confidence
            group = TrafficLightGroup()
            group.traffic_light_group_id = _group_id(observation.source_id)
            group.elements = [element]
            observation.signal = group
            message.traffic_lights.append(observation)
        self.traffic_light_publisher.publish(message)

    def _publish_cooperative(self, data):
        payload = self._payload(data)
        timestamp_s = float(payload.get("timestamp_s", self._timestamp(data)))
        message = CooperativeMessageArray()
        _set_time(message.header.stamp, timestamp_s)
        message.header.frame_id = _session_frame_id(data)
        message.schema_version = int(payload.get("schema_version", 1))
        message.sequence = int(payload.get("sequence", 0))
        for item in payload.get("lane_events", []):
            event = LaneEvent()
            event.id = str(item.get("id", ""))
            event.event_type = EVENT_TYPES.get(
                str(item.get("type", "")).strip().lower(), LaneEvent.UNKNOWN)
            event.source = str(item.get("source", ""))
            _set_time(event.source_stamp, item.get("timestamp_s", timestamp_s))
            _set_duration(event.ttl, item.get("ttl_s", 0.0))
            _fill_position(event.position, item.get("position", []))
            event.confidence = float(item.get("confidence", 1.0))
            message.lane_events.append(event)
        self.cooperative_publisher.publish(message)

    def _publish_session(self, data):
        """Publish only lifecycle and transport metadata, never safety flags."""
        message = String()
        message.data = json.dumps({
            "timestamp_s": self._timestamp(data),
            "session_id": str(data.get("session_id", "") or ""),
            "planner_config": dict(data.get("planner_config", {}) or {}),
            "cycle_send_started_wall_time_ns": int(
                data.get("cycle_send_started_wall_time_ns", 0) or 0),
            "tcp_received_wall_time_ns": int(
                dict(data.get("_timing", {}) or {}).get(
                    "tcp_received_wall_time_ns", 0) or 0),
        }, allow_nan=False, separators=(",", ":"))
        self.session_publisher.publish(message)

    def _create_debug_subscriptions(self):
        topics = {
            "localization": (Odometry, "/cpx/localization"),
            "perception": (TrackedObjects, "/cpx/perception"),
            "traffic_light": (TrafficLightObservationArray, "/cpx/traffic_light"),
            "v2x": (TrackedObjects, "/cpx/v2x"),
            "cp_obstacles": (TrackedObjects, "/cpx/cp_obstacles"),
            "cooperative": (CooperativeMessageArray, "/cpx/cooperative_messages"),
            "session": (String, "/cpx/planning/session"),
            "debug_output": (String, "/cpx/debug_output"),
            "planner_control": (Control, "/control/command/control_cmd"),
        }
        for name, (message_class, topic_name) in topics.items():
            self.create_subscription(
                message_class, topic_name,
                lambda message, data_type=name: self._log_debug_message(
                    data_type, message), 10)

    def _log_debug_message(self, data_type, message):
        if data_type == "planner_control":
            self.get_logger().info(
                "ROS control: acceleration={:.3f}, steering={:.3f}.".format(
                    message.longitudinal.acceleration,
                    message.lateral.steering_tire_angle))
        elif data_type == "debug_output":
            self.output_sequence += 1
            self.output_sender.send_latest({
                "schema_version": 1,
                "sequence": self.output_sequence,
                "timestamp_s": self.get_clock().now().nanoseconds / 1e9,
                "message_type": "debug_output",
                "data": ros_message_to_dict(message),
            }, key="__debug_output__")
            self.get_logger().info("ROS planner debug output received.")
        else:
            self.get_logger().info("{} data received.".format(data_type))

    def _forward_planner_control(self, message):
        received_ns = time.time_ns()
        try:
            control = json.loads(str(message.data))
            cycle_time_s = float(control["cycle_time_s"])
            control["ros_output_subscriber_received_wall_time_ns"] = received_ns
        except (KeyError, TypeError, ValueError):
            self.get_logger().warning("Invalid timed planner control received.")
            return

        self.output_sequence += 1
        enqueued_ns = time.time_ns()
        control["ros_output_tcp_enqueued_wall_time_ns"] = enqueued_ns
        self.output_sender.send_latest({
            "schema_version": 1,
            "sequence": self.output_sequence,
            "timestamp_s": cycle_time_s,
            "message_type": "planner_control",
            "data": control,
        }, key=str(control.get("session_id", "") or ""))

    def _output_send_completed(
        self,
        packet,
        enqueued_ns,
        send_started_ns,
        sent_ns,
        success,
    ):
        """Publish non-critical output timing after the socket worker returns."""

        if self.timing_publisher is not None:
            control = dict(dict(packet or {}).get("data", {}) or {})
            cycle_time_s = float(control.get("cycle_time_s", 0.0) or 0.0)
            received_ns = int(
                control.get("ros_output_subscriber_received_wall_time_ns", 0)
                or 0
            )
            event = String()
            event.data = json.dumps({
                "cycle_id": cycle_id_from_timestamp(cycle_time_s),
                "stream": "output_forwarder",
                "session_id": str(control.get("session_id", "") or ""),
                "output_topic_received_wall_time_ns": received_ns,
                "output_tcp_enqueued_wall_time_ns": int(enqueued_ns),
                "output_tcp_send_started_wall_time_ns": send_started_ns,
                "output_tcp_sent_wall_time_ns": sent_ns,
                "output_tcp_send_success": bool(success),
                "output_topic_to_tcp_send_ms": (sent_ns - received_ns) / 1e6,
                "output_callback_to_queue_ms": (
                    enqueued_ns - received_ns
                ) / 1e6,
                "output_queue_wait_ms": (
                    send_started_ns - enqueued_ns
                ) / 1e6,
                "output_callback_to_tcp_send_started_ms": (
                    send_started_ns - received_ns) / 1e6,
                "output_tcp_send_duration_ms": (
                    sent_ns - send_started_ns) / 1e6,
            }, allow_nan=False, separators=(",", ":"))
            self.timing_publisher.publish(event)

    def destroy_node(self):
        self._input_stop.set()
        self._input_wake.set()
        for receiver in self.receivers.values():
            receiver.close()
        input_worker = getattr(self, "_input_worker", None)
        if input_worker is not None and input_worker.is_alive():
            input_worker.join(timeout=2.0)
        self.output_sender.close()
        self.get_logger().info(
            "Python runtime pause summary: {}".format(
                self._runtime_pause_monitor.close()
            )
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OpenCDABridgeNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
