"""Receive the ROS planner output used by OpenCDA."""

import json
import math
import socket
import threading
import time


ROS_HOST = "127.0.0.1"
ROS_OUTPUT_PORT = 5060


def _cycle_key(timestamp_s):
    """Create a stable key for matching one OpenCDA and ROS planning cycle."""
    return "{:.6f}".format(float(timestamp_s))


def _as_dict(value):
    """Return a dictionary when the received JSON field is an object."""
    return value if isinstance(value, dict) else {}


class DataReceiver(object):
    """Receive same-cycle ROS debug data and numeric planner controls."""

    def __init__(self, host=ROS_HOST, port=ROS_OUTPUT_PORT):
        self.host = str(host)
        self.port = int(port)
        self.server_socket = None
        self.connection = None
        self.planner_controls = {}
        self.planner_control_condition = threading.Condition()
        self.planner_results = {}
        self.planner_result_condition = threading.Condition()
        self.stop_event = threading.Event()
        self.receiver_thread = None

    def start(self):
        """Start the TCP receiver in the background without blocking OpenCDA."""
        if self.receiver_thread is not None and self.receiver_thread.is_alive():
            return self
        self.stop_event.clear()
        self.receiver_thread = threading.Thread(target=self.run, name="ros-data-receiver", daemon=True)
        self.receiver_thread.start()
        return self

    def get_planner_result(self, cycle_time_s, timeout=2.0):
        """Wait for the ROS debug result belonging to one simulation cycle."""
        key = _cycle_key(cycle_time_s)
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.planner_result_condition:
            while key not in self.planner_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self.planner_result_condition.wait(remaining)
            return self.planner_results.pop(key)

    def get_planner_control(self, cycle_time_s, timeout=2.0):
        """Wait for the numeric ROS control belonging to one simulation cycle."""
        key = _cycle_key(cycle_time_s)
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self.planner_control_condition:
            while key not in self.planner_controls:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self.planner_control_condition.wait(remaining)
            return self.planner_controls.pop(key)

    def run(self):
        """Accept ROS connections until the receiver is closed."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(1.0)
        self.server_socket = server
        print("[ROS Data Receiver] Waiting on {}:{}.".format(self.host, self.port))
        while not self.stop_event.is_set():
            try:
                connection, address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self.stop_event.is_set():
                    break
                raise
            self.connection = connection
            connection.settimeout(1.0)
            print("[ROS Data Receiver] ROS connected from {}:{}.".format(address[0], address[1]))
            try:
                self._read_connection(connection)
            except (ConnectionResetError, OSError):
                pass
            finally:
                connection.close()
                self.connection = None

    def _read_connection(self, connection):
        """Split the TCP byte stream into newline-delimited JSON messages."""
        receive_buffer = b""
        while not self.stop_event.is_set():
            try:
                received = connection.recv(65536)
            except socket.timeout:
                continue
            if not received:
                return
            receive_buffer += received
            while b"\n" in receive_buffer:
                raw_message, receive_buffer = receive_buffer.split(b"\n", 1)
                if raw_message.strip():
                    self._receive_message(raw_message)

    def _receive_message(self, raw_message):
        """Store only planner controls and debug results; other ROS topics are ignored."""
        try:
            message = json.loads(raw_message.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            print("[ROS Data Receiver] Invalid JSON: {}.".format(error))
            return
        message_type = str(message.get("message_type", "")).strip().lower()
        if message_type == "planner_control":
            output_received_wall_time_ns = time.time_ns()
            output_received_perf_counter_ns = time.perf_counter_ns()
            control_data = _as_dict(message.get("data"))
            try:
                source_send_wall_time_ns = int(control_data.get("source_send_wall_time_ns", 0) or 0)
                planner_control = {
                    "sequence": int(message.get("sequence", 0) or 0),
                    "timestamp_s": float(control_data.get("cycle_time_s", message.get("timestamp_s", 0.0)) or 0.0),
                    "target_speed_mps": float(control_data.get("target_speed_mps", 0.0) or 0.0),
                    "acceleration_mps2": float(control_data["acceleration_mps2"]),
                    "steering_rad": float(control_data["steering_rad"]),
                    "planning_cycle_time_ms": float(control_data.get("planning_cycle_time_ms", 0.0) or 0.0),
                    "source_send_wall_time_ns": source_send_wall_time_ns,
                    "ros_output_created_wall_time_ns": int(control_data.get("ros_output_created_wall_time_ns", 0) or 0),
                    "ros_output_subscriber_received_wall_time_ns": int(control_data.get("ros_output_subscriber_received_wall_time_ns", 0) or 0),
                    "ros_output_tcp_send_started_wall_time_ns": int(control_data.get("ros_output_tcp_send_started_wall_time_ns", 0) or 0),
                    "source_to_output_created_ms": float(control_data.get("source_to_output_created_ms", 0.0) or 0.0),
                    "opencda_output_received_wall_time_ns": int(output_received_wall_time_ns),
                    "opencda_output_received_perf_counter_ns": int(output_received_perf_counter_ns),
                    "input_to_output_time_ms": (output_received_wall_time_ns - source_send_wall_time_ns) / 1000000.0 if source_send_wall_time_ns > 0 else 0.0,
                }
            except (KeyError, TypeError, ValueError) as error:
                print("[ROS Data Receiver] Invalid planner control: {}.".format(error))
                return
            if not all(math.isfinite(planner_control[name]) for name in ("target_speed_mps", "acceleration_mps2", "steering_rad", "planning_cycle_time_ms")):
                print("[ROS Data Receiver] Ignoring non-finite planner control.")
                return
            key = _cycle_key(planner_control["timestamp_s"])
            with self.planner_control_condition:
                self.planner_controls[key] = planner_control
                self._trim(self.planner_controls)
                self.planner_control_condition.notify_all()
            return
        if message_type == "debug_output":
            raw_payload = _as_dict(message.get("data")).get("data", "")
            try:
                planner_result = json.loads(str(raw_payload))
                key = _cycle_key(planner_result.get("cycle_time_s", 0.0))
            except (TypeError, ValueError) as error:
                print("[ROS Data Receiver] Invalid planner debug output: {}.".format(error))
                return
            with self.planner_result_condition:
                self.planner_results[key] = planner_result
                self._trim(self.planner_results)
                self.planner_result_condition.notify_all()

    @staticmethod
    def _trim(messages, max_count=100):
        """Bound queued results so a long simulation does not grow memory forever."""
        while len(messages) > int(max_count):
            messages.pop(sorted(messages.keys())[0], None)

    def close(self):
        """Stop the receiver and close its sockets."""
        self.stop_event.set()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.connection.close()
            except OSError:
                pass
            self.connection = None
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None
        if self.receiver_thread is not None and self.receiver_thread.is_alive() and self.receiver_thread is not threading.current_thread():
            self.receiver_thread.join(timeout=2.0)


def receive():
    """Run the receiver in the foreground until Ctrl+C."""
    receiver = DataReceiver()
    try:
        receiver.run()
    except KeyboardInterrupt:
        print("\n[ROS Data Receiver] Stopped.")
    finally:
        receiver.close()
