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
    """Receive ROS debug data and numeric planner controls.

    Exact-cycle waits remain available for offline equivalence checks. The
    runtime control path uses :meth:`pop_latest_planner_control` so CARLA does
    not block while a planner running at a lower rate computes its next plan.
    """

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

    def pop_latest_planner_control(self, cycle_time_s=None):
        """Return the newest available control without blocking.

        When ``cycle_time_s`` is provided, controls stamped in the future are
        retained for a later simulator tick. Older controls up to the chosen
        result are discarded because execution only needs the latest plan.
        """
        upper_bound = (
            None
            if cycle_time_s is None
            else float(cycle_time_s) + 1.0e-6
        )
        with self.planner_control_condition:
            eligible = [
                (key, control)
                for key, control in self.planner_controls.items()
                if upper_bound is None
                or float(control.get("timestamp_s", 0.0)) <= upper_bound
            ]
            if not eligible:
                return None

            _selected_key, selected = max(
                eligible,
                key=lambda item: (
                    float(item[1].get("timestamp_s", 0.0)),
                    int(item[1].get("sequence", 0)),
                ),
            )
            selected_time_s = float(selected.get("timestamp_s", 0.0))
            for key, control in list(self.planner_controls.items()):
                if (
                    float(control.get("timestamp_s", 0.0))
                    <= selected_time_s + 1.0e-9
                ):
                    self.planner_controls.pop(key, None)
            return dict(selected)

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
                    "stop_goal_active": bool(control_data.get("stop_goal_active", False)),
                    "mission_complete": bool(control_data.get("mission_complete", False)),
                    "emergency_stop": bool(control_data.get("emergency_stop", False)),
                    "session_id": str(control_data.get("session_id", "") or ""),
                    "emergency_stop_reason": str(control_data.get("emergency_stop_reason", "") or ""),
                    "safety_supervisor_reason": str(control_data.get("safety_supervisor_reason", "") or ""),
                    "emergency_brake_control_active": bool(control_data.get("emergency_brake_control_active", False)),
                    "front_gap_m": control_data.get("front_gap_m", ""),
                    "front_gap_actor_id": str(control_data.get("front_gap_actor_id", "") or ""),
                    "ego_x_m": control_data.get("ego_x_m", ""),
                    "ego_y_m": control_data.get("ego_y_m", ""),
                    "ego_heading_rad": control_data.get("ego_heading_rad", ""),
                    "behavior_decision": str(control_data.get("behavior_decision", "") or ""),
                    "behavior_fsm_state": str(control_data.get("behavior_fsm_state", "") or ""),
                    "mpc_status": str(control_data.get("mpc_status", "") or ""),
                    "mpc_fallback_reason": str(control_data.get("mpc_fallback_reason", "") or ""),
                    "mpc_replan_executed": bool(control_data.get("mpc_replan_executed", False)),
                    "reference_source": str(control_data.get("reference_source", "") or ""),
                    "route_sync_reason": str(control_data.get("route_sync_reason", "") or ""),
                    "route_progress_index": control_data.get("route_progress_index", ""),
                    "route_upcoming_turn_direction": str(control_data.get("route_upcoming_turn_direction", "") or ""),
                    "route_upcoming_turn_distance_m": control_data.get("route_upcoming_turn_distance_m", ""),
                    "route_remaining_distance_m": control_data.get("route_remaining_distance_m", ""),
                    "current_lane_id": control_data.get("current_lane_id", ""),
                    "route_physical_target_lane_id": control_data.get("route_physical_target_lane_id", ""),
                    "prediction_bridge_request_count": int(control_data.get("prediction_bridge_request_count", 0) or 0),
                    "prediction_bridge_success_count": int(control_data.get("prediction_bridge_success_count", 0) or 0),
                    "prediction_bridge_dropped_request_count": int(control_data.get("prediction_bridge_dropped_request_count", 0) or 0),
                    "prediction_bridge_attached_count": int(control_data.get("prediction_bridge_attached_count", 0) or 0),
                    "prediction_bridge_attached_mode_count": int(control_data.get("prediction_bridge_attached_mode_count", 0) or 0),
                    "prediction_bridge_latency_ms": float(control_data.get("prediction_bridge_latency_ms", 0.0) or 0.0),
                    "prediction_bridge_error": str(control_data.get("prediction_bridge_error", "") or ""),
                    "prediction_ros_revision": int(control_data.get("prediction_ros_revision", -1) or -1),
                    "prediction_ros_cached_object_count": int(control_data.get("prediction_ros_cached_object_count", 0) or 0),
                    "prediction_ros_cached_mode_count": int(control_data.get("prediction_ros_cached_mode_count", 0) or 0),
                    "prediction_ros_frame_object_count": int(control_data.get("prediction_ros_frame_object_count", 0) or 0),
                    "prediction_ros_frame_mode_count": int(control_data.get("prediction_ros_frame_mode_count", 0) or 0),
                    "prediction_ros_frame_age_s": control_data.get("prediction_ros_frame_age_s"),
                    "prediction_ros_scene_input_count": int(control_data.get("prediction_ros_scene_input_count", 0) or 0),
                    "prediction_ros_scene_object_count": int(control_data.get("prediction_ros_scene_object_count", 0) or 0),
                    "prediction_ros_scene_duplicate_count": int(control_data.get("prediction_ros_scene_duplicate_count", 0) or 0),
                    "prediction_ros_scene_track_ids": str(control_data.get("prediction_ros_scene_track_ids", "") or ""),
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
                # Keep the compact planner-control contract forward compatible.
                # The fields above normalize values required by execution, but
                # diagnostics must not disappear merely because a newer planner
                # schema added them before this receiver was updated.
                for name, value in control_data.items():
                    if name not in planner_control:
                        planner_control[name] = value
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
