"""Receive ROS data over TCP and build complete CP-X planner inputs."""

import json
import math
import queue
import socket
import threading
from types import SimpleNamespace


ROS_HOST = "127.0.0.1"
ROS_OUTPUT_PORT = 5060

REQUIRED_INPUTS = (
    "localization",
    "perception",
    "v2x",
    "traffic_light",
)

OBJECT_LABELS = {
    0: "unknown",
    1: "car",
    2: "truck",
    3: "bus",
    4: "trailer",
    5: "motorcycle",
    6: "bicycle",
    7: "pedestrian",
}

TRAFFIC_LIGHT_COLORS = {
    0: "unknown",
    1: "red",
    2: "yellow",
    3: "green",
    4: "white",
}


class _PlannerTransform(object):
    """Small transform object matching the planner's current input interface."""

    def __init__(self, x, y, z, heading_rad):
        self.location = SimpleNamespace(
            x=float(x),
            y=float(y),
            z=float(z),
        )
        self.rotation = SimpleNamespace(
            yaw=math.degrees(float(heading_rad)),
        )


class _PlannerPerceivedVehicle(object):
    """Plain Python replacement for one OpenCDA perceived vehicle."""

    def __init__(self, snapshot):
        snapshot = dict(snapshot)
        heading_rad = _as_float(snapshot.get("psi"))
        speed_mps = _as_float(snapshot.get("v"))
        length_m = _as_float(snapshot.get("length_m"))
        width_m = _as_float(snapshot.get("width_m"))
        height_m = _as_float(snapshot.get("height_m"))

        self.id = str(
            snapshot.get("id", snapshot.get("vehicle_id", ""))
        )
        self.carla_id = self.id
        self.type_id = str(snapshot.get("type", "vehicle"))
        self.confidence = _as_float(
            snapshot.get("confidence"),
            1.0,
        )
        self._transform = _PlannerTransform(
            snapshot.get("x", 0.0),
            snapshot.get("y", 0.0),
            snapshot.get("z", 0.0),
            heading_rad,
        )
        self._velocity = SimpleNamespace(
            x=speed_mps * math.cos(heading_rad),
            y=speed_mps * math.sin(heading_rad),
            z=0.0,
        )
        self.bounding_box = SimpleNamespace(
            extent=SimpleNamespace(
                x=length_m / 2.0,
                y=width_m / 2.0,
                z=height_m / 2.0,
            )
        )

    def get_transform(self):
        return self._transform

    def get_location(self):
        return self._transform.location

    def get_velocity(self):
        return self._velocity


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _ros_seconds(value, default=0.0):
    value = _as_dict(value)
    if not value:
        return float(default)
    return (
        _as_float(value.get("sec"))
        + _as_float(value.get("nanosec")) / 1000000000.0
    )


def _message_time(data, default=0.0):
    header = _as_dict(data.get("header"))
    if header:
        return _ros_seconds(header.get("stamp"), default)
    return _ros_seconds(data.get("stamp"), default)


def _yaw_from_quaternion(value):
    value = _as_dict(value)
    x_value = _as_float(value.get("x"))
    y_value = _as_float(value.get("y"))
    z_value = _as_float(value.get("z"))
    w_value = _as_float(value.get("w"), 1.0)
    return math.atan2(
        2.0 * (w_value * z_value + x_value * y_value),
        1.0 - 2.0 * (y_value * y_value + z_value * z_value),
    )


def _speed_from_vector(value):
    value = _as_dict(value)
    x_value = _as_float(value.get("x"))
    y_value = _as_float(value.get("y"))
    z_value = _as_float(value.get("z"))
    return math.sqrt(
        x_value * x_value
        + y_value * y_value
        + z_value * z_value
    )


def _uuid_string(value):
    values = _as_dict(value).get("uuid", [])
    try:
        return "".join(
            "{:02x}".format(int(item) & 0xFF)
            for item in values
        )
    except (TypeError, ValueError):
        return str(values)


def _object_type(classifications):
    classifications = [
        item
        for item in list(classifications or [])
        if isinstance(item, dict)
    ]
    if not classifications:
        return "unknown"
    best = max(
        classifications,
        key=lambda item: _as_float(item.get("probability")),
    )
    return OBJECT_LABELS.get(
        _as_int(best.get("label")),
        "unknown",
    )


def _planner_object(raw_object, source):
    raw_object = _as_dict(raw_object)
    kinematics = _as_dict(raw_object.get("kinematics"))

    pose_with_covariance = _as_dict(
        kinematics.get(
            "initial_pose_with_covariance",
            kinematics.get("pose_with_covariance"),
        )
    )
    pose = _as_dict(pose_with_covariance.get("pose"))
    position = _as_dict(pose.get("position"))

    twist_with_covariance = _as_dict(
        kinematics.get(
            "initial_twist_with_covariance",
            kinematics.get("twist_with_covariance"),
        )
    )
    twist = _as_dict(twist_with_covariance.get("twist"))

    dimensions = _as_dict(
        _as_dict(raw_object.get("shape")).get("dimensions")
    )
    object_id = _uuid_string(raw_object.get("object_id"))
    x_value = _as_float(position.get("x"))
    y_value = _as_float(position.get("y"))
    z_value = _as_float(position.get("z"))
    speed_mps = _speed_from_vector(twist.get("linear"))
    heading_rad = _yaw_from_quaternion(pose.get("orientation"))
    length_m = _as_float(dimensions.get("x"))
    width_m = _as_float(dimensions.get("y"))
    height_m = _as_float(dimensions.get("z"))

    return {
        "id": object_id,
        "vehicle_id": object_id,
        "type": _object_type(raw_object.get("classification")),
        "x": x_value,
        "y": y_value,
        "z": z_value,
        "v": speed_mps,
        "psi": heading_rad,
        "state": [x_value, y_value, speed_mps, heading_rad],
        "length_m": length_m,
        "width_m": width_m,
        "height_m": height_m,
        "shape": {
            "length_m": length_m,
            "width_m": width_m,
            "height_m": height_m,
        },
        "confidence": _as_float(
            raw_object.get("existence_probability"),
            1.0,
        ),
        "source": source,
        "provider_source": source,
    }


def _convert_localization(message):
    data = _as_dict(message.get("data"))
    pose = _as_dict(_as_dict(data.get("pose")).get("pose"))
    position = _as_dict(pose.get("position"))
    twist = _as_dict(_as_dict(data.get("twist")).get("twist"))

    x_value = _as_float(position.get("x"))
    y_value = _as_float(position.get("y"))
    z_value = _as_float(position.get("z"))
    speed_mps = _speed_from_vector(twist.get("linear"))
    heading_rad = _yaw_from_quaternion(pose.get("orientation"))

    return {
        "timestamp_s": _message_time(
            data,
            _as_float(message.get("timestamp_s")),
        ),
        "frame_id": str(
            _as_dict(data.get("header")).get("frame_id", "map")
        ),
        "ego_pose": {
            "x": x_value,
            "y": y_value,
            "z": z_value,
            "heading_rad": heading_rad,
        },
        "ego_speed_mps": speed_mps,
        "ego_state": [x_value, y_value, speed_mps, heading_rad],
    }


def _convert_objects(message, source, output_name):
    data = _as_dict(message.get("data"))
    return {
        "timestamp_s": _message_time(
            data,
            _as_float(message.get("timestamp_s")),
        ),
        "frame_id": str(
            _as_dict(data.get("header")).get("frame_id", "map")
        ),
        output_name: [
            _planner_object(item, source)
            for item in list(data.get("objects", []) or [])
            if isinstance(item, dict)
        ],
    }


def _signal_state(group):
    elements = [
        item
        for item in list(_as_dict(group).get("elements", []) or [])
        if isinstance(item, dict)
    ]
    if not elements:
        return "unknown", 0.0
    best = max(
        elements,
        key=lambda item: _as_float(item.get("confidence")),
    )
    return (
        TRAFFIC_LIGHT_COLORS.get(
            _as_int(best.get("color")),
            "unknown",
        ),
        _as_float(best.get("confidence")),
    )


def _convert_traffic(message):
    data = _as_dict(message.get("data"))
    timestamp_s = _message_time(
        data,
        _as_float(message.get("timestamp_s")),
    )
    controls = []

    for group in list(data.get("traffic_light_groups", []) or []):
        if not isinstance(group, dict):
            continue
        control_id = str(group.get("traffic_light_group_id", ""))
        state, confidence = _signal_state(group)
        controls.append({
            "id": control_id,
            "type": "traffic_light",
            "control_id": control_id,
            "state": state,
            "signal_state": state,
            "timestamp_s": timestamp_s,
            "ttl_s": 0.0,
            "confidence": confidence,
            "source": "ros_traffic_light",
            "provider_source": "ros_traffic_light",
        })

    return {
        "timestamp_s": timestamp_s,
        "control": controls,
    }


def _convert_message(message_type, message):
    if message_type == "localization":
        return _convert_localization(message)
    if message_type == "perception":
        return _convert_objects(
            message,
            "ros_perception",
            "objects",
        )
    if message_type == "v2x":
        return _convert_objects(
            message,
            "ros_v2x",
            "obstacles",
        )
    return _convert_traffic(message)


class DataReceiver(object):
    """Return data only after all four ROS planner inputs have arrived."""

    def __init__(self, host=ROS_HOST, port=ROS_OUTPUT_PORT):
        self.host = str(host)
        self.port = int(port)
        self.server_socket = None
        self.connection = None
        self.received_data = queue.Queue()
        self.pending_messages = {}
        self.sequence = 0
        self.stop_event = threading.Event()
        self.receiver_thread = None

    def start(self):
        """Start receiving in a background thread."""
        if (
            self.receiver_thread is not None
            and self.receiver_thread.is_alive()
        ):
            return self

        self.stop_event.clear()
        self.receiver_thread = threading.Thread(
            target=self.run,
            name="ros-data-receiver",
            daemon=True,
        )
        self.receiver_thread.start()
        return self

    def get_received_data(self, timeout=None):
        """Return the next complete planner input, or None on timeout."""
        try:
            if timeout is None:
                return self.received_data.get()
            return self.received_data.get(
                timeout=max(0.0, float(timeout))
            )
        except queue.Empty:
            return None

    def run(self):
        """Accept ROS connections until the receiver is closed."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(1.0)
        self.server_socket = server

        print(
            "[ROS Data Receiver] Waiting on {}:{}.".format(
                self.host,
                self.port,
            )
        )

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
            print(
                "[ROS Data Receiver] ROS connected from {}:{}.".format(
                    address[0],
                    address[1],
                )
            )
            try:
                self._read_connection(connection)
            except (ConnectionResetError, OSError):
                pass
            finally:
                connection.close()
                self.connection = None

    def _read_connection(self, connection):
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
                raw_message, receive_buffer = receive_buffer.split(
                    b"\n",
                    1,
                )
                if raw_message.strip():
                    self._receive_message(raw_message)

    def _receive_message(self, raw_message):
        try:
            message = json.loads(raw_message.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            print("[ROS Data Receiver] Invalid JSON: {}.".format(error))
            return

        message_type = str(
            message.get("message_type", "")
        ).strip().lower()
        if message_type not in REQUIRED_INPUTS:
            return

        self.pending_messages[message_type] = message
        missing = [
            name
            for name in REQUIRED_INPUTS
            if name not in self.pending_messages
        ]
        if missing:
            return

        converted = {
            name: _convert_message(name, self.pending_messages[name])
            for name in REQUIRED_INPUTS
        }
        self.pending_messages.clear()
        self.sequence += 1

        localization = converted["localization"]
        perception = converted["perception"]
        v2x = converted["v2x"]
        traffic = converted["traffic_light"]
        timestamp_s = _as_float(localization.get("timestamp_s"))

        ego_pose = dict(localization.get("ego_pose", {}))
        perception_objects = list(perception.get("objects", []))
        cp_payload = {
            "schema_version": 1,
            "sequence": self.sequence,
            "timestamp_s": timestamp_s,
            "obstacles": list(v2x.get("obstacles", [])),
            "lane_events": [],
            "control": list(traffic.get("control", [])),
        }
        planner_input = {
            "sequence": self.sequence,
            "timestamp_s": timestamp_s,
            "ego_pos": _PlannerTransform(
                ego_pose.get("x", 0.0),
                ego_pose.get("y", 0.0),
                ego_pose.get("z", 0.0),
                ego_pose.get("heading_rad", 0.0),
            ),
            # VehicleManager and update_information currently use km/h.
            "ego_spd": (
                _as_float(localization.get("ego_speed_mps")) * 3.6
            ),
            "objects": {
                "vehicles": [
                    _PlannerPerceivedVehicle(snapshot)
                    for snapshot in perception_objects
                ],
                "traffic_lights": [],
            },
            "cp_payload": cp_payload,
        }
        self.received_data.put(planner_input)

        print(
            "[ROS -> OpenCDA] Complete planner input {}.".format(
                self.sequence
            )
        )

    def close(self):
        """Stop the receiver and close open sockets."""
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

        if (
            self.receiver_thread is not None
            and self.receiver_thread.is_alive()
            and self.receiver_thread is not threading.current_thread()
        ):
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
