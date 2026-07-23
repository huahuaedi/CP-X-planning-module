"""Transmit OpenCDA localization and perception outputs to ROS 2."""

import atexit
import json
import math
import socket
import threading
import time


ROS_HOST = "127.0.0.1"

# Each future ROS publisher node listens on a different TCP port.
STREAM_PORTS = {
    "localization": 5051,
    "perception": 5052,
    "traffic_light": 5053,
}

_connections = {
    "localization": None,
    "perception": None,
    "traffic_light": None,
}

_sequence = 0
_connection_lock = threading.Lock()


def send(
    localization_transform,
    localization_speed_kmh,
    perception_objects,
    timestamp_s=None,
    timeout_s=1.0,
    raise_on_error=False,
):
    """
    Send actual OpenCDA localization and perception outputs to ROS.

    Parameters
    ----------
    localization_transform
        Output of:
        self.localizer.get_ego_pos()

    localization_speed_kmh
        Output of:
        self.localizer.get_ego_spd()

    perception_objects
        Output of:
        self.perception_manager.detect(ego_pos)

        Expected to contain:
        {
            "vehicles": [...],
            "traffic_lights": [...]
        }

    timestamp_s
        Optional timestamp. The current computer time is used if omitted.

    Returns
    -------
    dict
        Success status for each ROS publisher node.
    """
    global _sequence

    if localization_transform is None:
        raise ValueError("Localization transform is missing.")

    if not isinstance(perception_objects, dict):
        raise TypeError(
            "Perception output must be the dictionary returned by "
            "PerceptionManager.detect()."
        )

    if timestamp_s is None:
        timestamp_s = time.time()

    # ------------------------------------------------------------------
    # Convert OpenCDA localization output into plain numerical data.
    # ------------------------------------------------------------------
    location = localization_transform.location
    rotation = localization_transform.rotation

    localization_data = {
        "ego_state": {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
            "v": float(localization_speed_kmh) / 3.6,
            "psi": math.radians(float(rotation.yaw)),
        },
        "units": {
            "position": "m",
            "speed": "m/s",
            "heading": "rad",
        },
    }

    # ------------------------------------------------------------------
    # Convert OpenCDA perceived vehicles into plain numerical data.
    # ------------------------------------------------------------------
    nearby_objects = []

    for index, perceived_object in enumerate(
        perception_objects.get("vehicles", [])
    ):
        try:
            transform = perceived_object.get_transform()
            object_location = perceived_object.get_location()
            velocity = perceived_object.get_velocity()
            bounding_box = getattr(
                perceived_object,
                "bounding_box",
                None,
            )

            if object_location is None:
                continue

            speed_mps = 0.0
            if velocity is not None:
                speed_mps = math.sqrt(
                    float(velocity.x) ** 2
                    + float(velocity.y) ** 2
                    + float(velocity.z) ** 2
                )

            # Sensor-detected objects may not have a known heading.
            heading_rad = None
            if transform is not None:
                heading_rad = math.radians(
                    float(transform.rotation.yaw)
                )

            extent = getattr(bounding_box, "extent", None)

            length_m = None
            width_m = None
            height_m = None

            if extent is not None:
                length_m = 2.0 * float(extent.x)
                width_m = 2.0 * float(extent.y)
                height_m = 2.0 * float(extent.z)

            object_id = getattr(
                perceived_object,
                "carla_id",
                index,
            )

            nearby_objects.append({
                "id": str(object_id),
                "vehicle_id": str(object_id),
                "type": str(
                    getattr(
                        perceived_object,
                        "type_id",
                        "vehicle",
                    )
                ),
                "x": float(object_location.x),
                "y": float(object_location.y),
                "z": float(object_location.z),
                "v": float(speed_mps),
                "psi": heading_rad,
                "length_m": length_m,
                "width_m": width_m,
                "height_m": height_m,
                "confidence": float(
                    getattr(perceived_object, "confidence", 1.0)
                ),
            })

        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            print(
                "[Data Transmitter] Could not convert perceived object "
                "{}: {}".format(index, error)
            )

    perception_data = {
        "objects": nearby_objects,
        "units": {
            "position": "m",
            "speed": "m/s",
            "heading": "rad",
            "dimensions": "m",
        },
    }

    # ------------------------------------------------------------------
    # Traffic lights are included in the OpenCDA perception output.
    # ------------------------------------------------------------------
    traffic_lights = []

    for index, traffic_light in enumerate(
        perception_objects.get("traffic_lights", [])
    ):
        try:
            light_location = traffic_light.get_location()
            light_state = traffic_light.get_state()
            light_actor = getattr(traffic_light, "actor", None)

            if light_location is None:
                continue

            state = str(light_state)

            # Convert values such as "TrafficLightState.Red" to "red".
            if "." in state:
                state = state.split(".")[-1]

            state = state.strip().lower()

            traffic_lights.append({
                "id": str(
                    getattr(light_actor, "id", index)
                ),
                "type": "traffic_light",
                "state": state,
                "position": {
                    "x": float(light_location.x),
                    "y": float(light_location.y),
                    "z": float(light_location.z),
                },
                "confidence": 1.0,
            })

        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            print(
                "[Data Transmitter] Could not convert traffic light "
                "{}: {}".format(index, error)
            )

    traffic_light_data = {
        "traffic_lights": traffic_lights,
        "units": {
            "position": "m",
        },
    }

    stream_data = {
        "localization": localization_data,
        "perception": perception_data,
        "traffic_light": traffic_light_data,
    }

    results = {}

    # ------------------------------------------------------------------
    # Send each OpenCDA output to its corresponding ROS publisher node.
    # ------------------------------------------------------------------
    with _connection_lock:
        _sequence += 1
        current_sequence = int(_sequence)

        for stream_name, converted_data in stream_data.items():
            message = {
                "schema_version": 1,
                "message_type": stream_name,
                "sequence": current_sequence,
                "timestamp_s": float(timestamp_s),
                "frame_id": "map",
                "data": converted_data,
            }

            encoded_message = (
                json.dumps(
                    message,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")

            port = STREAM_PORTS[stream_name]
            sent_successfully = False
            last_error = None

            # Try twice so that a broken connection can be recreated.
            for attempt in range(2):
                try:
                    if _connections[stream_name] is None:
                        connection = socket.create_connection(
                            (ROS_HOST, port),
                            timeout=float(timeout_s),
                        )
                        connection.settimeout(float(timeout_s))
                        _connections[stream_name] = connection

                    _connections[stream_name].sendall(encoded_message)
                    sent_successfully = True
                    break

                except OSError as error:
                    last_error = error

                    if _connections[stream_name] is not None:
                        try:
                            _connections[stream_name].close()
                        except OSError:
                            pass

                    _connections[stream_name] = None

            results[stream_name] = sent_successfully

            if not sent_successfully:
                message = (
                    "Could not send {} output to ROS at {}:{}: {}"
                ).format(
                    stream_name,
                    ROS_HOST,
                    port,
                    last_error,
                )

                if raise_on_error:
                    raise ConnectionError(message)

                print("[Data Transmitter] {}".format(message))

    return results


def close():
    """Close all connections to the ROS publisher nodes."""
    with _connection_lock:
        for stream_name in _connections:
            connection = _connections[stream_name]

            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

                _connections[stream_name] = None


atexit.register(close)