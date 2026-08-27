"""Transmit OpenCDA localization, perception, V2X, and safety outputs to ROS 2."""

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
    "v2x": 5054,
    "cooperative": 5055,
    "safety": 5056,
}

_connections = {
    "localization": None,
    "perception": None,
    "traffic_light": None,
    "v2x": None,
    "cooperative": None,
    "safety": None,
}

_sequence = 0
_connection_lock = threading.Lock()


def _cp_obstacle_as_perception_object(obstacle):
    """Convert one CP obstacle into the same plain fields sent for perception objects."""
    if not isinstance(obstacle, dict):
        return None

    state = obstacle.get("state", [])
    state = list(state) if isinstance(state, (list, tuple)) else []
    x_m = obstacle.get("x", obstacle.get("x_m", state[0] if len(state) >= 1 else None))
    y_m = obstacle.get("y", obstacle.get("y_m", state[1] if len(state) >= 2 else None))
    speed_mps = obstacle.get("v", obstacle.get("speed_mps", state[2] if len(state) >= 3 else 0.0))
    heading_rad = obstacle.get("psi", obstacle.get("heading_rad", state[3] if len(state) >= 4 else None))
    if x_m is None or y_m is None:
        return None

    shape = obstacle.get("shape", {})
    shape = dict(shape) if isinstance(shape, dict) else {}
    raw_id = str(obstacle.get("id", obstacle.get("vehicle_id", ""))).strip()
    object_id = raw_id.rsplit(":", 1)[-1] if ":" in raw_id else raw_id
    if not object_id:
        return None

    return {
        "id": object_id,
        "vehicle_id": object_id,
        "type": str(obstacle.get("type", "unknown")),
        "x": float(x_m),
        "y": float(y_m),
        "z": float(obstacle.get("z", 0.0) or 0.0),
        "v": float(speed_mps or 0.0),
        "psi": None if heading_rad is None else float(heading_rad),
        "length_m": shape.get("length_m", obstacle.get("length_m")),
        "width_m": shape.get("width_m", obstacle.get("width_m")),
        "height_m": shape.get("height_m", obstacle.get("height_m")),
        "confidence": float(obstacle.get("confidence", 1.0)),
    }


def send(
    localization_transform,
    localization_speed_kmh,
    perception_objects,
    v2x_manager=None,
    safety_manager=None,
    final_destination=None,
    cp_payload=None,
    cooperative_payload=None,
    timestamp_s=None,
    timeout_s=1.0,
    raise_on_error=False,
):
    """
    Send actual OpenCDA localization, perception, and V2X outputs to ROS.

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

    v2x_manager
        The vehicle's updated OpenCDA V2XManager. Its ``cav_nearby``
        dictionary contains the other OpenCDA-managed CAVs currently within
        communication range.

    timestamp_s
        Optional timestamp. The current computer time is used if omitted.

    Returns
    -------
    dict
        Success status for each ROS publisher node.
    """
    global _sequence

    cycle_send_started_wall_time_ns = time.time_ns()

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
    }
    
    #adding final destination  to the localization data
    if isinstance(final_destination, dict):
        localization_data["final_destination"] = {
            "x": float(final_destination.get("x", 0.0)),
            "y": float(final_destination.get("y", 0.0)),
            "z": float(final_destination.get("z", 0.0)),
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
                "type": str(getattr(light_actor, "type_id", "traffic_light")),
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
    }

    # ------------------------------------------------------------------
    # Send CP obstacles through the same plain object boundary used by local
    # perception. ROS adds map and timing fields before the existing CP fusion.
    # ------------------------------------------------------------------
    raw_cp_payload = dict(cp_payload or {})
    cp_obstacles = []
    for obstacle in list(raw_cp_payload.get("obstacles", []) or []):
        converted_obstacle = _cp_obstacle_as_perception_object(obstacle)
        if converted_obstacle is not None:
            cp_obstacles.append(converted_obstacle)
    nearby_cavs = []
    v2x_neighbors = getattr(v2x_manager, "cav_nearby", {}) or {}
    for cav_id, nearby_vehicle_manager in dict(v2x_neighbors).items():
        try:
            nearby_v2x = nearby_vehicle_manager.v2x_manager
            nearby_transform = nearby_v2x.get_ego_pos()
            nearby_speed_kmh = nearby_v2x.get_ego_speed()
            if nearby_transform is None:
                continue
            nearby_location = nearby_transform.location
            nearby_rotation = nearby_transform.rotation
            nearby_actor = getattr(nearby_vehicle_manager, "vehicle", None)
            bounding_box = getattr(nearby_actor, "bounding_box", None)
            extent = getattr(bounding_box, "extent", None)
            length_m = 2.0 * float(extent.x) if extent is not None else None
            width_m = 2.0 * float(extent.y) if extent is not None else None
            height_m = 2.0 * float(extent.z) if extent is not None else None
            nearby_cavs.append({"id": str(cav_id), "vehicle_id": str(getattr(nearby_actor, "id", cav_id)), "type": str(getattr(nearby_actor, "type_id", "vehicle")), "source": "opencda_v2x", "x": float(nearby_location.x), "y": float(nearby_location.y), "z": float(nearby_location.z), "v": float(nearby_speed_kmh or 0.0) / 3.6, "psi": math.radians(float(nearby_rotation.yaw)), "length_m": length_m, "width_m": width_m, "height_m": height_m, "confidence": 1.0})
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
            print("[Data Transmitter] Could not convert V2X CAV {}: {}".format(cav_id, error))

    v2x_data = {
        "schema_version": int(raw_cp_payload.get("schema_version", 1) or 1),
        "timestamp_s": float(raw_cp_payload.get("timestamp_s", timestamp_s) or timestamp_s),
        "nearby_cavs": nearby_cavs,
        "cp_obstacles": cp_obstacles,
        "communication_range_m": float(getattr(v2x_manager, "communication_range", 0.0) or 0.0),
    }
    # Only lane events are sent through the cooperative topic for now.
    # Traffic-light information comes through /cpx/traffic_light.
    raw_cooperative_payload = (
        cooperative_payload
        if isinstance(cooperative_payload, dict)
        else raw_cp_payload
    )

    cooperative_data = {
        "schema_version": int(
            raw_cooperative_payload.get("schema_version", 1)
        ),
        "timestamp_s": float(
            raw_cooperative_payload.get(
                "timestamp_s",
                timestamp_s,
            )
        ),
        "lane_events": [
            dict(item)
            for item in list(
                raw_cooperative_payload.get("lane_events", raw_cooperative_payload.get("lane_closures", []))
            )
            if isinstance(item, dict)
        ],
    }

    safety_status = {}
    status_queue = getattr(safety_manager, "status_queue", None)
    if status_queue:
        try:
            _status_time, raw_safety_status = status_queue[-1]
            if isinstance(raw_safety_status, dict):
                safety_status = {str(key): bool(value) for key, value in raw_safety_status.items()}
        except (IndexError, TypeError, ValueError):
            safety_status = {}

    safety_data = {"status": safety_status}

    stream_data = {
        "localization": localization_data,
        "perception": perception_data,
        "traffic_light": traffic_light_data,
        "v2x": v2x_data,
        "cooperative": cooperative_data,
        "safety": safety_data,
    }

    results = {}

    # ------------------------------------------------------------------
    # Send each OpenCDA output to its corresponding ROS publisher node.
    # ------------------------------------------------------------------
    with _connection_lock:
        _sequence += 1
        current_sequence = int(_sequence)

        for stream_name, converted_data in stream_data.items():
            if stream_name == "cooperative":
                converted_data = dict(converted_data)
                converted_data["sequence"] = current_sequence
                
            message = {
                "schema_version": 1,
                "message_type": stream_name,
                "sequence": current_sequence,
                "timestamp_s": float(timestamp_s),
                "frame_id": "map",
                "cycle_send_started_wall_time_ns": int(cycle_send_started_wall_time_ns),
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
