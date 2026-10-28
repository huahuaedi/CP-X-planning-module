"""ROS prediction transport adapter for the unchanged CP-X planner core.

The prediction node runs asynchronously (normally at 5 Hz), while planning
continues to consume OpenCDA frames at 20 Hz. This module caches the newest
typed ``PredictionFrame`` and implements the existing planner
``prediction_bridge.attach()`` port. It owns no risk or mode-selection policy.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import quote, unquote


_SESSION_MARKER = "|cpx_session="


def prediction_frame_id(session_id: str = "") -> str:
    """Return the map frame plus an internal prediction request owner."""
    session_id = str(session_id or "")
    if not session_id:
        return "map"
    return "map{}{}".format(_SESSION_MARKER, quote(session_id, safe=""))


def prediction_session_id(message: Any) -> str:
    """Read the session copied from PredictionScene to PredictionFrame."""
    explicit_session_id = str(getattr(message, "session_id", "") or "")
    if explicit_session_id:
        return explicit_session_id
    frame_id = str(getattr(message.header, "frame_id", "") or "")
    if _SESSION_MARKER not in frame_id:
        return ""
    return unquote(frame_id.split(_SESSION_MARKER, 1)[1])


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _track_id(snapshot: Mapping[str, Any]) -> str:
    for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id"):
        value = snapshot.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _canonical_track_id(snapshot: Mapping[str, Any]) -> str:
    """Return the planner identity shared by local and cooperative objects.

    Native OpenCDA cooperative messages namespace their transport ID as
    ``<provider>:<actor-id>``.  The planner observation contract deliberately
    removes that provider prefix before tracking/fusion.  PredictionScene must
    use the same canonical identity; otherwise MTR returns valid modes under
    the transport ID and ``attach()`` cannot bind them to the fused object.
    """
    track_id = _track_id(snapshot)
    if not track_id:
        return ""
    if str(snapshot.get("cp_message_id", "") or "").strip():
        return track_id.rsplit(":", 1)[-1]
    provider_source = str(snapshot.get("provider_source", "") or "").lower()
    if (
        provider_source.startswith("native_opencda_multi_vantage")
        or track_id.startswith("native_opencda_multi_vantage:")
    ):
        return track_id.rsplit(":", 1)[-1]
    return track_id


def _yaw_from_quaternion(orientation: Any) -> float:
    return math.atan2(
        2.0 * (
            float(orientation.w) * float(orientation.z)
            + float(orientation.x) * float(orientation.y)
        ),
        1.0 - 2.0 * (
            float(orientation.y) ** 2 + float(orientation.z) ** 2
        ),
    )


def _snapshot_state(snapshot: Mapping[str, Any]) -> Dict[str, float]:
    state = snapshot.get("state", ())
    values = list(state) if isinstance(state, Sequence) and not isinstance(
        state, (str, bytes, bytearray)
    ) else []
    return {
        "x": float(snapshot.get("x", snapshot.get("x_m", values[0] if values else 0.0))),
        "y": float(snapshot.get("y", snapshot.get("y_m", values[1] if len(values) > 1 else 0.0))),
        "z": float(snapshot.get("z", snapshot.get("z_m", 0.0))),
        "v": max(0.0, float(snapshot.get("v", snapshot.get("speed_mps", values[2] if len(values) > 2 else 0.0)))),
        "psi": float(snapshot.get("psi", snapshot.get("heading_rad", values[3] if len(values) > 3 else 0.0))),
    }


def _object_type_code(snapshot: Mapping[str, Any], agent_type: Any) -> int:
    name = str(snapshot.get("object_type", snapshot.get("type", "unknown"))).lower()
    if name in {"pedestrian", "walker"}:
        return int(agent_type.TYPE_PEDESTRIAN)
    if name in {"cyclist", "bicycle", "bike"}:
        return int(agent_type.TYPE_CYCLIST)
    if name in {"vehicle", "car", "truck", "bus", "motorcycle", "cav"}:
        return int(agent_type.TYPE_VEHICLE)
    return int(agent_type.TYPE_UNKNOWN)


class ROSPredictionBridge:
    """Cache ROS MTR frames and expose the planner's prediction bridge port."""

    model_name = "mtr_ros"

    def __init__(
        self,
        *,
        max_stale_s: float = 0.5,
        scene_type: Any = None,
        map_type: Any = None,
        agent_type: Any = None,
        map_polyline_type: Any = None,
        point_type: Any = None,
        expected_session_id: str = "",
    ) -> None:
        self._max_stale_s = max(0.0, float(max_stale_s))
        self._scene_type = scene_type
        self._map_type = map_type
        self._agent_type = agent_type
        self._map_polyline_type = map_polyline_type
        self._point_type = point_type
        self._expected_session_id = str(expected_session_id or "")
        self._lock = threading.RLock()
        self._frame_timestamp_s = -float("inf")
        self._revision = -1
        self._modes_by_track: Dict[str, List[Dict[str, Any]]] = {}
        self.received_frame_count = 0
        self.accepted_frame_count = 0
        self.rejected_frame_count = 0
        self.published_scene_count = 0
        self.last_attached_count = 0
        self.last_attached_mode_count = 0
        self.last_map_polyline_count = 0
        self.last_frame_object_count = 0
        self.last_frame_mode_count = 0
        self.last_frame_age_s = float("inf")
        self.last_latency_ms = 0.0
        self.last_scene_input_count = 0
        self.last_scene_object_count = 0
        self.last_scene_duplicate_count = 0
        self.last_scene_track_ids: List[str] = []
        self.last_error = ""

    def reset(self) -> None:
        """Retire predictions when an OpenCDA session/time epoch changes."""
        with self._lock:
            self._frame_timestamp_s = -float("inf")
            self._revision = -1
            self._modes_by_track = {}
            self.last_attached_count = 0
            self.last_attached_mode_count = 0
            self.last_frame_object_count = 0
            self.last_frame_mode_count = 0
            self.last_frame_age_s = float("inf")
            self.last_latency_ms = 0.0
            self.last_error = ""

    def record_latency_ms(self, latency_ms: float) -> None:
        """Record scene-to-frame ROS latency without changing prediction data."""

        with self._lock:
            self.last_latency_ms = max(0.0, float(latency_ms))

    def update_frame(self, frame: Any) -> bool:
        """Normalize one typed PredictionFrame into transport-free values."""
        self.received_frame_count += 1
        try:
            frame_id = str(getattr(frame.header, "frame_id", "") or "").strip()
            base_frame_id = frame_id.split(_SESSION_MARKER, 1)[0]
            frame_session_id = prediction_session_id(frame)
            if base_frame_id and base_frame_id != "map":
                raise ValueError("prediction_frame_not_in_map")
            if (
                self._expected_session_id
                and frame_session_id != self._expected_session_id
            ):
                raise ValueError("prediction_frame_session_mismatch")
            timestamp_s = _stamp_seconds(frame.header.stamp)
            revision = int(frame.revision)
            normalized: Dict[str, List[Dict[str, Any]]] = {}
            for prediction in list(frame.objects or ()):
                track_id = str(prediction.track_id or "").strip()
                if not track_id:
                    continue
                modes = []
                for trajectory in list(prediction.trajectories or ()):
                    points = self._trajectory_points(
                        trajectory, frame_timestamp_s=timestamp_s
                    )
                    probability = max(0.0, float(trajectory.confidence))
                    if points and math.isfinite(probability):
                        modes.append({
                            "trajectory": points,
                            "probability": probability,
                            "maneuver": str(trajectory.mode_id or "unknown"),
                            "position_sigma_m": 0.5,
                        })
                if modes:
                    normalized[track_id] = modes
        except Exception as exc:
            with self._lock:
                self.rejected_frame_count += 1
                self.last_error = "%s:%s" % (type(exc).__name__, str(exc))
            return False

        with self._lock:
            time_reversed = timestamp_s < self._frame_timestamp_s - 1.0e-6
            if time_reversed:
                self._revision = -1
                self._modes_by_track = {}
            if (
                not time_reversed
                and timestamp_s <= self._frame_timestamp_s + 1.0e-6
                and revision <= self._revision
            ):
                self.rejected_frame_count += 1
                self.last_error = "out_of_order_prediction_frame"
                return False
            self._frame_timestamp_s = float(timestamp_s)
            self._revision = int(revision)
            self._modes_by_track = normalized
            self.last_frame_object_count = len(normalized)
            self.last_frame_mode_count = sum(
                len(modes) for modes in normalized.values()
            )
            self.accepted_frame_count += 1
            self.last_error = ""
        return True

    @staticmethod
    def _trajectory_points(
        trajectory: Any, *, frame_timestamp_s: float
    ) -> List[Dict[str, float]]:
        compact_points = list(getattr(trajectory, "points", ()) or ())
        if compact_points:
            points = []
            for compact in compact_points:
                point = {
                    "x": float(compact.x),
                    "y": float(compact.y),
                    "t_abs": (
                        float(frame_timestamp_s)
                        + float(compact.time_from_start_s)
                    ),
                    "v": max(0.0, float(compact.speed_mps)),
                    "psi": float(compact.yaw),
                }
                if all(math.isfinite(value) for value in point.values()):
                    points.append(point)
            return sorted(
                points, key=lambda point: float(point["t_abs"])
            )

        points = []
        velocities = list(getattr(trajectory, "velocities", ()) or ())
        for index, waypoint in enumerate(
            list(getattr(trajectory, "waypoints", ()) or ())
        ):
            velocity = velocities[index] if index < len(velocities) else None
            if velocity is None:
                speed_mps = 0.0
            else:
                linear = velocity.twist.linear
                speed_mps = math.sqrt(
                    float(linear.x) ** 2 + float(linear.y) ** 2
                    + float(linear.z) ** 2
                )
            point = {
                "x": float(waypoint.pose.position.x),
                "y": float(waypoint.pose.position.y),
                "t_abs": float(_stamp_seconds(waypoint.header.stamp)),
                "v": max(0.0, float(speed_mps)),
                "psi": float(_yaw_from_quaternion(waypoint.pose.orientation)),
            }
            if all(math.isfinite(value) for value in point.values()):
                points.append(point)
        return sorted(points, key=lambda point: float(point["t_abs"]))

    def attach(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        horizon_s: float,
        dt_s: float,
        ego_snapshot: Mapping[str, Any],
        timestamp_s: float,
        map_polylines: Sequence[Mapping[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        """Attach all fresh modes; selection/risk remains owned by Planner."""
        del dt_s, ego_snapshot
        timestamp_s = float(timestamp_s)
        with self._lock:
            frame_timestamp_s = float(self._frame_timestamp_s)
            revision = int(self._revision)
            modes_by_track = {
                key: [dict(mode) for mode in modes]
                for key, modes in self._modes_by_track.items()
            }
        age_s = timestamp_s - frame_timestamp_s
        self.last_frame_age_s = float(age_s)
        usable = bool(
            math.isfinite(frame_timestamp_s)
            and -0.1 <= age_s <= self._max_stale_s + 1.0e-6
        )
        attached_count = 0
        attached_mode_count = 0
        annotated = []
        for snapshot in list(object_snapshots or ()):
            updated = dict(snapshot)
            raw_modes = modes_by_track.get(_track_id(snapshot), ()) if usable else ()
            modes = self._rebase_modes(
                raw_modes, timestamp_s=timestamp_s, horizon_s=float(horizon_s)
            )
            if modes:
                primary = max(modes, key=lambda item: float(item["probability"]))
                updated["predicted_trajectory"] = list(primary["trajectory"])
                updated["trajectory_hypotheses"] = list(modes)
                updated["predicted_modes"] = [
                    {"path": list(mode["trajectory"]), "probability": float(mode["probability"])}
                    for mode in modes
                ]
                updated["prediction_timestamp_s"] = frame_timestamp_s
                updated["plan_revision"] = "mtr_ros:%d" % revision
                updated["prediction_source"] = self.model_name
                attached_count += 1
                attached_mode_count += len(modes)
            annotated.append(updated)
        self.last_attached_count = int(attached_count)
        self.last_attached_mode_count = int(attached_mode_count)
        self.last_map_polyline_count = len(list(map_polylines or ()))
        return annotated

    @staticmethod
    def _rebase_modes(raw_modes, *, timestamp_s: float, horizon_s: float):
        modes = []
        for raw_mode in list(raw_modes or ()):
            points = []
            for raw_point in list(raw_mode.get("trajectory", ()) or ()):
                relative_t_s = float(raw_point["t_abs"]) - float(timestamp_s)
                if relative_t_s <= 1.0e-6 or relative_t_s > horizon_s + 1.0e-6:
                    continue
                point = dict(raw_point)
                point.pop("t_abs", None)
                point["t"] = relative_t_s
                points.append(point)
            if points:
                modes.append({
                    "trajectory": points,
                    "probability": max(0.0, float(raw_mode.get("probability", 0.0))),
                    "maneuver": str(raw_mode.get("maneuver", "unknown")),
                    "position_sigma_m": max(0.0, float(raw_mode.get("position_sigma_m", 0.5))),
                })
        total = sum(float(mode["probability"]) for mode in modes)
        if modes and total <= 1.0e-9:
            total = float(len(modes))
            for mode in modes:
                mode["probability"] = 1.0
        for mode in modes:
            mode["probability"] = float(mode["probability"]) / total
        return modes

    def build_scene(
        self,
        *,
        stamp: Any,
        ego_pose: Mapping[str, float],
        ego_speed_mps: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        cp_obstacles: Sequence[Mapping[str, Any]],
        map_polylines: Sequence[Mapping[str, Any]],
        ego_track_id: str = "ego",
    ) -> Any:
        """Build the typed scene consumed by the external Prediction node."""
        if any(item is None for item in (
            self._scene_type, self._agent_type,
            self._map_polyline_type, self._point_type,
        )):
            raise RuntimeError("Prediction ROS message types are not configured")
        scene = self._scene_type()
        scene.header.frame_id = prediction_frame_id(
            self._expected_session_id
        )
        scene.header.stamp = stamp
        if hasattr(scene, "session_id"):
            scene.session_id = self._expected_session_id
        scene.ego = self._agent_message({
            "track_id": str(ego_track_id), "type": "vehicle",
            "x": float(ego_pose["x"]), "y": float(ego_pose["y"]),
            "z": float(ego_pose.get("z", 0.0)), "v": float(ego_speed_mps),
            "psi": float(ego_pose["heading_rad"]),
            "length_m": 4.8, "width_m": 2.0, "height_m": 1.6,
        })
        raw_objects = list(object_snapshots or ()) + list(cp_obstacles or ())
        unique: Dict[str, Mapping[str, Any]] = {}
        for snapshot in raw_objects:
            if not isinstance(snapshot, Mapping):
                continue
            track_id = _track_id(snapshot)
            if not track_id:
                continue
            canonical = _canonical_track_id(snapshot)
            if canonical not in unique or "perception" in str(snapshot.get("source", "")):
                normalized = dict(snapshot)
                # ``track_id`` has first priority in _track_id().  Carry the
                # canonical planner identity explicitly into PredictionScene
                # while leaving the source payload untouched.
                normalized["track_id"] = canonical
                unique[canonical] = normalized
        scene.objects = [self._agent_message(item) for item in unique.values()]
        self.last_scene_input_count = len(raw_objects)
        self.last_scene_object_count = len(unique)
        self.last_scene_duplicate_count = max(0, len(raw_objects) - len(unique))
        self.last_scene_track_ids = sorted(unique.keys())
        for raw_polyline in list(map_polylines or ()):
            raw_points = list(raw_polyline.get("points", ()) or ())
            if len(raw_points) < 2:
                continue
            polyline = self._map_polyline_type()
            polyline.global_type = int(raw_polyline.get("global_type", 0))
            for raw_point in raw_points:
                point = self._point_type()
                point.x = float(raw_point.get("x", 0.0))
                point.y = float(raw_point.get("y", 0.0))
                point.z = float(raw_point.get("z", 0.0))
                polyline.points.append(point)
            scene.map_polylines.append(polyline)
        self.published_scene_count += 1
        self.last_map_polyline_count = len(scene.map_polylines)
        return scene

    def build_map(
        self,
        *,
        stamp: Any,
        revision: int,
        map_polylines: Sequence[Mapping[str, Any]],
    ) -> Any:
        """Build static map context for the transient-local ROS topic."""
        if any(item is None for item in (
            self._map_type, self._map_polyline_type, self._point_type,
        )):
            raise RuntimeError("Prediction ROS map message types are not configured")
        message = self._map_type()
        message.header.frame_id = prediction_frame_id(
            self._expected_session_id
        )
        message.header.stamp = stamp
        message.revision = max(0, int(revision))
        for raw_polyline in list(map_polylines or ()):
            raw_points = list(raw_polyline.get("points", ()) or ())
            if len(raw_points) < 2:
                continue
            polyline = self._map_polyline_type()
            polyline.global_type = int(raw_polyline.get("global_type", 0))
            for raw_point in raw_points:
                point = self._point_type()
                point.x = float(raw_point.get("x", 0.0))
                point.y = float(raw_point.get("y", 0.0))
                point.z = float(raw_point.get("z", 0.0))
                polyline.points.append(point)
            message.map_polylines.append(polyline)
        self.last_map_polyline_count = len(message.map_polylines)
        return message

    def _agent_message(self, snapshot: Mapping[str, Any]) -> Any:
        agent = self._agent_type()
        state = _snapshot_state(snapshot)
        agent.track_id = _track_id(snapshot)
        agent.object_type = _object_type_code(snapshot, self._agent_type)
        agent.pose.position.x = state["x"]
        agent.pose.position.y = state["y"]
        agent.pose.position.z = state["z"]
        agent.pose.orientation.z = math.sin(0.5 * state["psi"])
        agent.pose.orientation.w = math.cos(0.5 * state["psi"])
        agent.twist.linear.x = state["v"] * math.cos(state["psi"])
        agent.twist.linear.y = state["v"] * math.sin(state["psi"])
        shape = snapshot.get("shape", {})
        shape = dict(shape) if isinstance(shape, Mapping) else {}
        agent.size.x = float(snapshot.get("length_m", shape.get("length_m", 4.5)))
        agent.size.y = float(snapshot.get("width_m", shape.get("width_m", 2.0)))
        agent.size.z = float(snapshot.get("height_m", shape.get("height_m", 1.6)))
        return agent

    @property
    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            timestamp_s = float(self._frame_timestamp_s)
            revision = int(self._revision)
            cached_object_count = len(self._modes_by_track)
            cached_mode_count = sum(
                len(modes) for modes in self._modes_by_track.values()
            )
            last_error = str(self.last_error)
        return {
            "prediction_bridge_model": self.model_name,
            "prediction_bridge_request_count": int(self.published_scene_count),
            "prediction_bridge_inference_request_count": 0,
            "prediction_bridge_success_count": int(self.accepted_frame_count),
            "prediction_bridge_dropped_request_count": int(self.rejected_frame_count),
            "prediction_bridge_queue_depth": 0,
            "prediction_bridge_attached_count": int(self.last_attached_count),
            "prediction_bridge_attached_mode_count": int(self.last_attached_mode_count),
            "prediction_bridge_latency_ms": float(self.last_latency_ms),
            "prediction_bridge_map_polyline_count": int(self.last_map_polyline_count),
            "prediction_bridge_error": last_error,
            "prediction_bridge_shared": False,
            "prediction_bridge_shared_consumer_count": 1,
            "prediction_ros_frame_timestamp_s": (
                timestamp_s if math.isfinite(timestamp_s) else None
            ),
            "prediction_ros_revision": revision,
            "prediction_ros_cached_object_count": int(cached_object_count),
            "prediction_ros_cached_mode_count": int(cached_mode_count),
            "prediction_ros_frame_object_count": int(self.last_frame_object_count),
            "prediction_ros_frame_mode_count": int(self.last_frame_mode_count),
            "prediction_ros_frame_age_s": (
                float(self.last_frame_age_s)
                if math.isfinite(self.last_frame_age_s) else None
            ),
            "prediction_ros_scene_input_count": int(self.last_scene_input_count),
            "prediction_ros_scene_object_count": int(self.last_scene_object_count),
            "prediction_ros_scene_duplicate_count": int(
                self.last_scene_duplicate_count
            ),
            "prediction_ros_scene_track_ids": ",".join(
                self.last_scene_track_ids
            ),
        }
