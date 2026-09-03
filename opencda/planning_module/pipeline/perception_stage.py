"""Build the immutable obstacle view consumed by one planning tick."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PerceptionStageResult:
    local_objects: Tuple[Mapping[str, Any], ...]
    fused_objects: Tuple[Mapping[str, Any], ...]
    mpc_objects: Tuple[Mapping[str, Any], ...]
    cp_payload: Mapping[str, Any]
    front_gap_m: Optional[float]
    front_actor_id: str
    front_actor_speed_mps: Optional[float]


class PerceptionStage:
    """Own obstacle fusion and the planner-facing front-object summary.

    The callable ports isolate currently legacy data implementations without
    giving this stage access to the bridge or its mutable state.
    """

    def __init__(
        self,
        *,
        collect_local: Callable[..., Sequence[Mapping[str, Any]]],
        fuse: Callable[..., Sequence[Mapping[str, Any]]],
        front_gap: Callable[..., Any],
        object_track_id: Callable[[Mapping[str, Any]], str],
        max_mpc_obstacles: int,
    ) -> None:
        self._collect_local = collect_local
        self._fuse = fuse
        self._front_gap = front_gap
        self._object_track_id = object_track_id
        self._max_mpc_obstacles = max(0, int(max_mpc_obstacles))

    def build(
        self,
        *,
        detected_objects: Any,
        cp_payload: Mapping[str, Any],
        ego_location: Any,
        ego_yaw_rad: float,
        timestamp_s: float,
        ignore_dynamic_objects: bool,
    ) -> PerceptionStageResult:
        local = list(self._collect_local(detected_objects=detected_objects))
        fused = list(self._fuse(
            local_object_snapshots=local,
            cp_obstacles=list(cp_payload.get("obstacles", ()) or ()),
            ego_location=ego_location,
            sim_time_s=float(timestamp_s),
        ))
        if bool(ignore_dynamic_objects):
            fused = []
        mpc_objects = self.limit_for_mpc(fused, ego_location=ego_location)
        gap_m, actor_id = self._front_gap(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            object_snapshots=fused,
            return_actor_id=True,
        )
        actor_id = str(actor_id or "")
        actor_speed_mps = None
        if actor_id:
            front = next(
                (
                    obstacle for obstacle in fused
                    if str(self._object_track_id(obstacle)) == actor_id
                ),
                None,
            )
            if front is not None:
                actor_speed_mps = max(
                    0.0,
                    float(front.get("v", front.get("speed_mps", 0.0)) or 0.0),
                )
        return PerceptionStageResult(
            local_objects=tuple(dict(item) for item in local),
            fused_objects=tuple(dict(item) for item in fused),
            mpc_objects=tuple(dict(item) for item in mpc_objects),
            cp_payload=dict(cp_payload),
            front_gap_m=None if gap_m is None else float(gap_m),
            front_actor_id=actor_id,
            front_actor_speed_mps=actor_speed_mps,
        )

    def limit_for_mpc(self, objects, *, ego_location):
        result = [
            dict(item) for item in list(objects or ())
            if isinstance(item, Mapping)
        ]
        if self._max_mpc_obstacles and len(result) > self._max_mpc_obstacles:
            result.sort(key=lambda item: (
                float(item.get("x", 0.0)) - float(ego_location.x)
            ) ** 2 + (
                float(item.get("y", 0.0)) - float(ego_location.y)
            ) ** 2)
            result = result[:self._max_mpc_obstacles]
        return result

    @staticmethod
    def normalize_local(snapshot):
        try:
            obstacle_id = str(
                snapshot.get("vehicle_id", snapshot.get("id", ""))
            ).strip()
            if not obstacle_id:
                return None
            return {
                "vehicle_id": obstacle_id, "id": obstacle_id,
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "v": float(snapshot.get("v", 0.0)),
                "psi": float(snapshot.get("psi", 0.0)),
                "length_m": float(snapshot.get("length_m", 4.5)),
                "width_m": float(snapshot.get("width_m", 2.0)),
                "source": str(snapshot.get("source", "opencda_perception")),
                "provider_source": str(snapshot.get(
                    "provider_source", "native_opencda_perception"
                )),
                "confidence": float(snapshot.get("confidence", 1.0)),
            }
        except Exception:
            return None

    @staticmethod
    def normalize_cp(obstacle):
        try:
            raw_id = str(
                obstacle.get("id", obstacle.get("vehicle_id", ""))
            ).strip()
            if not raw_id:
                return None
            state = obstacle.get("state", ())
            state_values = list(state) if (
                isinstance(state, Sequence)
                and not isinstance(state, (str, bytes, bytearray))
            ) else []
            x_m = obstacle.get(
                "x", obstacle.get("x_m", state_values[0] if state_values else None)
            )
            y_m = obstacle.get(
                "y", obstacle.get("y_m", state_values[1] if len(state_values) >= 2 else None)
            )
            if x_m is None or y_m is None:
                return None
            speed_mps = obstacle.get(
                "v", obstacle.get("speed_mps", state_values[2] if len(state_values) >= 3 else 0.0)
            )
            heading_rad = obstacle.get(
                "psi", obstacle.get("heading_rad", state_values[3] if len(state_values) >= 4 else 0.0)
            )
            shape = obstacle.get("shape", {})
            shape = dict(shape) if isinstance(shape, Mapping) else {}
            obstacle_id = raw_id.rsplit(":", 1)[-1]
            return {
                "vehicle_id": obstacle_id, "id": obstacle_id,
                "cp_message_id": raw_id,
                "x": float(x_m), "y": float(y_m),
                "v": float(speed_mps), "psi": float(heading_rad),
                "length_m": float(shape.get("length_m", obstacle.get("length_m", 4.5))),
                "width_m": float(shape.get("width_m", obstacle.get("width_m", 2.0))),
                "source": str(obstacle.get("source", "opencda_cp")),
                "provider_source": str(obstacle.get("provider_source", "opencda_cp")),
                "confidence": float(obstacle.get("confidence", 0.5)),
                "lane_id": int(float(obstacle.get("lane_id", 0) or 0)),
                "road_id": int(float(obstacle.get("road_id", 0) or 0)),
                "object_type": str(obstacle.get("type", "unknown")),
                "observed_by_cav_ids": list(obstacle.get("observed_by_cav_ids", ()) or ()),
                "not_observed_by_cav_ids": list(obstacle.get("not_observed_by_cav_ids", ()) or ()),
                "blind_spot_shared": bool(obstacle.get("blind_spot_shared", False)),
            }
        except Exception:
            return None
