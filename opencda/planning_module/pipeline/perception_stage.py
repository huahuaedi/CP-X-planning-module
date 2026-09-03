"""Build the immutable obstacle view consumed by one planning tick."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
        object_track_id: Callable[[Mapping[str, Any]], str],
        max_mpc_obstacles: int = 0,
        ego_length_m: float = 4.5,
        ego_width_m: float = 2.0,
        lane_width_m: float = 3.5,
        lane_change_boundary_overlap_m: float = 0.75,
    ) -> None:
        self._collect_local = collect_local
        self._object_track_id = object_track_id
        self._max_mpc_obstacles = max(0, int(max_mpc_obstacles))
        self._ego_half_length_m = 0.5 * max(0.0, float(ego_length_m))
        self._ego_width_m = max(0.5, float(ego_width_m))
        self._lane_width_m = max(1.0, float(lane_width_m))
        self._boundary_overlap_m = max(
            0.0, float(lane_change_boundary_overlap_m)
        )

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
        fused = self.fuse(
            local_objects=local,
            cp_obstacles=list(cp_payload.get("obstacles", ()) or ()),
            timestamp_s=float(timestamp_s),
        )
        if bool(ignore_dynamic_objects):
            fused = []
        mpc_objects = self.limit_for_mpc(fused, ego_location=ego_location)
        gap_m, actor_id = self.front_gap(
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

    def front_gap(
        self, *, ego_location, ego_yaw_rad, object_snapshots,
        lane_change_direction="", lane_change_progress=0.0,
        current_lane_id=None, lane_assignments=None, return_actor_id=False,
    ):
        cosine = math.cos(float(ego_yaw_rad))
        sine = math.sin(float(ego_yaw_rad))

        def nearest(min_lateral_m, max_lateral_m):
            best_gap, best_actor = None, None
            for obstacle in object_snapshots:
                actor_id = str(self._object_track_id(obstacle))
                if current_lane_id is not None and lane_assignments is not None:
                    assigned = int(lane_assignments.get(actor_id, 0) or 0)
                    if assigned != int(current_lane_id):
                        continue
                dx = float(obstacle.get("x", 0.0)) - float(ego_location.x)
                dy = float(obstacle.get("y", 0.0)) - float(ego_location.y)
                longitudinal = dx * cosine + dy * sine
                lateral = -dx * sine + dy * cosine
                if (
                    longitudinal <= 0.0
                    or lateral < float(min_lateral_m)
                    or lateral > float(max_lateral_m)
                ):
                    continue
                clearance = max(
                    0.0,
                    longitudinal - self._ego_half_length_m
                    - 0.5 * max(0.0, float(obstacle.get("length_m", 4.5) or 4.5)),
                )
                if best_gap is None or clearance < best_gap:
                    best_gap, best_actor = float(clearance), actor_id
            return best_gap, best_actor

        strict_lane = current_lane_id is not None and lane_assignments is not None
        direction = "" if strict_lane else str(lane_change_direction or "").lower()
        if direction not in {"left", "right"}:
            gap, actor = nearest(-2.5, 2.5)
        else:
            overlap = self._boundary_overlap_m
            if direction == "left":
                source_gap, source_actor = nearest(-2.5, overlap)
                target_gap, target_actor = nearest(-overlap, 2.5)
            else:
                source_gap, source_actor = nearest(-overlap, 2.5)
                target_gap, target_actor = nearest(-2.5, overlap)
            alpha_clear = min(
                0.95,
                0.5 + self._ego_width_m / (2.0 * self._lane_width_m),
            )
            alpha = max(0.0, min(1.0, float(lane_change_progress)))
            ramp = 0.0 if alpha <= alpha_clear else min(
                1.0, (alpha - alpha_clear) / max(1.0e-6, 1.0 - alpha_clear)
            )
            weight = ramp * ramp * (3.0 - 2.0 * ramp)
            unconstrained = 1.0e6
            source_value = unconstrained if source_gap is None else source_gap
            target_value = unconstrained if target_gap is None else target_gap
            blended = (1.0 - weight) * source_value + weight * target_value
            gap = None if blended >= 0.5 * unconstrained else float(blended)
            actor = target_actor if weight >= 0.5 else source_actor
        if return_actor_id:
            return gap, None if gap is None else str(actor)
        return gap

    def fuse(self, *, local_objects, cp_obstacles, timestamp_s):
        fused = {}
        priorities = {}
        for raw in list(local_objects or ()):
            normalized = self.normalize_local(raw)
            if normalized is not None:
                self._upsert(fused, priorities, normalized)
        for raw in list(cp_obstacles or ()):
            if not isinstance(raw, Mapping) or not self.message_is_fresh(
                raw, timestamp_s=float(timestamp_s)
            ):
                continue
            normalized = self.normalize_cp(raw)
            if normalized is None or self._duplicates_native_perception(
                normalized, fused.values()
            ):
                continue
            self._upsert(fused, priorities, normalized)
        return list(fused.values())

    @staticmethod
    def message_is_fresh(message, *, timestamp_s):
        try:
            valid_until_s = float(message.get("valid_until_s", "nan"))
            if math.isfinite(valid_until_s):
                return float(timestamp_s) <= valid_until_s
        except Exception:
            pass
        try:
            source_time_s = float(message.get("timestamp_s", timestamp_s))
            ttl_s = float(message.get("ttl_s", 0.0))
        except Exception:
            return True
        return bool(
            ttl_s <= 0.0 or float(timestamp_s) <= source_time_s + ttl_s
        )

    @staticmethod
    def _duplicates_native_perception(candidate, existing, max_delta_m=1.0):
        provider = str(candidate.get("provider_source", "")).lower()
        source = str(candidate.get("source", "")).lower()
        if "perception" not in provider and "perception" not in source:
            return False
        try:
            candidate_x = float(candidate.get("x", 0.0))
            candidate_y = float(candidate.get("y", 0.0))
        except Exception:
            return False
        for item in existing:
            item_provider = str(item.get("provider_source", "")).lower()
            item_source = str(item.get("source", "")).lower()
            if "perception" not in item_provider and "perception" not in item_source:
                continue
            try:
                distance_m = math.hypot(
                    candidate_x - float(item.get("x", 0.0)),
                    candidate_y - float(item.get("y", 0.0)),
                )
            except Exception:
                continue
            if distance_m <= float(max_delta_m):
                return True
        return False

    @staticmethod
    def _source_priority(snapshot):
        source = (
            str(snapshot.get("provider_source", ""))
            + " " + str(snapshot.get("source", ""))
        ).lower()
        if "perception" in source:
            return 100
        if "v2x" in source:
            return 80
        if "fallback" in source or "carla" in source:
            return 40
        return 60

    @classmethod
    def _upsert(cls, fused, priorities, snapshot):
        raw_id = str(
            snapshot.get("vehicle_id", snapshot.get("id", ""))
        ).strip()
        key = raw_id.rsplit(":", 1)[-1]
        if not key:
            return
        priority = cls._source_priority(snapshot)
        previous_priority = int(priorities.get(key, -1))
        previous = fused.get(key)
        previous_confidence = (
            float(previous.get("confidence", 0.0))
            if isinstance(previous, Mapping) else -1.0
        )
        confidence = float(snapshot.get("confidence", 0.0))
        if priority > previous_priority or (
            priority == previous_priority and confidence >= previous_confidence
        ):
            fused[key] = dict(snapshot)
            priorities[key] = int(priority)

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
