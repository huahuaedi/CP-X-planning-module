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
        limit_for_mpc: Callable[..., Sequence[Mapping[str, Any]]],
        front_gap: Callable[..., Any],
        object_track_id: Callable[[Mapping[str, Any]], str],
    ) -> None:
        self._collect_local = collect_local
        self._fuse = fuse
        self._limit_for_mpc = limit_for_mpc
        self._front_gap = front_gap
        self._object_track_id = object_track_id

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
        mpc_objects = list(self._limit_for_mpc(
            object_snapshots=fused,
            ego_location=ego_location,
        ))
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
