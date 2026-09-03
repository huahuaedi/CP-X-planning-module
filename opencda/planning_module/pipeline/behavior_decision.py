"""Typed, immutable output contract of behavior decision making."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class BehaviorConstraint:
    kind: str
    reason: str = ""
    value: object = None


@dataclass(frozen=True)
class BehaviorDecision:
    maneuver: str
    phase: str
    source_lane_id: int
    target_lane_id: int
    target_corridor_id: int
    direction: str
    speed_intent: str
    requested_speed_mps: float
    stop_required: bool
    route_required: bool
    traffic_signal_state: str = "unknown"
    boundary_recovery_active: bool = False
    stop_target: Optional[Mapping[str, object]] = None
    constraints: Tuple[BehaviorConstraint, ...] = ()
    reason: str = ""

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        default_speed_mps: float,
    ) -> "BehaviorDecision":
        row = dict(values or {})
        maneuver = str(row.get("decision", "lane_follow") or "lane_follow")
        direction = (
            "left" if maneuver.endswith("_left")
            else "right" if maneuver.endswith("_right")
            else ""
        )
        raw_stop_target = row.get("stop_target")
        stop_target = (
            MappingProxyType(dict(raw_stop_target))
            if isinstance(raw_stop_target, Mapping)
            else None
        )
        raw_constraints = list(row.get("constraints", ()) or ())
        constraints = tuple(
            item if isinstance(item, BehaviorConstraint)
            else BehaviorConstraint(
                kind=str(dict(item).get("kind", "constraint")),
                reason=str(dict(item).get("reason", "")),
                value=dict(item).get("value"),
            )
            for item in raw_constraints
            if isinstance(item, (BehaviorConstraint, Mapping))
        )
        requested_speed_mps = max(
            0.0,
            float(row.get("target_speed_mps", default_speed_mps) or 0.0),
        )
        stop_required = bool(
            row.get("stop_goal_active", False)
            or maneuver in {
                "destination_stop",
                "stop_at_intersection",
                "stop_sign",
                "emergency_brake",
            }
        )
        return cls(
            maneuver=maneuver,
            phase=str(row.get("lc_state", "LANE_KEEP") or "LANE_KEEP"),
            source_lane_id=int(row.get("current_lane_id", 0) or 0),
            target_lane_id=int(row.get("target_lane_id", 0) or 0),
            target_corridor_id=int(
                row.get("target_corridor_id", row.get("target_lane_id", 0)) or 0
            ),
            direction=direction,
            speed_intent=str(row.get("speed_intent", "track_target")),
            requested_speed_mps=requested_speed_mps,
            stop_required=stop_required,
            route_required=bool(row.get("route_required", False)),
            traffic_signal_state=str(
                row.get("traffic_signal_state", "unknown") or "unknown"
            ),
            boundary_recovery_active=bool(
                row.get("boundary_recovery_active", False)
            ),
            stop_target=stop_target,
            constraints=constraints,
            reason=str(row.get("behavior_override_reason", row.get("reason", ""))),
        )

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "decision": str(self.maneuver),
            "lc_state": str(self.phase),
            "current_lane_id": int(self.source_lane_id),
            "target_lane_id": int(self.target_lane_id),
            "target_corridor_id": int(self.target_corridor_id),
            "behavior_direction": str(self.direction),
            "behavior_speed_intent": str(self.speed_intent),
            "target_speed_mps": float(self.requested_speed_mps),
            "stop_goal_active": bool(self.stop_required),
            "behavior_route_required": bool(self.route_required),
            "traffic_signal_state": str(self.traffic_signal_state),
            "boundary_recovery_active": bool(self.boundary_recovery_active),
            "behavior_reason": str(self.reason),
        }
