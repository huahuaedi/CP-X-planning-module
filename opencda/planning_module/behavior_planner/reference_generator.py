"""Reference intent selection between behavior planning and MPC.

This module does not build CARLA waypoints directly.  It defines the contract
for what kind of local reference the MPC should track after the behavior
planner has selected a maneuver and target lane.
"""

from __future__ import annotations

from dataclasses import dataclass

from .planner import is_emergency_brake_decision, is_fixed_stop_decision, normalize_behavior_decision


ReferenceMode = str


@dataclass(frozen=True)
class ReferenceIntent:
    """Semantic contract consumed by the local reference generator and MPC."""

    mode: ReferenceMode
    target_lane_id: int
    follow_global_route_lane: bool
    reason: str


def select_reference_intent(
    *,
    behavior_decision: str,
    planner_fsm_state: str,
    ego_in_junction: bool,
    reference_target_lane_id: int,
    current_lane_id: int,
    route_optimal_lane_id: int,
    global_route_reference_allowed: bool,
    traffic_control_lane_lock_active: bool,
) -> ReferenceIntent:
    """Choose the reference semantics for the current planning tick.

    Layering policy:
    - Stop/follow-lead decisions control longitudinal behavior, but lateral
      reference remains lane based unless a final stop target is explicitly
      passed to the lower reference builder.
    - Lane-change decisions track a committed transition trajectory.
    - Junction lane-follow may track the global route branch because there may
      be no stable lane-center continuation through the connector.
    - Ordinary road lane-follow tracks the selected/current lane centerline.
    """

    normalized_decision = str(normalize_behavior_decision(behavior_decision))
    normalized_fsm = str(planner_fsm_state or "").strip().upper()
    target_lane_id = int(reference_target_lane_id or current_lane_id or route_optimal_lane_id or 0)

    if bool(is_fixed_stop_decision(normalized_decision)):
        return ReferenceIntent(
            mode="stop",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=False,
            reason=f"behavior_{normalized_decision}",
        )

    if bool(is_emergency_brake_decision(normalized_decision)):
        return ReferenceIntent(
            mode="follow_lead",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=False,
            reason="emergency_brake",
        )

    if normalized_decision in {"lane_change_left", "lane_change_right"} or normalized_fsm in {
        "PREPARE_LANE_CHANGE_LEFT",
        "PREPARE_LANE_CHANGE_RIGHT",
        "EXECUTE_LANE_CHANGE_LEFT",
        "EXECUTE_LANE_CHANGE_RIGHT",
    }:
        return ReferenceIntent(
            mode="lane_change",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=False,
            reason=f"decision_or_fsm_{normalized_decision}:{normalized_fsm}",
        )

    # The global route is a mission-level path, not a directly trackable MPC
    # horizon.  Its first sample can legitimately be behind the ego or on a
    # different junction connector after projection, which previously caused
    # temp_des/reference jumps and heading fallbacks.  Keep lane-follow local;
    # the runner still uses route context to choose maneuvers and lanes.
    if bool(global_route_reference_allowed) and int(route_optimal_lane_id) != 0:
        return ReferenceIntent(
            mode="lane_follow",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=False,
            reason="route_hint_local_lane_reference",
        )

    return ReferenceIntent(
        mode="lane_follow",
        target_lane_id=int(target_lane_id),
        follow_global_route_lane=False,
        reason="lane_center_follow",
    )
