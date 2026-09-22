"""Build one longitudinal proposal from the selected maneuver and perception."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .prediction import obstacle_track_id


@dataclass(frozen=True)
class SpeedPlanningRequest:
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    requested_speed_mps: float
    object_snapshots: Sequence[Mapping[str, object]]
    current_lane_id: int
    target_lane_id: int
    lane_assignments: Mapping[str, int]
    lane_change_progress: float
    lane_change_commitment_active: bool
    committed_source_lane_id: int
    scenario_decision: Any
    behavior_decision: str
    upcoming_turn_direction: str
    upcoming_turn_distance_m: float
    config: Mapping[str, object]


@dataclass(frozen=True)
class SpeedPlanningFrame:
    speed_plan: Any
    front_gap_m: Optional[float]
    front_actor_id: str
    front_obstacle_speed_mps: Optional[float]
    front_obstacle_lane_id: int
    front_obstacle_is_source_lane: bool

    @property
    def target_speed_mps(self) -> float:
        return float(self.speed_plan.target_speed_mps)

    @property
    def stop_goal_active(self) -> bool:
        return bool(self.speed_plan.stop_goal_active)


class SpeedPlanningStage:
    """Own obstacle attribution and the single SpeedPlanner proposal call."""

    def __init__(self, *, perception: Any, speed: Any) -> None:
        self._perception = perception
        self._speed = speed

    def run(self, request: SpeedPlanningRequest) -> SpeedPlanningFrame:
        decision = str(request.behavior_decision)
        front_gap_m, front_actor_id = self._perception.front_gap(
            ego_location=request.ego_location,
            ego_yaw_rad=float(request.ego_yaw_rad),
            object_snapshots=request.object_snapshots,
            current_lane_id=int(request.current_lane_id),
            lane_assignments=dict(request.lane_assignments),
            lane_change_direction=(
                "left" if decision == "lane_change_left"
                else "right" if decision == "lane_change_right"
                else ""
            ),
            lane_change_progress=float(request.lane_change_progress),
            return_actor_id=True,
        )
        front_actor_id = str(front_actor_id or "")
        front_speed_mps = None
        if front_actor_id:
            front = next(
                (
                    snapshot for snapshot in request.object_snapshots
                    if str(obstacle_track_id(snapshot)) == front_actor_id
                ),
                None,
            )
            if front is not None:
                front_speed_mps = max(
                    0.0,
                    float(front.get("v", front.get("speed_mps", 0.0)) or 0.0),
                )

        front_lane_id = int(
            request.lane_assignments.get(front_actor_id, 0) or 0
        ) if front_actor_id else 0
        source_lane_id = int(
            request.committed_source_lane_id
            if request.lane_change_commitment_active
            else request.current_lane_id
        )
        front_is_source_lane = bool(
            front_actor_id
            and front_lane_id != 0
            and front_lane_id == source_lane_id
            and (
                int(request.target_lane_id) != source_lane_id
                or request.lane_change_commitment_active
            )
        )
        turn_distance_m = (
            float(request.upcoming_turn_distance_m)
            if math.isfinite(float(request.upcoming_turn_distance_m))
            else None
        )
        speed_plan = self._speed.propose(
            scenario_decision=request.scenario_decision,
            behavior_decision=decision,
            requested_speed_mps=float(request.requested_speed_mps),
            ego_speed_mps=float(request.ego_speed_mps),
            config=dict(request.config),
            front_gap_m=front_gap_m,
            front_obstacle_speed_mps=front_speed_mps,
            upcoming_turn_direction=str(request.upcoming_turn_direction),
            upcoming_turn_distance_m=turn_distance_m,
            lane_change_commitment_active=bool(
                request.lane_change_commitment_active
            ),
            front_obstacle_is_source_lane=front_is_source_lane,
            additional_constraints=(),
        )
        return SpeedPlanningFrame(
            speed_plan=speed_plan,
            front_gap_m=(
                None if front_gap_m is None else float(front_gap_m)
            ),
            front_actor_id=front_actor_id,
            front_obstacle_speed_mps=front_speed_mps,
            front_obstacle_lane_id=front_lane_id,
            front_obstacle_is_source_lane=front_is_source_lane,
        )
