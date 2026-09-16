"""Pure observation-to-behavior risk policy.

This module owns no lifecycle state.  It translates one canonical front
observation into a typed intent; :class:`BehaviorStage` remains the only owner
that may turn that intent into a maneuver or stop command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .conflict_classifier import LANE_BLOCKAGE, NO_RISK, VRU_CONFLICT
from .observation_contract import normalize_object_type


@dataclass(frozen=True)
class SemanticBehaviorResponse:
    risk_kind: str = NO_RISK
    action: str = "NONE"
    object_type: str = "unknown"
    observation_source: str = "unknown"
    obstacle_id: str = ""
    distance_m: float = float("inf")
    reason: str = "no_relevant_front_observation"


def _observation_source(observation: Mapping[str, object]) -> str:
    local = bool(observation.get(
        "locally_observed", observation.get("observed_locally", False)
    ))
    cp = bool(observation.get(
        "cooperatively_observed", observation.get("observed_via_cp", False)
    ))
    if local and cp:
        return "local+cp"
    if cp:
        return "cp"
    if local:
        return "local"
    return str(observation.get("observation_source", "unknown") or "unknown")


def assess_front_observation(
    *, front_obstacle: Optional[Mapping[str, object]],
    ego_speed_mps: float, max_deceleration_mps2: float,
    route_lane_safety_score: float,
    config: Mapping[str, object], runtime_config: Mapping[str, object],
    object_track_id: Any,
) -> SemanticBehaviorResponse:
    """Return the semantic action for one canonical front observation."""

    if not isinstance(front_obstacle, Mapping):
        return SemanticBehaviorResponse()
    obstacle = dict(front_obstacle)
    object_type = normalize_object_type(
        obstacle.get("object_type", obstacle.get("actor_type", "unknown"))
    )
    source = _observation_source(obstacle)
    obstacle_id = str(object_track_id(obstacle))
    distance_m = max(0.0, float(
        obstacle.get("front_distance_m", float("inf"))
    ))
    speed_mps = max(0.0, float(
        obstacle.get("v", obstacle.get("speed_mps", 0.0))
    ))

    if object_type in {"pedestrian", "cyclist", "vru"}:
        deceleration = max(0.1, abs(float(max_deceleration_mps2)))
        reaction_s = max(0.0, float(config.get(
            "vru_yield_reaction_time_s", 1.0
        )))
        buffer_m = max(0.0, float(config.get("vru_yield_buffer_m", 4.0)))
        braking_reach_m = (
            float(ego_speed_mps) * reaction_s
            + float(ego_speed_mps) ** 2 / (2.0 * deceleration)
            + buffer_m
        )
        lookahead_m = max(
            braking_reach_m,
            float(config.get("vru_yield_min_lookahead_m", 15.0)),
        )
        inside_reach = distance_m <= lookahead_m
        return SemanticBehaviorResponse(
            risk_kind=VRU_CONFLICT,
            action="YIELD_STOP" if inside_reach else "NONE",
            object_type=object_type,
            observation_source=source,
            obstacle_id=obstacle_id,
            distance_m=distance_m,
            reason=(
                "vru_within_dynamic_stopping_reach"
                if inside_reach else "vru_outside_dynamic_stopping_reach"
            ),
        )

    stationary_threshold = float(config.get(
        "static_obstacle_speed_threshold_mps",
        runtime_config.get(
            "static_obstacle_speed_threshold_mps",
            runtime_config.get(
                "intersection_obstacle_moving_speed_threshold_mps", 0.5
            ),
        ),
    ))
    safety_threshold = float(config.get(
        "static_obstacle_replan_lane_safety_threshold",
        runtime_config.get(
            "static_obstacle_replan_lane_safety_threshold",
            runtime_config.get(
                "intersection_static_obstacle_replan_lane_safety_threshold",
                0.5,
            ),
        ),
    ))
    explicit_static_object = object_type == "static_object"
    static_lookahead_m = max(0.0, float(config.get(
        "static_obstacle_behavior_lookahead_m", 100.0
    )))
    stationary_vehicle_blockage = bool(
        object_type == "vehicle"
        and speed_mps <= stationary_threshold
        and float(route_lane_safety_score) < safety_threshold
    )
    blocks_lane = bool(
        (explicit_static_object and distance_m <= static_lookahead_m)
        or stationary_vehicle_blockage
    )
    if blocks_lane:
        return SemanticBehaviorResponse(
            risk_kind=LANE_BLOCKAGE,
            action="LANE_BLOCKAGE",
            object_type=object_type,
            observation_source=source,
            obstacle_id=obstacle_id,
            distance_m=distance_m,
            reason="typed_lane_blockage_on_current_corridor",
        )
    return SemanticBehaviorResponse(
        object_type=object_type,
        observation_source=source,
        obstacle_id=obstacle_id,
        distance_m=distance_m,
        reason="front_observation_requires_no_behavior_override",
    )
