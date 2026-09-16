"""Keep a VRU yield-stop commitment stable while the same hazard is tracked.

``assess_front_observation`` recomputes its dynamic stopping envelope from
the *current* ego speed every tick.  As the ego brakes in response to a
YIELD_STOP, that envelope shrinks, which can flip the action back to NONE
before the vehicle has actually stopped or the pedestrian has cleared --
releasing a commitment the ego is still in the middle of honoring.  This
latch holds YIELD_STOP for the same tracked obstacle id until it is no
longer reported as a VRU-typed front observation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .behavior_risk import SemanticBehaviorResponse
from .conflict_classifier import VRU_CONFLICT

_VRU_TYPES = {"pedestrian", "cyclist", "vru"}


@dataclass
class VRUYieldLatch:
    """Sole owner of VRU yield-stop commitment persistence."""

    _active_obstacle_id: str = ""

    def reset(self) -> None:
        self._active_obstacle_id = ""

    def update(
        self, response: SemanticBehaviorResponse
    ) -> SemanticBehaviorResponse:
        if response.risk_kind == VRU_CONFLICT and response.action == "YIELD_STOP":
            self._active_obstacle_id = str(response.obstacle_id)
            return response
        if (
            self._active_obstacle_id
            and response.object_type in _VRU_TYPES
            and str(response.obstacle_id) == self._active_obstacle_id
        ):
            return replace(
                response, action="YIELD_STOP",
                reason="vru_yield_commitment_latched",
            )
        self._active_obstacle_id = ""
        return response
