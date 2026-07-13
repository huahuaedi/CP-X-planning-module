"""Route wrapper objects exposed by the planner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .waypoint import Waypoint


@dataclass
class Route:
    """Store one planned route and the data most callers need to inspect."""

    raw_start: Dict[str, float]
    raw_goal: Dict[str, float]
    resolved_start: "Waypoint"
    resolved_goal: "Waypoint"
    lane_path: List[int]
    sampled_waypoints: List["Waypoint"]
    length_m: float
    sampling_resolution_m: float
    transition_types: List[str] = field(default_factory=list)

    def to_point_dicts(self) -> List[Dict[str, float]]:
        """Return the sampled route as CARLA-style points.

        input: none (`None`)
        output: sampled route points (`List[Dict[str, float]]`)
        """
        return [waypoint.position for waypoint in self.sampled_waypoints]
