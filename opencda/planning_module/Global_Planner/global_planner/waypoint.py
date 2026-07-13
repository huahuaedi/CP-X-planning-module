"""Waypoint wrapper objects exposed by the planner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .planner import GlobalPlanner


@dataclass
class Waypoint:
    """Represent one planner waypoint with CARLA-format and ENU coordinates."""

    position: Dict[str, float]
    enu_position: Tuple[float, float, float]
    ad_lane_id: int
    road_id: Optional[int]
    section_id: Optional[int]
    lane_id: Optional[int]
    parametric_offset: float
    heading: Optional[float]
    lane_length_m: Optional[float]
    lane_width_m: Optional[float]
    is_intersection: bool
    _planner: "GlobalPlanner" = field(repr=False, compare=False)

    @property
    def lane_width(self) -> Optional[float]:
        """Return lane width using the CARLA-style attribute name.

        input: none (`None`)
        output: lane width in meters (`float | None`)
        """
        return self.lane_width_m

    def left(self) -> Optional["Waypoint"]:
        """Return the adjacent same-direction left-lane waypoint if it exists.

        input: none (`None`)
        output: adjacent left waypoint or no result (`Waypoint | None`)
        """
        return self._planner._get_adjacent_waypoint(self, side="left")

    def right(self) -> Optional["Waypoint"]:
        """Return the adjacent same-direction right-lane waypoint if it exists.

        input: none (`None`)
        output: adjacent right waypoint or no result (`Waypoint | None`)
        """
        return self._planner._get_adjacent_waypoint(self, side="right")

    def next(self, distance_m: float) -> List["Waypoint"]:
        """Advance forward along the lane network by the requested distance.

        input: `distance_m` (`float`)
        output: reachable forward waypoints (`List[Waypoint]`)
        """
        return self._planner._step_waypoint(self, distance_m=distance_m, forward=True)

    def previous(self, distance_m: float) -> List["Waypoint"]:
        """Move backward along the lane network by the requested distance.

        input: `distance_m` (`float`)
        output: reachable backward waypoints (`List[Waypoint]`)
        """
        return self._planner._step_waypoint(self, distance_m=distance_m, forward=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return one JSON-friendly waypoint summary.

        input: none (`None`)
        output: serialized waypoint data (`Dict[str, object]`)
        """
        return {
            "position": dict(self.position),
            "enu_position": tuple(self.enu_position),
            "ad_lane_id": self.ad_lane_id,
            "road_id": self.road_id,
            "section_id": self.section_id,
            "lane_id": self.lane_id,
            "parametric_offset": self.parametric_offset,
            "heading": self.heading,
            "lane_length_m": self.lane_length_m,
            "lane_width_m": self.lane_width_m,
            "lane_width": self.lane_width,
            "is_intersection": self.is_intersection,
        }
