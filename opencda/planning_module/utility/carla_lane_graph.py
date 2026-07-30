"""
Shared CARLA lane-center waypoint graph helpers.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping

import numpy as np


INVALID_LANE_ID = 0


def direction_key(carla_lane_id: int) -> str:
    return "positive" if int(carla_lane_id) > 0 else "negative"


def round_xy(x_m: float, y_m: float) -> tuple[float, float]:
    return round(float(x_m), 3), round(float(y_m), 3)


def carla_waypoint_graph_key(waypoint) -> tuple[int, int, int, float]:
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        round(float(getattr(waypoint, "s", 0.0)), 3),
    )


def _carla_waypoint_group_key(waypoint) -> tuple[int, int, str]:
    return int(waypoint.road_id), int(waypoint.section_id), direction_key(int(waypoint.lane_id))


def is_driving_waypoint(waypoint) -> bool:
    if waypoint is None:
        return False
    lane_type = getattr(waypoint, "lane_type", None)
    if lane_type is None:
        return True
    lane_type_name = getattr(lane_type, "name", lane_type)
    normalized_name = str(lane_type_name).strip().upper()
    return normalized_name.endswith("DRIVING")


def _same_lane_group(base_waypoint, candidate_waypoint) -> bool:
    if base_waypoint is None or candidate_waypoint is None:
        return False
    if not is_driving_waypoint(candidate_waypoint):
        return False

    base_lane_id = int(getattr(base_waypoint, "lane_id", 0))
    candidate_lane_id = int(getattr(candidate_waypoint, "lane_id", 0))
    if base_lane_id == 0 or candidate_lane_id == 0:
        return False
    if base_lane_id * candidate_lane_id < 0:
        return False

    # Only road_id is checked — NOT section_id.
    # In CARLA/OpenDRIVE, parallel lanes on the same physical road commonly
    # have different section_ids because lane sections can start at different
    # longitudinal offsets (e.g. when a road widens or narrows).  Requiring
    # section_id equality causes get_right_lane()/get_left_lane() neighbours
    # to be rejected, making every lane appear to be the only lane in its
    # group → canonical_lane_id_for_waypoint() returns 1 for every lane on
    # that road, so optimal_lane_id always equals ego_lane_id even when the
    # global route goes through a different lane.
    return (
        int(getattr(base_waypoint, "road_id", 0)) == int(getattr(candidate_waypoint, "road_id", 0))
    )


def canonical_lane_waypoints(waypoint) -> List[object]:
    if waypoint is None:
        return []

    rightmost_waypoint = waypoint
    while True:
        get_right_lane = getattr(rightmost_waypoint, "get_right_lane", None)
        if not callable(get_right_lane):
            break
        right_waypoint = get_right_lane()
        if not _same_lane_group(waypoint, right_waypoint):
            break
        rightmost_waypoint = right_waypoint

    lanes: List[object] = [rightmost_waypoint]
    current_waypoint = rightmost_waypoint
    while True:
        get_left_lane = getattr(current_waypoint, "get_left_lane", None)
        if not callable(get_left_lane):
            break
        left_waypoint = get_left_lane()
        if not _same_lane_group(waypoint, left_waypoint):
            break
        lanes.append(left_waypoint)
        current_waypoint = left_waypoint
    return lanes


def canonical_lane_ids_for_waypoint(waypoint) -> List[int]:
    lane_waypoints = canonical_lane_waypoints(waypoint)
    return [
        int(index) + 1
        for index, lane_waypoint in enumerate(lane_waypoints)
        if int(getattr(lane_waypoint, "lane_id", 0)) != int(INVALID_LANE_ID)
    ]


def canonical_lane_id_for_waypoint(waypoint) -> int:
    if waypoint is None:
        return int(INVALID_LANE_ID)
    raw_lane_id = int(getattr(waypoint, "lane_id", INVALID_LANE_ID))
    if int(raw_lane_id) == int(INVALID_LANE_ID):
        return int(INVALID_LANE_ID)
    lane_waypoints = canonical_lane_waypoints(waypoint)
    for lane_index, lane_waypoint in enumerate(lane_waypoints):
        if int(getattr(lane_waypoint, "lane_id", INVALID_LANE_ID)) == int(raw_lane_id):
            return int(lane_index) + 1
    return int(INVALID_LANE_ID)


def _lane_identity_key(waypoint) -> tuple[int, int]:
    return (
        int(getattr(waypoint, "road_id", 0)),
        int(getattr(waypoint, "lane_id", 0)),
    )


def lane_hop_offset(from_waypoint, to_waypoint, max_hops: int = 8) -> int | None:
    """Count the signed ``get_left_lane()``/``get_right_lane()`` hops from
    ``from_waypoint`` to the same physical lane as ``to_waypoint``.

    ``canonical_lane_id_for_waypoint`` numbers lanes by counting how many
    driving lanes exist AT THE QUERIED POINT (rightmost = 1, increasing
    leftward). That count is only meaningful locally: wherever the total
    lane count changes along a road (a lane merges away, a turn-only lane
    appears/disappears), the same physical lane a vehicle never left gets a
    different number before and after. Comparing two such numbers computed
    at different points along a route -- e.g. "is the vehicle's current
    lane the same as the lane the route requires it to reach?" -- silently
    breaks in that case: the numbers can differ, or coincidentally match,
    without either reflecting whether the vehicle actually changed lanes.

    This instead proves the relationship by walking real lane adjacency, so
    it stays correct regardless of how the lane count changes in between.
    Matching is by ``(road_id, lane_id)`` rather than waypoint identity or
    position, since ``get_left_lane()``/``get_right_lane()`` neighbours can
    land at a different ``section_id``/longitudinal offset than the target
    waypoint (see ``_same_lane_group``).

    Returns ``None`` if the two waypoints are not connected within
    ``max_hops`` lane-to-lane steps (e.g. they are on different roads, such
    as across a turn onto a cross street, or a respawn/teleport) -- callers
    should fall back to a fresh, position-only lane count in that case,
    since there is no meaningful lane identity to carry across an
    unconnected jump.
    """

    if from_waypoint is None or to_waypoint is None:
        return None
    target_key = _lane_identity_key(to_waypoint)
    if target_key == _lane_identity_key(from_waypoint):
        return 0

    left_cursor = from_waypoint
    for hop in range(1, int(max_hops) + 1):
        get_left_lane = getattr(left_cursor, "get_left_lane", None)
        if not callable(get_left_lane):
            break
        candidate = get_left_lane()
        if not _same_lane_group(from_waypoint, candidate):
            break
        left_cursor = candidate
        if _lane_identity_key(candidate) == target_key:
            return int(hop)

    right_cursor = from_waypoint
    for hop in range(1, int(max_hops) + 1):
        get_right_lane = getattr(right_cursor, "get_right_lane", None)
        if not callable(get_right_lane):
            break
        candidate = get_right_lane()
        if not _same_lane_group(from_waypoint, candidate):
            break
        right_cursor = candidate
        if _lane_identity_key(candidate) == target_key:
            return -int(hop)

    return None


class StableLaneIdTracker:
    """Track a vehicle's canonical lane id continuously across ticks.

    See ``lane_hop_offset`` for why a freshly recomputed
    ``canonical_lane_id_for_waypoint`` on every tick is unstable. This
    tracker only changes its reported id when it can prove, via real lane
    adjacency, that the vehicle's raw lane actually changed -- so the id
    stays a stable identity for the whole time the vehicle occupies the
    same physical lane, independent of how many lanes exist locally.
    """

    def __init__(self) -> None:
        self._lane_id: int | None = None
        self._waypoint = None

    def update(self, waypoint) -> int:
        if waypoint is None:
            return int(self._lane_id or 1)
        if self._waypoint is not None and self._lane_id is not None:
            hop = lane_hop_offset(self._waypoint, waypoint)
            if hop is not None:
                self._lane_id = int(self._lane_id) + int(hop)
                self._waypoint = waypoint
                return int(self._lane_id)
        # First update, or the vehicle's raw lane is not connected to the
        # previously tracked one within a few hops (e.g. it just completed
        # a turn onto a cross street, or was respawned/teleported): there is
        # no continuity to preserve, so start fresh from a local count.
        fresh_id = int(canonical_lane_id_for_waypoint(waypoint)) or 1
        self._lane_id = int(fresh_id)
        self._waypoint = waypoint
        return int(self._lane_id)

    def reset(self) -> None:
        self._lane_id = None
        self._waypoint = None


def canonical_lane_waypoint_for_lane_id(waypoint, target_lane_id: int):
    lane_waypoints = canonical_lane_waypoints(waypoint)
    if len(lane_waypoints) == 0:
        return waypoint
    lane_index = int(target_lane_id) - 1
    if 0 <= int(lane_index) < len(lane_waypoints):
        return lane_waypoints[int(lane_index)]
    return waypoint


def raw_carla_lane_id_for_waypoint(waypoint) -> int:
    if waypoint is None:
        return int(INVALID_LANE_ID)
    return int(getattr(waypoint, "lane_id", INVALID_LANE_ID))


def _internal_lane_id(waypoint, lane_ids_by_group: Mapping[tuple[int, int, str], set[int]]) -> int:
    del lane_ids_by_group
    return int(canonical_lane_id_for_waypoint(waypoint))


def build_lane_center_waypoints(map_obj, carla, sample_distance_m: float) -> tuple[list[dict], dict]:
    sampled_waypoints = [
        waypoint
        for waypoint in map_obj.generate_waypoints(float(sample_distance_m))
        if waypoint.lane_type == carla.LaneType.Driving and int(waypoint.lane_id) != 0
    ]
    lane_ids_by_group: Dict[tuple[int, int, str], set[int]] = {}
    for waypoint in sampled_waypoints:
        lane_ids_by_group.setdefault(_carla_waypoint_group_key(waypoint), set()).add(int(waypoint.lane_id))

    valid_waypoints = []
    for waypoint in sampled_waypoints:
        canonical_lane_id = _internal_lane_id(waypoint, lane_ids_by_group)
        valid_waypoints.append(
            {
                "carla_waypoint": waypoint,
                "carla_waypoint_key": carla_waypoint_graph_key(waypoint),
                "position": [float(waypoint.transform.location.x), float(waypoint.transform.location.y)],
                "heading_rad": math.radians(float(waypoint.transform.rotation.yaw)),
                "lane_id": int(canonical_lane_id),
                "carla_lane_id": int(getattr(waypoint, "lane_id", 0)),
                "road_id": f"{int(waypoint.road_id)}:{int(waypoint.section_id)}",
                "direction": direction_key(int(waypoint.lane_id)),
                "lane_width_m": float(waypoint.lane_width),
                "is_intersection": bool(waypoint.is_junction),
                "maneuver": "straight",
            }
        )

    for item in valid_waypoints:
        carla_waypoint = item["carla_waypoint"]
        next_candidates = [
            waypoint
            for waypoint in carla_waypoint.next(float(sample_distance_m))
            if waypoint.lane_type == carla.LaneType.Driving
        ]
        if next_candidates:
            successor_positions_by_key: Dict[tuple[float, float], list[float]] = {}
            successor_keys = []
            for next_waypoint in next_candidates:
                successor_xy = [
                    float(next_waypoint.transform.location.x),
                    float(next_waypoint.transform.location.y),
                ]
                successor_positions_by_key[round_xy(successor_xy[0], successor_xy[1])] = successor_xy
                successor_key = carla_waypoint_graph_key(next_waypoint)
                if successor_key not in successor_keys:
                    successor_keys.append(successor_key)
            item["successors"] = list(successor_positions_by_key.values())
            item["successor_keys"] = list(successor_keys)

            preferred_next_candidates = [
                waypoint
                for waypoint in next_candidates
                if direction_key(int(waypoint.lane_id)) == direction_key(int(carla_waypoint.lane_id))
            ]
            selection_pool = preferred_next_candidates if len(preferred_next_candidates) > 0 else next_candidates
            next_waypoint = min(
                selection_pool,
                key=lambda waypoint: abs(
                    math.atan2(
                        math.sin(
                            math.radians(float(waypoint.transform.rotation.yaw))
                            - math.radians(float(carla_waypoint.transform.rotation.yaw))
                        ),
                        math.cos(
                            math.radians(float(waypoint.transform.rotation.yaw))
                            - math.radians(float(carla_waypoint.transform.rotation.yaw))
                        ),
                    )
                ),
            )
            item["next"] = [
                float(next_waypoint.transform.location.x),
                float(next_waypoint.transform.location.y),
            ]
            item["next_key"] = carla_waypoint_graph_key(next_waypoint)

    lane_width_m = (
        float(np.median([float(item["lane_width_m"]) for item in valid_waypoints]))
        if valid_waypoints
        else 4.0
    )
    lane_count = (
        max(len(lane_ids) for lane_ids in lane_ids_by_group.values())
        if len(lane_ids_by_group) > 0
        else 1
    )
    road_cfg = {
        "lane_width_m": float(lane_width_m),
        "lane_count": int(lane_count),
    }
    return valid_waypoints, road_cfg
