"""Serialize the planner's authoritative local map for prediction models."""

from __future__ import annotations

from typing import Any, Dict, List


def mtr_map_polylines(local_map: Any) -> List[Dict[str, Any]]:
    """Return world-frame lane and boundary polylines from LocalMapSnapshot.

    This adapter deliberately consumes only the immutable local-map contract;
    it does not query CARLA or AD-map again.  ``global_type`` follows Waymo's
    map feature ids used by MTR: 2 is a surface-street lane center and 15 is a
    road edge.  The MTR service owns centering, rotation and tensor packing.
    """

    if local_map is None or not bool(getattr(local_map, "valid", False)):
        return []
    rows: List[Dict[str, Any]] = []
    seen_lane_ids = set()
    for corridor in tuple(getattr(local_map, "corridors", ()) or ()):
        for geometry in tuple(getattr(corridor, "lane_geometries", ()) or ()):
            lane_id = int(getattr(geometry, "lane_id", 0) or 0)
            points = tuple(getattr(geometry, "centerline", ()) or ())
            if lane_id == 0 or lane_id in seen_lane_ids or len(points) < 2:
                continue
            seen_lane_ids.add(lane_id)

            def point_dict(point: Any, x_name: str, y_name: str) -> Dict[str, float]:
                return {
                    "x": float(getattr(point, x_name)),
                    "y": float(getattr(point, y_name)),
                    "z": 0.0,
                }

            rows.append({
                "lane_id": lane_id,
                "kind": "lane_center",
                "global_type": 2,
                "points": [point_dict(point, "x_m", "y_m") for point in points],
            })
            rows.append({
                "lane_id": lane_id,
                "kind": "left_road_edge",
                "global_type": 15,
                "points": [
                    point_dict(point, "left_boundary_x_m", "left_boundary_y_m")
                    for point in points
                ],
            })
            rows.append({
                "lane_id": lane_id,
                "kind": "right_road_edge",
                "global_type": 15,
                "points": [
                    point_dict(point, "right_boundary_x_m", "right_boundary_y_m")
                    for point in points
                ],
            })
    return rows
