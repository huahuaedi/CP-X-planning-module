"""Resolve scenario destinations by following the destination lane forward."""

import math


def _wrapped_angle_deg(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def _waypoint_yaw(waypoint):
    return float(waypoint.transform.rotation.yaw)


def _successor_score(current, successor):
    """Prefer the same lane and the smallest discontinuity at a road split."""

    same_lane = (
        getattr(current, "lane_id", None) == getattr(successor, "lane_id", None)
    )
    heading_jump = abs(_wrapped_angle_deg(
        _waypoint_yaw(successor) - _waypoint_yaw(current)
    ))
    current_location = current.transform.location
    successor_location = successor.transform.location
    segment_heading = math.degrees(math.atan2(
        float(successor_location.y) - float(current_location.y),
        float(successor_location.x) - float(current_location.x),
    ))
    geometry_jump = abs(_wrapped_angle_deg(segment_heading - _waypoint_yaw(current)))
    # lane_id is only a local CARLA identity, so use it as a preference rather
    # than a hard constraint; it is allowed to change at a road boundary.
    return (0 if same_lane else 1, heading_jump, geometry_jump)


def follow_lane_successor(start_waypoint, distance_m, step_m=2.0):
    """Follow a CARLA waypoint's longitudinal successors for ``distance_m``.

    This never calls ``get_left_lane``/``get_right_lane``.  Consequently an
    endpoint extension cannot introduce a lane change by itself.
    """

    remaining_m = max(0.0, float(distance_m))
    current = start_waypoint
    while current is not None and remaining_m > 1e-6:
        requested_step_m = min(max(0.25, float(step_m)), remaining_m)
        successors = list(current.next(requested_step_m) or [])
        if not successors:
            break
        successor = min(successors, key=lambda item: _successor_score(current, item))
        current_location = current.transform.location
        successor_location = successor.transform.location
        travelled_m = math.hypot(
            float(successor_location.x) - float(current_location.x),
            float(successor_location.y) - float(current_location.y),
        )
        if travelled_m <= 1e-4:
            break
        remaining_m -= travelled_m
        current = successor
    return current, max(0.0, remaining_m)


def resolve_destination_extension(
        carla_map, destination, extension_m, step_m=2.0, location_factory=None):
    """Return the destination extended along its projected driving lane."""

    extension_m = max(0.0, float(extension_m))
    destination = [float(value) for value in destination]
    if extension_m <= 1e-6:
        return destination, 0.0

    # Importing CARLA is intentionally left to the caller.  This utility stays
    # unit-testable with lightweight waypoint doubles.
    if location_factory is None:
        spawn_points = list(carla_map.get_spawn_points() or [])
        if not spawn_points:
            raise RuntimeError("CARLA map exposes no location type for destination projection")
        location_factory = type(spawn_points[0].location)
    location = location_factory(
        x=destination[0], y=destination[1], z=destination[2]
    )
    try:
        start_waypoint = carla_map.get_waypoint(location, project_to_road=True)
    except TypeError:
        start_waypoint = carla_map.get_waypoint(location)
    if start_waypoint is None:
        raise RuntimeError("could not project the configured destination to a road lane")

    end_waypoint, remaining_m = follow_lane_successor(
        start_waypoint, extension_m, step_m=step_m
    )
    if end_waypoint is None:
        raise RuntimeError("destination lane has no usable longitudinal successor")
    start_location = start_waypoint.transform.location
    end_location = end_waypoint.transform.location
    z_offset = destination[2] - float(start_location.z)
    return [
        float(end_location.x),
        float(end_location.y),
        float(end_location.z) + z_offset,
    ], remaining_m
