# -*- coding: utf-8 -*-
"""
Standalone diagnostic (no vehicle spawn): runs the exact same densification
logic as mdrive_test_left_turn._inject_literal_route() and prints the
resulting path's x/y/yaw sequence, so we can check for backtracking or
wrong-direction jumps before spending time on a live scenario run.
"""

import carla

MDRIVE_ROUTE_POINTS = [
    (-173.802, 91.368, 0.0), (-171.802, 91.370, 0.0), (-167.802, 91.375, 0.0),
    (-163.802, 91.379, 0.0), (-159.802, 91.384, 0.0), (-157.802, 91.386, 0.0),
    (-153.802, 91.391, 0.0), (-149.802, 91.395, 0.0), (-145.802, 91.400, 0.0),
    (-141.802, 91.404, 0.0), (-140.057, 91.359, 0.0), (-137.879, 91.085, 0.0),
    (-135.746, 90.570, 0.0), (-133.683, 89.819, 0.0), (-131.717, 88.842, 0.0),
    (-128.144, 86.187, 0.0), (-126.661, 84.448, 0.0), (-125.484, 82.490, 0.0),
    (-124.643, 80.365, 0.0), (-124.047, 75.892, 0.0), (-124.082, 71.892, 0.0),
    (-124.136, 65.893, 0.0), (-124.189, 59.893, 0.0), (-124.225, 55.893, 0.0),
    (-124.279, 49.893, 0.0), (-124.332, 43.893, 0.0), (-124.368, 39.894, 0.0),
    (-124.421, 33.894, 0.0), (-124.475, 27.894, 0.0),
]

DENSIFY_STEP_M = 1.5
DENSIFY_THRESHOLD_M = 2.5


def _densify_between(from_waypoint, to_location, step_m):
    inserted = []
    current = from_waypoint
    remaining_m = current.transform.location.distance(to_location)
    guard = 0
    while remaining_m > step_m and guard < 200:
        guard += 1
        candidates = list(current.next(step_m) or [])
        if not candidates:
            break
        current = min(candidates, key=lambda wp: wp.transform.location.distance(to_location))
        inserted.append(current)
        remaining_m = current.transform.location.distance(to_location)
    return inserted


def main() -> None:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != "Town05":
        world = client.load_world("Town05")
    carla_map = world.get_map()

    raw_waypoints = []
    for x, y, _yaw in MDRIVE_ROUTE_POINTS:
        location = carla.Location(x=x, y=y, z=0.0)
        raw_waypoints.append(carla_map.get_waypoint(location))

    densified = [raw_waypoints[0]]
    for waypoint in raw_waypoints[1:]:
        previous = densified[-1]
        gap_m = previous.transform.location.distance(waypoint.transform.location)
        if gap_m > DENSIFY_THRESHOLD_M:
            inserted = _densify_between(previous, waypoint.transform.location, DENSIFY_STEP_M)
            densified.extend(inserted)
        densified.append(waypoint)

    print(f"total densified points: {len(densified)}")
    print(f"{'idx':>3} {'x':>10} {'y':>10} {'yaw':>10} {'road_id':>7} {'lane_id':>7} {'s':>8} {'step_dist_m':>11}")
    prev_loc = None
    for index, waypoint in enumerate(densified):
        loc = waypoint.transform.location
        step_dist = "" if prev_loc is None else f"{loc.distance(prev_loc):.3f}"
        prev_loc = loc
        print(
            f"{index:>3} {loc.x:>10.3f} {loc.y:>10.3f} {waypoint.transform.rotation.yaw:>10.3f} "
            f"{waypoint.road_id:>7} {waypoint.lane_id:>7} {waypoint.s:>8.3f} {step_dist:>11}"
        )


if __name__ == "__main__":
    main()
