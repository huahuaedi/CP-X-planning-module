# -*- coding: utf-8 -*-
"""
Standalone diagnostic (no vehicle spawn, no tick loop): checks whether
carla_map.get_waypoint() snaps the MDrive Unprotected_Left_Turn/1 literal
route points to a continuous sequence of lanes, or jumps around at the
curve. Run this before re-running the full mdrive_test_left_turn scenario
whenever debugging why the CP-X MPC solver goes infeasible mid-turn.

Usage:
    PYTHONPATH=<carla-py3.7-egg> python opencda/scenario_testing/debug_mdrive_waypoint_snap.py
"""

import carla

MDRIVE_ROUTE_POINTS = [
    (-173.802, 91.368, 0.057296), (-171.802, 91.370, 0.057296), (-167.802, 91.375, 0.057296),
    (-163.802, 91.379, 0.057296), (-159.802, 91.384, 0.057296), (-157.802, 91.386, 0.057296),
    (-153.802, 91.391, 0.057296), (-149.802, 91.395, 0.057296), (-145.802, 91.400, 0.057296),
    (-141.802, 91.404, 0.057296), (-140.057, 91.359, -7.170340), (-137.879, 91.085, -7.170340),
    (-135.746, 90.570, -7.170340), (-133.683, 89.819, -7.170340), (-131.717, 88.842, -7.170340),
    (-128.144, 86.187, -7.170340), (-126.661, 84.448, -7.170340), (-125.484, 82.490, -7.170340),
    (-124.643, 80.365, -7.170340), (-124.047, 75.892, -7.170340), (-124.082, 71.892, -90.515734),
    (-124.136, 65.893, -90.515734), (-124.189, 59.893, -90.515734), (-124.225, 55.893, -90.515734),
    (-124.279, 49.893, -90.515734), (-124.332, 43.893, -90.515734), (-124.368, 39.894, -90.515734),
    (-124.421, 33.894, -90.515734), (-124.475, 27.894, -90.515734),
]


def main() -> None:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != "Town05":
        world = client.load_world("Town05")
    carla_map = world.get_map()

    header = f"{'idx':>3} {'mdrive_x':>10} {'mdrive_y':>10} {'mdrive_yaw':>10} | {'wp_x':>10} {'wp_y':>10} {'wp_yaw':>10} {'road_id':>7} {'lane_id':>7} {'s':>8} {'is_junction':>11}"
    print(header)
    prev_road_id = None
    for index, (x, y, yaw) in enumerate(MDRIVE_ROUTE_POINTS):
        location = carla.Location(x=x, y=y, z=0.0)
        waypoint = carla_map.get_waypoint(location)
        flag = ""
        if prev_road_id is not None and waypoint.road_id != prev_road_id:
            flag = "  <- road_id changed"
        prev_road_id = waypoint.road_id
        print(
            f"{index:>3} {x:>10.3f} {y:>10.3f} {yaw:>10.3f} | "
            f"{waypoint.transform.location.x:>10.3f} {waypoint.transform.location.y:>10.3f} "
            f"{waypoint.transform.rotation.yaw:>10.3f} {waypoint.road_id:>7} {waypoint.lane_id:>7} "
            f"{waypoint.s:>8.3f} {str(waypoint.is_junction):>11}{flag}"
        )


if __name__ == "__main__":
    main()
