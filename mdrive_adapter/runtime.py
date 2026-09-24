"""GT inputs and caller-owned route context for the CP-X planning generator."""
import copy
import math


def merge_config(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def collect_gt(world):
    """Current GT states keyed by actor ID; no future motion or occlusion model."""
    snapshots = []
    for actor in world.get_actors():
        kind = actor.type_id
        if not kind.startswith(("vehicle.", "walker.pedestrian.", "static.prop.")):
            continue
        if not actor.is_alive:
            continue
        transform = actor.get_transform()
        bbox = getattr(actor, "bounding_box", None)
        if bbox is None:
            continue
        velocity = actor.get_velocity()
        offset = bbox.location
        # CARLA Boost.Python Location objects cannot be copied/pickled.
        center = transform.transform(type(offset)(x=offset.x, y=offset.y, z=offset.z))
        speed = math.hypot(velocity.x, velocity.y)
        heading = math.radians(transform.rotation.yaw + bbox.rotation.yaw)
        if kind.startswith("walker.") and speed > 0.1:
            heading = math.atan2(velocity.y, velocity.x)
        snapshots.append({
            "vehicle_id": str(actor.id), "type": kind,
            "x": float(center.x), "y": float(center.y), "z": float(center.z),
            "v": speed, "psi": heading,
            "length_m": max(0.2, float(bbox.extent.x) * 2),
            "width_m": max(0.2, float(bbox.extent.y) * 2),
            "height_m": max(0.2, float(bbox.extent.z) * 2),
        })
    return snapshots


def make_planner(slot, actor, world, route, config, output):
    """Identical planner construction for in-process and process workers."""
    from pathlib import Path
    import carla
    from planning_runner import iter_planning_steps
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    context = ExternalContext(actor, route, config)
    message_path = str(output / "cp_message.json")
    scenario = copy.deepcopy(config["scenario"])
    scenario.update({"name": "cpx_gt_ego_%d" % slot, "_scenario_dir": str(output),
                     "camera": {"enabled": False}, "sumo": {"enabled": False},
                     "traffic_manager": {"enabled": False}, "runtime": {}, "obstacles": {},
                     "metrics": {"collision_sensor_enabled": False}})
    context.config = merge_config(config, {"mpc": {"behavior_planner_runtime": {
        "cooperative_message_path": message_path,
        "rule_based": {"cooperative_message_path": message_path}}}})
    return context, iter_planning_steps(None, world, scenario, carla, external=context)


class ExternalContext:
    def __init__(self, vehicle, route, config):
        if len(route) < 2:
            raise ValueError("CP-X requires a non-empty MDrive world route (at least two points)")
        self.vehicle = vehicle
        self.route = list(route)
        self.destination_transform = self.route[-1][0]
        self.config = config
        self.snapshots = []
        self.plan_time_s = None
        self.signal_query_key = "mdrive:%s:%s" % (getattr(vehicle, "id", "test"), id(self))
        self._speed_integral = 0.0
        self._control_time_s = None

    def configure_mpc(self, base):
        return merge_config(base, self.config.get("mpc", {}))

    def control_index(self, sim_time_s, dt_s):
        if self.plan_time_s is None:
            return 0
        return max(0, int(math.floor((sim_time_s - self.plan_time_s + 1e-8) / dt_s)))

    def close(self):
        from behavior_planner.traffic_light_stop import _clear_stop_target_query_state
        _clear_stop_target_query_state(self.signal_query_key)

    def track_control(self, control, trajectory, index, mpc_dt_s, sim_time_s):
        """Track MPC speed with feedback to overcome drivetrain resistance."""
        if not trajectory:
            return control
        settings = self.config.get("controller", {})
        lookahead = max(0.0, float(settings.get("speed_lookahead_s", 0.5)))
        target_index = min(len(trajectory) - 1, index + max(0, int(round(lookahead / mpc_dt_s)) - 1))
        target_speed = max(0.0, float(trajectory[target_index][2]))
        velocity = self.vehicle.get_velocity()
        speed = math.hypot(velocity.x, velocity.y)
        dt = 0.05 if self._control_time_s is None else max(0.0, min(0.2, sim_time_s - self._control_time_s))
        self._control_time_s = sim_time_s
        error = target_speed - speed
        self._speed_integral = max(-1.0, min(1.0, self._speed_integral + error * dt))
        if target_speed < 0.05:
            self._speed_integral = 0.0
        pedal = (float(control.throttle) - float(control.brake)
                 + float(settings.get("speed_kp", 1.0)) * error
                 + float(settings.get("speed_ki", 0.2)) * self._speed_integral)
        control.throttle = min(1.0, max(0.0, pedal))
        control.brake = min(1.0, max(0.0, -pedal))
        if target_speed < 0.05 and speed < 0.1:
            control.throttle = 0.0
            control.brake = max(0.3, control.brake)
        return control

    def obstacles(self):
        location = self.vehicle.get_location()
        radius = float(self.config.get("gt_range_m", 70.0))
        result = [dict(item) for item in self.snapshots
                  if item["vehicle_id"] != str(self.vehicle.id)
                  and math.hypot(item["x"] - location.x, item["y"] - location.y) <= radius]
        return sorted(result, key=lambda item: math.hypot(item["x"] - location.x, item["y"] - location.y))

    def install_route(self, planner, world_map, carla):
        from utility.global_planner import RoutePlanSummary
        from utility import canonical_lane_id_for_waypoint
        points, options, lanes, roads = [], [], [], []
        for transform, option in self.route:
            waypoint = world_map.get_waypoint(transform.location, project_to_road=True,
                                             lane_type=carla.LaneType.Driving)
            if waypoint is None:
                raise ValueError("MDrive route point has no driving waypoint: %s" % transform.location)
            points.append([float(transform.location.x), float(transform.location.y)])
            options.append(str(getattr(option, "name", option)).split(".")[-1])
            lanes.append(int(canonical_lane_id_for_waypoint(waypoint)))
            roads.append(str(waypoint.road_id))
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
        summary = RoutePlanSummary(
            route_found=True, start_road_id=roads[0], start_lane_id=lanes[0],
            goal_road_id=roads[-1], goal_lane_id=lanes[-1], optimal_lane_id=lanes[0],
            distance_to_destination_m=length, next_macro_maneuver="straight",
            route_waypoints=points, road_options=options, current_road_option=options[0],
        )
        planner.store_initial_route(summary, options, lanes)
        # Extend only the stop target so cars cross MDrive's final waypoint.
        clearance = max(0.0, float(self.config.get("route_end_clearance_m", 3.0)))
        if clearance > 0:
            continuations = waypoint.next(clearance)
            if continuations:
                end = self.route[-1][0]
                yaw = math.radians(end.rotation.yaw)
                expected_x = end.location.x + clearance * math.cos(yaw)
                expected_y = end.location.y + clearance * math.sin(yaw)
                continuation = min(continuations, key=lambda w: math.hypot(
                    w.transform.location.x - expected_x, w.transform.location.y - expected_y))
                self.destination_transform = continuation.transform
        return summary
