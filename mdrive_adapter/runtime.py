"""MDrive GT and control boundary for the current CP-X planning pipeline."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace


def merge_config(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def collect_gt(world):
    """Collect one current-state GT frame in the evaluator, shared by all egos."""
    snapshots = []
    for actor in world.get_actors():
        kind = actor.type_id
        if not kind.startswith(("vehicle.", "walker.pedestrian.", "static.prop.")) or not actor.is_alive:
            continue
        transform = actor.get_transform()
        bbox = getattr(actor, "bounding_box", None)
        if bbox is None:
            continue
        velocity = actor.get_velocity()
        offset = bbox.location
        center = transform.transform(type(offset)(x=offset.x, y=offset.y, z=offset.z))
        speed = math.hypot(velocity.x, velocity.y)
        heading = math.radians(transform.rotation.yaw + bbox.rotation.yaw)
        if kind.startswith("walker.") and speed > 0.1:
            heading = math.atan2(velocity.y, velocity.x)
        snapshots.append({
            "vehicle_id": str(actor.id), "type": kind,
            "object_type": ("pedestrian" if kind.startswith("walker.") else
                            "vehicle" if kind.startswith("vehicle.") else "static_object"),
            "source": "mdrive_gt", "provider_source": "current_state_oracle",
            "x": float(center.x), "y": float(center.y), "z": float(center.z),
            "v": speed, "psi": heading,
            "length_m": max(0.2, float(bbox.extent.x) * 2),
            "width_m": max(0.2, float(bbox.extent.y) * 2),
            "height_m": max(0.2, float(bbox.extent.z) * 2),
        })
    return snapshots


def make_planner(slot, actor, world, route, config, output):
    """Use the same stateful bridge in serial and spawned-worker execution."""
    context = ExternalContext(actor, route, config)
    return context, planning_steps(context, slot, world, Path(output))


class ExternalContext:
    def __init__(self, vehicle, route, config):
        if len(route) < 2:
            raise ValueError("CP-X requires at least two MDrive route points")
        self.vehicle = vehicle
        self.route = list(route)
        self.config = copy.deepcopy(config)
        self.snapshots = []
        self.bridge = None

    def obstacles(self):
        location = self.vehicle.get_location()
        radius = float(self.config.get("gt_range_m", 70.0))
        result = [dict(item) for item in self.snapshots
                  if item["vehicle_id"] != str(self.vehicle.id)
                  and math.hypot(item["x"] - location.x, item["y"] - location.y) <= radius]
        return sorted(result, key=lambda item: (math.hypot(item["x"] - location.x, item["y"] - location.y),
                                                item["vehicle_id"]))


def build_bridge_config(config, output, world_map):
    """Translate adapter settings without silently discarding MPC overrides."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    with (root / "opencda/planning_module/MPC/mpc.yaml").open() as stream:
        payload = yaml.safe_load(stream)
    payload["mpc"] = merge_config(payload["mpc"], config.get("mpc", {}))
    constraints = config.get("scenario", {}).get("constraints", {})
    payload["mpc"]["constraints"] = merge_config(payload["mpc"].get("constraints", {}), constraints)
    mpc_path = output / "mpc.yaml"
    mpc_path.write_text(yaml.safe_dump(payload))
    # The actual simulator map is the authoritative map, including local edits.
    xodr = output / "map.xodr"
    xodr.write_text(world_map.to_opendrive())
    bridge_config = merge_config({
        "target_speed_mps": float(constraints.get("max_velocity_mps", 7.0)),
        "max_mpc_obstacles": int(config.get("mpc", {}).get("obstacle_filter", {}).get("max_planning_obstacles", 64)),
        "publish_cp_message": False,
        "cav_intent_broadcast_enabled": False,
        "route_replan_enabled": False,
        "route_reached_distance_m": 0.5,
        "debug": False,
    }, config.get("bridge", {}))
    bridge_config.update({"mpc_config_path": str(mpc_path), "global_planner_xodr_path": str(xodr),
                          "global_planner_cache_root": str(output / "map_cache"),
                          "cp_message_path": str(output / "cp_message.json"),
                          "debug_output_dir": str(output), "scenario_name": "mdrive"})
    return bridge_config


def install_route(bridge, context):
    """Keep benchmark waypoints/options; append only a terminal stopping tail."""
    points = [[float(t.location.x), float(t.location.y), float(t.location.z)] for t, _ in context.route]
    options = [str(getattr(option, "name", option)).split(".")[-1] for _, option in context.route]
    clearance = max(0.0, float(context.config.get("route_end_clearance_m", 3.0)))
    if clearance:
        end = context.route[-1][0]
        yaw = math.radians(end.rotation.yaw)
        # Keep every benchmark point in order and make stopping happen past its finish line.
        points.append([points[-1][0] + clearance * math.cos(yaw),
                       points[-1][1] + clearance * math.sin(yaw), points[-1][2]])
        options.append("LANEFOLLOW")
    bridge.route_manager.install_external_route(points, options, allow_replan=False)


def planning_steps(context, slot, world, output):
    from opencda.core.actuation.pid_controller import Controller
    from opencda.planning_module.opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge
    actor = context.vehicle
    world_map = world.get_map()
    settings = context.config.get("controller", {})
    controller = Controller({"max_brake": 1.0, "max_throttle": 1.0, "max_steering": 1.0,
        "lon": {"k_p": float(settings.get("speed_kp", 1.0)) / 3.6,
                "k_i": float(settings.get("speed_ki", 0.2)) / 3.6, "k_d": 0.0},
        "lat": {"k_p": 0.0, "k_i": 0.0, "k_d": 0.0}, "dt": 0.05, "dynamic": False})
    def speed_kmh():
        velocity = actor.get_velocity()
        return 3.6 * math.hypot(velocity.x, velocity.y)
    manager = SimpleNamespace(vehicle=actor, carla_map=world_map, controller=controller,
        localizer=SimpleNamespace(get_ego_pos=actor.get_transform, get_ego_spd=speed_kmh),
        perception_manager=SimpleNamespace(objects={}), v2x_manager=None,
        _opencda_agent_finished=False)
    bridge = None
    try:
        bridge = CPXMPCPlannerBridge(manager, build_bridge_config(context.config, output, world_map))
        context.bridge = bridge
        install_route(bridge, context)
        while True:
            bridge.update_information(ego_transform=actor.get_transform(), ego_speed_kmh=speed_kmh(),
                                      detected_objects={"vehicles": context.obstacles()})
            # Use the full pipeline but propagate exceptions to the benchmark. Its normal
            # bounded fallback controls remain valid; programming errors must fail the run.
            result = bridge.execute_planning_pipeline()
            bridge.last_output = result
            bridge.last_debug = result.diagnostics_dict()
            bridge._record_debug(bridge.last_debug)
            yield {"done": bool(manager._opencda_agent_finished), "control": result.control,
                   "trajectory": [list(state) for state in result.planned_trajectory],
                   "status": dict(bridge.mpc.get_runtime_status()),
                   "behavior": str(result.behavior_command.decision)}
    finally:
        if bridge is not None:
            bridge.destroy()
        context.bridge = None
