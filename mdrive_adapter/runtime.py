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


def collect_gt(world, *, reference_points=(), max_static_range_m=None):
    """Collect one current-state GT frame in the evaluator, shared by all egos.

    ``reference_points`` (ego (x, y) locations) and ``max_static_range_m``
    are optional and only ever narrow the ``static.*`` category (see below);
    omitting either keeps the static category unfiltered; callers that need
    town-scale collection can retain that behavior explicitly. Vehicles/walkers are never range-filtered
    here -- they're always few enough to be cheap, and every ego's own
    ``context.obstacles()`` already does its own gt_range_m cut downstream.
    """
    snapshots = []
    for actor in world.get_actors():
        kind = actor.type_id
        # "static." (not just "static.prop.") also covers CARLA's map-furniture
        # actors -- e.g. static.pole -- which are real, queryable, collidable
        # actors distinct from the static OpenDRIVE map mesh. Excluding them
        # left the planner with zero perception of roadside poles/signs it can
        # still physically hit; confirmed via a live collision report:
        # "Agent collided against object with type=static.pole and id=0".
        if not kind.startswith(("vehicle.", "walker.pedestrian.", "static.")) or not actor.is_alive:
            continue
        transform = actor.get_transform()
        if (
            kind.startswith("static.")
            and reference_points
            and max_static_range_m is not None
        ):
            # A whole town's worth of streetlights/signs is not something
            # any ego needs to carry through Stage A/candidate evaluation
            # just because it exists -- cut it here, before the more
            # expensive per-actor velocity/bounding-box work below, rather
            # than only at each ego's later gt_range_m filter (which still
            # has to iterate every entry in this shared list to get there).
            loc = transform.location
            if not any(
                math.hypot(float(loc.x) - float(rx), float(loc.y) - float(ry))
                <= float(max_static_range_m)
                for rx, ry in reference_points
            ):
                continue
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
        # Peer CavIntent payloads received from the evaluator's previous-tick
        # broadcast, keyed by the peer's CARLA actor id (as a string, matching
        # cav_intent_codec's own actor_id-keyed lookups). Empty unless the
        # caller (parallel.py's planner_worker, or the serial CPXAgent loop)
        # feeds it -- see collect_own_cav_intent()/inject_cav_intents() below.
        self.cav_intents = {}
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
    # cav_nearby is rebuilt every tick from context.cav_intents (see below);
    # CPXMPCPlannerBridge._collect_cav_intents() only ever reads it through
    # this SimpleNamespace chain, so it never needs to know this is an
    # IPC-relayed payload rather than a same-process peer object.
    v2x_manager = SimpleNamespace(cav_nearby={})
    manager = SimpleNamespace(vehicle=actor, carla_map=world_map, controller=controller,
        localizer=SimpleNamespace(get_ego_pos=actor.get_transform, get_ego_spd=speed_kmh),
        perception_manager=SimpleNamespace(objects={}), v2x_manager=v2x_manager,
        _opencda_agent_finished=False)
    bridge = None
    try:
        bridge = CPXMPCPlannerBridge(manager, build_bridge_config(context.config, output, world_map))
        context.bridge = bridge
        # last_cav_intent_payload/cpx_planner is the same in-process attribute
        # chain _collect_cav_intents() reads for a same-process peer (see
        # mdrive-cpx-convert-to-ros-adapter-fixes: self._vehicle_manager.
        # cpx_planner = self._bridge). Setting it here lets a peer worker's
        # relayed payload be wrapped in an identical-shaped stub below.
        manager.cpx_planner = bridge
        install_route(bridge, context)
        while True:
            v2x_manager.cav_nearby = {
                str(peer_id): SimpleNamespace(
                    cpx_planner=SimpleNamespace(last_cav_intent_payload=dict(payload))
                )
                for peer_id, payload in (context.cav_intents or {}).items()
            }
            bridge.update_information(ego_transform=actor.get_transform(), ego_speed_kmh=speed_kmh(),
                                      detected_objects={"vehicles": context.obstacles()})
            # Use the full pipeline but propagate exceptions to the benchmark. Its normal
            # bounded fallback controls remain valid; programming errors must fail the run.
            result = bridge.execute_planning_pipeline()
            bridge.last_output = result
            bridge.last_debug = result.diagnostics_dict()
            bridge._record_debug(bridge.last_debug)
            own_intent = getattr(bridge, "last_cav_intent_payload", None)
            yield {"done": bool(manager._opencda_agent_finished), "control": result.control,
                   "trajectory": [list(state) for state in result.planned_trajectory],
                   "status": dict(bridge.mpc.get_runtime_status()),
                   "behavior": str(result.behavior_command.decision),
                   "cav_intent": dict(own_intent) if own_intent is not None else None}
    finally:
        if bridge is not None:
            bridge.destroy()
        context.bridge = None
