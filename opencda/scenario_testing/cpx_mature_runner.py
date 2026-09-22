# -*- coding: utf-8 -*-
"""Run CP-X planner on mature OpenCDA scenario layouts."""

import math
import os
import re
from collections.abc import Mapping

import carla
from omegaconf import OmegaConf

import opencda.scenario_testing.utils.customized_map_api as map_api
import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.planning_module.opencda_bridge.debug_viewer import OpenCDADebugViewer
from opencda.planning_module.utility.cp_messages import (
    CP_MESSAGE_PATH,
    reset_cp_message_payload,
    upsert_cp_item,
)
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.scripted_actor import spawn_scripted_actors
from opencda.scenario_testing.run_status import write_run_status
from opencda.scenario_testing.utils.yaml_utils import add_current_time


def _assign_scenario_actor_roles(scenario_params, script_name):
    """Give every scenario-owned actor a stable, run-independent identity."""

    scope = re.sub(r"[^a-zA-Z0-9_]+", "_", str(script_name)).strip("_").lower()
    prefix = "opencda_cpx_%s" % (scope or "scenario")
    scenario = scenario_params.get("scenario", {})
    roles = []
    for index, config in enumerate(scenario.get("single_cav_list", []) or []):
        name = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            str(config.get("name", "cav_%d" % index)),
        ).strip("_").lower()
        role = "%s_cav_%s" % (prefix, name or index)
        config["role_name"] = role
        roles.append(role)
    for index, config in enumerate(scenario.get("scripted_actors", []) or []):
        role = "%s_scripted_%d" % (prefix, index)
        semantic_type = re.sub(
            r"[^a-zA-Z0-9_]+", "_", str(config.get("object_type", ""))
        ).strip("_").lower()
        if semantic_type:
            # CARLA actors do not support arbitrary user metadata. Encode the
            # scenario contract in role_name so perception can recover it at
            # the simulator boundary instead of guessing from a blueprint.
            role = "%s_type_%s" % (role, semantic_type)
        config["role_name"] = role
        roles.append(role)
    return tuple(roles)


def _destroy_stale_scenario_actors(world, owned_roles):
    """Remove only actors owned by an earlier run of this exact scenario."""

    owned = {str(role).strip().lower() for role in owned_roles if str(role).strip()}
    destroyed = []
    if not owned:
        return destroyed
    for actor in world.get_actors():
        role = str(getattr(actor, "attributes", {}).get("role_name", "")).strip().lower()
        if role not in owned:
            continue
        actor_id = int(getattr(actor, "id", -1))
        try:
            actor.destroy()
        except (RuntimeError, AttributeError):
            continue
        destroyed.append(actor_id)
    if destroyed:
        print(
            "[CP-X scenario] removed %d stale owned actor(s): %s"
            % (len(destroyed), destroyed)
        )
    return destroyed


def _foreign_dynamic_actors(world, owned_roles):
    """Describe vehicles/walkers not owned by the active isolated fixture."""

    owned = {str(role).strip().lower() for role in owned_roles if str(role).strip()}
    foreign = []
    for actor in world.get_actors():
        type_id = str(getattr(actor, "type_id", "")).strip().lower()
        if not type_id.startswith(("vehicle.", "walker.")):
            continue
        role = str(getattr(actor, "attributes", {}).get("role_name", "")).strip()
        if role.lower() in owned:
            continue
        location = actor.get_location()
        foreign.append({
            "id": int(getattr(actor, "id", -1)),
            "type_id": type_id,
            "role_name": role,
            "x": round(float(location.x), 2),
            "y": round(float(location.y), 2),
        })
    return foreign


def _set_spectator_transform(spectator, ego_vehicle):
    transform = ego_vehicle.get_transform()
    mode = str(os.environ.get("OPENCDA_SPECTATOR_VIEW", "planner")).strip().lower()
    if mode in {"topdown", "bird", "birdview"}:
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=70.0),
            carla.Rotation(pitch=-90.0, yaw=float(transform.rotation.yaw)),
        ))
        return
    yaw_rad = math.radians(float(transform.rotation.yaw))
    distance_m = float(os.environ.get("OPENCDA_SPECTATOR_DISTANCE_M", "10.0"))
    height_m = float(os.environ.get("OPENCDA_SPECTATOR_HEIGHT_M", "4.5"))
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(
            x=-distance_m * math.cos(yaw_rad),
            y=-distance_m * math.sin(yaw_rad),
            z=height_m,
        ),
        carla.Rotation(
            pitch=float(os.environ.get("OPENCDA_SPECTATOR_PITCH_DEG", "-15.0")),
            yaw=float(transform.rotation.yaw),
            roll=0.0,
        ),
    ))


def _distance_to_destination(vehicle, destination):
    loc = vehicle.get_location()
    return math.hypot(float(loc.x) - float(destination[0]), float(loc.y) - float(destination[1]))


def _spawn_collision_sensor(world, vehicle, recorder):
    """Attach a CARLA collision sensor and forward events to ``recorder``.

    Collision ground truth belongs to the scenario/evaluator side, not the
    planner: a real vehicle has no CARLA collision sensor to read, so
    CPXMPCPlannerBridge must never depend on one. But a fixture run through
    real CARLA still needs an independent answer to "did the ego actually
    collide" that isn't itself computed by the planner under test.
    """

    blueprint = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(
        blueprint,
        carla.Transform(),
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    def _on_collision(event) -> None:
        other_actor = getattr(event, "other_actor", None)
        type_id = str(getattr(other_actor, "type_id", "unknown"))
        if "vehicle" in type_id:
            actor_type = "vehicle"
        elif "walker" in type_id or "pedestrian" in type_id:
            actor_type = "pedestrian"
        elif "static" in type_id or "prop" in type_id:
            actor_type = "static"
        else:
            actor_type = type_id[:40]

        ts = getattr(event, "timestamp", None)
        sim_time_s = (
            float(getattr(ts, "elapsed_seconds", 0.0)) if ts is not None else None
        )
        try:
            loc = vehicle.get_location()
            vel = vehicle.get_velocity()
            ego_x, ego_y = float(loc.x), float(loc.y)
            ego_speed_mps = float(math.hypot(vel.x, vel.y))
        except Exception:
            ego_x = ego_y = ego_speed_mps = None

        impulse = getattr(event, "normal_impulse", None)
        impulse_mag = (
            float((impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5)
            if impulse is not None
            else None
        )
        recorder.record_collision(
            event_id="%s:%s" % (
                getattr(event, "frame", ""), getattr(other_actor, "id", ""),
            ),
            sim_time_s=sim_time_s,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_speed_mps=ego_speed_mps,
            other_actor_type=actor_type,
            impulse_magnitude=impulse_mag,
        )

    sensor.listen(_on_collision)
    return sensor


def _manager_reached_destination(manager, destination, tolerance_m):
    """Use the active planner's terminal-stop contract when available.

    RouteManager owns progress, but entering its distance threshold is not a
    completed driving task: DestinationSpeedStage must still bring the
    vehicle to rest. CP-X publishes that single terminal fact on its vehicle
    manager. Non-CP-X managers keep the ordinary Euclidean completion rule.
    """

    planner = getattr(manager, "cpx_planner", None)
    if planner is not None:
        return bool(getattr(manager, "_opencda_agent_finished", False))
    return _distance_to_destination(
        manager.vehicle, destination
    ) <= float(tolerance_m)


def _destinations_reached(vehicle_managers, vehicle_configs, tolerance_m):
    """Return true only when every configured CAV reached its own goal."""

    if not vehicle_managers or len(vehicle_managers) != len(vehicle_configs):
        return False
    return all(
        _manager_reached_destination(
            manager, config["destination"], tolerance_m
        )
        for manager, config in zip(vehicle_managers, vehicle_configs)
    )


class _TargetBrakeStimulus(object):
    """Deterministic, state-triggered brake stimulus for prediction A/B runs.

    A fixed tick does not define an interaction: vehicle spawn settling and
    longitudinal control transients move the actual encounter by seconds.
    When trigger conditions are configured, arm once the ego is closing on
    the target inside the requested range.  Fixed ``start_tick`` remains the
    fallback for existing fixtures without trigger conditions.
    """

    def __init__(self, config):
        self.config = dict(config or {})
        self.started_tick = None

    @staticmethod
    def _speed_mps(vehicle):
        velocity = vehicle.get_velocity()
        return math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )

    def _triggered(self, tick, vehicle_managers):
        cfg = self.config
        target_position = cfg.get("trigger_target_position")
        if isinstance(target_position, Mapping):
            target_index = int(cfg.get("target_cav_index", 0))
            if not 0 <= target_index < len(vehicle_managers):
                return False
            axis = str(target_position.get("axis", "x")).strip().lower()
            if axis not in ("x", "y"):
                raise ValueError("trigger_target_position.axis must be x or y")
            location = vehicle_managers[target_index].vehicle.get_location()
            position = float(getattr(location, axis))
            threshold = float(target_position["value"])
            direction = str(
                target_position.get("direction", "increasing")
            ).strip().lower()
            if direction == "increasing":
                crossed = position >= threshold
            elif direction == "decreasing":
                crossed = position <= threshold
            else:
                raise ValueError(
                    "trigger_target_position.direction must be increasing or decreasing"
                )
            return bool(
                int(tick) >= max(0, int(cfg.get("minimum_start_tick", 0)))
                and crossed
            )

        trigger_gap = cfg.get("trigger_gap_m")
        if trigger_gap is None:
            return int(tick) >= max(0, int(cfg.get("start_tick", 0)))

        target_index = int(cfg.get("target_cav_index", 0))
        ego_index = int(cfg.get("ego_cav_index", 1))
        if not (
            0 <= target_index < len(vehicle_managers)
            and 0 <= ego_index < len(vehicle_managers)
            and target_index != ego_index
        ):
            return False
        target = vehicle_managers[target_index].vehicle
        ego = vehicle_managers[ego_index].vehicle
        target_location = target.get_location()
        ego_location = ego.get_location()
        gap_m = math.hypot(
            float(target_location.x) - float(ego_location.x),
            float(target_location.y) - float(ego_location.y),
        )
        ego_speed = self._speed_mps(ego)
        closing_speed = ego_speed - self._speed_mps(target)
        return bool(
            int(tick) >= max(0, int(cfg.get("minimum_start_tick", 0)))
            and gap_m <= float(trigger_gap)
            and ego_speed >= float(cfg.get("minimum_ego_speed_mps", 0.0))
            and closing_speed >= float(cfg.get("minimum_closing_speed_mps", 0.0))
        )

    def apply(self, control, *, tick, cav_index, vehicle_managers):
        cfg = self.config
        if not bool(cfg.get("enabled", False)):
            return control
        if int(cav_index) != int(cfg.get("target_cav_index", 0)):
            return control
        if self.started_tick is None and self._triggered(tick, vehicle_managers):
            self.started_tick = int(tick)
            print("[CP-X stimulus] target brake triggered at tick %d" % int(tick))
        duration = max(0, int(cfg.get("duration_ticks", 0)))
        if self.started_tick is None or not (
            self.started_tick <= int(tick) < self.started_tick + duration
        ):
            return control
        control.throttle = 0.0
        control.brake = max(float(control.brake), float(cfg.get("brake", 0.65)))
        return control


class _LaneClosureStimulus(object):
    """Publish one durable CP lane-closure event from scenario configuration.

    The fixture owns only when the external CP event becomes available. Route
    mapping, de-duplication, topology mutation and rollback remain entirely in
    :class:`CPXRouteManager`, exactly as they will for a real CP transport.
    """

    def __init__(self, config, *, message_path):
        # config is the raw OmegaConf node straight from scenario yaml --
        # dict(config) only converts the top level; a nested list (e.g.
        # message.position) stays an OmegaConf ListConfig, which
        # json.dump (inside write_cp_message_payload, via upsert_cp_item)
        # cannot serialize. Round-tripping through OmegaConf.create/
        # to_container recursively resolves the whole structure to plain
        # dict/list/scalar, and is a no-op for a plain dict (e.g. from a
        # test constructing this directly).
        self.config = OmegaConf.to_container(
            OmegaConf.create(config if config is not None else {}), resolve=True
        )
        self.message_path = str(message_path)
        self.published = False

    def _triggered(self, tick, vehicle_managers):
        trigger = dict(self.config.get("trigger_position", {}) or {})
        if not trigger:
            return int(tick) >= max(0, int(self.config.get("start_tick", 0)))
        cav_index = int(trigger.get("cav_index", 0))
        if not 0 <= cav_index < len(vehicle_managers):
            return False
        axis = str(trigger.get("axis", "x")).strip().lower()
        if axis not in ("x", "y"):
            raise ValueError("cp_lane_closure.trigger_position.axis must be x or y")
        position = float(getattr(vehicle_managers[cav_index].vehicle.get_location(), axis))
        threshold = float(trigger["value"])
        direction = str(trigger.get("direction", "increasing")).strip().lower()
        if direction == "increasing":
            return position >= threshold
        if direction == "decreasing":
            return position <= threshold
        raise ValueError(
            "cp_lane_closure.trigger_position.direction must be increasing or decreasing"
        )

    def publish_if_due(self, *, tick, sim_time_s, vehicle_managers):
        if self.published or not bool(self.config.get("enabled", False)):
            return False
        if not self._triggered(tick, vehicle_managers):
            return False
        message = dict(self.config.get("message", {}) or {})
        message.setdefault("type", "lane_closure")
        message.setdefault("source", "scenario_cp_fixture")
        if not str(message.get("id", "")).strip():
            raise ValueError("cp_lane_closure.message.id is required")
        if str(message.get("type", "")).strip().lower() != "lane_closure":
            raise ValueError("cp_lane_closure.message.type must be lane_closure")
        self.published = bool(upsert_cp_item(
            message_path=self.message_path,
            schema_version=1,
            list_name="lane_events",
            item=message,
            timestamp_s=float(sim_time_s),
        ))
        if self.published:
            print(
                "[CP-X lane closure] published id=%s position=%s ad_lane_id=%s"
                % (
                    str(message.get("id", "")),
                    message.get("position"),
                    message.get("ad_lane_id"),
                )
            )
        return bool(self.published)


def _scenario_manager_kwargs(scenario_params):
    mature_cfg = scenario_params.get("cpx_mature", {})
    map_mode = str(mature_cfg.get("map_mode", "town")).strip().lower()
    if map_mode == "2lane_freeway_simplified":
        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(
            current_path,
            "../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr",
        )
        return {"xodr_path": xodr_path}, map_api.spawn_helper_2lanefree
    town = str(mature_cfg.get("town", "Town06"))
    return {"town": town}, None


def _reset_cooperative_payloads(scenario_params):
    """Start every native scenario with an empty CP transport payload.

    The message file is transport state, not persistent scenario state.  In
    particular, a CP-off ablation must not consume obstacle messages left by
    the preceding CP-on run.  Reset every configured path once before any
    vehicle manager (and therefore any planner reader/provider) is created.
    """

    paths = {str(CP_MESSAGE_PATH)}
    planner_configs = [
        scenario_params.get("vehicle_base", {}).get("planner", {}),
    ]
    planner_configs.extend(
        cav.get("planner", {})
        for cav in scenario_params.get("scenario", {}).get(
            "single_cav_list", []
        )
    )
    for planner_config in planner_configs:
        configured_path = str(
            planner_config.get("cp_message_path", "") or ""
        ).strip()
        if configured_path:
            paths.add(configured_path)
    for message_path in sorted(paths):
        reset_cp_message_payload(message_path=message_path)


def _primary_cp_message_path(scenario_params):
    """Return the transport path consumed by the primary CAV planner."""

    vehicle_base_planner = dict(
        scenario_params.get("vehicle_base", {}).get("planner", {}) or {}
    )
    cavs = list(
        scenario_params.get("scenario", {}).get("single_cav_list", []) or []
    )
    ego_planner = dict(cavs[0].get("planner", {}) or {}) if cavs else {}
    return str(
        ego_planner.get(
            "cp_message_path",
            vehicle_base_planner.get("cp_message_path", CP_MESSAGE_PATH),
        )
        or CP_MESSAGE_PATH
    )


def _configure_synthetic_multimodal_prediction(
    *, scenario_params, single_cav_list, scripted_actor_list,
):
    """Bind a scenario's synthetic predictor to one physical target actor.

    The planner consumes actor ids, while scenario YAML uses stable list
    indices because CARLA assigns actor ids only after spawn.  A target may
    be either another managed CAV or a deterministic scripted road user; the
    prediction pipeline is identical after this one-time binding.
    """

    multimodal_cfg = dict(
        scenario_params.get("cpx_mature", {}).get(
            "synthetic_multimodal_prediction", {}
        ) or {}
    )
    if not bool(multimodal_cfg.get("enabled", False)):
        return None

    ego_index = int(multimodal_cfg.get("ego_cav_index", 0))
    if not 0 <= ego_index < len(single_cav_list):
        raise IndexError("synthetic multimodal ego_cav_index is out of range")

    target_planner = None
    target_scripted_actor = None
    target_label = ""
    if "target_scripted_actor_index" in multimodal_cfg:
        target_index = int(multimodal_cfg["target_scripted_actor_index"])
        if not 0 <= target_index < len(scripted_actor_list):
            raise IndexError(
                "synthetic multimodal target_scripted_actor_index is out of range"
            )
        target_actor = scripted_actor_list[target_index].vehicle
        target_scripted_actor = scripted_actor_list[target_index]
        target_label = "scripted[%d]" % target_index
    else:
        target_index = int(multimodal_cfg.get("target_cav_index", 0))
        if not 0 <= target_index < len(single_cav_list):
            raise IndexError("synthetic multimodal target_cav_index is out of range")
        if target_index == ego_index:
            raise ValueError("synthetic multimodal ego and target must differ")
        target_manager = single_cav_list[target_index]
        target_actor = target_manager.vehicle
        target_planner = getattr(target_manager, "cpx_planner", None)
        target_label = "cav[%d]" % target_index

    ego_planner = getattr(single_cav_list[ego_index], "cpx_planner", None)
    if ego_planner is None:
        raise RuntimeError("synthetic multimodal ego has no CP-X planner")

    mode_override = str(
        multimodal_cfg.get("prediction_mode", "synthetic_multimodal") or ""
    ).strip().lower()
    if mode_override:
        ego_planner._prediction_mode = mode_override
    target_actor_id = int(target_actor.id)
    activation_state = {str(target_actor_id): True}
    if target_scripted_actor is not None:
        activation_state[str(target_actor_id)] = bool(
            target_scripted_actor.prediction_active
        )
    ego_planner._synthetic_prediction_actor_activation = activation_state
    ego_planner.config["synthetic_prediction_actor_ids"] = [target_actor_id]
    ego_planner._prediction_snapshot_transform_cached = False
    ego_planner._prediction_snapshot_transform_fn = None
    if target_planner is not None:
        # The experiment must consume prediction hypotheses instead of the
        # target CAV's exact broadcast MPC solution.
        target_planner._cav_intent_broadcast_enabled = False
    print(
        "[CP-X multimodal] ego cav[%d] predicts %s actor=%d"
        % (ego_index, target_label, target_actor_id)
    )
    return target_scripted_actor, activation_state, str(target_actor_id)


def run_mature_scenario(opt, scenario_params, *, script_name):
    scenario_manager = None
    eval_manager = None
    debug_viewer = None
    single_cav_list = []
    bg_veh_list = []
    scripted_actor_list = []
    collision_sensors = []
    termination_reason = "initialization_failed"
    scenario_exception = ""
    completed_ticks = 0
    completion_mode = "first_cav"
    vehicle_configs = []
    try:
        scenario_params = add_current_time(scenario_params)
        _reset_cooperative_payloads(scenario_params)
        owned_actor_roles = _assign_scenario_actor_roles(
            scenario_params, script_name
        )
        cav_world = CavWorld(opt.apply_ml)
        manager_kwargs, map_helper = _scenario_manager_kwargs(scenario_params)
        scenario_manager = sim_api.ScenarioManager(
            scenario_params,
            opt.apply_ml,
            opt.version,
            cav_world=cav_world,
            **manager_kwargs,
        )
        _destroy_stale_scenario_actors(
            scenario_manager.world, owned_actor_roles
        )
        if opt.record:
            scenario_manager.client.start_recorder("%s.log" % script_name, True)

        single_cav_list = scenario_manager.create_vehicle_manager(
            application=["single"],
            map_helper=map_helper,
        )
        for manager in single_cav_list:
            planner = getattr(manager, "cpx_planner", None)
            if planner is None:
                continue
            if not bool(planner.config.get("record_evaluation_metrics", True)):
                continue
            collision_sensors.append(_spawn_collision_sensor(
                scenario_manager.world, manager.vehicle, planner.evaluation_metrics,
            ))
        if bool(
            scenario_params.get("cpx_mature", {}).get(
                "require_isolated_world", False
            )
        ):
            foreign_actors = _foreign_dynamic_actors(
                scenario_manager.world, owned_actor_roles
            )
            if foreign_actors:
                raise RuntimeError(
                    "isolated CP-X scenario found pre-existing dynamic "
                    "actors; restart CARLA once to clear legacy untagged "
                    "actors: %s" % foreign_actors
                )
        _, bg_veh_list = scenario_manager.create_traffic_carla()
        scripted_actor_list = spawn_scripted_actors(
            scenario_manager.world,
            scenario_params.get("scenario", {}).get("scripted_actors", []),
        )
        synthetic_prediction_runtime = _configure_synthetic_multimodal_prediction(
            scenario_params=scenario_params,
            single_cav_list=single_cav_list,
            scripted_actor_list=scripted_actor_list,
        )

        eval_manager = EvaluationManager(
            scenario_manager.cav_world,
            script_name=str(script_name),
            current_time=scenario_params["current_time"],
        )
        viewer_enabled = bool(
            scenario_params.get("debug_viewer", {}).get("enabled", False)
        )
        if OpenCDADebugViewer.enabled_from_env(viewer_enabled) and single_cav_list:
            debug_viewer = OpenCDADebugViewer(
                world=scenario_manager.world,
                carla_module=carla,
                ego_vehicle=single_cav_list[0].vehicle,
            )

        runtime_cfg = scenario_params.get("cpx_mature", {})
        max_ticks = int(runtime_cfg.get("max_ticks", 1200))
        destination_tolerance_m = float(runtime_cfg.get("destination_tolerance_m", 8.0))
        vehicle_configs = scenario_params["scenario"]["single_cav_list"]
        completion_mode = str(runtime_cfg.get("completion_mode", "first_cav"))
        termination_reason = "max_ticks_reached"
        scripted_brake_cfg = dict(
            runtime_cfg.get("scripted_target_brake", {}) or {}
        )
        target_brake_stimulus = _TargetBrakeStimulus(scripted_brake_cfg)
        lane_closure_stimulus = _LaneClosureStimulus(
            runtime_cfg.get("cp_lane_closure", {}),
            message_path=_primary_cp_message_path(scenario_params),
        )
        fixed_dt_s = float(
            scenario_params.get("world", {}).get("fixed_delta_seconds", 0.05)
        )
        spectator = scenario_manager.world.get_spectator()
        completed_auxiliary_indices = set()
        ego_finished = False
        for tick_index in range(max(1, max_ticks)):
            scenario_manager.tick()
            completed_ticks = int(tick_index + 1)
            ego_vehicle = single_cav_list[0].vehicle
            _ego_loc = ego_vehicle.get_location()
            _ego_xy = (float(_ego_loc.x), float(_ego_loc.y))
            for scripted_actor in scripted_actor_list:
                scripted_actor.step(fixed_dt_s, ego_xy=_ego_xy)
            lane_closure_stimulus.publish_if_due(
                tick=tick_index,
                sim_time_s=float(
                    scenario_manager.world.get_snapshot().timestamp.elapsed_seconds
                ),
                vehicle_managers=single_cav_list,
            )
            if synthetic_prediction_runtime is not None:
                scripted_target, activation_state, actor_id = (
                    synthetic_prediction_runtime
                )
                if scripted_target is not None:
                    activation_state[actor_id] = bool(
                        scripted_target.prediction_active
                    )
            _set_spectator_transform(spectator, ego_vehicle)
            if completion_mode == "all_cavs":
                reached_destination = _destinations_reached(
                    single_cav_list, vehicle_configs, destination_tolerance_m
                )
            else:
                reached_destination = _manager_reached_destination(
                    single_cav_list[0],
                    vehicle_configs[0]["destination"],
                    destination_tolerance_m,
                )
            if reached_destination:
                termination_reason = "route_destination_stopped"
                print("CP-X mature scenario reached the configured destination.")
                break
            for cav_index, single_cav in enumerate(single_cav_list):
                if cav_index in completed_auxiliary_indices:
                    single_cav.vehicle.apply_control(carla.VehicleControl(
                        throttle=0.0, brake=1.0, steer=0.0
                    ))
                    continue
                single_cav.update_info()
                try:
                    control = single_cav.run_step()
                except SystemExit as exc:
                    if int(getattr(exc, "code", 0) or 0) == 0:
                        if cav_index == 0:
                            print("CP-X mature scenario stopped by ego destination condition.")
                            single_cav._opencda_agent_finished = True
                            termination_reason = "route_destination_stopped"
                            ego_finished = True
                            break
                        single_cav._opencda_agent_finished = True
                        completed_auxiliary_indices.add(cav_index)
                        single_cav.vehicle.apply_control(carla.VehicleControl(
                            throttle=0.0, brake=1.0, steer=0.0
                        ))
                        print(
                            "CP-X auxiliary CAV %d reached destination; "
                            "ego evaluation continues." % cav_index
                        )
                        continue
                    raise
                control = target_brake_stimulus.apply(
                    control,
                    tick=tick_index,
                    cav_index=cav_index,
                    vehicle_managers=single_cav_list,
                )
                single_cav.vehicle.apply_control(control)
            if ego_finished:
                break
            if debug_viewer is not None:
                debug_viewer.render(single_cav_list)

    except BaseException as exc:
        termination_reason = "scenario_exception"
        scenario_exception = "%s: %s" % (type(exc).__name__, str(exc))
        raise

    finally:
        if single_cav_list:
            try:
                status_paths = write_run_status(
                    vehicle_managers=single_cav_list,
                    vehicle_configs=vehicle_configs,
                    termination_reason=termination_reason,
                    completed_ticks=completed_ticks,
                    scenario_name=str(script_name),
                    exception=scenario_exception,
                    completion_mode=completion_mode,
                    fallback_debug_output_dir=(
                        "opencda/planning_module/opencda_bridge/debug_cpx_default"
                    ),
                )
                if status_paths:
                    print(
                        "[CP-X scenario] run status written: %s"
                        % ", ".join(status_paths)
                    )
            except Exception as status_exc:
                print("[CP-X scenario] run status write failed: %s" % status_exc)
        if eval_manager is not None and bool(
            scenario_params.get("cpx_mature", {}).get("run_opencda_evaluation", False)
        ):
            eval_manager.evaluate()
        if opt.record and scenario_manager is not None:
            scenario_manager.client.stop_recorder()
        # Actors and their sensors must be destroyed while the scenario's
        # synchronous world is still active.  Restoring world settings first
        # can leave the previous run's ego alive at its destination, where a
        # following A/B run observes it as a real stopped lead vehicle.
        if debug_viewer is not None:
            debug_viewer.destroy()
        for sensor in collision_sensors:
            if bool(getattr(sensor, "is_alive", True)):
                sensor.destroy()
        for vehicle_manager in single_cav_list:
            vehicle_manager.destroy()
        for vehicle in bg_veh_list:
            vehicle.destroy()
        for scripted_actor in scripted_actor_list:
            actor = scripted_actor.vehicle
            if bool(getattr(actor, "is_alive", True)):
                actor.destroy()
        if scenario_manager is not None:
            scenario_manager.close()
