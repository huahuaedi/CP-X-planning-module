# -*- coding: utf-8 -*-
"""
Scenario testing: single vehicle behavior in intersection
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import json
import math
import os

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.planning_module.opencda_bridge.debug_viewer import OpenCDADebugViewer
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time

try:  # optional: prediction-knowledge ablation (scripted crossers + trace record)
    from opencda.scenario_testing.scripted_actor import spawn_scripted_actors
    from opencda.planning_module.pipeline.prediction_ablation import TraceRecorder
except Exception:  # pragma: no cover - keep base scenarios importable
    spawn_scripted_actors = None
    TraceRecorder = None


def _carla_lane_id(carla_map, location):
    try:
        wp = carla_map.get_waypoint(location, project_to_road=True)
        return int(wp.lane_id) if wp is not None else None
    except Exception:
        return None


def _actor_ground_truth(vehicle, carla_map):
    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    return {
        "x": float(tf.location.x),
        "y": float(tf.location.y),
        "v": float(math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)),
        "psi": math.radians(float(tf.rotation.yaw)),
        "lane_id": _carla_lane_id(carla_map, tf.location),
    }


def _spectator_view_mode():
    """Return the requested OpenCDA spectator camera mode."""

    mode = str(os.environ.get("OPENCDA_SPECTATOR_VIEW", "planner")).strip().lower()
    if mode in {"planner", "chase", "follow", "third_person"}:
        return "planner"
    if mode in {"topdown", "bird", "birdview", "opencda"}:
        return "topdown"
    return "planner"


def _set_spectator_transform(spectator, ego_vehicle):
    """Use the CP-X/planning-module style follow view by default."""

    transform = ego_vehicle.get_transform()
    mode = _spectator_view_mode()
    if mode == "topdown":
        spectator.set_transform(carla.Transform(
            transform.location + carla.Location(z=70),
            carla.Rotation(pitch=-90)))
        return

    yaw_rad = math.radians(float(transform.rotation.yaw))
    follow_distance_m = float(os.environ.get("OPENCDA_SPECTATOR_DISTANCE_M", "10.0"))
    follow_height_m = float(os.environ.get("OPENCDA_SPECTATOR_HEIGHT_M", "4.5"))
    location = transform.location + carla.Location(
        x=-follow_distance_m * math.cos(yaw_rad),
        y=-follow_distance_m * math.sin(yaw_rad),
        z=follow_height_m,
    )
    spectator.set_transform(carla.Transform(
        location,
        carla.Rotation(
            pitch=float(os.environ.get("OPENCDA_SPECTATOR_PITCH_DEG", "-15.0")),
            yaw=float(transform.rotation.yaw),
            roll=0.0,
        )
    ))


def _distance_to_destination(ego_vehicle, destination):
    location = ego_vehicle.get_location()
    return math.hypot(
        float(location.x) - float(destination[0]),
        float(location.y) - float(destination[1]),
    )


def run_scenario(opt, scenario_params):
    scenario_manager = None
    eval_manager = None
    debug_viewer = None
    single_cav_list = []
    bg_veh_list = []
    scripted_walkers = []
    termination_reason = "initialization_failed"
    completed_ticks = 0
    destination = None
    scenario_exception = ""
    cav_collision_counts = []
    try:
        scenario_params = add_current_time(scenario_params)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("single_town06_carla.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        cooperative_vru_cfg = (
            scenario_params.get("scenario", {}).get("cooperative_vru", {}) or {}
        )
        if bool(cooperative_vru_cfg.get("enabled", False)):
            walker_blueprint = scenario_manager.world.get_blueprint_library().find(
                str(cooperative_vru_cfg.get(
                    "blueprint",
                    "walker.pedestrian.0001",
                ))
            )
            spawn_position = list(
                cooperative_vru_cfg.get("spawn_position", [-2.0, 80.0, 0.6])
            )
            walker = scenario_manager.world.try_spawn_actor(
                walker_blueprint,
                carla.Transform(carla.Location(
                    x=float(spawn_position[0]),
                    y=float(spawn_position[1]),
                    z=float(spawn_position[2]),
                )),
            )
            if walker is None:
                print("[cooperative VRU] pedestrian spawn failed.")
            else:
                scripted_walkers.append(walker)
                print(
                    "[cooperative VRU] spawned pedestrian actor %s at "
                    "(%.1f, %.1f)."
                    % (
                        str(walker.id),
                        float(spawn_position[0]),
                        float(spawn_position[1]),
                    )
                )

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='single_intersection_town06_carla',
                              current_time=scenario_params['current_time'])
        viewer_enabled = bool(
            scenario_params.get("debug_viewer", {}).get("enabled", False)
        )
        if OpenCDADebugViewer.enabled_from_env(viewer_enabled) and single_cav_list:
            debug_viewer = OpenCDADebugViewer(
                world=scenario_manager.world,
                carla_module=carla,
                ego_vehicle=single_cav_list[0].vehicle,
            )

        spectator = scenario_manager.world.get_spectator()
        scenario_cfg = scenario_params.get("scenario", {})
        destination = scenario_cfg["single_cav_list"][0]["destination"]
        dynamic_reroute_cfg = scenario_cfg.get("dynamic_reroute", {}) or {}
        dynamic_reroute_enabled = bool(dynamic_reroute_cfg.get("enabled", False))
        dynamic_reroute_applied = False
        max_ticks = max(1, int(scenario_cfg.get("max_ticks", 3600)))
        destination_tolerance_m = max(
            0.5,
            float(scenario_cfg.get("destination_tolerance_m", 4.0)),
        )
        termination_reason = "max_ticks_reached"
        cav_collision_counts = [0 for _ in single_cav_list]
        cav_collision_active = [False for _ in single_cav_list]

        # --- prediction-knowledge ablation: scripted crossers + trace record --
        carla_map = scenario_manager.world.get_map()
        fixed_dt_s = float(
            scenario_params.get("world", {}).get("fixed_delta_seconds", 0.05)
        )
        scripted_actor_list = []
        scripted_cfgs = list(scenario_cfg.get("scripted_actors", []) or [])
        if scripted_cfgs and spawn_scripted_actors is not None:
            scripted_actor_list = spawn_scripted_actors(
                scenario_manager.world, scripted_cfgs
            )
        ablation_cfg = scenario_cfg.get("prediction_ablation", {}) or {}
        trace_recorder = None
        if bool(ablation_cfg.get("record", False)) and TraceRecorder is not None:
            trace_path = str(ablation_cfg.get(
                "trace_path",
                "opencda/planning_module/opencda_bridge/oracle_traces/"
                "single_intersection_town06_carla.jsonl",
            ))
            if not os.path.isabs(trace_path):
                trace_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.realpath(__file__)
                    ))),
                    trace_path,
                )
            trace_recorder = TraceRecorder(trace_path)
            print("[prediction ablation] recording ground-truth traces -> %s"
                  % trace_path)

        # run steps
        for tick_index in range(max_ticks):
            completed_ticks = int(tick_index) + 1
            scenario_manager.tick()
            for scripted in scripted_actor_list:
                scripted.step(fixed_dt_s)
            if trace_recorder is not None:
                # Match the bridge clock exactly: it reads sim time from
                # ``world.get_snapshot().timestamp.elapsed_seconds``.
                try:
                    world_time_s = float(
                        scenario_manager.world.get_snapshot()
                        .timestamp.elapsed_seconds
                    )
                except Exception:
                    world_time_s = float(tick_index) * fixed_dt_s
                gt_actors = {}
                for scripted in scripted_actor_list:
                    gt_actors[str(scripted.vehicle.id)] = _actor_ground_truth(
                        scripted.vehicle, carla_map
                    )
                for bg in bg_veh_list:
                    gt_actors[str(bg.id)] = _actor_ground_truth(bg, carla_map)
                trace_recorder.record(
                    sim_time_s=world_time_s,
                    actors=gt_actors,
                )
            for walker in list(scripted_walkers):
                direction = list(
                    cooperative_vru_cfg.get("direction", [1.0, 0.0, 0.0])
                )
                walker.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(
                        x=float(direction[0]),
                        y=float(direction[1]),
                        z=float(direction[2]),
                    ),
                    speed=float(cooperative_vru_cfg.get("speed_mps", 1.4)),
                    jump=False,
                ))
            _set_spectator_transform(spectator, single_cav_list[0].vehicle)

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                safety_queue = getattr(
                    getattr(single_cav, "safety_manager", None),
                    "status_queue",
                    None,
                )
                if safety_queue:
                    try:
                        latest_safety_status = dict(safety_queue[-1][1] or {})
                    except (IndexError, TypeError, ValueError):
                        latest_safety_status = {}
                    collision_now = bool(
                        latest_safety_status.get("collision", False)
                    )
                    if collision_now and not cav_collision_active[i]:
                        cav_collision_counts[i] += 1
                    cav_collision_active[i] = bool(collision_now)
                if (
                    i == 0
                    and dynamic_reroute_enabled
                    and not dynamic_reroute_applied
                ):
                    ego_location = single_cav.vehicle.get_location()
                    trigger_y_below = dynamic_reroute_cfg.get("trigger_y_below")
                    trigger_x_above = dynamic_reroute_cfg.get("trigger_x_above")
                    trigger_reached = (
                        trigger_y_below is not None
                        and float(ego_location.y) <= float(trigger_y_below)
                    )
                    if trigger_x_above is not None:
                        trigger_reached = bool(trigger_reached) and (
                            float(ego_location.x) >= float(trigger_x_above)
                        )
                    if trigger_reached:
                        reroute_destination = list(
                            dynamic_reroute_cfg["destination"]
                        )
                        new_destination = carla.Location(
                            x=float(reroute_destination[0]),
                            y=float(reroute_destination[1]),
                            z=float(reroute_destination[2]),
                        )
                        single_cav.set_destination(
                            ego_location,
                            new_destination,
                            clean=True,
                        )
                        destination = reroute_destination
                        dynamic_reroute_applied = True
                        print(
                            "[CP-X dynamic reroute] applied at "
                            "ego=(%.2f, %.2f), new_destination=(%.2f, %.2f)"
                            % (
                                float(ego_location.x),
                                float(ego_location.y),
                                float(new_destination.x),
                                float(new_destination.y),
                            )
                        )
                if bool(getattr(single_cav, "_opencda_agent_finished", False)):
                    control = carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0,
                        steer=0.0,
                    )
                else:
                    try:
                        control = single_cav.run_step()
                    except SystemExit as exc:
                        # OpenCDA's BehaviorAgent terminates the entire Python
                        # process when any vehicle reaches its destination.
                        # In a multi-CAV scenario, retire only that auxiliary
                        # CAV and let the primary CP-X vehicle continue.
                        if getattr(single_cav, "cpx_planner", None) is not None:
                            raise
                        if int(getattr(exc, "code", 0) or 0) != 0:
                            raise
                        single_cav._opencda_agent_finished = True
                        print(
                            "[multi-CAV] OpenCDA BehaviorAgent vehicle %s "
                            "reached its destination; holding it stopped."
                            % str(getattr(single_cav.vehicle, "id", ""))
                        )
                        control = carla.VehicleControl(
                            throttle=0.0,
                            brake=1.0,
                            steer=0.0,
                        )
                single_cav.vehicle.apply_control(control)

            if (
                bool(scenario_cfg.get("terminate_on_cav_collision", False))
                and any(count > 0 for count in cav_collision_counts)
            ):
                termination_reason = "cav_collision"
                break

            if debug_viewer is not None:
                debug_viewer.render(single_cav_list)
            if bool(
                getattr(single_cav_list[0], "_opencda_agent_finished", False)
            ):
                termination_reason = "route_destination_stopped"
                break
            primary_uses_cpx_planner = bool(
                getattr(single_cav_list[0], "cpx_planner", None) is not None
            )
            if (
                not primary_uses_cpx_planner
                and
                _distance_to_destination(
                    single_cav_list[0].vehicle,
                    destination,
                )
                <= destination_tolerance_m
            ):
                termination_reason = "destination_reached"
                break
        print(
            "[single_intersection_town06_carla] finished: "
            "reason=%s ticks=%d/%d distance_to_destination_m=%.2f"
            % (
                str(termination_reason),
                int(completed_ticks),
                int(max_ticks),
                float(
                    _distance_to_destination(
                        single_cav_list[0].vehicle,
                        destination,
                    )
                ),
            )
        )

    except BaseException as exc:
        termination_reason = "scenario_exception"
        scenario_exception = "%s: %s" % (type(exc).__name__, str(exc))
        raise

    finally:
        try:
            scenario_cfg = scenario_params.get("scenario", {})
            cav_cfgs = list(scenario_cfg.get("single_cav_list", []) or [])
            planner_cfg = (
                cav_cfgs[0].get("planner", {})
                if cav_cfgs
                else {}
            )
            debug_output_dir = str(
                planner_cfg.get(
                    "debug_output_dir",
                    "opencda/planning_module/opencda_bridge/debug_intersection",
                )
            )
            # Mirror cpx_mpc_planner._resolved_debug_output_dir(): the
            # prediction-ablation blind/oracle runs get the mode appended so
            # run_status.json lands next to that run's debug CSV.
            _pred_mode = str(
                scenario_params.get("vehicle_base", {})
                .get("planner", {})
                .get("prediction_mode", "cv")
            ).strip().lower()
            if _pred_mode and _pred_mode != "cv" and not debug_output_dir.endswith(
                "_" + _pred_mode
            ):
                debug_output_dir = debug_output_dir + "_" + _pred_mode
            if not os.path.isabs(debug_output_dir):
                debug_output_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(
                        os.path.realpath(__file__)
                    ))),
                    debug_output_dir,
                )
            os.makedirs(debug_output_dir, exist_ok=True)

            cav_states = []
            for index, vehicle_manager in enumerate(single_cav_list):
                vehicle = getattr(vehicle_manager, "vehicle", None)
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                cav_states.append({
                    "index": int(index),
                    "vehicle_id": int(getattr(vehicle, "id", -1)),
                    "planner": (
                        "cpx_mpc"
                        if getattr(vehicle_manager, "cpx_planner", None) is not None
                        else "opencda_behavior_agent"
                    ),
                    "x_m": float(transform.location.x),
                    "y_m": float(transform.location.y),
                    "speed_mps": float(math.sqrt(
                        velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
                    )),
                    "collision_count": int(
                        cav_collision_counts[index]
                        if index < len(cav_collision_counts)
                        else 0
                    ),
                    "agent_finished": bool(getattr(
                        vehicle_manager,
                        "_opencda_agent_finished",
                        False,
                    )),
                })
            distance_to_destination_m = None
            if single_cav_list and destination is not None:
                distance_to_destination_m = float(_distance_to_destination(
                    single_cav_list[0].vehicle,
                    destination,
                ))
            with open(
                os.path.join(debug_output_dir, "run_status.json"),
                "w",
                encoding="utf-8",
            ) as status_file:
                json.dump({
                    "termination_reason": str(termination_reason),
                    "exception": str(scenario_exception),
                    "completed_ticks": int(completed_ticks),
                    "distance_to_destination_m": distance_to_destination_m,
                    "cav_count": int(len(cav_states)),
                    "cav_states": cav_states,
                }, status_file, indent=2, sort_keys=True)
        except Exception as status_exc:
            print("[OpenCDA run status] write failed: %s" % str(status_exc))

        if eval_manager is not None:
            eval_manager.evaluate()

        if opt.record and scenario_manager is not None:
            scenario_manager.client.stop_recorder()

        if debug_viewer is not None:
            debug_viewer.destroy()

        if scenario_manager is not None:
            scenario_manager.close()

        try:
            _rec = locals().get("trace_recorder")
            if _rec is not None:
                _rec.close()
                print("[prediction ablation] trace rows written: %d"
                      % _rec.rows_written)
        except Exception:
            pass
        for scripted in (locals().get("scripted_actor_list", []) or []):
            try:
                scripted.vehicle.destroy()
            except Exception:
                pass

        for v in single_cav_list:
            v.destroy()
        for v in bg_veh_list:
            v.destroy()
        for walker in scripted_walkers:
            try:
                walker.destroy()
            except Exception:
                pass
