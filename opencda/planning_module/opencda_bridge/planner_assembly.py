"""Assemble a CPXMPCPlannerBridge: every stage, port and manager it owns.

Construction knowledge (which config keys feed which stage, in what order
they are built and wired) lives here so the bridge itself only converts
OpenCDA/ROS input and output.  The assembled objects are still attributes of
the bridge; the callbacks handed to stages are its bound methods.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional
import yaml
from opencda.planning_module.utility.carla_compat import carla
from opencda.planning_module.pipeline.route_context_stage import RouteContextStage
from opencda.planning_module.pipeline.boundary_recovery import BoundaryRecoveryTracker
from opencda.planning_module.pipeline.road_boundary_monitor import RoadBoundaryMonitor
from opencda.planning_module.pipeline.cooperative_arbitration_stage import CooperativeArbitrationStage
from opencda.planning_module.pipeline.traffic_light_memory import TrafficLightMemory
from opencda.planning_module.pipeline.reference_line_provider import ReferenceLineProvider
from opencda.planning_module.pipeline.reference_planning_stage import ReferencePlanningStage
from opencda.planning_module.pipeline.maneuver_manager import ManeuverManager
from opencda.planning_module.pipeline.nominal_trajectory import NominalTrajectoryGenerator
from opencda.planning_module.pipeline.fallback_manager import TrajectoryFallbackManager
from opencda.planning_module.pipeline.speed_planner import SpeedTargetPlanner
from opencda.planning_module.pipeline.destination_speed_stage import DestinationSpeedStage
from opencda.planning_module.pipeline.reference_publication_stage import ReferencePublicationStage
from opencda.planning_module.pipeline.mpc_entry_stage import MPCEntryStage
from opencda.planning_module.pipeline.mpc_cost_profile_stage import MPCCostProfileStage
from opencda.planning_module.pipeline.mpc_execution_stage import MPCExecutionStage
from opencda.planning_module.pipeline.perception_stage import PerceptionStage
from opencda.planning_module.pipeline.execution_pipeline import PlanningPipeline
from opencda.planning_module.pipeline.static_obstacle_stage import StaticObstacleStage
from opencda.planning_module.pipeline.control_safety_stage import ControlSafetyStage
from opencda.planning_module.pipeline.candidate_evaluation import CandidateTrajectoryEvaluator
from opencda.planning_module.pipeline.candidate_selection_stage import CandidateSelectionStage
from opencda.planning_module.pipeline.behavior_stage import BehaviorStage


class _WaypointMapAdapter:
    """Normalize waypoint queries across AD-map and CARLA map providers."""

    def __init__(self, map_planner: Any):
        self._map_planner = map_planner

    def get_waypoint(self, point: Any):
        get_waypoint = getattr(self._map_planner, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        if isinstance(point, Mapping):
            # The AD-map adapter consumes plain point mappings. Try that
            # platform-neutral form before constructing a CARLA Location.
            try:
                return get_waypoint(point)
            except Exception:
                pass
            location = carla.Location(
                x=float(point.get("x", 0.0)),
                y=float(point.get("y", 0.0)),
                z=float(point.get("z", 0.0)),
            )
            try:
                return get_waypoint(location)
            except Exception:
                return None
        try:
            return get_waypoint(point)
        except Exception:
            return None


def ensure_planning_module_import_path() -> None:
    """Expose planning_module-local imports used by legacy MPC modules.

    The standalone planning runner is usually launched from
    ``opencda/planning_module``, so imports like ``from utility...`` work.
    Native OpenCDA scenarios are launched from the repository root, where
    that directory is not on ``sys.path``.  Add it only when the bridge is
    constructed so the default OpenCDA path stays untouched.
    """

    planning_module_root = str(Path(__file__).resolve().parents[1])
    if planning_module_root not in sys.path:
        sys.path.insert(0, planning_module_root)


def load_mpc_config(config) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg_path = config.get("mpc_config_path")
    if not cfg_path:
        cfg_path = Path(__file__).resolve().parents[1] / "MPC" / "mpc.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    mpc_cfg = dict(payload.get("mpc", payload))
    road_cfg = dict(payload.get("road", {}))
    road_cfg.setdefault("lane_count", int(config.get("lane_count", 3)))
    road_cfg.setdefault("lane_width_m", float(config.get("lane_width_m", 3.5)))
    # "One switch": cav_conflict_enabled is documented (see __init__)
    # as also needing cost.corridor.enabled in mpc.yaml for the Stage-D
    # corridor to actually bind in the QP -- but nothing wired that
    # second half up, so cav_conflict_enabled=true alone still leaves
    # classify_conflicts/resolve_conflicts running with nowhere to
    # place their output. Confirmed via MDrive's Intersection_Deadlock_
    # Resolution/3: with the corridor left at mpc.yaml's off-by-default
    # and no per-agent yield/crossing mechanism outside it, a left-
    # turning ego crossed a straight-through ego's path with no
    # arbitration and collided. Only forces this on when the caller
    # opted into cav_conflict_enabled -- every existing config that
    # never sets it keeps mpc.yaml's own corridor setting untouched.
    if bool(config.get("cav_conflict_enabled", False)):
        cost_cfg = dict(mpc_cfg.get("cost", {}))
        corridor_cfg = dict(cost_cfg.get("corridor", {}))
        corridor_cfg["enabled"] = True
        cost_cfg["corridor"] = corridor_cfg
        mpc_cfg["cost"] = cost_cfg
    return mpc_cfg, road_cfg


def resolve_global_planner_xodr_path(config, map_planner) -> Path:
    raw_path = str(
        config.get(
            "global_planner_xodr_path",
            config.get("xodr_path", ""),
        )
        or ""
    ).strip()
    planning_module_root = Path(__file__).resolve().parents[1]
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = planning_module_root / path
        if path.exists():
            return path
        raise FileNotFoundError(f"Global planner xodr_path not found: {path}")

    map_name = str(config.get("global_planner_map_name", "") or "").strip()
    if not map_name:
        try:
            map_name = str(map_planner.name).split("/")[-1]
        except Exception:
            map_name = ""
    if not map_name:
        map_name = "Town06"
    candidates = []
    if map_name.endswith(".xodr"):
        candidates.append(planning_module_root / "Global_Planner" / "maps" / map_name)
    else:
        candidates.extend([
            planning_module_root / "Global_Planner" / "maps" / f"{map_name}.xodr",
            planning_module_root / "Global_Planner" / "maps" / f"{map_name}_Opt.xodr",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve custom global planner .xodr path; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def assemble_planner(
    bridge: Any,
    vehicle_manager: Any,
    config: Optional[Mapping[str, Any]],
    map_planner: Any,
) -> None:
    """Build every stage, port and manager of ``bridge`` in dependency order.

    Three phases, each seeing the construction-only values (``parts``) the
    earlier ones produced.  Nothing here runs after construction; the
    runtime-visible objects are set as attributes of ``bridge``.
    """

    ensure_planning_module_import_path()
    parts = _core_config_and_behavior_stages(
        bridge, vehicle_manager, config, map_planner,
    )
    _mpc_and_reference_generation(bridge, parts)
    _pipeline_route_and_finalization(bridge, parts)


def _core_config_and_behavior_stages(
    bridge, vehicle_manager: Any,
    config: Optional[Mapping[str, Any]], map_planner: Any,
) -> SimpleNamespace:
    """Phase 1. Returns the construction-only values later phases consume."""

    parts = SimpleNamespace()
    bridge.vehicle_manager = vehicle_manager
    from opencda.planning_module.pipeline.architecture_profile import (
        normalize_architecture_config,
    )

    bridge.config, bridge.architecture_profile = normalize_architecture_config(config)
    bridge.carla = carla
    bridge.map_planner = map_planner or getattr(vehicle_manager, "carla_map", None)
    bridge.waypoint_map_planner = bridge.map_planner
    bridge.enabled = bool(bridge.config.get("enabled", True))
    bridge.mode = str(bridge.config.get("mode", "full_cpx_mpc")).strip().lower()
    bridge.fallback_policy = str(
        bridge.config.get("fallback_policy", "emergency_stop")
    ).strip().lower()
    bridge.fallback_policy_warning = ""
    if bridge.fallback_policy == "opencda":
        bridge.fallback_policy = "emergency_stop"
        bridge.fallback_policy_warning = "opencda_fallback_disabled_in_full_cpx_mpc"
    bridge.use_opencda_global_route = bool(
        bridge.config.get("use_opencda_global_route", True)
    )
    bridge.opencda_global_route_reference_allowed = bool(
        bridge.config.get("opencda_global_route_reference_allowed", True)
    )
    bridge.target_speed_mps = float(bridge.config.get("target_speed_mps", 8.0))
    bridge.lookahead_m = float(bridge.config.get("lookahead_m", 18.0))
    bridge.min_front_gap_m = float(bridge.config.get("min_front_gap_m", 8.0))
    # min_front_gap_m alone is a flat distance that doesn't scale with
    # cruise speed: at 11.18 m/s the default 8.0m gave several
    # seconds of reaction margin before target_lane_prediction_risk
    # would trip, but at 20 m/s the same 8.0m is covered in half the
    # time -- confirmed via telemetry (Interactive_Lane_Change's
    # queued lane change missed again at 20 m/s cruise, tripping this
    # exact check, after being fixed at 11.18 m/s). min_front_gap_time_s
    # defaults to 8.0/11.18 so today's calibrated distance is
    # reproduced exactly at 11.18 m/s, while the effective floor grows
    # proportionally with whatever cruise speed is configured.
    bridge.min_front_gap_time_s = float(
        bridge.config.get("min_front_gap_time_s", 8.0 / 11.18)
    )
    parts.max_mpc_obstacles = max(0, int(bridge.config.get("max_mpc_obstacles", 4)))
    # Scenario-only isolation switch.  Map topology, route/reference
    # generation, road boundaries and MPC remain active; dynamic actor
    # observations are withheld from behavior, prediction and MPC so the
    # geometric planning chain can be tested deterministically.
    bridge.functional_test_ignore_dynamic_objects = bool(
        bridge.config.get("functional_test_ignore_dynamic_objects", False)
    )
    bridge.debug = bool(bridge.config.get("debug", True))
    bridge.last_debug: dict[str, Any] = {}
    bridge._last_accel_mps2 = 0.0
    bridge._last_steer_rad = 0.0
    # Multi-CAV interaction pipeline (classify -> assign -> corridor).
    # Off unless cav_conflict_enabled; also needs cost.corridor.enabled in
    # mpc.yaml for the corridor to bind in the QP.
    bridge._cav_conflict_enabled = bool(
        bridge.config.get("cav_conflict_enabled", False)
    )
    bridge._cav_transport_diagnostics: dict[str, Any] = {}
    # This CAV's own broadcast for nearby CP-X CAVs to read (its planned
    # trajectory + ResourceClaim + pose). Read peer-to-peer through
    # v2x_manager.cav_nearby; see _publish_cav_intent / _collect_cav_intents.
    bridge.last_cav_intent_payload = None
    bridge._cav_intent_sequence = 0
    bridge._cav_intent_broadcast_enabled = bool(
        bridge.config.get("cav_intent_broadcast_enabled", True)
    )
    bridge._warned = False
    bridge._route_context = RouteContextStage()
    bridge._stable_reference_line_provider = ReferenceLineProvider()
    bridge.maneuver_manager = ManeuverManager(bridge.config)
    bridge.nominal_trajectory_generator = NominalTrajectoryGenerator()
    speed_target_planner = SpeedTargetPlanner()
    behavior_stage = BehaviorStage()
    bridge._stop_release_temp_smooth_until_sim_time_s = 0.0
    bridge._full_signal_actor_id = ""
    bridge._full_traffic_memory = TrafficLightMemory(
        hold_unknown_s=float(bridge.config.get("full_traffic_unknown_hold_s", 1.5)),
        hold_green_unknown_s=float(
            bridge.config.get("full_traffic_green_unknown_hold_s", 0.25)
        ),
        green_confirm_s=float(bridge.config.get("full_traffic_green_confirm_s", 0.15)),
        hold_stop_unknown_until_green=bool(
            bridge.config.get(
                "full_traffic_hold_stop_unknown_until_green",
                False,
            )
        ),
        fail_safe_max_unknown_hold_s=float(
            bridge.config.get("full_traffic_fail_safe_max_unknown_hold_s", 6.0)
        ),
    )
    from opencda.planning_module.pipeline.scenario_manager import (
        CPXScenarioManager,
    )

    scenario_manager = CPXScenarioManager(bridge.config)
    bridge._boundary_recovery = BoundaryRecoveryTracker(bridge.config)
    fallback_manager = TrajectoryFallbackManager(
        max_hold_age_s=float(bridge.config.get("fallback_hold_last_valid_s", 0.35)),
        min_hold_arc_m=float(bridge.config.get("fallback_min_valid_arc_m", 2.0)),
        safe_stop_deceleration_mps2=float(
            bridge.config.get("fallback_safe_stop_deceleration_mps2", 2.0)
        ),
    )
    destination_speed_stage = DestinationSpeedStage(
        config=bridge.config,
        speed_planner=speed_target_planner,
        fallback_manager=fallback_manager,
        behavior_stage=behavior_stage,
    )
    from opencda.planning_module.pipeline.behavior_reference_execution_stage import (
        BehaviorReferenceExecutionStage,
    )
    behavior_reference_execution_stage = BehaviorReferenceExecutionStage(
        reference_provider=bridge._stable_reference_line_provider,
        fallback_manager=fallback_manager,
        behavior_stage=behavior_stage,
        lane_id_at_location=bridge._lane_id_at_location,
    )
    bridge._route_replan_last_attempt_s = -float("inf")
    bridge._route_replan_attempt_count = 0
    bridge._route_replan_last_reason = "route_replan_not_requested"
    bridge._static_obstacle_blocked_lane_id: object = ""
    bridge.draw_world_debug = bool(bridge.config.get("draw_world_debug", False))
    bridge.draw_world_debug_destination = bool(
        bridge.config.get("draw_world_debug_destination", False)
    )
    bridge.world_debug_life_time_s = float(bridge.config.get("world_debug_life_time_s", 0.15))
    parts.full_control_buffer_min_speed_mps = max(
        0.0,
        float(bridge.config.get("full_control_buffer_min_speed_mps", 1.5)),
    )
    bridge.full_lane_change_start_lock_s = max(
        0.0,
        float(bridge.config.get("full_lane_change_start_lock_s", 8.0)),
    )
    bridge.full_dense_traffic_lane_change_lock_enabled = bool(
        bridge.config.get("full_dense_traffic_lane_change_lock_enabled", True)
    )
    bridge.full_dense_traffic_object_count = max(
        0,
        int(bridge.config.get("full_dense_traffic_object_count", 8)),
    )
    bridge.full_dense_traffic_risky_lane_count = max(
        0,
        int(bridge.config.get("full_dense_traffic_risky_lane_count", 2)),
    )
    bridge.full_prepare_lane_change_reference_lock = bool(
        bridge.config.get("full_prepare_lane_change_reference_lock", True)
    )
    bridge.full_allow_opportunistic_lane_change = bool(
        bridge.config.get("full_allow_opportunistic_lane_change", False)
    )
    bridge.full_mpc_reference_stabilizer_enabled = bool(
        bridge.config.get("full_mpc_reference_stabilizer_enabled", True)
    )
    bridge.full_candidate_pipeline_enabled = bool(
        bridge.config.get("full_candidate_pipeline_enabled", True)
    )
    parts.full_candidate_reference_min_object_distance_m = max(
        0.0,
        float(bridge.config.get("full_candidate_reference_min_object_distance_m", 2.0)),
    )
    # The generic clearance above (default 2.0m, configured to 3.5m here)
    # sizes lateral gaps for negotiating with *moving* traffic. Applied
    # unmodified to a static-obstacle local-avoidance candidate it is
    # bridge-defeating: the whole point of that candidate is to pass close
    # to the very obstacle it is routing around, in a lane only ~3.5m
    # wide, so it always scores infeasible and the vehicle never moves
    # (confirmed via decision_veto_chain: all three lane-change variants
    # rejected on candidate_prediction_collision_risk ~1.1-1.3m, the
    # ego's own predicted clearance from the blocking obstacle, static
    # across assertive/normal/conservative timing since the obstacle
    # isn't moving). Use a tighter, still-conservative clearance just for
    # the candidate whose target lane matches the selected local-
    # avoidance lane; every other candidate keeps the full margin above.
    parts.static_obstacle_local_avoidance_min_object_distance_m = max(
        0.0,
        float(
            bridge.config.get(
                "static_obstacle_local_avoidance_min_object_distance_m",
                0.8,
            )
        ),
    )
    # Two same-lane candidates (e.g. full-speed "keep_lane" vs slowed
    # "yield_slow_down") build references at different speeds/extents, so
    # their own predicted-obstacle-distance estimates can differ by more
    # than this margin purely from that shape difference, not real
    # obstacle motion -- flipping which discrete risk bucket (and thus
    # which candidate) wins every other tick and reading to MPC as a
    # discontinuous reference. Keyed by candidate name so each logical
    # candidate slot keeps its own hysteresis state across ticks.
    parts.candidate_risk_hysteresis_margin_m = max(
        0.0,
        float(bridge.config.get("candidate_risk_hysteresis_margin_m", 1.5)),
    )
    parts.candidate_mpc_probe_enabled = bool(
        bridge.config.get("candidate_mpc_probe_enabled", True)
    )
    parts.candidate_mpc_probe_top_k = max(
        2,
        int(bridge.config.get("candidate_mpc_probe_top_k", 2)),
    )
    parts.candidate_mpc_probe_interval_s = max(
        0.05,
        float(bridge.config.get("candidate_mpc_probe_interval_s", 0.2)),
    )
    parts.candidate_trajectory_evaluator = CandidateTrajectoryEvaluator(
        mpc_probe_enabled=bool(parts.candidate_mpc_probe_enabled),
        mpc_probe_top_k=int(parts.candidate_mpc_probe_top_k),
        mpc_probe_interval_s=float(parts.candidate_mpc_probe_interval_s),
    )
    parts.strict_decision_ownership_enabled = bool(
        bridge.config.get("strict_decision_ownership_enabled", True)
    )
    bridge.strict_reference_validator_veto_enabled = bool(
        bridge.config.get("strict_reference_validator_veto_enabled", True)
    )
    parts.strict_explicit_fallback_speed_mps = max(
        0.0,
        float(bridge.config.get("strict_explicit_fallback_speed_mps", 0.8)),
    )
    parts.full_reference_stabilizer_min_forward_m = float(
        bridge.config.get("full_reference_stabilizer_min_forward_m", -0.25)
    )
    bridge.full_reference_stabilizer_min_spacing_m = max(
        0.0,
        float(bridge.config.get("full_reference_stabilizer_min_spacing_m", 0.35)),
    )
    parts.full_reference_stabilizer_max_heading_step_rad = max(
        0.0,
        float(bridge.config.get("full_reference_stabilizer_max_heading_step_rad", 0.75)),
    )
    bridge._debug_writer = None
    bridge._debug_csv_file = None
    bridge._debug_jsonl_file = None
    # CSV columns are derived from the typed diagnostics payload on the
    # first recorded frame; the bridge does not own a duplicate schema.

    parts.speed_target_planner = speed_target_planner
    parts.behavior_stage = behavior_stage
    parts.scenario_manager = scenario_manager
    parts.fallback_manager = fallback_manager
    parts.destination_speed_stage = destination_speed_stage
    parts.behavior_reference_execution_stage = behavior_reference_execution_stage
    return parts


def _mpc_and_reference_generation(bridge, parts: SimpleNamespace) -> None:
    ensure_planning_module_import_path()
    from opencda.planning_module.MPC.mpc import MPC
    from opencda.planning_module.behavior_planner import LaneSafetyScorer, RuleBasedBehaviorPlanner
    from opencda.planning_module.opencda_bridge.cp_provider import OpenCDACPProvider
    from opencda.planning_module.opencda_bridge.planner_input_adapter import OpenCDAPlanningAdapter
    from opencda.planning_module.pipeline.control_buffer import MPCControlBuffer
    from opencda.planning_module.pipeline.mpc_feedback import BehaviorMPCFeedback
    from opencda.planning_module.pipeline.mpc_command_extractor import (
        MPCCommandExtractor,
    )
    from opencda.planning_module.pipeline.route_manager import CPXRouteManager
    from opencda.planning_module.pipeline.reference_gate import FinalReferenceGate
    from opencda.planning_module.pipeline.reference_generator import ReferenceGenerator
    from opencda.planning_module.pipeline.reference_pipeline import (
        ReferencePipeline,
    )
    from opencda.planning_module.pipeline.safety_supervisor import SafetySupervisor
    from opencda.planning_module.pipeline.runtime_input_stage import RuntimeInputStage
    from opencda.planning_module.pipeline.velocity_steering_adapter import (
        OpenCDAVelocitySteeringAdapter,
    )
    from opencda.planning_module.pipeline.tracker import CPXObstacleTracker
    from opencda.planning_module.utility.evaluation_metrics import (
        EvaluationMetricsRecorder,
        write_planning_metrics_artifacts,
    )
    from opencda.planning_module.utility.global_planner import CustomGlobalPlannerAdapter

    mpc_cfg, road_cfg = load_mpc_config(bridge.config)
    bridge.mpc = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)
    # Vehicle geometry has one configuration owner.  Reference-contract
    # footprint checks and MPC lane/collision constraints must describe
    # the same body; mpc.yaml intentionally contains controller defaults,
    # while the platform-specific dimensions live in the bridge profile.
    bridge.mpc.ego_width_m = 2.0 * max(
        0.0, float(bridge.config.get("reference_vehicle_half_width_m", 1.0))
    )
    bridge.mpc.ego_length_m = 2.0 * max(
        0.0, float(bridge.config.get("reference_vehicle_half_length_m", 2.4))
    )
    # road_boundary_margin_m was baked in at MPC construction time from
    # mpc.yaml's ego_width_m default (0.0); recompute it now that the
    # bridge's real vehicle width is set, or the non-corridor road-
    # boundary constraint silently under-estimates the ego footprint.
    bridge.mpc.refresh_ego_footprint_margins()
    if bridge._cav_conflict_enabled:
        # One switch: cav_conflict_enabled also binds the Stage-C corridor
        # in the QP (else it is computed but ignored). mpc.yaml's
        # cost.corridor block still tunes w_slack / max_slack_m.
        bridge.mpc.corridor_constraint_enabled = True
    from opencda.planning_module.pipeline.actuator_mapper import CarlaActuatorMapper
    from opencda.planning_module.opencda_bridge.platform_ports import (
        ActuatorPort,
        MapLookupPort,
        VehicleDynamics,
    )
    fallback_actuator_steer_rad = float(bridge.config.get(
        "platform_actuator_max_steer_rad",
        math.radians(70.0),
    ))
    bridge.vehicle_dynamics = VehicleDynamics.from_carla_vehicle(
        bridge.vehicle_manager.vehicle,
        fallback_wheelbase_m=float(bridge.config.get(
            "platform_wheelbase_m", bridge.mpc.wheelbase_m
        )),
        fallback_actuator_max_steer_rad=fallback_actuator_steer_rad,
    )
    # MPC states use the physical axle geometry. Planning steering bounds
    # remain independent from the actuator's normalized full scale.
    bridge.mpc.wheelbase_m = float(bridge.vehicle_dynamics.wheelbase_m)
    bridge.mpc.l_r_m = 0.5 * float(bridge.vehicle_dynamics.wheelbase_m)
    bridge.actuator_mapper = CarlaActuatorMapper(bridge.config)
    bridge.actuator_port = ActuatorPort(
        actuator_mapper=bridge.actuator_mapper,
        constraints=bridge.mpc.constraints,
        carla_module=carla,
        clock=bridge._sim_time_s,
        actuator_max_steer_rad=(
            bridge.vehicle_dynamics.actuator_max_steer_rad
        ),
    )
    runtime_input_stage = RuntimeInputStage(bridge.actuator_mapper)
    vehicle_curvature_margin = min(
        1.0,
        max(
            0.1,
            float(
                bridge.config.get(
                    "reference_vehicle_curvature_safety_factor",
                    0.90,
                )
            ),
        ),
    )
    vehicle_max_curvature_1pm = bridge.mpc.maximum_path_curvature_1pm()
    bridge.config["reference_vehicle_max_curvature_1pm"] = (
        float(vehicle_curvature_margin)
        * float(vehicle_max_curvature_1pm)
    )
    bridge.reference_generator = ReferenceGenerator(
        config=bridge.config,
        mpc=bridge.mpc,
        map_planner=bridge.map_planner,
        map_waypoint_from_location=bridge._map_waypoint_from_location,
        lane_id_at_location=bridge._lane_id_at_location,
        body_frame_xy=bridge._body_frame_xy,
        target_speed_mps=float(bridge.target_speed_mps),
        lookahead_m=float(bridge.lookahead_m),
        drivable_waypoint_from_location=(
            bridge._drivable_waypoint_from_location
        ),
    )
    bridge.behavior_runtime_cfg = dict(mpc_cfg.get("behavior_planner_runtime", {}))
    bridge.lane_safety_scorer = LaneSafetyScorer()
    bridge.reference_map = _WaypointMapAdapter(bridge.map_planner)
    bridge.map_lookup_port = MapLookupPort(
        reference_map=bridge.reference_map,
        waypoint_map=bridge.waypoint_map_planner,
        carla_module=carla,
    )
    bridge.input_adapter = OpenCDAPlanningAdapter(bridge)
    bridge.tracker = CPXObstacleTracker(
        max_stale_s=float(bridge.config.get("tracker_max_stale_s", 0.5)),
        max_speed_mps=float(bridge.config.get("tracker_max_speed_mps", 45.0)),
        max_acceleration_mps2=float(
            bridge.config.get("tracker_max_acceleration_mps2", 12.0)
        ),
        max_position_jump_m=float(bridge.config.get("tracker_max_position_jump_m", 12.0)),
    )
    # Prediction-knowledge ablation (cv | blind | oracle). Built lazily on
    # first use so mpc.horizon_s / dt_s are settled; see
    # ``prediction_snapshot_transform``.
    bridge._prediction_mode = str(
        bridge.config.get("prediction_mode", "cv")
    ).strip().lower()
    bridge._prediction_snapshot_transform_cached = False
    bridge._prediction_snapshot_transform_fn = None
    bridge._synthetic_prediction_actor_activation = None
    bridge._oracle_trace_store = None
    if bridge._prediction_mode == "oracle":
        from opencda.planning_module.pipeline.prediction_ablation import (
            OracleTraceStore,
        )
        oracle_path = str(bridge.config.get("oracle_trace_path", "") or "")
        if not oracle_path:
            raise ValueError(
                "prediction_mode='oracle' requires config 'oracle_trace_path'"
            )
        if not os.path.isabs(oracle_path):
            # OpenCDA runs from the repo root; resolve against it so the
            # path matches what the scenario TraceRecorder wrote.
            repo_root = Path(__file__).resolve().parents[3]
            cand = repo_root / oracle_path
            oracle_path = str(cand if cand.is_file() else oracle_path)
        if not os.path.isfile(oracle_path):
            raise FileNotFoundError(
                "oracle_trace_path not found: %s -- run the recording "
                "scenario first (prediction_mode: cv, "
                "prediction_ablation.record: true)" % oracle_path
            )
        bridge._oracle_trace_store = OracleTraceStore.from_file(oracle_path)
    ego_extent = getattr(
        getattr(bridge.vehicle_manager.vehicle, "bounding_box", None),
        "extent", None,
    )
    perception_stage = PerceptionStage(
        obstacle_tracker=bridge.tracker,
        max_mpc_obstacles=int(parts.max_mpc_obstacles),
        ego_length_m=2.0 * float(getattr(ego_extent, "x", 2.25)),
        ego_width_m=2.0 * float(getattr(ego_extent, "y", 1.0)),
        lane_width_m=float(getattr(bridge.mpc, "lane_width_m", 3.5)),
        lane_change_boundary_overlap_m=float(
            bridge.config.get("lane_change_boundary_overlap_m", 0.75)
        ),
    )
    bridge.final_reference_gate = FinalReferenceGate(bridge.config)
    bridge.reference_pipeline = ReferencePipeline(
        config=bridge.config,
        generator=bridge.reference_generator,
        final_gate=bridge.final_reference_gate,
        horizon_steps=int(bridge.mpc.horizon_steps),
        dt_s=float(bridge.mpc.dt_s),
        default_speed_mps=float(bridge.target_speed_mps),
    )
    reference_publication_stage = ReferencePublicationStage(
        reference_pipeline=bridge.reference_pipeline,
        reference_provider=bridge._stable_reference_line_provider,
        config=bridge.config,
    )

    parts.runtime_input_stage = runtime_input_stage
    parts.perception_stage = perception_stage
    parts.reference_publication_stage = reference_publication_stage


def _pipeline_route_and_finalization(bridge, parts: SimpleNamespace) -> None:
    from opencda.planning_module.behavior_planner import RuleBasedBehaviorPlanner
    from opencda.planning_module.opencda_bridge.cp_provider import OpenCDACPProvider
    from opencda.planning_module.opencda_bridge.platform_ports import MapLookupPort
    from opencda.planning_module.pipeline.control_buffer import MPCControlBuffer
    from opencda.planning_module.pipeline.mpc_feedback import BehaviorMPCFeedback
    from opencda.planning_module.pipeline.mpc_command_extractor import (
        MPCCommandExtractor,
    )
    from opencda.planning_module.pipeline.route_manager import CPXRouteManager
    from opencda.planning_module.pipeline.safety_supervisor import SafetySupervisor
    from opencda.planning_module.pipeline.velocity_steering_adapter import (
        OpenCDAVelocitySteeringAdapter,
    )
    from opencda.planning_module.utility.evaluation_metrics import (
        EvaluationMetricsRecorder,
        write_planning_metrics_artifacts,
    )
    from opencda.planning_module.utility.global_planner import CustomGlobalPlannerAdapter

    mpc_entry_stage = MPCEntryStage(bridge.config)
    static_obstacle_stage = StaticObstacleStage({
        **bridge.behavior_runtime_cfg,
        **bridge.config,
    })
    bridge.safety_supervisor = SafetySupervisor(
        enabled=bool(bridge.config.get("safety_supervisor_enabled", True)),
        max_steer_delta=float(bridge.config.get("safety_max_steer_delta", 0.25)),
        max_throttle_delta=float(bridge.config.get("safety_max_throttle_delta", 0.45)),
        max_brake_delta=float(bridge.config.get("safety_max_brake_delta", 0.60)),
        stuck_release_min_accel_mps2=float(
            bridge.config.get("safety_stuck_release_min_accel_mps2", 0.01)
        ),
    )
    control_safety_stage = ControlSafetyStage(
        supervisor=bridge.safety_supervisor, config=bridge.config
    )
    candidate_selection_stage = CandidateSelectionStage(
        evaluator=parts.candidate_trajectory_evaluator,
        provider=bridge._stable_reference_line_provider,
        maneuver_manager=bridge.maneuver_manager,
        reference_pipeline=bridge.reference_pipeline,
        fallback_manager=parts.fallback_manager,
        static_obstacle_stage=static_obstacle_stage,
        mpc=bridge.mpc,
        config=bridge.config,
        map_epoch="admap",
        normal_clearance_m=parts.full_candidate_reference_min_object_distance_m,
        static_clearance_m=parts.static_obstacle_local_avoidance_min_object_distance_m,
        risk_hysteresis_margin_m=parts.candidate_risk_hysteresis_margin_m,
        strict_ownership=parts.strict_decision_ownership_enabled,
        target_speed_mps=bridge.target_speed_mps,
    )
    reference_planning_stage = ReferencePlanningStage(
        provider=bridge._stable_reference_line_provider,
        nominal_trajectory_generator=bridge.nominal_trajectory_generator,
        candidate_selection=candidate_selection_stage,
    )
    mpc_cost_profile_stage = MPCCostProfileStage(
        mpc=bridge.mpc,
        config=bridge.config,
        behavior_runtime_config=bridge.behavior_runtime_cfg,
    )
    bridge.pipeline = PlanningPipeline(
        runtime_input=parts.runtime_input_stage,
        perception=parts.perception_stage,
        behavior=parts.behavior_stage,
        scenario=parts.scenario_manager,
        static_obstacle=static_obstacle_stage,
        control_safety=control_safety_stage,
        speed=parts.speed_target_planner,
        destination_speed=parts.destination_speed_stage,
        reference_publication=parts.reference_publication_stage,
        mpc_entry=mpc_entry_stage,
        fallback=parts.fallback_manager,
        behavior_reference_execution=parts.behavior_reference_execution_stage,
        reference_planning=reference_planning_stage,
        mpc_cost_profile=mpc_cost_profile_stage,
    )
    # Construction-only scratch state threaded from the earlier
    # _init_* phases; nothing outside __init__ may depend on it.

    bridge.velocity_steering_adapter = OpenCDAVelocitySteeringAdapter(
        bridge.vehicle_manager.controller,
        actuator_max_steer_rad=(
            bridge.vehicle_dynamics.actuator_max_steer_rad
        ),
        speed_deadband_mps=float(bridge.actuator_mapper.speed_deadband_mps),
    )
    bridge.mpc_command_extractor = MPCCommandExtractor(
        preview_time_s=float(
            bridge.config.get("mpc_velocity_command_preview_time_s", 0.6)
        ),
        velocity_source=str(
            bridge.config.get("mpc_velocity_command_source", "preview")
        ),
        min_acceleration_mps2=float(
            bridge.mpc.constraints.min_acceleration_mps2
        ),
        max_acceleration_mps2=float(
            bridge.mpc.constraints.max_acceleration_mps2
        ),
    )
    route_sample_distance_m = float(bridge.config.get("route_sample_distance_m", 1.0))
    bridge.global_planner_backend = "custom_admap_dijkstra"
    bridge.global_planner_backend_warning = ""
    road_cfg_from_map = {"lane_count": 1, "lane_width_m": 3.5}
    global_planner_mode = str(
        bridge.config.get("global_planner_mode", "dij")
    ).strip().lower()
    if global_planner_mode not in {
        "custom", "custom_admap", "admap", "opendrive", "dijkstra", "dij"
    }:
        raise ValueError("global_planner_mode must select the AD-map backend")
    xodr_path = resolve_global_planner_xodr_path(bridge.config, bridge.map_planner)
    bridge.global_planner = CustomGlobalPlannerAdapter(
        xodr_path=str(xodr_path),
        cache_root=str(
            bridge.config.get(
                "global_planner_cache_root",
                Path(__file__).resolve().parents[1] / "Global_Planner" / "cache",
            )
        ),
        route_sample_distance_m=float(route_sample_distance_m),
        lane_change_penalty_m=(
            float(bridge.config["global_planner_lane_change_penalty_m"])
            if bridge.config.get("global_planner_lane_change_penalty_m") is not None
            else None
        ),
        ad_map_install_root=bridge.config.get("ad_map_install_root"),
    )
    bridge.global_planner.load(
        force_rebuild=bool(bridge.config.get("global_planner_force_rebuild", False))
    )
    print(
        "[CP-X OpenCDA Bridge] Using AD-map Dijkstra global planner: "
        f"{xodr_path}"
    )
    bridge.topology_map = bridge.global_planner
    # CARLA remains available to the simulation adapter, but every route,
    # lane and reference waypoint query is owned by AD-map.
    bridge.waypoint_map_planner = bridge.topology_map
    bridge.reference_map = _WaypointMapAdapter(bridge.waypoint_map_planner)
    bridge.map_lookup_port = MapLookupPort(
        reference_map=bridge.reference_map,
        waypoint_map=bridge.waypoint_map_planner,
        carla_module=carla,
    )
    bridge._stable_reference_line_provider.rebind_map_planner(
        bridge.waypoint_map_planner
    )
    bridge.waypoint_backend = "admap"
    bridge.route_manager = CPXRouteManager(
        global_planner=bridge.global_planner,
        route_sampling_resolution_m=float(
            bridge.config.get("route_sampling_resolution_m", 1.0)
        ),
        route_reference_smoothing_passes=int(
            bridge.config.get("route_reference_smoothing_passes", 3)
        ),
        route_turn_connector_smoothing_passes=int(
            bridge.config.get("route_turn_connector_smoothing_passes", 16)
        ),
        route_reference_boundary_aware=bool(
            bridge.config.get(
                "route_reference_boundary_aware",
                True,
            )
        ),
        route_reference_vehicle_half_width_m=float(
            bridge.config.get("reference_vehicle_half_width_m", 1.0)
        ),
        route_reference_boundary_margin_m=float(
            bridge.config.get(
                "reference_contract_turn_boundary_margin_m",
                0.15,
            )
        ),
        route_reference_tracking_reserve_m=float(
            bridge.config.get(
                "route_reference_tracking_reserve_m",
                0.20,
            )
        ),
        route_rejoin_min_lateral_m=float(
            bridge.config.get("route_rejoin_min_lateral_m", 0.35)
        ),
        route_rejoin_max_lateral_m=float(
            bridge.config.get("route_rejoin_max_lateral_m", 3.0)
        ),
        route_rejoin_distance_m=float(
            bridge.config.get("route_rejoin_distance_m", 8.0)
        ),
        reached_distance_m=float(bridge.config.get("route_reached_distance_m", 3.0)),
        stale_route_lateral_m=float(bridge.config.get("route_stale_lateral_m", 12.0)),
        turn_replan_max_length_ratio=float(
            bridge.config.get("turn_replan_max_length_ratio", 2.5)
        ),
        turn_replan_max_added_length_m=float(
            bridge.config.get("turn_replan_max_added_length_m", 100.0)
        ),
    )
    bridge._active_route_summary = None
    bridge.mpc_feedback = BehaviorMPCFeedback(
        enabled=bool(bridge.config.get("mpc_feedback_enabled", True)),
        hold_s=float(bridge.config.get("mpc_feedback_hold_s", 1.5)),
        min_failures=int(bridge.config.get("mpc_feedback_min_failures", 1)),
    )
    from opencda.planning_module.pipeline.control_finalization_stage import (
        ControlFinalizationStage,
    )
    bridge.pipeline.control_finalization = ControlFinalizationStage(
        mpc=bridge.mpc,
        command_extractor=bridge.mpc_command_extractor,
        feedback=bridge.mpc_feedback,
        control_safety=control_safety_stage,
        config=bridge.config,
    )
    bridge.control_buffer = MPCControlBuffer(
        enabled=bool(bridge.config.get("control_buffer_enabled", True)),
        replan_period_s=float(
            bridge.config.get(
                "mpc_replan_period_s",
                getattr(bridge.mpc, "trajectory_generation_period_s", 0.25),
            )
        ),
        max_reuse_s=float(bridge.config.get("control_buffer_max_reuse_s", 0.35)),
        max_reference_anchor_jump_m=float(
            bridge.config.get(
                "control_buffer_max_reference_anchor_jump_m",
                0.75,
            )
        ),
        max_reference_heading_jump_rad=float(
            bridge.config.get(
                "control_buffer_max_reference_heading_jump_rad",
                0.35,
            )
        ),
        max_predicted_speed_error_mps=float(
            bridge.config.get(
                "control_buffer_max_predicted_speed_error_mps",
                0.75,
            )
        ),
        max_target_speed_jump_mps=float(
            bridge.config.get(
                "control_buffer_max_target_speed_jump_mps",
                1.0,
            )
        ),
    )
    from opencda.planning_module.pipeline.lane_change_lifecycle_stage import (
        LaneChangeLifecycleStage,
    )
    bridge.lane_change_lifecycle_stage = LaneChangeLifecycleStage(
        provider=bridge._stable_reference_line_provider,
        maneuver_manager=bridge.maneuver_manager,
        reference_generator=bridge.reference_generator,
        route_manager=bridge.route_manager,
        mpc=bridge.mpc,
        control_buffer=bridge.control_buffer,
        config=bridge.config,
        target_speed_mps=bridge.target_speed_mps,
        map_epoch="admap",
        vehicle_extent=lambda: getattr(
            getattr(bridge.vehicle_manager.vehicle, "bounding_box", None),
            "extent",
            None,
        ),
    )
    candidate_selection_stage.set_lane_change_lifecycle(
        bridge.lane_change_lifecycle_stage
    )
    bridge.pipeline.mpc_execution = MPCExecutionStage(
        mpc=bridge.mpc,
        control_buffer=bridge.control_buffer,
        minimum_replan_speed_mps=float(parts.full_control_buffer_min_speed_mps),
    )
    bridge.cp_message_path = str(
        bridge.config.get(
            "cp_message_path",
            Path(__file__).resolve().parents[1] / "behavior_planner" / "cp_message.json",
        )
    )
    bridge.behavior_planner = RuleBasedBehaviorPlanner(
        cp_message_path=str(bridge.cp_message_path),
        cooperative_message_check_frequency_hz=float(
            bridge.config.get("cooperative_message_check_frequency_hz", 5.0)
        ),
    )
    bridge.cp_provider = None
    cp_message_path = bridge.config.get("cp_message_path")
    if not cp_message_path:
        cp_message_path = bridge.cp_message_path
    if bool(bridge.config.get("publish_cp_message", True)):
        bridge.cp_provider = OpenCDACPProvider(
            message_path=str(cp_message_path),
            schema_version=1,
            communication_range_m=float(bridge.config.get("communication_range_m", 80.0)),
            prediction_horizon_s=float(bridge.mpc.horizon_s),
            prediction_dt_s=float(bridge.mpc.dt_s),
            source="native_opencda",
            require_native_opencda=bool(
                bridge.config.get("require_native_opencda_cp", True)
            ),
            visibility_filter_enabled=bool(
                bridge.config.get("cp_visibility_filter_enabled", False)
            ),
            visibility_backend=str(
                bridge.config.get(
                    "cp_visibility_backend",
                    "actor_geometry",
                )
            ),
            visibility_sensor_height_m=float(
                bridge.config.get("cp_visibility_sensor_height_m", 1.6)
            ),
            visibility_target_tolerance_m=float(
                bridge.config.get("cp_visibility_target_tolerance_m", 0.75)
            ),
        )
    bridge._latest_opencda_update: dict[str, Any] = {}
    bridge.last_output = None
    bridge._write_planning_metrics_artifacts = write_planning_metrics_artifacts
    bridge.evaluation_metrics = EvaluationMetricsRecorder(
        ego_length_m=float(bridge.config.get("metrics_ego_length_m", 4.5)),
        lateral_conflict_width_m=float(
            bridge.config.get("metrics_lateral_conflict_width_m", 2.5)
        ),
        min_ego_speed_for_ttc_mps=float(
            bridge.config.get("metrics_min_ego_speed_for_ttc_mps", 0.5)
        ),
        pet_conflict_radius_m=float(
            bridge.config.get("metrics_pet_conflict_radius_m", 3.0)
        ),
        pet_bin_size_m=float(bridge.config.get("metrics_pet_bin_size_m", 3.0)),
    )
    bridge._road_boundary = RoadBoundaryMonitor(
        bridge.config,
        vehicle_provider=lambda: bridge.vehicle_manager.vehicle,
        generator_provider=lambda: bridge.reference_generator,
    )
    bridge._prediction_lane_step_resolved_count = 0
    bridge._prediction_lane_step_none_count = 0
    bridge._cooperative = CooperativeArbitrationStage(
        config=bridge.config,
        enabled=bridge._cav_conflict_enabled,
        pipeline=bridge.pipeline,
        mpc=bridge.mpc,
        route_manager=bridge.route_manager,
        maneuver_manager=bridge.maneuver_manager,
        record_stage_ms=bridge._accum_stage_ms,
    )
