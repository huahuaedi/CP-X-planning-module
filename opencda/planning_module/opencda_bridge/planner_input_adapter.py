"""OpenCDA-to-CP-X planner input adapter.

This module is the explicit boundary between OpenCDA's runtime data providers
and the CP-X planning pipeline.  OpenCDA still owns scenario execution,
VehicleManager.update_info(), localization, perception, V2X/CP publication,
and stable global-route generation.  The adapter converts those signals into
the PlannerInputFrame consumed by CP-X behavior, reference generation, and MPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from opencda.planning_module.utility.planning_context import (
    CPMessageContext,
    EgoPlanningState,
    MapLaneContext,
    PerceptionContext,
    PlannerInputFrame,
    PlanningContext,
    PredictionContext,
    RouteContext,
    TargetContext,
    TrafficControlContext,
)
from utility.global_planner import (
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoints,
)


@dataclass(frozen=True)
class PlannerInputAdapterOutput:
    """Planner-facing input and reusable per-tick adapter products."""

    frame: PlannerInputFrame
    ego_pose: Dict[str, float]
    current_state: List[float]
    current_lane_id: int
    lane_ids: List[int]
    ego_waypoint: Any
    ego_snapshot: Dict[str, float]
    lane_assignments: Dict[str, int]
    lane_safety_scores: Dict[int, float]
    front_distance_by_lane: Dict[int, float]
    route_points: List[List[float]]
    route_summary: Dict[str, object]
    route_optimal_lane_id: int
    route_reference_allowed: bool
    route_reference_gate_reason: str
    selected_traffic_control: Optional[Mapping[str, object]]
    signal_context: Dict[str, object]
    stop_target: Optional[Mapping[str, object]]
    source_quality: Dict[str, object]


class OpenCDARuntimePort:
    """Public input-facing port over the legacy bridge implementation.

    Only this compatibility class may call bridge-private data helpers.  The
    planner input adapter itself depends on named runtime capabilities, which
    makes those helpers movable without changing the input contract.
    """

    def __init__(self, bridge: Any):
        self._bridge = bridge

    def __getattr__(self, name: str):
        return getattr(self._bridge, name)

    def sim_time_s(self) -> float:
        return float(self._bridge._sim_time_s())

    def assign_obstacles_to_lanes(self, snapshots, **kwargs):
        return self._bridge._assign_obstacles_to_lanes(snapshots, **kwargs)

    def lane_step_fn(self):
        return self._bridge._obstacle_lane_step_fn()

    def nearest_front_distance_by_lane(self, **kwargs):
        return self._bridge._nearest_front_distance_by_lane(**kwargs)

    def active_global_route_points(self):
        return self._bridge._active_global_route_points()

    def planning_global_route_summary(self, **kwargs):
        return self._bridge._planning_module_global_route_summary(**kwargs)

    def local_map_snapshot(self):
        return getattr(self._bridge, "_local_map_snapshot", None)

    def load_cp_message_payload(self):
        return self._bridge._load_cp_message_payload()

    def select_relevant_traffic_control(self, **kwargs):
        return self._bridge._select_relevant_traffic_control(**kwargs)

    def traffic_context_from_cp_control(self, **kwargs):
        return self._bridge._traffic_context_from_cp_control(**kwargs)

    def record_lane_id_discontinuity(self, **kwargs):
        return self._bridge._record_lane_id_discontinuity(**kwargs)


class OpenCDAPlanningAdapter:
    """Build PlannerInputFrame from a native OpenCDA VehicleManager snapshot."""

    def __init__(self, bridge: Any):
        self.runtime = OpenCDARuntimePort(bridge)

    def build(
        self,
        *,
        ego_location: Any,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        cp_payload: Optional[Mapping[str, Any]],
    ) -> PlannerInputAdapterOutput:
        bridge = self.runtime
        sim_time_s = float(bridge.sim_time_s())
        ego_pose = {
            "x": float(ego_location.x),
            "y": float(ego_location.y),
            "z": float(ego_location.z),
            "heading_rad": float(ego_yaw_rad),
        }
        current_state = [
            float(ego_location.x),
            float(ego_location.y),
            float(ego_speed_mps),
            float(ego_yaw_rad),
        ]
        ego_waypoint = bridge.reference_map.get_waypoint(ego_pose)
        # One lane namespace is used end-to-end: the opaque AD-map lane id.
        # Direction is carried separately by topology offsets; numeric lane-id
        # ordering has no lateral meaning.
        current_lane_id = int(canonical_lane_id_for_waypoint(ego_waypoint) or 0)

        # Keep active-route progress synchronized throughout lane follow, so
        # intersection reference generation never performs a late first
        # lookup hundreds of metres into the route.
        bridge.route_manager.sync_route_progress(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
        )

        ego_snapshot = {
            "x": float(ego_location.x),
            "y": float(ego_location.y),
            "v": float(ego_speed_mps),
            "psi": float(ego_yaw_rad),
        }

        route_points = bridge.active_global_route_points()
        route_summary = bridge.planning_global_route_summary(
            ego_location=ego_location,
            ego_heading_rad=float(ego_yaw_rad),
            fallback_lane_id=int(current_lane_id),
            ego_waypoint=ego_waypoint,
        )
        local_map = bridge.local_map_snapshot()
        authoritative_lane_id = int(
            getattr(local_map, "ego_lane_id", 0)
            or route_summary.get("authoritative_current_lane_id", 0)
            or 0
        )
        authoritative_waypoint = getattr(
            bridge,
            "_authoritative_ego_waypoint",
            None,
        )
        if authoritative_lane_id != 0:
            current_lane_id = int(authoritative_lane_id)
        if authoritative_waypoint is not None:
            ego_waypoint = authoritative_waypoint
        lane_ids = [
            int(canonical_lane_id_for_waypoint(wp))
            for wp in list(canonical_lane_waypoints(ego_waypoint) or [])
            if int(canonical_lane_id_for_waypoint(wp)) != 0
        ]
        if local_map is not None and getattr(local_map, "frame_id", 0):
            local_corridors = {
                int(corridor.offset): list(corridor.lane_ids)
                for corridor in tuple(getattr(local_map, "corridors", ()) or ())
            }
        else:
            local_corridors = dict(
                route_summary.get("diagnostic_local_lane_frame", {}).get(
                    "corridors", {}
                )
                if isinstance(
                    route_summary.get("diagnostic_local_lane_frame", {}), Mapping
                )
                else {}
            )
        for corridor_lane_ids in local_corridors.values():
            for lane_id in list(corridor_lane_ids or []):
                if int(lane_id or 0) != 0:
                    lane_ids.append(int(lane_id))
        lane_ids = list(dict.fromkeys(lane_ids))
        if not lane_ids:
            lane_ids = [int(current_lane_id)] if current_lane_id != 0 else []
        snapshot_target_lane_id = int(
            getattr(local_map, "route_target_lane_id", 0) or 0
        )
        snapshot_target_in_frame = bool(
            getattr(local_map, "route_target_in_frame", False)
        )
        route_optimal_lane_id = int(
            snapshot_target_lane_id
            if snapshot_target_in_frame and snapshot_target_lane_id != 0
            else route_summary.get("optimal_lane_id", current_lane_id) or current_lane_id
        )
        route_reference_allowed = (
            bool(bridge.use_opencda_global_route)
            and bool(bridge.opencda_global_route_reference_allowed)
            and bool(route_summary.get("route_found", False))
            and len(route_points) >= 2
        )
        route_reference_gate_reason = (
            "opencda_global_route_enabled"
            if bool(route_reference_allowed)
            else str(route_summary.get("debug_reason", "opencda_global_route_unavailable"))
        )

        cp_payload = dict(cp_payload or bridge.load_cp_message_payload())
        cp_timestamp_s = _optional_float(cp_payload.get("timestamp_s", None))
        cp_age_s = (
            ""
            if cp_timestamp_s is None
            else max(0.0, float(sim_time_s) - float(cp_timestamp_s))
        )
        cp_valid = True
        if cp_age_s != "":
            cp_valid = float(cp_age_s) <= float(
                bridge.config.get("max_cp_message_age_s", 1.0)
            )
        traffic_controls = list(cp_payload.get("control", []) or [])
        lane_closures = list(
            cp_payload.get("lane_closures", cp_payload.get("lane_events", [])) or []
        )
        cp_obstacles = list(cp_payload.get("obstacles", []) or [])
        selected_control = bridge.select_relevant_traffic_control(
            traffic_controls=traffic_controls,
            ego_location=ego_location,
            ego_heading_rad=float(ego_yaw_rad),
            current_lane_id=int(current_lane_id),
            current_road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
            sim_time_s=float(sim_time_s),
        )
        signal_context, stop_target = bridge.traffic_context_from_cp_control(
            selected_control=selected_control,
            ego_location=ego_location,
        )
        if bool(bridge.config.get("ignore_traffic_control", False)):
            signal_context = {
                "signal_state": "unknown",
                "signal_source": "disabled_by_planner_config",
                "traffic_control_from_cp": False,
            }
            stop_target = None
        traffic_control_context = TrafficControlContext.from_signal_context(
            signal_context=signal_context,
            stop_target=stop_target,
        )
        tracked_obstacles = bridge.tracker.update(
            obstacle_snapshots=object_snapshots,
            timestamp_s=float(sim_time_s),
            signal_context=signal_context,
            stop_target=stop_target,
        )
        # Lane assignment, safety scoring, and front-gap extraction used to
        # run once on raw detections and then immediately run again on the
        # tracker output below.  Nothing consumed the first result.  Keep the
        # tracked snapshot as the single per-tick map-query owner so each
        # obstacle is projected onto the CARLA reference map only once.
        lane_assignments = bridge.assign_obstacles_to_lanes(
            tracked_obstacles,
            ego_waypoint=ego_waypoint,
            ego_lane_id=int(current_lane_id),
        )
        lane_safety_scores = bridge.lane_safety_scorer.compute_lane_scores(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=tracked_obstacles,
            lane_assignments=lane_assignments,
            ego_lane_id=int(current_lane_id),
            available_lane_ids=lane_ids,
            timestamp_s=float(sim_time_s),
        )
        bridge.lane_safety_scorer.cleanup_stale_obstacles(set(lane_assignments.keys()))
        front_dist_by_lane = bridge.nearest_front_distance_by_lane(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=tracked_obstacles,
            lane_assignments=lane_assignments,
            available_lane_ids=lane_ids,
        )
        # A flat min_front_gap_m doesn't scale with cruise speed -- give
        # it the same reaction-time margin regardless of how fast the
        # scenario is configured to cruise, floored at the configured
        # distance so low-speed/queued situations keep a sane minimum.
        # Scaled off the *configured* cruise target rather than ego's
        # live instantaneous speed: the conflict that matters here (a
        # queued lane change denied by this exact check) happens while
        # ego is still mid-acceleration toward that target, so scaling
        # off the live speed barely moved the effective floor at the
        # moment it mattered (confirmed via telemetry -- same denial,
        # same distances, down to the decimal, before and after that
        # version of the fix).
        speed_scaled_min_gap_m = max(
            float(bridge.min_front_gap_m),
            float(bridge.target_speed_mps) * float(bridge.min_front_gap_time_s),
        )
        prediction_frame = bridge.tracker.predict(
            ego_snapshot=ego_snapshot,
            lane_assignments=lane_assignments,
            available_lane_ids=lane_ids,
            horizon_s=float(bridge.mpc.horizon_s),
            dt_s=float(bridge.mpc.dt_s),
            min_front_gap_m=float(speed_scaled_min_gap_m),
            min_rear_gap_m=float(speed_scaled_min_gap_m),
            min_ttc_s=float(bridge.config.get("prediction_min_ttc_s", 2.0)),
            lane_step_fn=bridge.lane_step_fn(),
        )
        route_context = RouteContext(
            optimal_lane_id=int(route_optimal_lane_id),
            next_macro_maneuver=str(
                route_summary.get("next_macro_maneuver", "Continue Straight")
            ),
            current_road_option=str(route_summary.get("current_road_option", "")),
            next_macro_distance_m=float(
                route_summary.get("next_macro_distance_m", float("inf"))
                if route_summary.get("next_macro_distance_m", None) is not None
                else float("inf")
            ),
            remaining_distance_m=float(route_summary.get("remaining_distance_m", 0.0) or 0.0),
            remaining_points_count=len(route_points),
            route_found=bool(route_summary.get("route_found", False)),
        )
        frame = PlannerInputFrame(
            planning=PlanningContext(
                sim_time_s=float(sim_time_s),
                ego=EgoPlanningState(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    speed_mps=float(ego_speed_mps),
                    heading_rad=float(ego_yaw_rad),
                    lane_id=int(current_lane_id),
                    road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
                    section_id=int(getattr(ego_waypoint, "section_id", 0) or 0),
                    in_junction=bool(
                        getattr(
                            ego_waypoint,
                            "is_junction",
                            getattr(ego_waypoint, "is_intersection", False),
                        )
                    ),
                ),
                route=route_context,
                traffic_control=traffic_control_context,
                targets=TargetContext(stop_target=traffic_control_context.stop_target),
                global_route_reference_allowed=bool(route_reference_allowed),
                global_route_reference_gate_reason=str(route_reference_gate_reason),
            ),
            map_lane=MapLaneContext(
                lane_id=int(current_lane_id),
                road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
                section_id=int(getattr(ego_waypoint, "section_id", 0) or 0),
                lane_count=len(lane_ids),
                allowed_lane_ids=list(lane_ids),
                in_junction=bool(
                    getattr(
                        ego_waypoint,
                        "is_junction",
                        getattr(ego_waypoint, "is_intersection", False),
                    )
                ),
                route_lane_id=int(route_optimal_lane_id),
                route_maneuver=str(route_context.next_macro_maneuver),
            ),
            perception=PerceptionContext(
                dynamic_objects=[dict(obj) for obj in tracked_obstacles],
                planning_objects=[dict(obj) for obj in tracked_obstacles],
                source="native_opencda",
            ),
            prediction=PredictionContext(
                lane_assignments=dict(lane_assignments),
                lane_prediction_risks=dict(prediction_frame.lane_prediction_risks),
                obstacle_future_trajectories=dict(
                    prediction_frame.obstacle_future_trajectories
                ),
                model="constant_acceleration",
                horizon_s=float(bridge.mpc.horizon_s),
                dt_s=float(bridge.mpc.dt_s),
            ),
            cp_messages=CPMessageContext(
                message_path=str(bridge.cp_message_path),
                traffic_controls=traffic_controls,
                selected_traffic_control=selected_control,
                lane_closures=lane_closures,
                obstacles=cp_obstacles,
            ),
        )
        return PlannerInputAdapterOutput(
            frame=frame,
            ego_pose=ego_pose,
            current_state=current_state,
            current_lane_id=int(current_lane_id),
            lane_ids=list(lane_ids),
            ego_waypoint=ego_waypoint,
            ego_snapshot=ego_snapshot,
            lane_assignments=dict(lane_assignments),
            lane_safety_scores=dict(lane_safety_scores),
            front_distance_by_lane=dict(front_dist_by_lane),
            route_points=list(route_points),
            route_summary=dict(route_summary),
            route_optimal_lane_id=int(route_optimal_lane_id),
            route_reference_allowed=bool(route_reference_allowed),
            route_reference_gate_reason=str(route_reference_gate_reason),
            selected_traffic_control=selected_control,
            signal_context=dict(signal_context or {}),
            stop_target=stop_target,
            source_quality={
                "planner_input_frame_timestamp_s": float(sim_time_s),
                "cp_message_timestamp_s": "" if cp_timestamp_s is None else float(cp_timestamp_s),
                "cp_message_age_s": cp_age_s,
                "cp_message_valid": bool(cp_valid),
                **dict(bridge.tracker.diagnostics),
            },
        )


def _optional_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


PlannerInputAdapter = OpenCDAPlanningAdapter
