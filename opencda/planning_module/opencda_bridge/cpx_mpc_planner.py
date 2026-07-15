"""Bridge from native OpenCDA vehicle managers to the CP-X MPC planner.

The bridge is intentionally small: OpenCDA still owns simulation, localization,
perception, and V2X discovery. This class consumes a custom map planner and
returns a CARLA ``VehicleControl`` directly, replacing both
OpenCDA's behavior agent and PID controller when enabled.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import carla
import yaml


class CPXMPCPlannerBridge:
    """Direct-control planner used inside ``VehicleManager.run_step``."""

    def __init__(
        self,
        vehicle_manager: Any,
        config: Optional[Mapping[str, Any]] = None,
        *,
        map_planner: Any = None,
    ):
        self.vehicle_manager = vehicle_manager
        self.config = dict(config or {})
        self.map_planner = map_planner
        self.enabled = bool(self.config.get("enabled", True))
        self.target_speed_mps = float(self.config.get("target_speed_mps", 8.0))
        self.lookahead_m = float(self.config.get("lookahead_m", 18.0))
        self.min_front_gap_m = float(self.config.get("min_front_gap_m", 8.0))
        self.debug = bool(self.config.get("debug", True))
        self.last_debug: dict[str, Any] = {}
        self._last_accel_mps2 = 0.0
        self._last_steer_rad = 0.0
        self._warned = False

        self._ensure_planning_module_import_path()
        from opencda.planning_module.MPC.mpc import MPC

        mpc_cfg, road_cfg = self._load_mpc_config()
        self.mpc = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)

    @staticmethod
    def _ensure_planning_module_import_path() -> None:
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

    def run_step(self) -> carla.VehicleControl:
        """Plan and return a low-level CARLA control command."""

        ego_transform = self.vehicle_manager.localizer.get_ego_pos()
        ego_speed_kmh = float(self.vehicle_manager.localizer.get_ego_spd())
        ego_speed_mps = ego_speed_kmh / 3.6
        ego_location = ego_transform.location
        ego_yaw_rad = math.radians(float(ego_transform.rotation.yaw))

        object_snapshots = self._collect_object_snapshots()
        front_gap_m = self._front_gap_m(
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            object_snapshots=object_snapshots,
        )
        stop_goal_active = front_gap_m is not None and front_gap_m < self.min_front_gap_m
        speed_ref_mps = 0.0 if stop_goal_active else self.target_speed_mps

        destination_state, lane_center_reference = self._build_route_reference(
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            speed_ref_mps=speed_ref_mps,
        )

        current_state = [
            float(ego_location.x),
            float(ego_location.y),
            float(ego_speed_mps),
            float(ego_yaw_rad),
        ]

        try:
            self.mpc.plan_trajectory(
                current_state=current_state,
                destination_state=destination_state,
                object_snapshots=object_snapshots,
                current_acceleration_mps2=float(self._last_accel_mps2),
                current_steering_rad=float(self._last_steer_rad),
                lane_center_reference_samples=lane_center_reference,
                stop_goal_active=bool(stop_goal_active),
            )
            u_solution = getattr(self.mpc, "_last_u_solution", None)
            if u_solution is None or len(u_solution) == 0:
                raise RuntimeError("MPC did not expose a control solution")
            accel_mps2 = float(u_solution[0, 0])
            steer_rad = float(u_solution[0, 1])
            control = self._control_from_mpc(accel_mps2, steer_rad)
            fallback_reason = ""
        except Exception as exc:
            control = self._fallback_control(
                ego_transform=ego_transform,
                ego_speed_mps=ego_speed_mps,
                destination_state=destination_state,
                stop_goal_active=stop_goal_active,
            )
            accel_mps2 = self._last_accel_mps2
            steer_rad = self._last_steer_rad
            fallback_reason = str(exc)
            if not self._warned:
                print(f"[CP-X OpenCDA Bridge] MPC fallback active: {fallback_reason}")
                self._warned = True

        self._last_accel_mps2 = float(accel_mps2)
        self._last_steer_rad = float(steer_rad)
        self.last_debug = {
            "planner": "cpx_mpc",
            "object_count": len(object_snapshots),
            "v2x_nearby_count": len(getattr(self.vehicle_manager.v2x_manager, "cav_nearby", {}) or {}),
            "front_gap_m": "" if front_gap_m is None else float(front_gap_m),
            "stop_goal_active": bool(stop_goal_active),
            "destination_x": float(destination_state[0]),
            "destination_y": float(destination_state[1]),
            "target_speed_mps": float(speed_ref_mps),
            "mpc_status": str(getattr(self.mpc, "_last_status", "")),
            "mpc_solve_time_ms": float(getattr(self.mpc, "_last_solve_time_ms", 0.0)),
            "fallback_reason": fallback_reason,
        }
        return control

    def _load_mpc_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg_path = self.config.get("mpc_config_path")
        if not cfg_path:
            cfg_path = Path(__file__).resolve().parents[1] / "MPC" / "mpc.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        mpc_cfg = dict(payload.get("mpc", payload))
        road_cfg = dict(payload.get("road", {}))
        road_cfg.setdefault("lane_count", int(self.config.get("lane_count", 3)))
        road_cfg.setdefault("lane_width_m", float(self.config.get("lane_width_m", 3.5)))
        return mpc_cfg, road_cfg

    def _collect_object_snapshots(self) -> list[dict[str, float]]:
        objects = getattr(self.vehicle_manager.perception_manager, "objects", {}) or {}
        vehicles = list(objects.get("vehicles", []) or [])
        snapshots: list[dict[str, float]] = []
        for index, obj in enumerate(vehicles):
            actor = getattr(obj, "carla_actor", None) or getattr(obj, "vehicle", None) or obj
            if actor is None or not hasattr(actor, "get_transform"):
                continue
            try:
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                bbox = getattr(actor, "bounding_box", None)
                extent = getattr(bbox, "extent", None)
                speed_mps = math.sqrt(
                    float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
                )
                snapshots.append({
                    "vehicle_id": str(getattr(actor, "id", index)),
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "v": float(speed_mps),
                    "psi": math.radians(float(transform.rotation.yaw)),
                    "length_m": 2.0 * float(getattr(extent, "x", 2.2)),
                    "width_m": 2.0 * float(getattr(extent, "y", 0.9)),
                })
            except RuntimeError:
                continue
        return snapshots

    def _build_route_reference(
        self,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        speed_ref_mps: float,
    ):
        samples = self._route_samples_from_custom_planner(
            ego_location=ego_location,
        )
        if not samples:
            dest_x = float(ego_location.x + self.lookahead_m * math.cos(ego_yaw_rad))
            dest_y = float(ego_location.y + self.lookahead_m * math.sin(ego_yaw_rad))
            return [dest_x, dest_y, float(speed_ref_mps), float(ego_yaw_rad)], []

        destination = samples[-1]
        return [
            float(destination["x_ref_m"]),
            float(destination["y_ref_m"]),
            float(speed_ref_mps),
            float(destination["heading_rad"]),
        ], samples

    def _route_samples_from_custom_planner(self, *, ego_location: carla.Location):
        if self.map_planner is None:
            return []
        waypoint = self.map_planner.get_waypoint(
            {
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "z": float(ego_location.z),
            }
        )
        if waypoint is None:
            return []

        from utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad

        samples: list[dict[str, float]] = []
        traveled_m = 0.0
        current = waypoint
        step_m = max(1.0, min(3.0, float(self.lookahead_m)))
        while current is not None and traveled_m <= float(self.lookahead_m):
            lane_width_m = float(current.lane_width_m or 3.5)
            samples.append({
                "x_ref_m": float(current.position["x"]),
                "y_ref_m": float(current.position["y"]),
                "heading_rad": float(world_heading_rad(current) or 0.0),
                "lane_id": int(canonical_lane_id_for_waypoint(current)),
                "lane_width_m": lane_width_m,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * lane_width_m,
                "road_right_width_m": 0.5 * lane_width_m,
            })
            candidates = list(current.next(step_m) or [])
            if not candidates:
                break
            current = candidates[0]
            traveled_m += step_m
        return samples

    def _front_gap_m(
        self,
        ego_location: carla.Location,
        ego_yaw_rad: float,
        object_snapshots: Sequence[Mapping[str, Any]],
    ) -> Optional[float]:
        cos_h = math.cos(ego_yaw_rad)
        sin_h = math.sin(ego_yaw_rad)
        best_gap = None
        for snapshot in object_snapshots:
            dx = float(snapshot.get("x", 0.0)) - float(ego_location.x)
            dy = float(snapshot.get("y", 0.0)) - float(ego_location.y)
            longitudinal = dx * cos_h + dy * sin_h
            lateral = -dx * sin_h + dy * cos_h
            if longitudinal <= 0.0 or abs(lateral) > 2.5:
                continue
            best_gap = longitudinal if best_gap is None else min(best_gap, longitudinal)
        return best_gap

    def _control_from_mpc(self, acceleration_mps2: float, steering_angle_rad: float) -> carla.VehicleControl:
        max_accel = max(1e-6, float(self.mpc.constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        throttle = min(1.0, max(0.0, float(acceleration_mps2) / max_accel))
        brake = min(1.0, max(0.0, -float(acceleration_mps2) / max_brake))
        steer = min(1.0, max(-1.0, float(steering_angle_rad) / max_steer))
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)

    def _fallback_control(
        self,
        ego_transform: carla.Transform,
        ego_speed_mps: float,
        destination_state: Sequence[float],
        stop_goal_active: bool,
    ) -> carla.VehicleControl:
        if stop_goal_active:
            self._last_accel_mps2 = float(self.mpc.constraints.min_acceleration_mps2)
            self._last_steer_rad = 0.0
            return carla.VehicleControl(throttle=0.0, brake=0.8, steer=0.0)

        dx = float(destination_state[0]) - float(ego_transform.location.x)
        dy = float(destination_state[1]) - float(ego_transform.location.y)
        target_yaw = math.atan2(dy, dx)
        yaw_error = self._wrap_angle(target_yaw - math.radians(float(ego_transform.rotation.yaw)))
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        steer_rad = min(max_steer, max(-max_steer, 0.7 * yaw_error))
        speed_error = float(self.target_speed_mps) - float(ego_speed_mps)
        accel = min(
            float(self.mpc.constraints.max_acceleration_mps2),
            max(float(self.mpc.constraints.min_acceleration_mps2), 0.6 * speed_error),
        )
        self._last_accel_mps2 = float(accel)
        self._last_steer_rad = float(steer_rad)
        return self._control_from_mpc(accel, steer_rad)

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def cpx_planner_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether a vehicle config requests the CP-X planner bridge."""

    planner_cfg = dict(config.get("planner", {}) or {})
    planner_type = str(planner_cfg.get("type", "")).strip().lower()
    env_type = str(os.environ.get("OPENCDA_PLANNER", "")).strip().lower()
    return planner_type in {"cpx_mpc", "cp_x_mpc"} or env_type in {"cpx_mpc", "cp_x_mpc"}
