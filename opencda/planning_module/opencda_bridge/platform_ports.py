"""OpenCDA/CARLA platform ports used by the planner bridge.

This module is deliberately free of behavior, route and reference decisions.
It is the only owner of waypoint lookup compatibility and actuator conversion.
"""

from __future__ import annotations

import math
from typing import Any, Callable


class MapLookupPort:
    """Translate world locations into the configured AD-map waypoint API."""

    def __init__(self, *, reference_map: Any, waypoint_map: Any, carla_module: Any):
        self._reference_map = reference_map
        self._waypoint_map = waypoint_map
        self._carla = carla_module

    def waypoint(self, location: Any):
        if self._waypoint_map is None:
            return None
        get_waypoint = getattr(self._waypoint_map, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        point = self.location_to_point(location)
        try:
            return get_waypoint(point)
        except Exception:
            pass
        try:
            return get_waypoint(self._carla.Location(**point))
        except Exception:
            return None

    def drivable_waypoint(self, location: Any):
        get_drivable = getattr(self._waypoint_map, "get_drivable_waypoint", None)
        if callable(get_drivable):
            try:
                return get_drivable(self.location_to_point(location))
            except Exception:
                return None
        return None

    def lane_id(self, location: Any) -> int:
        waypoint = self._reference_map.get_waypoint(self.location_to_point(location))
        if waypoint is None:
            return 0
        try:
            from utility.global_planner import canonical_lane_id_for_waypoint

            return int(canonical_lane_id_for_waypoint(waypoint) or 0)
        except Exception:
            return int(getattr(waypoint, "ad_lane_id", 0) or 0)

    @classmethod
    def from_owner(cls, owner: Any) -> "MapLookupPort":
        """Build the port lazily for lightweight bridge test doubles."""

        waypoint_map = getattr(
            owner, "waypoint_map_planner", getattr(owner, "map_planner", None)
        )
        reference_map = getattr(owner, "reference_map", waypoint_map)
        return cls(
            reference_map=reference_map,
            waypoint_map=waypoint_map,
            carla_module=getattr(owner, "carla", None),
        )

    @staticmethod
    def location_to_point(location: Any) -> dict[str, float]:
        return {
            "x": float(getattr(location, "x", 0.0)),
            "y": float(getattr(location, "y", 0.0)),
            "z": float(getattr(location, "z", 0.0)),
        }

    @staticmethod
    def body_frame_xy(
        *, origin_x_m: float, origin_y_m: float, heading_rad: float,
        target_x_m: float, target_y_m: float,
    ) -> tuple[float, float]:
        dx_m = float(target_x_m) - float(origin_x_m)
        dy_m = float(target_y_m) - float(origin_y_m)
        cos_h = math.cos(float(heading_rad))
        sin_h = math.sin(float(heading_rad))
        return (
            float(dx_m * cos_h + dy_m * sin_h),
            float(-dx_m * sin_h + dy_m * cos_h),
        )


class ActuatorPort:
    """Own conversion between planner acceleration/steering and platform control."""

    def __init__(
        self, *, actuator_mapper: Any, constraints: Any, carla_module: Any,
        clock: Callable[[], float],
    ) -> None:
        self._mapper = actuator_mapper
        self._constraints = constraints
        self._carla = carla_module
        self._clock = clock
        self._ego_speed_mps = 0.0
        self._target_speed_mps = 0.0
        self._stop_goal_active = False

    def set_context(
        self, *, ego_speed_mps: float, target_speed_mps: float,
        stop_goal_active: bool,
    ) -> None:
        self._ego_speed_mps = float(ego_speed_mps)
        self._target_speed_mps = float(target_speed_mps)
        self._stop_goal_active = bool(stop_goal_active)

    def control(self, acceleration_mps2: float, steering_angle_rad: float):
        max_accel = max(1e-6, float(self._constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self._constraints.min_acceleration_mps2)))
        max_steer = max(1e-6, float(self._constraints.max_steer_rad))
        pedals = self._mapper.map_acceleration(
            acceleration_mps2=float(acceleration_mps2),
            max_acceleration_mps2=max_accel,
            min_acceleration_mps2=-max_brake,
            ego_speed_mps=self._ego_speed_mps,
            target_speed_mps=self._target_speed_mps,
            stop_goal_active=self._stop_goal_active,
            timestamp_s=float(self._clock()),
        )
        return self._carla.VehicleControl(
            throttle=float(pedals.throttle),
            brake=float(pedals.brake),
            steer=min(1.0, max(-1.0, float(steering_angle_rad) / max_steer)),
        )

    def acceleration(self, control: Any) -> float:
        max_accel = max(1e-6, float(self._constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self._constraints.min_acceleration_mps2)))
        return float(self._mapper.acceleration_from_command(
            throttle=float(getattr(control, "throttle", 0.0)),
            brake=float(getattr(control, "brake", 0.0)),
            max_acceleration_mps2=max_accel,
            min_acceleration_mps2=-max_brake,
            ego_speed_mps=self._ego_speed_mps,
            target_speed_mps=self._target_speed_mps,
            stop_goal_active=self._stop_goal_active,
        ))

    def steering(self, control: Any) -> float:
        return float(getattr(control, "steer", 0.0)) * max(
            1e-6, float(self._constraints.max_steer_rad)
        )
