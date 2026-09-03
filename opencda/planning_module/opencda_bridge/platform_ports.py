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


class WorldDebugPort:
    """CARLA-only visualization of already-frozen planner outputs."""

    def __init__(
        self, *, vehicle_manager: Any, carla_module: Any, mpc: Any,
        route_points: Callable[[], list], enabled: bool,
        draw_destination: bool, life_time_s: float,
        report_error: Callable[[str], None],
    ) -> None:
        self._vehicle_manager = vehicle_manager
        self._carla = carla_module
        self._mpc = mpc
        self._route_points = route_points
        self._enabled = bool(enabled)
        self._draw_destination = bool(draw_destination)
        self._life_time_s = max(0.05, float(life_time_s))
        self._report_error = report_error
        self._route_cache_signature = None
        self._route_cache = ()

    def mpc_trajectory_points(self) -> list[tuple[float, float]]:
        solution = getattr(self._mpc, "_last_x_solution", None)
        if solution is None:
            return []
        try:
            return [
                (float(state[0]), float(state[1]))
                for state in list(solution) if len(state) >= 2
            ]
        except Exception:
            return []

    def display_route_points(self) -> list[list[float]]:
        raw = [list(point) for point in self._route_points()]
        if len(raw) < 5:
            return raw
        indices = sorted({0, len(raw)//4, len(raw)//2, 3*len(raw)//4, len(raw)-1})
        signature = tuple(
            (len(raw), i, round(float(raw[i][0]), 3), round(float(raw[i][1]), 3))
            for i in indices
        )
        if signature == self._route_cache_signature:
            return [list(point) for point in self._route_cache]
        xy = [(float(p[0]), float(p[1])) for p in raw]
        radius = 6
        smoothed = []
        for index in range(len(xy)):
            if index in {0, len(xy) - 1}:
                smoothed.append(xy[index])
                continue
            first, last = max(0, index-radius), min(len(xy)-1, index+radius)
            weighted_x = weighted_y = total = 0.0
            for neighbor in range(first, last + 1):
                weight = float(radius + 1 - abs(neighbor - index))
                weighted_x += weight * xy[neighbor][0]
                weighted_y += weight * xy[neighbor][1]
                total += weight
            smoothed.append((weighted_x / total, weighted_y / total))
        display = []
        for index, (x_m, y_m) in enumerate(smoothed):
            other = smoothed[index+1] if index+1 < len(smoothed) else smoothed[index-1]
            base = smoothed[index] if index+1 < len(smoothed) else smoothed[index-1]
            heading = math.atan2(other[1]-base[1], other[0]-base[0])
            z_m = float(raw[index][2]) if len(raw[index]) >= 3 else 0.0
            display.append([x_m, y_m, z_m, heading])
        self._route_cache_signature = signature
        self._route_cache = tuple(tuple(point) for point in display)
        return display

    def draw(self, *, destination_state: Any, reference_samples: Any) -> None:
        if not self._enabled:
            return
        try:
            vehicle = self._vehicle_manager.vehicle
            debug = getattr(vehicle.get_world(), "debug", None)
            if debug is None:
                return
            z_m = float(getattr(vehicle.get_location(), "z", 0.0)) + 0.35
            self._polyline(debug, [(p[0], p[1]) for p in self.display_route_points()],
                           z_m+0.05, self._carla.Color(255, 210, 20), 0.08, 120)
            self._polyline(debug, [
                (s.get("x_ref_m", s.get("x", 0.0)), s.get("y_ref_m", s.get("y", 0.0)))
                for s in list(reference_samples or ())
            ], z_m+0.15, self._carla.Color(245, 245, 245), 0.06, 80)
            self._polyline(debug, self.mpc_trajectory_points(), z_m+0.25,
                           self._carla.Color(30, 230, 70), 0.10, 80)
            if self._draw_destination and destination_state and len(destination_state) >= 2:
                debug.draw_point(
                    self._carla.Location(x=float(destination_state[0]),
                                         y=float(destination_state[1]), z=z_m+0.55),
                    size=0.18, color=self._carla.Color(30, 145, 255),
                    life_time=self._life_time_s, persistent_lines=False,
                )
        except Exception as exc:
            self._report_error(str(exc))

    def _polyline(self, debug, points, z_m, color, thickness, max_segments):
        points = [(float(p[0]), float(p[1])) for p in list(points or ()) if len(p) >= 2]
        if len(points) < 2:
            return
        stride = max(1, int(len(points) / max(1, int(max_segments))))
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        for first, second in zip(sampled[:-1], sampled[1:]):
            if math.hypot(second[0]-first[0], second[1]-first[1]) < 1e-3:
                continue
            debug.draw_line(
                self._carla.Location(x=first[0], y=first[1], z=float(z_m)),
                self._carla.Location(x=second[0], y=second[1], z=float(z_m)),
                thickness=float(thickness), color=color,
                life_time=self._life_time_s, persistent_lines=False,
            )
