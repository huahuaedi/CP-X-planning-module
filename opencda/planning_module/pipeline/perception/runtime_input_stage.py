"""Normalize one OpenCDA runtime tick before planning starts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class RuntimeTickSnapshot:
    """Immutable ego state shared by every downstream planning stage."""

    timestamp_s: float
    ego_transform: Any
    ego_location: Any
    ego_speed_mps: float
    ego_yaw_rad: float
    measured_accel_mps2: float
    current_state: Tuple[float, float, float, float]


class RuntimeInputStage:
    """Own runtime-unit conversion and construction of the ego MPC state.

    A physically-bounded rate limit is applied to the reported ego speed
    *before* it reaches any downstream consumer (MPC state, the actuator
    mapper's measured acceleration, OpenCDA's longitudinal PID). CARLA's
    ``get_velocity()`` occasionally jumps the reported speed by ~0.5-1 m/s
    in a single tick while the vehicle is cruising steadily; that spike
    drives a brake-then-full-throttle overshoot in the PID and shows up as
    accel-command chatter / large jerk. A road car cannot accelerate faster
    than ~a few m/s^2 from cruise, so any upward jump beyond
    ``max_upward_accel_mps2`` per tick is rejected; downward changes pass
    through untouched so real hard braking / collisions are never masked.
    """

    def __init__(
        self,
        actuator_mapper: Any,
        *,
        max_upward_accel_mps2: float = 6.0,
    ) -> None:
        self._actuator_mapper = actuator_mapper
        self._max_upward_accel_mps2 = max(0.0, float(max_upward_accel_mps2))
        self._prev_speed_mps: Optional[float] = None
        self._prev_timestamp_s: Optional[float] = None

    def _rate_limited_speed_mps(
        self, raw_speed_mps: float, timestamp_s: float
    ) -> float:
        speed = max(0.0, float(raw_speed_mps))
        limit = float(self._max_upward_accel_mps2)
        if (
            limit <= 0.0
            or self._prev_speed_mps is None
            or self._prev_timestamp_s is None
        ):
            self._prev_speed_mps = speed
            self._prev_timestamp_s = float(timestamp_s)
            return speed
        dt_s = float(timestamp_s) - float(self._prev_timestamp_s)
        if dt_s <= 1.0e-3 or dt_s > 0.5:
            # Non-monotonic clock or a long gap (reset / first tick after a
            # pause): trust the raw reading.
            self._prev_speed_mps = speed
            self._prev_timestamp_s = float(timestamp_s)
            return speed
        max_speed = float(self._prev_speed_mps) + limit * dt_s
        limited = speed if speed <= max_speed else max_speed
        self._prev_speed_mps = limited
        self._prev_timestamp_s = float(timestamp_s)
        return limited

    def build(
        self,
        *,
        timestamp_s: float,
        ego_transform: Any,
        ego_speed_kmh: float,
    ) -> RuntimeTickSnapshot:
        raw_speed_mps = float(ego_speed_kmh) / 3.6
        speed_mps = self._rate_limited_speed_mps(raw_speed_mps, float(timestamp_s))
        location = ego_transform.location
        yaw_rad = math.radians(float(ego_transform.rotation.yaw))
        acceleration_mps2 = self._actuator_mapper.update_measurement(
            speed_mps=float(speed_mps),
            timestamp_s=float(timestamp_s),
        )
        return RuntimeTickSnapshot(
            timestamp_s=float(timestamp_s),
            ego_transform=ego_transform,
            ego_location=location,
            ego_speed_mps=float(speed_mps),
            ego_yaw_rad=float(yaw_rad),
            measured_accel_mps2=float(acceleration_mps2),
            current_state=(
                float(location.x),
                float(location.y),
                float(speed_mps),
                float(yaw_rad),
            ),
        )
