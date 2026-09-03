"""Normalize one OpenCDA runtime tick before planning starts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Tuple


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
    """Own runtime-unit conversion and construction of the ego MPC state."""

    def __init__(self, actuator_mapper: Any) -> None:
        self._actuator_mapper = actuator_mapper

    def build(
        self,
        *,
        timestamp_s: float,
        ego_transform: Any,
        ego_speed_kmh: float,
    ) -> RuntimeTickSnapshot:
        speed_mps = float(ego_speed_kmh) / 3.6
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
