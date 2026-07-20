"""Last-mile safety filtering for CP-X planner outputs."""

from __future__ import annotations

from typing import Any, Mapping, Tuple


class SafetySupervisor:
    """Filter planner controls before OpenCDA applies them."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_steer_delta: float = 0.25,
        max_throttle_delta: float = 0.45,
        max_brake_delta: float = 0.60,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_steer_delta = max(0.0, float(max_steer_delta))
        self.max_throttle_delta = max(0.0, float(max_throttle_delta))
        self.max_brake_delta = max(0.0, float(max_brake_delta))
        self._last_control = None

    def filter_control(
        self,
        *,
        control: Any,
        carla_module: Any,
        safety_manager: Any = None,
        input_frame: Any = None,
    ) -> Tuple[Any, str]:
        del input_frame
        if not bool(self.enabled):
            self._last_control = control
            return control, ""
        hazard_reason = self._hazard_reason(safety_manager)
        if hazard_reason:
            safe = carla_module.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
            self._last_control = safe
            return safe, "safety_supervisor_emergency_stop:" + hazard_reason
        if self._last_control is None:
            self._last_control = control
            return control, ""
        filtered = carla_module.VehicleControl(
            throttle=self._limit_delta(
                float(getattr(control, "throttle", 0.0)),
                float(getattr(self._last_control, "throttle", 0.0)),
                self.max_throttle_delta,
            ),
            brake=self._limit_delta(
                float(getattr(control, "brake", 0.0)),
                float(getattr(self._last_control, "brake", 0.0)),
                self.max_brake_delta,
            ),
            steer=self._limit_delta(
                float(getattr(control, "steer", 0.0)),
                float(getattr(self._last_control, "steer", 0.0)),
                self.max_steer_delta,
            ),
        )
        self._last_control = filtered
        if (
            abs(float(getattr(filtered, "throttle", 0.0)) - float(getattr(control, "throttle", 0.0))) > 1.0e-6
            or abs(float(getattr(filtered, "brake", 0.0)) - float(getattr(control, "brake", 0.0))) > 1.0e-6
            or abs(float(getattr(filtered, "steer", 0.0)) - float(getattr(control, "steer", 0.0))) > 1.0e-6
        ):
            return filtered, "safety_supervisor_rate_limit"
        return filtered, ""

    @staticmethod
    def _limit_delta(value: float, previous: float, max_delta: float) -> float:
        delta = float(value) - float(previous)
        if delta > float(max_delta):
            return float(previous) + float(max_delta)
        if delta < -float(max_delta):
            return float(previous) - float(max_delta)
        return float(value)

    @staticmethod
    def _hazard_reason(safety_manager: Any) -> str:
        queue = getattr(safety_manager, "status_queue", None)
        if not queue:
            return ""
        try:
            _, status = queue[-1]
        except Exception:
            return ""
        if not isinstance(status, Mapping):
            return ""
        active = [
            str(key)
            for key, value in dict(status).items()
            if bool(value)
        ]
        return ",".join(active)
