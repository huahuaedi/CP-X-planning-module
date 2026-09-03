"""Extract the platform tracking command from an MPC optimal solution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence


@dataclass(frozen=True)
class MPCTrackingCommand:
    target_velocity_mps: float
    target_steering_rad: float
    velocity_preview_time_s: float
    velocity_source_index: float
    valid: bool
    reason: str


class MPCCommandExtractor:
    """Single owner of the MPC-to-platform velocity/steering boundary."""

    def __init__(
        self,
        *,
        preview_time_s: float = 0.6,
        velocity_source: str = "preview",
        min_acceleration_mps2: float = -6.0,
        max_acceleration_mps2: float = 3.0,
    ) -> None:
        self.preview_time_s = max(0.0, float(preview_time_s))
        normalized_source = str(velocity_source or "preview").strip().lower()
        if normalized_source not in {"horizon", "preview"}:
            raise ValueError("velocity_source must be 'horizon' or 'preview'")
        self.velocity_source = normalized_source
        self.min_acceleration_mps2 = min(0.0, float(min_acceleration_mps2))
        self.max_acceleration_mps2 = max(0.0, float(max_acceleration_mps2))
        self._last_command: Optional[MPCTrackingCommand] = None
        self._last_timestamp_s: Optional[float] = None

    @property
    def last_command(self) -> Optional[MPCTrackingCommand]:
        return self._last_command

    def extract(
        self,
        *,
        state_solution: Sequence[Sequence[float]],
        steering_rad: float,
        solution_dt_s: float,
        timestamp_s: float,
        max_velocity_mps: float,
    ) -> MPCTrackingCommand:
        dt_s = max(1.0e-6, float(solution_dt_s))
        velocities = self._finite_velocities(state_solution)
        if not velocities:
            return self.hold(
                steering_rad=float(steering_rad),
                reason="mpc_velocity_solution_missing",
            )

        source_index = (
            float(len(velocities) - 1)
            if self.velocity_source == "horizon"
            else min(
                float(len(velocities) - 1),
                max(0.0, float(self.preview_time_s) / dt_s),
            )
        )
        lower_index = int(math.floor(source_index))
        upper_index = min(lower_index + 1, len(velocities) - 1)
        alpha = float(source_index) - float(lower_index)
        raw_velocity_mps = (
            (1.0 - alpha) * float(velocities[lower_index])
            + alpha * float(velocities[upper_index])
        )
        raw_velocity_mps = max(
            0.0, min(float(max_velocity_mps), float(raw_velocity_mps))
        )
        target_velocity_mps = self._continuity_limited_velocity(
            raw_velocity_mps=float(raw_velocity_mps),
            timestamp_s=float(timestamp_s),
        )
        command = MPCTrackingCommand(
            target_velocity_mps=float(target_velocity_mps),
            target_steering_rad=float(steering_rad),
            velocity_preview_time_s=float(source_index * dt_s),
            velocity_source_index=float(source_index),
            valid=True,
            reason=(
                "mpc_optimized_horizon_velocity"
                if self.velocity_source == "horizon"
                else "mpc_optimized_velocity_preview"
            ),
        )
        self._last_command = command
        self._last_timestamp_s = float(timestamp_s)
        return command

    def hold(self, *, steering_rad: float, reason: str) -> MPCTrackingCommand:
        previous = self._last_command
        if previous is None:
            return MPCTrackingCommand(
                target_velocity_mps=0.0,
                target_steering_rad=float(steering_rad),
                velocity_preview_time_s=float(self.preview_time_s),
                velocity_source_index=0.0,
                valid=False,
                reason=str(reason),
            )
        return MPCTrackingCommand(
            target_velocity_mps=float(previous.target_velocity_mps),
            target_steering_rad=float(steering_rad),
            velocity_preview_time_s=float(previous.velocity_preview_time_s),
            velocity_source_index=float(previous.velocity_source_index),
            valid=True,
            reason=str(reason),
        )

    def reset(self) -> None:
        self._last_command = None
        self._last_timestamp_s = None

    @staticmethod
    def platform_target_velocity(
        *,
        nominal_velocity_mps: float,
        stop_goal_active: bool,
        emergency_stop: bool,
        mpc_safety_cap_mps: Optional[float] = None,
    ) -> float:
        """Resolve the one velocity command sent to the platform PID.

        ``SpeedTargetPlanner`` owns nominal longitudinal intent.  The MPC
        state preview is deliberately not used as a second target: it already
        contains the plant's acceleration dynamics, and feeding that small
        near-term state to another longitudinal PID applies those dynamics a
        second time.  MPC may lower the command only through an explicit
        safety cap (for example a future prediction/ST constraint).
        """

        if bool(stop_goal_active) or bool(emergency_stop):
            return 0.0
        target_mps = max(0.0, float(nominal_velocity_mps))
        if mpc_safety_cap_mps is not None:
            cap_mps = max(0.0, float(mpc_safety_cap_mps))
            target_mps = min(float(target_mps), float(cap_mps))
        return float(target_mps)

    def _continuity_limited_velocity(
        self, *, raw_velocity_mps: float, timestamp_s: float
    ) -> float:
        previous = self._last_command
        previous_time_s = self._last_timestamp_s
        if previous is None or previous_time_s is None:
            return float(raw_velocity_mps)
        elapsed_s = max(0.0, float(timestamp_s) - float(previous_time_s))
        lower = (
            float(previous.target_velocity_mps)
            + float(self.min_acceleration_mps2) * elapsed_s
        )
        upper = (
            float(previous.target_velocity_mps)
            + float(self.max_acceleration_mps2) * elapsed_s
        )
        return max(0.0, min(float(upper), max(float(lower), float(raw_velocity_mps))))

    @staticmethod
    def _finite_velocities(
        state_solution: Sequence[Sequence[float]],
    ) -> list[float]:
        velocities = []
        rows = [] if state_solution is None else list(state_solution)
        for state in rows:
            try:
                velocity_mps = float(state[2])
            except (IndexError, TypeError, ValueError):
                return []
            if not math.isfinite(velocity_mps):
                return []
            velocities.append(float(velocity_mps))
        return velocities
