"""Boundary-recovery request: persistence, hysteresis and cooldown.

Turns each tick's road-boundary measurement into a ``BoundaryRecoveryRequest``.
A violation must persist for several frames before it latches, the latch holds
until the clearance recovers past a release margin, and repeated infeasible
recovery plans put the request on cooldown.  The measurement itself is the
road-boundary monitor's job; this class only owns the request lifecycle.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .scenario_manager import BoundaryRecoveryRequest


class BoundaryRecoveryTracker:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = config
        self.request = BoundaryRecoveryRequest()
        self.trigger_frames = 0
        self.infeasible_frames = 0
        self.cooldown_until_s = -float("inf")

    def update(
        self,
        *,
        boundary_snapshot: Mapping[str, object],
        behavior_decision: str,
        sim_time_s: float,
        recovery_planned: bool = False,
        recovery_reference_feasible: bool = True,
    ) -> None:
        if float(sim_time_s) < float(
            self.cooldown_until_s
        ):
            self.reset()
            return

        geometry_source = str(
            boundary_snapshot.get(
                "road_boundary_geometry_source",
                "",
            )
        )
        drivable_inside = boundary_snapshot.get(
            "road_boundary_drivable_inside",
            "",
        )
        if (
            geometry_source.startswith("drivable_footprint:")
            and drivable_inside in {True, "True", "true", "1", 1}
        ):
            # Consuming the soft boundary margin may request lower speed, but
            # it must not latch hard recovery while the complete footprint is
            # still on CARLA's driving-lane union.
            self.infeasible_frames = 0
            self.reset()
            return

        if bool(recovery_planned) and not bool(recovery_reference_feasible):
            self.infeasible_frames = (
                int(
                    self.infeasible_frames
                )
                + 1
            )
            max_failures = max(
                1,
                int(
                    self._config.get(
                        "boundary_recovery_max_infeasible_frames",
                        3,
                    )
                ),
            )
            if int(self.infeasible_frames) >= int(
                max_failures
            ):
                self.cooldown_until_s = (
                    float(sim_time_s)
                    + max(
                        0.1,
                        float(
                            self._config.get(
                                "boundary_recovery_cooldown_s",
                                2.0,
                            )
                        ),
                    )
                )
                self.reset()
            return
        self.infeasible_frames = 0

        try:
            valid = bool(
                boundary_snapshot.get(
                    "road_boundary_sample_valid",
                    False,
                )
            )
            clearance_m = float(
                boundary_snapshot.get("road_boundary_clearance_m", "")
            )
            lateral_offset_m = float(
                boundary_snapshot.get(
                    "road_boundary_lateral_offset_m",
                    "",
                )
            )
            heading_error_rad = float(
                boundary_snapshot.get(
                    "road_boundary_heading_error_rad",
                    "",
                )
            )
        except (TypeError, ValueError):
            valid = False
            clearance_m = float("inf")
            lateral_offset_m = 0.0
            heading_error_rad = 0.0
        if not bool(valid) or not all(
            math.isfinite(value)
            for value in (
                float(clearance_m),
                float(lateral_offset_m),
                float(heading_error_rad),
            )
        ):
            self.reset()
            return

        trigger_clearance_m = float(
            self._config.get(
                "boundary_recovery_trigger_clearance_m",
                -0.10,
            )
        )
        release_clearance_m = max(
            float(trigger_clearance_m),
            float(
                self._config.get(
                    "boundary_recovery_release_clearance_m",
                    0.10,
                )
            ),
        )
        if float(clearance_m) <= float(trigger_clearance_m):
            self.trigger_frames = (
                int(self.trigger_frames) + 1
            )
        else:
            self.trigger_frames = 0
        required_frames = max(
            1,
            int(
                self._config.get(
                    "boundary_recovery_trigger_frames",
                    3,
                )
            ),
        )
        previous_active = bool(
            self.request.active
        )
        active = bool(
            (
                bool(previous_active)
                and float(clearance_m) < float(release_clearance_m)
            )
            or int(self.trigger_frames)
            >= int(required_frames)
        )
        decision = str(behavior_decision or "").strip().lower()
        turn_direction = (
            "left"
            if decision.endswith("_left")
            else "right"
            if decision.endswith("_right")
            else ""
        )
        self.request = BoundaryRecoveryRequest(
            valid=True,
            active=bool(active),
            clearance_m=float(clearance_m),
            lateral_offset_m=float(lateral_offset_m),
            heading_error_rad=float(heading_error_rad),
            turn_direction=str(turn_direction),
            timestamp_s=float(sim_time_s),
            reason=(
                "boundary_recovery_latched"
                if bool(active)
                else "boundary_recovery_monitor"
            ),
        )

    def reset(self) -> None:
        self.trigger_frames = 0
        self.request = BoundaryRecoveryRequest()
