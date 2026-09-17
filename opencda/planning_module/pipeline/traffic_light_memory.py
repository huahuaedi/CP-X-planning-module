"""Temporal traffic-light state resolution for planning inputs."""

from __future__ import annotations

import math
from typing import Mapping


class TrafficLightMemory:
    """Debounce raw signal readings before scenario selection.

    This component is the single owner of red/yellow/green temporal memory.
    The scenario FSM receives its resolved state and must not add another
    signal debounce window.
    """

    def __init__(
        self,
        *,
        hold_unknown_s: float = 0.8,
        hold_green_unknown_s: float = 0.2,
        green_confirm_s: float = 0.0,
        hold_stop_unknown_until_green: bool = False,
        fail_safe_max_unknown_hold_s: float = 6.0,
    ) -> None:
        self.hold_unknown_s = max(0.0, float(hold_unknown_s))
        self.hold_green_unknown_s = max(0.0, float(hold_green_unknown_s))
        self.green_confirm_s = max(0.0, float(green_confirm_s))
        self.hold_stop_unknown_until_green = bool(
            hold_stop_unknown_until_green
        )
        # A stop-requiring reading that goes unknown (occlusion, a dropped
        # frame, sensor noise) must not resolve to "go" just because
        # hold_unknown_s's short debounce window passed -- that is a real
        # fail-open gap, not a design choice, for every scenario that
        # doesn't opt into the indefinite hold_stop_unknown_until_green.
        # This bounds the same fail-safe hold instead of leaving it off by
        # default: long enough to bridge a real, temporary detection gap,
        # but not indefinite, so a genuinely broken/never-confirms-green
        # signal path still releases eventually instead of deadlocking the
        # vehicle at the intersection forever.
        self.fail_safe_max_unknown_hold_s = max(
            0.0, float(fail_safe_max_unknown_hold_s)
        )
        self._last_stop_state = "unknown"
        self._last_stop_target: dict[str, object] | None = None
        self._hold_until_s = -float("inf")
        self._fail_safe_hold_until_s = -float("inf")
        self._green_since_s: float | None = None
        self._latched_stop_target: dict[str, object] | None = None
        self._latched_stop_state = "unknown"

    @property
    def latched_stop_target(self) -> dict[str, object] | None:
        return (
            dict(self._latched_stop_target)
            if self._latched_stop_target is not None
            else None
        )

    @property
    def latched_stop_state(self) -> str:
        return str(self._latched_stop_state)

    def latch_stop_target(
        self,
        *,
        traffic_state: str,
        stop_target: Mapping[str, object] | None,
        ego_x_m: float,
        ego_y_m: float,
        ego_yaw_rad: float,
        current_lane_id: int,
        virtual_stop_distance_m: float,
    ) -> tuple[dict[str, object] | None, str]:
        """Keep one world-fixed stop target throughout a red/yellow phase."""

        state = str(traffic_state or "unknown").strip().lower()
        if state not in {"red", "yellow"}:
            if self._latched_stop_target is not None:
                self._latched_stop_target = None
                self._latched_stop_state = state
                return None, "stop_target_latch_release"
            self._latched_stop_state = state
            return None, ""
        if (
            self._latched_stop_target is not None
            and self._latched_stop_state in {"red", "yellow"}
        ):
            return dict(self._latched_stop_target), "stop_target_latch_reuse"

        latched = None
        if isinstance(stop_target, Mapping):
            try:
                x_value = stop_target.get("x_m", stop_target.get("x"))
                y_value = stop_target.get("y_m", stop_target.get("y"))
                if x_value is not None and y_value is not None:
                    latched = dict(stop_target)
                    latched.update({
                        "x_m": float(x_value), "y_m": float(y_value),
                        "x": float(x_value), "y": float(y_value),
                        "source": str(latched.get("source", ""))
                        + ":latched_world_stop_target",
                    })
            except (TypeError, ValueError):
                latched = None
        if latched is None:
            distance_m = max(2.0, float(virtual_stop_distance_m))
            x_m = float(ego_x_m) + distance_m * math.cos(float(ego_yaw_rad))
            y_m = float(ego_y_m) + distance_m * math.sin(float(ego_yaw_rad))
            latched = {
                "x_m": x_m, "y_m": y_m, "x": x_m, "y": y_m,
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "distance_m": distance_m,
                "source": "latched_virtual_stop_target",
            }
        self._latched_stop_target = dict(latched)
        self._latched_stop_state = state
        return dict(latched), "stop_target_latch_create"

    def update(
        self,
        *,
        state: str,
        stop_target: Mapping[str, object] | None,
        sim_time_s: float,
    ) -> tuple[str, dict[str, object] | None, str]:
        normalized_state = str(state or "unknown").strip().lower()
        reason = ""
        if normalized_state in {"red", "yellow"}:
            self._green_since_s = None
            self._last_stop_state = str(normalized_state)
            self._last_stop_target = (
                dict(stop_target)
                if isinstance(stop_target, Mapping)
                else None
            )
            self._hold_until_s = float(sim_time_s) + float(self.hold_unknown_s)
            self._fail_safe_hold_until_s = float(sim_time_s) + float(
                self.fail_safe_max_unknown_hold_s
            )
            return str(normalized_state), self._last_stop_target, "raw_stop"

        if normalized_state == "green":
            if self._green_since_s is None:
                self._green_since_s = float(sim_time_s)
            if (
                self._last_stop_state in {"red", "yellow"}
                and float(sim_time_s) - float(self._green_since_s)
                < float(self.green_confirm_s)
            ):
                self._hold_until_s = max(
                    float(self._hold_until_s),
                    float(sim_time_s) + float(self.hold_unknown_s),
                )
                return (
                    str(self._last_stop_state),
                    self._last_stop_target,
                    "traffic_memory_wait_green_confirm",
                )
            self._last_stop_state = "green"
            self._last_stop_target = None
            self._hold_until_s = (
                float(sim_time_s) + float(self.hold_green_unknown_s)
            )
            reason = (
                "traffic_memory_green_release"
                if float(self.green_confirm_s) > 0.0
                else ""
            )
            return "green", None, reason

        if (
            normalized_state == "unknown"
            and float(sim_time_s) <= float(self._hold_until_s)
            and self._last_stop_state in {"red", "yellow"}
        ):
            reason = f"traffic_memory_hold_{self._last_stop_state}"
            return str(self._last_stop_state), self._last_stop_target, reason

        # Opt-in indefinite hold takes precedence over the bounded one right
        # below so a caller that explicitly asked for it keeps its exact
        # existing (unbounded) semantics and reason string.
        if (
            normalized_state == "unknown"
            and bool(self.hold_stop_unknown_until_green)
            and self._last_stop_state in {"red", "yellow"}
            and self._last_stop_target is not None
        ):
            return (
                str(self._last_stop_state),
                self._last_stop_target,
                f"traffic_memory_fail_safe_hold_{self._last_stop_state}_until_green",
            )

        # Bounded fail-safe hold: hold_unknown_s's debounce window above is
        # only meant to absorb a single dropped frame, not a real multi-
        # second occlusion/detection gap -- past it, this holds the same
        # stop-requiring state for up to fail_safe_max_unknown_hold_s before
        # falling through to plain "unknown" below. Unlike
        # hold_stop_unknown_until_green (indefinite, opt-in only), this is
        # bounded and unconditional: it closes the fail-open gap for every
        # caller that hasn't opted into the indefinite hold above, without
        # risking a permanent stop if the signal path is genuinely broken
        # rather than just briefly occluded.
        if (
            normalized_state == "unknown"
            and float(sim_time_s) <= float(self._fail_safe_hold_until_s)
            and self._last_stop_state in {"red", "yellow"}
            and self._last_stop_target is not None
        ):
            return (
                str(self._last_stop_state),
                self._last_stop_target,
                f"traffic_memory_fail_safe_hold_{self._last_stop_state}",
            )

        if (
            normalized_state == "unknown"
            and float(sim_time_s) <= float(self._hold_until_s)
            and self._last_stop_state == "green"
        ):
            return "green", None, "traffic_memory_hold_green"

        if normalized_state == "unknown":
            self._green_since_s = None
        return str(normalized_state), None, reason
