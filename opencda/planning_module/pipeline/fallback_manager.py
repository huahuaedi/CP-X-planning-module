"""Named trajectory degradation policies independent of route rebuilding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType


@dataclass(frozen=True)
class FallbackResult:
    mode: str
    trajectory: tuple
    target_speed_mps: float
    reason: str

    def mutable_trajectory(self):
        return [dict(sample) for sample in self.trajectory]


@dataclass(frozen=True)
class FailureReason:
    """Typed failure submitted by a planning stage; it has no control policy."""

    stage: str
    code: str
    severity: str = "degraded"
    recoverable: bool = True
    details: str = ""

    def label(self):
        label = "%s:%s" % (str(self.stage), str(self.code))
        return label if not self.details else "%s:%s" % (label, self.details)


@dataclass(frozen=True)
class FallbackRequest:
    sim_time_s: float
    route_revision: str
    current_speed_mps: float
    current_reference: tuple
    failure: FailureReason


class TrajectoryFallbackManager:
    """Select only hold-last-valid or bounded-safe-stop degradation."""

    def __init__(self, *, max_hold_age_s=0.35, min_hold_arc_m=2.0,
                 safe_stop_deceleration_mps2=2.0):
        self.max_hold_age_s = max(0.0, float(max_hold_age_s))
        self.min_hold_arc_m = max(0.0, float(min_hold_arc_m))
        self.safe_stop_deceleration_mps2 = max(
            0.1, float(safe_stop_deceleration_mps2)
        )
        self._last_valid = ()
        self._last_valid_time_s = -float("inf")
        self._last_valid_route_revision = ""

    def record_valid(self, trajectory, *, sim_time_s, route_revision):
        samples = [dict(sample) for sample in list(trajectory or [])]
        if len(samples) < 2:
            return False
        self._last_valid = tuple(
            MappingProxyType(dict(sample)) for sample in samples
        )
        self._last_valid_time_s = float(sim_time_s)
        self._last_valid_route_revision = str(route_revision or "")
        return True

    def resolve(self, *, sim_time_s, route_revision, current_speed_mps,
                current_reference, failure_reason):
        failure = (
            failure_reason
            if isinstance(failure_reason, FailureReason)
            else FailureReason(
                stage="planning",
                code="unspecified_failure",
                details=str(failure_reason),
            )
        )
        return self.arbitrate(FallbackRequest(
            sim_time_s=float(sim_time_s),
            route_revision=str(route_revision or ""),
            current_speed_mps=float(current_speed_mps),
            current_reference=tuple(dict(sample) for sample in list(current_reference or [])),
            failure=failure,
        ))

    def arbitrate(self, request):
        """Select the only allowed degradation policy for a typed failure."""

        sim_time_s = float(request.sim_time_s)
        route_revision = str(request.route_revision or "")
        current_speed_mps = float(request.current_speed_mps)
        current_reference = request.current_reference
        failure_reason = request.failure.label()
        age_s = max(0.0, float(sim_time_s) - self._last_valid_time_s)
        last_arc_m = _trajectory_arc_m(self._last_valid)
        if (
            self._last_valid
            and age_s <= self.max_hold_age_s
            and str(route_revision or "") == self._last_valid_route_revision
            and last_arc_m >= self.min_hold_arc_m
        ):
            speed = _first_speed(self._last_valid, current_speed_mps)
            return FallbackResult(
                mode="hold_last_valid",
                trajectory=self._last_valid,
                target_speed_mps=max(0.0, speed),
                reason="hold_last_valid:" + str(failure_reason),
            )
        safe_stop = _bounded_safe_stop(
            current_reference,
            current_speed_mps=float(current_speed_mps),
            deceleration_mps2=self.safe_stop_deceleration_mps2,
        )
        return FallbackResult(
            mode="bounded_safe_stop",
            trajectory=tuple(MappingProxyType(dict(sample)) for sample in safe_stop),
            target_speed_mps=0.0,
            reason="bounded_safe_stop:" + str(failure_reason),
        )

    def bounded_safe_stop(self, *, current_speed_mps, current_reference,
                          reason="requested"):
        """Explicit terminal stop; never selects hold-last-valid."""
        safe_stop = _bounded_safe_stop(
            current_reference,
            current_speed_mps=float(current_speed_mps),
            deceleration_mps2=self.safe_stop_deceleration_mps2,
        )
        return FallbackResult(
            mode="bounded_safe_stop",
            trajectory=tuple(
                MappingProxyType(dict(sample)) for sample in safe_stop
            ),
            target_speed_mps=0.0,
            reason="bounded_safe_stop:" + str(reason),
        )


def _xy(sample):
    return (
        float(sample.get("x_ref_m", sample.get("x", 0.0))),
        float(sample.get("y_ref_m", sample.get("y", 0.0))),
    )


def _trajectory_arc_m(samples):
    distance = 0.0
    for first, second in zip(samples[:-1], samples[1:]):
        ax, ay = _xy(first)
        bx, by = _xy(second)
        distance += math.hypot(bx - ax, by - ay)
    return float(distance)


def _first_speed(samples, default):
    if not samples:
        return float(default)
    return float(samples[0].get(
        "speed_ref_mps", samples[0].get("v_ref_mps", default)
    ))


def _bounded_safe_stop(reference, *, current_speed_mps, deceleration_mps2):
    samples = [dict(sample) for sample in list(reference or [])]
    if len(samples) < 2:
        return samples
    station_m = 0.0
    previous = samples[0]
    for index, sample in enumerate(samples):
        if index:
            ax, ay = _xy(previous)
            bx, by = _xy(sample)
            station_m += math.hypot(bx - ax, by - ay)
        speed_sq = max(
            0.0,
            float(current_speed_mps) ** 2
            - 2.0 * float(deceleration_mps2) * float(station_m),
        )
        speed_mps = math.sqrt(speed_sq)
        sample["speed_ref_mps"] = speed_mps
        sample["v_ref_mps"] = speed_mps
        sample["fallback_mode"] = "bounded_safe_stop"
        previous = sample
    samples[-1]["speed_ref_mps"] = 0.0
    samples[-1]["v_ref_mps"] = 0.0
    return samples
