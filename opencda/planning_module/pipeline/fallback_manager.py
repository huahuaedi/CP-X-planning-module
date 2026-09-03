"""Named trajectory degradation policies independent of route rebuilding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable, Optional, Sequence


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


@dataclass(frozen=True)
class CandidateFallbackResult:
    """Planning result when candidate selection cannot produce a trajectory."""

    decision: str
    target_lane_id: int
    target_speed_mps: float
    trajectory: tuple
    destination_state: tuple
    diagnostics: MappingProxyType

    def mutable_trajectory(self):
        return [dict(sample) for sample in self.trajectory]

    def mutable_destination_state(self):
        return list(self.destination_state)

    def mutable_diagnostics(self):
        return dict(self.diagnostics)


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

    def resolve_candidate_failure(
        self,
        *,
        candidate_results: Sequence[Any],
        baseline_decision: str,
        baseline_target_lane_id: int,
        current_lane_id: int,
        current_state: Sequence[float],
        ego_x_m: float,
        ego_y_m: float,
        reference_provider: Any,
        route_revision: str,
        sim_time_s: float,
        mpc_dt_s: float,
        horizon_steps: int,
        lane_change_min_first_forward_m: float,
        lane_follow_min_first_forward_m: float,
        summarize_candidates: Callable[[Sequence[Any]], str],
        maneuver_commitment: Optional[Any] = None,
        selection_reason: str = "",
    ) -> CandidateFallbackResult:
        """Own candidate rejection policy and select one bounded degradation."""

        committed = bool(
            maneuver_commitment is not None
            and bool(getattr(maneuver_commitment, "active", False))
        )
        baseline_normalized = str(baseline_decision or "").strip().lower()
        reference_mode = (
            "lane_change"
            if committed
            else "turn"
            if baseline_normalized in {
                "intersection_turn_left",
                "intersection_turn_right",
            }
            else "lane_follow"
        )
        snapshot = reference_provider.snapshot(reference_mode)
        current_reference = []
        if snapshot.active:
            window = reference_provider.window(
                reference_mode,
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                first_forward_m=max(
                    0.0,
                    float(
                        lane_change_min_first_forward_m
                        if reference_mode == "lane_change"
                        else lane_follow_min_first_forward_m
                    ),
                ),
                spacing_m=max(
                    0.1,
                    float(mpc_dt_s) * max(0.5, float(current_state[2])),
                ),
                count=int(horizon_steps),
                max_projection_advance_m=max(
                    2.0, 2.0 * max(0.5, float(current_state[2]))
                ),
            )
            current_reference = [dict(sample) for sample in window.samples]

        if len(current_reference) < 2:
            geometric_rows = [
                candidate
                for candidate in list(candidate_results or [])
                if len(list(getattr(candidate, "lane_center_reference", []) or []))
                >= 2
            ]
            contract_valid_rows = [
                candidate
                for candidate in geometric_rows
                if getattr(candidate, "contract_result", None) is not None
                and bool(getattr(candidate.contract_result, "valid", False))
            ]
            if contract_valid_rows or geometric_rows:
                source = (contract_valid_rows or geometric_rows)[0]
                current_reference = [
                    dict(sample)
                    for sample in list(
                        getattr(source, "lane_center_reference", []) or []
                    )
                ]

        collision_veto = any(
            "collision_risk" in str(getattr(candidate, "feasibility_reason", ""))
            for candidate in list(candidate_results or [])
        )
        failure = FailureReason(
            stage="candidate_selection",
            code="all_candidates_infeasible",
            severity="unsafe" if collision_veto else "degraded",
            recoverable=not collision_veto,
            details=str(selection_reason or "no_feasible_candidate"),
        )
        fallback = self.resolve(
            sim_time_s=float(sim_time_s),
            route_revision=str(route_revision or ""),
            current_speed_mps=float(current_state[2]),
            current_reference=current_reference,
            failure_reason=failure,
        )
        reference = fallback.mutable_trajectory()
        target_lane_id = int(
            getattr(maneuver_commitment, "target_lane_id", 0)
            if committed
            else baseline_target_lane_id
        ) or int(current_lane_id)
        retained_decision = (
            str(getattr(maneuver_commitment, "decision", baseline_decision))
            if committed
            else str(baseline_decision)
        )
        decision = "emergency_brake" if collision_veto else retained_decision
        destination = ()
        if reference:
            terminal = dict(reference[-1])
            destination = (
                float(terminal.get("x_ref_m", terminal.get("x", current_state[0]))),
                float(terminal.get("y_ref_m", terminal.get("y", current_state[1]))),
                float(fallback.target_speed_mps),
                float(terminal.get("heading_rad", current_state[3])),
                int(target_lane_id),
            )
        debug = {
            "stage": "fallback_manager",
            "intent_mode": str(decision),
            "fallback_reason": str(fallback.reason),
            "reference_source": "reference_line_provider:" + str(reference_mode),
            "candidate_pipeline_selected": str(fallback.mode),
            "candidate_pipeline_selected_status": "explicit_fallback",
            "candidate_pipeline_selected_reason": str(fallback.reason),
            "candidate_pipeline_count": int(len(candidate_results)),
            "candidate_pipeline_summary": str(
                summarize_candidates(candidate_results)
            ),
            "candidate_selected_decision": str(decision),
            "candidate_selected_lane_id": int(target_lane_id),
            "candidate_selection_failure_reason": str(failure.label()),
            "route_replan_attempted": False,
            "route_replan_succeeded": False,
        }
        if maneuver_commitment is not None:
            debug.update(maneuver_commitment.as_debug_fields())
        return CandidateFallbackResult(
            decision=str(decision),
            target_lane_id=int(target_lane_id),
            target_speed_mps=float(fallback.target_speed_mps),
            trajectory=tuple(
                MappingProxyType(dict(sample)) for sample in reference
            ),
            destination_state=tuple(destination),
            diagnostics=MappingProxyType(debug),
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
