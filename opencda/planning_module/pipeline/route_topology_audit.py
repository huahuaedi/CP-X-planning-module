"""Offline acceptance audit for the frozen route-topology layer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RouteTopologyAudit:
    valid: bool
    signature: str
    frame_count: int
    initial_progress_m: float
    final_progress_m: float
    initial_remaining_m: float
    final_remaining_m: float
    violations: tuple[str, ...]


def audit_route_topology_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_signature: str = "",
    progress_backstep_tolerance_m: float = 0.05,
    remaining_rise_tolerance_m: float = 1.0,
    require_no_replan: bool = True,
) -> RouteTopologyAudit:
    """Validate only topology/progress invariants; ignore behavior and MPC."""

    samples = [dict(row) for row in rows]
    violations: list[str] = []
    signatures = {
        str(row.get("route_topology_signature", "")).strip()
        for row in samples
        if str(row.get("route_topology_signature", "")).strip()
    }
    signature = next(iter(signatures)) if len(signatures) == 1 else ""
    if len(signatures) != 1:
        violations.append("topology_signature_changed_or_missing")
    elif expected_signature and signature != str(expected_signature):
        violations.append("topology_signature_not_golden")
    if any(not bool(row.get("route_topology_valid", False)) for row in samples):
        violations.append("topology_invalid_frame")
    if any(str(row.get("route_topology_errors", "")).strip() for row in samples):
        violations.append("topology_error_reported")

    progress = _finite_values(samples, "route_progress_s_m")
    remaining = _finite_values(samples, "route_remaining_distance_m")
    if not progress:
        violations.append("progress_missing")
    elif any(
        second < first - max(0.0, float(progress_backstep_tolerance_m))
        for first, second in zip(progress, progress[1:])
    ):
        violations.append("progress_backstep")
    if not remaining:
        violations.append("remaining_distance_missing")
    elif any(
        second > first + max(0.0, float(remaining_rise_tolerance_m))
        for first, second in zip(remaining, remaining[1:])
    ):
        violations.append("remaining_distance_increased")
    if require_no_replan and any(
        bool(row.get("route_replan_attempted", False)) for row in samples
    ):
        violations.append("unexpected_replan_attempt")

    return RouteTopologyAudit(
        valid=not violations,
        signature=signature,
        frame_count=len(samples),
        initial_progress_m=progress[0] if progress else 0.0,
        final_progress_m=progress[-1] if progress else 0.0,
        initial_remaining_m=remaining[0] if remaining else math.inf,
        final_remaining_m=remaining[-1] if remaining else math.inf,
        violations=tuple(violations),
    )


def _finite_values(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values
