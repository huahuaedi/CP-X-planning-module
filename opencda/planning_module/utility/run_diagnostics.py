"""Run-level diagnostic summary for planning artifacts.

The low-level CSV files are useful when debugging a specific tick, but they
are too noisy as a first look after a scenario run.  This module combines the
existing run status, metrics, reference-pipeline summary, and speed-cap summary
into a compact layer-by-layer report.
"""

from __future__ import annotations

from typing import Dict, Mapping


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int_value(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _health_from_rate(rate: float, *, warn: float, fail: float) -> str:
    if float(rate) >= float(fail):
        return "fail"
    if float(rate) >= float(warn):
        return "warn"
    return "ok"


def build_planning_debug_summary(
    *,
    scenario_name: str,
    run_status: Mapping[str, object],
    metrics_summary: Mapping[str, object] | None = None,
    reference_summary: Mapping[str, object] | None = None,
    speed_summary: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Build a compact diagnosis for the full planning run."""

    metrics = dict(metrics_summary or {})
    reference = dict(reference_summary or {})
    speed = dict(speed_summary or {})
    status = dict(run_status or {})

    reference_violation_rate = _float_value(reference.get("violation_rate", 0.0))
    reference_stabilized_rate = _float_value(reference.get("stabilized_rate", 0.0))
    speed_rise_limited_rate = _float_value(speed.get("rise_limited_rate", 0.0))
    collision_count = _int_value(metrics.get("collision_count", 0))
    mpc_success_rate = _float_value(metrics.get("mpc_plan_success_rate", 1.0), 1.0)

    likely_issues = []
    if collision_count > 0:
        likely_issues.append("collision_detected")
    if reference_violation_rate > 0.0:
        likely_issues.append("reference_pipeline_violation")
    if reference_stabilized_rate >= 0.10:
        likely_issues.append("reference_stabilizer_frequently_active")
    if str(speed.get("binding_caps", "")) and "reference_jump" in dict(speed.get("binding_caps", {}) or {}):
        likely_issues.append("speed_limited_by_reference_jump")
    if str(speed.get("binding_caps", "")) and "stop_profile" in dict(speed.get("binding_caps", {}) or {}):
        likely_issues.append("speed_limited_by_stop_profile")
    if speed_rise_limited_rate >= 0.25:
        likely_issues.append("speed_cap_rise_limited_often")
    if mpc_success_rate < 0.90:
        likely_issues.append("mpc_solver_success_rate_low")
    if not bool(status.get("finished", False)):
        likely_issues.append(f"scenario_not_finished:{status.get('reason', 'unknown')}")

    layer_health = {
        "mission": "ok" if bool(status.get("finished", False)) else "warn",
        "safety": "fail" if collision_count > 0 else "ok",
        "reference": _health_from_rate(reference_violation_rate, warn=0.01, fail=0.10),
        "speed": _health_from_rate(speed_rise_limited_rate, warn=0.25, fail=0.60),
        "mpc": "ok" if mpc_success_rate >= 0.97 else ("warn" if mpc_success_rate >= 0.90 else "fail"),
    }

    return {
        "scenario_name": str(scenario_name),
        "finished": bool(status.get("finished", False)),
        "reason": str(status.get("reason", "unknown")),
        "layer_health": layer_health,
        "likely_issues": likely_issues,
        "mission": {
            "sim_elapsed_s": status.get("sim_elapsed_s", ""),
            "destination_distance_m": status.get("destination_distance_m", ""),
            "final_behavior": status.get("final_behavior", ""),
            "final_fsm_state": status.get("final_fsm_state", ""),
        },
        "safety": {
            "collision_count": int(collision_count),
            "min_ttc_s": metrics.get("min_ttc_s", None),
            "max_drac_mps2": metrics.get("max_drac_mps2", None),
        },
        "reference": {
            "samples": reference.get("samples", 0),
            "violation_rate": float(reference_violation_rate),
            "stabilized_rate": float(reference_stabilized_rate),
            "max_reference_jump_m": reference.get("max_reference_jump_m", 0.0),
            "max_first_lateral_abs_m": reference.get("max_first_lateral_abs_m", 0.0),
            "top_violations": reference.get("violations_by_type", {}),
            "top_fallback_reasons": reference.get("fallback_reasons", {}),
            "lateral_sources": reference.get("lateral_sources", {}),
        },
        "speed": {
            "samples": speed.get("samples", 0),
            "rise_limited_rate": float(speed_rise_limited_rate),
            "min_final_cap_mps": speed.get("min_final_cap_mps", ""),
            "binding_caps": speed.get("binding_caps", {}),
            "active_caps": speed.get("active_caps", {}),
            "path_speed_cap_reasons": speed.get("path_speed_cap_reasons", {}),
        },
        "mpc": {
            "plan_attempts": metrics.get("mpc_plan_attempts", status.get("mpc_plan_attempts", 0)),
            "plan_success_rate": float(mpc_success_rate),
            "consecutive_failures": status.get("mpc_consecutive_failures", 0),
        },
    }
