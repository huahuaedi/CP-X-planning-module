#!/usr/bin/env python3
"""Evaluate one conflict run across perception, planning and execution.

The evaluator deliberately distinguishes an invalid scenario (the requested
conflict never occurred) from an unsafe run and from a planner failure.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

try:
    from opencda.scenario_testing.planner_debug_records import (
        flag, load_planner_records, number,
    )
except ModuleNotFoundError:  # direct ``python path/to/script.py`` execution
    from planner_debug_records import flag, load_planner_records, number


def _finite_values(records, key):
    values = [number(record, key) for record in records]
    return [value for value in values if math.isfinite(value)]


def _tags(record: Mapping[str, Any]) -> Iterable[str]:
    structured = record.get("cav_conflict_tags", {})
    if isinstance(structured, Mapping):
        return [str(value) for value in structured.values()]
    # Compatibility only for records produced before structured JSONL fields.
    summary = str(record.get("cav_conflict_summary", "") or "")
    return [part.split(":", 1)[1] for part in summary.split(";")
            if ":" in part and not part.startswith("corridor_")]


def _roles(record: Mapping[str, Any]) -> Iterable[str]:
    structured = record.get("cav_conflict_roles", {})
    if isinstance(structured, Mapping):
        return [str(value) for value in structured.values()]
    return []


def analyze_conflict_run(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_tags: Sequence[str] = (),
    require_constraint: bool = True,
) -> Dict[str, Any]:
    """Return deterministic metrics and an explicit run verdict."""

    rows = list(records or ())
    tag_counts = Counter(tag for row in rows for tag in _tags(row))
    role_counts = Counter(role for row in rows for role in _roles(row))
    times = _finite_values(rows, "sim_time_s")
    accels = _finite_values(rows, "measured_accel_mps2")
    gaps = _finite_values(rows, "nearest_ttc_bumper_gap_m")
    if not gaps:
        gaps = _finite_values(rows, "front_gap_m")
    ttcs = [value for value in _finite_values(rows, "nearest_ttc_s")
            if value >= 0.0]
    constraint_indices = [index for index, row in enumerate(rows)
                          if number(row, "cav_total_qp_row_count", 0.0) > 0.0]
    veto_indices = [index for index, row in enumerate(rows)
                    if number(row, "cav_credible_mode_veto_count", 0.0) > 0.0]
    fallback_ticks = sum(flag(row, "fallback_active") for row in rows)
    infeasible_ticks = sum(
        "infeasible" in str(row.get("mpc_status", "")).lower()
        for row in rows
    )
    collision_count = int(max(
        _finite_values(rows, "collision_count") or [0.0]
    ))
    jerk = []
    for index in range(1, min(len(times), len(accels))):
        dt_s = times[index] - times[index - 1]
        if dt_s > 1.0e-4:
            jerk.append(abs(accels[index] - accels[index - 1]) / dt_s)

    expected = {str(tag).upper() for tag in expected_tags}
    observed = {str(tag).upper() for tag in tag_counts}
    scenario_valid = bool(rows) and expected.issubset(observed)
    constraint_valid = bool(constraint_indices) if require_constraint else True
    if not rows:
        verdict = "NO_DATA"
    elif not scenario_valid:
        verdict = "INVALID_SCENARIO"
    elif collision_count > 0:
        verdict = "UNSAFE_COLLISION"
    elif fallback_ticks or infeasible_ticks:
        verdict = "PLANNER_FAILURE"
    elif not constraint_valid:
        verdict = "NO_CONSTRAINT_RESPONSE"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "scenario_valid": scenario_valid,
        "constraint_response_valid": constraint_valid,
        "ticks": len(rows),
        "duration_s": round(times[-1] - times[0], 3) if len(times) > 1 else 0.0,
        "tag_ticks": dict(sorted(tag_counts.items())),
        "role_ticks": dict(sorted(role_counts.items())),
        "constraint_ticks": len(constraint_indices),
        "first_constraint_tick": constraint_indices[0] if constraint_indices else None,
        "credible_veto_ticks": len(veto_indices),
        "first_credible_veto_tick": veto_indices[0] if veto_indices else None,
        "collision_count": collision_count,
        "min_bumper_gap_m": round(min(gaps), 3) if gaps else None,
        "min_ttc_s": round(min(ttcs), 3) if ttcs else None,
        "peak_decel_mps2": round(min(accels), 3) if accels else None,
        "peak_abs_jerk_mps3": round(max(jerk), 3) if jerk else None,
        "fallback_ticks": fallback_ticks,
        "mpc_infeasible_ticks": infeasible_ticks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("debug_dir", type=Path)
    parser.add_argument("--expect-tag", action="append", default=[])
    parser.add_argument("--allow-no-constraint", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_conflict_run(
        load_planner_records(args.debug_dir),
        expected_tags=args.expect_tag,
        require_constraint=not args.allow_no_constraint,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
