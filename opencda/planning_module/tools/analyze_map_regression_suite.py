"""Aggregate Map Contract results for the deterministic regression suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_map_matching import analyze


SCENARIOS = {
    "cpx_mapreg_town06_straight": "mapreg_town06_straight",
    "cpx_mapreg_town06_right_turn": "mapreg_town06_right_turn",
    "cpx_mapreg_town05_turn": "mapreg_town05_turn",
}

FAILURE_COUNTS = (
    "unmatched_rows",
    "local_frame_invariant_violation_count",
    "downstream_direction_conflict_count",
    "unauthorized_lateral_transition_count",
    "short_lane_bounce_count",
    "map_contract_schema_missing_field_count",
    "map_contract_violation_count",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug-root",
        type=Path,
        default=Path("opencda/planning_module/opencda_bridge"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-violations", action="store_true")
    args = parser.parse_args()

    results: dict[str, object] = {}
    missing: list[str] = []
    failed: list[str] = []
    for scenario, directory in SCENARIOS.items():
        csv_path = args.debug_root / directory / "opencda_planner_debug.csv"
        if not csv_path.is_file():
            missing.append(scenario)
            results[scenario] = {"status": "missing", "debug_csv": str(csv_path)}
            continue
        summary = analyze(csv_path)
        violations = {
            key: int(summary.get(key, 0) or 0)
            for key in FAILURE_COUNTS
        }
        passed = not any(violations.values())
        if not passed:
            failed.append(scenario)
        results[scenario] = {
            "status": "passed" if passed else "failed",
            "debug_csv": str(csv_path),
            "rows": int(summary.get("rows", 0) or 0),
            "minimum_confidence": summary.get("minimum_confidence"),
            "ad_lane_transition_count": int(
                summary.get("ad_lane_transition_count", 0) or 0
            ),
            "failure_counts": violations,
        }
    report = {
        "suite_passed": not missing and not failed,
        "scenario_count": len(SCENARIOS),
        "passed_count": len(SCENARIOS) - len(missing) - len(failed),
        "missing_scenarios": missing,
        "failed_scenarios": failed,
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_violations and not bool(report["suite_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
