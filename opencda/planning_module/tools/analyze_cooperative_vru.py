"""Summarize cooperative-VRU evidence from a CP-X debug CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _integer(row: dict, key: str) -> int:
    try:
        return int(float(row.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug-dir",
        default=(
            "opencda/planning_module/opencda_bridge/"
            "debug_cpx_c_vru_awareness"
        ),
    )
    args = parser.parse_args()
    debug_dir = Path(args.debug_dir)
    csv_path = debug_dir / "opencda_planner_debug.csv"
    status_path = debug_dir / "run_status.json"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No debug rows found in {csv_path}")
    if "cp_actor_evidence" not in rows[0]:
        raise SystemExit(
            "This log predates cooperative actor evidence fields; rerun the "
            "scenario with the current code."
        )

    actor_stats = defaultdict(
        lambda: {
            "actor_type": "unknown",
            "observed_frames": 0,
            "blind_shared_frames": 0,
            "prediction_used_frames": 0,
            "candidate_relevant_frames": 0,
            "observer_cav_ids": set(),
        }
    )
    for row in rows:
        try:
            evidence_items = json.loads(row.get("cp_actor_evidence") or "[]")
        except json.JSONDecodeError:
            evidence_items = []
        for item in evidence_items:
            actor_id = str(item.get("actor_id", ""))
            if not actor_id:
                continue
            stats = actor_stats[actor_id]
            stats["actor_type"] = str(item.get("actor_type", "unknown"))
            stats["observed_frames"] += 1
            stats["blind_shared_frames"] += int(
                bool(item.get("blind_spot_shared", False))
            )
            stats["prediction_used_frames"] += int(
                bool(item.get("used_by_prediction", False))
            )
            stats["candidate_relevant_frames"] += int(
                bool(item.get("candidate_relevant", False))
            )
            stats["observer_cav_ids"].update(
                str(value)
                for value in list(item.get("observer_cav_ids", []) or [])
            )

    pedestrian_stats = {
        actor_id: stats
        for actor_id, stats in actor_stats.items()
        if stats["actor_type"] == "pedestrian"
    }
    completed = False
    termination_reason = "missing_run_status"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        termination_reason = str(status.get("termination_reason", ""))
        completed = termination_reason == "destination_reached"

    has_pedestrian = bool(pedestrian_stats)
    has_blind_shared_pedestrian = any(
        stats["blind_shared_frames"] > 0
        for stats in pedestrian_stats.values()
    )
    has_predicted_pedestrian = any(
        stats["prediction_used_frames"] > 0
        for stats in pedestrian_stats.values()
    )
    has_candidate_relevant_pedestrian = any(
        stats["candidate_relevant_frames"] > 0
        for stats in pedestrian_stats.values()
    )
    if (
        completed
        and has_blind_shared_pedestrian
        and has_predicted_pedestrian
        and has_candidate_relevant_pedestrian
    ):
        verdict = "PASS"
    elif has_pedestrian:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    serializable_actors = []
    for actor_id, stats in sorted(pedestrian_stats.items()):
        serializable_actors.append({
            "actor_id": actor_id,
            "actor_type": stats["actor_type"],
            "observed_frames": stats["observed_frames"],
            "blind_shared_frames": stats["blind_shared_frames"],
            "prediction_used_frames": stats["prediction_used_frames"],
            "candidate_relevant_frames": stats["candidate_relevant_frames"],
            "observer_cav_ids": sorted(stats["observer_cav_ids"]),
        })
    summary = {
        "verdict": verdict,
        "termination_reason": termination_reason,
        "debug_rows": len(rows),
        "cp_valid_frames": sum(
            str(row.get("cp_message_valid", "")).lower() in {"true", "1"}
            for row in rows
        ),
        "three_cav_observer_frames": sum(
            _integer(row, "cp_observer_cav_count") >= 3 for row in rows
        ),
        "blind_shared_frames": sum(
            _integer(row, "cp_blind_spot_shared_count") > 0 for row in rows
        ),
        "pedestrian_detected": has_pedestrian,
        "blind_shared_pedestrian_detected": has_blind_shared_pedestrian,
        "pedestrian_used_by_prediction": has_predicted_pedestrian,
        "pedestrian_relevant_to_selected_candidate": (
            has_candidate_relevant_pedestrian
        ),
        "pedestrians": serializable_actors,
        "interpretation": (
            "candidate_relevant means the predicted actor entered the "
            "selected trajectory's 3 m safety envelope; it is not a "
            "counterfactual claim that removing the actor changes selection."
        ),
    }
    output_path = debug_dir / "cooperative_vru_evidence_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
