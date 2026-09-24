"""Combine MDrive route results and CP-X diagnostics without changing MDrive."""
import argparse
import json
from pathlib import Path


def summarize(run_dir):
    run_dir = Path(run_dir)
    results = []
    for path in sorted(run_dir.glob("results/ego_vehicle_*/results.json")):
        payload = json.loads(path.read_text())
        for record in payload.get("_checkpoint", {}).get("records", []):
            results.append({"ego": int(path.parent.name.rsplit("_", 1)[-1]),
                            "status": record.get("status"), "scores": record.get("scores"),
                            "infractions": record.get("infractions"), "meta": record.get("meta")})
    stats = {}
    for path in run_dir.glob("results/image/cpx_gt/steps.jsonl"):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                ego = row["ego"]
                item = stats.setdefault(ego, {"control_frames": 0, "first_xy": [row["x"], row["y"]],
                                             "last_xy": [], "solver_status_frames": {}, "max_step_ms": 0.0})
                item["control_frames"] += 1
                item["last_xy"] = [row["x"], row["y"]]
                item["max_step_ms"] = max(item["max_step_ms"], row["planning_ms"])
                status = row["status"]["solver_status"]
                item["solver_status_frames"][status] = item["solver_status_frames"].get(status, 0) + 1
    valid = bool(results) and all(
        r["ego"] in stats and stats[r["ego"]]["control_frames"] > 1
        and not any(word in str(r["status"]).lower() for word in ("crash", "reject", "setup"))
        for r in results)
    report = {"perception": "gt_current_state_oracle", "closed_loop_executed": valid,
              "route_results": results, "planner_diagnostics": stats,
              "note": "Successful execution does not imply collision-free driving or route completion. "
                      "solver_status_frames counts control frames, not separate MPC solves."}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), indent=2))
