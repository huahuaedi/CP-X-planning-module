"""Compare instrumented closed-loop runs; distinguish latency from episode duration."""
import argparse
import json
from pathlib import Path
import re
import statistics


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def latency(rows, key):
    values = sorted(row[key] for row in rows)
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": values[min(len(values) - 1, int(len(values) * .95))]}


def measure(run):
    run = Path(run)
    frames = read_rows(run / "results/image/cpx_gt/frames.jsonl")
    steady = frames[20:] if len(frames) > 20 else frames[1:]
    execution = json.loads((run / "execution.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    text = run.with_suffix(".log").read_text() if run.with_suffix(".log").exists() else ""
    match = re.search(r"run_scenario: ([\d.]+)s", text)
    return {"run_dir": str(run), "execution": execution,
            "closed_loop_executed": summary["closed_loop_executed"],
            "frame_count": len(frames), "simulation_s": frames[-1]["timestamp"],
            "run_scenario_wall_s": float(match.group(1)) if match else None,
            "first_agent_ms": frames[0]["agent_ms"],
            "steady_agent_ms": latency(steady, "agent_ms"),
            "steady_planning_barrier_ms": latency(steady, "planning_barrier_ms"),
            "steady_worker_sum_ms": latency(steady, "worker_sum_ms"),
            "route_results": [{"ego": r["ego"], "status": r["status"],
                               "route_completion": r["scores"]["score_route"]}
                              for r in summary["route_results"]]}


def compare(serial, parallel):
    a, b = measure(serial), measure(parallel)
    x = read_rows(Path(serial) / "results/image/cpx_gt/steps.jsonl")
    y = read_rows(Path(parallel) / "results/image/cpx_gt/steps.jsonl")
    index = {(round(r["timestamp"], 4), r["ego"]): r for r in y}
    shared = []
    for row in x:
        other = index.get((round(row["timestamp"], 4), row["ego"]))
        if other is not None:
            shared.append((row, other))
    differences = {key: max(abs(r[key] - s[key]) for r, s in shared)
                   for key in ("throttle", "brake", "steer", "x", "y")}
    return {"serial": a, "parallel": b,
            "steady_agent_speedup": a["steady_agent_ms"]["mean"] / b["steady_agent_ms"]["mean"],
            "steady_planning_speedup": a["steady_planning_barrier_ms"]["mean"] / b["steady_planning_barrier_ms"]["mean"],
            "evaluator_wall_speedup": a["execution"]["evaluator_wall_s"] / b["execution"]["evaluator_wall_s"],
            "matched_control_records": len(shared), "max_absolute_differences": differences,
            "note": "First 20 frames excluded from steady latency. Different trajectories/episode lengths "
                    "mean end-to-end ratios are descriptive, not equal-workload speedups. Single-run timings "
                    "on a shared machine are not a statistically repeated benchmark."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("parallel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.serial, args.parallel)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
