#!/usr/bin/env python3
"""Export reproducible evidence for the two-CAV ON/OFF experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEBUG = ROOT / "opencda" / "planning_module" / "opencda_bridge"
DEFAULT_DIRS = {
    "on_cav1": DEBUG / "debug_two_cav_merge_cav1",
    "on_cav2": DEBUG / "debug_two_cav_merge_cav2",
    "off_cav1": DEBUG / "debug_two_cav_merge_off_cav1",
    "off_cav2": DEBUG / "debug_two_cav_merge_off_cav2",
}


def _f(row, key, default=math.nan):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _rows(path: Path):
    csv_path = path / "opencda_planner_debug.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _joined_gap(first, second):
    by_time = {round(_f(r, "sim_time_s"), 2): r for r in second}
    result = []
    for row in first:
        t = round(_f(row, "sim_time_s"), 2)
        peer = by_time.get(t)
        if peer is None:
            continue
        xy = [_f(row, "x_m"), _f(row, "y_m"), _f(peer, "x_m"), _f(peer, "y_m")]
        if all(math.isfinite(v) for v in xy):
            result.append((t, math.hypot(xy[0] - xy[2], xy[1] - xy[3])))
    return result


def _prediction_errors(observer, peer):
    """Return (prediction-issued time, endpoint position error) samples."""

    peer_samples = [(_f(r, "sim_time_s"), r) for r in peer]
    peer_samples = [(t, r) for t, r in peer_samples if math.isfinite(t)]
    result = []
    for row in observer:
        horizon = _f(row, "cav_prediction_validation_horizon_s")
        px = _f(row, "cav_prediction_validation_x_m")
        py = _f(row, "cav_prediction_validation_y_m")
        t = _f(row, "sim_time_s")
        if not all(math.isfinite(v) for v in (horizon, px, py, t)):
            continue
        target_time = t + horizon
        nearest = min(peer_samples, key=lambda item: abs(item[0] - target_time), default=None)
        if nearest is None or abs(nearest[0] - target_time) > 0.08:
            continue
        actual = nearest[1]
        ax, ay = _f(actual, "x_m"), _f(actual, "y_m")
        if math.isfinite(ax) and math.isfinite(ay):
            result.append((t, math.hypot(px - ax, py - ay)))
    return result


def _metric(rows):
    def maximum(key):
        values = [_f(r, key) for r in rows]
        values = [v for v in values if math.isfinite(v)]
        return max(values, default=0.0)

    statuses = [str(r.get("mpc_feasibility_status", "")) for r in rows]
    summaries = [str(r.get("cav_conflict_summary", "")) for r in rows]
    return {
        "ticks": len(rows),
        "duration_s": round(_f(rows[-1], "sim_time_s") - _f(rows[0], "sim_time_s"), 2)
        if len(rows) > 1 else 0.0,
        "max_intent_count": int(maximum("cav_intent_count")),
        "max_shared_plan_count": int(maximum("cav_shared_plan_count")),
        "max_shared_plan_samples": int(maximum("cav_shared_plan_sample_count")),
        "max_longitudinal_rows": int(maximum("cav_longitudinal_qp_row_count")),
        "max_homotopy_rows": int(maximum("cav_homotopy_qp_row_count")),
        "qp_binding_ticks": sum(_f(r, "cav_total_qp_row_count", 0.0) > 0 for r in rows),
        "role_ticks": sum("cav" in s and "=" in s for s in summaries),
        "infeasible_ticks": sum("infeasible" in s.lower() for s in statuses),
        "fallback_ticks": sum(str(r.get("fallback_active", "")).lower() in {"true", "1", "1.0"} for r in rows),
        "hard_brake_ticks": sum(_f(r, "applied_brake", 0.0) >= 0.6 for r in rows),
        "lane_change_ticks": sum("lane_change" in str(r.get("behavior_decision", "")) for r in rows),
        "mean_abs_accel_mps2": round(
            sum(abs(_f(r, "post_supervisor_accel_cmd_mps2", 0.0)) for r in rows)
            / max(1, len(rows)), 3
        ),
    }


def _trajectory_delta(first, second):
    """Maximum same-time position delta between two experiment arms."""

    second_samples = [(_f(r, "sim_time_s"), r) for r in second]
    second_samples = [(t, r) for t, r in second_samples if math.isfinite(t)]
    deltas = []
    for row in first:
        t = _f(row, "sim_time_s")
        if not math.isfinite(t):
            continue
        nearest = min(second_samples, key=lambda item: abs(item[0] - t), default=None)
        if nearest is None or abs(nearest[0] - t) > 0.08:
            continue
        values = (_f(row, "x_m"), _f(row, "y_m"),
                  _f(nearest[1], "x_m"), _f(nearest[1], "y_m"))
        if all(math.isfinite(value) for value in values):
            deltas.append(math.hypot(values[0] - values[2], values[1] - values[3]))
    return max(deltas, default=0.0)


def _plot(data, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"cav1": "#1769aa", "cav2": "#d1495b"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for arm, style in (("off", "--"), ("on", "-")):
        for cav in ("cav1", "cav2"):
            rows = data[f"{arm}_{cav}"]
            axes[0, 0].plot([_f(r, "x_m") for r in rows], [_f(r, "y_m") for r in rows],
                            style, color=colors[cav], label=f"{arm.upper()} {cav}")
            axes[0, 1].plot([_f(r, "sim_time_s") for r in rows], [_f(r, "speed_mps") for r in rows],
                            style, color=colors[cav], label=f"{arm.upper()} {cav}")
        gap = _joined_gap(data[f"{arm}_cav1"], data[f"{arm}_cav2"])
        axes[1, 0].plot([x[0] for x in gap], [x[1] for x in gap], style, label=arm.upper())
    for cav in ("cav1", "cav2"):
        rows = data[f"on_{cav}"]
        t = [_f(r, "sim_time_s") for r in rows]
        axes[1, 1].plot(t, [_f(r, "cav_longitudinal_qp_row_count", 0.0) for r in rows],
                        color=colors[cav], label=f"{cav} longitudinal")
        axes[1, 1].plot(t, [_f(r, "cav_homotopy_qp_row_count", 0.0) for r in rows],
                        ":", color=colors[cav], label=f"{cav} homotopy")
    titles = ["Vehicle trajectories", "Speed response", "Inter-CAV center gap", "Active cooperative QP rows"]
    labels = [("x (m)", "y (m)"), ("time (s)", "speed (m/s)"),
              ("time (s)", "gap (m)"), ("time (s)", "row count")]
    for axis, title, (xlabel, ylabel) in zip(axes.flat, titles, labels):
        axis.set_title(title); axis.set_xlabel(xlabel); axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "cav_interaction_overview.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for cav in ("cav1", "cav2"):
        rows = data[f"on_{cav}"]; t = [_f(r, "sim_time_s") for r in rows]
        axes[0].plot(t, [_f(r, "cav_intent_count", 0.0) for r in rows], label=f"{cav} intents")
        axes[0].plot(t, [_f(r, "cav_shared_plan_count", 0.0) for r in rows], "--", label=f"{cav} shared plans")
        axes[1].plot(t, [_f(r, "post_supervisor_accel_cmd_mps2", 0.0) for r in rows], label=cav)
        axes[2].plot(t, [_f(r, "steer_cmd_rad", 0.0) for r in rows], label=cav)
    for axis, ylabel in zip(axes, ("received count", "accel (m/s²)", "steer (rad)")):
        axis.set_ylabel(ylabel); axis.grid(True, alpha=0.25); axis.legend()
    axes[2].set_xlabel("simulation time (s)")
    fig.suptitle("Prediction transport and MPC control response (ON)")
    fig.tight_layout()
    fig.savefig(output / "cav_prediction_and_control.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4.8))
    for observer, peer in (("cav1", "cav2"), ("cav2", "cav1")):
        errors = _prediction_errors(data[f"on_{observer}"], data[f"on_{peer}"])
        axis.plot([v[0] for v in errors], [v[1] for v in errors], label=f"{observer} predicts {peer}")
    axis.set_title("Shared MPC-plan prediction error at 1 s horizon")
    axis.set_xlabel("prediction issue time (s)"); axis.set_ylabel("position error (m)")
    axis.grid(True, alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(output / "cav_prediction_accuracy.png", dpi=170)
    plt.close(fig)


def _encode_video(frames: Path, output: Path, fps: int):
    if not frames.is_dir() or not next(frames.glob("frame_*.png"), None):
        return False
    frame_limit = None
    try:
        manifest = json.loads((frames / "capture_manifest.json").read_text(encoding="utf-8"))
        frame_limit = max(0, int(manifest.get("frame_count", 0)))
    except (OSError, ValueError, TypeError):
        pass
    command = ["ffmpeg", "-y", "-framerate", str(fps), "-i",
               str(frames / "frame_%06d.png")]
    if frame_limit:
        command += ["-frames:v", str(frame_limit)]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _combine_videos(left: Path, right: Path, output: Path):
    if not left.is_file() or not right.is_file():
        return False
    command = [
        "ffmpeg", "-y", "-i", str(left), "-i", str(right),
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS[left];[1:v]setpts=PTS-STARTPTS[right];"
        "[left][right]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-shortest", str(output),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "cav_interaction_report")
    parser.add_argument("--on-frames", type=Path)
    parser.add_argument("--off-frames", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    for key, value in DEFAULT_DIRS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=Path, default=value)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {key: _rows(getattr(args, key)) for key in DEFAULT_DIRS}
    if not all(data.values()):
        raise SystemExit("missing one or more ON/OFF CAV debug CSVs")
    metrics = {key: _metric(rows) for key, rows in data.items()}
    metrics["counterfactual"] = {
        cav: {
            "max_on_off_position_delta_m": round(
                _trajectory_delta(data[f"on_{cav}"], data[f"off_{cav}"]), 3
            )
        }
        for cav in ("cav1", "cav2")
    }
    for arm in ("on", "off"):
        gap = _joined_gap(data[f"{arm}_cav1"], data[f"{arm}_cav2"])
        minimum_gap = min((v for _, v in gap), default=None)
        metrics[arm] = {
            "minimum_inter_cav_gap_m": round(minimum_gap, 3) if minimum_gap is not None else None,
            "mean_inter_cav_gap_m": round(sum(v for _, v in gap) / len(gap), 3) if gap else None,
        }
    prediction_errors = []
    for observer, peer in (("cav1", "cav2"), ("cav2", "cav1")):
        values = _prediction_errors(data[f"on_{observer}"], data[f"on_{peer}"])
        metrics[f"on_{observer}"]["prediction_validation_samples"] = len(values)
        metrics[f"on_{observer}"]["prediction_rmse_m"] = (
            round(math.sqrt(sum(v * v for _, v in values) / len(values)), 3)
            if values else None
        )
        prediction_errors.extend(values)
    on = [metrics["on_cav1"], metrics["on_cav2"]]
    evidence = {
        "prediction_received": any(m["max_shared_plan_count"] > 0 for m in on),
        "prediction_validated_against_future_pose": bool(prediction_errors),
        "cooperative_role_activated": any(m["role_ticks"] > 0 for m in on),
        "longitudinal_corridor_bound": any(m["max_longitudinal_rows"] > 0 for m in on),
        "homotopy_halfspace_bound": any(m["max_homotopy_rows"] > 0 for m in on),
        "on_run_without_mpc_infeasible": all(m["infeasible_ticks"] == 0 for m in on),
        "on_off_control_response_differs": any(
            value["max_on_off_position_delta_m"] >= 0.25
            for value in metrics["counterfactual"].values()
        ),
    }
    metrics["evidence"] = evidence
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot(data, args.output)
    videos = []
    if args.on_frames and _encode_video(args.on_frames, args.output / "cooperative_on.mp4", args.fps):
        videos.append("cooperative_on.mp4")
    if args.off_frames and _encode_video(args.off_frames, args.output / "cooperative_off.mp4", args.fps):
        videos.append("cooperative_off.mp4")
    if _combine_videos(
        args.output / "cooperative_on.mp4",
        args.output / "cooperative_off.mp4",
        args.output / "cooperative_on_vs_off.mp4",
    ):
        videos.append("cooperative_on_vs_off.mp4")
    verdict = "PASS" if all(evidence.values()) else "NOT PROVEN"
    report = ["# Two-CAV prediction-aware MPC experiment", "", f"Verdict: **{verdict}**", "",
              "## Evidence"]
    report += [f"- {key}: `{value}`" for key, value in evidence.items()]
    report += [
        "", "## ON/OFF metrics", "",
        "| Run | QP binding ticks | Infeasible | Fallback | Hard brake | Lane change | Mean |a| |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("on", "off"):
        for cav in ("cav1", "cav2"):
            item = metrics[f"{arm}_{cav}"]
            report.append(
                f"| {arm.upper()} {cav} | {item['qp_binding_ticks']} | "
                f"{item['infeasible_ticks']} | {item['fallback_ticks']} | "
                f"{item['hard_brake_ticks']} | {item['lane_change_ticks']} | "
                f"{item['mean_abs_accel_mps2']:.3f} |"
            )
    report += ["", "## Prediction validation", ""]
    for observer in ("cav1", "cav2"):
        item = metrics[f"on_{observer}"]
        report.append(
            f"- {observer}: {item['prediction_validation_samples']} aligned "
            f"samples, 1 s position RMSE = {item['prediction_rmse_m']} m"
        )
    report += ["", "## Counterfactual response", ""]
    for cav, item in metrics["counterfactual"].items():
        report.append(
            f"- {cav}: max ON/OFF aligned position difference = "
            f"{item['max_on_off_position_delta_m']} m"
        )
    report += ["", "## Artifacts", "", "- `cav_interaction_overview.png`",
               "- `cav_prediction_and_control.png`", "- `cav_prediction_accuracy.png`",
               "- `metrics.json`"]
    report += [f"- `{name}`" for name in videos]
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"{verdict}: {args.output}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
