"""Single-frame MPC replay + QP-row audit for the interaction stack.

Purpose
-------
Investigate the "urgent conflict -> abnormal lateral deflection -> primal
infeasible -> persistent stop" failure *offline*, one frozen tick at a
time, without CARLA. Two things this gives you:

1. ``audit_rows`` -- reconstruct every Stage-D QP row to world coordinates
   and measure the angle between the row normal ``(a_x, a_y)`` and the
   local tangent of each candidate reference line at that stage's station.
   A longitudinal corridor band should have its normal *parallel* to the
   path tangent (``lateral_coupling_deg`` ~ 0); a lateral keep-out row
   should be *perpendicular* (~ 90). Anything in between means the row is
   pushing the solver sideways. The audit also reports the tangent
   disagreement between the reference the rows were *built* on and the
   reference the MPC actually *tracks* -- the suspected root cause.

2. ``replay`` / ``replay_matrix`` -- re-solve the exact frame with knobs:
   conflict rows on/off, obstacle repulsive potential on/off, previous-
   solution warm start on/off, and corridor rows rebuilt on the pre- vs
   post-publication reference. Compare the solved heading / lateral error
   to isolate which input causes the first deflection.

This module does not decide the root cause and applies no fix. It is the
measurement harness for plan steps 1-2.

Frame capture
-------------
``FrameCapture`` is a plain-data record of one tick's MPC + CAV-pipeline
inputs. Capture it live from the bridge (see ``frame_capture_ticks`` in
``cpx_mpc_planner``) or build one synthetically (see the test module).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Tests and existing tools mix ``from pipeline...`` / ``from MPC...`` with
# ``from opencda.planning_module...``; put both roots on the path so this
# file runs directly (``python tools/frame_replay.py ...``) as well as
# under pytest.
_PM_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PM_DIR.parents[1]
for _p in (str(_PM_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from pipeline.mpc_obstacle_relevance import _point_to_polyline, _polyline_xy  # noqa: E402
from pipeline.reference_geometry import pose_at_arc  # noqa: E402
from pipeline.spatiotemporal_corridor import Corridor  # noqa: E402
from pipeline.mpc_corridor_constraints import (  # noqa: E402
    corridor_rows as build_corridor_rows,
    homotopy_keepout_rows,
)

XY = Tuple[float, float]

# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------


@dataclass
class FrameCapture:
    """Everything needed to reproduce one tick's MPC solve offline."""

    tick: int = -1
    sim_time_s: float = 0.0

    # Ego. ``current_state`` is the MPC state vector [x, y, v, psi] in world
    # frame; ``ego_origin_xy`` is the world point Stage D was centred on
    # (must equal ``current_state[:2]`` for a consistent frame).
    current_state: Sequence[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    ego_origin_xy: XY = (0.0, 0.0)
    ego_yaw_rad: float = 0.0
    ego_speed_mps: float = 0.0
    current_acceleration_mps2: float = 0.0
    current_steering_rad: float = 0.0

    destination_state: Sequence[float] = field(default_factory=list)
    target_speed_mps: float = 0.0
    stop_goal_active: bool = False
    behavior_maneuver: str = "lane_follow"
    behavior_phase: str = ""

    # Two reference polylines. ``pre_publication_reference`` is what Stage
    # C/D actually used this tick (``local_lane_center_reference`` inside
    # ``_plan_behavior_and_reference``). ``published_reference`` is what
    # ``execute_mpc`` tracks (post reference-publication stage). When these
    # differ, corridor rows are built on one curve and tracked on another.
    pre_publication_reference: List[Dict[str, Any]] = field(default_factory=list)
    published_reference: List[Dict[str, Any]] = field(default_factory=list)

    # Obstacles fed to the MPC objective (repulsive potential + reference
    # obstacle-aware speed). Same schema as ``mpc.plan_trajectory``.
    mpc_object_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    # Stage-C semantic corridor (arc-length bands on the pre-publication
    # reference). Preferred over ``mpc_rows`` because it lets the replay
    # re-project onto either reference.
    corridor_s_lo: List[float] = field(default_factory=list)
    corridor_s_hi: List[float] = field(default_factory=list)
    corridor_binding: List[str] = field(default_factory=list)

    # Latched homotopy pass-side assignments (role == "proceed" only) and
    # per-stage CAV tracks, for ``homotopy_keepout_rows``.
    homotopy_assignments: List[Dict[str, Any]] = field(default_factory=list)
    cav_tracks_by_stage: Dict[str, List[XY]] = field(default_factory=dict)
    homotopy_d_safe_m: float = 3.0

    # Precomputed rows actually sent to MPC this tick (fallback when the
    # Stage-C corridor was not captured). Each row: dict with stage/a_x/a_y/
    # lower/upper/slack_group/tag.
    mpc_rows: List[Dict[str, Any]] = field(default_factory=list)

    # Previous tick's control solution, for a faithful warm start.
    prev_u_solution: Optional[List[List[float]]] = None

    # MPC config. ``mpc_config`` inline wins; else ``mpc_config_path``; else
    # the repo default MPC/mpc.yaml.
    mpc_config: Optional[Dict[str, Any]] = None
    mpc_config_path: Optional[str] = None

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    @staticmethod
    def from_json(path: str | Path) -> "FrameCapture":
        raw = json.loads(Path(path).read_text())
        known = {f for f in FrameCapture.__dataclass_fields__}  # type: ignore[attr-defined]
        return FrameCapture(**{k: v for k, v in raw.items() if k in known})

    # -- helpers ---------------------------------------------------------
    def reference(self, which: str) -> List[Dict[str, Any]]:
        if which == "pre":
            return list(self.pre_publication_reference)
        if which == "post":
            return list(self.published_reference or self.pre_publication_reference)
        raise ValueError("which must be 'pre' or 'post'")

    def has_corridor(self) -> bool:
        return len(self.corridor_s_hi) > 0 and len(self.corridor_s_hi) == len(
            self.corridor_s_lo
        )

    def corridor(self) -> Corridor:
        n = len(self.corridor_s_hi)
        binding = list(self.corridor_binding) + [""] * max(0, n - len(self.corridor_binding))
        return Corridor(
            s_lo=[float(v) for v in self.corridor_s_lo],
            s_hi=[float(v) for v in self.corridor_s_hi],
            binding=binding[:n],
        )


# ---------------------------------------------------------------------------
# Row audit
# ---------------------------------------------------------------------------


def _unit(vx: float, vy: float) -> XY:
    n = math.hypot(vx, vy)
    if n <= 1e-12:
        return (1.0, 0.0)
    return (vx / n, vy / n)


def _tangent_at_station(poly_world: Sequence[XY], station_m: float) -> XY:
    _, _, heading = pose_at_arc(list(poly_world), float(station_m))
    return (math.cos(heading), math.sin(heading))


def _angle_between_deg(a: XY, b: XY) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    return math.degrees(math.acos(abs(dot)))  # 0 == parallel, 90 == perpendicular


@dataclass
class RowAudit:
    stage: int
    slack_group: str
    tag: str
    a_x: float
    a_y: float
    lower: float
    upper: float
    kind: str  # "longitudinal_band" | "half_space"
    # World half-plane: a_x*(X-ox) + a_y*(Y-oy) in [lower, upper].
    world_expr: str
    station_on_pre_m: float
    station_on_post_m: float
    # Angle(row normal, reference tangent). Longitudinal band -> want ~0 on
    # its own reference; half-space -> want ~90.
    normal_vs_pre_tangent_deg: float
    normal_vs_post_tangent_deg: float
    # How far the two references' tangents have rotated apart at this row.
    pre_post_tangent_disagreement_deg: float
    # Signed lateral push implied by the row if the solver rides the post
    # reference: component of the (unit) normal perpendicular to the post
    # tangent. 0 for a clean longitudinal band.
    lateral_coupling_deg: float
    flagged: bool


def audit_rows(
    rows: Sequence[Any],
    ego_origin_xy: XY,
    *,
    pre_reference: Sequence[Any],
    post_reference: Sequence[Any],
    couple_flag_deg: float = 5.0,
    stage_stations_m: Optional[Mapping[int, float]] = None,
) -> List[RowAudit]:
    """Reconstruct each row to world coords and score its lateral coupling.

    ``stage_stations_m`` optionally maps MPC stage -> along-track station of
    the linearization rollout at that stage. When a real frame is replayed,
    pass ``ReplayResult`` rollout stations here so the tangent is sampled at
    the point the solver actually linearized about, instead of the finite
    bound magnitude (a coarse proxy that is exact only when every band
    shares one cap, e.g. a crossing yield).
    """

    ox, oy = float(ego_origin_xy[0]), float(ego_origin_xy[1])
    pre_world = _polyline_xy(pre_reference)
    post_world = _polyline_xy(post_reference)
    pre_ego = [(x - ox, y - oy) for (x, y) in pre_world]
    post_ego = [(x - ox, y - oy) for (x, y) in post_world]

    out: List[RowAudit] = []
    for row in rows:
        stage = int(_row_get(row, "stage"))
        a_x = float(_row_get(row, "a_x"))
        a_y = float(_row_get(row, "a_y"))
        lower = float(_row_get(row, "lower", -math.inf))
        upper = float(_row_get(row, "upper", math.inf))
        slack_group = str(_row_get(row, "slack_group", ""))
        tag = str(_row_get(row, "tag", ""))
        half_space = not (math.isfinite(lower) and math.isfinite(upper))
        kind = "half_space" if half_space else "longitudinal_band"

        normal = _unit(a_x, a_y)
        # Station of this row. Prefer a caller-supplied rollout station;
        # else the finite bound magnitude (the corridor cap), else the ego
        # origin's own projection.
        est_station = 0.0
        if stage_stations_m is not None and stage in stage_stations_m:
            est_station = float(stage_stations_m[stage])
        else:
            for bound in (upper, lower):
                if math.isfinite(bound):
                    est_station = abs(bound)
                    break
        s_pre = _project_station(pre_ego, est_station)
        s_post = _project_station(post_ego, est_station)

        t_pre = _tangent_at_station(pre_ego, s_pre)
        t_post = _tangent_at_station(post_ego, s_post)

        ang_pre = _angle_between_deg(normal, t_pre)
        ang_post = _angle_between_deg(normal, t_post)
        disagree = _angle_between_deg(t_pre, t_post)

        if half_space:
            # Want perpendicular to the tracked path: coupling is deviation
            # from 90 degrees.
            coupling = abs(90.0 - ang_post)
        else:
            # Want parallel to the tracked path: coupling is deviation from
            # 0 degrees.
            coupling = ang_post

        lo = "-inf" if not math.isfinite(lower) else f"{lower:.3f}"
        hi = "+inf" if not math.isfinite(upper) else f"{upper:.3f}"
        expr = f"{a_x:.4f}*(X-{ox:.2f}) + {a_y:.4f}*(Y-{oy:.2f}) in [{lo}, {hi}]"

        out.append(
            RowAudit(
                stage=stage,
                slack_group=slack_group,
                tag=tag,
                a_x=a_x,
                a_y=a_y,
                lower=lower,
                upper=upper,
                kind=kind,
                world_expr=expr,
                station_on_pre_m=float(s_pre),
                station_on_post_m=float(s_post),
                normal_vs_pre_tangent_deg=float(ang_pre),
                normal_vs_post_tangent_deg=float(ang_post),
                pre_post_tangent_disagreement_deg=float(disagree),
                lateral_coupling_deg=float(coupling),
                flagged=bool(coupling > float(couple_flag_deg)),
            )
        )
    return out


def _project_station(poly_ego: Sequence[XY], target: float) -> float:
    if len(poly_ego) < 2:
        return 0.0
    total = 0.0
    for i in range(len(poly_ego) - 1):
        total += math.hypot(
            poly_ego[i + 1][0] - poly_ego[i][0],
            poly_ego[i + 1][1] - poly_ego[i][1],
        )
    if target <= 0.0:
        # No finite bound: use the ego origin's own projection.
        return _point_to_polyline(0.0, 0.0, list(poly_ego))[1]
    return min(total, max(0.0, target))


def _row_get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def audit_summary(audits: Sequence[RowAudit]) -> Dict[str, Any]:
    bands = [a for a in audits if a.kind == "longitudinal_band"]
    spaces = [a for a in audits if a.kind == "half_space"]
    flagged = [a for a in audits if a.flagged]
    worst = max(audits, key=lambda a: a.lateral_coupling_deg, default=None)
    return {
        "row_count": len(audits),
        "longitudinal_band_count": len(bands),
        "half_space_count": len(spaces),
        "flagged_row_count": len(flagged),
        "max_lateral_coupling_deg": (worst.lateral_coupling_deg if worst else 0.0),
        "max_pre_post_tangent_disagreement_deg": max(
            (a.pre_post_tangent_disagreement_deg for a in audits), default=0.0
        ),
        "flagged_stages": sorted({a.stage for a in flagged}),
    }


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _load_mpc_config(capture: FrameCapture) -> Dict[str, Any]:
    if capture.mpc_config is not None:
        return json.loads(json.dumps(capture.mpc_config))  # deep copy
    path = (
        Path(capture.mpc_config_path)
        if capture.mpc_config_path
        else _PM_DIR / "MPC" / "mpc.yaml"
    )
    return yaml.safe_load(Path(path).read_text())


def _apply_toggles(
    cfg: Dict[str, Any], *, conflict_rows: bool, repulsive: bool, warm_start: bool
) -> Dict[str, Any]:
    mpc_cfg = cfg.setdefault("mpc", {})
    cost = mpc_cfg.setdefault("cost", {})
    cost.setdefault("corridor", {})
    cost["corridor"]["enabled"] = bool(conflict_rows)
    cost.setdefault("repulsive_potential", {})
    cost["repulsive_potential"]["enabled"] = bool(repulsive)
    mpc_cfg.setdefault("reference_rollout", {})
    mpc_cfg["reference_rollout"]["use_previous_solution_seed"] = bool(warm_start)
    return cfg


def _rows_for(
    capture: FrameCapture, corridor_ref: str
) -> Tuple[List[Any], List[Any]]:
    """(all_rows, longitudinal_only) for the chosen reference."""

    origin = tuple(capture.ego_origin_xy)
    if capture.has_corridor():
        ref = capture.reference(corridor_ref)
        longitudinal = build_corridor_rows(capture.corridor(), ref, ego_origin_xy=origin)
    elif corridor_ref == "pre" and capture.mpc_rows:
        longitudinal = [
            _MutRow(r) for r in capture.mpc_rows
            if math.isfinite(float(r.get("lower", "-inf") or "-inf"))
            or math.isfinite(float(r.get("upper", "inf") or "inf"))
        ]
    else:
        longitudinal = []
    lateral: List[Any] = []
    if capture.homotopy_assignments:
        tracks = {
            int(k): [tuple(p) for p in v]
            for k, v in capture.cav_tracks_by_stage.items()
        }
        lateral = homotopy_keepout_rows(
            [_MutRow(a) for a in capture.homotopy_assignments],
            tracks,
            ego_heading_rad=float(capture.ego_yaw_rad),
            ego_origin_xy=origin,
            d_safe_m=float(capture.homotopy_d_safe_m),
        )
    return list(longitudinal) + list(lateral), list(longitudinal)


class _MutRow(dict):
    """dict with attribute access, for feeding captured rows/assignments to
    the pure Stage-D builders that use ``getattr``."""

    __getattr__ = dict.get  # type: ignore[assignment]


@dataclass
class ReplayResult:
    conflict_rows: bool
    repulsive: bool
    warm_start: bool
    corridor_ref: str
    solver_status: str
    row_count: int
    solved: bool
    final_progress_m: float
    max_abs_heading_error_deg: float
    max_abs_lateral_error_m: float
    heading_error_deg_by_stage: List[float]
    lateral_error_m_by_stage: List[float]
    trajectory_world: List[List[float]]

    def label(self) -> str:
        return (
            f"rows={'on ' if self.conflict_rows else 'off'} "
            f"pot={'on ' if self.repulsive else 'off'} "
            f"warm={'on ' if self.warm_start else 'off'} "
            f"ref={self.corridor_ref}"
        )


def replay(
    capture: FrameCapture,
    *,
    conflict_rows: bool = True,
    repulsive: bool = True,
    warm_start: bool = True,
    corridor_ref: str = "pre",
) -> ReplayResult:
    """Re-solve the frame with one combination of toggles."""

    from MPC.mpc import MPC  # local import: heavy, path set above

    cfg = _apply_toggles(
        _load_mpc_config(capture),
        conflict_rows=conflict_rows,
        repulsive=repulsive,
        warm_start=warm_start,
    )
    mpc = MPC(cfg["mpc"], cfg.get("road", {}))

    if warm_start and capture.prev_u_solution:
        mpc._last_u_solution = np.asarray(capture.prev_u_solution, dtype=float)
    else:
        if hasattr(mpc, "clear_previous_solution_seed"):
            mpc.clear_previous_solution_seed()

    all_rows, _ = _rows_for(capture, corridor_ref)
    rows_arg = all_rows if conflict_rows else None

    tracked_ref = capture.reference("post")
    traj = mpc.plan_trajectory(
        current_state=list(capture.current_state),
        destination_state=list(capture.destination_state),
        object_snapshots=[dict(o) for o in capture.mpc_object_snapshots],
        current_acceleration_mps2=float(capture.current_acceleration_mps2),
        current_steering_rad=float(capture.current_steering_rad),
        lane_center_reference_samples=[dict(s) for s in tracked_ref],
        stop_goal_active=bool(capture.stop_goal_active),
        corridor_rows=rows_arg,
    )
    status = str(getattr(mpc, "_last_status", "") or "")
    solved = "solved" in status.lower() if status else bool(traj)

    ref_world = _polyline_xy(tracked_ref)
    heading_err: List[float] = []
    lateral_err: List[float] = []
    for state in traj or []:
        x, y = float(state[0]), float(state[1])
        psi = float(state[3]) if len(state) > 3 else 0.0
        perp, station = _point_to_polyline(x, y, ref_world)
        _, _, ref_heading = pose_at_arc(list(ref_world), station)
        dpsi = math.atan2(
            math.sin(psi - ref_heading), math.cos(psi - ref_heading)
        )
        heading_err.append(math.degrees(dpsi))
        lateral_err.append(perp)

    start = ref_world[0] if ref_world else (0.0, 0.0)
    final_progress = 0.0
    if traj:
        final_progress = _point_to_polyline(
            float(traj[-1][0]), float(traj[-1][1]), ref_world
        )[1] if ref_world else 0.0
    _ = start

    return ReplayResult(
        conflict_rows=conflict_rows,
        repulsive=repulsive,
        warm_start=warm_start,
        corridor_ref=corridor_ref,
        solver_status=status,
        row_count=len(all_rows),
        solved=bool(solved),
        final_progress_m=float(final_progress),
        max_abs_heading_error_deg=float(max((abs(v) for v in heading_err), default=0.0)),
        max_abs_lateral_error_m=float(max((abs(v) for v in lateral_err), default=0.0)),
        heading_error_deg_by_stage=[round(v, 4) for v in heading_err],
        lateral_error_m_by_stage=[round(v, 4) for v in lateral_err],
        trajectory_world=[[float(c) for c in state] for state in (traj or [])],
    )


def replay_matrix(
    capture: FrameCapture,
    *,
    warm_start_values: Sequence[bool] = (True, False),
    corridor_ref_values: Sequence[str] = ("pre", "post"),
) -> List[ReplayResult]:
    """The A/B grid: rows x potential x warm-start x corridor-ref."""

    results: List[ReplayResult] = []
    for rows_on in (True, False):
        for pot_on in (True, False):
            for warm in warm_start_values:
                refs = corridor_ref_values if rows_on else ("pre",)
                for ref in refs:
                    results.append(
                        replay(
                            capture,
                            conflict_rows=rows_on,
                            repulsive=pot_on,
                            warm_start=warm,
                            corridor_ref=ref,
                        )
                    )
    return results


def format_matrix(results: Sequence[ReplayResult]) -> str:
    header = (
        f"{'combo':<38}  {'rows':>4}  {'status':<16}  "
        f"{'|dpsi|max':>9}  {'|elat|max':>9}  {'progress':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.label():<38}  {r.row_count:>4}  {r.solver_status[:16]:<16}  "
            f"{r.max_abs_heading_error_deg:>9.2f}  {r.max_abs_lateral_error_m:>9.3f}  "
            f"{r.final_progress_m:>8.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="FrameCapture JSON file")
    parser.add_argument(
        "--audit", action="store_true", help="print the QP row audit and exit"
    )
    parser.add_argument(
        "--couple-flag-deg",
        type=float,
        default=5.0,
        help="lateral-coupling angle above which a row is flagged",
    )
    args = parser.parse_args(argv)

    capture = FrameCapture.from_json(args.capture)

    pre_rows, _ = _rows_for(capture, "pre")
    post_rows, _ = _rows_for(capture, "post") if capture.has_corridor() else (pre_rows, None)

    print(f"# frame: tick={capture.tick} sim_time_s={capture.sim_time_s}")
    print(
        f"# ego_origin_xy={tuple(round(v, 3) for v in capture.ego_origin_xy)} "
        f"current_state={[round(float(v), 3) for v in capture.current_state]}"
    )
    print(
        f"# pre_ref pts={len(capture.pre_publication_reference)} "
        f"post_ref pts={len(capture.published_reference)}"
    )

    audits = audit_rows(
        pre_rows,
        tuple(capture.ego_origin_xy),
        pre_reference=capture.reference("pre"),
        post_reference=capture.reference("post"),
        couple_flag_deg=args.couple_flag_deg,
    )
    print("\n## QP row audit (rows built on the PRE-publication reference)")
    print(json.dumps(audit_summary(audits), indent=2))
    for a in audits:
        mark = "  <-- FLAGGED" if a.flagged else ""
        print(
            f"  stage {a.stage:>2} {a.kind:<17} tag={a.tag or '-':<6} "
            f"couple={a.lateral_coupling_deg:6.2f} deg  "
            f"pre/post tangent disagree={a.pre_post_tangent_disagreement_deg:5.2f} deg{mark}"
        )
        print(f"      {a.world_expr}")

    if args.audit:
        return 0

    print("\n## Replay matrix")
    results = replay_matrix(capture)
    print(format_matrix(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
