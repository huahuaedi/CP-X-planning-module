"""Prediction-knowledge ablation support for the CP-X planning pipeline.

This module provides the ``snapshot_transform`` callables consumed by
``build_prediction_frame`` / ``CPXObstacleTracker.predict`` to run a controlled
A/B on how much the ego planner knows about other vehicles' futures:

* ``cv``     -- no transform (pipeline default: constant-velocity/acceleration
                kinematic rollout from each obstacle's current state).
* ``blind``  -- every obstacle is frozen at its current pose for the whole
                horizon: the ego knows *where* each vehicle is, but assumes it
                does not move ("no predicted trajectory").
* ``oracle`` -- each obstacle's future is replaced with its *recorded
                ground-truth* trajectory from an earlier run of the exact same
                scenario, plus a maneuver-intent label derived from that trace.

The oracle path is record-replay: run the scenario once with
``record_oracle_traces: true`` (any prediction mode) to dump
``TraceRecorder`` output, then run ``blind`` and ``oracle`` against that file.

All transforms return a NEW snapshot dict; they never mutate their input.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

SnapshotTransform = Callable[[dict, float], Mapping[str, Any]]

_POS_MATCH_TOLERANCE_M = 3.0
_HEADING_TURN_THRESHOLD_RAD = math.radians(18.0)
_LATERAL_LANE_CHANGE_THRESHOLD_M = 1.6


def _f(m: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in m and m[k] is not None:
            try:
                return float(m[k])
            except (TypeError, ValueError):
                return default
    return default


def _snapshot_xy(snapshot: Mapping[str, Any]) -> tuple[float, float]:
    return _f(snapshot, "x", "x_m"), _f(snapshot, "y", "y_m")


def _snapshot_speed(snapshot: Mapping[str, Any]) -> float:
    if "v" in snapshot or "speed" in snapshot or "speed_mps" in snapshot:
        return _f(snapshot, "v", "speed", "speed_mps")
    vx, vy = _f(snapshot, "vx"), _f(snapshot, "vy")
    return math.hypot(vx, vy)


def _snapshot_heading(snapshot: Mapping[str, Any]) -> float:
    if "psi" in snapshot or "heading" in snapshot or "yaw" in snapshot:
        return _f(snapshot, "psi", "heading", "yaw")
    vx, vy = _f(snapshot, "vx"), _f(snapshot, "vy")
    if math.hypot(vx, vy) > 0.1:
        return math.atan2(vy, vx)
    return 0.0


# --------------------------------------------------------------------------- #
# blind: freeze every obstacle at its current pose
# --------------------------------------------------------------------------- #
def frozen_snapshot_transform(
    *, horizon_s: float, dt_s: float
) -> SnapshotTransform:
    """Return a transform that replaces every obstacle's future with a
    stationary hold at its current position."""

    dt_s = max(1.0e-3, float(dt_s))
    count = max(1, int(math.ceil(max(dt_s, float(horizon_s)) / dt_s)))

    def _transform(snapshot: dict, _timestamp_s: float) -> dict:
        x0, y0 = _snapshot_xy(snapshot)
        points = [
            {"x": float(x0), "y": float(y0), "t": float(dt_s * step)}
            for step in range(1, count + 1)
        ]
        out = dict(snapshot)
        out["trajectory_hypotheses"] = [
            {"points": points, "probability": 1.0, "maneuver": "stationary_assumed"}
        ]
        out["predicted_maneuver"] = "stationary_assumed"
        out["prediction_source"] = "ablation_blind"
        out["v"] = 0.0
        return out

    return _transform


def synthetic_multimodal_snapshot_transform(
    *, horizon_s: float, dt_s: float, actor_ids: Sequence[int] = (),
    update_period_s: float = 0.2,
    actor_activation: Optional[Mapping[str, bool]] = None,
) -> SnapshotTransform:
    """Inject 5-Hz-style keep-lane/brake/lane-change hypotheses.

    The transform is called at the perception tick rate. Its cached payload
    models an external predictor whose output revision remains unchanged
    between prediction updates.
    """

    selected = {str(int(value)) for value in list(actor_ids or ())}
    step_s = max(1.0e-3, float(dt_s))
    count = max(2, int(math.ceil(max(step_s, float(horizon_s)) / step_s)) + 1)
    refresh_s = max(1.0e-3, float(update_period_s))
    cache: Dict[str, tuple[float, List[dict]]] = {}

    def _transform(snapshot: dict, timestamp_s: float) -> dict:
        out = dict(snapshot)
        actor_id = next((
            str(snapshot[key])
            for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id")
            if snapshot.get(key) is not None
        ), "")
        if selected and actor_id not in selected:
            return out
        if actor_activation is not None and not bool(
            actor_activation.get(str(actor_id), True)
        ):
            out["prediction_source"] = "synthetic_inactive"
            return out
        cached = cache.get(actor_id)
        if cached is not None:
            cached_time_s, cached_modes = cached
            age_s = float(timestamp_s) - float(cached_time_s)
            if -1.0e-9 <= age_s < refresh_s - 1.0e-9:
                out["trajectory_hypotheses"] = cached_modes
                out["prediction_source"] = "synthetic_multimodal"
                out["prediction_timestamp_s"] = float(cached_time_s)
                return out
        x0, y0 = _snapshot_xy(snapshot)
        v0 = max(0.0, _snapshot_speed(snapshot))
        heading = _snapshot_heading(snapshot)
        tx, ty = math.cos(heading), math.sin(heading)
        nx, ny = -ty, tx

        def _points(kind: str) -> list[dict]:
            rows = []
            for index in range(count):
                time_s = float(index) * step_s
                speed = max(0.0, v0 - 2.0 * time_s) if kind == "brake" else v0
                # Integrate deceleration only until rest; extending the
                # parabola past v=0 would make a stopped prediction reverse.
                moving_time_s = min(time_s, v0 / 2.0)
                distance = (
                    v0 * moving_time_s - moving_time_s ** 2
                    if kind == "brake" else v0 * time_s
                )
                u = min(1.0, time_s / max(1.0, 0.75 * float(horizon_s)))
                smooth = 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5
                lateral = 3.5 * smooth if kind == "lane_change_left" else 0.0
                rows.append({
                    "t": time_s,
                    "x": x0 + distance * tx + lateral * nx,
                    "y": y0 + distance * ty + lateral * ny,
                    "v": speed,
                    "psi": heading,
                })
            return rows

        modes = [
            {"maneuver": "lane_keep", "probability": 0.55,
             "position_sigma_m": 0.6, "points": _points("lane_keep")},
            {"maneuver": "brake", "probability": 0.25,
             "position_sigma_m": 0.8, "points": _points("brake")},
            {"maneuver": "lane_change_left", "probability": 0.20,
             "position_sigma_m": 1.0, "points": _points("lane_change_left")},
        ]
        cache[actor_id] = (float(timestamp_s), modes)
        out["trajectory_hypotheses"] = modes
        out["prediction_source"] = "synthetic_multimodal"
        out["prediction_timestamp_s"] = float(timestamp_s)
        return out

    return _transform


# --------------------------------------------------------------------------- #
# oracle: replay recorded ground-truth futures
# --------------------------------------------------------------------------- #
@dataclass
class _ActorTrace:
    actor_id: str
    ts: List[float]
    xs: List[float]
    ys: List[float]
    vs: List[float]
    psis: List[float]
    lane_ids: List[Optional[int]]

    def state_at(self, t: float) -> Optional[tuple[float, float, float, float, Optional[int]]]:
        if not self.ts or t < self.ts[0] - 1e-6 or t > self.ts[-1] + 1e-6:
            return None
        i = bisect_left(self.ts, t)
        if i <= 0:
            return self.xs[0], self.ys[0], self.vs[0], self.psis[0], self.lane_ids[0]
        if i >= len(self.ts):
            return self.xs[-1], self.ys[-1], self.vs[-1], self.psis[-1], self.lane_ids[-1]
        t0, t1 = self.ts[i - 1], self.ts[i]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        x = self.xs[i - 1] + a * (self.xs[i] - self.xs[i - 1])
        y = self.ys[i - 1] + a * (self.ys[i] - self.ys[i - 1])
        v = self.vs[i - 1] + a * (self.vs[i] - self.vs[i - 1])
        dpsi = math.atan2(
            math.sin(self.psis[i] - self.psis[i - 1]),
            math.cos(self.psis[i] - self.psis[i - 1]),
        )
        psi = self.psis[i - 1] + a * dpsi
        lane = self.lane_ids[i - 1] if a < 0.5 else self.lane_ids[i]
        return x, y, v, psi, lane


class OracleTraceStore:
    """Recorded ground-truth trajectories for every non-ego actor, keyed by
    actor id, queried by nearest current position (id-plumbing agnostic)."""

    def __init__(self, traces: Sequence[_ActorTrace]):
        self._traces = list(traces)

    # ---- loading ---------------------------------------------------------- #
    @classmethod
    def from_file(cls, path: str | Path) -> "OracleTraceStore":
        p = Path(path)
        by_actor: Dict[str, Dict[str, list]] = {}
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t = float(rec["t"])
                for actor_id, st in dict(rec.get("actors", {})).items():
                    slot = by_actor.setdefault(
                        str(actor_id),
                        {"t": [], "x": [], "y": [], "v": [], "psi": [], "lane": []},
                    )
                    slot["t"].append(t)
                    slot["x"].append(_f(st, "x", "x_m"))
                    slot["y"].append(_f(st, "y", "y_m"))
                    slot["v"].append(_snapshot_speed(st))
                    slot["psi"].append(_snapshot_heading(st))
                    lane = st.get("lane_id", st.get("lane"))
                    slot["lane"].append(int(lane) if lane is not None else None)
        traces = []
        for actor_id, slot in by_actor.items():
            order = sorted(range(len(slot["t"])), key=lambda k: slot["t"][k])
            traces.append(
                _ActorTrace(
                    actor_id=actor_id,
                    ts=[slot["t"][k] for k in order],
                    xs=[slot["x"][k] for k in order],
                    ys=[slot["y"][k] for k in order],
                    vs=[slot["v"][k] for k in order],
                    psis=[slot["psi"][k] for k in order],
                    lane_ids=[slot["lane"][k] for k in order],
                )
            )
        return cls(traces)

    # ---- query ---------------------------------------------------------- #
    def _match(self, x: float, y: float, t_now: float) -> Optional[_ActorTrace]:
        best, best_d = None, _POS_MATCH_TOLERANCE_M
        for tr in self._traces:
            st = tr.state_at(t_now)
            if st is None:
                continue
            d = math.hypot(st[0] - x, st[1] - y)
            if d < best_d:
                best, best_d = tr, d
        return best

    def future_for(
        self, snapshot: Mapping[str, Any], t_now: float, *, horizon_s: float, dt_s: float
    ) -> Optional[dict]:
        x, y = _snapshot_xy(snapshot)
        tr = self._match(x, y, t_now)
        if tr is None:
            return None
        dt_s = max(1.0e-3, float(dt_s))
        count = max(1, int(math.ceil(max(dt_s, float(horizon_s)) / dt_s)))
        pts, lanes, psis = [], [], []
        for step in range(1, count + 1):
            st = tr.state_at(t_now + dt_s * step)
            if st is None:
                break
            pts.append({"x": float(st[0]), "y": float(st[1]), "t": float(dt_s * step),
                        "v": float(st[2])})
            lanes.append(st[4])
            psis.append(st[3])
        if len(pts) < 2:
            return None
        st0 = tr.state_at(t_now)
        maneuver = _label_maneuver(
            psi0=(st0[3] if st0 else psis[0]),
            psis=psis,
            lane0=(st0[4] if st0 else lanes[0]),
            lanes=lanes,
            x0=x, y0=y, pts=pts,
            psi_ref=(st0[3] if st0 else psis[0]),
        )
        return {"points": pts, "maneuver": maneuver, "actor_id": tr.actor_id}

    def snapshot_transform(
        self, *, horizon_s: float, dt_s: float, freeze_unmatched: bool = False
    ) -> SnapshotTransform:
        frozen = frozen_snapshot_transform(horizon_s=horizon_s, dt_s=dt_s)

        def _transform(snapshot: dict, t_now: float) -> dict:
            fut = self.future_for(
                snapshot, float(t_now), horizon_s=horizon_s, dt_s=dt_s
            )
            if fut is None:
                return frozen(snapshot, t_now) if freeze_unmatched else dict(snapshot)
            out = dict(snapshot)
            out["trajectory_hypotheses"] = [
                {"points": fut["points"], "probability": 1.0, "maneuver": fut["maneuver"]}
            ]
            out["predicted_maneuver"] = fut["maneuver"]
            out["prediction_source"] = "ablation_oracle"
            out["prediction_timestamp_s"] = float(t_now)
            return out

        return _transform


def _label_maneuver(
    *, psi0: float, psis: Sequence[float], lane0: Optional[int],
    lanes: Sequence[list], x0: float, y0: float, pts: Sequence[Mapping[str, float]],
    psi_ref: float,
) -> str:
    """Derive an intent label from the recorded future window."""

    valid_lanes = [l for l in lanes if l is not None]
    if lane0 is not None and valid_lanes and valid_lanes[-1] != lane0:
        # Lane id changed within the horizon -> lane change; sign from lateral
        # displacement in the actor's own initial frame.
        cos_p, sin_p = math.cos(psi_ref), math.sin(psi_ref)
        lat = (pts[-1]["x"] - x0) * (-sin_p) + (pts[-1]["y"] - y0) * cos_p
        if abs(lat) >= _LATERAL_LANE_CHANGE_THRESHOLD_M:
            return "lane_change_left" if lat > 0 else "lane_change_right"
        return "lane_change"
    if psis:
        dpsi = math.atan2(math.sin(psis[-1] - psi0), math.cos(psis[-1] - psi0))
        if abs(dpsi) >= _HEADING_TURN_THRESHOLD_RAD:
            return "turn_left" if dpsi > 0 else "turn_right"
    return "lane_keep"


def build_snapshot_transform(
    *,
    prediction_mode: str,
    horizon_s: float,
    dt_s: float,
    oracle_store: Optional[OracleTraceStore] = None,
    freeze_unmatched_oracle: bool = False,
    synthetic_actor_ids: Sequence[int] = (),
    synthetic_update_period_s: float = 0.2,
    synthetic_actor_activation: Optional[Mapping[str, bool]] = None,
) -> Optional[SnapshotTransform]:
    """Factory used by the bridge: map a config mode string to a transform.

    Returns ``None`` for ``cv`` (pipeline default) and for any unknown mode.
    """

    mode = str(prediction_mode or "cv").strip().lower()
    if mode in {"", "cv", "default", "constant_velocity", "constant_acceleration"}:
        return None
    if mode in {"blind", "frozen", "none", "no_prediction"}:
        return frozen_snapshot_transform(horizon_s=float(horizon_s), dt_s=float(dt_s))
    if mode == "oracle":
        if oracle_store is None:
            raise ValueError(
                "prediction_mode='oracle' requires a recorded trace file "
                "(config 'oracle_trace_path'); run once with "
                "'record_oracle_traces: true' first."
            )
        return oracle_store.snapshot_transform(
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            freeze_unmatched=bool(freeze_unmatched_oracle),
        )
    if mode in {"synthetic_multimodal", "multimodal"}:
        return synthetic_multimodal_snapshot_transform(
            horizon_s=float(horizon_s), dt_s=float(dt_s),
            actor_ids=synthetic_actor_ids,
            update_period_s=float(synthetic_update_period_s),
            actor_activation=synthetic_actor_activation,
        )
    raise ValueError(f"unknown prediction_mode: {prediction_mode!r}")


# --------------------------------------------------------------------------- #
# recording pass helper (used by the scenario runner)
# --------------------------------------------------------------------------- #
class TraceRecorder:
    """Append one JSON line per tick with every tracked actor's ground truth.

    Line schema: ``{"t": <sim_s>, "actors": {"<id>": {"x","y","v","psi","lane_id"}}}``
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")
        self._n = 0

    def record(
        self,
        *,
        sim_time_s: float,
        actors: Mapping[str, Mapping[str, Any]],
    ) -> None:
        row = {"t": float(sim_time_s), "actors": {}}
        for actor_id, st in dict(actors or {}).items():
            row["actors"][str(actor_id)] = {
                "x": _f(st, "x", "x_m"),
                "y": _f(st, "y", "y_m"),
                "v": _snapshot_speed(st),
                "psi": _snapshot_heading(st),
                "lane_id": (
                    int(st["lane_id"]) if st.get("lane_id") is not None else None
                ),
            }
        self._fh.write(json.dumps(row) + "\n")
        self._n += 1

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    @property
    def rows_written(self) -> int:
        return self._n
