"""Prediction stage for the CP-X planning pipeline.

The runner already receives cooperative-perception obstacle snapshots from
CP-X or from the local CARLA/SUMO tracker.  This module makes the prediction
stage explicit: every obstacle gets a short-horizon future trajectory, then
each candidate lane receives a future-risk summary used by the behavior FSM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from behavior_planner.trajectory_risk import (
    lane_prediction_risk,
    obstacle_future_trajectory,
)


def obstacle_track_id(snapshot: Mapping[str, object]) -> str:
    """Public alias of the id resolution ``PredictionFrame`` keys are built
    with, so callers matching MPC obstacle snapshots against
    ``obstacle_future_trajectories`` use the exact same identity rule."""

    return _obstacle_id(snapshot)


def mpc_stage_trajectory(
    points: Sequence[Mapping[str, object]],
    *,
    fallback_heading_rad: float,
    horizon_steps: int,
    dt_s: float,
) -> List[List[float]]:
    """Convert ``obstacle_future_trajectory``-style ``{x, y, t, v}`` points
    into the ``[x, y, v, psi]``-per-stage list
    ``MPC._get_object_state_at_stage`` reads directly (see MPC/mpc.py).

    Without this, MPC's own obstacle-avoidance cost never sees this
    module's prediction at all: it only recognizes a ``predicted_trajectory``
    already shaped as one ``[x, y, v, psi]`` entry per stage, and silently
    falls back to its own constant-velocity extrapolation for anything else
    (including the ``{x, y, t}`` dict points this module produces). The
    heading is held constant at ``fallback_heading_rad`` because the
    constant-acceleration/constant-velocity models this module falls back to
    do not turn -- a real turning prediction would need to supply its own
    per-point heading in ``points``.
    """

    stages: List[List[float]] = []
    last_x: float | None = None
    last_y: float | None = None
    last_v = 0.0
    for step in range(max(0, int(horizon_steps))):
        if step < len(points):
            point = points[step]
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            v = float(point.get("v", last_v))
        elif last_x is not None:
            # The supplied trajectory is shorter than MPC's horizon (e.g. a
            # CP-supplied real prediction that stops early). Hold the last
            # known speed/heading rather than leaving later stages unset.
            x = float(last_x) + float(last_v) * math.cos(float(fallback_heading_rad)) * float(dt_s)
            y = float(last_y) + float(last_v) * math.sin(float(fallback_heading_rad)) * float(dt_s)
            v = float(last_v)
        else:
            break
        last_x, last_y, last_v = x, y, v
        stages.append([float(x), float(y), float(v), float(fallback_heading_rad)])
    return stages


def _obstacle_id(snapshot: Mapping[str, object]) -> str:
    for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id"):
        value = snapshot.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    try:
        return "xy:{:.1f}:{:.1f}".format(
            float(snapshot.get("x", snapshot.get("x_m", 0.0))),
            float(snapshot.get("y", snapshot.get("y_m", 0.0))),
        )
    except Exception:
        return ""


@dataclass(frozen=True)
class PredictionHypothesis:
    """One possible future motion mode and its calibrated uncertainty."""

    probability: float
    maneuver: str
    points: Tuple[Mapping[str, object], ...]
    position_sigma_m: float
    source: str

    def mutable_points(self) -> List[dict]:
        return [dict(point) for point in self.points]


@dataclass(frozen=True)
class PredictedObject:
    """Versioned probabilistic prediction for one tracked road user."""

    track_id: str
    source: str
    timestamp_s: float
    plan_revision: str
    hypotheses: Tuple[PredictionHypothesis, ...]

    @property
    def primary(self) -> PredictionHypothesis:
        return max(self.hypotheses, key=lambda item: float(item.probability))


@dataclass
class PredictionFrame:
    """Prediction output consumed by behavior decision and trajectory planning."""

    ego_snapshot: Dict[str, float]
    obstacle_snapshots: List[dict]
    obstacle_future_trajectories: Dict[str, List[dict]] = field(default_factory=dict)
    predicted_objects: Dict[str, PredictedObject] = field(default_factory=dict)
    lane_prediction_risks: Dict[int, Dict[str, object]] = field(default_factory=dict)
    timestamp_s: float = 0.0
    revision: str = ""

    def risk_for_lane(self, lane_id: int) -> Dict[str, object]:
        return dict(self.lane_prediction_risks.get(int(lane_id), {}))


def build_prediction_frame(
    *,
    ego_snapshot: Mapping[str, object],
    obstacle_snapshots: Sequence[Mapping[str, Any]],
    lane_assignments: Mapping[str, int],
    available_lane_ids: Sequence[int],
    horizon_s: float,
    dt_s: float,
    min_front_gap_m: float,
    min_rear_gap_m: float,
    min_ttc_s: float,
    prediction_model: str = "constant_acceleration",
    max_abs_acceleration_mps2: float = 4.0,
    lane_step_fn: Callable[[float, float, float], Any] | None = None,
    timestamp_s: float = 0.0,
    revision: str = "",
    risk_probability_threshold: float = 0.05,
) -> PredictionFrame:
    """Build an Apollo-style prediction frame for one planning tick.

    Existing CP-X messages may already include `predicted_trajectory`.  When
    they do not, the default fallback is a constant-acceleration prediction
    (`prediction_model="constant_acceleration"`).  With no acceleration field
    available this degenerates to constant velocity, so existing snapshots keep
    their previous behaviour.

    ``lane_step_fn``, when supplied, lets the fallback follow the obstacle's
    own lane centerline (curved) instead of a straight line -- see
    ``behavior_planner.trajectory_risk._lane_following_points``. Passing None
    (the default) preserves the exact previous straight-line behaviour.
    """

    normalized_ego = {
        "x": float(ego_snapshot.get("x", 0.0)),
        "y": float(ego_snapshot.get("y", 0.0)),
        "v": float(ego_snapshot.get("v", 0.0)),
        "psi": float(ego_snapshot.get("psi", 0.0)),
    }
    normalized_obstacles = [
        dict(snapshot)
        for snapshot in list(obstacle_snapshots or [])
        if isinstance(snapshot, Mapping)
    ]
    predicted_objects: Dict[str, PredictedObject] = {}
    obstacle_future_trajectories: Dict[str, List[dict]] = {}
    for snapshot in normalized_obstacles:
        obstacle_id = _obstacle_id(snapshot)
        if not obstacle_id:
            continue
        predicted = _predicted_object(
            obstacle_id=str(obstacle_id),
            snapshot=snapshot,
            timestamp_s=float(timestamp_s),
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            prediction_model=str(prediction_model),
            max_abs_acceleration_mps2=float(max_abs_acceleration_mps2),
            lane_step_fn=lane_step_fn,
        )
        predicted_objects[str(obstacle_id)] = predicted
        obstacle_future_trajectories[str(obstacle_id)] = (
            predicted.primary.mutable_points()
        )
    lane_prediction_risks = {
        int(lane_id): lane_prediction_risk(
            ego_snapshot=normalized_ego,
            obstacle_snapshots=normalized_obstacles,
            lane_assignments=lane_assignments,
            target_lane_id=int(lane_id),
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            min_front_gap_m=float(min_front_gap_m),
            min_rear_gap_m=float(min_rear_gap_m),
            min_ttc_s=float(min_ttc_s),
            prediction_model=str(prediction_model),
            max_abs_acceleration_mps2=float(max_abs_acceleration_mps2),
            lane_step_fn=lane_step_fn,
        )
        for lane_id in list(available_lane_ids or [])
    }
    for lane_id in list(available_lane_ids or []):
        probabilistic = _probabilistic_lane_risk(
            ego_snapshot=normalized_ego,
            predicted_objects=predicted_objects,
            lane_assignments=lane_assignments,
            target_lane_id=int(lane_id),
            horizon_s=float(horizon_s), dt_s=float(dt_s),
            min_front_gap_m=float(min_front_gap_m),
            min_rear_gap_m=float(min_rear_gap_m), min_ttc_s=float(min_ttc_s),
            risk_probability_threshold=float(risk_probability_threshold),
        )
        if probabilistic["hypothesis_count"]:
            lane_prediction_risks[int(lane_id)] = probabilistic
    return PredictionFrame(
        ego_snapshot=normalized_ego,
        obstacle_snapshots=normalized_obstacles,
        obstacle_future_trajectories=obstacle_future_trajectories,
        predicted_objects=predicted_objects,
        lane_prediction_risks=lane_prediction_risks,
        timestamp_s=float(timestamp_s),
        revision=str(revision or f"prediction:{float(timestamp_s):.3f}"),
    )


def _probabilistic_lane_risk(
    *, ego_snapshot, predicted_objects, lane_assignments, target_lane_id,
    horizon_s, dt_s, min_front_gap_m, min_rear_gap_m, min_ttc_s,
    risk_probability_threshold,
):
    probability = 0.0
    hypothesis_count = 0
    risky_ids = []
    front_gaps, rear_gaps, ttcs = [], [], []
    for track_id, predicted in dict(predicted_objects or {}).items():
        if int(lane_assignments.get(str(track_id), 0) or 0) != int(target_lane_id):
            continue
        for hypothesis in predicted.hypotheses:
            hypothesis_count += 1
            snapshot = {
                "vehicle_id": str(track_id),
                "x": predicted.primary.points[0].get("x", 0.0) if predicted.primary.points else 0.0,
                "y": predicted.primary.points[0].get("y", 0.0) if predicted.primary.points else 0.0,
                "v": hypothesis.points[0].get("v", 0.0) if hypothesis.points else 0.0,
                "predicted_trajectory": hypothesis.mutable_points(),
            }
            result = lane_prediction_risk(
                ego_snapshot=ego_snapshot, obstacle_snapshots=[snapshot],
                lane_assignments={str(track_id): int(target_lane_id)},
                target_lane_id=int(target_lane_id), horizon_s=float(horizon_s),
                dt_s=float(dt_s), min_front_gap_m=float(min_front_gap_m),
                min_rear_gap_m=float(min_rear_gap_m), min_ttc_s=float(min_ttc_s),
            )
            if bool(result.get("risk", False)):
                probability += float(hypothesis.probability)
                risky_ids.append(str(track_id))
            for key, target in (("min_front_gap_m", front_gaps),
                                ("min_rear_gap_m", rear_gaps),
                                ("min_ttc_s", ttcs)):
                value = result.get(key)
                if value is not None:
                    target.append(float(value))
    probability = min(1.0, max(0.0, probability))
    return {
        "risk": bool(probability >= max(0.0, float(risk_probability_threshold))),
        "collision_probability": float(probability),
        "risk_probability_threshold": float(risk_probability_threshold),
        "hypothesis_count": int(hypothesis_count),
        "target_lane_id": int(target_lane_id),
        "min_front_gap_m": min(front_gaps) if front_gaps else None,
        "min_rear_gap_m": min(rear_gaps) if rear_gaps else None,
        "min_ttc_s": min(ttcs) if ttcs else None,
        "risky_obstacle_id": ";".join(sorted(set(risky_ids))),
        "reason": "probabilistic_conflict" if probability else "",
    }


def _predicted_object(
    *, obstacle_id: str, snapshot: Mapping[str, object], timestamp_s: float,
    horizon_s: float, dt_s: float, prediction_model: str,
    max_abs_acceleration_mps2: float, lane_step_fn,
) -> PredictedObject:
    raw_modes = snapshot.get(
        "trajectory_hypotheses", snapshot.get("predicted_trajectories", ())
    )
    modes = list(raw_modes or ()) if isinstance(raw_modes, Sequence) and not isinstance(raw_modes, (str, bytes)) else []
    hypotheses = []
    for mode in modes:
        if not isinstance(mode, Mapping):
            continue
        points = mode.get("points", mode.get("trajectory", ()))
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            continue
        normalized_points = tuple(dict(point) for point in points if isinstance(point, Mapping))
        if not normalized_points:
            continue
        hypotheses.append(PredictionHypothesis(
            probability=max(0.0, float(mode.get("probability", 0.0) or 0.0)),
            maneuver=str(mode.get("maneuver", "unknown") or "unknown"),
            points=normalized_points,
            position_sigma_m=max(0.0, float(mode.get("position_sigma_m", 0.5) or 0.0)),
            source=str(snapshot.get("prediction_source", "v2x_plan") or "v2x_plan"),
        ))
    if not hypotheses:
        points = obstacle_future_trajectory(
            snapshot, horizon_s=float(horizon_s), dt_s=float(dt_s),
            model=str(prediction_model),
            max_abs_acceleration_mps2=float(max_abs_acceleration_mps2),
            lane_step_fn=lane_step_fn,
        )
        source = str(snapshot.get("prediction_source", "perception_model") or "perception_model")
        hypotheses = [PredictionHypothesis(
            probability=1.0,
            maneuver=str(snapshot.get("predicted_maneuver", "lane_keep") or "lane_keep"),
            points=tuple(dict(point) for point in points),
            position_sigma_m=max(0.0, float(snapshot.get("position_sigma_m", 1.0) or 0.0)),
            source=source,
        )]
    probability_sum = sum(float(item.probability) for item in hypotheses)
    if probability_sum <= 1.0e-9:
        probability_sum = float(len(hypotheses))
        probabilities = [1.0 / probability_sum for _ in hypotheses]
    else:
        probabilities = [float(item.probability) / probability_sum for item in hypotheses]
    normalized = tuple(
        PredictionHypothesis(
            probability=float(probability), maneuver=item.maneuver,
            points=item.points, position_sigma_m=item.position_sigma_m,
            source=item.source,
        )
        for item, probability in zip(hypotheses, probabilities)
    )
    return PredictedObject(
        track_id=str(obstacle_id),
        source=str(snapshot.get("prediction_source", normalized[0].source)),
        timestamp_s=float(snapshot.get("prediction_timestamp_s", timestamp_s) or timestamp_s),
        plan_revision=str(snapshot.get("plan_revision", "") or ""),
        hypotheses=normalized,
    )
