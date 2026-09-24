"""HTTP adapter from the external MTR service to planner predictions.

The adapter has one responsibility: annotate immutable tracker snapshots with
``predicted_trajectory``.  The existing ``CPXObstacleTracker`` remains the
single owner of ``PredictionFrame`` construction, freshness revisions, lane
risk, and the constant-acceleration fallback.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _track_id(snapshot: Mapping[str, Any]) -> str:
    for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id"):
        value = snapshot.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


class MTRPredictionBridge:
    """Fetch and attach MTR predictions without owning planner policy.

    The MTR checkpoint consumes 10 Hz history, while the CARLA control loop
    commonly runs at 20 Hz.  Requests are therefore rate-limited in simulation
    time and the latest response is time-rebased between refreshes.  A failed
    request never blocks the planning contract: cached output is held only for
    ``max_stale_s`` and then the caller naturally falls back to its internal
    prediction model.
    """

    model_name = "mtr_http"

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:8765",
        timeout_s: float = 2.0,
        update_hz: float = 5.0,
        history_update_hz: float = 10.0,
        max_stale_s: float = 0.5,
        asynchronous: bool = True,
    ) -> None:
        self._url = str(server_url).rstrip("/") + "/predict"
        self._timeout_s = max(0.01, float(timeout_s))
        self._minimum_interval_s = 1.0 / max(0.1, float(update_hz))
        self._history_interval_s = 1.0 / max(
            float(update_hz), float(history_update_hz), 0.1
        )
        self._max_stale_s = max(0.0, float(max_stale_s))
        self._last_request_timestamp_s = -float("inf")
        self._last_history_timestamp_s = -float("inf")
        self._prediction_timestamp_s = -float("inf")
        self._cached_predictions: Dict[str, List[Dict[str, float]]] = {}
        self._cached_prediction_modes: Dict[str, List[Dict[str, Any]]] = {}
        self._cached_track_ids = set()
        self._revision = 0
        self.last_attached_count = 0
        self.last_attached_mode_count = 0
        self.last_map_polyline_count = 0
        self.last_request_latency_ms = 0.0
        self.last_error = ""
        self.request_count = 0
        self.inference_request_count = 0
        self.success_count = 0
        self.dropped_request_count = 0
        self._asynchronous = bool(asynchronous)
        self._state_lock = threading.Lock()
        self._schedule_lock = threading.Lock()
        self._request_queue = None
        if self._asynchronous:
            self._request_queue = queue.Queue(maxsize=8)
            worker = threading.Thread(
                target=self._worker_loop,
                name="mtr-prediction-worker",
            )
            worker.daemon = True
            worker.start()

    def attach(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        horizon_s: float,
        dt_s: float,
        ego_snapshot: Mapping[str, Any],
        timestamp_s: float,
        map_polylines: Sequence[Mapping[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        del dt_s  # MTR owns its trained sampling interval.
        snapshots = [
            dict(snapshot) for snapshot in list(object_snapshots or ())
            if isinstance(snapshot, Mapping)
        ]
        timestamp_s = float(timestamp_s)
        track_ids = {
            _track_id(snapshot) for snapshot in snapshots if _track_id(snapshot)
        }
        ego_track_id = _track_id(ego_snapshot)
        if ego_track_id:
            # A shared scene bridge is called once by each ego planner.  The
            # observer changes, but the physical scene membership does not:
            # one ego's actor is present in every other ego's obstacle list.
            # Including ego in the signature makes that invariant explicit
            # and prevents N identical history/inference requests per tick.
            track_ids.add(ego_track_id)
        # Multiple CAV planners share this bridge and enter it sequentially (or
        # from different application threads) for the same world tick.  Keep
        # cadence detection and its timestamp update atomic so exactly one
        # caller publishes that scene to the prediction server.
        with self._schedule_lock:
            self._reset_after_time_discontinuity(timestamp_s)
            same_world_tick = bool(
                math.isfinite(self._last_history_timestamp_s)
                and abs(timestamp_s - self._last_history_timestamp_s) <= 1.0e-6
            )
            track_set_changed = bool(track_ids != self._cached_track_ids)
            inference_due = bool(
                not same_world_tick
                and (
                    timestamp_s - self._last_request_timestamp_s
                    >= self._minimum_interval_s - 1.0e-6
                    or track_set_changed
                )
            )
            history_due = bool(
                not same_world_tick
                and (
                    timestamp_s - self._last_history_timestamp_s
                    >= self._history_interval_s - 1.0e-6
                    or track_set_changed
                )
            )
            if history_due:
                self._schedule_request(
                    ego_snapshot=ego_snapshot,
                    obstacle_snapshots=snapshots,
                    timestamp_s=timestamp_s,
                    map_polylines=map_polylines,
                    run_inference=inference_due,
                    scene_track_ids=track_ids,
                )

        with self._state_lock:
            prediction_timestamp_s = float(self._prediction_timestamp_s)
            revision = int(self._revision)
            cached_predictions = self._cached_predictions
            cached_prediction_modes = self._cached_prediction_modes
        age_s = max(0.0, timestamp_s - prediction_timestamp_s)
        usable = bool(
            cached_predictions
            and math.isfinite(prediction_timestamp_s)
            and age_s <= self._max_stale_s + 1.0e-6
        )
        annotated = []
        attached_count = 0
        attached_mode_count = 0
        for snapshot in snapshots:
            updated = dict(snapshot)
            track_id = _track_id(snapshot)
            modes = self._rebased_modes(
                cached_prediction_modes.get(track_id, ()) if usable else (),
                age_s=age_s,
                horizon_s=float(horizon_s),
            )
            if not modes and usable:
                points = self._rebased_points(
                    cached_predictions.get(track_id, ()),
                    age_s=age_s,
                    horizon_s=float(horizon_s),
                )
                if points:
                    modes = [{
                        "trajectory": points,
                        "probability": 1.0,
                        "maneuver": "unknown",
                        "position_sigma_m": 0.5,
                    }]
            if modes:
                primary = max(
                    modes, key=lambda item: float(item["probability"])
                )
                updated["predicted_trajectory"] = list(primary["trajectory"])
                updated["trajectory_hypotheses"] = list(modes)
                updated["prediction_timestamp_s"] = float(
                    prediction_timestamp_s
                )
                updated["plan_revision"] = "mtr:%d" % revision
                updated["prediction_source"] = self.model_name
                attached_count += 1
                attached_mode_count += len(modes)
            annotated.append(updated)
        self.last_attached_count = int(attached_count)
        self.last_attached_mode_count = int(attached_mode_count)
        self.last_map_polyline_count = len(list(map_polylines or ()))
        return annotated

    @property
    def diagnostics(self) -> Dict[str, object]:
        with self._state_lock:
            success_count = int(self.success_count)
            latency_ms = float(self.last_request_latency_ms)
            last_error = str(self.last_error)
        queue_depth = (
            int(self._request_queue.qsize())
            if self._request_queue is not None else 0
        )
        return {
            "prediction_bridge_model": self.model_name,
            "prediction_bridge_request_count": int(self.request_count),
            "prediction_bridge_inference_request_count": int(
                self.inference_request_count
            ),
            "prediction_bridge_success_count": success_count,
            "prediction_bridge_dropped_request_count": int(
                self.dropped_request_count
            ),
            "prediction_bridge_queue_depth": queue_depth,
            "prediction_bridge_attached_count": int(self.last_attached_count),
            "prediction_bridge_attached_mode_count": int(
                self.last_attached_mode_count
            ),
            "prediction_bridge_latency_ms": latency_ms,
            "prediction_bridge_map_polyline_count": int(
                self.last_map_polyline_count
            ),
            "prediction_bridge_error": last_error,
            "prediction_bridge_shared": bool(
                getattr(self, "shared_consumer_count", 1) > 1
            ),
            "prediction_bridge_shared_consumer_count": int(
                getattr(self, "shared_consumer_count", 1)
            ),
        }

    def _schedule_request(
        self,
        *,
        ego_snapshot: Mapping[str, Any],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        timestamp_s: float,
        map_polylines: Sequence[Mapping[str, Any]],
        run_inference: bool,
        scene_track_ids: Sequence[str],
    ) -> None:
        self._last_history_timestamp_s = float(timestamp_s)
        if bool(run_inference):
            self._last_request_timestamp_s = float(timestamp_s)
            self.inference_request_count += 1
        self._cached_track_ids = set(scene_track_ids or ())
        request_data = {
            "ego_snapshot": dict(ego_snapshot),
            "obstacle_snapshots": [dict(row) for row in obstacle_snapshots],
            "timestamp_s": float(timestamp_s),
            "map_polylines": [dict(row) for row in map_polylines or ()],
            "run_inference": bool(run_inference),
        }
        self.request_count += 1
        if not self._asynchronous:
            self._request(request_data)
            return
        try:
            self._request_queue.put_nowait(request_data)
        except queue.Full:
            self.dropped_request_count += 1
            with self._state_lock:
                self.last_error = "prediction_request_queue_full"

    def _reset_after_time_discontinuity(self, timestamp_s: float) -> None:
        """Retire predictions from a previous simulation-time epoch."""

        if float(timestamp_s) >= self._last_history_timestamp_s - 1.0e-6:
            return
        self._last_request_timestamp_s = -float("inf")
        self._last_history_timestamp_s = -float("inf")
        self._cached_track_ids.clear()
        with self._state_lock:
            self._prediction_timestamp_s = -float("inf")
            self._cached_predictions = {}
            self._cached_prediction_modes = {}

    def _worker_loop(self) -> None:
        while True:
            request_data = self._request_queue.get()
            try:
                self._request(request_data)
            finally:
                self._request_queue.task_done()

    def _request(self, request_data: Mapping[str, Any]) -> None:
        body = json.dumps({
            "ego_snapshot": dict(request_data["ego_snapshot"]),
            "obstacle_snapshots": list(request_data["obstacle_snapshots"]),
            "timestamp_s": float(request_data["timestamp_s"]),
            "map_polylines": list(request_data["map_polylines"]),
            "run_inference": bool(request_data["run_inference"]),
        }).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                request, timeout=float(self._timeout_s)
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            inference_ran = bool(payload.get("inference_ran", True))
            if inference_ran:
                predictions = self._normalize_predictions(
                    payload.get("predictions", {})
                )
                prediction_modes = self._normalize_prediction_modes(
                    payload.get("prediction_modes", {})
                )
                with self._state_lock:
                    self._cached_predictions = predictions
                    self._cached_prediction_modes = prediction_modes
                    self._prediction_timestamp_s = float(
                        request_data["timestamp_s"]
                    )
                    self._revision += 1
                    self.success_count += 1
            with self._state_lock:
                self.last_error = ""
        except Exception as exc:
            # Preserve a still-fresh cached response; otherwise attach() leaves
            # snapshots untouched and the tracker uses its normal fallback.
            with self._state_lock:
                self.last_error = "%s:%s" % (
                    type(exc).__name__, str(exc)
                )
        finally:
            with self._state_lock:
                self.last_request_latency_ms = 1000.0 * (
                    time.monotonic() - started
                )

    @staticmethod
    def _normalize_predictions(value: object) -> Dict[str, List[Dict[str, float]]]:
        if not isinstance(value, Mapping):
            return {}
        result: Dict[str, List[Dict[str, float]]] = {}
        for raw_track_id, raw_points in value.items():
            points = []
            for raw in list(raw_points or ()):
                if not isinstance(raw, Mapping):
                    continue
                try:
                    point = {
                        "x": float(raw["x"]),
                        "y": float(raw["y"]),
                        "t": float(raw["t"]),
                        "v": max(0.0, float(raw.get("v", 0.0))),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
                if all(math.isfinite(number) for number in point.values()):
                    points.append(point)
            if points:
                result[str(raw_track_id)] = sorted(
                    points, key=lambda point: float(point["t"])
                )
        return result

    @classmethod
    def _normalize_prediction_modes(
        cls, value: object
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize probabilistic trajectories without owning risk policy."""

        if not isinstance(value, Mapping):
            return {}
        result: Dict[str, List[Dict[str, Any]]] = {}
        for raw_track_id, raw_modes in value.items():
            modes = []
            for raw_mode in list(raw_modes or ()):
                if not isinstance(raw_mode, Mapping):
                    continue
                points = cls._normalize_predictions({
                    "mode": raw_mode.get(
                        "trajectory", raw_mode.get("points", raw_mode.get("path", ()))
                    )
                }).get("mode", [])
                try:
                    probability = max(0.0, float(
                        raw_mode.get("probability", raw_mode.get("score", 0.0))
                    ))
                    sigma_m = max(0.0, float(
                        raw_mode.get("position_sigma_m", 0.5)
                    ))
                except (TypeError, ValueError):
                    continue
                if points and math.isfinite(probability) and math.isfinite(sigma_m):
                    modes.append({
                        "trajectory": points,
                        "probability": probability,
                        "maneuver": str(raw_mode.get("maneuver", "unknown")),
                        "position_sigma_m": sigma_m,
                    })
            probability_sum = sum(
                float(mode["probability"]) for mode in modes
            )
            if modes and probability_sum <= 1.0e-9:
                probability_sum = float(len(modes))
                for mode in modes:
                    mode["probability"] = 1.0
            if modes:
                for mode in modes:
                    mode["probability"] = (
                        float(mode["probability"]) / probability_sum
                    )
                result[str(raw_track_id)] = sorted(
                    modes,
                    key=lambda mode: -float(mode["probability"]),
                )
        return result

    @staticmethod
    def _rebased_points(
        points: Sequence[Mapping[str, float]],
        *,
        age_s: float,
        horizon_s: float,
    ) -> List[Dict[str, float]]:
        rebased = []
        for source in points:
            relative_t_s = float(source.get("t", 0.0)) - float(age_s)
            if relative_t_s <= 1.0e-6 or relative_t_s > float(horizon_s) + 1.0e-6:
                continue
            point = dict(source)
            point["t"] = float(relative_t_s)
            rebased.append(point)
        return rebased

    @classmethod
    def _rebased_modes(
        cls,
        modes: Sequence[Mapping[str, Any]],
        *,
        age_s: float,
        horizon_s: float,
    ) -> List[Dict[str, Any]]:
        rebased = []
        for source in modes:
            points = cls._rebased_points(
                source.get("trajectory", ()),
                age_s=float(age_s),
                horizon_s=float(horizon_s),
            )
            if not points:
                continue
            rebased.append({
                "trajectory": points,
                "probability": float(source.get("probability", 0.0)),
                "maneuver": str(source.get("maneuver", "unknown")),
                "position_sigma_m": float(
                    source.get("position_sigma_m", 0.5)
                ),
            })
        probability_sum = sum(
            float(mode["probability"]) for mode in rebased
        )
        if rebased and probability_sum <= 1.0e-9:
            probability_sum = float(len(rebased))
            for mode in rebased:
                mode["probability"] = 1.0
        for mode in rebased:
            mode["probability"] = float(mode["probability"]) / probability_sum
        return rebased
