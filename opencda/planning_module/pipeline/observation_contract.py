"""Canonical planner-facing observation contract.

Provider-specific decoding ends here.  Downstream perception, prediction and
risk stages keep using mappings for Python 3.7 compatibility, but every mapping
leaving this module has the same identity, semantic type and provenance fields.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def normalize_object_type(value):
    kind = str(value or "unknown").strip().lower()
    aliases = {
        "car": "vehicle", "cav": "vehicle", "truck": "vehicle",
        "bus": "vehicle", "motorcycle": "vehicle", "walker": "pedestrian",
        "bicycle": "cyclist", "static": "static_object",
        "roadway_object": "static_object", "debris": "static_object",
        "obstacle": "static_object",
    }
    return aliases.get(kind, kind or "unknown")


def add_provenance(snapshot, *, channel, received_timestamp_s):
    result = dict(snapshot)
    channel = "cp" if str(channel).lower() == "cp" else "local"
    try:
        source_time_s = float(result.get("timestamp_s", received_timestamp_s))
    except (TypeError, ValueError):
        source_time_s = float(received_timestamp_s)
    if not math.isfinite(source_time_s):
        source_time_s = float(received_timestamp_s)
    age_s = max(0.0, float(received_timestamp_s) - source_time_s)
    result.update({
        "object_type": normalize_object_type(
            result.get("object_type", result.get("type", "unknown"))
        ),
        "observation_sources": [channel],
        "locally_observed": channel == "local",
        "cooperatively_observed": channel == "cp",
        "received_timestamp_s": float(received_timestamp_s),
        "source_timestamp_s": source_time_s,
        "observation_age_s": age_s,
        channel + "_timestamp_s": source_time_s,
        channel + "_age_s": age_s,
    })
    return result


def merge_metadata(preferred, other):
    """Merge evidence and prediction while preserving preferred kinematics."""

    result = dict(preferred)
    result["observation_sources"] = _ordered_union(
        preferred.get("observation_sources", ()),
        other.get("observation_sources", ()),
    )
    result["locally_observed"] = bool(
        preferred.get("locally_observed") or other.get("locally_observed")
    )
    result["cooperatively_observed"] = bool(
        preferred.get("cooperatively_observed")
        or other.get("cooperatively_observed")
    )
    copied_fields = (
        "local_timestamp_s", "local_age_s", "cp_timestamp_s", "cp_age_s",
        "cp_message_id", "valid_until_s", "ttl_s", "plan_revision",
        "prediction_timestamp_s", "predicted_modes", "predicted_trajectory",
        "future_trajectory", "position_covariance", "velocity_covariance",
    )
    for key in copied_fields:
        if key not in result and key in other:
            result[key] = other[key]
    for key in ("observed_by_cav_ids", "not_observed_by_cav_ids"):
        values = _ordered_union(preferred.get(key, ()), other.get(key, ()))
        if values:
            result[key] = values
    result["blind_spot_shared"] = bool(
        preferred.get("blind_spot_shared") or other.get("blind_spot_shared")
    )
    if (
        str(result.get("object_type", "unknown")) == "unknown"
        and other.get("object_type")
    ):
        result["object_type"] = str(other["object_type"])
    return result


def normalize_local(snapshot):
    try:
        obstacle_id = str(
            snapshot.get("vehicle_id", snapshot.get("id", ""))
        ).strip()
        if not obstacle_id:
            return None
        result = {
            "vehicle_id": obstacle_id, "id": obstacle_id,
            "x": float(snapshot.get("x", 0.0)),
            "y": float(snapshot.get("y", 0.0)),
            "v": float(snapshot.get("v", 0.0)),
            "psi": float(snapshot.get("psi", 0.0)),
            "length_m": float(snapshot.get("length_m", 4.5)),
            "width_m": float(snapshot.get("width_m", 2.0)),
            "source": str(snapshot.get("source", "opencda_perception")),
            "provider_source": str(snapshot.get(
                "provider_source", "native_opencda_perception"
            )),
            "confidence": float(snapshot.get("confidence", 1.0)),
            "object_type": normalize_object_type(
                snapshot.get("object_type", snapshot.get("type", "vehicle"))
            ),
        }
        _copy_optional_prediction_fields(result, snapshot)
        return result
    except (TypeError, ValueError):
        return None


def normalize_cp(obstacle):
    try:
        raw_id = str(obstacle.get("id", obstacle.get("vehicle_id", ""))).strip()
        if not raw_id:
            return None
        state = obstacle.get("state", ())
        values = list(state) if (
            isinstance(state, Sequence)
            and not isinstance(state, (str, bytes, bytearray))
        ) else []
        x_m = obstacle.get("x", obstacle.get(
            "x_m", values[0] if values else None
        ))
        y_m = obstacle.get("y", obstacle.get(
            "y_m", values[1] if len(values) >= 2 else None
        ))
        if x_m is None or y_m is None:
            return None
        shape = obstacle.get("shape", {})
        shape = dict(shape) if isinstance(shape, Mapping) else {}
        obstacle_id = raw_id.rsplit(":", 1)[-1]
        result = {
            "vehicle_id": obstacle_id, "id": obstacle_id,
            "cp_message_id": raw_id,
            "x": float(x_m), "y": float(y_m),
            "v": float(obstacle.get("v", obstacle.get(
                "speed_mps", values[2] if len(values) >= 3 else 0.0
            ))),
            "psi": float(obstacle.get("psi", obstacle.get(
                "heading_rad", values[3] if len(values) >= 4 else 0.0
            ))),
            "length_m": float(shape.get(
                "length_m", obstacle.get("length_m", 4.5)
            )),
            "width_m": float(shape.get(
                "width_m", obstacle.get("width_m", 2.0)
            )),
            "source": str(obstacle.get("source", "opencda_cp")),
            "provider_source": str(obstacle.get(
                "provider_source", "opencda_cp"
            )),
            "confidence": float(obstacle.get("confidence", 0.5)),
            "lane_id": int(float(obstacle.get("lane_id", 0) or 0)),
            "road_id": int(float(obstacle.get("road_id", 0) or 0)),
            "object_type": normalize_object_type(obstacle.get(
                "object_type", obstacle.get("type", "unknown")
            )),
            "observed_by_cav_ids": list(
                obstacle.get("observed_by_cav_ids", ()) or ()
            ),
            "not_observed_by_cav_ids": list(
                obstacle.get("not_observed_by_cav_ids", ()) or ()
            ),
            "blind_spot_shared": bool(obstacle.get("blind_spot_shared", False)),
        }
        _copy_optional_prediction_fields(result, obstacle, include_validity=True)
        return result
    except (TypeError, ValueError):
        return None


def _copy_optional_prediction_fields(result, source, include_validity=False):
    keys = [
        "timestamp_s", "predicted_modes", "predicted_trajectory",
        "future_trajectory", "plan_revision", "prediction_timestamp_s",
        "position_covariance", "velocity_covariance",
    ]
    if include_validity:
        keys.extend(("valid_until_s", "ttl_s"))
    for key in keys:
        if key in source:
            result[key] = source[key]


def _ordered_union(first, second):
    values = []
    for sequence in (first, second):
        for value in list(sequence or ()):
            if value not in values:
                values.append(value)
    return values
