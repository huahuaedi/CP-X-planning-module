"""Summarize staged HD-map matching diagnostics from one CP-X debug CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MAP_CONTRACT_FIELDS = (
    "map_match_center_x_m",
    "map_match_center_y_m",
    "map_match_lane_width_m",
    "map_match_score",
    "map_match_candidate_count",
    "local_lane_frame_ego_ad_lane_id",
    "local_lane_frame_forward_distance_m",
    "local_lane_frame_backward_distance_m",
    "local_lane_frame_lane_to_offset",
    "local_lane_frame_route_target_ad_lane_id",
    "local_lane_frame_target_in_frame",
)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _corridor_offset(row: dict[str, str], lane_id: str) -> int | None:
    try:
        lane_to_offset = json.loads(
            row.get("local_lane_frame_lane_to_offset", "") or "{}"
        )
        if str(lane_id) in lane_to_offset:
            return int(lane_to_offset[str(lane_id)])
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    try:
        corridors = json.loads(row.get("local_lane_frame_corridors", "") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    for raw_offset, lane_ids in dict(corridors or {}).items():
        if str(lane_id) in {str(item) for item in list(lane_ids or [])}:
            try:
                return int(raw_offset)
            except (TypeError, ValueError):
                return None
    return None


def _lateral_action_active(*rows: dict[str, str]) -> bool:
    for row in rows:
        behavior = str(row.get("behavior_decision", "")).strip().lower()
        geometry = str(row.get("maneuver_geometry_type", "")).strip().lower()
        phase = str(row.get("lane_change_phase", "")).strip().lower()
        if (
            _truthy(row.get("lane_change_authorized", ""))
            or behavior in {"lane_change_left", "lane_change_right"}
            or geometry in {"lane_change", "lane_change_to_turn"}
            or phase in {"executing", "target_lane_stabilization"}
        ):
            return True
    return False


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def analyze(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    valid_rows = [row for row in rows if _truthy(row.get("map_match_valid", ""))]
    unmatched = len(rows) - len(valid_rows)
    lane_transitions = []
    unauthorized_lateral_transitions = []
    previous = None
    previous_row = None
    transition_indices = []
    for valid_index, row in enumerate(valid_rows):
        lane_id = str(row.get("map_match_ad_lane_id", ""))
        if previous is not None and lane_id != previous:
            offset = _corridor_offset(previous_row, lane_id) if previous_row else None
            transition_type = (
                "longitudinal" if offset == 0
                else "lateral_left" if offset is not None and offset > 0
                else "lateral_right" if offset is not None and offset < 0
                else "unclassified"
            )
            authorized = _lateral_action_active(previous_row or {}, row)
            transition = {
                "sim_time_s": row.get("sim_time_s", ""),
                "from_ad_lane_id": previous,
                "to_ad_lane_id": lane_id,
                "road_id": row.get("map_match_road_id", ""),
                "section_id": row.get("map_match_section_id", ""),
                "local_offset_before_transition": offset,
                "transition_type": transition_type,
                "lateral_action_active": authorized,
            }
            lane_transitions.append(transition)
            transition_indices.append((valid_index, previous, lane_id, transition))
            if transition_type.startswith("lateral_") and not authorized:
                unauthorized_lateral_transitions.append(dict(transition))
        previous = lane_id
        previous_row = row
    short_lane_bounces = []
    max_bounce_frames = 20
    for first, second in zip(transition_indices[:-1], transition_indices[1:]):
        first_index, lane_a, lane_b, first_transition = first
        second_index, lane_b_again, lane_a_again, second_transition = second
        if (
            lane_b == lane_b_again
            and lane_a == lane_a_again
            and int(second_index) - int(first_index) <= int(max_bounce_frames)
        ):
            short_lane_bounces.append({
                "from_ad_lane_id": lane_a,
                "temporary_ad_lane_id": lane_b,
                "return_ad_lane_id": lane_a_again,
                "start_sim_time_s": first_transition.get("sim_time_s", ""),
                "end_sim_time_s": second_transition.get("sim_time_s", ""),
                "duration_frames": int(second_index) - int(first_index),
                "first_transition_type": first_transition.get("transition_type", ""),
                "second_transition_type": second_transition.get("transition_type", ""),
            })
    schema = set(rows[0]) if rows else set()
    missing_contract_fields = [
        field for field in MAP_CONTRACT_FIELDS if field not in schema
    ]
    contract_violations = []
    previous_lane_id = ""
    for row in rows:
        reasons = []
        lane_id = str(row.get("map_match_ad_lane_id", ""))
        if _truthy(row.get("map_match_valid", "")):
            confidence = _number(row, "map_match_confidence")
            width = _number(row, "map_match_lane_width_m")
            lateral = _number(row, "map_match_lateral_offset_m")
            heading = _number(row, "map_match_heading_error_rad")
            candidates = _number(row, "map_match_candidate_count")
            if confidence is not None and confidence < 0.45:
                reasons.append("map_match_confidence_below_0.45")
            if width is not None and width <= 0.0:
                reasons.append("non_positive_lane_width")
            if width is not None and lateral is not None and abs(lateral) > 0.75 * width:
                reasons.append("lateral_offset_outside_lane_envelope")
            if heading is not None and abs(heading) > 0.7853981633974483:
                reasons.append("heading_error_exceeds_45deg")
            if candidates is not None and candidates < 1:
                reasons.append("valid_match_without_candidates")
        if previous_lane_id and lane_id != previous_lane_id and _truthy(
            row.get("local_lane_frame_cache_reused", "")
        ):
            reasons.append("cache_reused_across_lane_transition")
        try:
            corridors = json.loads(row.get("local_lane_frame_corridors", "") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            corridors = {}
            reasons.append("invalid_corridor_serialization")
        normalized_corridors = {
            int(offset): {str(item) for item in list(lane_ids or [])}
            for offset, lane_ids in dict(corridors or {}).items()
        }
        if _truthy(row.get("map_match_valid", "")) and lane_id not in normalized_corridors.get(0, set()):
            reasons.append("matched_lane_not_in_current_corridor")
        if any(abs(offset) > 1 for offset in normalized_corridors):
            reasons.append("corridor_offset_outside_supported_range")
        membership = {}
        for offset, lane_ids in normalized_corridors.items():
            for member in lane_ids:
                membership.setdefault(member, []).append(offset)
        try:
            lane_to_offset = {
                str(lane): int(offset)
                for lane, offset in dict(json.loads(
                    row.get("local_lane_frame_lane_to_offset", "") or "{}"
                ) or {}).items()
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            lane_to_offset = {}
            reasons.append("invalid_lane_to_offset_serialization")
        for member, offset in lane_to_offset.items():
            if abs(int(offset)) > 1:
                reasons.append("lane_to_offset_outside_supported_range")
            if member not in normalized_corridors.get(int(offset), set()):
                reasons.append("lane_to_offset_not_backed_by_corridor")
        if lane_id and lane_to_offset.get(lane_id) != 0:
            reasons.append("matched_lane_offset_is_not_zero")
        local_ego = str(row.get("local_lane_frame_ego_ad_lane_id", ""))
        if local_ego and lane_id and local_ego != lane_id:
            reasons.append("matcher_and_local_frame_ego_disagree")
        for key in (
            "local_lane_frame_forward_distance_m",
            "local_lane_frame_backward_distance_m",
        ):
            distance = _number(row, key)
            if distance is not None and distance < 99.0:
                reasons.append(f"{key}_below_99m")
        route_target = str(row.get("local_lane_frame_route_target_ad_lane_id", ""))
        if route_target and route_target != "0":
            observed = route_target in lane_to_offset
            reported = _truthy(row.get("local_lane_frame_target_in_frame", ""))
            if bool(observed) != bool(reported):
                reasons.append("route_target_membership_flag_mismatch")
        if reasons:
            contract_violations.append({
                "sim_time_s": row.get("sim_time_s", ""),
                "reasons": sorted(set(reasons)),
            })
        previous_lane_id = lane_id
    violations = []
    for row in rows:
        text = str(row.get("local_lane_frame_invariant_violations", "")).strip()
        if text:
            violations.append({"sim_time_s": row.get("sim_time_s", ""), "reason": text})
    direction_conflicts = []
    for row in rows:
        try:
            offset = int(float(row.get("local_lane_frame_target_offset", "0") or 0))
        except (TypeError, ValueError):
            offset = 0
        local_direction = "left" if offset > 0 else "right" if offset < 0 else ""
        authorization = str(row.get("lane_change_authorization_direction", "")).strip().lower()
        behavior = str(row.get("behavior_decision", "")).strip().lower()
        behavior_direction = (
            "left" if behavior == "lane_change_left"
            else "right" if behavior == "lane_change_right"
            else ""
        )
        geometry = str(row.get("maneuver_geometry_direction", "")).strip().lower()
        geometry_type = str(row.get("maneuver_geometry_type", "")).strip().lower()
        mismatches = []
        if local_direction and authorization and local_direction != authorization:
            mismatches.append(f"local={local_direction},authorization={authorization}")
        if (
            local_direction
            and geometry_type in {"lane_change", "lane_change_to_turn"}
            and geometry
            and local_direction != geometry
        ):
            mismatches.append(f"local={local_direction},geometry={geometry}")
        if local_direction and behavior_direction and local_direction != behavior_direction:
            mismatches.append(f"local={local_direction},behavior={behavior_direction}")
        if mismatches:
            direction_conflicts.append({
                "sim_time_s": row.get("sim_time_s", ""),
                "conflict": ";".join(mismatches),
            })
    confidences = []
    for row in valid_rows:
        try:
            confidences.append(float(row.get("map_match_confidence", "")))
        except (TypeError, ValueError):
            pass
    return {
        "rows": len(rows),
        "valid_map_match_rows": len(valid_rows),
        "unmatched_rows": unmatched,
        "minimum_confidence": min(confidences) if confidences else None,
        "mean_confidence": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        "ad_lane_transition_count": len(lane_transitions),
        "ad_lane_transitions": lane_transitions,
        "unauthorized_lateral_transition_count": len(unauthorized_lateral_transitions),
        "unauthorized_lateral_transitions": unauthorized_lateral_transitions[:50],
        "short_lane_bounce_count": len(short_lane_bounces),
        "short_lane_bounces": short_lane_bounces[:50],
        "map_contract_schema_missing_field_count": len(missing_contract_fields),
        "map_contract_schema_missing_fields": missing_contract_fields,
        "map_contract_violation_count": len(contract_violations),
        "map_contract_violations": contract_violations[:50],
        "local_frame_invariant_violation_count": len(violations),
        "local_frame_invariant_violations": violations[:50],
        "downstream_direction_conflict_count": len(direction_conflicts),
        "downstream_direction_conflicts": direction_conflicts[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("debug_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-violations", action="store_true")
    args = parser.parse_args()
    summary = analyze(args.debug_csv)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_violations and any((
        int(summary["unmatched_rows"]) > 0,
        int(summary["local_frame_invariant_violation_count"]) > 0,
        int(summary["downstream_direction_conflict_count"]) > 0,
        int(summary["unauthorized_lateral_transition_count"]) > 0,
        int(summary["short_lane_bounce_count"]) > 0,
        int(summary["map_contract_schema_missing_field_count"]) > 0,
        int(summary["map_contract_violation_count"]) > 0,
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
