"""Arc-length stable reference windows for receding-horizon planning."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class StableReferenceWindow:
    samples: tuple[dict[str, object], ...]
    projection_s_m: float
    start_s_m: float
    reason: str


def _xy(sample: Mapping[str, object]) -> tuple[float, float]:
    return (float(sample.get("x_ref_m", sample.get("x", 0.0))),
            float(sample.get("y_ref_m", sample.get("y", 0.0))))


def _arc(samples: Sequence[Mapping[str, object]]) -> list[float]:
    arc = [0.0]
    for first, second in zip(samples[:-1], samples[1:]):
        ax, ay = _xy(first)
        bx, by = _xy(second)
        arc.append(arc[-1] + math.hypot(bx - ax, by - ay))
    return arc


def _project_s(samples, arc, *, x_m, y_m, lower_s_m, upper_s_m=float("inf")):
    best_distance_sq = float("inf")
    best_s = max(0.0, float(lower_s_m))
    for index, (first, second) in enumerate(zip(samples[:-1], samples[1:])):
        if arc[index + 1] < lower_s_m - 1.0e-6:
            continue
        if arc[index] > float(upper_s_m) + 1.0e-6:
            break
        ax, ay = _xy(first)
        bx, by = _xy(second)
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        ratio = 0.0 if length_sq <= 1.0e-12 else (
            ((x_m - ax) * dx + (y_m - ay) * dy) / length_sq
        )
        ratio = min(1.0, max(0.0, ratio))
        station = arc[index] + ratio * (arc[index + 1] - arc[index])
        if station > float(upper_s_m) + 1.0e-6:
            station = float(upper_s_m)
            span = max(1.0e-9, arc[index + 1] - arc[index])
            ratio = min(1.0, max(0.0, (station - arc[index]) / span))
        px, py = ax + ratio * dx, ay + ratio * dy
        if station + 1.0e-6 < lower_s_m:
            continue
        distance_sq = (x_m - px) ** 2 + (y_m - py) ** 2
        if distance_sq < best_distance_sq:
            best_distance_sq, best_s = distance_sq, station
    return float(best_s)


def _interpolate(first, second, ratio, station_m):
    ratio = min(1.0, max(0.0, float(ratio)))
    result = dict(first if ratio < 0.5 else second)
    ax, ay = _xy(first)
    bx, by = _xy(second)
    result["x_ref_m"] = result["x"] = ax + ratio * (bx - ax)
    result["y_ref_m"] = result["y"] = ay + ratio * (by - ay)
    for key in ("curvature_1pm", "speed_ref_mps", "v_ref_mps", "speed_mps",
                "lane_change_progress", "frenet_d_m", "lane_width_m"):
        try:
            a = float(first.get(key, second.get(key, 0.0)))
            b = float(second.get(key, a))
            result[key] = a + ratio * (b - a)
        except (TypeError, ValueError):
            pass
    try:
        a = float(first.get("heading_rad", 0.0))
        b = float(second.get("heading_rad", a))
        delta = math.atan2(math.sin(b - a), math.cos(b - a))
        result["heading_rad"] = math.atan2(
            math.sin(a + ratio * delta), math.cos(a + ratio * delta)
        )
    except (TypeError, ValueError):
        pass
    result["reference_global_s_m"] = float(station_m)
    result["s_ref_m"] = float(station_m)
    result["progress_m"] = float(station_m)
    return result


class StableReferenceLineProvider:
    """Extract monotonic interpolated windows from an immutable master line."""

    def reference_for_lane(
            self, snapshot, *, lane_id, target_speed_mps=0.0):
        """Return one frozen AD-map lane centerline without ego anchoring."""
        if snapshot is None or not bool(getattr(snapshot, "valid", False)):
            return [], "local_map_snapshot_invalid"
        lane_id = int(lane_id or 0)
        geometry = snapshot.geometry_for_lane(lane_id)
        if geometry is None or not geometry.centerline:
            return [], "lane_geometry_missing"
        speed_mps = max(0.0, float(target_speed_mps))
        samples = [{
            "x_ref_m": float(point.x_m),
            "y_ref_m": float(point.y_m),
            "x": float(point.x_m),
            "y": float(point.y_m),
            "heading_rad": float(point.heading_rad),
            "curvature_1pm": float(point.curvature_1pm),
            "lane_width_m": float(point.lane_width_m),
            "left_boundary_x_m": float(point.left_boundary_x_m),
            "left_boundary_y_m": float(point.left_boundary_y_m),
            "right_boundary_x_m": float(point.right_boundary_x_m),
            "right_boundary_y_m": float(point.right_boundary_y_m),
            "road_left_width_m": math.hypot(
                float(point.left_boundary_x_m) - float(point.x_m),
                float(point.left_boundary_y_m) - float(point.y_m),
            ),
            "road_right_width_m": math.hypot(
                float(point.right_boundary_x_m) - float(point.x_m),
                float(point.right_boundary_y_m) - float(point.y_m),
            ),
            "boundary_source": str(point.boundary_source),
            "lane_id": lane_id,
            "speed_ref_mps": speed_mps,
            "v_ref_mps": speed_mps,
            "reference_geometry_owner": "local_map_snapshot",
        } for point in geometry.centerline]
        if len(samples) < 2:
            return [], "lane_geometry_too_short"
        return samples, "local_map_snapshot_lane:%d" % lane_id

    def reference_for_lane_chain(
            self, snapshot, *, lane_ids, target_speed_mps=0.0,
            maximum_join_distance_m=5.0):
        """Build one corridor from explicitly ordered AD-map lane segments."""
        ordered_ids = []
        for value in list(lane_ids or []):
            lane_id = int(value or 0)
            if lane_id and lane_id not in ordered_ids:
                ordered_ids.append(lane_id)
        combined, accepted = [], []
        for lane_id in ordered_ids:
            lane_samples, reason = self.reference_for_lane(
                snapshot, lane_id=lane_id, target_speed_mps=target_speed_mps)
            if not lane_samples:
                return [], "lane_chain_missing:%d:%s" % (lane_id, reason)
            if combined:
                ax, ay = _xy(combined[-1])
                bx, by = _xy(lane_samples[0])
                join_m = math.hypot(bx - ax, by - ay)
                if join_m > max(0.0, float(maximum_join_distance_m)):
                    return [], "lane_chain_disconnected:%d:%.3f" % (lane_id, join_m)
                if join_m <= 1.0e-4:
                    lane_samples = lane_samples[1:]
            combined.extend(dict(sample) for sample in lane_samples)
            accepted.append(lane_id)
        if len(combined) < 2:
            return [], "lane_chain_too_short"
        return combined, "local_map_snapshot_lane_chain:" + ">".join(
            str(lane_id) for lane_id in accepted)

    def reference_for_corridor(
            self, snapshot, *, offset, start_lane_id, target_speed_mps=0.0,
            maximum_join_distance_m=5.0):
        """Follow connected longitudinal segments inside one local corridor."""
        if snapshot is None or not bool(getattr(snapshot, "valid", False)):
            return [], "local_map_snapshot_invalid"
        corridor = next((item for item in snapshot.corridors
                         if int(item.offset) == int(offset)), None)
        if corridor is None:
            return [], "local_map_corridor_missing"
        remaining = {int(item.lane_id): item for item in corridor.lane_geometries}
        current = remaining.pop(int(start_lane_id), None)
        if current is None:
            return [], "local_map_corridor_start_missing"
        ordered = [int(start_lane_id)]
        while remaining:
            end = current.centerline[-1]
            choices = []
            for lane_id, geometry in remaining.items():
                start = geometry.centerline[0]
                distance = math.hypot(
                    float(start.x_m) - float(end.x_m),
                    float(start.y_m) - float(end.y_m),
                )
                choices.append((distance, lane_id, geometry))
            distance, lane_id, geometry = min(choices, key=lambda row: row[0])
            if distance > max(0.0, float(maximum_join_distance_m)):
                break
            ordered.append(int(lane_id))
            current = remaining.pop(int(lane_id))
        return self.reference_for_lane_chain(
            snapshot,
            lane_ids=ordered,
            target_speed_mps=target_speed_mps,
            maximum_join_distance_m=maximum_join_distance_m,
        )

    def reference_from_local_map(
            self, snapshot, *, start_lane_id, target_speed_mps=0.0,
            maximum_join_distance_m=5.0):
        """Build a topology-ordered master solely from LocalMapSnapshot."""
        if snapshot is None or not bool(getattr(snapshot, "valid", False)):
            return [], "local_map_snapshot_invalid"
        start_lane_id = int(start_lane_id or 0)
        route_sequence = list(getattr(snapshot, "route_lane_sequence", ()) or ())
        if start_lane_id in route_sequence:
            route_sequence = route_sequence[route_sequence.index(start_lane_id):]
        else:
            route_sequence = [start_lane_id]
        samples = []
        accepted_lanes = []
        for lane_id in route_sequence:
            geometry = snapshot.geometry_for_lane(int(lane_id))
            if geometry is None or not geometry.centerline:
                if int(lane_id) == start_lane_id:
                    return [], "start_lane_geometry_missing"
                break
            lane_samples = []
            for point in geometry.centerline:
                lane_samples.append({
                    "x_ref_m": float(point.x_m),
                    "y_ref_m": float(point.y_m),
                    "x": float(point.x_m),
                    "y": float(point.y_m),
                    "heading_rad": float(point.heading_rad),
                    "curvature_1pm": float(point.curvature_1pm),
                    "lane_width_m": float(point.lane_width_m),
                    "left_boundary_x_m": float(point.left_boundary_x_m),
                    "left_boundary_y_m": float(point.left_boundary_y_m),
                    "right_boundary_x_m": float(point.right_boundary_x_m),
                    "right_boundary_y_m": float(point.right_boundary_y_m),
                    "road_left_width_m": math.hypot(
                        float(point.left_boundary_x_m) - float(point.x_m),
                        float(point.left_boundary_y_m) - float(point.y_m),
                    ),
                    "road_right_width_m": math.hypot(
                        float(point.right_boundary_x_m) - float(point.x_m),
                        float(point.right_boundary_y_m) - float(point.y_m),
                    ),
                    "boundary_source": str(point.boundary_source),
                    "lane_id": int(lane_id),
                    "speed_ref_mps": max(0.0, float(target_speed_mps)),
                    "v_ref_mps": max(0.0, float(target_speed_mps)),
                    "reference_geometry_owner": "local_map_snapshot",
                    # Every point after the first AD-map lane segment belongs
                    # to a topology-approved longitudinal continuation.  The
                    # reference contract can therefore distinguish a normal
                    # lane-id boundary from an unauthorized lateral change.
                    "lane_transition_kind": (
                        "longitudinal_successor"
                        if accepted_lanes
                        else "lane_center"
                    ),
                })
            if samples:
                previous_x, previous_y = _xy(samples[-1])
                next_x, next_y = _xy(lane_samples[0])
                join_distance_m = math.hypot(
                    next_x - previous_x, next_y - previous_y
                )
                if join_distance_m > max(0.0, float(maximum_join_distance_m)):
                    break
                if join_distance_m <= 1.0e-4:
                    lane_samples = lane_samples[1:]
            samples.extend(lane_samples)
            accepted_lanes.append(int(lane_id))
        if len(samples) < 2:
            return [], "local_map_reference_too_short"
        return samples, "local_map_snapshot_route:" + ">".join(
            str(lane_id) for lane_id in accepted_lanes
        )

    def post_turn_master(
            self, snapshot, *, start_lane_id, target_speed_mps,
            horizon_steps, required_arc_m,
            maximum_join_distance_m=5.0):
        """Build and validate the topology-owned outgoing reference master."""
        samples, reason = self.reference_from_local_map(
            snapshot,
            start_lane_id=int(start_lane_id),
            target_speed_mps=float(target_speed_mps),
            maximum_join_distance_m=float(maximum_join_distance_m),
        )
        if len(samples) < int(horizon_steps):
            return [], "post_turn_master_short:%d<%d:%s" % (
                len(samples), int(horizon_steps), reason)
        arc_m = _arc(samples)[-1] if len(samples) >= 2 else 0.0
        if float(arc_m) + 1.0e-3 < float(required_arc_m):
            return [], "post_turn_master_arc_short:%.3f<%.3f:%s" % (
                float(arc_m), float(required_arc_m), reason)
        for sample in samples:
            sample.setdefault("lane_transition_kind", "post_turn_exit_centerline")
            sample["speed_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["v_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["speed_mps"] = max(0.0, float(target_speed_mps))
        return samples, "post_turn_master:" + str(reason)

    def target_lane_stabilization_master(
            self, *, ego_location, ego_yaw_rad, target_lane_id,
            horizon_steps, step_distance_m, route_points,
            target_speed_mps, max_curvature_1pm):
        """Build the immutable lane-change completion reference.

        The bridge supplies planning state only.  Geometry construction and
        conditioning stay behind the provider boundary so there is one
        reference owner during the lane-change-to-lane-follow handoff.
        """
        builder = self.builder
        if builder is None:
            return [], "target_lane_stabilization_builder_missing"
        samples = builder.target_lane_stabilization_samples(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            target_lane_id=int(target_lane_id),
            horizon_steps=int(horizon_steps),
            step_distance_m=float(step_distance_m),
            route_points=list(route_points or []),
        )
        samples, conditioning_reason = builder.curvature_feasible_samples(
            reference_samples=list(samples or []),
            ego_location=ego_location,
            ego_heading_rad=float(ego_yaw_rad),
            max_curvature_1pm=float(max_curvature_1pm),
            mode="target_lane_stabilization",
        )
        speed_mps = max(0.0, float(target_speed_mps))
        for sample in samples:
            sample["lane_id"] = int(target_lane_id)
            sample["lane_change_progress"] = 1.0
            sample["lane_transition_kind"] = "target_lane_stabilization"
            sample["speed_ref_mps"] = speed_mps
            sample["v_ref_mps"] = speed_mps
            sample["speed_mps"] = speed_mps
        return list(samples), str(conditioning_reason or "not_required")

    def preturn_lane_reference(
            self, snapshot, *, lane_id, ego_x_m, ego_y_m,
            target_speed_mps, first_forward_m, spacing_m, horizon_steps):
        """Produce the current-lane-only reference before connector handoff."""
        master, reason = self.reference_for_lane(
            snapshot,
            lane_id=int(lane_id),
            target_speed_mps=float(target_speed_mps),
        )
        if not master:
            return [], str(reason)
        window = self.window_from_reference(
            master,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            lower_s_m=0.0,
            first_forward_m=float(first_forward_m),
            spacing_m=max(0.05, float(spacing_m)),
            count=int(horizon_steps),
        )
        reference = [dict(sample) for sample in window.samples]
        # Do not cross a topology boundary just to fill the fixed MPC horizon.
        # Extend the terminal AD-map tangent while the connector is still
        # outside its fixed handoff arc.
        if len(reference) >= 2:
            while len(reference) < int(horizon_steps):
                previous, terminal = reference[-2], reference[-1]
                dx_m = float(terminal["x_ref_m"]) - float(previous["x_ref_m"])
                dy_m = float(terminal["y_ref_m"]) - float(previous["y_ref_m"])
                step_m = max(0.05, math.hypot(dx_m, dy_m))
                heading_rad = math.atan2(dy_m, dx_m)
                padded = dict(terminal)
                padded["x_ref_m"] = float(terminal["x_ref_m"]) + (
                    step_m * math.cos(heading_rad)
                )
                padded["y_ref_m"] = float(terminal["y_ref_m"]) + (
                    step_m * math.sin(heading_rad)
                )
                padded["x"] = float(padded["x_ref_m"])
                padded["y"] = float(padded["y_ref_m"])
                padded["heading_rad"] = float(heading_rad)
                padded["curvature_1pm"] = 0.0
                padded["lane_transition_kind"] = (
                    "current_ad_lane_terminal_tangent"
                )
                padded["corridor_center_x_m"] = float(padded["x_ref_m"])
                padded["corridor_center_y_m"] = float(padded["y_ref_m"])
                padded["corridor_heading_rad"] = float(heading_rad)
                reference.append(padded)
        speed_mps = max(0.0, float(target_speed_mps))
        for sample in reference:
            sample["v_ref_mps"] = speed_mps
            sample["speed_ref_mps"] = speed_mps
            sample["speed_mps"] = speed_mps
        return reference, "preturn_lane:" + str(reason)

    def lane_change_nominal(
            self, snapshot, *, ego_x_m, ego_y_m, ego_heading_rad,
            current_lane_id, target_lane_id, target_reference,
            fallback_source_reference, target_speed_mps, geometry_speed_mps,
            geometry_length_m, transition_duration_s, spacing_m,
            horizon_steps, lane_width_m):
        """Generate one Frenet d(s) lane-change path between AD-map corridors."""
        from .reference_geometry import (
            align_parallel_reference,
            build_reference_line,
            frenet_lane_change_path,
        )

        source_master, source_reason = self.reference_for_corridor(
            snapshot,
            offset=0,
            start_lane_id=int(current_lane_id),
            target_speed_mps=float(target_speed_mps),
        )
        source_count = max(
            int(horizon_steps),
            int(math.ceil(
                float(geometry_length_m) / max(0.05, float(spacing_m))
            )) + int(horizon_steps),
        )
        source_window = self.window_from_reference(
            source_master,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            lower_s_m=0.0,
            first_forward_m=float(spacing_m),
            spacing_m=float(spacing_m),
            count=int(source_count),
        ) if source_master else None
        physical_source = (
            [dict(sample) for sample in source_window.samples]
            if source_window is not None
            else [dict(sample) for sample in fallback_source_reference]
        )
        aligned_source = align_parallel_reference(
            target_reference,
            physical_source,
        )
        candidate_source = [{
            "x_ref_m": float(ego_x_m),
            "y_ref_m": float(ego_y_m),
            "lane_id": int(current_lane_id),
            "lane_width_m": float(lane_width_m),
        }] + [dict(sample) for sample in aligned_source]
        source_line = build_reference_line(
            candidate_source,
            spacing_m=float(spacing_m),
            default_lane_id=int(current_lane_id),
        )
        offsets = []
        target_rows = [dict(sample) for sample in target_reference]
        for source_sample, target_sample in zip(aligned_source, target_rows):
            sx, sy = _xy(source_sample)
            tx, ty = _xy(target_sample)
            heading = float(source_sample.get("heading_rad", ego_heading_rad))
            offsets.append(
                -math.sin(heading) * (tx - sx)
                + math.cos(heading) * (ty - sy)
            )
        debug = {"source_corridor_reason": str(source_reason)}
        if not offsets or not source_line.valid:
            return [], debug
        offsets.sort()
        lateral_offset_m = float(offsets[len(offsets) // 2])
        minimum_separation_m = max(0.5, 0.25 * float(lane_width_m))
        if abs(lateral_offset_m) < minimum_separation_m:
            debug["rejection"] = "target_corridor_not_laterally_separated"
            return [], debug
        first_source, first_target = aligned_source[0], target_rows[0]
        sx, sy = _xy(first_source)
        tx, ty = _xy(first_target)
        target_origin = {
            "x_ref_m": float(ego_x_m) + tx - sx,
            "y_ref_m": float(ego_y_m) + ty - sy,
            "lane_id": int(target_lane_id),
            "lane_width_m": float(lane_width_m),
        }
        target_line = build_reference_line(
            [target_origin] + target_rows,
            spacing_m=float(spacing_m),
            default_lane_id=int(target_lane_id),
        )
        reference = frenet_lane_change_path(
            source_line,
            lateral_offset_m=float(lateral_offset_m),
            transition_length_m=max(
                float(geometry_length_m),
                float(geometry_speed_mps) * float(transition_duration_s),
            ),
            target_lane_id=int(target_lane_id),
            target_speed_mps=float(target_speed_mps),
            target_reference_line=target_line,
        )
        debug["lateral_offset_m"] = float(lateral_offset_m)
        return [dict(sample) for sample in reference], debug

    def lane_change_completion_reference(
            self, snapshot, *, target_lane_id, target_speed_mps):
        """Return only the committed target corridor and its successors."""
        if snapshot is None:
            return [], "lane_change_completion_snapshot_missing"
        lane_to_offset = dict(
            getattr(snapshot, "lane_to_offset", {}) or {}
        )
        target_lane_id = int(target_lane_id)
        if target_lane_id not in lane_to_offset:
            return self.reference_for_lane(
                snapshot,
                lane_id=target_lane_id,
                target_speed_mps=float(target_speed_mps),
            )
        return self.reference_for_corridor(
            snapshot,
            offset=int(lane_to_offset[target_lane_id]),
            start_lane_id=target_lane_id,
            target_speed_mps=float(target_speed_mps),
        )

    def lane_change_target_reference(
            self, snapshot, *, target_lane_id, target_speed_mps):
        """Target corridor for exactly one committed lateral transition.

        The global route's next target may describe a later lateral edge.  It
        must never be appended to the current maneuver geometry.  Resolve the
        committed physical lane inside this snapshot and continue only through
        longitudinal segments carrying the same corridor offset.
        """
        return self.lane_change_completion_reference(
            snapshot,
            target_lane_id=int(target_lane_id),
            target_speed_mps=float(target_speed_mps),
        )

    def condition_lane_change_reference(
            self, reference, *, ego_location, ego_heading_rad,
            max_curvature_1pm, mode="committed_lane_change"):
        """Apply the provider-owned feasibility conditioning to an LC window."""
        samples, reason = self.builder.curvature_feasible_samples(
            reference_samples=[dict(sample) for sample in reference],
            ego_location=ego_location,
            ego_heading_rad=float(ego_heading_rad),
            max_curvature_1pm=float(max_curvature_1pm),
            mode=str(mode),
        )
        return [dict(sample) for sample in samples], str(reason or "")

    def window_from_reference(self, reference, *, ego_x_m, ego_y_m,
                              lower_s_m, first_forward_m, spacing_m, count,
                              max_projection_advance_m=float("inf")):
        samples = [dict(sample) for sample in list(reference or [])]
        if len(samples) < 2 or int(count) <= 0:
            return StableReferenceWindow((), float(lower_s_m), float(lower_s_m),
                                         "reference_unavailable")
        arc = _arc(samples)
        projection_s = _project_s(
            samples, arc, x_m=float(ego_x_m), y_m=float(ego_y_m),
            lower_s_m=max(0.0, float(lower_s_m)),
            upper_s_m=(
                float("inf")
                if not math.isfinite(float(max_projection_advance_m))
                else max(0.0, float(lower_s_m))
                + max(0.0, float(max_projection_advance_m))
            ),
        )
        start_s = min(arc[-1], max(float(lower_s_m), projection_s)
                      + max(0.0, float(first_forward_m)))
        spacing = max(0.05, float(spacing_m))
        # Never fill a fixed horizon by repeating the terminal master point.
        # Consumers that require a full horizon own geometric extrapolation;
        # duplicate XY samples violate the reference contract and used to
        # make a valid committed maneuver fail periodically as its window
        # approached the end of the immutable master.
        stations = []
        for index in range(max(1, int(count))):
            station = float(start_s) + float(index) * float(spacing)
            if station > float(arc[-1]) + 1.0e-9:
                break
            stations.append(min(float(arc[-1]), float(station)))
        if not stations:
            stations = [float(arc[-1])]
        result = []
        segment = 0
        for station in stations:
            while segment + 1 < len(arc) - 1 and arc[segment + 1] < station:
                segment += 1
            span = max(1.0e-9, arc[segment + 1] - arc[segment])
            result.append(_interpolate(samples[segment], samples[segment + 1],
                                       (station - arc[segment]) / span, station))
        bounded = math.isfinite(float(max_projection_advance_m))
        return StableReferenceWindow(
            tuple(result),
            projection_s,
            start_s,
            "arc_length_projection_stitched_bounded"
            if bounded
            else "arc_length_projection_stitched",
        )
