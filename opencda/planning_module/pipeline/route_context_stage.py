"""Map matching, local lane frame and authoritative route context.

Owns the per-tick map match, the frozen local lane graph/snapshot and the
authoritative ego waypoint.  It is the only writer of that state and asks
RouteManager -- the sole owner of route progress -- to sync and publish.
The planner, route manager and clock are passed per call rather than held,
so the stage never keeps a stale reference to a replaced collaborator.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .map_matching import (
    DiagnosticHDMapMatcher,
    LaneProjectionCandidate,
    local_lane_frame_invariants,
    topology_relation,
)
from .local_map_snapshot import LocalMapSnapshot, build_local_map_snapshot


class RouteContextStage:
    def __init__(self) -> None:
        self.hd_map_matcher = DiagnosticHDMapMatcher()
        self.map_matching: dict[str, object] = {}
        self.local_lane_frame: dict[str, object] = {}
        self.local_map_frame_id = 0
        self.local_map_snapshot = LocalMapSnapshot()
        self.authoritative_ego_waypoint: Any = None

    def build(
        self,
        *,
        global_planner: Any,
        route_manager: Any,
        clock: Callable[[], float],
        ego_location: Any,
        ego_heading_rad: float,
        fallback_lane_id: int,
    ) -> dict[str, object]:
        """The AD-map authoritative route summary for this tick."""

        self.authoritative_ego_waypoint = None
        matched_waypoint = None
        try:
            raw_candidates = list(
                global_planner.get_waypoint_candidates(
                    {
                        "x": float(ego_location.x),
                        "y": float(ego_location.y),
                        "z": float(getattr(ego_location, "z", 0.0)),
                    }
                )
                or []
            )
            previous_frame = dict(self.local_lane_frame or {})
            previous_corridors = {
                int(key): list(value or [])
                for key, value in dict(
                    previous_frame.get("corridors", {}) or {}
                ).items()
            }
            previous_lane_id = int(
                getattr(self.hd_map_matcher.previous, "ad_lane_id", 0)
                or 0
            )
            candidates = []
            waypoint_by_lane: dict[int, object] = {}
            from utility.global_planner import world_heading_rad

            for item in raw_candidates:
                waypoint = item.get("waypoint")
                if waypoint is None:
                    continue
                position = dict(getattr(waypoint, "position", {}) or {})
                ad_lane_id = int(item.get("ad_lane_id", 0) or 0)
                waypoint_by_lane.setdefault(ad_lane_id, waypoint)
                candidates.append(LaneProjectionCandidate(
                    ad_lane_id=ad_lane_id,
                    road_id=int(getattr(waypoint, "road_id", 0) or 0),
                    section_id=int(getattr(waypoint, "section_id", 0) or 0),
                    raw_lane_id=int(getattr(waypoint, "lane_id", 0) or 0),
                    center_x_m=float(position.get("x", ego_location.x)),
                    center_y_m=float(position.get("y", ego_location.y)),
                    heading_rad=float(world_heading_rad(waypoint) or 0.0),
                    lane_width_m=max(
                        0.1, float(getattr(waypoint, "lane_width_m", 3.5) or 3.5)
                    ),
                    snap_distance_m=float(item.get("snap_distance_m", 0.0)),
                    is_in_lane=bool(item.get("is_in_lane", False)),
                    probability=float(item.get("probability", 0.0)),
                    topology_relation=topology_relation(
                        candidate_lane_id=ad_lane_id,
                        previous_lane_id=previous_lane_id,
                        previous_corridors=previous_corridors,
                    ),
                ))
            matched = self.hd_map_matcher.update(
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
                ego_heading_rad=float(ego_heading_rad),
                candidates=candidates,
            )
            matched_waypoint = waypoint_by_lane.get(int(matched.ad_lane_id))
            self.map_matching = {
                **matched.as_dict(),
                "candidate_count": len(candidates),
            }
        except Exception as exc:
            self.map_matching = {
                "valid": False,
                "match_reason": f"diagnostic_map_match_failed:{exc}",
                "candidate_count": 0,
            }
        try:
            authoritative_lane_id = int(
                self.map_matching.get("ad_lane_id", 0) or 0
            )
            route_manager.sync_route_progress(
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
                ego_heading_rad=float(ego_heading_rad),
                current_lane_id=int(authoritative_lane_id),
            )
            # RouteManager owns the only runtime cursor.  The global
            # planner stores immutable topology and never advances a
            # second per-query nearest-node index for behavior.
            route_values = route_manager.get_route_info(
                x_m=float(ego_location.x),
                y_m=float(ego_location.y),
                query_key="authoritative_route_cursor",
                fallback_lane_id=int(authoritative_lane_id or fallback_lane_id),
                ego_waypoint=matched_waypoint,
            )
            summary = SimpleNamespace(**dict(route_values))
            summary.distance_to_destination_m = float(
                route_values.get("remaining_distance_m", 0.0) or 0.0
            )
        except Exception as exc:
            return {
                "route_found": False,
                "optimal_lane_id": int(fallback_lane_id),
                "current_road_option": "",
                "next_macro_maneuver": "Continue Straight",
                "next_macro_distance_m": float("inf"),
                "remaining_distance_m": 0.0,
                "debug_reason": f"admap_route_query_failed:{exc}",
            }
        ad_target_lane_id = int(
            getattr(summary, "optimal_lane_id", 0) or 0
        )
        local_target_lane_id = int(fallback_lane_id)
        # The continuity-aware matcher is the sole current-lane owner.
        # The local graph consumes that match to describe adjacency; it
        # must not independently replace the ego lane at an intersection.
        ad_current_lane_id = int(
            self.map_matching.get("ad_lane_id", 0) or 0
        )
        self.authoritative_ego_waypoint = matched_waypoint
        local_direction = ""
        local_offset = 0
        target_in_local_frame = False
        try:
            local_graph = global_planner.get_local_lane_graph(
                float(ego_location.x),
                float(ego_location.y),
                z_m=float(getattr(ego_location, "z", 0.0)),
                forward_distance_m=100.0,
                backward_distance_m=100.0,
                ego_waypoint=matched_waypoint,
            )
            # Freeze the actual AD-map centre geometry into this frame.
            # Downstream planning must not call the map again and derive a
            # different centreline from the same lane identity.
            lane_centerlines: dict[int, list[dict[str, float]]] = {}
            local_lane_ids = {
                int(lane_id)
                for lane_ids in dict(local_graph.get("corridors", {}) or {}).values()
                for lane_id in list(lane_ids or [])
            }
            try:
                raw_route_lane_sequence = (
                    global_planner.get_local_route_lane_sequence(
                        float(ego_location.x),
                        float(ego_location.y),
                        forward_distance_m=100.0,
                        backward_distance_m=100.0,
                    )
                )
            except Exception:
                raw_route_lane_sequence = []
            # Keep stored-route order. The corridor is a set-like lookup;
            # it must never be used to infer successor topology.
            local_graph["route_lane_sequence"] = [
                int(lane_id)
                for lane_id in list(raw_route_lane_sequence or [])
                if int(lane_id) in local_lane_ids
            ]
            for lane_id in sorted(local_lane_ids):
                try:
                    centerline_waypoints = global_planner.get_lane_centerline(
                        int(lane_id)
                    )
                except Exception:
                    centerline_waypoints = []
                samples: list[dict[str, float]] = []
                for waypoint in list(centerline_waypoints or []):
                    position = dict(getattr(waypoint, "position", {}) or {})
                    try:
                        sample = {
                            "x_m": float(position["x"]),
                            "y_m": float(position["y"]),
                            "lane_width_m": max(
                                0.1,
                                float(
                                    getattr(waypoint, "lane_width_m", 3.5)
                                    or 3.5
                                ),
                            ),
                        }
                        left_boundary = getattr(
                            waypoint, "left_boundary_position", None
                        )
                        right_boundary = getattr(
                            waypoint, "right_boundary_position", None
                        )
                        if isinstance(left_boundary, Mapping) and isinstance(
                            right_boundary, Mapping
                        ):
                            sample.update({
                                "left_boundary_x_m": float(left_boundary["x"]),
                                "left_boundary_y_m": float(left_boundary["y"]),
                                "right_boundary_x_m": float(right_boundary["x"]),
                                "right_boundary_y_m": float(right_boundary["y"]),
                            })
                        samples.append(sample)
                    except (KeyError, TypeError, ValueError):
                        continue
                if len(samples) >= 2:
                    lane_centerlines[int(lane_id)] = samples
            local_graph["lane_centerlines"] = lane_centerlines
            self.local_lane_frame = dict(local_graph)
            if int(ad_current_lane_id) == 0:
                # Compatibility/degraded-mode fallback only. In normal
                # AD-map operation the continuity matcher above is valid
                # and remains authoritative.
                ad_current_lane_id = int(
                    local_graph.get("ego_ad_lane_id", 0) or 0
                )
            lane_to_offset = dict(local_graph.get("lane_to_offset", {}) or {})
            target_in_local_frame = int(ad_target_lane_id) in {
                int(lane_id) for lane_id in lane_to_offset
            }
            offset = int(lane_to_offset.get(int(ad_target_lane_id), 0))
            local_offset = int(offset)
            local_direction = "left" if offset > 0 else "right" if offset < 0 else ""
            if bool(target_in_local_frame) and int(ad_target_lane_id) != 0:
                local_target_lane_id = int(ad_target_lane_id)
        except Exception:
            # Route information remains usable even if this tick's
            # topology-to-local-lane projection cannot be resolved.
            local_target_lane_id = int(fallback_lane_id)
        violations = local_lane_frame_invariants(
            matched_lane_id=int(
                self.map_matching.get("ad_lane_id", 0) or 0
            ),
            corridors={
                int(key): list(value or [])
                for key, value in dict(
                    self.local_lane_frame.get("corridors", {}) or {}
                ).items()
            },
            target_lane_id=int(ad_target_lane_id),
            reported_target_offset=int(local_offset),
        )
        self.local_lane_frame["invariant_violations"] = list(
            violations
        )
        self.local_lane_frame["route_target_offset"] = int(
            local_offset
        )
        self.local_lane_frame["route_target_ad_lane_id"] = int(
            ad_target_lane_id
        )
        self.local_lane_frame["route_target_in_frame"] = bool(
            target_in_local_frame
        )
        self.local_map_frame_id += 1
        self.local_map_snapshot = build_local_map_snapshot(
            frame_id=int(self.local_map_frame_id),
            timestamp_s=float(clock()),
            route_revision=str(
                getattr(route_manager, "route_revision", "") or ""
            ),
            match=self.map_matching,
            local_graph=self.local_lane_frame,
            route_target_lane_id=int(ad_target_lane_id),
            invariant_violations=violations,
        )
        # Compatibility mirror only. New planning consumers must read the
        # immutable snapshot, not mutate this dictionary.
        self.local_lane_frame = (
            self.local_map_snapshot.as_legacy_dict()
        )
        ad_current_lane_id = int(self.local_map_snapshot.ego_lane_id)
        target_in_local_frame = bool(
            self.local_map_snapshot.route_target_in_frame
        )
        local_offset = int(self.local_map_snapshot.route_target_offset)
        local_direction = (
            "left" if local_offset > 0 else "right" if local_offset < 0 else ""
        )
        if target_in_local_frame and int(ad_target_lane_id) != 0:
            local_target_lane_id = int(ad_target_lane_id)
        next_macro_maneuver = str(
            getattr(summary, "next_macro_maneuver", "Continue Straight")
        )
        normalized_macro = (
            next_macro_maneuver.strip().lower().replace("-", "_").replace(" ", "_")
        )
        if (
            normalized_macro in {"lane_change_left", "lane_change_right"}
            and int(ad_current_lane_id) != 0
            and int(ad_current_lane_id) == int(ad_target_lane_id)
            and int(local_offset) == 0
        ):
            # The route backend can keep reporting the consumed edge for
            # a few progress samples. Expose completion immediately so
            # behavior and ManeuverManager do not restart/retain it.
            next_macro_maneuver = "Lane Follow"
        route_result = {
            "route_found": bool(getattr(summary, "route_found", False)),
            "optimal_lane_id": int(local_target_lane_id),
            "authoritative_current_lane_id": int(ad_current_lane_id),
            "ad_current_lane_id": int(ad_current_lane_id),
            "ad_target_lane_id": int(ad_target_lane_id),
            "lane_change_direction": str(local_direction),
            "lane_change_offset": int(local_offset),
            "target_in_local_frame": bool(target_in_local_frame),
            "diagnostic_map_matching": dict(self.map_matching),
            "diagnostic_local_lane_frame": dict(
                self.local_lane_frame
            ),
            "current_road_option": str(getattr(summary, "current_road_option", "")),
            "next_macro_maneuver": str(next_macro_maneuver),
            "next_macro_distance_m": float(
                getattr(summary, "next_macro_distance_m", float("inf"))
            ),
            "remaining_distance_m": float(
                getattr(summary, "distance_to_destination_m", 0.0) or 0.0
            ),
            "debug_reason": "admap_topology_geometry_active",
        }
        # AD-map owns route identity, maneuver semantics, progress, and
        # destination completion in this backend.  Publish the exact
        # per-tick query consumed by behavior so RouteManagerStatus/CSV
        # cannot remain frozen at the last AD-map route rebuild.  CARLA's
        # independently synchronized index is geometry-only.
        publish = getattr(
            route_manager,
            "accept_authoritative_route_summary",
            None,
        )
        if callable(publish):
            publish(
                summary,
                debug_reason="admap_authoritative_route_active",
            )
        return route_result
