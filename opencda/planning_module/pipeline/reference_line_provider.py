"""Single lifecycle owner for all planner reference-line modes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from .behavior_decision import BehaviorDecision

from .stable_reference_line_provider import (
    StableReferenceLineProvider,
    StableReferenceWindow,
)


LANE_FOLLOW = "lane_follow"
LANE_CHANGE = "lane_change"
CONNECTOR = "connector"
TURN = "turn"
POST_TURN = "post_turn"

_MODES = {LANE_FOLLOW, LANE_CHANGE, CONNECTOR, TURN, POST_TURN}
_INSTALL_EVENTS = {
    "initial_route",
    "route_changed",
    "map_epoch_changed",
    "maneuver_started",
    "phase_transition",
    "same_route_extension",
}
_RELEASE_EVENTS = {
    "route_changed",
    "map_epoch_changed",
    "maneuver_completed",
    "maneuver_abandoned",
    "phase_transition",
    "reset",
}


@dataclass(frozen=True)
class ReferenceLineSnapshot:
    mode: str
    active: bool = False
    route_revision: str = ""
    map_epoch: str = ""
    geometry_revision: int = 0
    samples: tuple = ()
    activation_s_m: float = 0.0
    progress_s_m: float = 0.0
    source_lane_id: int = 0
    target_lane_id: int = 0
    maneuver_direction: str = ""
    install_event: str = ""
    build_reason: str = ""
    last_validation_failure: str = ""

    @property
    def travelled_s_m(self) -> float:
        return max(0.0, float(self.progress_s_m) - float(self.activation_s_m))

    def mutable_samples(self):
        return [dict(sample) for sample in self.samples]

    def trace_fields(self):
        return {
            "persistent_reference_active": bool(self.active),
            "persistent_reference_state": (
                "DEGRADED" if self.last_validation_failure
                else "ACTIVE" if self.active else "UNINITIALIZED"
            ),
            "persistent_reference_route_revision": str(self.route_revision),
            "persistent_reference_map_epoch": str(self.map_epoch),
            "persistent_reference_geometry_revision": int(self.geometry_revision),
            "persistent_reference_progress_s_m": float(self.progress_s_m),
            "persistent_reference_build_reason": str(self.build_reason),
            "persistent_reference_last_rejected_trigger": (
                "validation_failure:" + str(self.last_validation_failure)
                if self.last_validation_failure else ""
            ),
        }


@dataclass(frozen=True)
class ReferenceLineRequest:
    """Complete typed input contract for one reference publication."""

    local_map: Any
    route_cursor: Any
    behavior: BehaviorDecision
    route_revision: str
    map_epoch: str
    ego_x_m: float
    ego_y_m: float


@dataclass(frozen=True)
class ReferenceLineResult:
    mode: str
    samples: tuple
    accepted: bool
    reason: str
    geometry_revision: int

    def mutable_samples(self):
        return [dict(sample) for sample in self.samples]


@dataclass(frozen=True)
class LaneChangeWindowResult:
    samples: tuple
    projection_s_m: float
    start_s_m: float
    master_index: int
    reason: str

    def mutable_samples(self):
        return [dict(sample) for sample in self.samples]


class ReferenceLineProvider(StableReferenceLineProvider):
    """Own immutable masters and monotonic windows for every planning mode.

    Geometry can only be installed/released by explicit route, map, maneuver,
    or phase events. Validation/MPC failure is diagnostic and never rebuilds
    a master.
    """

    def __init__(self) -> None:
        self._builder = None
        self._snapshots = {
            mode: ReferenceLineSnapshot(mode=mode) for mode in _MODES
        }

    def attach_builder(self, builder: Any) -> None:
        self._builder = builder

    @property
    def builder(self) -> Any:
        if self._builder is None:
            raise RuntimeError("reference builder is not attached")
        return self._builder

    def snapshot(self, mode: str) -> ReferenceLineSnapshot:
        return self._snapshots[self._normalize_mode(mode)]

    def publish(
        self,
        request: ReferenceLineRequest,
        samples: Sequence[Mapping[str, object]],
        *,
        valid: bool,
        validation_reason: str = "",
        build_reason: str = "planning_reference",
    ) -> ReferenceLineResult:
        """Sole acceptance boundary for every reference delivered to MPC.

        Builders submit geometry; they cannot mutate lifecycle state or
        decide whether rejected geometry replaces a persistent master.
        """

        mode = self.mode_for(
            request.behavior,
            segment_kind=str(
                getattr(request.route_cursor, "segment_kind", "") or ""
            ),
        )
        rows = [dict(sample) for sample in list(samples or [])]
        current = self.snapshot(mode)
        if not bool(valid) or len(rows) < 2:
            self.mark_validation_failure(
                mode, str(validation_reason or "reference_invalid")
            )
            retained = current.samples if current.active else tuple(
                MappingProxyType(dict(sample)) for sample in rows
            )
            return ReferenceLineResult(
                mode=mode,
                samples=retained,
                accepted=False,
                reason=str(validation_reason or "reference_invalid"),
                geometry_revision=int(current.geometry_revision),
            )

        route_changed = bool(
            current.active
            and str(current.route_revision) != str(request.route_revision)
        )
        map_changed = bool(
            current.active and str(current.map_epoch) != str(request.map_epoch)
        )
        if not current.active:
            event = (
                "maneuver_started"
                if mode in {LANE_CHANGE, CONNECTOR, TURN, POST_TURN}
                else "initial_route"
            )
        elif route_changed:
            event = "route_changed"
        elif map_changed:
            event = "map_epoch_changed"
        else:
            # A planning tick is not a lifecycle event.  In particular, the
            # rolling local window must never increment geometry revision or
            # silently replace the persistent master.  The submitted window
            # is still returned to MPC; only route/map/maneuver transitions
            # are allowed to install a new master.
            return ReferenceLineResult(
                mode=mode,
                samples=tuple(
                    MappingProxyType(dict(sample)) for sample in rows
                ),
                accepted=True,
                reason="persistent_reference_retained",
                geometry_revision=int(current.geometry_revision),
            )
        installed, reason = self.install(
            mode,
            rows,
            route_revision=str(request.route_revision),
            map_epoch=str(request.map_epoch),
            event=event,
            source_lane_id=int(request.behavior.source_lane_id),
            target_lane_id=int(request.behavior.target_lane_id),
            maneuver_direction=str(request.behavior.direction),
            build_reason=str(build_reason),
            ego_x_m=float(request.ego_x_m),
            ego_y_m=float(request.ego_y_m),
        )
        snapshot = self.snapshot(mode)
        return ReferenceLineResult(
            mode=mode,
            samples=tuple(
                MappingProxyType(dict(sample)) for sample in rows
            ),
            accepted=bool(installed or snapshot.active),
            reason=str(reason),
            geometry_revision=int(snapshot.geometry_revision),
        )

    def turn_reference(
            self, *, local_map, config, horizon_steps, dt_s,
            ego_location, ego_yaw_rad, current_state, current_lane_id,
            target_lane_id, target_speed_mps, destination_state=None,
            lock_master=False, turn_direction="", route_revision="",
            map_epoch="admap"):
        """Produce and window the sole immutable AD-map turn reference."""
        turn_speed_mps = min(
            max(0.4, float(target_speed_mps)),
            float(config.get("waypoint_turn_speed_cap_mps", 2.2)),
        )
        step_distance_m = max(
            float(config.get("waypoint_turn_min_step_m", 0.35)),
            float(dt_s) * max(0.8, float(turn_speed_mps)),
        )
        reason = ""
        master = self.snapshot(TURN).mutable_samples()
        if not master:
            # GlobalRoute owns topology only. TURN geometry comes from the
            # topology-ordered AD-map lane and connector centre lines in the
            # local snapshot. The former route-polyline path was a second
            # geometry owner and exposed junction node discontinuities as a
            # lateral jump at PREPARE->TURN.
            master, reason = self.reference_from_local_map(
                local_map,
                start_lane_id=int(current_lane_id),
                target_speed_mps=float(turn_speed_mps),
                maximum_join_distance_m=float(
                    config.get("local_map_maximum_join_distance_m", 5.0)
                ),
            )
            if bool(lock_master) and master:
                installed, install_reason = self.install(
                    TURN, master,
                    route_revision=str(route_revision),
                    map_epoch=str(map_epoch or "admap"),
                    event="maneuver_started",
                    source_lane_id=int(current_lane_id),
                    target_lane_id=int(target_lane_id),
                    maneuver_direction=str(turn_direction or "").strip().lower(),
                    build_reason=str(reason),
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                )
                suffix = "turn_master_locked" if installed else str(install_reason)
                reason = ";".join(
                    item for item in (str(reason), suffix) if item
                )
        if not master:
            return [], list(destination_state or []), str(reason)
        if self.snapshot(TURN).active:
            window = self.window(
                TURN,
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
                first_forward_m=float(config.get(
                    "reference_contract_intersection_turn_min_first_forward_m",
                    0.2,
                )),
                spacing_m=float(step_distance_m),
                count=int(horizon_steps),
                max_projection_advance_m=max(2.0, 2.0 * float(step_distance_m)),
            )
            reference = [dict(sample) for sample in window.samples]
            reason = ";".join(
                item for item in (
                    str(reason), "turn_master_window:" + str(window.reason)
                ) if item
            )
        else:
            reference = [dict(sample) for sample in master]
        for sample in reference:
            sample["v_ref_mps"] = float(turn_speed_mps)
            sample["speed_ref_mps"] = float(turn_speed_mps)
            sample["speed_mps"] = float(turn_speed_mps)
        reference, curvature_reason = self.builder.curvature_feasible_turn_samples(
            reference_samples=reference,
            ego_location=ego_location,
            ego_heading_rad=float(ego_yaw_rad),
            max_curvature_1pm=float(
                config.get("reference_vehicle_max_curvature_1pm", 0.20)
            ),
        )
        if curvature_reason:
            reason = ";".join(
                item for item in (str(reason), str(curvature_reason)) if item
            )
        from opencda.planning_module.behavior_planner.reference_pipeline import (
            lane_center_destination_from_reference_arc_length,
        )
        seed = list(destination_state or [])
        if len(seed) < 5:
            seed = [
                float(current_state[0]), float(current_state[1]),
                float(turn_speed_mps), float(current_state[3]),
                int(target_lane_id or current_lane_id),
            ]
        seed[2] = float(turn_speed_mps)
        destination = lane_center_destination_from_reference_arc_length(
            destination_state=seed,
            lane_center_reference=reference,
            target_arc_length_m=float(
                config.get("waypoint_turn_destination_arc_m", 4.5)
            ),
        ) or seed
        return [dict(sample) for sample in reference], list(destination), str(reason)

    def lane_fallback_reference(
        self,
        *,
        ego_location,
        ego_yaw_rad,
        current_state,
        speed_ref_mps,
    ):
        """Build the provider-owned lane fallback used after pipeline errors.

        This geometry deliberately remains a transient fallback: an exception
        is not a lifecycle event and therefore must not replace a persistent
        lane-follow master.
        """

        return self.builder.build_lane_fallback(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            current_state=current_state,
            speed_ref_mps=float(speed_ref_mps),
        )

    def boundary_recovery_reference(
        self,
        *,
        ego_location,
        ego_yaw_rad,
        current_lane_id,
        base_reference_samples,
        target_speed_mps,
        horizon_steps,
        dt_s,
        max_curvature_1pm,
    ):
        """Build and condition the sole transient boundary-recovery line."""

        generated = self.builder.build_boundary_recovery(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            current_lane_id=int(current_lane_id),
            base_reference_samples=base_reference_samples,
            target_speed_mps=float(target_speed_mps),
            horizon_steps=int(horizon_steps),
            dt_s=float(dt_s),
        )
        samples = [dict(sample) for sample in list(generated.samples or [])]
        conditioning_reason = ""
        if samples:
            samples, conditioning_reason = self.builder.curvature_feasible_samples(
                reference_samples=samples,
                ego_location=ego_location,
                ego_heading_rad=float(ego_yaw_rad),
                max_curvature_1pm=float(max_curvature_1pm),
                mode="boundary_recovery",
            )
        for sample in samples:
            sample["speed_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["v_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["speed_mps"] = max(0.0, float(target_speed_mps))
            sample["reference_mode"] = "boundary_recovery"
        return generated, samples, str(conditioning_reason)

    @staticmethod
    def mode_for(behavior: BehaviorDecision, *, segment_kind: str = "") -> str:
        maneuver = str(behavior.maneuver or "").strip().lower()
        phase = str(behavior.phase or "").strip().upper()
        if maneuver in {"lane_change_left", "lane_change_right"} or (
            "LANE_CHANGE" in phase
        ):
            return LANE_CHANGE
        if maneuver in {"intersection_turn_left", "intersection_turn_right"}:
            return TURN
        if "POST_TURN" in phase or "TURN_EXIT" in phase:
            return POST_TURN
        if str(segment_kind) == "junction_connector":
            return CONNECTOR
        return LANE_FOLLOW

    def install(
        self,
        mode: str,
        samples: Sequence[Mapping[str, object]],
        *,
        route_revision: str,
        map_epoch: str,
        event: str,
        source_lane_id: int = 0,
        target_lane_id: int = 0,
        maneuver_direction: str = "",
        build_reason: str = "",
        ego_x_m: Optional[float] = None,
        ego_y_m: Optional[float] = None,
    ) -> tuple[bool, str]:
        normalized_mode = self._normalize_mode(mode)
        normalized_event = str(event or "")
        if normalized_event not in _INSTALL_EVENTS:
            return False, "reference_install_event_forbidden:" + normalized_event
        rows = [dict(sample) for sample in list(samples or [])]
        if len(rows) < 2:
            return False, "reference_too_short"
        current = self.snapshot(normalized_mode)
        if current.active and normalized_event == "initial_route":
            return False, "active_reference_cannot_reinitialize"
        if current.active and normalized_event == "route_changed" and (
            str(route_revision) == str(current.route_revision)
        ):
            return False, "route_revision_unchanged"
        if current.active and normalized_event == "map_epoch_changed" and (
            str(map_epoch) == str(current.map_epoch)
        ):
            return False, "map_epoch_unchanged"
        if current.active and normalized_event == "same_route_extension" and (
            str(route_revision) != str(current.route_revision)
            or str(map_epoch) != str(current.map_epoch)
        ):
            return False, "extension_owner_mismatch"
        activation_s_m = 0.0
        if ego_x_m is not None and ego_y_m is not None:
            activation = self.window_from_reference(
                rows,
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                lower_s_m=0.0,
                first_forward_m=0.0,
                spacing_m=0.5,
                count=1,
            )
            activation_s_m = float(activation.projection_s_m)
        immutable = tuple(MappingProxyType(dict(sample)) for sample in rows)
        self._snapshots[normalized_mode] = ReferenceLineSnapshot(
            mode=normalized_mode,
            active=True,
            route_revision=str(route_revision or ""),
            map_epoch=str(map_epoch or ""),
            geometry_revision=int(current.geometry_revision) + 1,
            samples=immutable,
            activation_s_m=float(activation_s_m),
            progress_s_m=float(activation_s_m),
            source_lane_id=int(source_lane_id),
            target_lane_id=int(target_lane_id),
            maneuver_direction=str(maneuver_direction or ""),
            install_event=normalized_event,
            build_reason=str(build_reason or normalized_event),
        )
        return True, "reference_installed:%s:%s" % (
            normalized_mode, normalized_event
        )

    def window(
        self,
        mode: str,
        *,
        ego_x_m: float,
        ego_y_m: float,
        first_forward_m: float,
        spacing_m: float,
        count: int,
        max_projection_advance_m: float = float("inf"),
        ego_heading_rad: Optional[float] = None,
        min_body_forward_m: Optional[float] = None,
    ) -> StableReferenceWindow:
        normalized_mode = self._normalize_mode(mode)
        current = self.snapshot(normalized_mode)
        if not current.active:
            return StableReferenceWindow((), 0.0, 0.0, "reference_inactive")
        result = self.window_from_reference(
            current.mutable_samples(),
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            lower_s_m=float(current.progress_s_m),
            first_forward_m=float(first_forward_m),
            spacing_m=float(spacing_m),
            count=int(count),
            max_projection_advance_m=float(max_projection_advance_m),
        )
        # Arc-length lookahead is not identical to longitudinal lookahead in
        # the ego body frame.  This matters near the end of a lane change,
        # where the vehicle heading and the immutable master can differ by a
        # few degrees: a nominal 0.25 m arc offset was observed as only
        # 0.15 m forward and intermittently violated the reference contract.
        # Drop only leading samples that are behind the requested body-frame
        # anchor; do not rebuild or otherwise mutate the persistent master.
        if ego_heading_rad is not None and min_body_forward_m is not None:
            cosine = math.cos(float(ego_heading_rad))
            sine = math.sin(float(ego_heading_rad))
            rows = [dict(sample) for sample in result.samples]
            first_kept = 0
            for index, sample in enumerate(rows):
                dx_m = float(sample.get("x_ref_m", sample.get("x", 0.0))) - float(ego_x_m)
                dy_m = float(sample.get("y_ref_m", sample.get("y", 0.0))) - float(ego_y_m)
                if dx_m * cosine + dy_m * sine >= float(min_body_forward_m):
                    first_kept = index
                    break
            else:
                first_kept = max(0, len(rows) - 1)
            if first_kept:
                rows = rows[first_kept:]
                result = StableReferenceWindow(
                    tuple(rows),
                    float(result.projection_s_m),
                    float(rows[0].get("s_ref_m", result.start_s_m)),
                    str(result.reason) + ":body_forward_anchored",
                )
        self._snapshots[normalized_mode] = ReferenceLineSnapshot(
            mode=current.mode,
            active=current.active,
            route_revision=current.route_revision,
            map_epoch=current.map_epoch,
            geometry_revision=current.geometry_revision,
            samples=current.samples,
            activation_s_m=current.activation_s_m,
            progress_s_m=max(
                float(current.progress_s_m), float(result.projection_s_m)
            ),
            source_lane_id=current.source_lane_id,
            target_lane_id=current.target_lane_id,
            maneuver_direction=current.maneuver_direction,
            install_event=current.install_event,
            build_reason=current.build_reason,
            last_validation_failure=current.last_validation_failure,
        )
        return result

    def locked_lane_change_window(
        self, *, ego_x_m, ego_y_m, ego_heading_rad, target_lane_id,
        target_speed_mps, spacing_m, count, min_first_forward_m,
        anchor_margin_m=0.05, max_projection_advance_m=float("inf")
    ) -> LaneChangeWindowResult:
        """Produce a complete MPC window from the immutable LC master."""

        master = self.snapshot(LANE_CHANGE).mutable_samples()
        if not master:
            return LaneChangeWindowResult(
                (), 0.0, 0.0, 0,
                "lane_change_window_missing_locked_reference",
            )
        spacing_m = max(0.05, float(spacing_m))
        window = self.window(
            LANE_CHANGE,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            first_forward_m=float(min_first_forward_m) + max(
                0.02, float(anchor_margin_m)
            ),
            spacing_m=float(spacing_m),
            count=int(count),
            max_projection_advance_m=float(max_projection_advance_m),
            ego_heading_rad=float(ego_heading_rad),
            min_body_forward_m=float(min_first_forward_m),
        )
        master_index = min(
            range(len(master)),
            key=lambda index: abs(
                float(master[index].get("s_ref_m", index * spacing_m))
                - float(window.start_s_m)
            ),
        )
        samples = window.mutable_samples() if hasattr(window, "mutable_samples") else [
            dict(sample) for sample in window.samples
        ]
        while samples and len(samples) < int(count):
            previous = dict(samples[-1])
            heading_rad = float(previous.get("heading_rad", ego_heading_rad))
            x_m = float(previous.get("x_ref_m", previous.get("x", ego_x_m)))
            y_m = float(previous.get("y_ref_m", previous.get("y", ego_y_m)))
            padded = dict(previous)
            padded["x_ref_m"] = x_m + spacing_m * math.cos(heading_rad)
            padded["y_ref_m"] = y_m + spacing_m * math.sin(heading_rad)
            padded["x"] = float(padded["x_ref_m"])
            padded["y"] = float(padded["y_ref_m"])
            padded["lane_id"] = int(target_lane_id)
            padded["lane_change_progress"] = 1.0
            samples.append(padded)
        for sample in samples:
            sample["speed_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["v_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["speed_mps"] = max(0.0, float(target_speed_mps))
        return LaneChangeWindowResult(
            tuple(samples), float(window.projection_s_m),
            float(window.start_s_m), int(master_index), str(window.reason)
        )

    def release(self, mode: str, *, event: str) -> tuple[bool, str]:
        normalized_mode = self._normalize_mode(mode)
        normalized_event = str(event or "")
        if normalized_event not in _RELEASE_EVENTS:
            return False, "reference_release_event_forbidden:" + normalized_event
        current = self.snapshot(normalized_mode)
        if not current.active:
            return False, "reference_already_inactive"
        self._snapshots[normalized_mode] = ReferenceLineSnapshot(
            mode=normalized_mode,
            geometry_revision=current.geometry_revision,
            build_reason="released:" + normalized_event,
        )
        return True, "reference_released:%s:%s" % (
            normalized_mode, normalized_event
        )

    def mark_validation_failure(self, mode: str, reason: str) -> None:
        normalized_mode = self._normalize_mode(mode)
        current = self.snapshot(normalized_mode)
        if not current.active:
            return
        self._snapshots[normalized_mode] = ReferenceLineSnapshot(
            mode=current.mode,
            active=current.active,
            route_revision=current.route_revision,
            map_epoch=current.map_epoch,
            geometry_revision=current.geometry_revision,
            samples=current.samples,
            activation_s_m=current.activation_s_m,
            progress_s_m=current.progress_s_m,
            source_lane_id=current.source_lane_id,
            target_lane_id=current.target_lane_id,
            maneuver_direction=current.maneuver_direction,
            install_event=current.install_event,
            build_reason=current.build_reason,
            last_validation_failure=str(reason),
        )

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized not in _MODES:
            raise ValueError("unsupported reference mode: " + normalized)
        return normalized
