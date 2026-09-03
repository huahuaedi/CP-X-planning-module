"""Semantic lifecycle owner for committed planning maneuvers.

XY reference geometry belongs exclusively to ``ReferenceLineProvider``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

from .stage_contracts import (
    LaneChangeContract,
    evaluate_lane_change_alignment,
)


@dataclass
class LaneChangeLifecycle:
    option: str = ""
    phase: str = "idle"
    source_lane_id: int = 0
    target_lane_id: int = 0
    target_speed_mps: float = 0.0
    progress: float = 0.0
    progress_index: int = 0
    progress_s_m: float = 0.0
    stabilization_frames: int = 0
    completion_stable_frames: int = 0
    geometry_completion_latched: bool = False
    completion_reference: list = field(default_factory=list)
    completion_debug: dict = field(default_factory=dict)
    completed_option: str = ""
    progress_pairs: list = field(default_factory=list)
    envelope_blocks: object = None
    envelope_epsilon0: float = 0.0
    commitment_invalid_frames: int = 0
    duration_comfort_reason: str = ""
    resolved_duration_s: float = 0.0
    committed_at_s: float = -float("inf")
    transition_to_turn_arc_m: float = 0.0
    transition_to_turn_step_m: float = 0.0
    required_target_lane_id: Optional[int] = None
    required_target_ad_lane_id: Optional[int] = None

    @property
    def active(self):
        return bool(self.option and self.phase != "idle")

    def reset(self, preserve_completed_option=True):
        completed = str(self.completed_option) if preserve_completed_option else ""
        self.option, self.phase = "", "idle"
        self.source_lane_id = self.target_lane_id = 0
        self.target_speed_mps = self.progress = self.progress_s_m = 0.0
        self.progress_index = self.stabilization_frames = 0
        self.completion_stable_frames = self.commitment_invalid_frames = 0
        self.geometry_completion_latched = False
        self.completion_reference, self.completion_debug = [], {}
        self.completed_option = completed
        self.progress_pairs, self.envelope_blocks = [], None
        self.envelope_epsilon0 = self.resolved_duration_s = 0.0
        self.duration_comfort_reason = ""
        self.committed_at_s = -float("inf")
        self.transition_to_turn_arc_m = self.transition_to_turn_step_m = 0.0
        self.required_target_lane_id = self.required_target_ad_lane_id = None


@dataclass
class TurnLifecycle:
    decision: str = ""
    phase: str = "idle"

    @property
    def active(self):
        return self.phase != "idle"

    def reset(self):
        self.decision, self.phase = "", "idle"


@dataclass(frozen=True)
class LaneChangeTransition:
    action: str
    reason: str = ""


class ManeuverManager:
    """Own identity, phase and monotonic progress, never reference samples."""

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.lane_change = LaneChangeLifecycle()
        self.turn = TurnLifecycle()
        self.last_release = {}
        self._route_lane_change_edge_id = ""
        self._completed_route_lane_change_edge_id = ""

    def reset(self, reason="reset"):
        self.last_release = {"reason": str(reason), "outcome": "reset"}
        self.lane_change.reset(preserve_completed_option=False)
        self.turn.reset()
        self._route_lane_change_edge_id = ""
        self._completed_route_lane_change_edge_id = ""

    def observe_route_lane_change_edge(self, edge_id):
        """Track one immutable route edge and retire completion on advance."""

        edge_id = str(edge_id or "")
        if (
            self._completed_route_lane_change_edge_id
            and edge_id != self._completed_route_lane_change_edge_id
        ):
            self._completed_route_lane_change_edge_id = ""
            self.lane_change.completed_option = ""
        self._route_lane_change_edge_id = edge_id
        return edge_id

    @property
    def route_lane_change_edge_completed(self):
        return bool(
            self._route_lane_change_edge_id
            and self._route_lane_change_edge_id
            == self._completed_route_lane_change_edge_id
        )

    @property
    def route_lane_change_edge_id(self):
        return str(self._route_lane_change_edge_id)

    @property
    def completed_route_lane_change_edge_id(self):
        return str(self._completed_route_lane_change_edge_id)

    def clear_turn(self, reason="turn_released"):
        if self.turn.active:
            self.last_release = {"reason": str(reason), "outcome": "turn_complete",
                                 "decision": str(self.turn.decision)}
        self.turn.reset()

    def resolve_post_turn_phase(self, decision, scenario_state,
                                turn_reference_active, post_turn_reference_active,
                                travelled_s_m, required_s_m, exit_aligned):
        lane_follow = str(decision).strip().lower() == "lane_follow"
        if (lane_follow and str(scenario_state).strip().upper() == "LANE_FOLLOW"
                and not post_turn_reference_active and turn_reference_active):
            self.turn.phase = "post_turn"
            return "activate"
        if not (lane_follow and post_turn_reference_active):
            return "inactive"
        self.turn.phase = "post_turn"
        if float(travelled_s_m) + 1e-3 >= float(required_s_m) and exit_aligned:
            self.clear_turn(reason="post_turn_complete")
            return "complete"
        return "hold"

    def begin_lane_change(self, option, phase, source_lane_id, target_lane_id,
                          target_speed_mps, completion_reference,
                          committed_at_s=None):
        state = self.lane_change
        state.reset(preserve_completed_option=True)
        state.option, state.phase = str(option), str(phase or "executing")
        state.source_lane_id, state.target_lane_id = int(source_lane_id), int(target_lane_id)
        state.target_speed_mps = max(0.0, float(target_speed_mps))
        # Diagnostic completion snapshot only; never returned as control geometry.
        state.completion_reference = [dict(x) for x in completion_reference or []]
        if committed_at_s is not None:
            state.committed_at_s = float(committed_at_s)
        return state

    def remember_required_lane_change(self, target_lane_id, target_ad_lane_id=None):
        self.lane_change.required_target_lane_id = int(target_lane_id)
        self.lane_change.required_target_ad_lane_id = (
            int(target_ad_lane_id) if target_ad_lane_id is not None else None
        )

    def clear_required_lane_change(self):
        self.lane_change.required_target_lane_id = None
        self.lane_change.required_target_ad_lane_id = None

    def lane_change_start_feasibility(
        self, *, authorization_allowed, distance_to_turn_m,
        geometry_arc_m, handoff_arc_m
    ):
        """Decide whether a new lane change can finish before a turn."""

        if not bool(authorization_allowed):
            return LaneChangeTransition("hold", "lane_change_not_authorized")
        if self.lane_change.active:
            return LaneChangeTransition("allow", "lane_change_already_active")
        try:
            turn_distance_m = float(distance_to_turn_m)
        except (TypeError, ValueError):
            turn_distance_m = float("inf")
        if not math.isfinite(turn_distance_m):
            return LaneChangeTransition("allow", "no_turn_in_route_horizon")
        required_arc_m = max(0.0, float(geometry_arc_m)) + max(
            0.0, float(handoff_arc_m)
        )
        if float(turn_distance_m) + 1.0e-3 < float(required_arc_m):
            return LaneChangeTransition(
                "deny",
                "required_lane_change_no_longer_feasible:"
                f"turn_distance={float(turn_distance_m):.2f}:"
                f"required_arc={float(required_arc_m):.2f}",
            )
        return LaneChangeTransition("allow", "lane_change_fits_before_turn")

    def transfer_lateral_ownership_to_turn(self, *, owner_state):
        """Release a lane-change commitment when turn geometry takes over."""

        state = str(owner_state or "").strip().upper()
        if state not in {
            "PREPARE_TURN", "INTERSECTION_TURN", "TURN_EXIT_STABILIZATION",
            "CREEP", "BOUNDARY_RECOVERY",
        }:
            return LaneChangeTransition("hold", "turn_does_not_own_lateral")
        if not self.lane_change.active:
            return LaneChangeTransition("hold", "lane_change_not_active")
        self.abandon_lane_change(
            f"lateral_ownership_transferred_to_{state.lower()}",
            suppress_recommit=True,
        )
        return LaneChangeTransition(
            "release", f"lateral_ownership_transferred_to_{state.lower()}"
        )

    def set_lane_change_transition_arc(self, *, arc_m, step_m):
        self.lane_change.transition_to_turn_arc_m = max(0.0, float(arc_m))
        self.lane_change.transition_to_turn_step_m = max(0.0, float(step_m))

    def advance_lane_change(self, progress=None, progress_index=None,
                            progress_s_m=None, phase=None):
        state = self.lane_change
        if progress is not None:
            state.progress = max(state.progress, min(1.0, float(progress)))
        if progress_index is not None:
            state.progress_index = max(state.progress_index, int(progress_index))
        if progress_s_m is not None:
            state.progress_s_m = max(state.progress_s_m, float(progress_s_m))
        if phase is not None:
            state.phase = str(phase)
        return state

    def finish_lane_change_lifecycle(self, completed):
        state, released = self.lane_change, str(self.lane_change.option)
        if completed:
            state.completed_option = released
            if self._route_lane_change_edge_id:
                self._completed_route_lane_change_edge_id = str(
                    self._route_lane_change_edge_id
                )
        state.reset(preserve_completed_option=True)
        return released

    def begin_lane_change_stabilization(self):
        state = self.lane_change
        state.phase = "target_lane_stabilization"
        state.progress_index = 0
        state.progress_s_m = 0.0
        state.stabilization_frames = state.completion_stable_frames = 0
        state.geometry_completion_latched = False
        return state

    def lane_change_handoff_transition(self, *, geometry_ready):
        state = self.lane_change
        if state.phase == "target_lane_stabilization":
            return LaneChangeTransition("hold", "already_stabilizing")
        if not bool(geometry_ready):
            return LaneChangeTransition("hold", "handoff_geometry_not_ready")
        return LaneChangeTransition("start_stabilization", "handoff_ready")

    def tick_lane_change_stabilization(self, *, timeout_frames):
        state = self.lane_change
        if state.phase != "target_lane_stabilization":
            return LaneChangeTransition("hold", "not_stabilizing")
        state.stabilization_frames += 1
        if state.stabilization_frames <= max(1, int(timeout_frames)):
            return LaneChangeTransition("hold", "stabilizing")
        self.abandon_lane_change(
            "stabilization_timeout", suppress_recommit=True
        )
        return LaneChangeTransition("abandon", "stabilization_timeout")

    def accept_lane_change_completion(
        self, *, stable_frames, debug, geometrically_complete,
        completion_reason, transition_progress_m, transition_arc_m
    ):
        self.record_lane_change_completion_evidence(
            stable_frames=int(stable_frames),
            debug=debug,
            geometrically_complete=bool(geometrically_complete),
        )
        if not self.lane_change.geometry_completion_latched:
            return LaneChangeTransition("hold", "completion_not_converged")
        if (
            float(transition_arc_m) > 0.0
            and float(transition_progress_m) + 1.0e-3
            < float(transition_arc_m)
        ):
            return LaneChangeTransition("hold", "transition_arc_incomplete")
        return LaneChangeTransition("complete", str(completion_reason))

    def evaluate_lane_change_completion(
        self,
        *,
        alignment,
        progress,
        target_lane_matches,
        footprint_clearance_m,
        contract,
    ):
        """Turn one geometric observation into the lifecycle contract result."""

        if not isinstance(contract, LaneChangeContract):
            contract = LaneChangeContract.from_config(self.config)
        return evaluate_lane_change_alignment(
            available=bool(getattr(alignment, "available", False)),
            lateral_error_m=float(alignment.lateral_error_m),
            heading_error_rad=float(alignment.heading_error_rad),
            progress=float(progress),
            previous_stable_frames=int(
                self.lane_change.completion_stable_frames
            ),
            target_lane_matches=bool(target_lane_matches),
            footprint_clearance_m=float(footprint_clearance_m),
            min_footprint_clearance_m=0.0,
            contract=contract,
        )

    def accept_evaluated_lane_change_completion(
        self,
        *,
        completion,
        contract,
        stabilization_geometry_ready,
        stabilization_lateral_error_m,
        stabilization_heading_error_rad,
    ):
        """Record diagnostics and decide release from one evaluated result."""

        transition_arc_m = max(
            0.0, float(self.lane_change.transition_to_turn_arc_m)
        )
        transition_progress_m = float(self.lane_change.progress_s_m)
        debug = {
            **contract.as_debug_fields(),
            "lane_change_stabilization_entry_lateral_error_m": float(
                stabilization_lateral_error_m
            ),
            "lane_change_stabilization_entry_heading_error_deg": math.degrees(
                float(stabilization_heading_error_rad)
            ),
            "lane_change_stabilization_geometry_ready": bool(
                stabilization_geometry_ready
            ),
            "lane_change_completion_reason": str(completion.reason),
            "lane_change_completion_stable_frames": int(
                completion.stable_frames
            ),
            "lane_change_completion_lateral_error_m": float(
                completion.target_lateral_error_m
            ),
            "lane_change_completion_heading_error_deg": math.degrees(
                float(completion.target_heading_error_rad)
            ),
            "lane_change_completion_target_lane_matches": bool(
                completion.target_lane_matches
            ),
            "lane_change_completion_footprint_clearance_m": float(
                completion.footprint_clearance_m
            ),
            "lane_change_to_turn_transition_arc_m": float(transition_arc_m),
            "lane_change_to_turn_transition_progress_m": float(
                transition_progress_m
            ),
        }
        transition = self.accept_lane_change_completion(
            stable_frames=int(completion.stable_frames),
            debug=debug,
            geometrically_complete=bool(completion.complete),
            completion_reason=str(completion.reason),
            transition_progress_m=float(transition_progress_m),
            transition_arc_m=float(transition_arc_m),
        )
        self.lane_change.completion_debug[
            "lane_change_geometry_completion_latched"
        ] = bool(self.lane_change.geometry_completion_latched)
        return transition

    def clear_completed_lane_change_if_route_advanced(self, route_option):
        if (self.lane_change.completed_option
                and str(route_option) != self.lane_change.completed_option):
            self.lane_change.completed_option = ""
            return True
        return False

    def record_lane_change_completion_evidence(self, stable_frames, debug,
                                               geometrically_complete=False):
        self.lane_change.completion_stable_frames = max(0, int(stable_frames))
        # Geometric completion is a one-way lifecycle event.  The following
        # fixed-arc handoff may temporarily move outside the tight completion
        # tolerance; that must not make an already completed crossing become
        # incomplete again.
        self.lane_change.geometry_completion_latched = bool(
            self.lane_change.geometry_completion_latched
            or geometrically_complete
        )
        self.lane_change.completion_debug = dict(debug or {})
        return self.lane_change

    def update_lane_change_target_speed(self, target_speed_mps):
        self.lane_change.target_speed_mps = max(0.0, float(target_speed_mps))
        return self.lane_change

    def complete_lane_change(self, reason):
        return self._release_lane_change("complete", reason, False)

    def abandon_lane_change(self, reason, suppress_recommit=False):
        return self._release_lane_change("abandoned", reason, suppress_recommit)

    def _release_lane_change(self, outcome, reason, preserve_completed_option):
        if not self.lane_change.active:
            return False
        self.last_release = {"reason": str(reason), "outcome": str(outcome),
                             "option": str(self.lane_change.option)}
        self.finish_lane_change_lifecycle(
            completed=(outcome == "complete" or bool(preserve_completed_option)))
        return True
