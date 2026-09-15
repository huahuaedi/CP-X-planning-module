"""Scenario-aware speed planning for the CP-X pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from .behavior_decision import BehaviorDecision

from opencda.planning_module.behavior_planner.car_follow import idm_acceleration


@dataclass(frozen=True)
class SpeedPlan:
    target_speed_mps: float
    speed_cap_mps: float
    stop_goal_active: bool
    reason: str = ""
    front_gap_m: Optional[float] = None
    desired_follow_gap_m: Optional[float] = None
    continuous_following_active: bool = False
    idm_acceleration_mps2: Optional[float] = None
    requested_speed_mps: float = 0.0
    scenario_cap_mps: Optional[float] = None
    turn_cap_mps: Optional[float] = None
    lane_change_cap_mps: Optional[float] = None
    following_cap_mps: Optional[float] = None
    turn_approach_cap_mps: Optional[float] = None
    upcoming_turn_distance_m: Optional[float] = None
    limiting_owner: str = "behavior_request"
    active_constraints: tuple[str, ...] = ()
    external_constraints: tuple = ()

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "speed_plan_target_mps": float(self.target_speed_mps),
            "speed_plan_cap_mps": float(self.speed_cap_mps),
            "speed_plan_stop_goal_active": bool(self.stop_goal_active),
            "speed_plan_reason": str(self.reason),
            "speed_plan_front_gap_m": (
                "" if self.front_gap_m is None else float(self.front_gap_m)
            ),
            "speed_plan_desired_follow_gap_m": (
                ""
                if self.desired_follow_gap_m is None
                else float(self.desired_follow_gap_m)
            ),
            "speed_plan_continuous_following_active": bool(
                self.continuous_following_active
            ),
            "speed_plan_idm_acceleration_mps2": (
                ""
                if self.idm_acceleration_mps2 is None
                else float(self.idm_acceleration_mps2)
            ),
            "speed_owner_requested_mps": float(self.requested_speed_mps),
            "speed_owner_scenario_cap_mps": (
                "" if self.scenario_cap_mps is None else float(self.scenario_cap_mps)
            ),
            "speed_owner_turn_cap_mps": (
                "" if self.turn_cap_mps is None else float(self.turn_cap_mps)
            ),
            "speed_owner_lane_change_cap_mps": (
                ""
                if self.lane_change_cap_mps is None
                else float(self.lane_change_cap_mps)
            ),
            "speed_owner_following_cap_mps": (
                "" if self.following_cap_mps is None else float(self.following_cap_mps)
            ),
            "speed_owner_turn_approach_cap_mps": (
                ""
                if self.turn_approach_cap_mps is None
                else float(self.turn_approach_cap_mps)
            ),
            "speed_owner_upcoming_turn_distance_m": (
                ""
                if self.upcoming_turn_distance_m is None
                else float(self.upcoming_turn_distance_m)
            ),
            "speed_owner_selected_target_mps": float(self.target_speed_mps),
            "speed_owner_limiting_owner": str(self.limiting_owner),
            "speed_owner_active_constraints": ";".join(self.active_constraints),
        }


@dataclass(frozen=True)
class SpeedCeilingResult:
    target_speed_mps: float
    destination_state: list[float]
    reference_samples: list[dict[str, object]]
    applied: bool
    reduction_mps: float


@dataclass(frozen=True)
class SpeedConstraint:
    """One named upper bound submitted to the longitudinal planner."""

    owner: str
    maximum_mps: float
    reason: str = ""


def conflict_corridor_speed_constraint(
    *,
    corridor: object,
    reference_samples: Sequence[Mapping[str, object]],
    ego_x_m: float,
    ego_y_m: float,
    comfortable_deceleration_mps2: float,
    corridor_dt_s: float = 0.1,
) -> Optional[SpeedConstraint]:
    """Convert an active Stage-C progress bound into a nominal speed limit.

    Stage D keeps the original per-stage half spaces as the final safety
    constraint.  This envelope gives the longitudinal planner the same
    information early enough to approach the bound with comfortable braking,
    instead of asking the QP to wait and then use its hard deceleration limit.
    An open corridor produces no constraint and therefore leaves the normal
    single-vehicle speed path unchanged.
    """

    from opencda.planning_module.pipeline.mpc_obstacle_relevance import (
        _polyline_xy,
        project_to_extended_polyline,
    )

    polyline = _polyline_xy(reference_samples)
    if len(polyline) < 2:
        return None
    upper_bounds = list(getattr(corridor, "s_hi", ()) or ())
    bindings = list(getattr(corridor, "binding", ()) or ())
    active = [
        (index, float(upper), str(bindings[index]))
        for index, upper in enumerate(upper_bounds)
        if index < len(bindings)
        and str(bindings[index])
        and math.isfinite(float(upper))
        and abs(float(upper)) < 1.0e8
    ]
    if not active:
        return None
    ego_station_m = project_to_extended_polyline(
        float(ego_x_m), float(ego_y_m), polyline
    )[1]
    deceleration_mps2 = max(0.1, float(comfortable_deceleration_mps2))
    dt_s = max(1.0e-3, float(corridor_dt_s))
    by_stage = {index: (upper, binding) for index, upper, binding in active}
    candidates = []
    for index, upper, binding in active:
        remaining_m = max(0.0, upper - float(ego_station_m))
        # A moving peer's station bound advances with it. Treating the
        # smallest row as a stationary stop line turns a perfectly parallel
        # make-gap proposal into a zero-speed command. The row's local slope
        # is the speed of that moving limit; static crossing rows have zero
        # slope and retain the original braking-distance approach.
        neighbor = by_stage.get(index + 1)
        neighbor_index = index + 1
        if neighbor is not None and neighbor[1] != binding:
            neighbor = None
        if neighbor is None:
            neighbor = by_stage.get(index - 1)
            neighbor_index = index - 1
            if neighbor is not None and neighbor[1] != binding:
                neighbor = None
        bound_velocity_mps = 0.0
        if neighbor is not None:
            neighbor_upper = neighbor[0]
            bound_velocity_mps = max(
                0.0,
                (neighbor_upper - upper)
                / ((neighbor_index - index) * dt_s),
            )
        candidates.append((
            bound_velocity_mps
            + math.sqrt(2.0 * deceleration_mps2 * remaining_m),
            binding, remaining_m, bound_velocity_mps,
        ))
    maximum_mps, binding, remaining_m, bound_velocity_mps = min(
        candidates, key=lambda item: item[0]
    )
    return SpeedConstraint(
        owner="cav_conflict",
        maximum_mps=float(maximum_mps),
        reason=(
            "cooperative_corridor_approach:"
            f"binding={binding},remaining_m={remaining_m:.3f},"
            f"bound_velocity_mps={bound_velocity_mps:.3f}"
        ),
    )


def cooperative_gap_speed_constraint(
    *,
    reference_samples: Sequence[Mapping[str, object]],
    ego_x_m: float,
    ego_y_m: float,
    ego_speed_mps: float,
    peer_x_m: float,
    peer_y_m: float,
    peer_speed_mps: float,
    peer_id: str,
    peer_length_m: float,
    ego_half_length_m: float,
    desired_bumper_gap_m: float,
    preparation_time_s: float,
    comfortable_deceleration_mps2: float,
    planning_dt_s: float,
) -> Optional[SpeedConstraint]:
    """Open a negotiated gap before an adjacent peer physically merges.

    A proposed resource claim has no MPC occupancy row. Its losing vehicle
    may nevertheless need to create longitudinal room before the winner can
    pass the ordinary lane-change safety gate. The target is the relative
    speed needed to reach the bumper gap over the maneuver preparation time;
    each tick may lower the current speed target only by a comfortable
    deceleration step. Physical collision limits remain Stage C/D's owner.
    """

    from opencda.planning_module.pipeline.mpc_obstacle_relevance import (
        _polyline_xy,
        project_to_extended_polyline,
    )

    polyline = _polyline_xy(reference_samples)
    if len(polyline) < 2:
        return None
    ego_station = project_to_extended_polyline(
        float(ego_x_m), float(ego_y_m), polyline
    )[1]
    peer_station = project_to_extended_polyline(
        float(peer_x_m), float(peer_y_m), polyline
    )[1]
    if peer_station <= ego_station:
        return None
    half_peer = (
        0.5 * float(peer_length_m)
        if float(peer_length_m) > 0.0
        else max(0.0, float(ego_half_length_m))
    )
    centre_gap_m = float(peer_station - ego_station)
    required_centre_gap_m = (
        max(0.0, float(desired_bumper_gap_m))
        + max(0.0, float(ego_half_length_m)) + half_peer
    )
    deficit_m = max(0.0, required_centre_gap_m - centre_gap_m)
    if deficit_m <= 0.0:
        return None
    desired_mps = max(
        0.0,
        float(peer_speed_mps)
        - deficit_m / max(0.1, float(preparation_time_s)),
    )
    reachable_mps = max(
        0.0,
        float(ego_speed_mps)
        - max(0.1, float(comfortable_deceleration_mps2))
        * max(0.01, float(planning_dt_s)),
    )
    return SpeedConstraint(
        owner="cooperative_gap",
        maximum_mps=max(desired_mps, reachable_mps),
        reason=(
            f"make_gap:peer={peer_id},centre_gap_m={centre_gap_m:.3f},"
            f"required_m={required_centre_gap_m:.3f}"
        ),
    )


@dataclass(frozen=True)
class SpeedTarget:
    """Final immutable longitudinal intent for one planning frame."""

    requested_mps: float
    target_mps: float
    limiting_owner: str
    stop_required: bool
    constraints: tuple[SpeedConstraint, ...] = ()
    revision: int = 0

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "speed_target_revision": int(self.revision),
            "speed_target_requested_mps": float(self.requested_mps),
            "speed_target_mps": float(self.target_mps),
            "speed_target_limiting_owner": str(self.limiting_owner),
            "speed_target_stop_required": bool(self.stop_required),
            "speed_target_constraints": ";".join(
                constraint.owner for constraint in self.constraints
            ),
        }


@dataclass(frozen=True)
class ResolvedSpeedFrame:
    """One frozen longitudinal target and its applied trajectory ceiling."""

    target: SpeedTarget
    ceiling: SpeedCeilingResult

    def trace_fields(self) -> dict[str, object]:
        fields = self.target.as_debug_fields()
        fields.update({
            "speed_owner_selected_target_mps": float(self.target.target_mps),
            "speed_owner_limiting_owner": str(self.target.limiting_owner),
            "speed_owner_active_constraints": ";".join(
                constraint.owner for constraint in self.target.constraints
            ),
            "speed_owner_proposed_post_plan_target_mps": float(
                self.target.target_mps
            ),
            "speed_owner_ceiling_applied": bool(self.ceiling.applied),
            "speed_owner_ceiling_reduction_mps": float(
                self.ceiling.reduction_mps
            ),
        })
        return fields


class SpeedTargetPlanner:
    """Sole owner that freezes behavior speed proposals into a frame target.

    Behavior and scenario modules may submit named ceilings.  They never edit
    the trajectory themselves.  Safety/fallback layers are deliberately not
    represented here: they may only reduce this target after validation.
    """

    def __init__(self):
        self._revision = 0
        self._destination_route_revision = ""
        self._destination_approach_active = False
        self._destination_approach_cap_mps = float("inf")

    @staticmethod
    def turn_curvature_constraint(
        curvature_1pm: Optional[float], config: Mapping[str, object]
    ) -> Optional[SpeedConstraint]:
        """Translate the persistent turn geometry into one speed ceiling.

        This is the only turn-speed policy.  ReferenceLineProvider measures
        the immutable master; SpeedTargetPlanner applies the lateral-accel
        relation ``v = sqrt(a_lat_max / abs(curvature))``.  No candidate or
        bridge branch owns a separate fixed turn speed.
        """

        if curvature_1pm is None:
            return None
        curvature = abs(float(curvature_1pm))
        minimum_curvature = max(0.0, float(config.get(
            "full_intersection_turn_curvature_min_curvature_1pm", 0.01
        )))
        if curvature <= minimum_curvature:
            return None
        lateral_accel_mps2 = max(0.1, float(config.get(
            "full_intersection_turn_lateral_accel_comfort_mps2", 2.5
        )))
        maximum_mps = math.sqrt(lateral_accel_mps2 / max(1.0e-6, curvature))
        return SpeedConstraint(
            owner="turn_master_curvature",
            maximum_mps=float(maximum_mps),
            reason=(
                "persistent_turn_master_lateral_acceleration_limit:"
                "curvature_1pm=%.6f" % float(curvature)
            ),
        )

    @staticmethod
    def constrain_plan(
        speed_plan: SpeedPlan,
        constraint: Optional[SpeedConstraint],
    ) -> SpeedPlan:
        """Return ``speed_plan`` with one named external ceiling applied.

        Late planning stages, such as cooperative conflict resolution, may
        discover a longitudinal limit only after the ordinary behavior speed
        proposal has been built.  They submit that limit here instead of
        editing ``speed_ref_mps`` or trajectory samples.  The final
        :meth:`resolve` call remains the sole owner of the executable target.
        """

        if constraint is None:
            return speed_plan
        constraint_owner = str(constraint.owner)
        maximum_mps = max(0.0, float(constraint.maximum_mps))
        retained = tuple(
            item
            for item in speed_plan.external_constraints
            if str(getattr(item, "owner", "")) != constraint_owner
        )
        constrained_target = min(
            max(0.0, float(speed_plan.target_speed_mps)),
            maximum_mps,
        )
        limiting_owner = str(speed_plan.limiting_owner)
        if constrained_target < float(speed_plan.target_speed_mps) - 1.0e-9:
            limiting_owner = constraint_owner
        active_constraints = tuple(speed_plan.active_constraints)
        if constraint_owner not in active_constraints:
            active_constraints += (constraint_owner,)
        return replace(
            speed_plan,
            target_speed_mps=float(constrained_target),
            speed_cap_mps=min(
                max(0.0, float(speed_plan.speed_cap_mps)), maximum_mps
            ),
            limiting_owner=limiting_owner,
            active_constraints=active_constraints,
            external_constraints=retained + (constraint,),
        )

    def destination_approach_constraint(
        self,
        *,
        route_revision: str,
        route_found: bool,
        route_reached_destination: bool,
        remaining_distance_m: float,
        ego_speed_mps: float,
        deceleration_mps2: float,
        buffer_m: float,
    ) -> tuple[Optional[SpeedConstraint], bool, float]:
        """Return a monotonic destination-speed phase for one route.

        Once the braking envelope is entered, slowing the vehicle must not
        shrink that envelope and release the constraint. The phase resets
        only when route identity changes or the route disappears.
        """

        revision = str(route_revision or "")
        if revision != self._destination_route_revision or not bool(route_found):
            self._destination_route_revision = revision
            self._destination_approach_active = False
            self._destination_approach_cap_mps = float("inf")
        deceleration = max(0.5, float(deceleration_mps2))
        buffer = max(0.0, float(buffer_m))
        required_distance_m = (
            max(0.0, float(ego_speed_mps)) ** 2 / (2.0 * deceleration)
            + buffer
        )
        finite_remaining = math.isfinite(float(remaining_distance_m))
        entered = bool(
            route_found
            and finite_remaining
            and float(remaining_distance_m) >= 0.0
            and float(remaining_distance_m) <= float(required_distance_m)
        )
        if entered:
            self._destination_approach_active = True
        if bool(route_reached_destination):
            self._destination_approach_active = False
            self._destination_approach_cap_mps = 0.0
            return None, False, float(required_distance_m)
        if not self._destination_approach_active:
            return None, False, float(required_distance_m)
        usable_distance_m = max(0.0, float(remaining_distance_m) - buffer)
        cap_mps = math.sqrt(2.0 * deceleration * usable_distance_m)
        self._destination_approach_cap_mps = min(
            float(self._destination_approach_cap_mps), float(cap_mps)
        )
        return SpeedConstraint(
            owner="destination_approach",
            maximum_mps=max(0.0, float(self._destination_approach_cap_mps)),
            reason="route_destination_approach_speed_profile",
        ), True, float(required_distance_m)

    @staticmethod
    def propose(**kwargs) -> SpeedPlan:
        """Build the typed longitudinal policy proposal for one frame."""

        return build_speed_plan(**kwargs)

    def resolve(
        self,
        *,
        behavior: "BehaviorDecision",
        speed_plan: Optional[SpeedPlan] = None,
        additional_constraints: Sequence[SpeedConstraint] = (),
    ) -> SpeedTarget:
        # SpeedPlan is a typed policy proposal. Debug dictionaries are output
        # only and must never be parsed back into the control path.
        requested = max(0.0, float(behavior.requested_speed_mps))
        target = requested
        limiting_owner = "behavior_request"
        constraints = []
        if speed_plan is not None:
            planned_target = max(0.0, float(speed_plan.target_speed_mps))
            target = min(float(requested), float(planned_target))
            limiting_owner = (
                "behavior_request"
                if float(requested) < float(planned_target)
                else str(speed_plan.limiting_owner)
            )
            for owner, maximum in (
                ("scenario_cap", speed_plan.scenario_cap_mps),
                ("turn_cap", speed_plan.turn_cap_mps),
                ("lane_change_cap", speed_plan.lane_change_cap_mps),
                (
                    "idm_following"
                    if speed_plan.continuous_following_active
                    and speed_plan.idm_acceleration_mps2 is not None
                    else "following_cap",
                    speed_plan.following_cap_mps,
                ),
                ("turn_approach_cap", speed_plan.turn_approach_cap_mps),
            ):
                if maximum is not None and math.isfinite(float(maximum)):
                    constraints.append(SpeedConstraint(
                        owner=owner,
                        maximum_mps=max(0.0, float(maximum)),
                    ))
            constraints.extend(speed_plan.external_constraints)
        constraints.extend(additional_constraints)
        for constraint in constraints:
            if float(constraint.maximum_mps) < target:
                target = max(0.0, float(constraint.maximum_mps))
                limiting_owner = str(constraint.owner)
        if behavior.stop_required or bool(
            speed_plan is not None and speed_plan.stop_goal_active
        ):
            target = 0.0
            limiting_owner = (
                str(speed_plan.limiting_owner)
                if speed_plan is not None and speed_plan.stop_goal_active
                else "behavior_stop"
            )
        self._revision += 1
        return SpeedTarget(
            requested_mps=requested,
            target_mps=float(target),
            limiting_owner=limiting_owner,
            stop_required=bool(behavior.stop_required),
            constraints=tuple(constraints),
            revision=self._revision,
        )

    @staticmethod
    def apply(
        target: SpeedTarget,
        *,
        destination_state: Sequence[float],
        reference_samples: Sequence[Mapping[str, object]],
    ) -> SpeedCeilingResult:
        return enforce_speed_ceiling(
            proposed_target_mps=float(target.requested_mps),
            ceiling_mps=float(target.target_mps),
            destination_state=destination_state,
            reference_samples=reference_samples,
        )

    def resolve_frame(
        self,
        *,
        behavior: "BehaviorDecision",
        speed_plan: Optional[SpeedPlan],
        additional_constraints: Sequence[SpeedConstraint],
        destination_state: Sequence[float],
        reference_samples: Sequence[Mapping[str, object]],
    ) -> ResolvedSpeedFrame:
        target = self.resolve(
            behavior=behavior,
            speed_plan=speed_plan,
            additional_constraints=additional_constraints,
        )
        return ResolvedSpeedFrame(
            target=target,
            ceiling=self.apply(
                target,
                destination_state=destination_state,
                reference_samples=reference_samples,
            ),
        )


def turn_approach_lookahead_m(
    *,
    cruise_speed_mps: float,
    config: Mapping[str, object],
) -> float:
    """Return enough route preview to decelerate to the turn-entry speed."""

    cruise_speed = max(0.0, float(cruise_speed_mps))
    comfortable_decel = max(
        0.1,
        float(config.get("turn_approach_comfort_decel_mps2", 2.5)),
    )
    braking_distance = max(
        0.0,
        cruise_speed * cruise_speed / (2.0 * comfortable_decel),
    )
    entry_buffer = max(
        0.0,
        float(config.get("turn_approach_entry_buffer_m", 5.0)),
    )
    preview_margin = max(
        0.0,
        float(config.get("turn_approach_preview_margin_m", 10.0)),
    )
    configured_minimum = max(
        0.0,
        float(config.get("scenario_turn_prepare_lookahead_m", 15.0)),
    )
    return max(
        configured_minimum,
        braking_distance + entry_buffer + preview_margin,
    )


def enforce_speed_ceiling(
    *,
    proposed_target_mps: float,
    ceiling_mps: float,
    destination_state: Sequence[float],
    reference_samples: Sequence[Mapping[str, object]],
) -> SpeedCeilingResult:
    """Apply the planner speed ceiling without changing reference geometry."""

    proposed = max(0.0, float(proposed_target_mps))
    ceiling = max(0.0, float(ceiling_mps))
    selected = min(proposed, ceiling)
    destination = list(destination_state)
    if len(destination) >= 3:
        destination[2] = min(max(0.0, float(destination[2])), ceiling)
    samples = [dict(sample) for sample in reference_samples]
    for sample in samples:
        for key in ("speed_ref_mps", "v_ref_mps", "speed_mps", "v"):
            if key not in sample:
                continue
            try:
                sample[key] = min(max(0.0, float(sample[key])), ceiling)
            except (TypeError, ValueError):
                continue
    reduction = max(0.0, proposed - selected)
    return SpeedCeilingResult(
        target_speed_mps=float(selected),
        destination_state=destination,
        reference_samples=samples,
        applied=bool(reduction > 1.0e-6),
        reduction_mps=float(reduction),
    )


def effective_emergency_gap_m(
    *,
    base_emergency_gap_m: float,
    ego_speed_mps: float,
    front_obstacle_speed_mps: Optional[float] = None,
    standstill_buffer_m: float = 1.0,
    time_headway_s: float = 1.5,
) -> float:
    """IDM/RSS-style emergency-stop floor: scales with closing speed.

    A flat distance floor can't tell "closing fast" from "queued
    nose-to-tail, neither vehicle moving" -- both read as "gap <= floor"
    even though only one is actually dangerous. Once several vehicles
    compress into a tight, genuinely stationary queue (a real
    construction-zone bottleneck), a flat floor latches every one of them
    at a hard stop forever, since the gap between two motionless vehicles
    never grows on its own. This shrinks the floor toward a small,
    still-safe standstill buffer as the closing speed (ego minus the
    obstacle ahead) drops toward zero, capped at ``base_emergency_gap_m``
    so a genuinely closing obstacle is never treated more permissively
    than a flat floor would. If the obstacle's own speed is unknown,
    assumes the least favorable case (stationary target, i.e. closing
    speed = ego speed) -- identical to a flat floor's implicit assumption.
    """
    closing_speed_mps = max(
        0.0,
        float(ego_speed_mps)
        - (
            0.0
            if front_obstacle_speed_mps is None
            else max(0.0, float(front_obstacle_speed_mps))
        ),
    )
    dynamic_distance_m = float(standstill_buffer_m) + float(time_headway_s) * float(
        closing_speed_mps
    )
    return float(min(float(base_emergency_gap_m), dynamic_distance_m))


def build_speed_plan(
    *,
    scenario_decision: object,
    behavior_decision: object,
    requested_speed_mps: float,
    ego_speed_mps: float,
    config: Mapping[str, object],
    front_gap_m: Optional[float] = None,
    front_obstacle_speed_mps: Optional[float] = None,
    upcoming_turn_direction: str = "",
    upcoming_turn_distance_m: Optional[float] = None,
    lane_change_commitment_active: bool = False,
    front_obstacle_is_source_lane: bool = False,
    previous_idm_acceleration_mps2: Optional[float] = None,
    additional_constraints: Sequence[SpeedConstraint] = (),
) -> SpeedPlan:
    """Return the speed target owned by the scenario/behavior layer."""

    requested = max(0.0, float(requested_speed_mps))
    scenario_cap = getattr(scenario_decision, "speed_cap_mps", None)
    scenario_state = str(getattr(scenario_decision, "state", "") or "").strip().upper()
    # PREPARE_TURN may become visible while a route-required lane change is
    # still executing. Its fixed scenario cap is not authority to clamp the
    # locked maneuver immediately; the distance-based turn-approach envelope
    # remains active so longitudinal braking can begin while the independently
    # locked lateral maneuver finishes. Traffic-control stops remain active.
    suppress_turn_preparation = bool(
        lane_change_commitment_active
        and scenario_state == "PREPARE_TURN"
        and not bool(getattr(scenario_decision, "stop_goal_active", False))
    )
    if bool(suppress_turn_preparation):
        scenario_cap = None
    scenario_cap_value = (
        None if scenario_cap is None else max(0.0, float(scenario_cap))
    )
    cap = requested if scenario_cap is None else min(requested, max(0.0, float(scenario_cap)))
    limiting_owner = "behavior_request"
    active_constraints = []
    if scenario_cap_value is not None:
        active_constraints.append("scenario_cap")
        if scenario_cap_value < requested:
            limiting_owner = "scenario_cap"
    stop_goal = bool(getattr(scenario_decision, "stop_goal_active", False))
    reason = str(getattr(scenario_decision, "reason", "") or "")
    decision = str(behavior_decision or "").strip().lower()
    if decision in {
        "stop_at_intersection",
        "stop_sign",
        "emergency_brake",
        "static_obstacle_stop",
    }:
        stop_goal = True
    turn_cap_mps = None
    turn_approach_cap_mps = None
    finite_turn_distance_m = None
    turn_direction = str(upcoming_turn_direction or "").strip().lower()
    if (
        turn_direction in {"left", "right"}
        and upcoming_turn_distance_m is not None
        and math.isfinite(float(upcoming_turn_distance_m))
    ):
        finite_turn_distance_m = max(0.0, float(upcoming_turn_distance_m))
    # This stage deliberately does not guess turn curvature with a fixed
    # speed. Once ReferenceLineProvider installs the immutable turn master,
    # SpeedTargetPlanner.turn_curvature_constraint owns that ceiling.
    lane_change_cap_mps = None
    if decision in {"lane_change_left", "lane_change_right"}:
        # A route-required lane change can start immediately after ego is
        # still accelerating from a stop (e.g. CHANGELANELEFT right at
        # spawn, with no straight lead-in to reach cruise speed first).
        # Without a cap, the lateral S-curve gets executed while
        # longitudinal speed is still an unsettled transient, and the
        # combined demand can press the actual footprint into the lane
        # boundary. Capping speed for the whole maneuver -- not just
        # smoothing the transition into it -- keeps the lateral maneuver's
        # dynamics decoupled from however unsettled the longitudinal speed
        # still is.
        # In curvature-aware mode the winning lane-change geometry owns this
        # cap. At this stage no candidate reference exists yet, so applying a
        # fixed cap here would discard the speed used to size the maneuver and
        # make duration/curvature selection internally inconsistent.
        if not bool(config.get("full_lane_change_dynamic_speed_cap_enabled", True)):
            lane_change_cap_mps = max(
                0.1, float(config.get("full_lane_change_speed_cap_mps", 3.0))
            )
            active_constraints.append("lane_change_cap")
            previous_cap = float(cap)
            cap = min(float(cap), float(lane_change_cap_mps))
            if float(cap) < previous_cap:
                limiting_owner = "lane_change_cap"
    following_active = False
    following_cap_mps = None
    following_gap_m = None
    desired_gap_m = None
    idm_acceleration_mps2 = None
    source_lane_following_suppressed = bool(
        bool(front_obstacle_is_source_lane)
        and (
            decision in {"lane_change_left", "lane_change_right"}
            or lane_change_commitment_active
        )
    )
    if bool(source_lane_following_suppressed):
        # Candidate/authorization has already established a safe target
        # corridor.  A lead reported in the source lane is no longer the
        # longitudinal owner while ego is crossing away from it.  Emergency
        # stop remains upstream and target-lane collision risk remains in the
        # candidate/prediction gate.
        active_constraints.append("source_lane_following_suppressed")
    if (
        front_gap_m is not None
        and math.isfinite(float(front_gap_m))
        and not stop_goal
        and not source_lane_following_suppressed
    ):
        gap_m = max(0.0, float(front_gap_m))
        following_gap_m = float(gap_m)
        emergency_gap_m = max(
            0.5,
            float(config.get("following_emergency_gap_m", 3.0)),
        )
        standstill_gap_m = max(
            emergency_gap_m,
            float(config.get("following_standstill_gap_m", 5.0)),
        )
        time_headway_s = max(
            0.1,
            float(config.get("following_time_headway_s", 1.5)),
        )
        free_gap_margin_m = max(
            1.0,
            float(config.get("following_free_gap_margin_m", 6.0)),
        )
        minimum_follow_speed_mps = max(
            0.0,
            float(config.get("following_minimum_speed_mps", 0.35)),
        )
        # Time-headway spacing keyed on ego's own speed alone can't tell
        # "leader cruising same speed" from "leader accelerating away" --
        # both look identical (same ego_speed_mps), so a real, growing gap
        # to a lead vehicle that is pulling away still reads as "too close
        # for my speed" and triggers a needless slowdown. Key it on closing
        # speed (ego minus the obstacle ahead) instead, floored at 0 so a
        # leader that is stationary or faster than ego never *shrinks* the
        # buffer below the plain standstill_gap_m -- this only ever relaxes
        # today's requirement when the leader is genuinely pulling away, the
        # same standard "constant time-gap" ACC policy uses.
        closing_speed_for_desired_gap_mps = max(
            0.0,
            float(ego_speed_mps)
            - (
                0.0
                if front_obstacle_speed_mps is None
                else max(0.0, float(front_obstacle_speed_mps))
            ),
        )
        desired_gap_m = standstill_gap_m + time_headway_s * closing_speed_for_desired_gap_mps
        free_gap_m = desired_gap_m + free_gap_margin_m
        emergency_standstill_buffer_m = max(
            0.0,
            float(config.get("following_emergency_standstill_buffer_m", 1.0)),
        )
        effective_gap_m = effective_emergency_gap_m(
            base_emergency_gap_m=float(emergency_gap_m),
            ego_speed_mps=float(ego_speed_mps),
            front_obstacle_speed_mps=front_obstacle_speed_mps,
            standstill_buffer_m=float(emergency_standstill_buffer_m),
            time_headway_s=float(time_headway_s),
        )
        if gap_m <= effective_gap_m:
            cap = 0.0
            stop_goal = True
            reason = _join_reason(reason, "speed_plan_obstacle_emergency_stop")
        elif (
            bool(config.get("following_idm_enabled", True))
            and front_obstacle_speed_mps is not None
            and math.isfinite(float(front_obstacle_speed_mps))
        ):
            # IDM owns normal car-following whenever the selected lead actor
            # has a usable speed estimate. Convert its acceleration command
            # into a short-horizon, kinematically reachable speed reference;
            # all previously computed scenario/turn/lane-change limits remain
            # hard ceilings. This lets a rear vehicle accelerate back toward
            # cruise after a lane change, while a closing lead continuously
            # lowers the same reference instead of engaging a second,
            # competing distance-only controller.
            idm_horizon_s = max(
                0.05,
                float(config.get("following_idm_target_horizon_s", 0.20)),
            )
            idm_max_acceleration_mps2 = max(
                0.05,
                float(config.get("following_idm_max_acceleration_mps2", 2.0)),
            )
            idm_comfort_deceleration_mps2 = max(
                0.05,
                float(
                    config.get(
                        "following_idm_comfort_deceleration_mps2", 3.0
                    )
                ),
            )
            relative_speed_mps = float(ego_speed_mps) - max(
                0.0, float(front_obstacle_speed_mps)
            )
            desired_gap_m = float(standstill_gap_m) + max(
                0.0,
                float(ego_speed_mps) * float(time_headway_s)
                + float(ego_speed_mps)
                * float(relative_speed_mps)
                / (
                    2.0
                    * math.sqrt(
                        float(idm_max_acceleration_mps2)
                        * float(idm_comfort_deceleration_mps2)
                    )
                ),
            )
            configured_catchup_delta_mps = max(
                0.0,
                float(config.get("following_max_catchup_delta_mps", 2.0)),
            )
            # requested is the normal cruise target, not a legal hard limit.
            # IDM continuously owns the interaction; avoid switching its
            # free-flow speed at a gap threshold because that discontinuity
            # reintroduces oscillation near the desired spacing.
            idm_desired_speed_mps = max(
                0.1,
                float(
                    config.get(
                        "following_idm_free_flow_speed_mps",
                        float(requested)
                        + max(
                            0.0,
                            float(
                                config.get(
                                    "following_idm_free_speed_headroom_mps", 10.0
                                )
                            ),
                        ),
                    )
                ),
            )
            idm_acceleration_mps2 = idm_acceleration(
                v=float(ego_speed_mps),
                v_lead=max(0.0, float(front_obstacle_speed_mps)),
                gap_m=float(gap_m),
                v_desired=float(idm_desired_speed_mps),
                a_max=float(idm_max_acceleration_mps2),
                b_comfort=float(idm_comfort_deceleration_mps2),
                time_headway_s=float(time_headway_s),
                min_gap_m=float(standstill_gap_m),
                delta=max(
                    1.0,
                    float(config.get("following_idm_acceleration_exponent", 4.0)),
                ),
            )
            if (
                previous_idm_acceleration_mps2 is not None
                and math.isfinite(float(previous_idm_acceleration_mps2))
            ):
                idm_control_dt_s = max(
                    0.01,
                    float(config.get("following_idm_control_dt_s", 0.05)),
                )
                idm_max_jerk_mps3 = max(
                    0.1,
                    float(config.get("following_idm_max_jerk_mps3", 3.0)),
                )
                maximum_acceleration_step = (
                    float(idm_max_jerk_mps3) * float(idm_control_dt_s)
                )
                idm_acceleration_mps2 = max(
                    float(previous_idm_acceleration_mps2) - maximum_acceleration_step,
                    min(
                        float(previous_idm_acceleration_mps2) + maximum_acceleration_step,
                        float(idm_acceleration_mps2),
                    ),
                )
            # Single longitudinal controller: IDM alone determines the next
            # reachable speed from ego speed, lead speed, and gap. Do not add
            # a second lead-speed/gap controller here; their branch switching
            # produced the high-frequency target-speed sawtooth diagnosed by
            # tools/test_idm_following.py.
            rate_limited_target_mps = max(
                0.0,
                float(ego_speed_mps)
                + float(idm_acceleration_mps2) * float(idm_horizon_s),
            )
            # Only the ordinary behavior-request ceiling may be relaxed for
            # catch-up. All explicit safety/context caps stay hard.
            following_hard_ceiling_mps = float(requested) + (
                float(configured_catchup_delta_mps)
            )
            for explicit_cap_mps in (
                scenario_cap_value,
                turn_approach_cap_mps,
                turn_cap_mps,
                lane_change_cap_mps,
            ):
                if explicit_cap_mps is not None:
                    following_hard_ceiling_mps = min(
                        float(following_hard_ceiling_mps),
                        max(0.0, float(explicit_cap_mps)),
                    )
            following_cap_mps = min(
                float(following_hard_ceiling_mps),
                float(rate_limited_target_mps),
            )
            active_constraints.append("idm_following")
            previous_cap = float(cap)
            cap = float(following_cap_mps)
            if abs(float(cap) - previous_cap) > 1.0e-6:
                limiting_owner = "idm_following"
            following_active = True
        elif gap_m < free_gap_m:
            # Conservative fallback for a detected lead whose velocity is
            # unavailable. Do not pretend that a stationary velocity sample
            # exists; retain the earlier distance-only behavior instead.
            ratio = (gap_m - emergency_gap_m) / max(
                1.0e-6,
                free_gap_m - emergency_gap_m,
            )
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            follow_cap = minimum_follow_speed_mps + smooth_ratio * max(
                0.0,
                float(cap) - minimum_follow_speed_mps,
            )
            following_cap_mps = float(follow_cap)
            active_constraints.append("following_cap")
            previous_cap = float(cap)
            cap = min(float(cap), float(follow_cap))
            if float(cap) < previous_cap:
                limiting_owner = "following_cap"
            following_active = True
    # Context modules submit named ceilings; they do not write the selected
    # speed directly. Apply them after following so they remain hard limits.
    for constraint in additional_constraints:
        maximum_mps = max(0.0, float(constraint.maximum_mps))
        active_constraints.append(str(constraint.owner))
        if maximum_mps < float(cap):
            cap = float(maximum_mps)
            limiting_owner = str(constraint.owner)
    if stop_goal:
        cap = 0.0
        active_constraints.append("stop_zero")
        limiting_owner = (
            "emergency_stop"
            if decision == "emergency_brake" or "obstacle_emergency_stop" in reason
            else "normal_stop"
        )
    if not math.isfinite(cap):
        cap = 0.0
    if decision in {"intersection_turn_left", "intersection_turn_right"}:
        reason = _join_reason(reason, "speed_plan_turn_cap")
    elif turn_approach_cap_mps is not None and turn_approach_cap_mps < requested:
        reason = _join_reason(reason, "speed_plan_turn_approach_cap")
    if (
        decision in {"lane_change_left", "lane_change_right"}
        and lane_change_cap_mps is not None
    ):
        reason = _join_reason(reason, "speed_plan_lane_change_cap")
    if stop_goal:
        reason = _join_reason(reason, "speed_plan_stop_zero")
    elif following_active:
        reason = _join_reason(reason, "speed_plan_continuous_following")
    elif scenario_cap is not None and float(cap) < float(requested):
        reason = _join_reason(reason, "speed_plan_scenario_cap")
    return SpeedPlan(
        target_speed_mps=float(cap),
        speed_cap_mps=float(cap),
        stop_goal_active=bool(stop_goal),
        reason=str(reason),
        front_gap_m=following_gap_m,
        desired_follow_gap_m=desired_gap_m,
        continuous_following_active=bool(following_active),
        idm_acceleration_mps2=idm_acceleration_mps2,
        requested_speed_mps=float(requested),
        scenario_cap_mps=scenario_cap_value,
        turn_cap_mps=turn_cap_mps,
        lane_change_cap_mps=lane_change_cap_mps,
        following_cap_mps=following_cap_mps,
        turn_approach_cap_mps=turn_approach_cap_mps,
        upcoming_turn_distance_m=finite_turn_distance_m,
        limiting_owner=str(limiting_owner),
        active_constraints=tuple(active_constraints),
        external_constraints=tuple(additional_constraints),
    )


def _join_reason(first: str, second: str) -> str:
    if not first:
        return str(second)
    if str(second) in str(first).split(";"):
        return str(first)
    return f"{first};{second}"
