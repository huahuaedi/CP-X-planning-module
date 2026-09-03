"""Route-level authorization for lane-change decisions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Mapping, Optional, Sequence


class RouteManeuver(Enum):
    LANE_FOLLOW = "lane_follow"
    GO_STRAIGHT = "go_straight"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LaneChangeAuthorization:
    allowed: bool
    direction: Optional[str]
    reason: str
    required_by_route: bool
    distance_to_maneuver_m: Optional[float]
    target_lane_id: int
    maneuver: str

    def as_debug_fields(self) -> Mapping[str, object]:
        return {
            "lane_change_authorized": bool(self.allowed),
            "lane_change_authorization_direction": "" if self.direction is None else str(self.direction),
            "lane_change_authorization_reason": str(self.reason),
            "lane_change_required_by_route": bool(self.required_by_route),
            "lane_change_distance_to_maneuver_m": (
                "" if self.distance_to_maneuver_m is None else float(self.distance_to_maneuver_m)
            ),
            "lane_change_authorized_target_lane_id": int(self.target_lane_id),
            "route_maneuver_normalized": str(self.maneuver),
        }


@dataclass(frozen=True)
class OpportunisticLaneChangeAuthorization:
    """Authorization independent of global-route geometry ownership."""

    allowed: bool
    reason: str


def authorize_opportunistic_lane_change(
    *,
    enabled: bool,
    start_lock_active: bool,
    dense_traffic_lock_active: bool,
) -> OpportunisticLaneChangeAuthorization:
    """Gate a locally motivated lane change using behavior constraints only.

    Global-route reference permission is intentionally absent: it controls
    XY geometry ownership, not whether an AD-map corridor may be selected.
    """
    if not bool(enabled):
        return OpportunisticLaneChangeAuthorization(
            False, "opportunistic_lane_change_disabled"
        )
    if bool(start_lock_active):
        return OpportunisticLaneChangeAuthorization(
            False, "opportunistic_lane_change_start_lock"
        )
    if bool(dense_traffic_lock_active):
        return OpportunisticLaneChangeAuthorization(
            False, "opportunistic_lane_change_dense_traffic_lock"
        )
    return OpportunisticLaneChangeAuthorization(
        True, "opportunistic_lane_change_authorized"
    )


class RouteLaneChangeAuthorizationLatch:
    """Keep a route-required maneuver stable after its entry gate is crossed."""

    _TRANSIENT_LAPSE_REASONS = {
        "explicit_lane_change_trigger_too_far",
        "maneuver_too_far_for_lane_change",
        "route_maneuver_does_not_require_lane_change",
        "already_in_required_lane",
    }

    def __init__(self) -> None:
        self._active: Optional[LaneChangeAuthorization] = None

    @property
    def active(self) -> Optional[LaneChangeAuthorization]:
        return self._active

    def reset(self) -> None:
        self._active = None

    def update(
        self,
        authorization: LaneChangeAuthorization,
        *,
        target_reached: bool,
        in_turn_connector: bool,
    ) -> LaneChangeAuthorization:
        if bool(target_reached) or bool(in_turn_connector):
            self.reset()
            return authorization
        if bool(authorization.allowed) and bool(authorization.required_by_route):
            self._active = authorization
            return authorization
        if (
            self._active is not None
            and str(authorization.reason) in self._TRANSIENT_LAPSE_REASONS
        ):
            active = self._active
            return replace(
                active,
                reason="route_lane_change_authorization_latched",
                distance_to_maneuver_m=(
                    authorization.distance_to_maneuver_m
                    if authorization.distance_to_maneuver_m is not None
                    else active.distance_to_maneuver_m
                ),
            )
        return authorization


def normalize_route_maneuver(value: object) -> RouteManeuver:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "none", "unknown"}:
        return RouteManeuver.UNKNOWN
    if text in {"lanefollow", "lane_follow", "follow", "keep_lane"}:
        return RouteManeuver.LANE_FOLLOW
    if text in {"continue_straight", "straight", "go_straight", "through"}:
        return RouteManeuver.GO_STRAIGHT
    if text in {"left", "turn_left", "left_turn", "leftturn"}:
        return RouteManeuver.TURN_LEFT
    if text in {"right", "turn_right", "right_turn", "rightturn"}:
        return RouteManeuver.TURN_RIGHT
    if text in {"lane_change_left", "change_left", "left_lane_change"}:
        return RouteManeuver.LANE_CHANGE_LEFT
    if text in {"lane_change_right", "change_right", "right_lane_change"}:
        return RouteManeuver.LANE_CHANGE_RIGHT
    return RouteManeuver.UNKNOWN


def suppress_lane_change_for_lateral_owner(
    authorization: LaneChangeAuthorization,
    *,
    owner_state: object,
) -> LaneChangeAuthorization:
    """Give an active turn/recovery state exclusive lateral authority."""

    state = str(owner_state or "").strip().upper()
    exclusive_states = {
        "PREPARE_TURN",
        "INTERSECTION_TURN",
        "TURN_EXIT_STABILIZATION",
        "CREEP",
        "BOUNDARY_RECOVERY",
    }
    if not bool(authorization.allowed) or state not in exclusive_states:
        return authorization
    return LaneChangeAuthorization(
        allowed=False,
        direction=authorization.direction,
        reason=f"scenario_lateral_owner:{state.lower()}",
        required_by_route=bool(authorization.required_by_route),
        distance_to_maneuver_m=authorization.distance_to_maneuver_m,
        target_lane_id=int(authorization.target_lane_id),
        maneuver=str(authorization.maneuver),
    )


def lane_change_target_reached(
    *,
    current_lane_id: int,
    remembered_target_lane_id: int,
    current_ad_lane_id: int = 0,
    remembered_target_ad_lane_id: int = 0,
    target_in_local_frame: bool = False,
    target_lane_offset: int = 0,
) -> bool:
    """Resolve completion by corridor relation, then stable identity."""

    if bool(target_in_local_frame) and int(target_lane_offset) == 0:
        return True
    if int(current_ad_lane_id or 0) != 0 and int(remembered_target_ad_lane_id or 0) != 0:
        return int(current_ad_lane_id) == int(remembered_target_ad_lane_id)
    return int(current_lane_id or 0) == int(remembered_target_lane_id or 0)


_LANE_CHANGE_MANEUVERS = {
    RouteManeuver.LANE_CHANGE_LEFT,
    RouteManeuver.LANE_CHANGE_RIGHT,
}
_TURN_MANEUVERS = {RouteManeuver.TURN_LEFT, RouteManeuver.TURN_RIGHT}
_NO_LANE_CHANGE_MANEUVERS = {
    RouteManeuver.GO_STRAIGHT,
    RouteManeuver.LANE_FOLLOW,
    RouteManeuver.UNKNOWN,
}

# Authorization advances through these phases; a failed predicate at any phase
# short-circuits to a `_denied(...)` whose reason names why. The phases exist
# so distance is consulted once (the IN_WINDOW gate) instead of scattered
# through the branch soup this replaced, and so the RouteGeometry direction
# and the legacy id-space path are one path, not two that can disagree.
LC_PHASE_IDLE = "idle"
LC_PHASE_ELIGIBLE = "eligible"           # route needs a lane change; direction known
LC_PHASE_IN_WINDOW = "in_window"         # within the trigger-distance window
LC_PHASE_AUTHORIZED = "authorized"       # target lane clear -> go


def _resolve_offset_direction(topology_lane_offset: int) -> str:
    offset = int(topology_lane_offset or 0)
    return "left" if offset > 0 else "right" if offset < 0 else ""


def _trigger_window_denial(
    *,
    maneuver: RouteManeuver,
    is_explicit_lane_change: bool,
    distance: Optional[float],
    explicit_lane_change_start_distance_m: Optional[float],
    preparation_start_distance_m: float,
    latest_start_distance_m: float,
    target_lane_id: int,
) -> Optional[LaneChangeAuthorization]:
    """ELIGIBLE -> IN_WINDOW: the one place trigger distance is consulted.

    Explicit lane changes only have a *too far* edge (drop in once inside
    ``explicit_lane_change_start_distance_m``, if one was given). A lane change
    made to line up for a turn also has a *too close* edge -- past
    ``latest_start_distance_m`` there is no room left to complete it before the
    connector.
    """
    if distance is None:
        return None
    if is_explicit_lane_change:
        if (
            explicit_lane_change_start_distance_m is not None
            and float(distance) > float(explicit_lane_change_start_distance_m)
        ):
            return _denied(
                "explicit_lane_change_trigger_too_far", maneuver, distance, target_lane_id
            )
        return None
    if float(distance) > float(preparation_start_distance_m):
        return _denied(
            "maneuver_too_far_for_lane_change", maneuver, distance, target_lane_id
        )
    if float(distance) < float(latest_start_distance_m):
        return _denied(
            "maneuver_too_close_for_lane_change", maneuver, distance, target_lane_id
        )
    return None


def authorize_route_lane_change(
    *,
    route_lane_change_allowed: bool,
    current_lane_id: int,
    route_required_lane_id: int,
    next_macro_maneuver: object,
    current_road_option: object,
    remaining_distance_m: Optional[float],
    available_lane_ids: Sequence[int],
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Mapping[int, Mapping[str, object]],
    preparation_start_distance_m: float,
    latest_start_distance_m: float,
    target_safety_threshold: float,
    require_adjacent: bool = True,
    explicit_lane_change_start_distance_m: Optional[float] = None,
    adjacent_lane_directions: Optional[Mapping[int, str]] = None,
    topology_current_lane_id: int = 0,
    topology_target_lane_id: int = 0,
    topology_lane_offset: int = 0,
    topology_target_in_local_frame: bool = True,
    route_geometry_direction: str = "",
    route_geometry_distance_m: Optional[float] = None,
) -> LaneChangeAuthorization:
    """Decide whether a route-required lane change is authorized *now*.

    A small forward state machine -- IDLE -> ELIGIBLE -> IN_WINDOW ->
    AUTHORIZED -- rather than a threshold gauntlet. The result feeds
    ``RouteLaneChangeAuthorizationLatch``, which is what keeps it stable once
    AUTHORIZED; this function only decides the *entry*.

    Direction comes from the strongest signal available, in order: the
    arc-length route model (``route_geometry_direction``), the rolling HD
    local frame's signed offset, an explicit adjacency map, else unknown.
    When the route model supplies a direction it is authoritative for
    existence and adjacency too -- the canonical/AD lane-id bookkeeping the
    legacy path leans on is unreliable across the CARLA->AD-map split (a
    small stable-tracker id vs opaque AD ids), so it is only consulted when
    the route model is silent.
    """
    # ------------------------------------------------------------- gate
    if not bool(route_lane_change_allowed):
        return _denied(
            "route_lane_change_not_allowed",
            next_macro_maneuver,
            remaining_distance_m,
            current_lane_id,
        )

    maneuver = normalize_route_maneuver(next_macro_maneuver)
    current_option = normalize_route_maneuver(current_road_option)
    current_lane_id = int(current_lane_id or 0)
    route_required_lane_id = int(route_required_lane_id or 0)

    geometry_direction = str(route_geometry_direction or "").strip().lower()
    if geometry_direction not in {"left", "right"}:
        geometry_direction = ""

    topology_requires_change = bool(
        int(topology_current_lane_id or 0) != 0
        and int(topology_target_lane_id or 0) != 0
        and int(topology_current_lane_id) != int(topology_target_lane_id)
        and int(topology_lane_offset or 0) != 0
    )
    adjacent_directions = {
        int(lane_id): str(direction).strip().lower()
        for lane_id, direction in dict(adjacent_lane_directions or {}).items()
        if str(direction).strip().lower() in {"left", "right"}
    }

    # --------------------------------------------- PHASE  IDLE -> ELIGIBLE
    # An active turn connector owns the lateral channel: no lane change.
    if current_option in _TURN_MANEUVERS:
        return _denied(
            "already_in_turn_connector", maneuver, remaining_distance_m, current_lane_id
        )

    if geometry_direction:
        # The route model saw a lane_change segment ahead. That is the
        # maneuver, whichever way ``next_macro`` reads.
        direction = geometry_direction
        active_maneuver = (
            RouteManeuver.LANE_CHANGE_LEFT
            if direction == "left"
            else RouteManeuver.LANE_CHANGE_RIGHT
        )
        reason_suffix = "_by_geometry"
        is_explicit_lane_change = True
        via_route_model = True
    else:
        if maneuver in _NO_LANE_CHANGE_MANEUVERS:
            return _denied(
                "route_maneuver_does_not_require_lane_change",
                maneuver,
                remaining_distance_m,
                current_lane_id,
            )
        active_maneuver = maneuver
        is_explicit_lane_change = maneuver in _LANE_CHANGE_MANEUVERS
        via_route_model = False
        if topology_requires_change and _resolve_offset_direction(topology_lane_offset):
            direction = _resolve_offset_direction(topology_lane_offset)
        else:
            direction = adjacent_directions.get(route_required_lane_id, "")
        reason_suffix = "_by_topology" if topology_requires_change else ""

    # A target the rolling HD frame cannot place is not one we can steer to.
    # The route model resolves this before anything else it does; the legacy
    # id path checks it only once it knows there is a required id at all.
    def _outside_local_frame(*, require_current_lane_known: bool) -> bool:
        return bool(
            (not require_current_lane_known or int(topology_current_lane_id or 0) != 0)
            and int(topology_target_lane_id or 0) != 0
            and int(topology_target_lane_id) != int(topology_current_lane_id or 0)
            and not bool(topology_target_in_local_frame)
        )

    # --------------------------------------- PHASE  resolve the target id
    # The id we act on and report. When the route model is driving, prefer a
    # value genuinely distinct from the current lane (the AD local-frame
    # target, else the canonical id) -- reporting the current lane makes the
    # commitment tracker complete the maneuver before any lateral motion.
    if via_route_model:
        if _outside_local_frame(require_current_lane_known=False):
            return _denied(
                "route_target_outside_local_frame",
                active_maneuver,
                route_geometry_distance_m,
                int(topology_target_lane_id),
            )
        target_lane_id = current_lane_id
        for candidate in (int(topology_target_lane_id or 0), route_required_lane_id):
            if candidate != 0 and candidate != current_lane_id:
                target_lane_id = candidate
                break
    else:
        if route_required_lane_id == 0:
            return _denied(
                "missing_required_lane_id", maneuver, remaining_distance_m, current_lane_id
            )
        if _outside_local_frame(require_current_lane_known=True):
            return _denied(
                "route_target_outside_local_frame",
                maneuver,
                remaining_distance_m,
                route_required_lane_id,
            )
        if route_required_lane_id == current_lane_id and not topology_requires_change:
            return LaneChangeAuthorization(
                allowed=False,
                direction=None,
                reason="already_in_required_lane",
                required_by_route=False,
                distance_to_maneuver_m=_finite_or_none(remaining_distance_m),
                target_lane_id=int(route_required_lane_id),
                maneuver=str(maneuver.value),
            )
        available = {
            _safe_int(lane_id)
            for lane_id in list(available_lane_ids or [])
            if _safe_int(lane_id)
        }
        if route_required_lane_id not in available:
            return _denied(
                "required_lane_not_available",
                maneuver,
                remaining_distance_m,
                route_required_lane_id,
            )
        target_lane_id = route_required_lane_id

    # ------------------------------------------- PHASE  direction / adjacency
    # The route model already proved the adjacent lane exists in ``direction``.
    if not via_route_model:
        if bool(require_adjacent) and direction not in {"left", "right"}:
            return _denied(
                "required_lane_not_adjacent", maneuver, remaining_distance_m, target_lane_id
            )
        _mismatch = {
            RouteManeuver.TURN_LEFT: ("left", "required_lane_direction_mismatch_left_turn"),
            RouteManeuver.TURN_RIGHT: ("right", "required_lane_direction_mismatch_right_turn"),
            RouteManeuver.LANE_CHANGE_LEFT: (
                "left", "required_lane_direction_mismatch_left_change"
            ),
            RouteManeuver.LANE_CHANGE_RIGHT: (
                "right", "required_lane_direction_mismatch_right_change"
            ),
        }.get(maneuver)
        if _mismatch is not None and direction != _mismatch[0]:
            return _denied(
                _mismatch[1], maneuver, remaining_distance_m, target_lane_id
            )

    # --------------------------------------- PHASE  ELIGIBLE -> IN_WINDOW
    distance = _finite_or_none(
        route_geometry_distance_m if via_route_model else remaining_distance_m
    )
    if distance is None and via_route_model:
        distance = _finite_or_none(remaining_distance_m)
    window_denial = _trigger_window_denial(
        maneuver=active_maneuver,
        is_explicit_lane_change=is_explicit_lane_change,
        distance=distance,
        explicit_lane_change_start_distance_m=explicit_lane_change_start_distance_m,
        preparation_start_distance_m=preparation_start_distance_m,
        latest_start_distance_m=latest_start_distance_m,
        target_lane_id=route_required_lane_id,
    )
    if window_denial is not None:
        return window_denial

    # --------------------------------------- PHASE  IN_WINDOW -> AUTHORIZED
    # Target-lane clearance. When the route model is driving, the safety-score
    # map may be keyed in a different id domain than the target we resolved;
    # a genuinely absent score is skipped rather than read as unsafe.
    if via_route_model:
        safety = lane_safety_scores.get(
            int(target_lane_id),
            lane_safety_scores.get(int(topology_target_lane_id or 0), None),
        )
    else:
        safety = lane_safety_scores.get(int(target_lane_id), 0.0)
    if safety is not None and float(safety) < float(target_safety_threshold):
        return _denied(
            "target_lane_safety_below_threshold", active_maneuver, distance, target_lane_id
        )
    if bool(dict(lane_prediction_risks.get(int(target_lane_id), {}) or {}).get("risk", False)):
        return _denied(
            "target_lane_prediction_risk", active_maneuver, distance, target_lane_id
        )

    # ------------------------------------------------------- AUTHORIZED
    return LaneChangeAuthorization(
        allowed=True,
        direction=str(direction) if direction in {"left", "right"} else str(geometry_direction),
        reason="route_lane_change_authorized" + reason_suffix,
        required_by_route=True,
        distance_to_maneuver_m=distance,
        target_lane_id=int(target_lane_id),
        maneuver=str(active_maneuver.value),
    )


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _direction_for_delta(lane_delta: int) -> str:
    # The project treats increasing canonical lane id as a leftward move in the
    # OpenCDA debug traces used for the integration scenarios.
    return "left" if int(lane_delta) > 0 else "right"


def _denied(
    reason: str,
    maneuver: object,
    distance: Optional[float],
    target_lane_id: int,
) -> LaneChangeAuthorization:
    normalized = normalize_route_maneuver(maneuver)
    return LaneChangeAuthorization(
        allowed=False,
        direction=None,
        reason=str(reason),
        required_by_route=False,
        distance_to_maneuver_m=_finite_or_none(distance),
        target_lane_id=int(target_lane_id or 0),
        maneuver=str(normalized.value),
    )


def _finite_or_none(value: object) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return float(number)
