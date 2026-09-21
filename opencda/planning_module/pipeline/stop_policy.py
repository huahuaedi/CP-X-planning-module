"""When a stop must be an emergency (full braking, wheels straight).

The MPC-path emergency decision lives here and only here.  Three things make
it one: a hard gate that names a stop hazard, the ``emergency_brake``
maneuver, and a corridor-infeasibility escalation.  Anything else that vetoes
the reference (a geometry or continuity contract failure) is a bounded
degradation, not an emergency.

Not in this module: the SafetySupervisor's hazard filter (collision, ran red
light, off-road, stuck from OpenCDA's safety manager), which is a separate,
later gate, and the last-resort stop when the pipeline itself raises, whose
policy is ``pipeline_failure_action`` below.
"""

from __future__ import annotations


def hard_gate_requires_emergency_stop(
    *,
    fallback_reason: str,
    behavior_decision: str,
    stop_goal_active: bool,
) -> bool:
    """Reserve full braking for hard gates that represent a stop hazard.

    A geometry/continuity contract veto means MPC must not consume that
    reference, but it is not evidence of an imminent collision.  Those
    failures use the bounded tracking fallback and remain subject to the
    downstream safety supervisor.  Collision, explicit stop, and emergency
    behavior retain deterministic full braking.
    """

    reason = str(fallback_reason or "").strip().lower()
    decision = str(behavior_decision or "").strip().lower()
    if not reason.startswith("candidate_hard_gate:"):
        return False
    if bool(stop_goal_active) or decision in {
        "emergency_brake",
        "stop_at_intersection",
        "stop_sign",
    }:
        return True
    hazard_tokens = (
        "collision_risk",
        "emergency_brake_direct_control",
        "stop_missing_target_hard_lock",
        # A geometry/continuity veto with nothing to fall back to is not
        # automatically harmless just because it isn't an explicit
        # collision-risk token -- confirmed on a real intersection turn
        # (MDrive Intersection_Deadlock_Resolution/3): the reference
        # pipeline hard-gated with "empty_reference;turn_swept_footprint:
        # no_corridor_geometry" at the turn exit, the bounded-tracking
        # fallback below applied throttle=0.2-0.23/brake=0.0 the whole
        # window with the reference still empty, and the ego collided with
        # unmodeled static scene geometry moments later -- then, still
        # inside this same hard-gate window, the post-impact speed drop
        # read as a large speed deficit against normal cruise and the
        # bounded-tracking path answered with full throttle (brake stayed
        # 0.0 throughout). "No usable reference at all" is at least as much
        # a "do not know it's safe to keep moving" case as the
        # stop-missing-target lock above; treat it the same way.
        "empty_reference",
        "no_corridor_geometry",
        "too_few_forward_samples",
    )
    return any(token in reason for token in hazard_tokens)


def emergency_stop_reason(
    *,
    fallback_reason: str,
    behavior_decision: str,
    stop_goal_active: bool,
    corridor_infeasible_escalate: bool,
) -> str:
    """Why this tick is an emergency, or ``""`` when it is not."""

    if hard_gate_requires_emergency_stop(
        fallback_reason=fallback_reason,
        behavior_decision=behavior_decision,
        stop_goal_active=stop_goal_active,
    ):
        return "hard_gate_stop_hazard"
    if str(behavior_decision or "").strip().lower() == "emergency_brake":
        return "emergency_brake_maneuver"
    if bool(corridor_infeasible_escalate):
        return "corridor_infeasible_escalation"
    return ""


def pipeline_failure_action(fallback_policy: str) -> str:
    """What to do when the planning pipeline raises: ``"raise"`` or ``"emergency_stop"``.

    ``"opencda"`` is listed for parity with the old inline check, but the
    bridge rewrites that policy to ``"emergency_stop"`` before it can get here.
    """

    if str(fallback_policy) in {"raise", "opencda"}:
        return "raise"
    return "emergency_stop"
