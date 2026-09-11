"""Fusion: chain Stage A (classify) -> Stage B (assign) -> Stage C (corridor)
into one per-tick call.

    resolution = resolve_conflicts(
        reference_samples=lane_center_reference,
        ego_snapshot={... "x","y","v","psi"},
        my_claim=ResourceClaim(...) or None,
        my_actor_id=<ego id>,
        obstacle_snapshots=[...],          # non-connected road users
        cav_intents=[CavIntent, ...],    # from collect_cav_intents(...)
        latch_state=<prev tick's>,
    )
    # -> resolution.corridor (Stage C), .assignments (Stage B),
    #    .tags (Stage A), .latch_state (feed back next tick), .diagnostics

Pure. The bridge is expected to: build ``cav_intents`` from its CP
payload, pass its own ``ResourceClaim`` when it is mid-maneuver (else
None), persist ``latch_state``, and feed ``resolution.corridor`` to the
MPC constraint builder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from opencda.planning_module.pipeline.conflict_classifier import (
    IGNORE,
    ClassifierParams,
    ConflictTag,
    classify_conflicts,
)
from opencda.planning_module.pipeline.cooperative_arbitration import (
    ArbitrationLatchEntry,
    ConflictAssignment,
    CavIntent,
    ResourceClaim,
    assign_conflict_roles,
)
from opencda.planning_module.pipeline.spatiotemporal_corridor import (
    Corridor,
    CorridorParams,
    aggregate_mode_corridors,
    build_longitudinal_corridor,
    retain_pending_corridor,
)
from opencda.planning_module.pipeline.prediction_modes import as_modes, single_mode
from opencda.planning_module.pipeline.mpc_obstacle_relevance import (
    _polyline_xy,
    project_to_extended_polyline,
)
from opencda.planning_module.pipeline.rss import RSSParams


@dataclass
class ConflictResolution:
    tags: List[ConflictTag]
    assignments: List[ConflictAssignment]
    corridor: Corridor
    latch_state: Dict[str, ArbitrationLatchEntry]
    tag_state: Dict[str, str] = field(default_factory=dict)
    veto_state: Dict[str, Any] = field(default_factory=dict)
    mpc_rows: List[Any] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _cav_to_agent_snapshot(cav: CavIntent) -> dict:
    track = [
        {"x": float(x), "y": float(y), "t": float(t), "v": float(v)}
        for (t, x, y, v) in cav.planned_path
    ]
    snapshot = {
        "id": int(cav.actor_id),
        "vehicle_id": int(cav.actor_id),
        "x": float(cav.position_xy[0]),
        "y": float(cav.position_xy[1]),
        "v": float(cav.speed_mps),
        "psi": float(cav.heading_rad),
        "cooperative": bool(cav.cooperative),
        "predicted_trajectory": track,
        "trajectory_source": "broadcast" if track else "current_pose",
    }
    if track:
        # One mode today (the committed broadcast plan). A real multi-modal
        # predictor drops in here as extra PredictedMode entries.
        snapshot["predicted_modes"] = list(single_mode(
            track, probability=max(0.0, min(1.0, float(cav.probability)))
        ))
    return snapshot


def _agent_id(agent: Mapping[str, Any]) -> str:
    return str(agent.get(
        "id",
        agent.get("vehicle_id", agent.get("track_id", agent.get("actor_id", ""))),
    ))


def _prediction_evidence(agent: Mapping[str, Any]) -> dict:
    modes = as_modes(agent.get("predicted_modes"))
    path = list(modes[0].path) if modes else list(
        agent.get("predicted_trajectory", ()) or ()
    )
    if not path:
        return {"sample_count": 0}
    last = path[-1]
    if isinstance(last, Mapping):
        return {
            "sample_count": len(path),
            "end_x": float(last.get("x", last.get("x_m", 0.0)) or 0.0),
            "end_y": float(last.get("y", last.get("y_m", 0.0)) or 0.0),
        }
    return {"sample_count": len(path)}


def _claim_diagnostics(claim: Optional[ResourceClaim]) -> dict:
    """Return the complete, JSON-safe Stage-B claim contract."""

    if claim is None:
        return {"participates": False, "phase": "none"}
    return {
        "participates": bool(claim.participates),
        "phase": str(claim.phase),
        "kind": str(claim.kind),
        "resource_id": str(claim.resource_id),
        "source_corridor_id": int(claim.source_corridor_id),
        "target_corridor_id": int(claim.target_corridor_id),
        "station_corridor_id": int(claim.station_corridor_id),
        "s_begin_m": claim.s_begin_m,
        "s_end_m": claim.s_end_m,
        "committed_at_s": float(claim.committed_at_s),
    }


def _apply_veto_hysteresis(
    *, mode_id, raw_dangerous, recovered, veto_state, release_ticks,
):
    """Asymmetric latch for the credible-mode veto: engage immediately on a
    real danger, release only after ``release_ticks`` consecutive clear
    evaluations AND the risk has comfortably receded.

    Without this the per-tick ``dangerous`` flag flips 0<->1 as the
    oscillating ego speed nudges ``first_risk_stage`` across the TTC
    threshold, which churns ``credible_mode_veto_count`` -> the MPC
    constraint-revision hash -> a forced replan every other tick -> a
    speed limit cycle (see fig2 POST-scheduler)."""

    prev = dict((veto_state or {}).get(str(mode_id), {}) or {})
    clear_streak = int(prev.get("clear_streak", 0))
    was_dangerous = bool(prev.get("dangerous", False))
    held = False
    if raw_dangerous:
        dangerous, clear_streak = True, 0
    elif was_dangerous:
        clear_streak += 1
        if clear_streak >= max(1, int(release_ticks)) and bool(recovered):
            dangerous, clear_streak = False, int(release_ticks)
        else:
            dangerous, held = True, True
    else:
        dangerous = False
    return dangerous, held, {
        "dangerous": bool(dangerous), "clear_streak": int(clear_streak),
    }


def _build_effective_corridor(
    *, reference_samples, ego_snapshot, all_agents, mode_groups,
    tag_by_id, assign_by_id, corridor_params, rss_params,
    credible_mode_probability_min, credible_mode_ttc_s,
    nominal_progress_limit_m=None,
    veto_state=None, veto_release_ticks=12, veto_release_margin_s=0.5,
):
    items = []
    for agent in all_agents:
        aid = _agent_id(agent)
        if "::mode" in aid:
            continue
        tag = tag_by_id.get(aid)
        if tag is not None and tag.tag != IGNORE:
            items.append((agent, tag, assign_by_id.get(aid)))
    corridor = build_longitudinal_corridor(
        reference_samples, ego_snapshot, items, corridor_params, rss_params
    )
    poly = _polyline_xy(reference_samples)
    ego_s0 = (
        project_to_extended_polyline(
            float(ego_snapshot.get("x", ego_snapshot.get("x_m", 0.0))),
            float(ego_snapshot.get("y", ego_snapshot.get("y_m", 0.0))), poly,
        )[1]
        if len(poly) >= 2 else 0.0
    )
    ego_v = max(0.0, float(ego_snapshot.get(
        "v", ego_snapshot.get("speed_mps", ego_snapshot.get("speed", 0.0))
    )))
    progress_limit = None
    if nominal_progress_limit_m is not None:
        try:
            candidate_limit = float(nominal_progress_limit_m)
            if math.isfinite(candidate_limit) and candidate_limit >= 0.0:
                progress_limit = candidate_limit
        except (TypeError, ValueError):
            progress_limit = None
    nominal_s = []
    for k in range(len(corridor.s_hi)):
        progress_m = ego_v * float(corridor_params.dt_s) * k
        if progress_limit is not None:
            progress_m = min(progress_m, progress_limit)
        nominal_s.append(ego_s0 + progress_m)
    dt_s = float(corridor_params.dt_s)
    credible_veto_count = 0
    held_veto_count = 0
    next_veto_state: Dict[str, Any] = {}
    for actor_id, group in mode_groups.items():
        mode_corridors = []
        for mode_agent, probability in group:
            mode_id = _agent_id(mode_agent)
            tag = tag_by_id.get(mode_id)
            if tag is None:
                continue
            mode_items = [] if tag.tag == IGNORE else [(mode_agent, tag, None)]
            mode_corridor = build_longitudinal_corridor(
                reference_samples, ego_snapshot, mode_items,
                corridor_params, rss_params, constrain_follow=True,
            )
            first_risk_stage = next((
                k for k, cap in enumerate(mode_corridor.s_hi)
                if k < len(nominal_s) and float(cap) < float(nominal_s[k])
            ), None)
            risk_t_s = (
                float("inf") if first_risk_stage is None
                else float(first_risk_stage) * dt_s
            )
            raw_dangerous = bool(
                float(probability) >= float(credible_mode_probability_min)
                and tag.tag != IGNORE
                and risk_t_s <= float(credible_mode_ttc_s)
            )
            recovered = (
                first_risk_stage is None
                or risk_t_s > float(credible_mode_ttc_s)
                + max(0.0, float(veto_release_margin_s))
            )
            dangerous, held, entry = _apply_veto_hysteresis(
                mode_id=mode_id, raw_dangerous=raw_dangerous,
                recovered=recovered, veto_state=veto_state,
                release_ticks=veto_release_ticks,
            )
            next_veto_state[str(mode_id)] = entry
            credible_veto_count += int(dangerous)
            held_veto_count += int(held)
            mode_corridors.append((mode_corridor, float(probability), dangerous, mode_id))
        aggregate = aggregate_mode_corridors(mode_corridors, nominal_s)
        for k, cap in enumerate(aggregate.s_hi):
            if cap < corridor.s_hi[k]:
                corridor.s_hi[k] = cap
                corridor.binding[k] = aggregate.binding[k] or actor_id
    corridor.clamp_and_check()
    return corridor, credible_veto_count, held_veto_count, next_veto_state


def resolve_conflicts(
    *,
    reference_samples: Sequence[Any],
    ego_snapshot: Mapping[str, Any],
    my_actor_id: int,
    my_claim: Optional[ResourceClaim] = None,
    obstacle_snapshots: Sequence[Mapping[str, Any]] = (),
    cav_intents: Sequence[CavIntent] = (),
    prediction_modes: Optional[Mapping[str, Sequence[Any]]] = None,
    latch_state: Optional[Mapping[str, ArbitrationLatchEntry]] = None,
    tag_state: Optional[Mapping[str, str]] = None,
    classifier_params: ClassifierParams = ClassifierParams(),
    corridor_params: CorridorParams = CorridorParams(),
    rss_params: RSSParams = RSSParams(),
    hysteresis_ticks: int = 3,
    mode_probability_floor: float = 0.05,
    credible_mode_probability_min: float = 0.15,
    credible_mode_ttc_s: float = 2.0,
    credible_mode_veto_release_ticks: int = 12,
    credible_mode_veto_release_margin_s: float = 0.5,
    nominal_progress_limit_m: Optional[float] = None,
    veto_state: Optional[Mapping[str, Any]] = None,
    refresh_assignments: bool = True,
    cached_assignments: Sequence[ConflictAssignment] = (),
    rebuild_corridor: bool = True,
    cached_corridor: Optional[Corridor] = None,
) -> ConflictResolution:
    cavs = list(cav_intents or [])
    # Only a peer carrying an actual shared plan owns future-trajectory data.
    # A pose/claim-only intent remains available to Stage B but must not erase
    # the prediction module's hypotheses for the same physical actor.
    shared_plan_cavs = [c for c in cavs if c.planned_path]
    cav_agents = [_cav_to_agent_snapshot(c) for c in shared_plan_cavs]
    # A connected vehicle normally also appears in perception.  Its shared
    # trajectory is the richer representation, so replace (rather than add
    # to) the perception track for the same actor.
    cav_ids = {str(c.actor_id) for c in shared_plan_cavs}
    raw_obstacles = list(obstacle_snapshots or [])
    modes_by_id = dict(prediction_modes or {})
    perception_agents: List[Mapping[str, Any]] = []
    for a in raw_obstacles:
        aid = _agent_id(a)
        if aid in cav_ids:
            continue
        # Fusion priority: a fresh broadcast plan (handled above as a CAV
        # agent) wins; every other road user gets the prediction module's
        # trajectory here, as a length-1 mode list today.
        if "predicted_modes" not in a and aid in modes_by_id:
            a = {**a, "predicted_modes": list(as_modes(modes_by_id[aid]))}
            a.setdefault("trajectory_source", "prediction")
        perception_agents.append(a)
    physical_agents: List[Mapping[str, Any]] = perception_agents + cav_agents
    all_agents: List[Mapping[str, Any]] = []
    mode_groups: Dict[str, List[Tuple[Mapping[str, Any], float]]] = {}
    retained_mode_count = 0
    for agent in physical_agents:
        modes = as_modes(agent.get("predicted_modes"))
        retained = [
            mode for mode in modes
            if float(mode.probability) >= max(0.0, float(mode_probability_floor))
        ]
        if len(retained) == 1:
            # Stage A consumes ``predicted_trajectory``. Do not silently drop
            # the prediction module's only hypothesis merely because no
            # probabilistic aggregation is needed.
            mode = retained[0]
            agent = dict(agent)
            agent.pop("predicted_modes", None)
            agent["predicted_trajectory"] = list(mode.path)
            agent["mode_probability"] = float(mode.probability)
            all_agents.append(agent)
            retained_mode_count += 1
            continue
        if not retained:
            all_agents.append(agent)
            continue
        actor_id = _agent_id(agent)
        mode_groups[actor_id] = []
        for index, mode in enumerate(retained):
            expanded = dict(agent)
            expanded.pop("predicted_modes", None)
            expanded["predicted_trajectory"] = list(mode.path)
            expanded["id"] = "%s::mode%d" % (actor_id, int(index))
            expanded["physical_actor_id"] = actor_id
            expanded["mode_probability"] = float(mode.probability)
            all_agents.append(expanded)
            mode_groups[actor_id].append((expanded, float(mode.probability)))
            retained_mode_count += 1

    # Stage A -----------------------------------------------------------------
    tags = classify_conflicts(
        reference_samples, ego_snapshot, all_agents, classifier_params,
        previous_tags=tag_state,
    )
    tag_by_id = {t.agent_id: t for t in tags}

    # Stage B (only cooperative cavs, only when ego holds an active claim) ---
    assignments: List[ConflictAssignment] = []
    arbitration_diagnostics: Dict[str, Any] = {}
    new_latch: Dict[str, ArbitrationLatchEntry] = dict(latch_state or {})
    current_tag_state = {tag.agent_id: tag.tag for tag in tags}
    tag_changed = bool(
        tag_state is not None and current_tag_state != dict(tag_state or {})
    )
    roles_refreshed = bool(refresh_assignments or tag_changed)
    if not roles_refreshed:
        live_ids = {int(c.actor_id) for c in cavs if bool(c.cooperative)}
        assignments = [
            assignment for assignment in cached_assignments or ()
            if int(assignment.cav_actor_id) in live_ids
        ]
    elif my_claim is not None and bool(my_claim.participates):
        conflicting_cavs = []
        for c in cavs:
            if not bool(c.cooperative):
                continue
            t = tag_by_id.get(str(c.actor_id))
            if t is None or t.tag == IGNORE:
                continue
            conflicting_cavs.append(c)
        assignments, new_latch = assign_conflict_roles(
            my_claim=my_claim,
            my_actor_id=int(my_actor_id),
            my_position_xy=(
                float(ego_snapshot.get("x", ego_snapshot.get("x_m", 0.0))),
                float(ego_snapshot.get("y", ego_snapshot.get("y_m", 0.0))),
            ),
            my_heading_rad=float(
                ego_snapshot.get("psi", ego_snapshot.get("heading_rad", 0.0))
            ),
            cavs=conflicting_cavs,
            latch_state=latch_state,
            hysteresis_ticks=int(hysteresis_ticks),
            diagnostics=arbitration_diagnostics,
        )
    elif roles_refreshed:
        arbitration_diagnostics["reason"] = "ego_claim_inactive"
    assign_by_id = {str(a.cav_actor_id): a for a in assignments}

    # Stage C ---------------------------------------------------------------
    corridor_rebuilt = bool(
        rebuild_corridor or tag_changed or roles_refreshed
    )
    incoming_veto_state = dict(veto_state or {})
    held_veto_count = 0
    if corridor_rebuilt or cached_corridor is None:
        (corridor, credible_veto_count, held_veto_count,
         new_veto_state) = _build_effective_corridor(
            reference_samples=reference_samples, ego_snapshot=ego_snapshot,
            all_agents=all_agents, mode_groups=mode_groups,
            tag_by_id=tag_by_id, assign_by_id=assign_by_id,
            corridor_params=corridor_params, rss_params=rss_params,
            credible_mode_probability_min=credible_mode_probability_min,
            credible_mode_ttc_s=credible_mode_ttc_s,
            nominal_progress_limit_m=nominal_progress_limit_m,
            veto_state=incoming_veto_state,
            veto_release_ticks=int(credible_mode_veto_release_ticks),
            veto_release_margin_s=float(credible_mode_veto_release_margin_s),
        )
        if cached_corridor is not None and not tag_changed:
            corridor = retain_pending_corridor(corridor, cached_corridor)
        corridor_rebuilt = True
    else:
        corridor = cached_corridor
        credible_veto_count = 0
        # Carry the veto latch unchanged while the cache is reused so a
        # later rebuild resumes from the last real state.
        new_veto_state = incoming_veto_state

    def _source(agent: Mapping[str, Any]) -> str:
        src = str(agent.get("trajectory_source", "") or "")
        if src:
            return src
        if as_modes(agent.get("predicted_modes")):
            return "prediction"
        if agent.get("predicted_trajectory") or agent.get("future_trajectory"):
            return "prediction"
        return "current_pose"

    source_counts: Dict[str, int] = {}
    for agent in all_agents:
        source_counts[_source(agent)] = source_counts.get(_source(agent), 0) + 1

    diagnostics = {
        "conflict_agent_count": len(physical_agents),
        "mode_conflict_count": len(all_agents),
        "deduplicated_agent_count": (
            len(raw_obstacles) + len(cav_agents) - len(physical_agents)
        ),
        "cav_count": len(cavs),
        "shared_plan_cav_count": sum(1 for c in cavs if c.planned_path),
        "shared_plan_sample_count": sum(len(c.planned_path) for c in cavs),
        "multimodal_agent_count": sum(
            1 for a in physical_agents if len(as_modes(a.get("predicted_modes"))) > 1
        ),
        "retained_prediction_mode_count": int(retained_mode_count),
        "credible_mode_veto_count": int(credible_veto_count),
        "credible_mode_veto_held_count": int(held_veto_count),
        "trajectory_source_counts": source_counts,
        "ego_claim_phase": (
            "none" if my_claim is None else str(my_claim.phase)
        ),
        "peer_claim_phases": {
            str(c.actor_id): str(c.claim.phase) for c in cavs
        },
        "ego_claim": _claim_diagnostics(my_claim),
        "peer_claims": {
            str(c.actor_id): _claim_diagnostics(c.claim) for c in cavs
        },
        "arbitration": arbitration_diagnostics,
        "non_ignore_count": sum(1 for t in tags if t.tag != IGNORE),
        "speed_owned_follow_count": sum(
            1 for t in tags if t.tag in ("FOLLOW", "LEAD_BRAKE")
        ),
        "tags": {t.agent_id: t.tag for t in tags},
        "tag_reasons": {t.agent_id: t.reason for t in tags},
        "agent_states": {
            _agent_id(agent): {
                "x": float(agent.get("x", agent.get("x_m", 0.0)) or 0.0),
                "y": float(agent.get("y", agent.get("y_m", 0.0)) or 0.0),
                "v": float(agent.get("v", agent.get("speed_mps", 0.0)) or 0.0),
                "psi": float(agent.get("psi", agent.get("heading_rad", 0.0)) or 0.0),
                "trajectory_source": _source(agent),
                "mode_count": len(as_modes(agent.get("predicted_modes"))),
                "prediction": _prediction_evidence(agent),
            }
            for agent in physical_agents
        },
        "roles": {str(a.cav_actor_id): a.role for a in assignments},
        "corridor_feasible": bool(corridor.feasible),
        "corridor_first_infeasible_stage": corridor.first_infeasible_stage,
        "corridor_binding": [b for b in corridor.binding if b],
        "coordination_roles_refreshed": bool(roles_refreshed),
        "coordination_refresh_reason": (
            "conflict_tag_changed" if tag_changed
            else "scheduled_refresh" if refresh_assignments
            else "cached_roles"
        ),
        "corridor_rebuilt": bool(corridor_rebuilt),
    }
    return ConflictResolution(
        tags=tags, assignments=assignments, corridor=corridor,
        latch_state=new_latch,
        tag_state=current_tag_state,
        veto_state=dict(new_veto_state or {}),
        diagnostics=diagnostics,
    )
