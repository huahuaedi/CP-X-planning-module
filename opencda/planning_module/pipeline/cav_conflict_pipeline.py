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
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from opencda.planning_module.pipeline.conflict_classifier import (
    CROSSING,
    CUT_IN,
    FOLLOW,
    IGNORE,
    LEAD_BRAKE,
    MERGE,
    ONCOMING,
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
    remove_actor_bounds,
)
from opencda.planning_module.pipeline.speed_planner import SpeedConstraint
from opencda.planning_module.pipeline.prediction_modes import as_modes, single_mode
from opencda.planning_module.pipeline.mpc_obstacle_relevance import (
    _polyline_xy,
    project_to_extended_polyline,
)
from opencda.planning_module.pipeline.rss import RSSParams
from opencda.planning_module.pipeline.reference_geometry import pose_at_arc


@dataclass
class ConflictResolution:
    tags: List[ConflictTag]
    assignments: List[ConflictAssignment]
    corridor: Corridor
    latch_state: Dict[str, ArbitrationLatchEntry]
    tag_state: Dict[str, str] = field(default_factory=dict)
    veto_state: Dict[str, Any] = field(default_factory=dict)
    # Fresh Stage-A evidence that permanently retires these actors' cached
    # bounds.  The schedule applies this to its original-time cache; doing it
    # only to the per-tick rebased copy lets the old constraint reappear on
    # the next tick.
    released_actor_ids: Tuple[str, ...] = ()
    speed_constraint: Optional[SpeedConstraint] = None
    # Stage C owns ``corridor`` in its classification-reference station
    # coordinates. Stage D owns this rebased view in the coordinates of the
    # reference that the MPC actually executes.
    constraint_corridor: Optional[Corridor] = None
    # Only a freshly computed Stage-C corridor may reset the cache's forecast
    # origin. ``corridor`` can also contain older, still-pending bounds.
    fresh_corridor: Optional[Corridor] = None
    mpc_rows: List[Any] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _cav_to_agent_snapshot(cav: CavIntent) -> dict:
    track = [
        {"x": float(x), "y": float(y), "t": float(t), "v": float(v)}
        for (t, x, y, v) in cav.planned_path
    ]
    # conflict_classifier's LEAD_BRAKE tag reads an agent's "a"/"acceleration"
    # /"acceleration_mps2" field (default 0.0 -- see _f); a broadcasting CAV
    # never carried one, so LEAD_BRAKE could not fire against another CAV no
    # matter how hard it actually braked (confirmed on a real scripted-brake
    # run: cav_conflict_tags stayed FOLLOW for the whole encounter). The
    # broadcast plan's own near-term samples are the CAV's committed intent,
    # a more reliable early signal than a raw instantaneous accelerometer
    # reading, and need no change to the broadcast message schema.
    acceleration_mps2 = 0.0
    if len(track) >= 2:
        dt_s = float(track[1]["t"]) - float(track[0]["t"])
        if dt_s > 1.0e-3:
            acceleration_mps2 = (
                float(track[1]["v"]) - float(track[0]["v"])
            ) / dt_s
    snapshot = {
        "id": int(cav.actor_id),
        "vehicle_id": int(cav.actor_id),
        "x": float(cav.position_xy[0]),
        "y": float(cav.position_xy[1]),
        "v": float(cav.speed_mps),
        "psi": float(cav.heading_rad),
        "length_m": float(cav.length_m),
        "width_m": float(cav.width_m),
        "acceleration_mps2": float(acceleration_mps2),
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


# Lower rank = kept preferentially by _cap_agents_by_severity. CROSSING/
# ONCOMING and MERGE/CUT_IN are genuine lateral-path conflicts; FOLLOW/
# LEAD_BRAKE are SpeedPlanner/IDM's nominal concern (build_longitudinal_
# corridor does not constrain on them by default -- see its own FOLLOW/
# LEAD_BRAKE branch), so they must never outrank a real crossing conflict
# merely because their trivial "closest approach" time is small.
_TAG_SEVERITY_RANK: Dict[str, int] = {
    CROSSING: 0, ONCOMING: 0,
    MERGE: 1, CUT_IN: 1,
    LEAD_BRAKE: 2, FOLLOW: 2,
    IGNORE: 3,
}


def _physical_agent_id(agent: Mapping[str, Any]) -> str:
    return str(agent.get("physical_actor_id", _agent_id(agent)))


def _physical_tag_id(tag: ConflictTag) -> str:
    return str(tag.agent_id).split("::mode", 1)[0]


def _cap_agents_by_severity(
    *,
    all_agents: Sequence[Mapping[str, Any]],
    tags: Sequence[ConflictTag],
    mode_groups: Mapping[str, List[Tuple[Mapping[str, Any], float]]],
    max_agents: int,
) -> Tuple[
    List[Mapping[str, Any]], List[ConflictTag],
    Dict[str, List[Tuple[Mapping[str, Any], float]]], int,
]:
    """Cap how many physical agents reach Stage B/C this tick, by severity.

    Runs *after* classify_conflicts, not before: ranking by Stage A's own
    tag (computed from each agent's full predicted track) means a
    currently-far-but-about-to-cross vehicle is never mistaken for
    irrelevant the way a pre-classification position filter mistook it
    (a crossing vehicle still on its own street has a large *current*
    lateral offset from ego's path, even seconds from actually crossing it).

    Ranked primarily by tag class, not conflict_t_s alone: CROSSING/ONCOMING
    (genuine lateral path conflicts -- Stage C's homotopy keepout exists
    specifically for these) and MERGE/CUT_IN outrank FOLLOW/LEAD_BRAKE,
    which corridor construction itself treats as SpeedPlanner/IDM's nominal
    concern, not a half-space it constrains on by default (see
    build_longitudinal_corridor's own FOLLOW/LEAD_BRAKE branch). Using raw
    conflict_t_s alone would rank a same-speed, constant-safe-gap FOLLOW
    agent (whose closest approach is trivially "now", t=0, precisely because
    it never gets closer) as more urgent than a real crossing conflict
    still several seconds out -- exactly backwards. Within a tag class,
    conflict_t_s (soonest first) breaks ties. IGNORE/untagged agents rank
    last: Stage A already decided they are not a physical conflict, so the
    budget only ever trims what Stage A itself found least urgent, never a
    classified conflict in favor of a merely-nearby one. Returns
    (kept_agents, kept_tags, kept_mode_groups, dropped_physical_agent_count).
    """

    physical_ids: List[str] = []
    seen = set()
    for agent in all_agents:
        pid = _physical_agent_id(agent)
        if pid not in seen:
            seen.add(pid)
            physical_ids.append(pid)
    if max_agents <= 0 or len(physical_ids) <= max_agents:
        return list(all_agents), list(tags), dict(mode_groups), 0

    best_severity: Dict[str, Tuple[int, float]] = {
        pid: (_TAG_SEVERITY_RANK[IGNORE], float("inf")) for pid in physical_ids
    }
    for tag in tags:
        pid = _physical_tag_id(tag)
        if pid not in best_severity:
            continue
        candidate = (
            _TAG_SEVERITY_RANK.get(tag.tag, _TAG_SEVERITY_RANK[IGNORE]),
            float("inf") if tag.conflict_t_s is None else float(tag.conflict_t_s),
        )
        if candidate < best_severity[pid]:
            best_severity[pid] = candidate
    # Stable sort: ties (typically every agent Stage A left at "no conflict")
    # keep perception order rather than an arbitrary reshuffle.
    ranked = sorted(physical_ids, key=lambda pid: best_severity[pid])
    kept = set(ranked[:max_agents])

    return (
        [agent for agent in all_agents if _physical_agent_id(agent) in kept],
        [tag for tag in tags if _physical_tag_id(tag) in kept],
        {pid: group for pid, group in mode_groups.items() if pid in kept},
        len(physical_ids) - len(kept),
    )


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


def _mode_diagnostics(
    mode_groups: Mapping[str, List[Tuple[Mapping[str, Any], float]]],
    tag_by_id: Mapping[str, ConflictTag],
) -> Dict[str, Dict[str, Any]]:
    """Describe retained hypotheses at the Stage A/C ownership boundary."""

    diagnostics: Dict[str, Dict[str, Any]] = {}
    for actor_id, group in mode_groups.items():
        for mode_agent, probability in group:
            mode_id = _agent_id(mode_agent)
            tag = tag_by_id.get(mode_id)
            evidence = _prediction_evidence(mode_agent)
            diagnostics[str(mode_id)] = {
                "physical_actor_id": str(actor_id),
                "probability": float(probability),
                "tag": "" if tag is None else str(tag.tag),
                "tag_reason": "" if tag is None else str(tag.reason),
                "conflict_s_m": (
                    None if tag is None or tag.conflict_s_m is None
                    else float(tag.conflict_s_m)
                ),
                "conflict_t_s": (
                    None if tag is None or tag.conflict_t_s is None
                    else float(tag.conflict_t_s)
                ),
                "sample_count": int(evidence.get("sample_count", 0) or 0),
                "end_x": evidence.get("end_x"),
                "end_y": evidence.get("end_y"),
                "included_in_expected_corridor": True,
                "raw_dangerous": False,
                "credible_veto_active": False,
                "credible_veto_held": False,
                "first_risk_stage": None,
                "first_risk_time_s": None,
                "active_corridor_stage_count": 0,
                "veto_binding_stage_count": 0,
            }
    return diagnostics


def _tags_on_corridor_reference(
    tags: Sequence[ConflictTag],
    source_reference: Sequence[Any],
    corridor_reference: Sequence[Any],
) -> Dict[str, ConflictTag]:
    """Express Stage-A conflict stations on Stage C's reference.

    Classification may intentionally use a proposed maneuver reference, but
    longitudinal ordering must use the path the ego is actually executing.
    Convert the one station-valued field through world XY; all semantic tag
    data and timing remain owned by Stage A.
    """

    source_poly = _polyline_xy(source_reference)
    corridor_poly = _polyline_xy(corridor_reference)
    out: Dict[str, ConflictTag] = {}
    for tag in tags:
        converted = tag
        if (
            tag.conflict_s_m is not None
            and len(source_poly) >= 2
            and len(corridor_poly) >= 2
        ):
            x_m, y_m, _ = pose_at_arc(
                source_poly, float(tag.conflict_s_m)
            )
            corridor_s_m = project_to_extended_polyline(
                float(x_m), float(y_m), corridor_poly
            )[1]
            converted = replace(tag, conflict_s_m=float(corridor_s_m))
        out[str(converted.agent_id)] = converted
    return out


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
    *, mode_id, raw_dangerous, recovered, veto_state,
    sim_time_s, release_duration_s,
):
    """Asymmetric latch for the credible-mode veto: engage immediately on a
    real danger, release only after the risk has remained comfortably clear
    for ``release_duration_s`` of simulation time.

    Time is part of this contract deliberately.  Counting evaluations made
    the hold duration depend on whether Stage C was rebuilt at the normal
    5 Hz cadence or additionally by 20 Hz Stage-A tag transitions.  Without
    this latch the per-tick ``dangerous`` flag flips 0<->1 as the
    oscillating ego speed nudges ``first_risk_stage`` across the TTC
    threshold, which churns ``credible_mode_veto_count`` -> the MPC
    constraint-revision hash -> a forced replan every other tick -> a
    speed limit cycle (see fig2 POST-scheduler)."""

    prev = dict((veto_state or {}).get(str(mode_id), {}) or {})
    was_dangerous = bool(prev.get("dangerous", False))
    now_s = float(sim_time_s)
    release_s = max(0.0, float(release_duration_s))
    clear_since_s = prev.get("clear_since_s")
    try:
        clear_since_s = (
            None if clear_since_s is None else float(clear_since_s)
        )
    except (TypeError, ValueError):
        clear_since_s = None
    held = False
    if raw_dangerous:
        dangerous, clear_since_s = True, None
    elif was_dangerous:
        if not bool(recovered):
            clear_since_s = None
            dangerous, held = True, True
        else:
            if clear_since_s is None or now_s < clear_since_s:
                clear_since_s = now_s
            clear_elapsed_s = max(0.0, now_s - clear_since_s)
            if clear_elapsed_s + 1.0e-9 >= release_s:
                dangerous, clear_since_s = False, None
            else:
                dangerous, held = True, True
    else:
        dangerous, clear_since_s = False, None
    return dangerous, held, {
        "dangerous": bool(dangerous),
        "clear_since_s": clear_since_s,
    }


def _build_effective_corridor(
    *, reference_samples, ego_snapshot, all_agents, mode_groups,
    tag_by_id, assign_by_id, corridor_params, rss_params,
    credible_mode_probability_min, credible_mode_ttc_s,
    nominal_progress_limit_m=None,
    veto_state=None, veto_release_duration_s=0.6,
    veto_release_margin_s=0.5, sim_time_s=0.0,
):
    mode_diagnostics = _mode_diagnostics(mode_groups, tag_by_id)
    items = []
    for agent in all_agents:
        aid = _agent_id(agent)
        if "::mode" in aid:
            continue
        tag = tag_by_id.get(aid)
        assignment = assign_by_id.get(aid)
        if tag is not None and tag.tag != IGNORE:
            items.append((agent, tag, assignment))
        elif tag is not None and str(getattr(assignment, "role", "")) == "make_gap":
            # A spatially-overlapping resource claim is future topology
            # evidence even while the peer is still geometrically adjacent.
            # Stage A remains an honest IGNORE for physical collision risk;
            # Stage C receives the negotiated merge semantic only for the
            # losing vehicle's make-gap corridor.
            items.append((agent, ConflictTag(
                agent_id=tag.agent_id,
                tag=MERGE,
                conflict_s_m=None,
                conflict_t_s=None,
                min_gap_m=tag.min_gap_m,
                min_lateral_m=tag.min_lateral_m,
                cooperative=True,
                reason="conflicting_resource_claim",
            ), assignment))
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
                sim_time_s=float(sim_time_s),
                release_duration_s=float(veto_release_duration_s),
            )
            next_veto_state[str(mode_id)] = entry
            credible_veto_count += int(dangerous)
            held_veto_count += int(held)
            mode_corridors.append((mode_corridor, float(probability), dangerous, mode_id))
            detail = mode_diagnostics.get(str(mode_id))
            if detail is not None:
                clear_since_s = entry.get("clear_since_s")
                clear_elapsed_s = (
                    0.0 if clear_since_s is None else max(
                        0.0, float(sim_time_s) - float(clear_since_s)
                    )
                )
                detail.update({
                    "raw_dangerous": bool(raw_dangerous),
                    "credible_veto_active": bool(dangerous),
                    "credible_veto_held": bool(held),
                    "credible_veto_clear_elapsed_s": float(clear_elapsed_s),
                    "credible_veto_release_remaining_s": (
                        max(
                            0.0,
                            float(veto_release_duration_s) - clear_elapsed_s,
                        ) if bool(held) else 0.0
                    ),
                    "first_risk_stage": first_risk_stage,
                    "first_risk_time_s": (
                        None if first_risk_stage is None else float(risk_t_s)
                    ),
                    "active_corridor_stage_count": sum(
                        1 for cap in mode_corridor.s_hi if float(cap) < 1.0e9
                    ),
                })
        aggregate = aggregate_mode_corridors(mode_corridors, nominal_s)
        for _mode_corridor, _probability, _dangerous, mode_id in mode_corridors:
            detail = mode_diagnostics.get(str(mode_id))
            if detail is not None:
                detail["veto_binding_stage_count"] = sum(
                    1 for binding in aggregate.binding
                    if str(binding) == str(mode_id)
                )
        for k, cap in enumerate(aggregate.s_hi):
            if cap < corridor.s_hi[k]:
                corridor.s_hi[k] = cap
                corridor.binding[k] = aggregate.binding[k] or actor_id
    corridor.clamp_and_check()
    return (
        corridor, credible_veto_count, held_veto_count,
        next_veto_state, mode_diagnostics,
    )


@dataclass(frozen=True)
class _ExpandedAgents:
    """Physical agents with multimodal prediction expanded into QP-visible rows."""

    physical_agents: List[Mapping[str, Any]]
    all_agents: List[Mapping[str, Any]]
    mode_groups: Dict[str, List[Tuple[Mapping[str, Any], float]]]
    retained_mode_count: int
    mode_capped_agent_count: int
    raw_obstacle_count: int
    cav_agent_count: int


def _build_physical_agents_with_modes(
    *,
    cavs: Sequence[CavIntent],
    obstacle_snapshots: Sequence[Mapping[str, Any]],
    prediction_modes: Optional[Mapping[str, Sequence[Any]]],
    mode_probability_floor: float,
    credible_mode_probability_min: float,
    max_modes_per_agent: int,
) -> _ExpandedAgents:
    """Fuse CAV broadcasts with perception, then expand retained modes.

    A connected vehicle's shared trajectory replaces its perception track (a
    broadcast plan is the richer representation); everything else keeps all
    retained prediction hypotheses.  An agent with more than one retained
    mode is expanded into one ``id::modeN`` row per hypothesis so Stage A/C
    see every mode as its own agent; single- or zero-mode agents pass through
    as one physical row so Stage A's ``predicted_trajectory`` contract still
    holds when no probabilistic aggregation is needed.
    """

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
        # agent) wins; every other road user gets all retained prediction
        # hypotheses here.
        if "predicted_modes" not in a and aid in modes_by_id:
            a = {**a, "predicted_modes": list(as_modes(modes_by_id[aid]))}
            a.setdefault("trajectory_source", "prediction")
        perception_agents.append(a)
    physical_agents: List[Mapping[str, Any]] = perception_agents + cav_agents
    all_agents: List[Mapping[str, Any]] = []
    mode_groups: Dict[str, List[Tuple[Mapping[str, Any], float]]] = {}
    retained_mode_count = 0
    mode_capped_agent_count = 0
    for agent in physical_agents:
        modes = as_modes(agent.get("predicted_modes"))
        retained = [
            mode for mode in modes
            if float(mode.probability) >= max(0.0, float(mode_probability_floor))
        ]
        pre_cap_count = len(retained)
        if int(max_modes_per_agent) > 0 and pre_cap_count > int(max_modes_per_agent):
            # A mode at or above credible_mode_probability_min is exactly what
            # the credible-danger veto below exists to catch even at low
            # probability -- dropping it here for budget would silently
            # blind that check to a hypothesis it was built to see. Only the
            # sub-credible tail is budget's to trim, ranked by probability
            # since among modes none of this tick's stages will veto on,
            # probability is the only signal left to rank by.
            credible = [
                mode for mode in retained
                if float(mode.probability) >= float(credible_mode_probability_min)
            ]
            non_credible = sorted(
                (
                    mode for mode in retained
                    if float(mode.probability) < float(credible_mode_probability_min)
                ),
                key=lambda mode: -float(mode.probability),
            )
            room = max(0, int(max_modes_per_agent) - len(credible))
            retained = credible + non_credible[:room]
            if len(retained) < pre_cap_count:
                mode_capped_agent_count += 1
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

    return _ExpandedAgents(
        physical_agents=physical_agents,
        all_agents=all_agents,
        mode_groups=mode_groups,
        retained_mode_count=retained_mode_count,
        mode_capped_agent_count=mode_capped_agent_count,
        raw_obstacle_count=len(raw_obstacles),
        cav_agent_count=len(cav_agents),
    )


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
    credible_mode_veto_release_s: float = 0.6,
    credible_mode_veto_release_margin_s: float = 0.5,
    sim_time_s: float = 0.0,
    nominal_progress_limit_m: Optional[float] = None,
    veto_state: Optional[Mapping[str, Any]] = None,
    refresh_assignments: bool = True,
    cached_assignments: Sequence[ConflictAssignment] = (),
    rebuild_corridor: bool = True,
    cached_corridor: Optional[Corridor] = None,
    corridor_reference_samples: Optional[Sequence[Any]] = None,
    max_relevant_agents: int = 6,
    max_modes_per_agent: int = 3,
) -> ConflictResolution:
    cavs = list(cav_intents or [])
    expanded = _build_physical_agents_with_modes(
        cavs=cavs,
        obstacle_snapshots=obstacle_snapshots,
        prediction_modes=prediction_modes,
        mode_probability_floor=mode_probability_floor,
        credible_mode_probability_min=credible_mode_probability_min,
        max_modes_per_agent=max_modes_per_agent,
    )
    physical_agents = expanded.physical_agents
    all_agents = expanded.all_agents
    mode_groups = expanded.mode_groups
    retained_mode_count = expanded.retained_mode_count
    mode_capped_agent_count = expanded.mode_capped_agent_count
    raw_obstacle_count = expanded.raw_obstacle_count
    cav_agent_count = expanded.cav_agent_count

    # Stage A -----------------------------------------------------------------
    # Every agent is classified before any count budget applies: a crossing
    # vehicle still on its own street (large *current* lateral offset from
    # ego's path) is exactly the case a pre-classification position filter
    # would misjudge as irrelevant, and classify_conflicts already reasons
    # over the full predicted track, not current position.
    tags = classify_conflicts(
        reference_samples, ego_snapshot, all_agents, classifier_params,
        previous_tags=tag_state,
    )
    all_agents, tags, mode_groups, dropped_agent_count = _cap_agents_by_severity(
        all_agents=all_agents, tags=tags, mode_groups=mode_groups,
        max_agents=int(max_relevant_agents),
    )
    tag_by_id = {t.agent_id: t for t in tags}
    corridor_reference = (
        reference_samples
        if corridor_reference_samples is None
        else corridor_reference_samples
    )
    corridor_tag_by_id = _tags_on_corridor_reference(
        tags, reference_samples, corridor_reference
    )

    # Stage B (only cooperative cavs, only when ego holds an active claim) ---
    assignments: List[ConflictAssignment] = []
    arbitration_diagnostics: Dict[str, Any] = {}
    current_tag_state = {tag.agent_id: tag.tag for tag in tags}
    tag_changed = bool(
        tag_state is not None and current_tag_state != dict(tag_state or {})
    )
    # Stage-A geometry runs every planning tick, so a tag transition must
    # rebuild the Stage-C safety corridor immediately.  It does *not* own
    # Stage-B cooperative priority: proceed/yield/make-gap changes only on
    # the coordination schedule (or an asynchronous claim-structure event
    # handled by that schedule).  Coupling tag churn to role refresh silently
    # promoted Stage B from 5 Hz to 20 Hz near a cut-in and restarted its
    # hysteresis latch on each geometric transition.
    roles_refreshed = bool(refresh_assignments)
    # A scheduled refresh is authoritative for membership.  Preserve the
    # previous latch only while roles are deliberately cached between
    # coordination ticks; otherwise disappeared/ineligible peers would keep
    # stale role state indefinitely and could inherit it if an actor id is
    # later reused.
    new_latch: Dict[str, ArbitrationLatchEntry] = (
        {} if roles_refreshed else dict(latch_state or {})
    )
    if not roles_refreshed:
        live_ids = {int(c.actor_id) for c in cavs if bool(c.cooperative)}
        assignments = [
            assignment for assignment in cached_assignments or ()
            if int(assignment.cav_actor_id) in live_ids
        ]
    elif my_claim is not None and bool(my_claim.participates):
        # Resource claims describe a future shared topology and therefore
        # enter arbitration before current geometry becomes a Stage-A
        # conflict. ``assign_conflict_roles`` remains the sole owner of
        # cooperative/active/spatial eligibility.
        conflicting_cavs = [c for c in cavs if bool(c.cooperative)]
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
    clearable_actor_ids = {
        str(tag.agent_id).split("::mode", 1)[0]
        for tag in tags
        if tag.tag == IGNORE and (
            str(tag.reason).startswith("min_lat=")
            or str(tag.reason) == "outside_longitudinal_window"
        )
    }
    active_actor_ids = {
        str(tag.agent_id).split("::mode", 1)[0]
        for tag in tags if tag.tag != IGNORE
    }
    negotiated_gap_actor_ids = {
        str(assignment.cav_actor_id)
        for assignment in assignments
        if str(assignment.role) == "make_gap"
    }
    cleared_actor_ids = (
        clearable_actor_ids - active_actor_ids - negotiated_gap_actor_ids
    )
    cached_corridor = remove_actor_bounds(
        cached_corridor, cleared_actor_ids,
    )
    incoming_veto_state = dict(veto_state or {})
    held_veto_count = 0
    if corridor_rebuilt or cached_corridor is None:
        (corridor, credible_veto_count, held_veto_count,
         new_veto_state, mode_diagnostics) = _build_effective_corridor(
            reference_samples=corridor_reference, ego_snapshot=ego_snapshot,
            all_agents=all_agents, mode_groups=mode_groups,
            tag_by_id=corridor_tag_by_id, assign_by_id=assign_by_id,
            corridor_params=corridor_params, rss_params=rss_params,
            credible_mode_probability_min=credible_mode_probability_min,
            credible_mode_ttc_s=credible_mode_ttc_s,
            nominal_progress_limit_m=nominal_progress_limit_m,
            veto_state=incoming_veto_state,
            veto_release_duration_s=float(credible_mode_veto_release_s),
            veto_release_margin_s=float(credible_mode_veto_release_margin_s),
            sim_time_s=float(sim_time_s),
        )
        fresh_corridor = corridor
        if cached_corridor is not None and not tag_changed:
            # A complete current Stage-C build supersedes older rows owned by
            # every actor present in that build.  Retention exists only to
            # bridge a temporarily missing observation; otherwise an open
            # new forecast would leave the same actor's obsolete cap active
            # until the old horizon expired (a several-second ghost stop).
            refreshed_actor_ids = {
                _physical_agent_id(agent) for agent in physical_agents
            }
            cached_corridor = remove_actor_bounds(
                cached_corridor, refreshed_actor_ids,
            )
            corridor = retain_pending_corridor(corridor, cached_corridor)
        corridor_rebuilt = True
    else:
        corridor = cached_corridor
        fresh_corridor = None
        credible_veto_count = 0
        mode_diagnostics = _mode_diagnostics(
            mode_groups, corridor_tag_by_id
        )
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
        # Post-severity-budget counts: how many physical agents (and their
        # mode-expanded entries) actually reached Stage B/C this tick.
        "conflict_agent_count": len({
            _physical_agent_id(agent) for agent in all_agents
        }),
        "mode_conflict_count": len(all_agents),
        "deduplicated_agent_count": (
            raw_obstacle_count + cav_agent_count - len(physical_agents)
        ),
        "relevant_agent_budget": int(max_relevant_agents),
        "relevant_agent_dropped_count": int(dropped_agent_count),
        "mode_budget_per_agent": int(max_modes_per_agent),
        "mode_budget_capped_agent_count": int(mode_capped_agent_count),
        "cav_count": len(cavs),
        "shared_plan_cav_count": sum(1 for c in cavs if c.planned_path),
        "shared_plan_sample_count": sum(len(c.planned_path) for c in cavs),
        "multimodal_agent_count": sum(
            1 for a in physical_agents if len(as_modes(a.get("predicted_modes"))) > 1
        ),
        "retained_prediction_mode_count": int(retained_mode_count),
        "prediction_modes": mode_diagnostics,
        "credible_mode_veto_count": int(credible_veto_count),
        "credible_mode_veto_held_count": int(held_veto_count),
        "credible_mode_veto_release_s": float(
            credible_mode_veto_release_s
        ),
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
        "semantic_risks": {t.agent_id: t.risk_kind for t in tags},
        "object_types": {t.agent_id: t.object_type for t in tags},
        "observation_sources": {
            t.agent_id: t.observation_source for t in tags
        },
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
            "scheduled_refresh_with_tag_change"
            if refresh_assignments and tag_changed
            else "scheduled_refresh" if refresh_assignments
            else "corridor_tag_changed" if tag_changed
            else "cached_roles"
        ),
        "corridor_rebuilt": bool(corridor_rebuilt),
        "released_actor_ids": tuple(sorted(cleared_actor_ids)),
    }
    return ConflictResolution(
        tags=tags, assignments=assignments, corridor=corridor,
        fresh_corridor=fresh_corridor,
        latch_state=new_latch,
        tag_state=current_tag_state,
        veto_state=dict(new_veto_state or {}),
        released_actor_ids=tuple(sorted(cleared_actor_ids)),
        diagnostics=diagnostics,
    )
