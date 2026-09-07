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
)
from opencda.planning_module.pipeline.prediction_modes import as_modes, single_mode
from opencda.planning_module.pipeline.mpc_obstacle_relevance import (
    _point_to_polyline,
    _polyline_xy,
)
from opencda.planning_module.pipeline.rss import RSSParams


@dataclass
class ConflictResolution:
    tags: List[ConflictTag]
    assignments: List[ConflictAssignment]
    corridor: Corridor
    latch_state: Dict[str, ArbitrationLatchEntry]
    tag_state: Dict[str, str] = field(default_factory=dict)
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
    arbitration_range_m: float = 40.0,
    hysteresis_ticks: int = 3,
    mode_probability_floor: float = 0.05,
    credible_mode_probability_min: float = 0.15,
    credible_mode_ttc_s: float = 2.0,
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
        if len(retained) <= 1:
            all_agents.append(agent)
            retained_mode_count += len(retained)
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
    new_latch: Dict[str, ArbitrationLatchEntry] = dict(latch_state or {})
    if my_claim is not None and bool(my_claim.participates):
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
            range_m=float(arbitration_range_m),
            hysteresis_ticks=int(hysteresis_ticks),
        )
    assign_by_id = {str(a.cav_actor_id): a for a in assignments}

    # Stage C ---------------------------------------------------------------
    items: List[Tuple[Mapping[str, Any], ConflictTag, Optional[ConflictAssignment]]] = []
    for agent in all_agents:
        aid = _agent_id(agent)
        if "::mode" in aid:
            continue
        t = tag_by_id.get(aid)
        if t is None or t.tag == IGNORE:
            continue
        items.append((agent, t, assign_by_id.get(aid)))

    corridor = build_longitudinal_corridor(
        reference_samples, ego_snapshot, items, corridor_params, rss_params
    )

    # A physical actor with several possible futures must not appear to the
    # optimizer as several simultaneous vehicles.  Build one corridor per
    # mode, then reduce them to one probability-aware effective corridor.
    poly = _polyline_xy(reference_samples)
    ego_s0 = 0.0
    if len(poly) >= 2:
        ego_s0 = _point_to_polyline(
            float(ego_snapshot.get("x", ego_snapshot.get("x_m", 0.0))),
            float(ego_snapshot.get("y", ego_snapshot.get("y_m", 0.0))),
            poly,
        )[1]
    ego_v = max(0.0, float(ego_snapshot.get(
        "v", ego_snapshot.get("speed_mps", ego_snapshot.get("speed", 0.0))
    )))
    nominal_s = [
        ego_s0 + ego_v * float(corridor_params.dt_s) * k
        for k in range(len(corridor.s_hi))
    ]
    credible_veto_count = 0
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
                corridor_params, rss_params, include_follow_bounds=True,
            )
            first_risk_stage = next((
                k for k, cap in enumerate(mode_corridor.s_hi)
                if k < len(nominal_s) and float(cap) < float(nominal_s[k])
            ), None)
            dangerous = bool(
                float(probability) >= float(credible_mode_probability_min)
                and tag.tag != IGNORE
                and first_risk_stage is not None
                and float(first_risk_stage) * float(corridor_params.dt_s)
                <= float(credible_mode_ttc_s)
            )
            credible_veto_count += int(dangerous)
            mode_corridors.append((
                mode_corridor, float(probability), dangerous, mode_id
            ))
        aggregate = aggregate_mode_corridors(mode_corridors, nominal_s)
        for k, cap in enumerate(aggregate.s_hi):
            if cap < corridor.s_hi[k]:
                corridor.s_hi[k] = cap
                corridor.binding[k] = aggregate.binding[k] or actor_id
    corridor.clamp_and_check()

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
        "trajectory_source_counts": source_counts,
        "ego_claim_phase": (
            "none" if my_claim is None else str(my_claim.phase)
        ),
        "peer_claim_phases": {
            str(c.actor_id): str(c.claim.phase) for c in cavs
        },
        "non_ignore_count": sum(1 for t in tags if t.tag != IGNORE),
        "speed_owned_follow_count": sum(
            1 for t in tags if t.tag in ("FOLLOW", "LEAD_BRAKE")
        ),
        "tags": {t.agent_id: t.tag for t in tags},
        "roles": {str(a.cav_actor_id): a.role for a in assignments},
        "corridor_feasible": bool(corridor.feasible),
        "corridor_first_infeasible_stage": corridor.first_infeasible_stage,
        "corridor_binding": [b for b in corridor.binding if b],
    }
    return ConflictResolution(
        tags=tags, assignments=assignments, corridor=corridor,
        latch_state=new_latch,
        tag_state={tag.agent_id: tag.tag for tag in tags},
        diagnostics=diagnostics,
    )
