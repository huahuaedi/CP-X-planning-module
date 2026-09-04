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
    build_longitudinal_corridor,
)
from opencda.planning_module.pipeline.prediction_modes import as_modes, single_mode
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
) -> ConflictResolution:
    cavs = list(cav_intents or [])
    cav_agents = [_cav_to_agent_snapshot(c) for c in cavs]
    # A connected vehicle normally also appears in perception.  Its shared
    # trajectory is the richer representation, so replace (rather than add
    # to) the perception track for the same actor.
    cav_ids = {str(c.actor_id) for c in cavs}
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
    all_agents: List[Mapping[str, Any]] = perception_agents + cav_agents

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
        t = tag_by_id.get(aid)
        if t is None or t.tag == IGNORE:
            continue
        items.append((agent, t, assign_by_id.get(aid)))

    corridor = build_longitudinal_corridor(
        reference_samples, ego_snapshot, items, corridor_params, rss_params
    )

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
        "conflict_agent_count": len(all_agents),
        "deduplicated_agent_count": len(raw_obstacles) + len(cav_agents) - len(all_agents),
        "cav_count": len(cavs),
        "shared_plan_cav_count": sum(1 for c in cavs if c.planned_path),
        "shared_plan_sample_count": sum(len(c.planned_path) for c in cavs),
        "multimodal_agent_count": sum(
            1 for a in all_agents if len(as_modes(a.get("predicted_modes"))) > 1
        ),
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
