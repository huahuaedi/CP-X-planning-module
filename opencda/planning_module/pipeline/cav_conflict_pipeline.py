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
from opencda.planning_module.pipeline.rss import RSSParams


@dataclass
class ConflictResolution:
    tags: List[ConflictTag]
    assignments: List[ConflictAssignment]
    corridor: Corridor
    latch_state: Dict[str, ArbitrationLatchEntry]
    mpc_rows: List[Any] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _cav_to_agent_snapshot(cav: CavIntent) -> dict:
    return {
        "id": int(cav.actor_id),
        "vehicle_id": int(cav.actor_id),
        "x": float(cav.position_xy[0]),
        "y": float(cav.position_xy[1]),
        "v": float(cav.speed_mps),
        "psi": float(cav.heading_rad),
        "cooperative": bool(cav.cooperative),
        "predicted_trajectory": [
            {"x": float(x), "y": float(y), "t": float(t)}
            for (t, x, y, _v) in cav.planned_path
        ],
    }


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
    latch_state: Optional[Mapping[str, ArbitrationLatchEntry]] = None,
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
    perception_agents = [a for a in raw_obstacles if _agent_id(a) not in cav_ids]
    all_agents: List[Mapping[str, Any]] = perception_agents + cav_agents

    # Stage A -----------------------------------------------------------------
    tags = classify_conflicts(
        reference_samples, ego_snapshot, all_agents, classifier_params
    )
    tag_by_id = {t.agent_id: t for t in tags}

    # Stage B (only cooperative cavs, only when ego holds an active claim) ---
    assignments: List[ConflictAssignment] = []
    new_latch: Dict[str, ArbitrationLatchEntry] = dict(latch_state or {})
    if my_claim is not None and bool(my_claim.active):
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

    diagnostics = {
        "conflict_agent_count": len(all_agents),
        "deduplicated_agent_count": len(raw_obstacles) + len(cav_agents) - len(all_agents),
        "cav_count": len(cavs),
        "non_ignore_count": sum(1 for t in tags if t.tag != IGNORE),
        "tags": {t.agent_id: t.tag for t in tags},
        "roles": {str(a.cav_actor_id): a.role for a in assignments},
        "corridor_feasible": bool(corridor.feasible),
        "corridor_first_infeasible_stage": corridor.first_infeasible_stage,
        "corridor_binding": [b for b in corridor.binding if b],
    }
    return ConflictResolution(
        tags=tags, assignments=assignments, corridor=corridor,
        latch_state=new_latch, diagnostics=diagnostics,
    )
