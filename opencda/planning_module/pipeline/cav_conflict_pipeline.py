"""Fusion: chain Stage A (classify) -> Stage B (assign) -> Stage C (corridor)
into one per-tick call.

    resolution = resolve_conflicts(
        reference_samples=lane_center_reference,
        ego_snapshot={... "x","y","v","psi"},
        my_claim=ResourceClaim(...) or None,
        my_actor_id=<ego id>,
        obstacle_snapshots=[...],          # non-connected road users
        peer_intents=[PeerIntent, ...],    # from collect_peer_intents(...)
        latch_state=<prev tick's>,
    )
    # -> resolution.corridor (Stage C), .assignments (Stage B),
    #    .tags (Stage A), .latch_state (feed back next tick), .diagnostics

Pure. The bridge is expected to: build ``peer_intents`` from its CP
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
    PeerIntent,
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
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _peer_to_agent_snapshot(peer: PeerIntent) -> dict:
    return {
        "id": int(peer.actor_id),
        "vehicle_id": int(peer.actor_id),
        "x": float(peer.position_xy[0]),
        "y": float(peer.position_xy[1]),
        "v": float(peer.speed_mps),
        "psi": float(peer.heading_rad),
        "cooperative": bool(peer.cooperative),
        "predicted_trajectory": [
            {"x": float(x), "y": float(y), "t": float(t)}
            for (t, x, y, _v) in peer.planned_path
        ],
    }


def resolve_conflicts(
    *,
    reference_samples: Sequence[Any],
    ego_snapshot: Mapping[str, Any],
    my_actor_id: int,
    my_claim: Optional[ResourceClaim] = None,
    obstacle_snapshots: Sequence[Mapping[str, Any]] = (),
    peer_intents: Sequence[PeerIntent] = (),
    latch_state: Optional[Mapping[str, ArbitrationLatchEntry]] = None,
    classifier_params: ClassifierParams = ClassifierParams(),
    corridor_params: CorridorParams = CorridorParams(),
    rss_params: RSSParams = RSSParams(),
    arbitration_range_m: float = 40.0,
    hysteresis_ticks: int = 3,
) -> ConflictResolution:
    peers = list(peer_intents or [])
    peer_agents = [_peer_to_agent_snapshot(p) for p in peers]
    all_agents: List[Mapping[str, Any]] = list(obstacle_snapshots or []) + peer_agents

    # Stage A -----------------------------------------------------------------
    tags = classify_conflicts(
        reference_samples, ego_snapshot, all_agents, classifier_params
    )
    tag_by_id = {t.agent_id: t for t in tags}

    # Stage B (only cooperative peers, only when ego holds an active claim) ---
    assignments: List[ConflictAssignment] = []
    new_latch: Dict[str, ArbitrationLatchEntry] = dict(latch_state or {})
    if my_claim is not None and bool(my_claim.active):
        conflicting_peers = []
        for p in peers:
            if not bool(p.cooperative):
                continue
            t = tag_by_id.get(str(p.actor_id))
            if t is None or t.tag == IGNORE:
                continue
            conflicting_peers.append(p)
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
            peers=conflicting_peers,
            latch_state=latch_state,
            range_m=float(arbitration_range_m),
            hysteresis_ticks=int(hysteresis_ticks),
        )
    assign_by_id = {str(a.peer_actor_id): a for a in assignments}

    # Stage C ---------------------------------------------------------------
    items: List[Tuple[Mapping[str, Any], ConflictTag, Optional[ConflictAssignment]]] = []
    for agent in all_agents:
        aid = str(
            agent.get("id", agent.get("vehicle_id", agent.get("track_id", agent.get("actor_id", ""))))
        )
        t = tag_by_id.get(aid)
        if t is None or t.tag == IGNORE:
            continue
        items.append((agent, t, assign_by_id.get(aid)))

    corridor = build_longitudinal_corridor(
        reference_samples, ego_snapshot, items, corridor_params, rss_params
    )

    diagnostics = {
        "conflict_agent_count": len(all_agents),
        "peer_count": len(peers),
        "non_ignore_count": sum(1 for t in tags if t.tag != IGNORE),
        "tags": {t.agent_id: t.tag for t in tags},
        "roles": {str(a.peer_actor_id): a.role for a in assignments},
        "corridor_feasible": bool(corridor.feasible),
        "corridor_first_infeasible_stage": corridor.first_infeasible_stage,
        "corridor_binding": [b for b in corridor.binding if b],
    }
    return ConflictResolution(
        tags=tags, assignments=assignments, corridor=corridor,
        latch_state=new_latch, diagnostics=diagnostics,
    )
