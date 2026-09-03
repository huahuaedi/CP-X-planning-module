"""C1: connected-cav intent -- schema, codec, and construction.

The bridge broadcasts the ego's own ``CavIntent`` on the V2X / CP channel
and receives cavs' intents back. This module is the pure boundary:

* ``build_ego_cav_intent`` -- assemble the ego broadcast from its plan;
* ``cav_intent_to_payload`` / ``cav_intent_from_payload`` -- flat,
  JSON-safe, schema-versioned dict <-> ``CavIntent`` (defensive parse:
  a malformed cav record yields ``None``, never an exception);
* ``collect_cav_intents`` -- pull every usable cav intent out of a
  CP-message-shaped list of per-actor records, skipping the ego;
* ``sample_cav_path_at`` -- linear-interpolate a cav's shared plan at a
  horizon time, for Stage A / Stage C.

``CavIntent`` / ``ResourceClaim`` themselves live in
``cooperative_arbitration`` so Stage B can consume them without importing
this module.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from opencda.planning_module.pipeline.cooperative_arbitration import (
    CavIntent,
    ResourceClaim,
)

SCHEMA_VERSION = 1
_PATH_SAMPLE_LEN = 4  # (t_rel_s, x, y, v)


def _f(m: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in m and m[k] is not None:
            try:
                return float(m[k])
            except (TypeError, ValueError):
                return default
    return default


def _actor_id(m: Mapping[str, Any]) -> Optional[int]:
    for k in ("actor_id", "vehicle_id", "id", "cav_id"):
        if k in m and m[k] is not None:
            try:
                return int(m[k])
            except (TypeError, ValueError):
                return None
    return None


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def build_ego_cav_intent(
    *,
    actor_id: int,
    position_xy: Tuple[float, float],
    heading_rad: float,
    speed_mps: float,
    claim: ResourceClaim,
    planned_states: Sequence[Sequence[float]] = (),
    dt_s: float = 0.1,
    t0_s: float = 0.0,
    max_samples: int = 30,
    cooperative: bool = True,
    generated_at_s: float = 0.0,
    valid_for_s: float = 0.5,
    sequence: int = 0,
    probability: float = 1.0,
) -> CavIntent:
    """Assemble the ego's broadcast.

    ``planned_states`` is the MPC state sequence ``[x, y, v, psi]`` (world
    frame); it is converted to ``(t_rel_s, x, y, v)`` samples spaced
    ``dt_s`` starting at ``t0_s``. Empty when the ego does not share a plan.
    """

    path: List[Tuple[float, float, float, float]] = []
    for k, st in enumerate(list(planned_states or [])[: max(0, int(max_samples))]):
        if not isinstance(st, Sequence) or len(st) < 3:
            continue
        path.append(
            (
                float(t0_s) + float(dt_s) * float(k),
                float(st[0]),
                float(st[1]),
                float(st[2]),
            )
        )
    return CavIntent(
        actor_id=int(actor_id),
        position_xy=(float(position_xy[0]), float(position_xy[1])),
        claim=claim,
        heading_rad=float(heading_rad),
        speed_mps=max(0.0, float(speed_mps)),
        planned_path=tuple(path),
        cooperative=bool(cooperative),
        generated_at_s=float(generated_at_s),
        valid_until_s=float(generated_at_s) + max(0.0, float(valid_for_s)),
        sequence=max(0, int(sequence)),
        probability=min(1.0, max(0.0, float(probability))),
    )


# --------------------------------------------------------------------------- #
# codec
# --------------------------------------------------------------------------- #
def cav_intent_to_payload(intent: CavIntent) -> dict:
    c = intent.claim
    return {
        "schema": SCHEMA_VERSION,
        "actor_id": int(intent.actor_id),
        "x": float(intent.position_xy[0]),
        "y": float(intent.position_xy[1]),
        "heading_rad": float(intent.heading_rad),
        "speed_mps": float(intent.speed_mps),
        "cooperative": bool(intent.cooperative),
        "generated_at_s": float(intent.generated_at_s),
        "valid_until_s": float(intent.valid_until_s),
        "sequence": int(intent.sequence),
        "probability": float(intent.probability),
        "claim": {
            "kind": str(c.kind),
            "resource_id": str(c.resource_id),
            "committed_at_s": float(c.committed_at_s),
            "active": bool(c.active),
            "require_ahead": bool(c.require_ahead),
        },
        "planned_path": [
            [float(t), float(x), float(y), float(v)]
            for (t, x, y, v) in intent.planned_path
        ],
    }


def cav_intent_from_payload(payload: Mapping[str, Any]) -> Optional[CavIntent]:
    """Defensive parse. Returns None for anything unusable."""

    if not isinstance(payload, Mapping):
        return None
    try:
        schema = int(payload.get("schema", SCHEMA_VERSION))
    except (TypeError, ValueError):
        return None
    if schema != SCHEMA_VERSION:
        return None
    aid = _actor_id(payload)
    if aid is None:
        return None
    raw_claim = payload.get("claim")
    if not isinstance(raw_claim, Mapping):
        return None
    kind = str(raw_claim.get("kind", "") or "")
    resource_id = str(raw_claim.get("resource_id", "") or "")
    if not kind or not resource_id:
        return None
    claim = ResourceClaim(
        kind=kind,
        resource_id=resource_id,
        committed_at_s=_f(raw_claim, "committed_at_s"),
        active=bool(raw_claim.get("active", False)),
        require_ahead=bool(raw_claim.get("require_ahead", True)),
    )
    path: List[Tuple[float, float, float, float]] = []
    for s in list(payload.get("planned_path", []) or []):
        if isinstance(s, Sequence) and not isinstance(s, (str, bytes)) and len(s) >= _PATH_SAMPLE_LEN:
            try:
                path.append((float(s[0]), float(s[1]), float(s[2]), float(s[3])))
            except (TypeError, ValueError):
                continue
    return CavIntent(
        actor_id=aid,
        position_xy=(_f(payload, "x", "x_m"), _f(payload, "y", "y_m")),
        claim=claim,
        heading_rad=_f(payload, "heading_rad", "psi", "yaw"),
        speed_mps=_f(payload, "speed_mps", "speed", "v"),
        planned_path=tuple(path),
        cooperative=bool(payload.get("cooperative", True)),
        generated_at_s=_f(payload, "generated_at_s", "timestamp_s"),
        valid_until_s=_f(payload, "valid_until_s", default=float("inf")),
        sequence=max(0, int(_f(payload, "sequence", default=0.0))),
        probability=min(1.0, max(0.0, _f(
            payload, "probability", "confidence", default=1.0
        ))),
    )


def collect_cav_intents(
    records: Iterable[Mapping[str, Any]],
    *,
    self_actor_id: int,
    intent_key: str = "cooperative_intent",
    now_s: Optional[float] = None,
    minimum_probability: float = 0.0,
) -> List[CavIntent]:
    """Extract cav intents from a CP-message-shaped iterable.

    Each record is one other vehicle; its cooperative intent is either the
    record itself (already payload-shaped) or a nested dict under
    ``intent_key``. Records for ``self_actor_id``, and records without a
    parseable intent, expired intents, and intents below the configured
    probability floor are dropped.  If several records exist for one actor,
    only the newest sequence is returned.
    """

    newest_by_actor = {}
    for rec in list(records or []):
        if not isinstance(rec, Mapping):
            continue
        if _actor_id(rec) == int(self_actor_id):
            continue
        nested = rec.get(intent_key)
        payload = nested if isinstance(nested, Mapping) else rec
        # Backfill actor_id/position from the outer record when the nested
        # intent omits them.
        if isinstance(nested, Mapping) and "actor_id" not in payload and _actor_id(rec) is not None:
            payload = {**payload, "actor_id": _actor_id(rec)}
        intent = cav_intent_from_payload(payload)
        if intent is None:
            continue
        if now_s is not None and float(intent.valid_until_s) < float(now_s):
            continue
        if float(intent.probability) < max(0.0, float(minimum_probability)):
            continue
        previous = newest_by_actor.get(int(intent.actor_id))
        if previous is None or (
            int(intent.sequence), float(intent.generated_at_s)
        ) > (
            int(previous.sequence), float(previous.generated_at_s)
        ):
            newest_by_actor[int(intent.actor_id)] = intent
    return [newest_by_actor[k] for k in sorted(newest_by_actor)]


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #
def sample_cav_path_at(
    intent: CavIntent, t_rel_s: float
) -> Optional[Tuple[float, float, float]]:
    """(x, y, v) of the cav's shared plan at ``t_rel_s`` (linear interp,
    clamped to the sample span). None if the cav shared no plan."""

    path = intent.planned_path
    if not path:
        return None
    t = float(t_rel_s)
    if t <= path[0][0]:
        return (path[0][1], path[0][2], path[0][3])
    if t >= path[-1][0]:
        return (path[-1][1], path[-1][2], path[-1][3])
    for (t0, x0, y0, v0), (t1, x1, y1, v1) in zip(path, path[1:]):
        if t0 <= t <= t1:
            a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            return (x0 + a * (x1 - x0), y0 + a * (y1 - y0), v0 + a * (v1 - v0))
    return (path[-1][1], path[-1][2], path[-1][3])
