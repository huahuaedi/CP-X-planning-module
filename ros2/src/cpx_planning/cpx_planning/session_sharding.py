"""Deterministic ownership for process-isolated multi-CAV Planner workers."""

from __future__ import annotations

import zlib


def session_shard_index(session_id, shard_count):
    """Return the stable worker index that owns ``session_id``.

    OpenCDA session identifiers end in the CARLA actor id.  Using that suffix
    keeps ownership easy to inspect in logs.  A CRC32 fallback covers custom
    session identifiers without relying on Python's randomized ``hash()``.
    """

    count = max(1, int(shard_count))
    value = str(session_id or "").strip()
    if not value:
        return 0
    try:
        owner_key = int(value.rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        owner_key = zlib.crc32(value.encode("utf-8"))
    return int(owner_key % count)


def owns_session(session_id, shard_count, shard_index):
    """Return whether one Planner process owns the given CAV session."""

    count = max(1, int(shard_count))
    index = int(shard_index)
    if index < 0 or index >= count:
        raise ValueError("planner shard index must be in [0, shard_count)")
    return session_shard_index(session_id, count) == index
