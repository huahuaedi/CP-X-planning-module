"""Bounded diagnostics for Python cyclic-garbage-collection pauses."""

from __future__ import annotations

from collections import defaultdict
import gc
import threading
import time


class RuntimePauseMonitor:
    """Record GC pauses without logging or writing from the control path."""

    def __init__(self, *, enabled=True):
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._active_cycle_id = 0
        self._starts_ns = {}
        self._count = 0
        self._total_ms = 0.0
        self._max_pause = {
            "duration_ms": 0.0,
            "cycle_id": 0,
            "generation": -1,
        }
        self._generation_count = defaultdict(int)
        self._generation_total_ms = defaultdict(float)
        self._callback = self._gc_callback
        if self.enabled:
            gc.callbacks.append(self._callback)

    def set_cycle(self, cycle_id):
        """Associate subsequent pauses with the active planning cycle."""
        self._active_cycle_id = int(cycle_id or 0)

    def clear_cycle(self):
        """Stop attributing process pauses to the completed cycle."""
        self._active_cycle_id = 0

    def _gc_callback(self, phase, info):
        generation = int(dict(info or {}).get("generation", -1) or 0)
        key = (threading.get_ident(), generation)
        now_ns = time.perf_counter_ns()
        if phase == "start":
            self._starts_ns[key] = now_ns
            return
        if phase != "stop":
            return
        started_ns = self._starts_ns.pop(key, None)
        if started_ns is None:
            return
        duration_ms = max(0.0, (now_ns - started_ns) / 1.0e6)
        with self._lock:
            self._count += 1
            self._total_ms += duration_ms
            self._generation_count[generation] += 1
            self._generation_total_ms[generation] += duration_ms
            if duration_ms > float(self._max_pause["duration_ms"]):
                self._max_pause = {
                    "duration_ms": duration_ms,
                    "cycle_id": int(self._active_cycle_id),
                    "generation": generation,
                }

    def snapshot(self):
        """Return one JSON-safe bounded aggregate."""
        with self._lock:
            generations = {}
            for generation in sorted(self._generation_count):
                count = int(self._generation_count[generation])
                total_ms = float(self._generation_total_ms[generation])
                generations[str(generation)] = {
                    "count": count,
                    "total_ms": total_ms,
                    "average_ms": total_ms / max(1, count),
                }
            return {
                "enabled": bool(self.enabled),
                "collection_count": int(self._count),
                "total_pause_ms": float(self._total_ms),
                "max_pause": dict(self._max_pause),
                "generations": generations,
                "automatic_gc_enabled": bool(gc.isenabled()),
                "gc_threshold": list(gc.get_threshold()),
                "gc_count": list(gc.get_count()),
            }

    def close(self):
        """Unregister the callback and return its final snapshot."""
        if self.enabled:
            try:
                gc.callbacks.remove(self._callback)
            except ValueError:
                pass
        return self.snapshot()
