"""Bounded diagnostics for Python cyclic-garbage-collection pauses."""

from collections import defaultdict
import gc
import threading
import time


class RuntimePauseMonitor:
    """Record GC pauses without logging from the ROS bridge hot path."""

    def __init__(self):
        self._lock = threading.RLock()
        self._starts_ns = {}
        self._count = 0
        self._total_ms = 0.0
        self._max_pause = {"duration_ms": 0.0, "generation": -1}
        self._generation_count = defaultdict(int)
        self._generation_total_ms = defaultdict(float)
        self._callback = self._gc_callback
        gc.callbacks.append(self._callback)

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
                    "generation": generation,
                }

    def snapshot(self):
        """Return one JSON-safe bounded aggregate."""
        with self._lock:
            return {
                "collection_count": int(self._count),
                "total_pause_ms": float(self._total_ms),
                "max_pause": dict(self._max_pause),
                "generations": {
                    str(generation): {
                        "count": int(self._generation_count[generation]),
                        "total_ms": float(
                            self._generation_total_ms[generation]
                        ),
                    }
                    for generation in sorted(self._generation_count)
                },
                "automatic_gc_enabled": bool(gc.isenabled()),
                "gc_threshold": list(gc.get_threshold()),
                "gc_count": list(gc.get_count()),
            }

    def close(self):
        """Unregister the callback and return its final snapshot."""
        try:
            gc.callbacks.remove(self._callback)
        except ValueError:
            pass
        return self.snapshot()
