"""Non-blocking, latest-value dispatch for MTR prediction scenes."""

from __future__ import annotations

import threading
from collections import OrderedDict


class LatestPredictionSceneDispatcher:
    """Publish prediction scenes without blocking the control worker.

    MTR performs inference in its ROS subscription callback. A reliable DDS
    writer can therefore experience back-pressure while that callback is
    occupied. Planning must never wait for prediction transport, so this
    dispatcher gives DDS its own daemon thread and retains only the newest
    scene while a previous publish is in progress.
    """

    def __init__(self, publisher, on_error=None, key_fn=None):
        self._publisher = publisher
        self._on_error = on_error
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._key_fn = key_fn or self._default_key
        self._pending = OrderedDict()
        self._submitted = 0
        self._published = 0
        self._replaced = 0
        self._failures = 0
        self._thread = threading.Thread(
            target=self._run,
            name="cpx-prediction-scene-publisher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, scene):
        """Queue ``scene`` without letting one session replace another."""
        key = self._key_fn(scene)
        with self._lock:
            self._submitted += 1
            if key in self._pending:
                self._replaced += 1
                self._pending.pop(key, None)
            self._pending[key] = scene
        self._ready.set()

    def discard_pending(self, key=None):
        """Discard one session's pending scene, or all scenes when key is None."""
        with self._lock:
            if key is None:
                self._pending.clear()
            else:
                self._pending.pop(str(key), None)
            if not self._pending:
                self._ready.clear()

    def stats(self):
        """Return bounded transport diagnostics."""
        with self._lock:
            return {
                "submitted_scenes": int(self._submitted),
                "published_scenes": int(self._published),
                "replaced_pending_scenes": int(self._replaced),
                "publish_failures": int(self._failures),
                "pending_scene": bool(self._pending),
                "pending_sessions": len(self._pending),
            }

    def close(self, timeout_s=2.0):
        """Stop background work and wait briefly for the worker."""
        self._stop.set()
        self._ready.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout_s)))
        return not self._thread.is_alive()

    def _run(self):
        while not self._stop.is_set():
            self._ready.wait(timeout=0.25)
            self._ready.clear()
            if self._stop.is_set():
                break
            with self._lock:
                scene = None
                if self._pending:
                    _, scene = self._pending.popitem(last=False)
                has_more = bool(self._pending)
            if has_more:
                self._ready.set()
            if scene is None:
                continue
            try:
                self._publisher.publish(scene)
            except Exception as exc:  # transport failure must not kill control
                with self._lock:
                    self._failures += 1
                if self._on_error is not None:
                    self._on_error(exc)
            else:
                with self._lock:
                    self._published += 1

    @staticmethod
    def _default_key(scene):
        header = getattr(scene, "header", None)
        frame_id = getattr(header, "frame_id", "") if header is not None else ""
        return str(frame_id or "__default__")
