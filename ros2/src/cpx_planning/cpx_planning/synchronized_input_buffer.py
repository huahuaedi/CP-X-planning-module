"""Assemble independently delivered ROS inputs into complete CARLA cycles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class SynchronizedInputFrame:
    """One complete planner input frame selected for execution."""

    session_id: str
    cycle_id: int
    messages: dict
    assembly_wait_ms: float


class SynchronizedInputBuffer:
    """Keep bounded per-cycle messages and return only the newest complete frame.

    ROS topics are delivered independently.  Storing only one latest message per
    topic can permanently misalign their timestamps whenever a planner solve
    overlaps more than one source tick.  This buffer instead keys messages by
    the common CARLA cycle ID.  A slow planner may skip old *complete* frames,
    but it never combines messages from different cycles.
    """

    def __init__(
        self,
        required_inputs,
        optional_inputs=(),
        max_frames=32,
        monotonic_ns=None,
    ):
        self.required_inputs = frozenset(str(name) for name in required_inputs)
        self.optional_inputs = frozenset(str(name) for name in optional_inputs)
        if not self.required_inputs:
            raise ValueError("At least one required input is needed.")
        self.max_frames = max(2, int(max_frames))
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        self._frames = OrderedDict()
        self._latest_optional = {}
        self._last_dispatched_cycle_ids = {}
        self._last_dispatched_monotonic_ns = {}
        self._last_selected_session_id = None
        self._dispatched_frames_by_session = {}
        self._condition = threading.Condition(threading.RLock())
        self._stats = {
            "received_messages": 0,
            "replaced_messages": 0,
            "stale_messages": 0,
            "overflow_frames": 0,
            "dispatched_frames": 0,
            "skipped_complete_frames": 0,
            "dropped_incomplete_frames": 0,
            "session_resets": 0,
        }

    def reset_for_new_session(self, session_id=None):
        """Discard ordering state for one session, or every session."""

        with self._condition:
            if session_id is None:
                self._frames.clear()
                self._latest_optional.clear()
                self._last_dispatched_cycle_ids.clear()
                self._last_dispatched_monotonic_ns.clear()
                self._last_selected_session_id = None
                self._dispatched_frames_by_session.clear()
            else:
                session_id = str(session_id or "")
                for key in list(self._frames):
                    if key[0] == session_id:
                        self._frames.pop(key, None)
                for key in list(self._latest_optional):
                    if key[0] == session_id:
                        self._latest_optional.pop(key, None)
                self._last_dispatched_cycle_ids.pop(session_id, None)
                self._last_dispatched_monotonic_ns.pop(session_id, None)
                self._dispatched_frames_by_session.pop(session_id, None)
                if self._last_selected_session_id == session_id:
                    self._last_selected_session_id = None
            self._stats["session_resets"] += 1
            self._condition.notify_all()

    def add(
        self,
        input_name,
        cycle_id,
        message,
        received_monotonic_ns=None,
        session_id="",
    ):
        """Insert one topic message without waiting for planner execution."""
        input_name = str(input_name)
        cycle_id = int(cycle_id)
        session_id = str(session_id or "")
        if input_name not in self.required_inputs and input_name not in self.optional_inputs:
            raise KeyError("Unknown synchronized input: {}".format(input_name))
        received_ns = int(received_monotonic_ns or self._monotonic_ns())
        with self._condition:
            self._stats["received_messages"] += 1
            last_dispatched = int(
                self._last_dispatched_cycle_ids.get(session_id, 0)
            )
            if cycle_id <= last_dispatched:
                self._stats["stale_messages"] += 1
                return False
            frame_key = (session_id, cycle_id)
            entry = self._frames.get(frame_key)
            if entry is None:
                entry = {
                    "first_received_ns": received_ns,
                    "messages": {},
                }
                self._frames[frame_key] = entry
            if input_name in entry["messages"]:
                self._stats["replaced_messages"] += 1
            entry["messages"][input_name] = message
            if input_name in self.optional_inputs:
                self._latest_optional[(session_id, input_name)] = (
                    cycle_id,
                    message,
                )
            self._trim_overflow_locked(session_id)
            complete = self._is_complete_locked(entry)
            if complete:
                self._condition.notify_all()
            return complete

    def pop_latest_complete(self):
        """Return the newest complete frame and discard all superseded frames."""
        with self._condition:
            return self._pop_latest_complete_locked()

    def wait_for_latest_complete(self, stop_event, timeout_s=0.25):
        """Wait until a complete frame exists or shutdown is requested."""
        timeout_s = max(0.001, float(timeout_s))
        with self._condition:
            while not stop_event.is_set():
                frame = self._pop_latest_complete_locked()
                if frame is not None:
                    return frame
                self._condition.wait(timeout_s)
            return None

    def wake_all(self):
        """Wake a waiting worker during node shutdown."""
        with self._condition:
            self._condition.notify_all()

    def stats(self):
        """Return counters plus the current bounded-buffer state."""
        with self._condition:
            result = dict(self._stats)
            result["buffered_frames"] = len(self._frames)
            result["last_dispatched_cycle_id"] = max(
                [0] + list(self._last_dispatched_cycle_ids.values())
            )
            result["active_sessions"] = len(
                set(key[0] for key in self._frames)
                | set(self._last_dispatched_cycle_ids)
            )
            result["last_dispatched_cycle_ids"] = dict(
                self._last_dispatched_cycle_ids
            )
            result["dispatched_frames_by_session"] = dict(
                self._dispatched_frames_by_session
            )
            return result

    def _is_complete_locked(self, entry):
        return self.required_inputs.issubset(entry["messages"])

    def _pop_latest_complete_locked(self):
        complete_by_session = {}
        for (session_id, cycle_id), entry in self._frames.items():
            if (
                cycle_id
                > int(self._last_dispatched_cycle_ids.get(session_id, 0))
                and self._is_complete_locked(entry)
            ):
                complete_by_session.setdefault(session_id, []).append(
                    cycle_id
                )
        if not complete_by_session:
            return None

        # Keep the newest frame for each session, then select sessions in
        # strict round-robin order. Comparing the arrival time of each
        # session's *newest* frame is not fair under overload: a session can
        # repeatedly replace its candidate just before selection and lose to
        # another session for several planner cycles. Round robin bounds that
        # scheduling gap to one turn per ready session while retaining the
        # drop-old-work policy within each session.
        candidates = {}
        for session_id, complete_ids in complete_by_session.items():
            newest_id = max(complete_ids)
            candidates[str(session_id)] = int(newest_id)
        ordered_sessions = sorted(candidates)

        # Earliest-deadline scheduling: serve the ready session that has
        # waited longest since its previous dispatch.  A strict lexical
        # round-robin is fair only while every session has a complete frame
        # on every turn.  If one CAV is briefly incomplete, it can miss its
        # turn and wait a second round even after becoming ready, which is
        # enough to expire a 0.5 s execution plan under four-CAV load.
        # Never-dispatched sessions receive the highest priority.  The
        # previous round-robin order remains the deterministic tie-breaker.
        def round_robin_rank(candidate_session):
            if self._last_selected_session_id is None:
                return ordered_sessions.index(candidate_session)
            later = [
                session
                for session in ordered_sessions
                if session > self._last_selected_session_id
            ]
            wrapped = [
                session
                for session in ordered_sessions
                if session <= self._last_selected_session_id
            ]
            return (later + wrapped).index(candidate_session)

        selected_session = min(
            ordered_sessions,
            key=lambda candidate_session: (
                int(self._last_dispatched_monotonic_ns.get(
                    candidate_session, -1
                )),
                round_robin_rank(candidate_session),
            ),
        )
        selected_id = candidates[selected_session]
        selected = self._frames[(selected_session, selected_id)]
        messages = dict(selected["messages"])
        for (session_id, name), (
            optional_cycle_id,
            message,
        ) in self._latest_optional.items():
            if session_id != selected_session:
                continue
            if name not in messages and optional_cycle_id <= selected_id:
                messages[name] = message

        self._stats["skipped_complete_frames"] += max(
            0, len(complete_by_session[selected_session]) - 1
        )
        for frame_key in list(self._frames):
            session_id, cycle_id = frame_key
            if session_id != selected_session:
                continue
            if cycle_id > selected_id:
                continue
            entry = self._frames.pop(frame_key)
            if cycle_id != selected_id and not self._is_complete_locked(entry):
                self._stats["dropped_incomplete_frames"] += 1
        self._last_dispatched_cycle_ids[selected_session] = int(selected_id)
        self._last_dispatched_monotonic_ns[selected_session] = int(
            self._monotonic_ns()
        )
        self._last_selected_session_id = str(selected_session)
        self._dispatched_frames_by_session[selected_session] = (
            int(self._dispatched_frames_by_session.get(selected_session, 0))
            + 1
        )
        self._stats["dispatched_frames"] += 1
        wait_ms = max(
            0.0,
            (int(self._monotonic_ns()) - int(selected["first_received_ns"]))
            / 1000000.0,
        )
        return SynchronizedInputFrame(
            session_id=str(selected_session),
            cycle_id=int(selected_id),
            messages=messages,
            assembly_wait_ms=float(wait_ms),
        )

    def _trim_overflow_locked(self, session_id):
        """Apply the configured capacity independently to each CAV."""
        session_keys = [
            key for key in self._frames if key[0] == str(session_id)
        ]
        while len(session_keys) > self.max_frames:
            oldest_key = session_keys.pop(0)
            entry = self._frames.pop(oldest_key)
            self._stats["overflow_frames"] += 1
            if not self._is_complete_locked(entry):
                self._stats["dropped_incomplete_frames"] += 1
