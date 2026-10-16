"""Wall-clock pacing for synchronous simulation ticks."""

import time


class WorldTickPacer(object):
    """Pace the single CARLA world tick owner without accumulating backlog."""

    def __init__(self, period_s, enabled, monotonic=None, sleeper=None):
        self.period_s = max(0.0, float(period_s))
        self.enabled = bool(enabled) and self.period_s > 0.0
        self._monotonic = monotonic or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._next_tick_wall_time_s = None

    def wait(self):
        if not self.enabled:
            return
        now_s = float(self._monotonic())
        if self._next_tick_wall_time_s is None:
            self._next_tick_wall_time_s = now_s
        elif now_s - self._next_tick_wall_time_s > self.period_s:
            # Do not issue a burst of catch-up ticks after a slow planning
            # cycle. Resume the cadence from the current wall-clock time.
            self._next_tick_wall_time_s = now_s
        delay_s = float(self._next_tick_wall_time_s) - now_s
        if delay_s > 0.0:
            self._sleeper(delay_s)
        self._next_tick_wall_time_s += self.period_s
