from opencda.scenario_testing.utils.tick_pacing import WorldTickPacer


class _FakeClock(object):
    def __init__(self):
        self.now_s = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now_s

    def sleep(self, duration_s):
        self.sleeps.append(float(duration_s))
        self.now_s += float(duration_s)


def test_world_tick_pacer_holds_requested_wall_rate():
    clock = _FakeClock()
    pacer = WorldTickPacer(
        0.05,
        True,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    pacer.wait()
    clock.now_s += 0.01
    pacer.wait()

    assert clock.sleeps == [0.04]
    assert abs(clock.now_s - 0.05) < 1.0e-9


def test_world_tick_pacer_does_not_burst_after_overrun():
    clock = _FakeClock()
    pacer = WorldTickPacer(
        0.05,
        True,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    pacer.wait()
    clock.now_s = 0.20
    pacer.wait()
    clock.now_s = 0.21
    pacer.wait()

    assert len(clock.sleeps) == 1
    assert abs(clock.sleeps[0] - 0.04) < 1.0e-9
