import math
from types import SimpleNamespace

import pytest

from pipeline.runtime_input_stage import RuntimeInputStage


class _ActuatorMapper:
    def __init__(self):
        self.calls = []

    def update_measurement(self, **kwargs):
        self.calls.append(dict(kwargs))
        return -0.75


def test_runtime_input_stage_builds_one_immutable_ego_state():
    mapper = _ActuatorMapper()
    stage = RuntimeInputStage(mapper)
    transform = SimpleNamespace(
        location=SimpleNamespace(x=12.0, y=-3.0, z=0.2),
        rotation=SimpleNamespace(yaw=90.0),
    )

    snapshot = stage.build(
        timestamp_s=4.5,
        ego_transform=transform,
        ego_speed_kmh=36.0,
    )

    assert snapshot.ego_speed_mps == pytest.approx(10.0)
    assert snapshot.ego_yaw_rad == pytest.approx(math.pi / 2.0)
    assert snapshot.measured_accel_mps2 == pytest.approx(-0.75)
    assert snapshot.current_state == pytest.approx((12.0, -3.0, 10.0, math.pi / 2.0))
    assert mapper.calls == [{"speed_mps": 10.0, "timestamp_s": 4.5}]

    with pytest.raises(Exception):
        snapshot.ego_speed_mps = 2.0


def _tf(yaw=0.0):
    return SimpleNamespace(
        location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        rotation=SimpleNamespace(yaw=yaw),
    )


def _speed(stage, kmh, t):
    return stage.build(timestamp_s=t, ego_transform=_tf(), ego_speed_kmh=kmh).ego_speed_mps


def test_first_tick_and_reset_gap_trust_the_raw_speed():
    stage = RuntimeInputStage(_ActuatorMapper(), max_upward_accel_mps2=6.0)
    assert _speed(stage, 3.6 * 11.4, 5.0) == pytest.approx(11.4)     # first tick
    # large clock gap -> treated as a reset, raw trusted even if it jumps
    assert _speed(stage, 3.6 * 20.0, 25.0) == pytest.approx(20.0)


def test_upward_speed_spike_is_clamped_to_the_accel_bound():
    stage = RuntimeInputStage(_ActuatorMapper(), max_upward_accel_mps2=6.0)
    _speed(stage, 3.6 * 11.40, 5.00)
    # CARLA reports +0.8 m/s in one 0.05 s tick (= +16 m/s^2): impossible.
    # Allowed step is 6.0 * 0.05 = 0.30 m/s.
    assert _speed(stage, 3.6 * 12.20, 5.05) == pytest.approx(11.70)
    # a persistent real change is tracked over a few ticks
    assert _speed(stage, 3.6 * 12.20, 5.10) == pytest.approx(12.00)
    assert _speed(stage, 3.6 * 12.20, 5.15) == pytest.approx(12.20)


def test_downward_change_passes_through_untouched():
    stage = RuntimeInputStage(_ActuatorMapper(), max_upward_accel_mps2=6.0)
    _speed(stage, 3.6 * 12.0, 5.00)
    # hard braking / collision: full drop must reach the planner immediately
    assert _speed(stage, 3.6 * 4.0, 5.05) == pytest.approx(4.0)
    assert _speed(stage, 3.6 * 0.0, 5.10) == pytest.approx(0.0)


def test_disabled_when_bound_is_zero():
    stage = RuntimeInputStage(_ActuatorMapper(), max_upward_accel_mps2=0.0)
    _speed(stage, 3.6 * 10.0, 5.00)
    assert _speed(stage, 3.6 * 30.0, 5.05) == pytest.approx(30.0)
