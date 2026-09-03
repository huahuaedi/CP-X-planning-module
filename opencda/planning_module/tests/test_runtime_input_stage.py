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
