"""ActuatorPort owns every stop-control conversion the bridge hands to the MPC
execution stage.  The expected values below are the arithmetic that used to
live inline in CPXMPCPlannerBridge's lambdas, so moving it must not change a
single output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opencda_bridge.platform_ports import ActuatorPort  # noqa: E402


class _Control:
    def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
        self.throttle, self.brake, self.steer = throttle, brake, steer


def _port(*, min_accel=-4.0, max_steer=0.5, actuator_max_steer=0.7):
    return ActuatorPort(
        actuator_mapper=object(),
        constraints=SimpleNamespace(
            max_acceleration_mps2=2.0,
            min_acceleration_mps2=min_accel,
            max_steer_rad=max_steer,
        ),
        carla_module=SimpleNamespace(VehicleControl=_Control),
        clock=lambda: 0.0,
        actuator_max_steer_rad=actuator_max_steer,
    )


def test_safe_stop_brake_is_deceleration_over_the_mpc_limit():
    control = _port(min_accel=-4.0).safe_stop_control(-2.0, 0.0)
    assert control.throttle == 0.0
    assert control.brake == 0.5
    assert _port(min_accel=-4.0).safe_stop_control(-9.0, 0.0).brake == 1.0
    assert _port(min_accel=-4.0).safe_stop_control(1.5, 0.0).brake == 0.0


def test_safe_stop_steer_uses_the_mpc_bound_not_the_actuator_full_scale():
    # max_steer 0.5 vs actuator 0.7: 0.25 rad -> 0.5 (MPC bound), not 0.357.
    control = _port(max_steer=0.5, actuator_max_steer=0.7).safe_stop_control(
        -1.0, 0.25
    )
    assert control.steer == 0.5
    assert _port(max_steer=0.5).safe_stop_control(-1.0, 9.0).steer == 1.0
    assert _port(max_steer=0.5).safe_stop_control(-1.0, -9.0).steer == -1.0


def test_emergency_stop_is_full_brake_and_straight():
    control = _port().emergency_stop_control()
    assert (control.throttle, control.brake, control.steer) == (0.0, 1.0, 0.0)
