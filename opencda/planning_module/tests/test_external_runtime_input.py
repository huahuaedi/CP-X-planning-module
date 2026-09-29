from types import SimpleNamespace

import pytest

from opencda.planning_module.opencda_bridge.cpx_mpc_planner import (
    CPXMPCPlannerBridge,
)


def _bridge_without_assembly():
    bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
    bridge._latest_opencda_update = {}
    return bridge


def test_external_input_normalizes_units_and_copies_mutable_payloads():
    bridge = _bridge_without_assembly()
    obstacle = {"actor_id": "peer", "x": 2.0, "y": 3.0, "v": 4.0}
    cp_payload = {"obstacles": [{"actor_id": "shared"}]}
    transform = SimpleNamespace(location=SimpleNamespace(x=1.0, y=2.0, z=0.0))

    bridge.update_external_information(
        ego_transform=transform,
        ego_speed_mps=5.0,
        object_snapshots=[obstacle],
        cp_payload=cp_payload,
        sim_time_s=12.5,
    )

    installed = bridge._latest_opencda_update
    assert installed["external_runtime"] is True
    assert installed["ego_speed_kmh"] == pytest.approx(18.0)
    assert installed["detected_objects"] == {"vehicles": [obstacle]}
    assert installed["detected_objects"]["vehicles"][0] is not obstacle
    assert installed["cp_payload"] == cp_payload
    assert installed["cp_payload"] is not cp_payload
    assert bridge._sim_time_s() == pytest.approx(12.5)


def test_native_input_does_not_latch_external_clock():
    bridge = _bridge_without_assembly()
    bridge._sim_time_s = lambda: 7.25

    bridge.update_information(
        ego_transform=object(),
        ego_speed_kmh=9.0,
        detected_objects={},
    )

    assert bridge._latest_opencda_update["external_runtime"] is False
    assert bridge._latest_opencda_update["sim_time_s"] == pytest.approx(7.25)
