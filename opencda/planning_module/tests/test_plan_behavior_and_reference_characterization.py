"""Characterization test for CPXMPCPlannerBridge._plan_behavior_and_reference.

This 925-line method is the sole per-tick behavior+reference orchestrator
(route context -> scenario observation -> conflict resolution -> behavior
command -> speed proposal -> cooperative arbitration -> candidate selection
-> turn/speed constraints -> reference finalize). Nothing in this suite
calls it today: existing tests construct the bridge via ``__new__`` and
exercise smaller private helpers directly. This test is the safety net for
splitting it into smaller phase methods -- it must keep passing, unmodified,
before and after any such refactor.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pytest

if "carla" not in sys.modules:
    fake_carla = types.ModuleType("carla")

    class _Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class _Rotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch = pitch
            self.yaw = yaw
            self.roll = roll

    class _Transform:
        def __init__(self, location=None, rotation=None):
            self.location = location or _Location()
            self.rotation = rotation or _Rotation()

    class _VehicleControl:
        def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
            self.throttle = throttle
            self.brake = brake
            self.steer = steer

    fake_carla.Location = _Location
    fake_carla.Rotation = _Rotation
    fake_carla.Transform = _Transform
    fake_carla.VehicleControl = _VehicleControl

    class _GenericStub:
        def __init__(self, *args, **kwargs):
            pass

    def _module_getattr(name):
        # See test_cpx_mpc_planner_init.py: whichever test file's fake carla
        # module wins sys.modules["carla"] must not break an unrelated test
        # collected afterwards that needs some other carla.X at import time.
        return _GenericStub

    fake_carla.__getattr__ = _module_getattr
    sys.modules["carla"] = fake_carla

from opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MAPS_DIR = REPO_ROOT / "opencda" / "planning_module" / "Global_Planner" / "maps"
_TOWN06_XODR = MAPS_DIR / "Town06.xodr"

# Same start/goal as test_layer0_route_topology.py's cpx_single_right_lane_turn
# fixture -- a known-good Town06 route, not chosen fresh for this test.
START_XYZ = {"x": 225.10, "y": -20.1, "z": 0.3}
GOAL_XYZ = {"x": 10.113184332893075, "y": -96.89902155894256, "z": 0.3}


class _Extent:
    x = 2.25
    y = 1.0


class _BoundingBox:
    extent = _Extent()


class _Vehicle:
    id = 1

    def __init__(self):
        self.bounding_box = _BoundingBox()

    def get_transform(self):
        import carla

        return carla.Transform()


class _Controller:
    def lon_run_step(self, *args, **kwargs):
        return 0.0, 0.0


class _VehicleManager:
    def __init__(self):
        self.vehicle = _Vehicle()
        self.controller = _Controller()
        self.carla_map = None


@pytest.fixture(scope="module")
def bridge():
    if not _TOWN06_XODR.is_file():
        pytest.skip(f"fixture map not found: {_TOWN06_XODR}")
    cache_root = tempfile.mkdtemp(prefix="cpx_plan_behavior_test_")
    config = {
        "enabled": True,
        "debug": False,
        "record_evaluation_metrics": False,
        "publish_cp_message": False,
        "global_planner_xodr_path": str(_TOWN06_XODR),
        "global_planner_cache_root": cache_root,
    }
    built = CPXMPCPlannerBridge(_VehicleManager(), config)
    built.route_manager.set_destination(
        start_point=START_XYZ, goal_point=GOAL_XYZ,
    )
    return built


def _ego_location():
    import carla

    return carla.Location(**START_XYZ)


def _call(bridge):
    return bridge._plan_behavior_and_reference(
        ego_location=_ego_location(),
        ego_yaw_rad=0.0,
        ego_speed_mps=5.0,
        speed_ref_mps=8.0,
        object_snapshots=[],
        stop_goal_active=False,
        cp_payload=None,
    )


def test_does_not_raise(bridge):
    result = _call(bridge)
    assert result is not None


def test_result_shape_is_stable(bridge):
    result = _call(bridge)
    assert isinstance(result.destination_state, tuple)
    assert isinstance(result.reference_samples, tuple)
    assert result.behavior_stage_result is not None
    assert isinstance(result.reference_debug, dict)


def test_reference_samples_are_dicts_with_xy(bridge):
    result = _call(bridge)
    assert len(result.reference_samples) > 0
    first = result.reference_samples[0]
    assert "x_ref_m" in first or "x" in first


def test_is_deterministic_for_the_same_frozen_input(bridge):
    first = _call(bridge)
    second = _call(bridge)
    assert first.destination_state == second.destination_state
    assert first.reference_samples == second.reference_samples
    assert (
        first.behavior_stage_result.decision.maneuver
        == second.behavior_stage_result.decision.maneuver
    )


def test_cav_conflict_governor_resets_when_the_route_changes():
    # A compute budget degraded by a complex intersection must not keep
    # constraining an unrelated later route -- confirmed against the real
    # bridge tick, not just CAVConflictComputeGovernor in isolation, since
    # the wiring (route_manager.route_revision -> governor.reset()) lives in
    # _plan_behavior_and_reference, not the governor itself.
    if not _TOWN06_XODR.is_file():
        pytest.skip(f"fixture map not found: {_TOWN06_XODR}")
    cache_root = tempfile.mkdtemp(prefix="cpx_governor_reset_test_")
    config = {
        "enabled": True, "debug": False, "record_evaluation_metrics": False,
        "publish_cp_message": False,
        "global_planner_xodr_path": str(_TOWN06_XODR),
        "global_planner_cache_root": cache_root,
        "cav_conflict_enabled": True,
    }
    bridge = CPXMPCPlannerBridge(_VehicleManager(), config)
    bridge.route_manager.set_destination(start_point=START_XYZ, goal_point=GOAL_XYZ)
    _call(bridge)
    assert bridge._cav_conflict_governor_route_revision == str(
        bridge.route_manager.route_revision
    )

    for _ in range(20):
        bridge._cav_conflict_governor.observe_stage_ms(500.0)
    assert bridge._cav_conflict_governor.current_max_relevant_agents < 6

    bridge.route_manager.set_destination(start_point=START_XYZ, goal_point=GOAL_XYZ)
    _call(bridge)
    assert bridge._cav_conflict_governor.current_max_relevant_agents == 6
