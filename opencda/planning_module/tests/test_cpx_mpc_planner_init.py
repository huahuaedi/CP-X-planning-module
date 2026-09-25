"""Characterization test for CPXMPCPlannerBridge.__init__.

Every other test in this suite constructs the bridge via
``CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)`` plus manually-set mock
attributes, bypassing ``__init__`` entirely -- so nothing exercises the real
constructor today. This test is the safety net for splitting that 791-line
method into smaller pieces: it must keep passing, unmodified, before and
after any such refactor.
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
        # Collection order across the test suite decides which test file's
        # fake `carla` module sys.modules["carla"] ends up holding (each
        # only sets it `if "carla" not in sys.modules`). A minimal fake here
        # that only covers this file's own needs can silently break an
        # unrelated test that happens to collect afterwards and needs some
        # other `carla.X` attribute at import time. Fall back to a generic
        # stub for anything not explicitly defined above instead.
        return _GenericStub

    fake_carla.__getattr__ = _module_getattr
    sys.modules["carla"] = fake_carla

from opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MAPS_DIR = REPO_ROOT / "opencda" / "planning_module" / "Global_Planner" / "maps"
_TOWN06_XODR = MAPS_DIR / "Town06.xodr"


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
    cache_root = tempfile.mkdtemp(prefix="cpx_mpc_planner_init_test_")
    config = {
        "enabled": True,
        "debug": False,
        "record_evaluation_metrics": False,
        "publish_cp_message": False,
        "global_planner_xodr_path": str(_TOWN06_XODR),
        "global_planner_cache_root": cache_root,
    }
    return CPXMPCPlannerBridge(_VehicleManager(), config)


def test_construction_does_not_raise(bridge):
    assert bridge is not None


def test_stage_timing_snapshot_and_waypoint_counters_are_per_tick(bridge):
    bridge._begin_stage_timing_cycle()
    bridge._accum_stage_ms("stage_a", 0.001)
    bridge._accum_stage_ms("stage_a", 0.002)
    bridge._accum_stage_ms("stage_b", 0.004)

    assert bridge._stage_timing_snapshot() == {
        "stage_a": pytest.approx(3.0),
        "stage_b": pytest.approx(4.0),
    }
    bridge._waypoint_cache_hits_current = 3
    bridge._waypoint_cache_misses_current = 1

    bridge._begin_stage_timing_cycle()
    assert bridge._stage_timing_snapshot() == {}
    assert bridge._waypoint_cache_hits_current == 0
    assert bridge._waypoint_cache_misses_current == 0


def test_mpc_is_configured_with_bridge_vehicle_geometry(bridge):
    assert bridge.mpc.ego_width_m == pytest.approx(2.0)
    assert bridge.mpc.ego_length_m == pytest.approx(4.8)


def test_reference_contract_reserves_closed_loop_curvature_authority(bridge):
    expected = 0.90 * bridge.mpc.maximum_path_curvature_1pm()
    assert bridge.config["reference_vehicle_max_curvature_1pm"] == pytest.approx(
        expected
    )


def test_pipeline_wires_all_stage_owners(bridge):
    pipeline = bridge.pipeline
    for attr in (
        "_runtime_input", "_perception", "behavior", "scenario",
        "static_obstacle", "control_safety", "speed", "destination_speed",
            "reference_publication", "mpc_entry", "mpc_cost_profile", "fallback",
        "behavior_reference_execution", "reference_planning", "cooperative",
        "control_finalization", "mpc_execution",
    ):
        assert getattr(pipeline, attr, None) is not None, (
            f"pipeline.{attr} was never assigned"
        )


def test_global_planner_and_route_manager_share_the_same_backend(bridge):
    assert bridge.global_planner is bridge.topology_map
    assert bridge.global_planner is bridge.waypoint_map_planner
    assert bridge.route_manager.global_planner is bridge.global_planner


def test_candidate_selection_stage_receives_lane_change_lifecycle(bridge):
    assert (
        bridge.pipeline.reference_planning.candidate_selection._lane_change_lifecycle
        is bridge.lane_change_lifecycle_stage
    )


def test_cooperative_stage_depends_on_explicit_ports_not_full_pipeline(bridge):
    cooperative = bridge.pipeline.cooperative
    assert not hasattr(bridge, "_cooperative")
    assert not hasattr(cooperative, "_pipeline")
    assert callable(cooperative._build_conflict_reference)
    assert callable(cooperative._resolve_interaction)
    assert cooperative._resolve_interaction.__module__.endswith(
        "cav_interaction_stage"
    )


def test_control_finalization_shares_mpc_command_extractor_and_control_safety(bridge):
    control_finalization = bridge.pipeline.control_finalization
    assert control_finalization._extractor is bridge.mpc_command_extractor
    assert control_finalization._safety is bridge.pipeline.control_safety


def test_cp_provider_disabled_by_config(bridge):
    assert bridge.cp_provider is None


def test_cav_conflict_governor_starts_at_the_configured_ceiling(bridge):
    governor = bridge.pipeline.cooperative.governor
    assert governor.current_max_relevant_agents == 6
    assert governor.current_max_modes_per_agent == 6
