"""Characterization test for CPXMPCPlannerBridge._run_full_cpx_pipeline_step.

This ~500-line method is the other per-tick orchestrator alongside
_plan_behavior_and_reference (see test_plan_behavior_and_reference_
characterization.py): OpenCDAPlanningAdapter -> PlanningPipeline ->
PlannerOutput, covering begin_cycle, execute_behavior_reference,
apply_destination, resolve_speed, prepare_trajectory_execution and
execute_mpc in one call. Nothing in this suite calls it (or its public
wrappers execute_planning_pipeline()/run_step()) today -- existing tests
construct the bridge via __new__ and exercise smaller private helpers
directly. This test is the safety net for splitting it into smaller phase
methods -- it must keep passing, unmodified, before and after any such
refactor.
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

# Same fixture route as test_plan_behavior_and_reference_characterization.py.
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

        return carla.Transform(carla.Location(**START_XYZ), carla.Rotation(yaw=0.0))

    def get_world(self):
        raise AssertionError(
            "cp_provider must be disabled (publish_cp_message=False) so "
            "the pipeline step never needs a real CARLA world"
        )


class _Controller:
    def lon_run_step(self, *args, **kwargs):
        return 0.0, 0.0


class _Localizer:
    def get_ego_pos(self):
        import carla

        return carla.Transform(carla.Location(**START_XYZ), carla.Rotation(yaw=0.0))

    def get_ego_spd(self):
        return 18.0  # km/h


class _PerceptionManager:
    objects = {"vehicles": [], "traffic_lights": []}


class _V2XManager:
    cav_nearby = {}


class _VehicleManager:
    def __init__(self):
        self.vehicle = _Vehicle()
        self.controller = _Controller()
        self.carla_map = None
        self.localizer = _Localizer()
        self.perception_manager = _PerceptionManager()
        self.v2x_manager = _V2XManager()


@pytest.fixture(scope="module")
def bridge():
    if not _TOWN06_XODR.is_file():
        pytest.skip(f"fixture map not found: {_TOWN06_XODR}")
    cache_root = tempfile.mkdtemp(prefix="cpx_run_full_step_test_")
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


def test_does_not_raise(bridge):
    output = bridge.execute_planning_pipeline()
    assert output is not None


def test_output_shape_is_stable(bridge):
    output = bridge.execute_planning_pipeline()
    assert output.control is not None
    diagnostics = output.diagnostics_dict()
    assert isinstance(diagnostics, dict)
    assert "sim_time_s" in diagnostics
    assert "accel_cmd_mps2" in diagnostics
    assert "steer_cmd_rad" in diagnostics


def test_run_step_returns_a_control_and_records_debug(bridge):
    control = bridge.run_step()
    assert hasattr(control, "throttle")
    assert hasattr(control, "brake")
    assert hasattr(control, "steer")
    assert bridge.last_debug
    assert bridge.last_output is not None


def test_consecutive_ticks_both_succeed(bridge):
    # Unlike _plan_behavior_and_reference (a pure function of its explicit
    # kwargs), this method takes no arguments and reads/advances mutable
    # bridge state every call -- MPC jerk-seeding from the previous tick's
    # commanded acceleration, route cursor progression, stage-timing
    # counters. Exact output equality across calls is therefore the wrong
    # invariant; what must hold is that a second tick, run right after the
    # first against the same otherwise-frozen fixture, still produces a
    # valid result instead of raising.
    first = bridge.execute_planning_pipeline()
    second = bridge.execute_planning_pipeline()
    assert first.control is not None
    assert second.control is not None
    assert isinstance(second.diagnostics_dict(), dict)


def test_reference_uses_the_freshly_built_local_map_not_a_stale_one(bridge):
    # Regression: PlanningTickAdapters.local_map_snapshot must be re-read
    # after this tick's route-context build runs, not captured once before
    # it. A stale/default snapshot degrades the published reference into a
    # near-empty placeholder (no lane_width_m/boundary_source/curvature --
    # see ReferenceLineProvider.lane_fallback_reference / straight_samples),
    # which still "does not raise" but silently feeds MPC geometry that
    # isn't the real candidate-selected lane.
    output = bridge.execute_planning_pipeline()
    assert output.reference_trajectory, "expected a published reference"
    first = output.reference_trajectory[0]
    assert float(first.get("lane_width_m", 0.0)) > 0.0
    assert str(first.get("boundary_source", "")) != ""


if __name__ == "__main__":
    import unittest

    unittest.main()
