"""Regression coverage for bounded nested planner-stage instrumentation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cpx_planning/pipeline/performance_profiler.py"
)
_SPEC = importlib.util.spec_from_file_location("cpx_performance_profiler", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PlannerStageProfiler = _MODULE.PlannerStageProfiler


class _ReferenceGenerator:
    def _turn_footprint_validation_cache_key(self):
        return "key"

    def _lane_corridor_geometry(self):
        return "lane"

    def _sample_corridor_geometry(self):
        return self._lane_corridor_geometry()

    def _compute_drivable_footprint_occupancy(self):
        return "inside"

    def drivable_footprint_occupancy(self):
        return self._compute_drivable_footprint_occupancy()

    def validate_turn_swept_footprint(self):
        self._turn_footprint_validation_cache_key()
        self._sample_corridor_geometry()
        self.drivable_footprint_occupancy()
        return "valid"

    def ensure_turn_swept_footprint(self):
        return self.validate_turn_swept_footprint()


def test_bridge_profiler_splits_turn_footprint_substages():
    generator = _ReferenceGenerator()
    bridge = SimpleNamespace(
        reference_generator=generator,
        reference_pipeline=None,
    )
    profiler = PlannerStageProfiler(enabled=True)

    profiler.instrument_bridge(bridge)

    assert generator.ensure_turn_swept_footprint() == "valid"
    summary = profiler.summary(planning_cycle_count=1)
    expected = {
        "profile.reference.turn_footprint_ensure",
        "profile.reference.turn_footprint_validation",
        "profile.reference.turn_footprint.cache_key",
        "profile.reference.turn_footprint.corridor_geometry",
        "profile.reference.turn_footprint.drivable_occupancy",
        "profile.reference.turn_footprint.drivable_occupancy_compute",
        "profile.reference.turn_footprint.lane_geometry_lookup",
    }
    assert expected.issubset(summary)
    assert all(summary[label]["call_count"] == 1 for label in expected)


def test_disabled_profiler_does_not_replace_methods():
    generator = _ReferenceGenerator()
    original = generator.validate_turn_swept_footprint
    profiler = PlannerStageProfiler(enabled=False)

    profiler.instrument_bridge(
        SimpleNamespace(reference_generator=generator, reference_pipeline=None)
    )

    assert generator.validate_turn_swept_footprint == original
