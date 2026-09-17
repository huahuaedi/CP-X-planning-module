from types import SimpleNamespace

from pipeline.performance_profiler import PlannerStageProfiler


class _ConflictPipeline:
    def cooperative_conflict_reference(self):
        return "reference"

    def resolve_cav_interaction(self):
        return "resolution"


def test_bridge_profiler_records_cooperative_conflict_boundaries():
    pipeline = _ConflictPipeline()
    bridge = SimpleNamespace(
        pipeline=pipeline,
        reference_generator=None,
        reference_pipeline=None,
    )
    profiler = PlannerStageProfiler(enabled=True)

    profiler.instrument_bridge(bridge)

    assert pipeline.cooperative_conflict_reference() == "reference"
    assert pipeline.resolve_cav_interaction() == "resolution"
    summary = profiler.summary(planning_cycle_count=1)
    assert summary["profile.cav_conflict.reference"]["call_count"] == 1
    assert summary["profile.cav_conflict.resolve_interaction"]["call_count"] == 1
