from types import SimpleNamespace

import pytest

from pipeline.execution_pipeline import PlanningPipeline
from pipeline.perception_stage import PerceptionStage
from pipeline.runtime_input_stage import RuntimeInputStage


class _Mapper:
    def update_measurement(self, **_kwargs):
        return 0.25


def test_pipeline_sequences_runtime_and_perception_without_bridge():
    pipeline = PlanningPipeline(
        runtime_input=RuntimeInputStage(_Mapper()),
        perception=PerceptionStage(max_mpc_obstacles=4),
    )
    transform = SimpleNamespace(
        location=SimpleNamespace(x=1.0, y=2.0, z=0.0),
        rotation=SimpleNamespace(yaw=0.0),
    )

    tick = pipeline.begin_tick(
        timestamp_s=3.0, ego_transform=transform, ego_speed_kmh=18.0
    )
    perception = pipeline.perceive(
        tick,
        detected_objects={"vehicles": [{
            "vehicle_id": "front", "x": 10.0, "y": 2.0, "v": 2.0,
        }]},
        cp_payload={},
        ignore_dynamic_objects=False,
    )

    assert tick.ego_speed_mps == pytest.approx(5.0)
    assert perception.front_actor_id == "front"
    assert len(perception.fused_objects) == 1
    assert not hasattr(pipeline, "bridge")
