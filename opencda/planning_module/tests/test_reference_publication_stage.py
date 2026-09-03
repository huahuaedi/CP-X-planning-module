from types import SimpleNamespace

from pipeline.behavior_decision import BehaviorDecision
from pipeline.reference_publication_stage import ReferencePublicationStage


class _Pipeline:
    def __init__(self, *, accepted=True):
        self.accepted = accepted

    def finalize(self, request):
        gate = SimpleNamespace(
            accepted=self.accepted,
            reason="" if self.accepted else "strict_reference_veto",
        )
        return SimpleNamespace(
            destination_state=list(request.destination_state),
            reference_samples=list(request.reference_samples),
            conditioning_reason="conditioned",
            gate=gate,
            as_debug_fields=lambda: {"reference_pipeline_stage": "validated"},
        )


class _Provider:
    def publish(self, request, samples, **kwargs):
        return SimpleNamespace(
            mode="lane_follow",
            reason="published" if kwargs["valid"] else "retained",
            geometry_revision=4,
            accepted=kwargs["valid"],
            mutable_samples=lambda: [dict(row) for row in samples],
        )


def _run(*, accepted):
    stage = ReferencePublicationStage(
        reference_pipeline=_Pipeline(accepted=accepted),
        reference_provider=_Provider(),
    )
    behavior = BehaviorDecision.from_mapping({
        "decision": "lane_follow",
        "current_lane_id": 1,
        "target_lane_id": 1,
        "target_speed_mps": 4.0,
    }, default_speed_mps=4.0)
    return stage.run(
        destination_state=[2.0, 0.0, 4.0, 0.0, 1],
        reference_samples=[
            {"x_ref_m": 1.0, "y_ref_m": 0.0},
            {"x_ref_m": 2.0, "y_ref_m": 0.0},
        ],
        current_state=[0.0, 0.0, 1.0, 0.0],
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.0,
        ego_speed_mps=1.0,
        target_speed_mps=4.0,
        behavior=behavior,
        stop_goal_active=False,
        route_points=[],
        local_map=SimpleNamespace(),
        route_cursor=SimpleNamespace(segment_kind="lane_follow"),
        route_revision="route-1",
        map_epoch="town06",
        reference_source="unit_reference",
        candidate_reason="candidate_ok",
    )


def test_reference_publication_stage_returns_published_reference():
    result = _run(accepted=True)
    assert result.gate.accepted
    assert result.debug_fields["reference_provider_geometry_revision"] == 4
    assert result.debug_fields["final_reference_geometry_source"] == "unit_reference"


def test_reference_publication_stage_submits_rejection_as_data():
    result = _run(accepted=False)
    assert not result.gate.accepted
    assert result.debug_fields["candidate_pipeline_selected_status"] == "infeasible"
    assert "strict_reference_veto" in result.stabilizer_reason
