from cpx_planning.pipeline.reference_gate import (
    REFERENCE_GATE_POLICY_REVISION,
    FinalReferenceGate,
)


def _reference(lateral_m):
    return [
        {
            "x_ref_m": float(index + 1),
            "y_ref_m": float(lateral_m),
            "heading_rad": 0.0,
            "lane_id": 1,
            "speed_ref_mps": 2.0,
        }
        for index in range(4)
    ]


def _validate(lateral_m):
    return FinalReferenceGate({}).validate(
        reference_samples=_reference(lateral_m),
        destination_state=[4.0, float(lateral_m), 2.0],
        ego_state=[0.0, 0.0, 2.0, 0.0],
        behavior_decision="intersection_turn_left",
        behavior_fsm_state="INTERSECTION_TURN_LEFT",
        current_lane_id=1,
        target_lane_id=2,
        stop_goal_active=False,
        horizon_steps=4,
        default_speed_mps=3.0,
    )


def test_policy_revision_is_reportable():
    assert REFERENCE_GATE_POLICY_REVISION == 2


def test_turn_accepts_near_spatial_anchor_despite_lateral_offset():
    result = _validate(1.5)
    assert result.accepted, result.reason
    assert result.mode == "intersection_turn"


def test_turn_rejects_distant_anchor_with_distance_reason():
    result = _validate(3.2)
    assert not result.accepted
    assert "first_lateral_out_of_contract" not in result.reason
    assert "first_distance_out_of_contract" in result.reason
