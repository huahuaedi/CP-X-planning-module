import pytest

from opencda.planning_module.pipeline.nominal_trajectory import (
    NominalTarget,
    NominalTrajectoryGenerator,
)


def test_nominal_target_names_legacy_slots_once():
    target = NominalTarget.from_legacy(
        [1.0, 2.0, 3.0, 0.4, 540156, 1.0, 42, 1.0]
    )
    assert target.lane_id == 540156
    assert target.road_id == 42
    assert target.entered_intersection is True
    assert target.as_legacy() == [1.0, 2.0, 3.0, 0.4, 540156, 1.0, 42, 1.0]


def test_generator_owns_immutable_reference_and_reset():
    owner = NominalTrajectoryGenerator()
    state = owner.update(
        target_state=[5.0, 0.0, 4.0, 0.0, 101],
        samples=[{"x_ref_m": 5.0, "y_ref_m": 0.0}],
        reference_freeze_count=2,
        source="test",
    )
    with pytest.raises(TypeError):
        state.samples[0]["x_ref_m"] = 9.0
    reset = owner.reset(source="route_changed")
    assert reset.target is None
    assert reset.samples == ()
    assert reset.reference_freeze_count == 0
