from types import SimpleNamespace

from pipeline.mpc_entry_stage import MPCEntryStage


def _behavior(maneuver="lane_follow", phase="LANE_KEEP", lane_id=7):
    return SimpleNamespace(
        maneuver=maneuver,
        phase=phase,
        target_lane_id=lane_id,
    )


def test_mode_transition_state_is_owned_by_mpc_entry_stage():
    stage = MPCEntryStage({})
    resets = []

    first = stage.apply_behavior_mode_transition(
        behavior=_behavior(),
        stop_goal_active=False,
        reset_control_buffer=lambda **kwargs: resets.append(kwargs),
    )
    repeated = stage.apply_behavior_mode_transition(
        behavior=_behavior(),
        stop_goal_active=False,
        reset_control_buffer=lambda **kwargs: resets.append(kwargs),
    )
    changed = stage.apply_behavior_mode_transition(
        behavior=_behavior("lane_change_left", "EXECUTE_LANE_CHANGE_LEFT", 8),
        stop_goal_active=False,
        reset_control_buffer=lambda **kwargs: resets.append(kwargs),
    )

    assert first == ""
    assert repeated == ""
    assert changed == (
        "mode_transition:lane_follow:7->lane_change_left:8:"
        "reset_control_buffer"
    )
    assert resets == [{"reason": "control_buffer_reset_mode_transition"}]


def test_stop_intent_has_one_mode_key_independent_of_lane():
    stage = MPCEntryStage({})

    assert stage.behavior_mode_key(
        decision="lane_follow",
        phase="LANE_KEEP",
        target_lane_id=4,
        stop_goal_active=True,
    ) == "stop"
    assert stage.behavior_mode_key(
        decision="emergency_brake",
        phase="FALLBACK",
        target_lane_id=9,
        stop_goal_active=False,
    ) == "stop"
