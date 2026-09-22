from types import SimpleNamespace

import pytest

from pipeline.speed_planning_stage import (
    SpeedPlanningRequest,
    SpeedPlanningStage,
)


class _Perception:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def front_gap(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _Speed:
    def __init__(self):
        self.calls = []

    def propose(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            target_speed_mps=kwargs["requested_speed_mps"],
            stop_goal_active=False,
        )


def _request(**overrides):
    values = dict(
        ego_location=SimpleNamespace(x=0.0, y=0.0),
        ego_yaw_rad=0.0,
        ego_speed_mps=5.0,
        requested_speed_mps=8.0,
        object_snapshots=(),
        current_lane_id=10,
        target_lane_id=10,
        lane_assignments={},
        lane_change_progress=0.0,
        lane_change_commitment_active=False,
        committed_source_lane_id=0,
        scenario_decision=SimpleNamespace(stop_goal_active=False),
        behavior_decision="lane_follow",
        upcoming_turn_direction="",
        upcoming_turn_distance_m=float("inf"),
        config={},
    )
    values.update(overrides)
    return SpeedPlanningRequest(**values)


def test_stage_builds_one_speed_proposal_from_the_selected_front_actor():
    perception = _Perception((12.0, "lead"))
    speed = _Speed()
    stage = SpeedPlanningStage(perception=perception, speed=speed)
    request = _request(
        object_snapshots=({"track_id": "lead", "v": 3.5},),
        lane_assignments={"lead": 10},
        upcoming_turn_direction="left",
        upcoming_turn_distance_m=20.0,
    )

    result = stage.run(request)

    assert result.front_gap_m == pytest.approx(12.0)
    assert result.front_actor_id == "lead"
    assert result.front_obstacle_speed_mps == pytest.approx(3.5)
    assert result.front_obstacle_lane_id == 10
    assert not result.front_obstacle_is_source_lane
    assert len(speed.calls) == 1
    assert speed.calls[0]["front_gap_m"] == pytest.approx(12.0)
    assert speed.calls[0]["upcoming_turn_distance_m"] == pytest.approx(20.0)


def test_committed_lane_change_marks_a_lead_in_the_source_lane():
    perception = _Perception((9.0, "source-lead"))
    speed = _Speed()
    stage = SpeedPlanningStage(perception=perception, speed=speed)

    result = stage.run(_request(
        behavior_decision="lane_change_left",
        current_lane_id=20,
        target_lane_id=30,
        lane_change_commitment_active=True,
        committed_source_lane_id=20,
        lane_assignments={"source-lead": 20},
        object_snapshots=({
            "vehicle_id": "source-lead", "speed_mps": 2.0,
        },),
    ))

    assert result.front_obstacle_is_source_lane
    assert speed.calls[0]["front_obstacle_is_source_lane"] is True
    assert speed.calls[0]["lane_change_commitment_active"] is True
    assert perception.calls[0]["lane_change_direction"] == "left"


def test_non_finite_turn_distance_is_removed_at_the_stage_boundary():
    perception = _Perception((None, None))
    speed = _Speed()
    stage = SpeedPlanningStage(perception=perception, speed=speed)

    result = stage.run(_request())

    assert result.front_actor_id == ""
    assert result.front_obstacle_speed_mps is None
    assert speed.calls[0]["upcoming_turn_distance_m"] is None
