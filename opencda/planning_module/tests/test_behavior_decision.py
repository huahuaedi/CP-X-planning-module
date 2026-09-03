from opencda.planning_module.pipeline.behavior_decision import BehaviorDecision


def test_behavior_decision_freezes_behavior_values():
    raw = {
        "decision": "lane_change_right",
        "lc_state": "EXECUTE_LANE_CHANGE_RIGHT",
        "current_lane_id": 101,
        "target_lane_id": 202,
        "target_speed_mps": 8.0,
        "route_required": True,
    }
    decision = BehaviorDecision.from_mapping(raw, default_speed_mps=12.0)
    raw["target_lane_id"] = 303
    assert decision.maneuver == "lane_change_right"
    assert decision.direction == "right"
    assert decision.target_lane_id == 202
    assert decision.requested_speed_mps == 8.0
    assert decision.route_required is True
