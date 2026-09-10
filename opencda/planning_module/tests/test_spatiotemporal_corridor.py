from pipeline.conflict_classifier import (
    CROSSING,
    FOLLOW,
    MERGE,
    ConflictTag,
)
from pipeline.cooperative_arbitration import ConflictAssignment
from pipeline.spatiotemporal_corridor import (
    Corridor,
    CorridorParams,
    build_longitudinal_corridor,
    rebase_corridor,
)

REF = [{"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(0, 121, 2)]
EGO = {"x": 0.0, "y": 0.0, "v": 10.0}
P = CorridorParams(horizon_steps=20, dt_s=0.1)

_BIG = 1.0e9


def _track(pts):
    return {"predicted_trajectory": [{"x": float(x), "y": float(y)} for x, y in pts]}


def _tag(agent_id, tag, s=None, t=None):
    return ConflictTag(
        agent_id=agent_id, tag=tag, conflict_s_m=s, conflict_t_s=t,
        min_gap_m=0.0, min_lateral_m=0.0, cooperative=False, reason="",
    )


def _assign(role):
    return ConflictAssignment(
        cav_actor_id=2, role=role, homotopy_side="left",
        cav_wins=(role != "proceed"), reason="",
    )


def test_no_conflicts_leaves_corridor_open():
    cor = build_longitudinal_corridor(REF, EGO, [], P)
    assert cor.feasible
    assert all(h >= _BIG for h in cor.s_hi)
    assert all(l <= -_BIG for l in cor.s_lo)


def test_cached_corridor_advances_time_and_rebases_station_to_current_reference():
    cached = Corridor(
        s_lo=[-_BIG] * 5,
        s_hi=[20.0 + k for k in range(5)],
        binding=["peer"] * 5,
    )
    current_ref = [
        {"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(5, 126, 2)
    ]
    rebased = rebase_corridor(
        cached, source_reference=REF, current_reference=current_ref,
        current_ego_xy=(5.0, 0.0), age_s=0.2, dt_s=0.1,
    )
    # Two prediction stages elapsed; the rolling reference now starts at ego,
    # so the old absolute cap is shifted back by ego's 5 m progress.
    assert rebased.s_hi[0] == 17.0
    assert rebased.binding[0] == "peer"


def test_follow_leaves_corridor_open_for_speed_planner():
    # lead sits at x=40 the whole horizon
    lead = {"x": 40.0, "v": 6.0, **_track([(40.0, 0.1)] * 21)}
    cor = build_longitudinal_corridor(REF, EGO, [(lead, _tag("lead", FOLLOW), None)], P)
    assert all(value >= _BIG for value in cor.s_hi)
    assert not any(cor.binding)
    assert cor.feasible


def test_crossing_yield_caps_s_hi_near_conflict_point_only_in_the_window():
    cor = build_longitudinal_corridor(
        REF, EGO,
        [({"x": 30.0, "v": 8.0}, _tag("x", CROSSING, s=30.0, t=1.0), _assign("yield"))],
        CorridorParams(horizon_steps=30, dt_s=0.1, crossing_clearance_time_s=0.5,
                       conflict_stop_buffer_m=4.0),
    )
    # window is t in [0.5, 1.5] s -> stages 5..15 capped at 30 - 4 = 26
    assert cor.s_hi[10] == 26.0
    assert cor.s_hi[0] >= _BIG      # before the window: uncapped
    assert cor.s_hi[25] >= _BIG     # after the window: uncapped


def test_crossing_proceed_adds_no_bound():
    cor = build_longitudinal_corridor(
        REF, EGO,
        [({"x": 30.0, "v": 8.0}, _tag("x", CROSSING, s=30.0, t=1.5), _assign("proceed"))],
        P,
    )
    assert all(h >= _BIG for h in cor.s_hi)


def test_imminent_crossing_overrides_stale_proceed_role():
    cor = build_longitudinal_corridor(
        REF, EGO,
        [({"x": 18.0, "v": 8.0},
          _tag("x", CROSSING, s=18.0, t=0.8), _assign("proceed"))],
        P,
    )
    assert min(cor.s_hi) == 14.0


def test_merge_make_gap_puts_ego_behind_cav():
    cav = {"x": 25.0, "v": 9.0, **_track([(25.0 + 0.9 * k, 0.2) for k in range(21)])}
    cor = build_longitudinal_corridor(
        REF, EGO, [(cav, _tag("cav", MERGE), _assign("make_gap"))], P
    )
    assert cor.s_hi[5] < 25.0 + 0.9 * 5
    assert cor.binding[5] == "cav"


def test_merge_proceed_adds_no_bound():
    cav = {"x": 25.0, "v": 9.0, **_track([(25.0, 0.2)] * 21)}
    cor = build_longitudinal_corridor(
        REF, EGO, [(cav, _tag("cav", MERGE), _assign("proceed"))], P
    )
    assert all(h >= _BIG for h in cor.s_hi)


def test_rear_merge_cannot_reverse_lead_vehicle_longitudinal_order():
    rear = {
        "x": -8.0, "y": 3.5, "v": 12.0,
        "predicted_trajectory": [
            {"x": -8.0 + 1.2 * k, "y": max(0.0, 3.5 - 0.3 * k)}
            for k in range(21)
        ],
    }
    cor = build_longitudinal_corridor(
        REF, EGO, [(rear, _tag("rear", MERGE, s=12.0, t=1.5), None)], P
    )
    assert all(h >= _BIG for h in cor.s_hi)


def test_unassigned_merge_has_a_safety_corridor_owner():
    cav = {"x": 18.0, "v": 8.0,
           **_track([(18.0 + 0.8 * k, 2.5 - 0.1 * k) for k in range(21)])}
    cor = build_longitudinal_corridor(
        REF, EGO, [(cav, _tag("cav", MERGE, s=24.0, t=0.8), None)], P
    )
    assert cor.binding[8] == "cav"
    assert cor.s_hi[8] < 18.0 + 0.8 * 8


def test_close_follow_does_not_create_a_second_stop_controller():
    lead = {"x": 3.0, "v": 0.0, **_track([(3.0, 0.1)] * 21)}
    cor = build_longitudinal_corridor(
        REF, {"x": 0.0, "v": 12.0}, [(lead, _tag("lead", FOLLOW), None)], P
    )
    assert all(value >= _BIG for value in cor.s_hi)
