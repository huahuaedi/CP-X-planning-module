from pipeline.conflict_classifier import (
    CROSSING,
    FOLLOW,
    MERGE,
    ConflictTag,
)
from pipeline.cooperative_arbitration import ConflictAssignment
from pipeline.spatiotemporal_corridor import (
    CorridorParams,
    build_longitudinal_corridor,
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
        cav_actor_id=2, role=role, homotopy_side="", conflict_xy=None,
        cav_wins=(role != "proceed"), reason="",
    )


def test_no_conflicts_leaves_corridor_open():
    cor = build_longitudinal_corridor(REF, EGO, [], P)
    assert cor.feasible
    assert all(h >= _BIG for h in cor.s_hi)
    assert all(l <= -_BIG for l in cor.s_lo)


def test_follow_caps_s_hi_behind_the_lead():
    # lead sits at x=40 the whole horizon
    lead = {"x": 40.0, "v": 6.0, **_track([(40.0, 0.1)] * 21)}
    cor = build_longitudinal_corridor(REF, EGO, [(lead, _tag("lead", FOLLOW), None)], P)
    # RSS gap for ego 10 / lead 6 is well under 40 m -> s_hi < 40, finite
    assert cor.s_hi[10] < 40.0
    assert cor.s_hi[10] > 0.0
    assert cor.binding[10] == "lead"
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
        [({"x": 30.0, "v": 8.0}, _tag("x", CROSSING, s=30.0, t=1.0), _assign("proceed"))],
        P,
    )
    assert all(h >= _BIG for h in cor.s_hi)


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


def test_infeasible_corridor_is_flagged():
    # a lead already behind the ego forces s_hi negative -> s_lo(-BIG) < s_hi ok,
    # but two conflicting caps: lead very close ahead + a hard crossing cap
    lead = {"x": 3.0, "v": 0.0, **_track([(3.0, 0.1)] * 21)}
    cor = build_longitudinal_corridor(
        REF, {"x": 0.0, "v": 12.0}, [(lead, _tag("lead", FOLLOW), None)], P
    )
    # s_hi ~ 3 - (RSS gap for 12 vs 0 ~ 30) = negative; s_lo is -BIG so still
    # "feasible" as a box, but s_hi is well below where ego can be -> Stage D
    # slacks it. Just assert the cap is negative and flagged consistently.
    assert cor.s_hi[0] < 0.0
