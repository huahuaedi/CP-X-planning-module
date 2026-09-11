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
    aggregate_mode_corridors,
    build_longitudinal_corridor,
    retain_pending_corridor,
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


def test_pending_corridor_can_tighten_but_not_revoke_future_rows():
    previous = Corridor(
        s_lo=[-_BIG] * 4,
        s_hi=[_BIG, 12.0, 18.0, _BIG],
        binding=["", "old", "old", ""],
    )
    refreshed = Corridor(
        s_lo=[-_BIG] * 4,
        s_hi=[_BIG, _BIG, 15.0, _BIG],
        binding=["", "", "new", ""],
    )
    retained = retain_pending_corridor(refreshed, previous)
    assert retained.s_hi == [_BIG, 12.0, 15.0, _BIG]
    assert retained.binding == ["", "old", "new", ""]


def test_follow_leaves_corridor_open_for_speed_planner():
    # lead sits at x=40 the whole horizon
    lead = {"x": 40.0, "v": 6.0, **_track([(40.0, 0.1)] * 21)}
    cor = build_longitudinal_corridor(REF, EGO, [(lead, _tag("lead", FOLLOW), None)], P)
    assert all(value >= _BIG for value in cor.s_hi)
    assert not any(cor.binding)
    assert cor.feasible


def test_prediction_follow_builds_time_indexed_safety_envelope():
    lead = {"x": 30.0, "v": 6.0,
            **_track([(30.0 + 0.6 * k, 0.1) for k in range(21)])}
    cor = build_longitudinal_corridor(
        REF, EGO, [(lead, _tag("lead", FOLLOW), None)], P,
        constrain_follow=True,
    )
    assert any(value < _BIG for value in cor.s_hi)
    assert any(binding == "lead" for binding in cor.binding)


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


def test_crossing_cap_tighter_than_braking_is_floored_at_the_reachable_station():
    # Reproduces the captured cpx_town05_crossing_late_conflict frame
    # (sim_time 136.559s): ego at 6.79 m/s, a CROSSING yield whose window
    # covers the whole horizon and whose geometric cap (conflict_s_m -
    # buffer) freezes at 5.1 m -- 4.52 m/s^2 to stop there, tighter than the
    # 3.0 m/s^2 MPC braking limit. Without the floor this cap reached every
    # stage unchanged and Stage D's tangent-only row let the QP satisfy it
    # by rotating heading instead of braking (the observed deflection).
    ego = {"x": 0.0, "y": 0.0, "v": 6.791947, "psi": 0.0}
    params = CorridorParams(
        horizon_steps=32, dt_s=0.1,
        crossing_clearance_time_s=5.0, conflict_stop_buffer_m=4.0,
        max_braking_mps2=3.0,
    )
    cor = build_longitudinal_corridor(
        REF, ego,
        [({"x": 9.2, "v": 6.0}, _tag("x", CROSSING, s=9.2, t=0.0), _assign("yield"))],
        params,
    )
    geometric_cap = 9.2 - 4.0  # == 5.2, close to the captured 5.1 m freeze
    assert any(h > geometric_cap + 1.0e-6 for h in cor.s_hi[1:]), (
        "expected at least one early stage floored above the geometric cap"
    )
    for k, value in enumerate(cor.s_hi):
        t = k * params.dt_s
        floor = 6.791947 * t - 0.5 * 3.0 * t * t if t < 6.791947 / 3.0 else (
            6.791947 ** 2 / (2 * 3.0)
        )
        assert value >= min(floor, geometric_cap) - 1.0e-6
    # Full stopping distance (v^2 / 2a) is reached well before the horizon
    # ends; from there the floor plateaus and stays below the geometric cap
    # forever, so the corridor is correctly and permanently flagged.
    assert not cor.feasible
    assert cor.first_infeasible_stage is not None


def test_crossing_cap_within_braking_limits_is_left_untouched():
    # Same geometry, slower ego: the geometric cap is comfortably reachable
    # by braking, so the floor must never engage.
    ego = {"x": 0.0, "y": 0.0, "v": 3.0, "psi": 0.0}
    params = CorridorParams(
        horizon_steps=32, dt_s=0.1,
        crossing_clearance_time_s=5.0, conflict_stop_buffer_m=4.0,
        max_braking_mps2=3.0,
    )
    cor = build_longitudinal_corridor(
        REF, ego,
        [({"x": 9.2, "v": 6.0}, _tag("x", CROSSING, s=9.2, t=0.0), _assign("yield"))],
        params,
    )
    geometric_cap = 9.2 - 4.0
    assert all(abs(h - geometric_cap) < 1.0e-6 for h in cor.s_hi[1:])
    assert cor.feasible
    assert cor.first_infeasible_stage is None


def test_aggregate_mode_corridors_carries_the_braking_infeasible_flag():
    # aggregate_mode_corridors rebuilds s_hi from scratch each call, so the
    # per-mode Corridor.feasible flag from build_longitudinal_corridor's
    # kinematic floor does not fall out of it automatically -- it has to be
    # carried across explicitly, or a multimodal scenario (synthetic
    # prediction always runs several modes) silently loses the signal this
    # pipeline's own diagnostics and warm-start invalidation rely on, even
    # though the aggregated s_hi numbers themselves stay correct (each is a
    # weighted combination of already-floored per-mode caps).
    feasible_mode = Corridor(s_lo=[-_BIG] * 3, s_hi=[10.0, 10.0, 10.0], binding=[""] * 3)
    infeasible_mode = Corridor(
        s_lo=[-_BIG] * 3, s_hi=[5.0, 5.0, 5.0], binding=[""] * 3,
        feasible=False, first_infeasible_stage=1,
    )
    out = aggregate_mode_corridors(
        [(feasible_mode, 0.5, False, "mode0"), (infeasible_mode, 0.5, False, "mode1")],
        nominal_s=[100.0, 100.0, 100.0],
    )
    assert not out.feasible
    assert out.first_infeasible_stage == 1


def test_aggregate_mode_corridors_stays_feasible_when_every_mode_is():
    a = Corridor(s_lo=[-_BIG] * 3, s_hi=[10.0, 10.0, 10.0], binding=[""] * 3)
    b = Corridor(s_lo=[-_BIG] * 3, s_hi=[12.0, 12.0, 12.0], binding=[""] * 3)
    out = aggregate_mode_corridors(
        [(a, 0.5, False, "mode0"), (b, 0.5, False, "mode1")],
        nominal_s=[100.0, 100.0, 100.0],
    )
    assert out.feasible
    assert out.first_infeasible_stage is None
