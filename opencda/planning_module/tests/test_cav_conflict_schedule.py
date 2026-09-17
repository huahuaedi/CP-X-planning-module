from types import SimpleNamespace

from pipeline.cav_conflict_schedule import CAVConflictSchedule
from pipeline.spatiotemporal_corridor import Corridor

_BIG = 1.0e9


def test_pending_corridor_keeps_original_time_and_expires_after_forecast():
    schedule = CAVConflictSchedule()
    reference = [
        {"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(20)
    ]
    fresh = Corridor(
        s_lo=[-_BIG] * 4, s_hi=[10.0] * 4,
        binding=["peer"] * 4,
    )
    effective = Corridor(
        s_lo=[-_BIG] * 4, s_hi=[10.0] * 4,
        binding=["peer"] * 4,
    )
    schedule.observe(
        sim_time_s=1.0, reference_samples=reference,
        result=SimpleNamespace(
            fresh_corridor=fresh, corridor=effective,
            diagnostics={"corridor_rebuilt": True},
        ),
    )
    assert schedule.corridor is fresh
    # A clear refresh can retain the previous forecast for its unexpired
    # stages, but may not stamp those old rows with a new publication time.
    schedule.observe(
        sim_time_s=1.1, reference_samples=reference,
        result=SimpleNamespace(
            fresh_corridor=Corridor(
                s_lo=[-_BIG] * 4, s_hi=[_BIG] * 4,
                binding=[""] * 4,
            ),
            corridor=effective,
            diagnostics={"corridor_rebuilt": True},
        ),
    )
    assert schedule.corridor is fresh
    assert schedule.corridor_time_s == 1.0
    pending = schedule.cached_corridor_for_tick(
        sim_time_s=1.2, reference_samples=reference,
        ego_x_m=0.0, ego_y_m=0.0, dt_s=0.1,
    )
    assert pending.s_hi[:2] == [10.0, 10.0]
    assert pending.s_hi[2:] == [_BIG, _BIG]
    assert schedule.cached_corridor_for_tick(
        sim_time_s=1.3, reference_samples=reference,
        ego_x_m=0.0, ego_y_m=0.0, dt_s=0.1,
    ) is None
    assert schedule.corridor is None


def test_fresh_clear_permanently_retires_cached_actor_bound():
    schedule = CAVConflictSchedule()
    reference = [
        {"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(20)
    ]
    cached = Corridor(
        s_lo=[-_BIG] * 4,
        s_hi=[10.0] * 4,
        binding=["peer-7"] * 4,
    )
    schedule.corridor = cached
    schedule.corridor_reference = tuple(reference)
    schedule.corridor_time_s = 1.0

    schedule.observe(
        sim_time_s=1.1,
        reference_samples=reference,
        result=SimpleNamespace(
            latch_state={}, tag_state={"peer-7": "IGNORE"},
            veto_state={}, assignments=(), released_actor_ids=("peer-7",),
            fresh_corridor=Corridor(
                s_lo=[-_BIG] * 4, s_hi=[_BIG] * 4, binding=[""] * 4,
            ),
            diagnostics={"corridor_rebuilt": True},
        ),
    )

    assert schedule.corridor is None
    assert schedule.cached_corridor_for_tick(
        sim_time_s=1.15,
        reference_samples=reference,
        ego_x_m=0.0,
        ego_y_m=0.0,
        dt_s=0.1,
    ) is None


def test_route_reset_clears_every_conflict_lifecycle_state():
    schedule = CAVConflictSchedule(
        latch_state={"7": object()},
        tag_state={"7": "CROSSING"},
        veto_state={"7::mode0": {"dangerous": True}},
        assignments=(object(),),
        corridor=Corridor(
            s_lo=[-_BIG], s_hi=[10.0], binding=["7"],
        ),
        corridor_reference=({"x_ref_m": 0.0, "y_ref_m": 0.0},),
        corridor_time_s=4.0,
    )
    old_revision = schedule.revision

    schedule.reset(reason="route_revision_changed:r2")

    assert schedule.latch_state == {}
    assert schedule.tag_state == {}
    assert schedule.veto_state == {}
    assert schedule.assignments == ()
    assert schedule.corridor is None
    assert schedule.corridor_reference == ()
    assert schedule.revision == old_revision + 1
    assert schedule.last_reset_reason == "route_revision_changed:r2"


def test_constraint_revision_ignores_scheduler_refresh_revision():
    base = {
        "roles": {7: "yield"}, "tags": {7: "FOLLOW"},
        "corridor_binding": ["agent:7"], "corridor_feasible": True,
        "longitudinal_qp_row_count": 6, "homotopy_qp_row_count": 0,
        "coordination_revision": 1,
    }
    refreshed = dict(base, coordination_revision=99, corridor_rebuilt=True)
    assert CAVConflictSchedule.constraint_revision(base) == (
        CAVConflictSchedule.constraint_revision(refreshed)
    )


def test_constraint_revision_changes_with_constraint_topology():
    base = {
        "roles": {7: "yield"}, "tags": {7: "FOLLOW"},
        "corridor_binding": ["agent:7"], "corridor_feasible": True,
        "longitudinal_qp_row_count": 6, "homotopy_qp_row_count": 0,
    }
    assert CAVConflictSchedule.constraint_revision(base) != (
        CAVConflictSchedule.constraint_revision(
            dict(base, longitudinal_qp_row_count=7)
        )
    )


def _proposal(target=20):
    return SimpleNamespace(
        maneuver="lane_change_left", target_corridor_id=target,
        committed=False,
    )


def test_coordination_runs_at_five_hz_between_unchanged_inputs():
    schedule = CAVConflictSchedule(coordination_period_s=0.2)
    claim = SimpleNamespace(
        resource_id="lane_change:10:20", phase="proposed",
        participates=True, committed_at_s=0.0,
    )
    first = schedule.decide(
        sim_time_s=1.0, prediction_revision="p1", claim=claim,
        peers=(), proposal=_proposal(),
    )
    assert first.refresh_roles
    schedule.observe(
        sim_time_s=1.0,
        result=SimpleNamespace(
            latch_state={}, tag_state={}, assignments=(),
            diagnostics={"coordination_roles_refreshed": True},
        ),
    )
    assert not schedule.decide(
        sim_time_s=1.05, prediction_revision="p1", claim=claim,
        peers=(), proposal=_proposal(),
    ).refresh_roles
    assert schedule.decide(
        sim_time_s=1.20, prediction_revision="p1", claim=claim,
        peers=(), proposal=_proposal(),
    ).refresh_roles


def test_prediction_revision_is_sampled_at_coordination_cadence():
    schedule = CAVConflictSchedule(coordination_period_s=0.2)
    claim = SimpleNamespace(
        resource_id="lane_change:10:20", phase="proposed",
        participates=True, committed_at_s=0.0,
    )
    schedule.decide(
        sim_time_s=1.0, prediction_revision="p1", claim=claim,
        peers=(), proposal=_proposal(),
    )
    schedule.observe(
        sim_time_s=1.0,
        result=SimpleNamespace(
            latch_state={}, tag_state={}, assignments=(),
            diagnostics={"coordination_roles_refreshed": True},
        ),
    )
    decision = schedule.decide(
        sim_time_s=1.05, prediction_revision="p2", claim=claim,
        peers=(), proposal=_proposal(),
    )
    assert not decision.refresh_roles
    decision = schedule.decide(
        sim_time_s=1.20, prediction_revision="p2", claim=claim,
        peers=(), proposal=_proposal(),
    )
    assert decision.refresh_roles
    assert decision.reason == "coordination_period_elapsed"


def test_claim_structure_change_forces_immediate_refresh():
    schedule = CAVConflictSchedule(coordination_period_s=0.2)
    claim = SimpleNamespace(
        resource_id="lane_change:10:20", phase="proposed",
        participates=True, committed_at_s=1.0,
    )
    schedule.decide(
        sim_time_s=1.0, prediction_revision="p1", claim=claim,
        peers=(), proposal=_proposal(),
    )
    schedule.observe(
        sim_time_s=1.0,
        result=SimpleNamespace(
            latch_state={}, tag_state={}, assignments=(),
            diagnostics={"coordination_roles_refreshed": True},
        ),
    )
    committed = SimpleNamespace(
        resource_id="lane_change:10:20", phase="committed",
        participates=True, committed_at_s=1.0,
    )
    decision = schedule.decide(
        sim_time_s=1.05, prediction_revision="p1", claim=committed,
        peers=(), proposal=_proposal(),
    )
    assert decision.refresh_roles
    assert decision.reason == "coordination_structure_changed"


def test_released_claim_timestamp_is_not_a_structure_change():
    schedule = CAVConflictSchedule(coordination_period_s=0.2)

    def released(timestamp_s):
        return SimpleNamespace(
            resource_id="lane_change", phase="released",
            participates=False, committed_at_s=timestamp_s,
        )

    schedule.decide(
        sim_time_s=1.0, prediction_revision="p1", claim=released(1.0),
        peers=(), proposal=SimpleNamespace(
            maneuver="lane_follow", target_corridor_id=10, committed=False,
        ),
    )
    schedule.observe(
        sim_time_s=1.0,
        result=SimpleNamespace(
            latch_state={}, tag_state={}, assignments=(),
            diagnostics={"coordination_roles_refreshed": True},
        ),
    )
    decision = schedule.decide(
        sim_time_s=1.05, prediction_revision="p2", claim=released(1.05),
        peers=(), proposal=SimpleNamespace(
            maneuver="lane_follow", target_corridor_id=10, committed=False,
        ),
    )
    assert not decision.refresh_roles
