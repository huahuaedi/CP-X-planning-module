from types import SimpleNamespace

from pipeline.cav_conflict_schedule import CAVConflictSchedule


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
