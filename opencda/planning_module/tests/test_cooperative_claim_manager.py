from pipeline.cooperative_arbitration import ConflictAssignment
from pipeline.cooperative_claim_manager import CooperativeClaimManager
from pipeline.cooperative_maneuver_proposal import CooperativeManeuverProposal


def _assignment(role):
    return ConflictAssignment(
        cav_actor_id=2, role=role, homotopy_side="left",
        cav_wins=(role != "proceed"), reason="test",
    )


def _proposal(
    maneuver="lane_change_left", source=1, target=2, active=False,
    committed_at_s=0.0, station=0, s_begin=None, s_end=None,
):
    proposal = CooperativeManeuverProposal.from_behavior(
        maneuver=maneuver, source_corridor_id=source,
        target_corridor_id=target, route_required=True,
        maneuver_active=active, committed_at_s=committed_at_s,
    )
    return proposal.with_station_interval(
        corridor_id=station, s_begin_m=s_begin, s_end_m=s_end,
    )


def test_proposal_waits_for_exchange_window_then_can_commit():
    manager = CooperativeClaimManager(enabled=True, proposal_dwell_s=0.25)
    claim = manager.claim(
        proposal=_proposal(), sim_time_s=1.0,
    )
    assert claim.phase == "proposed"
    assert manager.defer_candidate(sim_time_s=1.1, assignments=())
    assert not manager.defer_candidate(sim_time_s=1.3, assignments=())


def test_loser_remains_deferred_after_exchange_window():
    manager = CooperativeClaimManager(enabled=True, proposal_dwell_s=0.1)
    manager.claim(
        proposal=_proposal(maneuver="lane_change_right", target=3), sim_time_s=2.0,
    )
    assert manager.defer_candidate(
        sim_time_s=2.2, assignments=[_assignment("make_gap")]
    )


def test_maneuver_transition_publishes_committed_claim():
    manager = CooperativeClaimManager(enabled=True, proposal_dwell_s=0.1)
    manager.claim(
        proposal=_proposal(), sim_time_s=1.0,
    )
    claim = manager.claim(
        proposal=_proposal(active=True, committed_at_s=1.4), sim_time_s=1.5,
    )
    assert claim.phase == "committed"
    assert claim.committed_at_s == 1.4
    assert not manager.proposal_active


def test_non_lane_change_releases_proposal():
    manager = CooperativeClaimManager(enabled=True)
    manager.claim(
        proposal=_proposal(), sim_time_s=1.0,
    )
    claim = manager.claim(
        proposal=_proposal(maneuver="lane_follow", target=1), sim_time_s=1.1,
    )
    assert claim.phase == "released"
    assert not claim.participates


def test_proposal_contains_spatial_resource_without_resetting_its_clock():
    manager = CooperativeClaimManager(enabled=True)
    first = manager.claim(
        proposal=_proposal(source=10, target=20, station=20,
                           s_begin=40.0, s_end=90.0), sim_time_s=1.0,
    )
    updated = manager.claim(
        proposal=_proposal(source=10, target=20, station=20,
                           s_begin=42.0, s_end=92.0), sim_time_s=1.2,
    )
    assert updated.resource_id == "lane_change:10:20"
    assert updated.source_corridor_id == 10
    assert updated.target_corridor_id == 20
    assert updated.station_corridor_id == 20
    assert updated.s_begin_m == 42.0
    assert updated.s_end_m == 92.0
    assert updated.committed_at_s == first.committed_at_s == 1.0


def test_committed_claim_preserves_proposed_spatial_resource_on_publish():
    manager = CooperativeClaimManager(enabled=True)
    manager.claim(
        proposal=_proposal(maneuver="lane_change_right", source=20, target=30,
                           station=30, s_begin=100.0, s_end=150.0),
        sim_time_s=2.0,
    )
    committed = manager.claim(
        proposal=_proposal(maneuver="lane_change_right", source=0, target=30,
                           active=True, committed_at_s=2.4),
        sim_time_s=2.5,
    )
    assert committed.phase == "committed"
    assert committed.source_corridor_id == 20
    assert committed.target_corridor_id == 30
    assert committed.station_corridor_id == 30
    assert committed.s_begin_m == 100.0
    assert committed.s_end_m == 150.0
