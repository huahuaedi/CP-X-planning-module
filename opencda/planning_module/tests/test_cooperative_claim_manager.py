from pipeline.cooperative_arbitration import ConflictAssignment
from pipeline.cooperative_claim_manager import CooperativeClaimManager


def _assignment(role):
    return ConflictAssignment(
        cav_actor_id=2, role=role, homotopy_side="left",
        cav_wins=(role != "proceed"), reason="test",
    )


def test_proposal_waits_for_exchange_window_then_can_commit():
    manager = CooperativeClaimManager(enabled=True, proposal_dwell_s=0.25)
    claim = manager.claim(
        decision="lane_change_left", target_lane_id=2, sim_time_s=1.0,
        maneuver_active=False, committed_at_s=0.0,
    )
    assert claim.phase == "proposed"
    assert manager.defer_candidate(sim_time_s=1.1, assignments=())
    assert not manager.defer_candidate(sim_time_s=1.3, assignments=())


def test_loser_remains_deferred_after_exchange_window():
    manager = CooperativeClaimManager(enabled=True, proposal_dwell_s=0.1)
    manager.claim(
        decision="lane_change_right", target_lane_id=3, sim_time_s=2.0,
        maneuver_active=False, committed_at_s=0.0,
    )
    assert manager.defer_candidate(
        sim_time_s=2.2, assignments=[_assignment("make_gap")]
    )


def test_maneuver_transition_publishes_committed_claim():
    manager = CooperativeClaimManager(enabled=True, proposal_dwell_s=0.1)
    manager.claim(
        decision="lane_change_left", target_lane_id=2, sim_time_s=1.0,
        maneuver_active=False, committed_at_s=0.0,
    )
    claim = manager.claim(
        decision="lane_change_left", target_lane_id=2, sim_time_s=1.5,
        maneuver_active=True, committed_at_s=1.4,
    )
    assert claim.phase == "committed"
    assert claim.committed_at_s == 1.4
    assert not manager.proposal_active


def test_non_lane_change_releases_proposal():
    manager = CooperativeClaimManager(enabled=True)
    manager.claim(
        decision="lane_change_left", target_lane_id=2, sim_time_s=1.0,
        maneuver_active=False, committed_at_s=0.0,
    )
    claim = manager.claim(
        decision="lane_follow", target_lane_id=1, sim_time_s=1.1,
        maneuver_active=False, committed_at_s=0.0,
    )
    assert claim.phase == "released"
    assert not claim.participates


def test_proposal_contains_spatial_resource_without_resetting_its_clock():
    manager = CooperativeClaimManager(enabled=True)
    first = manager.claim(
        decision="lane_change_left", target_lane_id=20, sim_time_s=1.0,
        maneuver_active=False, committed_at_s=0.0,
        source_corridor_id=10, s_begin_m=40.0, s_end_m=90.0,
    )
    updated = manager.claim(
        decision="lane_change_left", target_lane_id=20, sim_time_s=1.2,
        maneuver_active=False, committed_at_s=0.0,
        source_corridor_id=10, s_begin_m=42.0, s_end_m=92.0,
    )
    assert updated.resource_id == "lane_change:10:20"
    assert updated.source_corridor_id == 10
    assert updated.target_corridor_id == 20
    assert updated.s_begin_m == 42.0
    assert updated.s_end_m == 92.0
    assert updated.committed_at_s == first.committed_at_s == 1.0


def test_committed_claim_preserves_proposed_spatial_resource_on_publish():
    manager = CooperativeClaimManager(enabled=True)
    manager.claim(
        decision="lane_change_right", target_lane_id=30, sim_time_s=2.0,
        maneuver_active=False, committed_at_s=0.0,
        source_corridor_id=20, s_begin_m=100.0, s_end_m=150.0,
    )
    committed = manager.claim(
        decision="lane_change_right", target_lane_id=30, sim_time_s=2.5,
        maneuver_active=True, committed_at_s=2.4,
    )
    assert committed.phase == "committed"
    assert committed.source_corridor_id == 20
    assert committed.target_corridor_id == 30
    assert committed.s_begin_m == 100.0
    assert committed.s_end_m == 150.0
