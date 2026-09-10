from pipeline.cooperative_maneuver_proposal import CooperativeManeuverProposal
from pipeline.behavior_stage import BehaviorStage
from types import SimpleNamespace


def test_lane_change_behavior_produces_typed_request():
    proposal = CooperativeManeuverProposal.from_behavior(
        maneuver="lane_change_left", source_corridor_id=10,
        target_corridor_id=20, route_required=True,
        maneuver_active=False, committed_at_s=0.0,
        reason="route_required",
    )
    assert proposal.requested
    assert not proposal.committed
    assert proposal.route_required


def test_lane_follow_behavior_releases_cooperative_resource():
    proposal = CooperativeManeuverProposal.from_behavior(
        maneuver="lane_follow", source_corridor_id=10,
        target_corridor_id=10, route_required=False,
        maneuver_active=False, committed_at_s=0.0,
    )
    assert not proposal.requested
    assert not proposal.committed


def test_geometry_enrichment_keeps_behavior_semantics_immutable():
    proposal = CooperativeManeuverProposal.from_behavior(
        maneuver="lane_change_right", source_corridor_id=20,
        target_corridor_id=10, route_required=False,
        maneuver_active=True, committed_at_s=3.0,
    )
    enriched = proposal.with_station_interval(
        corridor_id=10, s_begin_m=25.0, s_end_m=60.0,
    )
    assert proposal.station_corridor_id == 0
    assert enriched.committed
    assert enriched.station_corridor_id == 10
    assert enriched.s_begin_m == 25.0


def test_behavior_candidate_proposal_precedes_final_safety_decision():
    proposal = BehaviorStage._cooperative_proposal_from_candidate(
        authorization=SimpleNamespace(allowed=False),
        candidate_frame=SimpleNamespace(selected=SimpleNamespace(
            decision="lane_change_left", target_lane_id=20,
            reason="lower_progress_cost",
        )),
        current_lane_id=10,
        opportunistic_lane_change_allowed=True,
    )
    assert proposal.requested
    assert proposal.maneuver == "lane_change_left"
    assert proposal.target_corridor_id == 20
    assert not proposal.committed


def test_lifecycle_commitment_enriches_instead_of_redeciding_proposal():
    proposal = CooperativeManeuverProposal.from_behavior(
        maneuver="lane_follow", source_corridor_id=10,
        target_corridor_id=10, route_required=False,
        maneuver_active=False, committed_at_s=0.0,
    )
    committed = proposal.with_commitment(
        maneuver="lane_change_right", target_corridor_id=30,
        committed_at_s=12.5,
    )
    assert committed.requested
    assert committed.committed
    assert committed.maneuver == "lane_change_right"
    assert committed.target_corridor_id == 30
    assert committed.committed_at_s == 12.5
