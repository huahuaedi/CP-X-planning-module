from pipeline.cooperative_maneuver_proposal import CooperativeManeuverProposal


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
