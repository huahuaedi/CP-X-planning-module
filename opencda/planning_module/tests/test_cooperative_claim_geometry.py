from pipeline.cooperative_claim_geometry import project_claim_interval
from pipeline.local_map_snapshot import build_local_map_snapshot


def _snapshot():
    return build_local_map_snapshot(
        frame_id=1, timestamp_s=0.0, route_revision="r1",
        match={"valid": True, "ad_lane_id": 20},
        local_graph={
            "corridors": {0: [20]},
            "lane_to_offset": {20: 0},
            "lane_centerlines": {20: [
                {"x_m": 0.0, "y_m": 2.0},
                {"x_m": 50.0, "y_m": 2.0},
                {"x_m": 100.0, "y_m": 2.0},
            ]},
        },
    )


def test_claim_interval_uses_target_lane_arc_length():
    result = project_claim_interval(
        local_map=_snapshot(), corridor_id=20,
        x_m=40.0, y_m=0.0, lookbehind_m=10.0, lookahead_m=25.0,
    )
    assert result.valid
    assert result.corridor_id == 20
    assert result.s_begin_m == 30.0
    assert result.s_end_m == 65.0


def test_missing_target_geometry_does_not_invent_station_interval():
    result = project_claim_interval(
        local_map=_snapshot(), corridor_id=99,
        x_m=40.0, y_m=0.0, lookbehind_m=10.0, lookahead_m=25.0,
    )
    assert not result.valid
    assert result.s_begin_m is None
    assert result.s_end_m is None
