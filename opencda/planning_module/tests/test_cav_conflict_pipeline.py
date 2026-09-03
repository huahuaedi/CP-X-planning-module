import math

from pipeline.cav_conflict_pipeline import resolve_conflicts
from pipeline.conflict_classifier import CROSSING, FOLLOW, IGNORE, MERGE
from pipeline.cooperative_arbitration import CavIntent, ResourceClaim

REF = [{"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(0, 121, 2)]
EGO = {"x": 0.0, "y": 0.0, "v": 10.0, "psi": 0.0}
_BIG = 1.0e9


def _claim(committed_at_s=10.0, active=True):
    return ResourceClaim(
        kind="lane_change", resource_id="lane_change",
        committed_at_s=committed_at_s, active=active,
    )


def _cav(actor_id, xy, committed_at_s, *, path=(), speed=9.0, heading=0.0,
          cooperative=True):
    return CavIntent(
        actor_id=actor_id, position_xy=xy,
        claim=_claim(committed_at_s=committed_at_s),
        heading_rad=heading, speed_mps=speed,
        planned_path=tuple((0.1 * k, x, y, speed) for k, (x, y) in enumerate(path)),
        cooperative=cooperative,
    )


def test_end_to_end_follow_only_no_claim():
    lead = {"id": "lead", "x": 40.0, "y": 0.1, "v": 6.0,
            "predicted_trajectory": [{"x": 40.0, "y": 0.1}] * 21}
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=[lead], cav_intents=[],
    )
    assert r.diagnostics["tags"]["lead"] == FOLLOW
    assert r.assignments == []                       # no claim -> no Stage B
    assert r.corridor.s_hi[10] < 40.0
    assert r.corridor.feasible


def test_ignored_agent_never_reaches_the_corridor():
    adj = {"id": "adj", "x": 3.0, "y": 3.6, "v": 10.0, "psi": 0.0}
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1, my_claim=None,
        obstacle_snapshots=[adj], cav_intents=[],
    )
    assert r.diagnostics["tags"]["adj"] == IGNORE
    assert all(h >= _BIG for h in r.corridor.s_hi)


def test_cooperative_merge_cav_wins_ego_opens_gap():
    # cav merging in from the adjacent lane, committed earlier than ego
    path = [(25.0 + 0.9 * k, 3.4 - 0.16 * k) for k in range(20)]
    cav = _cav(2, (25.0, 3.4), committed_at_s=4.0, path=path, speed=9.0)
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=_claim(committed_at_s=10.0),
        obstacle_snapshots=[], cav_intents=[cav],
    )
    assert r.diagnostics["roles"].get("2") == "make_gap"
    # ego is bounded behind the cav's projected station
    assert any(h < _BIG for h in r.corridor.s_hi)
    assert "2" in r.corridor.binding


def test_cooperative_merge_cav_loses_ego_proceeds():
    path = [(25.0 + 0.9 * k, 3.4 - 0.16 * k) for k in range(20)]
    cav = _cav(2, (25.0, 3.4), committed_at_s=20.0, path=path, speed=9.0)
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=_claim(committed_at_s=10.0),
        obstacle_snapshots=[], cav_intents=[cav],
    )
    assert r.diagnostics["roles"].get("2") == "proceed"
    assert all(h >= _BIG for h in r.corridor.s_hi)


def test_non_connected_crosser_defaults_to_yield_without_assignment():
    path = [(30.0, -6.0 + 1.0 * k) for k in range(20)]
    crosser = {
        "id": "x", "x": 30.0, "y": -6.0, "v": 8.0, "psi": math.pi / 2.0,
        "predicted_trajectory": [{"x": x, "y": y} for x, y in path],
    }
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1, my_claim=None,
        obstacle_snapshots=[crosser], cav_intents=[],
    )
    assert r.diagnostics["tags"]["x"] == CROSSING
    assert r.assignments == []                       # non-connected -> no role
    assert any(h < _BIG for h in r.corridor.s_hi)   # still yielded


def test_latch_state_round_trips_and_holds():
    path = [(25.0 + 0.9 * k, 3.4 - 0.16 * k) for k in range(20)]
    r1 = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=_claim(committed_at_s=10.0),
        cav_intents=[_cav(2, (25.0, 3.4), committed_at_s=11.0, path=path)],
    )
    assert r1.diagnostics["roles"].get("2") == "proceed"
    # cav re-commits slightly earlier (non-decisive) -> latch holds proceed
    r2 = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=_claim(committed_at_s=10.0),
        cav_intents=[_cav(2, (25.0, 3.4), committed_at_s=9.8, path=path)],
        latch_state=r1.latch_state, hysteresis_ticks=3,
    )
    assert r2.diagnostics["roles"].get("2") == "proceed"


def test_connected_cav_replaces_same_actor_perception_track():
    path = [(20.0 + k, 0.0) for k in range(20)]
    cav = _cav(2, (20.0, 0.0), committed_at_s=4.0, path=path)
    duplicate_track = {
        "id": 2, "x": 200.0, "y": 0.0, "v": 0.0,
        "predicted_trajectory": [{"x": 200.0, "y": 0.0}] * 21,
    }
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[duplicate_track], cav_intents=[cav],
    )
    assert r.diagnostics["conflict_agent_count"] == 1
    assert r.diagnostics["deduplicated_agent_count"] == 1
    assert set(r.diagnostics["tags"]) == {"2"}
