import math

from pipeline.cav_conflict_pipeline import resolve_conflicts
from pipeline.conflict_classifier import CROSSING, FOLLOW, IGNORE, MERGE
from pipeline.cooperative_arbitration import CavIntent, ResourceClaim
from pipeline.spatiotemporal_corridor import Corridor

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


def test_end_to_end_follow_is_delegated_to_speed_planner():
    lead = {"id": "lead", "x": 40.0, "y": 0.1, "v": 6.0,
            "predicted_trajectory": [{"x": 40.0, "y": 0.1}] * 21}
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=[lead], cav_intents=[],
    )
    assert r.diagnostics["tags"]["lead"] == FOLLOW
    assert r.assignments == []                       # no claim -> no Stage B
    assert all(value >= _BIG for value in r.corridor.s_hi)
    assert r.diagnostics["speed_owned_follow_count"] == 1
    assert r.corridor.feasible


def test_stage_c_reuses_rebased_corridor_between_scheduled_updates():
    cached = Corridor(
        s_lo=[-_BIG] * 21, s_hi=[18.0] * 21,
        binding=["cached-peer"] * 21,
    )
    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=[], cav_intents=[],
        tag_state={}, refresh_assignments=False, cached_assignments=(),
        rebuild_corridor=False, cached_corridor=cached,
    )
    assert result.corridor is cached
    assert not result.diagnostics["corridor_rebuilt"]


def test_cooperative_make_gap_overrides_generic_follow_handoff():
    path = [(20.0 + 0.6 * k, 0.1) for k in range(21)]
    peer = _cav(2, (20.0, 0.1), committed_at_s=1.0, path=path, speed=6.0)
    ego_claim = ResourceClaim(
        kind="lane_change", resource_id="lane_change",
        committed_at_s=2.0, active=True, require_ahead=False,
        phase="proposed",
    )
    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=5,
        my_claim=ego_claim, obstacle_snapshots=[], cav_intents=[peer],
    )
    assert result.diagnostics["tags"]["2"] == FOLLOW
    assert result.diagnostics["roles"]["2"] == "make_gap"
    assert any(value < _BIG for value in result.corridor.s_hi)


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


def test_pose_only_cav_intent_does_not_replace_prediction_modes():
    crossing = [{"x": 30.0, "y": -6.0 + k} for k in range(20)]
    pose_only = _cav(2, (30.0, -6.0), committed_at_s=4.0, path=())
    perception = {
        "id": 2, "x": 30.0, "y": -6.0, "v": 8.0,
        "psi": math.pi / 2.0,
    }
    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[perception], cav_intents=[pose_only],
        prediction_modes={
            "2": [{"path": crossing, "probability": 1.0}],
        },
    )
    assert result.diagnostics["tags"]["2"] == CROSSING
    assert result.diagnostics["shared_plan_cav_count"] == 0
    assert result.diagnostics["trajectory_source_counts"]["prediction"] == 1


def test_prediction_modes_map_feeds_a_non_connected_agents_future():
    # Only the agent's *future* (from the prediction module, passed as a
    # length-1 mode) makes it a crossing conflict; its current pose alone
    # would not.
    crossing = [{"x": 30.0, "y": -6.0 + 1.0 * k} for k in range(20)]
    agent = {"id": "npc", "x": 30.0, "y": -6.0, "v": 8.0, "psi": math.pi / 2.0}
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1, my_claim=None,
        obstacle_snapshots=[agent],
        prediction_modes={"npc": [{"path": crossing, "probability": 1.0}]},
        cav_intents=[],
    )
    assert r.diagnostics["tags"]["npc"] == CROSSING
    assert any(h < _BIG for h in r.corridor.s_hi)
    assert r.diagnostics["trajectory_source_counts"].get("prediction") == 1


def test_single_prediction_mode_is_consumed_instead_of_current_pose_fallback():
    # Current pose/velocity alone remains adjacent and would be IGNORE. The
    # sole prediction hypothesis enters the lane and must drive Stage A.
    cut_in = [{"x": 10.0 + k, "y": 3.6 - 0.18 * k} for k in range(20)]
    agent = {"id": "npc", "x": 10.0, "y": 3.6, "v": 0.0, "psi": 0.0}
    result = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot=EGO,
        my_actor_id=1,
        obstacle_snapshots=[agent],
        prediction_modes={"npc": [{"path": cut_in, "probability": 1.0}]},
    )
    assert result.diagnostics["tags"]["npc"] == "CUT_IN"
    assert any(value < _BIG for value in result.corridor.s_hi)


def test_multimodal_corridors_use_expected_risk_and_credible_veto():
    stay = [{"x": 30.0, "y": -6.0}] * 20                     # off to the side
    cross = [{"x": 30.0, "y": -6.0 + 1.0 * k} for k in range(20)]
    base = {"id": "npc", "x": 30.0, "y": -6.0, "v": 8.0, "psi": math.pi / 2.0}

    r_cross = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot={**EGO, "v": 15.0},
        my_actor_id=1, my_claim=None,
        obstacle_snapshots=[{**base, "predicted_modes": [
            {"path": stay, "probability": 0.2},
            {"path": cross, "probability": 0.8}]}],
        cav_intents=[],
    )
    assert any(h < _BIG for h in r_cross.corridor.s_hi)      # used the cross mode
    assert r_cross.diagnostics["multimodal_agent_count"] == 1
    assert r_cross.diagnostics["credible_mode_veto_count"] == 1
    # The actor remains one physical conflict source; its modes are only
    # temporary classifier inputs and reduce to one final corridor.
    assert r_cross.diagnostics["conflict_agent_count"] == 1

    cached = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot={**EGO, "v": 15.0},
        my_actor_id=1, my_claim=None,
        obstacle_snapshots=[{**base, "predicted_modes": [
            {"path": stay, "probability": 0.2},
            {"path": cross, "probability": 0.8}]}],
        cav_intents=[], tag_state=r_cross.tag_state,
        refresh_assignments=False, rebuild_corridor=False,
        cached_corridor=r_cross.corridor,
    )
    assert cached.corridor is r_cross.corridor
    assert not cached.diagnostics["corridor_rebuilt"]

    r_stay = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot={**EGO, "v": 15.0},
        my_actor_id=1, my_claim=None,
        obstacle_snapshots=[{**base, "predicted_modes": [
            {"path": stay, "probability": 0.8},
            {"path": cross, "probability": 0.2}]}],
        cav_intents=[],
    )
    assert any(h < _BIG for h in r_stay.corridor.s_hi)       # p=0.2 crosser retained
    r_filtered = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1, my_claim=None,
        obstacle_snapshots=[{**base, "predicted_modes": [
            {"path": stay, "probability": 0.8},
            {"path": cross, "probability": 0.2}]}],
        cav_intents=[], mode_probability_floor=0.3,
    )
    assert all(h >= _BIG for h in r_filtered.corridor.s_hi)


def test_low_probability_distant_mode_does_not_bind_the_mpc_corridor():
    stay = [{"x": 30.0, "y": -6.0}] * 20
    distant_cross = [{"x": 60.0, "y": -6.0 + 1.0 * k} for k in range(20)]
    base = {"id": "npc", "x": 30.0, "y": -6.0, "v": 8.0,
            "psi": math.pi / 2.0, "predicted_modes": [
                {"path": stay, "probability": 0.9},
                {"path": distant_cross, "probability": 0.1},
            ]}
    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[base], cav_intents=[],
    )
    assert result.diagnostics["credible_mode_veto_count"] == 0
    assert all(value >= _BIG for value in result.corridor.s_hi)


def test_connected_cav_snapshot_carries_a_single_broadcast_mode():
    from pipeline.cav_conflict_pipeline import _cav_to_agent_snapshot
    from pipeline.prediction_modes import as_modes
    path = [(20.0 + k, 0.0) for k in range(10)]
    snap = _cav_to_agent_snapshot(_cav(2, (20.0, 0.0), committed_at_s=4.0, path=path))
    modes = as_modes(snap["predicted_modes"])
    assert len(modes) == 1
    assert modes[0].probability == 1.0
    assert snap["trajectory_source"] == "broadcast"


def test_multimodal_probability_boundaries_are_inclusive():
    stay = [{"x": 30.0, "y": -6.0}] * 20
    cross = [{"x": 30.0, "y": -6.0 + k} for k in range(20)]

    def resolve(cross_probability, *, floor=0.05, credible=0.15):
        return resolve_conflicts(
            reference_samples=REF,
            ego_snapshot={**EGO, "v": 15.0},
            my_actor_id=1,
            obstacle_snapshots=[{
                "id": "npc", "x": 30.0, "y": -6.0, "v": 8.0,
                "psi": math.pi / 2.0,
                "predicted_modes": [
                    {"path": stay, "probability": 1.0 - cross_probability},
                    {"path": cross, "probability": cross_probability},
                ],
            }],
            mode_probability_floor=floor,
            credible_mode_probability_min=credible,
        )

    assert resolve(0.049).diagnostics["retained_prediction_mode_count"] == 1
    assert resolve(0.050).diagnostics["retained_prediction_mode_count"] == 2
    assert resolve(0.051).diagnostics["retained_prediction_mode_count"] == 2
    assert resolve(0.149).diagnostics["credible_mode_veto_count"] == 0
    assert resolve(0.150).diagnostics["credible_mode_veto_count"] == 1
    assert resolve(0.151).diagnostics["credible_mode_veto_count"] == 1
