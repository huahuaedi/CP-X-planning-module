import math

import pytest

from pipeline.cav_conflict_pipeline import resolve_conflicts
from pipeline.conflict_classifier import CROSSING, FOLLOW, IGNORE, LEAD_BRAKE, MERGE
from pipeline.cooperative_arbitration import (
    ArbitrationLatchEntry,
    CavIntent,
    ResourceClaim,
)
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


def test_relevant_agent_budget_keeps_the_nearest_agents_ahead():
    # 8 obstacles all classified FOLLOW at increasing distance, so
    # conflict_t_s (closing time) grows monotonically with distance for this
    # simple case -- ranking by severity coincides with ranking by distance
    # here. Two-agent scenarios are not evidence four (let alone eight) hold
    # real-time: the budget must cap Stage B/C's input regardless of how
    # many objects perception reports.
    obstacles = [
        {"id": f"obs{i}", "x": 5.0 + 4.0 * i, "y": 0.1, "v": 6.0,
         "predicted_trajectory": [{"x": 5.0 + 4.0 * i, "y": 0.1}] * 21}
        for i in range(8)
    ]
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=obstacles, cav_intents=[],
    )
    assert r.diagnostics["conflict_agent_count"] == 6
    assert r.diagnostics["relevant_agent_budget"] == 6
    assert r.diagnostics["relevant_agent_dropped_count"] == 2
    kept_ids = set(r.diagnostics["tags"].keys())
    assert kept_ids == {f"obs{i}" for i in range(6)}


def test_relevant_agent_budget_is_configurable():
    obstacles = [
        {"id": f"obs{i}", "x": 5.0 + 4.0 * i, "y": 0.1, "v": 6.0,
         "predicted_trajectory": [{"x": 5.0 + 4.0 * i, "y": 0.1}] * 21}
        for i in range(8)
    ]
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=obstacles, cav_intents=[],
        max_relevant_agents=3,
    )
    assert r.diagnostics["conflict_agent_count"] == 3
    assert r.diagnostics["relevant_agent_dropped_count"] == 5


def test_severity_budget_keeps_a_far_crossing_vehicle_over_near_non_conflicts():
    # The exact danger a pre-classification position filter gets wrong: a
    # vehicle 6m off ego's path (outside the default 3m ignore_lateral_m
    # window) but on a predicted track crossing ego's path shortly, versus
    # several vehicles that are merely *near* ego's current position with no
    # real closing risk (moving at ego's own speed, so nothing is actually
    # converging). A position/distance-based pre-filter would keep the near,
    # harmless vehicles and could drop the real crossing conflict outright.
    # Ranking by classify_conflicts' own conflict_t_s (computed from each
    # agent's full predicted track, after classification has already run on
    # everyone) must keep the crossing vehicle instead.
    fillers = [
        {"id": f"filler{i}", "x": 35.0 + 4.0 * i, "y": 0.1, "v": 10.0,
         "predicted_trajectory": [
             {"x": 35.0 + 4.0 * i + 1.0 * k, "y": 0.1} for k in range(21)
         ]}
        for i in range(5)
    ]
    crossing_track = [{"x": 30.0, "y": -6.0 + 1.0 * k} for k in range(20)]
    crosser = {"id": "npc", "x": 30.0, "y": -6.0, "v": 8.0, "psi": math.pi / 2.0}
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1, my_claim=None,
        obstacle_snapshots=fillers + [crosser],
        prediction_modes={"npc": [{"path": crossing_track, "probability": 1.0}]},
        cav_intents=[], max_relevant_agents=3,
    )
    assert r.diagnostics["tags"].get("npc") == CROSSING
    assert r.diagnostics["relevant_agent_dropped_count"] == 3


def test_mode_budget_trims_only_the_sub_credible_tail():
    # credible_mode_probability_min defaults to 0.15, mode_probability_floor
    # defaults to 0.05 -- all 5 probabilities here clear the floor (so none
    # are dropped before the cap even runs), but only the first clears the
    # credible threshold. One mode (0.5) is credible and must survive
    # regardless of the mode budget; among the remaining sub-credible tail
    # (0.10/0.09/0.08/0.07), the budget (3) only has room for 2 more once
    # the credible one is seated, so the two highest-probability tail modes
    # survive and the rest are trimmed.
    agent = {
        "id": "multi", "x": 20.0, "y": 0.1, "v": 6.0,
        "predicted_modes": [
            {"path": [{"x": 20.0 + k, "y": 0.1} for k in range(21)],
             "probability": p}
            for p in (0.5, 0.10, 0.09, 0.08, 0.07)
        ],
    }
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=[agent], cav_intents=[],
        max_modes_per_agent=3,
    )
    assert r.diagnostics["mode_budget_per_agent"] == 3
    assert r.diagnostics["mode_budget_capped_agent_count"] == 1
    kept_mode_ids = {k for k in r.diagnostics["tags"] if k.startswith("multi::mode")}
    assert kept_mode_ids == {"multi::mode0", "multi::mode1", "multi::mode2"}


def test_mode_budget_never_drops_a_credible_mode_even_over_budget():
    # Two modes (0.5, 0.16) both clear credible_mode_probability_min (0.15).
    # A mode budget of 1 must not silently blind the credible-danger veto
    # to either one of them -- both survive, exceeding the nominal budget,
    # and only the definitely-sub-credible tail (0.05, 0.04) is trimmed.
    agent = {
        "id": "multi", "x": 20.0, "y": 0.1, "v": 6.0,
        "predicted_modes": [
            {"path": [{"x": 20.0 + k, "y": 0.1} for k in range(21)],
             "probability": p}
            for p in (0.5, 0.16, 0.05, 0.04)
        ],
    }
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=[agent], cav_intents=[],
        max_modes_per_agent=1,
    )
    kept_mode_ids = {k for k in r.diagnostics["tags"] if k.startswith("multi::mode")}
    assert kept_mode_ids == {"multi::mode0", "multi::mode1"}


def test_broadcast_cav_lead_brake_uses_its_own_planned_deceleration():
    # Regression: _cav_to_agent_snapshot used to build a broadcasting CAV's
    # agent dict with no acceleration field at all, so conflict_classifier's
    # LEAD_BRAKE tag (same-lane, a_accel <= decel_threshold_mps2) could never
    # fire against another CAV no matter how hard it actually braked --
    # confirmed on a real scripted hard-brake run where cav_conflict_tags
    # stayed FOLLOW for the whole encounter. The planned path's own v samples
    # now derive a near-term acceleration from the CAV's committed plan.
    decelerating_path = tuple(
        (0.1 * k, 40.0 + 0.6 * min(k, 1) + 0.1 * max(k - 1, 0), 0.1,
         6.0 if k == 0 else 1.0)
        for k in range(21)
    )
    lead = CavIntent(
        actor_id=2, position_xy=(40.0, 0.1),
        claim=_claim(committed_at_s=10.0),
        heading_rad=0.0, speed_mps=6.0,
        planned_path=decelerating_path,
        cooperative=True,
    )
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=[], cav_intents=[lead],
    )
    assert r.diagnostics["tags"]["2"] == LEAD_BRAKE


def test_broadcast_cav_at_constant_planned_speed_stays_follow():
    # Companion case: an unchanging planned speed derives zero acceleration
    # (as every other _cav()-built fixture in this file does), so the same
    # physical setup without deceleration stays FOLLOW, not LEAD_BRAKE.
    # Same spatial footprint as the deceleration case's post-brake plateau
    # (constant v=1.0 the whole time instead of only from sample 1 on), so
    # acceleration is the one thing that differs between the two tests.
    level_path = tuple(
        (0.1 * k, 40.0 + 0.1 * k, 0.1, 1.0) for k in range(21)
    )
    lead = CavIntent(
        actor_id=2, position_xy=(40.0, 0.1),
        claim=_claim(committed_at_s=10.0),
        heading_rad=0.0, speed_mps=1.0,
        planned_path=level_path,
        cooperative=True,
    )
    r = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=None, obstacle_snapshots=[], cav_intents=[lead],
    )
    assert r.diagnostics["tags"]["2"] == FOLLOW


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


def test_multimodal_follow_reduces_modes_to_prediction_safety_corridor():
    modes = [
        {"path": [{"x": 30.0 + 0.6 * k, "y": 0.1} for k in range(21)],
         "probability": probability}
        for probability in (0.55, 0.25, 0.20)
    ]
    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[{
            "id": "lead", "x": 30.0, "y": 0.1, "v": 6.0,
            "psi": 0.0, "predicted_modes": modes,
        }],
    )
    assert set(result.diagnostics["tags"].values()) == {FOLLOW}
    assert any(value < _BIG for value in result.corridor.s_hi)
    assert any(result.corridor.binding)


def test_multimodal_risk_beyond_route_end_does_not_constrain_mpc():
    modes = [
        {
            "path": [{"x": 15.0, "y": 0.1} for _ in range(21)],
            "probability": probability,
        }
        for probability in (0.55, 0.25, 0.20)
    ]
    result = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot={"x": 0.0, "y": 0.0, "v": 4.0, "psi": 0.0},
        my_actor_id=1,
        obstacle_snapshots=[{
            "id": "lead", "x": 15.0, "y": 0.1, "v": 0.0,
            "psi": 0.0, "predicted_modes": modes,
        }],
        nominal_progress_limit_m=5.0,
    )
    assert all(value >= _BIG for value in result.corridor.s_hi)
    assert result.diagnostics["credible_mode_veto_count"] == 0


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


def test_stage_c_refresh_retains_pending_rows_for_unchanged_conflict():
    cached = Corridor(
        s_lo=[-_BIG] * 21, s_hi=[18.0] * 21,
        binding=["cached-peer"] * 21,
    )
    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[], tag_state={}, rebuild_corridor=True,
        cached_corridor=cached,
    )
    assert result.diagnostics["corridor_rebuilt"]
    assert result.corridor.s_hi == cached.s_hi
    assert result.corridor.binding == cached.binding


def test_fresh_actor_forecast_supersedes_its_cached_pending_rows():
    cached = Corridor(
        s_lo=[-_BIG] * 21,
        s_hi=[18.0] * 21,
        binding=["lead::mode2"] * 21,
    )
    modes = [
        {
            "probability": probability,
            "trajectory": [
                {"t": 0.1 * k, "x": 30.0 + k, "y": 0.1, "v": 10.0}
                for k in range(21)
            ],
        }
        for probability in (0.6, 0.25, 0.15)
    ]
    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[{
            "id": "lead", "x": 30.0, "y": 0.1, "v": 10.0,
            "psi": 0.0, "predicted_modes": modes,
        }],
        tag_state={
            "lead::mode0": FOLLOW,
            "lead::mode1": FOLLOW,
            "lead::mode2": FOLLOW,
        },
        rebuild_corridor=True,
        cached_corridor=cached,
    )

    assert result.diagnostics["corridor_rebuilt"]
    assert all(value >= _BIG for value in result.fresh_corridor.s_hi)
    assert all(value >= _BIG for value in result.corridor.s_hi)
    assert not any(result.corridor.binding)


def test_stage_a_geometric_clearance_filters_cache_without_stage_c_refresh():
    cached = Corridor(
        s_lo=[-_BIG] * 21,
        s_hi=[18.0] * 21,
        binding=["walker"] * 20 + ["unseen_vehicle"],
    )
    walker = {
        "id": "walker", "x": 30.0, "y": 6.0,
        "v": 1.2, "psi": -math.pi / 2.0,
        "predicted_trajectory": [{"x": 30.0, "y": 6.0}] * 21,
    }
    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[walker], tag_state={"walker": IGNORE},
        refresh_assignments=False, rebuild_corridor=False,
        cached_corridor=cached,
    )

    assert result.diagnostics["tags"]["walker"] == IGNORE
    assert not result.diagnostics["corridor_rebuilt"]
    assert result.corridor.s_hi == [_BIG] * 20 + [18.0]
    assert cached.s_hi == [18.0] * 21


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


def test_conflicting_claims_arbitrate_before_geometric_merge_begins():
    # Both vehicles are still on adjacent parallel lanes, so Stage A is
    # correctly IGNORE. Their overlapping same-target claims nevertheless
    # need an early, deterministic role assignment before lateral motion.
    path = [(20.0 + 0.8 * k, 3.6) for k in range(21)]
    peer_claim = ResourceClaim(
        kind="lane_change", resource_id="lane_change:10:20",
        committed_at_s=1.0, active=True, require_ahead=False,
        phase="proposed", source_corridor_id=10, target_corridor_id=20,
        station_corridor_id=20, s_begin_m=0.0, s_end_m=80.0,
    )
    ego_claim = ResourceClaim(
        kind="lane_change", resource_id="lane_change:10:20",
        committed_at_s=2.0, active=True, require_ahead=False,
        phase="proposed", source_corridor_id=10, target_corridor_id=20,
        station_corridor_id=20, s_begin_m=0.0, s_end_m=80.0,
    )
    peer = CavIntent(
        actor_id=2, position_xy=(20.0, 3.6), claim=peer_claim,
        heading_rad=0.0, speed_mps=8.0,
        planned_path=tuple(
            (0.1 * k, x, y, 8.0) for k, (x, y) in enumerate(path)
        ),
    )

    result = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=ego_claim, cav_intents=[peer],
    )

    assert result.diagnostics["tags"]["2"] == IGNORE
    assert result.diagnostics["roles"]["2"] == "make_gap"
    # The claim reserves priority, but the peer's broadcast path still stays
    # in its own lane. No physical occupancy -> no MPC half-space or stop.
    assert all(value >= _BIG for value in result.corridor.s_hi)

    pending = Corridor(
        s_lo=[-_BIG] * 21, s_hi=[18.0] * 21,
        binding=["2"] * 21,
    )
    cached = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=ego_claim, cav_intents=[peer],
        tag_state={"2": IGNORE}, cached_corridor=pending,
        rebuild_corridor=False, refresh_assignments=True,
    )
    assert cached.diagnostics["roles"]["2"] == "make_gap"
    # The peer's complete current path supersedes its old cached forecast.
    # Its claim keeps the negotiated role, but cannot keep an obsolete
    # physical-occupancy row after the new path remains in the adjacent lane.
    assert all(value >= _BIG for value in cached.corridor.s_hi)


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


def test_tag_transition_rebuilds_corridor_without_refreshing_stage_b_roles():
    """20 Hz geometry must not silently promote 5 Hz arbitration to 20 Hz."""

    path = [(25.0 + 0.9 * k, 3.4 - 0.16 * k) for k in range(20)]
    peer = _cav(2, (25.0, 3.4), committed_at_s=11.0, path=path)
    initial = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=_claim(committed_at_s=10.0), cav_intents=[peer],
    )
    assert initial.assignments

    changed = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        my_claim=_claim(committed_at_s=10.0), cav_intents=[peer],
        tag_state={"2": IGNORE},
        latch_state=initial.latch_state,
        refresh_assignments=False,
        cached_assignments=initial.assignments,
        rebuild_corridor=False,
        cached_corridor=initial.corridor,
    )

    assert changed.diagnostics["tags"]["2"] != IGNORE
    assert changed.diagnostics["corridor_rebuilt"] is True
    assert changed.diagnostics["coordination_roles_refreshed"] is False
    assert changed.diagnostics["coordination_refresh_reason"] == (
        "corridor_tag_changed"
    )
    assert changed.assignments == initial.assignments
    assert changed.latch_state == initial.latch_state


def test_scheduled_role_refresh_prunes_a_disappeared_peer_latch():
    stale = {
        "99": ArbitrationLatchEntry(role="yield", side="left"),
    }
    result = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot=EGO,
        my_actor_id=1,
        my_claim=_claim(committed_at_s=10.0),
        cav_intents=[],
        latch_state=stale,
        refresh_assignments=True,
    )

    assert result.latch_state == {}
    assert result.assignments == []


def test_fresh_geometric_clear_reports_actor_for_cache_retirement():
    cached = Corridor(
        s_lo=[-_BIG] * len(REF),
        s_hi=[12.0] * len(REF),
        binding=["departed"] * len(REF),
    )
    departed = {
        "id": "departed",
        "x": 20.0,
        "y": 20.0,
        "v": 0.0,
        "predicted_trajectory": [
            {"x": 20.0, "y": 20.0} for _ in range(len(REF))
        ],
    }
    result = resolve_conflicts(
        reference_samples=REF,
        ego_snapshot=EGO,
        my_actor_id=1,
        obstacle_snapshots=[departed],
        cached_corridor=cached,
        rebuild_corridor=False,
        refresh_assignments=False,
    )

    assert result.diagnostics["tags"]["departed"] == IGNORE
    assert result.released_actor_ids == ("departed",)
    assert result.diagnostics["released_actor_ids"] == ("departed",)
    assert all(value >= _BIG for value in result.corridor.s_hi)


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
    mode_trace = r_cross.diagnostics["prediction_modes"]
    assert set(mode_trace) == {"npc::mode0", "npc::mode1"}
    assert mode_trace["npc::mode0"]["probability"] == pytest.approx(0.2)
    assert mode_trace["npc::mode0"]["tag"] == "IGNORE"
    assert mode_trace["npc::mode1"]["probability"] == pytest.approx(0.8)
    assert mode_trace["npc::mode1"]["tag"] == "CROSSING"
    assert mode_trace["npc::mode1"]["raw_dangerous"]
    assert mode_trace["npc::mode1"]["credible_veto_active"]
    assert mode_trace["npc::mode1"]["veto_binding_stage_count"] > 0
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


def test_six_mode_budget_reports_raw_and_retained_counts():
    modes = [
        {
            "path": [
                {"x": 20.0 + float(k), "y": 5.0 + 0.1 * float(index)}
                for k in range(20)
            ],
            "probability": probability,
        }
        for index, probability in enumerate(
            (0.30, 0.25, 0.14, 0.12, 0.10, 0.09)
        )
    ]
    obstacle = {
        "id": "six_mode_target", "x": 20.0, "y": 5.0,
        "v": 8.0, "psi": 0.0, "predicted_modes": modes,
    }

    capped = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[obstacle], max_modes_per_agent=3,
    )
    assert capped.diagnostics["raw_prediction_mode_count"] == 6
    assert capped.diagnostics["retained_prediction_mode_count"] == 3
    assert capped.diagnostics["mode_budget_capped_agent_count"] == 1

    all_modes = resolve_conflicts(
        reference_samples=REF, ego_snapshot=EGO, my_actor_id=1,
        obstacle_snapshots=[obstacle], max_modes_per_agent=6,
    )
    assert all_modes.diagnostics["raw_prediction_mode_count"] == 6
    assert all_modes.diagnostics["retained_prediction_mode_count"] == 6
    assert all_modes.diagnostics["mode_budget_capped_agent_count"] == 0


def test_credible_veto_hysteresis_holds_through_a_brief_flicker():
    from pipeline.cav_conflict_pipeline import _apply_veto_hysteresis

    state = {}
    # tick 0: real danger -> engage immediately
    d, held, e = _apply_veto_hysteresis(
        mode_id="p::mode1", raw_dangerous=True, recovered=False,
        veto_state=state, sim_time_s=0.0, release_duration_s=0.5)
    state = {"p::mode1": e}
    assert d and not held

    # ticks 1-4: raw flag drops but risk has NOT comfortably receded ->
    # veto is HELD, not released
    for tick in range(1, 5):
        d, held, e = _apply_veto_hysteresis(
            mode_id="p::mode1", raw_dangerous=False, recovered=False,
            veto_state=state, sim_time_s=0.1 * tick,
            release_duration_s=0.5)
        state = {"p::mode1": e}
        assert d and held

    # a single raw re-trigger resets the clear streak
    d, held, e = _apply_veto_hysteresis(
        mode_id="p::mode1", raw_dangerous=True, recovered=False,
        veto_state=state, sim_time_s=0.5, release_duration_s=0.5)
    state = {"p::mode1": e}
    assert d and not held and e["clear_since_s"] is None


def test_credible_veto_releases_after_sustained_clear_and_recovery():
    from pipeline.cav_conflict_pipeline import _apply_veto_hysteresis

    state = {"p::mode1": {"dangerous": True, "clear_since_s": None}}
    for i in range(5):
        d, held, e = _apply_veto_hysteresis(
            mode_id="p::mode1", raw_dangerous=False, recovered=True,
            veto_state=state, sim_time_s=1.0 + 0.1 * i,
            release_duration_s=0.4)
        state = {"p::mode1": e}
        if i < 4:
            assert d and held           # still holding
    assert not d and not held           # released after 0.4 s clear+recovered


def test_credible_veto_release_is_independent_of_evaluation_count():
    from pipeline.cav_conflict_pipeline import _apply_veto_hysteresis

    def release_time(evaluation_times):
        state = {"p::mode1": {"dangerous": True, "clear_since_s": None}}
        released_at = None
        for now_s in evaluation_times:
            dangerous, _held, entry = _apply_veto_hysteresis(
                mode_id="p::mode1", raw_dangerous=False, recovered=True,
                veto_state=state, sim_time_s=now_s,
                release_duration_s=0.6,
            )
            state = {"p::mode1": entry}
            if not dangerous:
                released_at = now_s
                break
        return released_at

    # Extra 20 Hz tag-driven rebuilds do not consume a release "tick".
    assert release_time([1.0, 1.2, 1.4, 1.6]) == pytest.approx(1.6)
    assert release_time([
        1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30,
        1.35, 1.40, 1.45, 1.50, 1.55, 1.60,
    ]) == pytest.approx(1.6)
