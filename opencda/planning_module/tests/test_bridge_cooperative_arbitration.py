"""Cooperative arbitration with a real peer, through the real bridge.

CPXMPCPlannerBridge._resolve_cooperative_arbitration turns one tick's proposal,
the peers' broadcast intents and the ego reference into (cav_resolution,
defer-the-lane-change).  Peers reach it only as serialized intent payloads on
``v2x_manager.cav_nearby[id].cpx_planner.last_cav_intent_payload`` -- the same
boundary a real V2X transport would use -- so these tests build genuine
payloads with the intent codec instead of mocking the resolver.

The earlier characterization tests ran this path with no peers at all, which
left every peer-dependent decision unprotected.
"""

from __future__ import annotations

import math
import sys
import tempfile
import types
from pathlib import Path

import pytest

if "carla" not in sys.modules:
    fake_carla = types.ModuleType("carla")

    class _Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

    class _Rotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch, self.yaw, self.roll = pitch, yaw, roll

    class _Transform:
        def __init__(self, location=None, rotation=None):
            self.location = location or _Location()
            self.rotation = rotation or _Rotation()

    class _VehicleControl:
        def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
            self.throttle, self.brake, self.steer = throttle, brake, steer

    fake_carla.Location = _Location
    fake_carla.Rotation = _Rotation
    fake_carla.Transform = _Transform
    fake_carla.VehicleControl = _VehicleControl

    class _GenericStub:
        def __init__(self, *args, **kwargs):
            pass

    fake_carla.__getattr__ = lambda name: _GenericStub
    sys.modules["carla"] = fake_carla

from opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge  # noqa: E402
from pipeline.cav_intent_codec import (  # noqa: E402
    build_ego_cav_intent,
    cav_intent_to_payload,
)
from pipeline.cooperative_arbitration import ResourceClaim  # noqa: E402
from pipeline.cooperative_maneuver_proposal import (  # noqa: E402
    CooperativeManeuverProposal,
)

_TOWN06_XODR = (
    Path(__file__).resolve().parents[3]
    / "opencda" / "planning_module" / "Global_Planner" / "maps" / "Town06.xodr"
)
# Known-good Town06 route; the ego drives toward -x from the start.
START_XYZ = {"x": 225.10, "y": -20.1, "z": 0.3}
GOAL_XYZ = {"x": 10.113184332893075, "y": -96.89902155894256, "z": 0.3}
EGO_YAW_RAD = math.pi
PEER_ID = 77


class _BoundingBox:
    class extent:
        x, y = 2.25, 1.0


class _Vehicle:
    id = 1

    def __init__(self):
        self.bounding_box = _BoundingBox()

    def get_transform(self):
        import carla

        return carla.Transform()


class _Controller:
    def lon_run_step(self, *args, **kwargs):
        return 0.0


def _released_claim():
    return ResourceClaim(
        kind="lane_change", resource_id="lane_change", committed_at_s=0.0,
        active=False, require_ahead=False, phase="released",
    )


def _peer(*, claim, gap_m, speed_mps=5.0, ego_x=START_XYZ["x"]):
    """A nearby CAV whose broadcast intent is a straight plan ahead of ego."""

    x = ego_x - gap_m  # ahead along the route (-x)
    states = [
        [x - speed_mps * 0.1 * k, START_XYZ["y"], speed_mps, math.pi]
        for k in range(30)
    ]
    intent = build_ego_cav_intent(
        actor_id=PEER_ID, position_xy=(x, START_XYZ["y"]),
        heading_rad=math.pi, speed_mps=speed_mps, claim=claim,
        planned_states=states, dt_s=0.1, generated_at_s=0.0,
        valid_for_s=5.0, sequence=1, length_m=4.5, width_m=2.0,
    )
    return types.SimpleNamespace(cpx_planner=types.SimpleNamespace(
        last_cav_intent_payload=cav_intent_to_payload(intent),
    ))


@pytest.fixture(scope="module")
def cache_root():
    return tempfile.mkdtemp(prefix="cpx_coop_arbitration_")


def _make_bridge(cache_root, **config_extra):
    if not _TOWN06_XODR.is_file():
        pytest.skip(f"fixture map not found: {_TOWN06_XODR}")
    vehicle_manager = types.SimpleNamespace(
        vehicle=_Vehicle(), controller=_Controller(), carla_map=None,
        v2x_manager=types.SimpleNamespace(cav_nearby={}),
    )
    config = {
        "enabled": True, "debug": False, "record_evaluation_metrics": False,
        "publish_cp_message": False,
        "global_planner_xodr_path": str(_TOWN06_XODR),
        "global_planner_cache_root": cache_root,
        "cav_conflict_enabled": True,
    }
    config.update(config_extra)
    bridge = CPXMPCPlannerBridge(vehicle_manager, config)
    bridge.route_manager.set_destination(
        start_point=START_XYZ, goal_point=GOAL_XYZ,
    )
    return bridge, vehicle_manager


def _ego_location():
    import carla

    return carla.Location(**START_XYZ)


def _frame(bridge):
    """The per-tick inputs the bridge feeds the arbitration, from a real tick."""

    result = bridge._plan_behavior_and_reference(
        ego_location=_ego_location(), ego_yaw_rad=EGO_YAW_RAD,
        ego_speed_mps=5.0, speed_ref_mps=8.0, object_snapshots=[],
        stop_goal_active=False, cp_payload=None,
    )
    return result, dict(
        current_state=[START_XYZ["x"], START_XYZ["y"], 5.0, EGO_YAW_RAD],
        ego_location=_ego_location(), ego_yaw_rad=EGO_YAW_RAD,
        ego_speed_mps=5.0,
        local_lane_center_reference=[dict(s) for s in result.reference_samples],
        local_map_snapshot=bridge._local_map_snapshot,
        object_snapshots=[], planned_speed_mps=8.0,
        planner_input_frame=types.SimpleNamespace(
            prediction=types.SimpleNamespace(revision="r1", predicted_objects=[])
        ),
    )


def _lane_change_lanes(bridge):
    """(source, target) lane ids: the ego lane and its neighbour to the right."""

    snapshot = bridge._local_map_snapshot
    source = int(snapshot.ego_lane_id)
    target = next(
        int(lane) for lane, _offset in snapshot.lane_to_offset_items
        if int(lane) != source
    )
    return source, target


def _proposal(bridge):
    source, target = _lane_change_lanes(bridge)
    return CooperativeManeuverProposal.from_behavior(
        maneuver="lane_change_right", source_corridor_id=source,
        target_corridor_id=target, route_required=True,
        maneuver_active=False, committed_at_s=0.0,
    )


def _peer_lane_change_claim(bridge, *, committed_at_s, phase="committed"):
    source, target = _lane_change_lanes(bridge)
    return ResourceClaim(
        kind="lane_change", resource_id=f"lane_change:{source}:{target}",
        committed_at_s=float(committed_at_s), active=True,
        require_ahead=False, phase=phase,
        source_corridor_id=source, target_corridor_id=target,
    )


def _arbitrate(bridge, frame, *, sim_time_s, proposal):
    return bridge._resolve_cooperative_arbitration(
        cooperative_proposal=proposal, sim_time_s=sim_time_s, **frame,
    )


def _roles(resolution):
    return {a.cav_actor_id: a.role for a in resolution.assignments}


def test_disabled_arbitration_returns_nothing_and_touches_no_claim_state(cache_root):
    bridge, _ = _make_bridge(cache_root, cav_conflict_enabled=False)
    _, frame = _frame(bridge)

    resolution, deferred = _arbitrate(
        bridge, frame, sim_time_s=0.0, proposal=_proposal(bridge)
    )

    assert resolution is None
    assert deferred is False
    assert bridge._cooperative.claims.current_claim is None


def test_same_lane_peer_ahead_is_followed_and_seen_over_the_transport(cache_root):
    bridge, vehicle_manager = _make_bridge(cache_root)
    vehicle_manager.v2x_manager.cav_nearby = {
        PEER_ID: _peer(claim=_released_claim(), gap_m=15.0, speed_mps=2.0)
    }

    result, _ = _frame(bridge)

    diagnostics = result.cav_resolution.diagnostics
    assert diagnostics["tags"] == {str(PEER_ID): "FOLLOW"}
    assert diagnostics["observation_sources"] == {str(PEER_ID): "peer_intent"}
    assert diagnostics["transport"]["payload_count"] == 1
    # Nobody claims a shared resource, so this is car-following, not a role.
    assert list(result.cav_resolution.assignments) == []
    assert diagnostics["arbitration"] == {"reason": "ego_claim_inactive"}


def test_first_proposal_tick_is_deferred_so_both_peers_see_the_same_claims(cache_root):
    bridge, _ = _make_bridge(cache_root)
    _, frame = _frame(bridge)

    resolution, deferred = _arbitrate(
        bridge, frame, sim_time_s=0.0, proposal=_proposal(bridge)
    )

    assert resolution is not None
    assert deferred is True
    assert bridge._cooperative.claims.current_claim.phase == "proposed"


def test_proposal_is_released_after_the_dwell_when_no_peer_conflicts(cache_root):
    bridge, _ = _make_bridge(cache_root)
    _, frame = _frame(bridge)
    proposal = _proposal(bridge)
    _arbitrate(bridge, frame, sim_time_s=0.0, proposal=proposal)

    _, deferred = _arbitrate(bridge, frame, sim_time_s=1.0, proposal=proposal)

    assert deferred is False


def _contest(bridge, vehicle_manager, frame, *, peer_claim, gap_m=8.0):
    vehicle_manager.v2x_manager.cav_nearby = {
        PEER_ID: _peer(claim=peer_claim, gap_m=gap_m)
    }
    proposal = _proposal(bridge)
    _arbitrate(bridge, frame, sim_time_s=0.0, proposal=proposal)
    return _arbitrate(bridge, frame, sim_time_s=1.0, proposal=proposal)


@pytest.mark.parametrize(
    "gap_m, peer_committed_at_s",
    [(8.0, -5.0), (-8.0, -5.0), (8.0, 5.0)],
    ids=["peer-ahead", "peer-behind", "peer-committed-later-still-wins"],
)
def test_a_committed_peer_claim_preempts_the_ego_proposal(
    cache_root, gap_m, peer_committed_at_s
):
    # "A physically committed maneuver cannot be pre-empted by a proposal":
    # commitment beats a proposal regardless of either timestamp.
    bridge, vehicle_manager = _make_bridge(cache_root)
    _, frame = _frame(bridge)

    resolution, deferred = _contest(
        bridge, vehicle_manager, frame, gap_m=gap_m,
        peer_claim=_peer_lane_change_claim(
            bridge, committed_at_s=peer_committed_at_s
        ),
    )

    assert _roles(resolution) == {PEER_ID: "make_gap"}
    assert resolution.assignments[0].cav_wins is True
    assert deferred is True, "the loser must not install its lane change yet"


def test_between_two_proposals_the_earlier_one_wins(cache_root):
    bridge, vehicle_manager = _make_bridge(cache_root)
    _, frame = _frame(bridge)

    resolution, deferred = _contest(
        bridge, vehicle_manager, frame,
        peer_claim=_peer_lane_change_claim(
            bridge, committed_at_s=5.0, phase="proposed"
        ),
    )

    assert resolution.assignments[0].cav_wins is False
    assert deferred is False


def test_between_two_proposals_the_later_one_makes_way(cache_root):
    bridge, vehicle_manager = _make_bridge(cache_root)
    _, frame = _frame(bridge)

    resolution, deferred = _contest(
        bridge, vehicle_manager, frame,
        peer_claim=_peer_lane_change_claim(
            bridge, committed_at_s=-5.0, phase="proposed"
        ),
    )

    assert _roles(resolution) == {PEER_ID: "make_gap"}
    assert resolution.assignments[0].cav_wins is True
    assert deferred is True


def test_route_change_retires_the_stale_claim_so_a_new_proposal_dwells_again(cache_root):
    bridge, _ = _make_bridge(cache_root)
    _, frame = _frame(bridge)
    proposal = _proposal(bridge)
    _arbitrate(bridge, frame, sim_time_s=0.0, proposal=proposal)
    _, past_dwell = _arbitrate(bridge, frame, sim_time_s=1.0, proposal=proposal)
    assert past_dwell is False  # the same proposal, 1 s later, is released

    bridge.route_manager.set_destination(
        start_point=START_XYZ, goal_point=GOAL_XYZ,
    )
    _, deferred = _arbitrate(bridge, frame, sim_time_s=1.0, proposal=proposal)

    assert deferred is True, "a new route must restart the proposal dwell"


def test_config_knobs_and_planner_limits_reach_the_interaction_resolver(cache_root):
    knobs = {
        "cav_conflict_comfort_deceleration_mps2": 0.77,
        "candidate_lane_change_normal_duration_s": 5.5,
        "prediction_mode_min_probability": 0.11,
        "prediction_credible_probability_min": 0.22,
        "prediction_credible_ttc_s": 3.3,
        "prediction_credible_veto_release_ticks": 9,
    }
    bridge, _ = _make_bridge(cache_root, **knobs)
    _, frame = _frame(bridge)
    captured = []
    real = bridge.pipeline.resolve_cav_interaction

    def spy(**kwargs):
        captured.append(kwargs)
        return real(**kwargs)

    bridge.pipeline.resolve_cav_interaction = spy

    _arbitrate(bridge, frame, sim_time_s=0.0, proposal=_proposal(bridge))

    assert len(captured) == 1
    seen = captured[0]
    assert seen["comfortable_deceleration_mps2"] == pytest.approx(0.77)
    assert seen["cooperative_preparation_time_s"] == pytest.approx(5.5)
    assert seen["mode_probability_floor"] == pytest.approx(0.11)
    assert seen["credible_mode_probability_min"] == pytest.approx(0.22)
    assert seen["credible_mode_ttc_s"] == pytest.approx(3.3)
    assert seen["credible_mode_veto_release_ticks"] == 9
    constraints = bridge.mpc.constraints
    assert seen["max_braking_mps2"] == pytest.approx(
        abs(constraints.min_acceleration_mps2)
    )
    assert seen["max_jerk_mps3"] == pytest.approx(constraints.max_jerk_mps3)
    assert seen["actor_id"] == 1
    assert seen["current_acceleration_mps2"] == pytest.approx(
        bridge._last_accel_mps2
    )
    assert seen["max_relevant_agents"] == (
        bridge._cooperative.governor.current_max_relevant_agents
    )


def test_unconfigured_resolver_knobs_keep_their_documented_defaults(cache_root):
    # These defaults shape cooperative behavior on every run that does not
    # override them, so changing one must be a deliberate, visible decision.
    bridge, _ = _make_bridge(cache_root)
    _, frame = _frame(bridge)
    captured = []
    real = bridge.pipeline.resolve_cav_interaction

    def spy(**kwargs):
        captured.append(kwargs)
        return real(**kwargs)

    bridge.pipeline.resolve_cav_interaction = spy

    _arbitrate(bridge, frame, sim_time_s=0.0, proposal=_proposal(bridge))

    seen = captured[0]
    assert seen["comfortable_deceleration_mps2"] == pytest.approx(1.5)
    assert seen["cooperative_preparation_time_s"] == pytest.approx(4.0)
    assert seen["mode_probability_floor"] == pytest.approx(0.05)
    assert seen["credible_mode_probability_min"] == pytest.approx(0.15)
    assert seen["credible_mode_ttc_s"] == pytest.approx(2.0)
    assert seen["credible_mode_veto_release_ticks"] == 12

