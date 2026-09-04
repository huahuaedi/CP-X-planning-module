from pipeline.cooperative_arbitration import CavIntent, ResourceClaim
from pipeline.cav_intent_codec import (
    SCHEMA_VERSION,
    build_ego_cav_intent,
    collect_cav_intents,
    cav_intent_from_payload,
    cav_intent_to_payload,
    sample_cav_path_at,
)


def _claim(committed_at_s=10.0):
    return ResourceClaim(
        kind="lane_change", resource_id="lane_change",
        committed_at_s=committed_at_s, active=True,
    )


def test_build_ego_intent_converts_state_sequence_to_timed_path():
    states = [[0.0, 0.0, 5.0, 0.0], [0.5, 0.1, 5.2, 0.0], [1.0, 0.3, 5.4, 0.0]]
    intent = build_ego_cav_intent(
        actor_id=7, position_xy=(0.0, 0.0), heading_rad=0.0, speed_mps=5.0,
        claim=_claim(), planned_states=states, dt_s=0.1, t0_s=0.0,
    )
    assert intent.actor_id == 7
    assert len(intent.planned_path) == 3
    assert intent.planned_path[0] == (0.0, 0.0, 0.0, 5.0)
    assert intent.planned_path[2] == (0.2, 1.0, 0.3, 5.4)


def test_codec_roundtrip():
    intent = build_ego_cav_intent(
        actor_id=3, position_xy=(12.0, -1.5), heading_rad=1.57, speed_mps=8.0,
        claim=_claim(6.0), planned_states=[[12.0, -1.5, 8.0, 1.57], [12.1, 0.4, 8.0, 1.57]],
    )
    payload = cav_intent_to_payload(intent)
    assert payload["schema"] == SCHEMA_VERSION
    back = cav_intent_from_payload(payload)
    assert back is not None
    assert back.actor_id == 3
    assert back.position_xy == (12.0, -1.5)
    assert back.claim.committed_at_s == 6.0
    assert back.claim.kind == "lane_change"
    assert back.claim.phase == "committed"
    assert len(back.planned_path) == 2


def test_from_payload_rejects_garbage():
    assert cav_intent_from_payload(None) is None
    assert cav_intent_from_payload({}) is None
    assert cav_intent_from_payload({"schema": 999, "actor_id": 1, "claim": {}}) is None
    assert cav_intent_from_payload({"actor_id": 1}) is None                 # no claim
    assert cav_intent_from_payload({"actor_id": 1, "claim": {"kind": ""}}) is None
    assert cav_intent_from_payload({"claim": {"kind": "k", "resource_id": "r"}}) is None  # no id


def test_collect_skips_self_and_unparseable():
    self_payload = cav_intent_to_payload(
        build_ego_cav_intent(actor_id=1, position_xy=(0, 0), heading_rad=0,
                              speed_mps=0, claim=_claim())
    )
    cav_payload = cav_intent_to_payload(
        build_ego_cav_intent(actor_id=2, position_xy=(10, 0), heading_rad=0,
                              speed_mps=6, claim=_claim(4.0))
    )
    records = [self_payload, cav_payload, {"actor_id": 3, "junk": True}]
    cavs = collect_cav_intents(records, self_actor_id=1)
    assert [p.actor_id for p in cavs] == [2]


def test_collect_reads_nested_intent_and_backfills_id():
    records = [
        {
            "vehicle_id": 5,
            "x": 20.0,
            "cooperative_intent": {
                "claim": {"kind": "lane_change", "resource_id": "lane_change",
                          "committed_at_s": 3.0, "active": True},
                "x": 20.0, "y": 0.0, "speed_mps": 7.0,
            },
        }
    ]
    cavs = collect_cav_intents(records, self_actor_id=1)
    assert len(cavs) == 1 and cavs[0].actor_id == 5
    assert cavs[0].claim.committed_at_s == 3.0


def test_collect_drops_expired_low_probability_and_keeps_newest_sequence():
    def payload(sequence, generated_at_s, valid_for_s, probability):
        return cav_intent_to_payload(build_ego_cav_intent(
            actor_id=2, position_xy=(10, 0), heading_rad=0, speed_mps=6,
            claim=_claim(), sequence=sequence, generated_at_s=generated_at_s,
            valid_for_s=valid_for_s, probability=probability,
        ))

    records = [
        payload(1, 9.0, 2.0, 1.0),
        payload(3, 10.0, 1.0, 1.0),
        payload(2, 10.0, 1.0, 1.0),
        payload(4, 8.0, 0.5, 1.0),
        payload(5, 10.0, 1.0, 0.01),
    ]
    cavs = collect_cav_intents(
        records, self_actor_id=1, now_s=10.25, minimum_probability=0.05
    )
    assert len(cavs) == 1
    assert cavs[0].sequence == 3


def test_codec_roundtrip_preserves_transport_metadata():
    intent = build_ego_cav_intent(
        actor_id=3, position_xy=(1, 2), heading_rad=0.2, speed_mps=4,
        claim=_claim(), generated_at_s=7.0, valid_for_s=0.4,
        sequence=12, probability=0.8,
    )
    back = cav_intent_from_payload(cav_intent_to_payload(intent))
    assert back is not None
    assert back.generated_at_s == 7.0
    assert back.valid_until_s == 7.4
    assert back.sequence == 12
    assert back.probability == 0.8


def test_sample_cav_path_interpolates_and_clamps():
    intent = CavIntent(
        actor_id=1, position_xy=(0, 0), claim=_claim(),
        planned_path=((0.0, 0.0, 0.0, 5.0), (1.0, 5.0, 0.0, 5.0), (2.0, 10.0, 2.0, 4.0)),
    )
    assert sample_cav_path_at(intent, -1.0) == (0.0, 0.0, 5.0)          # clamp low
    assert sample_cav_path_at(intent, 0.5) == (2.5, 0.0, 5.0)          # interp
    x, y, v = sample_cav_path_at(intent, 1.5)
    assert (round(x, 3), round(y, 3), round(v, 3)) == (7.5, 1.0, 4.5)
    assert sample_cav_path_at(intent, 99.0) == (10.0, 2.0, 4.0)        # clamp high
    assert sample_cav_path_at(CavIntent(actor_id=1, position_xy=(0, 0), claim=_claim()), 0.5) is None
