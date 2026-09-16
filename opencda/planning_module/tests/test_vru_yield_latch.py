from pipeline.behavior_risk import SemanticBehaviorResponse
from pipeline.conflict_classifier import NO_RISK, VRU_CONFLICT
from pipeline.vru_yield_latch import VRUYieldLatch


def _vru_response(action, obstacle_id="walker_1", object_type="pedestrian"):
    return SemanticBehaviorResponse(
        risk_kind=VRU_CONFLICT, action=action, object_type=object_type,
        obstacle_id=obstacle_id, distance_m=6.0,
        reason="vru_within_dynamic_stopping_reach"
        if action == "YIELD_STOP" else "vru_outside_dynamic_stopping_reach",
    )


def test_latch_passes_through_when_never_triggered():
    latch = VRUYieldLatch()
    response = _vru_response("NONE")
    assert latch.update(response) is response


def test_latch_holds_yield_stop_once_envelope_shrinks_for_same_obstacle():
    latch = VRUYieldLatch()
    triggered = _vru_response("YIELD_STOP")
    assert latch.update(triggered).action == "YIELD_STOP"

    # Ego braked in response; the dynamic envelope recomputed smaller and
    # flipped the pure classifier back to NONE for the *same* obstacle.
    shrunk = _vru_response("NONE")
    held = latch.update(shrunk)
    assert held.action == "YIELD_STOP"
    assert held.reason == "vru_yield_commitment_latched"


def test_latch_releases_once_obstacle_id_changes():
    latch = VRUYieldLatch()
    latch.update(_vru_response("YIELD_STOP", obstacle_id="walker_1"))

    other = _vru_response("NONE", obstacle_id="walker_2")
    released = latch.update(other)
    assert released.action == "NONE"
    assert released is other


def test_latch_releases_once_front_obstacle_is_no_longer_a_vru():
    latch = VRUYieldLatch()
    latch.update(_vru_response("YIELD_STOP"))

    no_risk = SemanticBehaviorResponse(risk_kind=NO_RISK, action="NONE")
    released = latch.update(no_risk)
    assert released.action == "NONE"
    assert released is no_risk


def test_reset_clears_the_latch():
    latch = VRUYieldLatch()
    latch.update(_vru_response("YIELD_STOP"))
    latch.reset()

    shrunk = _vru_response("NONE")
    assert latch.update(shrunk) is shrunk
