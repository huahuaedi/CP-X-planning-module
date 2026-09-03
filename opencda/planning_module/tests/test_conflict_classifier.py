import math

from pipeline.conflict_classifier import (
    CROSSING,
    CUT_IN,
    FOLLOW,
    IGNORE,
    LEAD_BRAKE,
    MERGE,
    ONCOMING,
    ClassifierParams,
    classify_conflicts,
)

# Ego reference: straight +x, 0..80 m.
REF = [{"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(0, 81, 2)]
EGO = {"x": 0.0, "y": 0.0, "v": 10.0, "psi": 0.0}
P = ClassifierParams(horizon_steps=20, dt_s=0.1)


def _one(agent):
    tags = classify_conflicts(REF, EGO, [agent], P)
    assert len(tags) == 1
    return tags[0]


def _track(pts):
    return {"predicted_trajectory": [{"x": float(x), "y": float(y)} for x, y in pts]}


def test_abeam_adjacent_lane_is_ignored():
    t = _one({"id": "adj", "x": 2.0, "y": 3.6, "v": 10.0, "psi": 0.0})
    assert t.tag == IGNORE


def test_lead_vehicle_same_lane_is_follow():
    t = _one({"id": "lead", "x": 25.0, "y": 0.1, "v": 8.0, "psi": 0.0})
    assert t.tag == FOLLOW
    assert t.conflict_s_m is not None


def test_braking_lead_is_lead_brake():
    t = _one({"id": "lead", "x": 25.0, "y": 0.1, "v": 8.0, "psi": 0.0,
              "a": -2.0})
    assert t.tag == LEAD_BRAKE


def test_cut_in_track_from_adjacent_lane_onto_path():
    pts = [(6 + 1.0 * k, 3.6 - 0.18 * k) for k in range(20)]  # y: 3.6 -> ~0.2
    t = _one({"id": "cut", "x": 6.0, "y": 3.6, "v": 10.0, "psi": -0.1,
              **_track(pts)})
    assert t.tag == CUT_IN


def test_slow_converging_from_adjacent_lane_is_merge():
    pts = [(6 + 1.0 * k, 3.6 - 0.06 * k) for k in range(20)]  # y: 3.6 -> ~2.5
    t = _one({"id": "mrg", "x": 6.0, "y": 3.6, "v": 10.0, "psi": -0.02,
              **_track(pts)})
    assert t.tag == MERGE


def test_perpendicular_crosser_is_crossing():
    # heading +y (north), starts south of the path, drives across it
    pts = [(20.0, -6.0 + 1.0 * k) for k in range(20)]
    t = _one({"id": "x", "x": 20.0, "y": -6.0, "v": 8.0, "psi": math.pi / 2.0,
              **_track(pts)})
    assert t.tag == CROSSING


def test_oncoming_vehicle_is_oncoming():
    pts = [(40.0 - 1.2 * k, 0.2) for k in range(20)]
    t = _one({"id": "onc", "x": 40.0, "y": 0.2, "v": 12.0, "psi": math.pi,
              **_track(pts)})
    assert t.tag == ONCOMING


def test_far_behind_is_ignored():
    t = _one({"id": "b", "x": -40.0, "y": 0.0, "v": 10.0, "psi": 0.0})
    assert t.tag == IGNORE


def test_cooperative_flag_is_carried_through():
    t = _one({"id": "c", "x": 25.0, "y": 0.1, "v": 8.0, "psi": 0.0,
              "cooperative": True})
    assert t.cooperative is True


def test_no_reference_tags_everything_follow():
    tags = classify_conflicts([{"x_ref_m": 0.0, "y_ref_m": 0.0}], EGO,
                              [{"id": "a", "x": 5.0, "y": 5.0}], P)
    assert tags[0].tag == FOLLOW and tags[0].reason == "no_reference"
