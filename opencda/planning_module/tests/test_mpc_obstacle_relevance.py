from pipeline.mpc_obstacle_relevance import split_relevant_mpc_obstacles


# Ego reference: straight line along +x from (0,0) to (60,0).
REF = [{"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(0, 61, 2)]


def _obs(oid, x, y, track=None):
    s = {"id": oid, "x": float(x), "y": float(y), "v": 8.0, "psi": 0.0}
    if track is not None:
        s["predicted_trajectory"] = [{"x": float(px), "y": float(py)} for px, py in track]
    return s


def _ids(snaps):
    return {s["id"] for s in snaps}


def test_abeam_adjacent_lane_vehicle_is_ignored():
    rel, ign = split_relevant_mpc_obstacles(
        [_obs("adj", 3.0, 3.6)], REF, lateral_gate_m=3.0
    )
    assert _ids(ign) == {"adj"} and rel == []


def test_lead_vehicle_same_lane_is_relevant():
    rel, ign = split_relevant_mpc_obstacles(
        [_obs("lead", 15.0, 0.2)], REF, lateral_gate_m=3.0
    )
    assert _ids(rel) == {"lead"} and ign == []


def test_cut_in_track_that_enters_the_band_is_relevant():
    # starts one lane over, predicted to merge onto the ego path
    track = [(4.0, 3.6), (6.0, 2.5), (8.0, 1.2), (10.0, 0.3)]
    rel, ign = split_relevant_mpc_obstacles(
        [_obs("cutin", 4.0, 3.6, track=track)], REF, lateral_gate_m=3.0
    )
    assert _ids(rel) == {"cutin"} and ign == []


def test_far_ahead_vehicle_beyond_longitudinal_gate_is_ignored():
    rel, ign = split_relevant_mpc_obstacles(
        [_obs("far", 80.0, 0.0)], REF, longitudinal_gate_m=40.0
    )
    assert _ids(ign) == {"far"}


def test_vehicle_far_behind_is_ignored():
    rel, ign = split_relevant_mpc_obstacles([_obs("behind", -30.0, 0.0)], REF)
    assert _ids(ign) == {"behind"}


def test_keep_ids_override_the_filter():
    rel, ign = split_relevant_mpc_obstacles(
        [_obs("adj", 3.0, 3.6)], REF, lateral_gate_m=3.0, keep_ids=["adj"]
    )
    assert _ids(rel) == {"adj"} and ign == []


def test_degenerate_reference_keeps_everything():
    rel, ign = split_relevant_mpc_obstacles([_obs("a", 3.0, 3.6)], [{"x_ref_m": 0.0, "y_ref_m": 0.0}])
    assert _ids(rel) == {"a"} and ign == []


def test_empty_input_is_safe():
    rel, ign = split_relevant_mpc_obstacles([], REF)
    assert rel == [] and ign == []


def test_track_with_any_in_band_sample_is_relevant_even_if_it_leaves():
    # oncoming/crossing: enters the path region then exits laterally
    track = [(20.0, 4.0), (20.0, 1.0), (20.0, -1.0), (20.0, -4.0)]
    rel, ign = split_relevant_mpc_obstacles(
        [_obs("cross", 20.0, 4.0, track=track)], REF, lateral_gate_m=3.0
    )
    assert _ids(rel) == {"cross"}
