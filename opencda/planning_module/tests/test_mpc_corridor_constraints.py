import math

from pipeline.cooperative_arbitration import ConflictAssignment
from pipeline.mpc_corridor_constraints import (
    corridor_rows,
    homotopy_keepout_rows,
)
from pipeline.spatiotemporal_corridor import Corridor

_BIG = 1.0e9
# straight +x reference
REF = [{"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(0, 101, 5)]


def test_open_corridor_yields_no_rows():
    cor = Corridor(s_lo=[-_BIG] * 5, s_hi=[_BIG] * 5, binding=[""] * 5)
    assert corridor_rows(cor, REF) == []


def test_s_hi_cap_becomes_an_upper_bound_along_the_tangent():
    cor = Corridor(s_lo=[-_BIG] * 5, s_hi=[_BIG, _BIG, 30.0, _BIG, _BIG],
                   binding=["", "", "lead", "", ""])
    rows = corridor_rows(cor, REF)
    assert len(rows) == 1
    r = rows[0]
    assert r.stage == 2
    # tangent of a +x line is (1, 0): the row is 1*x + 0*y <= 30
    assert math.isclose(r.a_x, 1.0, abs_tol=1e-9)
    assert math.isclose(r.a_y, 0.0, abs_tol=1e-9)
    assert r.upper == 30.0
    assert r.lower <= -_BIG
    assert r.tag == "lead"


def test_origin_offset_is_folded_into_the_frame():
    cor = Corridor(s_lo=[-_BIG, -_BIG], s_hi=[_BIG, 40.0], binding=["", ""])
    # ego sits 10 m along the path -> in the shifted frame the same world
    # cap x_world<=40 is x_shifted<=30
    rows = corridor_rows(cor, REF, ego_origin_xy=(10.0, 0.0))
    assert len(rows) == 1
    assert math.isclose(rows[0].upper, 30.0, abs_tol=1e-6)


def test_homotopy_left_keeps_ego_on_the_lefthand_half_space():
    a = ConflictAssignment(peer_actor_id=2, role="proceed", homotopy_side="left",
                           conflict_xy=None, peer_wins=False, reason="")
    # peer straight ahead on the x-axis, ego heading +x
    track = {2: [(10.0, 0.0), (12.0, 0.0)]}
    rows = homotopy_keepout_rows([a], track, ego_heading_rad=0.0, d_safe_m=2.0)
    assert len(rows) == 2
    r = rows[0]
    # left normal for heading 0 is (0, 1): 0*x + 1*y >= (0*10 + 1*0 + 2) = 2
    assert math.isclose(r.n_x, 0.0, abs_tol=1e-9)
    assert math.isclose(r.n_y, 1.0, abs_tol=1e-9)
    assert math.isclose(r.rhs, 2.0, abs_tol=1e-9)


def test_homotopy_right_flips_the_normal():
    a = ConflictAssignment(peer_actor_id=3, role="proceed", homotopy_side="right",
                           conflict_xy=None, peer_wins=False, reason="")
    rows = homotopy_keepout_rows([a], {3: [(10.0, 0.0)]}, ego_heading_rad=0.0,
                                 d_safe_m=2.0)
    assert math.isclose(rows[0].n_y, -1.0, abs_tol=1e-9)
    assert math.isclose(rows[0].rhs, 2.0, abs_tol=1e-9)


def test_assignment_without_a_side_produces_nothing():
    a = ConflictAssignment(peer_actor_id=4, role="yield", homotopy_side="",
                           conflict_xy=None, peer_wins=True, reason="")
    assert homotopy_keepout_rows([a], {4: [(1.0, 1.0)]}, ego_heading_rad=0.0) == []
