import math

from pipeline.mpc_corridor_constraints import corridor_rows
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
