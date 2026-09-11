import math

from pipeline.cooperative_arbitration import ConflictAssignment
from pipeline.mpc_corridor_constraints import corridor_rows, homotopy_keepout_rows
from pipeline.spatiotemporal_corridor import Corridor

_BIG = 1.0e9
# straight +x reference
REF = [{"x_ref_m": float(x), "y_ref_m": 0.0} for x in range(0, 101, 5)]


def test_open_corridor_yields_no_rows():
    cor = Corridor(s_lo=[-_BIG] * 5, s_hi=[_BIG] * 5, binding=[""] * 5)
    assert corridor_rows(cor, REF) == []


def test_curved_nonuniform_reference_cap_has_correct_world_boundary():
    # Station 15 is (10, 5), on the northbound segment, regardless of
    # extra samples on the first metre of the eastbound segment.
    for points in ([(0, 0), (10, 0), (10, 20)],
                   [(0, 0), (0.1, 0), (0.2, 0), (1, 0), (10, 0), (10, 20)]):
        reference = [{"x": x, "y": y} for x, y in points]
        corridor = Corridor(s_lo=[-_BIG, -_BIG], s_hi=[_BIG, 15],
                            binding=["", "crossing"])
        for ox, oy in [(0, 0), (3, 2)]:
            row = corridor_rows(corridor, reference, (ox, oy))[0]
            assert abs(row.a_x) < 1e-9
            assert abs(row.a_y - 1) < 1e-9
            assert abs(row.a_x * (10 - ox) + row.a_y * (5 - oy)
                       - row.upper) < 1e-9
            assert row.a_x * (10 - ox) + row.a_y * (6 - oy) > row.upper


def test_current_state_stage_is_never_constrained():
    cor = Corridor(
        s_lo=[-_BIG, -_BIG],
        s_hi=[-1.0, 5.0],
        binding=["already_violated", "future"],
    )
    rows = corridor_rows(cor, REF)
    assert [row.stage for row in rows] == [1]


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


def test_homotopy_half_space_uses_the_mpc_linear_row_schema():
    assignment = ConflictAssignment(
        cav_actor_id=2, role="proceed", homotopy_side="left",
        cav_wins=False, reason="",
    )
    rows = homotopy_keepout_rows(
        [assignment], {2: [(10.0, 0.0), (12.0, 0.0)]},
        ego_heading_rad=0.0, d_safe_m=2.0,
    )
    assert len(rows) == 1  # MPC constrains stages 1..N; stage zero is fixed.
    row = rows[0]
    assert row.stage == 1
    assert math.isclose(row.a_x, 0.0, abs_tol=1e-9)
    assert math.isclose(row.a_y, 1.0, abs_tol=1e-9)
    assert math.isclose(row.lower, 2.0, abs_tol=1e-9)
    assert row.upper >= _BIG


def test_homotopy_row_is_shifted_into_ego_origin_frame():
    assignment = ConflictAssignment(
        cav_actor_id=3, role="proceed", homotopy_side="right",
        cav_wins=False, reason="",
    )
    rows = homotopy_keepout_rows(
        [assignment], {3: [(5.0, 12.0), (5.0, 13.0)]},
        ego_heading_rad=0.0, ego_origin_xy=(5.0, 10.0), d_safe_m=1.0,
    )
    # Right normal is (0,-1); peer stage-1 is y_shift=3, so -y >= -2.
    assert math.isclose(rows[0].lower, -2.0, abs_tol=1e-9)
