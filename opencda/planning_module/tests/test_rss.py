import math

from pipeline.rss import (
    RSSParams,
    arrival_margin_s,
    lateral_safe_distance,
    longitudinal_safe_distance,
    longitudinal_safe_distance_opposite,
    longitudinal_violation,
    required_response_decel_mps2,
    time_to_point_s,
)

P = RSSParams()


def test_stationary_pair_needs_only_the_response_window_accel_distance():
    # RSS still assumes the rear car may floor it for one response time: the
    # distance is small but not zero.
    d = longitudinal_safe_distance(0.0, 0.0, P)
    expect = 0.5 * P.max_accel_mps2 * P.response_time_s ** 2 + (
        (P.response_time_s * P.max_accel_mps2) ** 2 / (2.0 * P.min_brake_mps2)
    )
    assert math.isclose(d, expect, rel_tol=1e-9)
    assert 0.0 < d < 1.5


def test_safe_distance_is_monotonic_in_rear_speed():
    d = [longitudinal_safe_distance(v, 10.0, P) for v in (0, 5, 10, 15, 20)]
    assert all(b >= a for a, b in zip(d, d[1:]))
    assert d[-1] > d[0]


def test_faster_front_car_reduces_required_distance():
    slow_front = longitudinal_safe_distance(15.0, 0.0, P)
    fast_front = longitudinal_safe_distance(15.0, 15.0, P)
    assert fast_front < slow_front


def test_hand_computed_same_direction_value():
    # v_r=10, v_f=0, rho=0.6, a_acc=2.5, b_min=3.5, b_max=8
    # rear = 10*0.6 + 0.5*2.5*0.36 + (10+1.5)^2/(2*3.5)
    #      = 6 + 0.45 + 132.25/7 = 6.45 + 18.892857... = 25.342857...
    d = longitudinal_safe_distance(10.0, 0.0, P)
    assert math.isclose(d, 6.0 + 0.45 + (11.5 ** 2) / 7.0, rel_tol=1e-9)


def test_opposite_direction_needs_more_than_same_direction():
    same = longitudinal_safe_distance(12.0, 12.0, P)
    opp = longitudinal_safe_distance_opposite(12.0, 12.0, P)
    assert opp > same


def test_lateral_distance_is_at_least_the_fixed_margin_and_grows_with_speed():
    assert lateral_safe_distance(0.0, 0.0, P) >= P.lateral_mu_m
    assert lateral_safe_distance(1.0, 1.0, P) > lateral_safe_distance(0.0, 0.0, P)


def test_violation_sign():
    # 8 m gap, ego 15 closing on a stopped car -> d_min ~ 40 m -> unsafe
    assert longitudinal_violation(8.0, 15.0, 0.0, P) > 0.0
    # 100 m gap -> safe
    assert longitudinal_violation(100.0, 15.0, 0.0, P) < 0.0


def test_required_decel_zero_when_safe_and_positive_capped_when_not():
    assert required_response_decel_mps2(100.0, 15.0, 0.0, P) == 0.0
    d = required_response_decel_mps2(6.0, 18.0, 0.0, P)
    assert 0.0 < d <= P.max_brake_mps2


def test_time_to_point():
    assert time_to_point_s(0.0, 5.0) == 0.0
    assert time_to_point_s(10.0, 0.0) is None            # never arrives
    assert math.isclose(time_to_point_s(10.0, 5.0), 2.0)
    # from rest at 2 m/s^2 over 4 m: t = 2 s
    assert math.isclose(time_to_point_s(4.0, 0.0, 2.0), 2.0)
    # decelerating and stops before the point
    assert time_to_point_s(100.0, 3.0, -1.0) is None


def test_arrival_margin_sign():
    # ego 20 m @ 10 m/s (t=2.0); other 30 m @ 6 m/s (t=5.0) -> +3.0, ego first
    assert math.isclose(
        arrival_margin_s(20.0, 10.0, 30.0, 6.0), 3.0
    )
    # other arrives first -> negative
    assert arrival_margin_s(40.0, 10.0, 10.0, 10.0) < 0.0
    # other never arrives -> None
    assert arrival_margin_s(20.0, 10.0, 10.0, 0.0) is None
