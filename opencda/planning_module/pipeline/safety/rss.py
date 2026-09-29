"""Responsibility-Sensitive Safety (RSS) distance and timing primitives.

Formulas from Shalev-Shwartz, Shammah, Shashua, "On a Formal Model of Safe
and Efficient Automated Driving" (2017), arXiv:1708.06374 -- longitudinal
same-direction (Def. 1 / Lemma 2), longitudinal opposite-direction
(Lemma 4), and lateral (Def. 2). Pure, unit-free beyond SI.

Used by the interaction-aware planning layers:
* Stage A conflict classifier -- is the current gap already unsafe, and by
  how much (the ``violation``);
* Stage B role assignment -- ``arrival_margin_s`` (who reaches the conflict
  point first, and decisively enough to skip hysteresis);
* Stage C corridor -- ``longitudinal_safe_distance`` sizes the yield / follow
  gap that becomes ``s_hi(t)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RSSParams:
    """Reaction time and acceleration bounds. Defaults are the mid-range
    values commonly used with RSS for an urban AV; tune per deployment."""

    response_time_s: float = 0.6          # rho
    max_accel_mps2: float = 2.5           # a_max,accel during the response
    min_brake_mps2: float = 3.5           # b_min,brake (guaranteed comfortable)
    max_brake_mps2: float = 8.0           # b_max,brake (front car worst case)
    lateral_max_accel_mps2: float = 0.4   # a_lat,max,accel
    lateral_min_brake_mps2: float = 0.8   # a_lat,min,brake
    lateral_mu_m: float = 0.5             # fixed lateral fluctuation margin


def _pos(x: float) -> float:
    return x if x > 0.0 else 0.0


def longitudinal_safe_distance(
    v_rear_mps: float,
    v_front_mps: float,
    p: RSSParams = RSSParams(),
) -> float:
    """Minimum safe longitudinal distance for a rear car following a front
    car travelling the same direction (RSS Lemma 2). Clamped at >= 0.

        d_min = [ v_r*rho + 0.5*a_acc*rho^2
                  + (v_r + rho*a_acc)^2 / (2*b_min) ]
                - v_f^2 / (2*b_max)
    """

    v_r = max(0.0, float(v_rear_mps))
    v_f = max(0.0, float(v_front_mps))
    rho = float(p.response_time_s)
    a_acc = float(p.max_accel_mps2)
    b_min = max(1e-6, float(p.min_brake_mps2))
    b_max = max(1e-6, float(p.max_brake_mps2))

    v_r_after = v_r + rho * a_acc
    rear_term = v_r * rho + 0.5 * a_acc * rho * rho + (v_r_after * v_r_after) / (2.0 * b_min)
    front_term = (v_f * v_f) / (2.0 * b_max)
    return _pos(rear_term - front_term)


def longitudinal_safe_distance_opposite(
    v_ego_mps: float,
    v_other_mps: float,
    p: RSSParams = RSSParams(),
) -> float:
    """Minimum safe distance between two vehicles approaching head-on
    (RSS Lemma 4). Both are assumed to brake at ``min_brake`` after their
    response; the wrong-way / crossing vehicle is treated with the same
    bound. Speeds are magnitudes of the closing components.
    """

    rho = float(p.response_time_s)
    a_acc = float(p.max_accel_mps2)
    b_min = max(1e-6, float(p.min_brake_mps2))

    def _half(v: float) -> float:
        v = max(0.0, float(v))
        v_after = v + rho * a_acc
        return (v + v_after) / 2.0 * rho + (v_after * v_after) / (2.0 * b_min)

    return _pos(_half(v_ego_mps) + _half(v_other_mps))


def lateral_safe_distance(
    v_ego_lat_mps: float,
    v_other_lat_mps: float,
    p: RSSParams = RSSParams(),
) -> float:
    """Minimum safe lateral distance between two vehicles whose lateral
    velocities point toward each other (RSS Def. 2). ``v_*_lat`` are the
    magnitudes of the approaching lateral components.
    """

    rho = float(p.response_time_s)
    a_acc = float(p.lateral_max_accel_mps2)
    b_min = max(1e-6, float(p.lateral_min_brake_mps2))

    def _half(v: float) -> float:
        v = max(0.0, float(v))
        v_after = v + rho * a_acc
        return (2.0 * v + rho * a_acc) / 2.0 * rho + (v_after * v_after) / (2.0 * b_min)

    return _pos(float(p.lateral_mu_m) + _half(v_ego_lat_mps) + _half(v_other_lat_mps))


def longitudinal_violation(
    gap_m: float,
    v_rear_mps: float,
    v_front_mps: float,
    p: RSSParams = RSSParams(),
    *,
    opposite: bool = False,
) -> float:
    """``d_min - gap`` : how far inside the RSS envelope the pair is
    (>0 unsafe, <=0 safe). ``gap_m`` is bumper-to-bumper."""

    d_min = (
        longitudinal_safe_distance_opposite(v_rear_mps, v_front_mps, p)
        if opposite
        else longitudinal_safe_distance(v_rear_mps, v_front_mps, p)
    )
    return float(d_min) - max(0.0, float(gap_m))


def required_response_decel_mps2(
    gap_m: float,
    v_rear_mps: float,
    v_front_mps: float,
    p: RSSParams = RSSParams(),
) -> float:
    """Constant deceleration the rear car must sustain, starting after its
    response time, to restore the RSS gap before contact. 0.0 when already
    safe. Capped at ``max_brake_mps2``.
    """

    if longitudinal_violation(gap_m, v_rear_mps, v_front_mps, p) <= 0.0:
        return 0.0
    v_r = max(0.0, float(v_rear_mps))
    v_f = max(0.0, float(v_front_mps))
    rho = float(p.response_time_s)
    # Distance the closing speed eats during the response, plus the residual
    # relative speed that must then be bled off within the remaining gap.
    closing = max(0.0, v_r - v_f)
    gap_after_response = max(1e-3, float(gap_m) - closing * rho)
    decel = (closing * closing) / (2.0 * gap_after_response)
    return float(min(decel, float(p.max_brake_mps2)))


def time_to_point_s(
    distance_m: float,
    speed_mps: float,
    accel_mps2: float = 0.0,
) -> Optional[float]:
    """Time to cover ``distance_m`` at ``speed_mps`` with constant
    ``accel_mps2``. None if it never arrives (stationary / decelerating to a
    stop first). ``distance_m <= 0`` -> 0.0 (already there / past)."""

    d = float(distance_m)
    if d <= 0.0:
        return 0.0
    v = max(0.0, float(speed_mps))
    a = float(accel_mps2)
    if abs(a) < 1e-9:
        return None if v < 1e-6 else d / v
    disc = v * v + 2.0 * a * d
    if disc < 0.0:
        return None
    t = (-v + math.sqrt(disc)) / a
    return float(t) if t > 0.0 else None


def arrival_margin_s(
    ego_distance_m: float,
    ego_speed_mps: float,
    other_distance_m: float,
    other_speed_mps: float,
    *,
    ego_accel_mps2: float = 0.0,
    other_accel_mps2: float = 0.0,
) -> Optional[float]:
    """``t_other - t_ego`` at a shared conflict point: >0 => ego arrives
    first (ego may have priority), <0 => the other arrives first. None if
    either vehicle never reaches the point."""

    t_ego = time_to_point_s(ego_distance_m, ego_speed_mps, ego_accel_mps2)
    t_other = time_to_point_s(other_distance_m, other_speed_mps, other_accel_mps2)
    if t_ego is None or t_other is None:
        return None
    return float(t_other) - float(t_ego)
