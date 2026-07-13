"""Trapezoidal stop speed profile for smooth deceleration to a stop line.

Mirrors the simplified Apollo speed DP used for stop-line approach planning.
Produces a time-indexed sequence of speed caps so that the vehicle arrives at
the stop line at zero speed with a comfortable deceleration profile.

Usage:
    profile = trapezoidal_stop_profile(current_v=8.0, distance_to_stop_m=30.0)
    v_cap_now = profile[0]   # allowable speed at this tick
"""

from __future__ import annotations

import math


def trapezoidal_stop_profile(
    current_v: float,
    distance_to_stop_m: float,
    *,
    a_decel: float = 2.5,
    stop_buffer_m: float = 1.5,
    n_steps: int = 20,
    dt_s: float = 0.1,
) -> list[float]:
    """Return per-step speed caps [m/s] for decelerating to a stop line.

    Element i is the maximum allowable speed at time (i+1)*dt_s from now.
    The profile uses the kinematic constraint v <= sqrt(2 * a_decel * d)
    iterated step-by-step so that gap is consumed accurately.

    Args:
        current_v: current ego speed [m/s]
        distance_to_stop_m: distance to the stop line [m]
        a_decel: deceleration magnitude [m/s^2]
        stop_buffer_m: buffer before stop line where speed must reach 0 [m]
        n_steps: number of time steps to project
        dt_s: time step duration [s]

    Returns:
        List of n_steps speed caps in m/s.  First element is the tightest
        (most immediate) constraint; subsequent elements relax as distance
        shrinks.
    """
    v = max(0.0, float(current_v))
    d = max(0.0, float(distance_to_stop_m) - float(stop_buffer_m))
    a_decel = max(0.1, float(a_decel))
    dt = max(1e-3, float(dt_s))

    profile: list[float] = []
    for _ in range(int(n_steps)):
        v_brake_cap = math.sqrt(max(0.0, 2.0 * a_decel * d))
        v = min(v, v_brake_cap)
        v = max(0.0, v)
        profile.append(float(v))
        d = max(0.0, d - v * dt)

    return profile
