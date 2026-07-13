"""Coordinate and point helpers shared by the planner package."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple, Union

CARLA_TO_ENU_Y_SIGN = -1.0


def normalize_position(position: Union[Mapping[str, float], Sequence[float]]) -> Tuple[float, float, float]:
    """Convert a CARLA-style point into a plain `(x, y, z)` tuple.

    input: `position` (`Dict[str, float] | List[float] | Tuple[float, float, float]`)
    output: normalized point (`Tuple[float, float, float]`)
    """
    if isinstance(position, Mapping):
        return (float(position["x"]), float(position["y"]), float(position["z"]))
    return (float(position[0]), float(position[1]), float(position[2]))


def as_carla_dict(position: Union[Mapping[str, float], Sequence[float]]) -> Dict[str, float]:
    """Return one CARLA-style point dictionary with `x`, `y`, and `z`.

    input: `position` (`Dict[str, float] | List[float] | Tuple[float, float, float]`)
    output: CARLA point (`Dict[str, float]`)
    """
    x, y, z = normalize_position(position)
    return {"x": x, "y": y, "z": z}


def to_enu_tuple(position: Union[Mapping[str, float], Sequence[float]]) -> Tuple[float, float, float]:
    """Convert one CARLA-world point into the AD-map ENU convention.

    input: `position` (`Dict[str, float] | List[float] | Tuple[float, float, float]`)
    output: ENU point (`Tuple[float, float, float]`)
    """
    x, y, z = normalize_position(position)
    return (x, CARLA_TO_ENU_Y_SIGN * y, z)


def to_carla_tuple(position: Union[Mapping[str, float], Sequence[float]]) -> Tuple[float, float, float]:
    """Convert one ENU point back into the CARLA-world convention.

    input: `position` (`Dict[str, float] | List[float] | Tuple[float, float, float]`)
    output: CARLA point (`Tuple[float, float, float]`)
    """
    x, y, z = normalize_position(position)
    return (x, CARLA_TO_ENU_Y_SIGN * y, z)


def to_carla_dict(position: Union[Mapping[str, float], Sequence[float]]) -> Dict[str, float]:
    """Convert one ENU point into a CARLA-style dictionary.

    input: `position` (`Dict[str, float] | List[float] | Tuple[float, float, float]`)
    output: CARLA point (`Dict[str, float]`)
    """
    x, y, z = to_carla_tuple(position)
    return {"x": x, "y": y, "z": z}


def points_are_close(
    first_point: Union[Mapping[str, float], Sequence[float]],
    second_point: Union[Mapping[str, float], Sequence[float]],
    tolerance: float = 0.05,
) -> bool:
    """Check whether two 3D points are nearly identical.

    input: two points (`Dict[str, float] | Tuple[float, float, float]`) and `tolerance` (`float`)
    output: whether the points are close (`bool`)
    """
    first_tuple = normalize_position(first_point)
    second_tuple = normalize_position(second_point)
    return all(abs(first_value - second_value) <= tolerance for first_value, second_value in zip(first_tuple, second_tuple))


def path_length(points: Sequence[Union[Mapping[str, float], Sequence[float]]]) -> float:
    """Measure the length of a point-by-point path.

    input: `points` (`List[Dict[str, float] | Tuple[float, float, float]]`)
    output: path length in meters (`float`)
    """
    if len(points) < 2:
        return 0.0
    total_length = 0.0
    for first_point, second_point in zip(points, points[1:]):
        total_length += math.dist(normalize_position(first_point), normalize_position(second_point))
    return total_length
