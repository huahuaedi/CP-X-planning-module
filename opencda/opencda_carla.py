"""Compatibility wrapper for OpenCDA modules expecting ``opencda_carla``.

Some upstream OpenCDA examples import CARLA geometry classes from
``opencda.opencda_carla``.  This project uses the standard CARLA Python API
directly, so this module re-exports the small set of classes those examples
need without changing their import sites.
"""

from __future__ import annotations

try:
    import carla
except ImportError as exc:  # pragma: no cover - exercised only without CARLA
    raise ImportError(
        "opencda.opencda_carla requires the CARLA Python API to be importable."
    ) from exc


Location = carla.Location
Rotation = carla.Rotation


class Transform(carla.Transform):
    """CARLA Transform with a default zero rotation for legacy OpenCDA code."""

    def __init__(self, location=None, rotation=None):
        super().__init__(
            location if location is not None else carla.Location(),
            rotation if rotation is not None else carla.Rotation(),
        )

