"""Per-agent multi-modal prediction interface for the interaction pipeline.

Today every producer emits exactly ONE mode:

* a connected vehicle's broadcast ``planned_path`` (its committed MPC plan), or
* the prediction module's single deterministic future for a non-connected
  road user.

The list-of-``PredictedMode`` shape exists now so a real multi-modal
predictor -- one mode per hypothesised driver intent, each with a
probability, e.g.::

    keep lane        p = 0.60
    change left      p = 0.30
    brake and follow p = 0.10

is a drop-in later (Contingency-MPC style) without touching Stage A / C / D.
Stage A/C currently consume the single highest-probability mode via
``primary_mode`` -> ``_obstacle_track_xy``.

Pure. No pipeline imports.
"""

from __future__ import annotations

from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple

XY = Tuple[float, float]


class PredictedMode(NamedTuple):
    """One hypothesised future for an agent.

    ``path`` elements are whatever ``_obstacle_track_xy`` already accepts:
    a mapping with ``x``/``y`` (optionally ``t``/``v``), or a sequence whose
    first two entries are ``x, y`` -- so a mode can wrap either a CAV
    ``planned_path`` ``(t, x, y, v)`` tuple stream or a prediction-module
    ``[{"x", "y"}, ...]`` stream unchanged.
    """

    path: Sequence[Any]
    probability: float = 1.0


def _as_mode(item: Any) -> Optional[PredictedMode]:
    # Structural, not isinstance -- pipeline modules are importable under two
    # roots (``pipeline.x`` and ``opencda.planning_module.pipeline.x``) so an
    # ``isinstance(item, PredictedMode)`` check misfires across them.
    if item is None:
        return None
    if isinstance(item, Mapping):
        path = item.get("path")
        if path is None:
            path = item.get("trajectory")
        raw_prob = item.get("probability", 1.0)
    else:
        path = getattr(item, "path", None)
        if path is None:
            path = getattr(item, "trajectory", None)
        raw_prob = getattr(item, "probability", 1.0)
    if path is None:
        return None
    try:
        prob = float(raw_prob)
    except (TypeError, ValueError):
        prob = 1.0
    return PredictedMode(path=path, probability=prob)


def as_modes(raw: Any) -> Tuple[PredictedMode, ...]:
    """Normalise ``snapshot['predicted_modes']`` to a tuple of PredictedMode."""

    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        return ()
    out: List[PredictedMode] = []
    for item in raw if isinstance(raw, Sequence) else ():
        mode = _as_mode(item)
        if mode is not None and mode.path is not None:
            out.append(mode)
    return tuple(out)


def primary_mode(raw: Any) -> Optional[PredictedMode]:
    """Highest-probability mode, or ``None`` when there are no modes."""

    modes = as_modes(raw)
    if not modes:
        return None
    return max(modes, key=lambda m: m.probability)


def mode_xy(path: Sequence[Any]) -> List[XY]:
    """Extract ``(x, y)`` samples from one mode's ``path``.

    Mirrors ``_obstacle_track_xy``'s element handling so mode paths and bare
    ``predicted_trajectory`` lists follow one rule.
    """

    pts: List[XY] = []
    for st in path or ():
        if isinstance(st, Mapping) and "x" in st and "y" in st:
            pts.append((float(st["x"]), float(st["y"])))
        elif (
            isinstance(st, Sequence)
            and not isinstance(st, (str, bytes))
            and len(st) >= 2
        ):
            pts.append((float(st[0]), float(st[1])))
    return pts


def single_mode(path: Sequence[Any], probability: float = 1.0) -> Tuple[PredictedMode]:
    """Wrap one deterministic trajectory as a length-1 mode list."""

    return (PredictedMode(path=path, probability=float(probability)),)
