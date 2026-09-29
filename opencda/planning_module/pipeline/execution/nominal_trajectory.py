"""Typed nominal-trajectory state and its single lifecycle owner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NominalTarget:
    x_m: float
    y_m: float
    speed_mps: float
    heading_rad: float
    lane_id: int = 0
    mode_value: float = 0.0
    road_id: Optional[int] = None
    entered_intersection: bool = False

    @classmethod
    def from_legacy(cls, values: Sequence[float]) -> "NominalTarget":
        row = list(values or [])
        if len(row) < 4:
            raise ValueError("nominal target requires x, y, speed, and heading")
        return cls(
            x_m=float(row[0]),
            y_m=float(row[1]),
            speed_mps=float(row[2]),
            heading_rad=float(row[3]),
            lane_id=int(row[4]) if len(row) >= 5 else 0,
            mode_value=float(row[5]) if len(row) >= 6 else 0.0,
            road_id=int(row[6]) if len(row) >= 7 else None,
            entered_intersection=(bool(float(row[7]) > 0.5) if len(row) >= 8 else False),
        )

    def as_legacy(self) -> list[float]:
        row = [
            float(self.x_m),
            float(self.y_m),
            float(self.speed_mps),
            float(self.heading_rad),
            int(self.lane_id),
            float(self.mode_value),
        ]
        if self.road_id is not None or self.entered_intersection:
            row.append(int(self.road_id or 0))
        if self.entered_intersection:
            row.append(1.0)
        return row


@dataclass(frozen=True)
class NominalTrajectory:
    target: Optional[NominalTarget] = None
    samples: Tuple[Mapping[str, object], ...] = ()
    reference_freeze_count: int = 0
    revision: int = 0
    source: str = ""

    def target_state(self) -> Optional[list[float]]:
        return None if self.target is None else self.target.as_legacy()

    def mutable_samples(self) -> list[dict[str, object]]:
        return [dict(sample) for sample in self.samples]


class NominalTrajectoryGenerator:
    """Own the accepted nominal target/reference across planning ticks.

    Existing generators may still emit the legacy numeric target at their
    boundary while migration is in progress.  Conversion happens here once;
    downstream state is typed and immutable.
    """

    def __init__(self) -> None:
        self._current = NominalTrajectory()

    @property
    def current(self) -> NominalTrajectory:
        return self._current

    def reset(self, *, source: str = "reset") -> NominalTrajectory:
        self._current = NominalTrajectory(
            revision=int(self._current.revision) + 1,
            source=str(source),
        )
        return self._current

    def update(
        self,
        *,
        target_state: Optional[Sequence[float]] = None,
        samples: Optional[Sequence[Mapping[str, object]]] = None,
        reference_freeze_count: Optional[int] = None,
        source: str,
    ) -> NominalTrajectory:
        current = self._current
        target = current.target
        if target_state is not None:
            target = NominalTarget.from_legacy(target_state)
        immutable_samples = current.samples
        if samples is not None:
            immutable_samples = tuple(
                MappingProxyType(dict(sample)) for sample in list(samples or [])
            )
        self._current = replace(
            current,
            target=target,
            samples=immutable_samples,
            reference_freeze_count=(
                int(current.reference_freeze_count)
                if reference_freeze_count is None
                else max(0, int(reference_freeze_count))
            ),
            revision=int(current.revision) + 1,
            source=str(source),
        )
        return self._current
