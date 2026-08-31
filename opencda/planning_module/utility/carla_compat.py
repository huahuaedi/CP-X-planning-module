"""CARLA type access that also works in a CARLA-free (ROS / Python 3.13) env.

`import carla` succeeds only in the legacy carla307 environment. Everything the
planning module touches on `carla` at runtime is a small set of value types
(`Location`, `Rotation`, `Transform`, `VehicleControl`) plus debug-draw enums
(`Color`, `LaneType`, `AttachmentType`, `TrafficLightState`).

Import `carla` from here instead of directly::

    from opencda.planning_module.utility.carla_compat import carla, CARLA_AVAILABLE

When the real package is importable you get it unchanged. Otherwise you get a
namespace with API-compatible stand-ins so control flow, geometry math and
diagnostics keep working; `CARLA_AVAILABLE` tells callers whether anything that
needs a real simulator (sensor/actor spawning) can run.
"""
from __future__ import annotations

import math
import types


class _Vector3D:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance(self, other: "_Vector3D") -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Location(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class _Rotation:
    __slots__ = ("pitch", "yaw", "roll")

    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)


class _Transform:
    __slots__ = ("location", "rotation")

    def __init__(self, location=None, rotation=None) -> None:
        self.location = location if location is not None else _Vector3D()
        self.rotation = rotation if rotation is not None else _Rotation()


class _VehicleControl:
    __slots__ = (
        "throttle", "steer", "brake", "hand_brake", "reverse",
        "manual_gear_shift", "gear",
    )

    def __init__(
        self,
        throttle: float = 0.0,
        steer: float = 0.0,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False,
        manual_gear_shift: bool = False,
        gear: int = 0,
    ) -> None:
        self.throttle = float(throttle)
        self.steer = float(steer)
        self.brake = float(brake)
        self.hand_brake = bool(hand_brake)
        self.reverse = bool(reverse)
        self.manual_gear_shift = bool(manual_gear_shift)
        self.gear = int(gear)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"VehicleControl(throttle={self.throttle:.3f}, steer={self.steer:.3f}, "
            f"brake={self.brake:.3f})"
        )


class _Color:
    __slots__ = ("r", "g", "b", "a")

    def __init__(self, r: int = 0, g: int = 0, b: int = 0, a: int = 255) -> None:
        self.r, self.g, self.b, self.a = int(r), int(g), int(b), int(a)


def _make_stub() -> types.SimpleNamespace:
    stub = types.SimpleNamespace()
    stub.Location = _Vector3D
    stub.Vector3D = _Vector3D
    stub.Rotation = _Rotation
    stub.Transform = _Transform
    stub.VehicleControl = _VehicleControl
    stub.Color = _Color
    stub.LaneType = types.SimpleNamespace(NONE=1, Driving=2, Any=-1)
    stub.AttachmentType = types.SimpleNamespace(Rigid=0, SpringArm=1)
    stub.TrafficLightState = types.SimpleNamespace(
        Red=0, Yellow=1, Green=2, Off=3, Unknown=4
    )
    stub.__is_carla_stub__ = True
    return stub


try:  # the real package, when this Python env has it
    import carla as _carla  # type: ignore
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _carla = _make_stub()  # type: ignore

carla = _carla
CARLA_AVAILABLE = not bool(getattr(carla, "__is_carla_stub__", False))

__all__ = ["carla", "CARLA_AVAILABLE"]
