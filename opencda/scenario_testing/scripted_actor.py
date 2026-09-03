# -*- coding: utf-8 -*-
"""Deterministic, ego-blind scripted background actors.

Used by the prediction-knowledge ablation: for a clean A/B the only thing
that may differ between the ``blind`` and ``oracle`` runs is what the ego
planner is told about other vehicles' futures -- so the other vehicle's
motion must be byte-identical across runs.  A CARLA Traffic-Manager car
reacts to ego and jitters its own speed; this driver does neither.

Each actor is advanced kinematically along a fixed world-XY polyline at a
fixed speed with ``set_transform`` (no physics, no autopilot).  CARLA
perception still reports it as a moving vehicle.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence


class ScriptedActor:
    """One kinematic waypoint-follower."""

    def __init__(
        self,
        vehicle: Any,
        *,
        path_xy: Sequence[Sequence[float]],
        speed_mps: float,
        z_m: float = 0.3,
        start_tick: int = 0,
        loop: bool = False,
    ):
        self.vehicle = vehicle
        self._path = [(float(p[0]), float(p[1])) for p in path_xy]
        self._speed = max(0.0, float(speed_mps))
        self._z = float(z_m)
        self._start_tick = int(start_tick)
        self._loop = bool(loop)
        self._seg = 0
        self._s_in_seg = 0.0
        self._done = len(self._path) < 2
        self._tick = 0
        if self._path:
            self._teleport(self._path[0], self._heading(0))

    # ------------------------------------------------------------------ #
    def _heading(self, seg: int) -> float:
        a = self._path[min(seg, len(self._path) - 2)]
        b = self._path[min(seg + 1, len(self._path) - 1)]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _seg_len(self, seg: int) -> float:
        a, b = self._path[seg], self._path[seg + 1]
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _teleport(self, xy, heading_rad: float) -> None:
        import carla

        self.vehicle.set_transform(
            carla.Transform(
                carla.Location(x=float(xy[0]), y=float(xy[1]), z=self._z),
                carla.Rotation(yaw=math.degrees(float(heading_rad))),
            )
        )
        try:  # keep reported velocity consistent with the scripted motion
            self.vehicle.set_target_velocity(
                carla.Vector3D(
                    x=float(self._speed * math.cos(heading_rad)),
                    y=float(self._speed * math.sin(heading_rad)),
                    z=0.0,
                )
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def step(self, dt_s: float) -> None:
        self._tick += 1
        if self._done or self._tick < self._start_tick:
            return
        remaining = self._speed * max(0.0, float(dt_s))
        while remaining > 1.0e-6 and not self._done:
            seg_len = self._seg_len(self._seg)
            room = seg_len - self._s_in_seg
            if remaining < room:
                self._s_in_seg += remaining
                remaining = 0.0
            else:
                remaining -= room
                self._seg += 1
                self._s_in_seg = 0.0
                if self._seg >= len(self._path) - 1:
                    if self._loop:
                        self._seg = 0
                    else:
                        self._done = True
                        self._teleport(self._path[-1], self._heading(len(self._path) - 2))
                        return
        a = self._path[self._seg]
        b = self._path[self._seg + 1]
        seg_len = max(1.0e-6, self._seg_len(self._seg))
        frac = self._s_in_seg / seg_len
        xy = (a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1]))
        self._teleport(xy, self._heading(self._seg))

    @property
    def finished(self) -> bool:
        return self._done


def spawn_scripted_actors(world: Any, actor_cfgs: Sequence[dict]) -> List[ScriptedActor]:
    """Spawn one :class:`ScriptedActor` per config entry.

    Config entry keys: ``blueprint`` (default ``vehicle.tesla.model3``),
    ``path`` (list of ``[x, y]``), ``speed_mps``, ``z`` (default 0.3),
    ``start_tick`` (default 0), ``loop`` (default False).
    """

    import carla

    blueprint_library = world.get_blueprint_library()
    out: List[ScriptedActor] = []
    for i, cfg in enumerate(list(actor_cfgs or [])):
        cfg = dict(cfg or {})
        path = list(cfg.get("path", []) or [])
        if len(path) < 2:
            print("[scripted_actor] entry %d skipped: path needs >= 2 points" % i)
            continue
        bp_name = str(cfg.get("blueprint", "vehicle.tesla.model3"))
        try:
            bp = blueprint_library.find(bp_name)
        except Exception:
            bp = blueprint_library.filter("vehicle.*")[0]
        bp.set_attribute("role_name", "scripted_%d" % i)
        z = float(cfg.get("z", 0.3))
        heading0 = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
        spawn_tf = carla.Transform(
            carla.Location(x=float(path[0][0]), y=float(path[0][1]), z=z),
            carla.Rotation(yaw=math.degrees(heading0)),
        )
        vehicle = world.try_spawn_actor(bp, spawn_tf)
        if vehicle is None:
            print("[scripted_actor] entry %d spawn failed at %s" % (i, path[0]))
            continue
        try:
            vehicle.set_simulate_physics(False)
        except Exception:
            pass
        out.append(
            ScriptedActor(
                vehicle,
                path_xy=path,
                speed_mps=float(cfg.get("speed_mps", 6.0)),
                z_m=z,
                start_tick=int(cfg.get("start_tick", 0)),
                loop=bool(cfg.get("loop", False)),
            )
        )
        print(
            "[scripted_actor] spawned %s id=%s speed=%.1f start_tick=%d points=%d"
            % (bp_name, vehicle.id, float(cfg.get("speed_mps", 6.0)),
               int(cfg.get("start_tick", 0)), len(path))
        )
    return out
