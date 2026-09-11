# -*- coding: utf-8 -*-
"""Deterministic, ego-blind scripted background actors.

Used by the prediction-knowledge ablation: for a clean A/B the only thing
that may differ between the ``blind`` and ``oracle`` runs is what the ego
planner is told about other vehicles' futures -- so the other vehicle's
motion must be byte-identical across runs.  A CARLA Traffic-Manager car
reacts to ego and jitters its own speed; this driver does neither.

Each actor is advanced deterministically along a fixed world-XY polyline at a
fixed speed with ``set_transform`` (no autopilot). Physics remains enabled so
CARLA exposes the commanded velocity to perception; the next scripted pose
still owns the trajectory and removes accumulated physical drift.
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
        acceleration_mps2: Any = None,
        z_m: float = 0.3,
        start_tick: int = 0,
        loop: bool = False,
        trigger: Any = None,
    ):
        self.vehicle = vehicle
        self._path = [(float(p[0]), float(p[1])) for p in path_xy]
        self._speed = max(0.0, float(speed_mps))
        self._acceleration = (
            None
            if acceleration_mps2 is None
            else max(1.0e-6, float(acceleration_mps2))
        )
        self._current_speed = 0.0
        self._z = float(z_m)
        self._start_tick = int(start_tick)
        self._loop = bool(loop)
        self._seg = 0
        self._s_in_seg = 0.0
        self._done = len(self._path) < 2
        self._tick = 0
        # Optional ego-position trigger: hold at path[0] until the ego crosses
        # a line, then start the schedule. Removes the ego's variable
        # spawn-settle + acceleration time from arrival-time alignment.
        #   trigger: {axis: "x"|"y", cross: <value>, from: "below"|"above",
        #             hold_ticks: <extra ticks after arming, default 0>}
        cfg = dict(trigger or {})
        self._trigger_axis = str(cfg.get("axis", "")).strip().lower()
        self._trigger_cross = (
            None if cfg.get("cross") is None else float(cfg["cross"])
        )
        self._trigger_from = str(cfg.get("from", "below")).strip().lower()
        self._trigger_hold_ticks = int(cfg.get("hold_ticks", 0))
        self._armed = self._trigger_axis not in ("x", "y") or self._trigger_cross is None
        self._arm_tick = 0
        if self._path:
            self._teleport(self._path[0], self._heading(0), speed_mps=0.0)

    # ------------------------------------------------------------------ #
    def _heading(self, seg: int) -> float:
        a = self._path[min(seg, len(self._path) - 2)]
        b = self._path[min(seg + 1, len(self._path) - 1)]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _seg_len(self, seg: int) -> float:
        a, b = self._path[seg], self._path[seg + 1]
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _teleport(self, xy, heading_rad: float, *, speed_mps=None) -> None:
        import carla

        self.vehicle.set_transform(
            carla.Transform(
                carla.Location(x=float(xy[0]), y=float(xy[1]), z=self._z),
                carla.Rotation(yaw=math.degrees(float(heading_rad))),
            )
        )
        try:  # keep reported velocity consistent with the scripted motion
            reported_speed = (
                self._current_speed if speed_mps is None else float(speed_mps)
            )
            self.vehicle.set_target_velocity(
                carla.Vector3D(
                    x=float(reported_speed * math.cos(heading_rad)),
                    y=float(reported_speed * math.sin(heading_rad)),
                    z=0.0,
                )
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _check_trigger(self, ego_xy) -> None:
        if self._armed or ego_xy is None:
            return
        value = float(ego_xy[0] if self._trigger_axis == "x" else ego_xy[1])
        crossed = (
            value >= self._trigger_cross if self._trigger_from == "below"
            else value <= self._trigger_cross
        )
        if crossed:
            self._armed = True
            self._arm_tick = self._tick

    def step(self, dt_s: float, ego_xy: Any = None) -> None:
        self._tick += 1
        self._check_trigger(ego_xy)
        if self._done or not self._armed:
            self._current_speed = 0.0
            if self._path and not self._done:
                self._teleport(self._path[0], self._heading(0), speed_mps=0.0)
            return
        if self._tick - self._arm_tick < max(
            self._start_tick if self._trigger_axis not in ("x", "y") else 0,
            self._trigger_hold_ticks,
        ):
            self._current_speed = 0.0
            return
        dt = max(0.0, float(dt_s))
        self._current_speed = (
            self._speed
            if self._acceleration is None
            else min(self._speed, self._current_speed + self._acceleration * dt)
        )
        remaining = self._current_speed * dt
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
                        self._teleport(
                            self._path[-1], self._heading(len(self._path) - 2),
                            speed_mps=0.0,
                        )
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

    @property
    def prediction_active(self) -> bool:
        """Whether this actor's configured motion has actually started.

        Scenario prediction fixtures use this as a validity signal.  A held
        actor remains observable, but must not advertise a future maneuver
        before its exogenous trigger has fired.
        """

        delay_ticks = max(
            self._start_tick if self._trigger_axis not in ("x", "y") else 0,
            self._trigger_hold_ticks,
        )
        return bool(
            not self._done
            and self._armed
            and self._tick - self._arm_tick >= delay_ticks
        )


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
        # ``set_target_velocity`` is not reflected by CARLA's get_velocity()
        # for a physics-disabled actor. Keep physics enabled so every planner
        # layer observes the same scripted kinematics; set_transform() on the
        # next tick remains the authoritative motion source.
        try:
            vehicle.set_simulate_physics(True)
        except Exception:
            pass
        out.append(
            ScriptedActor(
                vehicle,
                path_xy=path,
                speed_mps=float(cfg.get("speed_mps", 6.0)),
                acceleration_mps2=cfg.get("acceleration_mps2"),
                z_m=z,
                start_tick=int(cfg.get("start_tick", 0)),
                loop=bool(cfg.get("loop", False)),
                trigger=dict(cfg.get("trigger", {}) or {}),
            )
        )
        trig = dict(cfg.get("trigger", {}) or {})
        print(
            "[scripted_actor] spawned %s id=%s speed=%.1f start_tick=%d points=%d trigger=%s"
            % (bp_name, vehicle.id, float(cfg.get("speed_mps", 6.0)),
               int(cfg.get("start_tick", 0)), len(path), trig or "none")
        )
    return out
