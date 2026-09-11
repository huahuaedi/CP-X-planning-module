"""Bridge-side helper: dump one live tick as a ``FrameCapture`` JSON.

Kept out of ``cpx_mpc_planner`` so the planning hot path carries only a
guarded 2-line call. Off unless ``frame_capture`` is configured:

    frame_capture:
      out_dir: /abs/or/relative/dir        # required to arm
      sim_time_window_s: [34.0, 40.0]       # inclusive; omit = whole run
      min_row_count: 1                      # only dump ticks with corridor
                                             # rows active (0 = every tick)
      max_frames: 40                        # safety cap (default 60)

Each armed tick writes ``<out_dir>/frame_t<sim_time>.json`` which
``tools/frame_replay.py`` consumes directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class FrameCaptureConfig:
    def __init__(self, raw: Optional[Mapping[str, Any]]):
        raw = dict(raw or {})
        self.out_dir: Optional[str] = raw.get("out_dir")
        window = raw.get("sim_time_window_s")
        self.t_lo = float(window[0]) if window else float("-inf")
        self.t_hi = float(window[1]) if window else float("inf")
        self.max_frames = int(raw.get("max_frames", 60))
        self.min_row_count = int(raw.get("min_row_count", 0))

    @property
    def armed(self) -> bool:
        return bool(self.out_dir)

    def wants(self, sim_time_s: float) -> bool:
        return self.armed and self.t_lo <= float(sim_time_s) <= self.t_hi


def _rows_to_dicts(rows: Sequence[Any]) -> list:
    out = []
    for r in rows or ():
        if isinstance(r, Mapping):
            out.append(dict(r))
            continue
        out.append(
            {
                "stage": int(getattr(r, "stage", 0)),
                "a_x": float(getattr(r, "a_x", 0.0)),
                "a_y": float(getattr(r, "a_y", 0.0)),
                "lower": float(getattr(r, "lower", float("-inf"))),
                "upper": float(getattr(r, "upper", float("inf"))),
                "slack_group": str(getattr(r, "slack_group", "")),
                "tag": str(getattr(r, "tag", "")),
            }
        )
    return out


def _samples_to_dicts(samples: Sequence[Any]) -> list:
    out = []
    for s in samples or ():
        if isinstance(s, Mapping):
            out.append(
                {
                    k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in s.items()
                }
            )
        else:
            out.append({"x_ref_m": float(s[0]), "y_ref_m": float(s[1])})
    return out


def dump_execute_mpc_frame(
    config: FrameCaptureConfig,
    *,
    sim_time_s: float,
    current_state: Sequence[float],
    ego_origin_xy: Sequence[float],
    ego_yaw_rad: float,
    ego_speed_mps: float,
    current_acceleration_mps2: float,
    current_steering_rad: float,
    destination_state: Sequence[float],
    target_speed_mps: float,
    stop_goal_active: bool,
    behavior_maneuver: str,
    behavior_phase: str,
    pre_publication_reference: Sequence[Any],
    published_reference: Sequence[Any],
    mpc_object_snapshots: Sequence[Any],
    mpc_rows: Sequence[Any],
    cav_diagnostics: Optional[Mapping[str, Any]] = None,
    corridor: Any = None,
    prev_u_solution: Any = None,
    mpc_config_path: Optional[str] = None,
) -> Optional[Path]:
    """Serialize this tick if the window is armed and the cap is not hit."""

    if not config.wants(sim_time_s):
        return None
    if len(mpc_rows or ()) < config.min_row_count:
        return None

    # Local import: keeps this module importable without the pipeline on
    # the path (e.g. for unit tests of the config gate alone).
    from tools.frame_replay import FrameCapture

    out_dir = Path(config.out_dir)  # type: ignore[arg-type]
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_t*.json"))
    if len(existing) >= config.max_frames:
        return None

    s_lo, s_hi, binding = [], [], []
    if corridor is not None:
        s_lo = [float(v) for v in getattr(corridor, "s_lo", []) or []]
        s_hi = [float(v) for v in getattr(corridor, "s_hi", []) or []]
        binding = [str(v) for v in getattr(corridor, "binding", []) or []]

    prev_u = None
    if prev_u_solution is not None:
        try:
            prev_u = [[float(c) for c in row] for row in list(prev_u_solution)]
        except TypeError:
            prev_u = None

    capture = FrameCapture(
        tick=-1,
        sim_time_s=float(sim_time_s),
        current_state=[float(v) for v in current_state],
        ego_origin_xy=(float(ego_origin_xy[0]), float(ego_origin_xy[1])),
        ego_yaw_rad=float(ego_yaw_rad),
        ego_speed_mps=float(ego_speed_mps),
        current_acceleration_mps2=float(current_acceleration_mps2),
        current_steering_rad=float(current_steering_rad),
        destination_state=[float(v) for v in destination_state],
        target_speed_mps=float(target_speed_mps),
        stop_goal_active=bool(stop_goal_active),
        behavior_maneuver=str(behavior_maneuver),
        behavior_phase=str(behavior_phase),
        pre_publication_reference=_samples_to_dicts(pre_publication_reference),
        published_reference=_samples_to_dicts(published_reference),
        mpc_object_snapshots=[dict(o) for o in (mpc_object_snapshots or ())],
        corridor_s_lo=s_lo,
        corridor_s_hi=s_hi,
        corridor_binding=binding,
        mpc_rows=_rows_to_dicts(mpc_rows),
        prev_u_solution=prev_u,
        mpc_config_path=mpc_config_path,
    )
    path = out_dir / f"frame_t{float(sim_time_s):09.3f}.json"
    capture.to_json(path)
    if cav_diagnostics:
        (out_dir / f"frame_t{float(sim_time_s):09.3f}.cav_diag.json").write_text(
            __import__("json").dumps(dict(cav_diagnostics), indent=2, default=str)
        )
    return path
