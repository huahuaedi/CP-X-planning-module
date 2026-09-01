"""Factory for the single AD-map global-planner backend."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Mapping, Tuple

from .global_planner import CustomGlobalPlannerAdapter

CUSTOM_GLOBAL_PLANNER_MODES = {"custom", "custom_admap", "admap", "opendrive", "dijkstra", "dij"}


@dataclass(frozen=True)
class GlobalPlannerBackendSelection:
    """Result of global-planner backend construction."""

    planner: Any
    road_cfg: Dict[str, object]
    mode: str
    backend_name: str


def normalize_global_planner_mode(raw_mode: object) -> str:
    """Normalize scenario YAML planner mode."""

    return str(raw_mode if raw_mode is not None else "dij").strip().lower()


def create_global_planner_backend(
    *,
    planning_cfg: Mapping[str, object],
    scenario_cfg: Mapping[str, object],
    sumo_cfg: Mapping[str, object],
    world_map: Any,
    carla: Any,
    project_root: str,
    resolve_xodr_path_fn: Callable[..., str],
) -> GlobalPlannerBackendSelection:
    """Create the configured global planner backend.

    input: planning/scenario/map context
    output: backend selection with planner and road configuration
    """

    mode = normalize_global_planner_mode(planning_cfg.get("global_planner_mode", "dij"))
    if mode not in CUSTOM_GLOBAL_PLANNER_MODES:
        raise ValueError(
            f"Unsupported planning.global_planner_mode {mode!r}; AD-map is required."
        )
    sample_distance_m = float(planning_cfg.get("waypoint_sample_distance_m", 1.0))

    xodr_path = resolve_xodr_path_fn(scenario_cfg=scenario_cfg, sumo_cfg=sumo_cfg)
    print(f"[AD-MAP ROUTE] Using AD-map global planner (mode={mode}).")
    try:
        planner = CustomGlobalPlannerAdapter(
            xodr_path=xodr_path,
            cache_root=os.path.join(project_root, "Global_Planner", "cache"),
            route_sample_distance_m=float(sample_distance_m),
            lane_change_penalty_m=(
                float(planning_cfg["global_planner_lane_change_penalty_m"])
                if planning_cfg.get("global_planner_lane_change_penalty_m") is not None
                else None
            ),
            ad_map_install_root=planning_cfg.get("ad_map_install_root"),
        )
        planner.load()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The AD-map runtime is not installed for this Python environment."
        ) from exc
    return GlobalPlannerBackendSelection(
        planner=planner,
        road_cfg={"lane_count": 1, "lane_width_m": 3.5},
        mode=mode,
        backend_name="custom_admap",
    )
