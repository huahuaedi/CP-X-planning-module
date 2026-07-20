"""Factory for selecting global-planner backends.

The planning stack supports two route-planning systems:

* ``astar`` / ``legacy`` / ``carla_grp``: CARLA waypoint graph plus legacy A*.
  This is the baseline used for comparisons and does not require AD-map.
* ``custom`` / ``admap`` / ``opendrive``: custom OpenDRIVE planner backed by
  the compiled AD-map runtime.

Both backends expose the same runner-facing methods after construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Dict, Mapping, Tuple

from .carla_lane_graph import build_lane_center_waypoints
from .global_planner import CustomGlobalPlannerAdapter
from .legacy_global_planner import AStarGlobalPlanner


ASTAR_GLOBAL_PLANNER_MODES = {"", "astar", "legacy", "carla_grp"}
CUSTOM_GLOBAL_PLANNER_MODES = {"custom", "custom_admap", "admap", "opendrive"}


@dataclass(frozen=True)
class GlobalPlannerBackendSelection:
    """Result of global-planner backend construction."""

    planner: Any
    road_cfg: Dict[str, object]
    mode: str
    backend_name: str


def normalize_global_planner_mode(raw_mode: object) -> str:
    """Normalize scenario YAML planner mode."""

    return str(raw_mode if raw_mode is not None else "astar").strip().lower()


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

    sample_distance_m = float(planning_cfg.get("waypoint_sample_distance_m", 2.0))
    mode = normalize_global_planner_mode(planning_cfg.get("global_planner_mode", "astar"))

    if mode in CUSTOM_GLOBAL_PLANNER_MODES:
        xodr_path = resolve_xodr_path_fn(scenario_cfg=scenario_cfg, sumo_cfg=sumo_cfg)
        print(
            "[CARLA GLOBAL ROUTE OUTPUT] Using custom OpenDRIVE/AD-map "
            f"global planner (mode={mode})."
        )
        try:
            planner = CustomGlobalPlannerAdapter(
                xodr_path=xodr_path,
                cache_root=os.path.join(project_root, "Global_Planner", "cache"),
                route_sample_distance_m=float(sample_distance_m),
                ad_map_install_root=planning_cfg.get("ad_map_install_root"),
            )
            planner.load()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Custom global planner was requested, but the AD-map runtime "
                "is not installed for this Python environment. Build it with "
                "`PYTHON_BIN=\"$(command -v python)\" "
                "opencda/planning_module/Global_Planner/build_ad_map.sh --clean`, "
                "set `GLOBAL_PLANNER_AD_MAP_INSTALL`, or set "
                "`planning.global_planner_mode: astar` in the scenario YAML."
            ) from exc
        return GlobalPlannerBackendSelection(
            planner=planner,
            road_cfg={"lane_count": 1, "lane_width_m": 3.5},
            mode=mode,
            backend_name="custom_admap",
        )

    if mode in ASTAR_GLOBAL_PLANNER_MODES:
        print(
            "[CARLA GLOBAL ROUTE OUTPUT] Using legacy CARLA/A* global planner "
            f"(mode={mode or 'astar'})."
        )
        lane_center_waypoints, road_cfg = build_lane_center_waypoints(
            map_obj=world_map,
            carla=carla,
            sample_distance_m=float(sample_distance_m),
        )
        planner = AStarGlobalPlanner(
            lane_center_waypoints=lane_center_waypoints,
            world_map=world_map,
            route_sample_distance_m=float(sample_distance_m),
        )
        return GlobalPlannerBackendSelection(
            planner=planner,
            road_cfg=dict(road_cfg),
            mode=mode or "astar",
            backend_name="legacy_astar",
        )

    raise ValueError(
        "Unsupported planning.global_planner_mode "
        f"{mode!r}; expected astar/carla_grp/legacy or custom/admap/opendrive."
    )
