"""
OpenCDA SUMO-backed runner wrapper.
"""

from __future__ import annotations

from copy import deepcopy

from planning_runner import run_loaded_world as _run_loaded_world


def _strip_static_obstacle_spawner(scenario_cfg):
    normalized_cfg = deepcopy(dict(scenario_cfg or {}))
    obstacle_cfg = normalized_cfg.get("obstacles", None)
    if isinstance(obstacle_cfg, dict) and obstacle_cfg:
        normalized_cfg["obstacles"] = {
            key: value
            for key, value in obstacle_cfg.items()
            if str(key) != "spawner_module"
        }
    return normalized_cfg


def run_loaded_world(client, world, scenario_cfg, carla) -> int:
    """
    Delegate to the existing planning-module runner.

    The underlying runner understands the optional `sumo` block and will switch
    from plain `world.tick()` to OpenCDA SUMO co-simulation when it is enabled
    in the scenario YAML.

    OpenCDA scenarios intentionally disable the planning module's static
    obstacle spawners so these scenarios contain only the ego vehicle and SUMO
    traffic.
    """

    return int(
        _run_loaded_world(
            client=client,
            world=world,
            scenario_cfg=_strip_static_obstacle_spawner(scenario_cfg),
            carla=carla,
        )
    )
