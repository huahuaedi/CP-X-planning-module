"""Thin ROS runtime boundary over the current CP-X planner source.

No planning algorithm is copied into this package. The installed OpenCDA
planning module is the single behavior/reference/MPC implementation; this
module supplies only the small platform objects required while the planner is
being detached from OpenCDA's VehicleManager.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from opencda.planning_module.opencda_bridge.cpx_mpc_planner import (
    CPXMPCPlannerBridge,
)
from opencda.planning_module.utility.carla_compat import carla

_BRIDGE_MODULE = __import__(
    CPXMPCPlannerBridge.__module__, fromlist=["__file__"]
)
_PLANNING_MODULE_ROOT = Path(_BRIDGE_MODULE.__file__).resolve().parents[1]


class _ExternalWorld:
    def __init__(self) -> None:
        self.timestamp_s = 0.0
        self.debug = SimpleNamespace()

    def get_snapshot(self):
        return SimpleNamespace(
            timestamp=SimpleNamespace(elapsed_seconds=float(self.timestamp_s))
        )

    @staticmethod
    def get_actor(_actor_id):
        return None


class _ExternalVehicle:
    def __init__(self, *, actor_id: int, world: _ExternalWorld) -> None:
        self.id = int(actor_id)
        self._world = world
        self.bounding_box = SimpleNamespace(
            extent=SimpleNamespace(x=2.4, y=1.0, z=0.8)
        )

    def get_world(self):
        return self._world


class _CompatibilityLongitudinalController:
    """Construction-only OpenCDA PID facade."""

    max_throttle = 1.0
    max_brake = 1.0

    def __init__(self) -> None:
        self.current_speed = 0.0

    def lon_run_step(self, target_speed_kmh: float) -> float:
        error_kmh = float(target_speed_kmh) - float(self.current_speed)
        return max(-1.0, min(1.0, 0.08 * error_kmh))


class ExternalVehicleManager:
    """Minimal platform state required to assemble the shared planner core."""

    def __init__(self, *, map_planner: Any, actor_id: int = -1) -> None:
        self.world = _ExternalWorld()
        self.vehicle = _ExternalVehicle(actor_id=actor_id, world=self.world)
        self.controller = _CompatibilityLongitudinalController()
        self.carla_map = map_planner
        self.v2x_manager = None
        self.map_manager = None
        self._opencda_agent_finished = False


def _current_planning_module_root() -> Path:
    """Return the package root that owns the imported planner implementation."""

    return _PLANNING_MODULE_ROOT


class CurrentPlannerRuntime:
    """Run the repository's current planner from synchronized ROS values."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        map_planner: Any,
        actor_id: int = -1,
    ) -> None:
        runtime_config = dict(config or {})
        planning_module_root = _current_planning_module_root()
        # The ROS workspace may contain an old copied MPC tree. Runtime
        # behavior must come from the same package as CPXMPCPlannerBridge.
        runtime_config["mpc_config_path"] = str(
            planning_module_root / "MPC" / "mpc.yaml"
        )
        # ROS receives CP over topics. Never publish the file-backed native
        # OpenCDA CP payload from this process.
        runtime_config["publish_cp_message"] = False
        runtime_config["draw_world_debug"] = False
        runtime_config["draw_world_debug_destination"] = False
        self.vehicle_manager = ExternalVehicleManager(
            map_planner=map_planner,
            actor_id=int(actor_id),
        )
        self.bridge = CPXMPCPlannerBridge(
            vehicle_manager=self.vehicle_manager,
            config=runtime_config,
            map_planner=map_planner,
        )

    def set_destination(
        self,
        *,
        start: Mapping[str, float],
        goal: Mapping[str, float],
    ) -> None:
        self.bridge.set_destination(
            start_location=carla.Location(
                x=float(start["x"]),
                y=float(start["y"]),
                z=float(start.get("z", 0.0)),
            ),
            end_location=carla.Location(
                x=float(goal["x"]),
                y=float(goal["y"]),
                z=float(goal.get("z", 0.0)),
            ),
        )

    def step(
        self,
        *,
        sim_time_s: float,
        ego_pose: Mapping[str, float],
        ego_speed_mps: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        cp_payload: Mapping[str, Any],
        safety_manager: Any = None,
        v2x_manager: Any = None,
    ):
        timestamp_s = float(sim_time_s)
        self.vehicle_manager.world.timestamp_s = timestamp_s
        transform = carla.Transform(
            carla.Location(
                x=float(ego_pose["x"]),
                y=float(ego_pose["y"]),
                z=float(ego_pose.get("z", 0.0)),
            ),
            carla.Rotation(
                yaw=math.degrees(float(ego_pose["heading_rad"]))
            ),
        )
        self.bridge.update_external_information(
            ego_transform=transform,
            ego_speed_mps=float(ego_speed_mps),
            object_snapshots=object_snapshots,
            cp_payload=cp_payload,
            sim_time_s=timestamp_s,
            safety_manager=safety_manager,
            v2x_manager=v2x_manager,
        )
        output = self.bridge.execute_planning_pipeline()
        self.bridge.last_output = output
        self.bridge.last_debug = output.diagnostics_dict()
        return output

    def close(self) -> None:
        self.bridge.destroy()
