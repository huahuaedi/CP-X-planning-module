"""
OpenCDA SUMO co-simulation bridge for the planning-module runner.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Mapping


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANNING_MODULE_ROOT = os.path.dirname(CURRENT_DIR)
OPENCDA_PACKAGE_ROOT = os.path.dirname(PLANNING_MODULE_ROOT)
OPENCDA_PROJECT_ROOT = os.path.dirname(OPENCDA_PACKAGE_ROOT)

def _candidate_project_roots() -> list[str]:
    roots = [
        OPENCDA_PROJECT_ROOT,
        os.path.dirname(OPENCDA_PROJECT_ROOT),
        os.getcwd(),
        os.path.dirname(os.getcwd()),
        "/home/umd-user/Desktop/OpenCDA",
    ]
    env_root = os.environ.get("OPENCDA_ROOT", "").strip()
    if env_root:
        roots.insert(0, env_root)
    return list(dict.fromkeys(os.path.abspath(root) for root in roots if root))


def _has_opencda_package(project_root: str) -> bool:
    return os.path.isfile(os.path.join(project_root, "opencda", "__init__.py"))


for project_root in _candidate_project_roots():
    if _has_opencda_package(project_root) and project_root not in sys.path:
        sys.path.insert(0, project_root)
        break

if "SUMO_HOME" not in os.environ and os.path.isdir("/usr/share/sumo"):
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

SUMO_HOME = os.environ.get("SUMO_HOME", "")
SUMO_TOOLS_DIR = os.path.join(SUMO_HOME, "tools") if SUMO_HOME else ""
if SUMO_TOOLS_DIR and os.path.isdir(SUMO_TOOLS_DIR) and SUMO_TOOLS_DIR not in sys.path:
    sys.path.insert(0, SUMO_TOOLS_DIR)

import carla  # type: ignore
import traci  # type: ignore

from opencda.co_simulation.sumo_integration.bridge_helper import BridgeHelper
from opencda.co_simulation.sumo_integration.constants import INVALID_ACTOR_ID, SPAWN_OFFSET_Z
from opencda.co_simulation.sumo_integration.sumo_simulation import SumoSimulation


class OpenCDASumoBridge:
    """
    Mirror SUMO traffic into CARLA while keeping the ego vehicle under the
    planning module's control.
    """

    def __init__(
        self,
        *,
        client,
        world,
        sumo_cfg: Mapping[str, object],
        sumo_asset_dir: str,
    ) -> None:
        self.client = client
        self.world = world
        self.sumo_cfg = dict(sumo_cfg or {})
        self.sumo_asset_dir = str(sumo_asset_dir)
        self.map_basename = os.path.basename(self.sumo_asset_dir.rstrip(os.sep))
        # Start empty so vehicles that already exist in CARLA at bridge startup
        # (for example the ego vehicle and scenario obstacles) are exported
        # into SUMO on the first co-simulation tick.
        self._active_actors: set[int] = set()
        self.spawned_actors: set[int] = set()
        self.destroyed_actors: set[int] = set()
        self._tls: dict[str, object] = {}
        self._sumo_snapshot_overrides: dict[int, dict[str, object]] = {}
        self._carla_sync_radius_m = max(
            0.0,
            float(self.sumo_cfg.get("carla_sync_radius_m", 120.0)),
        )
        default_despawn_radius_m = (
            float(self._carla_sync_radius_m) + 20.0
            if float(self._carla_sync_radius_m) > 0.0
            else 0.0
        )
        self._carla_despawn_radius_m = max(
            0.0,
            float(self.sumo_cfg.get("carla_despawn_radius_m", default_despawn_radius_m)),
        )
        self._carla_sync_max_actors = max(
            0,
            int(self.sumo_cfg.get("carla_sync_max_actors", 80)),
        )

        for landmark in self.world.get_map().get_all_landmarks_of_type("1000001"):
            if landmark.id == "":
                continue
            traffic_light = self.world.get_traffic_light(landmark)
            if traffic_light is None:
                logging.warning(
                    "[SUMO COSIM] Landmark %s is not linked to a CARLA traffic light.",
                    landmark.id,
                )
                continue
            self._tls[landmark.id] = traffic_light

        sumocfg_path = os.path.join(self.sumo_asset_dir, f"{self.map_basename}.sumocfg")
        if not os.path.isfile(sumocfg_path):
            raise FileNotFoundError(
                f"SUMO config file was not found: {sumocfg_path}"
            )

        connect_existing_server = bool(self.sumo_cfg.get("connect_existing_server", False))
        sumo_host = None
        sumo_port = None
        if connect_existing_server:
            configured_host = str(self.sumo_cfg.get("host", "")).strip()
            configured_port = self.sumo_cfg.get("port", None)
            sumo_host = configured_host or None
            sumo_port = configured_port if configured_port not in ("", None) else None

        self.sumo = SumoSimulation(
            sumocfg_path,
            float(self.sumo_cfg.get("step_length", 0.05)),
            sumo_host,
            sumo_port,
            bool(self.sumo_cfg.get("gui", True)),
            int(self.sumo_cfg.get("client_order", 1)),
        )
        self.sumo.switch_off_traffic_lights()
        self.sumo2carla_ids: dict[str, int] = {}
        self.carla2sumo_ids: dict[int, str] = {}
        BridgeHelper.blueprint_library = self.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

    @property
    def traffic_light_ids(self) -> set[str]:
        return set(self._tls.keys())

    def get_traffic_light_state(self, landmark_id: str):
        traffic_light = self._tls.get(str(landmark_id))
        if traffic_light is None:
            return None
        return traffic_light.state

    def spawn_actor(self, blueprint, transform, role_name: str | None = None) -> int:
        transform = carla.Transform(
            transform.location + carla.Location(0.0, 0.0, SPAWN_OFFSET_Z),
            transform.rotation,
        )
        if role_name and hasattr(blueprint, "has_attribute") and blueprint.has_attribute("role_name"):
            try:
                blueprint.set_attribute("role_name", str(role_name))
            except Exception:
                pass
        batch = [
            carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetSimulatePhysics(carla.command.FutureActor, False)
            )
        ]
        response = self.client.apply_batch_sync(batch, False)[0]
        if response.error:
            logging.error("[SUMO COSIM] Failed to spawn mirrored CARLA actor: %s", response.error)
            return INVALID_ACTOR_ID
        return int(response.actor_id)

    def synchronize_vehicle(self, vehicle_id: int, transform) -> bool:
        actor = self.world.get_actor(int(vehicle_id))
        if actor is None:
            return False
        actor.set_transform(transform)
        return True

    def destroy_actor(self, actor_id: int) -> bool:
        self._sumo_snapshot_overrides.pop(int(actor_id), None)
        actor = self.world.get_actor(int(actor_id))
        if actor is None:
            return False
        return bool(actor.destroy())

    def _record_sumo_actor_state(self, *, carla_actor_id: int, sumo_actor_id: str, transform) -> None:
        previous_state = self._sumo_snapshot_overrides.get(int(carla_actor_id), None)
        step_length_s = max(1.0e-3, float(self.sumo_cfg.get("step_length", 0.05)))
        speed_mps = 0.0
        if isinstance(previous_state, Mapping):
            prev_x = float(previous_state.get("x", transform.location.x))
            prev_y = float(previous_state.get("y", transform.location.y))
            distance_m = ((float(transform.location.x) - prev_x) ** 2 + (float(transform.location.y) - prev_y) ** 2) ** 0.5
            speed_mps = float(distance_m / step_length_s)
        self._sumo_snapshot_overrides[int(carla_actor_id)] = {
            "vehicle_id": f"sumo_{str(sumo_actor_id)}",
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "psi": float(transform.rotation.yaw) * 3.141592653589793 / 180.0,
            "v": float(speed_mps),
        }

    def get_actor_snapshot_override(self, actor_id: int):
        return dict(self._sumo_snapshot_overrides.get(int(actor_id), {})) or None

    def _get_ego_location(self):
        for actor in self.world.get_actors().filter("vehicle.*"):
            role_name = str(getattr(actor, "attributes", {}).get("role_name", "")).strip().lower()
            if role_name == "ego":
                return actor.get_transform().location
        return None

    @staticmethod
    def _distance_sq_to_ego(*, x_m: float, y_m: float, ego_location) -> float:
        dx_m = float(x_m) - float(ego_location.x)
        dy_m = float(y_m) - float(ego_location.y)
        return float(dx_m) * float(dx_m) + float(dy_m) * float(dy_m)

    def _select_sumo_actor_ids_for_carla_sync(self, *, ego_location, excluded_sumo_ids: set[str]) -> list[str]:
        active_sumo_actor_ids = [
            str(actor_id)
            for actor_id in traci.vehicle.getIDList()
            if str(actor_id) not in excluded_sumo_ids
        ]
        if ego_location is None or float(self._carla_sync_radius_m) <= 0.0:
            if int(self._carla_sync_max_actors) > 0:
                return active_sumo_actor_ids[: int(self._carla_sync_max_actors)]
            return active_sumo_actor_ids

        sync_radius_sq = float(self._carla_sync_radius_m) * float(self._carla_sync_radius_m)
        ranked_actor_ids: list[tuple[float, str]] = []
        for sumo_actor_id in active_sumo_actor_ids:
            try:
                sumo_x_m, sumo_y_m = traci.vehicle.getPosition(str(sumo_actor_id))
            except Exception:
                continue
            distance_sq = self._distance_sq_to_ego(
                x_m=float(sumo_x_m),
                y_m=float(sumo_y_m),
                ego_location=ego_location,
            )
            if float(distance_sq) <= float(sync_radius_sq):
                ranked_actor_ids.append((float(distance_sq), str(sumo_actor_id)))

        ranked_actor_ids.sort(key=lambda item: float(item[0]))
        if int(self._carla_sync_max_actors) > 0:
            ranked_actor_ids = ranked_actor_ids[: int(self._carla_sync_max_actors)]
        return [actor_id for _distance_sq, actor_id in ranked_actor_ids]

    def tick(self) -> None:
        # SUMO -> CARLA
        self.sumo.tick()
        ego_location = self._get_ego_location()
        excluded_sumo_ids = {str(actor_id) for actor_id in self.carla2sumo_ids.values()}
        desired_sumo_actor_ids = set(
            self._select_sumo_actor_ids_for_carla_sync(
                ego_location=ego_location,
                excluded_sumo_ids=excluded_sumo_ids,
            )
        )

        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        for sumo_actor_id, carla_actor_id in list(self.sumo2carla_ids.items()):
            if sumo_actor_id not in desired_sumo_actor_ids:
                if (
                    ego_location is not None
                    and float(self._carla_despawn_radius_m) > 0.0
                ):
                    carla_actor = self.world.get_actor(int(carla_actor_id))
                    if carla_actor is not None:
                        carla_location = carla_actor.get_transform().location
                        distance_sq = self._distance_sq_to_ego(
                            x_m=float(carla_location.x),
                            y_m=float(carla_location.y),
                            ego_location=ego_location,
                        )
                        if float(distance_sq) > float(self._carla_despawn_radius_m) * float(self._carla_despawn_radius_m):
                            self.destroy_actor(int(carla_actor_id))
                            self.sumo2carla_ids.pop(sumo_actor_id, None)
                            try:
                                self.sumo.unsubscribe(str(sumo_actor_id))
                            except Exception:
                                pass
                            continue
            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            if sumo_actor is None:
                continue
            carla_transform = BridgeHelper.get_carla_transform(
                sumo_actor.transform,
                sumo_actor.extent,
            )
            self.synchronize_vehicle(carla_actor_id, carla_transform)
            self._record_sumo_actor_state(carla_actor_id=int(carla_actor_id), sumo_actor_id=str(sumo_actor_id), transform=carla_transform)

        for sumo_actor_id in sorted(desired_sumo_actor_ids):
            if sumo_actor_id in self.sumo2carla_ids:
                continue
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            if sumo_actor is None:
                continue
            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, False)
            if carla_blueprint is None:
                try:
                    self.sumo.unsubscribe(sumo_actor_id)
                except Exception:
                    pass
                continue
            carla_transform = BridgeHelper.get_carla_transform(
                sumo_actor.transform,
                sumo_actor.extent,
            )
            carla_actor_id = self.spawn_actor(carla_blueprint, carla_transform, role_name=f"sumo_{sumo_actor_id}")
            if carla_actor_id != INVALID_ACTOR_ID:
                self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
                self._record_sumo_actor_state(carla_actor_id=int(carla_actor_id), sumo_actor_id=str(sumo_actor_id), transform=carla_transform)

        # CARLA -> SUMO
        self.world.tick()
        current_actors = {
            int(actor.id)
            for actor in self.world.get_actors().filter("vehicle.*")
        }
        self.spawned_actors = current_actors.difference(self._active_actors)
        self.destroyed_actors = self._active_actors.difference(current_actors)
        self._active_actors = current_actors

        carla_spawned_actors = self.spawned_actors - set(self.sumo2carla_ids.values())
        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.world.get_actor(int(carla_actor_id))
            if carla_actor is None:
                continue
            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get("color", None)
            if type_id is None:
                continue
            sumo_actor_id = self.sumo.spawn_actor(type_id, color)
            if sumo_actor_id == INVALID_ACTOR_ID:
                continue
            self.carla2sumo_ids[int(carla_actor_id)] = sumo_actor_id
            self.sumo.subscribe(sumo_actor_id)

        for carla_actor_id in self.destroyed_actors:
            if int(carla_actor_id) in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(int(carla_actor_id)))

        for carla_actor_id, sumo_actor_id in list(self.carla2sumo_ids.items()):
            carla_actor = self.world.get_actor(int(carla_actor_id))
            if carla_actor is None:
                continue
            sumo_transform = BridgeHelper.get_sumo_transform(
                carla_actor.get_transform(),
                carla_actor.bounding_box.extent,
            )
            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, None)

        common_landmarks = self.sumo.traffic_light_ids & self.traffic_light_ids
        for landmark_id in common_landmarks:
            carla_tl_state = self.get_traffic_light_state(landmark_id)
            sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)
            self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

    def close(self) -> None:
        for carla_actor_id in list(self.sumo2carla_ids.values()):
            self.destroy_actor(carla_actor_id)
        self.sumo2carla_ids.clear()
        self._sumo_snapshot_overrides.clear()

        for sumo_actor_id in list(self.carla2sumo_ids.values()):
            self.sumo.destroy_actor(sumo_actor_id)
        self.carla2sumo_ids.clear()

        for actor in self.world.get_actors():
            if actor.type_id == "traffic.traffic_light":
                actor.freeze(False)

        self.sumo.close()
