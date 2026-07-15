"""OpenCDA-style cooperative-perception provider for CP-X scenarios.

This adapter lets the standalone planning-module scenarios consume a CP message
stream with the same contract used by the behavior planner.  It is deliberately
kept independent from the OpenCDA scenario runner: the provider reads CARLA
actors from the current world, packages them as cooperative-perception obstacle
messages, and writes them into ``cp_message.json``.  The existing CP-X planner
then consumes those messages through its normal CP pipeline.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from utility.cp_messages import replace_cp_list
from utility.global_planner import canonical_lane_id_for_waypoint


class OpenCDACPProvider:
    """Publish CARLA actor observations as OpenCDA-style CP obstacle messages."""

    def __init__(
        self,
        *,
        message_path: str,
        schema_version: int = 1,
        communication_range_m: float = 80.0,
        prediction_horizon_s: float = 3.0,
        prediction_dt_s: float = 0.2,
        source: str = "opencda_cp_provider",
    ) -> None:
        self.message_path = str(message_path)
        self.schema_version = int(schema_version)
        self.communication_range_m = max(0.0, float(communication_range_m))
        self.prediction_horizon_s = max(0.0, float(prediction_horizon_s))
        self.prediction_dt_s = max(1.0e-3, float(prediction_dt_s))
        self.source = str(source)
        self.last_publish_summary: dict[str, object] = {}

    def publish(
        self,
        *,
        world: Any,
        map_planner: Any,
        ego_vehicle: Any,
        sim_time_s: float,
    ) -> list[dict]:
        ego_location = ego_vehicle.get_location()
        messages: list[dict] = []
        for actor in world.get_actors().filter("*vehicle*"):
            if int(getattr(actor, "id", -1)) == int(getattr(ego_vehicle, "id", -2)):
                continue
            try:
                actor_location = actor.get_location()
                distance_m = float(actor_location.distance(ego_location))
            except RuntimeError:
                continue
            if self.communication_range_m > 0.0 and distance_m > self.communication_range_m:
                continue
            message = self._vehicle_actor_to_cp_message(
                actor=actor,
                map_planner=map_planner,
                sim_time_s=float(sim_time_s),
                distance_m=float(distance_m),
            )
            if isinstance(message, Mapping):
                messages.append(dict(message))

        replace_cp_list(
            message_path=self.message_path,
            schema_version=self.schema_version,
            list_name="obstacles",
            items=messages,
            timestamp_s=float(sim_time_s),
        )
        self.last_publish_summary = {
            "source": self.source,
            "obstacle_count": int(len(messages)),
            "communication_range_m": float(self.communication_range_m),
            "timestamp_s": float(sim_time_s),
        }
        return messages

    def _vehicle_actor_to_cp_message(
        self,
        *,
        actor: Any,
        map_planner: Any,
        sim_time_s: float,
        distance_m: float,
    ) -> Optional[dict]:
        try:
            transform = actor.get_transform()
            location = transform.location
            rotation = transform.rotation
            velocity = actor.get_velocity()
            bbox = actor.bounding_box
            extent = bbox.extent
        except RuntimeError:
            return None

        speed_mps = math.sqrt(
            float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
        )
        heading_rad = math.radians(float(rotation.yaw))
        lane_id = 0
        road_id = -1
        try:
            waypoint = map_planner.get_waypoint(
                {
                    "x": float(location.x),
                    "y": float(location.y),
                    "z": float(location.z),
                }
            )
            lane_id = int(canonical_lane_id_for_waypoint(waypoint))
            road_id = int(getattr(waypoint, "road_id", -1) or -1)
        except Exception:
            pass

        trajectory = self._constant_velocity_trajectory(
            x_m=float(location.x),
            y_m=float(location.y),
            speed_mps=float(speed_mps),
            heading_rad=float(heading_rad),
        )
        actor_id = str(getattr(actor, "id", ""))
        return {
            "id": f"opencda_cp_vehicle:{actor_id}",
            "type": "vehicle",
            "source": self.source,
            "provider_source": "opencda_style_v2x",
            "timestamp_s": float(sim_time_s),
            "ttl_s": max(0.2, 2.0 * float(self.prediction_dt_s)),
            "confidence": 1.0,
            "distance_m": float(distance_m),
            "state": [
                float(location.x),
                float(location.y),
                float(speed_mps),
                float(heading_rad),
            ],
            "z": float(location.z),
            "shape": {
                "length_m": 2.0 * float(getattr(extent, "x", 2.25)),
                "width_m": 2.0 * float(getattr(extent, "y", 1.0)),
                "height_m": 2.0 * float(getattr(extent, "z", 0.9)),
            },
            "road_id": int(road_id),
            "lane_id": int(lane_id),
            "trajectory": trajectory,
        }

    def _constant_velocity_trajectory(
        self,
        *,
        x_m: float,
        y_m: float,
        speed_mps: float,
        heading_rad: float,
    ) -> list[list[float]]:
        steps = max(1, int(round(self.prediction_horizon_s / self.prediction_dt_s)))
        cos_h = math.cos(float(heading_rad))
        sin_h = math.sin(float(heading_rad))
        trajectory: list[list[float]] = []
        for idx in range(steps + 1):
            t_s = float(idx) * float(self.prediction_dt_s)
            trajectory.append([
                float(x_m + speed_mps * cos_h * t_s),
                float(y_m + speed_mps * sin_h * t_s),
                float(speed_mps),
                float(heading_rad),
            ])
        return trajectory
