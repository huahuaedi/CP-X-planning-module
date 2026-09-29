"""Road-boundary measurement for the ego footprint.

Measures how far the ego's oriented footprint sits from the lane-corridor
strip (or, when enabled, the CARLA drivable-lane union), reports whether that
is a breach, and keeps the run's sample/breach counters.  The vehicle and the
reference generator are reached through provider callbacks and are only
touched inside the guarded measurement, so a missing collaborator degrades to
an invalid sample instead of raising.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


class RoadBoundaryMonitor:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        vehicle_provider: Callable[[], Any],
        generator_provider: Callable[[], Any],
    ) -> None:
        self._config = config
        self._vehicle_provider = vehicle_provider
        self._generator_provider = generator_provider
        self.sample_count = 0
        self.breach_count = 0

    def measure(
        self,
        ego_location: Any,
        *,
        record_sample: bool = True,
        ego_yaw_rad: float | None = None,
        reference_samples: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, object]:
        """Measure ego with the same route-corridor footprint contract."""

        result: dict[str, object] = {
            "road_boundary_sample_valid": False,
            "road_boundary_lateral_offset_m": "",
            "road_boundary_lane_width_m": "",
            "road_boundary_ego_half_width_m": "",
            "road_boundary_clearance_m": "",
            "road_boundary_breach": "",
            "road_boundary_heading_error_rad": "",
            "road_boundary_projection_segment_index": "",
            "road_boundary_projection_segment_ratio": "",
            "road_boundary_projection_raw_heading_rad": "",
            "road_boundary_projection_conditioned_heading_rad": "",
            "road_boundary_projection_continuity_limited": "",
            "road_boundary_projection_reason": "",
            "road_boundary_geometry_source": "",
            "road_boundary_drivable_inside": "",
        }
        try:
            bounding_box = getattr(self._vehicle_provider(), "bounding_box", None)
            extent = getattr(bounding_box, "extent", None)
            ego_half_width_m = float(
                getattr(
                    extent,
                    "y",
                    self._config.get("metrics_ego_half_width_m", 1.0),
                )
            )
            ego_half_length_m = float(
                getattr(
                    extent,
                    "x",
                    self._config.get("reference_vehicle_half_length_m", 2.4),
                )
            )
            if ego_yaw_rad is None:
                ego_yaw_rad = math.radians(
                    float(
                        self._vehicle_provider().get_transform().rotation.yaw
                    )
                )
            projection = None
            if len(list(reference_samples or [])) >= 2:
                projection = self._generator_provider().project_reference_corridor(
                    reference_samples=reference_samples,
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    ego_half_width_m=float(ego_half_width_m),
                    ego_half_length_m=float(ego_half_length_m),
                    safety_margin_m=float(
                        self._config.get(
                            "reference_contract_turn_boundary_margin_m",
                            0.15,
                        )
                    ),
                    max_heading_step_rad=float(
                        self._config.get(
                            "road_boundary_projection_max_heading_step_rad",
                            0.04,
                        )
                    ),
                    continuity_reset_distance_m=float(
                        self._config.get(
                            "road_boundary_projection_reset_distance_m",
                            2.5,
                        )
                    ),
                    max_position_step_m=float(
                        self._config.get(
                            "road_boundary_projection_max_position_step_m",
                            0.5,
                        )
                    ),
                )
                occupancy = projection.occupancy
            else:
                occupancy = self._generator_provider().lane_corridor_occupancy(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    ego_half_width_m=float(ego_half_width_m),
                    ego_half_length_m=float(ego_half_length_m),
                    safety_margin_m=float(
                        self._config.get(
                            "reference_contract_turn_boundary_margin_m",
                            0.15,
                        )
                    ),
                )
            if not bool(occupancy.valid):
                return result
            lateral_offset_m = float(occupancy.lateral_offset_m)
            lane_width_m = float(occupancy.lane_width_m)
            clearance_m = float(occupancy.footprint_clearance_m)
            geometry_source = "route_tangent_strip"
            drivable_inside: object = ""
            if bool(
                self._config.get(
                    "road_boundary_carla_drivable_footprint_enabled",
                    True,
                )
            ):
                drivable_occupancy = (
                    self._generator_provider().drivable_footprint_occupancy(
                        x_m=float(ego_location.x),
                        y_m=float(ego_location.y),
                        z_m=float(getattr(ego_location, "z", 0.0)),
                        heading_rad=float(ego_yaw_rad),
                        ego_half_width_m=float(ego_half_width_m),
                        ego_half_length_m=float(ego_half_length_m),
                        safety_margin_m=float(
                            self._config.get(
                                "reference_contract_turn_boundary_margin_m",
                                0.15,
                            )
                        ),
                    )
                )
                if bool(drivable_occupancy.valid):
                    clearance_m = float(
                        drivable_occupancy.min_clearance_m
                    )
                    geometry_source = str(
                        drivable_occupancy.reason
                    )
                    drivable_inside = bool(
                        drivable_occupancy.inside
                    )
            breach = (
                not bool(drivable_inside)
                if drivable_inside != ""
                else bool(clearance_m < 0.0)
            )
            if bool(record_sample):
                self.sample_count += 1
                if breach:
                    self.breach_count += 1
            result.update(
                {
                    "road_boundary_sample_valid": True,
                    "road_boundary_lateral_offset_m": float(lateral_offset_m),
                    "road_boundary_lane_width_m": float(lane_width_m),
                    "road_boundary_ego_half_width_m": float(ego_half_width_m),
                    "road_boundary_clearance_m": float(clearance_m),
                    "road_boundary_breach": bool(breach),
                    "road_boundary_heading_error_rad": float(
                        occupancy.heading_error_rad
                    ),
                    "road_boundary_projection_segment_index": (
                        int(projection.segment_index)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_segment_ratio": (
                        float(projection.segment_ratio)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_raw_heading_rad": (
                        float(projection.raw_heading_rad)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_conditioned_heading_rad": (
                        float(projection.conditioned_heading_rad)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_continuity_limited": (
                        bool(projection.continuity_limited)
                        if projection is not None
                        else False
                    ),
                    "road_boundary_projection_reason": (
                        str(projection.reason)
                        if projection is not None
                        else "lane_corridor_occupancy:map_fallback"
                    ),
                    "road_boundary_geometry_source": str(
                        geometry_source
                    ),
                    "road_boundary_drivable_inside": (
                        drivable_inside
                    ),
                }
            )
        except Exception:
            pass
        return result
