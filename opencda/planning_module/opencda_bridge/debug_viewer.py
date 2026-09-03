"""Pygame debug viewer for native OpenCDA scenarios."""

from __future__ import annotations

import math
import os
import queue
from typing import Any, Sequence

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover - optional debug dependency
    pygame = None


_HUD_TEXT_COLOR = (235, 235, 235)
_HUD_ALERT_COLOR = (255, 120, 60)

# Display-layer heuristics only (not enforcement thresholds -- nothing in
# the planning pipeline compares against these). Picked to flag a jump a
# human would find visually surprising on the topdown/chase view, not to
# match any contract/safety limit.


def _csv_count(value: object) -> int:
    return len([item for item in str(value or "").split(",") if item.strip()])


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ref_source_kind(cpx_debug: dict) -> str:
    """Name the single reference pipeline outcome for this tick."""

    if str(cpx_debug.get("reference_pipeline_stage", "")) == "explicit_fallback":
        return "explicit_fallback"
    if str(cpx_debug.get("final_reference_geometry_source", "")):
        return "provider"
    return "default"


class OpenCDADebugViewer:
    """Two-camera viewer with a bottom HUD, modeled after planning_runner."""

    def __init__(
        self,
        *,
        world: Any,
        carla_module: Any,
        ego_vehicle: Any,
        width_px: int = 640,
        height_px: int = 360,
        hud_height_px: int = 320,
        fov_deg: float = 90.0,
    ) -> None:
        if pygame is None:
            raise RuntimeError("pygame is not available")
        self.world = world
        self.carla = carla_module
        self.ego_vehicle = ego_vehicle
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.hud_height_px = int(hud_height_px)
        self.sensors = []
        self._stable_route_points: list[list[float]] = []
        self._stable_route_bounds: tuple[float, float, float, float] | None = None

        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (int(self.width_px * 2), int(self.height_px + self.hud_height_px))
        )
        pygame.display.set_caption("OpenCDA + CP-X Planner - Topdown | Chase")
        self.font = pygame.font.SysFont("monospace", 14)

        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(int(self.width_px)))
        blueprint.set_attribute("image_size_y", str(int(self.height_px)))
        blueprint.set_attribute("fov", str(float(fov_deg)))

        topdown_transform = carla_module.Transform(
            carla_module.Location(x=0.0, y=0.0, z=65.0),
            carla_module.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
        chase_transform = carla_module.Transform(
            carla_module.Location(x=-8.0, y=0.0, z=2.8),
            carla_module.Rotation(pitch=-10.0, yaw=0.0, roll=0.0),
        )
        self.topdown_sensor, self.topdown_queue = self._spawn_camera(
            blueprint, topdown_transform, parent=ego_vehicle
        )
        self.chase_sensor, self.chase_queue = self._spawn_camera(
            blueprint, chase_transform, parent=ego_vehicle
        )

    @staticmethod
    def enabled_from_env() -> bool:
        return str(os.environ.get("OPENCDA_DEBUG_VIEW", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _spawn_camera(self, blueprint: Any, transform: Any, parent: Any):
        image_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        sensor = self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=parent,
            attachment_type=self.carla.AttachmentType.Rigid,
        )

        def _on_image(image) -> None:
            if image_queue.full():
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    pass
            image_queue.put(image)

        sensor.listen(_on_image)
        self.sensors.append(sensor)
        return sensor, image_queue

    def render(self, vehicle_managers: Sequence[Any]) -> None:
        if pygame is None:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        topdown_image = self._latest_image(self.topdown_queue)
        chase_image = self._latest_image(self.chase_queue)
        self.display.fill((0, 0, 0))
        if topdown_image is not None:
            topdown_surface = self._surface_from_image(topdown_image)
            self._draw_topdown_planning_overlay(topdown_surface, vehicle_managers)
            self.display.blit(topdown_surface, (0, 0))
        if chase_image is not None:
            self.display.blit(self._surface_from_image(chase_image), (self.width_px, 0))

        hud_rect = pygame.Rect(0, self.height_px, self.width_px * 2, self.hud_height_px)
        pygame.draw.rect(self.display, (18, 18, 18), hud_rect)
        pygame.draw.line(self.display, (70, 70, 70), (0, self.height_px), (self.width_px * 2, self.height_px), 1)
        self._draw_hud_lines(self._build_hud_lines(vehicle_managers), hud_rect)
        pygame.display.flip()

    @staticmethod
    def _latest_image(image_queue: "queue.Queue[Any]"):
        image = None
        while True:
            try:
                image = image_queue.get_nowait()
            except queue.Empty:
                break
        return image

    @staticmethod
    def _surface_from_image(image: Any):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb = array[:, :, :3][:, :, ::-1]
        return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def _draw_topdown_planning_overlay(self, surface: Any, vehicle_managers: Sequence[Any]) -> None:
        if pygame is None or not vehicle_managers:
            return
        ego_vm = vehicle_managers[0]
        cpx_debug = getattr(getattr(ego_vm, "cpx_planner", None), "last_debug", {}) or {}
        try:
            ego_transform = ego_vm.vehicle.get_transform()
            ego_x = float(ego_transform.location.x)
            ego_y = float(ego_transform.location.y)
            ego_yaw = math.radians(float(ego_transform.rotation.yaw))
            horizontal_span_m = 2.0 * 65.0 * math.tan(math.radians(90.0) * 0.5)
            vertical_span_m = horizontal_span_m * float(self.height_px) / max(1.0, float(self.width_px))
            px_per_m_x = float(self.width_px) / max(1.0, horizontal_span_m)
            px_per_m_y = float(self.height_px) / max(1.0, vertical_span_m)

            def project(point: Sequence[float]):
                if point is None or len(point) < 2:
                    return None
                dx = float(point[0]) - ego_x
                dy = float(point[1]) - ego_y
                forward = dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)
                # CARLA camera coordinates use +Y to the vehicle's right.
                # The previous sign mirrored every non-ego overlay across
                # the image center, so CAV boxes appeared beside their cars.
                right = -dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)
                px = int(round(0.5 * self.width_px + right * px_per_m_x))
                py = int(round(0.5 * self.height_px - forward * px_per_m_y))
                if px < -20 or px > self.width_px + 20 or py < -20 or py > self.height_px + 20:
                    return None
                return px, py

            self._draw_cav_boxes(
                surface=surface,
                vehicle_managers=vehicle_managers,
                project=project,
            )
            if not cpx_debug:
                return

            mpc_points = self._project_points(cpx_debug.get("mpc_trajectory_points", []), project)

            self._draw_dotted_polyline(surface, mpc_points, color=(45, 185, 75), radius_px=3, dot_spacing_px=9)

            self._draw_stable_route_minimap(
                surface=surface,
                route_points=cpx_debug.get("global_route_points", []),
                ego_xy=(ego_x, ego_y),
            )
        except Exception:
            return

    def _draw_cav_boxes(
        self,
        *,
        surface: Any,
        vehicle_managers: Sequence[Any],
        project: Any,
    ) -> None:
        """Mark every managed CAV with an oriented red footprint and label."""

        for index, vehicle_manager in enumerate(list(vehicle_managers or [])):
            vehicle = getattr(vehicle_manager, "vehicle", None)
            if vehicle is None:
                continue
            try:
                transform = vehicle.get_transform()
                bbox = vehicle.bounding_box
                extent = bbox.extent
                offset = bbox.location
                yaw_rad = math.radians(float(transform.rotation.yaw))
                cos_yaw = math.cos(yaw_rad)
                sin_yaw = math.sin(yaw_rad)
                center_x = (
                    float(transform.location.x)
                    + float(offset.x) * cos_yaw
                    - float(offset.y) * sin_yaw
                )
                center_y = (
                    float(transform.location.y)
                    + float(offset.x) * sin_yaw
                    + float(offset.y) * cos_yaw
                )
                corners = []
                for local_x, local_y in (
                    (float(extent.x), float(extent.y)),
                    (float(extent.x), -float(extent.y)),
                    (-float(extent.x), -float(extent.y)),
                    (-float(extent.x), float(extent.y)),
                ):
                    world_x = center_x + local_x * cos_yaw - local_y * sin_yaw
                    world_y = center_y + local_x * sin_yaw + local_y * cos_yaw
                    pixel = project((world_x, world_y))
                    if pixel is None:
                        corners = []
                        break
                    corners.append(pixel)
                if len(corners) != 4:
                    continue

                pygame.draw.polygon(surface, (245, 45, 45), corners, width=3)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue

    def _draw_stable_route_minimap(
        self,
        *,
        surface: Any,
        route_points: Sequence[Any],
        ego_xy: tuple[float, float],
    ) -> None:
        normalized_route = [
            [float(point[0]), float(point[1])]
            for point in list(route_points or [])
            if hasattr(point, "__len__") and len(point) >= 2
        ]
        # Refresh the cached route on a genuine replan (start/end moved by
        # more than a couple of lane-widths), not on every tick's tiny
        # per-frame numerical jitter in the same route -- that jitter is
        # exactly what "stable" is protecting against.
        def _distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
            return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

        replanned = (
            len(normalized_route) >= 2
            and len(self._stable_route_points) >= 2
            and (
                _distance_2d(normalized_route[0], self._stable_route_points[0]) > 8.0
                or _distance_2d(normalized_route[-1], self._stable_route_points[-1]) > 8.0
            )
        )
        if len(normalized_route) >= 2 and (not self._stable_route_points or replanned):
            self._stable_route_points = [list(point) for point in normalized_route]
            xs = [float(point[0]) for point in self._stable_route_points]
            ys = [float(point[1]) for point in self._stable_route_points]
            pad_m = max(8.0, 0.08 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
            self._stable_route_bounds = (
                min(xs) - pad_m,
                max(xs) + pad_m,
                min(ys) - pad_m,
                max(ys) + pad_m,
            )
        if len(self._stable_route_points) < 2 or self._stable_route_bounds is None:
            return

        map_width = min(230, max(160, int(0.36 * self.width_px)))
        map_height = min(150, max(110, int(0.34 * self.height_px)))
        left = 12
        top = 12
        rect = pygame.Rect(left, top, map_width, map_height)
        overlay = pygame.Surface((map_width, map_height), pygame.SRCALPHA)
        overlay.fill((10, 12, 14, 135))
        pygame.draw.rect(overlay, (230, 230, 230, 150), overlay.get_rect(), width=1)

        min_x, max_x, min_y, max_y = self._stable_route_bounds
        span_x = max(1.0, float(max_x) - float(min_x))
        span_y = max(1.0, float(max_y) - float(min_y))

        def project_world_xy(x_m: float, y_m: float):
            px = int(round(8 + (float(x_m) - min_x) / span_x * (map_width - 16)))
            py = int(round(map_height - 8 - (float(y_m) - min_y) / span_y * (map_height - 16)))
            return px, py

        route_px = [
            project_world_xy(float(point[0]), float(point[1]))
            for point in self._stable_route_points
        ]
        self._draw_solid_polyline(
            overlay,
            route_px,
            color=(218, 186, 55),
            width_px=3,
        )
        pygame.draw.circle(overlay, (218, 186, 55), route_px[0], 4)
        pygame.draw.circle(overlay, (245, 235, 150), route_px[-1], 4)
        ego_px = project_world_xy(float(ego_xy[0]), float(ego_xy[1]))
        pygame.draw.circle(overlay, (45, 185, 255), ego_px, 5)
        pygame.draw.circle(overlay, (5, 8, 10), ego_px, 5, width=1)
        label = self.font.render("CARLA ROUTE GEOMETRY", True, (235, 235, 235)) if self.font else None
        if label is not None:
            overlay.blit(label, (8, 6))
        surface.blit(overlay, rect.topleft)

    @staticmethod
    def _project_points(points: Sequence[Any], project) -> list[tuple[int, int]]:
        projected: list[tuple[int, int]] = []
        for point in list(points or []):
            try:
                pixel = project(point)
            except Exception:
                pixel = None
            if pixel is not None:
                projected.append(pixel)
        return projected

    @staticmethod
    def _draw_solid_polyline(
        surface: Any,
        points_px: Sequence[tuple[int, int]],
        *,
        color: tuple[int, int, int],
        width_px: int,
    ) -> None:
        if pygame is None or len(points_px) < 2:
            return
        deduped: list[tuple[int, int]] = []
        for point in list(points_px):
            if not deduped or point != deduped[-1]:
                deduped.append(point)
        if len(deduped) >= 2:
            pygame.draw.lines(surface, color, False, deduped, max(1, int(width_px)))

    @staticmethod
    def _draw_dotted_polyline(
        surface: Any,
        points_px: Sequence[tuple[int, int]],
        *,
        color: tuple[int, int, int],
        radius_px: int,
        dot_spacing_px: int,
    ) -> None:
        if pygame is None or len(points_px) < 2:
            return
        spacing = max(1.0, float(dot_spacing_px))
        for start, end in zip(list(points_px)[:-1], list(points_px)[1:]):
            dx = float(end[0]) - float(start[0])
            dy = float(end[1]) - float(start[1])
            length = math.hypot(dx, dy)
            if length <= 1.0:
                continue
            count = max(1, int(length / spacing))
            for index in range(count + 1):
                ratio = float(index) / max(1.0, float(count))
                x = int(round(float(start[0]) + ratio * dx))
                y = int(round(float(start[1]) + ratio * dy))
                pygame.draw.circle(surface, color, (x, y), int(radius_px))

    def _build_hud_lines(
        self, vehicle_managers: Sequence[Any]
    ) -> list[tuple[str, bool]]:
        if not vehicle_managers:
            return [("OPEN-CDA + CP-X", False), ("No vehicle manager available", False)]
        ego_vm = vehicle_managers[0]
        vehicle = ego_vm.vehicle
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed_mps = float((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5)
        objects = getattr(ego_vm.perception_manager, "objects", {}) or {}
        cpx_debug = getattr(getattr(ego_vm, "cpx_planner", None), "last_debug", {}) or {}

        mpc_status = str(cpx_debug.get("mpc_status", ""))
        mpc_status_alert = bool(mpc_status) and "solved" not in mpc_status.lower()
        ref_fallback_alert = bool(str(cpx_debug.get("reference_pipeline_fallback", "")))

        return [
            ("OPEN-CDA + CP-X", False),
            (f"vehicle_id={vehicle.id}  planner={'CP-X MPC' if getattr(ego_vm, 'cpx_planner', None) else 'OpenCDA default'}", False),
            (f"speed={speed_mps:.2f} m/s  loc=({transform.location.x:.1f}, {transform.location.y:.1f}) yaw={transform.rotation.yaw:.1f}", False),
            (f"objects={len(objects.get('vehicles', []) or [])}  traffic_lights={len(objects.get('traffic_lights', []) or [])}  v2x_nearby={len(getattr(ego_vm.v2x_manager, 'cav_nearby', {}) or {})}", False),
            (f"cp_source={cpx_debug.get('cp_provider_source', '')}  native_cp={cpx_debug.get('native_opencda_available', '')}  cp_obs={cpx_debug.get('cp_obstacle_count', '')} cp_ctrl={cpx_debug.get('cp_control_count', '')}", False),
            (f"cp_shared observers={cpx_debug.get('cp_observer_cav_count', '')} ids={cpx_debug.get('cp_observer_cav_ids', '')} multi_seen={cpx_debug.get('cp_multi_observer_obstacle_count', '')}", False),
            (f"cp_visibility enabled={cpx_debug.get('cp_visibility_filter_enabled', '')} backend={cpx_debug.get('cp_visibility_backend', '')} blind_shared={cpx_debug.get('cp_blind_spot_shared_count', '')} actors={cpx_debug.get('cp_blind_spot_shared_actor_ids', '')}", False),
            (f"cp_vru pedestrians={cpx_debug.get('cp_pedestrian_count', '')} blind={cpx_debug.get('cp_blind_spot_pedestrian_count', '')} predicted={_csv_count(cpx_debug.get('cp_prediction_used_pedestrian_ids', ''))} candidate_relevant={_csv_count(cpx_debug.get('cp_candidate_relevant_pedestrian_ids', ''))}", False),
            (f"behavior={cpx_debug.get('behavior_decision', '')}  fsm={cpx_debug.get('behavior_fsm_state', '')}  target_lane={cpx_debug.get('behavior_target_lane_id', '')}", False),
            (f"scenario={cpx_debug.get('scenario_fsm_state', '')}  turn_ahead={cpx_debug.get('carla_upcoming_turn_direction', '')} dist={cpx_debug.get('carla_upcoming_turn_distance_m', '')}", False),
            (f"lane current={cpx_debug.get('current_lane_id', '')}  dest_lane={cpx_debug.get('destination_lane_id', '')}", False),
            (f"ref={cpx_debug.get('final_reference_geometry_source', cpx_debug.get('reference_source', ''))}  ref_source_kind={_ref_source_kind(cpx_debug)}  stage={cpx_debug.get('reference_pipeline_stage', '')}  intent={cpx_debug.get('reference_pipeline_intent', '')}", False),
            (f"ref_geom first_fwd={cpx_debug.get('reference_first_forward_m', '')} first_lat={cpx_debug.get('reference_first_lateral_m', '')} lane_pts={len(cpx_debug.get('lane_reference_points', []) or [])}", False),
            (f"ref_fallback={cpx_debug.get('reference_pipeline_fallback', '')}", ref_fallback_alert),
            (f"front_gap={cpx_debug.get('front_gap_m', '')}  stop_goal={cpx_debug.get('stop_goal_active', '')}", False),
            (f"traffic raw={cpx_debug.get('traffic_signal_raw_state', '')} memory={cpx_debug.get('traffic_signal_filtered_state', '')} behavior={cpx_debug.get('traffic_signal_behavior_state', cpx_debug.get('traffic_signal_state', ''))} from_cp={cpx_debug.get('traffic_control_from_cp', '')}", False),
            (f"planner_input cp_ctrl={cpx_debug.get('planner_input_cp_traffic_control_count', '')} pred_risk={cpx_debug.get('planner_input_prediction_risky_lane_count', '')} objs={cpx_debug.get('planner_input_perception_planning_count', '')}", False),
            (f"mpc_status={cpx_debug.get('mpc_status', '')}  solve_ms={cpx_debug.get('mpc_solve_time_ms', '')} profile={cpx_debug.get('mpc_cost_profile', '')}", mpc_status_alert),
            (f"decision={cpx_debug.get('decision_final_action', '')} source={cpx_debug.get('decision_control_source', '')} veto={cpx_debug.get('decision_veto_count', '')}", False),
            (f"cmd a={cpx_debug.get('accel_cmd_mps2', '')}  steer={cpx_debug.get('steer_cmd_rad', '')}", False),
            (f"target=({cpx_debug.get('destination_x', '')}, {cpx_debug.get('destination_y', '')}) v_ref={cpx_debug.get('target_speed_mps', '')}", False),
            (f"mpc_fallback={cpx_debug.get('mpc_fallback_reason', '')}", bool(str(cpx_debug.get('mpc_fallback_reason', '')))),
        ]

    def _draw_hud_lines(self, lines: Sequence[tuple[str, bool]], rect: Any) -> None:
        if self.font is None:
            return
        columns = 2 if int(rect.width) >= 1000 else 1
        column_width = max(1, int((rect.width - 24) / columns))
        line_height = max(16, int(self.font.get_linesize()))
        max_lines_per_column = max(1, int((rect.height - 20) / line_height))
        for index, (line, is_alert) in enumerate(lines):
            column = int(index / max_lines_per_column)
            if column >= columns:
                break
            row = int(index % max_lines_per_column)
            x = int(rect.x) + 12 + column * column_width
            y = int(rect.y) + 10 + row * line_height
            color = _HUD_ALERT_COLOR if is_alert else _HUD_TEXT_COLOR
            surface = self.font.render(str(line), True, color)
            self.display.blit(surface, (x, y))

    def destroy(self) -> None:
        for sensor in list(self.sensors):
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass
        self.sensors.clear()
        if pygame is not None:
            try:
                pygame.quit()
            except Exception:
                pass
