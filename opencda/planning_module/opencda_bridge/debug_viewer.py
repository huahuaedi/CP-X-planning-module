"""Pygame debug viewer for native OpenCDA scenarios."""

from __future__ import annotations

import os
import queue
from typing import Any, Sequence

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover - optional debug dependency
    pygame = None


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
        hud_height_px: int = 220,
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
            self.display.blit(self._surface_from_image(topdown_image), (0, 0))
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

    def _build_hud_lines(self, vehicle_managers: Sequence[Any]) -> list[str]:
        if not vehicle_managers:
            return ["OPEN-CDA + CP-X", "No vehicle manager available"]
        ego_vm = vehicle_managers[0]
        vehicle = ego_vm.vehicle
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed_mps = float((velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5)
        objects = getattr(ego_vm.perception_manager, "objects", {}) or {}
        cpx_debug = getattr(getattr(ego_vm, "cpx_planner", None), "last_debug", {}) or {}
        return [
            "OPEN-CDA + CP-X",
            f"vehicle_id={vehicle.id}  planner={'CP-X MPC' if getattr(ego_vm, 'cpx_planner', None) else 'OpenCDA default'}",
            f"speed={speed_mps:.2f} m/s  loc=({transform.location.x:.1f}, {transform.location.y:.1f}) yaw={transform.rotation.yaw:.1f}",
            f"objects={len(objects.get('vehicles', []) or [])}  traffic_lights={len(objects.get('traffic_lights', []) or [])}",
            f"v2x_nearby={len(getattr(ego_vm.v2x_manager, 'cav_nearby', {}) or {})}",
            f"front_gap={cpx_debug.get('front_gap_m', '')}  stop_goal={cpx_debug.get('stop_goal_active', '')}",
            f"mpc_status={cpx_debug.get('mpc_status', '')}  solve_ms={cpx_debug.get('mpc_solve_time_ms', '')}",
            f"target=({cpx_debug.get('destination_x', '')}, {cpx_debug.get('destination_y', '')}) v_ref={cpx_debug.get('target_speed_mps', '')}",
            f"fallback={cpx_debug.get('fallback_reason', '')}",
        ]

    def _draw_hud_lines(self, lines: Sequence[str], rect: Any) -> None:
        if self.font is None:
            return
        x = int(rect.x) + 12
        y = int(rect.y) + 10
        line_height = max(16, int(self.font.get_linesize()))
        for line in lines:
            if y + line_height > rect.y + rect.height:
                break
            surface = self.font.render(str(line), True, (235, 235, 235))
            self.display.blit(surface, (x, y))
            y += line_height

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
