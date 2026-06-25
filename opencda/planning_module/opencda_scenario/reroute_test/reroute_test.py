"""Standalone Town10 reroute test using the planning-module A* planner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import math
import os
from pathlib import Path
import platform
import queue
import sys
import time
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencda.planning_module.utility import (  # noqa: E402
    AStarGlobalPlanner,
    build_lane_center_waypoints,
    canonical_lane_id_for_waypoint,
)

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore[assignment]


EXPECTED_TOWN10_MAP = "Town10HD_Opt"
MARKER_NAMES: Tuple[str, str, str] = ("start", "end", "close")
DEFAULT_SAMPLE_DISTANCE_M = 2.0
DEFAULT_POLL_INTERVAL_S = 0.5
DEFAULT_DRAW_LIFE_S = 1.25
DEFAULT_CAMERA_IMAGE_WIDTH_PX = 840
DEFAULT_CAMERA_IMAGE_HEIGHT_PX = 680
DEFAULT_CAMERA_FOV_DEG = 90.0
DEFAULT_TOPDOWN_CAMERA_HEIGHT_M = 55.0
DEFAULT_TOPDOWN_CAMERA_PADDING_M = 20.0
DEFAULT_ROUTE_DEBUG_LIFE_S = 3600.0


@dataclass(frozen=True)
class MarkerWaypointState:
    marker_name: str
    matched_object_name: str
    marker_position_xy: Tuple[float, float]
    waypoint_position_xy: Tuple[float, float]
    road_id: int
    section_id: int
    carla_lane_id: int
    waypoint_key: Tuple[int, int, int, float]
    waypoint: Any


def _best_partial_match(candidates: List[Tuple[int, Any]]) -> Any | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _get_carla_egg_glob(carla_root: str) -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        platform_tag = "linux-x86_64" if machine in {"x86_64", "amd64"} else f"linux-{machine}"
    elif sys.platform == "win32":
        platform_tag = "win-amd64"
    else:
        platform_tag = "*"
    return os.path.join(
        carla_root,
        "PythonAPI",
        "carla",
        "dist",
        f"carla-*{sys.version_info.major}.{sys.version_info.minor}-{platform_tag}.egg",
    )


def _load_carla_api():
    try:
        import carla  # type: ignore

        return carla
    except Exception:
        pass

    carla_root = os.environ.get("CARLA_ROOT", "/home/umd-user/carla_source/carla")
    egg_matches = glob.glob(_get_carla_egg_glob(carla_root))
    if egg_matches:
        egg_path = egg_matches[0]
        if egg_path not in sys.path:
            sys.path.append(egg_path)
        import carla  # type: ignore

        return carla
    raise RuntimeError(
        "CARLA Python API could not be imported. Set CARLA_ROOT so the matching egg can be found."
    )


def _map_matches_town10(map_name: object) -> bool:
    normalized_name = str(map_name or "").strip()
    if not normalized_name:
        return False
    short_name = normalized_name.split("/")[-1]
    return short_name in {"Town10", "Town10HD", "Town10HD_Opt", EXPECTED_TOWN10_MAP}


def _find_environment_marker_by_name(world, carla, marker_name: str):
    marker_name_lower = str(marker_name).strip().lower()
    partial_candidates: List[Tuple[int, Any]] = []
    for env_obj in world.get_environment_objects(carla.CityObjectLabel.Any):
        env_name = str(getattr(env_obj, "name", "")).strip().lower()
        if env_name == marker_name_lower:
            return env_obj
        if marker_name_lower and marker_name_lower in env_name:
            partial_candidates.append((len(env_name), env_obj))
    return _best_partial_match(partial_candidates)


def _find_actor_by_name(world, object_name: str):
    object_name_lower = str(object_name).strip().lower()
    if not object_name_lower:
        return None

    partial_candidates: List[Tuple[int, Any]] = []
    for actor in list(world.get_actors() if hasattr(world, "get_actors") else []):
        raw_attributes = getattr(actor, "attributes", {})
        attr_name = str(raw_attributes.get("name", "")).strip().lower()
        role_name = str(raw_attributes.get("role_name", "")).strip().lower()
        type_id = str(getattr(actor, "type_id", "")).strip().lower()
        if (
            attr_name == object_name_lower
            or role_name == object_name_lower
            or type_id.endswith(object_name_lower)
        ):
            return actor
        if object_name_lower in attr_name:
            partial_candidates.append((len(attr_name), actor))
        if object_name_lower in role_name:
            partial_candidates.append((len(role_name), actor))
        if object_name_lower in type_id:
            partial_candidates.append((len(type_id), actor))
    return _best_partial_match(partial_candidates)


def _resolve_named_world_object(world, carla, object_name: str):
    marker = _find_environment_marker_by_name(world, carla, object_name)
    if marker is not None:
        return marker
    return _find_actor_by_name(world, object_name)


def _object_transform(world_object: Any):
    transform = getattr(world_object, "transform", None)
    if transform is None and hasattr(world_object, "get_transform"):
        try:
            transform = world_object.get_transform()
        except RuntimeError:
            transform = None
    return transform


def _matched_object_name(world_object: Any, fallback_name: str) -> str:
    if world_object is None:
        return str(fallback_name)
    raw_attributes = getattr(world_object, "attributes", {})
    attributes_get = getattr(raw_attributes, "get", lambda *_: "")
    for candidate in (
        getattr(world_object, "name", ""),
        getattr(world_object, "type_id", ""),
        attributes_get("role_name", ""),
        attributes_get("name", ""),
    ):
        normalized_candidate = str(candidate).strip()
        if normalized_candidate:
            return normalized_candidate
    return str(fallback_name)


def _waypoint_key(waypoint: Any) -> Tuple[int, int, int, float] | None:
    if waypoint is None:
        return None
    return (
        int(getattr(waypoint, "road_id", 0)),
        int(getattr(waypoint, "section_id", 0)),
        int(getattr(waypoint, "lane_id", 0)),
        round(float(getattr(waypoint, "s", 0.0)), 3),
    )


def _resolve_marker_waypoint_state(
    *,
    world,
    world_map,
    carla,
    marker_name: str,
) -> MarkerWaypointState | None:
    world_object = _resolve_named_world_object(world, carla, marker_name)
    if world_object is None:
        return None

    transform = _object_transform(world_object)
    location = getattr(transform, "location", None)
    if location is None:
        return None

    try:
        waypoint = world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        waypoint = None
    if waypoint is None:
        return None

    waypoint_location = getattr(getattr(waypoint, "transform", None), "location", None)
    if waypoint_location is None:
        return None

    key = _waypoint_key(waypoint)
    if key is None:
        return None

    return MarkerWaypointState(
        marker_name=str(marker_name),
        matched_object_name=_matched_object_name(world_object, marker_name),
        marker_position_xy=(float(location.x), float(location.y)),
        waypoint_position_xy=(float(waypoint_location.x), float(waypoint_location.y)),
        road_id=int(getattr(waypoint, "road_id", 0)),
        section_id=int(getattr(waypoint, "section_id", 0)),
        carla_lane_id=int(getattr(waypoint, "lane_id", 0)),
        waypoint_key=key,
        waypoint=waypoint,
    )


def _marker_snapshot_signature(marker_states: Iterable[MarkerWaypointState]) -> Tuple[Tuple[object, ...], ...]:
    return tuple(
        (
            str(marker_state.marker_name),
            round(float(marker_state.marker_position_xy[0]), 3),
            round(float(marker_state.marker_position_xy[1]), 3),
            tuple(marker_state.waypoint_key),
        )
        for marker_state in marker_states
    )


def _blocked_segments_for_close_waypoint(
    planner: AStarGlobalPlanner,
    close_waypoint: Any,
) -> List[Tuple[str, int]]:
    if close_waypoint is None:
        return []
    raw_segment_keys = list(
        planner.segment_keys_for_raw_carla_lane(
            road_id=int(getattr(close_waypoint, "road_id", 0)),
            section_id=int(getattr(close_waypoint, "section_id", 0)),
            lane_id=int(getattr(close_waypoint, "lane_id", 0)),
        )
    )
    if raw_segment_keys:
        return raw_segment_keys

    return list(
        planner.segment_keys_for_road_and_lane(
            road_id=f"{int(getattr(close_waypoint, 'road_id', 0))}:{int(getattr(close_waypoint, 'section_id', 0))}",
            lane_id=int(canonical_lane_id_for_waypoint(close_waypoint)),
        )
    )


def _draw_marker_points(world, carla, marker_states: Sequence[MarkerWaypointState], life_time_s: float) -> None:
    debug = getattr(world, "debug", None)
    if debug is None:
        return

    colors = {
        "start": carla.Color(0, 255, 0),
        "end": carla.Color(0, 170, 255),
        "close": carla.Color(255, 0, 0),
    }
    for marker_state in marker_states:
        color = colors.get(str(marker_state.marker_name).lower(), carla.Color(255, 255, 255))
        marker_location = carla.Location(
            x=float(marker_state.marker_position_xy[0]),
            y=float(marker_state.marker_position_xy[1]),
            z=0.8,
        )
        waypoint_location = carla.Location(
            x=float(marker_state.waypoint_position_xy[0]),
            y=float(marker_state.waypoint_position_xy[1]),
            z=0.5,
        )
        debug.draw_point(
            marker_location,
            size=0.18,
            color=color,
            life_time=float(life_time_s),
        )
        debug.draw_point(
            waypoint_location,
            size=0.14,
            color=color,
            life_time=float(life_time_s),
        )
        debug.draw_line(
            marker_location,
            waypoint_location,
            thickness=0.06,
            color=color,
            life_time=float(life_time_s),
        )
        debug.draw_string(
            marker_location,
            str(marker_state.marker_name),
            draw_shadow=False,
            color=color,
            life_time=float(life_time_s),
            persistent_lines=False,
        )


def _draw_dotted_route(world, carla, route_points: Sequence[Sequence[float]], life_time_s: float) -> None:
    if len(route_points) < 2:
        return
    debug = getattr(world, "debug", None)
    if debug is None:
        return

    yellow = carla.Color(255, 255, 0)
    elevated_points = [
        carla.Location(x=float(point_xy[0]), y=float(point_xy[1]), z=0.55)
        for point_xy in route_points
        if isinstance(point_xy, Sequence) and len(point_xy) >= 2
    ]
    for start_index in range(0, len(elevated_points) - 1, 2):
        first_point = elevated_points[start_index]
        second_point = elevated_points[start_index + 1]
        debug.draw_point(
            first_point,
            size=0.12,
            color=yellow,
            life_time=float(life_time_s),
        )
        debug.draw_point(
            second_point,
            size=0.12,
            color=yellow,
            life_time=float(life_time_s),
        )
        debug.draw_line(
            first_point,
            second_point,
            thickness=0.10,
            color=yellow,
            life_time=float(life_time_s),
        )


def _camera_blueprint(world, width_px: int, height_px: int, fov_deg: float):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(int(width_px)))
    blueprint.set_attribute("image_size_y", str(int(height_px)))
    blueprint.set_attribute("fov", str(float(fov_deg)))
    return blueprint


def _spawn_camera(world, carla, blueprint, transform):
    sensor = world.spawn_actor(blueprint, transform)
    image_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)

    def _on_image(image) -> None:
        if image_queue.full():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                pass
        image_queue.put(image)

    sensor.listen(_on_image)
    return sensor, image_queue


def _camera_calibration_matrix(width_px: int, height_px: int, fov_deg: float) -> np.ndarray:
    focal_length_px = float(width_px) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    calibration_matrix = np.identity(3)
    calibration_matrix[0, 0] = focal_length_px
    calibration_matrix[1, 1] = focal_length_px
    calibration_matrix[0, 2] = float(width_px) / 2.0
    calibration_matrix[1, 2] = float(height_px) / 2.0
    return calibration_matrix


def _project_world_to_image(
    camera_transform,
    calibration_matrix: np.ndarray,
    world_xyz: Sequence[float],
    image_width_px: int,
    image_height_px: int,
) -> tuple[int, int] | None:
    world_point = np.array(
        [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0],
        dtype=np.float64,
    )
    world_to_camera = np.array(camera_transform.get_inverse_matrix(), dtype=np.float64)
    point_camera = np.dot(world_to_camera, world_point)
    point_camera = np.array(
        [float(point_camera[1]), -float(point_camera[2]), float(point_camera[0])],
        dtype=np.float64,
    )
    if float(point_camera[2]) <= 1e-6:
        return None

    image_point = np.dot(calibration_matrix, point_camera)
    pixel_x = int(round(float(image_point[0] / image_point[2])))
    pixel_y = int(round(float(image_point[1] / image_point[2])))
    if pixel_x < 0 or pixel_x >= int(image_width_px) or pixel_y < 0 or pixel_y >= int(image_height_px):
        return None
    return pixel_x, pixel_y


def _draw_dotted_polyline(
    surface,
    points_px: Sequence[tuple[int, int]],
    color_rgb=(255, 220, 60),
    dot_spacing_px: int = 12,
    radius_px: int = 4,
) -> None:
    if pygame is None or len(points_px) < 2:
        return

    spacing_px = max(2, int(dot_spacing_px))
    radius_px = max(1, int(radius_px))
    for idx in range(len(points_px) - 1):
        x0_px, y0_px = points_px[idx]
        x1_px, y1_px = points_px[idx + 1]
        dx_px = float(x1_px - x0_px)
        dy_px = float(y1_px - y0_px)
        segment_length_px = math.hypot(dx_px, dy_px)
        if segment_length_px <= 1e-6:
            pygame.draw.circle(surface, color_rgb, (int(x0_px), int(y0_px)), radius_px)
            continue
        steps = max(1, int(segment_length_px / float(spacing_px)))
        for step_idx in range(steps + 1):
            alpha = float(step_idx) / float(steps)
            dot_x_px = int(round(float(x0_px) + alpha * dx_px))
            dot_y_px = int(round(float(y0_px) + alpha * dy_px))
            pygame.draw.circle(surface, color_rgb, (dot_x_px, dot_y_px), radius_px)


def _split_projected_polyline_segments(
    projected_points: Sequence[tuple[int, int] | None],
) -> List[List[tuple[int, int]]]:
    segments: List[List[tuple[int, int]]] = []
    current_segment: List[tuple[int, int]] = []
    for point_px in projected_points:
        if point_px is None:
            if len(current_segment) >= 2:
                segments.append(list(current_segment))
            current_segment = []
            continue
        current_segment.append((int(point_px[0]), int(point_px[1])))
    if len(current_segment) >= 2:
        segments.append(list(current_segment))
    return segments


def _split_route_world_segments(
    route_points: Sequence[Sequence[float]],
    *,
    max_gap_m: float = 12.0,
) -> List[List[tuple[float, float]]]:
    segments: List[List[tuple[float, float]]] = []
    current_segment: List[tuple[float, float]] = []
    previous_point_xy: tuple[float, float] | None = None
    gap_threshold_m = max(0.5, float(max_gap_m))

    for point_xy in route_points:
        if len(point_xy) < 2:
            continue
        current_point_xy = (float(point_xy[0]), float(point_xy[1]))
        if (
            previous_point_xy is not None
            and math.hypot(
                current_point_xy[0] - previous_point_xy[0],
                current_point_xy[1] - previous_point_xy[1],
            ) > gap_threshold_m
        ):
            if len(current_segment) >= 2:
                segments.append(list(current_segment))
            current_segment = []
        current_segment.append(current_point_xy)
        previous_point_xy = current_point_xy

    if len(current_segment) >= 2:
        segments.append(list(current_segment))
    return segments


def _topdown_camera_transform_from_target(carla, *, x_m: float, y_m: float, z_m: float, height_m: float):
    return carla.Transform(
        carla.Location(x=float(x_m), y=float(y_m), z=float(z_m) + float(height_m)),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )


def _draw_route_overlay(
    *,
    surface,
    camera_transform,
    calibration_matrix: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    overlay_z_m: float,
    route_points: Sequence[Sequence[float]] | None,
) -> None:
    if pygame is None or route_points is None or len(route_points) < 2:
        return
    for route_segment_world in _split_route_world_segments(route_points):
        projected_points_px = [
            _project_world_to_image(
                camera_transform=camera_transform,
                calibration_matrix=calibration_matrix,
                world_xyz=[float(point_xy[0]), float(point_xy[1]), float(overlay_z_m)],
                image_width_px=image_width_px,
                image_height_px=image_height_px,
            )
            for point_xy in route_segment_world
        ]
        for route_points_px in _split_projected_polyline_segments(projected_points_px):
            if len(route_points_px) < 2:
                continue
            _draw_dotted_polyline(surface, route_points_px)


def _draw_marker_overlay(
    *,
    surface,
    marker_states: Sequence[MarkerWaypointState],
    camera_transform,
    calibration_matrix: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    overlay_z_m: float,
) -> None:
    if pygame is None:
        return
    colors = {
        "start": (0, 255, 0),
        "end": (0, 170, 255),
        "close": (255, 0, 0),
    }
    for marker_state in marker_states:
        pixel = _project_world_to_image(
            camera_transform=camera_transform,
            calibration_matrix=calibration_matrix,
            world_xyz=[
                float(marker_state.marker_position_xy[0]),
                float(marker_state.marker_position_xy[1]),
                float(overlay_z_m),
            ],
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )
        if pixel is None:
            continue
        color = colors.get(str(marker_state.marker_name).lower(), (255, 255, 255))
        pygame.draw.circle(surface, color, pixel, 7)
        pygame.draw.circle(surface, (20, 20, 20), pixel, 7, width=1)


def _draw_hud_lines(surface, font, lines: Sequence[str], top_left_px: tuple[int, int]) -> None:
    if pygame is None or font is None or len(lines) == 0:
        return

    x0_px, y0_px = int(top_left_px[0]), int(top_left_px[1])
    line_height_px = int(font.get_linesize())
    padding_px = 6
    text_surfaces = [font.render(str(line), True, (255, 255, 255)) for line in lines]
    box_width_px = max(text_surface.get_width() for text_surface in text_surfaces) + 2 * padding_px
    box_height_px = len(text_surfaces) * line_height_px + 2 * padding_px
    box_surface = pygame.Surface((box_width_px, box_height_px), pygame.SRCALPHA)
    box_surface.fill((0, 0, 0, 140))
    surface.blit(box_surface, (x0_px, y0_px))

    for idx, text_surface in enumerate(text_surfaces):
        surface.blit(
            text_surface,
            (
                x0_px + padding_px,
                y0_px + padding_px + idx * line_height_px,
            ),
        )


def _render_topdown_camera(
    *,
    display,
    image,
    route_points: Sequence[Sequence[float]],
    marker_states: Sequence[MarkerWaypointState],
    camera_transform,
    calibration_matrix: np.ndarray,
    image_width_px: int,
    image_height_px: int,
    overlay_z_m: float,
    hud_lines: Sequence[str] | None,
    hud_font,
) -> None:
    if pygame is None or image is None:
        return
    image_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_array = image_array.reshape((image.height, image.width, 4))
    image_rgb = image_array[:, :, :3][:, :, ::-1]
    surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
    _draw_route_overlay(
        surface=surface,
        camera_transform=camera_transform,
        calibration_matrix=calibration_matrix,
        image_width_px=int(image_width_px),
        image_height_px=int(image_height_px),
        overlay_z_m=float(overlay_z_m),
        route_points=route_points,
    )
    _draw_marker_overlay(
        surface=surface,
        marker_states=marker_states,
        camera_transform=camera_transform,
        calibration_matrix=calibration_matrix,
        image_width_px=int(image_width_px),
        image_height_px=int(image_height_px),
        overlay_z_m=float(overlay_z_m),
    )
    display.blit(surface, (0, 0))
    if hud_lines:
        _draw_hud_lines(display, hud_font, hud_lines, (16, 16))
    pygame.display.flip()


def _normalize_route_points_with_endpoints(
    *,
    route_points: Sequence[Sequence[float]] | None,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
) -> List[List[float]]:
    normalized_route_points: List[List[float]] = []

    def _append_xy(xy: Sequence[float] | None) -> None:
        if xy is None or len(xy) < 2:
            return
        next_point = [float(xy[0]), float(xy[1])]
        if normalized_route_points:
            prev_point = normalized_route_points[-1]
            if math.hypot(float(prev_point[0]) - float(next_point[0]), float(prev_point[1]) - float(next_point[1])) <= 0.1:
                return
        normalized_route_points.append(next_point)

    _append_xy(start_xy)
    for route_point in list(route_points or []):
        _append_xy(route_point)
    _append_xy(goal_xy)
    return normalized_route_points


def _camera_height_for_centered_focus_points(
    *,
    center_xy: Sequence[float],
    focus_points_xy: Sequence[Sequence[float]],
    image_width_px: int,
    image_height_px: int,
    fov_deg: float,
    min_height_m: float,
    padding_m: float,
) -> float:
    if len(center_xy) < 2:
        return float(min_height_m)
    center_x_m = float(center_xy[0])
    center_y_m = float(center_xy[1])
    valid_points = [
        (float(point[0]), float(point[1]))
        for point in list(focus_points_xy or [])
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if len(valid_points) == 0:
        return float(min_height_m)

    half_span_x_m = max(abs(point_x_m - center_x_m) for point_x_m, _ in valid_points) + max(0.0, float(padding_m))
    half_span_y_m = max(abs(point_y_m - center_y_m) for _, point_y_m in valid_points) + max(0.0, float(padding_m))
    horizontal_fov_rad = math.radians(float(fov_deg))
    vertical_fov_rad = 2.0 * math.atan(
        math.tan(0.5 * horizontal_fov_rad) * float(image_height_px) / max(1.0, float(image_width_px))
    )
    required_height_x_m = half_span_x_m / max(1e-6, math.tan(0.5 * horizontal_fov_rad))
    required_height_y_m = half_span_y_m / max(1e-6, math.tan(0.5 * vertical_fov_rad))
    return float(max(float(min_height_m), float(required_height_x_m), float(required_height_y_m)))


def _plan_initial_route(
    *,
    planner: AStarGlobalPlanner,
    start_state: MarkerWaypointState,
    end_state: MarkerWaypointState,
    close_state: MarkerWaypointState,
):
    start_location = getattr(getattr(start_state.waypoint, "transform", None), "location", None)
    goal_location = getattr(getattr(end_state.waypoint, "transform", None), "location", None)
    carla_route_failure_reason = ""
    carla_blocked_edges = list(
        getattr(planner, "blocked_carla_graph_edges_for_waypoints", lambda _blocked_waypoints: [])(
            [close_state.waypoint]
        )
    )
    if start_location is not None and goal_location is not None and len(carla_blocked_edges) > 0:
        route_summary = planner.plan_route_from_locations_with_blocked_carla_waypoints(
            start_location=start_location,
            goal_location=goal_location,
            blocked_waypoints=[close_state.waypoint],
            fallback_start_xy=list(start_state.waypoint_position_xy),
            fallback_goal_xy=list(end_state.waypoint_position_xy),
        )
        if bool(getattr(route_summary, "route_found", False)):
            print("new path generated by global planner")
            return (
                route_summary,
                [f"edge:{int(edge_start)}->{int(edge_end)}" for edge_start, edge_end in carla_blocked_edges],
                "blocked_close_carla_edge",
            )
        carla_route_failure_reason = str(getattr(route_summary, "debug_reason", "") or "").strip()

    start_query = planner.nearest_waypoint_query(
        x_m=float(start_state.waypoint_position_xy[0]),
        y_m=float(start_state.waypoint_position_xy[1]),
    )
    goal_query = planner.nearest_waypoint_query(
        x_m=float(end_state.waypoint_position_xy[0]),
        y_m=float(end_state.waypoint_position_xy[1]),
    )
    close_query = planner.nearest_waypoint_query(
        x_m=float(close_state.waypoint_position_xy[0]),
        y_m=float(close_state.waypoint_position_xy[1]),
    )
    if start_query is None or goal_query is None or close_query is None:
        return (
            type(
                "RouteSummary",
                (),
                {
                    "route_found": False,
                    "route_waypoints": [],
                    "debug_reason": "Could not resolve start, end, or close to a sampled graph waypoint.",
                },
            )(),
            [f"node:{int(getattr(close_query, 'index', -1))}"] if close_query is not None else [],
            "blocked_close_waypoint",
        )

    blocked_node_indices = {int(close_query.index)}
    blocked_node_indices.discard(int(start_query.index))
    blocked_node_indices.discard(int(goal_query.index))
    route_summary = planner._route_with_endpoint_candidates(
        start_x_m=float(start_state.waypoint_position_xy[0]),
        start_y_m=float(start_state.waypoint_position_xy[1]),
        goal_x_m=float(end_state.waypoint_position_xy[0]),
        goal_y_m=float(end_state.waypoint_position_xy[1]),
        start_query=start_query,
        goal_query=goal_query,
        blocked_node_indices=blocked_node_indices,
    )
    if bool(getattr(route_summary, "route_found", False)):
        print("new path generated by global planner")
    elif carla_route_failure_reason:
        combined_reason = str(getattr(route_summary, "debug_reason", "") or "").strip()
        setattr(
            route_summary,
            "debug_reason",
            (
                f"carla_edge: {carla_route_failure_reason}; internal_node: {combined_reason}"
                if combined_reason
                else f"carla_edge: {carla_route_failure_reason}"
            ),
        )
    return (
        route_summary,
        [f"node:{int(node_index)}" for node_index in sorted(int(node_index) for node_index in blocked_node_indices)],
        "blocked_close_waypoint",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA port")
    parser.add_argument("--timeout-s", default=10.0, type=float, help="CARLA client timeout")
    parser.add_argument(
        "--sample-distance-m",
        default=2.0,
        type=float,
        help="Lane-center waypoint sample distance used to build the A* graph",
    )
    parser.add_argument(
        "--poll-interval-s",
        default=0.5,
        type=float,
        help="Marker polling interval",
    )
    parser.add_argument(
        "--draw-life-s",
        default=1.25,
        type=float,
        help="CARLA debug draw lifetime",
    )
    parser.add_argument(
        "--no-load-town10",
        action="store_true",
        help="Do not auto-load Town10HD_Opt when another map is active",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _runtime_cfg(scenario_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    scenario_cfg = dict(scenario_cfg or {})
    runtime_cfg = dict(scenario_cfg.get("runtime", {}) or {})
    planning_cfg = dict(scenario_cfg.get("planning", {}) or {})
    camera_cfg = dict(scenario_cfg.get("camera", {}) or {})
    topdown_cfg = dict(camera_cfg.get("topdown", {}) or {})
    return {
        "start_marker": str(runtime_cfg.get("start_marker", MARKER_NAMES[0])).strip() or MARKER_NAMES[0],
        "end_marker": str(runtime_cfg.get("end_marker", MARKER_NAMES[1])).strip() or MARKER_NAMES[1],
        "close_marker": str(runtime_cfg.get("close_marker", MARKER_NAMES[2])).strip() or MARKER_NAMES[2],
        "sample_distance_m": float(
            runtime_cfg.get(
                "sample_distance_m",
                planning_cfg.get("waypoint_sample_distance_m", DEFAULT_SAMPLE_DISTANCE_M),
            )
        ),
        "poll_interval_s": float(runtime_cfg.get("poll_interval_s", DEFAULT_POLL_INTERVAL_S)),
        "draw_life_s": float(runtime_cfg.get("draw_life_s", DEFAULT_DRAW_LIFE_S)),
        "camera_enabled": bool(camera_cfg.get("enabled", True)),
        "camera_image_width_px": int(camera_cfg.get("image_size_x", DEFAULT_CAMERA_IMAGE_WIDTH_PX)),
        "camera_image_height_px": int(camera_cfg.get("image_size_y", DEFAULT_CAMERA_IMAGE_HEIGHT_PX)),
        "camera_fov_deg": float(camera_cfg.get("fov", DEFAULT_CAMERA_FOV_DEG)),
        "camera_height_m": float(topdown_cfg.get("height", DEFAULT_TOPDOWN_CAMERA_HEIGHT_M)),
        "camera_padding_m": float(topdown_cfg.get("padding_m", DEFAULT_TOPDOWN_CAMERA_PADDING_M)),
        "route_debug_life_s": float(runtime_cfg.get("route_debug_life_s", DEFAULT_ROUTE_DEBUG_LIFE_S)),
    }


def _ensure_town10_world(client, *, load_if_needed: bool):
    world = client.get_world()
    world_map = world.get_map()
    if _map_matches_town10(getattr(world_map, "name", "")):
        return world
    if not bool(load_if_needed):
        raise RuntimeError(
            f"Expected Town10 but CARLA is currently running '{getattr(world_map, 'name', '')}'."
        )
    print(
        f"[REROUTE_TEST] loading {EXPECTED_TOWN10_MAP} "
        f"(current map: {getattr(world_map, 'name', '<unknown>')})"
    )
    return client.load_world(EXPECTED_TOWN10_MAP)


def _build_planner(world, carla, sample_distance_m: float) -> Tuple[Any, AStarGlobalPlanner]:
    world_map = world.get_map()
    print(
        "[REROUTE_TEST] building A* waypoint graph for "
        f"{getattr(world_map, 'name', '<unknown>')} with sample_distance_m={float(sample_distance_m):.2f}"
    )
    lane_center_waypoints, _ = build_lane_center_waypoints(
        map_obj=world_map,
        carla=carla,
        sample_distance_m=float(sample_distance_m),
    )
    planner = AStarGlobalPlanner(
        lane_center_waypoints=lane_center_waypoints,
        world_map=world_map,
        route_sample_distance_m=float(sample_distance_m),
    )
    return world_map, planner


def _initialize_topdown_camera(world, carla, runtime_cfg: dict[str, Any]):
    if not bool(runtime_cfg.get("camera_enabled", True)):
        return None, None, None, None, None
    if pygame is None:
        raise RuntimeError("pygame is required for the reroute_test top-down camera window.")

    image_width_px = int(runtime_cfg["camera_image_width_px"])
    image_height_px = int(runtime_cfg["camera_image_height_px"])
    camera_fov_deg = float(runtime_cfg["camera_fov_deg"])
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((int(image_width_px), int(image_height_px)))
    pygame.display.set_caption("CARLA reroute_test - close marker top-down")
    camera_blueprint = _camera_blueprint(world, image_width_px, image_height_px, camera_fov_deg)
    initial_transform = _topdown_camera_transform_from_target(
        carla,
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        height_m=float(runtime_cfg["camera_height_m"]),
    )
    camera_actor, camera_queue = _spawn_camera(world, carla, camera_blueprint, initial_transform)
    camera_calibration_matrix = _camera_calibration_matrix(
        width_px=image_width_px,
        height_px=image_height_px,
        fov_deg=camera_fov_deg,
    )
    hud_font = pygame.font.SysFont("monospace", 18)
    return display, camera_actor, camera_queue, camera_calibration_matrix, hud_font


def run_loaded_world(client, world, scenario_cfg, carla) -> int:
    runtime_cfg = _runtime_cfg(dict(scenario_cfg or {}))
    marker_names: Tuple[str, str, str] = (
        str(runtime_cfg["start_marker"]),
        str(runtime_cfg["end_marker"]),
        str(runtime_cfg["close_marker"]),
    )
    world_map, planner = _build_planner(
        world,
        carla,
        sample_distance_m=float(runtime_cfg["sample_distance_m"]),
    )
    active_map_name = str(getattr(world_map, "name", ""))

    last_status_key: Tuple[object, ...] | None = None
    cached_route_points: List[List[float]] = []
    route_found = False
    route_debug_reason = "route not planned yet"
    route_method = "uninitialized"
    route_initialized = False
    frozen_marker_states: List[MarkerWaypointState] = []
    frozen_close_state: MarkerWaypointState | None = None
    frozen_camera_height_m = float(runtime_cfg["camera_height_m"])
    persistent_route_drawn = False
    display = None
    topdown_camera = None
    topdown_queue = None
    topdown_calibration_matrix = None
    hud_font = None

    if bool(runtime_cfg.get("camera_enabled", True)):
        (
            display,
            topdown_camera,
            topdown_queue,
            topdown_calibration_matrix,
            hud_font,
        ) = _initialize_topdown_camera(world, carla, runtime_cfg)

    print(
        "[REROUTE_TEST] watching markers "
        f"{marker_names[0]}/{marker_names[1]}/{marker_names[2]}. "
        "The route will be planned once at startup and kept visible."
    )
    try:
        while True:
            if display is not None and pygame is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("[REROUTE_TEST] window closed.")
                        return 0
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                        print("[REROUTE_TEST] stopped.")
                        return 0
            world = client.get_world()
            current_world_map = world.get_map()
            current_map_name = str(getattr(current_world_map, "name", ""))
            if current_map_name != active_map_name:
                if not _map_matches_town10(current_map_name):
                    cached_route_points = []
                    route_found = False
                    route_initialized = False
                    route_debug_reason = f"wrong map: {current_map_name}"
                    route_method = "uninitialized"
                    frozen_marker_states = []
                    frozen_close_state = None
                    persistent_route_drawn = False
                    status_key = ("wrong_map", current_map_name)
                    if status_key != last_status_key:
                        print(f"[REROUTE_TEST] Town10 is required, current map is '{current_map_name}'.")
                        last_status_key = status_key
                    time.sleep(float(runtime_cfg["poll_interval_s"]))
                    continue
                world_map, planner = _build_planner(
                    world,
                    carla,
                    sample_distance_m=float(runtime_cfg["sample_distance_m"]),
                )
                active_map_name = current_map_name
                route_initialized = False
                route_found = False
                route_debug_reason = "route not planned after world reload"
                route_method = "uninitialized"
                frozen_marker_states = []
                frozen_close_state = None
                persistent_route_drawn = False
                if topdown_camera is not None:
                    try:
                        topdown_camera.stop()
                    except Exception:
                        pass
                    try:
                        topdown_camera.destroy()
                    except Exception:
                        pass
                    (
                        display,
                        topdown_camera,
                        topdown_queue,
                        topdown_calibration_matrix,
                        hud_font,
                    ) = _initialize_topdown_camera(world, carla, runtime_cfg)

            if not bool(route_initialized):
                marker_states: List[MarkerWaypointState] = []
                missing_markers: List[str] = []
                for marker_name in marker_names:
                    marker_state = _resolve_marker_waypoint_state(
                        world=world,
                        world_map=world_map,
                        carla=carla,
                        marker_name=marker_name,
                    )
                    if marker_state is None:
                        missing_markers.append(str(marker_name))
                        continue
                    marker_states.append(marker_state)

                if missing_markers:
                    cached_route_points = []
                    route_found = False
                    route_debug_reason = "markers not found yet"
                    route_method = "uninitialized"
                    status_key = ("missing", tuple(missing_markers))
                    if status_key != last_status_key:
                        print(
                            "[REROUTE_TEST] waiting for markers: "
                            + ", ".join(str(name) for name in missing_markers)
                        )
                        last_status_key = status_key
                    time.sleep(float(runtime_cfg["poll_interval_s"]))
                    continue

                state_by_name = _marker_state_dict(marker_states)
                start_state = state_by_name[marker_names[0]]
                end_state = state_by_name[marker_names[1]]
                close_state = state_by_name[marker_names[2]]
                route_summary, blocked_segments, route_method = _plan_initial_route(
                    planner=planner,
                    start_state=start_state,
                    end_state=end_state,
                    close_state=close_state,
                )
                route_found = bool(getattr(route_summary, "route_found", False))
                cached_route_points = (
                    _normalize_route_points_with_endpoints(
                        route_points=list(getattr(route_summary, "route_waypoints", []) or []),
                        start_xy=start_state.waypoint_position_xy,
                        goal_xy=end_state.waypoint_position_xy,
                    )
                    if bool(route_found)
                    else []
                )
                route_found = bool(route_found) and len(cached_route_points) >= 2
                route_debug_reason = str(getattr(route_summary, "debug_reason", ""))
                frozen_marker_states = list(marker_states)
                frozen_close_state = close_state
                frozen_camera_height_m = _camera_height_for_centered_focus_points(
                    center_xy=close_state.marker_position_xy,
                    focus_points_xy=(
                        list(cached_route_points)
                        if len(cached_route_points) > 0
                        else [
                            list(start_state.marker_position_xy),
                            list(end_state.marker_position_xy),
                            list(close_state.marker_position_xy),
                        ]
                    ),
                    image_width_px=int(runtime_cfg["camera_image_width_px"]),
                    image_height_px=int(runtime_cfg["camera_image_height_px"]),
                    fov_deg=float(runtime_cfg["camera_fov_deg"]),
                    min_height_m=float(runtime_cfg["camera_height_m"]),
                    padding_m=float(runtime_cfg["camera_padding_m"]),
                )
                status_key = (
                    "route",
                    bool(route_found),
                    str(route_method),
                    tuple(str(item) for item in blocked_segments),
                    str(route_debug_reason),
                )
                if status_key != last_status_key:
                    _print_route_status(
                        route_found=route_found,
                        route_point_count=len(cached_route_points),
                        close_state=close_state,
                        blocked_segments=blocked_segments,
                        route_reason=f"{route_method}: {route_debug_reason}",
                    )
                    last_status_key = status_key
                if bool(route_found) and not bool(persistent_route_drawn):
                    _draw_dotted_route(
                        world,
                        carla,
                        cached_route_points,
                        life_time_s=float(runtime_cfg["route_debug_life_s"]),
                    )
                    persistent_route_drawn = True
                route_initialized = True

            marker_states = list(frozen_marker_states)
            close_state = frozen_close_state
            if close_state is None:
                time.sleep(float(runtime_cfg["poll_interval_s"]))
                continue
            if topdown_camera is not None:
                close_waypoint_location = getattr(
                    getattr(getattr(close_state.waypoint, "transform", None), "location", None),
                    "z",
                    0.0,
                )
                topdown_camera.set_transform(
                    _topdown_camera_transform_from_target(
                        carla,
                        x_m=float(close_state.marker_position_xy[0]),
                        y_m=float(close_state.marker_position_xy[1]),
                        z_m=float(close_waypoint_location),
                        height_m=float(frozen_camera_height_m),
                    )
                )
            if display is not None and topdown_queue is not None and topdown_calibration_matrix is not None:
                topdown_image = None
                try:
                    topdown_image = topdown_queue.get_nowait()
                except queue.Empty:
                    pass
                if topdown_image is not None:
                    topdown_camera_transform = getattr(topdown_image, "transform", None)
                    if topdown_camera_transform is None and topdown_camera is not None:
                        topdown_camera_transform = topdown_camera.get_transform()
                    hud_lines = [
                        f"route_points={int(len(cached_route_points))} found={bool(route_found)}",
                        f"method={str(route_method)}",
                        f"close road={int(close_state.road_id)} section={int(close_state.section_id)} lane={int(close_state.carla_lane_id)} blocked={','.join(str(item) for item in blocked_segments) or 'n/a'}",
                        f"close xy=({float(close_state.marker_position_xy[0]):.1f}, {float(close_state.marker_position_xy[1]):.1f}) h={float(frozen_camera_height_m):.1f}m",
                    ]
                    if str(route_debug_reason).strip():
                        hud_lines.append(f"reason={str(route_debug_reason)}")
                    _render_topdown_camera(
                        display=display,
                        image=topdown_image,
                        route_points=cached_route_points,
                        marker_states=marker_states,
                        camera_transform=topdown_camera_transform,
                        calibration_matrix=topdown_calibration_matrix,
                        image_width_px=int(runtime_cfg["camera_image_width_px"]),
                        image_height_px=int(runtime_cfg["camera_image_height_px"]),
                        overlay_z_m=float(close_waypoint_location),
                        hud_lines=hud_lines,
                        hud_font=hud_font,
                    )
            _draw_marker_points(
                world,
                carla,
                marker_states,
                life_time_s=float(runtime_cfg["draw_life_s"]),
            )
            _draw_dotted_route(
                world,
                carla,
                cached_route_points,
                life_time_s=float(runtime_cfg["draw_life_s"]),
            )
            time.sleep(float(runtime_cfg["poll_interval_s"]))
    except KeyboardInterrupt:
        print("[REROUTE_TEST] stopped.")
        return 0
    finally:
        if topdown_camera is not None:
            try:
                topdown_camera.stop()
            except Exception:
                pass
            try:
                topdown_camera.destroy()
            except Exception:
                pass
        if pygame is not None and bool(runtime_cfg.get("camera_enabled", True)):
            pygame.quit()


def _marker_state_dict(marker_states: Sequence[MarkerWaypointState]) -> dict[str, MarkerWaypointState]:
    return {str(marker_state.marker_name): marker_state for marker_state in marker_states}


def _print_route_status(
    *,
    route_found: bool,
    route_point_count: int,
    close_state: MarkerWaypointState,
    blocked_segments: Sequence[object],
    route_reason: str,
) -> None:
    if route_found:
        print(
            "[REROUTE_TEST] route updated "
            f"blocking raw_lane={int(close_state.carla_lane_id)} "
            f"on road={int(close_state.road_id)} section={int(close_state.section_id)} "
            f"segment_keys={list(blocked_segments)} "
            f"route_points={int(route_point_count)}"
        )
        return
    print(
        "[REROUTE_TEST] no route found "
        f"blocking raw_lane={int(close_state.carla_lane_id)} "
        f"on road={int(close_state.road_id)} section={int(close_state.section_id)}. "
        f"Reason: {str(route_reason or 'unknown')}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        carla = _load_carla_api()
    except RuntimeError as exc:
        print(f"[REROUTE_TEST] {exc}")
        return 1

    client = carla.Client(str(args.host), int(args.port))
    client.set_timeout(float(args.timeout_s))

    try:
        world = _ensure_town10_world(client, load_if_needed=not bool(args.no_load_town10))
    except RuntimeError as exc:
        print(f"[REROUTE_TEST] {exc}")
        return 1
    scenario_cfg = {
        "name": "reroute_test",
        "planning": {
            "waypoint_sample_distance_m": float(args.sample_distance_m),
        },
        "camera": {
            "enabled": True,
            "image_size_x": DEFAULT_CAMERA_IMAGE_WIDTH_PX,
            "image_size_y": DEFAULT_CAMERA_IMAGE_HEIGHT_PX,
            "fov": DEFAULT_CAMERA_FOV_DEG,
            "topdown": {
                "height": DEFAULT_TOPDOWN_CAMERA_HEIGHT_M,
            },
        },
        "runtime": {
            "start_marker": MARKER_NAMES[0],
            "end_marker": MARKER_NAMES[1],
            "close_marker": MARKER_NAMES[2],
            "sample_distance_m": float(args.sample_distance_m),
            "poll_interval_s": float(args.poll_interval_s),
            "draw_life_s": float(args.draw_life_s),
        },
    }
    return int(
        run_loaded_world(
            client=client,
            world=world,
            scenario_cfg=scenario_cfg,
            carla=carla,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
