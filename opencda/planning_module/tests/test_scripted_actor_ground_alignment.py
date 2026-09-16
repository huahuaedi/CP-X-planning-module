"""Kinematic road vehicles must not preserve their spawn clearance in motion."""

from types import SimpleNamespace

import carla

from opencda.scenario_testing.scripted_actor import ScriptedActor


class _Vehicle:
    type_id = "vehicle.tesla.model3"

    def __init__(self):
        self.transforms = []

    def set_transform(self, transform):
        self.transforms.append(transform)


class _RoadMap:
    def get_waypoint(self, location, *, project_to_road):
        assert project_to_road is True
        return SimpleNamespace(transform=carla.Transform(
            carla.Location(z=0.5 if location.x > 1.0 else 0.0)
        ))


def test_scripted_vehicle_follows_road_height_not_spawn_clearance():
    vehicle = _Vehicle()
    actor = ScriptedActor(
        vehicle,
        path_xy=((0.0, 0.0), (2.0, 0.0)),
        speed_mps=2.0,
        z_m=0.3,
        road_map=_RoadMap(),
    )
    assert abs(vehicle.transforms[-1].location.z - 0.03) < 1.0e-6
    actor.step(1.0)
    assert abs(vehicle.transforms[-1].location.z - 0.53) < 1.0e-6


def test_scripted_vehicle_keeps_configured_height_without_map():
    vehicle = _Vehicle()
    ScriptedActor(
        vehicle,
        path_xy=((0.0, 0.0), (2.0, 0.0)),
        speed_mps=2.0,
        z_m=0.3,
    )
    assert abs(vehicle.transforms[-1].location.z - 0.3) < 1.0e-6


def test_scripted_vehicle_heading_does_not_jump_at_path_corner():
    vehicle = _Vehicle()
    actor = ScriptedActor(
        vehicle,
        path_xy=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
        speed_mps=6.0,
        road_map=_RoadMap(),
    )
    for _ in range(50):
        actor.step(0.05)
    headings = [transform.rotation.yaw for transform in vehicle.transforms]
    assert max(abs(right - left) for left, right in zip(
        headings, headings[1:]
    )) < 20.0
    assert headings[0] == 0.0
    assert abs(headings[-1] - 90.0) < 1.0e-6
