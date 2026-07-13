import math
import unittest

from planning_runner import _align_transform_to_lane, _spawn_vehicle


class _Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)


class _Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or _Location()
        self.rotation = rotation or _Rotation()


class _Carla:
    Location = _Location
    Rotation = _Rotation
    Transform = _Transform


class _Waypoint:
    position = {"x": 10.0, "y": 20.0, "z": 0.2}
    heading = math.radians(72.0)


class _Planner:
    def get_waypoint(self, position):
        self.query = dict(position)
        return _Waypoint()


class _Blueprint:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, name, value):
        self.attributes[str(name)] = str(value)

    def has_attribute(self, name):
        return str(name) == "color"


class _BlueprintLibrary:
    def __init__(self):
        self.blueprint = _Blueprint()

    def filter(self, pattern):
        del pattern
        return [self.blueprint]


class _ActorList(list):
    def filter(self, pattern):
        del pattern
        return []


class _World:
    def __init__(self, succeed_on_call):
        self.succeed_on_call = int(succeed_on_call)
        self.spawn_attempts = []
        self.vehicle = object()

    def get_actors(self):
        return _ActorList()

    def try_spawn_actor(self, blueprint, transform):
        del blueprint
        self.spawn_attempts.append(transform)
        if len(self.spawn_attempts) == self.succeed_on_call:
            return self.vehicle
        return None


class RunnerSpawnOrientationTests(unittest.TestCase):
    def test_alignment_uses_custom_global_planner_world_heading(self):
        planner = _Planner()
        anchor = _Transform(_Location(9.5, 19.5, 1.0), _Rotation(yaw=-15.0))

        aligned, waypoint = _align_transform_to_lane(planner, _Carla, anchor)

        self.assertIsInstance(waypoint, _Waypoint)
        self.assertEqual(planner.query, {"x": 9.5, "y": 19.5, "z": 1.0})
        self.assertAlmostEqual(aligned.location.x, 10.0)
        self.assertAlmostEqual(aligned.location.y, 20.0)
        self.assertAlmostEqual(aligned.rotation.yaw, 72.0)

    def test_anchor_position_fallback_keeps_custom_lane_yaw(self):
        # Six attempts use the lane-center position. The seventh is the first
        # anchor-position attempt and must still retain the custom lane yaw.
        world = _World(succeed_on_call=7)
        lane_transform = _Transform(_Location(10.0, 20.0, 0.2), _Rotation(yaw=72.0))
        anchor_transform = _Transform(_Location(9.5, 19.5, 1.0), _Rotation(yaw=-15.0))

        vehicle = _spawn_vehicle(
            world,
            _BlueprintLibrary(),
            {"ego": {"blueprint": "vehicle.test", "role_name": "ego", "spawn_z_offset_m": 1.0}},
            carla=_Carla,
            lane_transform=lane_transform,
            anchor_transform=anchor_transform,
        )

        self.assertIs(vehicle, world.vehicle)
        self.assertEqual(len(world.spawn_attempts), 7)
        self.assertAlmostEqual(world.spawn_attempts[-1].location.x, 9.5)
        self.assertAlmostEqual(world.spawn_attempts[-1].rotation.yaw, 72.0)
        self.assertTrue(all(abs(item.rotation.yaw - 72.0) < 1.0e-9 for item in world.spawn_attempts))


if __name__ == "__main__":
    unittest.main()
