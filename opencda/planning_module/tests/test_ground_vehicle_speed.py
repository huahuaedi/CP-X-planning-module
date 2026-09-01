import sys
import types
import unittest

_had_carla = "carla" in sys.modules
if not _had_carla:
    sys.modules["carla"] = types.SimpleNamespace(
        Color=lambda *args, **kwargs: (args, kwargs),
    )

from opencda.core.common.misc import get_speed

if not _had_carla:
    del sys.modules["carla"]


class _Velocity:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _Vehicle:
    def __init__(self, velocity):
        self._velocity = velocity

    def get_velocity(self):
        return self._velocity


class GroundVehicleSpeedTest(unittest.TestCase):
    def test_vertical_spawn_motion_is_not_reported_as_vehicle_speed(self):
        vehicle = _Vehicle(_Velocity(0.0, 0.0, -4.0))
        self.assertEqual(get_speed(vehicle, meters=True), 0.0)
        self.assertEqual(get_speed(vehicle), 0.0)

    def test_xy_velocity_is_preserved_in_mps_and_kmh(self):
        vehicle = _Vehicle(_Velocity(3.0, 4.0, 12.0))
        self.assertAlmostEqual(get_speed(vehicle, meters=True), 5.0)
        self.assertAlmostEqual(get_speed(vehicle), 18.0)


if __name__ == "__main__":
    unittest.main()
