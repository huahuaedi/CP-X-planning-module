import math
import unittest
from types import SimpleNamespace

from carla_scenario.runner import _collect_vehicle_snapshots


class _FakeVelocity:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _FakeActor:
    def __init__(self, actor_id, *, role_name="", x=0.0, y=0.0, yaw_deg=0.0, speed_mps=0.0):
        self.id = int(actor_id)
        self.attributes = {"role_name": str(role_name)}
        self.bounding_box = SimpleNamespace(extent=SimpleNamespace(x=2.0, y=1.0, z=0.8))
        self._transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
            rotation=SimpleNamespace(yaw=float(yaw_deg)),
        )
        self._velocity = _FakeVelocity(x=float(speed_mps), y=0.0, z=0.0)

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity


class _FakeActorList(list):
    def filter(self, pattern):
        if pattern != "vehicle.*":
            return []
        return list(self)


class _FakeWorld:
    def __init__(self, actors):
        self._actors = _FakeActorList(actors)

    def get_actors(self):
        return self._actors


class _FakeSumoBridge:
    def __init__(self, overrides):
        self._overrides = dict(overrides)

    def get_actor_snapshot_override(self, actor_id):
        return self._overrides.get(int(actor_id), None)


class RunnerSumoSnapshotTests(unittest.TestCase):
    def test_collect_vehicle_snapshots_falls_back_to_actor_id_when_role_name_is_empty(self):
        ego = _FakeActor(1, role_name="ego")
        obs = _FakeActor(2, role_name="", x=10.0, y=5.0, speed_mps=3.0)
        world = _FakeWorld([ego, obs])

        snapshots = _collect_vehicle_snapshots(world, ego)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["vehicle_id"], "2")
        self.assertAlmostEqual(float(snapshots[0]["v"]), 3.0, places=3)

    def test_collect_vehicle_snapshots_uses_sumo_override_for_vehicle_id_and_speed(self):
        ego = _FakeActor(1, role_name="ego")
        obs = _FakeActor(2, role_name="", x=10.0, y=5.0, yaw_deg=90.0, speed_mps=0.0)
        world = _FakeWorld([ego, obs])
        sumo_bridge = _FakeSumoBridge({
            2: {
                "vehicle_id": "sumo_veh_2",
                "v": 6.5,
                "psi": math.pi / 2.0,
            }
        })

        snapshots = _collect_vehicle_snapshots(world, ego, sumo_bridge=sumo_bridge)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["vehicle_id"], "sumo_veh_2")
        self.assertAlmostEqual(float(snapshots[0]["v"]), 6.5, places=3)
        self.assertAlmostEqual(float(snapshots[0]["psi"]), math.pi / 2.0, places=6)

    def test_collect_vehicle_snapshots_filters_out_distant_traffic(self):
        ego = _FakeActor(1, role_name="ego", x=0.0, y=0.0)
        near_obs = _FakeActor(2, x=40.0, y=0.0, speed_mps=2.0)
        far_obs = _FakeActor(3, x=150.0, y=0.0, speed_mps=4.0)
        world = _FakeWorld([ego, near_obs, far_obs])

        snapshots = _collect_vehicle_snapshots(
            world,
            ego,
            max_distance_m=100.0,
        )

        self.assertEqual([snapshot["vehicle_id"] for snapshot in snapshots], ["2"])

    def test_collect_vehicle_snapshots_limits_to_nearest_actors(self):
        ego = _FakeActor(1, role_name="ego", x=0.0, y=0.0)
        obs_a = _FakeActor(2, x=20.0, y=0.0)
        obs_b = _FakeActor(3, x=10.0, y=0.0)
        obs_c = _FakeActor(4, x=30.0, y=0.0)
        world = _FakeWorld([ego, obs_a, obs_b, obs_c])

        snapshots = _collect_vehicle_snapshots(
            world,
            ego,
            max_distance_m=100.0,
            max_snapshots=2,
        )

        self.assertEqual([snapshot["vehicle_id"] for snapshot in snapshots], ["3", "2"])


if __name__ == "__main__":
    unittest.main()
