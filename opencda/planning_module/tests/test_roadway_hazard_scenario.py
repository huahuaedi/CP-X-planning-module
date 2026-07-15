import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from carla_scenario import list_available_scenarios, load_carla_scenario
from carla_scenario.roadway_hazard.scenario import (
    filter_dynamic_obstacle_snapshots,
    initialize_runtime,
    maybe_replan_global_route,
)


class _FakeCityObjectLabel:
    Any = "Any"


class _FakeCarla:
    CityObjectLabel = _FakeCityObjectLabel


class _FakeEnvironmentObject:
    def __init__(self, name: str, x: float, y: float, z: float = 0.0):
        self.name = str(name)
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
            rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        )


class _FakeWorld:
    def __init__(self, *, environment_objects=None, actors=None):
        self._environment_objects = list(environment_objects or [])
        self._actors = list(actors or [])

    def get_environment_objects(self, _label):
        return list(self._environment_objects)

    def get_actors(self):
        return list(self._actors)


def _ego_transform(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(
        location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
        rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
    )


class RoadwayHazardScenarioTests(unittest.TestCase):
    def test_scenario_is_available_and_uses_expected_runtime_module(self):
        self.assertIn("roadway_hazard", list_available_scenarios())

        scenario_cfg = load_carla_scenario("roadway_hazard")

        self.assertEqual(
            str(scenario_cfg.get("obstacles", {}).get("spawner_module", "")),
            "carla_scenario.roadway_hazard.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "carla_scenario.roadway_hazard.scenario",
        )

    def test_hidden_obstacle_stays_filtered_before_cooperative_trigger(self):
        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "hidden_obstacle_id": "obstacle4",
                    "relay_obstacle_id": "obstacle6",
                    "reveal_distance_m": 20.0,
                }
            }
        )

        filtered_snapshots, next_runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            object_snapshots=[
                {"vehicle_id": "obstacle4", "x": 0.0, "y": 0.0},
                {"vehicle_id": "obstacle6", "x": 25.0, "y": 0.0},
                {"vehicle_id": "obstacle2", "x": 10.0, "y": 0.0},
            ],
        )

        filtered_ids = [str(snapshot.get("vehicle_id", "")) for snapshot in filtered_snapshots]
        self.assertNotIn("obstacle4", filtered_ids)
        self.assertIn("obstacle6", filtered_ids)
        self.assertFalse(bool(next_runtime_state.get("hidden_obstacle_revealed", False)))

    def test_hidden_obstacle_is_revealed_when_obstacle6_is_within_20m(self):
        runtime_state = initialize_runtime(
            scenario_cfg={
                "runtime": {
                    "hidden_obstacle_id": "obstacle4",
                    "relay_obstacle_id": "obstacle6",
                    "reveal_distance_m": 20.0,
                }
            }
        )

        filtered_snapshots, next_runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state=runtime_state,
            object_snapshots=[
                {"vehicle_id": "obstacle4", "x": 0.0, "y": 0.0},
                {"vehicle_id": "obstacle6", "x": 19.5, "y": 0.0},
            ],
        )

        filtered_ids = [str(snapshot.get("vehicle_id", "")) for snapshot in filtered_snapshots]
        self.assertIn("obstacle4", filtered_ids)
        self.assertTrue(bool(next_runtime_state.get("hidden_obstacle_revealed", False)))

    def test_hidden_obstacle_stays_visible_after_trigger_latches(self):
        filtered_snapshots, next_runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state={
                "hidden_obstacle_id": "obstacle4",
                "relay_obstacle_id": "obstacle6",
                "reveal_distance_m": 20.0,
                "hidden_obstacle_revealed": True,
            },
            object_snapshots=[
                {"vehicle_id": "obstacle4", "x": 0.0, "y": 0.0},
                {"vehicle_id": "obstacle6", "x": 40.0, "y": 0.0},
            ],
        )

        filtered_ids = [str(snapshot.get("vehicle_id", "")) for snapshot in filtered_snapshots]
        self.assertIn("obstacle4", filtered_ids)
        self.assertTrue(bool(next_runtime_state.get("hidden_obstacle_revealed", False)))

    def test_hazard_snapshot_stays_hidden_before_cp_trigger(self):
        filtered_snapshots, next_runtime_state = filter_dynamic_obstacle_snapshots(
            runtime_state={
                "mode": "hazard_cp",
                "hazard_vehicle_id": "hazard",
                "hazard_revealed": False,
            },
            object_snapshots=[
                {"vehicle_id": "hazard", "x": 10.0, "y": 0.0},
                {"vehicle_id": "sumo_1", "x": 15.0, "y": 0.0},
            ],
        )

        filtered_ids = [str(snapshot.get("vehicle_id", "")) for snapshot in filtered_snapshots]
        self.assertNotIn("hazard", filtered_ids)
        self.assertIn("sumo_1", filtered_ids)
        self.assertFalse(bool(next_runtime_state.get("hazard_revealed", False)))

    def test_hazard_message_is_inserted_once_when_ego_is_within_trigger_distance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump([], message_file)

            runtime_state = initialize_runtime(
                scenario_cfg={
                    "runtime": {
                        "hazard_name": "hazard",
                        "hazard_vehicle_id": "hazard",
                        "cooperative_message_trigger_distance_m": 20.0,
                        "cp_message_id": "roadway_hazard_message",
                        "cp_message_path": message_path,
                    }
                }
            )

            route_summary, route_points, next_runtime_state = maybe_replan_global_route(
                runtime_state=runtime_state,
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("hazard", x=30.0, y=0.0),
                    ]
                ),
                carla=_FakeCarla,
                ego_transform=_ego_transform(0.0, 0.0),
                sim_time_s=0.0,
                wall_time_s=0.0,
            )
            self.assertIsNone(route_summary)
            self.assertIsNone(route_points)
            self.assertFalse(bool(next_runtime_state.get("cp_message_inserted", False)))
            self.assertAlmostEqual(float(next_runtime_state.get("last_hazard_distance_m")), 30.0, places=3)

            route_summary, route_points, next_runtime_state = maybe_replan_global_route(
                runtime_state=next_runtime_state,
                world=_FakeWorld(
                    environment_objects=[
                        _FakeEnvironmentObject("hazard", x=30.0, y=0.0),
                    ]
                ),
                carla=_FakeCarla,
                ego_transform=_ego_transform(12.0, 0.0),
                sim_time_s=1.0,
                wall_time_s=1.0,
            )
            self.assertIsNone(route_summary)
            self.assertIsNone(route_points)
            self.assertTrue(bool(next_runtime_state.get("cp_message_inserted", False)))
            self.assertTrue(bool(next_runtime_state.get("hazard_revealed", False)))

            with open(message_path, "r", encoding="utf-8") as message_file:
                messages = json.load(message_file)

            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["type"], "hazard")
            self.assertEqual(messages[0]["hazard_vehicle_id"], "hazard")
            self.assertEqual(messages[0]["position"], [30.0, 0.0])

            filtered_snapshots, _ = filter_dynamic_obstacle_snapshots(
                runtime_state=next_runtime_state,
                object_snapshots=[
                    {"vehicle_id": "hazard", "x": 30.0, "y": 0.0},
                ],
            )
            self.assertEqual(len(filtered_snapshots), 1)
            self.assertEqual(filtered_snapshots[0]["vehicle_id"], "hazard")


if __name__ == "__main__":
    unittest.main()
