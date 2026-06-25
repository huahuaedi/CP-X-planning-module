import json
import os
import tempfile
import types
import unittest
from unittest import mock

from main import list_available_scenarios, load_any_scenario
from opencda_scenario.town10_scenario_5 import scenario as town10_scenario_5


class _FakeCarla:
    class CityObjectLabel:
        Any = object()

    class LaneType:
        Driving = object()

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class Rotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch = float(pitch)
            self.yaw = float(yaw)
            self.roll = float(roll)

    class Transform:
        def __init__(self, location, rotation):
            self.location = location
            self.rotation = rotation


class _FakeEnvironmentObject:
    def __init__(self, name: str, transform=None):
        self.name = str(name)
        self.transform = transform


class _FakeWaypoint:
    def __init__(self, x: float, y: float, *, road_id: int = 12, section_id: int = 3, lane_id: int = 1):
        self.transform = _make_transform(x, y)
        self.road_id = int(road_id)
        self.section_id = int(section_id)
        self.lane_id = int(lane_id)


class _FakeWorldMap:
    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        del project_to_road, lane_type
        return _FakeWaypoint(float(location.x), float(location.y))


class _FakeWorld:
    def __init__(self, actors=None, environment_objects=None):
        self._actors = list(actors or [])
        self._environment_objects = list(environment_objects or [])

    def get_actors(self):
        return list(self._actors)

    def get_environment_objects(self, _label):
        return list(self._environment_objects)


def _make_transform(x: float, y: float, z: float = 0.0, yaw: float = 0.0):
    return _FakeCarla.Transform(
        _FakeCarla.Location(x=x, y=y, z=z),
        _FakeCarla.Rotation(yaw=yaw),
    )


class Town10Scenario5Tests(unittest.TestCase):
    def test_scenario_is_available_and_uses_town10_scenario_5_modules(self):
        self.assertIn("town10_scenario_5", list_available_scenarios())

        scenario_cfg = load_any_scenario("town10_scenario_5")

        self.assertEqual(
            str(scenario_cfg.get("runner_module", "")),
            "opencda_scenario.town10_scenario_5.runner",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "opencda_scenario.town10_scenario_5.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("hazard_marker_prefix", "")),
            "hazard_",
        )

    def test_hazard_markers_register_lane_closures_once_each(self):
        world = _FakeWorld(
            environment_objects=[
                _FakeEnvironmentObject("hazard_1", transform=_make_transform(11.0, 2.0)),
                _FakeEnvironmentObject("hazard_2", transform=_make_transform(101.0, 2.0)),
            ],
        )
        spawn_requests = []

        def _fake_spawn_vehicle_at_marker(**kwargs):
            spawn_requests.append(
                {
                    "role_name": str(kwargs.get("role_name", "")),
                    "autopilot_enabled": bool(kwargs.get("autopilot_enabled", False)),
                }
            )
            return types.SimpleNamespace(id=300 + len(spawn_requests))

        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with mock.patch.object(
                town10_scenario_5.base,
                "_spawn_vehicle_at_marker",
                side_effect=_fake_spawn_vehicle_at_marker,
            ):
                runtime_state = town10_scenario_5.initialize_runtime(
                    scenario_cfg={
                        "runtime": {
                            "cp_message_path": message_path,
                            "hazard_marker_prefix": "hazard_",
                            "hazard_trigger_distance_m": 20.0,
                        },
                        "traffic_manager": {"port": 8000},
                    },
                    world=world,
                    world_map=_FakeWorldMap(),
                    carla=_FakeCarla,
                )

            self.assertEqual(
                [entry.get("name") for entry in runtime_state.get("hazard_markers", [])],
                ["hazard_1", "hazard_2"],
            )
            self.assertEqual(
                spawn_requests,
                [
                    {"role_name": "hazard_1", "autopilot_enabled": False},
                    {"role_name": "hazard_2", "autopilot_enabled": False},
                ],
            )
            self.assertEqual(runtime_state.get("hazard_vehicle_actor_ids", []), [301, 302])

            _route_summary, _route_points, runtime_state = town10_scenario_5.maybe_replan_global_route(
                runtime_state=runtime_state,
                world=world,
                world_map=_FakeWorldMap(),
                carla=_FakeCarla,
                ego_transform=_make_transform(11.5, 0.0),
                active_global_route_points=[],
                sim_time_s=5.0,
            )

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

            self.assertEqual(len(payload.get("lane_events", [])), 1)
            first_event = dict(payload["lane_events"][0])
            self.assertEqual(str(first_event.get("id", "")), "town10_scenario_5_lane_closure_1")
            self.assertEqual(str(first_event.get("type", "")), "lane_closure")
            self.assertEqual(first_event.get("lane_ids"), [1])
            self.assertEqual(first_event.get("road_id"), 12)
            self.assertEqual(first_event.get("section_id"), 3)
            self.assertEqual(first_event.get("position"), [11.0, 2.0])

            _route_summary, _route_points, runtime_state = town10_scenario_5.maybe_replan_global_route(
                runtime_state=runtime_state,
                world=world,
                world_map=_FakeWorldMap(),
                carla=_FakeCarla,
                ego_transform=_make_transform(101.5, 0.0),
                active_global_route_points=[],
                sim_time_s=6.0,
            )

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

        self.assertEqual(
            {entry.get("id") for entry in payload.get("lane_events", [])},
            {"town10_scenario_5_lane_closure_1", "town10_scenario_5_lane_closure_2"},
        )


if __name__ == "__main__":
    unittest.main()
