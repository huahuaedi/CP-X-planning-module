import json
import os
import tempfile
import types
import unittest
from unittest import mock

from main import list_available_scenarios, load_any_scenario
from opencda_scenario.sumo_assets import resolve_xodr_path
from opencda_scenario.town10_scenario_1 import runner as town10_scenario_1_runner
from opencda_scenario.town10_scenario_1 import scenario as town10_scenario_1


class _FakeWalkerControl:
    def __init__(self):
        self.direction = None
        self.speed = 0.0
        self.jump = False


class _FakeVector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _FakeCarla:
    WalkerControl = _FakeWalkerControl
    Vector3D = _FakeVector3D

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


class _FakeWalkerActor:
    def __init__(self, actor_id: int, role_name: str, transform):
        self.id = int(actor_id)
        self.attributes = {"role_name": str(role_name)}
        self.type_id = "walker.pedestrian.0001"
        self._transform = transform
        self.last_control = None
        self.bounding_box = types.SimpleNamespace(
            extent=types.SimpleNamespace(x=0.3, y=0.3, z=0.9)
        )

    def apply_control(self, control):
        self.last_control = control

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def destroy(self):
        return None


class _FakeVehicleActor:
    def __init__(self, actor_id: int, role_name: str):
        self.id = int(actor_id)
        self.attributes = {"role_name": str(role_name)}
        self.type_id = "vehicle.tesla.model3"
        self.autopilot_enabled = None

    def set_simulate_physics(self, _enabled):
        return None

    def set_autopilot(self, enabled, *_args):
        self.autopilot_enabled = bool(enabled)

    def apply_control(self, _control):
        return None

    def set_target_velocity(self, _value):
        return None

    def set_target_angular_velocity(self, _value):
        return None

    def destroy(self):
        return None


class _FakeWaypoint:
    def __init__(self, x: float, y: float, *, road_id: int = 1, section_id: int = 1, lane_id: int = 1):
        self.transform = _make_transform(x, y)
        self.road_id = int(road_id)
        self.section_id = int(section_id)
        self.lane_id = int(lane_id)


class _FakeTrafficLightActor:
    def __init__(
        self,
        actor_id: int,
        name: str,
        x: float,
        y: float,
        *,
        state: str,
        stop_waypoints=None,
    ):
        self.id = int(actor_id)
        self.attributes = {"name": str(name)}
        self.type_id = "traffic.traffic_light"
        self._transform = _make_transform(x, y)
        self._state = str(state)
        self._stop_waypoints = list(stop_waypoints or [])

    def get_transform(self):
        return self._transform

    def get_state(self):
        return self._state

    def get_stop_waypoints(self):
        return list(self._stop_waypoints)


class _FakeWorld:
    def __init__(self, actors=None, environment_objects=None):
        self._actors = list(actors or [])
        self._environment_objects = list(environment_objects or [])
        self.next_actor_id = 100

    def add_actor(self, actor):
        self._actors.append(actor)
        return actor

    def get_actors(self):
        return list(self._actors)

    def get_environment_objects(self, _label):
        return list(self._environment_objects)


def _make_transform(x: float, y: float, z: float = 0.0, yaw: float = 0.0):
    return _FakeCarla.Transform(
        _FakeCarla.Location(x=x, y=y, z=z),
        _FakeCarla.Rotation(yaw=yaw),
    )


class Town10Scenario1Tests(unittest.TestCase):
    def test_scenario_is_available_and_uses_town10(self):
        self.assertIn("town10_scenario_1", list_available_scenarios())

        scenario_cfg = load_any_scenario("town10_scenario_1")

        self.assertEqual(
            str(scenario_cfg.get("runner_module", "")),
            "opencda_scenario.town10_scenario_1.runner",
        )
        self.assertEqual(
            str(scenario_cfg.get("carla", {}).get("map", "")),
            "/Game/Carla/Maps/Town10HD_Opt",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "opencda_scenario.town10_scenario_1.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("anchors", {}).get("ego_spawn", "")),
            "town10_scenario_1_ego",
        )
        self.assertEqual(
            str(scenario_cfg.get("anchors", {}).get("final_destination", "")),
            "town10_scenario_1_final_destination",
        )

    def test_scenario_resolves_town10_xodr(self):
        scenario_cfg = load_any_scenario("town10_scenario_1")

        xodr_path = resolve_xodr_path(
            scenario_cfg=scenario_cfg,
            sumo_cfg=scenario_cfg.get("sumo", {}),
        )

        self.assertTrue(os.path.isfile(xodr_path))
        self.assertTrue(xodr_path.endswith("Town10HD_Opt.xodr"))

    def test_initialize_runtime_spawns_vehicles_and_town10_vrus(self):
        world = _FakeWorld(
            environment_objects=[
                _FakeEnvironmentObject("vehicle_1", transform=_make_transform(0.0, 0.0)),
                _FakeEnvironmentObject("vehicle_2", transform=_make_transform(10.0, 0.0)),
                _FakeEnvironmentObject("vehicle_52", transform=_make_transform(12.0, 0.0)),
                _FakeEnvironmentObject("vru_1", transform=_make_transform(11.0, 2.0)),
                _FakeEnvironmentObject("vru_2", transform=_make_transform(15.0, 2.0)),
                _FakeEnvironmentObject("vru_3", transform=_make_transform(100.0, 2.0)),
                _FakeEnvironmentObject("vru_4", transform=_make_transform(105.0, 2.0)),
            ],
        )
        vehicle_spawn_requests = []

        def _fake_spawn_vehicle_at_marker(**kwargs):
            vehicle_spawn_requests.append(
                {
                    "role_name": str(kwargs.get("role_name", "")),
                    "autopilot_enabled": bool(kwargs.get("autopilot_enabled", False)),
                }
            )
            actor = _FakeVehicleActor(
                actor_id=world.next_actor_id,
                role_name=str(kwargs.get("role_name", "")),
            )
            world.next_actor_id += 1
            return world.add_actor(actor)

        def _fake_spawn_vru(**kwargs):
            role_name = str(kwargs.get("role_name", ""))
            actor = _FakeWalkerActor(
                actor_id=world.next_actor_id,
                role_name=role_name,
                transform=kwargs.get("start_transform"),
            )
            world.next_actor_id += 1
            return world.add_actor(actor), None

        with mock.patch.object(
            town10_scenario_1.base,
            "_spawn_vehicle_at_marker",
            side_effect=_fake_spawn_vehicle_at_marker,
        ):
            with mock.patch.object(
                town10_scenario_1.base,
                "_spawn_vru",
                side_effect=_fake_spawn_vru,
            ):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    runtime_state = town10_scenario_1.initialize_runtime(
                        scenario_cfg={
                            "runtime": {
                                "cp_message_path": os.path.join(tmp_dir, "cp_message.json"),
                            },
                            "traffic_manager": {"port": 8000},
                        },
                        world=world,
                        world_map=None,
                        carla=_FakeCarla,
                    )

        self.assertTrue(bool(runtime_state.get("manual_vehicles_spawned", False)))
        self.assertEqual(
            vehicle_spawn_requests,
            [
                {"role_name": "vehicle_1", "autopilot_enabled": True},
                {"role_name": "vehicle_2", "autopilot_enabled": False},
                {"role_name": "vehicle_52", "autopilot_enabled": False},
            ],
        )
        self.assertEqual(runtime_state.get("delayed_autopilot_vehicle_actor_ids", {}), {52: 102})
        self.assertEqual(
            [entry.get("role_name") for entry in runtime_state.get("vru_states", [])],
            ["town10_scenario_1_vru_1", "town10_scenario_1_vru_3"],
        )

    def test_runtime_emits_town10_prefixed_control_messages(self):
        near_signal = _FakeTrafficLightActor(
            actor_id=201,
            name="near_signal",
            x=12.0,
            y=0.0,
            state="Red",
            stop_waypoints=[_FakeWaypoint(11.0, 0.0)],
        )
        world = _FakeWorld(
            actors=[near_signal],
            environment_objects=[
                _FakeEnvironmentObject("vehicle_52", transform=_make_transform(12.0, 0.0)),
                _FakeEnvironmentObject("vru_1", transform=_make_transform(11.0, 2.0)),
                _FakeEnvironmentObject("vru_2", transform=_make_transform(15.0, 2.0)),
                _FakeEnvironmentObject("stop_1", transform=_make_transform(11.0, -1.0)),
                _FakeEnvironmentObject("intersection_1", transform=_make_transform(11.0, 1.0)),
            ],
        )

        def _fake_spawn_vehicle_at_marker(**kwargs):
            actor = _FakeVehicleActor(
                actor_id=world.next_actor_id,
                role_name=str(kwargs.get("role_name", "")),
            )
            if bool(kwargs.get("autopilot_enabled", False)):
                actor.autopilot_enabled = True
            world.next_actor_id += 1
            return world.add_actor(actor)

        def _fake_spawn_vru(**kwargs):
            actor = _FakeWalkerActor(
                actor_id=world.next_actor_id,
                role_name=str(kwargs.get("role_name", "")),
                transform=kwargs.get("start_transform"),
            )
            world.next_actor_id += 1
            return world.add_actor(actor), None

        with mock.patch.object(
            town10_scenario_1.base,
            "_spawn_vehicle_at_marker",
            side_effect=_fake_spawn_vehicle_at_marker,
        ):
            with mock.patch.object(
                town10_scenario_1.base,
                "_spawn_vru",
                side_effect=_fake_spawn_vru,
            ):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    message_path = os.path.join(tmp_dir, "cp_message.json")
                    runtime_state = town10_scenario_1.initialize_runtime(
                        scenario_cfg={
                            "runtime": {
                                "cp_message_path": message_path,
                            },
                            "traffic_manager": {"port": 8000},
                        },
                        world=world,
                        world_map=None,
                        carla=_FakeCarla,
                    )

                    _route_summary, _route_points, runtime_state = town10_scenario_1.maybe_replan_global_route(
                        runtime_state=runtime_state,
                        world=world,
                        world_map=None,
                        carla=_FakeCarla,
                        ego_transform=_make_transform(11.5, 0.0),
                        active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                        sim_time_s=5.0,
                    )

                    delayed_actor = next(
                        actor for actor in world.get_actors() if getattr(actor, "attributes", {}).get("role_name") == "vehicle_52"
                    )
                    first_vru = next(
                        actor for actor in world.get_actors() if getattr(actor, "attributes", {}).get("role_name") == "town10_scenario_1_vru_1"
                    )

                    self.assertTrue(bool(delayed_actor.autopilot_enabled))
                    vru_states = {
                        int(entry.get("start_index", -1)): dict(entry)
                        for entry in runtime_state.get("vru_states", [])
                    }
                    self.assertTrue(bool(vru_states[1].get("movement_started", False)))
                    self.assertGreater(float(first_vru.last_control.speed), 0.0)

                    with open(message_path, "r", encoding="utf-8") as message_file:
                        payload = json.load(message_file)

        self.assertEqual(
            {entry.get("id") for entry in payload.get("control", [])},
            {"town10_scenario_1_stop_1", "town10_scenario_1_traffic_light_1"},
        )
        intersection_controls = [
            dict(entry)
            for entry in payload.get("control", [])
            if str(entry.get("id", "")) == "town10_scenario_1_traffic_light_1"
        ]
        self.assertEqual(len(intersection_controls), 1)
        self.assertEqual(str(intersection_controls[0].get("type", "")), "intersection")
        self.assertEqual(str(intersection_controls[0].get("state", "")), "red")
        self.assertTrue(bool(intersection_controls[0].get("stop", False)))

    def test_runner_aligns_final_destination_before_delegating(self):
        target_marker_name = "custom_destination_marker"
        world = _FakeWorld(
            environment_objects=[
                _FakeEnvironmentObject(
                    "final_destination",
                    transform=_make_transform(1.0, 2.0, yaw=5.0),
                ),
                _FakeEnvironmentObject(
                    target_marker_name,
                    transform=_make_transform(30.0, 40.0, yaw=90.0),
                ),
            ],
        )

        with mock.patch.object(
            town10_scenario_1_runner.base_runner,
            "run_loaded_world",
            return_value=17,
        ) as delegate_run:
            result = town10_scenario_1_runner.run_loaded_world(
                client=object(),
                world=world,
                scenario_cfg={
                    "anchors": {
                        "ego_spawn": "town10_scenario_1_ego",
                        "final_destination": target_marker_name,
                    }
                },
                carla=_FakeCarla,
            )

        self.assertEqual(result, 17)
        aligned_marker = next(
            env_obj for env_obj in world.get_environment_objects(_FakeCarla.CityObjectLabel.Any)
            if str(env_obj.name) == "final_destination"
        )
        self.assertEqual(float(aligned_marker.transform.location.x), 30.0)
        self.assertEqual(float(aligned_marker.transform.location.y), 40.0)
        self.assertEqual(float(aligned_marker.transform.rotation.yaw), 90.0)

        delegated_cfg = delegate_run.call_args.kwargs["scenario_cfg"]
        self.assertEqual(
            str(delegated_cfg.get("anchors", {}).get("final_destination", "")),
            target_marker_name,
        )


if __name__ == "__main__":
    unittest.main()
