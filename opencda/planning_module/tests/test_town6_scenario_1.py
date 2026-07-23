import json
import os
import tempfile
import types
import unittest
from unittest import mock

from main import list_available_scenarios, load_any_scenario
from opencda_scenario.sumo_assets import resolve_xodr_path
from opencda_scenario.town6_scenario_1 import scenario as town6_scenario_1


class _FakeVehicleControl:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


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
    VehicleControl = _FakeVehicleControl
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
        self.last_control = None

    def set_simulate_physics(self, _enabled):
        return None

    def set_autopilot(self, enabled, *_args):
        self.autopilot_enabled = bool(enabled)

    def apply_control(self, control):
        self.last_control = control

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


class Town6Scenario1Tests(unittest.TestCase):
    def test_scenario_is_available_and_uses_town06(self):
        self.assertIn("town6_scenario_1", list_available_scenarios())

        scenario_cfg = load_any_scenario("town6_scenario_1")

        self.assertEqual(str(scenario_cfg.get("runner_module", "")), "opencda_scenario.runner")
        self.assertEqual(
            str(scenario_cfg.get("carla", {}).get("map", "")),
            "/home/umd-user/carla_source/carla/Unreal/CarlaUE4/Content/Carla/Maps/Town06.umap",
        )
        self.assertEqual(
            str(scenario_cfg.get("runtime", {}).get("module", "")),
            "opencda_scenario.town6_scenario_1.scenario",
        )
        self.assertEqual(
            str(scenario_cfg.get("anchors", {}).get("ego_spawn", "")),
            "town6_scenario_1_ego",
        )
        self.assertEqual(
            str(scenario_cfg.get("anchors", {}).get("final_destination", "")),
            "final_destination",
        )

    def test_scenario_resolves_town06_xodr(self):
        scenario_cfg = load_any_scenario("town6_scenario_1")

        xodr_path = resolve_xodr_path(
            scenario_cfg=scenario_cfg,
            sumo_cfg=scenario_cfg.get("sumo", {}),
        )

        self.assertTrue(os.path.isfile(xodr_path))
        self.assertTrue(xodr_path.endswith("Town06.xodr"))

    def test_initialize_runtime_spawns_vehicles_and_static_vrus(self):
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
            town6_scenario_1.base,
            "_spawn_vehicle_at_marker",
            side_effect=_fake_spawn_vehicle_at_marker,
        ):
            with mock.patch.object(
                town6_scenario_1.base,
                "_spawn_vru",
                side_effect=_fake_spawn_vru,
            ):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    runtime_state = town6_scenario_1.initialize_runtime(
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
            ["town6_scenario_1_vru_1", "town6_scenario_1_vru_3"],
        )
        for entry in runtime_state.get("vru_states", []):
            self.assertIsNotNone(entry.get("actor_id", None))
            self.assertFalse(bool(entry.get("movement_started", False)))

    def test_runtime_triggers_control_messages_delayed_autopilot_and_vru_motion(self):
        world = _FakeWorld(
            environment_objects=[
                _FakeEnvironmentObject("vehicle_1", transform=_make_transform(0.0, 0.0)),
                _FakeEnvironmentObject("vehicle_52", transform=_make_transform(12.0, 0.0)),
                _FakeEnvironmentObject("vru_1", transform=_make_transform(11.0, 2.0)),
                _FakeEnvironmentObject("vru_2", transform=_make_transform(15.0, 2.0)),
                _FakeEnvironmentObject("vru_3", transform=_make_transform(100.0, 2.0)),
                _FakeEnvironmentObject("vru_4", transform=_make_transform(105.0, 2.0)),
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
            town6_scenario_1.base,
            "_spawn_vehicle_at_marker",
            side_effect=_fake_spawn_vehicle_at_marker,
        ):
            with mock.patch.object(
                town6_scenario_1.base,
                "_spawn_vru",
                side_effect=_fake_spawn_vru,
            ):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    message_path = os.path.join(tmp_dir, "cp_message.json")
                    runtime_state = town6_scenario_1.initialize_runtime(
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

                    ego_transform = _make_transform(11.5, 0.0)
                    _route_summary, _route_points, runtime_state = town6_scenario_1.maybe_replan_global_route(
                        runtime_state=runtime_state,
                        world=world,
                        world_map=None,
                        carla=_FakeCarla,
                        ego_transform=ego_transform,
                        active_global_route_points=[],
                        sim_time_s=5.0,
                    )

                    delayed_actor = next(
                        actor for actor in world.get_actors() if getattr(actor, "attributes", {}).get("role_name") == "vehicle_52"
                    )
                    first_vru = next(
                        actor for actor in world.get_actors() if getattr(actor, "attributes", {}).get("role_name") == "town6_scenario_1_vru_1"
                    )
                    second_vru = next(
                        actor for actor in world.get_actors() if getattr(actor, "attributes", {}).get("role_name") == "town6_scenario_1_vru_3"
                    )

                    self.assertTrue(bool(delayed_actor.autopilot_enabled))
                    vru_states = {
                        int(entry.get("start_index", -1)): dict(entry)
                        for entry in runtime_state.get("vru_states", [])
                    }
                    self.assertTrue(bool(vru_states[1].get("movement_started", False)))
                    self.assertFalse(bool(vru_states[3].get("movement_started", False)))
                    self.assertGreater(float(first_vru.last_control.speed), 0.0)
                    self.assertEqual(float(second_vru.last_control.speed), 0.0)

                    snapshots, _runtime_state = town6_scenario_1.filter_dynamic_obstacle_snapshots(
                        runtime_state=runtime_state,
                        world=world,
                        carla=_FakeCarla,
                        object_snapshots=[],
                    )

                    self.assertEqual(len(snapshots), 2)
                    self.assertEqual(
                        {snapshot.get("vehicle_id") for snapshot in snapshots},
                        {"town6_scenario_1_vru_1", "town6_scenario_1_vru_3"},
                    )

                    with open(message_path, "r", encoding="utf-8") as message_file:
                        payload = json.load(message_file)

        self.assertEqual(
            {entry.get("id") for entry in payload.get("control", [])},
            {"town6_scenario_1_stop_1", "town6_scenario_1_traffic_light_1"},
        )

    def test_intersection_cp_message_tracks_nearest_signal_until_crossing(self):
        near_signal = _FakeTrafficLightActor(
            actor_id=201,
            name="near_signal",
            x=12.0,
            y=0.0,
            state="Red",
            stop_waypoints=[_FakeWaypoint(11.0, 0.0)],
        )
        far_signal = _FakeTrafficLightActor(
            actor_id=202,
            name="far_signal",
            x=24.0,
            y=0.0,
            state="Green",
            stop_waypoints=[_FakeWaypoint(20.0, 0.0)],
        )
        world = _FakeWorld(
            actors=[near_signal, far_signal],
            environment_objects=[
                _FakeEnvironmentObject("intersection_1", transform=_make_transform(10.0, 0.0)),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            runtime_state = town6_scenario_1.initialize_runtime(
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

            ego_before_trigger = _make_transform(-15.0, 0.0)
            _route_summary, _route_points, runtime_state = town6_scenario_1.maybe_replan_global_route(
                runtime_state=runtime_state,
                world=world,
                world_map=None,
                carla=_FakeCarla,
                ego_transform=ego_before_trigger,
                active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                sim_time_s=1.0,
            )

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

            self.assertEqual(payload.get("control", []), [])

            ego_near_intersection = _make_transform(5.0, 0.0)
            _route_summary, _route_points, runtime_state = town6_scenario_1.maybe_replan_global_route(
                runtime_state=runtime_state,
                world=world,
                world_map=None,
                carla=_FakeCarla,
                ego_transform=ego_near_intersection,
                active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                sim_time_s=2.0,
            )

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

            self.assertEqual(len(payload.get("control", [])), 1)
            self.assertEqual(str(payload["control"][0]["id"]), "town6_scenario_1_traffic_light_1")
            self.assertEqual(str(payload["control"][0]["type"]), "intersection")
            self.assertEqual(str(payload["control"][0]["state"]), "red")
            self.assertTrue(bool(payload["control"][0]["stop"]))
            self.assertEqual(
                str(runtime_state.get("intersection_signal_actor_names", {}).get("intersection_1", "")),
                "near_signal",
            )

            near_signal._state = "Green"
            far_signal._state = "Red"
            ego_still_before_crossing = _make_transform(9.5, 0.0)
            _route_summary, _route_points, runtime_state = town6_scenario_1.maybe_replan_global_route(
                runtime_state=runtime_state,
                world=world,
                world_map=None,
                carla=_FakeCarla,
                ego_transform=ego_still_before_crossing,
                active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                sim_time_s=3.0,
            )

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

            self.assertEqual(len(payload.get("control", [])), 1)
            self.assertEqual(str(payload["control"][0]["type"]), "intersection")
            self.assertEqual(str(payload["control"][0]["state"]), "green")
            self.assertFalse(bool(payload["control"][0]["stop"]))
            self.assertEqual(
                str(runtime_state.get("intersection_signal_actor_names", {}).get("intersection_1", "")),
                "near_signal",
            )

            ego_after_crossing = _make_transform(13.0, 0.0)
            _route_summary, _route_points, runtime_state = town6_scenario_1.maybe_replan_global_route(
                runtime_state=runtime_state,
                world=world,
                world_map=None,
                carla=_FakeCarla,
                ego_transform=ego_after_crossing,
                active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                sim_time_s=4.0,
            )

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

        self.assertEqual(payload.get("control", []), [])
        self.assertEqual(runtime_state.get("crossed_intersection_marker_names", set()), {"intersection_1"})
        self.assertEqual(runtime_state.get("intersection_signal_actor_names", {}), {})

if __name__ == "__main__":
    unittest.main()
