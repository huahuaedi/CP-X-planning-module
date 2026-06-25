import json
import os
import tempfile
import types
import unittest

from opencda_scenario.all_usecase_scenario import scenario as all_usecase_scenario


class _FakeVehicleControl:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _FakeCarla:
    VehicleControl = _FakeVehicleControl

    class CityObjectLabel:
        Any = object()

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


class _FakeVehicleActor:
    def __init__(self, actor_id: int):
        self.id = int(actor_id)
        self.autopilot_enabled = None
        self.last_control = None

    def set_simulate_physics(self, _enabled):
        return None

    def set_autopilot(self, enabled, *_args):
        self.autopilot_enabled = bool(enabled)

    def apply_control(self, control):
        self.last_control = control


class _FakeWorld:
    def __init__(self, actors=None, environment_objects=None):
        self._actors = list(actors or [])
        self._environment_objects = list(environment_objects or [])

    def get_actors(self):
        return list(self._actors)

    def get_environment_objects(self, _label):
        return list(self._environment_objects)


class _FakeEnvironmentObject:
    def __init__(self, name: str, transform=None):
        self.name = str(name)
        self.transform = transform


class _FakeWaypoint:
    def __init__(self, x: float, y: float):
        self.transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=float(x), y=float(y), z=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )


class _FakeTrafficLightActor:
    def __init__(self, *, name: str, state: str, x: float, y: float, stop_waypoints=None):
        self.type_id = "traffic.traffic_light"
        self.attributes = {"name": str(name)}
        self._state = str(state)
        self._transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=float(x), y=float(y), z=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )
        self._stop_waypoints = list(stop_waypoints or [])

    def get_transform(self):
        return self._transform

    def get_state(self):
        return str(self._state)

    def get_stop_waypoints(self):
        return list(self._stop_waypoints)


class AllUsecaseScenarioTests(unittest.TestCase):
    def test_vehicle_spawn_attempts_prefer_exact_marker_position_over_waypoint_snap(self):
        marker_transform = _FakeCarla.Transform(
            _FakeCarla.Location(x=12.5, y=34.0, z=0.0),
            _FakeCarla.Rotation(yaw=5.0),
        )
        waypoint_transform = _FakeCarla.Transform(
            _FakeCarla.Location(x=13.0, y=35.0, z=0.4),
            _FakeCarla.Rotation(yaw=37.0),
        )

        attempts = all_usecase_scenario._spawn_attempts_for_marker(
            {
                "transform": marker_transform,
                "waypoint_transform": waypoint_transform,
            },
            _FakeCarla,
            z_offset_m=0.05,
        )

        self.assertGreater(len(attempts), 0)
        self.assertAlmostEqual(float(attempts[0].location.x), 12.5, places=3)
        self.assertAlmostEqual(float(attempts[0].location.y), 34.0, places=3)
        self.assertAlmostEqual(float(attempts[0].location.z), 0.45, places=3)
        self.assertAlmostEqual(float(attempts[0].rotation.yaw), 37.0, places=3)

    def test_vehicle_spawn_attempts_can_prefer_closest_waypoint_transform(self):
        marker_transform = _FakeCarla.Transform(
            _FakeCarla.Location(x=12.5, y=34.0, z=0.0),
            _FakeCarla.Rotation(yaw=5.0),
        )
        waypoint_transform = _FakeCarla.Transform(
            _FakeCarla.Location(x=13.0, y=35.0, z=0.4),
            _FakeCarla.Rotation(yaw=37.0),
        )

        attempts = all_usecase_scenario._spawn_attempts_for_marker(
            {
                "transform": marker_transform,
                "waypoint_transform": waypoint_transform,
            },
            _FakeCarla,
            z_offset_m=0.05,
            prefer_waypoint_transform=True,
        )

        self.assertGreater(len(attempts), 0)
        self.assertAlmostEqual(float(attempts[0].location.x), 13.0, places=3)
        self.assertAlmostEqual(float(attempts[0].location.y), 35.0, places=3)
        self.assertAlmostEqual(float(attempts[0].location.z), 0.45, places=3)
        self.assertAlmostEqual(float(attempts[0].rotation.yaw), 37.0, places=3)

    def test_find_markers_by_prefix_matches_vehicle_markers_like_hazard_lookup(self):
        world = _FakeWorld(
            environment_objects=[
                _FakeEnvironmentObject("foo_vehicle_aus_2_bar"),
                _FakeEnvironmentObject("vehicle_aus_1"),
                _FakeEnvironmentObject("vehicle_aus_1_duplicate_longer_name"),
                _FakeEnvironmentObject("something_else"),
            ],
        )

        markers = all_usecase_scenario._find_markers_by_prefix(
            world,
            _FakeCarla,
            "vehicle_aus_",
        )

        self.assertEqual(
            [str(getattr(marker, "name", "")) for marker in markers],
            ["vehicle_aus_1", "foo_vehicle_aus_2_bar"],
        )

    def test_initialize_runtime_spawns_vehicle_markers_immediately_at_closest_waypoint(self):
        world = _FakeWorld(
            environment_objects=[
                _FakeEnvironmentObject(
                    "hazard_aus_1",
                    transform=_FakeCarla.Transform(
                        _FakeCarla.Location(x=0.0, y=0.0, z=0.0),
                        _FakeCarla.Rotation(yaw=0.0),
                    ),
                ),
                _FakeEnvironmentObject(
                    "vehicle_aus_1",
                    transform=_FakeCarla.Transform(
                        _FakeCarla.Location(x=10.0, y=0.0, z=0.0),
                        _FakeCarla.Rotation(yaw=0.0),
                    ),
                ),
                _FakeEnvironmentObject(
                    "foo_vehicle_aus_11_bar",
                    transform=_FakeCarla.Transform(
                        _FakeCarla.Location(x=20.0, y=0.0, z=0.0),
                        _FakeCarla.Rotation(yaw=0.0),
                    ),
                ),
            ],
        )
        spawn_requests = []

        def _fake_spawn_vehicle_at_marker(**kwargs):
            spawn_requests.append(
                {
                    "role_name": str(kwargs.get("role_name", "")),
                    "autopilot_enabled": bool(kwargs.get("autopilot_enabled", False)),
                    "prefer_waypoint_transform": bool(kwargs.get("prefer_waypoint_transform", False)),
                }
            )
            return types.SimpleNamespace(id=200 + len(spawn_requests))

        with unittest.mock.patch.object(
            all_usecase_scenario,
            "_spawn_vehicle_at_marker",
            side_effect=_fake_spawn_vehicle_at_marker,
        ):
            with tempfile.TemporaryDirectory() as tmp_dir:
                runtime_state = all_usecase_scenario.initialize_runtime(
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
        self.assertEqual(runtime_state.get("manual_vehicle_actor_ids", []), [202, 203])
        self.assertEqual(runtime_state.get("delayed_autopilot_vehicle_actor_ids", {}), {11: 203})
        self.assertEqual(
            spawn_requests,
            [
                {
                    "role_name": "hazard_aus_1",
                    "autopilot_enabled": False,
                    "prefer_waypoint_transform": False,
                },
                {
                    "role_name": "vehicle_aus_1",
                    "autopilot_enabled": True,
                    "prefer_waypoint_transform": True,
                },
                {
                    "role_name": "foo_vehicle_aus_11_bar",
                    "autopilot_enabled": False,
                    "prefer_waypoint_transform": True,
                },
            ],
        )

    def test_unknown_intersection_signal_keeps_existing_cp_message_state(self):
        marker = {
            "name": "intersection_aus_1",
            "index": 1,
            "position_xy": [10.0, 0.0],
            "waypoint_transform": None,
            "road_id": 1,
            "lane_id": 1,
        }
        existing_message = all_usecase_scenario._control_message_from_marker(
            prefix="all_usecase",
            message_type="traffic_light",
            state="red",
            marker=marker,
        )
        ego_transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 0.0,
                        "obstacles": [],
                        "lane_events": [],
                        "control": [existing_message],
                    },
                    message_file,
                )

            runtime_state = {
                "cp_message_path": message_path,
                "cp_schema_version": 1,
                "intersection_markers": [marker],
                "triggered_intersection_marker_names": {"intersection_aus_1"},
                "crossed_intersection_marker_names": set(),
                "intersection_signal_states": {"intersection_aus_1": "red"},
                "control_trigger_distance_m": 20.0,
                "traffic_light_stop_waypoint_match_distance_m": 12.0,
                "traffic_light_actor_position_match_distance_m": 40.0,
                "message_prefix": "all_usecase",
            }

            original_signal_helper = all_usecase_scenario._intersection_signal_state_for_marker
            try:
                all_usecase_scenario._intersection_signal_state_for_marker = lambda **_: ("unknown", "")
                all_usecase_scenario._maybe_register_intersection_control(
                    runtime_state=runtime_state,
                    world=object(),
                    ego_transform=ego_transform,
                    active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                    sim_time_s=1.0,
                )
            finally:
                all_usecase_scenario._intersection_signal_state_for_marker = original_signal_helper

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

        self.assertEqual(len(payload.get("control", [])), 1)
        self.assertEqual(str(payload.get("control", [])[0].get("state", "")), "red")
        self.assertEqual(str(payload.get("control", [])[0].get("type", "")), "intersection")
        self.assertTrue(bool(payload.get("control", [])[0].get("stop", False)))
        self.assertEqual(
            str(runtime_state.get("intersection_signal_states", {}).get("intersection_aus_1", "")),
            "red",
        )

    def test_intersection_control_is_not_removed_until_ego_passes_stop_point(self):
        marker = {
            "name": "intersection_aus_2",
            "index": 2,
            "position_xy": [5.0, 0.0],
            "waypoint_transform": _FakeCarla.Transform(
                _FakeCarla.Location(x=10.0, y=0.0, z=0.0),
                _FakeCarla.Rotation(yaw=0.0),
            ),
            "road_id": 1,
            "lane_id": 1,
        }
        existing_message = all_usecase_scenario._control_message_from_marker(
            prefix="all_usecase",
            message_type="traffic_light",
            state="red",
            marker=marker,
        )
        ego_transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=8.0, y=0.0, z=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 0.0,
                        "obstacles": [],
                        "lane_events": [],
                        "control": [existing_message],
                    },
                    message_file,
                )

            runtime_state = {
                "cp_message_path": message_path,
                "cp_schema_version": 1,
                "intersection_markers": [marker],
                "triggered_intersection_marker_names": {"intersection_aus_2"},
                "crossed_intersection_marker_names": set(),
                "intersection_signal_states": {"intersection_aus_2": "red"},
                "intersection_signal_actor_names": {"intersection_aus_2": "signal_2"},
                "control_trigger_distance_m": 20.0,
                "traffic_light_stop_waypoint_match_distance_m": 12.0,
                "traffic_light_actor_position_match_distance_m": 40.0,
                "message_prefix": "all_usecase",
            }

            original_signal_helper = all_usecase_scenario._intersection_signal_state_for_marker
            try:
                all_usecase_scenario._intersection_signal_state_for_marker = lambda **_: ("red", "signal_2")
                all_usecase_scenario._maybe_register_intersection_control(
                    runtime_state=runtime_state,
                    world=object(),
                    ego_transform=ego_transform,
                    active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                    sim_time_s=1.0,
                )
            finally:
                all_usecase_scenario._intersection_signal_state_for_marker = original_signal_helper

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

        self.assertEqual(len(payload.get("control", [])), 1)
        self.assertEqual(str(payload.get("control", [])[0].get("state", "")), "red")
        self.assertEqual(str(payload.get("control", [])[0].get("type", "")), "intersection")
        self.assertTrue(bool(payload.get("control", [])[0].get("stop", False)))
        self.assertEqual(runtime_state.get("crossed_intersection_marker_names", set()), set())

    def test_intersection_control_updates_to_new_closest_signal_state(self):
        marker = {
            "name": "intersection_aus_3",
            "index": 3,
            "position_xy": [10.0, 0.0],
            "waypoint_transform": _FakeCarla.Transform(
                _FakeCarla.Location(x=10.0, y=0.0, z=0.0),
                _FakeCarla.Rotation(yaw=0.0),
            ),
            "road_id": 1,
            "lane_id": 1,
        }
        red_signal = _FakeTrafficLightActor(
            name="red_signal",
            state="Red",
            x=10.0,
            y=0.0,
            stop_waypoints=[_FakeWaypoint(10.0, 0.0)],
        )
        green_signal = _FakeTrafficLightActor(
            name="green_signal",
            state="Green",
            x=12.0,
            y=0.0,
            stop_waypoints=[_FakeWaypoint(12.0, 0.0)],
        )
        ego_transform = types.SimpleNamespace(
            location=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=types.SimpleNamespace(yaw=0.0),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            runtime_state = {
                "cp_message_path": message_path,
                "cp_schema_version": 1,
                "intersection_markers": [marker],
                "triggered_intersection_marker_names": {"intersection_aus_3"},
                "crossed_intersection_marker_names": set(),
                "intersection_signal_states": {},
                "intersection_signal_actor_names": {},
                "control_trigger_distance_m": 20.0,
                "traffic_light_stop_waypoint_match_distance_m": 12.0,
                "traffic_light_actor_position_match_distance_m": 40.0,
                "message_prefix": "all_usecase",
            }

            world = _FakeWorld(actors=[red_signal, green_signal])
            all_usecase_scenario._maybe_register_intersection_control(
                runtime_state=runtime_state,
                world=world,
                ego_transform=ego_transform,
                active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                sim_time_s=1.0,
            )

            green_signal._stop_waypoints = [_FakeWaypoint(10.1, 0.0)]
            red_signal._stop_waypoints = [_FakeWaypoint(15.0, 0.0)]
            all_usecase_scenario._maybe_register_intersection_control(
                runtime_state=runtime_state,
                world=world,
                ego_transform=ego_transform,
                active_global_route_points=[[0.0, 0.0], [20.0, 0.0]],
                sim_time_s=2.0,
            )

            with open(message_path, "r", encoding="utf-8") as message_file:
                payload = json.load(message_file)

        self.assertEqual(len(payload.get("control", [])), 1)
        self.assertEqual(str(payload["control"][0]["state"]), "green")
        self.assertFalse(bool(payload["control"][0]["stop"]))
        self.assertEqual(str(payload["control"][0]["signal_actor_name"]), "green_signal")
        self.assertEqual(
            str(runtime_state.get("intersection_signal_actor_names", {}).get("intersection_aus_3", "")),
            "green_signal",
        )

    def test_manual_vehicle_spawn_uses_immediate_and_delayed_autopilot_groups(self):
        runtime_state = {
            "manual_vehicles_spawned": False,
            "vehicle_markers": [
                {"name": "vehicle_aus_1", "index": 1},
                {"name": "vehicle_aus_3", "index": 3},
                {"name": "vehicle_aus_5", "index": 5},
                {"name": "vehicle_aus_10", "index": 10},
                {"name": "vehicle_aus_11", "index": 11},
            ],
            "manual_vehicle_blueprint": "vehicle.tesla.model3",
            "manual_vehicle_color_rgb": "90,90,90",
            "vehicle_spawn_z_offset_m": 0.05,
            "autopilot_vehicle_indices": {1, 2, 3},
            "delayed_autopilot_vehicle_indices": {11},
            "traffic_manager_port": 8000,
        }
        spawned_modes = {}

        def _fake_spawn_vehicle_at_marker(**kwargs):
            marker = dict(kwargs.get("marker", {}) or {})
            marker_index = int(marker.get("index", -1))
            spawned_modes[marker_index] = (
                bool(kwargs.get("autopilot_enabled", False)),
                bool(kwargs.get("prefer_waypoint_transform", False)),
            )
            return types.SimpleNamespace(id=100 + int(marker_index))

        with unittest.mock.patch.object(
            all_usecase_scenario,
            "_spawn_vehicle_at_marker",
            side_effect=_fake_spawn_vehicle_at_marker,
        ):
            all_usecase_scenario._maybe_spawn_manual_vehicles(
                runtime_state=runtime_state,
                world=object(),
                carla=object(),
                ego_xy=[0.0, 0.0],
            )

        self.assertTrue(bool(runtime_state["manual_vehicles_spawned"]))
        self.assertEqual(
            spawned_modes,
            {
                1: (True, True),
                3: (True, True),
                5: (False, True),
                10: (False, True),
                11: (False, True),
            },
        )
        self.assertEqual(runtime_state["delayed_autopilot_vehicle_actor_ids"], {11: 111})

    def test_delayed_autopilot_vehicle_activates_when_ego_is_near_indexed_stopping_point(self):
        vehicle_11 = _FakeVehicleActor(actor_id=211)
        vehicle_12 = _FakeVehicleActor(actor_id=212)
        runtime_state = {
            "delayed_autopilot_vehicle_actor_ids": {11: 211, 12: 212},
            "delayed_autopilot_vehicle_activated_indices": set(),
            "delayed_autopilot_trigger_distance_m": 40.0,
            "traffic_manager_port": 8000,
            "intersection_markers": [
                {"name": "intersection_aus_11", "index": 11, "position_xy": [30.0, 0.0], "waypoint_transform": None},
                {"name": "intersection_aus_12", "index": 12, "position_xy": [80.0, 0.0], "waypoint_transform": None},
            ],
            "stop_markers": [],
        }

        all_usecase_scenario._maybe_activate_delayed_autopilot_vehicles(
            runtime_state=runtime_state,
            world=_FakeWorld(actors=[vehicle_11, vehicle_12]),
            carla=_FakeCarla,
            ego_xy=[0.0, 0.0],
        )

        self.assertTrue(bool(vehicle_11.autopilot_enabled))
        self.assertIsNotNone(vehicle_11.last_control)
        self.assertIsNone(vehicle_12.autopilot_enabled)
        self.assertEqual(runtime_state["delayed_autopilot_vehicle_actor_ids"], {12: 212})
        self.assertEqual(runtime_state["delayed_autopilot_vehicle_activated_indices"], {11})


if __name__ == "__main__":
    unittest.main()
