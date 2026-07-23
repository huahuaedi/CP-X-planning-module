import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

PLANNING_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLANNING_MODULE_ROOT not in sys.path:
    sys.path.insert(0, PLANNING_MODULE_ROOT)

from utility.coordinate_obstacle_spawner import (
    _step_pedestrian_along_waypoints,
    _step_vehicle_along_waypoints,
    maybe_replan_global_route,
)


def _ego_transform(x_m: float, y_m: float):
    return SimpleNamespace(location=SimpleNamespace(x=float(x_m), y=float(y_m)))


class _FakeWalkerControl:
    def __init__(self, *, direction, speed):
        self.direction = direction
        self.speed = float(speed)


class _FakeVector3D:
    def __init__(self, *, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _FakeVehicleControl:
    def __init__(self, *, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.throttle = float(throttle)
        self.steer = float(steer)
        self.brake = float(brake)
        self.hand_brake = bool(hand_brake)
        self.reverse = bool(reverse)


class _FakeCarlaModule:
    WalkerControl = _FakeWalkerControl
    VehicleControl = _FakeVehicleControl
    Vector3D = _FakeVector3D


class _FakeWalkerActor:
    def __init__(self, *, x_m: float, y_m: float, role_name: str = ""):
        self._transform = SimpleNamespace(location=SimpleNamespace(x=float(x_m), y=float(y_m)))
        self.attributes = {"role_name": role_name}
        self.applied_controls = []

    def get_transform(self):
        return self._transform

    def apply_control(self, control):
        self.applied_controls.append(control)

    def move_to(self, x_m: float, y_m: float):
        self._transform = SimpleNamespace(location=SimpleNamespace(x=float(x_m), y=float(y_m)))


class _FakeVehicleActor:
    def __init__(self, *, x_m: float, y_m: float, yaw_deg: float = 0.0, speed_mps: float = 0.0, role_name: str = ""):
        self._transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x_m), y=float(y_m)),
            rotation=SimpleNamespace(yaw=float(yaw_deg)),
        )
        self._velocity = SimpleNamespace(x=float(speed_mps), y=0.0, z=0.0)
        self.attributes = {"role_name": role_name}
        self.applied_controls = []

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    def apply_control(self, control):
        self.applied_controls.append(control)


class _FakeActorList(list):
    def filter(self, _pattern):
        return self


class MaybeReplanGlobalRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cp_message_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.message_path = os.path.join(self.tmp_dir, "cp_message.json")
        self.scenario_cfg = {
            "obstacles": {
                "cp_message_path": self.message_path,
                "cooperative_message_trigger_distance_m": 20.0,
                "static_actors": [
                    {"role_name": "entity_1", "x": 0.0, "y": 0.0},
                    {"role_name": "entity_2", "x": 100.0, "y": 0.0},
                ],
            }
        }

    def _lane_events(self):
        if not os.path.isfile(self.message_path):
            return []
        with open(self.message_path, "r", encoding="utf-8") as file:
            return json.load(file).get("lane_events", [])

    def test_no_message_written_when_ego_is_far_from_all_obstacles(self):
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=_ego_transform(500.0, 500.0),
        )

        self.assertEqual(self._lane_events(), [])
        self.assertEqual(state["coordinate_hazard_inserted_ids"], [])

    def test_hazard_message_inserted_once_ego_is_within_trigger_distance(self):
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=_ego_transform(5.0, 0.0),
        )

        lane_events = self._lane_events()
        self.assertEqual(len(lane_events), 1)
        self.assertEqual(lane_events[0]["id"], "entity_1")
        self.assertEqual(lane_events[0]["type"], "hazard")
        self.assertEqual(lane_events[0]["position"], [0.0, 0.0])
        self.assertEqual(state["coordinate_hazard_inserted_ids"], ["entity_1"])

    def test_hazard_message_not_duplicated_on_subsequent_calls(self):
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=_ego_transform(5.0, 0.0),
        )
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=state,
            ego_transform=_ego_transform(1.0, 0.0),
        )

        self.assertEqual(len(self._lane_events()), 1)
        self.assertEqual(state["coordinate_hazard_inserted_ids"], ["entity_1"])

    def test_each_obstacle_triggers_independently_by_distance(self):
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=_ego_transform(5.0, 0.0),
        )
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=state,
            ego_transform=_ego_transform(95.0, 0.0),
        )

        lane_events = self._lane_events()
        self.assertEqual({event["id"] for event in lane_events}, {"entity_1", "entity_2"})
        self.assertEqual(set(state["coordinate_hazard_inserted_ids"]), {"entity_1", "entity_2"})

    def test_no_ego_transform_is_a_safe_noop(self):
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=None,
        )

        self.assertEqual(self._lane_events(), [])
        self.assertEqual(state.get("coordinate_hazard_inserted_ids"), None)


class StepPedestrianAlongWaypointsTests(unittest.TestCase):
    def setUp(self):
        self.carla = _FakeCarlaModule()
        self.waypoints = [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]

    def test_walks_toward_current_target_when_far_away(self):
        actor = _FakeWalkerActor(x_m=0.0, y_m=0.0)

        next_index = _step_pedestrian_along_waypoints(
            carla=self.carla,
            actor=actor,
            waypoints=self.waypoints,
            current_index=0,
            arrival_threshold_m=1.0,
            speed_mps=1.2,
        )

        self.assertEqual(next_index, 0)
        self.assertEqual(len(actor.applied_controls), 1)
        control = actor.applied_controls[0]
        self.assertAlmostEqual(control.direction.x, 1.0)
        self.assertAlmostEqual(control.direction.y, 0.0)
        self.assertAlmostEqual(control.speed, 1.2)

    def test_advances_to_next_waypoint_on_arrival(self):
        actor = _FakeWalkerActor(x_m=9.5, y_m=0.0)

        next_index = _step_pedestrian_along_waypoints(
            carla=self.carla,
            actor=actor,
            waypoints=self.waypoints,
            current_index=0,
            arrival_threshold_m=1.0,
            speed_mps=1.2,
        )

        self.assertEqual(next_index, 1)
        control = actor.applied_controls[0]
        self.assertAlmostEqual(control.direction.x, 1.0)  # heading toward waypoint[1] = (20, 0)

    def test_stops_after_reaching_final_waypoint(self):
        actor = _FakeWalkerActor(x_m=29.5, y_m=0.0)

        next_index = _step_pedestrian_along_waypoints(
            carla=self.carla,
            actor=actor,
            waypoints=self.waypoints,
            current_index=2,
            arrival_threshold_m=1.0,
            speed_mps=1.2,
        )

        self.assertEqual(next_index, 2)
        control = actor.applied_controls[0]
        self.assertAlmostEqual(control.speed, 0.0)


class MaybeReplanGlobalRoutePedestrianIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cp_message_ped_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.carla = _FakeCarlaModule()
        self.actor = _FakeWalkerActor(x_m=0.0, y_m=0.0, role_name="ped_1")
        self.world = SimpleNamespace(get_actors=lambda: _FakeActorList([self.actor]))
        self.scenario_cfg = {
            "obstacles": {
                "cp_message_path": os.path.join(self.tmp_dir, "cp_message.json"),
                "pedestrians": [
                    {
                        "role_name": "ped_1",
                        "waypoints": [[10.0, 0.0], [20.0, 0.0]],
                        "speed_mps": 1.2,
                        "arrival_threshold_m": 1.0,
                    }
                ],
            }
        }

    def test_finds_and_steps_pedestrian_by_role_name(self):
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=None,
            world=self.world,
            carla=self.carla,
        )

        self.assertEqual(len(self.actor.applied_controls), 1)
        self.assertEqual(state["pedestrian_progress"]["ped_1"], 0)
        self.assertIs(state["pedestrian_actor_cache"]["ped_1"], self.actor)

    def test_actor_lookup_is_cached_across_calls(self):
        lookups = []
        original_get_actors = self.world.get_actors

        def _tracking_get_actors():
            lookups.append(True)
            return original_get_actors()

        self.world.get_actors = _tracking_get_actors

        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=None,
            world=self.world,
            carla=self.carla,
        )
        maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=state,
            ego_transform=None,
            world=self.world,
            carla=self.carla,
        )

        self.assertEqual(len(lookups), 1)


class StepVehicleAlongWaypointsTests(unittest.TestCase):
    def setUp(self):
        self.carla = _FakeCarlaModule()
        self.waypoints = [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]

    def test_steers_toward_target_when_off_heading(self):
        # Actor faces +90deg (north) but the target is due east (+x) of it.
        actor = _FakeVehicleActor(x_m=0.0, y_m=0.0, yaw_deg=90.0, speed_mps=0.0)

        next_index = _step_vehicle_along_waypoints(
            carla=self.carla,
            actor=actor,
            waypoints=self.waypoints,
            current_index=0,
            arrival_threshold_m=3.0,
            target_speed_mps=8.0,
        )

        self.assertEqual(next_index, 0)
        control = actor.applied_controls[0]
        self.assertNotEqual(control.steer, 0.0)
        self.assertGreater(control.throttle, 0.0)

    def test_advances_to_next_waypoint_on_arrival(self):
        actor = _FakeVehicleActor(x_m=9.0, y_m=0.0, yaw_deg=0.0, speed_mps=8.0)

        next_index = _step_vehicle_along_waypoints(
            carla=self.carla,
            actor=actor,
            waypoints=self.waypoints,
            current_index=0,
            arrival_threshold_m=3.0,
            target_speed_mps=8.0,
        )

        self.assertEqual(next_index, 1)

    def test_brakes_after_reaching_final_waypoint(self):
        actor = _FakeVehicleActor(x_m=29.0, y_m=0.0, yaw_deg=0.0, speed_mps=8.0)

        next_index = _step_vehicle_along_waypoints(
            carla=self.carla,
            actor=actor,
            waypoints=self.waypoints,
            current_index=2,
            arrival_threshold_m=3.0,
            target_speed_mps=8.0,
        )

        self.assertEqual(next_index, 2)
        control = actor.applied_controls[0]
        self.assertAlmostEqual(control.throttle, 0.0)
        self.assertAlmostEqual(control.brake, 1.0)

    def test_coasts_once_target_speed_is_reached(self):
        actor = _FakeVehicleActor(x_m=0.0, y_m=0.0, yaw_deg=0.0, speed_mps=8.0)

        _step_vehicle_along_waypoints(
            carla=self.carla,
            actor=actor,
            waypoints=self.waypoints,
            current_index=0,
            arrival_threshold_m=3.0,
            target_speed_mps=8.0,
        )

        control = actor.applied_controls[0]
        self.assertAlmostEqual(control.throttle, 0.0)


class MaybeReplanGlobalRouteNpcIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="cp_message_npc_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.carla = _FakeCarlaModule()
        self.actor = _FakeVehicleActor(x_m=0.0, y_m=0.0, yaw_deg=0.0, role_name="npc_1")
        self.world = SimpleNamespace(get_actors=lambda: _FakeActorList([self.actor]))
        self.scenario_cfg = {
            "obstacles": {
                "cp_message_path": os.path.join(self.tmp_dir, "cp_message.json"),
                "npc_vehicles": [
                    {
                        "role_name": "npc_1",
                        "follow_mode": "literal",
                        "waypoints": [[10.0, 0.0], [20.0, 0.0]],
                        "speed_mps": 8.0,
                        "arrival_threshold_m": 3.0,
                    }
                ],
            }
        }

    def test_literal_follow_mode_drives_the_npc_vehicle(self):
        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=None,
            world=self.world,
            carla=self.carla,
        )

        self.assertEqual(len(self.actor.applied_controls), 1)
        self.assertEqual(state["npc_vehicle_progress"]["npc_1"], 0)

    def test_autopilot_follow_mode_is_not_driven_by_this_hook(self):
        self.scenario_cfg["obstacles"]["npc_vehicles"][0]["follow_mode"] = "autopilot"

        _route, _points, state = maybe_replan_global_route(
            scenario_cfg=self.scenario_cfg,
            runtime_state=None,
            ego_transform=None,
            world=self.world,
            carla=self.carla,
        )

        self.assertEqual(len(self.actor.applied_controls), 0)
        self.assertEqual(state["npc_vehicle_progress"], {})


if __name__ == "__main__":
    unittest.main()
