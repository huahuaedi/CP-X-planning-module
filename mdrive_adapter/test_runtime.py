import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".runtime/deps"))
sys.path.insert(0, str(ROOT / "opencda/planning_module"))
from mdrive_adapter.runtime import ExternalContext, collect_gt, merge_config


class RuntimeTests(unittest.TestCase):
    def test_control_time_grid_not_simulation_tick_count(self):
        context = ExternalContext(None, [(None, None), (None, None)], {})
        context.plan_time_s = 12.0
        self.assertEqual([context.control_index(t, .1) for t in (12., 12.05, 12.1, 12.15, 12.2)],
                         [0, 0, 1, 1, 2])
        context.plan_time_s = 12.25
        self.assertEqual(context.control_index(12.25, .1), 0)

    def test_ego_filter_keeps_other_egos_and_independent_snapshots(self):
        def actor(identifier, x):
            return types.SimpleNamespace(id=identifier, get_location=lambda: types.SimpleNamespace(x=x, y=0))
        a = ExternalContext(actor(1, 0), [(None, None), (None, None)], {"gt_range_m": 20})
        b = ExternalContext(actor(2, 10), [(None, None), (None, None)], {"gt_range_m": 20})
        a.snapshots = b.snapshots = [{"vehicle_id": str(i), "x": x, "y": 0}
                                   for i, x in ((1, 0), (2, 10), (3, 100))]
        self.assertEqual([x["vehicle_id"] for x in a.obstacles()], ["2"])
        self.assertEqual([x["vehicle_id"] for x in b.obstacles()], ["1"])
        a.obstacles()[0]["x"] = -999
        self.assertEqual(a.snapshots[1]["x"], 10)

    def test_gt_includes_walkers_and_uses_unique_actor_ids(self):
        def actor(identifier, kind):
            transform = types.SimpleNamespace(rotation=types.SimpleNamespace(yaw=0),
                transform=lambda point: point)
            return types.SimpleNamespace(id=identifier, type_id=kind, is_alive=True,
                attributes={"role_name": "autopilot"}, get_transform=lambda: transform,
                get_velocity=lambda: types.SimpleNamespace(x=0, y=2),
                bounding_box=types.SimpleNamespace(location=types.SimpleNamespace(x=0,y=0,z=1),
                    rotation=types.SimpleNamespace(yaw=0), extent=types.SimpleNamespace(x=1,y=1,z=1)))
        world = types.SimpleNamespace(get_actors=lambda: [actor(1,"vehicle.a"), actor(2,"vehicle.a"),
                                                        actor(3,"walker.pedestrian.0001")])
        objects = collect_gt(world)
        self.assertEqual([o["vehicle_id"] for o in objects], ["1", "2", "3"])
        self.assertAlmostEqual(objects[-1]["psi"], math.pi / 2)

    def test_config_overrides_do_not_mutate_other_egos(self):
        base = {"a": {"b": 2, "c": [1]}}
        result = merge_config(base, {"a": {"b": 3}})
        result["a"]["c"].append(2)
        self.assertEqual(base, {"a": {"b": 2, "c": [1]}})

    def test_benchmark_route_keeps_intermediate_turn_and_order(self):
        route = [(types.SimpleNamespace(location=types.SimpleNamespace(x=x,y=y,z=0)),
                  types.SimpleNamespace(name=option))
                 for x, y, option in ((0,0,"LANEFOLLOW"),(10,0,"LEFT"),(10,10,"LANEFOLLOW"))]
        context = ExternalContext(None, route, {"route_end_clearance_m": 0})
        planner = mock.Mock()
        world_map = mock.Mock()
        world_map.get_waypoint.return_value = types.SimpleNamespace(road_id=12)
        carla = types.SimpleNamespace(LaneType=types.SimpleNamespace(Driving=1))
        with mock.patch("utility.canonical_lane_id_for_waypoint", return_value=2):
            summary = context.install_route(planner, world_map, carla)
        self.assertEqual(summary.route_waypoints, [[0,0],[10,0],[10,10]])
        self.assertEqual(summary.road_options, ["LANEFOLLOW","LEFT","LANEFOLLOW"])
        self.assertEqual(summary.distance_to_destination_m, 20)
        planner.store_initial_route.assert_called_once_with(summary, summary.road_options, [2,2,2])

    def test_finish_clearance_moves_only_stop_target(self):
        def transform(x):
            return types.SimpleNamespace(location=types.SimpleNamespace(x=x,y=0,z=0),
                                         rotation=types.SimpleNamespace(yaw=0))
        finish = transform(32)
        continuation = types.SimpleNamespace(transform=transform(35))
        waypoint = types.SimpleNamespace(road_id=12, next=lambda distance: [continuation])
        route = [(transform(0), "LANEFOLLOW"), (finish, "LANEFOLLOW")]
        context = ExternalContext(None, route, {"route_end_clearance_m": 3})
        world_map = mock.Mock()
        world_map.get_waypoint.return_value = waypoint
        carla = types.SimpleNamespace(LaneType=types.SimpleNamespace(Driving=1))
        with mock.patch("utility.canonical_lane_id_for_waypoint", return_value=1):
            summary = context.install_route(mock.Mock(), world_map, carla)
        self.assertEqual(summary.route_waypoints[-1], [32,0])
        self.assertIs(context.route[-1][0], finish)
        self.assertEqual(context.destination_transform.location.x, 35)

    def test_standalone_driver_applies_yielded_controls_and_closes(self):
        import planning_runner
        vehicle = mock.Mock()
        closed = []
        def steps():
            try:
                yield {"vehicle": vehicle, "control": "first"}
                yield {"vehicle": vehicle, "control": "second"}
            finally:
                closed.append(True)
        with mock.patch.object(planning_runner, "iter_planning_steps", return_value=steps()):
            self.assertEqual(planning_runner.run_loaded_world(None, None, {}, None), 0)
        self.assertEqual(vehicle.apply_control.call_args_list, [mock.call("first"), mock.call("second")])
        self.assertEqual(closed, [True])

    def test_speed_feedback_can_start_then_brake_without_changing_steering(self):
        actor = types.SimpleNamespace(id=1, get_velocity=lambda: types.SimpleNamespace(x=0, y=0))
        context = ExternalContext(actor, [(None, None), (None, None)], {})
        command = types.SimpleNamespace(throttle=.13, brake=0, steer=.12)
        path = [[i*.1, 0, .1*(i+1), 0] for i in range(20)]
        command = context.track_control(command, path, 0, .1, 0)
        self.assertGreater(command.throttle, .3)
        self.assertEqual(command.brake, 0)
        self.assertEqual(command.steer, .12)
        actor.get_velocity = lambda: types.SimpleNamespace(x=3, y=0)
        command = context.track_control(command, [[0,0,0,0]] * 20, 0, .1, .05)
        self.assertEqual(command.throttle, 0)
        self.assertGreater(command.brake, 0)

    def test_stopped_car_can_approach_distant_stop_goal(self):
        import yaml
        from MPC.mpc import MPC
        from planning_runner import _apply_stop_target_speed_cap
        base = yaml.safe_load((ROOT / "opencda/planning_module/MPC/mpc.yaml").read_text())["mpc"]
        adapter = yaml.safe_load((ROOT / "mdrive_adapter/cpx_gt.yaml").read_text())
        config = merge_config(base, adapter["mpc"])
        config["constraints"] = adapter["scenario"]["constraints"]
        destination, cap = _apply_stop_target_speed_cap(
            temporary_destination_state=[32,0,0,0,1], ego_state=[0,0,0,0],
            stop_target_distance_m=32, original_max_velocity_mps=7,
            braking_deceleration_mps2=10, stop_buffer_m=0)
        self.assertGreater(cap, 0)
        planner = MPC(config, {"lane_count": 1, "lane_width_m": 3.5})
        trajectory = planner.plan_trajectory([0,0,0,0], destination, [], 0, 0, stop_goal_active=True)
        self.assertEqual(planner.get_runtime_status()["solver_status"], "solved")
        self.assertGreater(max(state[2] for state in trajectory), .1)


if __name__ == "__main__":
    unittest.main()
