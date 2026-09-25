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

    def test_benchmark_route_keeps_turns_and_only_appends_stop_tail(self):
        from mdrive_adapter.runtime import install_route
        def transform(x, y, yaw):
            return types.SimpleNamespace(location=types.SimpleNamespace(x=x, y=y, z=1),
                                         rotation=types.SimpleNamespace(yaw=yaw))
        route = [(transform(0, 0, 0), "LANEFOLLOW"),
                 (transform(10, 0, 90), "LEFT"),
                 (transform(10, 10, 90), "LANEFOLLOW")]
        context = ExternalContext(None, route, {"route_end_clearance_m": 3})
        bridge = mock.Mock()
        install_route(bridge, context)
        points, options = bridge.route_manager.install_external_route.call_args[0]
        self.assertEqual(points[:3], [[0, 0, 1], [10, 0, 1], [10, 10, 1]])
        self.assertAlmostEqual(points[-1][0], 10)
        self.assertAlmostEqual(points[-1][1], 13)
        self.assertEqual(options, ["LANEFOLLOW", "LEFT", "LANEFOLLOW", "LANEFOLLOW"])
        self.assertEqual(len(context.route), 3)
        self.assertFalse(bridge.route_manager.install_external_route.call_args[1]["allow_replan"])

    def test_bridge_config_preserves_constraints_and_isolates_artifacts(self):
        import tempfile
        import yaml
        from mdrive_adapter.runtime import build_bridge_config
        config = {"mpc": {"horizon_s": 2.5},
                  "scenario": {"constraints": {"max_velocity_mps": 6}},
                  "bridge": {"max_mpc_obstacles": 17}}
        world_map = types.SimpleNamespace(to_opendrive=lambda: "<OpenDRIVE/>")
        with tempfile.TemporaryDirectory() as directory:
            a = build_bridge_config(config, Path(directory) / "ego0", world_map)
            b = build_bridge_config(config, Path(directory) / "ego1", world_map)
            self.assertNotEqual(a["cp_message_path"], b["cp_message_path"])
            self.assertNotEqual(a["global_planner_cache_root"], b["global_planner_cache_root"])
            payload = yaml.safe_load(Path(a["mpc_config_path"]).read_text())
            self.assertEqual(payload["mpc"]["horizon_s"], 2.5)
            self.assertEqual(payload["mpc"]["constraints"]["max_velocity_mps"], 6)
            self.assertEqual(a["target_speed_mps"], 6)
            self.assertEqual(a["max_mpc_obstacles"], 17)
            self.assertFalse(a["publish_cp_message"])

    def test_pipeline_exceptions_propagate_and_cleanup_runs(self):
        from mdrive_adapter.runtime import make_planner
        actor = mock.Mock()
        actor.get_velocity.return_value = types.SimpleNamespace(x=0, y=0)
        world = mock.Mock()
        with mock.patch("mdrive_adapter.runtime.build_bridge_config", return_value={}), \
             mock.patch("mdrive_adapter.runtime.install_route"), \
             mock.patch("opencda.planning_module.opencda_bridge.cpx_mpc_planner.CPXMPCPlannerBridge") as cls:
            bridge = cls.return_value
            bridge.execute_planning_pipeline.side_effect = ValueError("broken pipeline")
            context, steps = make_planner(0, actor, world, [(None,None)] * 2, {}, "/unused")
            with self.assertRaisesRegex(ValueError, "broken pipeline"):
                next(steps)
            bridge.destroy.assert_called_once_with()
            self.assertIsNone(context.bridge)

    def test_disabled_rerouting_does_not_block_benchmark_lane(self):
        from opencda.planning_module.opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge
        bridge = CPXMPCPlannerBridge.__new__(CPXMPCPlannerBridge)
        bridge.config = {"route_replan_enabled": False}
        bridge.global_planner = mock.Mock()
        result = bridge._attempt_static_obstacle_route_replan(
            ego_location=None, obstacle={"x": 1, "y": 2})
        self.assertEqual(result, (False, False, "route_replan_disabled"))
        bridge.global_planner.block_lane_at_position.assert_not_called()


class SummaryTests(unittest.TestCase):
    def test_missing_ego_is_not_a_successful_parallel_run(self):
        import json
        import tempfile
        from mdrive_adapter.summarize import summarize
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / 'results/ego_vehicle_0/results.json'
            result.parent.mkdir(parents=True)
            result.write_text(json.dumps({'_checkpoint': {'records': [
                {'status': 'Completed', 'scores': {}, 'infractions': {}}]}}))
            steps = root / 'results/image/cpx_gt/steps.jsonl'
            steps.parent.mkdir(parents=True)
            row = {'ego': 0, 'x': 0, 'y': 0, 'planning_ms': 1,
                   'status': {'solver_status': 'solved'}}
            steps.write_text((json.dumps(row) + '\n') * 2)
            self.assertTrue(summarize(root, expected_egos=1)['closed_loop_executed'])
            self.assertFalse(summarize(root, expected_egos=2)['closed_loop_executed'])


if __name__ == "__main__":
    unittest.main()
