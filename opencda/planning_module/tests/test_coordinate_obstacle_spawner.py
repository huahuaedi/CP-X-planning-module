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

from utility.coordinate_obstacle_spawner import maybe_replan_global_route


def _ego_transform(x_m: float, y_m: float):
    return SimpleNamespace(location=SimpleNamespace(x=float(x_m), y=float(y_m)))


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
        self.assertEqual(state, {})


if __name__ == "__main__":
    unittest.main()
