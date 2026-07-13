import json
import os
import tempfile
import unittest

from behavior_planner.reroute import (
    lane_closure_messages,
    load_cp_messages,
    remove_cp_messages_by_id,
    reroute_from_lane_closure_messages,
    write_cp_messages,
)


class _RejectingPlanner:
    active_route = None

    def get_waypoint(self, position):
        raise AssertionError("get_waypoint must not run without a closure position")


class BehaviorRerouteTests(unittest.TestCase):
    def test_lane_closure_messages_filters_non_closures_and_missing_ids(self):
        filtered = lane_closure_messages(
            [
                {"id": "keep", "type": "lane_closure", "position": [1.0, 2.0]},
                {"id": "skip", "type": "speed_limit", "position": [1.0, 2.0]},
                {"type": "lane_closure", "position": [1.0, 2.0]},
            ]
        )

        self.assertEqual([message["id"] for message in filtered], ["keep"])

    def test_remove_cp_messages_keeps_unhandled_lane_closures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "cp.json")
            write_cp_messages(
                [
                    {"id": "handled", "type": "lane_closure", "position": [1.0, 2.0]},
                    {"id": "pending", "type": "lane_closure", "position": [3.0, 4.0]},
                ],
                message_path=path,
            )

            remaining = remove_cp_messages_by_id(["handled"], message_path=path)

            self.assertEqual([message["id"] for message in remaining], ["pending"])
            self.assertEqual([message["id"] for message in load_cp_messages(path)], ["pending"])

    def test_positionless_reroute_message_is_rejected_and_unhandled(self):
        result = reroute_from_lane_closure_messages(
            messages=[{"id": "bad", "type": "lane_closure", "road_id": 1, "lane_id": 2}],
            global_planner=_RejectingPlanner(),
            ego_position=[0.0, 0.0],
            goal_position=[10.0, 0.0],
        )

        self.assertTrue(result["reroute_failed"])
        self.assertEqual(result["handled_message_ids"], [])
        self.assertIn("closure position", result["debug_reason"].lower())

    def test_write_cp_messages_uses_lane_event_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "cp.json")
            write_cp_messages(
                [{"id": "closure", "type": "lane_closure", "position": [1.0, 2.0]}],
                message_path=path,
            )
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["lane_events"][0]["id"], "closure")


if __name__ == "__main__":
    unittest.main()
