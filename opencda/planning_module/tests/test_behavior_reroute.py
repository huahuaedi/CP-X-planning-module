import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from behavior_planner.planner import RuleBasedBehaviorPlanner
from behavior_planner.reroute import (
    lane_closure_messages,
    load_cp_messages,
    remove_cp_messages_by_id,
    reroute_from_lane_closure_messages,
)


class _DummyLaneType:
    Driving = "Driving"


class _DummyCarla:
    LaneType = _DummyLaneType

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)


class _DummyWaypoint:
    def __init__(self, road_id, section_id, lane_id, x=0.0, y=0.0, s=None):
        self.road_id = int(road_id)
        self.section_id = int(section_id)
        self.lane_id = int(lane_id)
        self.lane_type = _DummyLaneType.Driving
        self.lane_width = 3.5
        self.is_junction = False
        self.s = float(x if s is None else s)
        self.transform = SimpleNamespace(
            location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )
        self._left_lane = None
        self._right_lane = None
        self._next_lanes = []
        self._previous_lanes = []

    def get_left_lane(self):
        return self._left_lane

    def get_right_lane(self):
        return self._right_lane

    def next(self, distance):
        del distance
        return list(self._next_lanes)

    def previous(self, distance):
        del distance
        return list(self._previous_lanes)

    def set_neighbors(
        self,
        *,
        left=None,
        right=None,
        next_lane=None,
        next_lanes=None,
        previous_lane=None,
        previous_lanes=None,
    ):
        self._left_lane = left
        self._right_lane = right
        if next_lanes is not None:
            self._next_lanes = list(next_lanes)
        elif next_lane is not None:
            self._next_lanes = [next_lane]
        else:
            self._next_lanes = []
        if previous_lanes is not None:
            self._previous_lanes = list(previous_lanes)
        elif previous_lane is not None:
            self._previous_lanes = [previous_lane]
        else:
            self._previous_lanes = []


class _DummyWorldMap:
    def __init__(self, default_waypoint, waypoint_by_xy=None):
        self._default_waypoint = default_waypoint
        self._waypoint_by_xy = dict(waypoint_by_xy or {})

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        del project_to_road
        del lane_type
        key = (
            round(float(getattr(location, "x", 0.0)), 3),
            round(float(getattr(location, "y", 0.0)), 3),
        )
        if key in self._waypoint_by_xy:
            return self._waypoint_by_xy[key]
        return self._default_waypoint


class _DummyReroutePlanner:
    def __init__(self):
        self.replaced_routes = []

    def replace_stored_route(self, *, summary, per_waypoint_options, per_waypoint_lane_ids):
        self.replaced_routes.append(
            {
                "summary": summary,
                "per_waypoint_options": list(per_waypoint_options),
                "per_waypoint_lane_ids": [int(lane_id) for lane_id in list(per_waypoint_lane_ids)],
            }
        )


def _build_two_lane_bypass_world(*, ego_on_closed_lane=False, negative_lane_ids=False):
    lane_1_id = -1 if negative_lane_ids else 1
    lane_2_id = -2 if negative_lane_ids else 2
    start_road_id = 12 if ego_on_closed_lane else 7

    start_lane_1 = _DummyWaypoint(start_road_id, 0, lane_1_id, x=0.0, y=0.0, s=0.0)
    start_lane_2 = _DummyWaypoint(start_road_id, 0, lane_2_id, x=0.0, y=3.5, s=0.0)
    blocked_lane = _DummyWaypoint(12 if not negative_lane_ids else 20, 0, lane_1_id, x=5.0, y=0.0, s=5.0)
    bypass_lane = _DummyWaypoint(12 if not negative_lane_ids else 20, 0, lane_2_id, x=5.0, y=3.5, s=5.0)
    goal_lane_1 = _DummyWaypoint(20 if not negative_lane_ids else 21, 0, lane_1_id, x=10.0, y=0.0, s=10.0)
    goal_lane_2 = _DummyWaypoint(20 if not negative_lane_ids else 21, 0, lane_2_id, x=10.0, y=3.5, s=10.0)

    start_lane_1.set_neighbors(left=start_lane_2, next_lane=blocked_lane)
    start_lane_2.set_neighbors(right=start_lane_1, next_lane=bypass_lane)
    blocked_lane.set_neighbors(left=bypass_lane, next_lane=goal_lane_1, previous_lane=start_lane_1)
    bypass_lane.set_neighbors(right=blocked_lane, next_lane=goal_lane_2, previous_lane=start_lane_2)
    goal_lane_1.set_neighbors(left=goal_lane_2, previous_lane=blocked_lane)
    goal_lane_2.set_neighbors(right=goal_lane_1, previous_lane=bypass_lane)

    waypoint_by_xy = {
        (0.0, 0.0): start_lane_1,
        (0.0, 3.5): start_lane_2,
        (5.0, 0.0): blocked_lane,
        (5.0, 3.5): bypass_lane,
        (10.0, 0.0): goal_lane_1,
        (10.0, 3.5): goal_lane_2,
    }
    return _DummyWorldMap(start_lane_1, waypoint_by_xy=waypoint_by_xy)


def _build_closed_two_lane_world():
    lane_1 = _DummyWaypoint(12, 0, 1, x=0.0, y=0.0, s=0.0)
    lane_2 = _DummyWaypoint(12, 0, 2, x=0.0, y=3.5, s=0.0)
    goal_lane_1 = _DummyWaypoint(12, 0, 1, x=10.0, y=0.0, s=10.0)
    goal_lane_2 = _DummyWaypoint(12, 0, 2, x=10.0, y=3.5, s=10.0)

    lane_1.set_neighbors(left=lane_2, next_lane=goal_lane_1)
    lane_2.set_neighbors(right=lane_1, next_lane=goal_lane_2)
    goal_lane_1.set_neighbors(left=goal_lane_2, previous_lane=lane_1)
    goal_lane_2.set_neighbors(right=goal_lane_1, previous_lane=lane_2)

    waypoint_by_xy = {
        (0.0, 0.0): lane_1,
        (0.0, 3.5): lane_2,
        (10.0, 0.0): goal_lane_1,
        (10.0, 3.5): goal_lane_2,
    }
    return _DummyWorldMap(lane_1, waypoint_by_xy=waypoint_by_xy)


def _build_single_lane_blocked_world():
    start_waypoint = _DummyWaypoint(12, 0, 1, x=0.0, y=0.0, s=0.0)
    blocked_waypoint = _DummyWaypoint(12, 0, 1, x=5.0, y=0.0, s=5.0)
    goal_waypoint = _DummyWaypoint(12, 0, 1, x=10.0, y=0.0, s=10.0)

    start_waypoint.set_neighbors(next_lane=blocked_waypoint)
    blocked_waypoint.set_neighbors(next_lane=goal_waypoint, previous_lane=start_waypoint)
    goal_waypoint.set_neighbors(previous_lane=blocked_waypoint)

    return _DummyWorldMap(
        start_waypoint,
        waypoint_by_xy={
            (0.0, 0.0): start_waypoint,
            (5.0, 0.0): blocked_waypoint,
            (10.0, 0.0): goal_waypoint,
        },
    )


def _build_extended_two_lane_bypass_world():
    lane_1_waypoints = [
        _DummyWaypoint(12, 0, 1, x=-10.0, y=0.0, s=-10.0),
        _DummyWaypoint(12, 0, 1, x=-5.0, y=0.0, s=-5.0),
        _DummyWaypoint(12, 0, 1, x=0.0, y=0.0, s=0.0),
        _DummyWaypoint(12, 0, 1, x=5.0, y=0.0, s=5.0),
        _DummyWaypoint(12, 0, 1, x=10.0, y=0.0, s=10.0),
        _DummyWaypoint(12, 0, 1, x=15.0, y=0.0, s=15.0),
    ]
    lane_2_waypoints = [
        _DummyWaypoint(12, 0, 2, x=-10.0, y=3.5, s=-10.0),
        _DummyWaypoint(12, 0, 2, x=-5.0, y=3.5, s=-5.0),
        _DummyWaypoint(12, 0, 2, x=0.0, y=3.5, s=0.0),
        _DummyWaypoint(12, 0, 2, x=5.0, y=3.5, s=5.0),
        _DummyWaypoint(12, 0, 2, x=10.0, y=3.5, s=10.0),
        _DummyWaypoint(12, 0, 2, x=15.0, y=3.5, s=15.0),
    ]

    for index, lane_1_waypoint in enumerate(lane_1_waypoints):
        lane_2_waypoint = lane_2_waypoints[index]
        next_lane_1 = lane_1_waypoints[index + 1] if index + 1 < len(lane_1_waypoints) else None
        next_lane_2 = lane_2_waypoints[index + 1] if index + 1 < len(lane_2_waypoints) else None
        previous_lane_1 = lane_1_waypoints[index - 1] if index > 0 else None
        previous_lane_2 = lane_2_waypoints[index - 1] if index > 0 else None

        lane_1_waypoint.set_neighbors(
            left=lane_2_waypoint,
            next_lane=next_lane_1,
            previous_lane=previous_lane_1,
        )
        lane_2_waypoint.set_neighbors(
            right=lane_1_waypoint,
            next_lane=next_lane_2,
            previous_lane=previous_lane_2,
        )

    waypoint_by_xy = {}
    for waypoint in list(lane_1_waypoints) + list(lane_2_waypoints):
        waypoint_by_xy[
            (
                round(float(waypoint.transform.location.x), 3),
                round(float(waypoint.transform.location.y), 3),
            )
        ] = waypoint

    return _DummyWorldMap(lane_1_waypoints[0], waypoint_by_xy=waypoint_by_xy)


class BehaviorRerouteTests(unittest.TestCase):
    def test_payload_lane_event_message_is_received_by_behavior_planner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 0.0,
                        "obstacles": [],
                        "lane_events": [
                            {
                                "id": "closure_payload_1",
                                "type": "lane_closure",
                                "position": [10.0, 20.0],
                                "road_id": 12,
                                "section_id": 0,
                                "lane_id": 1,
                                "carla_lane_id": 1,
                            }
                        ],
                        "control": [],
                    },
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
            )

            self.assertEqual(result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in result["reroute_messages"]],
                ["closure_payload_1"],
            )

    def test_payload_lane_event_message_triggers_reroute_in_intersection_mode_too(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 0.0,
                        "obstacles": [],
                        "lane_events": [
                            {
                                "id": "closure_payload_intersection",
                                "type": "lane_closure",
                                "position": [10.0, 20.0],
                                "road_id": 12,
                                "section_id": 0,
                                "lane_id": 1,
                                "carla_lane_id": 1,
                            }
                        ],
                        "control": [],
                    },
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="INTERSECTION",
                current_time_s=0.0,
            )

            self.assertEqual(result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in result["reroute_messages"]],
                ["closure_payload_intersection"],
            )

    def test_behavior_planner_suppresses_lane_closure_when_hazard_is_off_active_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 0.0,
                        "obstacles": [],
                        "lane_events": [
                            {
                                "id": "closure_off_route",
                                "type": "lane_closure",
                                "position": [20.0, 10.0],
                                "road_id": 12,
                                "section_id": 0,
                                "lane_id": 1,
                                "carla_lane_id": 1,
                            }
                        ],
                        "control": [],
                    },
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
                ego_position_xy=[0.0, 0.0],
                global_route_points=[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
            )

            self.assertEqual(result["decision"], "lane_follow")
            self.assertNotIn("reroute_messages", result)

    def test_behavior_planner_suppresses_lane_closure_when_hazard_is_behind_ego(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 0.0,
                        "obstacles": [],
                        "lane_events": [
                            {
                                "id": "closure_behind",
                                "type": "lane_closure",
                                "position": [2.0, 0.0],
                                "road_id": 12,
                                "section_id": 0,
                                "lane_id": 1,
                                "carla_lane_id": 1,
                            }
                        ],
                        "control": [],
                    },
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
                ego_position_xy=[10.0, 0.0],
                global_route_points=[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
            )

            self.assertEqual(result["decision"], "lane_follow")
            self.assertNotIn("reroute_messages", result)

    def test_lane_closure_messages_are_not_removed_when_reroute_decision_is_emitted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [
                        {"id": "closure_1", "type": "lane_closure", "position": [10.0, 20.0]},
                        {"id": "closure_2", "type": "lane_closure", "position": [11.0, 21.0]},
                        {"id": "note_1", "type": "speed_limit", "position": [12.0, 22.0]},
                    ],
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
            )

            self.assertEqual(result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in result["reroute_messages"]],
                ["closure_1", "closure_2"],
            )

            remaining_messages = load_cp_messages(message_path=message_path)
            self.assertEqual(
                [message["id"] for message in remaining_messages],
                ["closure_1", "closure_2", "note_1"],
            )

    def test_acknowledge_reroute_success_keeps_lane_closure_messages_in_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 0.0,
                        "obstacles": [],
                        "lane_events": [
                            {
                                "id": "closure_ack",
                                "type": "lane_closure",
                                "position": [10.0, 0.0],
                            }
                        ],
                        "control": [],
                    },
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
            )

            self.assertEqual(result["decision"], "reroute")
            planner.acknowledge_reroute_success(["closure_ack"])

            remaining_messages = load_cp_messages(message_path=message_path)
            self.assertEqual([message["id"] for message in remaining_messages], ["closure_ack"])

    def test_behavior_planner_rechecks_all_persisted_lane_events_on_later_polls(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    {
                        "schema_version": 1,
                        "sequence": 1,
                        "timestamp_s": 0.0,
                        "obstacles": [],
                        "lane_events": [
                            {
                                "id": "closure_repeat_1",
                                "type": "lane_closure",
                                "position": [10.0, 0.0],
                            },
                            {
                                "id": "closure_repeat_2",
                                "type": "lane_closure",
                                "position": [20.0, 0.0],
                            },
                        ],
                        "control": [],
                    },
                    message_file,
                )

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            first_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
                ego_position_xy=[0.0, 0.0],
                global_route_points=[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
            )
            self.assertEqual(first_result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in first_result["reroute_messages"]],
                ["closure_repeat_1", "closure_repeat_2"],
            )

            planner.acknowledge_reroute_success(["closure_repeat_1", "closure_repeat_2"])

            second_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=1.0,
                ego_position_xy=[0.0, 0.0],
                global_route_points=[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
            )
            self.assertEqual(second_result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in second_result["reroute_messages"]],
                ["closure_repeat_1", "closure_repeat_2"],
            )

    def test_lane_closure_messages_follow_poll_frequency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump([], message_file)

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            first_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
            )
            self.assertEqual(first_result["decision"], "lane_follow")

            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [{"id": "closure_late", "type": "lane_closure", "position": [3.0, 4.0]}],
                    message_file,
                )

            second_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.5,
            )
            self.assertEqual(second_result["decision"], "lane_follow")

            third_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=1.0,
            )
            self.assertEqual(third_result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in third_result["reroute_messages"]],
                ["closure_late"],
            )

    def test_lane_closure_messages_follow_wall_time_not_simulation_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump([], message_file)

            planner = RuleBasedBehaviorPlanner(
                cp_message_path=message_path,
                cooperative_message_check_frequency_hz=1.0,
            )
            first_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=0.0,
                wall_time_s=100.0,
            )
            self.assertEqual(first_result["decision"], "lane_follow")

            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [{"id": "closure_wall_time", "type": "lane_closure", "position": [3.0, 4.0]}],
                    message_file,
                )

            second_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=50.0,
                wall_time_s=100.5,
            )
            self.assertEqual(second_result["decision"], "lane_follow")

            third_result = planner.update(
                lane_safety_scores={1: 1.0},
                ego_lane_id=1,
                mode="NORMAL",
                current_time_s=100.0,
                wall_time_s=101.0,
            )
            self.assertEqual(third_result["decision"], "reroute")
            self.assertEqual(
                [message["id"] for message in third_result["reroute_messages"]],
                ["closure_wall_time"],
            )

    def test_remove_cp_messages_by_id_keeps_unhandled_messages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            message_path = os.path.join(tmp_dir, "cp_message.json")
            with open(message_path, "w", encoding="utf-8") as message_file:
                json.dump(
                    [
                        {"id": "m1", "type": "lane_closure", "position": [1.0, 2.0]},
                        {"id": "m2", "type": "lane_closure", "position": [3.0, 4.0]},
                    ],
                    message_file,
                )

            remaining = remove_cp_messages_by_id(["m1"], message_path=message_path)
            self.assertEqual([message["id"] for message in remaining], ["m2"])

    def test_lane_closure_messages_filters_non_closure_and_missing_id(self):
        filtered = lane_closure_messages(
            [
                {"id": "ok", "type": "lane_closure", "position": [1.0, 2.0]},
                {"id": "skip", "type": "speed_limit", "position": [1.0, 2.0]},
                {"type": "lane_closure", "position": [1.0, 2.0]},
            ]
        )
        self.assertEqual([message["id"] for message in filtered], ["ok"])

    def test_reroute_blocks_only_hazard_waypoint_and_updates_stored_route(self):
        planner = _DummyReroutePlanner()
        world_map = _build_two_lane_bypass_world()

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_lane_1",
                    "type": "lane_closure",
                    "position": [5.0, 0.0],
                    "road_id": 12,
                    "section_id": 0,
                    "lane_id": 1,
                    "carla_lane_id": 1,
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=10.0, y=0.0, z=0.0),
        )

        self.assertIsNotNone(result["route_summary"])
        self.assertTrue(bool(result["route_summary"].route_found))
        self.assertEqual(result["handled_message_ids"], ["closure_lane_1"])
        self.assertEqual(result["route_points"][0], [0.0, 0.0])
        self.assertIn([0.0, 3.5], result["route_points"])
        self.assertIn([5.0, 3.5], result["route_points"])
        self.assertEqual(
            result["resolved_messages"][0]["blocked_waypoint_key"],
            [12, 0, 1, 5.0],
        )
        self.assertEqual(len(planner.replaced_routes), 1)
        self.assertEqual(
            planner.replaced_routes[0]["summary"].route_waypoints,
            result["route_points"],
        )

    def test_reroute_normalizes_non_carla_lane_id_from_message_position(self):
        planner = _DummyReroutePlanner()
        world_map = _build_two_lane_bypass_world(negative_lane_ids=True)

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_negative_lane",
                    "type": "lane_closure",
                    "position": [5.0, 3.5],
                    "road_id": 999,
                    "section_id": 0,
                    "lane_id": 2,
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=10.0, y=0.0, z=0.0),
        )

        self.assertIsNotNone(result["route_summary"])
        self.assertTrue(bool(result["route_summary"].route_found))
        self.assertEqual(
            result["resolved_messages"][0]["blocked_waypoint_key"],
            [20, 0, -2, 5.0],
        )
        self.assertEqual(result["resolved_messages"][0]["road_id"], 20)
        self.assertEqual(result["resolved_messages"][0]["lane_id"], 2)
        self.assertEqual(result["resolved_messages"][0]["carla_lane_id"], -2)
        self.assertTrue(bool(result["resolved_messages"][0]["lane_id_matches_canonical"]))
        self.assertFalse(bool(result["resolved_messages"][0]["lane_ids_match_carla"]))
        self.assertTrue(bool(result["resolved_messages"][0]["normalized_from_position"]))

    def test_reroute_keeps_ego_origin_when_ego_starts_on_closed_lane(self):
        planner = _DummyReroutePlanner()
        world_map = _build_two_lane_bypass_world(ego_on_closed_lane=True)

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_start_lane",
                    "type": "lane_closure",
                    "position": [5.0, 0.0],
                    "road_id": 12,
                    "section_id": 0,
                    "lane_id": 1,
                    "carla_lane_id": 1,
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=10.0, y=0.0, z=0.0),
        )

        self.assertIsNotNone(result["route_summary"])
        self.assertEqual(result["route_points"][0], [0.0, 0.0])
        self.assertIn([0.0, 3.5], result["route_points"])

    def test_reroute_ignores_whole_road_flag_and_blocks_only_hazard_waypoint(self):
        planner = _DummyReroutePlanner()
        world_map = _build_two_lane_bypass_world()

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_whole_road",
                    "type": "lane_closure",
                    "position": [5.0, 0.0],
                    "road_id": 12,
                    "section_id": 0,
                    "block_entire_road": True,
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=10.0, y=0.0, z=0.0),
        )

        self.assertIsNotNone(result["route_summary"])
        self.assertIn([5.0, 3.5], result["route_points"])
        self.assertTrue(bool(result["resolved_messages"][0]["block_entire_road"]))
        self.assertEqual(
            result["resolved_messages"][0]["blocked_waypoint_key"],
            [12, 0, 1, 5.0],
        )

    def test_reroute_blocks_waypoints_for_all_hazard_messages(self):
        planner = _DummyReroutePlanner()
        world_map = _build_two_lane_bypass_world()

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_lane_1",
                    "type": "lane_closure",
                    "position": [5.0, 0.0],
                    "road_id": 12,
                    "section_id": 0,
                    "lane_id": 1,
                    "carla_lane_id": 1,
                },
                {
                    "id": "closure_lane_2",
                    "type": "lane_closure",
                    "position": [5.0, 3.5],
                    "road_id": 12,
                    "section_id": 0,
                    "lane_id": 2,
                    "carla_lane_id": 2,
                },
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=10.0, y=0.0, z=0.0),
        )

        self.assertIsNone(result["route_summary"])
        self.assertEqual(result["route_points"], [])
        self.assertEqual(result["handled_message_ids"], ["closure_lane_1", "closure_lane_2"])
        self.assertEqual(
            result["blocked_waypoint_keys"],
            [
                [12, 0, 1, 5.0],
                [20, 0, 1, 10.0],
                [7, 0, 1, 0.0],
                [12, 0, 2, 5.0],
                [20, 0, 2, 10.0],
                [7, 0, 2, 0.0],
            ],
        )

    def test_reroute_blocks_two_forward_and_two_behind_waypoints_around_hazard(self):
        planner = _DummyReroutePlanner()
        world_map = _build_extended_two_lane_bypass_world()

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_extended_lane_1",
                    "type": "lane_closure",
                    "position": [0.0, 0.0],
                    "road_id": 12,
                    "section_id": 0,
                    "lane_id": 1,
                    "carla_lane_id": 1,
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=-10.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=15.0, y=0.0, z=0.0),
        )

        self.assertIsNotNone(result["route_summary"])
        self.assertEqual(
            result["resolved_messages"][0]["blocked_waypoint_keys"],
            [
                [12, 0, 1, 0.0],
                [12, 0, 1, 5.0],
                [12, 0, 1, 10.0],
                [12, 0, 1, -5.0],
                [12, 0, 1, -10.0],
            ],
        )
        self.assertEqual(
            result["blocked_waypoint_keys"],
            [
                [12, 0, 1, 0.0],
                [12, 0, 1, 5.0],
                [12, 0, 1, 10.0],
                [12, 0, 1, -5.0],
                [12, 0, 1, -10.0],
            ],
        )
        self.assertIn([-10.0, 3.5], result["route_points"])
        self.assertIn([10.0, 3.5], result["route_points"])

    def test_reroute_fails_when_blocked_hazard_waypoint_is_the_only_path(self):
        planner = _DummyReroutePlanner()
        world_map = _build_single_lane_blocked_world()

        result = reroute_from_lane_closure_messages(
            messages=[
                {
                    "id": "closure_single_path",
                    "type": "lane_closure",
                    "position": [5.0, 0.0],
                    "road_id": 12,
                    "section_id": 0,
                    "lane_id": 1,
                    "carla_lane_id": 1,
                }
            ],
            world_map=world_map,
            carla=_DummyCarla,
            global_planner=planner,
            ego_transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            goal_location=SimpleNamespace(x=10.0, y=0.0, z=0.0),
        )

        self.assertIsNone(result["route_summary"])
        self.assertEqual(result["route_points"], [])
        self.assertEqual(result["handled_message_ids"], ["closure_single_path"])
        self.assertEqual(len(planner.replaced_routes), 0)


if __name__ == "__main__":
    unittest.main()
