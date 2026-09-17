import math
import os
import sys
import types
import unittest


PLANNING_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PLANNING_MODULE_ROOT not in sys.path:
    sys.path.insert(0, PLANNING_MODULE_ROOT)

if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.ModuleType("numpy")

from behavior_planner.trajectory_risk import lane_prediction_risk, obstacle_future_trajectory
from behavior_planner.planner import RuleBasedBehaviorPlanner


class TrajectoryRiskTests(unittest.TestCase):
    def test_target_lane_rear_vehicle_future_closing_is_risky(self):
        result = lane_prediction_risk(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 8.0, "psi": 0.0},
            obstacle_snapshots=[
                {
                    "vehicle_id": "rear_fast",
                    "x": -12.0,
                    "y": 3.5,
                    "v": 18.0,
                    "psi": 0.0,
                    "predicted_trajectory": [
                        {"x": -12.0, "y": 3.5, "t": 0.0},
                        {"x": 6.0, "y": 3.5, "t": 1.0},
                        {"x": 24.0, "y": 3.5, "t": 2.0},
                    ],
                }
            ],
            lane_assignments={"rear_fast": 2},
            target_lane_id=2,
            horizon_s=3.0,
            min_rear_gap_m=10.0,
            min_ttc_s=2.5,
        )

        self.assertTrue(result["risk"])
        self.assertEqual(result["risky_obstacle_id"], "rear_fast")
        self.assertIn(result["reason"], {"rear_gap", "ttc", "front_gap"})

    def test_target_lane_without_future_conflict_is_not_risky(self):
        result = lane_prediction_risk(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 8.0, "psi": 0.0},
            obstacle_snapshots=[
                {
                    "vehicle_id": "front_far",
                    "x": 80.0,
                    "y": 3.5,
                    "v": 8.0,
                    "psi": 0.0,
                }
            ],
            lane_assignments={"front_far": 2},
            target_lane_id=2,
            horizon_s=3.0,
            min_front_gap_m=12.0,
            min_rear_gap_m=8.0,
            min_ttc_s=2.5,
        )

        self.assertFalse(result["risk"])
        self.assertEqual(result["target_lane_id"], 2)

    def test_behavior_planner_blocks_lane_change_when_prediction_risk_is_high(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        result = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={
                2: {
                    "risk": True,
                    "target_lane_id": 2,
                    "reason": "ttc",
                    "risky_obstacle_id": "rear_fast",
                }
            },
        )

        self.assertEqual(result["decision"], "lane_follow")
        self.assertEqual(result["target_lane_id"], 1)
        self.assertTrue(
            result["traffic_light_debug"]["lane_change_blocked_by_prediction"]
        )

    def test_behavior_planner_prepares_then_executes_lane_change(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
        )

        first = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": False}},
        )
        self.assertEqual(first["decision"], "lane_follow")
        self.assertEqual(first["lc_state"], "PREPARE_LANE_CHANGE_LEFT")
        self.assertTrue(first["traffic_light_debug"]["lane_change_preparing"])

        second = planner.update(
            lane_safety_scores={1: 0.0, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": False}},
        )
        self.assertEqual(second["decision"], "lane_change_left")
        self.assertEqual(second["lc_state"], "EXECUTE_LANE_CHANGE_LEFT")
        self.assertEqual(second["target_lane_id"], 2)

    def test_behavior_planner_aborts_execute_when_target_lane_degrades(self):
        planner = RuleBasedBehaviorPlanner(
            hysteresis_delta=0.05,
            lane_change_target_safety_threshold=0.10,
            lane_change_abort_safety_threshold=0.50,
        )

        planner.update(
            lane_safety_scores={1: 0.0, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": False}},
        )
        planner.update(
            lane_safety_scores={1: 0.0, 2: 0.95},
            ego_lane_id=1,
            selected_lane_id=1,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": False}},
        )

        aborted = planner.update(
            lane_safety_scores={1: 0.8, 2: 0.1},
            ego_lane_id=1,
            selected_lane_id=2,
            mode="NORMAL",
            route_optimal_lane_id=1,
            lane_prediction_risks={2: {"risk": True, "reason": "ttc"}},
        )

        self.assertEqual(aborted["decision"], "lane_follow")
        self.assertEqual(aborted["lc_state"], "ABORT_LANE_CHANGE")
        self.assertTrue(aborted["traffic_light_debug"]["lane_change_aborted"])


class LaneFollowingObstaclePredictionTests(unittest.TestCase):
    """Verify obstacle_future_trajectory follows a curved lane instead of
    straight-line extrapolation when a lane_step_fn is supplied, and falls
    back exactly to the previous straight-line behaviour otherwise."""

    @staticmethod
    def _circular_lane_step_fn(radius_m):
        # A road curving along a circle of `radius_m` centered at (0, radius_m),
        # starting at the origin heading east -- stands in for a real map's
        # position -> lane-centerline snapping (any (x, y) near the arc maps
        # back to an unambiguous arc-length s, the same way a real waypoint
        # lookup resolves position to a lane point).
        def _step(x_m, y_m, distance_m):
            s = radius_m * math.atan2(x_m, radius_m - y_m)
            new_s = s + distance_m
            return (
                radius_m * math.sin(new_s / radius_m),
                radius_m * (1.0 - math.cos(new_s / radius_m)),
                new_s / radius_m,
            )

        return _step

    def test_follows_the_curve_instead_of_a_straight_line(self):
        snapshot = {"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0, "a": 0.0}
        straight = obstacle_future_trajectory(
            snapshot, horizon_s=2.0, dt_s=0.5, model="constant_acceleration"
        )
        curved = obstacle_future_trajectory(
            snapshot,
            horizon_s=2.0,
            dt_s=0.5,
            model="constant_acceleration",
            lane_step_fn=self._circular_lane_step_fn(20.0),
        )

        self.assertEqual(len(curved), len(straight))
        # The straight-line model never leaves y=0; the curved one must.
        self.assertAlmostEqual(straight[-1]["y"], 0.0)
        self.assertGreater(curved[-1]["y"], 0.5)

    def test_none_lane_step_fn_result_falls_back_to_straight_line_exactly(self):
        snapshot = {"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0, "a": 0.0}
        straight = obstacle_future_trajectory(
            snapshot, horizon_s=2.0, dt_s=0.5, model="constant_acceleration"
        )
        fallback = obstacle_future_trajectory(
            snapshot,
            horizon_s=2.0,
            dt_s=0.5,
            model="constant_acceleration",
            lane_step_fn=lambda x, y, d: None,
        )

        self.assertEqual(fallback, straight)

    def test_pedestrians_ignore_lane_step_fn_and_stay_straight_line(self):
        # A pedestrian crossing a road is not constrained to the driving
        # lane network -- snapping its prediction onto the nearest lane
        # would predict it walking along the road instead of across it.
        pedestrian_snapshot = {
            "x": 0.0,
            "y": 0.0,
            "v": 1.4,
            "psi": 0.0,
            "a": 0.0,
            "object_type": "pedestrian",
        }
        straight = obstacle_future_trajectory(
            pedestrian_snapshot, horizon_s=2.0, dt_s=0.5, model="constant_acceleration"
        )
        with_lane_step_fn = obstacle_future_trajectory(
            pedestrian_snapshot,
            horizon_s=2.0,
            dt_s=0.5,
            model="constant_acceleration",
            lane_step_fn=self._circular_lane_step_fn(20.0),
        )

        self.assertEqual(with_lane_step_fn, straight)

    def test_omitting_lane_step_fn_keeps_previous_behaviour(self):
        snapshot = {"x": 0.0, "y": 0.0, "v": 5.0, "psi": 0.0, "a": 0.0}
        without_param = obstacle_future_trajectory(
            snapshot, horizon_s=2.0, dt_s=0.5, model="constant_acceleration"
        )
        with_none = obstacle_future_trajectory(
            snapshot,
            horizon_s=2.0,
            dt_s=0.5,
            model="constant_acceleration",
            lane_step_fn=None,
        )

        self.assertEqual(without_param, with_none)


class PredictionPointKinematicsTests(unittest.TestCase):
    """A supplied ``predicted_trajectory`` must keep or derive v/psi/a --
    see behavior_planner.trajectory_risk._trajectory_points /
    _fill_missing_kinematics. Without this, a real per-point speed a
    CP message or tracker attaches gets silently thrown away, and anything
    reading the normalized points back (pipeline.prediction.mpc_stage_trajectory
    -> MPC._get_object_state_at_stage) sees no v/psi at all instead of the
    real ones."""

    def test_supplied_v_and_psi_are_kept_as_is_not_recomputed(self):
        snapshot = {
            "x": 0.0, "y": 0.0, "v": 999.0, "psi": 999.0,
            "predicted_trajectory": [
                {"x": 10.0, "y": 0.0, "t": 1.0, "v": 3.0, "psi": 0.7, "a": 0.4},
            ],
        }
        points = obstacle_future_trajectory(snapshot, horizon_s=2.0, dt_s=1.0)
        self.assertEqual(points[0]["v"], 3.0)
        self.assertEqual(points[0]["psi"], 0.7)
        self.assertEqual(points[0]["a"], 0.4)

    def test_missing_v_and_psi_are_derived_from_the_trajectory_not_zero(self):
        # Straight line east at 4 m/s -- x advances by 4 every 1s, y constant.
        snapshot = {
            "x": 0.0, "y": 0.0, "v": 999.0, "psi": 999.0,
            "predicted_trajectory": [
                {"x": 4.0, "y": 0.0, "t": 1.0},
                {"x": 8.0, "y": 0.0, "t": 2.0},
                {"x": 12.0, "y": 0.0, "t": 3.0},
            ],
        }
        points = obstacle_future_trajectory(snapshot, horizon_s=3.0, dt_s=1.0)
        self.assertEqual(len(points), 3)
        for point in points:
            # None of these come from the (deliberately wrong) snapshot
            # v/psi=999 -- every point is derived from x/y/t, including the
            # first, which has no predecessor to backward-difference against.
            self.assertAlmostEqual(point["v"], 4.0)
            self.assertAlmostEqual(point["psi"], 0.0)

    def test_single_point_trajectory_uses_the_snapshots_current_state(self):
        snapshot = {
            "x": 0.0, "y": 0.0, "v": 7.0, "psi": 1.2,
            "predicted_trajectory": [{"x": 4.0, "y": 0.0, "t": 1.0}],
        }
        points = obstacle_future_trajectory(snapshot, horizon_s=2.0, dt_s=1.0)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["v"], 7.0)
        self.assertEqual(points[0]["psi"], 1.2)


if __name__ == "__main__":
    unittest.main()
