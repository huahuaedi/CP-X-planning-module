import unittest

from utility.planning_context import (
    CPMessageContext,
    EgoPlanningState,
    MapLaneContext,
    PerceptionContext,
    PlannerInputFrame,
    PlanningContext,
    PredictionContext,
    RouteContext,
    TargetContext,
    TrafficControlContext,
)


class PlannerInputFrameTests(unittest.TestCase):
    def test_trace_fields_expose_upstream_input_counts(self):
        planning = PlanningContext(
            sim_time_s=1.0,
            ego=EgoPlanningState(x_m=0.0, y_m=0.0, speed_mps=2.0, heading_rad=0.0, lane_id=1),
            route=RouteContext(
                optimal_lane_id=2,
                next_macro_maneuver="left",
                remaining_points_count=10,
                route_found=True,
            ),
            traffic_control=TrafficControlContext(signal_state="green", from_cp=True),
            targets=TargetContext(),
        )
        frame = PlannerInputFrame(
            planning=planning,
            map_lane=MapLaneContext(
                lane_id=1,
                road_id=7,
                section_id=0,
                lane_count=2,
                allowed_lane_ids=[1, 2],
                in_junction=False,
                route_lane_id=2,
                route_maneuver="left",
            ),
            perception=PerceptionContext(
                dynamic_objects=[{"id": "veh_1"}],
                static_objects=[{"id": "cone_1"}],
                planning_objects=[{"id": "veh_1"}, {"id": "cone_1"}],
                source="cp",
            ),
            prediction=PredictionContext(
                lane_assignments={"veh_1": 1},
                lane_prediction_risks={1: {"risk": True}},
                obstacle_future_trajectories={"veh_1": [[0.0, 0.0, 2.0, 0.0]]},
                model="constant_acceleration",
                horizon_s=3.0,
                dt_s=0.1,
            ),
            cp_messages=CPMessageContext(
                message_path="cp_message.json",
                traffic_controls=[{"id": "tl_1", "control_id": "tl_1"}],
                selected_traffic_control={"id": "tl_1", "control_id": "tl_1"},
                lane_closures=[{"id": "closure_1"}],
                obstacles=[{"id": "hazard_1"}],
                generated_traffic_light_control={"id": "tl_1"},
            ),
        )

        fields = frame.trace_fields()

        self.assertEqual(fields["planning_context_signal_state"], "green")
        self.assertEqual(fields["planner_input_lane_id"], 1)
        self.assertEqual(fields["planner_input_perception_dynamic_count"], 1)
        self.assertEqual(fields["planner_input_perception_static_count"], 1)
        self.assertEqual(fields["planner_input_perception_planning_count"], 2)
        self.assertEqual(fields["planner_input_prediction_risky_lane_count"], 1)
        self.assertEqual(fields["planner_input_cp_traffic_control_count"], 1)
        self.assertEqual(fields["planner_input_cp_lane_closure_count"], 1)
        self.assertEqual(fields["planner_input_cp_obstacle_count"], 1)
        self.assertEqual(fields["planner_input_cp_selected_control_id"], "tl_1")
        self.assertEqual(fields["planner_input_cp_generated_tl"], 1)


if __name__ == "__main__":
    unittest.main()
