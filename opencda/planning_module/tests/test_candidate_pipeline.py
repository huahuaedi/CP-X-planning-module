import importlib.util
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "pipeline"
for package_name in ("opencda", "opencda.planning_module", "opencda.planning_module.pipeline"):
    if package_name not in sys.modules:
        module = types.ModuleType(package_name)
        module.__path__ = [str(PIPELINE_ROOT)]
        sys.modules[package_name] = module

REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "opencda.planning_module.pipeline.reference_contract",
    PIPELINE_ROOT / "reference_contract.py",
)
reference_contract = importlib.util.module_from_spec(REFERENCE_SPEC)
sys.modules[REFERENCE_SPEC.name] = reference_contract
REFERENCE_SPEC.loader.exec_module(reference_contract)

CANDIDATE_SPEC = importlib.util.spec_from_file_location(
    "opencda.planning_module.pipeline.candidate_pipeline",
    PIPELINE_ROOT / "candidate_pipeline.py",
)
candidate_pipeline = importlib.util.module_from_spec(CANDIDATE_SPEC)
sys.modules[CANDIDATE_SPEC.name] = candidate_pipeline
CANDIDATE_SPEC.loader.exec_module(candidate_pipeline)


class _ContractResult:
    def __init__(self, valid=True, reason=""):
        self.valid = bool(valid)
        self._reason = str(reason)

    def reason(self):
        return self._reason


class CandidatePipelineTest(unittest.TestCase):
    def test_stop_candidate_has_priority(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=True,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].decision, "stop_at_intersection")
        self.assertEqual(intents[0].target_speed_mps, 0.0)

    def test_lane_change_candidate_requires_authorization(self):
        denied = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=False,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )
        self.assertFalse(any("lane_change" in intent.decision for intent in denied))

        allowed = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )
        self.assertTrue(any(intent.decision == "lane_change_left" for intent in allowed))

    def test_intersection_turn_candidate_does_not_include_plain_keep_lane(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="intersection_turn_right",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].decision, "intersection_turn_right")
        self.assertEqual(intents[0].reason, "route_option_turn_required")

    def test_selects_feasible_candidate_over_invalid_contract(self):
        bad_intent = candidate_pipeline.CandidateBehaviorIntent(
            name="bad",
            decision="lane_follow",
            target_lane_id=1,
            target_speed_mps=3.0,
            base_cost=0.0,
        )
        good_intent = candidate_pipeline.CandidateBehaviorIntent(
            name="good",
            decision="lane_follow",
            target_lane_id=1,
            target_speed_mps=2.0,
            base_cost=10.0,
        )
        bad = candidate_pipeline.CandidateReferenceResult(
            intent=bad_intent,
            destination_state=[5.0, 0.0, 3.0, 0.0],
            lane_center_reference=[{"x_ref_m": 1.0, "y_ref_m": 0.0}],
            contract_result=_ContractResult(valid=False, reason="bad_reference"),
        )
        good = candidate_pipeline.CandidateReferenceResult(
            intent=good_intent,
            destination_state=[5.0, 0.0, 2.0, 0.0],
            lane_center_reference=[{"x_ref_m": 1.0, "y_ref_m": 0.0}],
            contract_result=_ContractResult(valid=True),
        )
        evaluated = [
            candidate_pipeline.evaluate_candidate_reference(
                candidate=bad,
                ego_state=[0.0, 0.0, 0.0, 0.0],
                object_snapshots=[],
                current_lane_id=1,
            ),
            candidate_pipeline.evaluate_candidate_reference(
                candidate=good,
                ego_state=[0.0, 0.0, 0.0, 0.0],
                object_snapshots=[],
                current_lane_id=1,
            ),
        ]
        selected = candidate_pipeline.select_best_candidate(evaluated)
        self.assertEqual(selected.intent.name, "good")

    def test_prediction_trajectory_blocks_candidate(self):
        intent = candidate_pipeline.CandidateBehaviorIntent(
            name="keep",
            decision="lane_follow",
            target_lane_id=1,
            target_speed_mps=3.0,
            base_cost=0.0,
        )
        candidate = candidate_pipeline.CandidateReferenceResult(
            intent=intent,
            destination_state=[5.0, 0.0, 3.0, 0.0],
            lane_center_reference=[
                {"x_ref_m": 1.0, "y_ref_m": 0.0},
                {"x_ref_m": 2.0, "y_ref_m": 0.0},
            ],
            contract_result=_ContractResult(valid=True),
        )
        evaluated = candidate_pipeline.evaluate_candidate_reference(
            candidate=candidate,
            ego_state=[0.0, 0.0, 0.0, 0.0],
            object_snapshots=[],
            prediction_trajectories={
                "obstacle": [
                    {"x": 1.2, "y": 0.0},
                    {"x": 2.1, "y": 0.0},
                ]
            },
            current_lane_id=1,
            min_object_distance_m=1.0,
        )
        self.assertFalse(evaluated.feasible)
        self.assertIn("candidate_prediction_collision_risk", evaluated.feasibility_reason)

    def test_turn_curvature_contract_violation_is_softened(self):
        intent = candidate_pipeline.CandidateBehaviorIntent(
            name="turn",
            decision="intersection_turn_left",
            target_lane_id=1,
            target_speed_mps=1.0,
            base_cost=0.0,
        )
        candidate = candidate_pipeline.CandidateReferenceResult(
            intent=intent,
            destination_state=[4.0, 2.0, 1.0, 0.0],
            lane_center_reference=[
                {"x_ref_m": 1.0, "y_ref_m": 0.0},
                {"x_ref_m": 2.0, "y_ref_m": 0.4},
            ],
            contract_result=_ContractResult(
                valid=False,
                reason="curvature_out_of_contract",
            ),
        )
        evaluated = candidate_pipeline.evaluate_candidate_reference(
            candidate=candidate,
            ego_state=[0.0, 0.0, 0.0, 0.0],
            object_snapshots=[],
            current_lane_id=1,
        )
        self.assertTrue(evaluated.feasible)
        self.assertIn("turn_contract_softened", evaluated.feasibility_reason)


if __name__ == "__main__":
    unittest.main()
