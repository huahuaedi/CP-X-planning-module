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
    def test_traffic_stop_candidate_has_hard_priority(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=1,
            current_lane_id=1,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=True,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
        )
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].decision, "stop_at_intersection")
        self.assertEqual(intents[0].target_speed_mps, 0.0)

    def test_obstacle_stop_keeps_authorized_lane_change_candidates(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_follow",
            selected_target_lane_id=2,
            current_lane_id=2,
            target_speed_mps=3.0,
            candidate_lane_ids=[2, 1],
            lane_safety_scores={2: 0.4, 1: 1.0},
            lane_prediction_risks={},
            stop_goal_active=True,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=1,
            allow_lane_change_candidates=True,
            lane_change_authorization_source="route",
        )

        self.assertTrue(any(intent.name == "obstacle_stop" for intent in intents))
        self.assertTrue(any(
            intent.decision == "lane_change_right"
            for intent in intents
        ))

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
        lane_change_intents = [
            intent for intent in allowed if intent.decision == "lane_change_left"
        ]
        self.assertEqual(
            {intent.trajectory_variant for intent in lane_change_intents},
            {"assertive", "normal", "conservative"},
        )
        self.assertEqual(
            {intent.lane_change_duration_s for intent in lane_change_intents},
            {3.2, 4.0, 5.5},
        )

    def test_route_required_lane_change_penalizes_keep_lane_defer(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_change_right",
            selected_target_lane_id=1,
            current_lane_id=2,
            target_speed_mps=3.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=1,
            allow_lane_change_candidates=True,
            lane_change_authorization_source="route",
            lane_change_defer_cost=10.0,
        )

        keep = next(intent for intent in intents if intent.name == "keep_lane")
        yield_intent = next(
            intent for intent in intents if intent.name == "yield_slow_down"
        )
        normal = next(
            intent for intent in intents
            if intent.trajectory_variant == "normal"
        )
        self.assertEqual(keep.reason, "defer_route_lane_change")
        self.assertGreater(keep.base_cost, normal.base_cost)
        self.assertEqual(
            yield_intent.reason,
            "conservative_yield_defer_route_lane_change",
        )
        self.assertGreater(yield_intent.base_cost, normal.base_cost)

    def test_lane_change_reference_uses_quintic_time_blend(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0, "heading_rad": 0.0}
            for index in range(6)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 3.5, "heading_rad": 0.0}
            for index in range(6)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=4.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
        )

        self.assertEqual(len(shaped), 6)
        self.assertGreater(shaped[0]["y_ref_m"], 0.0)
        self.assertLess(shaped[0]["y_ref_m"], shaped[-1]["y_ref_m"])
        self.assertLess(shaped[-1]["y_ref_m"], 3.5)
        self.assertTrue(all(
            row["lane_transition_kind"] == "lateral_lane_change"
            for row in shaped
        ))

    def test_lane_change_reference_preserves_current_lateral_progress(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0}
            for index in range(6)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 4.0}
            for index in range(6)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=4.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
            ego_x_m=1.0,
            ego_y_m=2.0,
        )

        self.assertAlmostEqual(shaped[0]["lane_change_initial_progress"], 0.5)
        self.assertGreater(shaped[0]["y_ref_m"], 2.0)
        self.assertTrue(all(
            second["lane_change_progress"] >= first["lane_change_progress"]
            for first, second in zip(shaped[:-1], shaped[1:])
        ))

    def test_lane_change_reference_progress_floor_prevents_regression(self):
        source = [
            {"x_ref_m": float(index + 1), "y_ref_m": 0.0}
            for index in range(6)
        ]
        target = [
            {"x_ref_m": float(index + 1), "y_ref_m": 4.0}
            for index in range(6)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            target_reference=target,
            source_reference=source,
            duration_s=4.0,
            dt_s=0.5,
            current_lane_id=1,
            target_lane_id=2,
            target_speed_mps=3.0,
            ego_x_m=1.0,
            ego_y_m=0.5,
            initial_progress_floor=0.6,
        )

        self.assertAlmostEqual(
            shaped[0]["lane_change_initial_progress"],
            0.6,
        )

    def test_selected_lane_change_has_only_configured_variants(self):
        intents = candidate_pipeline.build_candidate_intents(
            selected_decision="lane_change_left",
            selected_target_lane_id=2,
            current_lane_id=1,
            target_speed_mps=4.0,
            candidate_lane_ids=[1, 2],
            lane_safety_scores={1: 1.0, 2: 1.0},
            lane_prediction_risks={},
            stop_goal_active=False,
            traffic_stop_active=False,
            lane_change_authorized=True,
            lane_change_authorized_target_lane_id=2,
            allow_lane_change_candidates=True,
            lane_change_assertive_duration_s=3.0,
            lane_change_normal_duration_s=4.5,
            lane_change_conservative_duration_s=6.0,
            lane_change_authorization_source="opportunistic",
        )

        lane_changes = [
            intent for intent in intents if intent.decision == "lane_change_left"
        ]
        self.assertEqual(len(lane_changes), 3)
        self.assertEqual(
            [intent.lane_change_duration_s for intent in lane_changes],
            [3.0, 4.5, 6.0],
        )
        self.assertTrue(all(
            intent.name.startswith("opportunistic_lane_change_left_")
            for intent in lane_changes
        ))
        self.assertTrue(all(
            intent.reason == "opportunistic_lane_change_authorized"
            for intent in lane_changes
        ))

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

    def test_prediction_trajectory_on_current_lane_becomes_follow_cost(self):
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
        self.assertTrue(evaluated.feasible)
        self.assertIn("candidate_prediction_lead_follow", evaluated.feasibility_reason)

    def test_prediction_trajectory_blocks_lane_change_candidate(self):
        intent = candidate_pipeline.CandidateBehaviorIntent(
            name="change",
            decision="lane_change_left",
            target_lane_id=2,
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

    def test_route_lane_change_does_not_bypass_unified_cost_ranking(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
                base_cost=0.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=1.0,
        )
        change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.7,
                base_cost=10.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="mpc_probe_solved",
            total_cost=20.0,
        )
        selected = candidate_pipeline.select_best_candidate([keep, change])
        self.assertEqual(selected.intent.name, "keep_lane")

    def test_committed_lane_change_cannot_be_replaced_by_keep_lane(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=1.0,
        )
        change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="committed_lane_change_continuation",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=20.0,
        )
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=0.45,
            reference_locked=True,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [keep, change],
            commitment=commitment,
        )

        self.assertEqual(outcome.status, "selected_committed")
        self.assertIs(outcome.selected, change)

    def test_route_required_lane_change_outranks_soft_keep_lane_cost(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="mpc_probe_solved",
            total_cost=13.0,
        )
        change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right_assertive",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="mpc_probe_solved",
            total_cost=49.0,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [keep, change],
            commitment=candidate_pipeline.ManeuverCommitment(),
            required_decision="lane_change_right",
            required_target_lane_id=1,
        )

        self.assertEqual(outcome.status, "selected_route_required")
        self.assertEqual(outcome.reason, "feasible_route_required_candidate")
        self.assertIs(outcome.selected, change)

    def test_route_required_lane_change_defers_when_candidate_is_infeasible(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="mpc_probe_solved",
            total_cost=13.0,
        )
        change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right_normal",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.7,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="infeasible",
            total_cost=1000.0,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [keep, change],
            commitment=candidate_pipeline.ManeuverCommitment(),
            required_decision="lane_change_right",
            required_target_lane_id=1,
        )

        self.assertEqual(outcome.status, "selected")
        self.assertEqual(outcome.reason, "route_required_candidate_infeasible_defer")
        self.assertIs(outcome.selected, keep)

    def test_committed_lane_change_requests_locked_reference_when_rerank_fails(self):
        keep = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="keep_lane",
                decision="lane_follow",
                target_lane_id=2,
                target_speed_mps=3.0,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=1.0,
        )
        invalid_change = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="infeasible",
            feasibility_reason="curvature_out_of_contract",
            total_cost=10020.0,
        )
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=0.45,
            reference_locked=True,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [keep, invalid_change],
            commitment=commitment,
        )

        self.assertEqual(outcome.status, "committed_reference_required")
        self.assertIsNone(outcome.selected)
        self.assertEqual(
            outcome.reason,
            "committed_maneuver_missing_locked_continuation",
        )

    def test_committed_lane_change_rejects_new_variant_when_locked_is_invalid(self):
        locked = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="committed_lane_change_continuation",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="infeasible",
            total_cost=1000.0,
        )
        regenerated = candidate_pipeline.CandidateReferenceResult(
            intent=candidate_pipeline.CandidateBehaviorIntent(
                name="route_lane_change_right_normal",
                decision="lane_change_right",
                target_lane_id=1,
                target_speed_mps=2.5,
            ),
            destination_state=[],
            lane_center_reference=[],
            feasibility_status="feasible",
            total_cost=1.0,
        )
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=0.45,
            reference_locked=True,
        )

        outcome = candidate_pipeline.select_candidate_with_commitment(
            [locked, regenerated],
            commitment=commitment,
        )

        self.assertIsNone(outcome.selected)
        self.assertEqual(outcome.reason, "locked_maneuver_reference_infeasible")

    def test_progress_alone_does_not_end_locked_commitment(self):
        commitment = candidate_pipeline.ManeuverCommitment(
            state="COMMITTED",
            decision="lane_change_right",
            source_lane_id=2,
            target_lane_id=1,
            progress=1.0,
            reference_locked=True,
        )

        self.assertTrue(commitment.active)

    def test_lane_change_alignment_removes_longitudinal_sample_offset(self):
        source = [
            {"x_ref_m": float(i), "y_ref_m": 0.0}
            for i in range(1, 21)
        ]
        target = [
            {"x_ref_m": float(i) + 2.0, "y_ref_m": -3.5}
            for i in range(1, 21)
        ]
        shaped = candidate_pipeline.shape_lane_change_reference(
            source_reference=source,
            target_reference=target,
            duration_s=4.0,
            dt_s=0.1,
            current_lane_id=2,
            target_lane_id=1,
            target_speed_mps=3.0,
        )
        self.assertEqual(len(shaped), 20)
        self.assertLess(abs(float(shaped[0]["x_ref_m"]) - 1.0), 0.05)
        self.assertLess(float(shaped[-1]["y_ref_m"]), -1.5)

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

    def test_mpc_probe_result_controls_candidate_feasibility(self):
        intent = candidate_pipeline.CandidateBehaviorIntent(
            name="lane_change",
            decision="lane_change_left",
            target_lane_id=2,
            target_speed_mps=2.0,
        )
        candidate = candidate_pipeline.CandidateReferenceResult(
            intent=intent,
            destination_state=[5.0, 1.0, 2.0, 0.0],
            lane_center_reference=[],
            contract_result=_ContractResult(valid=True),
            feasibility_status="feasible",
        )
        candidate_pipeline.apply_mpc_probe_result(
            candidate=candidate,
            solved=False,
            status="primal infeasible",
            solve_time_ms=2.0,
        )
        self.assertFalse(candidate.feasible)
        self.assertIn("mpc_probe:primal infeasible", candidate.feasibility_reason)


if __name__ == "__main__":
    unittest.main()
