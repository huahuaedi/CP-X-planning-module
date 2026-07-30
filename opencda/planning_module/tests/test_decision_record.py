import json
import unittest

from opencda.planning_module.pipeline.decision_record import build_decision_record


class DecisionRecordTest(unittest.TestCase):
    def test_final_reference_gate_has_independent_veto_owner(self):
        record = build_decision_record(
            reference_stabilizer_reason="rebuilt_current_lane_reference",
            final_reference_gate_reason="first_forward_before_contract",
        )

        rows = [row.as_dict() for row in record.veto_chain]
        self.assertTrue(any(
            row["owner"] == "FinalReferenceGate"
            and row["effect"] == "reject_reference"
            and row["reason"] == "first_forward_before_contract"
            for row in rows
        ))

    def test_candidate_rejections_and_top_k_pruning_enter_veto_chain(self):
        summary = json.dumps([
            {
                "name": "keep_lane",
                "feasibility_status": "mpc_probe_solved",
                "feasibility_reason": "",
            },
            {
                "name": "route_lane_change_assertive",
                "feasibility_status": "mpc_probe_infeasible",
                "feasibility_reason": "mpc_probe:primal infeasible",
            },
            {
                "name": "route_lane_change_conservative",
                "feasibility_status": "mpc_probe_skipped",
                "feasibility_reason": "",
            },
            {
                "name": "yield_slow_down",
                "feasibility_status": "infeasible",
                "feasibility_reason": "candidate_prediction_collision_risk:1.20",
            },
        ])

        record = build_decision_record(
            candidate_selected_name="keep_lane",
            candidate_selected_decision="lane_follow",
            candidate_selected_status="mpc_probe_solved",
            candidate_pipeline_summary=summary,
            candidate_mpc_probe_summary=(
                "keep_lane:solved|route_lane_change_assertive:primal infeasible"
            ),
        )

        rows = [row.as_dict() for row in record.veto_chain]
        self.assertTrue(any(
            row["owner"] == "CandidateMPCFeasibility"
            and row["effect"] == "reject_candidate"
            and "assertive" in row["reason"]
            for row in rows
        ))
        self.assertTrue(any(
            row["owner"] == "CandidateMPCFeasibility"
            and row["effect"] == "prune_candidate"
            and "conservative" in row["reason"]
            for row in rows
        ))
        self.assertTrue(any(
            row["owner"] == "CandidateEvaluator"
            and row["effect"] == "reject_candidate"
            and "prediction_collision" in row["reason"]
            for row in rows
        ))


if __name__ == "__main__":
    unittest.main()
