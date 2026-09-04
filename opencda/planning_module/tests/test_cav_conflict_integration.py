"""CARLA-free end-to-end check of the interaction-aware stack:

    resolve_conflicts  (Stage A classify -> B assign -> C corridor)
        -> corridor_rows  (Stage D constraint builder)
            -> MPC.plan_trajectory(corridor_rows=..., cost.corridor.enabled)

Asserts the solved ego trajectory actually yields to a crossing vehicle
and opens a gap for a cooperative merging cav -- with the real OSQP
solve, no bridge, no CARLA.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from MPC.mpc import MPC
from pipeline.cav_conflict_pipeline import resolve_conflicts
from pipeline.conflict_classifier import ClassifierParams
from pipeline.cooperative_arbitration import CavIntent, ResourceClaim
from pipeline.mpc_corridor_constraints import corridor_rows
from pipeline.execution_pipeline import PlanningPipeline
from pipeline.spatiotemporal_corridor import CorridorParams

# Ego reference: straight +y, 0..60 m.
REF = [{"x_ref_m": 0.0, "y_ref_m": float(y)} for y in range(0, 61, 2)]


def _mpc(corridor_enabled):
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "MPC" / "mpc.yaml").read_text()
    )
    cfg["mpc"].setdefault("cost", {})["corridor"] = {
        "enabled": bool(corridor_enabled), "w_slack": 20000.0, "max_slack_m": 0.0,
    }
    return MPC(cfg["mpc"], cfg.get("road", {}))


def _lane_ref(mpc, speed_ref_mps):
    return [
        {"x_ref_m": 0.0, "y_ref_m": 0.7 * (k + 1), "x": 0.0, "y": 0.7 * (k + 1),
         "heading_rad": np.pi / 2.0, "lane_id": 1, "lane_width_m": 3.5,
         "speed_ref_mps": float(speed_ref_mps)}
        for k in range(mpc.horizon_steps)
    ]


def _solve_progress(mpc, ego_xy, ego_speed, rows):
    out = mpc.plan_trajectory(
        current_state=[ego_xy[0], ego_xy[1], float(ego_speed), np.pi / 2.0],
        destination_state=[0.0, ego_xy[1] + 40.0, float(ego_speed), np.pi / 2.0, 1],
        object_snapshots=[],
        current_acceleration_mps2=0.0,
        current_steering_rad=0.0,
        lane_center_reference_samples=_lane_ref(mpc, ego_speed),
        corridor_rows=rows,
    )
    return float(out[-1][1])   # final y == arc-length along the +y reference


class CavConflictIntegrationTests(unittest.TestCase):
    def _params(self, mpc):
        return (
            ClassifierParams(horizon_steps=mpc.horizon_steps, dt_s=mpc.dt_s),
            CorridorParams(horizon_steps=mpc.horizon_steps, dt_s=mpc.dt_s,
                           crossing_clearance_time_s=3.0, conflict_stop_buffer_m=4.0),
        )

    def test_ego_yields_to_a_non_connected_crosser(self):
        mpc = _mpc(corridor_enabled=True)
        cls_p, cor_p = self._params(mpc)
        ego = {"x": 0.0, "y": 0.0, "v": 9.0, "psi": np.pi / 2.0}
        # crosser drives +x across the ego path at y = 18 m
        crosser = {
            "id": "x", "x": -8.0, "y": 18.0, "v": 7.0, "psi": 0.0,
            "predicted_trajectory": [
                {"x": -8.0 + 0.7 * k, "y": 18.0} for k in range(mpc.horizon_steps + 1)
            ],
        }
        r = resolve_conflicts(
            reference_samples=REF, ego_snapshot=ego, my_actor_id=1, my_claim=None,
            obstacle_snapshots=[crosser], cav_intents=[],
            classifier_params=cls_p, corridor_params=cor_p,
        )
        self.assertEqual(r.diagnostics["tags"]["x"], "CROSSING")
        self.assertTrue(any(h < 1e8 for h in r.corridor.s_hi))

        rows = corridor_rows(r.corridor, REF, ego_origin_xy=(0.0, 0.0))
        capped = _solve_progress(mpc, (0.0, 0.0), 9.0, rows)
        free = _solve_progress(_mpc(corridor_enabled=True), (0.0, 0.0), 9.0, None)

        self.assertLess(capped, 18.0)           # did not reach the crossing point
        self.assertLess(capped, free - 2.0)     # and clearly less than unyielded

    def test_ego_opens_a_gap_for_a_cooperative_merging_cav(self):
        mpc = _mpc(corridor_enabled=True)
        cls_p, cor_p = self._params(mpc)
        ego = {"x": 0.0, "y": 0.0, "v": 9.0, "psi": np.pi / 2.0}
        # cav one lane over (x=+3.4), ahead, merging toward x=0, committed
        # EARLIER than ego -> cav wins -> ego opens the gap behind it.
        cav_path = tuple(
            (0.1 * k, 3.4 - 0.12 * k, 12.0 + 9.0 * 0.1 * k, 9.0)
            for k in range(mpc.horizon_steps + 1)
        )
        cav = CavIntent(
            actor_id=2, position_xy=(3.4, 12.0),
            claim=ResourceClaim(kind="lane_change", resource_id="lane_change",
                                committed_at_s=3.0, active=True),
            heading_rad=np.pi / 2.0, speed_mps=9.0, planned_path=cav_path,
        )
        my_claim = ResourceClaim(kind="lane_change", resource_id="lane_change",
                                 committed_at_s=10.0, active=True)
        r = resolve_conflicts(
            reference_samples=REF, ego_snapshot=ego, my_actor_id=1,
            my_claim=my_claim, obstacle_snapshots=[], cav_intents=[cav],
            classifier_params=cls_p, corridor_params=cor_p,
        )
        self.assertEqual(r.diagnostics["roles"].get("2"), "make_gap")
        self.assertTrue(any(h < 1e8 for h in r.corridor.s_hi))

        rows = corridor_rows(r.corridor, REF, ego_origin_xy=(0.0, 0.0))
        capped = _solve_progress(mpc, (0.0, 0.0), 9.0, rows)
        free = _solve_progress(_mpc(corridor_enabled=True), (0.0, 0.0), 9.0, None)
        # ego holds back behind the cav's projected station (starts at y=12)
        self.assertLess(capped, free - 1.0)

    def test_runtime_pipeline_sends_longitudinal_and_homotopy_rows_to_mpc(self):
        mpc = _mpc(corridor_enabled=True)
        cav_path = tuple(
            (mpc.dt_s * k, 3.4 - 0.12 * k, 12.0 + 0.9 * k, 9.0)
            for k in range(mpc.horizon_steps + 1)
        )
        cav = CavIntent(
            actor_id=2, position_xy=(3.4, 12.0),
            claim=ResourceClaim(
                kind="lane_change", resource_id="lane_change",
                committed_at_s=20.0, active=True,
            ),
            heading_rad=np.pi / 2.0, speed_mps=9.0, planned_path=cav_path,
        )
        result = PlanningPipeline.resolve_cav_interaction(
            reference_samples=REF,
            ego_location=SimpleNamespace(x=0.0, y=0.0),
            ego_yaw_rad=np.pi / 2.0,
            ego_speed_mps=9.0,
            actor_id=1,
            claim=ResourceClaim(
                kind="lane_change", resource_id="lane_change",
                committed_at_s=10.0, active=True,
            ),
            obstacle_snapshots=[{
                "id": "lead", "x": 0.0, "y": 15.0, "v": 2.0,
                "psi": np.pi / 2.0,
                "predicted_trajectory": [
                    {"x": 0.0, "y": 15.0 + 0.2 * k}
                    for k in range(mpc.horizon_steps + 1)
                ],
            }],
            cav_intents=[cav], latch_state={},
            horizon_steps=mpc.horizon_steps, dt_s=mpc.dt_s,
        )
        self.assertGreater(result.diagnostics["longitudinal_qp_row_count"], 0)
        self.assertEqual(
            result.diagnostics["homotopy_qp_row_count"], mpc.horizon_steps
        )
        self.assertEqual(
            len(result.mpc_rows), result.diagnostics["total_qp_row_count"]
        )
        self.assertEqual(result.diagnostics["shared_plan_cav_count"], 1)
        self.assertEqual(
            result.diagnostics["shared_plan_sample_count"], len(cav_path)
        )
        self.assertEqual(result.diagnostics["prediction_validation_actor_id"], 2)
        self.assertAlmostEqual(
            result.diagnostics["prediction_validation_horizon_s"], 1.0
        )
        validation_stage = int(round(1.0 / mpc.dt_s))
        self.assertAlmostEqual(
            result.diagnostics["prediction_validation_x_m"],
            cav_path[validation_stage][1],
        )
        self.assertAlmostEqual(
            result.diagnostics["prediction_validation_y_m"],
            cav_path[validation_stage][2],
        )
        groups = {row.slack_group for row in result.mpc_rows}
        self.assertEqual(groups, {"corridor", "cav_homotopy"})

        _solve_progress(mpc, (0.0, 0.0), 9.0, result.mpc_rows)
        self.assertEqual(mpc._last_status, "solved")


if __name__ == "__main__":
    unittest.main()
