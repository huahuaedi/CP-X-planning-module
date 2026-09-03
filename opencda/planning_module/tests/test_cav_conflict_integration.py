"""CARLA-free end-to-end check of the interaction-aware stack:

    resolve_conflicts  (Stage A classify -> B assign -> C corridor)
        -> corridor_rows  (Stage D constraint builder)
            -> MPC.plan_trajectory(corridor_rows=..., cost.corridor.enabled)

Asserts the solved ego trajectory actually yields to a crossing vehicle
and opens a gap for a cooperative merging peer -- with the real OSQP
solve, no bridge, no CARLA.
"""

import unittest
from pathlib import Path

import numpy as np
import yaml

from MPC.mpc import MPC
from pipeline.cav_conflict_pipeline import resolve_conflicts
from pipeline.conflict_classifier import ClassifierParams
from pipeline.cooperative_arbitration import PeerIntent, ResourceClaim
from pipeline.mpc_corridor_constraints import corridor_rows
from pipeline.spatiotemporal_corridor import CorridorParams

# Ego reference: straight +y, 0..60 m.
REF = [{"x_ref_m": 0.0, "y_ref_m": float(y)} for y in range(0, 61, 2)]


def _mpc(corridor_enabled):
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "MPC" / "mpc.yaml").read_text()
    )
    cfg["mpc"].setdefault("cost", {})["corridor"] = {
        "enabled": bool(corridor_enabled), "w_slack": 20000.0, "max_slack_m": 1.0,
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
            obstacle_snapshots=[crosser], peer_intents=[],
            classifier_params=cls_p, corridor_params=cor_p,
        )
        self.assertEqual(r.diagnostics["tags"]["x"], "CROSSING")
        self.assertTrue(any(h < 1e8 for h in r.corridor.s_hi))

        rows = corridor_rows(r.corridor, REF, ego_origin_xy=(0.0, 0.0))
        capped = _solve_progress(mpc, (0.0, 0.0), 9.0, rows)
        free = _solve_progress(_mpc(corridor_enabled=True), (0.0, 0.0), 9.0, None)

        self.assertLess(capped, 18.0)           # did not reach the crossing point
        self.assertLess(capped, free - 2.0)     # and clearly less than unyielded

    def test_ego_opens_a_gap_for_a_cooperative_merging_peer(self):
        mpc = _mpc(corridor_enabled=True)
        cls_p, cor_p = self._params(mpc)
        ego = {"x": 0.0, "y": 0.0, "v": 9.0, "psi": np.pi / 2.0}
        # peer one lane over (x=+3.4), ahead, merging toward x=0, committed
        # EARLIER than ego -> peer wins -> ego opens the gap behind it.
        peer_path = tuple(
            (0.1 * k, 3.4 - 0.12 * k, 12.0 + 9.0 * 0.1 * k, 9.0)
            for k in range(mpc.horizon_steps + 1)
        )
        peer = PeerIntent(
            actor_id=2, position_xy=(3.4, 12.0),
            claim=ResourceClaim(kind="lane_change", resource_id="lane_change",
                                committed_at_s=3.0, active=True),
            heading_rad=np.pi / 2.0, speed_mps=9.0, planned_path=peer_path,
        )
        my_claim = ResourceClaim(kind="lane_change", resource_id="lane_change",
                                 committed_at_s=10.0, active=True)
        r = resolve_conflicts(
            reference_samples=REF, ego_snapshot=ego, my_actor_id=1,
            my_claim=my_claim, obstacle_snapshots=[], peer_intents=[peer],
            classifier_params=cls_p, corridor_params=cor_p,
        )
        self.assertEqual(r.diagnostics["roles"].get("2"), "make_gap")
        self.assertTrue(any(h < 1e8 for h in r.corridor.s_hi))

        rows = corridor_rows(r.corridor, REF, ego_origin_xy=(0.0, 0.0))
        capped = _solve_progress(mpc, (0.0, 0.0), 9.0, rows)
        free = _solve_progress(_mpc(corridor_enabled=True), (0.0, 0.0), 9.0, None)
        # ego holds back behind the peer's projected station (starts at y=12)
        self.assertLess(capped, free - 1.0)


if __name__ == "__main__":
    unittest.main()
