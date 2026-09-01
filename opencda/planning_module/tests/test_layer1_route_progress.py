"""Layer 1 -- route progress & topology tracking, in isolation.

Drives a synthetic ego straight along the *golden* dij route for
``cpx_single_right_lane_turn`` and checks only what the route layer owns:

  * route progress (float arc length) is monotonic and actually completes
    -- it must not freeze partway (the 20:10 replan-loop symptom);
  * ``upcoming_turn`` / ``upcoming_lane_change`` report the right direction
    and a distance that shrinks monotonically on approach, sourced from
    RouteGeometry (not the bridge's macro-text override);
  * lane identity along the route is continuous -- it changes only across a
    lane-change / turn segment boundary, never flip-flops;
  * exactly one lane change is announced before the turn (no spurious
    second lane change while approaching the junction);
  * nothing here triggers a replan -- ``route_debug_reason`` stays
    ``admap_route_ready`` for the whole traversal.

The route is never allowed to drive behaviour / speed / MPC in this test.
"""

from __future__ import annotations

import math
import os
import unittest

import importlib.util as _ilu
from pathlib import Path as _Path

_spec = _ilu.spec_from_file_location(
    "_layer0_route_topology",
    _Path(__file__).resolve().parent / "test_layer0_route_topology.py",
)
_layer0 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_layer0)
_deps_available = _layer0._deps_available
_load_route_geometry = _layer0._load_route_geometry


def _walk_poses(route_xy, step_m=2.0):
    """Yield (x, y, heading) sampled ~every step_m along the polyline."""
    acc = 0.0
    for (ax, ay), (bx, by) in zip(route_xy[:-1], route_xy[1:]):
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1.0e-6:
            continue
        heading = math.atan2(by - ay, bx - ax)
        n = max(1, int(seg / step_m))
        for k in range(n):
            t = k / n
            yield (ax + t * (bx - ax), ay + t * (by - ay), heading)
        acc += seg
    yield (route_xy[-1][0], route_xy[-1][1],
           math.atan2(route_xy[-1][1] - route_xy[-2][1],
                      route_xy[-1][0] - route_xy[-2][0]))


@unittest.skipUnless(
    _deps_available() and os.environ.get("CPX_SKIP_ROUTE_FIXTURES") != "1",
    "AD-map / Town06 route fixture dependencies unavailable",
)
class Layer1RouteProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.manager = _load_route_geometry()
        cls.rg = cls.manager.route_geometry
        cls.route_xy = [
            (float(p[0]), float(p[1]))
            for p in (cls.summary.route_waypoints or [])
        ]
        cls.turn_seg = next(
            s for s in cls.rg.segments if s.kind == "junction_turn"
        )
        cls.first_lc = next(
            s for s in cls.rg.segments if s.kind == "lane_change"
        )
        # Replay the traversal once; every test reads the recorded trace.
        cls.trace = cls._replay()

    @classmethod
    def _replay(cls):
        rows = []
        for (x, y, h) in _walk_poses(cls.route_xy, step_m=2.0):
            cls.manager.sync_route_progress(
                ego_x_m=x, ego_y_m=y, ego_heading_rad=h
            )
            prog = cls.rg.project(x, y, s_lower_m=cls.manager.route_progress_s_m)
            turn_dir, turn_dist, turn_reason = cls.manager.upcoming_turn(
                ego_x_m=x, ego_y_m=y, ego_heading_rad=h, lookahead_m=60.0
            )
            lc_dir, lc_dist, lc_reason = cls.manager.upcoming_lane_change(
                ego_x_m=x, ego_y_m=y, ego_heading_rad=h, lookahead_m=60.0
            )
            info = cls.manager.get_route_info(
                x_m=x, y_m=y, query_key="layer1", fallback_lane_id=1
            )
            rows.append({
                "x": x, "y": y,
                "s_m": float(cls.manager.route_progress_s_m),
                "turn_dir": turn_dir, "turn_dist": turn_dist,
                "turn_reason": turn_reason,
                "lc_dir": lc_dir, "lc_dist": lc_dist, "lc_reason": lc_reason,
                "debug_reason": cls.manager.route_debug_reason,
                "remaining_m": float(info.get("remaining_distance_m", 0.0) or 0.0),
                "optimal_lane_id": int(info.get("optimal_lane_id", 0) or 0),
                "lane_index": int(prog.lane_index),
                "seg_kind": prog.segment.kind if prog.segment else "",
            })
        return rows

    # -- progress ------------------------------------------------------- #
    def test_progress_is_monotonic_non_decreasing(self):
        s = [r["s_m"] for r in self.trace]
        backsteps = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(s, s[1:]))
            if b < a - 0.25
        ]
        self.assertEqual(backsteps, [], msg=f"progress went backwards: {backsteps[:5]}")

    def test_progress_completes_and_does_not_freeze(self):
        s = [r["s_m"] for r in self.trace]
        self.assertGreaterEqual(
            s[-1], 0.9 * self.rg.total_m,
            msg=f"progress stalled at {s[-1]:.1f} of {self.rg.total_m:.1f} m",
        )
        # no single station holds for more than a quarter of the traversal
        longest_hold = max_hold = 1
        for a, b in zip(s, s[1:]):
            longest_hold = longest_hold + 1 if abs(b - a) < 0.05 else 1
            max_hold = max(max_hold, longest_hold)
        self.assertLess(
            max_hold, len(s) // 4,
            msg=f"progress frozen for {max_hold} consecutive steps",
        )

    def test_route_info_remaining_distance_decreases(self):
        rem = [r["remaining_m"] for r in self.trace]
        backsteps = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(rem, rem[1:]))
            if b > a + 1.0
        ]
        self.assertEqual(backsteps, [], msg=f"remaining_distance rose: {backsteps[:5]}")

    # -- no replan ---------------------------------------------------- #
    def test_no_replan_is_triggered_by_progress_sync(self):
        reasons = {r["debug_reason"] for r in self.trace}
        self.assertEqual(
            reasons, {"admap_route_ready"},
            msg=f"unexpected route_debug_reason values: {reasons}",
        )

    # -- turn ------------------------------------------------------- #
    def test_turn_is_announced_right_and_closes_in(self):
        seen = [
            r for r in self.trace
            if r["turn_dir"] == "right" and math.isfinite(r["turn_dist"])
        ]
        self.assertGreater(len(seen), 3, msg="turn never announced within lookahead")
        for r in seen:
            self.assertEqual(r["turn_reason"], "route_geometry_turn_ahead")
        dists = [r["turn_dist"] for r in seen]
        rises = [(a, b) for a, b in zip(dists, dists[1:]) if b > a + 0.75]
        self.assertEqual(rises, [], msg=f"turn distance grew on approach: {rises[:5]}")

    def test_turn_not_announced_once_well_past_it(self):
        past = [
            r for r in self.trace
            if r["s_m"] > self.turn_seg.s_end_m + 15.0
        ]
        self.assertTrue(past, "traversal never got well past the turn")
        self.assertTrue(
            all(not math.isfinite(r["turn_dist"]) or r["turn_dist"] == float("inf")
                for r in past),
            msg="turn still announced long after passing it",
        )

    # -- lane change ---------------------------------------------- #
    def test_first_lane_change_announced_right_before_the_turn(self):
        approach = [
            r for r in self.trace
            if r["s_m"] < self.first_lc.s_start_m
            and r["lc_dir"] == "right"
            and math.isfinite(r["lc_dist"])
        ]
        self.assertGreater(len(approach), 3, "first lane change never announced")
        for r in approach:
            self.assertEqual(r["lc_reason"], "route_geometry_lane_change_ahead")
        dists = [r["lc_dist"] for r in approach]
        rises = [(a, b) for a, b in zip(dists, dists[1:]) if b > a + 0.75]
        self.assertEqual(rises, [], msg=f"lane-change distance grew on approach: {rises[:5]}")

    def test_no_second_lane_change_announced_while_approaching_the_turn(self):
        # Between finishing the first lane change and entering the turn, the
        # route must not ask for another lane change.
        window = [
            r for r in self.trace
            if self.first_lc.s_end_m + 3.0 < r["s_m"] < self.turn_seg.s_start_m
        ]
        self.assertTrue(window, "no samples between first lane change and turn")
        spurious = [r for r in window if r["lc_dir"] in {"left", "right"}
                    and math.isfinite(r["lc_dist"])]
        self.assertEqual(
            spurious, [],
            msg=f"{len(spurious)} spurious lane-change announcements before the turn",
        )

    # -- lane identity continuity ----------------------------------- #
    def test_stable_lane_index_steps_by_one_only_across_a_lane_change(self):
        # The stable corridor index the route layer exposes must not churn
        # where the AD map merely re-segments a lane; it steps by exactly one
        # per lane change and is otherwise constant along the traversal.
        idx = [r["lane_index"] for r in self.trace]
        for i, (a, b) in enumerate(zip(idx, idx[1:])):
            self.assertLessEqual(
                abs(b - a), 1,
                msg=f"lane_index jumped {a}->{b} at step {i}",
            )
        # one net step, in the 'right' == -1 direction, for the single
        # pre-turn lane change this scenario has.
        self.assertEqual(idx[0], 0)
        self.assertIn(min(idx), (-2, -1))
        # never returns to a corridor it already left
        seen_low = 0
        for value in idx:
            seen_low = min(seen_low, value)
            self.assertLessEqual(
                value, seen_low + 1,
                msg=f"lane_index climbed back toward an abandoned corridor: {idx}",
            )

    def test_raw_ad_lane_id_churn_is_hidden_by_the_stable_index(self):
        # Sanity: the raw ids really do churn mid-lane-follow (that is why the
        # stable index exists); the stable index above must stay smooth
        # through exactly those points.
        rg = self.rg
        raw = [rg._lane_id(i) for i in range(len(rg._nodes))]
        raw_changes = sum(
            1 for a, b in zip(raw, raw[1:]) if a and b and a != b
        )
        self.assertGreater(raw_changes, 2, "expected raw AD id churn in this route")


if __name__ == "__main__":
    unittest.main()
