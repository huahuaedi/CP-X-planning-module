"""Layer 0 -- golden route-topology fixture.

Bottom of the layered rebuild (see the 6-layer plan). This layer owns exactly
one thing: the AD-map / dij route for the ``cpx_single_right_lane_turn``
scenario must keep the same *shape*. Nothing here touches behavior, speed, or
MPC -- it is a pure structural snapshot so that any later change which
silently re-routes the vehicle is caught here first.

If the golden values below need to move, that is a deliberate route change:
update the constants in the same commit and say why.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAPS_DIR = REPO_ROOT / "opencda" / "planning_module" / "Global_Planner" / "maps"
DIJ_CACHE = (
    REPO_ROOT
    / "opencda"
    / "planning_module"
    / "Global_Planner"
    / "m0_compare"
    / "_dij_cache"
)

# cpx_single_right_lane_turn.yaml: spawn_position / destination.
SCENARIO_MAP = "Town06"
START_XYZ = {"x": 225.10, "y": -20.1, "z": 0.3}
GOAL_XYZ = {"x": 10.113184332893075, "y": -96.89902155894256, "z": 0.3}

# --- GOLDEN (dij route, route_sample_distance_m = 1.0) --------------------- #
GOLDEN_ENTRY_COUNT = 294
GOLDEN_TOTAL_M = 289.36
GOLDEN_SIGNATURE = (
    "lane_follow",
    "lane_change:right",
    "lane_follow",
    "junction_turn:right",
    "lane_follow",
)
# (kind, direction, s_start_m, s_end_m) -- arc positions checked with a
# tolerance; the point is drift detection, not sub-metre exactness.
GOLDEN_SEGMENTS = [
    ("lane_follow", "", 0.0, 128.0),
    ("lane_change", "right", 129.0, 140.0),
    ("lane_follow", "", 141.0, 213.0),
    ("junction_turn", "right", 214.0, 218.0),
    ("lane_follow", "", 219.0, 289.36),
]
ARC_TOL_M = 6.0


def _load_route_geometry():
    from utility.global_planner import CustomGlobalPlannerAdapter
    from pipeline.route_manager import CPXRouteManager

    adapter = CustomGlobalPlannerAdapter(
        xodr_path=str(MAPS_DIR / f"{SCENARIO_MAP}.xodr"),
        cache_root=str(DIJ_CACHE),
        route_sample_distance_m=1.0,
    )
    adapter.load()
    manager = CPXRouteManager(global_planner=adapter)
    summary = manager.set_destination(start_point=START_XYZ, goal_point=GOAL_XYZ)
    return summary, manager


def _deps_available() -> bool:
    if not (MAPS_DIR / f"{SCENARIO_MAP}.xodr").is_file():
        return False
    try:
        from Global_Planner.global_planner.runtime import import_ad_map_access
        import_ad_map_access()
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


@unittest.skipUnless(
    _deps_available() and os.environ.get("CPX_SKIP_ROUTE_FIXTURES") != "1",
    "AD-map / Town06 route fixture dependencies unavailable",
)
class Layer0RouteTopologyGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.manager = _load_route_geometry()
        cls.rg = cls.manager.route_geometry

    def test_route_is_found_with_expected_size(self):
        self.assertTrue(getattr(self.summary, "route_found", False))
        self.assertEqual(self.manager.route_debug_reason, "admap_route_ready")
        self.assertAlmostEqual(
            len(self.manager._route_entries), GOLDEN_ENTRY_COUNT, delta=6
        )

    def test_route_geometry_is_valid_and_positive_length(self):
        self.assertIsNotNone(self.rg)
        self.assertTrue(self.rg.valid)
        self.assertAlmostEqual(self.rg.total_m, GOLDEN_TOTAL_M, delta=8.0)

    def test_topology_validation_passes_clean(self):
        report = self.rg.validate_topology()
        self.assertTrue(report.valid, msg=f"errors={report.errors}")
        self.assertEqual(report.errors, ())

    def test_segment_signature_matches_golden(self):
        report = self.rg.validate_topology()
        self.assertEqual(tuple(report.signature), GOLDEN_SIGNATURE)

    def test_segment_kinds_directions_and_arc_ranges(self):
        segments = self.rg.segments
        self.assertEqual(len(segments), len(GOLDEN_SEGMENTS))
        for seg, (kind, direction, s0, s1) in zip(segments, GOLDEN_SEGMENTS):
            self.assertEqual(seg.kind, kind)
            self.assertEqual(seg.direction, direction)
            self.assertLessEqual(
                abs(seg.s_start_m - s0), ARC_TOL_M,
                msg=f"{kind} start {seg.s_start_m:.1f} vs golden {s0}",
            )
            self.assertLessEqual(
                abs(seg.s_end_m - s1), ARC_TOL_M,
                msg=f"{kind} end {seg.s_end_m:.1f} vs golden {s1}",
            )

    def test_exactly_one_junction_turn_to_the_right(self):
        turns = [s for s in self.rg.segments if s.kind == "junction_turn"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].direction, "right")

    def test_exactly_one_lane_change_before_the_turn(self):
        turn = next(s for s in self.rg.segments if s.kind == "junction_turn")
        pre_turn_changes = [
            s
            for s in self.rg.segments
            if s.kind == "lane_change" and s.s_end_m <= turn.s_start_m
        ]
        self.assertEqual(
            len(pre_turn_changes), 1,
            msg=f"pre-turn lane changes: "
            f"{[(s.s_start_m, s.s_end_m, s.direction) for s in pre_turn_changes]}",
        )
        self.assertEqual(pre_turn_changes[0].direction, "right")

    def test_cumulative_arc_length_is_monotonic(self):
        cum = self.rg._cum_m
        self.assertTrue(all(b >= a - 1.0e-9 for a, b in zip(cum, cum[1:])))
        self.assertGreater(cum[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
