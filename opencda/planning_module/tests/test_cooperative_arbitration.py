import math
import unittest

from opencda.planning_module.pipeline.cooperative_arbitration import (
    CavIntent,
    ResourceClaim,
    assign_conflict_roles,
)


def _claim(kind="lane_change", committed_at_s=10.0, active=True, require_ahead=True):
    return ResourceClaim(
        kind=kind, resource_id=kind, committed_at_s=committed_at_s,
        active=active, require_ahead=require_ahead,
    )


def _cav(actor_id, xy, committed_at_s, *, kind="lane_change", active=True,
          cooperative=True):
    return CavIntent(
        actor_id=actor_id, position_xy=xy,
        claim=_claim(kind=kind, committed_at_s=committed_at_s, active=active),
        cooperative=cooperative,
    )


class ConflictRoleAssignmentTest(unittest.TestCase):
    def _assign(self, cavs, *, my_commit=10.0, kind="lane_change", latch=None,
                **kw):
        return assign_conflict_roles(
            my_claim=_claim(kind=kind, committed_at_s=my_commit),
            my_actor_id=5,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            cavs=cavs,
            latch_state=latch,
            **kw,
        )

    def test_earlier_cav_on_merge_makes_ego_open_a_gap(self):
        a, _ = self._assign([_cav(2, (12.0, 0.0), committed_at_s=5.0)])
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0].role, "make_gap")
        self.assertTrue(a[0].cav_wins)

    def test_earlier_cav_on_non_merge_makes_ego_yield(self):
        a, _ = self._assign(
            [_cav(2, (12.0, 0.0), committed_at_s=5.0, kind="junction_entry")],
            kind="junction_entry",
        )
        self.assertEqual(a[0].role, "yield")

    def test_later_cav_lets_ego_proceed(self):
        a, _ = self._assign([_cav(2, (12.0, 0.0), committed_at_s=20.0)])
        self.assertEqual(a[0].role, "proceed")
        self.assertFalse(a[0].cav_wins)

    def test_non_cooperative_cav_is_not_assigned(self):
        a, _ = self._assign(
            [_cav(2, (12.0, 0.0), committed_at_s=5.0, cooperative=False)]
        )
        self.assertEqual(a, [])

    def test_inactive_or_out_of_range_or_behind_cav_skipped(self):
        self.assertEqual(
            self._assign([_cav(2, (12.0, 0.0), 5.0, active=False)])[0], []
        )
        self.assertEqual(
            self._assign([_cav(2, (500.0, 0.0), 5.0)])[0], []
        )
        self.assertEqual(
            self._assign([_cav(2, (-12.0, 0.0), 5.0)])[0], []
        )

    def test_hysteresis_holds_role_through_a_transient_flip(self):
        # tick 1: cav commits later -> ego proceeds, latched.
        a1, latch = self._assign(
            [_cav(2, (12.0, 0.0), committed_at_s=11.0)], hysteresis_ticks=3
        )
        self.assertEqual(a1[0].role, "proceed")
        # tick 2: cav re-commits slightly earlier (non-decisive 0.2s margin).
        a2, latch = self._assign(
            [_cav(2, (12.0, 0.0), committed_at_s=9.8)],
            latch=latch, hysteresis_ticks=3,
        )
        self.assertEqual(a2[0].role, "proceed")  # held
        a3, latch = self._assign(
            [_cav(2, (12.0, 0.0), committed_at_s=9.8)],
            latch=latch, hysteresis_ticks=3,
        )
        self.assertEqual(a3[0].role, "proceed")  # still held
        a4, latch = self._assign(
            [_cav(2, (12.0, 0.0), committed_at_s=9.8)],
            latch=latch, hysteresis_ticks=3,
        )
        self.assertEqual(a4[0].role, "make_gap")  # 3 consistent ticks -> switch

    def test_decisive_margin_switches_immediately(self):
        a1, latch = self._assign(
            [_cav(2, (12.0, 0.0), committed_at_s=11.0)], hysteresis_ticks=5
        )
        self.assertEqual(a1[0].role, "proceed")
        # cav now clearly earlier (margin 8s >> decisive_margin_s default 1.0)
        a2, latch = self._assign(
            [_cav(2, (12.0, 0.0), committed_at_s=2.0)],
            latch=latch, hysteresis_ticks=5,
        )
        self.assertEqual(a2[0].role, "make_gap")

    def test_symmetric_resolution_between_two_cavs(self):
        # ego id 5 commits at 10; cav id 2 commits at 7 -> cav wins.
        ego_a, _ = assign_conflict_roles(
            my_claim=_claim(committed_at_s=10.0), my_actor_id=5,
            my_position_xy=(0.0, 0.0), my_heading_rad=0.0,
            cavs=[_cav(2, (12.0, 0.0), committed_at_s=7.0)],
        )
        cav_a, _ = assign_conflict_roles(
            my_claim=_claim(committed_at_s=7.0), my_actor_id=2,
            my_position_xy=(12.0, 0.0), my_heading_rad=math.pi,
            cavs=[_cav(5, (0.0, 0.0), committed_at_s=10.0)],
        )
        self.assertEqual(ego_a[0].role, "make_gap")   # loser opens gap
        self.assertEqual(cav_a[0].role, "proceed")   # winner proceeds

if __name__ == "__main__":
    unittest.main()
