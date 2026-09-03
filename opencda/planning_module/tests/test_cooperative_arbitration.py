import math
import unittest

from opencda.planning_module.pipeline.cooperative_arbitration import (
    ConflictAssignment,
    PeerIntent,
    ResourceClaim,
    assign_conflict_roles,
    lateral_side,
    should_yield,
)


class CooperativeArbitrationTest(unittest.TestCase):
    def test_yields_to_active_peer_claim_ahead(self):
        my_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=10.0,
            active=False,
        )
        peer_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=5.0,
            active=True,
        )
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=1,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(2, peer_claim, (10.0, 0.0))],
        )
        self.assertIsNotNone(reason)
        self.assertIn("peer=2", reason)

    def test_does_not_yield_to_peer_behind(self):
        my_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=10.0,
            active=False,
        )
        peer_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=5.0,
            active=True,
        )
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=1,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(2, peer_claim, (-10.0, 0.0))],
        )
        self.assertIsNone(reason)

    def test_does_not_yield_beyond_range(self):
        my_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=10.0,
            active=False,
        )
        peer_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=5.0,
            active=True,
        )
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=1,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(2, peer_claim, (100.0, 0.0))],
            range_m=40.0,
        )
        self.assertIsNone(reason)

    def test_ignores_different_resource_id(self):
        my_claim = ResourceClaim(
            kind="avoidance_lane",
            resource_id="lane_3",
            committed_at_s=10.0,
            active=True,
        )
        peer_claim = ResourceClaim(
            kind="avoidance_lane",
            resource_id="lane_5",
            committed_at_s=5.0,
            active=True,
        )
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=1,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(2, peer_claim, (10.0, 0.0))],
        )
        self.assertIsNone(reason)

    def test_ignores_different_kind(self):
        my_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=10.0,
            active=True,
        )
        peer_claim = ResourceClaim(
            kind="junction_entry",
            resource_id="junction_7",
            committed_at_s=5.0,
            active=True,
        )
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=1,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(2, peer_claim, (10.0, 0.0))],
        )
        self.assertIsNone(reason)

    def test_ignores_inactive_peer_claim(self):
        my_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=10.0,
            active=False,
        )
        peer_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=5.0,
            active=False,
        )
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=1,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(2, peer_claim, (10.0, 0.0))],
        )
        self.assertIsNone(reason)

    def test_simultaneous_commit_breaks_tie_on_actor_id(self):
        my_claim = ResourceClaim(
            kind="junction_entry",
            resource_id="junction_7",
            committed_at_s=10.0,
            active=True,
            require_ahead=False,
        )
        lower_id_peer = ResourceClaim(
            kind="junction_entry",
            resource_id="junction_7",
            committed_at_s=10.0,
            active=True,
        )
        higher_id_peer = ResourceClaim(
            kind="junction_entry",
            resource_id="junction_7",
            committed_at_s=10.0,
            active=True,
        )
        # Peer with the lower actor id wins a same-timestamp tie.
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=5,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(2, lower_id_peer, (10.0, 0.0))],
        )
        self.assertIsNotNone(reason)
        # Peer with the higher actor id loses the tie -- ego proceeds.
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=5,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(9, higher_id_peer, (10.0, 0.0))],
        )
        self.assertIsNone(reason)

    def test_require_ahead_false_yields_regardless_of_direction(self):
        my_claim = ResourceClaim(
            kind="junction_entry",
            resource_id="junction_7",
            committed_at_s=10.0,
            active=True,
            require_ahead=False,
        )
        peer_claim = ResourceClaim(
            kind="junction_entry",
            resource_id="junction_7",
            committed_at_s=5.0,
            active=True,
        )
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=1,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=[(2, peer_claim, (-10.0, 0.0))],
        )
        self.assertIsNotNone(reason)


def _claim(kind="lane_change", committed_at_s=10.0, active=True, require_ahead=True):
    return ResourceClaim(
        kind=kind, resource_id=kind, committed_at_s=committed_at_s,
        active=active, require_ahead=require_ahead,
    )


def _peer(actor_id, xy, committed_at_s, *, kind="lane_change", active=True,
          cooperative=True):
    return PeerIntent(
        actor_id=actor_id, position_xy=xy,
        claim=_claim(kind=kind, committed_at_s=committed_at_s, active=active),
        cooperative=cooperative,
    )


class ConflictRoleAssignmentTest(unittest.TestCase):
    def _assign(self, peers, *, my_commit=10.0, kind="lane_change", latch=None,
                **kw):
        return assign_conflict_roles(
            my_claim=_claim(kind=kind, committed_at_s=my_commit),
            my_actor_id=5,
            my_position_xy=(0.0, 0.0),
            my_heading_rad=0.0,
            peers=peers,
            latch_state=latch,
            **kw,
        )

    def test_earlier_peer_on_merge_makes_ego_open_a_gap(self):
        a, _ = self._assign([_peer(2, (12.0, 0.0), committed_at_s=5.0)])
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0].role, "make_gap")
        self.assertTrue(a[0].peer_wins)

    def test_earlier_peer_on_non_merge_makes_ego_yield(self):
        a, _ = self._assign(
            [_peer(2, (12.0, 0.0), committed_at_s=5.0, kind="junction_entry")],
            kind="junction_entry",
        )
        self.assertEqual(a[0].role, "yield")

    def test_later_peer_lets_ego_proceed(self):
        a, _ = self._assign([_peer(2, (12.0, 0.0), committed_at_s=20.0)])
        self.assertEqual(a[0].role, "proceed")
        self.assertFalse(a[0].peer_wins)

    def test_homotopy_side_left_and_right(self):
        a_left, _ = self._assign([_peer(2, (12.0, 5.0), committed_at_s=5.0)])
        a_right, _ = self._assign([_peer(3, (12.0, -5.0), committed_at_s=5.0)])
        self.assertEqual(a_left[0].homotopy_side, "left")
        self.assertEqual(a_right[0].homotopy_side, "right")

    def test_non_cooperative_peer_is_not_assigned(self):
        a, _ = self._assign(
            [_peer(2, (12.0, 0.0), committed_at_s=5.0, cooperative=False)]
        )
        self.assertEqual(a, [])

    def test_inactive_or_out_of_range_or_behind_peer_skipped(self):
        self.assertEqual(
            self._assign([_peer(2, (12.0, 0.0), 5.0, active=False)])[0], []
        )
        self.assertEqual(
            self._assign([_peer(2, (500.0, 0.0), 5.0)])[0], []
        )
        self.assertEqual(
            self._assign([_peer(2, (-12.0, 0.0), 5.0)])[0], []
        )

    def test_hysteresis_holds_role_through_a_transient_flip(self):
        # tick 1: peer commits later -> ego proceeds, latched.
        a1, latch = self._assign(
            [_peer(2, (12.0, 0.0), committed_at_s=11.0)], hysteresis_ticks=3
        )
        self.assertEqual(a1[0].role, "proceed")
        # tick 2: peer re-commits slightly earlier (non-decisive 0.2s margin).
        a2, latch = self._assign(
            [_peer(2, (12.0, 0.0), committed_at_s=9.8)],
            latch=latch, hysteresis_ticks=3,
        )
        self.assertEqual(a2[0].role, "proceed")  # held
        a3, latch = self._assign(
            [_peer(2, (12.0, 0.0), committed_at_s=9.8)],
            latch=latch, hysteresis_ticks=3,
        )
        self.assertEqual(a3[0].role, "proceed")  # still held
        a4, latch = self._assign(
            [_peer(2, (12.0, 0.0), committed_at_s=9.8)],
            latch=latch, hysteresis_ticks=3,
        )
        self.assertEqual(a4[0].role, "make_gap")  # 3 consistent ticks -> switch

    def test_decisive_margin_switches_immediately(self):
        a1, latch = self._assign(
            [_peer(2, (12.0, 0.0), committed_at_s=11.0)], hysteresis_ticks=5
        )
        self.assertEqual(a1[0].role, "proceed")
        # peer now clearly earlier (margin 8s >> decisive_margin_s default 1.0)
        a2, latch = self._assign(
            [_peer(2, (12.0, 0.0), committed_at_s=2.0)],
            latch=latch, hysteresis_ticks=5,
        )
        self.assertEqual(a2[0].role, "make_gap")

    def test_symmetric_resolution_between_two_cavs(self):
        # ego id 5 commits at 10; peer id 2 commits at 7 -> peer wins.
        ego_a, _ = assign_conflict_roles(
            my_claim=_claim(committed_at_s=10.0), my_actor_id=5,
            my_position_xy=(0.0, 0.0), my_heading_rad=0.0,
            peers=[_peer(2, (12.0, 0.0), committed_at_s=7.0)],
        )
        peer_a, _ = assign_conflict_roles(
            my_claim=_claim(committed_at_s=7.0), my_actor_id=2,
            my_position_xy=(12.0, 0.0), my_heading_rad=math.pi,
            peers=[_peer(5, (0.0, 0.0), committed_at_s=10.0)],
        )
        self.assertEqual(ego_a[0].role, "make_gap")   # loser opens gap
        self.assertEqual(peer_a[0].role, "proceed")   # winner proceeds

    def test_lateral_side_helper(self):
        self.assertEqual(lateral_side((0.0, 0.0), 0.0, (5.0, 2.0)), "left")
        self.assertEqual(lateral_side((0.0, 0.0), 0.0, (5.0, -2.0)), "right")


if __name__ == "__main__":
    unittest.main()
