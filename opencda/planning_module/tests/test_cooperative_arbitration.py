import unittest

from opencda.planning_module.pipeline.cooperative_arbitration import (
    ResourceClaim,
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


if __name__ == "__main__":
    unittest.main()
