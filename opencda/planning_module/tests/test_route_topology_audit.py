import unittest

from pipeline.route_topology_audit import audit_route_topology_rows


class RouteTopologyAuditTests(unittest.TestCase):
    @staticmethod
    def _row(progress, remaining, signature="lane_follow -> junction_turn:right"):
        return {
            "route_topology_valid": True,
            "route_topology_signature": signature,
            "route_topology_errors": "",
            "route_progress_s_m": progress,
            "route_remaining_distance_m": remaining,
            "route_replan_attempted": False,
        }

    def test_accepts_stable_monotonic_route(self):
        report = audit_route_topology_rows([
            self._row(0.0, 10.0), self._row(4.0, 6.0), self._row(9.0, 1.0),
        ])
        self.assertTrue(report.valid)
        self.assertEqual(report.violations, ())

    def test_rejects_signature_change_progress_backstep_and_replan(self):
        rows = [self._row(4.0, 6.0), self._row(2.0, 8.0, "other")]
        rows[-1]["route_replan_attempted"] = True
        report = audit_route_topology_rows(rows)
        self.assertFalse(report.valid)
        self.assertIn("topology_signature_changed_or_missing", report.violations)
        self.assertIn("progress_backstep", report.violations)
        self.assertIn("remaining_distance_increased", report.violations)
        self.assertIn("unexpected_replan_attempt", report.violations)


if __name__ == "__main__":
    unittest.main()
