import unittest

from opencda_bridge.debug_viewer import _ref_source_kind, _safe_float


class RefSourceKindTests(unittest.TestCase):
    """`ref=` alone silently carries a different underlying source
    depending on which path produced it; `_ref_source_kind` names that
    path explicitly instead of leaving the reader to infer it."""

    def test_explicit_fallback_stage_wins_even_if_maneuver_source_is_stale(self):
        kind = _ref_source_kind({
            "reference_pipeline_stage": "explicit_fallback",
            "final_reference_geometry_source": "unified_maneuver_geometry",
        })

        self.assertEqual(kind, "explicit_fallback")

    def test_maneuver_geometry_source_present_without_fallback_stage(self):
        kind = _ref_source_kind({
            "reference_pipeline_stage": "",
            "final_reference_geometry_source": "unified_maneuver_geometry",
        })

        self.assertEqual(kind, "maneuver")

    def test_neither_signal_present_is_default(self):
        kind = _ref_source_kind({
            "reference_pipeline_stage": "",
            "final_reference_geometry_source": "",
        })

        self.assertEqual(kind, "default")

    def test_missing_keys_are_default(self):
        self.assertEqual(_ref_source_kind({}), "default")


class SafeFloatTests(unittest.TestCase):
    def test_parses_numeric_string(self):
        self.assertEqual(_safe_float("1.25"), 1.25)

    def test_parses_float_and_int(self):
        self.assertEqual(_safe_float(3), 3.0)
        self.assertEqual(_safe_float(2.5), 2.5)

    def test_empty_string_returns_none(self):
        self.assertIsNone(_safe_float(""))

    def test_non_numeric_string_returns_none(self):
        self.assertIsNone(_safe_float("not_a_number"))

    def test_none_returns_none(self):
        self.assertIsNone(_safe_float(None))

    def test_non_finite_returns_none(self):
        self.assertIsNone(_safe_float(float("inf")))
        self.assertIsNone(_safe_float(float("nan")))


if __name__ == "__main__":
    unittest.main()
