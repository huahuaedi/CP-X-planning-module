import unittest
from types import SimpleNamespace

from Global_Planner.global_planner import admap_backend


class _FakeMatcher:
    def getMapMatchedPositions(self, enu_point, distance, probability):
        del enu_point, distance, probability
        return []


class _MatcherFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return _FakeMatcher()


class AdMapBackendMatcherCacheTest(unittest.TestCase):
    def setUp(self):
        self._previous_ad = admap_backend.ad
        self._previous_matcher = admap_backend._map_matcher
        self.factory = _MatcherFactory()
        admap_backend.ad = SimpleNamespace(
            map=SimpleNamespace(
                match=SimpleNamespace(AdMapMatching=self.factory),
            ),
            physics=SimpleNamespace(
                Distance=lambda value: value,
                Probability=lambda value: value,
            ),
        )
        admap_backend._reset_map_matcher()

    def tearDown(self):
        admap_backend.ad = self._previous_ad
        admap_backend._map_matcher = self._previous_matcher

    def test_matcher_is_reused_until_map_cache_is_reset(self):
        point = object()

        self.assertEqual(admap_backend.get_map_matches(point), [])
        self.assertEqual(admap_backend.get_map_matches(point), [])
        self.assertEqual(self.factory.calls, 1)

        admap_backend._reset_map_matcher()
        self.assertEqual(admap_backend.get_map_matches(point), [])
        self.assertEqual(self.factory.calls, 2)


if __name__ == "__main__":
    unittest.main()
