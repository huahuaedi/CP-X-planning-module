import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from opencda.planning_module.opencda_bridge.planner_assembly import (
    _mtr_prediction_bridge,
)
from opencda.planning_module.opencda_bridge.prediction_bridge import (
    MTRPredictionBridge,
)
from opencda.planning_module.pipeline.prediction import mpc_stage_trajectory
from opencda.planning_module.pipeline.tracker import CPXObstacleTracker


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({
            "predictions": {
                "veh-1": [
                    {"x": 9.0, "y": 8.0, "t": 0.1, "v": 2.0},
                    {"x": 9.2, "y": 8.0, "t": 0.2, "v": 2.0},
                ]
            },
            "prediction_modes": {
                "veh-1": [
                    {
                        "probability": 0.6,
                        "maneuver": "lane_keep",
                        "trajectory": [
                            {"x": 9.0, "y": 8.0, "t": 0.1, "v": 2.0},
                            {"x": 9.2, "y": 8.0, "t": 0.2, "v": 2.0},
                        ],
                    },
                    {
                        "probability": 0.3,
                        "maneuver": "turn_left",
                        "trajectory": [
                            {"x": 9.0, "y": 8.1, "t": 0.1, "v": 1.8},
                            {"x": 9.1, "y": 8.3, "t": 0.2, "v": 1.6},
                        ],
                    },
                    {
                        "probability": 0.1,
                        "maneuver": "brake",
                        "trajectory": [
                            {"x": 8.9, "y": 8.0, "t": 0.1, "v": 1.0},
                            {"x": 8.9, "y": 8.0, "t": 0.2, "v": 0.0},
                        ],
                    },
                ]
            }
        }).encode("utf-8")


class MTRPredictionBridgeTest(unittest.TestCase):
    def setUp(self):
        self.urlopen = mock.patch(
            "opencda.planning_module.opencda_bridge.prediction_bridge.urllib.request.urlopen",
            return_value=_FakeResponse(),
        ).start()

    def tearDown(self):
        mock.patch.stopall()

    def test_http_prediction_reaches_tracker_frame_and_mpc_format(self):
        bridge = MTRPredictionBridge(update_hz=10.0, asynchronous=False)
        obstacles = [{
            "track_id": "veh-1", "x": 1.0, "y": 2.0,
            "v": 3.0, "psi": 0.0,
        }]
        annotated = bridge.attach(
            obstacles,
            horizon_s=1.0,
            dt_s=0.1,
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 4.0, "psi": 0.0},
            timestamp_s=5.0,
        )
        tracker = CPXObstacleTracker()
        tracker.update(obstacle_snapshots=obstacles, timestamp_s=5.0)
        frame = tracker.predict(
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 4.0, "psi": 0.0},
            obstacle_snapshots=annotated,
            lane_assignments={"veh-1": 1},
            available_lane_ids=[1],
            horizon_s=1.0,
            dt_s=0.1,
            min_front_gap_m=2.0,
            min_rear_gap_m=2.0,
            min_ttc_s=1.0,
        )

        points = frame.obstacle_future_trajectories["veh-1"]
        self.assertEqual(points[0]["x"], 9.0)
        self.assertEqual(points[0]["v"], 2.0)
        hypotheses = frame.predicted_objects["veh-1"].hypotheses
        self.assertEqual(len(hypotheses), 3)
        self.assertEqual(
            [round(item.probability, 2) for item in hypotheses],
            [0.6, 0.3, 0.1],
        )
        self.assertIn("veh-1", frame.revision)
        self.assertIn("mtr:1", frame.revision)
        stages = mpc_stage_trajectory(
            points, fallback_heading_rad=0.0, horizon_steps=2, dt_s=0.1
        )
        self.assertEqual(stages[0][0:3], [9.0, 8.0, 2.0])
        request = self.urlopen.call_args[0][0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(sent["timestamp_s"], 5.0)
        self.assertEqual(sent["ego_snapshot"]["v"], 4.0)
        self.assertEqual(bridge.last_attached_count, 1)
        self.assertEqual(bridge.last_attached_mode_count, 3)

    def test_planners_in_one_cav_world_share_one_bridge_instance(self):
        cav_world = SimpleNamespace()

        def planner_bridge():
            return SimpleNamespace(
                config={"mtr_prediction_async": False},
                vehicle_manager=SimpleNamespace(
                    v2x_manager=SimpleNamespace(cav_world=cav_world)
                ),
            )

        first = _mtr_prediction_bridge(planner_bridge(), MTRPredictionBridge)
        second = _mtr_prediction_bridge(planner_bridge(), MTRPredictionBridge)

        self.assertIs(first, second)
        self.assertEqual(first.shared_consumer_count, 2)
        self.assertTrue(first.diagnostics["prediction_bridge_shared"])

    def test_rate_limit_rebases_cached_prediction(self):
        bridge = MTRPredictionBridge(update_hz=5.0, asynchronous=False)
        kwargs = {
            "object_snapshots": [{"track_id": "veh-1"}],
            "horizon_s": 1.0,
            "dt_s": 0.1,
            "ego_snapshot": {"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
        }
        first = bridge.attach(timestamp_s=1.0, **kwargs)
        second = bridge.attach(timestamp_s=1.05, **kwargs)

        self.assertEqual(self.urlopen.call_count, 1)
        self.assertAlmostEqual(first[0]["predicted_trajectory"][0]["t"], 0.1)
        self.assertAlmostEqual(second[0]["predicted_trajectory"][0]["t"], 0.05)

    def test_shared_scene_is_requested_once_for_multiple_egos_per_tick(self):
        bridge = MTRPredictionBridge(update_hz=5.0, asynchronous=False)
        actors = {
            actor_id: {
                "actor_id": actor_id, "track_id": str(actor_id),
                "x": float(actor_id), "y": 0.0, "v": 2.0, "psi": 0.0,
            }
            for actor_id in (1, 2, 3)
        }

        bridge.attach(
            [actors[2], actors[3]], horizon_s=1.0, dt_s=0.1,
            ego_snapshot=actors[1], timestamp_s=1.0,
        )
        bridge.attach(
            # A second ego can have a narrower local detection set. This is
            # still the same world tick and may not multiply MTR requests.
            [actors[1]], horizon_s=1.0, dt_s=0.1,
            ego_snapshot=actors[2], timestamp_s=1.0,
        )

        self.assertEqual(self.urlopen.call_count, 1)
        self.assertEqual(bridge.request_count, 1)
        self.assertEqual(bridge.inference_request_count, 1)

    def test_simulation_time_rollback_retires_previous_epoch_cache(self):
        bridge = MTRPredictionBridge(update_hz=5.0, asynchronous=False)
        kwargs = {
            "object_snapshots": [{"track_id": "veh-1"}],
            "horizon_s": 1.0, "dt_s": 0.1,
            "ego_snapshot": {
                "actor_id": 1, "x": 0.0, "y": 0.0,
                "v": 0.0, "psi": 0.0,
            },
        }
        bridge.attach(timestamp_s=10.0, **kwargs)
        bridge.attach(timestamp_s=0.0, **kwargs)

        self.assertEqual(self.urlopen.call_count, 2)
        request = json.loads(self.urlopen.call_args[0][0].data.decode("utf-8"))
        self.assertEqual(request["timestamp_s"], 0.0)

    def test_history_updates_at_ten_hz_while_inference_stays_at_five_hz(self):
        bridge = MTRPredictionBridge(
            update_hz=5.0, history_update_hz=10.0, asynchronous=False
        )
        kwargs = {
            "object_snapshots": [{"track_id": "veh-1"}],
            "horizon_s": 1.0,
            "dt_s": 0.1,
            "ego_snapshot": {"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
        }

        bridge.attach(timestamp_s=1.0, **kwargs)
        bridge.attach(timestamp_s=1.1, **kwargs)
        bridge.attach(timestamp_s=1.2, **kwargs)

        self.assertEqual(self.urlopen.call_count, 3)
        requests = [
            json.loads(call[0][0].data.decode("utf-8"))
            for call in self.urlopen.call_args_list
        ]
        self.assertEqual(
            [request["run_inference"] for request in requests],
            [True, False, True],
        )
        self.assertEqual(bridge.inference_request_count, 2)

    def test_unreachable_server_leaves_fallback_contract_untouched(self):
        self.urlopen.side_effect = OSError("connection refused")
        bridge = MTRPredictionBridge(
            server_url="http://127.0.0.1:1", timeout_s=0.05,
            asynchronous=False,
        )
        annotated = bridge.attach(
            [{"track_id": "veh-1", "x": 1.0, "y": 2.0}],
            horizon_s=1.0,
            dt_s=0.1,
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
            timestamp_s=1.0,
        )

        self.assertNotIn("predicted_trajectory", annotated[0])
        self.assertTrue(bridge.last_error)

    def test_async_bridge_never_blocks_planning_tick(self):
        request_started = threading.Event()
        release_response = threading.Event()

        def delayed_response(*_args, **_kwargs):
            request_started.set()
            release_response.wait(timeout=1.0)
            return _FakeResponse()

        self.urlopen.side_effect = delayed_response
        bridge = MTRPredictionBridge(update_hz=5.0, asynchronous=True)
        kwargs = {
            "object_snapshots": [{"track_id": "veh-1"}],
            "horizon_s": 1.0,
            "dt_s": 0.1,
            "ego_snapshot": {"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
        }

        first = bridge.attach(timestamp_s=1.0, **kwargs)

        # The worker's response is deliberately unavailable. If attach()
        # shared the HTTP critical path, this assertion could only be reached
        # after the one-second server timeout. It must instead return the
        # unannotated fallback contract immediately.
        self.assertEqual(bridge.success_count, 0)
        self.assertNotIn("predicted_trajectory", first[0])
        self.assertTrue(request_started.wait(timeout=1.0))
        release_response.set()
        deadline = time.monotonic() + 1.0
        while bridge.success_count < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        second = bridge.attach(timestamp_s=1.05, **kwargs)
        self.assertIn("predicted_trajectory", second[0])
        self.assertEqual(bridge.dropped_request_count, 0)


    def test_mid_run_disconnect_expires_cache_then_recovers(self):
        # Point 5 in the multimodal-integration review: the server working,
        # then dying mid-run, is a different case from never having worked
        # (test_unreachable_server_leaves_fallback_contract_untouched) --
        # here there is a fresh cache the bridge must hold for max_stale_s
        # and then stop trusting, not discard immediately or keep forever.
        bridge = MTRPredictionBridge(
            server_url="http://127.0.0.1:1", timeout_s=0.05,
            max_stale_s=0.3, asynchronous=False,
        )
        kwargs = dict(
            object_snapshots=[{"track_id": "veh-1", "x": 1.0, "y": 2.0}],
            horizon_s=1.0, dt_s=0.1,
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
        )

        # Tick 0: server up, response cached.
        annotated = bridge.attach(timestamp_s=0.0, **kwargs)
        self.assertIn("predicted_trajectory", annotated[0])
        self.assertEqual(bridge.last_error, "")

        # Server dies. Immediately after (well inside max_stale_s) the cache
        # is still trusted.
        self.urlopen.side_effect = OSError("connection refused")
        annotated = bridge.attach(timestamp_s=0.1, **kwargs)
        self.assertIn("predicted_trajectory", annotated[0])
        self.assertTrue(bridge.last_error)

        # Past max_stale_s with no successful refresh: the cache is no
        # longer trusted and the caller sees the plain fallback contract,
        # not a stale trajectory reported as current.
        annotated = bridge.attach(timestamp_s=1.0, **kwargs)
        self.assertNotIn("predicted_trajectory", annotated[0])

        # Server recovers: a normal successful response resumes attachment.
        self.urlopen.side_effect = None
        self.urlopen.return_value = _FakeResponse()
        annotated = bridge.attach(timestamp_s=1.1, **kwargs)
        self.assertIn("predicted_trajectory", annotated[0])

    def test_sudden_mode_probability_shift_is_not_smoothed_by_the_bridge(self):
        # Point 5: a sudden mode-probability change is a scenario-level
        # stability question for the corridor/credible-veto hysteresis
        # downstream, but that hysteresis can only reason about it correctly
        # if it knows this layer hands through the server's latest
        # probabilities verbatim (renormalized, not blended with the
        # previous tick's). Lock that contract here rather than leaving it
        # to be discovered as a surprise inside the corridor's own tests.
        bridge = MTRPredictionBridge(
            server_url="http://127.0.0.1:1", update_hz=100.0,
            asynchronous=False,
        )
        kwargs = dict(
            object_snapshots=[{"track_id": "veh-1", "x": 1.0, "y": 2.0}],
            horizon_s=1.0, dt_s=0.1,
            ego_snapshot={"x": 0.0, "y": 0.0, "v": 0.0, "psi": 0.0},
        )
        bridge.attach(timestamp_s=0.0, **kwargs)
        modes = bridge._cached_prediction_modes["veh-1"]
        self.assertAlmostEqual(modes[0]["probability"], 0.6, places=6)

        def flipped_response(*_args, **_kwargs):
            class _R(_FakeResponse):
                def read(self):
                    return json.dumps({
                        "predictions": {},
                        "prediction_modes": {
                            "veh-1": [
                                {
                                    "probability": 0.05,
                                    "maneuver": "lane_keep",
                                    "trajectory": [
                                        {"x": 9.0, "y": 8.0, "t": 0.1, "v": 2.0},
                                    ],
                                },
                                {
                                    "probability": 0.95,
                                    "maneuver": "brake",
                                    "trajectory": [
                                        {"x": 8.9, "y": 8.0, "t": 0.1, "v": 0.0},
                                    ],
                                },
                            ]
                        },
                    }).encode("utf-8")
            return _R()

        self.urlopen.side_effect = flipped_response
        bridge.attach(timestamp_s=0.02, **kwargs)
        modes = sorted(
            bridge._cached_prediction_modes["veh-1"],
            key=lambda m: -m["probability"],
        )
        self.assertAlmostEqual(modes[0]["probability"], 0.95, places=6)
        self.assertEqual(modes[0]["maneuver"], "brake")


if __name__ == "__main__":
    unittest.main()
