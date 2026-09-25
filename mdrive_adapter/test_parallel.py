import os
from pathlib import Path
import tempfile
import time
import unittest

from mdrive_adapter.parallel import PlannerPool, choose_cpus, physical_cpus


def fake_worker(pipe, cpu, settings):
    count = 0
    pipe.send({"ok": True, "ready": True})
    while True:
        command = pipe.recv()
        if command["kind"] == "close":
            break
        count += 1
        if settings.get("fail"):
            pipe.send({"ok": False, "error": "intentional failure"})
            continue
        started = time.monotonic()
        time.sleep(settings.get("delay", 0))
        pipe.send({"ok": True, "result": {"frame": command["frame"] + settings.get("offset", 0),
            "value": command["value"] + count, "started": started, "ended": time.monotonic(), "pid": os.getpid()}})
    pipe.close()


class ParallelTests(unittest.TestCase):
    def test_physical_core_selection_excludes_smt_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cpu, socket, core in ((0,0,0), (1,0,1), (2,0,0), (3,1,0)):
                path = root / ("cpu%d" % cpu) / "topology"
                path.mkdir(parents=True)
                (path / "physical_package_id").write_text(str(socket))
                (path / "core_id").write_text(str(core))
            self.assertEqual(physical_cpus([0,1,2,3], root), [0,1,3])

    def test_cpu_count_validation(self):
        with self.assertRaises(ValueError):
            choose_cpus(2, "0")

    def test_workers_overlap_and_retain_independent_state(self):
        pool = PlannerPool([{"delay": .15}] * 2, [0,1], target=fake_worker)
        try:
            jobs = {0: {"frame": 1, "value": 2}, 1: {"frame": 1, "value": 7}}
            pool.step(jobs)  # Warm up spawn/import cost before testing overlap.
            parallel = pool.step(jobs)
            serial = pool.step(jobs, concurrent=False)
            self.assertLess(max(r["started"] for r in parallel.values()),
                            min(r["ended"] for r in parallel.values()))
            self.assertGreaterEqual(serial[1]["started"], serial[0]["ended"])
            self.assertEqual([r["value"] for r in parallel.values()], [4,9])
            self.assertEqual([r["value"] for r in serial.values()], [5,10])
            self.assertNotEqual(parallel[0]["pid"], parallel[1]["pid"])
        finally:
            workers = [p for p, _ in pool.workers]
            pool.close()
        self.assertTrue(all(not p.is_alive() for p in workers))

    def test_errors_and_stale_frames_are_not_silently_accepted(self):
        for config in ({"fail": True}, {"offset": -1}):
            pool = PlannerPool([config], [0], target=fake_worker)
            try:
                with self.assertRaises(RuntimeError):
                    pool.step({0: {"frame": 1, "value": 0}})
            finally:
                pool.close()

    def test_timeout_is_bounded(self):
        pool = PlannerPool([{"delay": .3}], [0], target=fake_worker)
        pool.timeout_s = .02
        try:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                pool.step({0: {"frame": 1, "value": 0}})
        finally:
            pool.close()

    def test_failed_barrier_reaps_every_worker(self):
        pool = PlannerPool([{"fail": True}, {"delay": .1}], [0, 1], target=fake_worker)
        workers = [process for process, _ in pool.workers]
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            pool.step({0: {"frame": 1, "value": 0}, 1: {"frame": 1, "value": 0}})
        self.assertEqual(pool.workers, [])
        self.assertTrue(all(not process.is_alive() for process in workers))

    def test_mixed_frames_rejected_before_dispatch(self):
        pool = PlannerPool([{}, {}], [0, 1], target=fake_worker)
        try:
            with self.assertRaisesRegex(ValueError, "same frame"):
                pool.step({0: {"frame": 1, "value": 0}, 1: {"frame": 2, "value": 0}})
            result = pool.step({0: {"frame": 3, "value": 0}})
            self.assertEqual(result[0]["value"], 1)
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
