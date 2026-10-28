from unittest.mock import patch

from cpx_planning.runtime_pause_monitor import RuntimePauseMonitor


def test_runtime_pause_monitor_records_max_pause_and_cycle():
    monitor = RuntimePauseMonitor(enabled=False)
    monitor.enabled = True
    monitor.set_cycle(12345)

    with patch(
        "cpx_planning.runtime_pause_monitor.time.perf_counter_ns",
        side_effect=[1_000_000, 6_500_000],
    ):
        monitor._gc_callback("start", {"generation": 2})
        monitor._gc_callback("stop", {"generation": 2})

    snapshot = monitor.snapshot()
    assert snapshot["collection_count"] == 1
    assert snapshot["max_pause"] == {
        "duration_ms": 5.5,
        "cycle_id": 12345,
        "generation": 2,
    }
    assert snapshot["generations"]["2"]["count"] == 1
