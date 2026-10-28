"""Focused regression tests for thread-safe timing aggregation."""

import json

from cpx_planning.timing import CycleTimingRecorder


def test_duration_and_metadata_are_independent(tmp_path):
    """Metadata updates must not reference duration-only local variables."""
    summary_path = tmp_path / "timing.json"
    recorder = CycleTimingRecorder(enabled=True, summary_path=summary_path)

    recorder.record_duration(1, "planner.example", 2.5)
    recorder.set_metadata("input_frame_sync", {"dispatched_frames": 1})
    recorder.finish_cycle(1)
    recorder.write_summary()

    assert recorder._totals_ms["planner.example"] == 2.5
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["diagnostics"]["input_frame_sync"]["dispatched_frames"] == 1


def test_atomic_bridge_event_expands_timestamps_without_five_ros_callbacks(
    tmp_path,
):
    recorder = CycleTimingRecorder(
        enabled=True,
        summary_path=tmp_path / "timing.json",
    )
    recorder.record_external_event({
        "cycle_id": 42,
        "stream": "planning_cycle",
        "logical_streams": [
            "localization",
            "perception",
            "traffic_light",
            "v2x",
            "cooperative",
        ],
        "source_send_wall_time_ns": 1_000_000,
        "tcp_received_wall_time_ns": 2_000_000,
        "ros_publish_started_wall_time_ns": 3_000_000,
        "ros_publish_finished_wall_time_ns": 4_000_000,
    })

    events = recorder._event_times_ns[42]
    for stream in (
        "localization",
        "perception",
        "traffic_light",
        "v2x",
        "cooperative",
    ):
        assert events["transport.{}.source_send".format(stream)] == 1_000_000
        assert (
            events["transport.{}.ros_publish_finished".format(stream)]
            == 4_000_000
        )


def test_finish_cycle_never_writes_summary_on_planning_thread(
    tmp_path,
    monkeypatch,
):
    """The former 100-cycle write caused synchronized multi-shard stalls."""
    recorder = CycleTimingRecorder(
        enabled=True,
        summary_path=tmp_path / "timing.json",
    )
    writes = []
    monkeypatch.setattr(recorder, "write_summary", lambda: writes.append(True))

    for cycle_id in range(1, 201):
        recorder.finish_cycle(cycle_id)

    assert writes == []
    assert recorder._successful_cycle_count == 200


def test_summary_reports_maximum_stage_latency_and_cycle(tmp_path):
    summary_path = tmp_path / "timing.json"
    recorder = CycleTimingRecorder(enabled=True, summary_path=summary_path)

    recorder.record_duration(10, "node_scheduling.input_critical_path", 4.0)
    recorder.record_duration(11, "node_scheduling.input_critical_path", 175.0)
    recorder.record_duration(12, "node_scheduling.input_critical_path", 8.0)
    recorder.finish_cycle(12)
    recorder.write_summary()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    maximum = summary["diagnostics"]["maximum_stage_latency"][
        "node_scheduling.input_critical_path"
    ]
    assert maximum == {
        "duration_ms": 175.0,
        "cycle_id": 11,
        "sample_count": 3,
    }
    assert summary["diagnostics"]["timing_collection"] == {
        "periodic_summary_write_on_planning_thread": False,
        "final_summary_written_on_shutdown": True,
    }
