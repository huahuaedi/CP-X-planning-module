import json
import os
from pathlib import Path

from opencda.scenario_testing.compare_cav_interaction import _status


def test_status_ignores_artifact_older_than_debug_csv(tmp_path: Path):
    status = tmp_path / "run_status.json"
    debug = tmp_path / "opencda_planner_debug.csv"
    status.write_text(json.dumps({"termination_reason": "old"}), encoding="utf-8")
    debug.write_text("sim_time_s\n1.0\n", encoding="utf-8")
    os.utime(status, ns=(1_000_000_000, 1_000_000_000))
    os.utime(debug, ns=(2_000_000_000, 2_000_000_000))
    assert _status(tmp_path) == {}


def test_status_reads_artifact_written_after_debug_csv(tmp_path: Path):
    status = tmp_path / "run_status.json"
    debug = tmp_path / "opencda_planner_debug.csv"
    debug.write_text("sim_time_s\n1.0\n", encoding="utf-8")
    status.write_text(json.dumps({"termination_reason": "goal"}), encoding="utf-8")
    os.utime(debug, ns=(1_000_000_000, 1_000_000_000))
    os.utime(status, ns=(2_000_000_000, 2_000_000_000))
    assert _status(tmp_path)["termination_reason"] == "goal"
