import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencda.scenario_testing.run_status import write_run_status


class _Vector(object):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _Vehicle(object):
    def __init__(self, actor_id, x, y, speed):
        self.id = actor_id
        self._location = _Vector(x, y, 0.0)
        self._velocity = _Vector(speed, 0.0, 0.0)

    def get_location(self):
        return self._location

    def get_transform(self):
        return SimpleNamespace(location=self._location)

    def get_velocity(self):
        return self._velocity


class _Recorder(object):
    def __init__(self, collision_count, breaches, samples, attempts, successes):
        self._summary = {
            "collision_count": collision_count,
            "road_boundary_breach_count": breaches,
            "road_boundary_sample_count": samples,
            "road_boundary_breach_rate": breaches / float(samples),
            "mpc_plan_attempts": attempts,
            "mpc_plan_successes": successes,
            "mpc_plan_success_rate": successes / float(attempts),
        }

    def summary(self):
        return dict(self._summary)


class _Planner(object):
    def __init__(self, output_dir, recorder):
        self._output_dir = Path(output_dir)
        self.evaluation_metrics = recorder

    def _resolved_debug_output_dir(self):
        return self._output_dir


def _manager(output_dir, actor_id, x, y, collision_count, finished):
    manager = SimpleNamespace(
        vehicle=_Vehicle(actor_id, x, y, 3.0),
        cpx_planner=_Planner(
            output_dir,
            _Recorder(collision_count, 1, 10, 5, 4),
        ),
        _opencda_agent_finished=finished,
    )
    return manager


def test_write_run_status_emits_same_multi_cav_truth_in_each_debug_dir(tmp_path):
    managers = [
        _manager(tmp_path / "cav1", 11, 9.0, 0.0, 0, True),
        _manager(tmp_path / "cav2", 22, 18.0, 0.0, 1, True),
    ]

    paths = write_run_status(
        vehicle_managers=managers,
        vehicle_configs=[
            {"destination": [10.0, 0.0, 0.0]},
            {"destination": [20.0, 0.0, 0.0]},
        ],
        termination_reason="route_destination_stopped",
        completed_ticks=123,
        scenario_name="two_cav_test",
        completion_mode="all_cavs",
    )

    assert paths == (
        str(tmp_path / "cav1" / "run_status.json"),
        str(tmp_path / "cav2" / "run_status.json"),
    )
    first = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    second = json.loads(Path(paths[1]).read_text(encoding="utf-8"))
    assert first["subject_cav_index"] == 0
    assert second["subject_cav_index"] == 1
    assert first["cav_states"] == second["cav_states"]
    assert first["all_cavs_finished"] is True
    assert first["collision_count"] == 1
    assert first["road_boundary_breach_count"] == 2
    assert first["distance_to_destination_m"] == pytest.approx(1.0)
    assert first["cav_states"][1]["distance_to_destination_m"] == pytest.approx(2.0)
    assert first["cav_states"][0]["mpc_plan_success_rate"] == pytest.approx(0.8)


def test_write_run_status_preserves_exception_and_partial_completion(tmp_path):
    manager = _manager(tmp_path / "ego", 7, 1.0, 2.0, 0, False)
    path = write_run_status(
        vehicle_managers=[manager],
        vehicle_configs=[{"destination": [5.0, 2.0, 0.0]}],
        termination_reason="scenario_exception",
        completed_ticks=8,
        scenario_name="failure_test",
        exception="RuntimeError: fixture failed",
    )[0]

    status = json.loads(Path(path).read_text(encoding="utf-8"))
    assert status["termination_reason"] == "scenario_exception"
    assert status["exception"] == "RuntimeError: fixture failed"
    assert status["completed_ticks"] == 8
    assert status["all_cavs_finished"] is False
    assert status["distance_to_destination_m"] == pytest.approx(4.0)


def test_write_run_status_identifies_external_ros_planner(tmp_path):
    ros_output_dir = tmp_path / "ros_ego"
    manager = SimpleNamespace(
        vehicle=_Vehicle(31, 1.0, 0.0, 2.0),
        cpx_planner=None,
        ros_planner_enabled=True,
        _ros_planner_config={"debug_output_dir": str(ros_output_dir)},
        _scenario_evaluation_metrics=_Recorder(2, 0, 10, 1, 1),
        _opencda_agent_finished=True,
    )
    path = write_run_status(
        vehicle_managers=[manager],
        vehicle_configs=[{"destination": [1.0, 0.0, 0.0]}],
        termination_reason="route_destination_stopped",
        completed_ticks=10,
        scenario_name="ros_planner_test",
        fallback_debug_output_dir=tmp_path,
    )[0]

    status = json.loads(Path(path).read_text(encoding="utf-8"))
    assert path == str(ros_output_dir / "run_status.json")
    assert status["all_cavs_finished"] is True
    assert status["cav_states"][0]["planner"] == "cpx_mpc_ros"
    assert status["collision_count"] == 2


def test_ros_run_status_preserves_simulator_collision_event_details(tmp_path):
    recorder = _Recorder(1, 0, 10, 1, 1)
    recorder.collision_events = [{
        "event_id": "42:99",
        "sim_time_s": 12.5,
        "other_actor_type": "vehicle",
        "impulse_magnitude": 321.0,
    }]
    manager = SimpleNamespace(
        vehicle=_Vehicle(42, 1.0, 0.0, 0.0),
        cpx_planner=None,
        ros_planner_enabled=True,
        _ros_planner_config={"debug_output_dir": str(tmp_path / "ros")},
        _scenario_evaluation_metrics=recorder,
        _opencda_agent_finished=False,
    )

    path = write_run_status(
        vehicle_managers=[manager],
        vehicle_configs=[{"destination": [5.0, 0.0, 0.0]}],
        termination_reason="max_ticks_reached",
        completed_ticks=20,
        scenario_name="collision_detail_test",
    )[0]

    state = json.loads(Path(path).read_text(encoding="utf-8"))["cav_states"][0]
    assert state["collision_events"] == recorder.collision_events
