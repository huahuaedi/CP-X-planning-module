from pathlib import Path
from types import SimpleNamespace

from opencda.scenario_testing.evaluations.evaluate_manager import (
    EvaluationManager,
)


class _CavWorld:
    def __init__(self, manager):
        self._manager = manager

    def get_vehicle_managers(self):
        return {1: self._manager}


def test_kinematics_evaluation_skips_cpx_vehicle_without_behavior_agent(tmp_path):
    manager = SimpleNamespace(
        vehicle=SimpleNamespace(id=7),
        agent=None,
    )
    evaluator = EvaluationManager.__new__(EvaluationManager)
    evaluator.cav_world = _CavWorld(manager)
    evaluator.eval_save_path = str(tmp_path)

    log_path = Path(tmp_path) / "log.txt"
    evaluator.kinematics_eval(str(log_path))

    assert "CP-X planner owns this vehicle" in log_path.read_text()
