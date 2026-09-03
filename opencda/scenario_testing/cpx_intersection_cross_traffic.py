"""Prediction-knowledge ablation: ego crosses a Town06 intersection while a
deterministic, ego-blind vehicle crosses its path.

Run three times against the same config:

  1. record the oracle:  prediction_mode: cv   + prediction_ablation.record: true
  2. blind:              prediction_mode: blind
  3. oracle:             prediction_mode: oracle + oracle_trace_path: <file>

then diff with ``compare_prediction_ablation.py``.
"""

from opencda.scenario_testing.single_intersection_town06_carla import run_scenario
