"""Constant-velocity baseline for the two-vehicle prediction experiment."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(opt, scenario_params, script_name="cpx_two_cav_prediction_cv")
