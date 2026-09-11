"""Synthetic-multimodal CUT_IN validation scenario."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt,
        scenario_params,
        script_name="cpx_two_vehicle_cut_in_multimodal",
    )
