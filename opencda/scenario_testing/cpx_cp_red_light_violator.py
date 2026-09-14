"""CP warning for a potential red-light violator at a Town05 crossing."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_cp_red_light_violator"
    )
