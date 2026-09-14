"""CP-off control for the Town05 potential red-light violator scenario."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_cp_red_light_violator_off"
    )
