"""Map-layer regression: Town05 AD-map turn and road-segment transitions."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(opt, scenario_params, script_name="cpx_mapreg_town05_turn")
