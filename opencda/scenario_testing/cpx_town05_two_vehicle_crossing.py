"""Town05 perpendicular crossing with multimodal target prediction."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_town05_two_vehicle_crossing"
    )
