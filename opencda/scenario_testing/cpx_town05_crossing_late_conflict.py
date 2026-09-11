"""Short arrival-window crossing regression using the shared scenario runner."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(opt, scenario_params,
                        script_name="cpx_town05_crossing_late_conflict")
