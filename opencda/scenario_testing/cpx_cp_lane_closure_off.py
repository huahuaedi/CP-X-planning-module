"""CP-off control arm for the high-level lane-closure reroute scenario."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt,
        scenario_params,
        script_name="cpx_cp_lane_closure_off",
    )
