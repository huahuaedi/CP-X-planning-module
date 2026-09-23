"""Real-MTR multimodal three-CAV validation scenario."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt,
        scenario_params,
        script_name="cpx_three_cav_merge_mtr_multimodal",
    )
