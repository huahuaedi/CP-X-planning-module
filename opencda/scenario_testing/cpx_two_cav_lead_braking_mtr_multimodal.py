"""Real-MTR multimodal LEAD_BRAKE validation scenario."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt,
        scenario_params,
        script_name="cpx_two_cav_lead_braking_mtr_multimodal",
    )
