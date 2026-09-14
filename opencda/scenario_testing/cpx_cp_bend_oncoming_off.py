# -*- coding: utf-8 -*-
"""CP-disabled control arm for bend-oncoming validation."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_cp_bend_oncoming_off"
    )
