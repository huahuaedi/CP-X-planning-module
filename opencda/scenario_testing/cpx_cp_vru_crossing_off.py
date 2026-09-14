# -*- coding: utf-8 -*-
"""CP-disabled control arm for pedestrian-crossing validation."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_cp_vru_crossing_off"
    )
