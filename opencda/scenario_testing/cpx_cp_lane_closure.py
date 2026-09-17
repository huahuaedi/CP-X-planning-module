# -*- coding: utf-8 -*-
"""CP-F lane-closure reroute validation through the native OpenCDA entry."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_cp_lane_closure"
    )
