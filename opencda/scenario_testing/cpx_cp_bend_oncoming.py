# -*- coding: utf-8 -*-
"""Cooperative-perception oncoming-vehicle validation around a bend."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_cp_bend_oncoming"
    )
