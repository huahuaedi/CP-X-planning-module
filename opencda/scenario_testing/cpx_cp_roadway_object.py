# -*- coding: utf-8 -*-
"""Cooperative-perception roadway-object validation."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_cp_roadway_object"
    )
