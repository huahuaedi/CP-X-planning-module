# -*- coding: utf-8 -*-
"""CP-X on mature Town06 intersection approach, ending before the turn."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt,
        scenario_params,
        script_name="cpx_mature_intersection_approach_town06_carla",
    )

