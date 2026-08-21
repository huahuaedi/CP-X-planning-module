# -*- coding: utf-8 -*-
"""CP-X Town05 turn scenario using the AD-map (dij) global planner backend.

See ``config_yaml/cpx_highspeed_town05_turn.yaml`` for spawn/destination and
planner configuration.
"""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt,
        scenario_params,
        script_name="cpx_highspeed_town05_turn",
    )
