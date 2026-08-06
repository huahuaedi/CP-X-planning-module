# -*- coding: utf-8 -*-
"""Dedicated CP-X lane-change test on OpenCDA's mature 2-lane freeway layout.

Reuses the same map/coordinates as `cpx_mature_2lanefree_carla`
(``opencda/assets/2lane_freeway_simplified``, ego spawn/destination already
proven across four upstream OpenCDA demo scenarios), runs the full CP-X
pipeline (``mode: full_cpx_mpc``), and places a single slow vehicle directly
in ego's own lane to force a lane-change decision. See
``config_yaml/cpx_lane_change_2lanefree.yaml`` for the scenario-specific
reasoning.
"""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt,
        scenario_params,
        script_name="cpx_lane_change_2lanefree",
    )
