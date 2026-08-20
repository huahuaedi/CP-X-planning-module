# -*- coding: utf-8 -*-
"""Dedicated CP-X lane-change test on OpenCDA's 2-lane freeway layout.

Uses ``opencda/assets/2lane_freeway_simplified`` and coordinates validated by
the upstream OpenCDA freeway scenarios, runs the full CP-X
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
