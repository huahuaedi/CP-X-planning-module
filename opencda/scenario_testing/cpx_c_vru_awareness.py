# -*- coding: utf-8 -*-
"""Three-CAV Town06 cooperative VRU-awareness validation."""

from opencda.scenario_testing.single_intersection_town06_carla import (
    run_scenario as _run_town06_scenario,
)


def run_scenario(opt, scenario_params):
    # Keep this experiment's logs separate from the generic three-CAV run.
    cav1 = scenario_params["scenario"]["single_cav_list"][0]
    cav1["planner"]["debug_output_dir"] = (
        "opencda/planning_module/opencda_bridge/"
        "debug_cpx_c_vru_awareness"
    )
    _run_town06_scenario(opt, scenario_params)
