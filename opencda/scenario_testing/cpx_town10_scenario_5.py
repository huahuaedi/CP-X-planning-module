# -*- coding: utf-8 -*-
"""Ported scenario: ``opencda_scenario/town10_scenario_5``.

Runs the legacy planning-module scenario through the official OpenCDA
VehicleManager / CPXMPCPlannerBridge pipeline instead of the standalone
``opencda/planning_module/main.py`` runner. World bootstrapping (obstacle/
marker spawning, SUMO co-simulation if enabled, scripted traffic lights,
proactive CP messages) is provided unchanged by the original scenario module;
see ``opencda/scenario_testing/utils/cpx_scenario_bridge.py`` for the shared
integration and important scope-reduction notes (no pygame HUD window is
reproduced here).
"""

from opencda.scenario_testing.utils.cpx_scenario_bridge import run_legacy_scenario_port


def run_scenario(opt, scenario_params):
    run_legacy_scenario_port(
        opt,
        scenario_params,
        loader_name="opencda_scenario",
        legacy_scenario_name="town10_scenario_5",
        script_name="cpx_town10_scenario_5",
    )
