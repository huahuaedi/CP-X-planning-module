# -*- coding: utf-8 -*-
"""Deterministic CP-X smoke test: Town06 traffic-light approach/stop/release."""

from opencda.scenario_testing.cpx_smoke_runner import run_smoke_scenario


def run_scenario(opt, scenario_params):
    run_smoke_scenario(
        opt,
        scenario_params,
        town="Town06",
        script_name="cpx_smoke_town06_traffic_light",
    )

