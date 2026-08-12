"""Isolated CP-X lane-change validation at 12 m/s."""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(opt, scenario_params, script_name="cpx_lane_change_speed_12")
