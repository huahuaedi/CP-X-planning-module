"""Baseline for cpx_two_cav_merge_conflict: identical two-CAV freeway merge
with the multi-CAV interaction pipeline OFF (cav_conflict_enabled: false).
Run both and diff with opencda/scenario_testing/compare_cav_interaction.py to
show the algorithm's effect.
"""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_two_cav_merge_conflict_baseline"
    )
