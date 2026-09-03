"""Two CP-X CAVs on the 2-lane freeway, both route-required to merge into the
same inner lane at ~20 m/s. The leading CAV commits its merge first and gets
role `proceed`; the trailing CAV loses the arbitration, gets `make_gap`, and
its Stage-C corridor caps it behind the leader's projected station in the
target lane.

Interaction pipeline is enabled via cav_conflict_enabled: true in the yaml.
Watch the debug CSVs' cav_conflict_summary / cav_corridor_binding columns,
and diff against cpx_two_cav_merge_conflict_baseline with
opencda/scenario_testing/compare_cav_interaction.py.
"""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_two_cav_merge_conflict"
    )
