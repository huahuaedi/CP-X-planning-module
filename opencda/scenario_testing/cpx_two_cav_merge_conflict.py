"""Two CP-X CAVs, same lane, both route-required to change into the same
adjacent lane. The leading CAV commits first and proceeds; the trailing CAV
loses the arbitration, gets role make_gap, and its Stage-C corridor caps it
behind the leader's projected station in the target lane.

Enable with cav_conflict_enabled: true (set in the yaml). Watch the debug
CSVs' cav_conflict_summary / cav_corridor_binding columns.
"""

from opencda.scenario_testing.single_intersection_town06_carla import run_scenario
