"""Two CP-X CAVs competing for a clear passing lane behind slow traffic.

While both lateral maneuvers are committed, deterministic resource
arbitration assigns one CAV ``proceed`` and the other ``make_gap``.  Stage C
then bounds longitudinal progress and Stage D adds the winner's latched
homotopy half-spaces to the MPC QP.

Interaction pipeline is enabled via cav_conflict_enabled: true in the yaml.
The baseline has identical physical inputs with this pipeline disabled.
``export_cav_interaction_report.py`` produces the A/B evidence package.
"""

from opencda.scenario_testing.cpx_mature_runner import run_mature_scenario


def run_scenario(opt, scenario_params):
    run_mature_scenario(
        opt, scenario_params, script_name="cpx_two_cav_merge_conflict"
    )
