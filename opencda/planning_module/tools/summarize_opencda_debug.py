#!/usr/bin/env python
"""Summarize CP-X OpenCDA debug CSV files for smoke-test reporting."""

from __future__ import annotations

import argparse
import csv
from collections import Counter


def _float(row, key, default=0.0):
    try:
        return float(row.get(key) or default)
    except Exception:
        return float(default)


def _counter(rows, key):
    return Counter(str(row.get(key, "")) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    print("rows:", len(rows))
    if not rows:
        return
    for key in [
        "scenario_fsm_state",
        "behavior_decision",
        "mpc_status",
        "mpc_feasibility_status",
        "traffic_signal_state",
        "traffic_signal_filtered_state",
        "route_current_road_option",
        "reference_source",
        "route_turn_reference_reason",
        "route_debug_reason",
        "route_sync_reason",
        "speed_plan_reason",
        "decision_final_action",
        "decision_control_source",
        "decision_veto_count",
        "control_guard_reason",
        "safety_supervisor_reason",
        "decision_veto_chain_text",
    ]:
        if key in rows[0]:
            print("%s:" % key, _counter(rows, key).most_common(10))
    print("max speed:", max(_float(row, "speed_mps") for row in rows))
    print("avg speed:", sum(_float(row, "speed_mps") for row in rows) / len(rows))
    for key in [
        "destination_lateral_m",
        "reference_first_lateral_m",
        "traffic_stop_forward_m",
    ]:
        if key in rows[0]:
            print("max abs %s:" % key, max(abs(_float(row, key)) for row in rows))
    print("last:")
    last = rows[-1]
    for key in [
        "scenario_fsm_state",
        "scenario_fsm_reason",
        "behavior_decision",
        "mpc_status",
        "mpc_fallback_reason",
        "mpc_feasibility_status",
        "mpc_feasibility_reason",
        "traffic_signal_state",
        "speed_plan_reason",
        "traffic_stop_forward_m",
        "destination_lateral_m",
        "reference_first_lateral_m",
        "reference_source",
        "route_turn_reference_reason",
        "route_debug_reason",
        "route_sync_reason",
        "route_progress_index",
        "decision_owner_summary",
        "decision_veto_chain_text",
    ]:
        if key in last:
            print("  %s:" % key, last.get(key))


if __name__ == "__main__":
    main()
