# Planning Module

This directory contains the CP-X planning stack used by OpenCDA and the
standalone CARLA development scenarios. The authoritative integrated
architecture and decision ownership table are in
[`docs/planning_architecture.md`](docs/planning_architecture.md).

## Entry Point

The supported OpenCDA integration entrypoint is run from the repository root:

```bash
python opencda.py -t <scenario_name> -v 0.9.12
```

`main.py` remains available for standalone planner development scenarios. Run
it from this directory:

```bash
python main.py <scenario_name>
```

Examples:

```bash
python main.py town10
python main.py roadway_hazard
python main.py traffic_light_stop
python main.py high_level_route_planning
```

Run without arguments to list available scenarios:

```bash
python main.py
```

## CP-X Validation Scenarios

Three Town06 CARLA scenarios validate the CP-X planning stack through the
OpenCDA integration entrypoint (`opencda.py`, not `main.py`). Each spawns a
single CAV; unless noted, background traffic and vehicle config are
inherited from `single_intersection_town06_carla.yaml`.

| Scenario | What it validates |
|---|---|
| `cpx_single_right_lane_turn` | Route-required `CHANGELANERIGHT` before an intersection, followed by the continuous right-turn connector onto the outgoing road. |
| `cpx_single_left_lane_turn` | Route-required `CHANGELANELEFT` right at spawn, followed by the continuous left-turn connector. The destination sits ~40m past the turn's outgoing lane instead of ending right at the turn, and this scenario overrides `carla_traffic_manager`'s spawn range so background traffic actually spawns near this ego (the base range doesn't cover this route's x/y footprint). |
| `single_intersection_town06_carla` | Straight-line urban traffic-light behavior: the ego follows the southbound arterial through three signalized intersections with CARLA Traffic Manager background traffic. |

Run any of them from the repository root:

```bash
export PYTHONPATH="$PWD/opencda/planning_module:$PWD:$PYTHONPATH"
export OPENCDA_DEBUG_VIEW=1        # pygame debug HUD/overlay
export OPENCDA_SPECTATOR_VIEW=planner  # or "topdown"

python opencda.py -t cpx_single_right_lane_turn -v 0.9.12
python opencda.py -t cpx_single_left_lane_turn -v 0.9.12
python opencda.py -t single_intersection_town06_carla -v 0.9.12
```

Each run writes per-tick planner debug CSV/JSONL to the scenario's
`planner.debug_output_dir` (see its config yaml under
`scenario_testing/config_yaml/`). Turn that CSV into plots and a metrics
report with:

```bash
python -m opencda.planning_module.tools.export_full_run_plots <debug_csv> <output_dir>
```

## Directory Layout

- `main.py`: discovers and runs scenarios.
- `planning_runner.py`: shared CARLA runtime loop, actor spawning, route generation, behavior planner calls, MPC calls, camera/debug overlays, and metric export.
- `MPC/`: trajectory optimization and local-goal generation.
- `behavior_planner/`: rule-based behavior planner, lane-safety scoring, future trajectory risk gate, stop/reroute handling, and temporary destination selection.
- `utility/`: global planner, CARLA lane graph extraction, tracker, cooperative message helpers, config loading, and evaluation metrics.
- `carla_scenario/`: CARLA-only scenarios.
- `opencda_scenario/`: scenarios with OpenCDA/SUMO-style runtime logic.
- `tests/`: unit tests for planner behavior, scenario loading, route logic, and metrics.

## Behavior Planner

The rule-based behavior planner uses a finite state machine instead of a binary lane-change flag. Lane changes are separated into preparation, execution, cancellation, and abort states:

- `LANE_KEEP`
- `PREPARE_LANE_CHANGE_LEFT`
- `PREPARE_LANE_CHANGE_RIGHT`
- `EXECUTE_LANE_CHANGE_LEFT`
- `EXECUTE_LANE_CHANGE_RIGHT`
- `ABORT_LANE_CHANGE`
- `CANCEL_LANE_CHANGE`
- `REROUTE`
- `STOP`
- `YIELD`

Before executing a lane change, the planner checks:

- current lane safety score,
- target lane safety score,
- future obstacle trajectories,
- front/rear predicted gaps,
- predicted TTC risk.

This addresses the concern that lane-change decisions should not depend only on a static lane safety score.

## MPC

The MPC module receives the selected behavior mode, lane target, temporary destination, route context, obstacle predictions, and cost configuration. It records cost terms such as:

- reference tracking cost,
- lane center cost,
- road/lane boundary cost,
- obstacle repulsive cost,
- control cost,
- solver status and solve time.

## Metrics

The runner can export:

- collision count and collision rate,
- minimum TTC,
- minimum PET,
- maximum DRAC,
- MPC success rate,
- per-tick time-series metrics.

Generated metric, CSV, PNG, and log files are runtime artifacts and should not be committed.

## Notes

The default target environment is CARLA 0.9.12 with Python 3.7. Some original code was written with newer Python syntax, so the checked-in code avoids Python 3.8+ syntax where it affects runtime compatibility.
