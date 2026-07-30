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
