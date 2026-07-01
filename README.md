# CP-X Planning Module

This repository is a OpenCDA-based CARLA planning stack focused on behavior planning, global routing, MPC trajectory planning, and evaluation metrics.

The main development target is CARLA 0.9.12 with Python 3.7 so that it can be aligned with the OpenCDA 0.9.12 ecosystem and later integrated with the MDrive planner interface.

## Current Status

The module has been tested locally with:

- CARLA 0.9.12
- Python 3.7
- `town10` CARLA scenario
- MPC runtime loop generating `mpc_cost_history.csv` and `mpc_cost_plot.png`
- behavior-planner tests for future trajectory risk and finite-state lane-change behavior
- evaluation-metrics tests


## What This Repo Contains

- `opencda/planning_module/main.py`: scenario entry point.
- `opencda/planning_module/planning_runner.py`: shared CARLA runtime loop.
- `opencda/planning_module/MPC/`: MPC trajectory planner and local-goal logic.
- `opencda/planning_module/behavior_planner/`: rule-based behavior planner.
- `opencda/planning_module/utility/`: global planner, lane graph, tracker, metrics, and config helpers.
- `opencda/planning_module/carla_scenario/`: CARLA-only scenarios.
- `opencda/planning_module/opencda_scenario/`: OpenCDA/SUMO-style scenarios.
- `opencda/planning_module/tests/`: unit tests for planner logic and scenario configuration.
- `opencda/co_simulation/`: minimal SUMO bridge dependency used by `opencda_scenario`.


## Main Planning Changes
 The current lane-change states include:

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

Lane-change decisions now consider both current lane safety and predicted future obstacle motion. The trajectory-risk gate checks future front/rear gaps and time-to-collision before a lane change can move from preparation to execution.

The planning runner also records evaluation metrics, including collision count/rate, minimum TTC, minimum PET, maximum DRAC, MPC solve status, and per-tick time-series output.

## Prerequisites

Install or prepare:

- CARLA 0.9.12 Linux package.
- Conda or Miniforge.
- Python 3.7 environment.
- A working CARLA PythonAPI egg matching Python 3.7.
- Optional: SUMO and `traci` for SUMO-based scenarios.

This project was developed against a local CARLA path like:

```bash
$HOME/Downloads/MDrive/carla912
```

If your CARLA path is different, update `CARLA_ROOT` in the commands below.

## Environment Setup

Recommended environment:

```bash
conda create -n opencda_planning python=3.7 -y
conda activate opencda_planning
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r opencda/planning_module/requirements.txt
pip install traci
```

Set CARLA paths for CARLA 0.9.12:

```bash
export CARLA_ROOT="$HOME/Downloads/MDrive/carla912"
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

If CARLA reports missing `libtiff.so.5` or `libomp.so.5`, install them inside the conda environment:

```bash
conda install -c conda-forge "libtiff=4.4.0" llvm-openmp -y
ln -s "$CONDA_PREFIX/lib/libomp.so" "$CONDA_PREFIX/lib/libomp.so.5"
```

## Quick Start

Start CARLA in one terminal:

```bash
conda activate opencda_planning
export CARLA_ROOT="$HOME/Downloads/MDrive/carla912"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd "$CARLA_ROOT"
./CarlaUE4.sh
```

Run a planning scenario in another terminal from the repository root:

```bash
conda activate opencda_planning
export CARLA_ROOT="$HOME/Downloads/MDrive/carla912"
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd opencda/planning_module
python main.py town10
```

List available scenarios:

```bash
python main.py
```

Useful scenarios:

```bash
python main.py town10
python main.py roadway_hazard
python main.py traffic_light_stop
python main.py high_level_route_planning
python main.py town10_scenario_1
python main.py town10_scenario_5
python main.py town10_scenario_6
```

The plain `town10` scenario can fall back to CARLA map spawn points if custom route anchors are missing from the loaded map.

## Scenarios

There are two families of scenarios. Both are launched with `python main.py <name>` from `opencda/planning_module/`.

### `carla_scenario/` — Pure CARLA (no SUMO, no dynamic traffic)

These scenarios use only CARLA actors. Traffic is either static or scripted. They are good for validating a single planning feature in isolation.

| Scenario | Difficulty | What it tests | Map |
|----------|-----------|---------------|-----|
| `town10` | ⭐ Easy | Baseline lane-keeping and MPC trajectory tracking with static obstacle cubes on Town10. Start here. | Town10HD_Opt |
| `traffic_light_stop` | ⭐⭐ Medium | Stop-line detection, red-light compliance, and smooth deceleration profile at a signalised intersection. | Town10HD_Opt |
| `roadway_hazard` | ⭐⭐ Medium | Obstacle avoidance when a parked vehicle partially blocks the lane. Tests lane-change trigger and re-merge. | Town10HD_Opt |
| `high_level_route_planning` | ⭐⭐⭐ Hard | Multi-waypoint route with a scripted workzone marker. Validates global re-routing and CP-message handoff. | Town10HD_Opt |
| `workzone` | ⭐⭐⭐ Hard | Narrow construction-zone passage on a custom map. Tests tight boundary constraints in the MPC. | Custom (workzone) |
| `custom_map2` | ⭐⭐⭐ Hard | Free-drive on a custom map with no pre-configured route anchors. Requires the custom map to be loaded in CARLA. | Custom (custom_map2) |

> `workzone` and `custom_map2` require their custom `.umap` files to be imported into the CARLA UE4 content folder. They will fail to launch if the map asset is missing.

### `opencda_scenario/` — CARLA + SUMO (dynamic NPC traffic and pedestrians)

These scenarios use the SUMO co-simulation bridge to spawn realistic background traffic. SUMO must be running (the runner starts it automatically if `sumo.enabled: true`).

| Scenario | Difficulty | What it tests |
|----------|-----------|---------------|
| `town10_scenario_1` | ⭐⭐ Medium | Town10 with NPC vehicles and pedestrians. NPCs activate immediately at scenario start. |
| `town10_scenario_2` | ⭐⭐ Medium | Same as scenario_1 but NPCs activate only when ego is within 20 m — tests late-appearing actors. |
| `town10_scenario_3` | ⭐⭐⭐ Hard | NPCs activate at 50 m range. Tests longer-horizon prediction and earlier lane-change decisions. |
| `town10_scenario_4` | ⭐⭐⭐ Hard | Intersection-focused route with triggered NPC vehicles. Tests gap-acceptance and yield behaviour. |
| `town10_scenario_5` | ⭐⭐⭐⭐ Very Hard | Adds scripted hazard vehicles (dark Tesla Model 3) that cut into the ego lane at 60 m trigger range, on top of SUMO traffic and pedestrians. |
| `town10_scenario_6` | ⭐⭐⭐⭐ Very Hard | Same hazard-vehicle setup as scenario_5 on a different Town10 route with more turns. |

**Recommended test order:**
1. `town10` — confirm the MPC loop works
2. `traffic_light_stop` — confirm stop-line logic
3. `roadway_hazard` — confirm obstacle avoidance
4. `town10_scenario_1` — add live traffic
5. `town10_scenario_5` or `6` — full stress test

### Analysing results

After any run, output files are written to the scenario directory. Use the bundled analysis script:

```bash
cd opencda/planning_module

# analyse the most recent run
python analyze_run.py

# analyse a specific run
python analyze_run.py opencda_scenario/town10_scenario_5

# compare multiple runs side by side
python analyze_run.py --compare \
  opencda_scenario/town10_scenario_1 \
  opencda_scenario/town10_scenario_5 \
  opencda_scenario/town10_scenario_6

# list all runs that have result files
python analyze_run.py --list
```

The script produces `analysis_report.txt` (pass/warn/fail for each metric) and `analysis_plots.png` (8-panel figure). The `--compare` flag produces an additional `comparison_plots.png` bar chart across all selected scenarios.

`analysis_plots.png` panels:

| Panel | Title | What to look for |
|-------|-------|-----------------|
| Speed | ego speed over time | Unexpected stops or oscillation |
| TTC | nearest time-to-collision | Drops below 3 s threshold line |
| DRAC | deceleration rate to avoid collision | Spikes above 3 m/s² |
| MPC cost terms | per-component cost over time | `RoadBoundary` spikes at turns = lane pressing |
| Solver status | solved vs failed pie | Large failed slice = planner instability |
| Solve time | histogram of OSQP wall time | Long tail beyond 50 ms |
| **Boundary cost vs curvature** | `Cost_RoadBoundary` (red, left axis) overlaid with trajectory curvature κ (blue, right axis); breach intervals shaded | **Correlated peaks prove the vehicle presses the line specifically at curves, not on straights — quantitative evidence without video** |
| **Trajectory breach map** | ego path coloured green→red by boundary cost; × markers at every position where a breach occurred | **Spatial map of where on the route the lane pressing happens** |

Key metrics to watch:

| Metric | Target | Warning | Basis |
|--------|--------|---------|-------|
| Collisions | 0 | > 0 | Any collision is a hard failure in safety-critical systems. |
| Min TTC (s) | > 3 s | < 2 s | NHTSA forward-collision warning research uses 2.5 s; ISO 22179 (FSRA) uses 2.0 s as the minimum acceptable headway. 3 s is the commonly cited "comfortable" threshold in AV safety literature; < 2 s is classified as critical in multiple standards. |
| Max DRAC (m/s²) | < 3 | > 4 | Comfortable braking is typically 1.5–2.5 m/s²; emergency braking is 6–8 m/s². 3 m/s² is the boundary between comfortable and uncomfortable deceleration used in passenger-vehicle ride-quality assessments. |
| Boundary breach % | < 5 % | > 10 % | Internal judgment: minor deviations at tight turns are tolerated up to 5 % of planning ticks. Above 10 % indicates a systematic control error (the baseline run measured 41.5 %, which confirmed the threshold is meaningful). No published standard directly defines this metric. |
| MPC success rate | > 97 % | < 90 % | Engineering judgment: OSQP occasionally reaches its iteration limit under heavy obstacle fields; 3 failures per 100 planning cycles is acceptable. Below 90 % the fallback open-loop control becomes dominant, increasing risk. |
| Max solve time (ms) | < 30 ms | > 50 ms | CARLA runs at 20 Hz (50 ms per tick). The MPC replans at 4 Hz but executes in the same Python thread. 30 ms keeps solver overhead below 60 % of one tick, leaving margin for the rest of the loop. 50 ms would consume a full tick and risk dropping frames. |

> **Note on MDrive integration:** if the module is later evaluated on the MDrive leaderboard, replace the TTC and DRAC thresholds with MDrive's own infraction categories, which have their own severity classification. The values above are internal development targets only.

## Tests

Run the lightweight tests used for the current cleanup:

```bash
conda activate opencda_planning
cd <repo-root>
python -m unittest \
  opencda/planning_module/tests/test_trajectory_risk.py \
  opencda/planning_module/tests/test_evaluation_metrics.py
```

Python 3.7 syntax check:

```bash
python -m py_compile \
  opencda/planning_module/planning_runner.py \
  opencda/planning_module/MPC/mpc.py \
  opencda/planning_module/MPC/local_goal.py \
  opencda/planning_module/behavior_planner/planner.py \
  opencda/planning_module/behavior_planner/trajectory_risk.py \
  opencda/planning_module/utility/evaluation_metrics.py
```

## Outputs

Scenario runs can generate:

- `mpc_cost_history.csv`
- `mpc_cost_plot.png`
- `planning_metrics.json`
- `planning_metrics_timeseries.csv`
- `behavior_planner/cp_message.json`
- `carla_launch.log`


## License

This project is based on OpenCDA. Keep the original OpenCDA license and citation requirements when publishing or sharing the code.
