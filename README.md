# CP-X Planning Module

This repository is a cleaned OpenCDA-based CARLA planning stack focused on behavior planning, global routing, MPC trajectory planning, and evaluation metrics.

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
- Optional for CARLA-only scenarios, but required for SUMO-backed scenarios: SUMO, `netconvert`, and `traci`.

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

For SUMO-backed `opencda_scenario` runs, install SUMO and set `SUMO_HOME`:

```bash
sudo apt update
sudo apt install -y sumo sumo-tools
export SUMO_HOME=/usr/share/sumo
```

Check that the SUMO tools are visible:

```bash
which sumo
which sumo-gui
which netconvert
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
export SUMO_HOME=/usr/share/sumo

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

For a SUMO-backed scenario, run:

```bash
python main.py town10_sumo
```

SUMO-backed scenarios use `scripts/netconvert_carla.py` and `scripts/data/opendrive_netconvert.typ.xml` to generate SUMO network assets from CARLA OpenDRIVE files when `auto_generate_assets: true` is enabled in the scenario YAML.

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

These are runtime artifacts and should not be committed.

## Common Issues

### `ImportError: libtiff.so.5`

Install an older libtiff runtime inside the conda environment:

```bash
conda install -c conda-forge "libtiff=4.4.0" -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### `libomp.so.5: cannot open shared object file`

Install OpenMP runtime and add a compatibility symlink if needed:

```bash
conda install -c conda-forge llvm-openmp -y
ln -s "$CONDA_PREFIX/lib/libomp.so" "$CONDA_PREFIX/lib/libomp.so.5"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### `Anchor 'cav_spawn' was not found`

The plain CARLA `Town10HD_Opt` map does not contain custom route anchor cubes. The runner includes a fallback that selects CARLA map spawn points when these anchors are missing.

### `OpenCDA netconvert helper was not found`

SUMO-backed scenarios need:

```text
scripts/netconvert_carla.py
scripts/data/opendrive_netconvert.typ.xml
```

Keep these files in the repository. They are required for automatic SUMO asset generation.

### `please declare environment variable 'SUMO_HOME'`

Set:

```bash
export SUMO_HOME=/usr/share/sumo
```

## Known Limitations

- CARLA 0.9.12 + Python 3.7 is the recommended setup. CARLA 0.9.16 + Python 3.10 is not the current target for this cleaned repo.
- SUMO scenarios may require extra local SUMO/netconvert setup and valid `.xodr` paths.
- The MDrive planner is not integrated yet; this repository keeps the current rule-based behavior planner plus MPC stack and documents the intended adapter path.


## License

This project is based on OpenCDA. Keep the original OpenCDA license and citation requirements when publishing or sharing the code.
