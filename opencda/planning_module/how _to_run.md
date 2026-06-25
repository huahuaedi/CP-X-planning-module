# OpenCDA Planning Module

This folder contains the custom planning-module workflow that sits on top of the full OpenCDA repository. It includes:

- CARLA-only scenarios under `carla_scenario/`
- CARLA + SUMO scenarios under `opencda_scenario/`
- the runtime entrypoint `main.py`
- the shared runtime loop in `planning_runner.py`
- MPC configuration in `MPC/mpc.yaml`
- planning-module-only Python dependencies in `requirements.txt`

This README is written for a new lab member who clones the whole `OpenCDA` repository and needs to set it up on a different machine.

## 1. Repository layout to know first

Relevant files and folders:

- repo root dependencies: `../../requirements.txt`
- planning-module dependencies: `requirements.txt`
- planning-module entrypoint: `main.py`
- CARLA-only scenarios: `carla_scenario/`
- CARLA + SUMO scenarios: `opencda_scenario/`
- SUMO assets used by planning-module scenarios: `opencda_scenario/assets/`
- MPC tuning file: `MPC/mpc.yaml`
- cooperative message file written at runtime: `behavior_planner/cp_message.json`
- planning-module tests: `tests/`

## 2. What needs to be installed

### Required software

1. Python
   Use a Python version that matches the CARLA Python API available on the machine.
   Check the files under:
   `CARLA_ROOT/PythonAPI/carla/dist/`

   Example:
   if you see a file like `carla-...-py3.10-...`, use Python 3.10.

2. CARLA
   This planning-module branch is currently written around a local CARLA installation or CARLA source build.

3. SUMO
   Required only for the scenarios under `opencda_scenario/`.

4. Unreal Engine root
   Needed only if you want this code to auto-launch CARLA with `launch_mode: ue4editor_map`.

### Required Python packages

Install both requirements files. The planning module depends on packages from the repo root and from `opencda/planning_module/requirements.txt`.

From the repository root `/home/umd-user/Desktop/OpenCDA`:

```bash
conda create -n opencda_planning python=3.10 -y
conda activate opencda_planning
pip install -r requirements.txt
pip install -r opencda/planning_module/requirements.txt
pip install traci
```

If your CARLA build is not for Python 3.10, create the environment with the matching Python version instead.

Notes:

- `traci` is required for SUMO-backed scenarios.
- The code will try to import the CARLA Python API from the active environment first, then fall back to the CARLA egg under `CARLA_ROOT`.
- The old repo-root `setup.sh` is not the main setup path for this planning-module workflow. The current `main.py` resolves CARLA directly from `CARLA_ROOT` or from an installed package.

## 3. Environment variables to set

These are the main environment variables a new user should set in `~/.bashrc`, `~/.zshrc`, or in the terminal before running:

```bash
export OPENCDA_ROOT=/path/to/OpenCDA
export CARLA_ROOT=/path/to/carla
export SUMO_HOME=/usr/share/sumo
export UE4_ROOT=/path/to/UnrealEngine
```

What each one is for:

- `OPENCDA_ROOT`
  Recommended. Helps the SUMO asset generation helpers find the repo root.

- `CARLA_ROOT`
  Used by `main.py` and `utility/global_planner.py` to find the CARLA Python API and CARLA map files.

- `SUMO_HOME`
  Required for SUMO-backed scenarios so the code can find SUMO tools such as `randomTrips.py`.

- `UE4_ROOT`
  Needed only when the scenario is configured to auto-launch CARLA with `launch_mode: ue4editor_map`.

If CARLA is already running manually on `127.0.0.1:2000`, the code can connect to it directly and may not need to auto-launch CARLA. In that case `UE4_ROOT` may not be needed.

## 4. Files that must be changed on a new machine

The scenario YAML files currently contain machine-specific paths such as:

- `/home/umd-user/carla_source/carla`
- `/home/umd-user/carla_source/carla/Unreal/CarlaUE4/Content/Carla/Maps/...`

On a new machine, search these files and update the CARLA and SUMO-related paths:

```bash
rg -n "carla_root:|xodr_path:|map:" opencda/planning_module/carla_scenario opencda/planning_module/opencda_scenario -g '*.yaml'
```

### Fields to update

In the YAML files under `carla_scenario/` and `opencda_scenario/`, check these fields:

- `carla.carla_root`
  Change this to the local CARLA installation root.

- `carla.map`
  Some scenarios use built-in CARLA map names like `/Game/Carla/Maps/Town10HD_Opt`.
  Others use absolute file paths such as the `Town06.umap` path in `opencda_scenario/town6_scenario_1/town6_scenario_1.yaml`.
  If a scenario uses an absolute `.umap` path, update it for the new machine.

- `sumo.xodr_path`
  Update this to the local `.xodr` file for the map.

- `sumo.asset_root`
  Optional. By default the planning module stores generated SUMO assets under `opencda_scenario/assets/`.

- `carla.host` and `carla.port`
  Change only if CARLA is running on a different address or port.

### Scenario folders that currently contain local CARLA paths

CARLA-only scenarios:

- `carla_scenario/town10/town10.yaml`
- `carla_scenario/high_level_route_planning/high_level_route_planning.yaml`
- `carla_scenario/roadway_hazard/roadway_hazard.yaml`
- `carla_scenario/traffic_light_stop/traffic_light_stop.yaml`
- `carla_scenario/workzone/workzone.yaml`
- `carla_scenario/custom_map2/custom_map2.yaml`

SUMO-backed scenarios:

- `opencda_scenario/town10/town10_sumo.yaml`
- `opencda_scenario/high_level_route_planning/high_level_route_planning_sumo.yaml`
- `opencda_scenario/roadway_hazard/roadway_hazard.yaml`
- `opencda_scenario/town6_scenario_1/town6_scenario_1.yaml`
- `opencda_scenario/town10_scenario_1/town10_scenario_1.yaml`
- `opencda_scenario/town10_scenario_2/town10_scenario_2.yaml`
- `opencda_scenario/town10_scenario_3/town10_scenario_3.yaml`
- `opencda_scenario/town10_scenario_4/town10_scenario_4.yaml`
- `opencda_scenario/town10_scenario_5/town10_scenario_5.yaml`
- `opencda_scenario/town10_scenario_6/town10_scenario_6.yaml`
- `opencda_scenario/all_usecase_scenario/all_usecase_scenario.yaml`
- `opencda_scenario/reroute_test/reroute_test.yaml`



## 6. How to run the planning module

Always run from the repository root:

```bash
cd /home/umd-user/Desktop/OpenCDA
conda activate opencda_planning
cd opencda/planning_module
```

### List available scenarios

```bash
python main.py
```

### Run a CARLA-only scenario

```bash
python main.py high_level_route_planning
```

Other CARLA-only examples:

```bash
python main.py town10
python main.py roadway_hazard
python main.py traffic_light_stop
```

### Run a CARLA + SUMO scenario

```bash
python main.py town10_sumo
```

Other SUMO-backed examples:

```bash
python main.py town6_scenario_1
python main.py town10_scenario_1
python main.py town10_scenario_5
python main.py all_usecase_scenario
python main.py reroute_test
```

## 7. What happens when a scenario starts

1. `main.py` loads the scenario YAML.
2. It connects to CARLA on the configured host and port.
3. If CARLA is not already running and the scenario allows it, the code tries to launch CARLA automatically.
4. It loads the requested map.
5. If the scenario has `sumo.enabled: true`, the code prepares or regenerates SUMO assets and starts the SUMO bridge.
6. It runs the shared planning loop from `planning_runner.py`.

## 8. Where outputs and runtime files go

Important runtime files:

- CARLA launch log:
  `opencda/planning_module/carla_launch.log`

- cooperative planning message file:
  `opencda/planning_module/behavior_planner/cp_message.json`

- MPC cost artifacts:
  the runner writes `mpc_cost_history.csv` and `mpc_cost_plot.png` into the scenario folder that is being used

Examples:

- `opencda/planning_module/opencda_scenario/town10_scenario_1/mpc_cost_history.csv`
- `opencda/planning_module/opencda_scenario/town10_scenario_1/mpc_cost_plot.png`

## 9. How SUMO assets work

For SUMO-backed scenarios, the code uses:

- `opencda/planning_module/opencda_scenario/sumo_assets.py`
- `scripts/netconvert_carla.py`
- SUMO's `randomTrips.py`

If the SUMO config, net, and route files are missing, the code can generate them automatically when:

- `sumo.enabled: true`
- `sumo.auto_generate_assets: true`

The generated assets live under:

- `opencda/planning_module/opencda_scenario/assets/Town10HD_Opt/`
- `opencda/planning_module/opencda_scenario/assets/Town06/`



## 12. Suggested setup checklist 

1. Clone the whole OpenCDA repository.
2. Create a clean conda environment with the Python version that matches CARLA.
3. Install `requirements.txt`, `opencda/planning_module/requirements.txt`, and `traci`.
4. Set `OPENCDA_ROOT`, `CARLA_ROOT`, `SUMO_HOME`, and optionally `UE4_ROOT`.
5. Search the scenario YAML files and replace the old `/home/umd-user/...` CARLA paths.
6. Run `python main.py` from `opencda/planning_module` to list scenarios.


## 13. Useful reference files

- planning-module entrypoint: `main.py`
- runtime loop: `planning_runner.py`
- MPC config: `MPC/mpc.yaml`
- tracking config: `utility/tracker.yaml`
- OpenCDA install notes: `../../docs/md_files/installation.md`

