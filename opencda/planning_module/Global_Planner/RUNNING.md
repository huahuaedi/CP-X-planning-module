# Global Planner Setup and Run Guide

This guide explains how to build and run the Global Planner from a clean environment.

The current Global Planner is a CARLA-independent OpenDRIVE route planner. It uses AD-map to read `.xodr` maps and performs a custom shortest-path search over a lane-level topology graph. The search uses accumulated path cost without a heuristic, so it is a Dijkstra-style search rather than A*.

## 1. Directory Structure

The Global Planner is located at:

```text
opencda/planning_module/Global_Planner/
├── build_ad_map.sh          # Builds AD-map and its Python bindings
├── global_planner/          # Global Planner Python package
├── maps/                    # OpenDRIVE maps
├── map_repo/                # AD-map source and local build artifacts
└── cache/                   # Runtime-generated map and lane caches
```

The standalone example entry point is:

```text
global_planner/main.py
```

Its default map and query are:

```python
XODR_PATH = PROJECT_ROOT / "maps" / "Town10HD_Opt.xodr"
START_POINT = {"x": -86.8, "y": 133.5, "z": 0.0}
GOAL_POINT = {"x": 59.4, "y": 137.8, "z": 0.0}
```

## 2. Requirements

The AD-map v2.3.0 Python bindings used by this project require:

- Ubuntu/Linux
- Conda or Miniforge
- Python 3.7
- GCC/G++ 9 through 12; GCC/G++ 11 is recommended
- CMake
- CastXML
- Git and standard C/C++ build tools

Create and activate a compatible Conda environment:

```bash
conda create -n opencda_planning python=3.7 -y
conda activate opencda_planning
```

Verify the active Python interpreter:

```bash
python --version
command -v python
```

The reported version must be Python `3.7.x`.

## 3. Install Ubuntu Build Dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git curl castxml \
  gcc-11 g++-11 \
  libpugixml-dev libproj-dev libspdlog-dev libfmt-dev \
  libosmium2-dev liblapacke-dev libgtest-dev python3-dev
```

Verify the important build tools:

```bash
cmake --version
castxml --version
gcc-11 --version
g++-11 --version
```

A newer system-default GCC does not prevent the build, provided that `CASTXML_CXX`, `CC`, and `CXX` explicitly select a compatible compiler.

## 4. Build AD-map

Activate the environment and move to the repository root:

```bash
cd /home/umd-user/Downloads/CP-X-planning-module
conda activate opencda_planning
```

Run an initial build:

```bash
CASTXML_CXX=/usr/bin/g++-11 \
CC=/usr/bin/gcc-11 \
CXX=/usr/bin/g++-11 \
PYTHON_BIN="$(command -v python)" \
opencda/planning_module/Global_Planner/build_ad_map.sh
```

If an earlier build stopped with an error, remove its generated artifacts and rebuild:

```bash
CASTXML_CXX=/usr/bin/g++-11 \
CC=/usr/bin/gcc-11 \
CXX=/usr/bin/g++-11 \
PYTHON_BIN="$(command -v python)" \
opencda/planning_module/Global_Planner/build_ad_map.sh --clean
```

`--clean` deletes the generated `build`, `log`, and `install` directories but preserves the downloaded AD-map source checkout. It therefore avoids downloading the source and submodules again.

Use the following only if the source checkout itself is damaged:

```bash
opencda/planning_module/Global_Planner/build_ad_map.sh --clean-all
```

`--clean-all` also deletes the source checkout, so the next build requires network access.

The build script performs these steps:

1. Checks Python 3.7 and the required system tools.
2. Downloads or reuses the pinned AD-map v2.3.0 source.
3. Configures GCC/G++ 11 for CastXML.
4. Builds the bundled PROJ 4.9.3 dependency.
5. Builds Boost.Python for the active Python 3.7 ABI.
6. Builds `ad_map_access` with Colcon.
7. Installs the runtime under `Global_Planner/map_repo/install`.
8. Performs an `ad_map_access` import smoke test.

A successful build ends with output similar to:

```text
AD-map import OK: ...
AD-map runtime installed at .../Global_Planner/map_repo/install
```

Verify that the installation exists:

```bash
ls opencda/planning_module/Global_Planner/map_repo/install
```

## 5. Run the Standalone Global Planner

The planner core does not depend on CARLA, so a CARLA server is not required for this standalone test.

```bash
cd /home/umd-user/Downloads/CP-X-planning-module/opencda/planning_module/Global_Planner
conda activate opencda_planning
python global_planner/main.py
```

Example output from the default query:

```text
149.94323683720603
```

This value is the planned route length in meters. The example therefore represents a route of approximately `149.94 m`.

On the first map load, the planner creates AD-map and Python lane caches under `Global_Planner/cache/`. Later runs with the same map and configuration reuse those caches.

## 6. Change the Map, Start, or Goal

Edit:

```text
opencda/planning_module/Global_Planner/global_planner/main.py
```

Change these values as needed:

```python
XODR_PATH = PROJECT_ROOT / "maps" / "Town10HD_Opt.xodr"
START_POINT = {"x": -86.8, "y": 133.5, "z": 0.0}
GOAL_POINT = {"x": 59.4, "y": 137.8, "z": 0.0}
```

Public inputs and outputs use CARLA-style coordinates:

```python
{"x": ..., "y": ..., "z": ...}
```

The planner converts them to ENU coordinates internally:

```text
enu_x = carla_x
enu_y = -carla_y
enu_z = carla_z
```

## 7. Use the Planner from Python

Run this from the `Global_Planner` directory or from an environment where that directory is on `PYTHONPATH`:

```python
from global_planner import GlobalPlanner

planner = GlobalPlanner(
    "maps/Town10HD_Opt.xodr",
    cache_root="cache",
)

planner.load()
try:
    route = planner.trace_route(
        {"x": -86.8, "y": 133.5, "z": 0.0},
        {"x": 59.4, "y": 137.8, "z": 0.0},
    )
    print("Route length:", route.length_m)
    print("Sample count:", len(route.sampled_waypoints))
finally:
    planner.close()
```

A context manager can ensure that AD-map is closed correctly:

```python
from global_planner import GlobalPlanner

with GlobalPlanner("maps/Town10HD_Opt.xodr", cache_root="cache") as planner:
    route = planner.trace_route(
        {"x": -86.8, "y": 133.5, "z": 0.0},
        {"x": 59.4, "y": 137.8, "z": 0.0},
    )
    print(route.length_m)
```

## 8. Run It as Part of the Complete Planning Module

The complete scenario runtime uses the same Global Planner through `CustomGlobalPlannerAdapter`, together with the behavior planner, MPC, and CARLA scenario logic.

Start the CARLA server first. Then open another terminal and run:

```bash
conda activate opencda_planning
export CARLA_ROOT="/path/to/CARLA"
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/<matching-carla-egg>:$PYTHONPATH"

cd /home/umd-user/Downloads/CP-X-planning-module/opencda/planning_module
python main.py town10
```

The CARLA Python egg ABI must match the Python version in the active environment.

The `reroute_test` scenario is the older A* rerouting test. It is not the standalone entry point for the current custom Dijkstra-style Global Planner.

## 9. Troubleshooting

### 9.1 `Missing build command: cmake`

CMake is not installed:

```bash
sudo apt install -y cmake
```

### 9.2 `needs a GCC 9-12 C++ compiler`

CastXML cannot use the newer system compiler:

```bash
sudo apt install -y gcc-11 g++-11
```

Then explicitly select GCC/G++ 11 during the build:

```bash
CASTXML_CXX=/usr/bin/g++-11 \
CC=/usr/bin/gcc-11 \
CXX=/usr/bin/g++-11 \
PYTHON_BIN="$(command -v python)" \
opencda/planning_module/Global_Planner/build_ad_map.sh --clean
```

### 9.3 CMake 4 Compatibility Errors

Possible errors include:

```text
Compatibility with CMake < 3.5 has been removed
```

or:

```text
Policy CMP0022 may not be set to OLD behavior
```

The repository's `build_ad_map.sh` includes the required CMake 4 compatibility handling:

- It passes `CMAKE_POLICY_VERSION_MINIMUM=3.5` during configuration.
- It changes `CMP0022` to `NEW` in the ignored, pinned PROJ source checkout.

Use `--clean` to rebuild after updating the script.

### 9.4 `No module named 'global_planner.cache'`

The required planner cache source module is missing. The repository must contain:

```text
Global_Planner/global_planner/cache.py
```

Update the current branch or restore that file if it is absent.

### 9.5 `Could not find the AD-map install folder`

The AD-map build has not completed successfully, or the runtime was installed in a non-default location.

The default installation path is:

```text
Global_Planner/map_repo/install
```

For a custom installation path, set one of these environment variables:

```bash
export GLOBAL_PLANNER_AD_MAP_INSTALL="/path/to/install"
```

or:

```bash
export AD_MAP_INSTALL_ROOT="/path/to/install"
```

### 9.6 The Program Prints Only One Floating-Point Number

This is expected. `global_planner/main.py` is a minimal example that prints only:

```python
route.length_m
```

To inspect the generated route points, also print:

```python
print(route.sampled_waypoints)
```

## 10. Minimal End-to-End Procedure

Install dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git curl castxml gcc-11 g++-11 \
  libpugixml-dev libproj-dev libspdlog-dev libfmt-dev \
  libosmium2-dev liblapacke-dev libgtest-dev python3-dev
```

Build AD-map:

```bash
cd /home/umd-user/Downloads/CP-X-planning-module
conda activate opencda_planning

CASTXML_CXX=/usr/bin/g++-11 \
CC=/usr/bin/gcc-11 \
CXX=/usr/bin/g++-11 \
PYTHON_BIN="$(command -v python)" \
opencda/planning_module/Global_Planner/build_ad_map.sh --clean
```

Run the planner:

```bash
cd opencda/planning_module/Global_Planner
python global_planner/main.py
```

Successful example output:

```text
149.94323683720603
```
