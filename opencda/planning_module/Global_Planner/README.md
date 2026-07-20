# CARLA-Independent Global Planner

This folder contains the custom OpenDRIVE global planner and its reproducible AD-map build.

## Included

```text
Global_Planner/
├── global_planner/
├── maps/
├── build_ad_map.sh
├── map_repo/
└── README.md
```

- `global_planner/`
  The planner package itself.
- `build_ad_map.sh`
  Downloads the pinned AD-map v2.3.0 source and builds bindings for Python 3.7.
- `map_repo/install/`
  Generated locally by the build script; it is intentionally not committed.
- `map_repo/log/`
  Colcon build logs; generated locally and intentionally not committed.
- `map_repo/source/`
  Downloaded third-party AD-map source checkout; generated locally and intentionally not committed.

## Not Included

Do not push these directories:

```text
map_repo/install/
map_repo/log/
map_repo/source/
```

They are build/download artifacts, not repository source. `install/` contains
native Python extensions and shared libraries tied to the local Python ABI,
operating system, compiler, dependency versions, and absolute build paths.
`log/` is disposable build output. `source/` is the pinned upstream AD-map
checkout downloaded by `build_ad_map.sh`; preserving it locally makes rebuilds
faster, but it should not be copied through this repository.

After cloning this repository on a new machine, rebuild the runtime locally with
the command below.

## Build after cloning

Activate the `carla307` Python 3.7 environment that will run OpenCDA, then run from the repository root:

```bash
PYTHON_BIN="$(command -v python)" opencda/planning_module/Global_Planner/build_ad_map.sh --clean
```

The script checks the Python ABI and development headers, clones the pinned upstream source, builds the Python bindings, installs them under `map_repo/install`, and performs an import smoke test.

`--clean` removes build/install output but preserves the downloaded
`map_repo/source` checkout, so a rebuild does not depend on GitHub being
available again. Use `--clean-all` only when that source checkout must also be
deleted and downloaded again. If the first download reports `Could not resolve
host: github.com`, restore DNS/network access and rerun the same command.

Do not copy or commit `map_repo/source`, `build`, `log`, or `install`; those trees contain generated and machine-specific files.

## If Your Other Project Uses A Different Runtime Location

If you do not want to keep `map_repo/install` at that default location, you can still use the planner by passing:

```python
GlobalPlanner(..., ad_map_install_root="/path/to/install")
```

or by setting either environment variable:

- `GLOBAL_PLANNER_AD_MAP_INSTALL`
- `AD_MAP_INSTALL_ROOT`

## Basic Usage

```python
from global_planner import GlobalPlanner

planner = GlobalPlanner(
    xodr_path="/path/to/your_map.xodr",
    cache_root="/path/to/cache",
)

planner.load()
route = planner.trace_route(
    {"x": -86.8, "y": 133.5, "z": 0.0},
    {"x": 59.4, "y": 137.8, "z": 0.0},
)
print(route.length_m)
planner.close()
```

## Notes

- The planner package itself is small.
- The main extra runtime dependency is the built AD-map install.
- The bundle does not require CARLA.
- For full planner API details, read `global_planner/README.md`.
