# Independent Global Planner

This folder contains the minimum parts from this repository that are needed to run the CARLA-independent global planner in another Python project.

## Included

The planning stack imports this package as `planner.Global_Planner`.

- `global_planner/`
  The planner package itself.
- `map_repo/install/`
  The built AD-map runtime and Python bindings required by the planner.

## Not Included

This bundle does not include:
- `maps/`
- `cache/`
- `examples/`
- CARLA-specific utilities
- the full `map_repo` source tree

That means the target project must provide:
- the `.xodr` map file path
- a cache directory path if caching is desired

The bundled native runtime must contain Python 3.7 bindings for the OpenCDA
environment. A binding built for Python 3.12 cannot be loaded by Python 3.7.

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
from planner.Global_Planner import GlobalPlanner

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
