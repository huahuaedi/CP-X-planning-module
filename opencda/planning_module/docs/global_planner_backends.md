# Global Planner Backends

The planning module supports two global-planner backends. They are selected by
the scenario YAML field:

```yaml
planning:
  global_planner_mode: astar
```

## Backend A: legacy CARLA/A* baseline

Modes:

- `astar`
- `legacy`
- `carla_grp`

Purpose:

- Baseline for comparison experiments.
- Uses CARLA map waypoints and the legacy A* route planner.
- Does not require the AD-map runtime.
- Returns CARLA `Waypoint` objects internally.

Recommended use:

- Use this when comparing against the previous planning behavior.
- Use this when running in the Python 3.7 CARLA/OpenCDA environment.

## Backend B: custom OpenDRIVE/AD-map planner

Modes:

- `custom`
- `custom_admap`
- `admap`
- `opendrive`

Purpose:

- Experimental custom global planner.
- Uses OpenDRIVE geometry through the compiled AD-map runtime.
- Returns custom planner `Waypoint` objects internally.

Runtime requirement:

```bash
PYTHON_BIN="$(command -v python)" opencda/planning_module/Global_Planner/build_ad_map.sh --clean
```

or set:

```bash
export GLOBAL_PLANNER_AD_MAP_INSTALL=/path/to/ad_map/install
```

Notes:

- The bundled AD-map build script requires Python 3.10-3.13.
- Do not use this backend in the Python 3.7 CARLA environment unless a
  compatible AD-map runtime is available.

## Shared interface

Both backends must expose the runner-facing methods used by behavior planning,
reference generation, and MPC:

- `get_waypoint()`
- `get_local_lane_context()`
- `get_current_route_info()`
- `plan_route_from_locations()`
- `trace_route()`
- `register_imported_route()`
- `block_lane_at_position()`
- `load()`
- `close()`

Downstream modules should not directly assume `waypoint.position` or
`waypoint.ad_lane_id`. CARLA waypoints and custom AD-map waypoints use different
object layouts, so waypoint access must go through the shared helper functions.

## Comparison workflow

Run the same scenario twice, changing only:

```yaml
planning:
  global_planner_mode: astar
```

versus:

```yaml
planning:
  global_planner_mode: custom
```

Then compare:

- route shape and selected lane
- temporary destination stability
- reference tracking error
- MPC solve status and solve time
- collisions / boundary breaches / TTC / DRAC
- stop-line behavior near traffic lights
