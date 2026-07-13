# Global Planner Package

This folder contains the standalone global planner package used by this project.

The package is CARLA-independent in its core logic:
- it does not import `carla`
- it reads OpenDRIVE through AD-map
- it snaps positions to lane centers
- it finds a lane-level shortest route
- it returns route points and waypoints in CARLA-style coordinates

The public package entry point is:

```python
from global_planner import GlobalPlanner
```

## What This Package Does

Given a start position and a goal position, the planner:

1. converts the input position from CARLA-style coordinates to AD-map ENU
2. finds nearby drivable lane matches
3. snaps the query to the closest lane-center point
4. searches a legal lane-by-lane route using a custom shortest-path search
5. builds the final route geometry, including lane changes
6. samples the final route into output waypoints
7. returns those route points back in CARLA-style coordinates

This package also supports:
- nearest lane-center waypoint lookup
- adjacent-lane waypoint lookup
- forward and backward waypoint stepping
- lane-centerline extraction for any cached lane

## Folder Contents

```text
global_planner/
├── __init__.py
├── admap_backend.py
├── cache.py
├── geometry.py
├── main.py
├── planner.py
├── README.md
├── route.py
├── runtime.py
└── waypoint.py
```

- `planner.py`
  Main public `GlobalPlanner` class and the custom route-search logic.
- `waypoint.py`
  `Waypoint` wrapper with lane and movement helpers such as `left()`, `right()`, `next()`, and `previous()`.
- `route.py`
  `Route` wrapper returned by `trace_route()`.
- `runtime.py`
  Automatically prepares the AD-map runtime so you do not need to manually source shell scripts before importing the planner.
- `admap_backend.py`
  Thin helper layer over `ad_map_access`.
- `geometry.py`
  Coordinate conversion and general point helpers.
- `cache.py`
  Cache metadata and planner cache file handling.
- `main.py`
  Small direct-run example entry point.

## Requirements

This package requires:
- a built AD-map install under `map_repo/install/`, or another valid install path
- the Python bindings for `ad_map_access`
- the compiled AD-map shared libraries
- a valid `.xodr` OpenDRIVE map file

You do not need CARLA to use this package.

## Coordinate Convention

Public input and output use CARLA-style coordinates:

```python
{"x": ..., "y": ..., "z": ...}
```

Internal route computation uses ENU coordinates.

The conversion used by this package is:

```text
enu_x = carla_x
enu_y = -carla_y
enu_z = carla_z
```

So:
- outside the package, use CARLA-style points
- inside the package, route logic works in ENU

## Quick Start

Run from the project root:

```python
from global_planner import GlobalPlanner

planner = GlobalPlanner("maps/Town10HD_Opt.xodr", cache_root="cache")
planner.load()

route = planner.trace_route(
    {"x": -86.8, "y": 133.5, "z": 0.0},
    {"x": 59.4, "y": 137.8, "z": 0.0},
)

print("Route length:", route.length_m)
print("Number of points:", len(route.sampled_waypoints))

planner.close()
```

You can also use the context-manager form:

```python
from global_planner import GlobalPlanner

with GlobalPlanner("maps/Town10HD_Opt.xodr", cache_root="cache") as planner:
    route = planner.trace_route(
        {"x": -86.8, "y": 133.5, "z": 0.0},
        {"x": 59.4, "y": 137.8, "z": 0.0},
    )
    print(route.length_m)
```

## Direct Script Run

This folder also contains `main.py` as a minimal example.

From inside `global_planner/`:

```bash
python main.py
```

That script:
- creates a planner
- loads the map
- finds one route
- prints the route length

## Public API

### `GlobalPlanner`

Constructor:

```python
GlobalPlanner(
    xodr_path,
    cache_root="cache",
    centerline_spacing_m=3.0,
    route_engine="custom",
    default_search_radius_m=8.0,
    default_lane_change_penalty_m=2.0,
    lane_change_distance_m=8.0,
    overlap_margin=0.05,
    ad_map_install_root=None,
)
```

Important constructor arguments:

- `xodr_path`
  Path to the OpenDRIVE file.
- `cache_root`
  Root folder where the `.adm` cache and Python planner cache are stored.
- `centerline_spacing_m`
  Spacing used when building cached lane-centerline samples.
- `default_search_radius_m`
  Default map-matching radius for start, goal, and waypoint queries.
- `default_lane_change_penalty_m`
  Fixed cost added to each lane change during route search.
- `lane_change_distance_m`
  Controls how far along a lane the planner tries to place a lane change.
- `overlap_margin`
  Margin passed into AD-map OpenDRIVE loading.
- `ad_map_install_root`
  Optional explicit path to the AD-map install folder if it is not at the default location.

### `load(force_rebuild=False) -> None`

Loads the map and planner cache.

Behavior:
- if a valid cache already exists, it loads the saved `.adm` map and Python lane cache
- otherwise it loads the `.xodr`, builds the lane cache, and saves both caches

You must call `load()` before route or waypoint queries.

### `close() -> None`

Unloads the active AD-map singleton state.

Call this when finished with the planner.

### `get_waypoint(position, search_radius_m=None) -> Waypoint | None`

Returns the nearest lane-center waypoint for the given CARLA-style position.

Input:
- `position` may be a dict like `{"x": ..., "y": ..., "z": ...}` or a tuple/list `(x, y, z)`
- `search_radius_m` overrides the default search radius

Output:
- a `Waypoint` on the chosen lane center
- `None` if no drivable lane is found nearby

Important detail:
- this does not just return a point on the lane surface
- it returns the closest point on the lane center of the selected matched lane

Example:

```python
waypoint = planner.get_waypoint({"x": -86.8, "y": 133.5, "z": 0.0})
print(waypoint.position)
print(waypoint.road_id, waypoint.lane_id)
```

### `get_lane_centerline(lane_id) -> list[Waypoint]`

Returns the cached lane-centerline waypoints for one AD lane id.

Example:

```python
centerline = planner.get_lane_centerline(380047)
for waypoint in centerline[:5]:
    print(waypoint.position)
```

### `trace_route(start, goal, sampling_resolution_m=3.0, search_radius_m=None, lane_change_penalty_m=None, goal_append_distance_threshold_m=1.0) -> Route`

Finds the shortest custom route between two positions.

Input:
- `start`
- `goal`
- `sampling_resolution_m`
  Distance between sampled output points
- `search_radius_m`
  Search radius for start and goal lane candidates
- `lane_change_penalty_m`
  Fixed lane-change cost override
- `goal_append_distance_threshold_m`
  If the sampled route ends farther than this distance from the resolved goal waypoint, that goal waypoint is appended once at the end

Output:
- a `Route` object

Example:

```python
route = planner.trace_route(
    {"x": -86.8, "y": 133.5, "z": 0.0},
    {"x": 59.4, "y": 137.8, "z": 0.0},
)

print(route.length_m)
print(route.lane_path)
print(route.to_point_dicts()[:3])
```

### `to_enu(position) -> tuple[float, float, float]`

Converts a CARLA-style point into ENU.

### `to_carla(position) -> dict[str, float]`

Converts an ENU point back into CARLA-style coordinates.

## `Waypoint` Object

Returned by:
- `get_waypoint()`
- `trace_route().resolved_start`
- `trace_route().resolved_goal`
- `trace_route().sampled_waypoints`
- `get_lane_centerline()`

Main fields:
- `position`
  CARLA-style point dict
- `enu_position`
  Internal ENU tuple
- `ad_lane_id`
  AD-map lane id
- `road_id`
  Decoded OpenDRIVE road id
- `section_id`
  Decoded lane-section index
- `lane_id`
  Decoded OpenDRIVE lane id
- `parametric_offset`
  Longitudinal location on the lane from `0.0` to `1.0`
- `heading`
  Lane heading at the waypoint if available
- `lane_length_m`
  Full lane length
- `lane_width_m`
  Width at that waypoint if available
- `is_intersection`
  Whether the lane belongs to an intersection

### `waypoint.left() -> Waypoint | None`

Returns the nearest same-direction waypoint on the left adjacent lane, using the same longitudinal offset.

### `waypoint.right() -> Waypoint | None`

Returns the nearest same-direction waypoint on the right adjacent lane, using the same longitudinal offset.

### `waypoint.next(distance_m) -> list[Waypoint]`

Moves forward by the requested distance along the lane network.

Important detail:
- if the lane continues into multiple possible successor lanes, this can return multiple waypoints

### `waypoint.previous(distance_m) -> list[Waypoint]`

Moves backward by the requested distance along the lane network.

### `waypoint.to_dict() -> dict`

Returns a JSON-friendly summary of the waypoint.

Example:

```python
waypoint = planner.get_waypoint({"x": -86.8, "y": 133.5, "z": 0.0})
print(waypoint.left())
print(waypoint.next(10.0))
print(waypoint.to_dict())
```

## `Route` Object

Returned by `trace_route()`.

Main fields:
- `raw_start`
  Original input start position
- `raw_goal`
  Original input goal position
- `resolved_start`
  Snapped lane-center waypoint used as the route start
- `resolved_goal`
  Snapped lane-center waypoint used as the route goal
- `lane_path`
  List of AD lane ids along the selected route
- `sampled_waypoints`
  Final output route waypoints
- `length_m`
  Analytical route length in meters
- `sampling_resolution_m`
  Sampling resolution used for output points
- `transition_types`
  Per-transition labels such as `"forward"` or `"lane_change"`

### `route.to_point_dicts() -> list[dict[str, float]]`

Returns the sampled route points as CARLA-style point dictionaries.

## Runtime Helpers

The package exports two advanced helpers from `runtime.py`:

### `prepare_ad_map_runtime(ad_map_install_root=None) -> Path`

Prepares the Python path and shared-library path for AD-map.

### `import_ad_map_access(ad_map_install_root=None)`

Prepares the runtime and returns the imported `ad_map_access` module.

Most users do not need to call these directly because `GlobalPlanner` already does it automatically.

## How The Planner Works

### 1. Runtime bootstrap

When `GlobalPlanner` is created, the package automatically:
- finds the AD-map install folder
- adds the AD-map Python packages to `sys.path`
- preloads the required shared libraries
- imports `ad_map_access`

This is why you no longer need to manually source `activate.sh` just to use the planner.

### 2. Map load and cache reuse

When `load()` is called:
- the planner computes a signature for the `.xodr` file
- it checks whether a matching cache already exists
- if yes, it loads the saved `.adm` map and Python lane cache
- if not, it parses the `.xodr`, builds the lane cache, and saves both

The cache key depends on:
- the map file content
- the map file metadata
- the lane-centerline spacing
- the planner cache version

### 3. Lane cache build

For each routable lane, the planner stores:
- decoded road, section, and lane ids
- lane length
- lane direction
- intersection flag
- same-direction left and right adjacent lanes
- forward connected lanes
- backward connected lanes
- sampled lane-center points

This makes later waypoint and route queries much faster.

### 4. Waypoint snapping

For a query position:
- AD-map finds nearby drivable lane matches
- the planner samples the lane center at each matched longitudinal position
- the candidates are sorted mainly by true distance from the raw point to that lane-center point

The sort preference is:
1. smaller lane-center snap distance
2. matches that are actually inside the lane
3. higher match probability
4. lane id as a stable tie-breaker

The best candidate becomes the snapped waypoint.

### 5. Lane-level shortest-path search

The planner runs a custom lane-level shortest-path search with a priority queue.

It is effectively a Dijkstra-style search because:
- it uses accumulated cost only
- it does not use a heuristic term

Allowed transitions:
- forward continuation into connected same-direction lanes
- lane changes into same-direction adjacent lanes

Blocked transitions:
- non-routable lanes
- backward travel on a lane
- lane changes inside intersections
- lane changes between opposite-direction lanes

### 6. Cost model

The route cost is:
- forward transition cost = full length of the next lane
- lane-change transition cost = fixed `lane_change_penalty_m`

This means:
- a low lane-change penalty makes lane changes easier to choose
- a high lane-change penalty makes the planner stay in the current lane longer

### 7. Route geometry build

After the lane path is found, the planner converts it into route steps:
- lane segments along lane centers
- lane-change connectors between lanes

Current lane-change behavior:
- earlier lane changes are diagonal
- the last two lane changes are lateral

That behavior comes from the route-building logic in `planner.py`.

### 8. Route sampling

Finally, the planner samples the route into output waypoints:
- lane segments are sampled at the requested spacing
- lane changes are sampled by interpolating between the two lane-center connector endpoints
- duplicate points are skipped

The planner also ensures:
- the resolved start waypoint is included at the beginning if needed
- the resolved goal waypoint is appended at the end if the final sampled point is farther than the threshold

## Route Length

`Route.length_m` is computed analytically from the route steps:
- lane-segment length = traveled fraction of lane length
- lane-change length = Euclidean distance between the connector start and end points

This is not simply the sum of the sampled output points, although the two are usually close.

## Cache Files

When a cache is built, the planner writes files like:

```text
cache/<map_name>_<hash>/
├── map.adm
├── map.adm.txt
├── metadata.json
└── planner_cache.pkl
```

- `map.adm`
  Saved AD-map binary data
- `map.adm.txt`
  AD-map config file used to reload the cached map
- `planner_cache.pkl`
  Python-side lane metadata and centerline cache
- `metadata.json`
  Cache validation metadata

## Common Usage Examples

### Find the nearest lane-center waypoint

```python
waypoint = planner.get_waypoint({"x": 10.0, "y": 20.0, "z": 0.0})
if waypoint is not None:
    print(waypoint.position)
    print(waypoint.road_id, waypoint.lane_id)
```

### Move to adjacent lanes

```python
left_waypoint = waypoint.left()
right_waypoint = waypoint.right()
```

### Move forward or backward

```python
next_waypoints = waypoint.next(15.0)
previous_waypoints = waypoint.previous(15.0)
```

### Get a lane centerline

```python
centerline = planner.get_lane_centerline(waypoint.ad_lane_id)
for point in centerline[:10]:
    print(point.position)
```

### Get route output points only

```python
route = planner.trace_route(start_point, goal_point)
point_list = route.to_point_dicts()
```

## Important Notes

- Always call `load()` before `get_waypoint()`, `get_lane_centerline()`, or `trace_route()`.
- Always call `close()` when finished, unless you use the context-manager form.
- This package uses AD-map global state, so it is effectively single-map-at-a-time inside one Python process.
- Public inputs and outputs use CARLA-style coordinates even though the package itself is CARLA-independent.
- The planner returns lane-center waypoints, not arbitrary road-surface points.
- If no legal route exists between nearby snapped candidates, `trace_route()` raises a `RuntimeError`.

## Recommended Entry Point For Normal Use

For most users, the only class you need is:

```python
from global_planner import GlobalPlanner
```

Then use:
1. `planner = GlobalPlanner(...)`
2. `planner.load()`
3. `planner.get_waypoint(...)` or `planner.trace_route(...)`
4. `planner.close()`
