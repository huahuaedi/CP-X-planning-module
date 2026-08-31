# M0 — global-route backend comparison (`dij` vs `astar`)

Goal of M0: produce a ranked list of scenarios where the **custom AD-map /
OpenDRIVE Dijkstra** planner (`global_planner_mode: dij`) disagrees with the
**legacy CARLA `GlobalRoutePlanner` + A\*** planner (`global_planner_mode:
astar`), so the AD-map junction / lane-change topology can be fixed *before*
`dij` is made the default and CARLA is removed from the route path.

Neither backend is ground truth. The diff just surfaces disagreement.

## Why two runs in two environments

`carla` has no wheel for Python 3.13 (the target ROS env), so the `astar`
backend can only run in the legacy `carla307` / Python 3.7 environment. The
`dij` backend needs the AD-map native runtime, which runs on Python 3.10-3.13.
So each backend is run where it works and the two JSON outputs are diffed
anywhere.

```
run_backend.py --backend dij   ->  out_dij.json     (new env)
run_backend.py --backend astar ->  out_astar.json   (legacy carla env)
diff_backends.py               ->  M0_REPORT.md     (anywhere)
```

## 1. `dij` side (new env)

The AD-map runtime can come from either source:

- **pip wheel (supported path, verified working):**
  ```bash
  python -m venv .venv && . .venv/bin/activate
  pip install ad-map-access==3.0.0 numpy pyyaml
  python run_backend.py --backend dij --out out_dij.json
  ```
  `runtime.import_ad_map_access()` now falls back to an importable
  `ad_map_access` (the wheel bundles its own native libs) when no
  folder-layout install (`GLOBAL_PLANNER_AD_MAP_INSTALL` /
  `AD_MAP_INSTALL_ROOT` / `map_repo/install`) is found.

- **built runtime:** point `GLOBAL_PLANNER_AD_MAP_INSTALL` at an install root
  (the folder-layout bootstrap then takes precedence over the wheel). The
  committed `build_ad_map.sh` still targets Python 3.7 and does not build for
  3.10-3.13 — prefer the wheel.

## 2. CARLA side (legacy carla env)

Two backends, both build an **offline** `carla.Map(name, xodr_string)` from
`Global_Planner/maps/<map>.xodr` (no CARLA server needed).

### `carla_grp` — recommended baseline

```bash
python run_backend.py --backend carla_grp --out out_carla_grp.json
```

Uses the **vendored** `opencda.core.plan.global_route_planner.GlobalRoutePlanner`
— exactly what `route_manager.py::_build_carla_route` uses to drive route
geometry today, in both `astar` and `dij` modes. Only needs `carla` +
`networkx` (no `agents`), old-style `GlobalRoutePlannerDAO(wmap, res)`, so it is
version-robust. This is the most relevant "what CARLA gives us" reference.

### `astar` — the `global_planner_mode: astar` backend

```bash
python run_backend.py --backend astar --carla-root /path/to/CARLA \
    [--agents-path /path/to/PythonAPI/carla] --out out_astar.json
```

Runs `AStarGlobalPlanner` (the factory's `astar` backend). Additionally needs
CARLA's bundled `agents.navigation.global_route_planner` on `sys.path`
(`<carla-root>/PythonAPI/carla/agents/…`, or point `--agents-path` at the dir
that contains `agents/`). NOTE its `_build_carla_route_planner` calls the
*new* `GlobalRoutePlanner(wmap, res)` signature — fails on CARLA < 0.9.13.

Diff either one against dij:
```bash
python diff_backends.py --dij out_dij.json --astar out_carla_grp.json --out M0_REPORT.md
```

## 3. diff

```bash
python diff_backends.py --dij out_dij.json --astar out_astar.json --out M0_REPORT.md
```

Per-case verdict: `MATCH` / `MINOR` / `DIVERGENT` / `DIJ-FAIL` / `ASTAR-FAIL`,
ranked worst first. Signals: route length %, max lateral corridor deviation,
lane-change count delta, per-backend route-point gap count.

## 4. same-environment validation (rules out a wheel-version artifact)

The first `dij` run used the `ad-map-access 3.0.0` wheel on Python 3.13; the
CARLA baseline ran on Python 3.7. To confirm the divergences are real planner
behaviour and not an AD-map 3.0.0-vs-2.3.0 difference, build AD-map 2.3.0 in
the Python 3.7 CARLA env and re-run `dij` there:

```bash
# in / for the py3.7 carla env (e.g. miniforge3/envs/opencda_planning)
sudo apt install build-essential cmake git curl castxml libpugixml-dev \
  libproj-dev libspdlog-dev libfmt-dev libosmium2-dev liblapacke-dev libgtest-dev
PYTHON_BIN=/home/umd-user/miniforge3/envs/opencda_planning/bin/python \
  opencda/planning_module/Global_Planner/build_ad_map.sh --clean       # ~30-60 min

# then, with that env active (map_repo/install now exists, beats the wheel):
python run_backend.py --backend dij --out out_dij_py37.json
python diff_backends.py --dij out_dij_py37.json --astar out_carla_grp.json \
  --out M0_REPORT_py37.md
```

Same DIVERGENT cases in `M0_REPORT_py37.md` → planner-logic issues, proceed to
M1. Different → the wheel is not equivalent to the source build; resolve first.

## cases.json

`start`/`goal` are CARLA world `[x, y, z]` from each scenario dir yaml
(`ego_spawn_xyz_yaw` / `final_destination_xyz_yaw`). Scenarios whose anchors
live only as CARLA level markers (all `cpx_*` scenario_testing yamls, the
`mdrive_*` ones, `town6_scenario_1`) are in `unresolved[]` — to include them,
capture the runtime-resolved anchors from a `cpx_scenario_bridge` log and add
them as cases.

## Status / caveats

- `dij` side: **runs, 10/10 cases trace** via the pip wheel, through
  `CustomGlobalPlannerAdapter` (same entry point `route_manager.py` uses), so
  `road_option_seq` carries the real macro labels
  (`LANEFOLLOW` / `STRAIGHT` / `CHANGELANELEFT` / `CHANGELANERIGHT`).
  Route search is resolution-stable (identical lane_seq + length at 1.0 m and
  2.0 m sampling).
- `Town10HD_Opt` loads with `generateCenterLine() Invalid geometry definition`
  warnings from AD-map (3×) — a candidate root cause for junction topology
  disagreement; not yet investigated.
- Early signal from the dij run: `town10_cp_passing_merge` /
  `town10_cp_roadway_object` route inserts **3 lane changes** over 363 m;
  worth checking against astar for spurious lane changes.
- CARLA side: **not yet captured.** First `astar` run failed 10/10 —
  `agents.navigation.global_route_planner` was not importable
  (`No module named 'agents'`), so `AStarGlobalPlanner._carla_route_planner`
  was None and every route returned `route_found=false`. Offline
  `carla.Map` itself works (`generate_waypoints` = 3066 on Town10HD, start
  map-match = road 5 / lane -1, matching dij). Use `--backend carla_grp`
  (no `agents` dependency) or pass a real `--carla-root` / `--agents-path`.
- `run_backend.py`'s `lane_id_renumber_events` counter also counts legitimate
  lane changes; cross-check against `n_lane_changes` / `transition_types`.

## carla_free_route_smoke.py

Separate check (not a dij-vs-CARLA diff): proves the route + reference layer
runs with **no CARLA at all**. Per Town10 case it builds
`CustomGlobalPlannerAdapter` + `CPXRouteManager(carla_map=None, carla_api=None)`
+ `ReferenceGenerator` (adapter as `map_planner`, adapter waypoint queries as
the map callbacks) and asserts:

- `set_destination` finds the route; `get_route_info` remaining distance is
  monotonic; `geometry_route_points` / `carla_waypoint_reference` non-empty and
  near ego; `upcoming_turn` does not raise.
- `ReferenceGenerator.build_route_reference` returns >=2 samples near ego with
  finite headings; `lane_center_samples` returns >=2.

10/10 pass. MPC tracking of a reference is already covered CARLA-free by
`tests/test_mpc_lane_reference.py` etc.

## carla_free_runstep_smoke.py

End-to-end: constructs the full `CPXMPCPlannerBridge` with a
`CustomGlobalPlannerAdapter` as `map_planner` and a minimal fake
`vehicle_manager` (SimpleNamespace tree), `set_destination`, then 12 ticks of
`update_information()` + `run_step()`. Asserts every tick returns a finite
`VehicleControl` and the route came from the in-house path
(`route_manager.carla_route_debug_reason == "inhouse_route_ready"`).

Needs `pip install osqp scipy` (MPC QP solver). 10/10 construct + run.
6/10 produce control with zero fallback ticks; 4 (the junction-heavy
`passing_merge` / `roadway_object` / `gp_readme` and `scenario_2/3`) hit the
planner's own reference/feasibility gates on some ticks — an artifact of the
crude smoke (ego snapped exactly onto route points, fixed 5 km/h, no dynamics),
not a CARLA-coupling failure.

