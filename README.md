# CP-X Planning Module

CP-X is an OpenCDA-based decision and trajectory-planning stack for CARLA,
cooperative perception, connected-vehicle intent sharing, probabilistic motion
prediction, and MPC trajectory generation. The supported CARLA target is
OpenCDA/CARLA 0.9.12 with Python 3.7.

`opencda.py` is the only supported OpenCDA scenario entry point. MDrive is an
external runtime adapter and does not modify MDrive source files; see
[mdrive_adapter/README.md](mdrive_adapter/README.md).

## Frozen planner baseline

The planner core is frozen at the behavior represented by commit `125c723`
(`Wire MDrive peer intents and static GT`). Commit `6aaae98` safely reverted a
later maneuver-lifecycle experiment, so the `convert_to_ros` planner tree is
equivalent to that baseline. Documentation and transport-adapter work may
continue without changing the frozen planning semantics.

Validated on the frozen baseline:

- 1,530 planning-module tests pass in `opencda_planning`;
- isolated left turn, right turn, and 12 m/s bidirectional lane change complete
  with no collision, fallback, MPC infeasibility, or final-reference hard gate;
- the two-CAV merge completes for both vehicles with deterministic
  `proceed`/`make_gap` arbitration and active longitudinal corridor rows in the
  MPC QP;
- the MTR-backed two-CAV merge completes without blocking the control loop;
- MDrive accepts peer intents and static ground-truth obstacles through its
  adapter.

This is a validated fixed-scenario baseline, not a claim of unrestricted
production autonomy. In particular, the merge test confirms MTR transport and
cooperative planning stability, but its fresh peer `planned_path` has priority
over model hypotheses; use a non-cooperative cut-in/crossing fixture to prove
that multiple MTR modes change the resulting plan.

## System architecture

```text
OpenCDA / MDrive / ROS 2 Bridge
        |
        | ego state, map/route, perceived objects, CP messages,
        | peer intents, prediction modes, safety state
        v
Runtime input adapter
  OpenCDAPlanningAdapter or simulator-neutral update_external_information()
        v
PlannerInputFrame
        |
        +--> Perception + tracking + local-map snapshot
        +--> AD-map RouteManager (route and monotonic progress)
        +--> Prediction bridge (CV, HTTP MTR, or in-process MTR)
        v
Scenario context -> Behavior candidates -> SpeedTargetPlanner
        v
CAV interaction pipeline
  Stage A classify -> Stage B arbitrate -> Stage C corridor
        v
ReferenceGenerator -> ReferencePipeline -> FinalReferenceGate
        v
Stage D corridor/keep-out rows -> MPC QP -> MPCControlBuffer
        v
Velocity/steering adapter -> SafetySupervisor
        v
PlannerOutput
  control + behavior + reference + planned path + predictions + diagnostics
```

The planner core does not own simulation, sensor drivers, neural-network model
loading, ROS spinning, or direct V2X transport. Those responsibilities end at
the input adapter. The same normalized planner contract is used by native
OpenCDA, the MDrive adapter, and future ROS nodes.

### Planner input and output contracts

`PlannerInputFrame` is the boundary between data providers and the planner. It
contains ego state, timestamp, destination and traffic-control context; current
map/lane/route state; perceived dynamic and static objects; prediction model,
revision, trajectories, and probabilistic modes; CP lane events, controls, and
obstacles; and CAV peer poses, claims, and broadcast `planned_path (t,x,y,v)`.

`CPXMPCPlannerBridge.update_information()` adapts native OpenCDA state.
`update_external_information()` accepts already-normalized simulator-neutral
state for MDrive/ROS integration without imitating an OpenCDA VehicleManager.

`PlannerOutput` returns the platform control object, typed behavior command,
accepted reference, MPC planned trajectory, predictions used by the tick,
acceleration/road-wheel steering commands, and structured diagnostics. OpenCDA
consumes `control`; integration code should use the typed output instead of
reading behavior back from debug strings.

### Cooperative interaction pipeline

1. **Stage A — classification:** project every retained agent trajectory onto
   the ego reference and classify it as `IGNORE`, `FOLLOW`, `LEAD_BRAKE`,
   `CUT_IN`, `MERGE`, `CROSSING`, or `ONCOMING`.
2. **Stage B — arbitration:** cooperative peers with active resource claims are
   assigned `proceed`, `yield`, or `make_gap`; role and pass-side hysteresis
   prevents tick-to-tick flips.
3. **Stage C — space-time corridor:** translate the interaction into
   longitudinal `s_lo[k]`/`s_hi[k]` bounds. Probabilistic modes use expected
   progress plus a credible-danger veto.
4. **Stage D — MPC constraints:** convert corridor bounds and pass-side choices
   into soft linear QP rows. Penalized slack keeps the QP feasible; a clear
   scene contributes no cooperative constraint.

A peer carrying a fresh broadcast plan replaces its perception prediction for
that actor. Other road users retain prediction-module hypotheses. Agent and
mode budgets are applied only after Stage A severity classification, so a
distant future crossing conflict is not removed merely because it is not
currently close.

### Prediction integration

```text
CP-X Planner -- PredictionScene (history + map + session_id) --> MTR ROS node
CP-X Planner <-- PredictionFrame (six modes per object) ------- MTR ROS node
      |
      v
PredictionContext -> Stage A/C -> Stage D -> MPC
```

Supported simulation transports are localhost HTTP, direct in-process import,
and typed ROS 2 topics. The ROS adapter publishes `PredictionScene`, caches
asynchronous `PredictionFrame` results, and translates all modes into the same
normalized planner contract. Rate limiting, staleness checks, time rebasing,
mode probabilities, and risk policy remain outside the transport boundary;
the ROS node does not copy planning policy.

### Runtime cadence

| Component | Nominal cadence | Behavior between updates |
|---|---:|---|
| CARLA / execution / safety | 20 Hz | Consumes the newest buffered MPC solution every tick |
| Stage A conflict geometry | 20 Hz | Re-evaluates geometry as ego and agents move |
| CAV Stage B role arbitration | 5 Hz or claim/topology event | Holds the latched role |
| Stage C corridor | 5 Hz, plus tag/role changes | Time-rebases cached bounds |
| MTR history sampling | 10 Hz | Deduplicated once per world tick across CAVs |
| MTR inference request | 5 Hz | Asynchronous latest-result cache with stale timeout |
| MPC trajectory generation | 5 Hz by default | Buffer serves time-aligned commands between solves |

The MPC horizon is adaptive by behavior, normally bounded between 1 and 5 s,
with a 0.1 s discretization. These are horizon samples, not simulator ticks.

## ROS 2 closed-loop integration

The current implementation is a working ROS 2 Jazzy closed loop, not only a
message-definition prototype. OpenCDA, the Planner, and MTR intentionally run
in separate Python environments:

| Runtime | Python | Responsibility |
|---|---:|---|
| OpenCDA host environment | 3.7 | CARLA ticks, simulated perception, scenario actors, actuator application |
| ROS 2 Jazzy container | 3.12 | TCP/ROS Bridge, exact-frame synchronization, per-CAV Planner workers, control return |
| MTR virtual environment | 3.12 | CUDA inference and six-mode `PredictionFrame` publication |

Python versions do not have to match across ROS nodes. ROS messages are the
ABI boundary. The OpenCDA side never imports `rclpy`, and the Planner process
never imports the CARLA Python 3.7 runtime.

### Data flow and ownership

```text
OpenCDA/CARLA at 20 Hz
  localization + perception + traffic light + V2X + cooperative messages
             |
             | TCP 5051-5055, one session-tagged simulation cycle
             v
Unified OpenCDA Bridge
  independent TCP input worker -> typed ROS topics
             |
             v
Planner shard selected by session_id
  exact-frame buffer -> isolated route/behavior/reference/MPC state
             |
             +---- PredictionScene ----> MTR node at 5 Hz inference
             |<--- PredictionFrame ----- six modes per tracked object
             |
             v
PlannerOutput -> ROS control topic -> Bridge -> TCP 5060 -> OpenCDA actuator
```

Each CAV owns an independent Planner runtime. Route state, Behavior FSM,
reference memory, prediction cache, MPC warm start, and control buffer are not
shared between CAV sessions. The returned control keeps its `session_id`, so
the Bridge sends it only to the matching vehicle.

Simulator `SafetyManager` telemetry is deliberately not a Planner ROS input.
It is evaluation data specific to OpenCDA. Deployment safety must instead be
provided by the vehicle/platform safety supervisor and real sensor/status
interfaces.

### One-time configuration

Copy `CP-ROS-WS/.env.example` to `CP-ROS-WS/.env` and update installation
paths. CPU isolation is optional, but it is recommended when CARLA, MTR, and
multiple Planner workers share one workstation. The validated 24-core layout
is:

```bash
CPX_MTR_CPUSET=16-19
CPX_PLANNER_CPUSETS="20;21"
CPX_BRIDGE_CPUSET=22
```

CARLA and OpenCDA must then stay on the complementary cores `0-15`. Do not copy
these core numbers to a machine with a different CPU layout without adapting
them.

### Complete startup and shutdown sequence

Start CARLA on the host:

```bash
cd "$HOME/Downloads/MDrive/carla912"
taskset -c 0-15 ./CarlaUE4.sh
```

Start MTR, the Bridge, and all Planner shards inside the ROS container:

```bash
docker exec -it ubuntu24-gpu-container /bin/bash
cd /root/CP-ROS-WS
./scripts/cpx_ros_stack.sh build     # required after source changes
./scripts/cpx_ros_stack.sh start
./scripts/cpx_ros_stack.sh health
```

Run one OpenCDA scenario on the host:

```bash
cd "$HOME/Downloads/CP-X-planning-module-convert_to_ros"
conda activate opencda_planning
taskset -c 0-15 python opencda.py \
  -t cpx_two_cav_merge_conflict_mtr_multimodal \
  -v 0.9.12
```

Inspect or stop the managed ROS stack:

```bash
./scripts/cpx_ros_stack.sh status
./scripts/cpx_ros_stack.sh health-live  # run while OpenCDA is publishing
./scripts/cpx_ros_stack.sh logs
./scripts/cpx_ros_stack.sh stop
```

Do not start `planning.launch.py`, `communication.launch.py`, and
`opencda_integration.launch.py` simultaneously. The integration launch already
owns the Bridge ports and Planner processes; duplicate launches cause
`Address already in use` and cross-session output.

### Repeatable ROS regression commands

The experiment runner keeps one ROS/MTR stack alive across scenarios so map
switching and session leakage are visible:

```bash
# Diverse functional matrix
taskset -c 0-15 python \
  opencda/planning_module/tools/ros_integration_suite.py run \
  --cases smoke_right_turn,mtr_cut_in,mtr_lead_braking,C_on,C_off \
  --trials 1 \
  --python "$CONDA_PREFIX/bin/python" \
  --continue-on-failure

# Repeated multi-session merge regression
taskset -c 0-15 python \
  opencda/planning_module/tools/ros_integration_suite.py run \
  --cases mtr_merge_2cav \
  --trials 3 \
  --python "$CONDA_PREFIX/bin/python" \
  --continue-on-failure
```

Individual commands used by the current validation are:

```bash
python opencda.py -t cpx_single_right_lane_turn -v 0.9.12
python opencda.py -t cpx_two_vehicle_cut_in_mtr_six_modes -v 0.9.12
python opencda.py -t cpx_two_cav_lead_braking_mtr_multimodal -v 0.9.12
python opencda.py -t cpx_cp_vru_crossing -v 0.9.12
python opencda.py -t cpx_cp_vru_crossing_off -v 0.9.12
python opencda.py -t cpx_two_cav_merge_conflict_mtr_multimodal -v 0.9.12
```

### Validated results (October 27, 2026)

With CPU isolation enabled, three consecutive two-CAV MTR merge runs produced:

- 6/6 CAV completions, zero collisions, and zero stale-control frames;
- six attached MTR modes per predicted object;
- two distinct session IDs per run with no route, FSM, reference, or MPC-state leakage;
- mean end-to-end input/output time of 22.6-23.1 ms;
- maximum Planner cycle reduced from approximately 1.17 s without isolation
  to 0.22-0.24 s with isolated workers.

The diverse functional matrix also completed the right turn, MTR cut-in,
two-CAV lead braking, and VRU CP ON/OFF scenarios. Cut-in, lead braking, and
both VRU cases had zero stale control and zero collisions. Full machine-readable
reports are stored locally at:

- `artifacts/ros_integration_runs/20261027_155344/summary.md` — repeated merge;
- `artifacts/ros_integration_runs/20261027_175342/summary.md` — right turn,
  cut-in, lead braking, and VRU ON;
- `artifacts/ros_integration_runs/20261027_201419/summary.md` — VRU OFF.

### Problems already solved

- The old seven-process OpenCDA input bridge was replaced by one unified
  Bridge with one owner for TCP ports 5051-5055 and 5060.
- High-rate TCP input publication was moved off the ROS executor, eliminating
  observed Bridge scheduling stalls of about 1.5 s.
- Inputs and outputs are session-tagged, and each CAV uses an isolated Planner
  runtime and MTR request ownership.
- MTR CUDA extensions were rebuilt for Python 3.12, PyTorch 2.11/CUDA 12.8,
  and `sm_120`; Planner and MTR no longer need to share one Python environment.
- MTR CPU thread count and Planner numerical threads are bounded.
- Optional CPU affinity prevents CARLA/MTR load from starving the Bridge and
  multi-session Planner workers.
- Startup health checks detect missing packages, nodes, publishers, CUDA
  imports, and occupied Bridge ports before a scenario starts.

### Remaining limitations and required work

1. **Cold map/session activation:** the first Town06 right-turn run observed a
   roughly 2.0 s first-control delay. The overall stale rate was only 0.11%,
   but map loading should be pre-warmed or readiness-gated before real-vehicle
   operation.
2. **Traffic-light evaluation:** the right-turn run reached its destination
   without collision but OpenCDA printed `ran_light=True`. The current ROS
   regression analyzer reports this run as safety PASS because its hard gate
   currently checks collisions and road-boundary failures, not the
   `ran_light` SafetyManager flag. This analyzer gap and the underlying signal
   behavior both require correction.
3. **Latency headroom:** the repeated isolated merge passed with zero stale
   frames, but observed maximum end-to-end delays were 274-347 ms; the
   lead-braking run reached 362.5 ms. These remain below the current 0.5 s
   control expiry but need stricter worst-case testing for a real vehicle.
4. **Coverage:** the latest diverse matrix is one trial per scenario. All A-F
   cooperative-perception pairs, three/four-CAV scale cases, packet loss,
   delayed prediction, node restart, and long-duration soak tests still need
   repeated runs.
5. **Deployment boundary:** these results validate CARLA/OpenCDA + localhost
   ROS communication. Real sensors, vehicle CAN/control adapters, DDS network
   QoS, clock synchronization, and platform emergency handling have not yet
   been validated on the vehicle.

## Repository layout

| Path | Responsibility |
|---|---|
| `opencda.py` | Only supported OpenCDA scenario launcher |
| `opencda/scenario_testing/` | Scenario runners, actors, and inheritable YAML |
| `opencda/planning_module/opencda_bridge/` | Runtime assembly and OpenCDA/MTR adapters |
| `opencda/planning_module/pipeline/core/` | Typed contracts, architecture profile, output records |
| `pipeline/perception/` | Input normalization, tracking, map matching, local map |
| `pipeline/route/` | AD-map route, progress, route context, authorization |
| `pipeline/behavior/` | Scenario context, candidates, maneuvers, speed planning |
| `pipeline/interaction/` | Peer intents, prediction modes, Stage A–C, scheduling |
| `pipeline/reference/` | Stable reference, geometry, contracts, final gate |
| `pipeline/execution/` | Stage D, MPC execution, buffering, platform adaptation |
| `pipeline/safety/` | RSS, boundary monitoring, fallback, SafetySupervisor |
| `pipeline/diagnostics/` | Structured diagnostics and stage profiling |
| `opencda/planning_module/MPC/` | Vehicle model, SQP/QP, constraints, objective |
| `opencda/planning_module/Global_Planner/` | OpenDRIVE/AD-map routing backend |
| `mdrive_adapter/` | External MDrive runtime port |
| `CP-ROS-WS/` | ROS 2 transport workspace; not planning-policy owner |
| `artifacts/` | Ignored local logs, plots, and videos |

Detailed ownership and reference-contract rules are in
[planning_architecture.md](opencda/planning_module/docs/planning_architecture.md).


## Custom Global Planner

This repository includes a custom CARLA-independent global planner under `opencda/planning_module/Global_Planner/`.

It reads the OpenDRIVE map and provides route, waypoint, and lane-context queries used by the planning runner and behavior planner.

CARLA is still used for simulation, actor state, actuation, and visualization, but route search and lane-level planning context come from the custom planner.

From project root run this command to build and installs the required AD-map Python bindings and native libraries for the active Python environment.
```bash
PYTHON_BIN="$(command -v python)" \
  opencda/planning_module/Global_Planner/build_ad_map.sh --clean

```

The AD-map runtime is built locally under:

```text
opencda/planning_module/Global_Planner/map_repo/install
```

The following directories are local build/download artifacts and are intentionally not committed or pushed:

```text
opencda/planning_module/Global_Planner/map_repo/install
opencda/planning_module/Global_Planner/map_repo/log
opencda/planning_module/Global_Planner/map_repo/source
```

`install/` contains native Python extensions and shared libraries tied to the active Python ABI, operating system, compiler, system libraries, and local paths. `log/` is build output. `source/` is the downloaded third-party AD-map checkout. Rebuild them on each machine with `build_ad_map.sh` instead of copying them through git.



## Ownership boundaries

The frozen stack uses one owner for each decision:

- `RouteManager` owns the global route and monotonic route progress.
- `BehaviorPlanner` and candidate evaluation own maneuver selection and
  lane-change authorization.
- `SpeedTargetPlanner` owns routine target speed and named speed constraints.
- `ReferenceGenerator` owns geometry; `ReferencePipeline` may condition it
  once; `FinalReferenceGate` is the only final validity authority.
- the interaction pipeline owns conflict classification, cooperative roles,
  and the space-time corridor, but not vehicle control.
- MPC tracks the accepted reference and constraints; it does not choose route
  or traffic-light policy.
- `MPCControlBuffer` owns time alignment between 5 Hz solves and 20 Hz control.
- `SafetySupervisor` owns the final collision, signal, boundary, and actuator
  veto immediately before control leaves the planner.

Every ownership transition is available in structured diagnostics, including
`decision_veto_chain`, solver status, corridor rows, collision/TTC/PET/DRAC,
road-boundary state, and per-stage timing.

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

python opencda.py -t cpx_cp_roadway_object -v 0.9.12
```

Run the external MDrive adapter from the same repository:

```bash
./run_mdrive.sh --graphics-adapter 1
./run_mdrive.sh --graphics-adapter 1 --record-video
./run_mdrive.sh --help
```

It defaults to sibling `../MDrive` and `../envs/mdrive_tcp/bin/python` (or
`python` on `PATH`). Override these with `MDRIVE_ROOT` and `CPX_PYTHON`.

## OpenCDA CP-X Entry Point

The CP-X MPC planner integrates as an OpenCDA vehicle-manager planner and is
always launched from the repository root with `opencda.py`:

```bash
conda activate opencda_planning
export CARLA_ROOT="$HOME/Downloads/MDrive/carla912"
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg:$PWD/opencda/planning_module:$PWD:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python opencda.py -t <scenario_name> -v 0.9.12
```

Optional environment variables for interactive debugging:

```bash
export OPENCDA_DEBUG_VIEW=1            # pygame debug HUD/overlay
export OPENCDA_SPECTATOR_VIEW=planner  # chase view (default); "topdown" is the alternative
```

### Recommended CP-X/OpenCDA scenario suite

Every runnable scenario below has a same-named Python entry point in
`opencda/scenario_testing/` and a same-named YAML file in
`opencda/scenario_testing/config_yaml/`. Scenario YAML files may inherit common
settings through `base_config`; the child YAML contains only the values that
differ from its parent.

| Group | Scenario | Purpose | Configuration base |
|---|---|---|---|
| Route turn | `cpx_single_left_lane_turn_isolated` | Deterministic route-required left turn without background-traffic delay. | `cpx_single_left_lane_turn.yaml` |
| Route turn | `cpx_single_right_lane_turn` | Route-required right lane change followed by a continuous right-turn connector. | `single_intersection_town06_carla.yaml` |
| Static obstacle | `cpx_single_left_lane_turn_blocked` | Local avoidance, lane borrowing, stopping fallback, and recovery around a deterministic blocker. | `cpx_single_left_lane_turn.yaml` |
| Cooperative driving | `cpx_two_cav_merge_conflict` | Two connected vehicles exchange planned paths and claims; validates `proceed`/`make_gap` plus corridor QP rows. | `cpx_two_cav_merge_conflict.yaml` |
| MTR integration | `cpx_two_cav_merge_conflict_mtr_multimodal` | Same merge with the real asynchronous MTR prediction bridge. | `cpx_two_cav_merge_conflict.yaml` |
| Speed sweep | `cpx_lane_change_speed_08` | Lane change at 8 m/s. | `cpx_lane_change_2lanefree.yaml` |
| Speed sweep | `cpx_lane_change_speed_12` | Lane change at 12 m/s. | `cpx_lane_change_2lanefree.yaml` |
| Speed sweep | `cpx_lane_change_speed_15` | Lane change at 15 m/s. | `cpx_lane_change_2lanefree.yaml` |

The frozen-baseline smoke suite is:

```bash
python opencda.py -t cpx_single_left_lane_turn_isolated -v 0.9.12
python opencda.py -t cpx_single_right_lane_turn -v 0.9.12
python opencda.py -t cpx_lane_change_speed_12 -v 0.9.12
python opencda.py -t cpx_two_cav_merge_conflict -v 0.9.12
python opencda.py -t cpx_two_cav_merge_conflict_mtr_multimodal -v 0.9.12
```

The MTR scenario requires either the configured HTTP prediction server or the
in-process predictor path. A completed MTR request alone is not evidence that
multiple modes changed the plan; verify retained mode count, conflict tags,
corridor binding, and QP row diagnostics.

The shared `single_intersection_town06_carla.py`, `cpx_mature_runner.py`,
`default.yaml`, `single_intersection_town06_carla.yaml`,
`cpx_lane_change_2lanefree.yaml`, and `cpx_profile_full_default.yaml` files are
supporting infrastructure for these scenarios and must not be removed.

Each run writes per-tick planner debug CSV/JSONL to the scenario's
`planner.debug_output_dir` (see its config under
`opencda/scenario_testing/config_yaml/`). Turn that CSV into plots and a
metrics report with:

```bash
python -m opencda.planning_module.tools.export_full_run_plots <debug_csv> <output_dir>
```

See [`opencda/planning_module/README.md`](opencda/planning_module/README.md)
for the CP-X planning stack's architecture, directory layout, and behavior
planner/MPC internals.

### Analysing results

After any run, `planning_metrics.json` and its time-series CSV are written next to the planner debug log (the scenario's `planner.debug_output_dir`). Use the bundled analysis script:

```bash
cd opencda/planning_module

# analyse the most recent run
python analyze_run.py

# analyse a specific run
python analyze_run.py <run_dir>

# compare multiple runs side by side
python analyze_run.py --compare <run_dir_a> <run_dir_b> <run_dir_c>

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
| Max solve time (ms) | < 30 ms | > 50 ms | CARLA runs at 20 Hz (50 ms per tick), while MPC trajectory generation defaults to 5 Hz and buffered commands are consumed at 20 Hz. A 30 ms solve preserves same-tick headroom; 50 ms consumes a full simulation tick. |

> **Note on MDrive integration:** if the module is later evaluated on the MDrive leaderboard, replace the TTC and DRAC thresholds with MDrive's own infraction categories, which have their own severity classification. The values above are internal development targets only.

## Tests

Run the complete planning-module suite in the supported environment:

```bash
conda run -n opencda_planning \
  python -m pytest -q opencda/planning_module/tests
```

The frozen baseline currently reports `1530 passed`. For a small syntax-only
check of the main boundaries:

```bash
conda run -n opencda_planning python -m py_compile \
  opencda/planning_module/MPC/mpc.py \
  opencda/planning_module/opencda_bridge/cpx_mpc_planner.py \
  opencda/planning_module/pipeline/core/output.py \
  opencda/planning_module/pipeline/interaction/cav_conflict_pipeline.py \
  opencda/planning_module/pipeline/execution/mpc_execution_stage.py
```

## Outputs

Scenario runs write to the configured local `artifacts/debug/<run>/` directory.
Typical outputs are:

- `opencda_planner_debug.jsonl`: complete per-tick decision and timing record;
- `opencda_planner_control_output.jsonl`: emitted control commands;
- `planning_metrics_timeseries.csv`: reporting-oriented safety/performance data;
- `planned_trajectory.csv`: planned and executed trajectory samples;
- `collision_events.csv`: collision records;
- optional plots and video produced by the reporting/recording tools.

`/artifacts/` is ignored by Git. Copy only final report figures into
`docs/results/` when they need to be versioned.


## License

This project is based on OpenCDA. Keep the original OpenCDA license and citation requirements when publishing or sharing the code.
