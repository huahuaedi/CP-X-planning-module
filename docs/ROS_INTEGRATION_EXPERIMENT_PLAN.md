# CP-X ROS Integration Experiment Plan

## Goal

Validate the complete closed loop rather than an isolated algorithm:

`OpenCDA/CARLA → ROS Bridge → Planner ↔ MTR Prediction → ROS Bridge → vehicle control`.

ROS Planner, Bridge, and MTR should remain alive between cases. Restarting them
between scenarios can hide session, map, prediction-cache, route, and MPC warm
start leakage.

## Experiment layers

| Layer | Suite | Main question | Required repetitions |
|---|---|---|---:|
| 1 | Basic motion smoke | Do straight, left/right turn, and lane-change functions still work through ROS? | 3 |
| 2 | `cp_af` | Does shared perception create a measurable benefit under identical physical conditions? | 3 |
| 3 | `prediction` | Do fresh six-mode MTR trajectories reach the Planner and affect obstacle handling? | 5 |
| 4 | `scale` | Are prediction latency and per-CAV state stable at 2, 3, and 4 CAVs? | 3 |
| 5 | `lifecycle` | Can maps and CP ON/OFF cases run sequentially without restarting ROS or leaking state? | 5 |

## A–F controlled comparisons

| Pair | Hazard | ON evidence | OFF/control interpretation |
|---|---|---|---|
| A | Red-light violator | Earlier conflict binding/deceleration | Same physical actor, later local reaction |
| B | Occluded oncoming vehicle | Collision-free completion | Unsafe/non-completing baseline is reported, never counted as a safety pass |
| C | Crossing VRU | Earlier relevant-obstacle/risk activation | Both may remain safe; compare awareness and braking onset |
| D | Roadway object | Early lane change and completion | Safe stop or late response without shared perception |
| E | Converging CAVs | Independent sessions plus cooperative conflict resolution | Same geometry without cooperative arbitration |
| F | Lane closure | Route changes from lane 630148 to 630149 and completes | Must start clean on 630148 and must not inherit the ON closure |

## Acceptance signals

Every controlled CAV log must show:

- ROS control became active and used one unique `session_id`;
- stale-control rate at or below 2%;
- zero scenario exceptions;
- zero road-boundary breaches;
- zero collisions, except an explicitly labeled unsafe baseline used only to
  quantify CP benefit;
- expected completion/non-completion outcome;
- distinct session IDs for every CAV in a multi-CAV run.

Prediction cases additionally require:

- at least six attached trajectory modes;
- advancing prediction revisions;
- prediction age within the configured 0.5 s freshness window when attached;
- no cross-ego session ownership errors.

The report includes mean and maximum latency. P95 is intentionally not used.

## Commands

Keep CARLA and one ROS integration stack running, then use the OpenCDA Python
3.7 environment:

```bash
conda activate opencda_planning

python opencda/planning_module/tools/ros_integration_suite.py preflight
python opencda/planning_module/tools/ros_integration_suite.py list

# First gate: same-map and cross-map lifecycle isolation
python opencda/planning_module/tools/ros_integration_suite.py run \
  --suite lifecycle --trials 1

# Main A–F evidence
python opencda/planning_module/tools/ros_integration_suite.py run \
  --suite cp_af --trials 3

# MTR functional and scaling evidence
python opencda/planning_module/tools/ros_integration_suite.py run \
  --suite prediction --trials 5
python opencda/planning_module/tools/ros_integration_suite.py run \
  --suite scale --trials 3
```

Results are archived under `artifacts/ros_integration_runs/<timestamp>/` with
the console output, raw per-CAV logs, `summary.json`, and a PPT-friendly
`summary.md`.

Existing live outputs can be checked without running CARLA:

```bash
python opencda/planning_module/tools/ros_integration_suite.py analyze \
  --suite cp_af --live
```

## Reporting metrics

Use success rate across repetitions instead of presenting one favorable run.
Report:

- completion rate, collisions, boundary breaches, and remaining distance;
- first hazard-awareness time/distance and first braking/deceleration point;
- minimum front gap or TTC when available;
- mean and maximum Planner input-to-output latency;
- stale-control frame rate;
- attached MTR mode count, revision progress, and prediction age;
- per-CAV session IDs and multi-CAV completion;
- ON–OFF differences for A–F, not only absolute ON performance.
