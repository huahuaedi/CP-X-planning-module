# CP-X planner in MDrive: setup, reproduction, and the slow-run problem

This folder contains everything needed to run the CP-X planner (`convert_to_ros`
branch) inside MDrive, reproduce the scenario results listed below, and start
debugging the slow multi-agent runs.

- `mdrive_cpx_integration.patch` — the MDrive-side changes (15 files).
- This README — how to apply it, the exact commands, measured results, and what is known.

All results below were measured on 2026-09-21 with CP-X commit `78ada3a`. The only
later commit (`20a2ec1`) changes a test file, so the runtime code is the same.
Hardware: 1 GPU (24 GB), 24 CPU cores, CARLA 0.9.12.

## 1. Setup

**Requirements**

- MDrive at commit `9e083770` (`ucla-mobility/MDrive`, "Improve NPC actor behavior, ..."). The patch was generated against this commit and applies cleanly to a clean export of it.
- This repo checked out on branch `convert_to_ros`. The Town01/02/03 OpenDRIVE maps are already included under `opencda/planning_module/Global_Planner/maps/`. Without them, scenarios on those maps crash at agent setup.
- The MDrive conda environment (we used one named `tcp_codriving_clean`, Python 3.7) with the CARLA 0.9.12 egg.

**Apply the MDrive patch**

```bash
cd $MDRIVE_ROOT
git checkout 9e083770
git apply $CPX_REPO/docs/mdrive_handoff/mdrive_cpx_integration.patch
git status --short        # 9 modified files + 6 new files
```

The patch touches `simulation/leaderboard/team_code/` (`perception_swap_agent.py`,
`cpx_planner_adapter.py`, `rule_planner.py`, `auto_pilot.py`, `base_agent.py`, four agent
configs including `cpx_gt_primary_only.yaml`), `simulation/leaderboard/leaderboard/`
(`scenario_manager.py`, `leaderboard_evaluator_parameter.py`, `sensors/fixed_sensors.py`), and
`tools/` (`run_custom_eval.py`, `perception_runner.py`). It includes local MDrive changes that
were in place when the results below were produced, not only CP-X-specific ones (for example,
`scenario_manager.py` applies a final hard stop to an ego that has completed its route instead of
removing the vehicle). The Tier-2 `soft_complete` termination itself is already in the base commit.

**Environment variables** (set these in every shell)

```bash
conda activate <mdrive-env>
export CARLA_ROOT="$MDRIVE_ROOT/carla912"
export CPX_PLANNING_REPO="$CPX_REPO"        # REQUIRED, see note
export PYTHONPATH="$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

> **Note:** if `CPX_PLANNING_REPO` is unset, the adapter falls back to `../Planning module`
> next to the MDrive checkout, which is a different (older) copy of the planner.

## 2. Run commands

Run from `$MDRIVE_ROOT`. Each command starts its own CARLA on port 14000; run them one at a
time. Results land in `results/results_driving_custom/<results-tag>/`.

```bash
run() { python tools/run_custom_eval.py --planner cpx_gt_primary_only \
  --routes-dir "scenarioset/interaction/$1" --results-tag "$2" \
  --port 14000 --start-carla --overwrite; }

run Highway_On-Ramp_Merge/1             repro_highway
run Overtaking_on_Two-Lane_Road/1       repro_overtaking
run Roundabout_Navigation/1             repro_roundabout
run Pedestrian_Crosswalk/1              repro_pedestrian
run Blocked_Lane_Obstacle/1             repro_blocked
run Construction_Zone/1                 repro_construction
run Major_Minor_Unsignalized_Entry/1    repro_majorminor
run Interactive_Lane_Change/3           repro_ilc3        # route 3, not the default /1

# The two slow scenarios (see section 4). Expect these to take a very long time.
run Unprotected_Left_Turn/1             repro_unprotectedleft
run Intersection_Deadlock_Resolution/1  repro_deadlock
```

`cpx_gt_primary_only` (`agent_config/cpx_gt_primary_only.yaml`, `controlled_ego_count: 1`) runs
the full CP-X pipeline on ego 0 only; the other egos use MDrive's rule-based planner and still
broadcast their intent.

**Where to look after a run**

| What | Where |
|---|---|
| Termination decision | `grep TERM_DECISION results/results_driving_custom/<tag>/_planner_logs/cpx_gt_primary_only.log` |
| Score / infractions / timing | `<tag>/cpx_gt_primary_only/ego_vehicle_0/results.json` (`duration_game`, `duration_system`) |
| Per-tick planner trace | `<tag>/cpx_gt_primary_only/image/cpx_planner/ego_0/opencda_planner_debug.jsonl` |
| Per-stage timing | `grep cpx_stage_timing <tag>/_planner_logs/cpx_gt_primary_only.log` |

## 3. Measured results (ego 0 only)

`rc` is the route completion printed in `TERM_DECISION`. "Slowdown" is
`duration_system / duration_game` (wall time over simulated time).

| Scenario | Egos | Outcome | rc | Sim time | Wall time | Slowdown |
|---|---|---|---|---|---|---|
| Highway_On-Ramp_Merge/1 | 3 | Stopped, no collision | 61.7% | 45.3 s | 98.3 s | 2.2x |
| Overtaking_on_Two-Lane_Road/1 | 2 | Stopped, no collision | 46.8% | 41.0 s | 95.9 s | 2.3x |
| Roundabout_Navigation/1 | 2 | Stopped 2.6 m short of the goal, no collision | 94.6% | 44.6 s | 162.4 s | 3.6x |
| Pedestrian_Crosswalk/1 | 2 | Stopped 2.7 m short; run ends by route timeout, no collision | 96.8% (HUGSIM) | 75.1 s | 296.1 s | 3.9x |
| Blocked_Lane_Obstacle/1 | 3 | Stopped, no collision (see note) | 12.7% | 36.4 s | 146.3 s | 4.0x |
| Construction_Zone/1 | 4 | Collision at 4.2 s | 2.3% | 17.3 s | 119.0 s | 6.9x |
| Major_Minor_Unsignalized_Entry/1 | 3 | Collision at 3.2 s | 3.2% | 12.8 s | 39.8 s | 3.1x |
| Interactive_Lane_Change/3 | 5 | Collision at 13.9 s | 95.1% | 14.0 s | 58.1 s | 4.2x |

Blocked and Pedestrian were each run twice with identical results. Every run finishes without a
startup crash. All eight are 2-7x slower than real time but complete in under 5 minutes.

## 4. The slow-run problem

**Scenarios:** `Unprotected_Left_Turn/1` and `Intersection_Deadlock_Resolution/1` (6 egos).
These were last observed on 2026-09-15 and were **not re-run** in the batch above. In that
observation, more than 25 minutes of wall time produced only about 15 s of simulated time.
The process was not deadlocked: it alternated R/D state at ~111% CPU and the tick count in the
debug jsonl kept climbing (121 -> 225 ticks over several minutes). In the debug trace,
`cav_conflict_agent_count` was 10 there (4 in the 5-ego `Interactive_Lane_Change/3`).

**What is known**

1. `simulation/leaderboard/team_code/perception_swap_agent.py` calls each controlled ego's
   `planner.run_step()` in a plain sequential `for primary_idx in range(self.ego_vehicles_num)`
   loop in one Python process, so per-tick work serializes on one core (24 cores were idle).
   Parallelizing it has to respect the `begin_tick` / `commit_tick` staged-message contract in
   `MDriveCAVRegistry` (`cpx_planner_adapter.py`), which assumes sequential execution within a tick.
2. `controlled_ego_count: 1` limits how many egos run the full pipeline (5-ego ILC3 finished in
   about 1 minute), but it does **not** remove the growth of a single ego's per-tick cost with
   the number of nearby perceived vehicles.
3. In a 3-ego run (Highway), QP trajectory planning averaged about 15 ms per replan and route
   summary about 14 ms per call, so the QP is not obviously the dominant cost there. This is one
   data point in a 3-ego scenario; it says nothing yet about the 6-ego cases.

**What is not known:** which stage inside a single `run_step()` grows super-linearly. Candidates
(unverified): `classify_conflicts` in `opencda/planning_module/pipeline/conflict_classifier.py`
(at least O(agents x horizon)), candidate evaluation, and the cooperative-arbitration step.
`py-spy` could not attach on our machine (ptrace blocked, no passwordless sudo).

**Fastest ways to find it** (the slowdown is visible in the first ~200 ticks, so there is no
need to wait 25 minutes):

1. `py-spy dump --pid <python pid>` a few times during the slow phase (needs
   `kernel.yama.ptrace_scope=0` or root).
2. Without privileges: wrap `MDriveCPXPlanner.run_step` in `cProfile` for the first ~200 ticks and
   sort by cumulative time.
3. Compare `[cpx_stage_timing]` output (printed roughly every 50 ticks from
   `opencda/planning_module/opencda_bridge/cpx_mpc_planner.py`) between `Interactive_Lane_Change/3`
   (fast) and `Unprotected_Left_Turn/1` (slow).
4. Check whether cost tracks `cav_conflict_agent_count` by distance-filtering the perceived agents
   (for example to 50 m) and re-timing.

## 5. Known issues per scenario (who owns what)

| Scenario | Cause | Owner |
|---|---|---|
| Overtaking | Passing the parked truck needs the opposing lane. `_same_lane_group()` in `opencda/planning_module/utility/carla_lane_graph.py` rejects lanes of opposite sign, so every lane-change reference for that lane is empty (`empty_reference`, all three candidate variants rejected) and the ego stops. Missing capability, not a bug. | CP-X |
| Blocked | Same `empty_reference` failure on the lane-borrow candidates. The third static truck (`entity_1_3`) also fails to spawn (184 attempts); this happened on earlier runs too. | CP-X + scenario |
| Highway | MPC reports infeasible during plain lane-follow, enters `bounded_safe_stop`, and never recovers. Diagnosed, not fixed. | CP-X |
| Roundabout, Pedestrian | Planner declares `route_reached_destination` and stops about 2.6 m before the end of the route, in both scenarios (2.63 and 2.60 m). Cause of the offset not located yet. | CP-X (probable) |
| Pedestrian (timeout) | Same stop position as an earlier run that ended by Tier-2 `soft_complete`; this time `soft_complete` never fired (needs rc >= 95% and speed < 0.1 m/s for 5 s, in `scenario_manager.py`). Cause not determined. | Unknown / MDrive side |
| Construction, Major_Minor, ILC3 | Collisions. Not root-caused. For ILC3 an earlier analysis suggested a faster following vehicle rear-ending the ego during destination braking, but that was not concluded. | Not diagnosed |
| Unprotected_Left_Turn, Intersection_Deadlock | Slow-run problem above. | Both |

## 6. Unit tests

```bash
cd $CPX_REPO
PYTHONPATH="$PWD:$PWD/opencda/planning_module" python -m pytest opencda/planning_module/tests -q
# expected: 1446 passed
```
