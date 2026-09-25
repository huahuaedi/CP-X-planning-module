# MDrive adapter

Run CP-X through MDrive's external agent interface without modifying MDrive.
Perception uses current-state GT within 70 m. The adapter calls the current
`CPXMPCPlannerBridge` pipeline with AD-map topology, prediction, behavior planning,
MPC, and the velocity/steering control boundary. MDrive owns simulation ticks, controls and scoring.

## Setup

Use a working MDrive environment with CARLA 0.9.12 / Python 3.7. From the CP-X root:

```bash
export MDRIVE_ROOT=/path/to/MDrive
export CPX_PYTHON=/path/to/mdrive/environment/bin/python
"$CPX_PYTHON" -m pip install --no-deps --target .runtime/deps -r mdrive_adapter/requirements.txt
```

Build the current branch's native AD-map backend with the same Python ABI:

```bash
PYTHON_BIN="$CPX_PYTHON" BUILD_JOBS=6 \
  opencda/planning_module/Global_Planner/build_ad_map.sh
```

See `opencda/planning_module/Global_Planner/RUNNING.md` for build dependencies.
`GLOBAL_PLANNER_AD_MAP_INSTALL` can point at an existing compatible installation.
The launcher checks backend imports before starting CARLA.

## Run

```bash
./run_mdrive.sh --graphics-adapter 1
./run_mdrive.sh --graphics-adapter 1 --record-video
```

The root script defaults to r26, parallel execution and a managed CARLA server.
All paths and options can be overridden; use `./run_mdrive.sh --help`.

- `--execution serial`: plan each ego in the evaluator process.
- `--execution parallel`: one persistent process per ego, each on a distinct
  physical core. All egos receive the same GT frame; the evaluator waits for all
  controls before advancing. Workers use separate read-only CARLA connections.
- `--worker-cpus 2,3,4,5,6,7`: explicit affinity instead of automatic selection.
  Affinity does not reserve cores; numerical libraries use one thread per process.
- `--record-video`: overhead RGB video; requires OpenCV and ffmpeg with libx264.
  Images are only for visualization and do not enter planning.
- `--routes-dir mdrive_adapter/smoke_routes`: short two-ego integration test.
- `--config PATH`, `--run-dir PATH`: override planning config or output directory.
- `--existing-carla`: use an existing server. Choose unused ports and an
  available Vulkan `--graphics-adapter`; `--gpus` only controls CUDA visibility.

The root script invokes `mdrive_adapter/run.py --direct`, which runs MDrive's
evaluator for one scenario and is required for parallel
execution. Without it, the launcher forwards extra options to `run_custom_eval.py`.
The diagnostic `worker-serial` mode dispatches the same workers sequentially.

## Results

Each run copies routes and config into `.runtime/run_<timestamp>/`. Outputs stay
outside MDrive; `.runtime/` is ignored by Git.

| Path inside the run directory | Contents |
| --- | --- |
| `summary.json` | Route results and solver diagnostics |
| `results/ego_vehicle_*/results.json` | Authoritative MDrive scores and infractions |
| `results/image/cpx_gt/steps.jsonl` | Per-ego controls, trajectories and timing |
| `results/image/cpx_gt/frames.jsonl` | Per-frame agent and planning latency |
| `results/image/cpx_gt/workers.json` | Worker IDs and assigned cores |
| `results/image/cpx_gt/episode.mp4` | Optional video at simulation speed |
| `execution.json` | Evaluator wall time, including initialization and cleanup |

Compare sequentially executed runs with identical configs and video settings:

```bash
"$CPX_PYTHON" mdrive_adapter/compare_runs.py \
  /path/to/serial_run /path/to/parallel_run --output /path/to/comparison.json
```

## Semantics and limits

- This is a GT/oracle baseline: no occlusion or communication model, and no GT future trajectories.
- Each ego has independent planner state; parallel execution is not joint optimization.
- Benchmark waypoints are preserved and global rerouting is disabled. An internal
  3 m stopping tail is appended after the final waypoint so cars can cross
  MDrive's finish line; benchmark routes and scoring are unchanged.
- Decision timers and the MPC control buffer use simulation time. The current
  bridge maps MPC velocity/steering through longitudinal PID and physical
  steering geometry. No legacy standalone runner is used.
- Every parallel barrier receives the same GT frame. Workers never tick the
  simulator or apply controls. A stale frame, worker error, or timeout fails
  the barrier and closes the pool. Planning exceptions propagate to MDrive.
- `mpc` and `scenario.constraints` settings are merged into a private per-ego
  MPC YAML. Current bridge options can be supplied under `bridge`. Map caches,
  message paths, controller state and diagnostics are isolated per ego.
- This GT baseline does not exchange cooperative intents between workers.
- Direct mode uses MDrive's no-NPC configuration, which forces traffic lights green.
  Episode recovery and open-loop replay are unsupported.
- `closed_loop_executed` means execution succeeded, not collision-free completion.
  Use MDrive's collision metrics: the bridge does not own a collision sensor,
  so its auxiliary collision counters are not authoritative.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 "$CPX_PYTHON" -m unittest \
  mdrive_adapter.test_runtime mdrive_adapter.test_parallel
```
