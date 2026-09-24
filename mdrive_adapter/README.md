# MDrive adapter

Run CP-X through MDrive's external agent interface without modifying MDrive.
Perception uses current-state GT within 70 m; prediction, behavior planning and
MPC remain CP-X's own pipeline. MDrive owns simulation ticks, controls and scoring.

## Setup

Use a working MDrive environment with CARLA 0.9.12 / Python 3.7. From the CP-X root:

```bash
export MDRIVE_ROOT=/path/to/MDrive
export CPX_PYTHON=/path/to/mdrive/environment/bin/python
"$CPX_PYTHON" -m pip install --no-deps --target .runtime/deps -r mdrive_adapter/requirements.txt
```

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

One r26 six-ego comparison (video off) measured 408 → 152 ms/frame after excluding
the first 20 frames, and 489 → 256 s evaluator wall time. Both ran 826 frames;
all 4,956 controls and positions matched. Collisions and blocked vehicles remained.
These are single-run measurements, not a general performance guarantee.

## Semantics and limits

- This is a GT/oracle baseline: no occlusion or communication model, and no GT future trajectories.
- Each ego has independent planner state; parallel execution is not joint optimization.
- Benchmark waypoints are preserved and global rerouting is disabled. The stop
  target extends 3 m past the endpoint so cars can cross MDrive's finish line.
- Decision timers and MPC control indexing use simulation time. Longitudinal
  feedback tracks MPC speed; steering remains the MPC output.
- Direct mode uses MDrive's no-NPC configuration, which forces traffic lights green.
  Episode recovery and open-loop replay are unsupported.
- `closed_loop_executed` means execution succeeded, not collision-free completion.
  Use MDrive's collision metrics; auxiliary CP-X collision fields are null.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 "$CPX_PYTHON" -m unittest \
  mdrive_adapter.test_runtime mdrive_adapter.test_parallel
```
