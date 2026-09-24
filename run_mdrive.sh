#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MDRIVE_ROOT="${MDRIVE_ROOT:-$ROOT/../MDrive}"
python="${CPX_PYTHON:-$ROOT/../envs/mdrive_tcp/bin/python}"
if [[ -z "${CPX_PYTHON:-}" && ! -x "$python" ]]; then
    python=python
fi

server_args=(--start-carla)
args=()
for arg in "$@"; do
    case "$arg" in
        --existing-carla) server_args=() ;;
        -h|--help)
            cat <<'EOF'
Usage: ./run_mdrive.sh [options]

Defaults: r26, GT perception, parallel planning, automatic core selection,
          managed CARLA server on port 23010 (traffic manager: 23015).

  --routes-dir PATH       Select a scenario directory
  --execution serial     Use serial planning
  --worker-cpus LIST      Physical-core CPU IDs, e.g. 2,3,4,5,6,7 (default: auto)
  --record-video          Record overhead RGB video
  --graphics-adapter N    Select CARLA's Vulkan GPU (default: 0)
  --port N               Override CARLA port
  --traffic-manager-port N
  --existing-carla        Connect to an existing server without starting one
  --config PATH          Override planner configuration
  --run-dir PATH         Set a new output directory

Environment: MDRIVE_ROOT, CPX_PYTHON. Other options pass through to the launcher.
EOF
            exit 0 ;;
        *) args+=("$arg") ;;
    esac
done

export PYTHONDONTWRITEBYTECODE=1
exec "$python" "$ROOT/mdrive_adapter/run.py" \
    --direct "${server_args[@]}" --mdrive-root "$MDRIVE_ROOT" \
    --routes-dir "$MDRIVE_ROOT/scenarioset/interdrive/r26_town05_ins_chaos" \
    --execution parallel --worker-cpus auto \
    --port 23010 --traffic-manager-port 23015 "${args[@]}"
