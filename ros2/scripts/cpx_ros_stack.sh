#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_default=$(cd -- "${script_dir}/.." && pwd)
planner_default=$(cd -- "${workspace_default}/.." && pwd)
config_file=${CPX_STACK_ENV_FILE:-${workspace_default}/.env}

if [[ -f "${config_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${config_file}"
  set +a
fi

ROS_DISTRO=${ROS_DISTRO:-jazzy}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}
FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}
CPX_ROS_WS=${CPX_ROS_WS:-${workspace_default}}
CPX_PLANNER_ROOT=${CPX_PLANNER_ROOT:-${planner_default}}
CPX_PLANNER_DEPS=${CPX_PLANNER_DEPS:-/root/cpx-python312-deps}
CPX_PREDICTION_ROOT=${CPX_PREDICTION_ROOT:-/root/CP-X-Prediction}
CPX_PREDICTION_ROS_WS=${CPX_PREDICTION_ROS_WS:-${CPX_PREDICTION_ROOT}/ROS2-Code}
MTR_VENV=${MTR_VENV:-/root/mtr-ros-py312}
MTR_ROOT=${MTR_ROOT:-${CPX_PREDICTION_ROOT}/Intersection-Code/Intersection-MTR/Prediction/MTR}
CPX_PLANNER_SHARDS=${CPX_PLANNER_SHARDS:-2}
CPX_PREDICTION_ENABLED=${CPX_PREDICTION_ENABLED:-true}
CPX_MTR_NUM_MODES=${CPX_MTR_NUM_MODES:-6}
CPX_MTR_HISTORY_HZ=${CPX_MTR_HISTORY_HZ:-10.0}
CPX_MTR_INFERENCE_HZ=${CPX_MTR_INFERENCE_HZ:-5.0}
CPX_MTR_CPU_THREADS=${CPX_MTR_CPU_THREADS:-4}
CPX_PLANNER_CPU_THREADS=${CPX_PLANNER_CPU_THREADS:-1}
CPX_MTR_CPUSET=${CPX_MTR_CPUSET:-}
CPX_BRIDGE_CPUSET=${CPX_BRIDGE_CPUSET:-}
CPX_PLANNER_CPUSETS=${CPX_PLANNER_CPUSETS:-}
CPX_PREDICTION_MAX_AGE_S=${CPX_PREDICTION_MAX_AGE_S:-0.5}
CPX_CONTROL_MAX_PLAN_AGE_S=${CPX_CONTROL_MAX_PLAN_AGE_S:-0.5}
CPX_DEBUG=${CPX_DEBUG:-false}
CPX_DEBUG_TIME=${CPX_DEBUG_TIME:-true}

runtime_dir=${CPX_ROS_WS}/.runtime
log_dir=${runtime_dir}/logs

export ROS_DOMAIN_ID ROS_AUTOMATIC_DISCOVERY_RANGE FASTDDS_BUILTIN_TRANSPORTS
export CPX_ROS_WS CPX_PLANNER_ROOT CPX_PLANNER_DEPS CPX_PREDICTION_ROOT
export CPX_PREDICTION_ROS_WS MTR_VENV MTR_ROOT CPX_PLANNER_SHARDS

source_setup() {
  # ROS-generated setup files may probe variables without ${name:-} guards.
  # Keep strict mode for this script, but disable nounset only while sourcing.
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

source_runtime() {
  local file
  for file in \
    "/opt/ros/${ROS_DISTRO}/setup.bash" \
    "${CPX_PREDICTION_ROS_WS}/install/setup.bash" \
    "${CPX_ROS_WS}/install/setup.bash"; do
    if [[ ! -r ${file} ]]; then
      printf 'Missing required setup file: %s\n' "${file}" >&2
      exit 2
    fi
    source_setup "${file}"
  done
  export PYTHONPATH="${CPX_PLANNER_DEPS}:${CPX_PLANNER_ROOT}:${PYTHONPATH:-}"
}

pid_alive() {
  local file=$1 pid
  [[ -r ${file} ]] || return 1
  read -r pid <"${file}"
  [[ ${pid} =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

assert_ports_free() {
  local port
  for port in 5051 5052 5053 5054 5055; do
    if ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .; then
      printf 'TCP port %s is already in use. Stop the old ROS Bridge first.\n' "${port}" >&2
      exit 3
    fi
  done
}

start_stack() {
  mkdir -p "${log_dir}"
  if pid_alive "${runtime_dir}/planner.pid" || pid_alive "${runtime_dir}/mtr.pid"; then
    printf 'CP-X ROS stack is already running; use status or stop first.\n' >&2
    exit 4
  fi
  assert_ports_free
  source_runtime

  (
    export PATH="${MTR_VENV}/bin:${PATH}"
    export PYTHONPATH="${MTR_ROOT}/live_inference:${MTR_ROOT}:${PYTHONPATH}"
    # PyTorch otherwise consumes every host CPU for intra-op work. With two
    # Planner shards that can starve the Bridge receive/publish threads for
    # hundreds of milliseconds even though CUDA inference itself is fast.
    export OMP_NUM_THREADS="${CPX_MTR_CPU_THREADS}"
    export OMP_THREAD_LIMIT="${CPX_MTR_CPU_THREADS}"
    export MKL_NUM_THREADS="${CPX_MTR_CPU_THREADS}"
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS="${CPX_MTR_CPU_THREADS}"
    mtr_command=(ros2 run mtr_prediction_msgs mtr_prediction_node --ros-args \
      -p num_modes:="${CPX_MTR_NUM_MODES}" \
      -p history_hz:="${CPX_MTR_HISTORY_HZ}" \
      -p inference_hz:="${CPX_MTR_INFERENCE_HZ}" \
      -p session_idle_timeout_s:=30.0)
    if [[ -n ${CPX_MTR_CPUSET} ]]; then
      exec setsid taskset -c "${CPX_MTR_CPUSET}" "${mtr_command[@]}"
    fi
    exec setsid "${mtr_command[@]}"
  ) >"${log_dir}/mtr.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"${runtime_dir}/mtr.pid"

  (
    # Each shard has its own process. Keep numerical kernels single-threaded
    # so both shards and the Bridge retain deterministic CPU access.
    export OMP_NUM_THREADS="${CPX_PLANNER_CPU_THREADS}"
    export OMP_THREAD_LIMIT="${CPX_PLANNER_CPU_THREADS}"
    export MKL_NUM_THREADS="${CPX_PLANNER_CPU_THREADS}"
    export OPENBLAS_NUM_THREADS="${CPX_PLANNER_CPU_THREADS}"
    export NUMEXPR_NUM_THREADS="${CPX_PLANNER_CPU_THREADS}"
    exec setsid ros2 launch cpx_planning opencda_integration.launch.py \
      prediction_enabled:="${CPX_PREDICTION_ENABLED}" \
      prediction_max_age_s:="${CPX_PREDICTION_MAX_AGE_S}" \
      execution_control_max_plan_age_s:="${CPX_CONTROL_MAX_PLAN_AGE_S}" \
      planner_shard_count:="${CPX_PLANNER_SHARDS}" \
      planner_cpu_sets:="${CPX_PLANNER_CPUSETS}" \
      bridge_cpu_set:="${CPX_BRIDGE_CPUSET}" \
      ros_domain_id:="${ROS_DOMAIN_ID}" \
      discovery_range:="${ROS_AUTOMATIC_DISCOVERY_RANGE}" \
      planner_pythonpath:="${CPX_PLANNER_DEPS}:${CPX_PLANNER_ROOT}" \
      debug:="${CPX_DEBUG}" \
      debug_time:="${CPX_DEBUG_TIME}"
  ) >"${log_dir}/planner.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"${runtime_dir}/planner.pid"

  printf 'Started MTR and Planner. Logs: %s\n' "${log_dir}"
  sleep 8
  if ! health_stack health; then
    printf 'Startup health-check failed; stopping the partial stack.\n' >&2
    tail -n 40 "${log_dir}/mtr.log" "${log_dir}/planner.log" >&2 || true
    stop_stack
    exit 5
  fi
}

stop_one() {
  local name=$1 file=${runtime_dir}/$1.pid pid
  [[ -r ${file} ]] || return 0
  read -r pid <"${file}"
  if [[ ${pid} =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    printf 'Stopping %s (launch PID %s)\n' "${name}" "${pid}"
    if [[ ${name} == planner ]]; then
      # Signal only the managed launch parent. ROS launch forwards SIGINT to
      # its children and waits for destroy_node() itself. Signalling the
      # complete process group here gives every child a second SIGINT while it
      # is writing timing summaries or closing AD-map state.
      kill -INT "${pid}" 2>/dev/null || true
    else
      # ros2 run does not supervise its child like ros2 launch. Signal the MTR
      # process group so the actual node exits together with the CLI wrapper.
      kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
    fi
    for _ in {1..80}; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "${pid}" 2>/dev/null; then
      printf '%s did not stop cleanly within 20 seconds; terminating its process group.\n' "${name}" >&2
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f -- "${file}"
}

stop_stack() {
  stop_one planner
  stop_one mtr
  printf 'CP-X ROS stack stopped.\n'
}

status_stack() {
  local result=0
  if pid_alive "${runtime_dir}/planner.pid"; then
    printf 'Planner: running (PID %s)\n' "$(<"${runtime_dir}/planner.pid")"
  else
    printf 'Planner: stopped\n'; result=1
  fi
  if pid_alive "${runtime_dir}/mtr.pid"; then
    printf 'MTR: running (PID %s)\n' "$(<"${runtime_dir}/mtr.pid")"
  else
    printf 'MTR: stopped\n'; result=1
  fi
  return "${result}"
}

health_stack() {
  source_runtime
  "${script_dir}/health_check.sh" "${@:2}"
}

build_stack() {
  # Force Jazzy's system Python even if the caller entered with Conda active.
  export PATH="/usr/bin:/bin:${PATH}"
  source_setup "/opt/ros/${ROS_DISTRO}/setup.bash"
  (
    cd "${CPX_PREDICTION_ROS_WS}"
    colcon build --symlink-install --packages-select mtr_prediction_msgs
  )
  source_setup "${CPX_PREDICTION_ROS_WS}/install/setup.bash"
  (
    cd "${CPX_ROS_WS}"
    colcon build --symlink-install \
      --packages-select cpx_interfaces cpx_comm_test cpx_planning
  )
}

logs_stack() {
  mkdir -p "${log_dir}"
  touch "${log_dir}/mtr.log" "${log_dir}/planner.log"
  tail -n 80 -F "${log_dir}/mtr.log" "${log_dir}/planner.log"
}

usage() {
  printf 'Usage: %s {build|start|stop|restart|status|health|health-live|logs|manifest}\n' "$0"
}

case ${1:-} in
  build) build_stack ;;
  start) start_stack ;;
  stop) stop_stack ;;
  restart) stop_stack; start_stack ;;
  status) status_stack ;;
  health) health_stack "$@" ;;
  health-live) health_stack health --live ;;
  logs) logs_stack ;;
  manifest) "${script_dir}/capture_runtime_manifest.sh" ;;
  *) usage; exit 64 ;;
esac
