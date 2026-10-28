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

CPX_ROS_WS=${CPX_ROS_WS:-${workspace_default}}
CPX_PLANNER_ROOT=${CPX_PLANNER_ROOT:-${planner_default}}
CPX_PLANNER_DEPS=${CPX_PLANNER_DEPS:-/root/cpx-python312-deps}
CPX_PREDICTION_ROOT=${CPX_PREDICTION_ROOT:-/root/CP-X-Prediction}
MTR_VENV=${MTR_VENV:-/root/mtr-ros-py312}
output=${1:-${CPX_ROS_WS}/runtime-manifest.txt}

revision() {
  local repo=$1
  if [[ -d "${repo}/.git" ]]; then
    git -c safe.directory="${repo}" -C "${repo}" rev-parse HEAD 2>/dev/null || printf 'unknown'
  else
    printf 'not-a-git-checkout'
  fi
}

{
  printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'ros_distro=%s\n' "${ROS_DISTRO:-jazzy}"
  printf 'system_python=%s\n' "$(/usr/bin/python3 --version 2>&1)"
  printf 'cpx_ros_ws_revision=%s\n' "$(revision "${CPX_ROS_WS}")"
  printf 'planner_revision=%s\n' "$(revision "${CPX_PLANNER_ROOT}")"
  printf 'prediction_revision=%s\n' "$(revision "${CPX_PREDICTION_ROOT}")"
  printf 'planner_overlay=%s\n' "${CPX_PLANNER_DEPS}"
  printf 'mtr_venv=%s\n' "${MTR_VENV}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'gpu='
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1
  fi
  printf '\n[planner-python]\n'
  PYTHONPATH="${CPX_PLANNER_DEPS}:${CPX_PLANNER_ROOT}" /usr/bin/python3 - <<'PY'
from importlib import metadata

for distribution in ('numpy', 'scipy', 'osqp', 'qdldl', 'PyYAML', 'setuptools'):
    try:
        print('{}=={}'.format(distribution, metadata.version(distribution)))
    except metadata.PackageNotFoundError:
        print('{}=MISSING'.format(distribution))
PY
  printf '\n[mtr-python]\n'
  if [[ -x "${MTR_VENV}/bin/python" ]]; then
    "${MTR_VENV}/bin/python" - <<'PY'
from importlib import metadata

for distribution in ('torch', 'numpy', 'scipy', 'PyYAML', 'empy', 'setuptools'):
    try:
        print('{}=={}'.format(distribution, metadata.version(distribution)))
    except metadata.PackageNotFoundError:
        print('{}=MISSING'.format(distribution))
PY
  fi
} >"${output}"

printf 'Wrote %s\n' "${output}"
