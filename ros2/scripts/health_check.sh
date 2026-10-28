#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_default=$(cd -- "${script_dir}/.." && pwd)
planner_default=$(cd -- "${workspace_default}/.." && pwd)

CPX_ROS_WS=${CPX_ROS_WS:-${workspace_default}}
CPX_PLANNER_ROOT=${CPX_PLANNER_ROOT:-${planner_default}}
CPX_PLANNER_DEPS=${CPX_PLANNER_DEPS:-/root/cpx-python312-deps}
CPX_PREDICTION_ROOT=${CPX_PREDICTION_ROOT:-/root/CP-X-Prediction}
CPX_PREDICTION_ROS_WS=${CPX_PREDICTION_ROS_WS:-${CPX_PREDICTION_ROOT}/ROS2-Code}
MTR_VENV=${MTR_VENV:-/root/mtr-ros-py312}
MTR_ROOT=${MTR_ROOT:-${CPX_PREDICTION_ROOT}/Intersection-Code/Intersection-MTR/Prediction/MTR}
CPX_PLANNER_SHARDS=${CPX_PLANNER_SHARDS:-2}

live=false
if [[ ${1:-} == "--live" ]]; then
  live=true
fi

failures=0
warnings=0

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf '[FAIL] %s\n' "$*"; failures=$((failures + 1)); }

require_path() {
  if [[ -e $1 ]]; then pass "$2"; else fail "$2: missing $1"; fi
}

require_path "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" "ROS ${ROS_DISTRO:-jazzy} installation"
require_path "${CPX_ROS_WS}/install/setup.bash" "CP-X ROS 2 install space"
require_path "${CPX_PREDICTION_ROS_WS}/install/setup.bash" "Prediction ROS interface install space"
require_path "${MTR_VENV}/bin/python" "MTR Python environment"

if PYTHONPATH="${CPX_PLANNER_DEPS}:${CPX_PLANNER_ROOT}:${PYTHONPATH:-}" /usr/bin/python3 -c \
  'import rclpy, numpy, scipy, osqp, yaml; import opencda' >/dev/null 2>&1; then
  pass "Planner Python imports"
else
  fail "Planner Python imports"
fi

if PATH="${MTR_VENV}/bin:${PATH}" PYTHONPATH="${MTR_ROOT}/live_inference:${MTR_ROOT}:${PYTHONPATH:-}" \
  "${MTR_VENV}/bin/python" -c \
  'import torch; from mtr.ops.knn import knn_cuda; from mtr.ops.attention import attention_cuda' \
  >/dev/null 2>&1; then
  pass "MTR PyTorch and CUDA extension imports"
else
  fail "MTR PyTorch and CUDA extension imports"
fi

nodes=$(ros2 node list 2>/dev/null || true)
for suffix in /opencda_bridge_node /mtr_prediction_node; do
  if grep -qE "${suffix}$" <<<"${nodes}"; then pass "ROS node ${suffix}"; else fail "ROS node ${suffix}"; fi
done

planner_count=$(grep -cE '/planner_node$' <<<"${nodes}" || true)
global_count=$(grep -cE '/global_planner_node$' <<<"${nodes}" || true)
if (( planner_count >= CPX_PLANNER_SHARDS )); then
  pass "Planner workers ${planner_count}/${CPX_PLANNER_SHARDS}"
else
  fail "Planner workers ${planner_count}/${CPX_PLANNER_SHARDS}"
fi
if (( global_count >= CPX_PLANNER_SHARDS )); then
  pass "Global-planner workers ${global_count}/${CPX_PLANNER_SHARDS}"
else
  fail "Global-planner workers ${global_count}/${CPX_PLANNER_SHARDS}"
fi

for topic in /cpx/localization /cpx/planner_control_output /prediction/scene /prediction/frame; do
  info=$(ros2 topic info "${topic}" 2>/dev/null || true)
  if grep -q 'Publisher count: [1-9]' <<<"${info}"; then
    pass "Topic publisher ${topic}"
  else
    if [[ ${topic} == /cpx/localization ]]; then
      warn "${topic} has no publisher yet (normal before OpenCDA connects)"
    else
      fail "Topic publisher ${topic}"
    fi
  fi
done

for port in 5051 5052 5053 5054 5055; do
  if ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .; then
    pass "Bridge TCP port ${port} listening"
  else
    fail "Bridge TCP port ${port} not listening"
  fi
done

if ${live}; then
  for topic in /cpx/localization /prediction/frame /cpx/planner_control_output; do
    sample=$(timeout 4 ros2 topic hz "${topic}" --window 20 2>&1 || true)
    if grep -q 'average rate:' <<<"${sample}"; then
      rate=$(grep 'average rate:' <<<"${sample}" | tail -1 | awk '{print $3}')
      pass "Live ${topic}: ${rate} Hz"
    else
      fail "No live samples on ${topic} within 4 s"
    fi
  done
else
  warn "Message-rate checks skipped; use '$0 --live' while OpenCDA is running"
fi

printf '\nHealth summary: %d failure(s), %d warning(s)\n' "${failures}" "${warnings}"
(( failures == 0 ))
