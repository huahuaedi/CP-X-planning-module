#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_REPO_DIR="${SCRIPT_DIR}/map_repo"
SOURCE_DIR="${MAP_REPO_DIR}/source"
BUILD_DIR="${MAP_REPO_DIR}/build"
LOG_DIR="${MAP_REPO_DIR}/log"
INSTALL_DIR="${MAP_REPO_DIR}/install"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UPSTREAM_URL="https://github.com/carla-simulator/map.git"
UPSTREAM_TAG="v3.0.0"
BOOST_VERSION="1.83.0"
BOOST_PACKAGE_BASENAME="boost_${BOOST_VERSION//./_}"
BOOST_ARCHIVE="${BUILD_DIR}/${BOOST_PACKAGE_BASENAME}.tar.gz"
BOOST_SOURCE_DIR="${BUILD_DIR}/${BOOST_PACKAGE_BASENAME}"
BOOST_INSTALL_DIR="${INSTALL_DIR}/boost"
BUILD_PYTHON_VENV="${BUILD_DIR}/python-venv"
BOOST_URL="https://archives.boost.io/release/${BOOST_VERSION}/source/${BOOST_PACKAGE_BASENAME}.tar.gz"
BOOST_SHA256="c0685b68dd44cc46574cce86c4e17c0f611b15e195be9848dfd0769a0a207628"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf "${SOURCE_DIR}" "${BUILD_DIR}" "${LOG_DIR}" "${INSTALL_DIR}"
fi

for command_name in git cmake c++ castxml curl tar sha256sum; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing build command: ${command_name}" >&2
    echo "On Ubuntu install: build-essential cmake git curl castxml libpugixml-dev libproj-dev libspdlog-dev libfmt-dev libosmium2-dev liblapacke-dev libgtest-dev python3-dev" >&2
    exit 1
  fi
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
RUNTIME_PYTHON_EXECUTABLE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
PYTHON_INCLUDE_DIR="$(${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_path("include"))')"
${PYTHON_BIN} - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit(
        f"AD-map v3.0.0 requires Python 3.10-3.13; active interpreter is {sys.version.split()[0]}"
    )
PY

if [[ ! -f "${PYTHON_INCLUDE_DIR}/Python.h" ]]; then
  echo "Python development headers do not match ${RUNTIME_PYTHON_EXECUTABLE}." >&2
  echo "Missing: ${PYTHON_INCLUDE_DIR}/Python.h" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${LOG_DIR}" "${INSTALL_DIR}"
"${RUNTIME_PYTHON_EXECUTABLE}" -m venv "${BUILD_PYTHON_VENV}"
BUILD_PYTHON_EXECUTABLE="${BUILD_PYTHON_VENV}/bin/python"
BUILD_PYTHON_PREFIX="$(${BUILD_PYTHON_EXECUTABLE} -c 'import sys; print(sys.prefix)')"

"${BUILD_PYTHON_EXECUTABLE}" -m pip install --upgrade \
  'setuptools<80' colcon-common-extensions wheel pygccxml pyplusplus xmlrunner

if [[ -e "${SOURCE_DIR}" && ! -d "${SOURCE_DIR}/.git" ]]; then
  echo "${SOURCE_DIR} exists but is not the expected Git checkout; move it or run with --clean." >&2
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --branch "${UPSTREAM_TAG}" --depth 1 --shallow-submodules --recurse-submodules \
    "${UPSTREAM_URL}" "${SOURCE_DIR}"
else
  git -C "${SOURCE_DIR}" -c fetch.recurseSubmodules=false fetch --depth 1 --force \
    origin "refs/tags/${UPSTREAM_TAG}:refs/tags/${UPSTREAM_TAG}"
  git -C "${SOURCE_DIR}" checkout --detach "${UPSTREAM_TAG}"
  git -C "${SOURCE_DIR}" submodule sync --recursive
  git -C "${SOURCE_DIR}" submodule update --init --recursive --depth 1
fi

# Ubuntu's libboost-python-dev targets the distribution's default Python only.
# Build Boost.Python privately so the native bindings always match the exact
# ABI that runs OpenCDA (for example, Python 3.10 in the carla310 environment).
# The generated build venv intentionally has no NumPy: AD-map does not require
# Boost.NumPy, and Boost 1.83's optional NumPy binding is incompatible with
# NumPy 2.x that may already be installed in the OpenCDA runtime environment.
if [[ ! -d "${BOOST_SOURCE_DIR}" ]]; then
  curl --fail --location --retry 3 --output "${BOOST_ARCHIVE}" "${BOOST_URL}"
  printf '%s  %s\n' "${BOOST_SHA256}" "${BOOST_ARCHIVE}" | sha256sum --check -
  tar -xzf "${BOOST_ARCHIVE}" -C "${BUILD_DIR}"
fi

(
  cd "${BOOST_SOURCE_DIR}"
  ./bootstrap.sh \
    --prefix="${BOOST_INSTALL_DIR}" \
    --with-libraries=python,filesystem,system,program_options \
    --with-python="${BUILD_PYTHON_EXECUTABLE}" \
    --with-python-version="${PYTHON_VERSION}" \
    --with-python-root="${BUILD_PYTHON_PREFIX}"
  ./b2 \
    --user-config=/dev/null \
    --prefix="${BOOST_INSTALL_DIR}" \
    --with-python \
    --with-filesystem \
    --with-system \
    --with-program_options \
    -j"${BUILD_JOBS}" \
    python="${PYTHON_VERSION}" \
    cxxflags="-fPIC -I${PYTHON_INCLUDE_DIR}" \
    link=shared \
    runtime-link=shared \
    variant=release \
    install
)

BOOST_CMAKE_DIR="$(find "${BOOST_INSTALL_DIR}" -type d -path '*/cmake/Boost-1.83.0' -print -quit)"
if [[ -z "${BOOST_CMAKE_DIR}" ]]; then
  echo "The private Boost installation has no BoostConfig.cmake directory." >&2
  exit 1
fi

"${BUILD_PYTHON_VENV}/bin/colcon" --log-base "${LOG_DIR}" build \
  --base-paths "${SOURCE_DIR}" \
  --build-base "${BUILD_DIR}" \
  --install-base "${INSTALL_DIR}" \
  --packages-up-to ad_map_access \
  --metas "${SOURCE_DIR}/colcon_python.meta" \
  --cmake-args \
    -DBUILD_TESTING=OFF \
    -DBUILD_PYTHON_BINDING=ON \
    -DDISABLE_WARNINGS_AS_ERRORS=ON \
    -DBoost_USE_STATIC_LIBS=OFF \
    "-DBoost_DIR=${BOOST_CMAKE_DIR}" \
    "-DCMAKE_BUILD_RPATH=${BOOST_INSTALL_DIR}/lib" \
    "-DCMAKE_INSTALL_RPATH=${BOOST_INSTALL_DIR}/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    "-DPYTHON_BINDING_VERSION=${PYTHON_VERSION}" \
    "-DPython3_EXECUTABLE:FILEPATH=${BUILD_PYTHON_EXECUTABLE}"

PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${RUNTIME_PYTHON_EXECUTABLE}" - <<'PY'
from global_planner import import_ad_map_access
module = import_ad_map_access()
print("AD-map import OK:", module.__file__)
PY

echo "AD-map runtime installed at ${INSTALL_DIR}"
