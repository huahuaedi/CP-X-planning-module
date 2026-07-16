#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_REPO_DIR="${SCRIPT_DIR}/map_repo"
SOURCE_DIR="${MAP_REPO_DIR}/source"
BUILD_DIR="${MAP_REPO_DIR}/build"
LOG_DIR="${MAP_REPO_DIR}/log"
INSTALL_DIR="${MAP_REPO_DIR}/install"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UPSTREAM_URL="${AD_MAP_UPSTREAM_URL:-https://github.com/carla-simulator/map.git}"
UPSTREAM_TAG="v2.3.0"
BOOST_VERSION="1.71.0"
BOOST_PACKAGE_BASENAME="boost_${BOOST_VERSION//./_}"
BOOST_ARCHIVE="${BUILD_DIR}/${BOOST_PACKAGE_BASENAME}.tar.gz"
BOOST_SOURCE_DIR="${BUILD_DIR}/${BOOST_PACKAGE_BASENAME}"
BOOST_INSTALL_DIR="${INSTALL_DIR}/boost"
PROJ_BUILD_DIR="${BUILD_DIR}/proj4"
PROJ_INSTALL_DIR="${INSTALL_DIR}/PROJ4"
BUILD_PYTHON_VENV="${BUILD_DIR}/python-venv"
BOOST_URL="https://archives.boost.io/release/${BOOST_VERSION}/source/${BOOST_PACKAGE_BASENAME}.tar.gz"
BOOST_SHA256="96b34f7468f26a141f6020efb813f1a2f3dfb9797ecf76a7d7cbd843cc95f5bd"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CASTXML_CXX="${CASTXML_CXX:-}"

case "${1:-}" in
  "")
    ;;
  --clean)
    # Rebuild generated artifacts but retain the expensive source checkout so
    # transient GitHub/DNS outages do not prevent an otherwise local rebuild.
    rm -rf "${BUILD_DIR}" "${LOG_DIR}" "${INSTALL_DIR}"
    ;;
  --clean-all)
    # Use only when the pinned source checkout itself must be downloaded again.
    rm -rf "${SOURCE_DIR}" "${BUILD_DIR}" "${LOG_DIR}" "${INSTALL_DIR}"
    ;;
  -h|--help)
    echo "Usage: $0 [--clean|--clean-all]"
    echo "  --clean      rebuild while preserving the downloaded AD-map source"
    echo "  --clean-all  delete and redownload the AD-map source before rebuilding"
    exit 0
    ;;
  *)
    echo "Unknown option: ${1}" >&2
    echo "Usage: $0 [--clean|--clean-all]" >&2
    exit 2
    ;;
esac

for command_name in git cmake c++ castxml curl tar sha256sum sed; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing build command: ${command_name}" >&2
    echo "On Ubuntu install: build-essential cmake git curl castxml libpugixml-dev libproj-dev libspdlog-dev libfmt-dev libosmium2-dev liblapacke-dev libgtest-dev python3-dev" >&2
    exit 1
  fi
done

if [[ -z "${CASTXML_CXX}" ]]; then
  for compiler_candidate in g++-11 g++-12 g++-10 g++-9; do
    if command -v "${compiler_candidate}" >/dev/null 2>&1; then
      CASTXML_CXX="$(command -v "${compiler_candidate}")"
      break
    fi
  done
fi
if [[ -z "${CASTXML_CXX}" ]]; then
  echo "AD-map v2.3's CastXML generator needs a GCC 9-12 C++ compiler (GCC 11 recommended)." >&2
  echo "Install g++-11 or set CASTXML_CXX to a compatible compiler path." >&2
  exit 1
fi
CASTXML_CC="${CASTXML_CC:-${CASTXML_CXX/g++/gcc}}"
if [[ ! -x "${CASTXML_CC}" ]]; then
  echo "Matching C compiler not found: ${CASTXML_CC}" >&2
  exit 1
fi
BOOST_GCC_VERSION="${CASTXML_CXX##*g++-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
RUNTIME_PYTHON_EXECUTABLE="$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
RUNTIME_PYTHON_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
PYTHON_INCLUDE_DIR="$(${PYTHON_BIN} -c 'import sysconfig; print(sysconfig.get_path("include"))')"
${PYTHON_BIN} - <<'PY'
import sys
if sys.version_info[:2] != (3, 7):
    raise SystemExit(
        f"This AD-map v2.3.0 build targets Python 3.7; active interpreter is {sys.version.split()[0]}"
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
"${BUILD_PYTHON_EXECUTABLE}" -m pip install --upgrade \
  'pip<24.1' 'setuptools<69' 'wheel<0.39'
"${BUILD_PYTHON_EXECUTABLE}" -m pip install \
  'colcon-common-extensions==0.3.0' \
  'pygccxml==3.0.2' \
  'pyplusplus==1.8.7' \
  'unittest-xml-reporting==3.2.0'

if [[ -e "${SOURCE_DIR}" && ! -d "${SOURCE_DIR}/.git" ]]; then
  echo "${SOURCE_DIR} exists but is not the expected Git checkout; move it or run with --clean-all." >&2
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  clone_ok=0
  for clone_attempt in 1 2 3; do
    rm -rf "${SOURCE_DIR}"
    echo "Cloning AD-map ${UPSTREAM_TAG} (attempt ${clone_attempt}/3)..."
    if git clone --branch "${UPSTREAM_TAG}" --depth 1 --shallow-submodules --recurse-submodules \
      "${UPSTREAM_URL}" "${SOURCE_DIR}"; then
      clone_ok=1
      break
    fi
    sleep 2
  done
  if [[ "${clone_ok}" -ne 1 ]]; then
    echo "Could not download AD-map from ${UPSTREAM_URL}." >&2
    echo "Check DNS/network access (for example: getent hosts github.com), then rerun this command." >&2
    echo "Once downloaded, normal and --clean builds reuse map_repo/source and work without fetching it again." >&2
    exit 1
  fi
else
  if ! git -C "${SOURCE_DIR}" rev-parse --verify --quiet "${UPSTREAM_TAG}^{commit}" >/dev/null; then
    echo "Pinned tag ${UPSTREAM_TAG} is not cached; downloading it..."
    git -C "${SOURCE_DIR}" -c fetch.recurseSubmodules=false fetch --depth 1 --force \
      origin "refs/tags/${UPSTREAM_TAG}:refs/tags/${UPSTREAM_TAG}"
  fi
  source_head="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
  pinned_head="$(git -C "${SOURCE_DIR}" rev-parse "${UPSTREAM_TAG}^{commit}")"
  if [[ "${source_head}" != "${pinned_head}" ]]; then
    git -C "${SOURCE_DIR}" checkout --detach "${UPSTREAM_TAG}"
  else
    echo "Reusing cached AD-map ${UPSTREAM_TAG} source at ${SOURCE_DIR}."
  fi

  missing_submodules="$(git -C "${SOURCE_DIR}" submodule status --recursive | sed -n '/^-/p')"
  if [[ -n "${missing_submodules}" ]]; then
    echo "Completing missing AD-map submodules..."
    git -C "${SOURCE_DIR}" submodule sync --recursive
    git -C "${SOURCE_DIR}" submodule update --init --recursive --depth 1
  fi
fi

# The v2.3 wrapper helper hardcodes /usr/bin/g++. GCC 13+ headers cannot be
# parsed by its CastXML generator, so point that generated dependency at the
# explicitly selected older compiler. This changes only the ignored checkout.
WRAPPER_HELPER="${SOURCE_DIR}/cmake/python/python_wrapper_helper.py"
sed -i \
  "s|compiler_path = \"/usr/bin/g++\"|compiler_path = \"${CASTXML_CXX}\"|" \
  "${WRAPPER_HELPER}"
if ! grep -Fq "compiler_path = \"${CASTXML_CXX}\"" "${WRAPPER_HELPER}"; then
  echo "Could not configure the AD-map CastXML compiler in ${WRAPPER_HELPER}." >&2
  exit 1
fi

# AD-map 2.3 bundles the legacy PROJ 4.9 source required by its OpenDRIVE
# reader, but that submodule is not a colcon package. Build it first so the
# reader does not accidentally pick a newer system PROJ without proj_api.h.
cmake -S "${SOURCE_DIR}/dependencies/PROJ" -B "${PROJ_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PROJ_INSTALL_DIR}" \
  -DBUILD_LIBPROJ_SHARED=ON \
  -DPROJ4_TESTS=OFF
cmake --build "${PROJ_BUILD_DIR}" --parallel "${BUILD_JOBS}"
cmake --install "${PROJ_BUILD_DIR}"

# Ubuntu's libboost-python-dev targets the distribution's default Python only.
# Build Boost.Python privately so the native bindings always match the exact
# ABI that runs OpenCDA (Python 3.7 in the carla307 environment).
# The generated build venv intentionally has no NumPy: AD-map does not require
# Boost.NumPy, and Boost's optional NumPy binding is incompatible with
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
    --with-python-root="${RUNTIME_PYTHON_PREFIX}"
  ./b2 \
    --user-config=/dev/null \
    --prefix="${BOOST_INSTALL_DIR}" \
    --with-python \
    --with-filesystem \
    --with-system \
    --with-program_options \
    -j"${BUILD_JOBS}" \
    toolset="gcc-${BOOST_GCC_VERSION}" \
    python="${PYTHON_VERSION}" \
    cxxflags="-fPIC -I${PYTHON_INCLUDE_DIR}" \
    link=shared \
    runtime-link=shared \
    variant=release \
    install
)

BOOST_CMAKE_DIR="$(find "${BOOST_INSTALL_DIR}" -type d -path "*/cmake/Boost-${BOOST_VERSION}" -print -quit)"
if [[ -z "${BOOST_CMAKE_DIR}" ]]; then
  echo "The private Boost installation has no BoostConfig.cmake directory." >&2
  exit 1
fi

PATH="${BUILD_PYTHON_VENV}/bin:${PATH}" \
CC="${CASTXML_CC}" \
CXX="${CASTXML_CXX}" \
CMAKE_PREFIX_PATH="${RUNTIME_PYTHON_PREFIX}:${BOOST_INSTALL_DIR}:${PROJ_INSTALL_DIR}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" \
"${BUILD_PYTHON_VENV}/bin/colcon" --log-base "${LOG_DIR}" build \
  --base-paths "${SOURCE_DIR}" \
  --build-base "${BUILD_DIR}" \
  --install-base "${INSTALL_DIR}" \
  --packages-up-to ad_map_access \
  --metas "${SOURCE_DIR}/colcon.meta" \
  --cmake-clean-cache \
  --cmake-args \
    -DBUILD_TESTING=OFF \
    -DBUILD_PYTHON_BINDING=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_CXX_FLAGS=-include cstdint" \
    -DDISABLE_WARNINGS_AS_ERRORS=ON \
    -DBoost_USE_STATIC_LIBS=OFF \
    "-DBoost_DIR=${BOOST_CMAKE_DIR}" \
    "-DCMAKE_INCLUDE_PATH=${PROJ_INSTALL_DIR}/include" \
    "-DCMAKE_LIBRARY_PATH=${PROJ_INSTALL_DIR}/lib" \
    "-DCMAKE_BUILD_RPATH=${BOOST_INSTALL_DIR}/lib" \
    "-DCMAKE_INSTALL_RPATH=${BOOST_INSTALL_DIR}/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    "-DPYTHON_BINDING_VERSIONS=${PYTHON_VERSION}" \
    "-DPYTHON_EXECUTABLE:FILEPATH=${BUILD_PYTHON_EXECUTABLE}"

PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${RUNTIME_PYTHON_EXECUTABLE}" - <<'PY'
from global_planner import import_ad_map_access
module = import_ad_map_access()
print("AD-map import OK:", module.__file__)
PY

echo "AD-map runtime installed at ${INSTALL_DIR}"
