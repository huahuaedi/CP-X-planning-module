# AD Map Access runtime

`install/` is generated locally and is intentionally not committed. Native
Python extensions are tied to the active Python ABI, operating system, system
libraries, and build paths.

From the repository root, build the pinned runtime with:

```bash
PYTHON_BIN="$(command -v python)" \
  opencda/planning_module/Global_Planner/build_ad_map.sh --clean
```

The script builds `carla-simulator/map` v3.0.0 and installs its isolated colcon
packages under this directory's `install/`. It also builds Boost 1.83 privately
against the selected interpreter, avoiding the Python 3.10/3.12 mismatch in
Ubuntu's system Boost.Python package. Native build tools are installed in a
generated venv with the same Python ABI; the active environment's packages and
Python version are not changed, and Boost is not installed globally.
`global_planner/runtime.py` locates the generated install directory
automatically. `GLOBAL_PLANNER_AD_MAP_INSTALL` or `AD_MAP_INSTALL_ROOT` may
override it.
