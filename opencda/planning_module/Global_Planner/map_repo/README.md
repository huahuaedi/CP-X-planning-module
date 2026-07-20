# AD Map Access runtime

`install/`, `log/`, and `source/` are generated locally and are intentionally
not committed or pushed.

- `install/` contains native Python extensions and shared libraries tied to the
  active Python ABI, operating system, compiler, system libraries, and build
  paths.
- `log/` contains disposable colcon build logs.
- `source/` contains the downloaded pinned upstream AD-map checkout. Keeping it
  locally avoids re-downloading source on `--clean` rebuilds, but it should not
  be stored in this repository.

From the repository root, build the pinned runtime with:

```bash
PYTHON_BIN="$(command -v python)" \
  opencda/planning_module/Global_Planner/build_ad_map.sh --clean
```

The script builds `carla-simulator/map` v2.3.0 and installs its isolated colcon
packages under this directory's `install/`. It also builds Boost 1.71 privately
against the selected Python 3.7 interpreter, avoiding mismatches with Ubuntu's
system Boost.Python package. Native build tools are installed in a
generated venv with the same Python ABI; the active environment's packages and
Python version are not changed, and Boost is not installed globally.
`global_planner/runtime.py` locates the generated install directory
automatically. `GLOBAL_PLANNER_AD_MAP_INSTALL` or `AD_MAP_INSTALL_ROOT` may
override it.

`--clean` preserves the downloaded `source/` checkout and rebuilds only the
generated outputs. Use `--clean-all` only to force a fresh source download.
This lets later clean rebuilds proceed without contacting GitHub.
