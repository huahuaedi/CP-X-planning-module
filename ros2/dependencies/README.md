# Reproducible runtime baseline

The working deployment deliberately uses two Python 3.12 environments:

- ROS/Planner: `/usr/bin/python3` plus the read-only overlay named by
  `CPX_PLANNER_DEPS`.
- MTR: the virtual environment named by `MTR_VENV`, containing PyTorch and the
  MTR CUDA extensions.

Do not activate a Conda base environment before building ROS interfaces. It can
make CMake select a different Python interpreter from the one used by Jazzy.

## Planner overlay

```bash
mkdir -p /root/cpx-python312-deps
/usr/bin/python3 -m pip install --target /root/cpx-python312-deps \
  -r dependencies/planner-python312.lock.txt
```

The overlay is placed at the front of `PYTHONPATH` by the launcher. This is
required because Ubuntu Noble does not provide the `python3-osqp` package used
by the Planner.

## MTR environment

Create a Python 3.12 virtual environment, install the versions in
`mtr-python312.lock.txt` using the matching PyTorch CUDA wheel index, and then
rebuild both MTR CUDA extensions inside that environment. A copied `.so` from a
different Python/PyTorch/CUDA ABI is not valid even if its filename looks
compatible.

Validated hardware/runtime snapshot:

- Ubuntu 24.04 (Noble)
- ROS 2 Jazzy
- Python 3.12.3
- PyTorch 2.11.0+cu128
- NVIDIA RTX PRO 4000 Blackwell, compute capability 12.0
- NVIDIA driver 595.91.07

## Source ownership

The `ros2/` workspace does not vendor the Prediction repository. Its Planner
core is the parent repository. Run
`scripts/capture_runtime_manifest.sh` after installation to record the exact
source revisions and package versions used by a test. The generated
`runtime-manifest.txt` is intentionally ignored; attach it to test artifacts.
