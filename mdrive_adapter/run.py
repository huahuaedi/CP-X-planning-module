"""Launch MDrive with CP-X GT planning; copy routes before MDrive densifies them."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import signal
import socket
import time
from xml.etree import ElementTree

if __package__:
    from .parallel import THREAD_ENV
    from .summarize import summarize
else:
    from parallel import THREAD_ENV
    from summarize import summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mdrive-root", type=Path, default=ROOT.parent / "MDrive")
    parser.add_argument("--routes-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "mdrive_adapter/cpx_gt.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--record-video", action="store_true", help="Record spectator RGB without using images for planning")
    parser.add_argument("--execution", choices=("serial", "parallel", "worker-serial"), default="serial")
    parser.add_argument("--worker-cpus", default="auto", help="One logical CPU from each distinct physical core, comma separated")
    parser.add_argument("--direct", action="store_true",
                        help="Call the same MDrive evaluator directly (one leaf scenario; external CARLA)")
    args, extra = parser.parse_known_args()
    if args.execution != "serial" and not args.direct:
        parser.error("Process workers currently require --direct for an explicit CARLA endpoint")
    forbidden = ("--agent", "--agent-config", "--planner", "--track", "--results-tag",
                 "--results-subdir", "--output-root", "--recovery-mode", "--openloop")
    for token in extra:
        if token.split("=", 1)[0] in forbidden:
            parser.error("%s is owned by this closed-loop CP-X launcher" % token)
    mdrive = args.mdrive_root.resolve()
    source = args.routes_dir.resolve()
    config = args.config.resolve()
    if not source.is_dir() or not config.is_file():
        parser.error("routes directory or agent configuration does not exist")
    run_dir = (args.run_dir or ROOT / ".runtime" / ("run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if run_dir == mdrive or mdrive in run_dir.parents:
        parser.error("run-dir must be outside MDrive")
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(str(source), str(run_dir / "routes"))
    shutil.copyfile(str(config), str(run_dir / "agent.yaml"))
    env = os.environ.copy()
    env["CPX_RECORD_VIDEO"] = "1" if args.record_video else "0"
    env["CPX_SCENE_NAME"] = source.name
    env["CPX_EXECUTION"] = args.execution
    env["CPX_WORKER_CPUS"] = args.worker_cpus
    # Avoid MDrive starting a second conda scheduler and reallocating ports.
    env.pop("CONDA_DEFAULT_ENV", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(run_dir / "matplotlib")
    for key in THREAD_ENV:
        env[key] = "1"
    env.setdefault("CARLA_ROOT", str(mdrive / "carla912"))
    paths = [ROOT / ".runtime/deps", ROOT, ROOT / "opencda/planning_module",
             mdrive, mdrive / "simulation/leaderboard", mdrive / "simulation/scenario_runner",
             Path(env["CARLA_ROOT"]) / "PythonAPI/carla"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get("PYTHONPATH", "")
    # Fail before launching a simulator if the current planning backend is missing.
    subprocess.check_call([sys.executable, "-c",
        "import sys, yaml; "
        "from opencda.planning_module.opencda_bridge.cpx_mpc_planner import CPXMPCPlannerBridge; "
        "from opencda.planning_module.Global_Planner.global_planner.runtime import import_ad_map_access; "
        "config = yaml.safe_load(open(sys.argv[1])) or {}; "
        "import_ad_map_access(config.get('bridge', {}).get('ad_map_install_root'))",
        str(config)], cwd=str(ROOT), env=env)
    agent_args = ["--agent", str(ROOT / "mdrive_adapter/cpx_agent.py"),
                  "--agent-config", str(run_dir / "agent.yaml"), "--track", "MAP"]
    server_command = None
    if args.direct:
        direct_parser = argparse.ArgumentParser(add_help=False)
        direct_parser.add_argument("--port", default="2000")
        direct_parser.add_argument("--host", default="127.0.0.1")
        direct_parser.add_argument("--traffic-manager-port", default="8000")
        direct_parser.add_argument("--gpus", default=None)
        direct_parser.add_argument("--timeout", default="120")
        direct_parser.add_argument("--start-carla", action="store_true")
        direct_parser.add_argument("--graphics-adapter", default="0",
                                   help="Vulkan adapter index for CARLA; independent of CUDA_VISIBLE_DEVICES")
        direct_args = direct_parser.parse_args(extra)
        env["CPX_CARLA_HOST"] = direct_args.host
        env["CPX_CARLA_PORT"] = direct_args.port
        routes = run_dir / "routes"
        route_groups = [ElementTree.parse(str(path)).getroot().findall("route")
                        for path in routes.glob("*.xml")]
        manifest = routes / "actors_manifest.json"
        if manifest.exists():
            with manifest.open() as stream:
                ego_count = len(json.load(stream).get("ego", []))
            env["CUSTOM_ACTOR_MANIFEST"] = str(manifest)
            env["CUSTOM_ACTOR_ROOT"] = str(routes)
        else:
            ego_count = sum(any(r.get("role", "ego") == "ego" for r in group)
                            for group in route_groups)
        if ego_count < 1:
            parser.error("--direct expects one leaf scenario containing ego route XMLs")
        if direct_args.start_carla:
            if direct_args.host not in ("127.0.0.1", "localhost"):
                parser.error("--start-carla requires a local host")
            towns = {route.get("town") for group in route_groups for route in group}
            if len(towns) != 1 or not next(iter(towns)):
                parser.error("--direct --start-carla requires a single map")
            binary = Path(env["CARLA_ROOT"]) / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
            # Build MDrive's existing RPC compatibility shim outside MDrive.
            shim_source = mdrive / "tools/close_ebadf_suppress.c"
            if shim_source.exists():
                shim = run_dir / "close_ebadf_suppress.so"
                subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", str(shim), str(shim_source), "-ldl"])
            else:
                shim = None
            server_command = [str(binary), "CarlaUE4", "/Game/Carla/Maps/" + next(iter(towns)).split("/")[-1],
                              "-RenderOffScreen", "-nosound", "-quality-level=Low",
                              "-carla-rpc-port=" + direct_args.port,
                              "-graphicsadapter=" + direct_args.graphics_adapter]
        results = run_dir / "results"
        (results / "image").mkdir(parents=True)
        env.update({"RESULT_ROOT": str(results), "SAVE_PATH": str(results / "image"),
                    "ROUTES": str(routes), "ROUTES_DIR": str(routes),
                    "CHECKPOINT_ENDPOINT": str(results / "results.json"),
                    "CARLA_RECOVERY_MODE": "off", "CUSTOM_ACTOR_CONTROL_MODE": "policy",
                    "CUSTOM_EGO_NORMALIZE_Z": "1"})
        env.pop("CUSTOM_EGO_LOG_REPLAY", None)
        env.pop("CUSTOM_USE_PRECOMPUTED_DENSE_ROUTE", None)
        if direct_args.gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = direct_args.gpus
        command = [sys.executable, str(mdrive / "simulation/leaderboard/leaderboard/leaderboard_evaluator_parameter.py"),
                   "--routes_dir", str(routes), "--ego-num", str(ego_count)] + agent_args + [
                   "--scenario_parameter", str(mdrive / "simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_no_npc.yaml"),
                   "--scenarios", str(mdrive / "simulation/leaderboard/data/scenarios/no_scenarios.json"),
                   "--checkpoint", str(results / "results.json"),
                   "--host", direct_args.host, "--port", direct_args.port,
                   "--trafficManagerPort", direct_args.traffic_manager_port,
                   "--timeout", direct_args.timeout]
    else:
        command = [sys.executable, str(mdrive / "tools/run_custom_eval.py"),
                   "--planner", "minimal"] + agent_args + [
                   "--routes-dir", str(run_dir / "routes"),
                   "--results-tag", str(run_dir / "results"), "--output-root", str(run_dir / "prepared"),
                   "--disable-checkpoint-recovery", "--scenario-retry-limit", "1",
                   "--planner-retry-limit", "1"] + extra
    print("CP-X GT run directory:", run_dir, flush=True)
    with (run_dir / "command.txt").open("w") as stream:
        stream.write(" ".join(shlex.quote(token) for token in command) + "\n")
    server = None
    server_log = None
    try:
        if server_command is not None:
            with socket.socket() as sock:
                if sock.connect_ex((direct_args.host, int(direct_args.port))) == 0:
                    raise RuntimeError("CARLA port is already in use; omit --start-carla to use that server")
            server_env = env.copy()
            if shim is not None:
                server_env["LD_PRELOAD"] = str(shim) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
            server_log = open(str(run_dir / "carla.log"), "w")
            server = subprocess.Popen(server_command, cwd=str(run_dir), env=server_env,
                                      stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError("CARLA exited at startup; see " + str(run_dir / "carla.log"))
                with socket.socket() as sock:
                    sock.settimeout(1)
                    if sock.connect_ex((direct_args.host, int(direct_args.port))) == 0:
                        break
                if time.monotonic() > deadline:
                    raise RuntimeError("CARLA startup timed out")
                time.sleep(1)
        evaluation_started = time.perf_counter()
        returncode = subprocess.call(command, cwd=str(mdrive), env=env)
        if args.direct:
            (run_dir / "execution.json").write_text(json.dumps({
                "execution": args.execution, "worker_cpus": args.worker_cpus,
                "record_video": args.record_video, "evaluator_wall_s": time.perf_counter() - evaluation_started,
                "returncode": returncode, "thread_limit": 1}, indent=2))
            report = summarize(run_dir, expected_egos=ego_count)
            print("CP-X closed-loop execution:", report["closed_loop_executed"], flush=True)
            if not report["closed_loop_executed"] and returncode == 0:
                returncode = 2
        return returncode
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    sys.exit(main())
