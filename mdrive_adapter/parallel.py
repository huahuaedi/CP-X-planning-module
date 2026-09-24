"""Spawned per-ego planners. Only the benchmark process advances CARLA."""
import multiprocessing
import os
from pathlib import Path
import time
import traceback


THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def physical_cpus(allowed=None, root=Path("/sys/devices/system/cpu")):
    """One allowed logical CPU per distinct (socket, physical core)."""
    allowed = sorted(os.sched_getaffinity(0) if allowed is None else allowed)
    selected, seen = [], set()
    for cpu in allowed:
        topology = root / ("cpu%d" % cpu) / "topology"
        key = tuple((topology / name).read_text().strip()
                    for name in ("physical_package_id", "core_id"))
        if key not in seen:
            selected.append(cpu)
            seen.add(key)
    return selected


def choose_cpus(count, spec="auto"):
    available = physical_cpus()
    if spec == "auto":
        # Leave the first two physical cores for the evaluator / OS when possible.
        cpus = available[2:2 + count] if len(available) >= count + 2 else available[:count]
    else:
        cpus = [int(v.strip()) for v in spec.split(",")]
    if len(cpus) != count or len(set(cpus)) != count:
        raise ValueError("Need exactly one distinct physical CPU per ego (%d)" % count)
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        raise ValueError("Worker CPUs are outside the current affinity mask")
    if len(physical_cpus(cpus)) != count:
        raise ValueError("Worker CPUs must not be SMT siblings on the same physical core")
    return cpus


def pack_route(route):
    return [(t.location.x, t.location.y, t.location.z, t.rotation.pitch,
             t.rotation.yaw, t.rotation.roll, str(getattr(option, "name", option)))
            for t, option in route]


def planner_worker(connection, cpu, settings):
    """No inherited simulator handles: spawn, bind, then open a fresh read-only client."""
    steps = None
    try:
        os.sched_setaffinity(0, {cpu})
        for key in THREAD_ENV:
            os.environ[key] = "1"
        import carla
        from mdrive_adapter.runtime import make_planner
        client = carla.Client(settings["host"], settings["port"])
        client.set_timeout(30.0)
        world = client.get_world()
        world.get_snapshot()  # Subscribe before the harness performs sensor warmup ticks.
        route = [(carla.Transform(carla.Location(x=p[0], y=p[1], z=p[2]),
                  carla.Rotation(pitch=p[3], yaw=p[4], roll=p[5])), p[6]) for p in settings["route"]]
        context = None
        actor = None
        connection.send({"ok": True, "ready": True})
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                break
            frame = command["frame"]
            started = time.perf_counter()
            deadline = started + 10
            while world.get_snapshot().frame < frame and time.perf_counter() < deadline:
                time.sleep(.001)
            if world.get_snapshot().frame != frame:
                raise RuntimeError("Worker %s simulator frame %s != requested %s" %
                                   (settings["slot"], world.get_snapshot().frame, frame))
            if context is None:
                actor = world.get_actor(command["actor_id"])
                if actor is None:
                    raise RuntimeError("Ego actor is missing at worker initialization")
                context, steps = make_planner(settings["slot"], actor, world, route,
                                              settings["config"], settings["output"])
            if actor.id != command["actor_id"]:
                raise RuntimeError("Ego identity changed without resetting the planner")
            context.snapshots = command["snapshots"]
            try:
                result = next(steps)
            except StopIteration:
                result = {"done": True, "trajectory": [], "status": {},
                          "control": carla.VehicleControl(brake=1.0)}
            result.pop("vehicle", None)
            control = result.pop("control")
            result["control_values"] = {key: getattr(control, key) for key in
                ("throttle", "steer", "brake", "hand_brake", "reverse", "manual_gear_shift", "gear")}
            if world.get_snapshot().frame != frame:
                raise RuntimeError("Simulator advanced before the planning barrier")
            result.update({"frame": frame, "worker_cpu": cpu, "worker_pid": os.getpid(),
                           "planning_ms": (time.perf_counter() - started) * 1000})
            connection.send({"ok": True, "result": result})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if steps is not None:
            steps.close()
        connection.close()


class PlannerPool:
    def __init__(self, configs, cpus, timeout_s=90, target=planner_worker):
        if len(configs) != len(cpus):
            raise ValueError("Each planner needs a CPU assignment")
        self.workers = []
        self.timeout_s = timeout_s
        context = multiprocessing.get_context("spawn")
        for key in THREAD_ENV:
            os.environ[key] = "1"
        try:
            for config, cpu in zip(configs, cpus):
                parent, child = context.Pipe()
                process = context.Process(target=target, args=(child, cpu, config))
                process.start()
                child.close()
                self.workers.append((process, parent))
            deadline = time.monotonic() + timeout_s
            for process, pipe in self.workers:
                if not pipe.poll(max(0, deadline - time.monotonic())):
                    raise RuntimeError("Planner worker did not initialize")
                response = pipe.recv()
                if not response.get("ready"):
                    raise RuntimeError("Planner worker initialization failed: %s" % response)
        except BaseException:
            self.close()
            raise

    def step(self, jobs, concurrent=True):
        results = {}
        deadline = time.monotonic() + self.timeout_s

        def receive(slot, job):
            process, pipe = self.workers[slot]
            if not pipe.poll(max(0, deadline - time.monotonic())):
                raise RuntimeError("CP-X worker %d timed out (pid %s)" % (slot, process.pid))
            try:
                response = pipe.recv()
            except EOFError:
                raise RuntimeError("CP-X worker %d exited unexpectedly" % slot)
            if not response["ok"]:
                raise RuntimeError("CP-X worker %d failed:\n%s" % (slot, response["error"]))
            result = response["result"]
            if result["frame"] != job["frame"]:
                raise RuntimeError("CP-X received a stale planning result")
            results[slot] = result

        # Send ALL requests before awaiting any result in parallel mode.
        for slot, job in jobs.items():
            self.workers[slot][1].send(dict(job, kind="step"))
            if not concurrent:
                receive(slot, job)
        if concurrent:
            for slot, job in jobs.items():
                receive(slot, job)
        return results

    def close(self):
        workers, self.workers = self.workers, []
        for process, pipe in workers:
            try:
                pipe.send({"kind": "close"})
            except (OSError, BrokenPipeError):
                pass
        deadline = time.monotonic() + 30
        for process, pipe in workers:
            process.join(max(0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            pipe.close()
