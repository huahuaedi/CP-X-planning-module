"""Load with MDrive --agent /absolute/path/to/mdrive_adapter/cpx_agent.py."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# Resolve legacy imports locally without installing over MDrive's OpenCDA.
for path in (ROOT, ROOT / ".runtime/deps", ROOT / "opencda/planning_module"):
    sys.path.insert(0, str(path))

import carla
import yaml
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from mdrive_adapter.runtime import collect_gt, make_planner
from opencda.planning_module.MPC.mpc import _OSQP_AVAILABLE


def get_entry_point():
    return "CPXAgent"


class CPXAgent(AutonomousAgent):
    def setup(self, path_to_conf_file, ego_vehicles_num=1):
        if not _OSQP_AVAILABLE:
            raise RuntimeError("CP-X requires OSQP; install mdrive_adapter/requirements.txt")
        with open(path_to_conf_file) as stream:
            self.config = yaml.safe_load(stream) or {}
        if self.config.get("perception") != "gt":
            raise ValueError("This adapter explicitly supports perception: gt only")
        self.track = Track.MAP
        self.agent_name = "cpx_gt"
        self.ego_vehicles_num = int(ego_vehicles_num)
        self._routes = None
        self._contexts = [None] * self.ego_vehicles_num
        self._steps = [None] * self.ego_vehicles_num
        self._done = [False] * self.ego_vehicles_num
        self.last_planned_waypoints_world_by_ego = {}
        self._last_frame = None
        self._last_controls = None
        self._video = None
        self._pool = None
        self.execution = os.environ.get("CPX_EXECUTION", "serial")
        if self.execution not in ("serial", "parallel", "worker-serial"):
            raise ValueError("Unknown CP-X execution mode: " + self.execution)
        self._save_root = Path(os.environ.get("SAVE_PATH", str(ROOT / ".runtime/results"))) / "cpx_gt"
        self._save_root.mkdir(parents=True, exist_ok=True)
        self._log = open(str(self._save_root / "steps.jsonl"), "w", buffering=1)
        self._timing_log = open(str(self._save_root / "frames.jsonl"), "w", buffering=1)
        with open(str(self._save_root / "config.json"), "w") as stream:
            json.dump(self.config, stream, indent=2)

    def sensors(self):
        return [{"type": "sensor.speedometer", "id": "speed", "reading_frequency": 20}]

    def set_global_plan(self, gps_plan, world_plan):
        # Preserve every supplied waypoint, including turns and elevation.
        self._close_planners()
        self._routes = [list(route) for route in world_plan]
        if len(self._routes) != self.ego_vehicles_num or any(len(route) < 2 for route in self._routes):
            raise ValueError("MDrive must supply a route with at least two points for every ego")
        self._global_plan_world_coord = self._routes
        self._global_plan_world_coord_all = self._routes
        self._global_plan = gps_plan
        self._last_frame = None
        self._last_controls = None
        # Connect before warmup: a paused world cannot supply a new client's first snapshot.
        if self.execution != "serial" and self._routes:
            self._start_pool()

    def _start(self, slot, actor, world):
        if self._routes is None or slot >= len(self._routes):
            raise RuntimeError("MDrive did not provide a route for ego %s" % slot)
        output = self._save_root / ("ego_%d" % slot)
        self._contexts[slot], self._steps[slot] = make_planner(
            slot, actor, world, self._routes[slot], self.config, output)

    def _start_pool(self):
        from mdrive_adapter.parallel import PlannerPool, choose_cpus, pack_route
        cpus = choose_cpus(self.ego_vehicles_num, os.environ.get("CPX_WORKER_CPUS", "auto"))
        configs = [{"slot": slot, "route": pack_route(self._routes[slot]),
                    "config": self.config, "output": str(self._save_root / ("ego_%d" % slot)),
                    "host": os.environ["CPX_CARLA_HOST"], "port": int(os.environ["CPX_CARLA_PORT"])}
                   for slot in range(self.ego_vehicles_num)]
        self._pool = PlannerPool(configs, cpus)
        (self._save_root / "workers.json").write_text(json.dumps({
            "execution": self.execution, "physical_core_cpus": cpus,
            "worker_pids": [p.pid for p, _ in self._pool.workers],
            "threads_per_worker": 1}, indent=2))

    def run_step(self, input_data, timestamp):
        frame_started = time.perf_counter()
        world = CarlaDataProvider.get_world()
        if world is None:
            raise RuntimeError("MDrive world is not available")
        frame = world.get_snapshot().frame
        if frame == self._last_frame:
            return self._last_controls
        if self._last_frame is not None and frame < self._last_frame:
            raise RuntimeError("Simulator frame moved backwards without resetting the route")
        snapshots = collect_gt(world)
        gt_ms = (time.perf_counter() - frame_started) * 1000
        live_ids = {item["vehicle_id"] for item in snapshots}
        actors = {slot: CarlaDataProvider.get_hero_actor(hero_id=slot)
                  for slot in range(self.ego_vehicles_num)}
        live_actors = {slot: actor for slot, actor in actors.items()
                       if actor is not None and str(actor.id) in live_ids}
        if os.environ.get("CPX_RECORD_VIDEO") == "1":
            if self._video is None:
                from mdrive_adapter.video import EpisodeVideo
                self._video = EpisodeVideo(world, self._routes, self._save_root / "episode.mp4", carla)
            else:
                self._video.capture(world, live_actors.items())
        active = {slot: actor for slot, actor in live_actors.items() if not self._done[slot]}
        planning_started = time.perf_counter()
        results = {}
        if self.execution != "serial":
            if self._pool is None:
                self._start_pool()
            jobs = {slot: {"frame": frame, "actor_id": actor.id, "snapshots": snapshots}
                    for slot, actor in active.items()}
            results = self._pool.step(jobs, concurrent=self.execution == "parallel")
            for result in results.values():
                result["control"] = carla.VehicleControl(**result.pop("control_values"))
        else:
            for slot, actor in active.items():
                started = time.perf_counter()
                if self._steps[slot] is None:
                    self._start(slot, actor, world)
                if self._contexts[slot].vehicle.id != actor.id:
                    raise RuntimeError("Ego identity changed without resetting the planner")
                self._contexts[slot].snapshots = snapshots
                try:
                    result = next(self._steps[slot])
                except StopIteration:
                    self._done[slot] = True
                    continue
                result["planning_ms"] = (time.perf_counter() - started) * 1000
                results[slot] = result
        planning_ms = (time.perf_counter() - planning_started) * 1000
        if world.get_snapshot().frame != frame:
            raise RuntimeError("World advanced before all ego controls were ready")
        controls = [carla.VehicleControl(brake=1.0) for _ in range(self.ego_vehicles_num)]
        for slot, result in results.items():
            actor = active[slot]
            self._done[slot] = bool(result["done"])
            control = result["control"]
            controls[slot] = control
            trajectory = result["trajectory"]
            self.last_planned_waypoints_world_by_ego[slot] = [state[:2] for state in trajectory]
            record = {key: value for key, value in result.items() if key not in ("vehicle", "control")}
            transform = actor.get_transform()
            record.update({"frame": frame, "timestamp": timestamp, "ego": slot,
                           "actor_id": actor.id, "perception": "gt",
                           "x": transform.location.x, "y": transform.location.y,
                           "throttle": control.throttle, "brake": control.brake, "steer": control.steer})
            self._log.write(json.dumps(record) + "\n")
        self._timing_log.write(json.dumps({"frame": frame, "timestamp": timestamp,
            "execution": self.execution, "active_egos": len(active), "gt_ms": gt_ms,
            "planning_barrier_ms": planning_ms,
            "worker_sum_ms": sum(r["planning_ms"] for r in results.values()),
            "worker_max_ms": max((r["planning_ms"] for r in results.values()), default=0),
            "agent_ms": (time.perf_counter() - frame_started) * 1000}) + "\n")
        self._last_frame = frame
        self._last_controls = controls
        return controls

    def _close_planners(self):
        if getattr(self, "_pool", None) is not None:
            self._pool.close()
            self._pool = None
        for steps in getattr(self, "_steps", []):
            if steps is not None:
                steps.close()
        self._steps = [None] * self.ego_vehicles_num
        self._contexts = [None] * self.ego_vehicles_num
        self._done = [False] * self.ego_vehicles_num

    def destroy(self):
        with ExitStack() as cleanup:
            # Close in reverse order, even if planner or video cleanup fails.
            for name in ("_timing_log", "_log", "_video"):
                resource = getattr(self, name, None)
                if resource is not None:
                    cleanup.callback(resource.close)
                    setattr(self, name, None)
            self._close_planners()
