"""Optional spectator RGB recording; images never enter the planning pipeline."""
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess

import cv2
import numpy as np


class EpisodeVideo:
    def __init__(self, world, routes, output, carla):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = 1280, 960
        points = [t.location for route in routes for t, _ in route]
        xs, ys = [p.x for p in points], [p.y for p in points]
        altitude = max(40.0, (max(xs) - min(xs)) / 1.5 * 1.35,
                       (max(ys) - min(ys)) / 2.0 * 1.35)
        transform = carla.Transform(carla.Location(x=(min(xs) + max(xs)) / 2,
            y=(min(ys) + max(ys)) / 2, z=max(p.z for p in points) + altitude),
            carla.Rotation(pitch=-90))
        self.inverse = np.array(transform.get_inverse_matrix())
        dt = world.get_settings().fixed_delta_seconds
        if not dt or dt <= 0:
            raise RuntimeError("Video recording requires a fixed simulation time step")
        self.fps = 1.0 / dt
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("--record-video requires ffmpeg on PATH")
        self.log = self.output.with_suffix(".ffmpeg.log").open("w")
        self.encoder = subprocess.Popen([executable, "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (self.width, self.height),
            "-r", str(self.fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "2",
            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self.output)], stdin=subprocess.PIPE, stderr=self.log)
        self.camera = None
        self.frames = 0
        self.first_time = None
        self.last_frame = None
        self.queue = queue.Queue()
        try:
            blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
            for key, value in {"image_size_x": str(self.width), "image_size_y": str(self.height),
                               "fov": "90", "sensor_tick": "0", "motion_blur_intensity": "0"}.items():
                blueprint.set_attribute(key, value)
            self.camera = world.spawn_actor(blueprint, transform)
            self.camera.listen(self.queue.put)
        except Exception:
            self.close()
            raise

    def capture(self, world, actors):
        snapshot = world.get_snapshot()
        if self.last_frame == snapshot.frame:
            return
        while True:
            try:
                image = self.queue.get(timeout=10)
            except queue.Empty:
                raise RuntimeError("Spectator camera did not deliver simulation frame %s" % snapshot.frame)
            if image.frame >= snapshot.frame:
                break
        if image.frame != snapshot.frame:
            raise RuntimeError("Spectator camera frame is ahead of the evaluator")
        frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3].copy()
        if self.first_time is None:
            self.first_time = image.timestamp
        elapsed = image.timestamp - self.first_time
        cv2.rectangle(frame, (0, 0), (self.width, 74), (25, 25, 25), -1)
        cv2.putText(frame, "CP-X + GT | %s | spectator RGB" % os.environ.get("CPX_SCENE_NAME", "episode"), (20, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
        cv2.putText(frame, "Simulation: %.2f s | %.0f FPS | cameras are visualization only" % (elapsed, self.fps),
                    (20, 59), cv2.FONT_HERSHEY_SIMPLEX, .65, (210, 210, 210), 1)
        colors = [(70, 210, 255), (100, 255, 100), (255, 160, 90),
                  (220, 110, 255), (60, 140, 255), (255, 240, 80)]
        for slot, actor in actors:
            location = actor.get_location()
            p = self.inverse.dot([location.x, location.y, location.z + 1, 1])
            if p[0] <= 0:
                continue
            u = int(self.width / 2 + self.width / 2 * p[1] / p[0])
            v = int(self.height / 2 - self.width / 2 * p[2] / p[0])
            if 0 <= u < self.width and 74 <= v < self.height:
                velocity = actor.get_velocity()
                label = "E%d %.1f m/s" % (slot, math.hypot(velocity.x, velocity.y))
                color = colors[slot % len(colors)]
                cv2.circle(frame, (u, v), 15, color, 2)
                cv2.putText(frame, label, (u + 18, v - 9), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
                cv2.putText(frame, label, (u + 18, v - 9), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        self.encoder.stdin.write(frame.tobytes())
        self.frames += 1
        self.last_frame = snapshot.frame

    def close(self):
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.destroy()
            except RuntimeError as exc:
                # The harness may already have removed sensors during cleanup.
                # Still finalize the MP4 so the recorded episode is reviewable.
                print("[CP-X video] Camera cleanup: %s" % exc)
            finally:
                self.camera = None
        if self.encoder.stdin and not self.encoder.stdin.closed:
            self.encoder.stdin.close()
            try:
                code = self.encoder.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.encoder.kill()
                self.encoder.wait()
                raise RuntimeError("Video encoder did not finish")
            finally:
                self.log.close()
            self.output.with_suffix(".json").write_text(json.dumps({
                "frames": self.frames, "fps": self.fps, "duration_s": self.frames / self.fps,
                "encoder_exit_code": code, "view": "CARLA spectator RGB", "perception": "gt"}, indent=2))
            if code:
                raise RuntimeError("Video encoding failed; see " + str(self.output.with_suffix(".ffmpeg.log")))
