"""ROS boundary for the copied CP-X planning pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import unquote

from autoware_control_msgs.msg import Control
from autoware_perception_msgs.msg import TrackedObjects
from builtin_interfaces.msg import Time
from cpx_interfaces.msg import CooperativeMessageArray, PlannerInputFrame, TrafficLightObservationArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

from cpx_planning.component_interfaces import RemoteGlobalPlannerProxy, cycle_id_from_timestamp, encode_json, fill_header, load_planner_configuration
from cpx_planning.control_execution_cache import SessionControlExecutionCache
from cpx_planning.current_planner_runtime import CurrentPlannerRuntime
from cpx_planning.prediction_scene_dispatcher import LatestPredictionSceneDispatcher
from cpx_planning.prediction_ros_adapter import (
    ROSPredictionBridge,
    prediction_session_id,
)
from opencda.planning_module.pipeline.reference.reference_gate import (
    REFERENCE_GATE_POLICY_REVISION,
)
from opencda.planning_module.opencda_bridge.planner_assembly import (
    resolve_global_planner_xodr_path,
)
from cpx_planning.ros_input_adapter import ROSInputAdapter
from cpx_planning.ros_output_adapter import ROSOutputAdapter
from cpx_planning.runtime_pause_monitor import RuntimePauseMonitor
from cpx_planning.session_sharding import owns_session
from cpx_planning.synchronized_input_buffer import SynchronizedInputBuffer
from cpx_planning.timing import CycleTimingRecorder, default_timing_summary_path
from cpx_planning.utility.config_loader import deep_merge_dicts


# Enable detailed timing without enabling the much larger planner debug logs.
debug_time = False

PLANNER_DATA_INPUTS = (
    "localization",
    "perception",
    "v2x",
    "cp_obstacles",
    "traffic_lights",
    "final_destination",
)
REQUIRED_PLANNER_INPUTS = PLANNER_DATA_INPUTS + ("session",)
OPTIONAL_PLANNER_INPUTS = ("cooperative",)
INPUT_APPLY_ORDER = PLANNER_DATA_INPUTS + OPTIONAL_PLANNER_INPUTS


@dataclass
class _PlannerSessionContext:
    """All mutable planning state owned by one OpenCDA CAV session."""

    session_id: str
    planner_config: dict
    planner_runtime: object
    planner_bridge: object
    input_adapter: object
    prediction_bridge: object = None
    latest_planner_output: object = None
    latest_control_message: object = None
    latest_adapter_output: object = None
    latest_planning_cycle_time_ms: float = 0.0
    last_frame_timestamp_s: object = None
    waiting_message_printed: bool = False
    last_prediction_scene_timestamp_s: float = -float("inf")
    prediction_map_signature: object = None
    prediction_map_revision: int = 0
    prediction_scene_submitted_ns: object = None
    last_debug_sample_timestamp_s: float = -float("inf")
    profiler_instrumented: bool = False

    def __post_init__(self):
        if self.prediction_scene_submitted_ns is None:
            self.prediction_scene_submitted_ns = {}


def _default_planner_input_log_path():
    """Keep the comparison log in the workspace root for a symlink build."""
    source_path = Path(__file__).resolve()
    for parent in source_path.parents:
        if parent.name == "src":
            return parent.parent / "ros_planner_input_adapter_output.jsonl"
    return Path.cwd() / "ros_planner_input_adapter_output.jsonl"


def _json_safe(value):
    """Convert planner contracts and custom waypoints into JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value.__class__.__name__ == "Waypoint" and callable(getattr(value, "to_dict", None)):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return _json_safe(list_method())
        except Exception:
            pass
    return str(value)


def _strict_json_dumps(value, **kwargs):
    """Serialize a ROS-bound payload without leaking NaN/Inf values.

    Planner diagnostics legitimately use infinity for "no finite distance".
    Those values are valid inside the planner but forbidden by the strict JSON
    transport shared with OpenCDA.  Sanitize the complete payload at the
    boundary so adding a diagnostic field can never suppress control output.
    """

    kwargs["allow_nan"] = False
    return json.dumps(_json_safe(value), **kwargs)


def _time_message(timestamp_s):
    """Use simulation time in both the planner output and the returned TCP control."""
    timestamp_s = max(0.0, float(timestamp_s))
    message = Time()
    message.sec = int(timestamp_s)
    message.nanosec = int(round((timestamp_s - message.sec) * 1000000000.0))
    if message.nanosec >= 1000000000:
        message.sec += 1
        message.nanosec -= 1000000000
    return message


def _input_message_timestamp_s(input_name, message):
    """Read the simulation timestamp used to match one raw callback to a planning cycle."""
    if str(input_name) == "session":
        try:
            return float(json.loads(str(message.data or "{}")).get("timestamp_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0.0
    return float(stamp.sec) + float(stamp.nanosec) / 1000000000.0


def _input_message_session_id(input_name, message):
    """Read the CAV session carried by one internal planner input topic."""
    if str(input_name) == "session":
        try:
            return str(
                json.loads(str(message.data or "{}")).get("session_id", "")
                or ""
            )
        except (TypeError, ValueError):
            return ""
    header = getattr(message, "header", None)
    frame_id = str(getattr(header, "frame_id", "") or "")
    marker = "|cpx_session="
    if marker not in frame_id:
        return ""
    return unquote(frame_id.split(marker, 1)[1])


class CPXPlannerNode(Node):
    """Receive raw ROS data and run the same CP-X bridge used by OpenCDA."""

    def __init__(
        self,
        local_map_planner=None,
        local_bus=None,
        map_owner=None,
    ):
        super().__init__("planner_node")
        self._compact_schema_revision = 2
        package_root = Path(__file__).resolve().parent
        self.package_root = package_root
        self.local_bus = local_bus
        self._map_owner = map_owner
        self.declare_parameter("planner_input_log_path", str(_default_planner_input_log_path()))
        self.declare_parameter("control_topic", "/control/command/control_cmd")
        self.declare_parameter("debug", False)
        self.declare_parameter("debug_time", debug_time)
        self.declare_parameter("debug_sample_hz", 2.0)
        self.declare_parameter("input_frame_buffer_size", 32)
        self.declare_parameter("timing_summary_path", str(default_timing_summary_path()))
        self.declare_parameter("prediction_enabled", False)
        self.declare_parameter("prediction_scene_topic", "/prediction/scene")
        self.declare_parameter("prediction_map_topic", "/prediction/map")
        self.declare_parameter("prediction_frame_topic", "/prediction/frame")
        self.declare_parameter("prediction_max_age_s", 0.5)
        self.declare_parameter("prediction_scene_hz", 10.0)
        self.declare_parameter("execution_control_max_plan_age_s", 0.5)
        self.declare_parameter("planner_shard_count", 1)
        self.declare_parameter("planner_shard_index", 0)
        self.debug = bool(self.get_parameter("debug").value)
        self.debug_time = bool(self.get_parameter("debug_time").value)
        self._debug_sample_interval_s = 1.0 / max(
            0.1, float(self.get_parameter("debug_sample_hz").value)
        )
        self._last_debug_sample_timestamp_s = -float("inf")
        self.prediction_enabled = bool(self.get_parameter("prediction_enabled").value)
        self._planner_shard_count = max(
            1, int(self.get_parameter("planner_shard_count").value)
        )
        self._planner_shard_index = int(
            self.get_parameter("planner_shard_index").value
        )
        if not 0 <= self._planner_shard_index < self._planner_shard_count:
            raise ValueError(
                "planner_shard_index must be in [0, planner_shard_count)"
            )
        self._prediction_scene_interval_s = 1.0 / max(
            0.1, float(self.get_parameter("prediction_scene_hz").value)
        )
        self._last_prediction_scene_timestamp_s = -float("inf")
        self._prediction_scene_time_lock = threading.Lock()
        self._prediction_scene_submitted_ns = {}
        self._input_callback_group = ReentrantCallbackGroup()
        self._prediction_callback_group = ReentrantCallbackGroup()
        self._timing_callback_group = ReentrantCallbackGroup()
        self.timing_recorder = CycleTimingRecorder(enabled=self.debug_time, summary_path=str(self.get_parameter("timing_summary_path").value))
        self._runtime_pause_monitor = RuntimePauseMonitor(enabled=self.debug_time)
        if local_bus is not None:
            local_bus.timing_recorder = self.timing_recorder
        planner_config = load_planner_configuration(package_root)
        planner_config["debug"] = bool(self.debug)
        planner_config["record_debug"] = bool(self.debug)
        planner_config["record_evaluation_metrics"] = bool(self.debug)
        if self.prediction_enabled:
            # ROS owns prediction transport in this mode. Never start the
            # legacy HTTP/in-process MTR bridge at the same time.
            planner_config["use_mtr_prediction_bridge"] = False
        self._base_planner_config = dict(planner_config)
        self._planner_config = dict(planner_config)
        self._owns_global_planner_proxy = local_map_planner is None
        self.global_planner = local_map_planner or RemoteGlobalPlannerProxy(
            self, local_bus=local_bus
        )
        self._map_planner = self.global_planner
        self.planner_runtime = CurrentPlannerRuntime(
            config=planner_config,
            map_planner=self.global_planner,
        )
        self.planner_bridge = self.planner_runtime.bridge
        self.prediction_bridge = None
        self._prediction_bridge_kwargs = None
        self.prediction_scene_publisher = None
        self.prediction_map_publisher = None
        self._prediction_map_signature = None
        self._prediction_map_revision = 0
        self._prediction_scene_dispatcher = None
        self.prediction_frame_subscription = None
        if self.prediction_enabled:
            try:
                from geometry_msgs.msg import Point
                from mtr_prediction_msgs.msg import (
                    AgentState, MapPolyline, PredictionFrame, PredictionMap,
                    PredictionScene,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "prediction_enabled=true requires mtr_prediction_msgs "
                    "to be built and sourced"
                ) from exc
            self.prediction_bridge = ROSPredictionBridge(
                max_stale_s=float(self.get_parameter("prediction_max_age_s").value),
                scene_type=PredictionScene,
                map_type=PredictionMap,
                agent_type=AgentState,
                map_polyline_type=MapPolyline,
                point_type=Point,
            )
            self._prediction_bridge_kwargs = {
                "max_stale_s": float(
                    self.get_parameter("prediction_max_age_s").value
                ),
                "scene_type": PredictionScene,
                "map_type": PredictionMap,
                "agent_type": AgentState,
                "map_polyline_type": MapPolyline,
                "point_type": Point,
            }
            self.planner_bridge.prediction_bridge = self.prediction_bridge
            prediction_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=8,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            prediction_frame_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=8,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self.prediction_scene_publisher = self.create_publisher(
                PredictionScene,
                str(self.get_parameter("prediction_scene_topic").value),
                prediction_qos,
            )
            prediction_map_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.prediction_map_publisher = self.create_publisher(
                PredictionMap,
                str(self.get_parameter("prediction_map_topic").value),
                prediction_map_qos,
            )
            self._prediction_scene_dispatcher = LatestPredictionSceneDispatcher(
                self.prediction_scene_publisher,
                on_error=lambda exc: self.get_logger().error(
                    "Could not publish MTR PredictionScene: {}".format(exc)
                ),
            )
            self.prediction_frame_subscription = self.create_subscription(
                PredictionFrame,
                str(self.get_parameter("prediction_frame_topic").value),
                self._receive_prediction_frame,
                prediction_frame_qos,
                callback_group=self._prediction_callback_group,
            )
        self.input_adapter = ROSInputAdapter(bridge=self.planner_bridge)
        self.latest_planner_output = None
        self.latest_control_message = None
        self.latest_adapter_output = None
        self.latest_planning_cycle_time_ms = 0.0
        self._last_frame_timestamp_s = None
        self._active_session_id = None
        self._active_run_id = ""
        self._session_contexts = {}
        self._bootstrap_runtime = self.planner_runtime
        self._buffer_session_id = None
        self._frame_session_payload = {}
        self._waiting_message_printed = False
        self._input_worker_stop = threading.Event()
        self._input_frame_buffer = SynchronizedInputBuffer(
            required_inputs=REQUIRED_PLANNER_INPUTS,
            optional_inputs=OPTIONAL_PLANNER_INPUTS,
            max_frames=int(self.get_parameter("input_frame_buffer_size").value),
        )
        self._stage_profile_dispatch_baseline = 0
        self._planning_lock = threading.RLock()
        self._execution_control_cache = SessionControlExecutionCache(
            max_plan_age_s=float(
                self.get_parameter("execution_control_max_plan_age_s").value
            )
        )
        self._latest_execution_ego_states = {}
        self._execution_ego_state_lock = threading.RLock()
        # Input assembly, prediction delivery and optional timing diagnostics
        # are independent and thread-safe. Keeping all of them in the node's
        # default mutually-exclusive group serialized hundreds of callbacks
        # per second in multi-CAV runs and allowed observability traffic to
        # delay control reuse. Separate reentrant groups let the executor use
        # its worker threads while the planning algorithm stays on its own
        # dedicated worker.

        self.ros_output_adapter = ROSOutputAdapter()
        self.control_publisher = self.create_publisher(Control, str(self.get_parameter("control_topic").value), 10)
        compact_control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=8,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.tcp_control_output_publisher = self.create_publisher(
            String,
            "/cpx/planner_control_output",
            compact_control_qos,
        )
        self.debug_output_publisher = self.create_publisher(String, "/cpx/debug_output", 10) if self.debug else None
        input_frame_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.RELIABLE)
        source_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.input_frame_publisher = self.create_publisher(PlannerInputFrame, "/cpx/planning/input_frame", input_frame_qos)
        # Callbacks assemble timestamp-matched frames while planning runs on a
        # worker, so one slow solve cannot destroy cross-topic frame alignment.
        self.localization_subscription = self.create_subscription(Odometry, "/cpx/localization", lambda message: self._receive("localization", message), source_qos, callback_group=self._input_callback_group)
        self.perception_subscription = self.create_subscription(TrackedObjects, "/cpx/perception", lambda message: self._receive("perception", message), source_qos, callback_group=self._input_callback_group)
        self.v2x_subscription = self.create_subscription(TrackedObjects, "/cpx/v2x", lambda message: self._receive("v2x", message), source_qos, callback_group=self._input_callback_group)
        self.cp_obstacles_subscription = self.create_subscription(TrackedObjects, "/cpx/cp_obstacles", lambda message: self._receive("cp_obstacles", message), source_qos, callback_group=self._input_callback_group)
        self.traffic_light_subscription = self.create_subscription(TrafficLightObservationArray, "/cpx/traffic_light", lambda message: self._receive("traffic_lights", message), source_qos, callback_group=self._input_callback_group)
        self.cooperative_subscription = self.create_subscription(CooperativeMessageArray, "/cpx/cooperative_messages", lambda message: self._receive("cooperative", message), source_qos, callback_group=self._input_callback_group)
        self.session_subscription = self.create_subscription(String, "/cpx/planning/session", lambda message: self._receive("session", message), source_qos, callback_group=self._input_callback_group)
        self.destination_subscription = self.create_subscription(PoseStamped, "/cpx/final_destination", lambda message: self._receive("final_destination", message), source_qos, callback_group=self._input_callback_group)
        timing_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=32,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.timing_subscription = self.create_subscription(
            String,
            "/cpx/timing/events",
            self._receive_timing_event,
            timing_qos,
            callback_group=self._timing_callback_group,
        ) if self.debug_time else None

        self.planner_input_log_path = Path(str(self.get_parameter("planner_input_log_path").value)).expanduser().resolve()
        if self.debug:
            self.planner_input_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.planner_input_log_path.write_text("", encoding="utf-8")
        self._input_worker = threading.Thread(
            target=self._planning_worker_loop,
            name="cpx-planning-worker",
            daemon=True,
        )
        self._input_worker.start()
        self.get_logger().info(
            "CP-X planner coordinator buffers exact OpenCDA cycles at 20 Hz "
            "and executes only the newest complete frame for shard {}/{}; compact control "
            "schema revision {}; reference-gate policy revision {} ({}).".format(
                self._planner_shard_index,
                self._planner_shard_count,
                self._compact_schema_revision,
                REFERENCE_GATE_POLICY_REVISION,
                Path(__import__(
                    "opencda.planning_module.pipeline.reference.reference_gate",
                    fromlist=["__file__"],
                ).__file__).resolve(),
            )
        )

    def _receive(self, input_name, message):
        """Add one raw input to its CARLA cycle without waiting for planning."""
        callback_started_wall_time_ns = time.time_ns()
        incoming_session_id = _input_message_session_id(input_name, message)
        if str(input_name) == "session":
            session_payload = self._session_payload_from_messages({
                "session": message,
            })
            incoming_session_id = str(
                session_payload.get("session_id", incoming_session_id) or ""
            ).strip()
            if incoming_session_id:
                self._buffer_session_id = incoming_session_id
        elif not incoming_session_id:
            # Compatibility with an older single-session bridge. New bridge
            # messages always carry session ownership in ``header.frame_id``.
            incoming_session_id = str(self._buffer_session_id or "")
        if not owns_session(
            incoming_session_id,
            self._planner_shard_count,
            self._planner_shard_index,
        ):
            return
        callback_cycle_id = cycle_id_from_timestamp(_input_message_timestamp_s(input_name, message))
        self.timing_recorder.record_event_time(callback_cycle_id, "input.{}.callback_started".format(input_name), callback_started_wall_time_ns)
        buffer_started = time.perf_counter()
        self._input_frame_buffer.add(
            input_name=input_name,
            cycle_id=callback_cycle_id,
            message=message,
            session_id=incoming_session_id,
        )
        if str(input_name) == "localization" and incoming_session_id:
            pose = message.pose.pose
            twist = message.twist.twist
            orientation = pose.orientation
            ego_state = {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "heading_rad": math.atan2(
                    2.0
                    * (
                        float(orientation.w) * float(orientation.z)
                        + float(orientation.x) * float(orientation.y)
                    ),
                    1.0
                    - 2.0
                    * (
                        float(orientation.y) ** 2
                        + float(orientation.z) ** 2
                    ),
                ),
                "speed_mps": math.hypot(
                    float(twist.linear.x), float(twist.linear.y)
                ),
            }
            with self._execution_ego_state_lock:
                self._latest_execution_ego_states[incoming_session_id] = ego_state
        self.timing_recorder.record_duration(
            callback_cycle_id,
            "callback_message_waiting.{}.frame_buffer".format(input_name),
            (time.perf_counter() - buffer_started) * 1000.0,
        )
        # Return a bounded-age execution command at the source 20 Hz cadence.
        # This callback never waits for the full planner lock, so one slow
        # multi-CAV solve cannot make every vehicle emergency-brake merely
        # because no new ROS control packet arrived. The cache itself rejects
        # plans older than execution_control_max_plan_age_s.
        if str(input_name) == "session" and incoming_session_id:
            self._publish_reused_execution_control(session_payload)

    def _planning_worker_loop(self):
        """Run planning outside ROS callbacks using the newest complete frame."""
        while not self._input_worker_stop.is_set():
            frame = self._input_frame_buffer.wait_for_latest_complete(
                self._input_worker_stop,
                timeout_s=0.25,
            )
            if frame is None:
                continue
            self.timing_recorder.record_duration(
                frame.cycle_id,
                "input_sync.complete_frame_assembly",
                frame.assembly_wait_ms,
            )
            self.timing_recorder.set_metadata(
                "input_frame_sync",
                self._input_frame_buffer.stats(),
            )
            try:
                with self._planning_lock:
                    session_activation_started = time.perf_counter()
                    self._activate_frame_session(
                        frame.messages,
                        session_id=frame.session_id,
                    )
                    self.timing_recorder.record_duration(
                        frame.cycle_id,
                        "planner.session_context_activation",
                        (time.perf_counter() - session_activation_started)
                        * 1000.0,
                    )
                    input_apply_started = time.perf_counter()
                    for input_name in INPUT_APPLY_ORDER:
                        message = frame.messages.get(input_name)
                        if message is not None:
                            self._apply_message(input_name, message)
                    self.timing_recorder.record_duration(
                        frame.cycle_id,
                        "planner.session_input_apply",
                        (time.perf_counter() - input_apply_started) * 1000.0,
                    )
                    self.run_planning_cycle()
            except Exception as exc:
                self.get_logger().error(
                    "Could not process synchronized input frame {}: {}".format(
                        frame.cycle_id,
                        exc,
                    )
                )

    @staticmethod
    def _session_payload_from_messages(messages):
        """Return lifecycle metadata carried independently of planner data."""
        message = dict(messages or {}).get("session")
        if message is None:
            return {}
        try:
            payload = json.loads(str(message.data or "{}"))
        except (TypeError, ValueError):
            return {}
        return dict(payload or {}) if isinstance(payload, Mapping) else {}

    @staticmethod
    def _session_actor_id(session_id):
        """Return the CARLA actor suffix embedded by OpenCDA."""
        try:
            return int(str(session_id).rsplit(":", 1)[-1])
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _session_run_id(session_id):
        """Return the scenario-run UUID shared by all CAV sessions."""

        value = str(session_id or "").strip()
        if not value:
            return ""
        prefix, separator, _actor_id = value.rpartition(":")
        return prefix if separator and prefix else value

    def _new_prediction_bridge(self, session_id=""):
        if self._prediction_bridge_kwargs is None:
            return None
        kwargs = dict(self._prediction_bridge_kwargs)
        kwargs["expected_session_id"] = str(session_id or "")
        return ROSPredictionBridge(**kwargs)

    def _save_active_session_context(self):
        context = self._session_contexts.get(self._active_session_id)
        if context is None:
            return
        context.planner_config = dict(self._planner_config)
        context.planner_runtime = self.planner_runtime
        context.planner_bridge = self.planner_bridge
        context.input_adapter = self.input_adapter
        context.prediction_bridge = self.prediction_bridge
        context.latest_planner_output = self.latest_planner_output
        context.latest_control_message = self.latest_control_message
        context.latest_adapter_output = self.latest_adapter_output
        context.latest_planning_cycle_time_ms = float(
            self.latest_planning_cycle_time_ms
        )
        context.last_frame_timestamp_s = self._last_frame_timestamp_s
        context.waiting_message_printed = bool(self._waiting_message_printed)
        context.last_prediction_scene_timestamp_s = float(
            self._last_prediction_scene_timestamp_s
        )
        context.prediction_map_signature = self._prediction_map_signature
        context.prediction_map_revision = int(self._prediction_map_revision)
        context.prediction_scene_submitted_ns = dict(
            self._prediction_scene_submitted_ns
        )
        context.last_debug_sample_timestamp_s = float(
            self._last_debug_sample_timestamp_s
        )

    def _load_session_context(self, context):
        self._planner_config = dict(context.planner_config)
        self.planner_runtime = context.planner_runtime
        self.planner_bridge = context.planner_bridge
        self.input_adapter = context.input_adapter
        self.prediction_bridge = context.prediction_bridge
        self.latest_planner_output = context.latest_planner_output
        self.latest_control_message = context.latest_control_message
        self.latest_adapter_output = context.latest_adapter_output
        self.latest_planning_cycle_time_ms = float(
            context.latest_planning_cycle_time_ms
        )
        self._last_frame_timestamp_s = context.last_frame_timestamp_s
        self._waiting_message_printed = bool(context.waiting_message_printed)
        self._last_prediction_scene_timestamp_s = float(
            context.last_prediction_scene_timestamp_s
        )
        self._prediction_map_signature = context.prediction_map_signature
        self._prediction_map_revision = int(context.prediction_map_revision)
        self._prediction_scene_submitted_ns = dict(
            context.prediction_scene_submitted_ns
        )
        self._last_debug_sample_timestamp_s = float(
            context.last_debug_sample_timestamp_s
        )
        self._active_session_id = str(context.session_id)

    def _create_session_context(self, session_id, planner_config):
        fork_route_context = getattr(
            self._map_planner, "fork_route_context", None
        )
        session_map_planner = (
            fork_route_context()
            if callable(fork_route_context)
            else self._map_planner
        )
        runtime = CurrentPlannerRuntime(
            config=planner_config,
            map_planner=session_map_planner,
            actor_id=self._session_actor_id(session_id),
        )
        prediction_bridge = self._new_prediction_bridge(session_id)
        if prediction_bridge is not None:
            runtime.bridge.prediction_bridge = prediction_bridge
        return _PlannerSessionContext(
            session_id=str(session_id),
            planner_config=dict(planner_config),
            planner_runtime=runtime,
            planner_bridge=runtime.bridge,
            input_adapter=ROSInputAdapter(bridge=runtime.bridge),
            prediction_bridge=prediction_bridge,
        )

    def _activate_frame_session(self, messages, session_id=None):
        """Give each OpenCDA vehicle execution a fresh planner lifecycle.

        ROS nodes intentionally remain alive between CARLA runs. A vehicle
        actor id changes when the scenario is spawned again, so it provides a
        deterministic session boundary without position or timing thresholds.
        """
        session_payload = self._session_payload_from_messages(messages)
        self._frame_session_payload = session_payload
        session_id = str(
            session_id or session_payload.get("session_id", "") or ""
        ).strip()
        if not session_id:
            return
        incoming_run_id = self._session_run_id(session_id)
        run_changed = bool(
            self._active_run_id
            and incoming_run_id
            and incoming_run_id != self._active_run_id
        )
        if session_id == self._active_session_id and not run_changed:
            self._frame_session_payload = session_payload
            return
        previous_session_id = self._active_session_id
        self._save_active_session_context()
        if run_changed:
            for stale_context in self._session_contexts.values():
                stale_context.planner_runtime.close()
            self._session_contexts.clear()
            self._execution_control_cache.clear()
            with self._execution_ego_state_lock:
                self._latest_execution_ego_states.clear()
            self._active_session_id = None
            self._input_frame_buffer.reset_for_new_session()
            if self._prediction_scene_dispatcher is not None:
                self._prediction_scene_dispatcher.discard_pending()
            self.get_logger().info(
                "Retired planner sessions from OpenCDA run {} before {}.".format(
                    self._active_run_id,
                    incoming_run_id,
                )
            )
        if incoming_run_id:
            self._active_run_id = incoming_run_id

        # A session id is immutable for one OpenCDA actor in one run. Once
        # its isolated runtime exists, switching back to it only requires
        # restoring that context. Re-running configuration merge, map-path
        # resolution and activate_map() on every round-robin turn adds
        # lifecycle work to the control path without changing any state.
        # New sessions and new runs still take the complete lifecycle path
        # below, including map replacement and cache invalidation.
        existing_context = self._session_contexts.get(session_id)
        if existing_context is not None and not run_changed:
            self._load_session_context(existing_context)
            self._frame_session_payload = session_payload
            return

        scenario_planner_config = session_payload.get("planner_config", {})
        if not isinstance(scenario_planner_config, Mapping):
            scenario_planner_config = {}
        planner_config = deep_merge_dicts(
            dict(self._base_planner_config),
            dict(scenario_planner_config),
        )
        if self.prediction_enabled:
            planner_config["use_mtr_prediction_bridge"] = False

        desired_xodr_path = resolve_global_planner_xodr_path(
            planner_config,
            self._map_planner,
        )
        map_activator = getattr(self._map_owner, "activate_map", None)
        if not callable(map_activator):
            map_activator = getattr(self._map_planner, "activate_map", None)
        if not callable(map_activator):
            raise RuntimeError(
                "ROS global planner does not expose the required "
                "session-level activate_map() lifecycle operation."
            )
        map_changed = bool(map_activator(str(desired_xodr_path)))
        if run_changed and not map_changed:
            run_resetter = getattr(self._map_owner, "reset_for_new_run", None)
            if not callable(run_resetter):
                raise RuntimeError(
                    "ROS global planner does not expose the required "
                    "reset_for_new_run() lifecycle operation."
                )
            run_resetter()
        if run_changed and self.local_bus is not None:
            self.local_bus.reset_map_query_cache()
        if map_changed and self._session_contexts:
            # Route-context views share the process-wide AD-map core.  A map
            # replacement invalidates every waypoint/lane id held by older
            # sessions, so discard them before constructing the new context.
            for stale_context in self._session_contexts.values():
                stale_context.planner_runtime.close()
            self._session_contexts.clear()
            self._active_session_id = None
        context = self._session_contexts.get(session_id)
        context_event = ""
        if context is None:
            context = self._create_session_context(
                session_id,
                planner_config,
            )
            self._session_contexts[session_id] = context
            context_event = "created"
        elif dict(context.planner_config) != dict(planner_config):
            context.planner_runtime.close()
            context = self._create_session_context(
                session_id,
                planner_config,
            )
            self._session_contexts[session_id] = context
            context_event = "rebuilt"
        self._load_session_context(context)
        self._frame_session_payload = session_payload

        if self._bootstrap_runtime is not None:
            self._bootstrap_runtime.close()
            self._bootstrap_runtime = None
        stage_profiler = getattr(self, "stage_profiler", None)
        if stage_profiler is not None and not context.profiler_instrumented:
            stage_profiler.instrument_route_manager(
                self.planner_bridge.route_manager
            )
            stage_profiler.instrument_reference_map(
                getattr(self.planner_bridge, "reference_map", None)
            )
            stage_profiler.instrument_bridge(self.planner_bridge)
            stage_profiler.instrument_mpc(self.planner_bridge.mpc)
            context.profiler_instrumented = True
        if context_event or map_changed:
            self.get_logger().info(
                "Activated isolated planner session {} -> {} ({}) "
                "with {} scenario overrides; active contexts={}; AD-map={} ({}).".format(
                    previous_session_id or "initial",
                    session_id,
                    context_event or "map_changed",
                    len(scenario_planner_config),
                    len(self._session_contexts),
                    desired_xodr_path,
                    "reloaded" if map_changed else "reused",
                )
            )

    def _receive_timing_event(self, message):
        """Store transport timing published by the TCP/ROS boundary nodes."""
        try:
            event = json.loads(str(message.data or "{}"))
        except (TypeError, ValueError):
            return
        if isinstance(event, dict):
            event_session_id = str(event.get("session_id", "") or "")
            if event_session_id and not owns_session(
                event_session_id,
                self._planner_shard_count,
                self._planner_shard_index,
            ):
                return
            self.timing_recorder.record_external_event(event)

    def _receive_prediction_frame(self, message):
        """Cache a 5 Hz prediction frame outside the 20 Hz input barrier."""
        frame_session_id = prediction_session_id(message)
        # PredictionFrame is broadcast on one shared ROS topic, so every
        # process-level planner shard receives every session's result.  Drop
        # frames owned by another shard before looking up local contexts.  A
        # warning here used to be emitted for every valid foreign frame; with
        # N shards that produced O(N) log traffic and consumed enough CPU to
        # perturb the control schedule it was intended to protect.
        if frame_session_id and not owns_session(
            frame_session_id,
            self._planner_shard_count,
            self._planner_shard_index,
        ):
            return
        with self._planning_lock:
            if frame_session_id:
                context = self._session_contexts.get(frame_session_id)
            elif len(self._session_contexts) == 1:
                context = next(iter(self._session_contexts.values()))
            else:
                context = None
            prediction_bridge = (
                context.prediction_bridge if context is not None else None
            )
        if prediction_bridge is None:
            self.get_logger().warning(
                "Ignored PredictionFrame without a known session owner."
            )
            return
        if not prediction_bridge.update_frame(message):
            self.get_logger().warning(
                "Ignored PredictionFrame for session {}: {}".format(
                    frame_session_id or "legacy-single-session",
                    prediction_bridge.last_error or "invalid_frame",
                )
            )
            return
        frame_timestamp_s = _input_message_timestamp_s(
            "prediction_frame", message
        )
        frame_key = cycle_id_from_timestamp(frame_timestamp_s)
        with self._prediction_scene_time_lock:
            if context.session_id == self._active_session_id:
                submitted_times = self._prediction_scene_submitted_ns
            else:
                submitted_times = context.prediction_scene_submitted_ns
            submitted_ns = submitted_times.pop(frame_key, None)
        if submitted_ns is not None:
            prediction_bridge.record_latency_ms(
                (time.time_ns() - submitted_ns) / 1.0e6
            )

    def _publish_prediction_scene(self, timestamp_s, runtime_inputs):
        """Publish current observations/history input for the MTR ROS node."""
        if self.prediction_scene_publisher is None:
            return False
        timestamp_s = float(timestamp_s)
        if timestamp_s < self._last_prediction_scene_timestamp_s - 1.0e-6:
            self._last_prediction_scene_timestamp_s = -float("inf")
        if (
            timestamp_s - self._last_prediction_scene_timestamp_s
            < self._prediction_scene_interval_s - 1.0e-6
        ):
            return False
        local_map = getattr(self.planner_bridge, "_local_map_snapshot", None)
        self._publish_prediction_map(
            timestamp_s=timestamp_s,
            local_map=local_map,
        )

        scene = self.prediction_bridge.build_scene(
            stamp=_time_message(timestamp_s),
            ego_pose=dict(runtime_inputs["ego_pose"]),
            ego_speed_mps=float(runtime_inputs["ego_speed_mps"]),
            object_snapshots=list(
                runtime_inputs.get("local_object_snapshots", ()) or ()
            ),
            cp_obstacles=list(
                dict(runtime_inputs.get("cp_payload", {}) or {}).get(
                    "obstacles", ()
                ) or ()
            ),
            # Static map context is latched independently on /prediction/map.
            # Keep this field empty to avoid serializing the same large map at
            # every 10 Hz scene update. MTR still accepts inline maps from
            # older publishers for backward compatibility.
            map_polylines=(),
            ego_track_id="ego:{}".format(
                int(getattr(
                    self.planner_runtime.vehicle_manager.vehicle, "id", -1
                ))
            ),
        )
        # DDS reliability/back-pressure is isolated from the 20 Hz control
        # worker. If MTR is still inferring, only the newest scene is kept.
        scene_key = cycle_id_from_timestamp(timestamp_s)
        with self._prediction_scene_time_lock:
            self._prediction_scene_submitted_ns[scene_key] = time.time_ns()
            while len(self._prediction_scene_submitted_ns) > 64:
                oldest_key = next(iter(self._prediction_scene_submitted_ns))
                self._prediction_scene_submitted_ns.pop(oldest_key, None)
        self._prediction_scene_dispatcher.submit(scene)
        self._last_prediction_scene_timestamp_s = timestamp_s
        return True

    @staticmethod
    def _local_map_signature(local_map):
        """Return a stable signature for the geometry sent to Prediction."""
        if local_map is None or not bool(getattr(local_map, "valid", False)):
            return None
        corridor_rows = []
        for corridor in tuple(getattr(local_map, "corridors", ()) or ()):
            geometry_rows = []
            for geometry in tuple(
                getattr(corridor, "lane_geometries", ()) or ()
            ):
                points = tuple(getattr(geometry, "centerline", ()) or ())
                if not points:
                    continue
                first = points[0]
                last = points[-1]
                geometry_rows.append((
                    int(getattr(geometry, "lane_id", 0) or 0),
                    len(points),
                    round(float(getattr(first, "x_m", 0.0)), 2),
                    round(float(getattr(first, "y_m", 0.0)), 2),
                    round(float(getattr(last, "x_m", 0.0)), 2),
                    round(float(getattr(last, "y_m", 0.0)), 2),
                    round(float(getattr(first, "lane_width_m", 0.0)), 2),
                    round(float(getattr(last, "lane_width_m", 0.0)), 2),
                ))
            corridor_rows.append((
                int(getattr(corridor, "offset", 0) or 0),
                tuple(geometry_rows),
            ))
        return (
            str(getattr(local_map, "route_revision", "") or ""),
            tuple(corridor_rows),
        )

    def _publish_prediction_map(self, *, timestamp_s, local_map):
        """Latch map context only when the authoritative geometry changes."""
        if self.prediction_map_publisher is None:
            return False
        signature = self._local_map_signature(local_map)
        if signature is None or signature == self._prediction_map_signature:
            return False
        from opencda.planning_module.opencda_bridge.prediction_map_context import (
            mtr_map_polylines,
        )

        map_polylines = mtr_map_polylines(local_map)
        if not map_polylines:
            return False
        revision = self._prediction_map_revision + 1
        message = self.prediction_bridge.build_map(
            stamp=_time_message(timestamp_s),
            revision=revision,
            map_polylines=map_polylines,
        )
        self.prediction_map_publisher.publish(message)
        self._prediction_map_revision = revision
        self._prediction_map_signature = signature
        return True

    def _apply_message(self, input_name, message):
        """Call the matching ROS adapter update method without changing the data."""
        methods = {
            "localization": self.input_adapter.update_localization,
            "perception": self.input_adapter.update_perception,
            "v2x": self.input_adapter.update_v2x,
            "cp_obstacles": self.input_adapter.update_cp_obstacles,
            "traffic_lights": self.input_adapter.update_traffic_lights,
            "cooperative": self.input_adapter.update_cooperative_messages,
            "final_destination": self.input_adapter.update_final_destination,
        }
        methods[input_name](message)

    def run_planning_cycle(self):
        """Run one current-source planner cycle for each synchronized ROS frame."""
        with self._planning_lock:
            if self.input_adapter is None or not self.input_adapter.ready():
                if self.debug and not self._waiting_message_printed:
                    self.get_logger().info("Waiting for localization, perception, CP/V2X, traffic-light, destination, and session data.")
                    self._waiting_message_printed = True
                return
            timestamp_s = float(self.input_adapter.latest_timestamp_s())
            if timestamp_s == self._last_frame_timestamp_s:
                return
            planning_started_monotonic = time.perf_counter()
            cycle_id = cycle_id_from_timestamp(timestamp_s)
            self._runtime_pause_monitor.set_cycle(cycle_id)
            self.timing_recorder.start_cycle(cycle_id, timestamp_s)
            self.timing_recorder.record_event_time(cycle_id, "planner.cycle_started", time.time_ns())
            source_send_wall_time_ns = int(self._frame_session_payload.get("cycle_send_started_wall_time_ns", 0) or 0)
            if source_send_wall_time_ns > 0:
                self.timing_recorder.record_duration(cycle_id, "transport.opencda_send_to_ros_inputs_ready", (time.time_ns() - source_send_wall_time_ns) / 1000000.0)
            try:
                stage_started = time.perf_counter()
                runtime_inputs = self.input_adapter.runtime_inputs()
                self.timing_recorder.record_duration(cycle_id, "input.runtime_inputs", (time.perf_counter() - stage_started) * 1000.0)
                stage_started = time.perf_counter()
                self._publish_prediction_scene(timestamp_s, runtime_inputs)
                self.timing_recorder.record_duration(cycle_id, "prediction.publish_scene", (time.perf_counter() - stage_started) * 1000.0)
                stage_started = time.perf_counter()
                self._publish_input_frame(cycle_id, timestamp_s, runtime_inputs)
                self.timing_recorder.record_duration(cycle_id, "input.publish_frame", (time.perf_counter() - stage_started) * 1000.0)
                stage_started = time.perf_counter()
                planner_output = self.planner_runtime.step(
                    sim_time_s=timestamp_s,
                    ego_pose=dict(runtime_inputs["ego_pose"]),
                    ego_speed_mps=float(runtime_inputs["ego_speed_mps"]),
                    object_snapshots=runtime_inputs["local_object_snapshots"],
                    cp_payload=dict(runtime_inputs["cp_payload"] or {}),
                )
                self.timing_recorder.record_duration(cycle_id, "planner.full_pipeline", (time.perf_counter() - stage_started) * 1000.0)
                stage_snapshot = getattr(
                    self.planner_bridge, "_stage_timing_snapshot", None
                )
                if callable(stage_snapshot):
                    for stage_name, duration_ms in stage_snapshot().items():
                        self.timing_recorder.record_duration(
                            cycle_id,
                            "planner_internal.{}".format(stage_name),
                            duration_ms,
                        )
            except Exception as exc:
                self.timing_recorder.record_value(cycle_id, "failure_reason", str(exc))
                self.timing_recorder.record_duration(cycle_id, "planner.cycle_to_failure", (time.perf_counter() - planning_started_monotonic) * 1000.0)
                self.timing_recorder.finish_cycle(cycle_id, {"success": False})
                self.get_logger().error("Could not run the current CP-X planning cycle: {}".format(exc))
                self._runtime_pause_monitor.clear_cycle()
                return
            self._last_frame_timestamp_s = timestamp_s
            self._waiting_message_printed = False
            self.latest_adapter_output = runtime_inputs
            self.latest_planner_output = planner_output
            stage_started = time.perf_counter()
            control_message = self.ros_output_adapter.build_control_message(planner_output=planner_output, stamp=_time_message(timestamp_s))
            self.timing_recorder.record_duration(cycle_id, "output.build_autoware_control", (time.perf_counter() - stage_started) * 1000.0)
            planning_cycle_time_ms = (time.perf_counter() - planning_started_monotonic) * 1000.0
            self.latest_planning_cycle_time_ms = float(planning_cycle_time_ms)
            output_created_wall_time_ns = time.time_ns()
            source_to_output_created_ms = (output_created_wall_time_ns - source_send_wall_time_ns) / 1000000.0 if source_send_wall_time_ns > 0 else None
            output_publish_started_wall_time_ns = time.time_ns()
            diagnostics = planner_output.diagnostics_dict()
            prediction_diagnostics = (
                dict(self.prediction_bridge.diagnostics)
                if self.prediction_bridge is not None else {}
            )
            front_gap_m = diagnostics.get("front_gap_m")
            if not isinstance(front_gap_m, (int, float)) or not math.isfinite(front_gap_m):
                front_gap_m = None
            self.timing_recorder.record_event_time(cycle_id, "output.control_topics_publish_started", output_publish_started_wall_time_ns)
            tcp_control_output = String()
            tcp_control_output.data = _strict_json_dumps({"cycle_time_s": float(timestamp_s), "session_id": str(self._active_session_id or ""), "target_speed_mps": float(control_message.longitudinal.velocity), "stop_goal_active": bool(diagnostics.get("stop_goal_active", False)), "emergency_stop": bool(diagnostics.get("emergency_stop_reason", "") or diagnostics.get("emergency_brake_control_active", False)), "emergency_stop_reason": str(diagnostics.get("emergency_stop_reason", "") or ""), "safety_supervisor_reason": str(diagnostics.get("safety_supervisor_reason", "") or ""), "emergency_brake_control_active": bool(diagnostics.get("emergency_brake_control_active", False)), "front_gap_m": front_gap_m, "front_gap_actor_id": str(diagnostics.get("proximity_front_actor_id", "") or diagnostics.get("front_gap_actor_id", "") or ""), "proximity_emergency_gap_m": diagnostics.get("proximity_emergency_gap_m"), "proximity_front_actor_source": str(diagnostics.get("proximity_front_actor_source", "") or ""), "proximity_front_actor_provider_source": str(diagnostics.get("proximity_front_actor_provider_source", "") or ""), "proximity_front_actor_type": str(diagnostics.get("proximity_front_actor_type", "") or ""), "proximity_front_actor_x_m": diagnostics.get("proximity_front_actor_x_m", ""), "proximity_front_actor_y_m": diagnostics.get("proximity_front_actor_y_m", ""), "proximity_front_actor_speed_mps": diagnostics.get("proximity_front_actor_speed_mps", ""), "proximity_front_actor_length_m": diagnostics.get("proximity_front_actor_length_m", ""), "proximity_front_actor_width_m": diagnostics.get("proximity_front_actor_width_m", ""), "acceleration_mps2": float(control_message.longitudinal.acceleration), "steering_rad": float(control_message.lateral.steering_tire_angle), "planning_cycle_time_ms": float(planning_cycle_time_ms), "source_send_wall_time_ns": int(source_send_wall_time_ns), "ros_output_created_wall_time_ns": int(output_created_wall_time_ns), "ros_output_publish_started_wall_time_ns": int(output_publish_started_wall_time_ns), "source_to_output_created_ms": source_to_output_created_ms}, separators=(",", ":"))
            compact_output = json.loads(tcp_control_output.data)
            compact_output.update({
                "compact_schema_revision": int(
                    self._compact_schema_revision
                ),
                "reference_gate_policy_revision": int(
                    REFERENCE_GATE_POLICY_REVISION
                ),
                "ego_x_m": float(runtime_inputs["ego_pose"]["x"]),
                "ego_y_m": float(runtime_inputs["ego_pose"]["y"]),
                "ego_heading_rad": float(runtime_inputs["ego_pose"]["heading_rad"]),
                "behavior_decision": str(planner_output.behavior_command.decision),
                "behavior_fsm_state": str(planner_output.behavior_command.fsm_state),
                "behavior_target_lane_id": diagnostics.get(
                    "behavior_target_lane_id", ""
                ),
                "route_tracking_lane_change_target_lane_id": diagnostics.get(
                    "route_tracking_lane_change_target_lane_id", ""
                ),
                "mpc_status": str(diagnostics.get("mpc_status", "")),
                "mpc_fallback_reason": str(diagnostics.get("mpc_fallback_reason", "")),
                "mpc_replan_executed": bool(
                    diagnostics.get("mpc_replan_executed", False)
                ),
                "mpc_obstacle_input_count": diagnostics.get(
                    "mpc_obstacle_input_count", 0
                ),
                "mpc_obstacle_relevant_count": diagnostics.get(
                    "mpc_obstacle_relevant_count", 0
                ),
                "mpc_obstacle_ignored_count": diagnostics.get(
                    "mpc_obstacle_ignored_count", 0
                ),
                "mpc_obstacle_ignored_ids": diagnostics.get(
                    "mpc_obstacle_ignored_ids", []
                ),
                "mpc_infeasibility_diagnostic": diagnostics.get(
                    "mpc_infeasibility_diagnostic", {}
                ),
                "reference_source": str(
                    diagnostics.get(
                        "reference_source",
                        diagnostics.get("lateral_reference_source", ""),
                    )
                    or ""
                ),
                "route_sync_reason": str(
                    diagnostics.get(
                        "route_sync_reason",
                        diagnostics.get("carla_route_sync_reason", ""),
                    )
                    or ""
                ),
                "route_progress_index": diagnostics.get(
                    "route_progress_index",
                    diagnostics.get("carla_route_progress_index", ""),
                ),
                "route_upcoming_turn_direction": str(
                    diagnostics.get("route_upcoming_turn_direction", "") or ""
                ),
                "route_upcoming_turn_distance_m": diagnostics.get(
                    "route_upcoming_turn_distance_m", ""
                ),
                "route_remaining_distance_m": diagnostics.get(
                    "route_remaining_distance_m",
                    diagnostics.get("distance_to_destination_m", ""),
                ),
                "mission_complete": bool(diagnostics.get("mission_complete", False)),
                "current_lane_id": diagnostics.get("current_lane_id", ""),
                "route_physical_target_lane_id": diagnostics.get(
                    "route_physical_target_lane_id", ""
                ),
            })
            # Keep the TCP response compact, but expose the reference
            # lifecycle evidence needed to diagnose a hard-gate stop.  These
            # scalar fields let OpenCDA distinguish an exhausted immutable
            # master from a newly rejected candidate without transporting the
            # full reference trajectory on the control-critical path.
            for diagnostic_key in (
                "planner_output_reference_count",
                "final_reference_gate_valid",
                "final_reference_gate_reason",
                "reference_pipeline_conditioning_reason",
                "persistent_reference_active",
                "persistent_reference_state",
                "persistent_reference_progress_s_m",
                "persistent_reference_geometry_revision",
                "persistent_reference_build_reason",
                "persistent_reference_last_rejected_trigger",
                "candidate_lane_change_lock_reason",
                "candidate_lane_change_window_reason",
                "candidate_selected_lane_change_duration_s",
                "candidate_selected_lane_change_initial_progress",
                "candidate_selected_lane_change_terminal_progress",
                "candidate_selected_lane_change_geometry_design_speed_mps",
                "candidate_selected_lane_change_geometry_speed_mps",
                "candidate_selected_lane_change_geometry_length_m",
                "candidate_selected_lane_change_geometry_step_m",
                "reference_max_curvature_1pm",
                "reference_first_forward_m",
                "reference_first_lateral_m",
                "destination_forward_m",
                "destination_lateral_m",
                "lane_change_authorization_reason",
                "lane_change_authorization_direction",
                "lane_change_authorization_distance_m",
                "route_geometry_lane_change_direction",
                "route_geometry_lane_change_distance_m",
                "route_geometry_lane_change_reason",
                "route_current_road_option",
                "route_next_macro_maneuver",
                "route_lane_change_edge_id",
                "lane_change_phase",
                "lane_change_completion_reason",
                "lane_change_completion_stable_frames",
                "lane_change_completion_lateral_error_m",
                "lane_change_completion_heading_error_deg",
                "lane_change_completion_predicted_lateral_error_m",
                "lane_change_completion_target_lane_matches",
                "lane_change_completion_footprint_clearance_m",
                "lane_change_stabilization_entry_lateral_error_m",
                "lane_change_stabilization_entry_heading_error_deg",
                "lane_change_stabilization_geometry_ready",
                "lane_change_geometry_completion_latched",
                "lane_change_to_turn_transition_arc_m",
                "lane_change_to_turn_transition_progress_m",
                "maneuver_commitment_progress",
                "map_match_ad_lane_id",
                "route_turn_reference_reason",
                "cav_conflict_summary",
                "cav_conflict_reference_sample_count",
                "cav_conflict_tags",
                "cav_conflict_tag_reasons",
                "cav_conflict_roles",
                "cav_arbitration",
                "cav_ego_claim",
                "cav_peer_claims",
                "cav_shared_plan_count",
                "cav_intent_count",
                "cav_total_qp_row_count",
                "cav_anticipatory_speed_cap_mps",
                "cav_anticipatory_speed_constraint_owner",
                "cav_anticipatory_speed_constraint_reason",
            ):
                if diagnostic_key in diagnostics:
                    compact_output[diagnostic_key] = diagnostics[diagnostic_key]
            compact_output.update(prediction_diagnostics)
            compact_output.update({
                "planner_cycle_executed": True,
                "control_reused": False,
                "control_reuse_reason": "fresh_completed_plan",
                "control_plan_age_s": 0.0,
                "control_plan_cycle_time_s": float(timestamp_s),
            })
            tcp_control_output.data = _strict_json_dumps(
                compact_output,
                separators=(",", ":"),
            )
            stage_started = time.perf_counter()
            self.tcp_control_output_publisher.publish(tcp_control_output)
            self._execution_control_cache.update(
                self._active_session_id,
                compact_output,
                created_wall_time_ns=output_created_wall_time_ns,
                planned_trajectory=planner_output.planned_trajectory,
                wheelbase_m=float(self.planner_bridge.mpc.wheelbase_m),
                min_steer_rad=float(
                    self.planner_bridge.mpc.constraints.min_steer_rad
                ),
                max_steer_rad=float(
                    self.planner_bridge.mpc.constraints.max_steer_rad
                ),
                min_steer_rate_rps=float(
                    self.planner_bridge.mpc.constraints.min_steer_rate_rps
                ),
                max_steer_rate_rps=float(
                    self.planner_bridge.mpc.constraints.max_steer_rate_rps
                ),
            )
            self.timing_recorder.record_duration(cycle_id, "output_publishing.ros_compact_control_topic", (time.perf_counter() - stage_started) * 1000.0)
            stage_started = time.perf_counter()
            self.control_publisher.publish(control_message)
            self.timing_recorder.record_duration(cycle_id, "output_publishing.autoware_control_topic", (time.perf_counter() - stage_started) * 1000.0)
            output_publish_finished_wall_time_ns = time.time_ns()
            self.timing_recorder.record_event_time(cycle_id, "output.control_topics_publish_finished", output_publish_finished_wall_time_ns)
            self.timing_recorder.record_duration(cycle_id, "output.publish", (output_publish_finished_wall_time_ns - output_publish_started_wall_time_ns) / 1000000.0)
            self.timing_recorder.record_duration(cycle_id, "output_publishing.ros_control_topics_total", (output_publish_finished_wall_time_ns - output_publish_started_wall_time_ns) / 1000000.0)
            self.timing_recorder.record_duration(cycle_id, "planner.planning_cycle_time", planning_cycle_time_ms)
            if source_to_output_created_ms is not None:
                self.timing_recorder.record_duration(cycle_id, "transport.opencda_send_to_ros_output_created", source_to_output_created_ms)
            self._refresh_stage_profile_metadata()
            self.timing_recorder.finish_cycle(cycle_id, {"success": True})
            self.latest_control_message = control_message
            # Debug serialization, disk I/O and console logging are
            # observability work, not part of the control critical path.
            # Sample them after publishing control so enabling ``debug``
            # cannot delay the command for the same simulator cycle.
            if self._debug_sample_due(timestamp_s):
                self.write_planner_input_adapter_output(runtime_inputs)
                self.publish_debug_output(runtime_inputs, planner_output)
                self._log_cycle(runtime_inputs, planner_output)
            self._runtime_pause_monitor.clear_cycle()

    def _publish_reused_execution_control(self, session_payload):
        """Publish the latest safe completed plan for one new source tick."""

        session_id = str(dict(session_payload or {}).get("session_id", "") or "")
        with self._execution_ego_state_lock:
            ego_state = deepcopy(
                self._latest_execution_ego_states.get(session_id)
            )
        payload = self._execution_control_cache.reused_for_session(
            session_payload,
            ego_state=ego_state,
        )
        if payload is None:
            return False
        message = String()
        message.data = _strict_json_dumps(payload, separators=(",", ":"))
        self.tcp_control_output_publisher.publish(message)
        return True

    def _debug_sample_due(self, timestamp_s):
        """Return whether low-rate debug artifacts are due for this cycle."""
        if not self.debug:
            return False
        timestamp_s = float(timestamp_s)
        if timestamp_s < self._last_debug_sample_timestamp_s - 1.0e-6:
            self._last_debug_sample_timestamp_s = -float("inf")
        if (
            timestamp_s - self._last_debug_sample_timestamp_s
            < self._debug_sample_interval_s - 1.0e-6
        ):
            return False
        self._last_debug_sample_timestamp_s = timestamp_s
        return True

    def _refresh_stage_profile_metadata(self, force=False):
        """Publish bounded profiler aggregates without per-call forwarding.

        Profiling remains local to the planning worker. Updating the JSON
        metadata every 100 dispatched frames makes the breakdown observable
        while keeping measurement overhead outside hot geometry loops.
        """
        if not self.debug_time:
            return
        stage_profiler = getattr(self, "stage_profiler", None)
        if stage_profiler is None:
            return
        dispatched = int(
            self._input_frame_buffer.stats().get("dispatched_frames", 0)
        )
        session_dispatched = max(
            0,
            dispatched - int(self._stage_profile_dispatch_baseline),
        )
        if not force and session_dispatched % 100 != 0:
            return
        self.timing_recorder.set_metadata(
            "planner_stage_profile",
            stage_profiler.summary(session_dispatched),
        )

    def _publish_input_frame(self, cycle_id, timestamp_s, runtime_inputs):
        """Publish the simulator-neutral current-planner input for recording."""
        message = PlannerInputFrame()
        fill_header(message.header, timestamp_s)
        message.cycle_id = int(cycle_id)
        message.sim_time_s = float(timestamp_s)
        message.planner_input_adapter_output_json = ""
        message.runtime_inputs_json = encode_json(runtime_inputs)
        message.metadata_json = encode_json(
            {
                "planner_source": "opencda.current",
                "session_id": str(self._active_session_id or ""),
                "v2x_nearby_count": int(
                    runtime_inputs.get("v2x_nearby_count", 0) or 0
                ),
            }
        )
        if self.local_bus is not None:
            self.local_bus.store_input_frame(message)
        self.input_frame_publisher.publish(message)

    def publish_debug_output(self, runtime_inputs, planner_output):
        """Publish the normalized ROS input and current planner output."""
        diagnostics = planner_output.diagnostics_dict()
        ego_pose = dict(runtime_inputs["ego_pose"])
        local_objects = list(runtime_inputs.get("local_object_snapshots", []) or [])
        cp_payload = dict(runtime_inputs.get("cp_payload", {}) or {})
        cp_objects = list(cp_payload.get("obstacles", []) or [])
        reference_xy = [[float(dict(sample).get("x_ref_m", dict(sample).get("x", 0.0))), float(dict(sample).get("y_ref_m", dict(sample).get("y", 0.0)))] for sample in list(planner_output.reference_trajectory or []) if isinstance(sample, dict)]
        planned_trajectory = [[float(value) for value in list(state)[:4]] for state in list(planner_output.planned_trajectory or [])]
        payload = {
            "schema_version": 1,
            "cycle_time_s": float(runtime_inputs["sim_time_s"]),
            "planner_source": "opencda.current",
            "input": {
                "ego_state": [
                    float(ego_pose["x"]),
                    float(ego_pose["y"]),
                    float(runtime_inputs["ego_speed_mps"]),
                    float(ego_pose["heading_rad"]),
                ],
                "local_objects": _json_safe(local_objects),
                "cp_objects": _json_safe(cp_objects),
            },
            "behavior": planner_output.behavior_command.as_dict(),
            "predictions": _json_safe(planner_output.predictions),
            "mpc": {
                "acceleration_mps2": float(planner_output.acceleration_mps2),
                "steering_rad": float(planner_output.steering_rad),
                "status": str(diagnostics.get("mpc_status", "")),
                "fallback_reason": str(diagnostics.get("mpc_fallback_reason", "")),
                "replan_executed": bool(diagnostics.get("mpc_replan_executed", False)),
            },
            "reference_xy": reference_xy,
            "planned_trajectory": planned_trajectory,
        }
        message = String()
        message.data = _strict_json_dumps(payload, separators=(",", ":"))
        if self.debug_output_publisher is not None:
            self.debug_output_publisher.publish(message)

    def write_planner_input_adapter_output(self, runtime_inputs):
        """Append the normalized input handed to the current planner."""
        record = {
            "schema_version": 1,
            "source": "ros",
            "planner_source": "opencda.current",
            "cycle_time_s": float(runtime_inputs["sim_time_s"]),
            "runtime_inputs": _json_safe(runtime_inputs),
        }
        with self.planner_input_log_path.open("a", encoding="utf-8") as output_file:
            output_file.write(_strict_json_dumps(record, separators=(",", ":")) + "\n")

    def _log_cycle(self, runtime_inputs, planner_output):
        """Print a short status line for the cycle that was just published."""
        ego_pose = dict(runtime_inputs["ego_pose"])
        cp_payload = dict(runtime_inputs.get("cp_payload", {}) or {})
        object_count = len(runtime_inputs.get("local_object_snapshots", []) or [])
        object_count += len(cp_payload.get("obstacles", []) or [])
        behavior = planner_output.behavior_command
        self.get_logger().info("CP-X cycle {:.3f}: ego=({:.2f},{:.2f}) objects={} decision={} fsm={} accel={:.3f} steer={:.4f}".format(float(runtime_inputs["sim_time_s"]), float(ego_pose["x"]), float(ego_pose["y"]), int(object_count), str(behavior.decision), str(behavior.fsm_state), float(planner_output.acceleration_mps2), float(planner_output.steering_rad)))

    def destroy_node(self):
        """Close the current planner and any proxy owned by this node."""
        self._input_worker_stop.set()
        self._input_frame_buffer.wake_all()
        input_worker = getattr(self, "_input_worker", None)
        if input_worker is not None and input_worker.is_alive():
            input_worker.join(timeout=5.0)
        prediction_dispatcher = getattr(
            self, "_prediction_scene_dispatcher", None
        )
        if prediction_dispatcher is not None:
            dispatcher_stopped = prediction_dispatcher.close(timeout_s=2.0)
            dispatcher_stats = prediction_dispatcher.stats()
            self.timing_recorder.set_metadata(
                "prediction_scene_dispatch", dispatcher_stats
            )
            self.get_logger().info(
                "Prediction scene dispatcher summary: {}".format(
                    dispatcher_stats
                )
            )
            if not dispatcher_stopped:
                self.get_logger().warning(
                    "Prediction scene publisher did not stop within 2 seconds; "
                    "DDS transport may still be blocked."
                )
        self.get_logger().info(
            "Synchronized input buffer summary: {}".format(
                self._input_frame_buffer.stats()
            )
        )
        self._save_active_session_context()
        closed_runtime_ids = set()
        for context in self._session_contexts.values():
            runtime = context.planner_runtime
            if runtime is None or id(runtime) in closed_runtime_ids:
                continue
            runtime.close()
            closed_runtime_ids.add(id(runtime))
        if (
            self._bootstrap_runtime is not None
            and id(self._bootstrap_runtime) not in closed_runtime_ids
        ):
            self._bootstrap_runtime.close()
            closed_runtime_ids.add(id(self._bootstrap_runtime))
        if self._owns_global_planner_proxy:
            self.global_planner.close()
        self._refresh_stage_profile_metadata(force=True)
        self.timing_recorder.set_metadata(
            "python_runtime_pauses",
            self._runtime_pause_monitor.close(),
        )
        self.timing_recorder.write_summary()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CPXPlannerNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
