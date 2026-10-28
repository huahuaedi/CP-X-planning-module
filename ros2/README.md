# OpenCDA-to-ROS 2 Data Bridge

## Overview

This project receives data produced by the OpenCDA localization and
perception and V2X modules. One unified bridge node owns all TCP connections,
publishes the typed ROS input topics, and forwards planner control back to
OpenCDA.

OpenCDA runs with Python 3.7, while ROS 2 Jazzy uses Python 3.12. The two
processes therefore communicate through local TCP connections using JSON.

```text
OpenCDA (Python 3.7)
  ├── TCP 5051: localization + destination
  ├── TCP 5052: perception
  ├── TCP 5053: traffic lights
  ├── TCP 5054: V2X + CP obstacles
  └── TCP 5055: cooperative messages
             │
             ▼
      opencda_bridge_node
             │ typed /cpx/* ROS topics
             ▼
        planning_system
             │ planner control
             └── TCP 5060 ──> OpenCDA
```

The ROS nodes do not read CARLA directly. They only receive outputs already
produced by the OpenCDA Localization, Perception, and V2X modules.

## Reproducible quick start

The repository now provides one lifecycle command for the Planner, Bridge, and
MTR nodes. Run it **inside the ROS/GPU container**; CARLA and OpenCDA remain in
their existing host environment.

```bash
cd /path/to/CP-X-planning-module/ros2
cp .env.example .env        # once; edit only paths that differ
./scripts/cpx_ros_stack.sh build
./scripts/cpx_ros_stack.sh start
```

The default starts one unified OpenCDA Bridge, two isolated CP-X Planner
workers, and one shared six-mode MTR process. It refuses to start if an older
Bridge still owns TCP ports 5051-5055.

Before OpenCDA connects:

```bash
./scripts/cpx_ros_stack.sh health
```

While an OpenCDA scenario is running:

```bash
./scripts/cpx_ros_stack.sh health-live
```

Operational commands are:

```bash
./scripts/cpx_ros_stack.sh status
./scripts/cpx_ros_stack.sh logs
./scripts/cpx_ros_stack.sh stop
./scripts/cpx_ros_stack.sh restart
./scripts/cpx_ros_stack.sh manifest
```

`stop` signals only process groups recorded under the ignored `.runtime/`
directory; it does not use broad `pkill` patterns. Exact validated dependency
versions and environment ownership are documented in `dependencies/`.

## Files

The ROS package is located in `src/cpx_comm_test`.

| File | Purpose |
|---|---|
| `opencda_bridge_node.py` | Production bridge owning TCP ports 5051-5055, all input topic publishers, the lifecycle topic, and the TCP 5060 control return path. |
| `tcp_json_receiver.py` | Shared method for receiving newline-separated JSON over TCP. |
| `localization_publisher.py` | Receives localization data on port 5051 and publishes `/cpx/localization`. |
| `perception_publisher.py` | Receives nearby-object data on port 5052 and publishes `/cpx/perception`. |
| `traffic_light_publisher.py` | Receives traffic-light data on port 5053 and publishes `/cpx/traffic_light`. |
| `v2x_publisher.py` | Receives V2X data on port 5054; publishes `nearby_cavs` on `/cpx/v2x` and `cp_obstacles` on `/cpx/cp_obstacles`. |
| `cooperative_message_publisher.py` | Receives lane events and traffic controls on port 5055 and publishes `/cpx/cooperative_messages`. |
| Legacy publisher and subscriber files | Retained as compatibility/debug executables; the unified launch does not start them. |

OpenCDA's simulator-only `SafetyManager` remains local to scenario evaluation.
It is not transmitted to ROS and cannot trigger planner control. The bridge
publishes `/cpx/planning/session` from localization-envelope metadata. The
same session owner is encoded on the internal input headers, so equal CARLA
timestamps from multiple CAVs cannot be combined into one planning frame.

The planner keeps one synchronized input buffer and one complete runtime per
`session_id`. Route state, behavior memory, reference state, MPC solver/warm
start, prediction cache, and last control are therefore independent for each
CAV. Compact control output retains `session_id`, allowing the shared TCP 5060
receiver to return each command only to its owning vehicle.

When ROS prediction is enabled, both `PredictionScene` and `PredictionFrame`
carry an explicit `session_id` (with the encoded header retained as a rollout
compatibility fallback). MTR keeps a separate history window,
history/inference timestamp, pending latest scene, and revision counter per
session while sharing the loaded GPU model and the static map cache.

Custom ROS messages are defined in `src/cpx_interfaces`.

| Message | Purpose |
|---|---|
| `LaneEvent.msg` | Describes a lane closure, roadway hazard, or work zone. |
| `TrafficControl.msg` | Describes a cooperative traffic-light state. |
| `CooperativeMessageArray.msg` | Carries lane events and traffic controls in one synchronized update. |

## ROS topics and message types

| Topic | ROS message type |
|---|---|
| `/cpx/localization` | `nav_msgs/msg/Odometry` |
| `/cpx/perception` | `autoware_perception_msgs/msg/PredictedObjects` |
| `/cpx/traffic_light` | `autoware_perception_msgs/msg/TrafficLightGroupArray` |
| `/cpx/v2x` | `autoware_perception_msgs/msg/TrackedObjects` |
| `/cpx/cooperative_messages` | `cpx_interfaces/msg/CooperativeMessageArray` |
| `/cpx/planning/session` | `std_msgs/msg/String` lifecycle metadata |

## Custom message format

`LaneEvent.msg` contains:

```text
uint8 UNKNOWN=0
uint8 LANE_CLOSURE=1
uint8 ROAD_HAZARD=2
uint8 WORK_ZONE=3

string id
uint8 event_type
string source
builtin_interfaces/Time source_stamp
builtin_interfaces/Duration ttl
geometry_msgs/Point position
float32 confidence
```

`TrafficControl.msg` contains:

```text
string id
string source
builtin_interfaces/Time source_stamp
builtin_interfaces/Duration ttl
autoware_perception_msgs/TrafficLightGroup signal
float32 confidence
```

 The
`signal` field reuses Autoware's traffic-light format for the light ID, color,
shape, status, and confidence.

`CooperativeMessageArray.msg` puts both custom message lists into one update:

```text
std_msgs/Header header
uint16 schema_version
uint64 sequence
cpx_interfaces/LaneEvent[] lane_events
cpx_interfaces/TrafficControl[] traffic_controls
```



## Data format


### Localization

```json
{
  "ego_state": {
    "x": 100.5,
    "y": 25.2,
    "z": 0.3,
    "v": 5.5,
    "psi": 1.57
  }
}
```

### Perception

```json
{
  "objects": [
    {
      "id": "25",
      "vehicle_id": "25",
      "type": "vehicle.tesla.model3",
      "x": 110.2,
      "y": 25.4,
      "z": 0.3,
      "v": 4.8,
      "psi": 1.57,
      "length_m": 4.5,
      "width_m": 1.8,
      "height_m": 1.5,
      "confidence": 1.0
    }
  ]
}
```


### Traffic light

```json
{
  "traffic_lights": [
    {
      "id": "12",
      "type": "traffic_light",
      "state": "red",
      "position": {
        "x": 125.0,
        "y": 25.0,
        "z": 3.0
      },
      "confidence": 1.0
    }
  ]
}
```

### V2X

```json
{
  "nearby_cavs": [
    {
      "id": "cav-id",
      "vehicle_id": "42",
      "type": "vehicle.lincoln.mkz_2017",
      "source": "opencda_v2x",
      "x": 100.0,
      "y": 25.0,
      "z": 0.3,
      "v": 5.0,
      "psi": 1.57,
      "length_m": 4.9,
      "width_m": 1.9,
      "height_m": 1.5,
      "confidence": 1.0
    }
  ],
  "communication_range_m": 100.0
}
```

### Cooperative lane events and traffic controls

This is the JSON input expected on TCP port `5055`. The
`cooperative_message_publisher` converts it into
`cpx_interfaces/msg/CooperativeMessageArray`.

```json
{
  "schema_version": 1,
  "sequence": 4,
  "timestamp_s": 20.5,
  "lane_events": [
    {
      "id": "closure-1",
      "type": "lane_closure",
      "source": "roadside_unit",
      "timestamp_s": 20.5,
      "ttl_s": 10.0,
      "position": [100.0, 25.0, 0.0],
      "confidence": 1.0
    }
  ],
  "control": [
    {
      "id": "signal-12",
      "control_id": "12",
      "type": "traffic_light",
      "signal_state": "red",
      "source": "roadside_unit",
      "timestamp_s": 20.5,
      "ttl_s": 2.0,
      "confidence": 1.0
    }
  ]
}
```



## Requirements

- Ubuntu with ROS 2 Jazzy installed.
- ROS interface generation tools:
  `sudo apt install ros-jazzy-rosidl-default-generators`
- Planner numerical packages from
  `dependencies/planner-python312.lock.txt` installed into the configured
  Python overlay.
- MTR packages from `dependencies/mtr-python312.lock.txt`, with CUDA extensions
  rebuilt in the configured MTR virtual environment.
- OpenCDA project configured with Python 3.7 and CARLA 0.9.12.
- The OpenCDA `data_transmitter.py` connected to
  `VehicleManager.update_info()`.
- The ROS workspace and OpenCDA must run on the same computer because the TCP
  receiver uses `127.0.0.1`.

## Run OpenCDA

First, start the CARLA 0.9.12 server in a separate terminal (change the path):

```bash
export UE4_ROOT=/home/umd-user/carla_source/UnrealEngine_4.26

"$UE4_ROOT/Engine/Binaries/Linux/UE4Editor" \
  /home/umd-user/carla_source/carla/Unreal/CarlaUE4/CarlaUE4.uproject \
  /Game/Carla/Maps/Town10HD_Opt \
  -game -vulkan -quality-level=Low -ResX=800 -ResY=600
```

Wait until the CARLA server is ready on port 2000. Then open another terminal
and run the scenario:

```bash
conda activate carla307
cd /path/to/CP-X-planning-module
python opencda.py -t cpx_town10_scenario_1 -v 0.9.12
```
Note: Change the name of the conda environment.
## Build this ROS workspace

The planner now consumes the typed messages owned by the Prediction
repository. Build and source that interface package first (or place both
packages under one colcon workspace):

```bash
source /opt/ros/jazzy/setup.bash
cd ~/Downloads/CP-X-Prediction/ROS2-Code
colcon build --symlink-install --packages-select mtr_prediction_msgs
source install/setup.bash
```

Then build the planning workspace in the same shell so colcon can resolve
`mtr_prediction_msgs`:

```bash
cd /path/to/CP-X-planning-module/ros2
colcon build --symlink-install \
  --packages-select cpx_interfaces cpx_comm_test cpx_planning
source install/setup.bash
```

Rebuild the workspace after changing any ROS source file.

## Run

Source ROS and the workspace in one ROS terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Downloads/CP-X-Prediction/ROS2-Code/install/setup.bash
source /path/to/CP-X-planning-module/ros2/install/setup.bash
```

### 1. Start the unified bridge

```bash
ros2 launch cpx_comm_test communication.launch.py debug:=false debug_time:=true
```

For the complete planner integration, use the single top-level launch instead:

```bash
ros2 launch cpx_planning opencda_integration.launch.py debug:=false debug_time:=true
```

Do not start the legacy publisher/subscriber executables alongside either launch;
they would compete for the same TCP ports.

### 2. Start CARLA and run the OpenCDA scenario

After the CARLA 0.9.12 server is ready, open the OpenCDA terminal:

```bash
conda activate carla307
cd /path/to/CP-X-planning-module
python opencda.py -t cpx_town10_scenario_1 -v 0.9.12
```

With `debug:=true`, the bridge logs the arrival of localization, perception,
traffic-light, V2X, CP, and cooperative messages.

## MTR Prediction ROS integration

The integration is an asynchronous two-topic loop:

```text
OpenCDA 20 Hz inputs -> CP-X Planner -> /prediction/scene
                                         |
                               MTR history 10 Hz / inference 5 Hz
                                         |
                       /prediction/frame -> latest-result cache -> Planner
```

`/prediction/frame` is intentionally not one of the exact 20 Hz synchronized
Planner inputs. The Planner reads the newest frame whose simulation-time age
is at most 0.5 s, rebases every future waypoint to the current planning tick,
and preserves all six modes and their probabilities. If no fresh result is
available, the existing constant-acceleration prediction is used; planning
does not wait for inference.

Start the MTR node in the environment that contains PyTorch, the compiled MTR
operators, `rclpy`, and the generated message package:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Downloads/CP-X-Prediction/ROS2-Code/install/setup.bash
export CPX_PREDICTION_ROOT=~/Downloads/CP-X-Prediction
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ros2 run mtr_prediction_msgs mtr_prediction_node --ros-args \
  -p num_modes:=6 -p history_hz:=10.0 -p inference_hz:=5.0
```

In the ROS planning/container environment, source both workspaces and enable
the adapter explicitly:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Downloads/CP-X-Prediction/ROS2-Code/install/setup.bash
source /path/to/CP-X-planning-module/ros2/install/setup.bash
export PYTHONPATH=/root/cpx-python312-deps:/root/CP-X-planning-module:$PYTHONPATH
export ROS_DOMAIN_ID=42
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ros2 launch cpx_planning opencda_integration.launch.py \
  prediction_enabled:=true debug:=false debug_time:=true
```

All processes must use the same `ROS_DOMAIN_ID`. If Prediction runs in a
different container or computer, use `discovery_range:=SUBNET` and ensure DDS
UDP traffic is reachable; do not set discovery to `LOCALHOST` across machines.

Verify the live contract before evaluating vehicle behavior:

```bash
ros2 topic hz /prediction/scene
ros2 topic hz /prediction/frame
ros2 topic echo /prediction/frame --once
ros2 topic hz /cpx/planner_control_output
```

Expected rates are approximately 20 Hz for the scene and Planner input,
5 Hz for prediction frames, and close to 20 Hz for reused Planner control
output. The first scene may have no map polylines; subsequent scenes use the
Planner's authoritative local AD-map snapshot.
