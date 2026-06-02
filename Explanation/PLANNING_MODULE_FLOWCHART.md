# Complete Planning Module Flowchart

## Overview

This document provides a comprehensive flowchart of the entire planning module in OpenCDA, showing all steps, decision points, and the files where each operation is performed.

---

## System Architecture Overview

```
VehicleManager (opencda/core/common/vehicle_manager.py)
    │
    ├─→ BehaviorAgent (opencda/core/plan/behavior_agent.py)
    │       │
    │       ├─→ GlobalRoutePlanner (opencda/core/plan/global_route_planner.py)
    │       ├─→ LocalPlanner (opencda/core/plan/local_planner_behavior.py)
    │       └─→ CollisionChecker (opencda/core/plan/collision_check.py)
    │
    └─→ ControlManager (opencda/core/actuation/control_manager.py)
            └─→ PID Controller (opencda/core/actuation/pid_controller.py)
```

---

## Phase 1: Initialization

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: INITIALIZATION                                     │
│ File: opencda/core/common/vehicle_manager.py                │
└─────────────────────────────────────────────────────────────┘

VehicleManager.__init__()
  ↓
  ├─→ Create BehaviorAgent
  │     File: opencda/core/plan/behavior_agent.py
  │     Method: BehaviorAgent.__init__()
  │     │
  │     ├─→ Initialize ego state (_ego_pos, _ego_speed)
  │     ├─→ Load config parameters (speed, safety, etc.)
  │     ├─→ Create CollisionChecker
  │     │     File: opencda/core/plan/collision_check.py
  │     │     Method: CollisionChecker.__init__()
  │     │
  │     ├─→ Create LocalPlanner
  │     │     File: opencda/core/plan/local_planner_behavior.py
  │     │     Method: LocalPlanner.__init__()
  │     │     │
  │     │     ├─→ Initialize waypoints_queue (deque, maxlen=20000)
  │     │     ├─→ Initialize _waypoint_buffer (deque, maxlen=buffer_size)
  │     │     ├─→ Initialize _trajectory_buffer (deque, maxlen=30)
  │     │     └─→ Initialize _history_buffer (deque, maxlen=3)
  │     │
  │     └─→ Initialize GlobalRoutePlanner (lazy, created on first route)
  │           File: opencda/core/plan/global_route_planner.py
  │           (Not created yet, will be created in set_destination)
  │
  └─→ Create ControlManager
        File: opencda/core/actuation/control_manager.py
        Method: ControlManager.__init__()
          │
          └─→ Create PID Controller
                File: opencda/core/actuation/pid_controller.py
                Method: Controller.__init__()
```

---

## Phase 2: Route Setup (One-Time or When Destination Changes)

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: ROUTE SETUP                                        │
│ File: opencda/core/plan/behavior_agent.py                  │
│ Method: set_destination()                                   │
└─────────────────────────────────────────────────────────────┘

START: set_destination(start_location, end_location)
  ↓
  STEP 1: Clean buffers if needed
  │       File: opencda/core/plan/local_planner_behavior.py
  │       Method: LocalPlanner.get_waypoints_queue().clear()
  │       Method: LocalPlanner.get_waypoint_buffer().clear()
  │
  STEP 2: Convert locations to waypoints
  │       File: opencda/core/plan/behavior_agent.py
  │       Method: _map.get_waypoint()
  │       │
  │       ├─→ start_waypoint = _map.get_waypoint(start_location)
  │       └─→ end_waypoint = _map.get_waypoint(end_location)
  │
  STEP 3: Ensure start waypoint is ahead of vehicle
  │       File: opencda/core/plan/behavior_agent.py
  │       Method: cal_distance_angle() (from opencda/core/common/misc.py)
  │       │
  │       └─→ While angle > 90°:
  │             start_waypoint = start_waypoint.next(1)[0]
  │
  STEP 4: Generate global route
  │       File: opencda/core/plan/behavior_agent.py
  │       Method: _trace_route(start_waypoint, end_waypoint)
  │       │
  │       ├─→ If _global_planner is None:
  │       │     │
  │       │     ├─→ Create GlobalRoutePlannerDAO
  │       │     │     File: opencda/core/plan/global_route_planner_dao.py
  │       │     │     Method: GlobalRoutePlannerDAO.__init__()
  │       │     │
  │       │     ├─→ Create GlobalRoutePlanner
  │       │     │     File: opencda/core/plan/global_route_planner.py
  │       │     │     Method: GlobalRoutePlanner.__init__()
  │       │     │
  │       │     └─→ Setup global planner
  │       │           Method: GlobalRoutePlanner.setup()
  │       │           │
  │       │           ├─→ Get topology from DAO
  │       │           │     Method: _dao.get_topology()
  │       │           │
  │       │           ├─→ Build graph representation
  │       │           │     Method: _build_graph()
  │       │           │     │
  │       │           │     ├─→ Convert topology to NetworkX graph
  │       │           │     ├─→ Create id_map (coordinates → node IDs)
  │       │           │     └─→ Create road_id_to_edge mapping
  │       │           │
  │       │           ├─→ Find loose ends
  │       │           │     Method: _find_loose_ends()
  │       │           │
  │       │           └─→ Add lane change links
  │       │                 Method: _lane_change_link()
  │       │
  │       └─→ Generate route
  │             Method: GlobalRoutePlanner.trace_route()
  │             │
  │             ├─→ Find path through graph
  │             │     Method: _path_search(origin, destination)
  │             │     │
  │             │     ├─→ Localize origin and destination
  │             │     │     Method: _localize(location)
  │             │     │
  │             │     └─→ A* pathfinding
  │             │           Method: nx.astar_path()
  │             │
  │             └─→ Convert path to waypoint trace
  │                   Method: trace_route()
  │                   │
  │                   ├─→ For each edge in route:
  │                   │     │
  │                   │     ├─→ Determine turn decision
  │                   │     │     Method: _turn_decision(i, route)
  │                   │     │
  │                   │     └─→ Add waypoints with RoadOption
  │                   │
  │                   └─→ Return: List[(Waypoint, RoadOption)]
  │
  STEP 5: Set route in LocalPlanner
  │       File: opencda/core/plan/local_planner_behavior.py
  │       Method: LocalPlanner.set_global_plan(route_trace, clean)
  │       │
  │       ├─→ Add waypoints to waypoints_queue
  │       └─→ Optionally refill _waypoint_buffer
  │
  END: Route ready for planning
```

---

## Phase 3: Main Planning Loop (Every Simulation Step)

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: MAIN PLANNING LOOP                                 │
│ File: opencda/core/common/vehicle_manager.py                │
│ Method: VehicleManager.run_step()                           │
└─────────────────────────────────────────────────────────────┘

START: VehicleManager.run_step()
  ↓
  ┌─────────────────────────────────────────────────────────┐
  │ STEP 1: Update Information                              │
  │ File: opencda/core/common/vehicle_manager.py            │
  │ Method: VehicleManager.update_info()                    │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ Get ego position from LocalizationManager
  ├─→ Get ego speed from LocalizationManager
  ├─→ Get objects from PerceptionManager
  │
  └─→ Update BehaviorAgent
        File: opencda/core/plan/behavior_agent.py
        Method: BehaviorAgent.update_information()
        │
        ├─→ Store ego_pos, ego_speed
        ├─→ Calculate break_distance
        │     Formula: ego_speed / 3.6 * emergency_param
        │
        ├─→ Update LocalPlanner
        │     File: opencda/core/plan/local_planner_behavior.py
        │     Method: LocalPlanner.update_information()
        │     │
        │     └─→ Store ego_pos, ego_speed in LocalPlanner
        │
        ├─→ Filter obstacles (white list matching)
        │     Method: white_list_match(obstacle_vehicles)
        │     │
        │     └─→ Remove platoon members from obstacle list
        │
        ├─→ Update traffic light state
        │     Method: vehicle.get_traffic_light_state()
        │
        └─→ Update debug helper
              Method: debug_helper.update(ego_speed, ttc)

  ┌─────────────────────────────────────────────────────────┐
  │ STEP 2: Behavior Planning                               │
  │ File: opencda/core/plan/behavior_agent.py               │
  │ Method: BehaviorAgent.run_step()                        │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ Initialize step variables
  │     ├─→ Reset ttc = 1000
  │     ├─→ Decrement overtake_counter if > 0
  │     └─→ Decrement destination_push_flag if > 0
  │
  ├─→ Detect intersection
  │     Method: is_intersection(objects, waypoint_buffer)
  │     │
  │     └─→ Check if any traffic light within 20m of future waypoints
  │
  ├─→ DECISION POINT 1: Destination Reached?
  │     Method: is_close_to_destination()
  │     │
  │     ├─→ YES: Exit simulation (sys.exit(0))
  │     └─→ NO: Continue
  │
  ├─→ DECISION POINT 2: Traffic Light / Stop Sign?
  │     Method: traffic_light_manager(ego_vehicle_wp)
  │     │
  │     ├─→ If red light or stop sign:
  │     │     │
  │     │     ├─→ Stop sign: Wait 60 frames (~2 seconds)
  │     │     └─→ Red light: Return (0, None) → Stop command
  │     │
  │     └─→ If green: Continue
  │
  ├─→ DECISION POINT 3: Temporary Route Finished?
  │     Check: waypoints_queue empty AND waypoint_buffer <= 2
  │     │
  │     ├─→ YES: Reset to global route
  │     │     │
  │     │     ├─→ Reset flags (overtake_allowed, lane_change_allowed)
  │     │     └─→ Call set_destination() with original end_waypoint
  │     │
  │     └─→ NO: Continue
  │
  ├─→ Update overtake permission based on intersection
  │     ├─→ If intersection: overtake_allowed = False
  │     └─→ Else: overtake_allowed = True (if originally allowed)
  │
  ┌─────────────────────────────────────────────────────────┐
  │ STEP 3: Path Generation                                  │
  │ File: opencda/core/plan/local_planner_behavior.py        │
  │ Method: LocalPlanner.generate_path()                     │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ Filter problematic waypoints
  │     Method: buffer_filter()
  │     │
  │     ├─→ Remove waypoints behind vehicle (angle > 90°)
  │     └─→ Remove waypoints too close during lane changes (< 4.5m)
  │
  ├─→ Get current vehicle state
  │     ├─→ current_location = _ego_pos.location
  │     ├─→ current_yaw = _ego_pos.rotation.yaw
  │     └─→ current_wpt = _map.get_waypoint(current_location).next(1)[0]
  │
  ├─→ Detect lane changes
  │     ├─→ Get future and past waypoints
  │     ├─→ Calculate lateral offset
  │     │     Method: cal_distance_angle()
  │     │
  │     ├─→ Check lane_id_change
  │     │     Compare: future_wpt.lane_id != current_wpt.lane_id
  │     │
  │     └─→ Check lane_lateral_change
  │           Compare: lateral_diff > vehicle_width
  │
  ├─→ Collect waypoints for spline
  │     ├─→ Add history waypoints (if applicable)
  │     ├─→ Add current position or waypoint
  │     └─→ Add future waypoints from buffer (filter duplicates)
  │
  ├─→ Create spline
  │     File: opencda/core/plan/spline.py
  │     Method: Spline2D(x, y)
  │     │
  │     ├─→ Create 2D cubic spline from waypoint coordinates
  │     └─→ Spline internally creates sx and sy (1D splines)
  │
  └─→ Interpolate smooth path
        │
        ├─→ Generate arc length values (s) from current to end
        │     Step: ds = 0.1 meters
        │
        ├─→ For each s:
        │     │
        │     ├─→ Calculate position: (x, y) = sp.calc_position(s)
        │     ├─→ Calculate yaw: ryaw = sp.calc_yaw(s)
        │     └─→ Calculate curvature: rk = sp.calc_curvature(s)
        │           (Clamped to [-0.2, 0.2])
        │
        └─→ Return: rx, ry, rk, ryaw

  ┌─────────────────────────────────────────────────────────┐
  │ STEP 4: Lane Change Permission Check                     │
  │ File: opencda/core/plan/behavior_agent.py                │
  │ Method: check_lane_change_permission()                   │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ Check curvature constraint
  │     If mean(|rk|) > 0.04: lane_change_allowed = False
  │
  ├─→ Check if lane change is enabled
  │     Conditions:
  │     ├─→ collision_detector_enabled = True
  │     ├─→ LocalPlanner.lane_id_change = True
  │     ├─→ LocalPlanner.lane_lateral_change = True
  │     ├─→ overtake_counter <= 0
  │     └─→ destination_push_flag = 0
  │
  └─→ If enabled: Check adjacent lane safety
        Method: lane_change_management()
        │
        ├─→ Find target waypoint in adjacent lane
        ├─→ Generate adjacent lane path
        │     File: opencda/core/plan/collision_check.py
        │     Method: adjacent_lane_collision_check()
        │
        └─→ Collision check on adjacent lane
              Method: collision_manager(adjacent_check=True)

  ┌─────────────────────────────────────────────────────────┐
  │ STEP 5: Collision Detection                              │
  │ File: opencda/core/plan/behavior_agent.py                │
  │ Method: collision_manager()                               │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ Initialize: vehicle_state = False, min_distance = 100000
  │
  ├─→ For each obstacle vehicle:
  │     │
  │     ├─→ Check collision on planned path
  │     │     File: opencda/core/plan/collision_check.py
  │     │     Method: collision_circle_check()
  │     │     │
  │     │     ├─→ Calculate check distance
  │     │     │     Formula: min(max(time_ahead * speed / 0.1, 90), len(path))
  │     │     │
  │     │     ├─→ For sampled points along path (every 1m):
  │     │     │     │
  │     │     │     ├─→ Place collision circles
  │     │     │     │     Positions: [-1.0, 0, 1.0] meters from path point
  │     │     │     │
  │     │     │     ├─→ Get obstacle bounding box
  │     │     │     │
  │     │     │     └─→ Check if circles intersect bounding box
  │     │     │           Method: spatial.distance.cdist()
  │     │     │
  │     │     └─→ Return: collision_free (boolean)
  │     │
  │     └─→ If collision detected:
  │           ├─→ vehicle_state = True
  │           ├─→ Calculate distance (subtract 3m for vehicle length)
  │           └─→ Track closest obstacle
  │
  └─→ Return: (is_hazard, target_vehicle, min_distance)

  ┌─────────────────────────────────────────────────────────┐
  │ STEP 6: Decision Making                                  │
  │ File: opencda/core/plan/behavior_agent.py                │
  │ Method: BehaviorAgent.run_step() (continued)             │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ DECISION POINT 4: Lane Change Blocked?
  │     Conditions:
  │     ├─→ lane_change_allowed = False
  │     ├─→ potential_curved_road = True
  │     ├─→ destination_push_flag = 0
  │     └─→ overtake_counter <= 0
  │     │
  │     ├─→ YES: Push Destination
  │     │     │
  │     │     ├─→ Get push destination
  │     │     │     Method: get_push_destination()
  │     │     │     │
  │     │     │     ├─→ If intersection: Use waypoint from buffer
  │     │     │     └─→ Else: Use ego waypoint advanced by 3×speed
  │     │     │
  │     │     ├─→ Set destination_push_flag = 90
  │     │     ├─→ Set new temporary destination
  │     │     │     Method: set_destination(clean=True, end_reset=False)
  │     │     │
  │     │     └─→ Regenerate path
  │     │           Method: LocalPlanner.generate_path()
  │     │
  │     └─→ NO: Continue to next decision
  │
  ├─→ DECISION POINT 5: Hazard Detected?
  │     │
  │     ├─→ NO: Go to Normal Driving
  │     │
  │     └─→ YES: Check overtake conditions
  │           │
  │           ├─→ DECISION POINT 6: Overtake Allowed?
  │           │     Conditions:
  │           │     ├─→ overtake_allowed = True
  │           │     ├─→ overtake_counter <= 0
  │           │     └─→ NOT potential_curved_road
  │           │     │
  │           │     ├─→ NO: Go to Car Following
  │           │     │
  │           │     └─→ YES: Check same lane and speed
  │           │           │
  │           │           ├─→ If ego_lane_id == obstacle_lane_id:
  │           │           │     │
  │           │           │     └─→ If ego_speed >= obstacle_speed - 5:
  │           │           │           │
  │           │           │           └─→ Attempt Overtake
  │           │           │                 Method: overtake_management()
  │           │           │                 │
  │           │           │                 ├─→ Check left lane availability
  │           │           │                 │     Method: adjacent_lane_collision_check()
  │           │           │                 │
  │           │           │                 ├─→ Collision check on left lane
  │           │           │                 │
  │           │           │                 ├─→ If safe: Plan route through left lane
  │           │           │                 │     Method: set_destination()
  │           │           │                 │     Set: overtake_counter = 100
  │           │           │                 │
  │           │           │                 └─→ If left fails: Try right lane
  │           │           │
  │           │           └─→ Else: Go to Car Following
  │           │
  │           └─→ If overtake not attempted or failed:
  │                 └─→ Go to Car Following

  ┌─────────────────────────────────────────────────────────┐
  │ STEP 7: Behavior Execution                               │
  │ File: opencda/core/plan/behavior_agent.py                │
  │ Method: BehaviorAgent.run_step() (continued)             │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ DECISION POINT 7: Car Following Mode?
  │     │
  │     ├─→ YES: Car Following
  │     │     │
  │     │     ├─→ Check emergency stop
  │     │     │     If distance < max(break_distance, 3):
  │     │     │       Return (0, None) → Emergency stop
  │     │     │
  │     │     ├─→ Calculate following speed
  │     │     │     Method: car_following_manager()
  │     │     │     │
  │     │     │     ├─→ Calculate TTC (Time To Collision)
  │     │     │     │     Formula: distance / delta_v
  │     │     │     │
  │     │     │     ├─→ If 0 < TTC < safety_time:
  │     │     │     │     └─→ target_speed = min(lead_speed - speed_decrease, target)
  │     │     │     │
  │     │     │     └─→ Else:
  │     │     │           └─→ target_speed = min(lead_speed + 1, target)
  │     │     │
  │     │     └─→ Generate trajectory with following speed
  │     │           Method: LocalPlanner.run_step()
  │     │
  │     └─→ NO: Normal Driving
  │           │
  │           └─→ Generate trajectory with max speed
  │                 Method: LocalPlanner.run_step()
  │                 target_speed = max_speed - speed_lim_dist

  ┌─────────────────────────────────────────────────────────┐
  │ STEP 8: Trajectory Generation                            │
  │ File: opencda/core/plan/local_planner_behavior.py        │
  │ Method: LocalPlanner.run_step()                          │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ Refill waypoint buffer if needed
  │     If len(_waypoint_buffer) < waypoint_update_freq:
  │       └─→ Pop waypoints from waypoints_queue
  │
  ├─→ Generate trajectory if needed
  │     Conditions:
  │     ├─→ No pre-generated trajectory
  │     ├─→ trajectory_buffer length < trajectory_update_freq
  │     └─→ NOT in following mode (for platooning)
  │     │
  │     └─→ If conditions met:
  │           Method: generate_trajectory(rx, ry, rk)
  │           │
  │           ├─→ Calculate curvature-based speed limit
  │           │     Formula: min(target_speed, sqrt(5.0 / mean_k) * 3.6)
  │           │
  │           ├─→ Calculate acceleration
  │           │     Formula: clamp((target_speed - current_speed) / dt, -6.5, 3.5)
  │           │
  │           ├─→ Sample trajectory for 2 seconds
  │           │     For each time step (dt = 0.1s):
  │           │     │
  │           │     ├─→ Update distance: s += v*dt + 0.5*a*dt²
  │           │     ├─→ Update speed: v += a*dt
  │           │     ├─→ Map distance to path index
  │           │     │     index = int(s / 0.1 - 1)
  │           │     │
  │           │     └─→ Add trajectory point
  │           │           (Transform(x, y, z), target_speed)
  │           │
  │           └─→ Store in _trajectory_buffer
  │
  ├─→ Get target waypoint
  │     From _trajectory_buffer[min(1, len-1)]
  │     (Prefer index 1 for slight lookahead)
  │
  ├─→ Clean passed waypoints
  │     Method: pop_buffer(vehicle_transform)
  │     │
  │     ├─→ Remove waypoints within min_distance
  │     └─→ Move to history_buffer (if distance > 4.5m)
  │
  └─→ Return: (target_speed, target_location)

  ┌─────────────────────────────────────────────────────────┐
  │ STEP 9: Control Command Generation                       │
  │ File: opencda/core/actuation/control_manager.py          │
  │ Method: ControlManager.run_step()                        │
  └─────────────────────────────────────────────────────────┘
  │
  ├─→ Update controller state
  │     File: opencda/core/actuation/pid_controller.py
  │     Method: Controller.update_info(ego_pos, ego_speed)
  │
  ├─→ Generate control command
  │     Method: Controller.run_step(target_speed, target_location)
  │     │
  │     ├─→ Emergency stop check
  │     │     If target_speed == 0 or target_location is None:
  │     │       └─→ Return: brake = 1.0, throttle = 0.0
  │     │
  │     ├─→ Longitudinal control (speed)
  │     │     Method: lon_run_step(target_speed)
  │     │     │
  │     │     ├─→ Calculate speed error
  │     │     ├─→ PID control: acceleration = Kp*e + Kd*de + Ki*ie
  │     │     └─→ Return: acceleration
  │     │
  │     ├─→ Lateral control (steering)
  │     │     Method: lat_run_step(target_location)
  │     │     │
  │     │     ├─→ Calculate cross-track error
  │     │     ├─→ Calculate heading error
  │     │     ├─→ PID control: steering = Kp*e + Kd*de + Ki*ie
  │     │     └─→ Return: steering angle
  │     │
  │     └─→ Generate VehicleControl
  │           │
  │           ├─→ If acceleration >= 0:
  │           │     ├─→ throttle = min(acceleration, max_throttle)
  │           │     └─→ brake = 0.0
  │           │
  │           ├─→ Else:
  │           │     ├─→ throttle = 0.0
  │           │     └─→ brake = min(abs(acceleration), max_brake)
  │           │
  │           ├─→ Apply steering smoothing
  │           │     Limit steering change to ±0.2 per step
  │           │
  │           └─→ Return: carla.VehicleControl
  │
  └─→ Apply control to vehicle
        Method: vehicle.apply_control(control)

END: Control applied, vehicle moves
  ↓
(Repeat loop for next simulation step)
```

---

## Complete Decision Tree

```
START: run_step()
  ↓
  ├─→ Destination reached? → YES → Exit simulation
  │                         NO  ↓
  │
  ├─→ Red light/Stop sign? → YES → Return (0, None) → Stop
  │                          NO  ↓
  │
  ├─→ Route finished? → YES → Reset to global route
  │                   NO  ↓
  │
  ├─→ Generate path (always)
  │     ↓
  │
  ├─→ Check lane change permission
  │     ↓
  │
  ├─→ Collision check
  │     ↓
  │
  ├─→ Lane change blocked? → YES → Push destination → Regenerate path
  │                          NO  ↓
  │
  ├─→ Hazard detected? → NO → Normal driving
  │                    YES ↓
  │
  ├─→ Overtake allowed? → NO → Car following
  │                      YES ↓
  │
  ├─→ Same lane & speed OK? → NO → Car following
  │                            YES ↓
  │
  ├─→ Overtake safe? → YES → Execute overtake → New route
  │                   NO  ↓
  │
  └─→ Car following
        ↓
        ├─→ Too close? → YES → Emergency stop
        │                NO  ↓
        │
        └─→ Calculate following speed
              ↓
              └─→ Generate trajectory
                    ↓
                    └─→ Return (target_speed, target_location)
```

---

## File Reference Map

### Core Planning Files

| File | Purpose | Key Methods |
|------|---------|-------------|
| `behavior_agent.py` | Main decision maker | `run_step()`, `collision_manager()`, `overtake_management()`, `car_following_manager()` |
| `local_planner_behavior.py` | Path smoothing & trajectory | `generate_path()`, `generate_trajectory()`, `run_step()` |
| `global_route_planner.py` | Route planning | `trace_route()`, `_path_search()`, `_turn_decision()` |
| `collision_check.py` | Collision detection | `collision_circle_check()`, `adjacent_lane_collision_check()` |
| `spline.py` | Spline interpolation | `Spline2D()`, `calc_position()`, `calc_curvature()` |

### Supporting Files

| File | Purpose | Key Methods |
|------|---------|-------------|
| `global_route_planner_dao.py` | Map data access | `get_topology()`, `get_waypoint()` |
| `planer_debug_helper.py` | Debug visualization | `update()`, plotting methods |
| `control_manager.py` | Control interface | `run_step()` |
| `pid_controller.py` | PID control | `lon_run_step()`, `lat_run_step()` |

### Integration Files

| File | Purpose | Key Methods |
|------|---------|-------------|
| `vehicle_manager.py` | System coordinator | `run_step()`, `update_info()` |
| `misc.py` | Utility functions | `cal_distance_angle()`, `distance_vehicle()` |

---

## Data Flow Summary

```
Perception Data → BehaviorAgent.update_information()
  ↓
  ├─→ Filter obstacles (white list)
  ├─→ Update ego state
  └─→ Update LocalPlanner state
        ↓
        BehaviorAgent.run_step()
          ↓
          ├─→ LocalPlanner.generate_path()
          │     ↓
          │     ├─→ Filter waypoints
          │     ├─→ Detect lane changes
          │     ├─→ Create spline
          │     └─→ Interpolate path
          │           Returns: rx, ry, rk, ryaw
          │
          ├─→ CollisionChecker.collision_circle_check()
          │     ↓
          │     Returns: collision_free
          │
          ├─→ Decision making
          │     ↓
          │     ├─→ Push destination (if needed)
          │     ├─→ Overtake (if safe)
          │     ├─→ Car following (if hazard)
          │     └─→ Normal driving (else)
          │
          └─→ LocalPlanner.run_step()
                ↓
                ├─→ Generate trajectory
                │     ├─→ Sample path with time
                │     ├─→ Assign speeds
                │     └─→ Apply curvature limits
                │
                └─→ Return: (target_speed, target_location)
                      ↓
                      ControlManager.run_step()
                        ↓
                        ├─→ PID Controller
                        │     ├─→ Longitudinal control (speed)
                        │     └─→ Lateral control (steering)
                        │
                        └─→ Return: carla.VehicleControl
                              ↓
                              Vehicle applies control
```

---

## Key Decision Points Summary

1. **Destination Check**: `is_close_to_destination()` → Exit if reached
2. **Traffic Light**: `traffic_light_manager()` → Stop if red
3. **Route Reset**: Check if temporary route finished → Reset to global
4. **Lane Change Permission**: `check_lane_change_permission()` → Multiple conditions
5. **Collision Detection**: `collision_manager()` → Detect hazards
6. **Lane Change Blocked**: Push destination if blocked
7. **Hazard Response**: Overtake vs Car Following decision
8. **Emergency Stop**: Distance check in car following
9. **Curvature Speed Limit**: Applied in trajectory generation

---

## Timing and Frequency

- **Every Simulation Step** (typically 20-30 Hz):
  - `update_information()` - Update state
  - `run_step()` - Main planning loop
  - `generate_path()` - Smooth path generation
  - `collision_manager()` - Collision checking
  - `run_step()` (LocalPlanner) - Trajectory generation
  - `run_step()` (Controller) - Control command

- **On Route Change**:
  - `set_destination()` - Route planning
  - `_trace_route()` - Global route generation
  - `set_global_plan()` - Update waypoints

- **Periodic Updates**:
  - Trajectory buffer: Regenerated when < `trajectory_update_freq`
  - Waypoint buffer: Refilled when < `waypoint_update_freq`
  - Counters: Decremented every step (overtake_counter, push_flag)

---

This flowchart provides a complete view of the planning module, showing every step, decision point, and the files where each operation occurs.

