```markdown
# Explanation: `behavior_agent.py`

## Overview

`BehaviorAgent` is the brains of OpenCDA’s planning stack. It:
- Tracks ego pose/speed from localization.
- Owns the global planner, local planner, and collision checker.
- Decides when to stop, follow, overtake, push destinations, or drive normally.

To mirror your preferred style, every method now contains:
1. **Purpose** – natural-language summary.
2. **Code with Comments** – snippet annotated with data types and small examples.
3. **Logic Explanation** – why the code is structured this way.
4. **Flowchart** – step-by-step view.

---

## Class Outline

```python
BehaviorAgent
├── __init__()
├── update_information()
├── add_white_list()
├── white_list_match()
├── set_destination()
├── get_local_planner()
├── reroute()
├── _trace_route()
├── traffic_light_manager()
├── collision_manager()
├── overtake_management()
├── lane_change_management()
├── car_following_manager()
├── is_intersection()
├── is_close_to_destination()
├── check_lane_change_permission()
├── get_push_destination()
└── run_step()
```

---

## Method 1: `__init__()`

### Purpose
Attach to the CARLA vehicle/map, load all behavior configuration, and construct helper modules (global planner, local planner, collision checker, debugging utilities).

### Code with Comments

```python
class BehaviorAgent(object):
    def __init__(self, vehicle, carla_map, config_yaml):
        self.vehicle = vehicle
        # CURRENT EGO STATE ----------------------------------------------------
        self._ego_pos = None          # carla.Transform, filled by update_information()
        self._ego_speed = 0.0         # float km/h
        self._map = carla_map         # carla.Map for waypoint queries

        # SPEED / SAFETY PARAMS (all floats from YAML) -------------------------
        self.max_speed = config_yaml['max_speed']                # e.g., 70 km/h
        self.tailgate_speed = config_yaml['tailgate_speed']      # fallback speed
        self.speed_lim_dist = config_yaml['speed_lim_dist']      # reduce near limit
        self.speed_decrease = config_yaml['speed_decrease']      # braking delta
        self.safety_time = config_yaml['safety_time']            # seconds
        self.emergency_param = config_yaml['emergency_param']    # braking scaler
        self.break_distance = 0                                  # meters, updated later
        self.ttc = 1000                                          # default Time-To-Collision

        # HELPERS --------------------------------------------------------------
        self._collision_check = CollisionChecker(
            time_ahead=config_yaml['collision_time_ahead'])
        self._local_planner = LocalPlanner(
            self, carla_map, config_yaml['local_planner'])

        # GLOBAL PLAN STATE ----------------------------------------------------
        self._global_planner = None
        self.start_waypoint = None
        self.end_waypoint = None
        self._sampling_resolution = config_yaml['sample_resolution']  # meters

        # BEHAVIOR FLAGS -------------------------------------------------------
        self.ignore_traffic_light = config_yaml['ignore_traffic_light']
        self.overtake_allowed = config_yaml['overtake_allowed']
        self.overtake_allowed_origin = self.overtake_allowed    # remember default
        self.overtake_counter = 0                              # cooldown frames
        self.hazard_flag = False
        self.light_state = "Red"
        self.light_id_to_ignore = -1
        self.stop_sign_wait_count = 0
        self.car_following_flag = False
        self.lane_change_allowed = True
        self.destination_push_flag = 0

        # COLLABORATION STRUCTURES ---------------------------------------------
        self.white_list = []               # list[VehicleManager]
        self.obstacle_vehicles = []        # filtered obstacles
        self.objects = {}                  # latest perception dict

        # DEBUG ----------------------------------------------------------------
        self.debug_helper = PlanDebugHelper(self.vehicle.id)
        self.debug = config_yaml.get('debug', False)
        self.initial_global_route = None
```

### Logic Explanation
- Keep everything strongly typed (Transforms, km/h floats) to avoid accidental mixing of units.
- Store both current behavior flags and “origin” values so temporary overrides (e.g., in intersections) can be reset easily.
- Lazily instantiate the heavy global planner; only build the local planner and collision checker immediately because they are lightweight and always needed.

### Flowchart

```
Load vehicle/map references
  ↓
Read speed & safety config
  ↓
Instantiate CollisionChecker + LocalPlanner
  ↓
Initialize planning state (global, local, flags)
  ↓
Prepare collaboration + debug helpers
  ↓
Done (agent ready)
```

---

## Method 2: `update_information()`

### Purpose
Refresh ego pose/speed from localization, pass them to the local planner, filter perceived obstacles, and update traffic-light state.

### Code with Comments

```python
def update_information(self, ego_pos, ego_speed, objects):
    self._ego_speed = ego_speed        # float km/h
    self._ego_pos = ego_pos            # carla.Transform
    self.break_distance = self._ego_speed / 3.6 * self.emergency_param
    # Example: ego_speed = 54 km/h (15 m/s), emergency_param = 2.5 ⇒ break_distance ≈ 37.5 m

    self.get_local_planner().update_information(ego_pos, ego_speed)

    self.objects = objects             # dict with keys like 'vehicles', 'traffic_lights'
    obstacle_vehicles = objects['vehicles']
    self.obstacle_vehicles = self.white_list_match(obstacle_vehicles)
    # Removes platoon teammates by comparing white-list entries

    self.debug_helper.update(ego_speed, self.ttc)

    self.light_state = "Green" if self.ignore_traffic_light \
        else str(self.vehicle.get_traffic_light_state())
```

### Logic Explanation
- Braking distance is recalculated every frame so downstream modules (car following) know how close is too close.
- Passing state to LocalPlanner keeps spline generation aligned with the latest pose; otherwise the smooth path would start from an old location.
- Filtering obstacles via `white_list_match` avoids braking for fellow platoon members that we plan to merge with.

### Flowchart

```
Store ego_pos / ego_speed
  ↓
Compute braking distance
  ↓
Notify LocalPlanner
  ↓
Filter obstacles against white list
  ↓
Update debug helper + traffic lights
  ↓
Done
```

---

## Method 3: `add_white_list()`

### Purpose
Add a vehicle manager to the “ignore” list (platoon teammates).

### Code with Comments

```python
def add_white_list(self, vm):
    self.white_list.append(vm)   # vm: VehicleManager with v2x_manager
```

### Logic Explanation
Teammates are added once and can later be removed from obstacle lists. No extra logic is required.

### Flowchart

```
Append VehicleManager to white_list
```

---

## Method 4: `white_list_match()`

### Purpose
Remove known teammates from obstacle detections so we don’t brake for partners.

### Code with Comments

```python
def white_list_match(self, obstacles):
    new_obstacle_list = []
    for o in obstacles:
        flag = False
        o_waypoint = self._map.get_waypoint(o.get_location())
        for vm in self.white_list:
            pos = vm.v2x_manager.get_ego_pos()
            w_waypoint = self._map.get_waypoint(pos.location)

            if o_waypoint.lane_id != w_waypoint.lane_id:
                continue  # different lane ⇒ cannot be same car

            # POSITION THRESHOLD: 3 meters in X/Y (approx half a car length)
            if abs(pos.location.x - o.get_location().x) <= 3.0 and \
               abs(pos.location.y - o.get_location().y) <= 3.0:
                flag = True
                break
        if not flag:
            new_obstacle_list.append(o)
    return new_obstacle_list
```

### Logic Explanation
- Narrow lane ID comparison short-circuits most checks.
- The ±3 m window covers GPS noise and ensures that only vehicles physically close AND in the same lane get filtered out.

### Flowchart

```
For each obstacle:
  ├─ Compare lane_id with each white-list member
  ├─ If same lane and within 3 m ⇒ flag as teammate
  └─ If not flagged ⇒ keep
Return filtered list
```

---

## Method 5: `set_destination()`

### Purpose
Compute a new route from `start_location` to `end_location`, refreshing buffers when requested.

### Code with Comments

```python
def set_destination(self, start_location, end_location,
                    clean=False, end_reset=True, clean_history=False):
    if clean:
        self.get_local_planner().get_waypoints_queue().clear()
        self.get_local_planner().get_trajectory().clear()
        self.get_local_planner().get_waypoint_buffer().clear()
    if clean_history:
        self.get_local_planner().get_history_buffer().clear()

    self.start_waypoint = self._map.get_waypoint(start_location)

    # Ensure start waypoint lies ahead of the ego vehicle
    if self._ego_pos:
        cur_loc = self._ego_pos.location
        cur_yaw = self._ego_pos.rotation.yaw
        _, angle = cal_distance_angle(
            self.start_waypoint.transform.location, cur_loc, cur_yaw)
        while angle > 90:
            self.start_waypoint = self.start_waypoint.next(1)[0]
            _, angle = cal_distance_angle(
                self.start_waypoint.transform.location, cur_loc, cur_yaw)

    end_waypoint = self._map.get_waypoint(end_location)
    if end_reset:
        self.end_waypoint = end_waypoint

    route_trace = self._trace_route(self.start_waypoint, end_waypoint)
    if self.initial_global_route is None:
        self.initial_global_route = route_trace

    self._local_planner.set_global_plan(route_trace, clean)
```

### Logic Explanation
- Cleaning buffers lets us completely swap routes (e.g., after reroute).
- The “angle > 90°” loop nudges the starting waypoint forward until it’s in front of the ego car, preventing backward paths.
- Storing `initial_global_route` keeps a copy for debugging.

### Flowchart

```
Optional: clear buffers/history
  ↓
Convert start/end locations to waypoints
  ↓
If ego pose known:
  └─ Advance start waypoint until in front of ego
  ↓
Call _trace_route() to get list of (Waypoint, RoadOption)
  ↓
Store as initial route if first time
  ↓
Pass plan to LocalPlanner (clean flag matches our own)
```

---

## Method 6: `get_local_planner()`

### Purpose
Expose the LocalPlanner instance (used by BehaviorAgent internals and other modules).

### Code with Comments

```python
def get_local_planner(self):
    return self._local_planner
```

### Logic Explanation
Simple accessor. Keeps encapsulation while letting other methods call LocalPlanner methods.

### Flowchart

```
Return _local_planner reference
```

---

## Method 7: `reroute()`

### Purpose
When the current destination is nearly reached, pick a new random destination so the simulation keeps going.

### Code with Comments

```python
def reroute(self, spawn_points):
    if self.debug:
        print("Target almost reached, setting new destination...")
    random.shuffle(spawn_points)
    new_start = self._local_planner.waypoints_queue[-1][0].transform.location
    destination = spawn_points[0].location if \
        spawn_points[0].location != new_start else spawn_points[1].location
    if self.debug:
        print("New destination:", destination)
    self.set_destination(new_start, destination)
```

### Logic Explanation
- Uses the last waypoint from the current queue as a “handoff” point to avoid jumps.
- Ensures the new destination differs from the start (fall back to second spawn point if necessary).

### Flowchart

```
Shuffle spawn points
  ↓
Pick last planned waypoint as new start
  ↓
Choose random destination ≠ start
  ↓
Call set_destination()
```

---

## Method 8: `_trace_route()`

### Purpose
Lazy-initialize the global planner and ask it for a waypoint+RoadOption route between two waypoints.

### Code with Comments

```python
def _trace_route(self, start_waypoint, end_waypoint):
    if self._global_planner is None:
        wld = self.vehicle.get_world()
        dao = GlobalRoutePlannerDAO(
            wld.get_map(), sampling_resolution=self._sampling_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        self._global_planner = grp

    route = self._global_planner.trace_route(
        start_waypoint.transform.location,
        end_waypoint.transform.location)
    return route      # list[(carla.Waypoint, RoadOption)]
```

### Logic Explanation
Global planner construction is expensive; the lazy check avoids recreating it. After creation, `trace_route` is just a method call.

### Flowchart

```
If _global_planner is None:
  ├─ Build DAO with map + sampling resolution
  ├─ Create planner and call setup()
  └─ Store in _global_planner
Call trace_route(start, end)
  ↓
Return list of edges
```

---

## Method 9: `traffic_light_manager()`

### Purpose
Decide whether to stop for a red light or stop sign, with a cooldown so we don’t stop twice for the same intersection.

### Code with Comments

```python
def traffic_light_manager(self, waypoint):
    light_id = self.vehicle.get_traffic_light().id \
        if self.vehicle.get_traffic_light() is not None else -1

    # Handle stop-sign waiting window (2 seconds @30 FPS ≈ 60 frames)
    if 60 <= self.stop_sign_wait_count < 240:
        self.stop_sign_wait_count += 1
    elif self.stop_sign_wait_count >= 240:
        self.stop_sign_wait_count = 0

    if self.light_state == "Red":
        if light_id == -1:                      # stop sign
            if self.stop_sign_wait_count < 60:
                self.stop_sign_wait_count += 1
                return 1                        # keep stopping
            else:
                return 0                        # done waiting

        if not waypoint.is_junction and (
                self.light_id_to_ignore != light_id or light_id == -1):
            return 1                            # red light outside junction
        elif waypoint.is_junction and light_id != -1:
            self.light_id_to_ignore = light_id  # ignore same semaphore going forward

    if self.light_id_to_ignore != light_id:
        self.light_id_to_ignore = -1
    return 0
```

### Logic Explanation
- `light_id_to_ignore` prevents braking again for the same traffic light right after passing it.
- The stop-sign counter enforces a 2-second stop even if CARLA flips the light instantly.

### Flowchart

```
Read traffic light ID (-1 if stop sign)
  ↓
Update stop-sign wait counter
  ↓
If light_state == Red:
  ├─ If stop sign:
  │    ├─ Wait < 60 ⇒ increment & return 1
  │    └─ Else return 0
  ├─ If not in junction and new red light ⇒ return 1
  └─ If in junction ⇒ remember light_id
Reset light_id_to_ignore if new light encountered
  ↓
Return 0
```

---

## Method 10: `collision_manager()`

### Purpose
Scan the planned path for obstacle vehicles, returning whether a collision is predicted and which vehicle is closest.

### Code with Comments

```python
def collision_manager(self, rx, ry, ryaw, waypoint, adjacent_check=False):
    def dist(v):
        return v.get_location().distance(waypoint.transform.location)

    vehicle_state = False
    min_distance = 100000.0
    target_vehicle = None

    for vehicle in self.obstacle_vehicles:
        collision_free = self._collision_check.collision_circle_check(
            rx, ry, ryaw, vehicle, self._ego_speed / 3.6, self._map,
            adjacent_check=adjacent_check)
        if not collision_free:
            vehicle_state = True
            distance = positive(dist(vehicle) - 3)   # subtract 3 m ≈ car length
            if distance < min_distance:
                min_distance = distance
                target_vehicle = vehicle

    return vehicle_state, target_vehicle, min_distance
```

### Logic Explanation
- Uses the smooth path (`rx`, `ry`, `ryaw`) and multiple circles along it to approximate the ego footprint.
- Subtracting 3 m from distance gives bumper-to-bumper clearance, making it easier to decide when to brake.

### Flowchart

```
min_distance = large
vehicle_state = False
target_vehicle = None
  ↓
For each perceived obstacle:
  ├─ Run collision_circle_check()
  ├─ If collision predicted:
  │    ├─ vehicle_state = True
  │    ├─ distance = dist(vehicle) - 3
  │    └─ Update closest target_vehicle
Return (vehicle_state, target_vehicle, min_distance)
```

---

## Method 11: `overtake_management()`

### Purpose
Attempt to change into the left or right adjacent lane to pass a slower vehicle if it is safe to do so.

### Code with Comments

```python
def overtake_management(self, obstacle_vehicle):
    obstacle_vehicle_wpt = self._map.get_waypoint(obstacle_vehicle.get_location())
    left_wpt = obstacle_vehicle_wpt.get_left_lane()
    right_wpt = obstacle_vehicle_wpt.get_right_lane()
    left_turn = obstacle_vehicle_wpt.left_lane_marking.lane_change
    right_turn = obstacle_vehicle_wpt.right_lane_marking.lane_change

    def try_side(target_wpt, lane_change_flag):
        if not target_wpt:
            return True
        if lane_change_flag not in (carla.LaneChange.Left, carla.LaneChange.Both,
                                    carla.LaneChange.Right):
            return True
        if obstacle_vehicle_wpt.lane_id * target_wpt.lane_id <= 0:
            return True
        if target_wpt.lane_type != carla.LaneType.Driving:
            return True

        rx, ry, ryaw = self._collision_check.adjacent_lane_collision_check(
            ego_loc=self._ego_pos.location,
            target_wpt=target_wpt,
            carla_map=self._map,
            overtake=True,
            world=self.vehicle.get_world())
        vehicle_state, _, _ = self.collision_manager(
            rx, ry, ryaw, self._map.get_waypoint(self._ego_pos.location), True)
        if vehicle_state:
            return True   # unsafe ⇒ treat as “keep following”

        self.overtake_counter = 100                      # ~3 seconds cooldown
        next_wpt_list = target_wpt.next(self._ego_speed / 3.6 * 6)
        if not next_wpt_list:
            return True  # nothing to go back to

        next_wpt = next_wpt_list[0]
        target_wpt = target_wpt.next(5)[0]
        self.set_destination(
            target_wpt.transform.location,
            next_wpt.transform.location,
            clean=True,
            end_reset=False)
        return False     # False signals “overtake succeeded”

    if try_side(left_wpt, left_turn) is False:
        print("left overtake is operated")
        return False
    if try_side(right_wpt, right_turn) is False:
        print("right overtake is operated")
        return False
    return True
```

### Logic Explanation
- Factorized into `try_side` to share checks between left/right.
- Cooldown counter prevents immediate repeated overtakes.
- Uses `set_destination()` with a short left/right trajectory so the local planner handles the maneuver naturally.

### Flowchart

```
Check left side (lane marking, lane type, collision-free path)
  ├─ If safe: set short detour, start cooldown, return False
Check right side similarly
  └─ Otherwise return True (no overtake performed)
```

---

## Method 12: `lane_change_management()`

### Purpose
Before executing a planned lane change, ensure the adjacent lane path is free of collisions.

### Code with Comments

```python
def lane_change_management(self):
    ego_wpt = self._map.get_waypoint(self._ego_pos.location)
    ego_lane_id = ego_wpt.lane_id
    target_wpt = None
    for wpt, _ in self.get_local_planner().get_waypoint_buffer():
        if wpt.lane_id != ego_lane_id:
            target_wpt = wpt
            break
    if not target_wpt:
        return True      # no lane change planned

    rx, ry, ryaw = self._collision_check.adjacent_lane_collision_check(
        ego_loc=self._ego_pos.location,
        target_wpt=target_wpt,
        overtake=False,
        carla_map=self._map,
        world=self.vehicle.get_world())
    vehicle_state, _, _ = self.collision_manager(
        rx, ry, ryaw, self._map.get_waypoint(
            self._ego_pos.location), adjacent_check=True)
    return not vehicle_state
```

### Logic Explanation
This is the “are we sure?” gate before lane change: LocalPlanner might propose one, but we only allow it if the adjacent lane is currently safe.

### Flowchart

```
Find next waypoint with different lane_id
  ↓
If none ⇒ return True
  ↓
Generate adjacent-lane path
  ↓
Run collision_manager with adjacent_check=True
  ↓
Return True if no hazard, else False
```

---

## Method 13: `car_following_manager()`

### Purpose
Given the lead vehicle and distance, compute a safe target speed for the ego car.

### Code with Comments

```python
def car_following_manager(self, vehicle, distance, target_speed=None):
    if not target_speed:
        target_speed = self.max_speed - self.speed_lim_dist

    vehicle_speed = get_speed(vehicle)               # km/h
    delta_v = max(1, (self._ego_speed - vehicle_speed) / 3.6)  # m/s
    ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
    self.ttc = ttc

    if 0.0 < ttc < self.safety_time:
        target_speed = min(positive(vehicle_speed - self.speed_decrease),
                           target_speed)
    else:
        target_speed = 0 if vehicle_speed == 0 else \
            min(vehicle_speed + 1, target_speed)
    return target_speed
```

### Logic Explanation
- Time-to-collision (TTC) is distance divided by relative speed.
- If TTC is below the configured safety window, reduce speed aggressively; otherwise try to match the lead car’s speed (plus 1 km/h to close the gap gently).

### Flowchart

```
Compute delta_v and TTC
  ↓
If 0 < TTC < safety_time:
  └─ target_speed = lead_speed - speed_decrease
Else:
  └─ target_speed = min(lead_speed + 1, target_speed) (or 0 if lead stopped)
Return target_speed
```

---

## Method 14: `is_intersection()`

### Purpose
Detect whether any upcoming waypoint is within 20 m of a detected traffic light.

### Code with Comments

```python
def is_intersection(self, objects, waypoint_buffer):
    for tl in objects['traffic_lights']:
        for wpt, _ in waypoint_buffer:
            distance = tl.get_location().distance(wpt.transform.location)
            if distance < 20:
                return True
    return False
```

### Logic Explanation
Proximity to a traffic light is a cheaper proxy than using CARLA’s junction flag, and it keeps behavior consistent even when the HD map has small errors.

### Flowchart

```
For each traffic light:
  ├─ For each buffered waypoint:
  │    ├─ If distance < 20 m ⇒ return True
Return False
```

---

## Method 15: `is_close_to_destination()`

### Purpose
Check whether the ego car is within a 10 m × 10 m box around the destination.

### Code with Comments

```python
def is_close_to_destination(self):
    return abs(self._ego_pos.location.x - self.end_waypoint.transform.location.x) <= 10 and \
           abs(self._ego_pos.location.y - self.end_waypoint.transform.location.y) <= 10
```

### Logic Explanation
A rectangular check is cheap and good enough for simulation end detection.

### Flowchart

```
If |Δx| ≤ 10 AND |Δy| ≤ 10 ⇒ True else False
```

---

## Method 16: `check_lane_change_permission()`

### Purpose
Determine if lane changes should be allowed right now, considering curvature, collision detector availability, and other flags.

### Code with Comments

```python
def check_lane_change_permission(self, lane_change_allowed,
                                 collision_detector_enabled, rk):
    if len(rk) > 2 and np.mean(np.abs(np.array(rk))) > 0.04:
        lane_change_allowed = False    # heavy curvature ⇒ unsafe to change lanes

    lane_change_enabled_flag = collision_detector_enabled and \
                               self.get_local_planner().lane_id_change and \
                               self.get_local_planner().lane_lateral_change and \
                               self.overtake_counter <= 0 and \
                               not self.destination_push_flag
    if lane_change_enabled_flag:
        lane_change_allowed = lane_change_allowed and self.lane_change_management()
        if not lane_change_allowed:
            print("lane change not allowed")
    return lane_change_allowed
```

### Logic Explanation
- The curvature threshold (0.04 m⁻¹) roughly corresponds to a 25 m radius curve; lane changes on tighter curves are disallowed.
- Only when the LocalPlanner indicates a lane change and all other conditions are satisfied do we run the full collision-based lane-change management.

### Flowchart

```
If mean |rk| > 0.04 ⇒ lane_change_allowed = False
  ↓
lane_change_enabled_flag = collision detector ON AND local planner wants lane change AND no overtake cooldown AND no destination push
  ↓
If flag True:
  └─ lane_change_allowed &= lane_change_management()
Return lane_change_allowed
```

---

## Method 17: `get_push_destination()`

### Purpose
When a planned lane change can’t be executed (blocked), choose a temporary destination further along the current path to “push” the car forward safely.

### Code with Comments

```python
def get_push_destination(self, ego_vehicle_wp, is_intersection):
    waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
    reset_index = len(waypoint_buffer) // 2   # midway into buffer

    if is_intersection:
        reset_target = waypoint_buffer[reset_index][0].next(
            max(self._ego_speed / 3.6, 10.0))[0]
    else:
        reset_target = ego_vehicle_wp.next(
            max(self._ego_speed / 3.6 * 3, 10.0))[0]
    if self.debug:
        print(f"Vehicle id: {self.vehicle.id} destination pushed to {reset_target.transform.location}")
    return reset_target
```

### Logic Explanation
- In intersections, we look ahead using the buffered waypoint to stay in the correct lane (since lane IDs can change rapidly).
- Outside intersections, move 3 seconds ahead (speed * 3) but at least 10 m so we definitely clear the blockage.

### Flowchart

```
reset_index = len(buffer) // 2
  ↓
If intersection:
  └─ reset_target = future waypoint after max(speed, 10 m)
Else:
  └─ reset_target = ego waypoint advanced by max(3*speed, 10 m)
Return reset_target
```

---

## Method 18: `run_step()`

### Purpose
Execute one full decision cycle: traffic compliance, path generation, collision handling, lane decisions, and output of target speed/location for the controller.

### Code with Comments

```python
def run_step(self, target_speed=None,
             collision_detector_enabled=True,
             lane_change_allowed=True):
    ego_vehicle_loc = self._ego_pos.location
    ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
    waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
    self.ttc = 1000
    if self.overtake_counter > 0:
        self.overtake_counter -= 1
    if self.destination_push_flag > 0:
        self.destination_push_flag -= 1

    is_intersection = self.is_intersection(self.objects, waypoint_buffer)

    # 0. END CONDITION --------------------------------------------------------
    if self.is_close_to_destination():
        print('Simulation is Over')
        sys.exit(0)

    # 1. TRAFFIC LIGHT --------------------------------------------------------
    if self.traffic_light_manager(ego_vehicle_wp) != 0:
        return 0, None

    # 2. ROUTE RESET ----------------------------------------------------------
    if len(self.get_local_planner().get_waypoints_queue()) == 0 \
            and len(self.get_local_planner().get_waypoint_buffer()) <= 2:
        if self.debug:
            print('Destination Reset!')
        self.overtake_allowed = True and self.overtake_allowed_origin
        self.lane_change_allowed = True
        self.destination_push_flag = 0
        self.set_destination(
            ego_vehicle_loc,
            self.end_waypoint.transform.location,
            clean=True,
            clean_history=True)

    # 3. INTERSECTION RULE ----------------------------------------------------
    self.overtake_allowed = False if is_intersection else self.overtake_allowed_origin

    # 4. PATH GENERATION / LANE CHANGE ---------------------------------------
    rx, ry, rk, ryaw = self._local_planner.generate_path()
    self.lane_change_allowed = self.check_lane_change_permission(
        lane_change_allowed, collision_detector_enabled, rk)

    # 5. COLLISION CHECK ------------------------------------------------------
    is_hazard = False
    if collision_detector_enabled:
        is_hazard, obstacle_vehicle, distance = self.collision_manager(
            rx, ry, ryaw, ego_vehicle_wp)
    car_following_flag = False
    if not is_hazard:
        self.hazard_flag = False

    # 6. PUSH CASE ------------------------------------------------------------
    if not self.lane_change_allowed and \
            self.get_local_planner().potential_curved_road and \
            not self.destination_push_flag and \
            self.overtake_counter <= 0:
        self.overtake_allowed = False
        reset_target = self.get_push_destination(ego_vehicle_wp, is_intersection)
        self.destination_push_flag = 90
        self.set_destination(
            ego_vehicle_loc,
            reset_target.transform.location,
            clean=True,
            end_reset=False)
        rx, ry, rk, ryaw = self._local_planner.generate_path()

    # 7. HAZARD HANDLING ------------------------------------------------------
    elif is_hazard and (not self.overtake_allowed or
                        self.overtake_counter > 0 or
                        self.get_local_planner().potential_curved_road):
        car_following_flag = True
    elif is_hazard and self.overtake_allowed and self.overtake_counter <= 0:
        obstacle_speed = get_speed(obstacle_vehicle)
        obstacle_lane_id = self._map.get_waypoint(obstacle_vehicle.get_location()).lane_id
        ego_lane_id = ego_vehicle_wp.lane_id
        if ego_lane_id == obstacle_lane_id:
            self.hazard_flag = is_hazard
            if self._ego_speed >= obstacle_speed - 5:
                car_following_flag = self.overtake_management(obstacle_vehicle)
            else:
                car_following_flag = True

    # 8. CAR FOLLOWING --------------------------------------------------------
    if car_following_flag:
        if distance < max(self.break_distance, 3):
            return 0, None     # emergency brake
        target_speed = self.car_following_manager(obstacle_vehicle, distance, target_speed)
        target_speed, target_loc = self._local_planner.run_step(
            rx, ry, rk, target_speed=target_speed)
        return target_speed, target_loc

    # 9. NORMAL DRIVING -------------------------------------------------------
    target_speed, target_loc = self._local_planner.run_step(
        rx, ry, rk, target_speed=self.max_speed - self.speed_lim_dist
        if not target_speed else target_speed)
    return target_speed, target_loc
```

### Logic Explanation
1. Handle termination/traffic compliance immediately.
2. Keep the local planner fed with fresh paths, and reset to the global plan if a temporary route (push/overtake) finishes.
3. Disallow overtakes in intersections.
4. Use LocalPlanner for geometry; keep lane-change permission synchronized with curvature/collision info.
5. Prioritize push -> follow -> overtake -> normal driving to maintain predictability.

### Flowchart

```
Update counters & detect intersection
  ↓
Destination reached? → exit
  ↓
Red light? → stop
  ↓
Temporary route finished? → reset to global route
  ↓
Disable overtake in intersection
  ↓
Generate path, check lane change permission
  ↓
Collision check → hazard?
  ↓
If lane change blocked & push available → push destination
Else if hazard & cannot overtake → follow
Else if hazard & can overtake → attempt overtake
  ↓
If following:
  ├─ Too close → emergency stop
  └─ Else compute follow speed, run LocalPlanner
Else:
  └─ Normal run_step with desired speed
  ↓
Return (target_speed, target_location)
```

---

## Overall Flow (File Level)

```
Initialization (__init__)
  ↓
Loop each simulation tick:
  ├─ update_information()
  │    └─ Refresh ego pose/speed, obstacles, traffic state
  └─ run_step()
       ├─ Manage lights/destination resets
       ├─ Generate smooth path via LocalPlanner
       ├─ Detect collisions & lane-change safety
       ├─ Choose behavior (push / follow / overtake / normal)
       └─ Output target speed & waypoint for control
```

---

This rewrite now mirrors the earlier explanation style: code section + data types, explicit “why/logic” paragraphs, concrete examples, and a flowchart after every method. Let me know if you’d like similar treatment for other files.```

```python
    def set_destination(self, start_location, end_location,
                        clean=False, end_reset=True, clean_history=False):
        """
        ... (original docstring) ...
        """
        if clean:
            self.get_local_planner().get_waypoints_queue().clear()
            self.get_local_planner().get_trajectory().clear()
            self.get_local_planner().get_waypoint_buffer().clear()
        if clean_history:
            self.get_local_planner().get_history_buffer().clear()

        self.start_waypoint = self._map.get_waypoint(start_location)

        # Ensure start waypoint lies ahead of ego to avoid planning backwards
        if self._ego_pos:
            cur_loc = self._ego_pos.location
            cur_yaw = self._ego_pos.rotation.yaw
            _, angle = cal_distance_angle(
                self.start_waypoint.transform.location, cur_loc, cur_yaw)

            while angle > 90:
                self.start_waypoint = self.start_waypoint.next(1)[0]
                _, angle = cal_distance_angle(
                    self.start_waypoint.transform.location, cur_loc, cur_yaw)
                # Example: ego at x=100, start waypoint 2m behind => angle 180°, advance 1m until ≤90°

        end_waypoint = self._map.get_waypoint(end_location)
        if end_reset:
            self.end_waypoint = end_waypoint

        route_trace = self._trace_route(self.start_waypoint, end_waypoint)

        if self.initial_global_route is None:
            self.initial_global_route = route_trace

        self._local_planner.set_global_plan(route_trace, clean)
```

### Flowchart

```
Optional: clean buffers/history
  ↓
Convert start/end locations to waypoints
  ↓
If ego pose known:
  ├─ While start waypoint behind ego ⇒ advance forward
  └─ Ensures planner starts ahead
  ↓
Route = _trace_route(start, end)
  ↓
Store initial route if first time
  ↓
Pass route to LocalPlanner
  ↓
End
```

---

## Method 6: `get_local_planner()`

```python
    def get_local_planner(self):
        """
        return the local planner
        """
        return self._local_planner
```

### Flowchart

```
Return reference to LocalPlanner
```

---

## Method 7: `reroute()`

```python
    def reroute(self, spawn_points):
        """
        ... (original docstring) ...
        """
        if self.debug:
            print("Target almost reached, setting new destination...")
        random.shuffle(spawn_points)
        new_start = self._local_planner.waypoints_queue[-1][0].transform.location
        destination = spawn_points[0].location if \
            spawn_points[0].location != new_start else spawn_points[1].location
        if self.debug:
            print("New destination: " + str(destination))

        self.set_destination(new_start, destination)
```

**Why:** When near destination, pick a new random destination to keep simulation running (e.g., benchmarking).

### Flowchart

```
Shuffle spawn points
  ↓
Use last planned waypoint as new start
  ↓
Pick random destination different from start
  ↓
Call set_destination(new_start, destination)
  ↓
End
```

---

## Method 8: `_trace_route()`

```python
    def _trace_route(self, start_waypoint, end_waypoint):
        """
        ... (original docstring) ...
        """
        if self._global_planner is None:
            wld = self.vehicle.get_world()
            dao = GlobalRoutePlannerDAO(
                wld.get_map(), sampling_resolution=self._sampling_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._global_planner = grp

        route = self._global_planner.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)
        return route
```

**Why:** Lazily instantiate the heavy global planner once, reuse for subsequent route queries.

### Flowchart

```
If _global_planner is None:
  ├─ Build DAO with map + sampling resolution
  ├─ Create GlobalRoutePlanner and setup()
  └─ Store reference
Call global planner trace_route(start, end)
  ↓
Return list of (Waypoint, RoadOption)
```

---

## Method 9: `traffic_light_manager()`

```python
    def traffic_light_manager(self, waypoint):
        """
        ... (original docstring) ...
        """
        light_id = self.vehicle.get_traffic_light(
        ).id if self.vehicle.get_traffic_light() is not None else -1

        if 60 <= self.stop_sign_wait_count < 240:
            self.stop_sign_wait_count += 1
        elif self.stop_sign_wait_count >= 240:
            self.stop_sign_wait_count = 0

        if self.light_state == "Red":
            if light_id == -1:
                if self.stop_sign_wait_count < 60:
                    self.stop_sign_wait_count += 1
                    return 1    # instruct stop (emergency flag)
                else:
                    return 0    # done waiting at stop sign

            if not waypoint.is_junction and (
                    self.light_id_to_ignore != light_id or light_id == -1):
                return 1        # approaching red light outside junction
            elif waypoint.is_junction and light_id != -1:
                self.light_id_to_ignore = light_id  # ignore same semaphore once passed
        if self.light_id_to_ignore != light_id:
            self.light_id_to_ignore = -1
        return 0
```

**Example:** Stop sign encountered (light_id=-1), wait 60 frames (~2 seconds at 30 FPS) before resuming.

### Flowchart

```
Get current light ID (or -1 if stop sign)
  ↓
Update stop_sign_wait_count
  ↓
If light_state == "Red":
  ├─ If stop sign (light_id=-1):
  │    ├─ If wait_count < 60 ⇒ increment & return 1 (stop)
  │    └─ Else return 0 (done waiting)
  ├─ Else if not junction and new red light ⇒ return 1
  └─ Else if junction ⇒ remember light_id to ignore later
Reset light_id_to_ignore if new light
  ↓
Return 0 (no stop needed)
```

---

## Method 10: `collision_manager()`

```python
    def collision_manager(self, rx, ry, ryaw, waypoint, adjacent_check=False):
        """
        ... (original docstring) ...
        """
        def dist(v):
            return v.get_location().distance(waypoint.transform.location)

        vehicle_state = False
        min_distance = 100000          # TYPE: float meters
        target_vehicle = None

        for vehicle in self.obstacle_vehicles:
            collision_free = self._collision_check.collision_circle_check(
                rx, ry, ryaw, vehicle, self._ego_speed / 3.6, self._map,
                adjacent_check=adjacent_check)
            if not collision_free:
                vehicle_state = True
                distance = positive(dist(vehicle) - 3)
                # WHY subtract 3m: approximate vehicle length to measure bumper gap.
                if distance < min_distance:
                    min_distance = distance
                    target_vehicle = vehicle

        return vehicle_state, target_vehicle, min_distance
```

**Why:** iterate over perceived obstacles, run geometric collision check along planned path, and return the closest threat.

### Flowchart

```
Set min_distance = large, vehicle_state = False
  ↓
For each obstacle vehicle:
  ├─ Run collision_circle_check on path (rx, ry, ryaw)
  ├─ If collision predicted:
  │    ├─ vehicle_state = True
  │    ├─ distance = positive(dist - 3)
  │    └─ Update closest target_vehicle/min_distance
Return (vehicle_state, target_vehicle, min_distance)
```

---

## Method 11: `overtake_management()`

```python
    def overtake_management(self, obstacle_vehicle):
        """
        ... (original docstring) ...
        """
        obstacle_vehicle_loc = obstacle_vehicle.get_location()
        obstacle_vehicle_wpt = self._map.get_waypoint(obstacle_vehicle_loc)
        left_turn = obstacle_vehicle_wpt.left_lane_marking.lane_change
        right_turn = obstacle_vehicle_wpt.right_lane_marking.lane_change
        left_wpt = obstacle_vehicle_wpt.get_left_lane()
        right_wpt = obstacle_vehicle_wpt.get_right_lane()

        if (left_turn == carla.LaneChange.Left or left_turn ==
            carla.LaneChange.Both) and \
                left_wpt and \
                obstacle_vehicle_wpt.lane_id * left_wpt.lane_id > 0 and \
                left_wpt.lane_type == carla.LaneType.Driving:
            rx, ry, ryaw = self._collision_check.adjacent_lane_collision_check(
                ego_loc=self._ego_pos.location, target_wpt=left_wpt,
                carla_map=self._map,
                overtake=True, world=self.vehicle.get_world())
            vehicle_state, _, _ = self.collision_manager(
                rx, ry, ryaw, self._map.get_waypoint(
                    self._ego_pos.location), True)
            if not vehicle_state:
                print("left overtake is operated")
                self.overtake_counter = 100          # ~100 frames cooldown (~3 sec)
                next_wpt_list = left_wpt.next(self._ego_speed / 3.6 * 6)
                if len(next_wpt_list) == 0:
                    return True

                next_wpt = next_wpt_list[0]
                left_wpt = left_wpt.next(5)[0]
                self.set_destination(
                    left_wpt.transform.location,
                    next_wpt.transform.location,
                    clean=True,
                    end_reset=False)
                return vehicle_state

        if (right_turn == carla.LaneChange.Right or right_turn ==
            carla.LaneChange.Both) and \
                right_wpt and \
                obstacle_vehicle_wpt.lane_id * right_wpt.lane_id > 0 \
                and right_wpt.lane_type == carla.LaneType.Driving:
            rx, ry, ryaw = self._collision_check.adjacent_lane_collision_check(
                ego_loc=self._ego_pos.location,
                target_wpt=right_wpt,
                overtake=True,
                carla_map=self._map,
                world=self.vehicle.get_world())

            vehicle_state, _, _ = self.collision_manager(
                rx, ry, ryaw, self._map.get_waypoint(
                    self._ego_pos.location), True)
            if not vehicle_state:
                print("right overtake is operated")
                self.overtake_counter = 100
                next_wpt_list = right_wpt.next(self._ego_speed / 3.6 * 6)
                if len(next_wpt_list) == 0:
                    return True

                next_wpt = next_wpt_list[0]
                right_wpt = right_wpt.next(5)[0]
                self.set_destination(
                    right_wpt.transform.location,
                    next_wpt.transform.location,
                    clean=True,
                    end_reset=False)
                return vehicle_state

        return True
```

**Why:** Evaluate left/right adjacent lanes for safe overtake by generating a hypothetical path and collision-checking it. If safe, re-plan route into adjacent lane for a short distance.

### Flowchart

```
Compute obstacle's lane + adjacent lanes
  ↓
If left lane change allowed & safe:
  ├─ Generate path in left lane
  ├─ Collision check (adjacent)
  ├─ If safe:
  │    ├─ Set overtake cooldown = 100
  │    ├─ Plan short route through left lane
  │    └─ Return False (no hazard after replan)
If right lane change allowed & safe:
  (same steps as left)
Else:
  Return True (no safe overtake ⇒ follow)
```

---

## Method 12: `lane_change_management()`

```python
    def lane_change_management(self):
        """
        ... (original docstring) ...
        """
        ego_wpt = self._map.get_waypoint(self._ego_pos.location)
        ego_lane_id = ego_wpt.lane_id
        target_wpt = None

        for wpt in self.get_local_planner().get_waypoint_buffer():
            if wpt[0].lane_id != ego_lane_id:
                target_wpt = wpt[0]
                break
        if not target_wpt:
            return True   # nothing to change into

        rx, ry, ryaw = self._collision_check.adjacent_lane_collision_check(
            ego_loc=self._ego_pos.location,
            target_wpt=target_wpt,
            overtake=False,
            carla_map=self._map,
            world=self.vehicle.get_world())
        vehicle_state, _, _ = self.collision_manager(
            rx, ry, ryaw, self._map.get_waypoint(
                self._ego_pos.location), adjacent_check=True)
        return not vehicle_state
```

**Why:** Pre-check adjacent lane for collisions before executing planned lane change.

### Flowchart

```
Find first waypoint in buffer with different lane_id
  ↓
If none ⇒ return True (no lane change planned)
  ↓
Generate adjacent-lane path & collision-check it
  ↓
If collision predicted ⇒ return False
Else ⇒ return True
```

---

## Method 13: `car_following_manager()`

```python
    def car_following_manager(self, vehicle, distance, target_speed=None):
        """
        ... (original docstring) ...
        """
        if not target_speed:
            target_speed = self.max_speed - self.speed_lim_dist

        vehicle_speed = get_speed(vehicle)   # km/h
        delta_v = max(1, (self._ego_speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / \
                                                      np.nextafter(0., 1.)
        self.ttc = ttc
        if self.safety_time > ttc > 0.0:
            target_speed = min(positive(vehicle_speed - self.speed_decrease),
                               target_speed)
        else:
            target_speed = 0 if vehicle_speed == 0 else \
                min(vehicle_speed + 1, target_speed)
        return target_speed
```

**Example:** Ego 60 km/h, lead 40 km/h, distance 20 m → delta_v ≈ 5.5 m/s, ttc ≈ 3.6 s. If safety_time=4s, target_speed reduced to lead_speed - speed_decrease.

### Flowchart

```
Set default target_speed
  ↓
Measure lead vehicle speed
  ↓
Compute closing speed delta_v and TTC
  ↓
If 0 < TTC < safety_time:
  └─ target_speed = min(lead_speed - speed_decrease, current target)
Else:
  └─ target_speed = min(lead_speed + 1, current target) (or 0 if lead stopped)
  ↓
Return target_speed
```

---

## Method 14: `is_intersection()`

```python
    def is_intersection(self, objects, waypoint_buffer):
        """
        ... (original docstring) ...
        """
        for tl in objects['traffic_lights']:
            for wpt, _ in waypoint_buffer:
                distance = tl.get_location().distance(wpt.transform.location)
                if distance < 20:
                    return True
        return False
```

### Flowchart

```
For each traffic light detected:
  ├─ For each future waypoint:
  │    ├─ If distance < 20m ⇒ return True
Return False (no intersection nearby)
```

---

## Method 15: `is_close_to_destination()`

```python
    def is_close_to_destination(self):
        """
        ... (original docstring) ...
        """
        flag = abs(self._ego_pos.location.x - self.end_waypoint.transform.location.x) <= 10 and \
               abs(self._ego_pos.location.y - self.end_waypoint.transform.location.y) <= 10
        return flag
```

**Why:** simple bounding box of ±10 meters in X/Y.

### Flowchart

```
If |x - x_dest| ≤ 10 AND |y - y_dest| ≤ 10 ⇒ True
Else ⇒ False
```

---

## Method 16: `check_lane_change_permission()`

```python
    def check_lane_change_permission(self, lane_change_allowed,
                                     collision_detector_enabled, rk):
        """
        ... (original docstring) ...
        """
        if len(rk) > 2 and np.mean(np.abs(np.array(rk))) > 0.04:
            lane_change_allowed = False   # WHY: high curvature ⇒ unsafe

        lane_change_enabled_flag = collision_detector_enabled and \
                                   self.get_local_planner().lane_id_change and \
                                   self.get_local_planner().lane_lateral_change and \
                                   self.overtake_counter <= 0 and \
                                   not self.destination_push_flag
        if lane_change_enabled_flag:
            lane_change_allowed = lane_change_allowed and self.lane_change_management()
            if not lane_change_allowed:
                print("lane change not allowed")

        return lane_change_allowed
```

### Flowchart

```
If mean |rk| > 0.04 ⇒ lane_change_allowed = False
  ↓
lane_change_enabled_flag = collision detector ON AND local planner reports lane change AND no overtake cooldown AND no destination push
  ↓
If flag True:
  ├─ lane_change_allowed &= lane_change_management()
  └─ If False ⇒ print warning
Return lane_change_allowed
```

---

## Method 17: `get_push_destination()`

```python
    def get_push_destination(self, ego_vehicle_wp, is_intersection):
        """
        ... (original docstring) ...
        """
        waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
        reset_index = len(waypoint_buffer) // 2

        if is_intersection:
            reset_target = waypoint_buffer[reset_index][0].next(
                max(self._ego_speed / 3.6, 10.0))[0]
        else:
            reset_target = ego_vehicle_wp.next(max(self._ego_speed / 3.6 * 3,
                                                   10.0))[0]
        if self.debug:
            print(
                'Vehicle id: %d :destination pushed forward ...' %
                (self.vehicle.id, ...))
        return reset_target
```

**Why:** When planned lane change blocked, temporarily “push” destination forward to stay in current lane.

### Flowchart

```
reset_index = half of waypoint buffer
  ↓
If in intersection:
  └─ reset_target = future waypoint after max(speed,10)m
Else:
  └─ reset_target = ego waypoint advanced by max(3*speed,10)m
  ↓
Return reset_target
```

---

## Method 18: `run_step()`

```python
    def run_step(self,
                 target_speed=None,
                 collision_detector_enabled=True,
                 lane_change_allowed=True):
        """
        ... (original docstring) ...
        """
        ego_vehicle_loc = self._ego_pos.location
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        waipoint_buffer = self.get_local_planner().get_waypoint_buffer()
        self.ttc = 1000
        if self.overtake_counter > 0:
            self.overtake_counter -= 1
        if self.destination_push_flag > 0:
            self.destination_push_flag -= 1

        is_intersection = self.is_intersection(self.objects, waipoint_buffer)

        # 0. Simulation end
        if self.is_close_to_destination():
            print('Simulation is Over')
            sys.exit(0)

        # 1. Traffic light
        if self.traffic_light_manager(ego_vehicle_wp) != 0:
            return 0, None

        # 2. Route reset if temporary plan finished
        if len(self.get_local_planner().get_waypoints_queue()) == 0 \
                and len(self.get_local_planner().get_waypoint_buffer()) <= 2:
            if self.debug:
                print('Destination Reset!')
            self.overtake_allowed = True and self.overtake_allowed_origin
            self.lane_change_allowed = True
            self.destination_push_flag = 0
            self.set_destination(
                ego_vehicle_loc,
                self.end_waypoint.transform.location,
                clean=True,
                clean_history=True)

        # 3. Intersection tweaks
        if is_intersection:
            self.overtake_allowed = False
        else:
            self.overtake_allowed = True and self.overtake_allowed_origin

        # 4. Path generation
        rx, ry, rk, ryaw = self._local_planner.generate_path()

        # 5. Lane change permission
        self.lane_change_allowed = self.check_lane_change_permission(
            lane_change_allowed, collision_detector_enabled, rk)

        # 6. Collision checking
        is_hazard = False
        if collision_detector_enabled:
            is_hazard, obstacle_vehicle, distance = self.collision_manager(
                rx, ry, ryaw, ego_vehicle_wp)
        car_following_flag = False
        if not is_hazard:
            self.hazard_flag = False

        # 7. Push case
        if not self.lane_change_allowed and \
                self.get_local_planner().potential_curved_road \
                and not self.destination_push_flag and \
                self.overtake_counter <= 0:
            self.overtake_allowed = False
            reset_target = self.get_push_destination(ego_vehicle_wp, is_intersection)
            self.destination_push_flag = 90
            self.set_destination(
                ego_vehicle_loc,
                reset_target.transform.location,
                clean=True,
                end_reset=False)
            rx, ry, rk, ryaw = self._local_planner.generate_path()

        # 8. Hazard handling (follow/overtake)
        elif is_hazard and (not self.overtake_allowed or
                            self.overtake_counter > 0
                            or self.get_local_planner().potential_curved_road):
            car_following_flag = True
        elif is_hazard and self.overtake_allowed and \
                self.overtake_counter <= 0:
            obstacle_speed = get_speed(obstacle_vehicle)
            obstacle_lane_id = self._map.get_waypoint(obstacle_vehicle.get_location()).lane_id
            ego_lane_id = self._map.get_waypoint(
                self._ego_pos.location).lane_id
            if ego_lane_id == obstacle_lane_id:
                self.hazard_flag = is_hazard
                if self._ego_speed >= obstacle_speed - 5:
                    car_following_flag = self.overtake_management(obstacle_vehicle)
                else:
                    car_following_flag = True

        # 9. Car following response
        if car_following_flag:
            if distance < max(self.break_distance, 3):
                return 0, None

            target_speed = self.car_following_manager(obstacle_vehicle, distance, target_speed)
            target_speed, target_loc = self._local_planner.run_step(
                rx, ry, rk, target_speed=target_speed)
            return target_speed, target_loc

        # 10. Normal driving
        target_speed, target_loc = self._local_planner.run_step(
            rx, ry, rk, target_speed=self.max_speed - self.speed_lim_dist
            if not target_speed else target_speed)
        return target_speed, target_loc
```

**Key Logic:**
- Early exits for destination or red lights
- Maintains route continuity (reset when temporary route finished)
- Generates smooth path and checks lane-change permission
- Handles hazard cases: push destination, car following, or overtake
- Returns target speed/location for control module

### Flowchart

```
Get ego pose & buffers
  ↓
Update counters (overtake cooldown, push flag)
  ↓
Detect intersection, destination reached?
  ├─ If reached ⇒ exit simulation
  └─ If red light ⇒ return stop command
  ↓
If temporary route finished ⇒ reset to global route
  ↓
Disable overtake inside intersection, enable otherwise
  ↓
Generate path (LocalPlanner)
  ↓
Update lane_change_allowed via curvature + collision flags
  ↓
Run collision_manager ⇒ is_hazard? (with obstacle + distance)
  ↓
If lane change blocked & push allowed ⇒ push destination and replan
Else if hazard and overtake not allowed ⇒ follow
Else if hazard and overtake allowed ⇒ attempt overtake
  ↓
If following:
  ├─ If too close ⇒ emergency stop
  └─ Else compute following speed and call LocalPlanner.run_step()
Else:
  └─ Normal driving: run_step() with default speed
  ↓
Return (target_speed, target_location)
```

---

## Overall File Flowchart

```
Initialization (__init__)
  ↓
Main Loop (per simulation tick)
  ├─ update_information()
  │    ├─ Store ego state, obstacles, traffic lights
  │    └─ Notify local planner
  │
  ├─ run_step()
  │    1. Handle destination / red lights / route reset
  │    2. Generate smooth path via LocalPlanner
  │    3. Collision detection & lane-change gating
  │    4. Decide behavior:
  │         - Push destination (if lane change blocked)
  │         - Car following (distance < threshold)
  │         - Overtake (safe adjacent lane)
  │         - Normal driving
  │    5. Produce target speed/location
  │
  └─ Control module receives target, applies PID control
```

---

## Summary

- `BehaviorAgent` coordinates global planning, local smoothing, and behavior decisions.
- Handles traffic rules (lights, stops), obstacle avoidance (collision checker), and advanced behaviors (overtake, push destinations).
- Delegates geometric smoothing to `LocalPlanner` and collision checks to `CollisionChecker`.
- Provides deterministic flowcharts to understand how each method fits in the decision pipeline.
```

