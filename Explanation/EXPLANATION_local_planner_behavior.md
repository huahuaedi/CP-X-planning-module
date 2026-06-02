# Explanation: `local_planner_behavior.py`

## Overview

The `LocalPlanner` class is responsible for generating smooth, drivable trajectories from high-level waypoints. It converts discrete waypoints from the global route planner into continuous, smooth paths that vehicles can follow using spline interpolation. It also handles trajectory sampling with speed assignment based on curvature constraints.

**Key Purpose**: Bridge the gap between high-level route planning (waypoints) and low-level vehicle control (smooth trajectories with speed profiles).

**Relationship to Other Modules**:
- **Input**: Receives waypoints from `GlobalRoutePlanner` via `BehaviorAgent`
- **Output**: Provides smooth trajectories to `ControlManager` via `BehaviorAgent`
- **Used by**: `BehaviorAgent.run_step()` calls `generate_path()` and `run_step()`

---

## Class Structure

```python
LocalPlanner
├── __init__()              # Initialize planner with configuration
├── set_global_plan()       # Set waypoints from global route
├── update_information()    # Update ego position and speed
├── generate_path()          # Generate smooth spline path from waypoints
├── generate_trajectory()    # Sample path with speed assignment
├── buffer_filter()         # Remove problematic waypoints
├── pop_buffer()            # Remove passed waypoints
└── run_step()              # Main execution step
```

---

## RoadOption Enum

### Purpose
Defines the possible navigation instructions when moving between road segments.

### Code with Detailed Comments

```python
class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations
    when moving from a segment of lane to other.
    
    WHY ENUM:
    - Type safety: Prevents invalid values (can't use 999 as road option)
    - Readability: RoadOption.LEFT is clearer than 1
    - Maintainability: Easy to add new options
    """
    VOID = -1
    # MEANING: Invalid or empty road option
    # USAGE: Used when no valid route option exists
    # EXAMPLE: Vehicle reached dead end, no valid path
    
    LEFT = 1
    # MEANING: Turn left at intersection
    # USAGE: Vehicle needs to turn left at upcoming intersection
    # EXAMPLE: At 4-way intersection, turn left instead of going straight
    
    RIGHT = 2
    # MEANING: Turn right at intersection
    # USAGE: Vehicle needs to turn right at upcoming intersection
    # EXAMPLE: At T-junction, turn right to continue route
    
    STRAIGHT = 3
    # MEANING: Go straight through intersection
    # USAGE: Continue straight at intersection (not turning)
    # EXAMPLE: At 4-way intersection, go straight through
    
    LANEFOLLOW = 4
    # MEANING: Continue following current lane (no turn, no lane change)
    # USAGE: Most common option - just drive forward in lane
    # EXAMPLE: Driving on highway, following lane markings
    
    CHANGELANELEFT = 5
    # MEANING: Change to left lane
    # USAGE: Vehicle needs to move to left adjacent lane
    # EXAMPLE: Preparing for left turn, or overtaking on right
    
    CHANGELANERIGHT = 6
    # MEANING: Change to right lane
    # USAGE: Vehicle needs to move to right adjacent lane
    # EXAMPLE: Preparing for right turn, or returning to right lane after overtake
```

---

## Method 1: `__init__()`

### Purpose
Initialize the LocalPlanner with configuration parameters and create data structures for storing waypoints and trajectories.

### Code with Detailed Comments

```python
def __init__(self, agent, carla_map, config_yaml):
    # ============================================================
    # PARAMETER: agent
    # ============================================================
    # TYPE: carla.agent (BehaviorAgent instance)
    # MEANING: Reference to the behavior agent that owns this planner
    # WHY: Need access to vehicle object and agent methods
    # 
    # EXAMPLE:
    # - agent.vehicle: The CARLA vehicle object to control
    # - agent.get_waypoints_queue(): Access to global route waypoints
    # 
    # RELATIONSHIP:
    # - BehaviorAgent creates LocalPlanner
    # - LocalPlanner needs agent to access vehicle and route data
    self._vehicle = agent.vehicle
    self._map = carla_map
    
    # ============================================================
    # INITIALIZE EGO STATE
    # ============================================================
    # TYPE: carla.Transform, float
    # MEANING: Current position and speed of ego vehicle
    # WHY: Need to know where vehicle is to plan from current location
    # 
    # INITIAL VALUES:
    # - None: Will be updated by update_information() before use
    # - None: Will be updated by update_information() before use
    # 
    # UPDATE FREQUENCY:
    # - Updated every simulation step via update_information()
    # - Called by BehaviorAgent before generate_path()
    self._ego_pos = None  # carla.Transform: (x, y, z, roll, pitch, yaw)
    self._ego_speed = None  # float: speed in km/h
    
    # ============================================================
    # PARAMETER: min_dist
    # ============================================================
    # TYPE: float
    # MEANING: Minimum distance threshold for waypoint removal
    # WHY: Remove waypoints when vehicle gets close enough (has "reached" them)
    # 
    # DEFAULT: Typically 2.0-3.0 meters
    # EXAMPLE:
    # - min_dist = 2.5m
    # - Vehicle at (100, 200), waypoint at (102, 200)
    # - Distance = 2.0m < 2.5m → Remove waypoint (vehicle reached it)
    # 
    # TRADE-OFF:
    # - Too small (< 1m): Waypoints removed too early, jerky behavior
    # - Too large (> 5m): Waypoints removed too late, vehicle overshoots
    # - 2-3m is good balance for smooth following
    self._min_distance = config_yaml['min_dist']
    
    # ============================================================
    # PARAMETER: buffer_size
    # ============================================================
    # TYPE: int
    # MEANING: Maximum number of waypoints to keep in working buffer
    # WHY: Limit memory usage and keep only relevant waypoints
    # 
    # DEFAULT: Typically 10-20 waypoints
    # EXAMPLE:
    # - buffer_size = 15
    # - Keep next 15 waypoints in _waypoint_buffer
    # - When buffer < 15, refill from waypoints_queue
    # 
    # WHY BUFFER:
    # - Performance: Don't process entire route at once
    # - Smoothness: Need several waypoints ahead for spline generation
    # - Memory: Limit to reasonable number
    self._buffer_size = config_yaml['buffer_size']
    
    # ============================================================
    # DATA STRUCTURE: waypoints_queue
    # ============================================================
    # TYPE: deque (double-ended queue)
    # MEANING: Complete global route waypoints from origin to destination
    # WHY: Store entire route, feed waypoints to buffer as needed
    # 
    # STRUCTURE: deque of tuples (waypoint, RoadOption)
    # EXAMPLE:
    #   waypoints_queue = [
    #       (Waypoint(100, 200, 0), RoadOption.LANEFOLLOW),
    #       (Waypoint(102, 200, 0), RoadOption.LANEFOLLOW),
    #       (Waypoint(104, 200, 0), RoadOption.CHANGELANERIGHT),
    #       ...
    #   ]
    # 
    # WHY DEQUE:
    # - Fast append/pop from both ends: O(1) operations
    # - Efficient for queue operations (FIFO)
    # - maxlen prevents unbounded growth
    # 
    # MAXLEN: 20000 waypoints
    # - Typical route: 100-1000 waypoints
    # - 20000 is safety limit (very long routes)
    self.waypoints_queue = deque(maxlen=20000)
    
    # ============================================================
    # DATA STRUCTURE: _waypoint_buffer
    # ============================================================
    # TYPE: deque
    # MEANING: Working buffer of next waypoints to process
    # WHY: Keep small subset of waypoints for current planning
    # 
    # RELATIONSHIP:
    # - waypoints_queue: Complete route (source)
    # - _waypoint_buffer: Next N waypoints (working set)
    # - When buffer empty/low, refill from queue
    # 
    # SIZE: buffer_size (typically 10-20)
    # EXAMPLE:
    #   _waypoint_buffer = [
    #       (Waypoint(100, 200, 0), RoadOption.LANEFOLLOW),  # Next
    #       (Waypoint(102, 200, 0), RoadOption.LANEFOLLOW),
    #       (Waypoint(104, 200, 0), RoadOption.LANEFOLLOW),  # Last
    #   ]
    self._waypoint_buffer = deque(maxlen=self._buffer_size)
    
    # ============================================================
    # DATA STRUCTURE: _trajectory_buffer
    # ============================================================
    # TYPE: deque
    # MEANING: Generated trajectory points with speed assignments
    # WHY: Store sampled trajectory for vehicle to follow
    # 
    # STRUCTURE: deque of tuples (Transform, speed)
    # EXAMPLE:
    #   _trajectory_buffer = [
    #       (Transform(Location(100.1, 200.1, 0.5), Rotation(...)), 20.5),
    #       (Transform(Location(100.2, 200.2, 0.5), Rotation(...)), 21.0),
    #       ...
    #   ]
    # 
    # WHY BUFFER:
    # - Pre-generate trajectory points (smooth, continuous)
    # - Vehicle follows these points instead of waypoints directly
    # - Reduces computation per step (generate once, use many times)
    # 
    # SIZE: 30 points (typically 3 seconds at 0.1s intervals)
    # - Each point is 0.1s apart
    # - 30 points = 3 seconds of trajectory
    self._trajectory_buffer = deque(maxlen=30)
    
    # ============================================================
    # DATA STRUCTURE: _history_buffer
    # ============================================================
    # TYPE: deque
    # MEANING: Recently passed waypoints (for spline generation)
    # WHY: Need past waypoints to create smooth spline curve
    # 
    # WHY HISTORY:
    # - Spline needs points before and after current position
    # - Past waypoints help create smooth transition
    # - Prevents sudden direction changes
    # 
    # SIZE: 3 waypoints
    # - Small buffer: Only need recent history
    # - Used in generate_path() for spline interpolation
    self._history_buffer = deque(maxlen=3)
    
    # ============================================================
    # PARAMETER: trajectory_update_freq
    # ============================================================
    # TYPE: int
    # MEANING: Minimum number of trajectory points before regenerating
    # WHY: Don't regenerate trajectory every step (expensive)
    # 
    # EXAMPLE:
    # - trajectory_update_freq = 10
    # - Regenerate when buffer has < 10 points
    # - Otherwise, reuse existing trajectory
    # 
    # WHY THRESHOLD:
    # - Performance: Trajectory generation is expensive
    # - Smoothness: Keep enough points ahead for smooth following
    # - Balance: Regenerate before running out
    self.trajectory_update_freq = config_yaml['trajectory_update_freq']
    
    # ============================================================
    # PARAMETER: waypoint_update_freq
    # ============================================================
    # TYPE: int
    # MEANING: Minimum number of waypoints in buffer before refilling
    # WHY: Don't refill buffer every step (unnecessary)
    # 
    # EXAMPLE:
    # - waypoint_update_freq = 5
    # - Refill when buffer has < 5 waypoints
    # - Otherwise, keep using current buffer
    self.waypoint_update_freq = config_yaml['waypoint_update_freq']
    
    # ============================================================
    # PARAMETER: trajectory_dt
    # ============================================================
    # TYPE: float
    # MEANING: Time step between trajectory points (seconds)
    # WHY: Control temporal resolution of trajectory
    # 
    # DEFAULT: Typically 0.1 seconds
    # EXAMPLE:
    # - dt = 0.1s
    # - Trajectory points: t=0.0s, t=0.1s, t=0.2s, ...
    # - Vehicle follows points at 10 Hz (10 points per second)
    # 
    # TRADE-OFF:
    # - Smaller (0.05s): More points, smoother, but more computation
    # - Larger (0.2s): Fewer points, faster, but less smooth
    # - 0.1s is good balance
    self.dt = config_yaml['trajectory_dt']
    
    # ============================================================
    # LANE CHANGE DETECTION FLAGS
    # ============================================================
    # TYPE: boolean
    # MEANING: Indicators for detecting lane changes and curved roads
    # WHY: Need to handle lane changes differently (smoother splines)
    # 
    # potential_curved_road:
    # - True: Road is curved or lane change is happening
    # - Used to adjust spline generation (include more history)
    # 
    # lane_id_change:
    # - True: Lane ID changed (definite lane change)
    # - Detected by comparing waypoint lane_ids
    # 
    # lane_lateral_change:
    # - True: Large lateral movement detected (possible lane change)
    # - Detected by measuring lateral offset between waypoints
    self.potential_curved_road = False
    self.lane_id_change = False
    self.lane_lateral_change = False
    
    # ============================================================
    # DEBUG FLAGS
    # ============================================================
    # TYPE: boolean
    # MEANING: Enable/disable debug visualization
    # WHY: Help visualize planning for debugging
    # 
    # debug: Draw waypoint buffer and history buffer
    # debug_trajectory: Draw generated trajectory
    self.debug = config_yaml['debug']
    self.debug_trajectory = config_yaml['debug_trajectory']
```

### Flowchart

```
Start
  ↓
Store agent reference (for vehicle access)
  ↓
Initialize ego state (None, will be updated)
  ↓
Load configuration parameters:
  - min_dist (waypoint removal threshold)
  - buffer_size (working buffer size)
  - trajectory_update_freq (regeneration threshold)
  - waypoint_update_freq (refill threshold)
  - trajectory_dt (time step)
  ↓
Create data structures:
  - waypoints_queue (complete route, maxlen=20000)
  - _waypoint_buffer (working buffer, maxlen=buffer_size)
  - _trajectory_buffer (generated trajectory, maxlen=30)
  - _history_buffer (recent waypoints, maxlen=3)
  ↓
Initialize lane change detection flags (all False)
  ↓
Set debug flags from config
  ↓
End (LocalPlanner ready to use)
```

---

## Method 2: `set_global_plan()`

### Purpose
Set the global route plan (waypoints) from the global route planner. This is called when a new route is calculated or updated.

### Code with Detailed Comments

```python
def set_global_plan(self, current_plan, clean=False):
    """
    Sets new global plan.
    
    PARAMETERS:
    - current_plan: list of (waypoint, RoadOption) tuples
    - clean: boolean, whether to clear existing buffers
    
    WHY THIS METHOD:
    - Separates route setting from route generation
    - Allows updating route without recreating planner
    - Can clear old waypoints when new route set
    """
    
    # ============================================================
    # STEP 1: Add All Waypoints to Queue
    # ============================================================
    # LOGIC: Add each waypoint from plan to waypoints_queue
    # MEANING: Store complete route for future use
    # 
    # WHY LOOP:
    # - current_plan is list of (waypoint, RoadOption) tuples
    # - Need to add each tuple to queue
    # 
    # EXAMPLE:
    #   current_plan = [
    #       (Waypoint(100, 200, 0), RoadOption.LANEFOLLOW),
    #       (Waypoint(102, 200, 0), RoadOption.LANEFOLLOW),
    #       (Waypoint(104, 200, 0), RoadOption.CHANGELANERIGHT),
    #   ]
    #   After loop:
    #   waypoints_queue = deque([...all waypoints...])
    for elem in current_plan:
        # TYPE: elem is tuple (carla.Waypoint, RoadOption)
        # EXAMPLE: (Waypoint(100, 200, 0), RoadOption.LANEFOLLOW)
        # 
        # WHY APPEND:
        # - Add to end of queue (FIFO: First In, First Out)
        # - Waypoints processed in order
        self.waypoints_queue.append(elem)
    
    # ============================================================
    # STEP 2: Optionally Clear and Refill Buffer
    # ============================================================
    # LOGIC: If clean=True, clear old buffer and refill with new waypoints
    # MEANING: Start fresh with new route
    # 
    # WHY CLEAN OPTION:
    # - New route: Old waypoints might be invalid
    # - Route update: Need to replace old waypoints
    # - Route reset: Clear everything and start over
    # 
    # WHEN TO USE clean=True:
    # - New destination set
    # - Route recalculated
    # - Route reset/reroute
    # 
    # WHEN TO USE clean=False:
    # - Adding waypoints to existing route
    # - Route extension
    if clean:
        # Clear existing buffer (remove old waypoints)
        # WHY: Old waypoints might be from different route
        self._waypoint_buffer.clear()
        
        # Refill buffer with new waypoints
        # WHY: Need waypoints ready for immediate use
        # 
        # LOOP: Add up to buffer_size waypoints
        # WHY: Don't add more than buffer can hold
        for _ in range(self._buffer_size):
            # Check if waypoints available
            if self.waypoints_queue:
                # Pop from queue (remove from queue)
                # Add to buffer (add to working set)
                # 
                # WHY POPLEFT:
                # - Get first waypoint (FIFO order)
                # - Remove from queue (don't duplicate)
                # 
                # EXAMPLE:
                #   waypoints_queue: [W1, W2, W3, ...]
                #   After popleft: waypoints_queue: [W2, W3, ...]
                #   _waypoint_buffer: [W1]
                self._waypoint_buffer.append(
                    self.waypoints_queue.popleft())
            else:
                # No more waypoints available
                # WHY BREAK: Can't add waypoints that don't exist
                # 
                # EXAMPLE:
                # - Route has only 5 waypoints
                # - buffer_size = 15
                # - After 5 iterations, queue empty
                # - Break (don't try to add more)
                break
```

### Flowchart

```
Start
  ↓
For each waypoint in current_plan:
  ├─ Append to waypoints_queue
  └─ Continue
  ↓
If clean == True:
  ├─ Clear _waypoint_buffer
  ├─ For i in range(buffer_size):
  │   ├─ If waypoints_queue not empty:
  │   │   ├─ Pop waypoint from queue (popleft)
  │   │   └─ Append to buffer
  │   └─ Else: break
  └─ End
  ↓
End (Route set, ready for planning)
```

---

## Method 3: `update_information()`

### Purpose
Update the ego vehicle's current position and speed. Called every simulation step before path generation.

### Code with Detailed Comments

```python
def update_information(self, ego_pos, ego_speed):
    """
    Update the ego position and speed for trajectory planner.
    
    PARAMETERS:
    - ego_pos: carla.Transform (position and orientation)
    - ego_speed: float (speed in km/h)
    
    WHY THIS METHOD:
    - Separates state update from planning logic
    - Called every step by BehaviorAgent
    - Provides current state for path generation
    """
    
    # ============================================================
    # UPDATE EGO POSITION
    # ============================================================
    # TYPE: carla.Transform
    # STRUCTURE: Contains location (x,y,z) and rotation (roll,pitch,yaw)
    # 
    # EXAMPLE:
    #   ego_pos = Transform(
    #       Location(x=100.5, y=200.3, z=0.2),
    #       Rotation(roll=0, pitch=0, yaw=45.0)
    #   )
    # 
    # WHY UPDATE:
    # - Vehicle moves every step
    # - Need current position to plan from
    # - Used in generate_path() to start spline from current location
    self._ego_pos = ego_pos
    
    # ============================================================
    # UPDATE EGO SPEED
    # ============================================================
    # TYPE: float
    # UNIT: kilometers per hour (km/h)
    # 
    # EXAMPLE:
    #   ego_speed = 60.0  # 60 km/h = 16.67 m/s
    # 
    # WHY UPDATE:
    # - Speed changes during acceleration/deceleration
    # - Used in generate_trajectory() for speed assignment
    # - Used for curvature-based speed limiting
    # 
    # CONVERSION:
    # - Input: km/h (from localization)
    # - Internal: Sometimes converted to m/s for calculations
    # - Formula: m/s = km/h / 3.6
    self._ego_speed = ego_speed
```

### Flowchart

```
Start
  ↓
Store ego_pos (Transform: location + rotation)
  ↓
Store ego_speed (float: km/h)
  ↓
End (State updated, ready for path generation)
```

---

## Method 4: `generate_path()`

### Purpose
Generate a smooth, continuous path from waypoints using cubic spline interpolation. This is the core path generation method.

### Code with Detailed Comments

```python
def generate_path(self):
    """
    Generate the smooth path using cubic spline.
    
    RETURNS:
    - rx: list of float (x coordinates)
    - ry: list of float (y coordinates)
    - ryaw: list of float (yaw angles in radians)
    - rk: list of float (curvatures in 1/m)
    
    WHY SPLINE:
    - Waypoints are discrete points
    - Vehicles need smooth, continuous paths
    - Spline creates smooth curve through waypoints
    - Provides curvature for speed control
    """
    
    # ============================================================
    # STEP 1: Initialize Spline Node Lists
    # ============================================================
    # LOGIC: Collect waypoint coordinates for spline interpolation
    # MEANING: These points will define the smooth curve
    # 
    # WHY LISTS:
    # - Spline needs list of (x, y) points
    # - Will add waypoints to these lists
    # - Then create spline from these points
    x = []  # List[float]: x coordinates of spline nodes
    y = []  # List[float]: y coordinates of spline nodes
    
    # ============================================================
    # STEP 2: Filter Problematic Waypoints
    # ============================================================
    # LOGIC: Remove waypoints that could cause bad vehicle dynamics
    # MEANING: Filter out waypoints with dramatic yaw changes
    # 
    # WHY FILTER:
    # - Some waypoints might have sharp turns
    # - Sharp turns cause jerky vehicle motion
    # - Filter prevents bad waypoints from entering spline
    # 
    # WHAT IT DOES:
    # - Removes waypoints behind vehicle
    # - Removes waypoints too close for lane changes
    # - Keeps only good waypoints for smooth path
    self.buffer_filter()
    
    # ============================================================
    # STEP 3: Set Interpolation Resolution
    # ============================================================
    # LOGIC: Define spacing between interpolated points
    # MEANING: How fine-grained the output path will be
    # 
    # VALUE: 0.1 meters
    # WHY 0.1m:
    # - Fine enough for smooth following
    # - Not too fine (avoids excessive computation)
    # - Good balance: ~10 points per meter
    # 
    # EXAMPLE:
    # - Waypoints 2m apart
    # - Interpolate with 0.1m spacing
    # - Get 20 points between each waypoint pair
    ds = 0.1  # [m] distance of each interpolated points
    
    # ============================================================
    # STEP 4: Get Current Vehicle State
    # ============================================================
    # LOGIC: Extract current location and orientation
    # MEANING: Need to know where vehicle is to start path from here
    # 
    # current_location:
    # - TYPE: carla.Location
    # - STRUCTURE: (x, y, z) coordinates
    # - EXAMPLE: Location(x=100.5, y=200.3, z=0.2)
    # 
    # current_yaw:
    # - TYPE: float
    # - UNIT: degrees
    # - MEANING: Vehicle heading direction
    # - EXAMPLE: 45.0 (pointing northeast)
    current_location = self._ego_pos.location
    current_yaw = self._ego_pos.rotation.yaw
    
    # ============================================================
    # STEP 5: Get Current Waypoint
    # ============================================================
    # LOGIC: Find waypoint on map corresponding to current location
    # MEANING: Get road-aligned waypoint near vehicle
    # 
    # WHY GET WAYPOINT:
    # - Vehicle might be slightly off-center in lane
    # - Waypoint is always centered in lane
    # - Use waypoint for smoother path (centered)
    # 
    # next(1)[0]:
    # - next(1): Get next waypoint 1 meter ahead
    # - [0]: Get first waypoint from list
    # - WHY: Want waypoint slightly ahead (not exactly at vehicle)
    current_wpt = self._map.get_waypoint(current_location).next(1)[0]
    current_wpt_loc = current_wpt.transform.location
    
    # ============================================================
    # STEP 6: Get Future and Past Waypoints
    # ============================================================
    # LOGIC: Get waypoints for lane change detection
    # MEANING: Check if vehicle is changing lanes or on curved road
    # 
    # future_wpt:
    # - TYPE: carla.Waypoint
    # - MEANING: Last waypoint in buffer (far ahead)
    # - WHY: Compare with current to detect lane changes
    # 
    # previous_wpt:
    # - TYPE: carla.Waypoint
    # - MEANING: First waypoint in history (recently passed)
    # - WHY: Compare with future to detect lateral movement
    # 
    # FALLBACK:
    # - If no history, use current_wpt
    # - WHY: Need at least one waypoint for comparison
    future_wpt = self._waypoint_buffer[-1][0]  # Last waypoint in buffer
    previous_wpt = self._history_buffer[0][0] if len(
        self._history_buffer) > 0 else current_wpt
    
    # ============================================================
    # STEP 7: Calculate Lateral Offset
    # ============================================================
    # LOGIC: Measure lateral (sideways) movement between waypoints
    # MEANING: Detect if vehicle is changing lanes (large lateral movement)
    # 
    # HOW IT WORKS:
    # 1. Calculate distance and angle from previous to future waypoint
    # 2. Use angle to compute lateral component
    # 3. Large lateral = lane change happening
    # 
    # vec_norm:
    # - TYPE: float
    # - MEANING: Distance from previous to future waypoint
    # - EXAMPLE: 10.5 meters
    # 
    # angle:
    # - TYPE: float
    # - MEANING: Angle between waypoint direction and line connecting them
    # - EXAMPLE: 15.0 degrees
    vec_norm, angle = cal_distance_angle(previous_wpt.transform.location,
                                         future_wpt.transform.location,
                                         future_wpt.transform.rotation.yaw)
    
    # Calculate lateral distance (perpendicular to road)
    # FORMULA: lateral = distance × sin(angle)
    # 
    # WHY SIN:
    # - sin(angle) gives perpendicular component
    # - cos(angle) would give parallel component
    # 
    # ANGLE ADJUSTMENT:
    # - angle - 1 if angle > 90: Adjust for angles > 90°
    # - angle + 1 if angle <= 90: Adjust for angles <= 90°
    # - WHY: Handle angle wrapping (0-180° range)
    # 
    # EXAMPLE:
    # - vec_norm = 10m, angle = 15°
    # - lateral_diff = 10 × sin(15°) = 10 × 0.259 = 2.59m
    # - Large lateral = lane change detected
    lateral_diff = abs(
        vec_norm *
        math.sin(
            math.radians(
                angle - 1 if angle > 90 else angle + 1)))
    
    # ============================================================
    # STEP 8: Compare Lateral Offset with Vehicle Width
    # ============================================================
    # LOGIC: Check if lateral movement exceeds vehicle width
    # MEANING: Large lateral movement = lane change
    # 
    # WHY COMPARE WITH WIDTH:
    # - Vehicle width ~2 meters
    # - Lateral movement > width = definitely lane change
    # - Lateral movement < width = might be just lane drift
    # 
    # veh_width:
    # - TYPE: float
    # - CALCULATION: 2 × (bounding_box.location.y - extent.y)
    # - MEANING: Total vehicle width
    # - EXAMPLE: 2.0 meters (typical car)
    # 
    # lane_width:
    # - TYPE: float
    # - MEANING: Width of current lane
    # - EXAMPLE: 3.5 meters (standard lane)
    boundingbox = self._vehicle.bounding_box
    veh_width = 2 * abs(boundingbox.location.y - boundingbox.extent.y)
    lane_width = current_wpt.lane_width
    
    # lane_lateral_change:
    # - True: Lateral movement > vehicle width
    # - MEANING: Significant sideways movement detected
    # - EXAMPLE: lateral_diff = 2.5m, veh_width = 2.0m → True
    self.lane_lateral_change = veh_width < lateral_diff
    
    # ============================================================
    # STEP 9: Check Lane ID Changes
    # ============================================================
    # LOGIC: Detect lane changes by comparing lane IDs
    # MEANING: Different lane_id = different lane
    # 
    # WHY CHECK LANE ID:
    # - Most reliable indicator of lane change
    # - Lane ID changes when crossing lane boundary
    # - More reliable than lateral offset (handles edge cases)
    # 
    # CONDITIONS:
    # - future_wpt.lane_id != current_wpt.lane_id: Future waypoint in different lane
    # - previous_wpt.lane_id != future_wpt.lane_id: Lane changed between past and future
    # 
    # EXAMPLE:
    # - current_wpt.lane_id = -1 (left lane)
    # - future_wpt.lane_id = -2 (right lane)
    # - lane_id_change = True (lane change detected)
    self.lane_id_change = (
            future_wpt.lane_id != current_wpt.lane_id or
            previous_wpt.lane_id != future_wpt.lane_id)
    
    # ============================================================
    # STEP 10: Set Curved Road Flag
    # ============================================================
    # LOGIC: Combine lane change indicators
    # MEANING: Road is curved OR lane change happening
    # 
    # WHY THIS FLAG:
    # - Curved roads need different spline handling
    # - Lane changes need smoother splines
    # - Both cases: Include more history waypoints
    # 
    # USAGE:
    # - If True: Include history waypoints in spline
    # - If False: Use simpler spline (straight road)
    self.potential_curved_road = self.lane_id_change or self.lane_lateral_change
    
    # ============================================================
    # STEP 11: Calculate Angle to First Waypoint
    # ============================================================
    # LOGIC: Check if first waypoint in buffer is ahead or behind
    # MEANING: Determine if waypoint is in front of vehicle
    # 
    # WHY CHECK:
    # - Need to know if waypoint is reachable
    # - Waypoints behind vehicle should be ignored
    # - Angle < 90° = waypoint ahead
    # - Angle > 90° = waypoint behind
    _, angle = cal_distance_angle(
        self._waypoint_buffer[0][0].transform.location,
        current_location,
        current_yaw)
    
    # ============================================================
    # STEP 12: Add History Waypoints to Spline
    # ============================================================
    # LOGIC: Include recently passed waypoints for smooth spline
    # MEANING: Past waypoints help create smooth transition
    # 
    # WHY HISTORY:
    # - Spline needs points before current position
    # - Creates smooth curve (no sudden direction changes)
    # - Especially important for curved roads/lane changes
    # 
    # LOOP: Process each waypoint in history buffer
    index = 0
    for i in range(len(self._history_buffer)):
        prev_wpt = self._history_buffer[i][0].transform.location
        
        # Calculate angle from history waypoint to current location
        _, angle = cal_distance_angle(
            prev_wpt, current_location, current_yaw)
        
        # CONDITION 1: Waypoint is behind AND not curved road
        # - angle > 90: Waypoint is behind vehicle
        # - not curved: Only use if road is straight
        # - WHY: On straight roads, behind waypoints help smoothness
        # - On curved roads, might cause issues (use different logic)
        if angle > 90 and not self.potential_curved_road:
            x.append(prev_wpt.x)
            y.append(prev_wpt.y)
            index += 1
        
        # CONDITION 2: Curved road (always include history)
        # - WHY: Curved roads need history for smooth curves
        # - Lane changes need history for smooth transitions
        if self.potential_curved_road:
            x.append(prev_wpt.x)
            y.append(prev_wpt.y)
            index += 1
    
    # ============================================================
    # STEP 13: Add Current Position to Spline
    # ============================================================
    # LOGIC: Add current vehicle position or nearby waypoint
    # MEANING: Start spline from current location
    # 
    # WHY TWO CASES:
    # - Curved road: Use actual position (more accurate)
    # - Straight road: Use waypoint (centered in lane)
    if self.potential_curved_road:
        # CURVED ROAD CASE:
        # - Use actual vehicle position
        # - WHY: More accurate for lane changes
        # - Check if we have any points yet
        _, angle = cal_distance_angle(
            self._waypoint_buffer[0][0].transform.location,
            current_location,
            current_yaw)
        
        # If no points added yet, add current position
        # WHY: Need at least one point for spline
        if len(x) == 0 or len(y) == 0:
            x.append(current_location.x)
            y.append(current_location.y)
    else:
        # STRAIGHT ROAD CASE:
        # - Prefer waypoint (centered in lane)
        # - WHY: Waypoint is always centered, vehicle might be off-center
        _, angle = cal_distance_angle(
            current_wpt_loc, current_location, current_yaw)
        
        # If waypoint is ahead, use waypoint
        # If waypoint is behind, use actual position
        # WHY: Don't use waypoint if it's behind us
        if angle < 90:
            # Waypoint ahead: Use waypoint (centered)
            x.append(current_wpt_loc.x)
            y.append(current_wpt_loc.y)
        else:
            # Waypoint behind: Use actual position
            x.append(current_location.x)
            y.append(current_location.y)
    
    # ============================================================
    # STEP 14: Filter and Add Future Waypoints
    # ============================================================
    # LOGIC: Add waypoints from buffer, filtering too-close ones
    # MEANING: Add future waypoints for spline interpolation
    # 
    # WHY FILTER:
    # - Waypoints too close together cause spline issues
    # - Minimum spacing: 0.5 meters
    # - Prevents numerical problems in spline calculation
    # 
    # INDEX ADJUSTMENT:
    # - For curved roads: Use index-1 (skip last history point)
    # - WHY: Avoid duplicate point at transition
    # - For straight roads: Use index (keep all points)
    index = max(0, index - 1) if self.potential_curved_road else index
    prev_x = x[index]
    prev_y = y[index]
    
    # Loop through waypoint buffer
    for i in range(len(self._waypoint_buffer)):
        cur_x = self._waypoint_buffer[i][0].transform.location.x
        cur_y = self._waypoint_buffer[i][0].transform.location.y
        
        # Check if waypoint is too close to previous
        # WHY: Prevent spline numerical issues
        if abs(prev_x - cur_x) < 0.5 and abs(prev_y - cur_y) < 0.5:
            continue  # Skip this waypoint (too close)
        
        # Update previous coordinates
        prev_x = cur_x
        prev_y = cur_y
        
        # Add waypoint to spline nodes
        x.append(cur_x)
        y.append(cur_y)
    
    # ============================================================
    # STEP 15: Create Spline and Interpolate
    # ============================================================
    # LOGIC: Generate smooth path using cubic spline
    # MEANING: Create continuous curve through waypoints
    # 
    # VALIDATION:
    # - Need at least 2 points for spline
    # - WHY: Spline needs at least 2 points to define curve
    # - If insufficient, return empty lists
    if len(x) < 2 or len(y) < 2:
        return rx, ry, rk, ryaw
    
    # Create 2D spline from waypoint coordinates
    # Spline2D creates smooth curve through (x, y) points
    # 
    # HOW IT WORKS:
    # - Spline2D internally creates two 1D splines:
    #   - sx: x-coordinate as function of arc length s
    #   - sy: y-coordinate as function of arc length s
    # - Arc length s: Distance along curve from start
    sp = Spline2D(x, y)
    
    # Calculate distance from current position to spline start
    # WHY: Only interpolate points after current position
    # 
    # diff_x, diff_y: Difference in coordinates
    # diff_s: Euclidean distance
    diff_x = current_location.x - sp.sx.y[0]  # sp.sx.y[0] = first x coordinate
    diff_y = current_location.y - sp.sy.y[0]  # sp.sy.y[0] = first y coordinate
    diff_s = np.hypot(diff_x, diff_y)
    
    # Generate parameter values for interpolation
    # s: Arc length along spline
    # Start from diff_s (current position)
    # End at sp.s[-1] (spline end)
    # Step by ds (0.1 meters)
    s = np.arange(diff_s, sp.s[-1], ds)
    
    # Initialize output lists
    self._long_plan_debug = []  # For debug visualization
    rx, ry, ryaw, rk = [], [], [], []
    
    # ============================================================
    # STEP 16: Interpolate Points Along Spline
    # ============================================================
    # LOGIC: Generate fine-grained points along spline curve
    # MEANING: Create smooth, continuous path
    # 
    # LOOP: For each arc length value s
    for (i, i_s) in enumerate(s):
        # Calculate position at arc length i_s
        # ix, iy: (x, y) coordinates on spline
        ix, iy = sp.calc_position(i_s)
        
        # Skip point if too close to first waypoint
        # WHY: Avoid duplicate point at start
        if abs(ix - x[index]) <= ds and abs(iy - y[index]) <= ds:
            continue
        
        # Store first half for debug visualization
        # WHY: Don't need to visualize entire path (too many points)
        if i <= len(s) // 2:
            self._long_plan_debug.append(
                carla.Transform(carla.Location(ix, iy, 0)))
        
        # Add coordinates to output
        rx.append(ix)
        ry.append(iy)
        
        # Calculate and clamp curvature
        # WHY: Limit extreme curvatures (unrealistic)
        # 
        # calc_curvature():
        # - Returns curvature in 1/meters
        # - Positive = left turn, Negative = right turn
        # - Large value = sharp turn
        # 
        # CLAMPING:
        # - min(..., 0.2): Maximum curvature 0.2 (sharp left)
        # - max(..., -0.2): Minimum curvature -0.2 (sharp right)
        # - WHY: Prevent unrealistic sharp turns
        rk.append(max(min(sp.calc_curvature(i_s), 0.2), -0.2))
        
        # Calculate yaw (heading direction)
        # calc_yaw(): Returns angle in radians
        # - 0 rad = East, π/2 rad = North, π rad = West, -π/2 rad = South
        ryaw.append(sp.calc_yaw(i_s))
    
    # Return interpolated path
    return rx, ry, rk, ryaw
```

### Flowchart

```
Start
  ↓
Initialize x, y lists (spline nodes)
  ↓
Filter problematic waypoints (buffer_filter)
  ↓
Get current location and yaw
  ↓
Get current waypoint from map
  ↓
Get future and past waypoints
  ↓
Calculate lateral offset (detect lane change)
  ↓
Compare with vehicle width → lane_lateral_change
  ↓
Check lane ID changes → lane_id_change
  ↓
Set potential_curved_road flag
  ↓
Add history waypoints to x, y:
  ├─ If angle > 90 and not curved: Add
  └─ If curved: Always add
  ↓
Add current position:
  ├─ If curved: Use actual position
  └─ If straight: Use waypoint if ahead, else position
  ↓
Filter and add future waypoints (skip if < 0.5m apart)
  ↓
If len(x) < 2: Return empty lists
  ↓
Create Spline2D from (x, y) points
  ↓
Calculate distance from current to spline start
  ↓
Generate arc length values s (from current to end, step 0.1m)
  ↓
For each s:
  ├─ Calculate position (ix, iy)
  ├─ Skip if too close to first waypoint
  ├─ Add to rx, ry
  ├─ Calculate and clamp curvature → rk
  └─ Calculate yaw → ryaw
  ↓
Return rx, ry, rk, ryaw
```

---

## Method 5: `generate_trajectory()`

### Purpose
Sample the smooth path and assign speed to each trajectory point based on curvature constraints and acceleration limits.

### Code with Detailed Comments

```python
def generate_trajectory(self, rx, ry, rk):
    """
    Sampling the generated path and assign speed to each point.
    
    PARAMETERS:
    - rx: list of float (x coordinates from generate_path)
    - ry: list of float (y coordinates from generate_path)
    - rk: list of float (curvatures from generate_path)
    
    WHY THIS METHOD:
    - Path from generate_path() has no speed information
    - Need to assign speeds based on:
      * Curvature (slow down for turns)
      * Acceleration limits (smooth speed changes)
      * Current speed (start from current)
    """
    
    # ============================================================
    # STEP 1: Set Interpolation Parameters
    # ============================================================
    # ds: Spatial resolution (distance between path points)
    # - VALUE: 0.1 meters
    # - MEANING: Path points are 0.1m apart
    # - WHY: Fine enough for smooth following
    ds = 0.1
    
    # dt: Temporal resolution (time between trajectory points)
    # - VALUE: From config (typically 0.1 seconds)
    # - MEANING: Trajectory points are dt seconds apart
    # - WHY: Control temporal spacing of trajectory
    dt = self.dt
    
    # ============================================================
    # STEP 2: Get Speed Targets
    # ============================================================
    # target_speed: Desired speed from behavior agent
    # - TYPE: float (km/h)
    # - SOURCE: Set by BehaviorAgent based on road conditions
    # - EXAMPLE: 60.0 km/h (speed limit)
    # 
    # current_speed: Current vehicle speed
    # - TYPE: float (km/h)
    # - SOURCE: From update_information()
    # - EXAMPLE: 55.0 km/h (current speed)
    target_speed = self._target_speed
    current_speed = self._ego_speed
    
    # ============================================================
    # STEP 3: Calculate Number of Samples
    # ============================================================
    # LOGIC: Sample trajectory for 2 seconds ahead
    # MEANING: Generate 2 seconds of trajectory points
    # 
    # FORMULA: sample_num = 2.0 / dt
    # - If dt = 0.1s: sample_num = 20 points
    # - If dt = 0.05s: sample_num = 40 points
    # 
    # WHY 2 SECONDS:
    # - Enough lookahead for smooth control
    # - Not too long (avoids outdated predictions)
    # - Good balance for real-time planning
    sample_num = 2.0 // dt
    
    # ============================================================
    # STEP 4: Initialize Variables
    # ============================================================
    break_flag = False  # Flag to stop sampling
    current_speed = current_speed / 3.6  # Convert km/h to m/s
    sample_resolution = 0  # Distance traveled along path
    
    # ============================================================
    # STEP 5: Calculate Curvature-Based Speed Limit
    # ============================================================
    # LOGIC: Limit speed based on path curvature
    # MEANING: Slow down for sharp turns
    # 
    # WHY CURVATURE LIMIT:
    # - Sharp turns require lower speeds
    # - High speed + high curvature = unsafe (vehicle might skid)
    # - Physics: v² ≤ a_lat_max / curvature
    # 
    # mean_k: Average curvature of path
    # - TYPE: float (1/meters)
    # - CALCULATION: statistics.mean(rk)
    # - FALLBACK: 0.0001 if less than 2 points (avoid division by zero)
    # 
    # EXAMPLE:
    # - rk = [0.01, 0.02, 0.015, 0.01] (gentle curve)
    # - mean_k = 0.01375 1/m
    # - Speed limit = sqrt(5.0 / 0.01375) * 3.6 ≈ 69.7 km/h
    # 
    # EXAMPLE (sharp turn):
    # - rk = [0.1, 0.15, 0.12] (sharp curve)
    # - mean_k = 0.123 1/m
    # - Speed limit = sqrt(5.0 / 0.123) * 3.6 ≈ 22.9 km/h
    # 
    # FORMULA: v ≤ sqrt(a_lat_max / k)
    # - a_lat_max = 5.0 m/s² (maximum lateral acceleration)
    # - k = curvature (1/m)
    # - Result in m/s, convert to km/h (× 3.6)
    mean_k = 0.0001 if len(rk) < 2 else abs(statistics.mean(rk))
    target_speed = min(target_speed, np.sqrt(5.0 / (mean_k + 10e-6)) * 3.6)
    
    # ============================================================
    # STEP 6: Calculate Acceleration
    # ============================================================
    # LOGIC: Determine acceleration to reach target speed
    # MEANING: How fast to change speed
    # 
    # WHY ACCELERATION LIMIT:
    # - Smooth speed changes (not instant)
    # - Realistic vehicle dynamics
    # - Comfortable for passengers
    # 
    # FORMULA: a = (v_target - v_current) / dt
    # - v_target: Target speed (m/s)
    # - v_current: Current speed (m/s)
    # - dt: Time step (s)
    # 
    # CLAMPING:
    # - max_acc = 3.5 m/s² (maximum acceleration)
    # - min acceleration = -6.5 m/s² (maximum deceleration/braking)
    # 
    # EXAMPLE:
    # - target_speed = 60 km/h = 16.67 m/s
    # - current_speed = 50 km/h = 13.89 m/s
    # - dt = 0.1s
    # - a = (16.67 - 13.89) / 0.1 = 27.8 m/s²
    # - Clamped to 3.5 m/s² (max acceleration)
    max_acc = 3.5
    acceleration = max(
        min(max_acc, (target_speed / 3.6 - current_speed) / dt), -6.5)
    
    # ============================================================
    # STEP 7: Sample Trajectory Points
    # ============================================================
    # LOGIC: Generate trajectory points with speed assignments
    # MEANING: Create time-stamped path points vehicle will follow
    # 
    # LOOP: For each time step (up to sample_num)
    for i in range(1, int(sample_num) + 1):
        # ============================================================
        # STEP 7a: Update Distance Traveled
        # ============================================================
        # LOGIC: Calculate how far vehicle travels in this time step
        # FORMULA: s = v₀t + ½at²
        # - v₀: Initial speed (current_speed)
        # - t: Time step (dt)
        # - a: Acceleration
        # 
        # EXAMPLE:
        # - current_speed = 15 m/s
        # - acceleration = 1.0 m/s²
        # - dt = 0.1s
        # - sample_resolution += 15 × 0.1 + 0.5 × 1.0 × 0.1²
        # - sample_resolution += 1.5 + 0.005 = 1.505m
        sample_resolution += current_speed * dt + \
                             0.5 * acceleration * dt ** 2
        
        # ============================================================
        # STEP 7b: Update Current Speed
        # ============================================================
        # LOGIC: Speed changes due to acceleration
        # FORMULA: v = v₀ + at
        # 
        # EXAMPLE:
        # - current_speed = 15 m/s
        # - acceleration = 1.0 m/s²
        # - dt = 0.1s
        # - current_speed = 15 + 1.0 × 0.1 = 15.1 m/s
        current_speed += acceleration * dt
        
        # ============================================================
        # STEP 7c: Find Corresponding Path Point
        # ============================================================
        # LOGIC: Map distance traveled to path point index
        # MEANING: Which path point corresponds to this distance?
        # 
        # CALCULATION:
        # - sample_resolution: Distance traveled (meters)
        # - ds: Distance between path points (0.1m)
        # - Index = sample_resolution / ds - 1
        # 
        # WHY -1:
        # - Index 0 = first point (at distance 0)
        # - sample_resolution / ds gives next point
        # - Subtract 1 to get current point
        # 
        # EXAMPLE:
        # - sample_resolution = 1.5m
        # - ds = 0.1m
        # - Index = 1.5 / 0.1 - 1 = 15 - 1 = 14
        # - Use path point at index 14
        if int(sample_resolution // ds - 1) >= len(rx):
            # Reached end of path
            # Use last path point
            sample_x = rx[-1]
            sample_y = ry[-1]
            break_flag = True
        else:
            # Get path point at calculated index
            # max(0, ...): Ensure index is not negative
            sample_x = rx[max(0, int(sample_resolution // ds - 1))]
            sample_y = ry[max(0, int(sample_resolution // ds - 1))]
        
        # ============================================================
        # STEP 7d: Create Trajectory Point
        # ============================================================
        # LOGIC: Create trajectory point with position and speed
        # MEANING: Point vehicle should reach at specific time
        # 
        # STRUCTURE: (Transform, speed)
        # - Transform: Position and orientation
        # - speed: Target speed at this point (km/h)
        # 
        # Z COORDINATE:
        # - Use waypoint z coordinate + 0.5m
        # - WHY: Slight elevation for visualization
        # - 0.5m above ground
        self._trajectory_buffer.append(
            (carla.Transform(
                carla.Location(
                    sample_x,
                    sample_y,
                    self._waypoint_buffer[0][0].transform.location.z +
                    0.5)),
             target_speed))
        
        # Break if reached end of path
        if break_flag:
            break
```

### Flowchart

```
Start
  ↓
Set ds = 0.1m, dt from config
  ↓
Get target_speed and current_speed
  ↓
Calculate sample_num = 2.0 / dt
  ↓
Convert current_speed to m/s
  ↓
Calculate mean curvature from rk
  ↓
Limit target_speed by curvature: min(target, sqrt(5.0/k)*3.6)
  ↓
Calculate acceleration: clamp((target-current)/dt, -6.5, 3.5)
  ↓
For i = 1 to sample_num:
  ├─ Update distance: s += v×dt + 0.5×a×dt²
  ├─ Update speed: v += a×dt
  ├─ Calculate path index: int(s/ds - 1)
  ├─ If index >= len(rx):
  │   ├─ Use last point
  │   └─ Set break_flag = True
  ├─ Else:
  │   └─ Get point at index
  ├─ Create Transform with (x, y, z+0.5)
  ├─ Append (Transform, target_speed) to buffer
  └─ If break_flag: break
  ↓
End (Trajectory generated)
```

---

## Method 6: `buffer_filter()`

### Purpose
Remove waypoints from the buffer that could cause bad vehicle dynamics, such as waypoints behind the vehicle or waypoints too close during lane changes.

### Code with Detailed Comments

```python
def buffer_filter(self):
    """
    Remove the waypoints in the global route plan which has dramatic
    change of yaw angle. Such waypoint can cause bad vehicle dynamics.
    
    WHY THIS METHOD:
    - Some waypoints are problematic:
      * Behind vehicle (already passed)
      * Too close during lane changes (causes sharp steering)
    - Filter removes these before path generation
    - Prevents jerky, unstable vehicle motion
    """
    
    # ============================================================
    # STEP 1: Initialize
    # ============================================================
    prev_wpt = None  # Previous waypoint for comparison
    tmp = self._waypoint_buffer.copy()  # Copy to iterate safely
    
    # WHY COPY:
    # - Modifying list while iterating is dangerous
    # - Copy allows safe iteration
    # - Original buffer can be modified during iteration
    
    # ============================================================
    # STEP 2: Check First Few Waypoints
    # ============================================================
    # LOGIC: Only check first 3 waypoints (most critical)
    # MEANING: These are the immediate next waypoints
    # 
    # WHY ONLY 3:
    # - First waypoints most likely to cause issues
    # - Far waypoints less critical (will be filtered later)
    # - Performance: Don't check entire buffer
    for i, (waypoint, _) in enumerate(tmp):
        if i >= 3:
            break  # Only check first 3 waypoints
        
        # ============================================================
        # STEP 3: Calculate Index in Original Buffer
        # ============================================================
        # LOGIC: Buffer might have been modified, need correct index
        # MEANING: Find index in actual buffer (not copy)
        # 
        # WHY CALCULATE:
        # - Copy has original length
        # - Original buffer might be shorter (elements removed)
        # - Need to map copy index to original index
        # 
        # FORMULA: j = i - (len(copy) - len(original))
        # - If no removals: j = i
        # - If 1 removed before i: j = i - 1
        j = i - (len(tmp) - len(self._waypoint_buffer))
        
        # ============================================================
        # STEP 4: Check if Waypoint is Behind Vehicle
        # ============================================================
        # LOGIC: Remove waypoints that vehicle has already passed
        # MEANING: Waypoints behind vehicle are useless
        # 
        # HOW IT WORKS:
        # - Calculate angle from waypoint to vehicle
        # - Angle > 90° means waypoint is behind vehicle
        # - Remove such waypoints
        # 
        # EXAMPLE:
        # - Vehicle heading: 0° (North)
        # - Waypoint at 180° (South) relative to vehicle
        # - Angle = 180° > 90° → Behind vehicle → Remove
        _, angle = cal_distance_angle(
            waypoint.transform.location,
            self._ego_pos.location, self._ego_pos.rotation.yaw)
        
        if angle > 90:
            # Waypoint is behind vehicle
            # Remove from buffer
            del self._waypoint_buffer[j]
            continue  # Skip to next waypoint
        
        # ============================================================
        # STEP 5: Initialize Previous Waypoint
        # ============================================================
        # LOGIC: Need previous waypoint for distance check
        # MEANING: First waypoint becomes "previous" for next iteration
        if prev_wpt is None:
            prev_wpt = waypoint
            continue  # First waypoint, no comparison yet
        
        # ============================================================
        # STEP 6: Check Lane Change Distance
        # ============================================================
        # LOGIC: Remove waypoints too close during lane changes
        # MEANING: Prevent sharp steering during lane changes
        # 
        # WHY THIS CHECK:
        # - Lane changes need gradual steering
        # - Waypoint too close = sharp steering angle
        # - Sharp steering = unstable, jerky motion
        # 
        # CONDITION 1: Different lane IDs
        # - prev_wpt.lane_id != waypoint.lane_id
        # - MEANING: Waypoint is in different lane (lane change)
        # 
        # CONDITION 2: Buffer has at least 2 waypoints
        # - len(self._waypoint_buffer) >= 2
        # - WHY: Need waypoints ahead to replace removed one
        if prev_wpt.lane_id != waypoint.lane_id and \
                len(self._waypoint_buffer) >= 2:
            # Calculate distance between waypoints
            dist = compute_distance(waypoint.transform.location,
                                    prev_wpt.transform.location)
            
            # If distance <= 4.5 meters, remove waypoint
            # WHY 4.5m:
            # - Typical waypoint spacing: 2m
            # - 4.5m = ~2 waypoints spacing
            # - Too close = sharp turn required
            # 
            # EXAMPLE:
            # - prev_wpt at (100, 200) in lane -1
            # - waypoint at (102, 200) in lane -2 (different lane)
            # - Distance = 2m < 4.5m → Remove waypoint
            # - Vehicle will use next waypoint (further, smoother)
            if dist <= 4.5:
                del self._waypoint_buffer[j]
        
        # Update previous waypoint for next iteration
        prev_wpt = waypoint
```

### Flowchart

```
Start
  ↓
Initialize prev_wpt = None
  ↓
Create copy of _waypoint_buffer
  ↓
For i = 0 to 2 (first 3 waypoints):
  ├─ Calculate index j in original buffer
  ├─ Calculate angle from waypoint to vehicle
  ├─ If angle > 90°:
  │   ├─ Delete waypoint from buffer
  │   └─ Continue to next
  ├─ If prev_wpt is None:
  │   ├─ Set prev_wpt = waypoint
  │   └─ Continue to next
  ├─ If lane_id changed AND buffer has >= 2 waypoints:
  │   ├─ Calculate distance between waypoints
  │   ├─ If distance <= 4.5m:
  │   │   └─ Delete waypoint from buffer
  │   └─ Update prev_wpt = waypoint
  └─ End loop
  ↓
End (Problematic waypoints removed)
```

---

## Method 7: `pop_buffer()`

### Purpose
Remove waypoints and trajectory points that the vehicle has already passed (reached). This keeps the buffers clean and up-to-date.

### Code with Detailed Comments

```python
def pop_buffer(self, vehicle_transform):
    """
    Remove waypoints the ego vehicle has achieved.
    
    PARAMETERS:
    - vehicle_transform: carla.Transform (current vehicle position)
    
    WHY THIS METHOD:
    - Vehicle moves forward, passes waypoints
    - Need to remove passed waypoints (no longer needed)
    - Keeps buffers clean and efficient
    - Moves waypoints to history buffer for spline generation
    """
    
    # ============================================================
    # STEP 1: Find Maximum Index of Passed Waypoints
    # ============================================================
    # LOGIC: Find all waypoints vehicle has reached
    # MEANING: Waypoints within min_distance threshold
    # 
    # max_index: Highest index of waypoint vehicle has reached
    # - Start at -1 (no waypoints reached yet)
    # - Will be updated as we find reached waypoints
    max_index = -1
    
    # Loop through waypoint buffer
    for i, (waypoint, _) in enumerate(self._waypoint_buffer):
        # Calculate distance from vehicle to waypoint
        # distance_vehicle(): Returns 2D Euclidean distance
        # 
        # EXAMPLE:
        # - Vehicle at (100, 200)
        # - Waypoint at (102, 200)
        # - Distance = 2.0m
        # - min_distance = 2.5m
        # - 2.0 < 2.5 → Waypoint reached
        if distance_vehicle(
                waypoint, vehicle_transform) < self._min_distance:
            # Vehicle has reached this waypoint
            # Update max_index to this index
            max_index = i
    
    # ============================================================
    # STEP 2: Remove Passed Waypoints
    # ============================================================
    # LOGIC: Remove all waypoints up to max_index
    # MEANING: Remove waypoints vehicle has passed
    # 
    # WHY REMOVE:
    # - Waypoints already reached are no longer needed
    # - Keeps buffer clean
    # - Moves waypoints to history (for spline generation)
    if max_index >= 0:
        # Remove waypoints from index 0 to max_index
        for i in range(max_index + 1):
            # Check if history buffer exists
            if self._history_buffer:
                # Get last waypoint in history
                prev_wpt = self._history_buffer[-1]
                
                # Get waypoint being removed
                incoming_wpt = self._waypoint_buffer.popleft()
                
                # ============================================================
                # STEP 2a: Check Distance Between History and Incoming
                # ============================================================
                # LOGIC: Only add to history if significant distance
                # MEANING: Avoid duplicate waypoints in history
                # 
                # WHY CHECK DISTANCE:
                # - If waypoints are very close, skip adding
                # - Prevents history buffer from having duplicate points
                # - 4.5m threshold: ~2 waypoint spacing
                # 
                # EXAMPLE:
                # - prev_wpt at (100, 200)
                # - incoming_wpt at (102, 200)
                # - Distance = 2m < 4.5m → Don't add to history
                # 
                # EXAMPLE (far apart):
                # - prev_wpt at (100, 200)
                # - incoming_wpt at (110, 200)
                # - Distance = 10m > 4.5m → Add to history
                if abs(
                        prev_wpt[0].transform.location.x -
                        incoming_wpt[0].transform.location.x) > 4.5 or abs(
                    prev_wpt[0].transform.location.y -
                    incoming_wpt[0].transform.location.y) > 4.5:
                    # Waypoints are far apart, add to history
                    self._history_buffer.append(incoming_wpt)
            else:
                # No history yet, add first waypoint
                # WHY: History buffer needs at least one waypoint
                self._history_buffer.append(
                    self._waypoint_buffer.popleft())
    
    # ============================================================
    # STEP 3: Remove Passed Trajectory Points
    # ============================================================
    # LOGIC: Also remove trajectory points vehicle has passed
    # MEANING: Clean trajectory buffer as well
    # 
    # WHY SEPARATE CHECK:
    # - Trajectory points are different from waypoints
    # - Different distance threshold (min_distance - 1, minimum 1m)
    # - Trajectory points are more frequent (every 0.1m)
    if self._trajectory_buffer:
        max_index = -1
        
        # Find passed trajectory points
        for i, (waypoint, _,) in enumerate(self._trajectory_buffer):
            # Distance threshold: max(min_distance - 1, 1)
            # WHY: Trajectory points closer together, use smaller threshold
            # 
            # EXAMPLE:
            # - min_distance = 2.5m
            # - Threshold = max(2.5 - 1, 1) = 1.5m
            if distance_vehicle(
                    waypoint, vehicle_transform) < \
                    max(self._min_distance - 1, 1):
                max_index = i
        
        # Remove passed trajectory points
        if max_index >= 0:
            for i in range(max_index + 1):
                self._trajectory_buffer.popleft()
```

### Flowchart

```
Start
  ↓
Initialize max_index = -1
  ↓
For each waypoint in _waypoint_buffer:
  ├─ Calculate distance to vehicle
  ├─ If distance < min_distance:
  │   └─ Update max_index = i
  └─ Continue
  ↓
If max_index >= 0:
  ├─ For i = 0 to max_index:
  │   ├─ If history_buffer exists:
  │   │   ├─ Get last history waypoint
  │   │   ├─ Pop waypoint from buffer
  │   │   ├─ Calculate distance between history and waypoint
  │   │   ├─ If distance > 4.5m:
  │   │   │   └─ Add waypoint to history
  │   │   └─ Else: Skip (too close)
  │   └─ Else:
  │       └─ Add waypoint to history (first one)
  └─ End
  ↓
If trajectory_buffer exists:
  ├─ Find max_index of passed trajectory points
  ├─ If max_index >= 0:
  │   └─ Remove points 0 to max_index
  └─ End
  ↓
End (Buffers cleaned)
```

---

## Method 8: `run_step()`

### Purpose
Main execution method called every simulation step. Manages waypoint buffering, trajectory generation, and returns the next target for vehicle control.

### Code with Detailed Comments

```python
def run_step(
        self,
        rx,
        ry,
        rk,
        target_speed=None,
        trajectory=None,
        following=False):
    """
    Execute one step of local planning.
    
    PARAMETERS:
    - rx, ry, rk: Path coordinates and curvatures from generate_path()
    - target_speed: Desired speed (km/h)
    - trajectory: Pre-generated trajectory (for platooning)
    - following: Boolean, whether vehicle is car-following
    
    RETURNS:
    - target_speed: float (km/h)
    - target_location: carla.Location
    
    WHY THIS METHOD:
    - Main interface for behavior agent
    - Called every simulation step
    - Coordinates all planning activities
    """
    
    # ============================================================
    # STEP 1: Store Target Speed
    # ============================================================
    # LOGIC: Save target speed for trajectory generation
    # MEANING: Desired speed from behavior agent
    self._target_speed = target_speed
    
    # ============================================================
    # STEP 2: Refill Waypoint Buffer if Needed
    # ============================================================
    # LOGIC: Keep waypoint buffer filled for path generation
    # MEANING: Ensure enough waypoints available
    # 
    # CONDITION: Buffer has fewer waypoints than update frequency
    # WHY: Don't refill every step (unnecessary)
    # 
    # EXAMPLE:
    # - waypoint_update_freq = 5
    # - Buffer has 3 waypoints
    # - 3 < 5 → Refill buffer
    if len(self._waypoint_buffer) < self.waypoint_update_freq:
        # Calculate how many waypoints to add
        # Add until buffer is full (buffer_size)
        for i in range(self._buffer_size - len(self._waypoint_buffer)):
            if self.waypoints_queue:
                # Pop waypoint from queue
                # Add to buffer
                self._waypoint_buffer.append(
                    self.waypoints_queue.popleft())
            else:
                # No more waypoints available
                break
    
    # ============================================================
    # STEP 3: Generate Trajectory if Needed
    # ============================================================
    # LOGIC: Generate trajectory when buffer is low
    # MEANING: Create smooth trajectory points with speeds
    # 
    # CONDITION 1: No pre-generated trajectory
    # - trajectory is None (not provided)
    # 
    # CONDITION 2: Trajectory buffer is low
    # - len(_trajectory_buffer) < trajectory_update_freq
    # 
    # CONDITION 3: Not in following mode
    # - following = False
    # 
    # WHY THESE CONDITIONS:
    # - If trajectory provided, use it (platooning)
    # - If buffer full, don't regenerate (wasteful)
    # - If following, use provided trajectory
    if not trajectory and len(
            self._trajectory_buffer) < self.trajectory_update_freq and \
            not following:
        # Clear old trajectory
        self._trajectory_buffer.clear()
        
        # Check if path is valid
        if len(rx) == 0:
            # No path available
            # Return zero speed and no target
            return 0, None
        
        # Generate new trajectory from path
        self.generate_trajectory(rx, ry, rk)
    elif trajectory:
        # Pre-generated trajectory provided (platooning)
        # Use it directly
        self._trajectory_buffer = trajectory.copy()
    
    # ============================================================
    # STEP 4: Get Target Waypoint
    # ============================================================
    # LOGIC: Get next waypoint vehicle should follow
    # MEANING: Immediate target for vehicle control
    # 
    # INDEX: min(1, len - 1)
    # - Prefer index 1 (second point, slight lookahead)
    # - If only 1 point, use index 0
    # 
    # WHY INDEX 1:
    # - Index 0 might be too close (already reached)
    # - Index 1 provides slight lookahead (smoother)
    # - Still close enough for responsive control
    self.target_waypoint, self._target_speed = \
        self._trajectory_buffer[min(1, len(self._trajectory_buffer) - 1)]
    
    # ============================================================
    # STEP 5: Clean Passed Waypoints
    # ============================================================
    # LOGIC: Remove waypoints vehicle has reached
    # MEANING: Keep buffers clean and up-to-date
    vehicle_transform = self._ego_pos
    self.pop_buffer(vehicle_transform)
    
    # ============================================================
    # STEP 6: Debug Visualization (Optional)
    # ============================================================
    # LOGIC: Draw trajectory and waypoints for debugging
    # MEANING: Visualize planning in CARLA
    if self.debug_trajectory:
        # Draw generated path (green)
        draw_trajetory_points(self._vehicle.get_world(),
                              self._long_plan_debug,
                              color=carla.Color(0, 255, 0),
                              size=0.05,
                              lt=0.1)
    
    if self.debug:
        # Draw waypoint buffer (blue)
        draw_trajetory_points(self._vehicle.get_world(),
                              self._waypoint_buffer,
                              z=0.1,
                              size=0.1,
                              color=carla.Color(0, 0, 255),
                              lt=0.2)
        # Draw history buffer (magenta)
        draw_trajetory_points(self._vehicle.get_world(),
                              self._history_buffer,
                              z=0.1,
                              size=0.1,
                              color=carla.Color(255, 0, 255),
                              lt=0.2)
    
    # ============================================================
    # STEP 7: Return Target
    # ============================================================
    # LOGIC: Return target speed and location for control
    # MEANING: What vehicle should do next
    # 
    # target_waypoint:
    # - Has 'is_junction' attribute: Use transform.location
    # - No 'is_junction': Use location directly
    # 
    # WHY CHECK ATTRIBUTE:
    # - Different waypoint types have different structures
    # - Need to handle both cases
    return self._target_speed, \
           self.target_waypoint.transform.location if hasattr(
               self.target_waypoint,
               'is_junction') else self.target_waypoint.location
```

### Flowchart

```
Start
  ↓
Store target_speed
  ↓
If waypoint_buffer length < waypoint_update_freq:
  ├─ Refill buffer from waypoints_queue
  └─ End
  ↓
If no trajectory AND buffer low AND not following:
  ├─ Clear trajectory buffer
  ├─ If rx is empty: Return (0, None)
  ├─ Generate trajectory from path
  └─ End
Else if trajectory provided:
  └─ Use provided trajectory
  ↓
Get target waypoint from trajectory buffer (index 1 or 0)
  ↓
Clean passed waypoints (pop_buffer)
  ↓
If debug enabled: Draw visualization
  ↓
Return (target_speed, target_location)
```

---

## Overall File Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                    LocalPlanner Lifecycle                    │
└─────────────────────────────────────────────────────────────┘

INITIALIZATION (__init__)
  ↓
  Create data structures (queues, buffers)
  ↓
  Load configuration parameters
  ↓

SETUP (set_global_plan)
  ↓
  Receive waypoints from global planner
  ↓
  Store in waypoints_queue
  ↓
  Optionally refill _waypoint_buffer
  ↓

MAIN LOOP (Every Simulation Step)
  ↓
  ┌─────────────────────────────────────┐
  │  UPDATE STATE (update_information)  │
  │  - Update ego position              │
  │  - Update ego speed                 │
  └─────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────┐
  │  GENERATE PATH (generate_path)      │
  │  1. Filter waypoints (buffer_filter)│
  │  2. Detect lane changes             │
  │  3. Collect waypoint coordinates    │
  │  4. Create spline                   │
  │  5. Interpolate smooth path         │
  │  6. Calculate curvature              │
  │  Returns: rx, ry, rk, ryaw          │
  └─────────────────────────────────────┘
  ↓
  ┌─────────────────────────────────────┐
  │  EXECUTE STEP (run_step)            │
  │  1. Refill waypoint buffer if needed │
  │  2. Generate trajectory if needed    │
  │     (generate_trajectory)            │
  │     - Sample path with time         │
  │     - Assign speeds                 │
  │     - Apply curvature limits        │
  │  3. Get target waypoint              │
  │  4. Clean passed waypoints           │
  │     (pop_buffer)                     │
  │  5. Return target for control        │
  └─────────────────────────────────────┘
  ↓
  Control module uses target
  ↓
  Vehicle moves
  ↓
  (Repeat main loop)
```

---

## Data Flow Diagram

```
┌─────────────────┐
│ Global Planner  │
│  (waypoints)    │
└────────┬────────┘
         │
         │ set_global_plan()
         ↓
┌─────────────────┐
│ waypoints_queue │ (Complete route)
└────────┬────────┘
         │
         │ Refill when needed
         ↓
┌─────────────────┐
│_waypoint_buffer │ (Working set, ~15 waypoints)
└────────┬────────┘
         │
         │ generate_path()
         ↓
┌─────────────────┐
│  Spline Path    │ (rx, ry, rk, ryaw)
│  (smooth curve) │
└────────┬────────┘
         │
         │ generate_trajectory()
         ↓
┌─────────────────┐
│_trajectory_buffer│ (Time-stamped points with speeds)
└────────┬────────┘
         │
         │ run_step()
         ↓
┌─────────────────┐
│  Control Module │ (Vehicle follows trajectory)
└─────────────────┘
```

---

## Key Concepts Summary

### 1. **Waypoint Management**
- **waypoints_queue**: Complete route (source)
- **_waypoint_buffer**: Working set (next ~15 waypoints)
- **_history_buffer**: Recently passed waypoints (for spline)

### 2. **Path Generation**
- **Discrete waypoints** → **Smooth spline path**
- Uses cubic spline interpolation
- Handles lane changes and curved roads specially

### 3. **Trajectory Generation**
- **Smooth path** → **Time-stamped trajectory with speeds**
- Speed limited by curvature
- Acceleration constraints applied

### 4. **Buffer Management**
- Filter problematic waypoints
- Remove passed waypoints
- Refill buffers as needed

### 5. **Integration**
- Called by BehaviorAgent every step
- Provides target for ControlManager
- Handles platooning trajectories

---

## Relationship to Other Modules

```
BehaviorAgent
    │
    ├─→ Calls: generate_path() → Returns smooth path
    │
    ├─→ Calls: run_step() → Returns target for control
    │
    └─→ Manages: set_global_plan(), update_information()

LocalPlanner
    │
    ├─→ Uses: Spline2D (spline interpolation)
    │
    ├─→ Uses: cal_distance_angle() (geometry calculations)
    │
    └─→ Provides: Trajectory to ControlManager
```

---

This completes the explanation of `local_planner_behavior.py`. The LocalPlanner is a critical component that bridges high-level route planning with low-level vehicle control, generating smooth, drivable trajectories that vehicles can follow safely and comfortably.

