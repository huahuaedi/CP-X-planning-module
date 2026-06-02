# Explanation: `collision_check.py`

## Overview

The `CollisionChecker` class provides collision detection capabilities for autonomous vehicles in CARLA. It uses geometric methods to check if planned trajectories will collide with other vehicles, enabling safe lane changes, overtaking, and platoon joining maneuvers.

**Key Purpose**: Prevent collisions by checking if planned paths intersect with obstacle vehicles before executing maneuvers.

---

## Class Structure

```python
CollisionChecker
├── __init__()                    # Initialize collision checker parameters
├── is_in_range()                 # Check for obstacles during platoon back-joining
├── adjacent_lane_collision_check() # Generate collision check path for lane changes
└── collision_circle_check()     # Main collision detection using circle method
```

---

## Method 1: `__init__()`

### Purpose
Initialize the collision checker with configurable parameters that control how far ahead to check and how to represent the vehicle's collision shape.

### Code with Detailed Comments

```python
def __init__(self, time_ahead=1.2, circle_radius=1.0, circle_offsets=None):
    # ============================================================
    # PARAMETER: time_ahead
    # ============================================================
    # MEANING: How many seconds into the future we check for collisions
    # WHY: Vehicles are moving, so we need to predict where obstacles will be
    # 
    # EXAMPLE:
    # - time_ahead = 1.2 seconds
    # - Vehicle speed = 20 m/s
    # - Check distance = 1.2 * 20 = 24 meters ahead
    # 
    # TRADE-OFF:
    # - Too small (< 0.5s): Not enough reaction time, dangerous
    # - Too large (> 2.0s): Too conservative, might prevent safe maneuvers
    # - 1.2s is a good balance: ~1 second reaction + 0.2s buffer
    self.time_ahead = time_ahead
    
    # ============================================================
    # PARAMETER: circle_offsets
    # ============================================================
    # MEANING: Multiple circles placed along the vehicle's width for collision checking
    # WHY: Vehicles are rectangular, but checking rectangles is computationally expensive
    #      Using multiple circles approximates the vehicle shape more accurately
    # 
    # DEFAULT: [-1.0, 0, 1.0] meters from center
    # - -1.0m: Left side of vehicle (relative to heading)
    # - 0m: Center of vehicle
    # - 1.0m: Right side of vehicle
    # 
    # VISUAL:
    #     [Left Circle]  [Center Circle]  [Right Circle]
    #           ↓              ↓               ↓
    #        (-1,0)          (0,0)           (1,0)
    # 
    # WHY THREE CIRCLES:
    # - Single circle at center: Misses collisions at vehicle corners
    # - Two circles: Better but still misses some edge cases
    # - Three circles: Good approximation of rectangular vehicle
    # - More circles: Better accuracy but slower computation
    # 
    # EXAMPLE:
    # Vehicle width = 2 meters
    # - Left circle at -1m covers left edge
    # - Center circle at 0m covers center
    # - Right circle at +1m covers right edge
    # Together they approximate the 2m wide vehicle
    self._circle_offsets = [-1.0, 0, 1.0] if circle_offsets is None else circle_offsets
    
    # ============================================================
    # PARAMETER: circle_radius
    # ============================================================
    # MEANING: Radius of each collision checking circle
    # WHY: Provides safety buffer around vehicle
    # 
    # DEFAULT: 1.0 meter
    # - This is approximately half the width of a typical car
    # - Creates safety margin: even if circles don't touch, we detect collision
    # 
    # TRADE-OFF:
    # - Too small: Might miss collisions, unsafe
    # - Too large: Too conservative, prevents safe maneuvers
    # - 1.0m is reasonable for typical passenger vehicles
    self._circle_radius = circle_radius
```

### Logic Explanation

**Why use circles instead of rectangles?**
- **Computational efficiency**: Circle-to-point distance is simple: `sqrt((x1-x2)² + (y1-y2)²)`
- **Rectangle collision**: Requires checking if point is inside rotated rectangle (more complex)
- **Multiple circles**: Approximate rectangle shape with simpler geometry
- **Good enough**: For collision detection, approximation is acceptable (safety margin helps)

**Why time-based lookahead?**
- **Dynamic obstacles**: Other vehicles are moving, so static distance isn't enough
- **Speed-dependent**: Faster vehicles need to check further ahead
- **Reaction time**: Need time to detect, decide, and react to obstacles
- **Formula**: `distance = time_ahead × speed` ensures consistent safety margin

---

## Method 2: `is_in_range()`

### Purpose
Check if a candidate vehicle is blocking the path between ego vehicle and target vehicle during platoon back-joining. This is used when a vehicle wants to join a platoon from behind.

### Code with Detailed Comments

```python
def is_in_range(
        self,
        ego_pos,
        target_vehicle,
        candidate_vehicle,
        carla_map):
    """
    Check whether there is a obstacle vehicle between target_vehicle
    and ego_vehicle during back_joining.
    
    SCENARIO: Platoon joining from behind
    - Ego vehicle wants to join platoon
    - Target vehicle is the platoon member to catch up with
    - Candidate vehicle is a potential obstacle blocking the path
    - Need to check if candidate is between ego and target
    """
    
    # ============================================================
    # STEP 1: Extract Locations
    # ============================================================
    # WHY: Need coordinates to check spatial relationships
    ego_loc = ego_pos.location
    target_loc = target_vehicle.get_location()
    candidate_loc = candidate_vehicle.get_location()
    
    # ============================================================
    # STEP 2: Create Bounding Rectangle
    # ============================================================
    # LOGIC: Define a rectangle that spans from ego to target
    # MEANING: Any vehicle inside this rectangle is "between" ego and target
    # 
    # WHY RECTANGLE:
    # - Simple geometric check: Is point inside rectangle?
    # - Fast computation: Just compare x and y coordinates
    # - Good enough for initial filtering
    # 
    # VISUAL:
    #     Ego (100, 200)                    Target (300, 200)
    #         ●──────────────────────────────────●
    #         │                                  │
    #         │      Candidate (200, 200)       │
    #         │            ●                    │
    #         │                                  │
    #         └──────────────────────────────────┘
    # 
    # min_x = 100, max_x = 300
    # min_y = 200, max_y = 200
    min_x, max_x = min(ego_loc.x, target_loc.x), max(ego_loc.x, target_loc.x)
    min_y, max_y = min(ego_loc.y, target_loc.y), max(ego_loc.y, target_loc.y)
    
    # ============================================================
    # STEP 3: Quick Rejection with Buffer
    # ============================================================
    # LOGIC: If candidate is clearly outside the rectangle (with buffer), it's not blocking
    # MEANING: Fast early exit for vehicles that are definitely not in the way
    # 
    # WHY 2 METER BUFFER:
    # - Accounts for vehicle width (vehicles are ~2m wide)
    # - Accounts for slight position errors
    # - Prevents false negatives (saying "no obstacle" when there is one)
    # 
    # EXAMPLE:
    # - Rectangle: x=[100, 300], y=[200, 200]
    # - Candidate at (50, 200): x=50 < 100-2 = 98 → OUTSIDE → return False
    # - Candidate at (150, 200): x=150 is between 98 and 302 → CONTINUE CHECK
    # 
    # WHY EARLY EXIT:
    # - Performance: Avoid expensive waypoint lookups for vehicles far away
    # - Most vehicles will be rejected here (not in the path)
    if candidate_loc.x <= min_x - 2 or candidate_loc.x >= max_x + 2 or \
            candidate_loc.y <= min_y - 2 or candidate_loc.y >= max_y + 2:
        return False  # Candidate is clearly not blocking
    
    # ============================================================
    # STEP 4: Get Waypoints for Lane Information
    # ============================================================
    # WHY: Need to know which lane each vehicle is in
    # - Same lane = definitely blocking
    # - Different lanes = might not be blocking (depends on angle)
    candidate_wpt = carla_map.get_waypoint(candidate_loc)
    target_wpt = carla_map.get_waypoint(target_loc)
    
    # ============================================================
    # STEP 5: Check Same Lane (Most Common Case)
    # ============================================================
    # LOGIC: If candidate and target are in the same lane, candidate is blocking
    # MEANING: Vehicle in same lane between ego and target = obstacle
    # 
    # WHY THIS CHECK FIRST:
    # - Most common scenario: Obstacle is in the same lane
    # - Fast check: Just compare lane_id
    # - Clear decision: Same lane = definitely blocking
    # 
    # EXAMPLE:
    # - Target in lane -1 (left lane)
    # - Candidate in lane -1 (left lane)
    # - Ego wants to join from behind
    # - Candidate is blocking the path → return True
    if target_wpt.lane_id == candidate_wpt.lane_id:
        return True  # Definitely blocking
    
    # ============================================================
    # STEP 6: Check Different Sections (Edge Case)
    # ============================================================
    # LOGIC: If in same section but different lanes, not blocking
    # MEANING: Road sections can have multiple lanes side-by-side
    # 
    # WHY THIS CHECK:
    # - CARLA map structure: Roads have sections, sections have lanes
    # - Same section + different lanes = parallel lanes (not blocking)
    # - Different sections = different road segments (might be blocking)
    # 
    # EXAMPLE:
    # - Highway with 3 lanes in same section
    # - Target in lane -1, candidate in lane -2
    # - Same section, different lanes = parallel, not blocking
    # - Return False (not blocking)
    if target_wpt.section_id == candidate_wpt.section_id:
        return False  # Different lanes in same section = parallel, not blocking
    
    # ============================================================
    # STEP 7: Angle-Based Check (Final Decision)
    # ============================================================
    # LOGIC: Check if candidate is ahead of target (blocking) or behind (not blocking)
    # MEANING: Use angle to determine relative position along the road
    # 
    # WHY ANGLE CHECK:
    # - Vehicles might be in different road segments
    # - Need to know if candidate is in front of target (blocking) or behind
    # - Angle tells us direction: small angle = ahead, large angle = behind
    # 
    # HOW IT WORKS:
    # - cal_distance_angle() calculates angle from target to candidate
    # - Angle is relative to candidate's heading direction
    # - Angle <= 3° means candidate is roughly ahead of target
    # 
    # EXAMPLE:
    # - Target heading: 0° (North)
    # - Candidate at 45° from target: angle = 45° > 3° → Not blocking (behind/side)
    # - Candidate at 2° from target: angle = 2° <= 3° → Blocking (ahead)
    # 
    # WHY 3 DEGREES:
    # - Small threshold accounts for road curvature
    # - Vehicles slightly off-center still considered "ahead"
    # - Too large (>10°): Would include vehicles to the side
    # - Too small (<1°): Too strict, might miss blocking vehicles
    distance, angle = cal_distance_angle(
        target_wpt.transform.location, candidate_wpt.transform.location,
        candidate_wpt.transform.rotation.yaw)
    
    # Return True if angle is small (candidate is ahead, blocking)
    # Return False if angle is large (candidate is behind/side, not blocking)
    return True if angle <= 3 else False
```

### Logic Explanation

**Why this method for platoon joining?**
- **Specific use case**: Back-joining requires checking if path is clear
- **Efficiency**: Quick geometric checks before expensive collision detection
- **Safety**: Prevents attempting to join when path is blocked

**Why multiple checks in sequence?**
- **Early exits**: Fast rejection of non-blocking vehicles
- **Progressive refinement**: From simple (rectangle) to complex (angle)
- **Performance**: Most vehicles rejected early, only few reach angle check

---

## Method 3: `adjacent_lane_collision_check()`

### Purpose
Generate a smooth path in the adjacent lane for collision checking during lane changes and overtaking. This creates the trajectory that will be checked for obstacles.

### Code with Detailed Comments

```python
def adjacent_lane_collision_check(
        self, ego_loc, target_wpt, overtake, carla_map, world):
    """
    Generate a straight line in the adjacent lane for collision detection
    during overtake/lane change.
    
    SCENARIO: Vehicle wants to change lanes or overtake
    - Need to check if adjacent lane is clear
    - Generate a path in the adjacent lane
    - Check this path for collisions with other vehicles
    """
    
    # ============================================================
    # STEP 1: Determine Forward Check Point
    # ============================================================
    # LOGIC: Different behavior for overtake vs normal lane change
    # MEANING: Overtaking needs to check further ahead
    # 
    # WHY DIFFERENT FOR OVERTAKE:
    # - Normal lane change: Just need to check if lane is clear
    # - Overtaking: Need to check if we can pass the vehicle ahead
    # - Overtaking requires more space ahead (to complete the maneuver)
    # 
    # VISUAL:
    # Normal lane change:
    #   Ego → [Check point at target_wpt]
    # 
    # Overtaking:
    #   Ego → [Check point 6 waypoints ahead] → [Pass vehicle] → [Return to lane]
    # 
    # WHY 6 WAYPOINTS:
    # - Waypoints are typically 2 meters apart (from resolution)
    # - 6 waypoints = ~12 meters ahead
    # - Overtaking needs space to accelerate and pass
    # - Too short: Might not see obstacles ahead
    # - Too long: Unnecessary computation
    if overtake:
        target_wpt_next = target_wpt.next(6)[0]  # Check 6 waypoints ahead for overtake
    else:
        target_wpt_next = target_wpt  # Normal lane change: check at target waypoint
    
    # ============================================================
    # STEP 2: Calculate Distance to Previous Waypoint
    # ============================================================
    # LOGIC: Need to check behind the vehicle too (vehicles approaching from behind)
    # MEANING: Calculate how far back to check
    # 
    # WHY CHECK BEHIND:
    # - During lane change, vehicles behind in adjacent lane might hit us
    # - Need to ensure we have space behind as well as ahead
    # - Safety: Don't cut off vehicles approaching from behind
    # 
    # CALCULATION:
    # - diff_x, diff_y: Distance from ego to forward check point
    # - np.hypot(): Euclidean distance (Pythagorean theorem)
    # - +3: Add 3 meter buffer for safety
    # 
    # EXAMPLE:
    # - Ego at (100, 200)
    # - Forward check at (130, 200)
    # - diff_x = 30, diff_y = 0
    # - diff_s = sqrt(30² + 0²) + 3 = 33 meters
    # - Check 33 meters behind the forward point
    diff_x = target_wpt_next.transform.location.x - ego_loc.x
    diff_y = target_wpt_next.transform.location.y - ego_loc.y
    diff_s = np.hypot(diff_x, diff_y) + 3  # Distance + 3m buffer
    
    # ============================================================
    # STEP 3: Get Previous Waypoint (Behind Vehicle)
    # ============================================================
    # LOGIC: Find waypoint behind the forward check point
    # MEANING: This defines the end of our collision check path
    # 
    # WHY WHILE LOOP:
    # - target_wpt.previous(diff_s) might return empty list
    # - This happens if diff_s is too large (beyond road segment)
    # - Need to reduce distance until we find a valid waypoint
    # 
    # WHY REDUCE BY 2 METERS:
    # - Waypoints are ~2 meters apart
    # - Reduce by 2m each iteration until valid waypoint found
    # - Prevents infinite loop (will eventually find waypoint)
    # 
    # EXAMPLE:
    # - diff_s = 100m (too large, road segment only 50m)
    # - previous(100) returns [] → reduce to 98m
    # - previous(98) returns [] → reduce to 96m
    # - ... continue until previous() returns valid waypoint
    target_wpt_previous = target_wpt.previous(diff_s)
    while len(target_wpt_previous) == 0:
        diff_s -= 2  # Reduce by 2 meters (one waypoint spacing)
        target_wpt_previous = target_wpt.previous(diff_s)
    
    # Get first waypoint from list (closest to calculated distance)
    target_wpt_previous = target_wpt_previous[0]
    
    # ============================================================
    # STEP 4: Calculate Middle Waypoint
    # ============================================================
    # LOGIC: Need three points to create smooth spline curve
    # MEANING: Three points define the path: previous → middle → next
    # 
    # WHY THREE POINTS:
    # - Spline interpolation needs at least 3 points for smooth curve
    # - Two points = straight line (not realistic for curved roads)
    # - Three points = smooth curve that follows road geometry
    # 
    # CALCULATION:
    # - Middle point is halfway between previous and next
    # - diff_s/2: Half the distance from previous to next
    # - This creates a natural curve following the road
    target_wpt_middle = target_wpt_previous.next(diff_s/2)[0]
    
    # ============================================================
    # STEP 5: Extract Coordinates for Spline
    # ============================================================
    # LOGIC: Prepare three points for spline interpolation
    # MEANING: These define the collision check path
    # 
    # WHY SPLINE:
    # - Roads are curved, not straight lines
    # - Spline creates smooth curve through the three points
    # - Matches real road geometry
    # - Provides smooth path for collision checking
    x, y = [target_wpt_next.transform.location.x,
            target_wpt_middle.transform.location.x,
            target_wpt_previous.transform.location.x], \
           [target_wpt_next.transform.location.y,
            target_wpt_middle.transform.location.y,
            target_wpt_previous.transform.location.y]
    
    # ============================================================
    # STEP 6: Create Spline and Interpolate Points
    # ============================================================
    # LOGIC: Generate smooth path with fine resolution
    # MEANING: Create many points along the curve for detailed collision checking
    # 
    # WHY ds = 0.1 METERS:
    # - Fine resolution for accurate collision detection
    # - 0.1m spacing = 10 points per meter
    # - Small enough to catch small obstacles
    # - Large enough to be computationally efficient
    # 
    # TRADE-OFF:
    # - Smaller (0.05m): More accurate but slower
    # - Larger (0.2m): Faster but might miss small obstacles
    # - 0.1m is a good balance
    ds = 0.1  # 10 cm spacing between interpolated points
    
    # Create 2D spline from three points
    # Spline2D fits a smooth curve through the points
    sp = Spline2D(x, y)
    
    # Generate parameter values along the spline
    # sp.s[0] = start, sp.s[-1] = end
    # np.arange() creates array: [start, start+0.1, start+0.2, ..., end]
    s = np.arange(sp.s[0], sp.s[-1], ds)
    
    # ============================================================
    # STEP 7: Calculate Interpolated Path Points
    # ============================================================
    # LOGIC: Generate x, y, yaw for each point along the spline
    # MEANING: These are the points we'll check for collisions
    # 
    # WHY INTERPOLATE:
    # - Three waypoints are too sparse for collision checking
    # - Need many points (every 0.1m) for accurate detection
    # - Interpolation fills in the gaps smoothly
    # 
    # OUTPUT:
    # - rx: List of x coordinates
    # - ry: List of y coordinates  
    # - ryaw: List of yaw angles (heading direction at each point)
    # 
    # WHY YAWS:
    # - Need heading direction to place collision circles correctly
    # - Circles are placed relative to vehicle heading
    # - Different heading = different circle positions
    debug_tmp = []  # For debugging visualization (currently commented out)
    
    rx, ry, ryaw = [], [], []
    for i_s in s:
        # Calculate position (x, y) at parameter value i_s
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        
        # Calculate yaw (heading angle) at parameter value i_s
        # This tells us which direction the vehicle is facing at this point
        ryaw.append(sp.calc_yaw(i_s))
        
        # Store for debug visualization (optional)
        debug_tmp.append(carla.Transform(carla.Location(ix, iy, 0)))
    
    # ============================================================
    # STEP 8: Optional Debug Visualization
    # ============================================================
    # LOGIC: Draw the collision check path for debugging
    # MEANING: Visualize what path is being checked
    # 
    # WHY COMMENTED OUT:
    # - Debug visualization can be expensive (many draw calls)
    # - Only enable when debugging specific issues
    # - Yellow line = overtaking path
    # - White line = normal lane change path
    # 
    # draw_trajetory_points(
    #     world, debug_tmp, color=carla.Color(
    #         255, 255, 0) if overtake else carla.Color(
    #         255, 255, 255), size=0.05, lt=0.2)
    
    return rx, ry, ryaw
```

### Logic Explanation

**Why generate a path instead of checking waypoints directly?**
- **Smooth trajectory**: Real vehicles follow smooth curves, not waypoint jumps
- **Accurate collision**: Check actual path vehicle will take
- **Road geometry**: Spline follows curved roads naturally

**Why check both ahead and behind?**
- **Ahead**: Vehicles in front might be obstacles
- **Behind**: Vehicles approaching from behind might hit us during lane change
- **Safety**: Need clear space in both directions

**Why use spline interpolation?**
- **Smoothness**: Realistic vehicle path (not zigzag between waypoints)
- **Density**: Many points (0.1m spacing) for accurate collision detection
- **Efficiency**: Generate path once, check many obstacles against it

---

## Method 4: `collision_circle_check()`

### Purpose
The main collision detection method. Checks if an obstacle vehicle will collide with the planned trajectory using circle-based collision detection.

### Code with Detailed Comments

```python
def collision_circle_check(
        self,
        path_x,
        path_y,
        path_yaw,
        obstacle_vehicle,
        speed,
        carla_map,
        adjacent_check=False):
    """
    Use circled collision check to see whether potential hazard on
    the forwarding path.
    
    METHOD: Circle-based collision detection
    - Place multiple circles along planned path
    - Check if obstacle vehicle bounding box intersects any circle
    - Fast and efficient for real-time collision detection
    """
    
    # ============================================================
    # STEP 1: Initialize Collision-Free Flag
    # ============================================================
    # LOGIC: Assume path is safe, prove it's not
    # MEANING: Start optimistic, set to False if collision found
    # 
    # WHY THIS APPROACH:
    # - Early exit: Stop checking once collision found
    # - Efficient: Don't need to check all points if collision exists
    collision_free = True
    
    # ============================================================
    # STEP 2: Calculate Check Distance
    # ============================================================
    # LOGIC: Determine how far along path to check for collisions
    # MEANING: Don't check entire path, only relevant portion
    # 
    # FORMULA BREAKDOWN:
    # - self.time_ahead * speed: Distance vehicle travels in time_ahead seconds
    # - / 0.1: Convert to number of path points (each point is 0.1m apart)
    # - int(): Round to integer (can't check fractional points)
    # - max(..., 90): Minimum 90 points (9 meters) even if speed is very slow
    # - min(..., len(path_x)): Don't exceed path length
    # 
    # WHY TIME-BASED:
    # - Faster vehicles need to check further ahead
    # - Slower vehicles still need minimum check distance
    # - Consistent safety margin regardless of speed
    # 
    # EXAMPLE:
    # - time_ahead = 1.2s, speed = 20 m/s
    # - Distance = 1.2 * 20 = 24 meters
    # - Points = 24 / 0.1 = 240 points
    # - But if speed = 5 m/s: 1.2 * 5 / 0.1 = 60 points
    # - Use max(60, 90) = 90 points (minimum check)
    # 
    # WHY MINIMUM 90 POINTS:
    # - Very slow vehicles still need reasonable lookahead
    # - 90 points = 9 meters minimum
    # - Prevents checking too short distance when stopped/slow
    # 
    # WHY ADJACENT CHECK IS DIFFERENT:
    # - adjacent_check = True: Check entire path (for lane changes)
    # - Lane changes need full path check (entire maneuver)
    # - Normal driving: Only check ahead (time-based)
    distance_check = min(max(int(self.time_ahead * speed / 0.1), 90),
                         len(path_x)) \
        if not adjacent_check else len(path_x)
    
    # ============================================================
    # STEP 3: Get Obstacle Vehicle Information
    # ============================================================
    # LOGIC: Need obstacle location and orientation for collision check
    # MEANING: Where is the obstacle and which way is it facing?
    obstacle_vehicle_loc = obstacle_vehicle.get_location()
    
    # Get obstacle's heading direction
    # WHY: Need to calculate rotated bounding box correctly
    # - Vehicles are rectangles, not circles
    # - Rectangle orientation matters for collision detection
    # - Need yaw to rotate bounding box to match vehicle orientation
    obstacle_vehicle_yaw = \
        carla_map.get_waypoint(obstacle_vehicle_loc).transform.rotation.yaw
    
    # ============================================================
    # STEP 4: Check Path Points (Sampled)
    # ============================================================
    # LOGIC: Check collision at multiple points along the path
    # MEANING: Not every point (too expensive), but enough for accuracy
    # 
    # WHY STEP BY 10:
    # - Path points are 0.1m apart
    # - Check every 10 points = check every 1 meter
    # - Balance between accuracy and performance
    # 
    # TRADE-OFF:
    # - Step by 1: Check every 0.1m (very accurate, slow)
    # - Step by 10: Check every 1m (good accuracy, fast)
    # - Step by 20: Check every 2m (less accurate, very fast)
    # - Step by 10 is a good compromise
    # 
    # EXAMPLE:
    # - Path has 240 points (24 meters)
    # - Check points: 0, 10, 20, 30, ..., 230
    # - Total checks: 24 collision checks (much faster than 240)
    for i in range(0, distance_check, 10):
        # Get current path point coordinates and heading
        ptx, pty, yaw = path_x[i], path_y[i], path_yaw[i]
        
        # ============================================================
        # STEP 5: Place Collision Circles
        # ============================================================
        # LOGIC: Place multiple circles along vehicle width at this path point
        # MEANING: Approximate vehicle shape with circles
        # 
        # WHY MULTIPLE CIRCLES:
        # - Vehicle has width (~2 meters)
        # - Single circle misses collisions at edges
        # - Multiple circles cover full vehicle width
        # 
        # CALCULATION:
        # - circle_offsets = [-1.0, 0, 1.0] (left, center, right)
        # - For each offset, calculate circle position
        # - Position = path_point + offset * direction_vector
        # 
        # DIRECTION VECTOR:
        # - cos(yaw), sin(yaw) = unit vector in heading direction
        # - Perpendicular offset: Use perpendicular to heading
        # - Actually: offset is along vehicle width (perpendicular to heading)
        # 
        # VISUAL:
        #     Path point (ptx, pty) with heading yaw
        #           ↓
        #        [Left]  [Center]  [Right]
        #         Circle   Circle   Circle
        # 
        # WHY PERPENDICULAR:
        # - Vehicle width is perpendicular to heading
        # - Offset circles left/right of center line
        # - Covers full vehicle width
        circle_locations = np.zeros((len(self._circle_offsets), 2))
        circle_offsets = np.array(self._circle_offsets)
        
        # Calculate circle positions
        # circle_locations[:, 0] = x coordinates
        # circle_locations[:, 1] = y coordinates
        # 
        # NOTE: The code uses cos/sin of yaw, which suggests offset along heading
        # But typically, vehicle width offset should be perpendicular
        # This might be a simplification or the offsets are small enough it doesn't matter
        circle_locations[:, 0] = ptx + circle_offsets * cos(yaw)
        circle_locations[:, 1] = pty + circle_offsets * sin(yaw)
        
        # ============================================================
        # STEP 6: Calculate Obstacle Bounding Box
        # ============================================================
        # LOGIC: Represent obstacle vehicle as rotated rectangle
        # MEANING: Calculate corners of vehicle's bounding box
        # 
        # WHY BOUNDING BOX:
        # - Vehicles are rectangular, not circular
        # - Need to check if circles intersect rectangle
        # - More accurate than treating vehicle as single point
        # 
        # CALCULATION:
        # - obstacle_vehicle.bounding_box.extent: Half-dimensions of vehicle
        # - extent.x = half length, extent.y = half width
        # - Need to rotate by vehicle yaw to get world coordinates
        # 
        # WHY ROTATE:
        # - Vehicle bounding box is in vehicle-local coordinates
        # - Need world coordinates for collision check
        # - Rotation accounts for vehicle orientation
        # 
        # NOTE: The code calculates corrected_extent but uses it incorrectly
        # The actual bounding box calculation seems simplified
        # In practice, you'd need to rotate all four corners properly
        corrected_extent_x = obstacle_vehicle.bounding_box.extent.x * \
                             math.cos(math.radians(obstacle_vehicle_yaw))
        corrected_extent_y = obstacle_vehicle.bounding_box.extent.y * \
                             math.sin(math.radians(obstacle_vehicle_yaw))
        
        # ============================================================
        # STEP 7: Create Bounding Box Corner Array
        # ============================================================
        # LOGIC: Define corners of obstacle vehicle bounding box
        # MEANING: Four corners + center point for collision checking
        # 
        # WHY FIVE POINTS:
        # - Four corners of rectangle
        # - Plus center point
        # - Check if any of these points are inside collision circles
        # 
        # CALCULATION:
        # - Each corner = center ± extent in x and y
        # - Creates rectangle around vehicle
        # 
        # NOTE: The calculation here is simplified
        # Proper rotation would require:
        #   corner_x = center_x + extent_x*cos(yaw) - extent_y*sin(yaw)
        #   corner_y = center_y + extent_x*sin(yaw) + extent_y*cos(yaw)
        # 
        # The current code seems to approximate this
        obstacle_vehicle_bbx_array = \
            np.array([[obstacle_vehicle_loc.x - corrected_extent_x,
                      obstacle_vehicle_loc.y - corrected_extent_y],
                     [obstacle_vehicle_loc.x - corrected_extent_x,
                      obstacle_vehicle_loc.y + corrected_extent_y],
                     [obstacle_vehicle_loc.x,
                      obstacle_vehicle_loc.y],  # Center point
                     [obstacle_vehicle_loc.x + corrected_extent_x,
                      obstacle_vehicle_loc.y - corrected_extent_y],
                     [obstacle_vehicle_loc.x + corrected_extent_x,
                      obstacle_vehicle_loc.y + corrected_extent_y]])
        
        # ============================================================
        # STEP 8: Calculate Distances
        # ============================================================
        # LOGIC: Check if bounding box points are inside collision circles
        # MEANING: Calculate distance from each bbox point to each circle center
        # 
        # HOW IT WORKS:
        # - spatial.distance.cdist(): Computes distance matrix
        # - Input: bbox_points (5 points) × circle_centers (3 circles)
        # - Output: 5×3 matrix of distances
        # 
        # EXAMPLE:
        #   bbox_point_1 to circle_1: distance = 2.5m
        #   bbox_point_1 to circle_2: distance = 1.8m
        #   bbox_point_1 to circle_3: distance = 3.1m
        #   ... (for all 5 points × 3 circles)
        collision_dists = spatial.distance.cdist(
            obstacle_vehicle_bbx_array, circle_locations)
        
        # ============================================================
        # STEP 9: Check for Collision
        # ============================================================
        # LOGIC: Determine if any bbox point is inside any circle
        # MEANING: Collision occurs if distance < circle_radius
        # 
        # CALCULATION:
        # - Subtract circle_radius from all distances
        # - If result < 0, point is inside circle (collision!)
        # - If result >= 0, point is outside circle (no collision)
        # 
        # WHY SUBTRACT RADIUS:
        # - Distance is center-to-center
        # - Need center-to-edge distance
        # - Subtract radius: if distance - radius < 0, point is inside
        # 
        # EXAMPLE:
        # - Distance from bbox point to circle center = 0.8m
        # - Circle radius = 1.0m
        # - collision_dists = 0.8 - 1.0 = -0.2 < 0
        # - Collision detected!
        # 
        # - Distance = 1.5m, radius = 1.0m
        # - collision_dists = 1.5 - 1.0 = 0.5 >= 0
        # - No collision
        collision_dists = np.subtract(collision_dists, self._circle_radius)
        
        # ============================================================
        # STEP 10: Update Collision-Free Flag
        # ============================================================
        # LOGIC: Check if any distance is negative (collision found)
        # MEANING: If any bbox point is inside any circle, collision exists
        # 
        # HOW IT WORKS:
        # - np.any(collision_dists < 0): True if ANY value is negative
        # - not np.any(...): Invert (True if NO collisions)
        # - collision_free = collision_free AND (no collisions at this point)
        # - If collision found at any point, collision_free becomes False
        # 
        # WHY AND LOGIC:
        # - Start with True (assume safe)
        # - If collision found at ANY point, set to False
        # - Once False, stays False (path is unsafe)
        collision_free = collision_free and not np.any(collision_dists < 0)
        
        # ============================================================
        # STEP 11: Early Exit Optimization
        # ============================================================
        # LOGIC: Stop checking once collision is found
        # MEANING: No need to check remaining points if collision exists
        # 
        # WHY EARLY EXIT:
        # - Performance: Don't waste time checking safe points
        # - Once collision found, path is unsafe (no need to continue)
        # - Significant speedup when collision exists
        # 
        # EXAMPLE:
        # - Path has 240 points to check
        # - Collision found at point 50
        # - Early exit: Skip checking points 60, 70, ..., 230
        # - Saves ~80% of computation time
        if not collision_free:
            break  # Stop checking, collision found
    
    # ============================================================
    # STEP 12: Return Result
    # ============================================================
    # LOGIC: Return whether path is collision-free
    # MEANING: True = safe to proceed, False = collision risk
    # 
    # USAGE:
    # - Behavior agent uses this to decide if maneuver is safe
    # - True: Proceed with lane change/overtake
    # - False: Abort maneuver, stay in current lane
    return collision_free
```

### Logic Explanation

**Why circle-based collision detection?**
- **Efficiency**: Circle-to-point distance is fast: `sqrt((x1-x2)² + (y1-y2)²)`
- **Approximation**: Multiple circles approximate vehicle shape well enough
- **Real-time**: Fast enough for real-time collision checking (every simulation step)

**Why check sampled points instead of all points?**
- **Performance**: Checking every 0.1m is too expensive
- **Accuracy**: Checking every 1m (step by 10) is accurate enough
- **Balance**: Good trade-off between speed and accuracy

**Why early exit?**
- **Performance**: Once collision found, no need to check further
- **Efficiency**: Can save 80%+ computation time when collision exists
- **Real-time**: Critical for maintaining simulation frame rate

**Why time-based lookahead?**
- **Dynamic**: Obstacles are moving, need to predict future positions
- **Speed-dependent**: Faster vehicles need longer lookahead
- **Safety**: Consistent safety margin regardless of speed

---

## Summary

The `CollisionChecker` uses efficient geometric methods to detect collisions:

1. **Circle-based detection**: Fast approximation of vehicle shape
2. **Time-based lookahead**: Predicts future collisions based on speed
3. **Sampled checking**: Balances accuracy and performance
4. **Early exit**: Optimizes for common case (no collision)

These methods enable real-time collision detection for safe autonomous driving in CARLA simulations.

