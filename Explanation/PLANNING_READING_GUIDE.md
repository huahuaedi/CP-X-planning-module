# Decision Making and Path Planning - Reading Guide

This guide provides a step-by-step reading order to understand the decision making and path planning system in OpenCDA.

## 🚀 Quick Start (TL;DR)

**If you only have 2 hours:**
1. Read `behavior_agent.py` → `run_step()` method (lines 751-902) - **THE CORE**
2. Read `local_planner_behavior.py` → `generate_path()` method (lines 216-373)
3. Skim `collision_check.py` → `collision_circle_check()` method

**If you have 6-8 hours:**
Follow Phases 1-5 in order (skip Phase 7 unless you need platooning)

**If you want complete understanding:**
Follow all phases in order (8-12 hours total)

## 📚 Reading Order

### **Phase 1: Foundation & Utilities** (Start Here)
*Understand the basic building blocks and helper functions*

1. **`opencda/core/common/misc.py`** ⭐ ESSENTIAL
   - **Why**: Contains fundamental utility functions used throughout planning
   - **Key Functions to Understand**:
     - `cal_distance_angle()` - Calculates relative distance and angle (used in collision checks)
     - `distance_vehicle()` - Distance from waypoint to vehicle
     - `compute_distance()` - Euclidean distance calculation
     - `get_speed()` - Vehicle speed extraction
     - `draw_trajetory_points()` - Visualization helper
   - **Reading Time**: 30-45 minutes
   - **Focus**: Understand how spatial calculations work

2. **`opencda/core/plan/spline.py`** ⭐ ESSENTIAL
   - **Why**: Core path smoothing algorithm - generates smooth trajectories
   - **Key Classes**:
     - `Spline2D` - 2D cubic spline for path generation
     - `Spline` - 1D cubic spline (used by Spline2D)
   - **Key Methods**:
     - `calc_position()` - Get position along spline
     - `calc_curvature()` - Calculate path curvature
     - `calc_yaw()` - Calculate heading angle
   - **Reading Time**: 45-60 minutes
   - **Focus**: Understand how waypoints are converted to smooth paths

---

### **Phase 2: Global Route Planning** 
*Understand how high-level routes are planned*

3. **`opencda/core/plan/global_route_planner_dao.py`** 
   - **Why**: Data access layer - fetches map data from CARLA
   - **Key Methods**:
     - `get_topology()` - Retrieves road network topology
     - `get_waypoint()` - Gets waypoint at location
   - **Reading Time**: 20-30 minutes
   - **Focus**: Understand how map data is accessed

4. **`opencda/core/plan/global_route_planner.py`** ⭐ ESSENTIAL
   - **Why**: High-level route planning from start to destination
   - **Key Methods** (with line numbers):
     - `setup()` (lines 62-70) - Builds graph representation of map
     - **`trace_route()` (lines 433-434+)** - Main method to get route from origin to destination
     - **`_path_search()` (lines 285-305)** - A* search algorithm implementation
     - **`_turn_decision()` (lines 330-397)** - Determines turn type (LEFT, RIGHT, STRAIGHT)
     - `abstract_route_plan()` (lines 399-419) - Generates turn-by-turn navigation plan
   - **Reading Time**: 60-90 minutes
   - **Focus**: Understand A* pathfinding and route generation
   - **Note**: This uses NetworkX for graph operations

---

### **Phase 3: Local Path Planning**
*Understand how smooth local trajectories are generated*

5. **`opencda/core/plan/local_planner_behavior.py`** ⭐ ESSENTIAL
   - **Why**: Generates smooth local trajectories from waypoints
   - **Key Classes**:
     - `RoadOption` (Enum, lines 21-32) - Road navigation options
     - `LocalPlanner` - Main local planning class
   - **Key Methods** (with line numbers):
     - `set_global_plan()` (lines 128-151) - Sets the global route waypoints
     - **`generate_path()` (lines 216-373)** - **CRITICAL** - Generates smooth spline path
     - `generate_trajectory()` (lines 375-445) - Samples path and assigns speeds
     - `buffer_filter()` (lines 447-491) - Filters problematic waypoints
     - `run_step()` (lines 535-639) - Executes one planning step, returns target speed/location
   - **Reading Time**: 90-120 minutes
   - **Focus**: Understand how waypoints → spline → trajectory conversion works
   - **Key Concepts**: 
     - Waypoint buffers and queues
     - Lane change detection
     - Trajectory sampling with speed assignment

---

### **Phase 4: Collision Checking & Safety**
*Understand how safety is ensured*

6. **`opencda/core/plan/collision_check.py`** ⭐ ESSENTIAL
   - **Why**: Validates planned paths for safety
   - **Key Class**: `CollisionChecker`
   - **Key Methods** (with line numbers):
     - **`collision_circle_check()` (lines 179-263)** - Checks if path is collision-free
     - **`adjacent_lane_collision_check()` (lines 107-177)** - Checks adjacent lanes for lane changes
     - `is_in_range()` (lines 40-105) - Checks if vehicle is in range (for platooning)
   - **Reading Time**: 45-60 minutes
   - **Focus**: Understand circle-based collision detection algorithm

---

### **Phase 5: Decision Making (Behavior Planning)**
*Understand high-level decision making*

7. **`opencda/core/plan/behavior_agent.py`** ⭐⭐ MOST IMPORTANT
   - **Why**: Main decision-making brain - coordinates everything
   - **Key Class**: `BehaviorAgent`
   - **Key Methods** (Read in this order with line numbers):
     1. `__init__()` (lines 86-148) - Initialization, understand all components
     2. `update_information()` (lines 152-187) - How perception/localization data flows in
     3. `set_destination()` (lines 242-303) - How routes are set
     4. `_trace_route()` (lines 334-361) - Calls global planner
     5. **`run_step()` (lines 751-902)** - **CRITICAL** - Main decision loop, read carefully!
     6. `collision_manager()` (lines 414-458) - Collision detection coordination
     7. `overtake_management()` (lines 460-548) - Overtaking decision logic
     8. `lane_change_management()` (lines 550-580) - Lane change safety check
     9. `car_following_manager()` (lines 582-626) - Car-following speed control
     10. `traffic_light_manager()` (lines 363-412) - Traffic light handling
   - **Reading Time**: 2-3 hours (this is the core!)
   - **Focus**: Understand the decision flow in `run_step()`:
     ```
     1. Traffic light check
     2. Route reset if needed
     → 3. Generate path (calls LocalPlanner.generate_path())
     → 4. Check lane change permission
     → 5. Collision check
     → 6. Handle push case (lane change blocked)
     → 7. Handle overtake case
     → 8. Handle car following case
     → 9. Normal driving case
     ```
   - **Key Attributes to Understand**:
     - `_local_planner` - Local path planner instance
     - `_global_planner` - Global route planner instance
     - `_collision_check` - Collision checker instance
     - `obstacle_vehicles` - Detected obstacles
     - `hazard_flag`, `car_following_flag` - State flags

---

### **Phase 6: Integration & System Flow**
*Understand how everything connects*

8. **`opencda/core/common/vehicle_manager.py`**
   - **Why**: Shows how planning integrates with other modules
   - **Key Section**: `__init__()` method (lines ~76-130)
   - **Focus**: See how `BehaviorAgent` is created and connected to:
     - Perception manager
     - Localization manager
     - V2X manager
     - Controller
   - **Reading Time**: 30-45 minutes
   - **Key Understanding**: How `agent.run_step()` is called in the main loop

9. **`opencda/core/plan/planer_debug_helper.py`**
   - **Why**: Debugging and visualization tools
   - **Reading Time**: 15-20 minutes
   - **Focus**: Optional, but helpful for understanding performance metrics

---

### **Phase 7: Advanced - Platooning** (Optional)
*Specialized decision making for platooning scenarios*

10. **`opencda/core/application/platooning/fsm.py`**
    - **Why**: Defines platooning states
    - **Reading Time**: 15 minutes
    - **Focus**: Understand the state machine states

11. **`opencda/core/application/platooning/platoon_behavior_agent.py`**
    - **Why**: Extends BehaviorAgent for platooning
    - **Key Methods**:
      - `run_step()` - Overrides parent, adds FSM logic
      - `run_step_cut_in_joining()` - Cut-in joining algorithm
      - `run_step_back_joining()` - Back joining algorithm
      - `run_step_front_joining()` - Front joining algorithm
      - `platooning_following_manager()` - Gap-keeping car following
    - **Reading Time**: 90-120 minutes
    - **Focus**: Understand how platooning states modify behavior

---

## 🎯 Quick Reference: Key File Dependencies

```
misc.py (utilities)
    ↓
spline.py (path smoothing)
    ↓
global_route_planner_dao.py → global_route_planner.py (route planning)
    ↓
collision_check.py (safety)
    ↓
local_planner_behavior.py (local trajectory)
    ↓
behavior_agent.py (decision making) ← Uses all above
    ↓
vehicle_manager.py (integration)
```

---

## 🔄 Execution Flow (How It All Works Together)

### **Main Simulation Loop** (in scenario files like `single_town06_carla.py`):
```python
while True:
    world.tick()  # CARLA simulation step
    
    # For each vehicle:
    vehicle_manager.update_info()  # Step 1: Get sensor data
    control = vehicle_manager.run_step()  # Step 2: Plan & control
    vehicle.apply_control(control)  # Step 3: Execute
```

### **Step-by-Step Execution Flow**:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. vehicle_manager.update_info()                            │
│    ├─> localizer.localize()                                 │
│    ├─> perception_manager.detect()                          │
│    └─> agent.update_information(ego_pos, ego_speed, objects)│
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. vehicle_manager.run_step()                               │
│    ├─> map_manager.run_step()  (optional visualization)     │
│    ├─> agent.run_step()  ⭐ MAIN DECISION POINT             │
│    │   │                                                    │
│    │   ├─> [Check traffic lights]                          │
│    │   ├─> [Check if route needs reset]                   │
│    │   ├─> local_planner.generate_path()  ⭐ PATH GEN      │
│    │   │   ├─> buffer_filter()                            │
│    │   │   ├─> Spline2D()  (from spline.py)               │
│    │   │   └─> Returns: rx, ry, ryaw, rk                  │
│    │   │                                                    │
│    │   ├─> check_lane_change_permission()                  │
│    │   ├─> collision_manager()  ⭐ SAFETY CHECK            │
│    │   │   └─> collision_check.collision_circle_check()  │
│    │   │                                                    │
│    │   ├─> [Decision Branch]                               │
│    │   │   ├─> If hazard + overtake_allowed:               │
│    │   │   │   └─> overtake_management()                   │
│    │   │   ├─> If hazard + no overtake:                    │
│    │   │   │   └─> car_following_manager()                 │
│    │   │   └─> Else: normal driving                        │
│    │   │                                                    │
│    │   └─> local_planner.run_step(rx, ry, rk)              │
│    │       ├─> generate_trajectory()  (adds speed)         │
│    │       └─> Returns: target_speed, target_location     │
│    │                                                    │
│    └─> controller.run_step(target_speed, target_pos)       │
│        └─> Returns: carla.VehicleControl                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. vehicle.apply_control(control)                           │
│    └─> Vehicle executes the control command                 │
└─────────────────────────────────────────────────────────────┘
```

### **Key Method Call Chain in `behavior_agent.run_step()`**:

```python
# Line 751-902 in behavior_agent.py

def run_step():
    # 1. Setup & checks
    ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
    
    # 2. Traffic light check (line 799)
    if self.traffic_light_manager(ego_vehicle_wp) != 0:
        return 0, None  # Stop for red light
    
    # 3. Route reset if needed (line 803-816)
    if waypoints_queue is empty:
        self.set_destination(...)  # Reset to global route
    
    # 4. Generate smooth path (line 825) ⭐
    rx, ry, rk, ryaw = self._local_planner.generate_path()
    #    ↓ This calls:
    #    - buffer_filter() - removes bad waypoints
    #    - Spline2D(x, y) - creates smooth spline
    #    - Returns interpolated path points
    
    # 5. Check lane change permission (line 828)
    self.lane_change_allowed = self.check_lane_change_permission(...)
    
    # 6. Collision check (line 834) ⭐
    is_hazard, obstacle_vehicle, distance = self.collision_manager(rx, ry, ryaw, ...)
    #    ↓ This calls:
    #    - collision_check.collision_circle_check() for each obstacle
    
    # 7. Decision branches (line 838-901)
    if is_hazard and self.overtake_allowed:
        # Overtake path
        car_following_flag = self.overtake_management(obstacle_vehicle)
    elif is_hazard:
        # Car following
        car_following_flag = True
        target_speed = self.car_following_manager(obstacle_vehicle, distance)
    else:
        # Normal driving
        target_speed = self.max_speed
    
    # 8. Execute local planner (line 894 or 899)
    target_speed, target_loc = self._local_planner.run_step(rx, ry, rk, target_speed)
    #    ↓ This calls:
    #    - generate_trajectory() - samples path with speed
    #    - pop_buffer() - removes passed waypoints
    #    - Returns next target waypoint
    
    return target_speed, target_loc
```

---

## 📖 Recommended Deep Dive Sequence

### **For Understanding Path Generation:**
1. `spline.py` → `local_planner_behavior.py` (focus on `generate_path()`)

### **For Understanding Decision Making:**
1. `behavior_agent.py` → Focus on `run_step()` method
2. Trace through each decision branch
3. Understand how collision checks influence decisions

### **For Understanding Route Planning:**
1. `global_route_planner_dao.py` → `global_route_planner.py`
2. Focus on `_path_search()` (A* algorithm)
3. Understand `_turn_decision()` logic

---

## 🔍 Key Concepts to Master

1. **Hierarchical Planning**:
   - Global (route) → Local (trajectory) → Control (execution)

2. **Waypoint Management**:
   - `waypoints_queue` (global route)
   - `_waypoint_buffer` (local planning window)
   - `_trajectory_buffer` (execution trajectory)

3. **Decision Flow**:
   - Perception → Collision Check → Behavior Decision → Path Generation → Control

4. **State Management**:
   - Flags: `hazard_flag`, `car_following_flag`, `lane_change_allowed`
   - Counters: `overtake_counter`, `destination_push_flag`

---

## 💡 Tips for Reading

1. **Start with `behavior_agent.py` `run_step()`** - This is the main entry point
2. **Use a debugger** - Set breakpoints in `run_step()` to trace execution
3. **Read method docstrings** - They explain the logic
4. **Follow the data flow** - See how waypoints flow: queue → buffer → trajectory
5. **Understand the state flags** - They control decision branches
6. **Look at example scenarios** - Check `opencda/scenario_testing/` for usage examples

---

## 🚀 Next Steps After Reading

1. Run a simple scenario and trace through the code
2. Modify decision parameters and observe behavior changes
3. Add custom decision logic in `behavior_agent.py`
4. Experiment with different path smoothing parameters in `local_planner_behavior.py`

---

## 📝 Notes

- **Total Reading Time Estimate**: 8-12 hours for complete understanding
- **Minimum Essential Reading**: Phases 1-5 (6-8 hours)
- **Most Critical File**: `behavior_agent.py` - spend most time here!

---

*Last Updated: Based on OpenCDA codebase structure*

