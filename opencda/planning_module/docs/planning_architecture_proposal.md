# Planning Architecture Proposal

This proposal is for moving the current planner from bug-by-bug fixes to a
layered, inspectable planning stack. The reference style is close to common
open-source autonomous-driving stacks such as Apollo planning, Autoware
behavior/path planning, and CARLA's BehaviorAgent: separate route intent,
behavior decisions, path generation, speed decisions, and low-level control.

## 0. Current Implementation Status (as of 2026-06-30)

This section tracks how much of the target architecture already exists in
`opencda/planning_module/`. Status tags used below: **DONE**, **PARTIAL**,
**MISSING**.

| Layer / Item | Status | Where |
| --- | --- | --- |
| Route layer (separate module) | MISSING | Route info (`route_optimal_lane_id`, `next_macro_maneuver`) flows directly into `behavior_planner/planner.py`; no isolated route-corridor/segment module |
| Behavior layer FSM + priority order | DONE | `behavior_planner/planner.py` (`RuleBasedBehaviorPlanner.update`) implements emergency brake -> traffic/CP stop -> reroute -> lane-change candidate evaluation in that priority order |
| Behavior command object (typed contract) | MISSING | `update()` still returns a loose `Dict[str, Any]`; `planning_runner.py` reads it with `.get(...)` rather than a typed `BehaviorCommand` |
| Red-light / stop-sign invariants | DONE | `behavior_planner/traffic_light_stop.py` (`find_relevant_signal_context`, `should_stop_for_signal`) handles ego-association vs stop-waypoint vs actor-position priority and avoids stale-red via explicit `is not None` checks |
| TEMP_DES target-lane consistency during lane change | DONE | `temp_destination.py` (`_start_wp_for_decision`, `_follow_route_lane_for_decision`) routes both the rolling anchor and reference samples off the behavior-selected lane during `lane_change_*` decisions |
| Path layer continuity (reference path vs jumping point) | PARTIAL | `_build_lane_reference_samples_to_target` / `_build_route_reference_samples_from_anchor` build multi-step reference samples, but there is no single dedicated "Path Layer" module decoupled from `temp_destination.py` |
| Speed layer (unified speed envelope) | PARTIAL | Stop profile and obstacle following speed live inside `MPC/mpc.py` (`final_stop_speed_cap`, reference-rollout obstacle speed); curvature cap is computed and applied pre-solve in `planning_runner.py` (~line 5570-5596); there is no single `speed_envelope` object passed into MPC |
| MPC hard/soft constraint split | DONE | `MPC/mpc.py`: hard bounds in `MPCConstraintSpec`; soft costs (`Cost_LaneCenter`, `Cost_RoadBoundary`, `Cost_Repulsive`, `Cost_Control`) configured via `MPCComfortCostSpec` / `MPCSafetyCostSpec` / `MPCRepulsivePotentialSpec`, weights in `mpc.yaml` |
| MPC fail-safe fallback | PARTIAL | On solver failure the rollout/seed is reset and retried; after repeated failures it returns the reference rollout — there is no explicit "brake gently" / emergency-stop escalation distinct from just falling back to the reference path |
| Diagnostics artifacts | MOSTLY DONE | Produced: `planning_metrics_timeseries.csv`, `control_timeseries.csv`, `lane_reference_timeseries.csv`, `fsm_transition_log.csv`, `mpc_cost_history.csv`, `planned_trajectory.csv`, `analysis_report.txt`, `analysis_plots.png`, `tracking_dashboard.png`. Missing: `tracking_error_timeseries.csv` (referenced by `analyze_run.py` but never written) and `mpc_diagnostics.json` |
| Scenario regression suite | DONE | `opencda/planning_module/tests/` has ~26 files covering red light, intersection behavior, reroute/lane-closure, lane keeping, lane reference, route planning, and multiple town/SUMO scenarios |

Net read: **Phase 1 (diagnostics) is essentially done** modulo two missing
artifacts. **Phase 4 (MPC hard/soft split, curvature pre-cap) is done**, with
the fail-safe escalation still incomplete. **Phase 2 (behavior command
object)** and the dedicated **Route layer** are the biggest structural gaps —
the behavior layer already makes the right decisions, but downstream code
still consumes them as an untyped dict instead of a typed contract. **Phase
3 (path/speed split)** is partially there: lane-following reference
continuity exists, but speed-envelope logic is scattered across MPC and the
runner rather than isolated in its own layer.

## 1. Target Layering

### Route Layer

Input:

- Global route points
- Current ego pose
- Map topology and lane graph

Output:

- Route corridor
- Current route segment
- Route-preferred lane
- Upcoming maneuver: straight, left, right, merge, junction

Rule:

- The route layer never decides stop, lane change, or speed.
- It only says where the mission wants the vehicle to go.

### Behavior Layer

Input:

- Route corridor
- Ego lane state
- Traffic light and stop controls
- Static and dynamic obstacles
- Lane safety / prediction risk

Output:

- Behavior state:
  - `LANE_KEEP`
  - `PREPARE_LANE_CHANGE_LEFT`
  - `EXECUTE_LANE_CHANGE_LEFT`
  - `PREPARE_LANE_CHANGE_RIGHT`
  - `EXECUTE_LANE_CHANGE_RIGHT`
  - `STOP_FOR_RED_LIGHT`
  - `STOP_FOR_STOP_SIGN`
  - `FOLLOW_LEAD`
  - `REROUTE`
  - `EMERGENCY_BRAKE`
- Target lane id
- Stop target, if any
- Desired speed envelope
- Reason/debug fields

Hard priority order:

1. Emergency collision risk
2. Red light / required stop before stop line
3. Stop sign wait logic
4. Blocked lane / reroute
5. Lane-change safety
6. Route preference
7. Cruise / car following

Important invariant:

- If behavior says stop for red light, the target lane must be the ego lane.
- If behavior says execute lane change, TEMP_DES and MPC reference must both
  use the target lane, not the route-preferred lane.

### Path Layer

Input:

- Behavior target lane
- Route corridor
- Stop/follow target
- Map lane centerline

Output:

- Reference path samples:
  - `x_ref_m`
  - `y_ref_m`
  - `heading_rad`
  - `lane_id`
  - road boundary widths

Rule:

- TEMP_DES is only a rolling anchor.
- MPC should primarily track a continuous reference path, not chase a jumping
  point.
- Reference path continuity must be measured every run:
  - reference jump
  - lateral tracking error
  - heading error
  - boundary cost

### Speed Layer

Input:

- Behavior state
- Lead vehicle
- Stop line distance
- Curvature
- Speed limit / configured max speed

Output:

- Speed envelope:
  - desired cruise speed
  - max speed cap
  - stop profile
  - following profile

Rule:

- Red light stop and stop sign stop are speed-profile problems, not lane
  selection problems.
- Curvature cap should be applied before MPC solve.
- IDM should not force `vmax=0` unless the behavior layer is in a true stop
  context.

### MPC Layer

Input:

- Reference path
- Speed envelope
- Obstacle predictions
- Vehicle state

Output:

- Planned trajectory
- Control sequence
- Feasibility/cost diagnostics

Recommended decoupling:

- Hard constraints:
  - vehicle dynamics
  - acceleration / steering bounds
  - terminal stop only when behavior explicitly says stop
- Soft constraints:
  - lane center cost
  - road boundary slack
  - obstacle repulsive cost
  - control smoothness
- Pre-MPC shaping:
  - curvature speed cap
  - stop speed profile
  - lane-change reference path
- Post-MPC diagnostics:
  - solver status
  - lateral error
  - heading error
  - road boundary cost
  - obstacle collision cost

## 2. Current Problems Mapped To Layers

### Red Light Still Moving

Likely causes:

- CP control type mismatch
- stale or unknown signal blocking fallback
- stop target assigned to neighboring lane
- traffic-light fallback disabled when CP obstacle pipeline is active

Required invariant:

- Red/yellow control produces `STOP_FOR_RED_LIGHT`.
- Stop target is on ego lane.
- Green clears the latch.
- Unknown signal does not preserve stale red.

### TEMP_DES On Wrong Lane During Lane Change

Likely causes:

- TEMP_DES smoothing blends old lane and target lane.
- route-optimal lane overrides behavior-selected lane.
- reference lane is overwritten by blue-dot lane.

Required invariant:

- During `EXECUTE_LANE_CHANGE_*`, both TEMP_DES and MPC reference use the
  behavior target lane.

### Curve Boundary Breach

Likely causes:

- speed cap does not activate early enough
- reference jumps on curve
- road boundary slack too permissive
- MPC is solving many goals at once: lane center, route progress, obstacle,
  stop, and speed

Required invariant:

- Curve speed cap must be active before entering the bend.
- Road boundary breach must be visible in every run report.
- Boundary cost should not be the first signal that the vehicle is already
  outside the lane.

## 3. Diagnostics Required For Every Run

Artifacts now expected:

- `planning_metrics_timeseries.csv`
- `control_timeseries.csv`
- `lane_reference_timeseries.csv`
- `fsm_transition_log.csv`
- `mpc_cost_history.csv`
- `planned_trajectory.csv`
- `tracking_error_timeseries.csv`
- `mpc_diagnostics.json`
- `analysis_report.txt`
- `analysis_plots.png`
- `tracking_dashboard.png`

Minimum dashboard panels:

- actual path vs reference path
- lateral/position error
- speed, acceleration, MPC vmax
- throttle/brake/steer
- FSM timeline
- MPC solver/cost timeline

## 4. Refactor Plan

Phase 1: Diagnostics first — **MOSTLY DONE**

- Keep current planner running.
- Make every run produce tracking/error/control/FSM/MPC reports.
- Use the reports to identify layer ownership for each failure.
- Remaining: `tracking_error_timeseries.csv` and `mpc_diagnostics.json` are
  referenced/expected but not currently written.

Phase 2: Behavior decision contract — **NOT STARTED**

- Replace implicit strings with a behavior command object:
  - state
  - target lane
  - stop target
  - speed target
  - reason
  - priority
- Enforce invariants at the interface before path generation.
- `RuleBasedBehaviorPlanner.update()` already computes all of these fields
  (state/decision, target lane, stop target, candidate reasons/priority via
  `_simple_candidate_evaluation` / `_set_candidate_evaluation`) but returns
  them as a loose `Dict[str, Any]`, and `planning_runner.py` reads them with
  `.get(...)`. The data already has the right shape; it just needs a typed
  wrapper (e.g. dataclass) and call-site updates.

Phase 3: Path and speed split — **PARTIAL**

- TEMP_DES becomes a display/anchor output.
- MPC receives a continuous reference path plus speed envelope.
- Stop profile and curvature cap live in speed layer.
- Reference-path continuity exists (`_build_lane_reference_samples_to_target`,
  `_build_route_reference_samples_from_anchor` in `temp_destination.py`).
  Curvature cap is already computed and applied before the MPC solve in
  `planning_runner.py`. What's missing is a single speed-envelope object —
  stop profile, follow profile, and curvature cap are currently computed in
  separate places (`MPC/mpc.py`, `planning_runner.py`) rather than merged
  into one speed-layer output consumed by MPC.

Phase 4: MPC simplification — **MOSTLY DONE**

- Keep dynamics and actuator limits hard.
- Move red-light/stop/following/curve logic outside MPC.
- Keep lane center, boundary, obstacle, and smoothness as weighted soft costs.
- Add fail-safe fallback:
  - if solver fails, brake gently and hold last safe path for one cycle
  - if repeated failure, emergency stop
- Hard/soft constraint split and pre-solve curvature capping are implemented
  in `MPC/mpc.py` / `mpc.yaml`. Remaining gap: on solver failure the code
  resets and retries, then falls back to the reference rollout, but there is
  no explicit "brake gently" / escalating-emergency-stop behavior distinct
  from that fallback.

Phase 5: Scenario regression suite — **DONE**

- Red light stop/release
- Green light pass
- Lane change target consistency
- Curve boundary compliance
- Static blocked lane reroute
- Dynamic lead vehicle following
- Mixed traffic with SUMO/autopilot actors
- `opencda/planning_module/tests/` already has ~26 test files covering most
  of these cases (red light, intersection behavior, reroute, lane keeping,
  lane reference, route planning, multi-town/SUMO scenarios).

