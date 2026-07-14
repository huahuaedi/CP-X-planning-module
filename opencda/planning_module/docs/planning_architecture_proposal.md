# Planning Architecture Proposal

This document describes the current planning stack and the target direction
for making it more stable, inspectable, and easier to optimize. The practical
goal is to stop treating each scenario failure as an isolated bug and instead
assign every failure to a clear layer: route, behavior, prediction, path,
speed, MPC, or diagnostics.

The style is close to common autonomous-driving stacks such as Apollo,
Autoware, and CARLA BehaviorAgent: route intent, behavior decisions, path
generation, speed shaping, and low-level control are separate responsibilities.

## 0. Current Status

Status tags:

- **DONE**: implemented and covered by tests or runtime artifacts.
- **PARTIAL**: implemented but still mixed with another layer or not fully
  typed.
- **MISSING**: still mostly implicit or embedded in runner logic.

| Layer / Item | Status | Current implementation |
| --- | --- | --- |
| Planning context / world-model boundary | PARTIAL | `utility/planning_context.py` now defines typed per-cycle context objects for ego, route, traffic control, and targets. It is integrated into `planning_runner.py` for behavior input normalization and diagnostics, but the runner still owns most context construction. |
| Route layer | PARTIAL | Global route points, route-optimal lane, and maneuver labels are available, but route corridor logic is still embedded in `planning_runner.py` and `behavior_planner/temp_destination.py`. |
| Behavior FSM | DONE | `behavior_planner/planner.py` chooses emergency brake, traffic/stop-sign stop, reroute, lane-change, or lane follow. FSM states include `IDLE`, `LANE_KEEP`, `PREPARE_LANE_CHANGE_*`, `EXECUTE_LANE_CHANGE_*`, and abort/reset states. |
| Behavior command contract | PARTIAL | `behavior_planner/contract.py` documents the command shape. Runtime still passes a loose dict, but `_enforce_behavior_command_invariants()` corrects unsafe lane/stop inconsistencies before output. |
| Traffic-light stop logic | PARTIAL | Red/yellow/green detection, stop target matching, signal memory, unknown release, and raw actor state logging are implemented. Recent issue: red was detected but speed profile still allowed high speed; this is now addressed with conservative stop speed shaping. |
| Other-vehicle prediction | PARTIAL | Behavior risk checks use CP `predicted_trajectory` when provided and constant-acceleration fallback otherwise. This is implemented in `behavior_planner/trajectory_risk.py` and `pipeline/prediction.py`. |
| Temp destination | PARTIAL | `temp_des` is stabilized and guarded against behind-ego / wrong-lane jumps, but it is still both a display point and a local goal input to MPC. Stop target movement is now step-limited. |
| Reference intent contract | PARTIAL | `behavior_planner/reference_generator.py` now chooses whether MPC should track lane center, route branch, lane-change, stop, or follow-lead reference. This is the first step toward separating `temp_des` from the actual MPC reference. |
| Lane-center reference | PARTIAL | `lane_center_reference_samples` are generated for MPC. Non-lane-change jumps are frozen, and repeated freeze can re-anchor to ego heading. Route-branch reference is no longer treated as ordinary lane-follow noise inside junctions. |
| Speed layer | PARTIAL | Curvature, IDM, stop, and reference-jump caps are stacked before MPC using `utility/speed_profile.py`, but there is not yet a first-class speed-envelope object. |
| MPC cost profiles | DONE | Different behavior modes use different cost profiles. Profile switching now has hysteresis/min-hold and weight blending to reduce oscillation. |
| MPC hard/soft split | DONE | Vehicle bounds are hard constraints. Lane center, road boundary, obstacle, attractive, and control terms are soft costs. |
| MPC fail-safe | DONE | Solver failure falls back to a path-holding braking trajectory and escalates to stronger braking after repeated failures. |
| Diagnostics | DONE | Runtime writes temp destination, lane reference, control, cost, FSM, planned trajectory, metrics, and run status artifacts. |
| Regression tests | DONE | Tests cover traffic lights, temp destination guards, speed caps, prediction risk, planning pipeline, MPC fail-safe, route/reference behavior, and scenario helpers. |

## 1. Layered Runtime Flow

Current high-level flow:

```text
CARLA/SUMO/CP inputs
  -> PlanningContext: ego, route, traffic control, separated targets
  -> prediction frame and lane safety
  -> behavior planner command
  -> temp_des and lane_center_reference_samples
  -> speed caps and MPC cost profile selection
  -> MPC trajectory and control sequence
  -> CARLA VehicleControl
  -> diagnostics CSV/JSON artifacts
```

The most important rule is that `temp_des` is not the whole plan. MPC also
uses lane-center reference samples, speed caps, cost profile, obstacle
predictions, warm-start state, and hard vehicle constraints.

The new `PlanningContext` boundary separates target semantics:

```text
stop_target: longitudinal stopping target for traffic control / speed layer
local_goal / temp_des: rolling local goal and blue-dot visualization
lane_reference: continuous lateral/path tracking reference for MPC
final_goal: mission completion target
global_route: mission-level route hint, not a direct MPC reference
```

This is intentionally a boundary object, not a new planner. Its purpose is to
make the layer inputs inspectable and prevent a stop line, global route point,
and MPC tracking reference from being treated as the same object.

## 1.2 Reference Intent Contract

Current file:

- `behavior_planner/reference_generator.py`

Purpose:

- Convert a behavior decision plus route/lane state into a first-class
  reference intent before MPC reference samples are generated.
- Prevent `temp_des`, selected lane, stop target, and global route from
  competing for the same meaning.

Inputs:

- Behavior decision: lane follow, lane change, stop, emergency/follow lead.
- Behavior FSM state: idle, prepare lane change, execute lane change, abort.
- Current lane id, selected/reference target lane id, route-optimal lane id.
- Junction flag and traffic-control lock flag.
- Global-route-reference gate result.

Outputs:

- `ReferenceIntent.mode`
  - `lane_follow`: track selected/current lane centerline.
  - `route_branch_follow`: track global route branch through a junction or
    route rejoin segment.
  - `lane_change`: track a committed lane-change transition reference.
  - `stop`: keep lateral reference lane-based; stop target belongs to speed
    layer unless final-stop snapping is explicitly active.
  - `follow_lead`: keep lateral reference lane-based while speed/follow target
    handles longitudinal behavior.
- `ReferenceIntent.target_lane_id`
- `ReferenceIntent.follow_global_route_lane`
- `ReferenceIntent.reason` for CSV diagnostics.

Current integration:

- `planning_runner.py` calls `select_reference_intent()` after route-reference
  gating.
- `lane_reference_timeseries.csv` records `reference_intent_mode` and
  `reference_intent_reason`.
- Route-branch intent bypasses ordinary lane-follow first-sample/lateral
  reanchor guards, because the desired route branch can be laterally offset in
  a junction.

Key invariant:

- MPC should track the reference generated from `ReferenceIntent`, not chase a
  raw global-route point or a jumpy `temp_des`.

## 1.1 Planning Context Boundary

Current file:

- `utility/planning_context.py`

Inputs:

- Ego state and lane context.
- Current route summary and active route points.
- CP/CARLA traffic-control context.
- Stop target mapping, if available.
- Temporary destination and final goal.
- Behavior/FSM state and reference gate status.

Outputs:

- `EgoPlanningState`
- `TrafficControlContext`
- `RouteContext`
- `TargetContext`
- `PlanningContext.trace_fields()` for CSV diagnostics.

Required invariants:

- `TrafficControlContext.stop_target` is only a longitudinal stop target.
- `RouteContext` is a mission-level hint and must not directly pull MPC.
- `TargetContext.local_goal`, `stop_target`, and `final_goal` must remain
  separate.
- MPC should receive `lane_center_reference_samples`; global route reference
  is allowed only if the route-reference gate explicitly passes.

Current integration:

- `planning_runner.py` builds a `PlanningContext` before behavior planning.
- Behavior traffic-signal input is read from `PlanningContext`.
- After reference gating, `PlanningContext` records reference priority and
  global-route-reference gate result.
- `temporary_destination_timeseries.csv` and `lane_reference_timeseries.csv`
  include flattened planning-context fields.

Remaining gap:

- Context construction still lives inside `planning_runner.py`. The next
  cleanup step is to move context construction into a dedicated
  `PlanningContextBuilder` so the runner becomes an event loop rather than a
  planning-policy owner.

## 2. Route Layer

Current files:

- `utility/global_planner.py`
- `behavior_planner/temp_destination.py`
- route handling inside `planning_runner.py`

Inputs:

- Global route points.
- Current ego pose and lane context.
- CARLA map topology.
- Final destination.

Outputs:

- Current route summary.
- Route-optimal lane id.
- Upcoming macro maneuver: `straight`, `left`, `right`, etc.
- Global route visualization points.
- Route-follow latch used after reroute.

Constraints / invariants:

- The route layer must not decide stop, emergency brake, or lane-change safety.
- Route-preferred lane must not override a committed behavior lane during
  active lane change.
- During strict lane follow, the reference lane should stay on the current ego
  lane rather than being pulled back to the global route lane.

Recent optimizations:

- Route fallback is disabled when it would pull non-lane-change reference into
  the wrong lane.
- Reference generation can reject a first sample whose lane id does not match
  the expected lane.
- Reroute latch is cleared when an active lane change is executing toward a
  lane that is not the route-optimal lane.

Remaining gap:

- There is still no standalone route-corridor module. Route decisions are
  spread across the runner and temp destination builder.

## 3. Prediction And Risk Layer

Current files:

- `pipeline/prediction.py`
- `behavior_planner/trajectory_risk.py`
- lane-safety logic in `behavior_planner/lane_safety.py`

Inputs:

- Dynamic object snapshots from CARLA/SUMO/CP.
- CP-provided `predicted_trajectory`, when available.
- Object position, heading, speed, and optional acceleration.
- Ego lane context and target lane candidate.

Outputs:

- Prediction frame.
- Per-lane prediction risk.
- Future front/rear gap.
- Future TTC.
- Risk debug summary consumed by behavior planner and artifacts.

Constraints / invariants:

- CP `predicted_trajectory` has priority when provided.
- If no trajectory is provided, use constant-acceleration fallback.
- If acceleration is unavailable, the model degenerates to constant velocity.
- Prediction risk can block or abort lane change, but it should not directly
  change MPC constraints.

Recent optimizations:

- Added constant-acceleration fallback prediction.
- Added configurable prediction model:
  - `lane_change_prediction_model: constant_acceleration`
  - `lane_change_prediction_max_abs_acceleration_mps2: 4.0`
- Lane-change candidate evaluation now considers predicted future gaps and
  TTC, not only instantaneous lane safety.

Remaining gap:

- Long-horizon interaction prediction is still simple. Current prediction is
  sufficient for candidate risk, but not a full multi-agent planner.

## 4. Behavior Layer

Current files:

- `behavior_planner/planner.py`
- `behavior_planner/contract.py`
- `behavior_planner/traffic_light_stop.py`

Inputs:

- Route summary and route-optimal lane.
- Ego state and lane context.
- Lane safety scores.
- Prediction risk.
- Traffic-light / stop-sign context.
- CP control messages.
- Front/follow target.
- Current FSM state and timers.

Outputs:

- Behavior decision:
  - `lane_follow`
  - `lane_change_left`
  - `lane_change_right`
  - `stop_at_intersection`
  - `stop_sign`
  - `reroute`
  - `emergency_brake`
- FSM state.
- Target lane id.
- Selected lane id.
- Stop target or follow target.
- Candidate summaries and risk debug.
- Traffic-light debug.

Priority order:

1. Emergency brake / imminent collision.
2. Traffic-light or stop-sign stop.
3. CP stop / intersection control.
4. Reroute / blocked lane.
5. Lane-change candidate evaluation.
6. Route preference.
7. Lane follow / car following.

Constraints / invariants:

- Stop and emergency brake must use ego lane, not a neighboring lane.
- Active lane-change commands must use behavior-selected target lane.
- Stop latch must release on green or when the stop target has been passed.
- Unknown traffic-light state must not preserve stale red forever.
- Unknown traffic-light state should not create a new stop unless there is a
  valid current stop target/latch.

Recent optimizations:

- Added short traffic-light memory:
  - `unknown_hold_s`
  - `unknown_release_s`
- Added raw signal actor state logging.
- Avoided stale-red stop when no current stop target exists.
- Added lane-change cooldown/commit windows after stop release, lane-change
  completion, and abort.
- Added stronger guard for unknown signal: an unknown signal with
  `should_stop_now=False` cannot newly trigger `stop_at_intersection`.

Remaining gap:

- The command is still a dict. A dataclass command should eventually replace
  loose `.get(...)` access in `planning_runner.py`.

## 5. Temp Destination Layer

Current files:

- `behavior_planner/temp_destination.py`
- destination shaping helpers in `planning_runner.py`

Inputs:

- Behavior decision.
- Target lane id.
- Current ego pose.
- Route points.
- Stop target or follow target.
- Previous temp destination.
- Mode context: normal / intersection.
- Dynamic lookahead configuration.

Outputs:

- Temporary destination state:
  - `x`
  - `y`
  - `v_ref`
  - `heading`
  - `lane_id`
  - `mode`
  - `road_id`
  - `entered_intersection`
- Debug history in `temporary_destination_timeseries.csv`.

Constraints / invariants:

- `temp_des` must stay ahead of ego unless it is a fixed stop target.
- During active lane change, `temp_des` must use behavior target lane.
- During `PREPARE_LANE_CHANGE_*`, `temp_des` should remain on current lane.
- During stop, target speed should be shaped by stop distance, not remain at
  cruise speed.
- A new stop target should not make `temp_des` jump tens of meters in one
  replan cycle.

Recent optimizations:

- Added temp destination smoothing and forward guard.
- Added junction lane-follow lock to avoid blue-dot lane hopping in
  intersections.
- Added post-stop lookahead smoothing.
- Added exact final-destination snap near final goal.
- Added stop target movement limiter:
  - `stop_target_max_destination_step_m: 5.0`
- Added conservative stop profile:
  - `stop_profile_braking_deceleration_mps2: 1.2`
  - `stop_profile_buffer_m: 5.0`

Important note:

- A stable `temp_des` is necessary but not sufficient for a stable MPC
  trajectory. MPC also tracks the whole lane reference, speed envelope,
  profile weights, obstacle costs, and warm-start state.

## 6. Path Reference Layer

Current files:

- `behavior_planner/temp_destination.py`
- reference fallback/stabilization helpers in `planning_runner.py`

Inputs:

- Ego pose.
- Behavior decision.
- Target lane id.
- Route points.
- Stop/follow target.
- Previous reference samples.
- Current maneuver and mode.

Outputs:

- `lane_center_reference_samples`, each with:
  - `x_ref_m`
  - `y_ref_m`
  - `heading_rad`
  - `lane_id`
  - road boundary width / offset fields
- `lane_reference_timeseries.csv`.

Constraints / invariants:

- Non-lane-change reference must not jump to a different lane.
- First reference sample must not be behind ego or laterally far away.
- Stop references should approach and hold the stop target.
- Lane-change reference can move toward target lane only after execute state.
- Reference should be continuous enough that MPC is not solving a new geometry
  every replan.

Recent optimizations:

- Added first-sample invalid checks:
  - behind ego
  - lateral jump
  - reversed heading
  - lane mismatch
  - non-lane-change discontinuity
- Added route fallback only when route-follow is allowed.
- Added heading fallback when route fallback is unsafe.
- Added reference stabilization:
  - `reference_stabilization_freeze_on_jump: true`
  - `reference_stabilization_jump_threshold_m: 2.25`
- Added re-anchor after repeated freezes:
  - `reference_freeze_reanchor_after_replans: 4`
- Added reference-jump speed cap via `reference_jump_speed_cap_mps()`.

Remaining gap:

- Freeze is still a guard, not a full reference smoother. The next structural
  step is to blend reference horizons or anchor them to the previous solved
  MPC trajectory.

## 7. Speed Layer

Current files:

- `utility/speed_profile.py`
- speed-cap application in `planning_runner.py`
- final stop cap inside `MPC/mpc.py`

Inputs:

- Behavior decision.
- Ego speed.
- Stop target distance.
- Lead vehicle gap and speed.
- Curvature estimate from reference samples.
- Reference jump magnitude.
- Configured maximum speed.

Outputs:

- Effective MPC max velocity.
- Active speed cap names.
- Temporary destination speed.
- Control trace fields:
  - `mpc_vmax_mps`
  - `temporary_destination_v_mps`
  - `path_speed_cap_active`
  - `path_speed_cap_reason`

Constraints / invariants:

- Effective max speed is the minimum of active caps.
- Red-light / stop-sign handling is a speed-profile problem once behavior has
  selected a stop.
- Curvature cap should activate before entering the curve.
- IDM should cap speed for following but should not create a false stop unless
  behavior is truly stopping.

Current caps:

- IDM following cap.
- Stop profile cap.
- Curvature cap.
- Reference-jump cap.
- Final-goal stop cap.

Recent optimizations:

- Extracted speed cap helpers into `utility/speed_profile.py`.
- Added sequential speed-cap application.
- Added conservative stop profile for red lights / stop signs.
- Added curvature-aware speed cap for bends.
- Added reference-jump cap to avoid chasing discontinuous path at speed.

Remaining gap:

- There is no explicit `SpeedEnvelope` object. Caps are computed and applied in
  sequence in the runner.

## 8. MPC Layer

Current files:

- `MPC/mpc.py`
- `MPC/mpc.yaml`

Inputs:

- Current ego state.
- Temporary destination state.
- Lane-center reference samples.
- Dynamic object snapshots and predictions.
- Current acceleration / steering.
- Active speed upper bound.
- Cost profile selected by behavior/mode.

Outputs:

- Planned trajectory.
- Control sequence.
- Runtime solver status.
- Cost breakdown.
- Lateral and heading error diagnostics.
- Fail-safe fallback trajectory when needed.

Hard constraints:

- Velocity bounds.
- Acceleration bounds.
- Jerk bounds.
- Steering angle bounds.
- Steering-rate bounds.
- Terminal velocity constraint when active.
- Kinematic model equality constraints.

Soft costs:

- Attractive destination cost.
- Lane-center cost.
- Road-boundary cost and slack.
- Obstacle repulsive cost.
- Control effort and rate smoothness.

Cost profiles:

- `lane_follow`
- `intersection_turn`
- `stop`
- `prepare_lane_change`
- `execute_lane_change`
- `recovery`

Constraints / invariants:

- MPC should not decide behavior. It should optimize the command produced by
  behavior/path/speed layers.
- Traffic-light state should not directly enter MPC; it should be converted
  into stop target and speed profile first.
- Cost profiles should not switch every frame.
- Solver failure must return a safe braking plan, not arbitrary controls.

Recent optimizations:

- Added `mode_cost_profiles` in `mpc.yaml`.
- Added profile blending:
  - `mode_cost_profile_blend_alpha: 0.18`
- Added runner-side cost profile hysteresis:
  - `mpc_cost_profile_min_hold_s: 1.5`
- Added stop/recovery safety preemption.
- Added fail-safe fallback with gentle braking and emergency escalation.
- Added planned trajectory trace:
  - `planned_trajectory_timeseries.csv`

Remaining gap:

- MPC still receives both `temp_des` and lane reference. This is acceptable
  short term, but the longer-term design should make `temp_des` a local anchor
  and make the continuous reference/speed envelope the primary MPC contract.

## 9. Diagnostics Layer

Runtime artifacts:

- `run_status.json`
- `planning_metrics.json`
- `planning_metrics_timeseries.csv`
- `temporary_destination_timeseries.csv`
- `lane_reference_timeseries.csv`
- `control_timeseries.csv`
- `mpc_cost_history.csv`
- `fsm_transition_log.csv`
- `planned_trajectory.csv`
- `planned_trajectory_timeseries.csv`
- `collision_events.csv`

Post-run artifacts:

- `analysis_report.txt`
- `analysis_plots.png`
- `tracking_dashboard.png`
- `tracking_error_timeseries.csv`
- `mpc_diagnostics.json`

Key fields for diagnosing instability:

- Temp destination:
  - `temp_destination_jump_m`
  - `temp_v_mps`
  - `effective_lookahead_m`
  - `traffic_signal_state`
  - `traffic_should_stop_now`
  - `stop_target_distance_m`
- Reference:
  - `reference_jump_m`
  - `reference_stabilized`
  - `reference_fallback_reason`
- MPC profile:
  - `mpc_cost_profile`
  - `requested_mpc_cost_profile`
  - `mpc_cost_profile_switch_reason`
  - `mpc_cost_profile_elapsed_s`
- MPC result:
  - `solver_status`
  - `Cost_ref`
  - `Cost_Control`
  - `Cost_RoadBoundary`
  - `Cost_Repulsive`
  - planned trajectory stage errors in `planned_trajectory_timeseries.csv`

Recommended dashboard panels:

- Actual path vs reference path.
- Temp destination history.
- Lane reference first-sample history.
- Speed, acceleration, MPC vmax.
- Control commands.
- FSM timeline.
- Behavior decision timeline.
- Cost profile requested vs applied.
- MPC solver status and costs.
- Traffic signal state vs stop target distance.

## 10. Recent Optimization Summary

Traffic-light / stop handling:

- Added raw signal state logging.
- Added short unknown memory and unknown release.
- Blocked new `stop_at_intersection` from unknown signal when
  `should_stop_now=False`.
- Added stop target passed release and green release.
- Added conservative stop speed profile so red-light detection actually
  reduces `temp_v_mps` / `mpc_vmax_mps` before the stop line.

Temp destination:

- Added smoothing.
- Added forward guard.
- Added junction lane-follow lock.
- Added stop target step limit.
- Added post-stop release smoothing.

Reference:

- Added first-sample validation.
- Disabled unsafe route fallback.
- Added freeze-on-jump for non-lane-change reference.
- Added freeze timeout re-anchor to ego heading.
- Added reference jump speed cap.

Behavior / lane change:

- Added candidate hysteresis and lane-change cooldown.
- Added prepare/execute separation so `PREPARE_LANE_CHANGE_*` does not pull
  the blue dot into the target lane too early.
- Added prediction risk with constant-acceleration fallback.
- Added abort/commit cooldown to avoid immediate reattempt oscillation.

MPC:

- Added behavior-mode cost profiles.
- Added profile min-hold and blending.
- Added fail-safe fallback and repeated-failure escalation.
- Added planned trajectory trace for direct comparison against reference.

Speed:

- Extracted speed caps into reusable helpers.
- Applied IDM, stop, curvature, and reference-jump caps sequentially.
- Added curvature speed cap for turns.
- Added stop speed cap before MPC solve.

## 11. Remaining Structural Work

High priority:

- Create a typed `BehaviorCommand` dataclass and remove loose dict access from
  the runner.
- Create a dedicated `PathReference` module that owns reference continuity,
  branch lock, smoothing, and stop/turn/lane-change reference generation.
- Create a `SpeedEnvelope` object with all active caps and reasons.
- Make route corridor a first-class module instead of mixing route logic into
  temp destination and runner.

Medium priority:

- Replace hard reference freeze with horizon blending.
- Anchor reference to previous solved trajectory when safe.
- Add explicit stop-state phases: approach, hold, release.
- Add per-mode MPC constraint/cost sanity checks.
- Add scenario-level regression thresholds for:
  - max temp destination jump
  - max reference jump
  - MPC infeasible count
  - max control jump
  - red-light stop distance

Low priority:

- Convert diagnostics into a single HTML dashboard.
- Add CI jobs for deterministic non-CARLA unit tests.
- Add replay scripts that compare two runs side by side.

## 12. Practical Debugging Rules

If the car ignores red:

1. Check `traffic_signal_state`, `traffic_should_stop_now`, and
   `stop_target_distance_m`.
2. If red is detected but `temp_v_mps` / `mpc_vmax_mps` remains high, the
   problem is speed profile.
3. If no stop target exists, the problem is traffic-light association.
4. If behavior is not `stop_at_intersection`, the problem is behavior gating.

If `temp_des` is stable but MPC trajectory is unstable:

1. Check `reference_jump_m`.
2. Check `reference_stabilized`.
3. Check requested vs applied MPC profile.
4. Check solver status.
5. Check `Cost_Control` and planned trajectory stage jumps.

If the vehicle cuts a curve or presses lane boundaries:

1. Check curvature cap activation.
2. Check reference heading and first sample.
3. Check road boundary cost.
4. Check speed entering the curve.

If lane change flickers:

1. Check FSM transitions.
2. Check candidate summaries.
3. Check prediction risk summary.
4. Check lane-change cooldown/hold reason.
5. Check whether `PREPARE` is pulling target lane too early.
