# CP-X Planning Architecture

## Runtime path

```text
OpenCDA scenario / VehicleManager
  -> OpenCDAPlanningAdapter
  -> PlannerInputFrame
  -> ScenarioManager
  -> Behavior proposal
  -> Candidate generation and prediction-aware evaluation
  -> ReferenceGenerator
  -> ReferencePipeline (condition -> recover once -> final gate)
  -> MPC tracking
  -> Platform velocity/steering adapter
  -> SafetySupervisor (final control authority)
  -> PlannerOutput / carla.VehicleControl
```

`python opencda.py -t <scenario> -v 0.9.12` is the only supported scenario
entrypoint. OpenCDA owns simulation, localization, perception, V2X, map
lifecycle, and `apply_control()`. CP-X owns planning after `PlannerInputFrame`.

## Decision authority

| Stage | May do | Must not do |
| --- | --- | --- |
| ScenarioManager | Select lane-follow, approach, stop, turn, creep, or recovery context | Generate steering or rewrite map geometry |
| BehaviorPlanner | Propose behavior, lane, and intent speed | Apply CARLA control |
| CandidateEvaluator | Rank candidates and reject unsafe/prediction-infeasible candidates | Repair a rejected reference |
| CPXRouteManager | Own the global route, monotonic progress, route-segment branch matching, and the latched CARLA junction connector | Generate MPC controls or select behavior |
| ReferenceGenerator | Generate straight, lane-change, curvature-feasible turn, stop, recovery, and emergency-stop geometry | Select behavior, retain control state, or bypass the final gate |
| ReferencePipeline | Select mode-specific generation intent and velocity semantics | Reimplement waypoint/polyline geometry or choose an unrelated behavior |
| FinalReferenceGate | Apply the mode-specific reference contract immediately before MPC | Repair, reinterpret, or silently accept an invalid reference |
| MPC | Test dynamic feasibility and track an accepted reference | Select a route or traffic-light policy |
| SafetySupervisor | Enforce collision stop, signal-stop envelope, road-boundary stop, and final actuator/rate limits | Create lane changes or routine speed plans |

The decision chain is recorded in `decision_veto_chain`. A rejected candidate
does not enter MPC. Every reference must pass `FinalReferenceGate` after all
conditioning and immediately before the solver. A stop candidate without a
valid reference produces an emergency stop. SafetySupervisor is the final
control authority.

## Stop semantics

- `stop_at_intersection` and `stop_sign` set `normal_stop=True`, request a
  zero-terminal-speed reference, and use bounded approach/hold braking.
- Once a committed normal stop falls below the configured low-speed capture
  threshold, MPC solving is suspended and a deterministic light hold brake is
  used until the stop is released. Emergency braking remains a separate path.
- `emergency_brake` sets `emergency_brake=True` and authorizes immediate
  maximum braking.
- Candidate hard gates and SafetySupervisor collision vetoes may escalate the
  final control to emergency braking without rewriting the original behavior.
  Diagnostics record `normal_stop_requested`, `emergency_brake_requested`, and
  `emergency_brake_control_active` separately.

## Single-owner rules

`full_cpx_mpc` uses the `unified_full_v1` architecture profile:

- Reference validity owner: `FinalReferenceGate`.
- Reference geometry owner: `ReferenceGenerator`.
- Reference repair owner: `ReferencePipeline`, using `ReferenceGenerator`
  geometry APIs before the gate.
- Lane-change authorization owner: `BehaviorPlanner` and `CandidateEvaluator`.
- Routine speed owner: `SpeedPlanner`.
- Traffic-light temporal state owner: `TrafficLightMemory`.
- Control time-alignment owner: `MPCControlBuffer`.
- Final hazard and actuator-limit owner: `SafetySupervisor`.

The platform velocity/steering adapter runs before signal, road-boundary, and
hazard supervision. No actuation mapper is allowed to regenerate a command
after `SafetySupervisor`; the supervised command is the command returned to
OpenCDA for `apply_control()`.

The profile disables the OpenCDA-style stateful conditioner, low-speed lateral recovery, lane-follow
speed recovery, negative-acceleration release, bridge-level dense-traffic
lane-change lock, bridge-level overspeed correction, and the legacy
pre-gate strict reference veto. Pre-gate validators may diagnose or repair;
only `FinalReferenceGate` may reject the final reference. The overlapping
state and veto authority were the main sources of oscillation and
scenario-specific switch combinations.
Legacy `mode2`, `opencda_reference_mpc`, and their reference/trajectory
memories have been removed. `full_cpx_mpc` is the only CP-X planner mode.

The obsolete bridge-private launch, overspeed, negative-acceleration release,
and lateral-recovery control guards have been removed. Signal debounce now
lives in `pipeline/traffic_light_memory.py`; `ScenarioManager` consumes the
resolved state without adding a second debounce timer.

## Reference generation boundary

`pipeline/reference_generator.py` owns all geometric sampling used by MPC.
Its public API returns either `GeneratedReference` (named samples,
destination, source, and reason) or explicitly named sample collections:

- route and current-lane center sampling;
- quintic lane-change and lane-recovery connectors;
- ego-anchored intersection-turn connectors;
- route-aligned polyline projection and resampling;
- independent traffic-control stop references;
- straight and ego-heading emergency-stop references;
- heading, spacing, and lateral geometry helpers.

Callers do not unpack anonymous `(destination, samples)` tuples and do not call
generator-private waypoint helpers. This prevents a destination vector from
being silently treated as a trajectory.

`pipeline/reference_pipeline.py` is the sole post-generation owner. Both
candidate evaluation and final MPC input use the same `ReferencePipelineRequest`.
The pipeline:

1. resolves the contract mode from behavior intent;
2. generates mandatory stop/emergency geometry;
3. removes non-finite, behind-ego, and duplicate samples;
4. bounds turn curvature against the MPC wheelbase/steering envelope;
5. validates the mode-specific contract;
6. performs at most one mode-specific recovery;
7. applies `FinalReferenceGate` to the exact reference sent to MPC.

A short lane-follow recovery hysteresis is updated only by `finalize()`.
Candidate conditioning remains side-effect free. This prevents the final local
destination and reference source from alternating when raw geometry crosses a
contract threshold on adjacent frames.

A committed lane change is never rebuilt as lane follow. If its locked
trajectory is invalid, the candidate boundary must explicitly continue or
abort the maneuver. The bridge no longer owns active reference stabilization
or final-gate orchestration.

The reference contract derives its curvature ceiling from
`tan(max_steer_rad) / wheelbase` with a safety factor. A wider
mode-specific limit cannot override this vehicle limit. Candidate turn
geometry is conditioned before evaluation and checked again at the final
gate.

Road-boundary warning, minor breach recovery, and critical breach are separate
SafetySupervisor outcomes. A warning uses a closed-loop speed envelope. A
minor breach first stabilizes and then permits low-speed recovery creep along
the MPC correction. A critical breach and every collision retain hard braking,
so a small map-envelope crossing is not an absorbing stopped state.

## Map boundary

Planning algorithms consume mapping-based points and poses. `utility/map_api.py`
is the sole compatibility boundary for native CARLA `world_map`,
`carla.Location`, and `Transform` objects. Route and traffic-control helpers
therefore use one algorithm for OpenCDA, CARLA, and imported MDrive scenarios.

## Remaining migration boundary

`CPXMPCPlannerBridge.execute_planning_pipeline()` is the temporary public
lifecycle port while the executable stages are migrated out of the bridge.
The large legacy stage body remains behind that method while individual stages
are moved into pipeline services. `OpenCDARuntimePort` is the only class allowed
to call legacy bridge-private input helpers.
