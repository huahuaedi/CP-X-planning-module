# Map-Layer Regression Suite

This suite validates the complete planning map interface before Behavior,
reference generation, speed planning, or MPC tuning is evaluated.

## Scenarios

| Scenario | Map contract coverage |
|---|---|
| `cpx_mapreg_town06_straight` | Longitudinal identity and road/section transitions without lateral motion |
| `cpx_mapreg_town06_right_turn` | Right adjacency, route-required lane change, junction connector, and exit lane |
| `cpx_mapreg_town05_turn` | A second CARLA town, turn topology, and cross-segment continuity |

Run each scenario separately against CARLA 0.9.12:

```bash
python opencda.py -t cpx_mapreg_town06_straight -v 0.9.12
python opencda.py -t cpx_mapreg_town06_right_turn -v 0.9.12
python opencda.py -t cpx_mapreg_town05_turn -v 0.9.12
```

Then aggregate all results:

```bash
python -m opencda.planning_module.tools.analyze_map_regression_suite \
  --output opencda/planning_module/opencda_bridge/map_regression_summary.json \
  --fail-on-violations
```

The suite passes only when every scenario has:

- a valid pose-to-HD-map match on every recorded frame;
- valid geometry, confidence, lane width, candidate evidence, and heading;
- a matched lane contained in rolling corridor `0`;
- consistent matcher and local-frame ego identities;
- a rolling map window of at least 99 m forward and backward;
- a unique deterministic `lane_to_offset` decision backed by the reachable
  corridor graph (shared downstream lanes at merges are allowed);
- consistent Global Route target membership, offset, and physical direction;
- no cache reuse across a lane transition;
- no unauthorized lateral transition;
- no short `A -> B -> A` lane bounce.
