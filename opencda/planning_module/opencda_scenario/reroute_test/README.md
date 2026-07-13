# Reroute Test

Standalone CARLA-only Town10 reroute visualizer.

It watches three named markers:

- `start`
- `end`
- `close`

The script snaps each marker to the nearest driving waypoint, blocks only the
sampled graph waypoint that corresponds to `close`, and runs the planning
module's A* global planner to find a route from `start` to `end`. The route is
computed once at startup and kept visible as a yellow dotted path in CARLA
debug view. It also opens a pygame window with a top-down RGB camera centered
over the `close` marker and zoomed to keep the route in view.

Run it from the repository root with:

```bash
python -m opencda.planning_module.opencda_scenario.reroute_test --host 127.0.0.1 --port 2000
```

Or from `opencda/planning_module`, just like the other scenarios:

```bash
python main.py reroute_test
```

By default it will load `Town10HD_Opt` if a different map is currently open.
