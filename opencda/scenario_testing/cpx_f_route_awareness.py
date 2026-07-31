# -*- coding: utf-8 -*-
"""CP-X use case F: cooperative perception for high-level path/route planning
awareness (complex road condition -- special event / emergency vehicle
blockage requiring an informed reroute decision).

Built on
``opencda/planning_module/carla_scenario/high_level_route_planning/emergency_vehicle_blockage.yaml``
(Town10HD_Opt) -- see that file for why it reuses the same mechanism/marker
as the plain ``high_level_route_planning`` (workzone) scenario rather than a
new map asset.

This script adds two more real CAVs positioned near that scenario's
"workzone" marker (resolved live against the running CARLA world, since a
marker's position is baked into the map and not known statically -- see
``opencda/scenario_testing/utils/cpx_scenario_bridge.py``'s
``resolve_marker_transform``/``align_transform_to_lane``), so OpenCDA's
native V2X (``opencda_bridge/cp_provider.py``) has more than one real
perceiving agent, on top of the legacy scenario's own file-based proactive
``lane_closure`` CP message.

NOTE -- not verified against a live CARLA session: the workzone marker's
own baked-in orientation is not guaranteed to point "upstream" toward where
cav1 approaches from versus "downstream" past it -- CARLA was not reachable
while writing this to confirm. `_UPSTREAM_OFFSET_M`/`_DOWNSTREAM_OFFSET_M`
below deliberately place one observer on each side of the marker along the
lane, so at least one of the two should end up positioned before cav1
reaches the workzone regardless of which way the lane heading actually
points; confirm and simplify once run in the simulator.
"""

from opencda.scenario_testing.utils.cpx_scenario_bridge import (
    add_observer_cav,
    align_observer_to_adjacent_lane,
    align_transform_to_lane,
    offset_transform_along_heading,
    resolve_marker_transform,
    run_legacy_scenario_port,
)

_WORKZONE_MARKER_NAME = "workzone"
_UPSTREAM_OFFSET_M = -20.0
_DOWNSTREAM_OFFSET_M = 20.0


def _configure_extra_cavs(scenario_params, world, carla_module, context):
    workzone_transform = resolve_marker_transform(world, carla_module, _WORKZONE_MARKER_NAME)
    if workzone_transform is None:
        print(
            "[cpx_f_route_awareness] Could not resolve the "
            f"'{_WORKZONE_MARKER_NAME}' marker; skipping observer CAVs."
        )
        return

    aligned_transform = align_transform_to_lane(context.global_planner, carla_module, workzone_transform)
    if aligned_transform is None:
        aligned_transform = workzone_transform

    for name, offset_m in (
        ("cav2_route_observer_a", _UPSTREAM_OFFSET_M),
        ("cav3_route_observer_b", _DOWNSTREAM_OFFSET_M),
    ):
        observer_transform = offset_transform_along_heading(aligned_transform, carla_module, offset_m)
        observer_transform = align_observer_to_adjacent_lane(
            world,
            carla_module,
            observer_transform,
        )
        add_observer_cav(
            scenario_params,
            name=name,
            spawn_transform=observer_transform,
            carla_module=carla_module,
            forward_offset_m=60.0,
            target_speed_mps=4.0,
        )


def run_scenario(opt, scenario_params):
    run_legacy_scenario_port(
        opt,
        scenario_params,
        loader_name="carla_scenario",
        legacy_scenario_name="emergency_vehicle_blockage",
        script_name="cpx_f_route_awareness",
        configure_extra_cavs=_configure_extra_cavs,
    )
