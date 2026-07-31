# -*- coding: utf-8 -*-
"""CP-X use case B: cooperative perception for geometric/line-of-sight
occlusion (curved roadway / unprotected turn, occluded oncoming vehicle).

Built on
``opencda/planning_module/carla_scenario/mdrive_unprotected_left_turn_1``
(Town05). cav1 makes an unprotected left turn; that legacy scenario's own
yaml already has a scripted oncoming vehicle
(``obstacles.npc_vehicles: [oncoming_left_turn_traffic]``, ``follow_mode:
literal`` so it actually crosses regardless of any traffic-light logic --
see that yaml's inline NOTE for how its coordinates were derived without a
live CARLA session).

This script adds two more real CAVs positioned further down the same
east-west road the oncoming vehicle travels, so OpenCDA's native V2X
(``opencda_bridge/cp_provider.py``) can share genuinely earlier detection of
it with cav1 -- the same rationale as
``opencda/scenario_testing/cpx_c_vru_awareness.py``, see that file's
docstring.

NOTE -- none of the coordinates below (cav1's own turn geometry, the
oncoming vehicle's path, or these two observer positions) have been verified
against a live CARLA session; CARLA was not reachable while writing this.
Treat every position here as a starting point to confirm/retune once run in
the simulator, not as ground truth.
"""

from opencda.scenario_testing.utils.cpx_scenario_bridge import (
    add_observer_cav,
    align_observer_to_adjacent_lane,
    run_legacy_scenario_port,
)

_ONCOMING_ROAD_Y = 94.868
# Near the oncoming vehicle's own scripted start point (-64.475, 94.868) --
# close enough that its own 50m ground-truth perception radius should cover
# the oncoming vehicle from tick zero.
_OBSERVER_FAR_XYZ_YAW = (-80.0, _ONCOMING_ROAD_Y, 0.3, 0.0)
# Closer to cav1's own turn point (~x=-124.475), as a second, closer-range
# corroborating/relaying observer as cav1 approaches the intersection.
_OBSERVER_NEAR_XYZ_YAW = (-115.0, _ONCOMING_ROAD_Y, 0.3, 0.0)


def _transform_from_xyz_yaw(carla_module, xyz_yaw):
    x, y, z, yaw = xyz_yaw
    return carla_module.Transform(
        carla_module.Location(x=x, y=y, z=z),
        carla_module.Rotation(yaw=yaw),
    )


def _configure_extra_cavs(scenario_params, world, carla_module, context):
    del context
    far_transform = align_observer_to_adjacent_lane(
        world,
        carla_module,
        _transform_from_xyz_yaw(carla_module, _OBSERVER_FAR_XYZ_YAW),
    )
    near_transform = align_observer_to_adjacent_lane(
        world,
        carla_module,
        _transform_from_xyz_yaw(carla_module, _OBSERVER_NEAR_XYZ_YAW),
    )
    add_observer_cav(
        scenario_params,
        name="cav2_occlusion_observer_far",
        spawn_transform=far_transform,
        carla_module=carla_module,
        forward_offset_m=60.0,
        target_speed_mps=4.0,
    )
    add_observer_cav(
        scenario_params,
        name="cav3_occlusion_observer_near",
        spawn_transform=near_transform,
        carla_module=carla_module,
        forward_offset_m=60.0,
        target_speed_mps=4.0,
    )


def run_scenario(opt, scenario_params):
    run_legacy_scenario_port(
        opt,
        scenario_params,
        loader_name="carla_scenario",
        legacy_scenario_name="mdrive_unprotected_left_turn_1",
        script_name="cpx_b_occlusion_awareness",
        configure_extra_cavs=_configure_extra_cavs,
    )
