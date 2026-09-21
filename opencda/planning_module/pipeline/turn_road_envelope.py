"""Rolling MPC road envelope for an intersection turn.

Builds the per-turn road-envelope payload the MPC uses to keep the ego body
inside a tube around the turn reference.  Pure: everything it needs is passed
in, and it returns ``None`` outside a turn, when disabled, or when the
reference yields no envelope blocks.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def rolling_turn_envelope_payload_world(
    *,
    config: Mapping[str, Any],
    mpc: Any,
    vehicle: Any,
    behavior_decision: str,
    reference_samples: Sequence[Mapping[str, object]],
    ego_x_m: float,
    ego_y_m: float,
) -> Optional[Mapping[str, object]]:
    """Build an MPC road envelope for only the current turn horizon."""

    if str(behavior_decision or "").strip().lower() not in {
        "intersection_turn_left",
        "intersection_turn_right",
    }:
        return None
    if not bool(config.get("turn_mpc_road_envelope_enabled", True)):
        return None
    from opencda.planning_module.pipeline.candidate_pipeline import (
        build_turn_reference_envelope_blocks,
    )
    from opencda.planning_module.MPC.lane_keep import (
        road_envelope_conservativeness_correction,
    )

    extent = getattr(getattr(vehicle, "bounding_box", None), "extent", None)
    ego_half_width_m = max(
        0.1,
        float(
            getattr(
                extent,
                "y",
                config.get("metrics_ego_half_width_m", 1.0),
            )
        ),
    )
    blocks = build_turn_reference_envelope_blocks(
        reference_samples=reference_samples,
        ego_half_width_m=float(ego_half_width_m),
        ego_x_m=float(ego_x_m),
        ego_y_m=float(ego_y_m),
        safety_margin_m=max(
            0.0,
            float(
                config.get(
                    "turn_mpc_road_envelope_safety_margin_m",
                    config.get(
                        "reference_contract_turn_boundary_margin_m",
                        0.15,
                    ),
                )
            ),
        ),
        default_lane_width_m=float(getattr(mpc, "lane_width_m", 3.5)),
        longitudinal_overlap_m=max(
            0.0,
            float(config.get("turn_mpc_road_envelope_overlap_m", 0.75)),
        ),
    )
    if not blocks:
        return None
    rho = float(getattr(mpc, "road_envelope_rho", -8.0))
    return {
        "blocks": blocks,
        "epsilon0": road_envelope_conservativeness_correction(
            blocks,
            rho=float(rho),
        ),
        "rho": float(rho),
        # This is recovery slack, not extra drivable width.  Keeping the
        # 10k envelope penalty means MPC still prefers the body-safe tube,
        # while the ceiling prevents a tracking error at the turn apex
        # from making the entire QP mathematically infeasible. A finite
        # 1.5m ceiling was observed going infeasible on a real
        # intersection turn (MDrive Intersection_Deadlock_Resolution/3,
        # confirmed via MPC._last_infeasibility_diagnostic:
        # road_envelope_term_active with road_boundary/corridor/terminal
        # all inactive).
        #
        # An unbounded ceiling is NOT the fix, though it was tried first:
        # this envelope is derived from the reference/lane geometry, not
        # from live sensing of curbs, poles, medians or other static
        # scene geometry -- MDrive's ground-truth perception feed here
        # only ever supplies vehicles + traffic lights (see
        # _MDriveVehicleManager.perception_manager.objects in
        # cpx_planner_adapter.py), never static world obstacles. With no
        # independent check against real geometry, letting MPC accept
        # an arbitrarily large deviation from the reference tube has no
        # backstop: re-tested on the same scenario with the ceiling
        # unbounded, the earlier infeasibility/stall was gone but the
        # vehicle then drifted far enough off its reference during
        # recovery to collide with unmodeled static scene geometry
        # partway through the turn exit (confirmed via a sudden speed
        # collapse -- 2.34 m/s to under 0.2 m/s in one control tick, at
        # zero commanded curvature and no MPC-side warning beforehand --
        # not a planner-commanded stop). 3.0m is a middle ground: about
        # 2x the original ceiling (enough headroom for the turn geometry
        # that made 1.5m infeasible) while still bounding how far MPC
        # can let the solution drift from the reference tube in the
        # absence of any static-obstacle sensing to catch it.
        "max_slack_m": (
            lambda configured: (
                float(configured)
                if configured is not None and float(configured) > 0.0
                else 3.0
            )
        )(
            config.get("turn_mpc_road_envelope_recovery_slack_m")
        ),
    }
