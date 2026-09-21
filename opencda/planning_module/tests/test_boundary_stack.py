"""Road-boundary monitoring, boundary-recovery latching and the turn envelope.

Three pieces of safety-relevant logic that used to live inside the bridge:
  * the boundary-recovery state machine (persistence, hysteresis, cooldown),
  * the road-boundary measurement (footprint vs. lane corridor / drivable union),
  * the rolling MPC road envelope built for an intersection turn.

Only the FIXTURE section below knows where that logic lives.  The assertions
are written against behaviour so they can move with the code unchanged; the
existing tests in test_opencda_bridge_input_fusion.py covered only the happy
path of the first two and nothing at all of the envelope.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pipeline.boundary_recovery import BoundaryRecoveryTracker  # noqa: E402
from pipeline.road_boundary_monitor import RoadBoundaryMonitor  # noqa: E402
from pipeline.turn_road_envelope import rolling_turn_envelope_payload_world  # noqa: E402

# =========================== FIXTURES (the only part tied to the layout) ===========================


class RecoveryHarness:
    def __init__(self, config):
        self._tracker = BoundaryRecoveryTracker(dict(config))

    def update(self, **kwargs):
        self._tracker.update(**kwargs)

    def reset(self):
        self._tracker.reset()

    @property
    def request(self):
        return self._tracker.request

    @property
    def trigger_frames(self):
        return self._tracker.trigger_frames

    @property
    def infeasible_frames(self):
        return self._tracker.infeasible_frames

    @property
    def cooldown_until_s(self):
        return self._tracker.cooldown_until_s

    def set_counters(self, *, trigger=None, infeasible=None, cooldown=None):
        if trigger is not None:
            self._tracker.trigger_frames = trigger
        if infeasible is not None:
            self._tracker.infeasible_frames = infeasible
        if cooldown is not None:
            self._tracker.cooldown_until_s = cooldown


class MonitorHarness:
    def __init__(self, config, vehicle, generator):
        self._monitor = RoadBoundaryMonitor(
            dict(config),
            vehicle_provider=lambda: vehicle,
            generator_provider=lambda: generator,
        )

    def measure(self, ego_location, **kwargs):
        return self._monitor.measure(ego_location, **kwargs)

    @property
    def sample_count(self):
        return self._monitor.sample_count

    @property
    def breach_count(self):
        return self._monitor.breach_count


def envelope(config, mpc, vehicle, **kwargs):
    return rolling_turn_envelope_payload_world(config=dict(config), mpc=mpc, vehicle=vehicle, **kwargs)


# =================================== shared builders ===================================


def _snapshot(**overrides):
    base = {
        "road_boundary_sample_valid": True,
        "road_boundary_clearance_m": -0.2,
        "road_boundary_lateral_offset_m": 0.3,
        "road_boundary_heading_error_rad": -0.2,
    }
    base.update(overrides)
    return base


def _good_frame(clearance=0.5):
    return _snapshot(road_boundary_clearance_m=clearance)


# ====================================== recovery ======================================


def test_reset_clears_the_request_and_trigger_frames_only():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1})
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=1.0)
    assert harness.request.active is True
    harness.set_counters(infeasible=2, cooldown=99.0)

    harness.reset()

    assert harness.request.valid is False and harness.request.active is False
    assert harness.trigger_frames == 0
    assert harness.infeasible_frames == 2, "reset must not clear the infeasible streak"
    assert harness.cooldown_until_s == 99.0, "reset must not clear the cooldown"


def test_frames_below_the_requirement_count_up_but_do_not_latch():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 3})

    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.0)
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.1)

    assert harness.trigger_frames == 2
    assert harness.request.active is False
    assert harness.request.reason == "boundary_recovery_monitor"
    assert harness.request.valid is True


def test_a_frame_above_the_trigger_clearance_restarts_the_persistence_count():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 3, "boundary_recovery_trigger_clearance_m": -0.10})
    for step in range(2):
        harness.update(boundary_snapshot=_snapshot(road_boundary_clearance_m=-0.2),
                       behavior_decision="intersection_turn_left", sim_time_s=float(step))
    assert harness.trigger_frames == 2

    harness.update(boundary_snapshot=_good_frame(-0.05), behavior_decision="intersection_turn_left", sim_time_s=2.0)

    assert harness.trigger_frames == 0
    assert harness.request.active is False


def test_clearance_exactly_at_the_trigger_counts_as_a_violation():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1, "boundary_recovery_trigger_clearance_m": -0.10})
    harness.update(boundary_snapshot=_good_frame(-0.10), behavior_decision="intersection_turn_left", sim_time_s=0.0)
    assert harness.request.active is True


def test_required_frames_is_floored_at_one():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 0})

    # A healthy frame must not latch: a floor of zero would latch on nothing.
    harness.update(boundary_snapshot=_good_frame(0.5), behavior_decision="intersection_turn_left", sim_time_s=0.0)
    assert harness.request.active is False

    # ...while a single violating frame is enough.
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.1)
    assert harness.request.active is True


def test_latched_recovery_holds_until_the_release_clearance():
    harness = RecoveryHarness({
        "boundary_recovery_trigger_frames": 1,
        "boundary_recovery_trigger_clearance_m": -0.10,
        "boundary_recovery_release_clearance_m": 0.10,
    })
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.0)
    assert harness.request.active is True

    harness.update(boundary_snapshot=_good_frame(0.05), behavior_decision="intersection_turn_left", sim_time_s=1.0)
    assert harness.request.active is True, "between trigger and release the latch holds"
    assert harness.request.reason == "boundary_recovery_latched"

    harness.update(boundary_snapshot=_good_frame(0.10), behavior_decision="intersection_turn_left", sim_time_s=2.0)
    assert harness.request.active is False, "at the release clearance the latch drops"


def test_release_clearance_can_never_sit_below_the_trigger_clearance():
    harness = RecoveryHarness({
        "boundary_recovery_trigger_frames": 1,
        "boundary_recovery_trigger_clearance_m": -0.10,
        "boundary_recovery_release_clearance_m": -0.50,   # nonsense config
    })
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.0)
    assert harness.request.active is True

    harness.update(boundary_snapshot=_good_frame(-0.05), behavior_decision="intersection_turn_left", sim_time_s=1.0)

    assert harness.request.active is False, "release is floored at the trigger clearance (-0.10)"


@pytest.mark.parametrize("decision, direction", [
    ("intersection_turn_left", "left"),
    ("intersection_turn_right", "right"),
    ("  INTERSECTION_TURN_RIGHT ", "right"),
    ("lane_follow", ""),
    ("", ""),
    (None, ""),
])
def test_turn_direction_comes_from_the_behavior_suffix(decision, direction):
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1})
    harness.update(boundary_snapshot=_snapshot(), behavior_decision=decision, sim_time_s=0.0)
    assert harness.request.turn_direction == direction


def test_the_request_carries_the_measured_values_and_timestamp():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1})
    harness.update(
        boundary_snapshot=_snapshot(road_boundary_clearance_m=-0.31,
                                    road_boundary_lateral_offset_m=0.42,
                                    road_boundary_heading_error_rad=-0.17),
        behavior_decision="intersection_turn_left", sim_time_s=7.5)
    request = harness.request
    assert (request.clearance_m, request.lateral_offset_m, request.heading_error_rad) == (-0.31, 0.42, -0.17)
    assert request.timestamp_s == 7.5
    assert request.reason == "boundary_recovery_latched"


@pytest.mark.parametrize("snapshot", [
    _snapshot(road_boundary_sample_valid=False),
    _snapshot(road_boundary_clearance_m=float("nan")),
    _snapshot(road_boundary_lateral_offset_m=float("inf")),
    _snapshot(road_boundary_heading_error_rad=float("-inf")),
    _snapshot(road_boundary_clearance_m=""),
    _snapshot(road_boundary_clearance_m="not-a-number"),
    {"road_boundary_sample_valid": True},
    {key: value for key, value in _snapshot().items() if key != "road_boundary_sample_valid"},
], ids=["invalid-flag", "nan-clearance", "inf-offset", "inf-heading", "empty-clearance", "text-clearance", "missing-keys",
        "no-valid-flag"])
def test_an_unusable_sample_drops_any_latched_request(snapshot):
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1})
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.0)
    assert harness.request.active is True

    harness.update(boundary_snapshot=snapshot, behavior_decision="intersection_turn_left", sim_time_s=1.0)

    assert harness.request.valid is False and harness.request.active is False
    assert harness.trigger_frames == 0


@pytest.mark.parametrize("inside", [True, "True", "true", "1", 1])
def test_a_footprint_fully_on_the_drivable_union_never_latches(inside):
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1})
    harness.set_counters(infeasible=2)

    harness.update(
        boundary_snapshot=_snapshot(road_boundary_geometry_source="drivable_footprint:carla_driving_lane_union",
                                    road_boundary_drivable_inside=inside),
        behavior_decision="intersection_turn_left", sim_time_s=0.0)

    assert harness.request.active is False
    assert harness.infeasible_frames == 0, "the infeasible streak is cleared as well"


@pytest.mark.parametrize("source, inside", [
    ("route_tangent_strip", True),                                        # wrong geometry source
    ("drivable_footprint:carla_driving_lane_union", False),               # outside the union
    ("drivable_footprint:carla_driving_lane_union", "false"),
    ("drivable_footprint:carla_driving_lane_union", ""),
])
def test_only_a_drivable_footprint_that_is_inside_suppresses_the_latch(source, inside):
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1})
    harness.update(
        boundary_snapshot=_snapshot(road_boundary_geometry_source=source, road_boundary_drivable_inside=inside),
        behavior_decision="intersection_turn_left", sim_time_s=0.0)
    assert harness.request.active is True


def test_an_infeasible_recovery_below_the_streak_limit_keeps_the_request_untouched():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1, "boundary_recovery_max_infeasible_frames": 3})
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.0)
    before = harness.request

    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.1,
                   recovery_planned=True, recovery_reference_feasible=False)

    assert harness.infeasible_frames == 1
    assert harness.request is before, "an early return must not rebuild or reset the request"


def test_a_feasible_frame_restarts_the_infeasible_streak():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1, "boundary_recovery_max_infeasible_frames": 3})
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.0,
                   recovery_planned=True, recovery_reference_feasible=False)
    assert harness.infeasible_frames == 1

    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.1,
                   recovery_planned=True, recovery_reference_feasible=True)

    assert harness.infeasible_frames == 0


def test_infeasibility_is_ignored_when_no_recovery_was_planned():
    harness = RecoveryHarness({"boundary_recovery_trigger_frames": 1})
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=0.0,
                   recovery_planned=False, recovery_reference_feasible=False)
    assert harness.infeasible_frames == 0
    assert harness.request.active is True


def test_hitting_the_infeasible_limit_starts_a_cooldown_that_blocks_recovery():
    harness = RecoveryHarness({
        "boundary_recovery_trigger_frames": 1,
        "boundary_recovery_max_infeasible_frames": 2,
        "boundary_recovery_cooldown_s": 4.0,
    })
    for step in range(2):
        harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left",
                       sim_time_s=10.0 + step, recovery_planned=True, recovery_reference_feasible=False)
    assert harness.cooldown_until_s == pytest.approx(15.0), "cooldown = time of the failing frame + 4 s"
    assert harness.request.active is False

    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=14.9)
    assert harness.request.active is False, "still cooling down"

    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=15.0)
    assert harness.request.active is True, "cooldown expiry re-enables recovery"


def test_unconfigured_infeasible_limit_is_three_frames_and_cooldown_two_seconds():
    harness = RecoveryHarness({})
    for step in range(2):
        harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left",
                       sim_time_s=20.0 + step, recovery_planned=True, recovery_reference_feasible=False)
    assert harness.cooldown_until_s == -float("inf"), "two failures are still within the default limit"

    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left",
                   sim_time_s=22.0, recovery_planned=True, recovery_reference_feasible=False)

    assert harness.cooldown_until_s == pytest.approx(24.0), "default cooldown is 2.0 s from the third failure"


def test_the_cooldown_has_a_floor_and_the_streak_limit_is_at_least_one():
    harness = RecoveryHarness({
        "boundary_recovery_max_infeasible_frames": 0,
        "boundary_recovery_cooldown_s": 0.0,
    })
    harness.update(boundary_snapshot=_snapshot(), behavior_decision="intersection_turn_left", sim_time_s=5.0,
                   recovery_planned=True, recovery_reference_feasible=False)
    assert harness.cooldown_until_s == pytest.approx(5.1)


def test_default_recovery_thresholds():
    harness = RecoveryHarness({})
    for step in range(2):
        harness.update(boundary_snapshot=_snapshot(road_boundary_clearance_m=-0.10),
                       behavior_decision="intersection_turn_left", sim_time_s=float(step))
    assert harness.request.active is False, "default needs 3 consecutive frames"
    harness.update(boundary_snapshot=_snapshot(road_boundary_clearance_m=-0.10),
                   behavior_decision="intersection_turn_left", sim_time_s=2.0)
    assert harness.request.active is True, "default trigger clearance is -0.10 m"
    harness.update(boundary_snapshot=_good_frame(0.09), behavior_decision="intersection_turn_left", sim_time_s=3.0)
    assert harness.request.active is True, "default release clearance is +0.10 m"
    harness.update(boundary_snapshot=_good_frame(0.10), behavior_decision="intersection_turn_left", sim_time_s=4.0)
    assert harness.request.active is False


# ====================================== monitor ======================================


class StubGenerator:
    """Stands in for ReferenceGenerator; records what the monitor asks of it."""

    def __init__(self, occupancy=None, projection=None, drivable=None, raises=None):
        self.occupancy = occupancy
        self.projection = projection
        self.drivable = drivable
        self.raises = raises
        self.calls = []

    def _hit(self, name, kwargs):
        self.calls.append((name, kwargs))
        if self.raises:
            raise self.raises

    def project_reference_corridor(self, **kwargs):
        self._hit("project", kwargs)
        return self.projection

    def lane_corridor_occupancy(self, **kwargs):
        self._hit("occupancy", kwargs)
        return self.occupancy

    def drivable_footprint_occupancy(self, **kwargs):
        self._hit("drivable", kwargs)
        return self.drivable

    def called(self, name):
        return [kwargs for n, kwargs in self.calls if n == name]


def _occupancy(*, valid=True, offset=0.3, width=3.5, clearance=0.4, heading=0.05):
    return SimpleNamespace(valid=valid, lateral_offset_m=offset, lane_width_m=width,
                           footprint_clearance_m=clearance, heading_error_rad=heading)


def _projection(occupancy=None, **kw):
    fields = dict(occupancy=occupancy or _occupancy(), segment_index=3, segment_ratio=0.25,
                  raw_heading_rad=0.11, conditioned_heading_rad=0.04, continuity_limited=True,
                  reason="projection:ok")
    fields.update(kw)
    return SimpleNamespace(**fields)


def _drivable(*, valid=True, clearance=0.6, reason="drivable_footprint:carla_driving_lane_union", inside=True):
    return SimpleNamespace(valid=valid, min_clearance_m=clearance, reason=reason, inside=inside)


def _vehicle(x=2.4, y=1.0, yaw_deg=90.0):
    return SimpleNamespace(
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=x, y=y)),
        get_transform=lambda: SimpleNamespace(rotation=SimpleNamespace(yaw=yaw_deg)),
    )


def _samples(count=3):
    return [{"x_ref_m": float(i), "y_ref_m": 0.0} for i in range(count)]


_NO_DRIVABLE = {"road_boundary_carla_drivable_footprint_enabled": False}
_LOCATION = SimpleNamespace(x=1.0, y=2.0, z=0.7)

_DEFAULT_KEYS = {
    "road_boundary_sample_valid", "road_boundary_lateral_offset_m", "road_boundary_lane_width_m",
    "road_boundary_ego_half_width_m", "road_boundary_clearance_m", "road_boundary_breach",
    "road_boundary_heading_error_rad", "road_boundary_projection_segment_index",
    "road_boundary_projection_segment_ratio", "road_boundary_projection_raw_heading_rad",
    "road_boundary_projection_conditioned_heading_rad", "road_boundary_projection_continuity_limited",
    "road_boundary_projection_reason", "road_boundary_geometry_source", "road_boundary_drivable_inside",
}


def test_an_invalid_occupancy_returns_the_blank_result_and_counts_nothing():
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(), StubGenerator(occupancy=_occupancy(valid=False)))

    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0)

    assert set(result) == _DEFAULT_KEYS
    assert result["road_boundary_sample_valid"] is False
    assert all(value == "" for key, value in result.items() if key != "road_boundary_sample_valid")
    assert monitor.sample_count == 0 and monitor.breach_count == 0


def test_a_generator_failure_is_swallowed_into_the_blank_result():
    monitor = MonitorHarness({}, _vehicle(), StubGenerator(raises=RuntimeError("map gone")))
    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0)
    assert result["road_boundary_sample_valid"] is False
    assert set(result) == _DEFAULT_KEYS
    assert monitor.sample_count == 0


def test_without_a_reference_it_measures_against_the_lane_corridor():
    generator = StubGenerator(occupancy=_occupancy(offset=0.3, width=3.5, clearance=0.4, heading=0.05))
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(), generator)

    result = monitor.measure(_LOCATION, ego_yaw_rad=0.2, reference_samples=_samples(1))

    assert generator.called("project") == []
    assert len(generator.called("occupancy")) == 1
    assert result["road_boundary_sample_valid"] is True
    assert result["road_boundary_lateral_offset_m"] == 0.3
    assert result["road_boundary_lane_width_m"] == 3.5
    assert result["road_boundary_clearance_m"] == 0.4
    assert result["road_boundary_heading_error_rad"] == 0.05
    assert result["road_boundary_projection_segment_index"] == ""
    assert result["road_boundary_projection_segment_ratio"] == ""
    assert result["road_boundary_projection_raw_heading_rad"] == ""
    assert result["road_boundary_projection_conditioned_heading_rad"] == ""
    assert result["road_boundary_projection_continuity_limited"] is False
    assert result["road_boundary_projection_reason"] == "lane_corridor_occupancy:map_fallback"
    assert result["road_boundary_geometry_source"] == "route_tangent_strip"
    assert result["road_boundary_drivable_inside"] == ""


def test_with_a_reference_it_projects_and_reports_the_projection():
    generator = StubGenerator(projection=_projection(_occupancy(offset=0.7, width=3.2, clearance=0.15, heading=0.09)))
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(), generator)

    result = monitor.measure(_LOCATION, ego_yaw_rad=0.2, reference_samples=_samples(3))

    assert generator.called("occupancy") == []
    assert len(generator.called("project")) == 1
    assert result["road_boundary_lateral_offset_m"] == 0.7
    assert result["road_boundary_lane_width_m"] == 3.2
    assert result["road_boundary_clearance_m"] == 0.15
    assert result["road_boundary_heading_error_rad"] == 0.09
    assert result["road_boundary_projection_segment_index"] == 3
    assert result["road_boundary_projection_segment_ratio"] == 0.25
    assert result["road_boundary_projection_raw_heading_rad"] == 0.11
    assert result["road_boundary_projection_conditioned_heading_rad"] == 0.04
    assert result["road_boundary_projection_continuity_limited"] is True
    assert result["road_boundary_projection_reason"] == "projection:ok"


def test_the_projection_receives_the_pose_the_footprint_and_the_configured_limits():
    generator = StubGenerator(projection=_projection())
    monitor = MonitorHarness({
        **_NO_DRIVABLE,
        "reference_contract_turn_boundary_margin_m": 0.33,
        "road_boundary_projection_max_heading_step_rad": 0.07,
        "road_boundary_projection_reset_distance_m": 4.4,
        "road_boundary_projection_max_position_step_m": 0.9,
    }, _vehicle(x=2.6, y=1.1), generator)
    samples = _samples(4)

    monitor.measure(_LOCATION, ego_yaw_rad=0.2, reference_samples=samples)

    (call,) = generator.called("project")
    assert call["reference_samples"] is samples
    assert (call["x_m"], call["y_m"], call["heading_rad"]) == (1.0, 2.0, 0.2)
    assert (call["ego_half_width_m"], call["ego_half_length_m"]) == (1.1, 2.6)
    assert call["safety_margin_m"] == 0.33
    assert call["max_heading_step_rad"] == 0.07
    assert call["continuity_reset_distance_m"] == 4.4
    assert call["max_position_step_m"] == 0.9


def test_the_projection_limits_default_when_unconfigured():
    generator = StubGenerator(projection=_projection())
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(), generator)

    monitor.measure(_LOCATION, ego_yaw_rad=0.0, reference_samples=_samples(3))

    (call,) = generator.called("project")
    assert call["safety_margin_m"] == 0.15
    assert call["max_heading_step_rad"] == 0.04
    assert call["continuity_reset_distance_m"] == 2.5
    assert call["max_position_step_m"] == 0.5


def test_the_corridor_query_receives_the_pose_the_footprint_and_the_margin():
    generator = StubGenerator(occupancy=_occupancy())
    monitor = MonitorHarness({**_NO_DRIVABLE, "reference_contract_turn_boundary_margin_m": 0.25},
                             _vehicle(x=2.6, y=1.1), generator)

    monitor.measure(_LOCATION, ego_yaw_rad=0.2)

    (call,) = generator.called("occupancy")
    assert (call["x_m"], call["y_m"], call["heading_rad"]) == (1.0, 2.0, 0.2)
    assert (call["ego_half_width_m"], call["ego_half_length_m"], call["safety_margin_m"]) == (1.1, 2.6, 0.25)


def test_footprint_size_falls_back_to_config_when_the_vehicle_has_no_bounding_box():
    generator = StubGenerator(occupancy=_occupancy())
    monitor = MonitorHarness({**_NO_DRIVABLE, "metrics_ego_half_width_m": 0.9, "reference_vehicle_half_length_m": 2.2},
                             SimpleNamespace(), generator)

    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0)

    (call,) = generator.called("occupancy")
    assert (call["ego_half_width_m"], call["ego_half_length_m"]) == (0.9, 2.2)
    assert result["road_boundary_ego_half_width_m"] == 0.9


def test_footprint_size_defaults_without_config_or_bounding_box():
    generator = StubGenerator(occupancy=_occupancy())
    monitor = MonitorHarness(_NO_DRIVABLE, SimpleNamespace(), generator)

    monitor.measure(_LOCATION, ego_yaw_rad=0.0)

    (call,) = generator.called("occupancy")
    assert (call["ego_half_width_m"], call["ego_half_length_m"]) == (1.0, 2.4)


def test_a_missing_yaw_is_read_from_the_vehicle_transform_in_radians():
    generator = StubGenerator(occupancy=_occupancy())
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(yaw_deg=90.0), generator)

    monitor.measure(_LOCATION)

    (call,) = generator.called("occupancy")
    assert call["heading_rad"] == pytest.approx(math.pi / 2.0)


def test_without_the_drivable_check_a_negative_clearance_is_a_breach():
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(), StubGenerator(occupancy=_occupancy(clearance=-0.05)))
    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0)
    assert result["road_boundary_breach"] is True
    assert (monitor.sample_count, monitor.breach_count) == (1, 1)


def test_without_the_drivable_check_a_zero_clearance_is_not_a_breach():
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(), StubGenerator(occupancy=_occupancy(clearance=0.0)))
    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0)
    assert result["road_boundary_breach"] is False
    assert (monitor.sample_count, monitor.breach_count) == (1, 0)


def test_the_drivable_union_replaces_clearance_source_and_decides_the_breach():
    generator = StubGenerator(occupancy=_occupancy(clearance=-0.5),
                              drivable=_drivable(clearance=0.6, inside=True))
    monitor = MonitorHarness({}, _vehicle(), generator)

    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0)

    assert result["road_boundary_clearance_m"] == 0.6, "clearance now comes from the drivable footprint"
    assert result["road_boundary_geometry_source"] == "drivable_footprint:carla_driving_lane_union"
    assert result["road_boundary_drivable_inside"] is True
    assert result["road_boundary_breach"] is False, "inside the union is never a breach, whatever the strip says"


def test_being_outside_the_drivable_union_is_a_breach_even_with_positive_clearance():
    generator = StubGenerator(occupancy=_occupancy(clearance=0.5), drivable=_drivable(clearance=0.5, inside=False))
    monitor = MonitorHarness({}, _vehicle(), generator)

    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0)

    assert result["road_boundary_breach"] is True
    assert monitor.breach_count == 1


def test_an_invalid_drivable_result_falls_back_to_the_route_strip():
    generator = StubGenerator(occupancy=_occupancy(clearance=0.4), drivable=_drivable(valid=False, inside=False))
    monitor = MonitorHarness({}, _vehicle(), generator)

    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0)

    assert result["road_boundary_clearance_m"] == 0.4
    assert result["road_boundary_geometry_source"] == "route_tangent_strip"
    assert result["road_boundary_drivable_inside"] == ""
    assert result["road_boundary_breach"] is False


def test_the_drivable_query_gets_the_pose_height_footprint_and_margin():
    generator = StubGenerator(occupancy=_occupancy(), drivable=_drivable())
    monitor = MonitorHarness({"reference_contract_turn_boundary_margin_m": 0.21}, _vehicle(x=2.6, y=1.1), generator)

    monitor.measure(_LOCATION, ego_yaw_rad=0.2)

    (call,) = generator.called("drivable")
    assert (call["x_m"], call["y_m"], call["z_m"], call["heading_rad"]) == (1.0, 2.0, 0.7, 0.2)
    assert (call["ego_half_width_m"], call["ego_half_length_m"], call["safety_margin_m"]) == (1.1, 2.6, 0.21)


def test_the_drivable_check_is_on_by_default_and_can_be_turned_off():
    on = StubGenerator(occupancy=_occupancy(), drivable=_drivable())
    MonitorHarness({}, _vehicle(), on).measure(_LOCATION, ego_yaw_rad=0.0)
    assert len(on.called("drivable")) == 1

    off = StubGenerator(occupancy=_occupancy(), drivable=_drivable())
    MonitorHarness(_NO_DRIVABLE, _vehicle(), off).measure(_LOCATION, ego_yaw_rad=0.0)
    assert off.called("drivable") == []


def test_a_location_without_height_queries_the_drivable_union_at_zero():
    generator = StubGenerator(occupancy=_occupancy(), drivable=_drivable())
    MonitorHarness({}, _vehicle(), generator).measure(SimpleNamespace(x=1.0, y=2.0), ego_yaw_rad=0.0)
    (call,) = generator.called("drivable")
    assert call["z_m"] == 0.0


def test_record_sample_false_measures_without_counting():
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(), StubGenerator(occupancy=_occupancy(clearance=-1.0)))

    result = monitor.measure(_LOCATION, ego_yaw_rad=0.0, record_sample=False)

    assert result["road_boundary_breach"] is True
    assert (monitor.sample_count, monitor.breach_count) == (0, 0)


def test_counters_accumulate_across_calls():
    generator = StubGenerator(occupancy=_occupancy(clearance=-1.0))
    monitor = MonitorHarness(_NO_DRIVABLE, _vehicle(), generator)
    monitor.measure(_LOCATION, ego_yaw_rad=0.0)
    generator.occupancy = _occupancy(clearance=1.0)
    monitor.measure(_LOCATION, ego_yaw_rad=0.0)
    monitor.measure(_LOCATION, ego_yaw_rad=0.0)
    assert (monitor.sample_count, monitor.breach_count) == (3, 1)


# ====================================== envelope ======================================

_BLOCKS_FN = "opencda.planning_module.pipeline.candidate_pipeline.build_turn_reference_envelope_blocks"
_CORRECTION_FN = "opencda.planning_module.MPC.lane_keep.road_envelope_conservativeness_correction"


def _mpc(**overrides):
    fields = dict(lane_width_m=3.4, road_envelope_rho=-6.0)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _turn_envelope(config=None, mpc=None, vehicle=None, decision="intersection_turn_left",
                   blocks=("BLOCK",), correction=0.42):
    captured = {}

    def fake_blocks(**kwargs):
        captured["blocks_kwargs"] = kwargs
        return list(blocks)

    def fake_correction(built, *, rho):
        captured["correction_args"] = (built, rho)
        return correction

    with patch(_BLOCKS_FN, fake_blocks), patch(_CORRECTION_FN, fake_correction):
        payload = envelope(
            config or {}, mpc or _mpc(), vehicle if vehicle is not None else _vehicle(),
            behavior_decision=decision, reference_samples=_samples(5), ego_x_m=1.0, ego_y_m=2.0)
    return payload, captured


@pytest.mark.parametrize("decision", ["lane_follow", "lane_change_left", "stop_at_intersection", "", None])
def test_only_intersection_turns_get_an_envelope(decision):
    payload, captured = _turn_envelope(decision=decision)
    assert payload is None
    assert "blocks_kwargs" not in captured


@pytest.mark.parametrize("decision", ["intersection_turn_left", "intersection_turn_right", "  INTERSECTION_TURN_LEFT "])
def test_turn_decisions_are_accepted_regardless_of_case_and_whitespace(decision):
    payload, _ = _turn_envelope(decision=decision)
    assert payload is not None


def test_the_envelope_can_be_disabled_by_config():
    payload, captured = _turn_envelope(config={"turn_mpc_road_envelope_enabled": False})
    assert payload is None and "blocks_kwargs" not in captured


def test_no_blocks_means_no_envelope():
    payload, captured = _turn_envelope(blocks=())
    assert payload is None
    assert "correction_args" not in captured


def test_the_payload_reports_blocks_the_correction_and_rho():
    payload, captured = _turn_envelope(mpc=_mpc(road_envelope_rho=-6.0), blocks=("A", "B"), correction=0.42)
    assert payload["blocks"] == ["A", "B"]
    assert payload["epsilon0"] == 0.42
    assert payload["rho"] == -6.0
    assert captured["correction_args"] == (["A", "B"], -6.0)


def test_rho_defaults_when_the_mpc_has_none():
    payload, _ = _turn_envelope(mpc=SimpleNamespace(lane_width_m=3.4))
    assert payload["rho"] == -8.0


@pytest.mark.parametrize("configured, expected", [(None, 3.0), (0.0, 3.0), (-2.0, 3.0), (1.5, 1.5), ("2.5", 2.5)])
def test_recovery_slack_is_bounded_and_defaults_to_three_meters(configured, expected):
    config = {} if configured is None else {"turn_mpc_road_envelope_recovery_slack_m": configured}
    payload, _ = _turn_envelope(config=config)
    assert payload["max_slack_m"] == expected


def test_the_block_builder_receives_the_reference_pose_and_derived_limits():
    payload, captured = _turn_envelope(
        config={"turn_mpc_road_envelope_safety_margin_m": 0.3, "turn_mpc_road_envelope_overlap_m": 1.25},
        mpc=_mpc(lane_width_m=3.4), vehicle=_vehicle(y=1.1))
    kwargs = captured["blocks_kwargs"]
    assert kwargs["reference_samples"] == _samples(5)
    assert (kwargs["ego_x_m"], kwargs["ego_y_m"]) == (1.0, 2.0)
    assert kwargs["ego_half_width_m"] == 1.1
    assert kwargs["safety_margin_m"] == 0.3
    assert kwargs["default_lane_width_m"] == 3.4
    assert kwargs["longitudinal_overlap_m"] == 1.25


def test_envelope_limits_default_when_unconfigured():
    _, captured = _turn_envelope(mpc=SimpleNamespace(), vehicle=SimpleNamespace())
    kwargs = captured["blocks_kwargs"]
    assert kwargs["ego_half_width_m"] == 1.0
    assert kwargs["safety_margin_m"] == 0.15
    assert kwargs["default_lane_width_m"] == 3.5
    assert kwargs["longitudinal_overlap_m"] == 0.75


def test_the_envelope_margin_falls_back_to_the_turn_boundary_margin_and_is_never_negative():
    _, captured = _turn_envelope(config={"reference_contract_turn_boundary_margin_m": 0.2})
    assert captured["blocks_kwargs"]["safety_margin_m"] == 0.2

    _, captured = _turn_envelope(config={"turn_mpc_road_envelope_safety_margin_m": -1.0})
    assert captured["blocks_kwargs"]["safety_margin_m"] == 0.0


def test_the_overlap_is_never_negative_and_the_half_width_has_a_floor():
    _, captured = _turn_envelope(config={"turn_mpc_road_envelope_overlap_m": -2.0}, vehicle=_vehicle(y=0.01))
    kwargs = captured["blocks_kwargs"]
    assert kwargs["longitudinal_overlap_m"] == 0.0
    assert kwargs["ego_half_width_m"] == 0.1


def test_the_half_width_falls_back_to_config_when_the_vehicle_has_no_extent():
    _, captured = _turn_envelope(config={"metrics_ego_half_width_m": 0.8}, vehicle=SimpleNamespace())
    assert captured["blocks_kwargs"]["ego_half_width_m"] == 0.8


def test_a_real_turn_reference_yields_a_real_envelope():
    # Quarter turn: 20 m straight then a 90-degree left arc, sampled every metre.
    samples = [{"x_ref_m": float(i), "y_ref_m": 0.0, "heading_rad": 0.0, "lane_width_m": 3.5} for i in range(20)]
    radius = 12.0
    for k in range(1, 16):
        angle = (math.pi / 2.0) * k / 15.0
        samples.append({
            "x_ref_m": 20.0 + radius * math.sin(angle),
            "y_ref_m": radius * (1.0 - math.cos(angle)),
            "heading_rad": angle, "lane_width_m": 3.5,
        })

    payload = envelope({}, _mpc(), _vehicle(), behavior_decision="intersection_turn_left",
                       reference_samples=samples, ego_x_m=0.0, ego_y_m=0.0)

    assert payload is not None
    assert len(payload["blocks"]) > 0
    assert math.isfinite(payload["epsilon0"])
    assert payload["max_slack_m"] == 3.0
