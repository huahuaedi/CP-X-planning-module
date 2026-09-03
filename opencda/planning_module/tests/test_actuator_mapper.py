from opencda.planning_module.pipeline.actuator_mapper import CarlaActuatorMapper


def test_compensation_adds_feedforward_and_speed_error_boost():
    mapper = CarlaActuatorMapper({})
    command = mapper.map_acceleration(
        acceleration_mps2=1.0,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=1.5,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )
    assert command.throttle > 1.0 / 3.0
    assert command.throttle == 0.55
    assert command.brake == 0.0


def test_stop_intent_disables_positive_pedal_command():
    mapper = CarlaActuatorMapper({})
    command = mapper.map_acceleration(
        acceleration_mps2=1.0,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=0.0,
        target_speed_mps=3.0,
        stop_goal_active=True,
    )
    assert command.throttle == 0.0
    assert command.brake == 0.0


def test_braking_is_not_mixed_with_throttle():
    mapper = CarlaActuatorMapper({})
    command = mapper.map_acceleration(
        acceleration_mps2=-1.5,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=2.0,
        target_speed_mps=0.0,
        stop_goal_active=True,
    )
    assert command.throttle == 0.0
    assert command.brake == 0.15


def test_tracking_brake_uses_carla_calibration_and_cap():
    mapper = CarlaActuatorMapper({})
    moderate = mapper.map_acceleration(
        acceleration_mps2=-2.1,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=2.0,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )
    saturated = mapper.map_acceleration(
        acceleration_mps2=-10.0,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=2.0,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )

    assert abs(moderate.brake - 0.15) < 1.0e-9
    assert saturated.brake == 0.22


def test_calibrated_brake_round_trips_to_requested_acceleration():
    mapper = CarlaActuatorMapper({})
    command = mapper.map_acceleration(
        acceleration_mps2=-1.4,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=2.0,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )
    recovered = mapper.acceleration_from_command(
        throttle=command.throttle,
        brake=command.brake,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=2.0,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )

    assert abs(recovered + 1.4) < 1.0e-9


def test_high_brake_pedal_inverse_is_clamped_to_the_vehicle_limit():
    # The fallback / bounded-stop path sets the brake pedal directly near
    # 1.0. Inverting it with the forward pedal-mapping gain would report an
    # impossible decel (~-14 m/s^2) that then poisons the next MPC jerk seed.
    mapper = CarlaActuatorMapper({})
    recovered = mapper.acceleration_from_command(
        throttle=0.0,
        brake=1.0,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=8.0,
        target_speed_mps=6.0,
        stop_goal_active=False,
    )
    assert recovered == -3.0
    # a moderate pedal still inverts linearly (unclamped)
    mid = mapper.acceleration_from_command(
        throttle=0.0, brake=0.1,
        max_acceleration_mps2=3.0, min_acceleration_mps2=-3.0,
        ego_speed_mps=8.0, target_speed_mps=6.0, stop_goal_active=False,
    )
    assert -3.0 < mid < 0.0


def test_measured_acceleration_uses_speed_delta_and_filter():
    mapper = CarlaActuatorMapper({"actuator_measured_accel_alpha": 1.0})
    assert mapper.update_measurement(speed_mps=1.0, timestamp_s=2.0) == 0.0
    assert abs(
        mapper.update_measurement(speed_mps=1.2, timestamp_s=2.1) - 2.0
    ) < 1.0e-9


def test_compensated_command_round_trips_to_requested_acceleration():
    mapper = CarlaActuatorMapper({})
    command = mapper.map_acceleration(
        acceleration_mps2=0.5,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=1.5,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )
    recovered = mapper.acceleration_from_command(
        throttle=command.throttle,
        brake=command.brake,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=1.5,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )
    assert abs(recovered - 0.5) < 1.0e-9


def test_positive_acceleration_tapers_throttle_just_past_overspeed():
    # 0.3 m/s over target: past the 0.15 deadband, but well inside the
    # default 1.0 m/s taper band, so throttle rolls off smoothly instead
    # of stepping straight to zero (a hard cutoff here was a real
    # disturbance at higher cruise speeds -- see actuator_mapper.py's
    # overspeed_taper_mps docstring).
    mapper = CarlaActuatorMapper({})
    command = mapper.map_acceleration(
        acceleration_mps2=1.5,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=3.3,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )
    assert 0.0 < command.throttle < 0.5
    assert command.brake == 0.0


def test_positive_acceleration_coasts_well_past_overspeed_taper():
    mapper = CarlaActuatorMapper({})
    command = mapper.map_acceleration(
        acceleration_mps2=1.5,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=4.5,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )
    assert command.throttle == 0.0
    assert command.brake == 0.0


def test_small_mpc_acceleration_clears_carla_launch_deadzone():
    mapper = CarlaActuatorMapper({})
    command = mapper.map_acceleration(
        acceleration_mps2=0.06,
        max_acceleration_mps2=3.0,
        min_acceleration_mps2=-3.0,
        ego_speed_mps=0.02,
        target_speed_mps=3.0,
        stop_goal_active=False,
    )
    assert 0.29 <= command.throttle <= 0.31
