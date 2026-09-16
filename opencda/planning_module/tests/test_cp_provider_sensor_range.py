"""CP observation range and kinematic source contract tests."""

from types import SimpleNamespace

import carla

from opencda.planning_module.opencda_bridge.cp_provider import OpenCDACPProvider
from opencda.scenario_testing.scripted_actor import spawn_scripted_actors


def _actor(actor_id, x_m):
    location = carla.Location(x=x_m, y=0.0, z=0.3)
    return SimpleNamespace(
        id=actor_id,
        type_id="vehicle.tesla.model3",
        get_location=lambda: location,
    )


def test_cp_sensor_range_is_independent_of_exchange_radius(tmp_path):
    ego = _actor(1, 0.0)
    observer = _actor(2, 45.0)
    target = _actor(3, 50.0)
    ego_vm = SimpleNamespace(
        vehicle=ego,
        perception_manager=SimpleNamespace(deactivated_detection_range_m=20.0),
    )
    observer_vm = SimpleNamespace(
        vehicle=observer,
        perception_manager=SimpleNamespace(deactivated_detection_range_m=60.0),
    )
    ego_vm.v2x_manager = SimpleNamespace(cav_nearby={"observer": observer_vm})

    class Actors:
        def filter(self, pattern):
            return [ego, observer, target] if pattern == "*vehicle*" else []

    world = SimpleNamespace(get_actors=lambda: Actors())
    provider = OpenCDACPProvider(
        message_path=str(tmp_path / "cp.json"), communication_range_m=60.0
    )
    provider._object_to_cp_message = lambda **kwargs: {
        "id": str(kwargs["obj"].id),
        "observed_by_cav_ids": kwargs["observed_by_cav_ids"],
        "not_observed_by_cav_ids": kwargs["not_observed_by_cav_ids"],
        "visibility_by_cav_id": kwargs["visibility_by_cav_id"],
    }
    messages = provider._native_opencda_messages(
        world=world, vehicle_manager=ego_vm, map_planner=None, sim_time_s=1.0
    )
    target_message = next(item for item in messages if item["id"] == "3")
    assert target_message["observed_by_cav_ids"] == ["2"]
    assert target_message["not_observed_by_cav_ids"] == ["1"]
    assert target_message["visibility_by_cav_id"]["1"] == "out_of_sensor_range"
    assert target_message["blind_spot_shared"] is True


def test_cp_recovers_kinematic_actor_velocity_from_poses(tmp_path):
    provider = OpenCDACPProvider(message_path=str(tmp_path / "cp.json"))
    assert provider._observed_motion(
        actor_id="3", x_m=0.0, y_m=0.0,
        timestamp_s=0.0, heading_rad=0.0,
    ) == (0.0, 0.0)
    speed, heading = provider._observed_motion(
        actor_id="3", x_m=1.2, y_m=0.0,
        timestamp_s=0.2, heading_rad=0.0,
    )
    assert abs(speed - 6.0) < 1.0e-6
    assert abs(heading) < 1.0e-6


def test_cp_message_uses_recovered_kinematic_vehicle_speed(tmp_path):
    provider = OpenCDACPProvider(message_path=str(tmp_path / "cp.json"))

    class Vehicle:
        id = 3
        type_id = "vehicle.tesla.model3"

        def __init__(self):
            self.location = carla.Location(x=0.0, y=0.0, z=0.3)
            self.bounding_box = SimpleNamespace(
                extent=carla.Vector3D(x=2.0, y=1.0, z=0.8)
            )

        def get_transform(self):
            return carla.Transform(self.location, carla.Rotation(yaw=0.0))

        def get_velocity(self):
            return carla.Vector3D(x=0.0, y=0.0, z=0.0)

    vehicle = Vehicle()
    kwargs = dict(
        obj=vehicle,
        map_planner=None,
        ego_location=carla.Location(x=-10.0, y=0.0, z=0.3),
        source="test",
        provider_source="test",
        fallback_id="3",
        object_type="vehicle",
        skip_range_filter=True,
    )
    provider._object_to_cp_message(sim_time_s=0.0, **kwargs)
    vehicle.location = carla.Location(x=1.2, y=0.0, z=0.3)
    message = provider._object_to_cp_message(sim_time_s=0.2, **kwargs)
    assert abs(message["state"][2] - 6.0) < 1.0e-6
    assert message["trajectory"][-1][0] > vehicle.location.x


def test_scripted_vehicle_has_only_kinematic_motion_owner():
    class Blueprint:
        def has_attribute(self, name):
            return False

    class Vehicle:
        id = 3
        type_id = "vehicle.tesla.model3"

        def __init__(self):
            self.physics_calls = []
            self.transforms = []

        def set_simulate_physics(self, enabled):
            self.physics_calls.append(enabled)

        def set_transform(self, transform):
            self.transforms.append(transform)

        def set_target_velocity(self, velocity):
            raise AssertionError("kinematic scripted actor must not use CARLA dynamics")

    vehicle = Vehicle()
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: Blueprint()),
        try_spawn_actor=lambda blueprint, transform: vehicle,
    )
    actors = spawn_scripted_actors(world, [{
        "path": [[0.0, 0.0], [10.0, 0.0]],
        "speed_mps": 6.0,
    }])
    assert len(actors) == 1
    assert vehicle.physics_calls == [False]
    actors[0].step(0.2)
    assert abs(vehicle.transforms[-1].location.x - 1.2) < 1.0e-6
