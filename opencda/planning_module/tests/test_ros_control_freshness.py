import time

from opencda.core.common.vehicle_manager import VehicleManager


class _SingleControlReceiver(object):
    def __init__(self, control):
        self.control = dict(control)

    def pop_latest_planner_control(self, _cycle_time_s, session_id=None):
        control, self.control = self.control, None
        return control


def _manager_with_control(control, max_age_s=0.5):
    manager = VehicleManager.__new__(VehicleManager)
    manager.ros_receiver = _SingleControlReceiver(control)
    manager._ros_planner_session_id = "vehicle-test"
    manager._latest_ros_output = None
    manager._ros_control_max_age_s = float(max_age_s)
    return manager


def test_ros_control_freshness_uses_wall_time_not_simulation_stamp_lag():
    manager = _manager_with_control({
        "timestamp_s": 10.0,
        "source_send_wall_time_ns": time.time_ns(),
    })

    control, is_new, age_s = manager._latest_valid_ros_control(20.0)

    assert control is not None
    assert is_new is True
    assert age_s < 0.5
    assert control["ros_control_simulation_age_s"] == 10.0
    assert control["ros_control_wall_age_s"] < 0.5


def test_ros_control_rejects_result_old_in_wall_time():
    manager = _manager_with_control({
        "timestamp_s": 20.0,
        "source_send_wall_time_ns": time.time_ns() - int(1.0e9),
    })

    control, is_new, age_s = manager._latest_valid_ros_control(20.0)

    assert control is None
    assert is_new is True
    assert age_s >= 1.0
