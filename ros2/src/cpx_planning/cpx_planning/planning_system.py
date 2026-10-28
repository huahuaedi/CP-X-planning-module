"""Run the six topic-connected CP-X ROS nodes in one Python process."""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from cpx_planning.component_interfaces import LocalComponentBus
from cpx_planning.global_planner_node import GlobalPlannerNode
from cpx_planning.planner_node import CPXPlannerNode
from cpx_planning.pipeline.performance_profiler import PlannerStageProfiler


def main(args=None):
    """Run map ownership and the current planner behind ROS boundaries."""
    rclpy.init(args=args)
    local_bus = LocalComponentBus()
    global_planner_node = GlobalPlannerNode(local_bus=local_bus)
    planner_node = CPXPlannerNode(
        local_map_planner=global_planner_node.map_planner,
        local_bus=local_bus,
        map_owner=global_planner_node,
    )
    stage_profiler = PlannerStageProfiler(
        enabled=bool(planner_node.debug_time),
        external_recorder=lambda label, duration_ms: planner_node.timing_recorder.record_duration(0, label, duration_ms),
    )
    stage_profiler.instrument_global_planner(global_planner_node.map_planner)
    stage_profiler.instrument_route_manager(planner_node.planner_bridge.route_manager)
    stage_profiler.instrument_reference_map(getattr(planner_node.planner_bridge, "reference_map", None))
    stage_profiler.instrument_bridge(planner_node.planner_bridge)
    stage_profiler.instrument_mpc(planner_node.planner_bridge.mpc)
    planner_node.stage_profiler = stage_profiler
    local_bus.mirror_markers = bool(planner_node.debug)
    local_bus.mirror_payloads = bool(planner_node.debug)
    nodes = [global_planner_node, planner_node]
    # Planning stages now run inside the imported current planner. Workers
    # serve ROS input/output callbacks; they do not split algorithm ownership.
    executor = MultiThreadedExecutor(num_threads=4)
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
