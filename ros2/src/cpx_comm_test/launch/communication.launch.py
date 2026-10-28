"""Start the unified OpenCDA-to-ROS communication boundary."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start one node that owns all TCP and ROS boundary I/O."""
    debug = LaunchConfiguration("debug")
    debug_time = LaunchConfiguration("debug_time")
    bridge_cpu_set = LaunchConfiguration("bridge_cpu_set")

    def bridge_node(context):
        cpu_set = bridge_cpu_set.perform(context).strip()
        return [Node(
            package="cpx_comm_test",
            executable="opencda_bridge",
            name="opencda_bridge_node",
            output="screen",
            parameters=[{"debug": debug, "debug_time": debug_time}],
            prefix=("taskset -c {}".format(cpu_set) if cpu_set else ""),
        )]

    return LaunchDescription([
        DeclareLaunchArgument("debug", default_value="false"),
        DeclareLaunchArgument("debug_time", default_value="false"),
        DeclareLaunchArgument(
            "bridge_cpu_set",
            default_value="",
            description="Optional CPU set reserved for the unified Bridge.",
        ),
        OpaqueFunction(function=bridge_node),
    ])
